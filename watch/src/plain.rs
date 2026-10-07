//! --plain: one line per event, for CI logs, pipes and terminals where a full-screen view won't do.

use crate::event::Ev;
use crate::fmt::{clean, hms, secs};
use crate::state::{State, short_name};

/// The line for an event, or None for events that only update the totals. `state` has already
/// seen the event, so it can name the call or run an event belongs to.
pub fn line(ev: &Ev, state: &State) -> Option<String> {
    let call_name = |key: &str| {
        state
            .calls
            .iter()
            .rev()
            .find(|c| c.key == key)
            .map(|c| short_name(&c.function).to_string())
            .unwrap_or_else(|| "(thunc.call)".into())
    };
    let agent =
        |key: &str| state.runs.iter().find(|r| r.key == key).map(|r| r.agent.clone()).unwrap_or_else(|| "agent".into());
    let text = match ev {
        Ev::CallStart { .. } | Ev::AgentReply { .. } | Ev::AgentTool { .. } => return None,
        Ev::CallAttempt { ok: true, .. } => return None,
        Ev::CallAttempt { key, n, seconds, problem, .. } => {
            format!(
                "retry  {}  attempt {n} rejected after {}: {}",
                call_name(key),
                secs(*seconds),
                problem.as_deref().unwrap_or("invalid")
            )
        }
        Ev::CallEnd { key, ok: true, cached: true, value, .. } => {
            format!("call   {}  → {}  (cached)", call_name(key), value.as_deref().unwrap_or(""))
        }
        Ev::CallEnd { key, ok: true, seconds, attempts, value, .. } => {
            let tries = if *attempts > 1 { format!(", {attempts} attempts") } else { String::new() };
            format!("call   {}  → {}  ({}{tries})", call_name(key), value.as_deref().unwrap_or(""), secs(*seconds))
        }
        Ev::CallEnd { key, seconds, error, .. } => {
            format!("FAIL   {}  {}  ({})", call_name(key), error.as_deref().unwrap_or("failed"), secs(*seconds))
        }
        Ev::AgentStart { key, .. } => {
            let line = state.runs.iter().find(|r| &r.key == key).map(|r| r.call_line()).unwrap_or_default();
            format!("agent  {}  started {line}", agent(key))
        }
        Ev::AgentStep { key, n, tool, target, result, seconds, denied, .. } => {
            let what = if *denied { format!("DENIED {result}") } else { result.chars().take(80).collect() };
            format!("step   {}  #{n} {tool} {target}  {}  ({})", agent(key), what, secs(*seconds))
        }
        Ev::AgentEnd { key, ok: true, steps, value, .. } => {
            format!("agent  {}  finished in {steps} steps → {}", agent(key), value.as_deref().unwrap_or(""))
        }
        Ev::AgentEnd { key, error, .. } => format!("FAIL   {}  {}", agent(key), error.as_deref().unwrap_or("failed")),
    };
    Some(format!("{}  {}", hms(ev.t()), clean(&text)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::parse_event_line;

    #[test]
    fn lines_name_the_function_and_the_outcome() {
        let mut s = State::new(0.0);
        let mut seen = Vec::new();
        for l in [
            r#"{"v":1,"event":"call.start","t":10,"pid":1,"id":1,"function":"urgency"}"#,
            r#"{"v":1,"event":"call.attempt","t":12,"pid":1,"id":1,"n":1,"seconds":2,"ok":false,"problem":"not an int"}"#,
            r#"{"v":1,"event":"call.attempt","t":13,"pid":1,"id":1,"n":2,"seconds":1,"ok":true}"#,
            r#"{"v":1,"event":"call.end","t":13,"pid":1,"id":1,"ok":true,"attempts":2,"seconds":3,"value":"4"}"#,
        ] {
            let e = parse_event_line(l).unwrap();
            s.apply(e.clone());
            seen.extend(line(&e, &s));
        }
        assert_eq!(seen.len(), 2);
        assert!(seen[0].ends_with("retry  urgency  attempt 1 rejected after 2.0s: not an int"));
        assert!(seen[1].ends_with("call   urgency  → 4  (3.0s, 2 attempts)"));
    }
}
