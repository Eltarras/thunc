//! The end-of-run report: the same tables as `thunc run --profile`, built from the events. Shown on
//! the summary screen and printed when the dashboard closes.

use std::collections::BTreeMap;

use crate::fmt::{fit, secs};
use crate::state::{Link, State, short_name};

/// The report's lines; a line about one function or agent task links to it.
pub type Lines = Vec<(Kind, String, Option<Link>)>;

struct Out(Lines);

impl Out {
    fn push(&mut self, (kind, text): (Kind, String)) {
        self.0.push((kind, text, None));
    }
    fn link(&mut self, kind: Kind, text: String, link: Link) {
        self.0.push((kind, text, Some(link)));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Title,
    Section,
    Header,
    Row,
    Warn,
    Note,
    Blank,
}

/// `one_program`: the events come from one program's run (not an agents folder or a shared events
/// file), so its wall time means something.
pub fn lines(state: &State, status: &str, one_program: bool) -> Lines {
    let title = if one_program {
        format!("thunc watch: {} wall time{status}", secs(state.wall()))
    } else {
        let n = |count: usize, one: &str, many: &str| format!("{count} {}", if count == 1 { one } else { many });
        format!(
            "thunc watch: {}, {}{status}",
            n(state.calls.len(), "call", "calls"),
            n(state.runs.len(), "agent run", "agent runs")
        )
    };
    let mut out = Out(vec![(Kind::Title, title, None)]);
    if state.functions.is_empty() && state.runs.is_empty() {
        out.push((Kind::Note, "No thunc calls or agent runs yet.".into()));
        return out.0;
    }

    let mut functions: Vec<_> = state.functions.iter().filter(|f| f.calls > 0).collect();
    functions.sort_by(|a, b| b.total.total_cmp(&a.total));
    if !functions.is_empty() {
        let w = functions.iter().map(|f| short_name(&f.name).chars().count()).max().unwrap_or(8).clamp(8, 28);
        out.push((Kind::Blank, String::new()));
        out.push((Kind::Section, "CALLS".into()));
        out.push((
            Kind::Header,
            format!(
                "{}  {:>5}  {:>6}  {:>7}  {:>6}  {:>8}  {:>7}  {:>7}  {:>7}  {:>8}",
                fit("FUNCTION", w),
                "CALLS",
                "CACHED",
                "RETRIES",
                "FAILED",
                "TOTAL",
                "MEAN",
                "P95",
                "MAX",
                "MODEL"
            ),
        ));
        for f in functions {
            let opt = |v: Option<f64>| v.map(secs).unwrap_or_else(|| "-".into());
            let row = format!(
                "{}  {:>5}  {:>6}  {:>7}  {:>6}  {:>8}  {:>7}  {:>7}  {:>7}  {:>8}",
                fit(short_name(&f.name), w),
                f.calls,
                f.cached,
                f.retries,
                f.failed,
                secs(f.total),
                opt(f.mean()),
                opt(f.percentile(95.0)),
                opt(f.max()),
                secs(f.model)
            );
            out.link(if f.failed > 0 { Kind::Warn } else { Kind::Row }, row, Link::Function(f.name.clone()));
        }
    }

    if !state.runs.is_empty() {
        let mut tasks: BTreeMap<String, Vec<&crate::state::Run>> = BTreeMap::new();
        for r in &state.runs {
            tasks.entry(format!("{} · {}", r.agent, short_name(&r.task))).or_default().push(r);
        }
        let w = tasks.keys().map(|k| k.chars().count()).max().unwrap_or(4).clamp(4, 36);
        out.push((Kind::Blank, String::new()));
        out.push((Kind::Section, "AGENT RUNS".into()));
        out.push((
            Kind::Header,
            format!(
                "{}  {:>4}  {:>5}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
                fit("AGENT · TASK", w),
                "RUNS",
                "STEPS",
                "FAILED",
                "TOTAL",
                "MAX",
                "MODEL",
                "TOOLS"
            ),
        ));
        let mut rows: Vec<_> = tasks.into_iter().collect();
        rows.sort_by(|a, b| {
            let total = |rs: &Vec<&crate::state::Run>| rs.iter().map(|r| r.seconds(state.now)).sum::<f64>();
            total(&b.1).total_cmp(&total(&a.1))
        });
        for (name, runs) in rows {
            let times: Vec<f64> = runs.iter().map(|r| r.seconds(state.now)).collect();
            let failed = runs.iter().filter(|r| r.end.as_ref().is_some_and(|e| !e.ok)).count();
            let row = format!(
                "{}  {:>4}  {:>5}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}",
                fit(&name, w),
                runs.len(),
                runs.iter().map(|r| r.step_count()).sum::<usize>(),
                failed,
                secs(times.iter().sum()),
                secs(times.iter().cloned().fold(0.0, f64::max)),
                if runs.iter().all(|r| r.measured()) { secs(runs.iter().map(|r| r.model).sum()) } else { "-".into() },
                if runs.iter().all(|r| r.measured()) { secs(runs.iter().map(|r| r.tools).sum()) } else { "-".into() },
            );
            out.link(if failed > 0 { Kind::Warn } else { Kind::Row }, row, Link::Run(runs[0].key.clone()));
        }

        let mut tools: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for r in state.runs.iter().filter(|r| r.measured()) {
            for s in &r.steps {
                if s.tool != "finish" && s.tool != "reply" {
                    tools.entry(&s.tool).or_default().push(s.seconds);
                }
            }
        }
        if !tools.is_empty() {
            out.push((Kind::Blank, String::new()));
            out.push((Kind::Section, "AGENT TOOLS".into()));
            out.push((
                Kind::Header,
                format!("{}  {:>5}  {:>8}  {:>8}  {:>8}", fit("TOOL", 8), "CALLS", "TOTAL", "MEAN", "MAX"),
            ));
            let mut tools: Vec<_> = tools.into_iter().collect();
            tools.sort_by(|a, b| b.1.iter().sum::<f64>().total_cmp(&a.1.iter().sum::<f64>()));
            for (tool, times) in tools {
                let total: f64 = times.iter().sum();
                out.push((
                    Kind::Row,
                    format!(
                        "{}  {:>5}  {:>8}  {:>8}  {:>8}",
                        fit(tool, 8),
                        times.len(),
                        secs(total),
                        secs(total / times.len() as f64),
                        secs(times.iter().cloned().fold(0.0, f64::max))
                    ),
                ));
            }
        }
    }

    let (busy, summed) = state.busy();
    let wall = state.wall();
    let model: f64 = state.functions.iter().map(|f| f.model).sum::<f64>()
        + state.runs.iter().filter(|r| r.measured()).map(|r| r.model).sum::<f64>();
    let share = |part: f64, whole: f64| {
        if whole > 0.0 { format!("{:.0}%", 100.0 * part / whole) } else { "-".into() }
    };
    out.push((Kind::Blank, String::new()));
    if one_program {
        out.push((
            Kind::Row,
            format!("In thunc:      {} of {} wall time ({})", secs(busy), secs(wall), share(busy, wall)),
        ));
    }
    if model > 0.0 {
        out.push((
            Kind::Row,
            format!("Model time:    {}, {} of the time in calls and runs", secs(model), share(model, summed)),
        ));
    }
    if busy > 0.0 && summed > busy * 1.05 {
        out.push((
            Kind::Row,
            format!("Concurrency:   overlapped {:.1}x on average (thunc.map or threads)", summed / busy),
        ));
    }
    let slowest_call = state.calls.iter().map(|c| (short_name(&c.function).to_string(), c.seconds(state.now)));
    let slowest_run =
        state.runs.iter().map(|r| (format!("{} · {}", r.agent, short_name(&r.task)), r.seconds(state.now)));
    if let Some((name, s)) = slowest_call.chain(slowest_run).max_by(|a, b| a.1.total_cmp(&b.1)) {
        out.push((Kind::Row, format!("Slowest:       {name} took {}", secs(s))));
    }
    if state.dropped_calls > 0 {
        out.push((Kind::Note, format!("(The {} oldest calls are counted but no longer listed.)", state.dropped_calls)));
    }
    out.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::parse_event_line;

    #[test]
    fn the_report_has_the_profile_sections() {
        let mut s = State::new(20.0);
        for l in [
            r#"{"v":1,"event":"call.start","t":10,"pid":1,"id":1,"function":"urgency"}"#,
            r#"{"v":1,"event":"call.attempt","t":12,"pid":1,"id":1,"n":1,"seconds":2,"ok":true}"#,
            r#"{"v":1,"event":"call.end","t":12,"pid":1,"id":1,"ok":true,"attempts":1,"seconds":2,"value":"4"}"#,
            r#"{"v":1,"event":"agent.start","t":10,"pid":1,"id":2,"agent":"repo-guide","task":"tests_for"}"#,
            r#"{"v":1,"event":"agent.reply","t":10,"pid":1,"id":2,"n":1,"seconds":0}"#,
            r#"{"v":1,"event":"agent.step","t":13,"pid":1,"id":2,"n":1,"tool":"run","target":"pytest","seconds":3}"#,
            r#"{"v":1,"event":"agent.end","t":16,"pid":1,"id":2,"ok":true,"steps":2}"#,
        ] {
            s.apply(parse_event_line(l).unwrap());
        }
        let text: Vec<String> = lines(&s, "", true).into_iter().map(|(_, l, _)| l).collect();
        let all = text.join("\n");
        for needle in [
            "CALLS",
            "urgency",
            "AGENT RUNS",
            "repo-guide · tests_for",
            "AGENT TOOLS",
            "run",
            "Slowest:       repo-guide · tests_for took 6.0s",
        ] {
            assert!(all.contains(needle), "missing {needle:?} in\n{all}");
        }
    }
}
