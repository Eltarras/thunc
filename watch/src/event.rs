//! What the dashboard is told: events from THUNC_EVENTS (see thunc/events.py), and the same events
//! read back from an agent's session records in .thunc_agents/<agent>/sessions/*.jsonl.

use serde_json::Value;

/// Calls and runs are keyed by "pid:id" (one events file can hold several processes), or by the
/// session path for runs read from an agents folder.
pub type Key = String;

#[derive(Debug, Clone, PartialEq)]
pub enum Ev {
    CallStart {
        key: Key,
        t: f64,
        function: String,
        backend: String,
        model: String,
        inputs: Vec<(String, String)>,
    },
    CallAttempt {
        key: Key,
        t: f64,
        n: u32,
        seconds: f64,
        ok: bool,
        problem: Option<String>,
        reply: Option<String>,
    },
    CallEnd {
        key: Key,
        t: f64,
        ok: bool,
        cached: bool,
        attempts: u32,
        seconds: f64,
        value: Option<String>,
        error: Option<String>,
    },
    AgentStart {
        key: Key,
        t: f64,
        agent: String,
        task: String,
        returns: String,
        session: Option<String>,
        inputs: Vec<(String, String)>,
    },
    AgentReply {
        key: Key,
        t: f64,
        n: u32,
        seconds: f64,
    },
    AgentTool {
        key: Key,
        t: f64,
        n: u32,
        tool: String,
        target: String,
    },
    AgentStep {
        key: Key,
        t: f64,
        n: u32,
        tool: String,
        target: String,
        result: String,
        seconds: f64,
        denied: bool,
    },
    AgentEnd {
        key: Key,
        t: f64,
        ok: bool,
        steps: u32,
        seconds: f64,
        files_changed: Vec<String>,
        value: Option<String>,
        error: Option<String>,
    },
}

impl Ev {
    pub fn t(&self) -> f64 {
        match self {
            Ev::CallStart { t, .. }
            | Ev::CallAttempt { t, .. }
            | Ev::CallEnd { t, .. }
            | Ev::AgentStart { t, .. }
            | Ev::AgentReply { t, .. }
            | Ev::AgentTool { t, .. }
            | Ev::AgentStep { t, .. }
            | Ev::AgentEnd { t, .. } => *t,
        }
    }
}

fn text(v: &Value, field: &str) -> Option<String> {
    match v.get(field)? {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

fn num(v: &Value, field: &str) -> f64 {
    v.get(field).and_then(Value::as_f64).unwrap_or(0.0)
}

fn count(v: &Value, field: &str) -> u32 {
    v.get(field).and_then(Value::as_u64).unwrap_or(0) as u32
}

fn flag(v: &Value, field: &str) -> bool {
    v.get(field).and_then(Value::as_bool).unwrap_or(false)
}

fn strings(v: &Value, field: &str) -> Vec<String> {
    v.get(field)
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect())
        .unwrap_or_default()
}

fn inputs(v: &Value) -> Vec<(String, String)> {
    v.get("inputs")
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .map(|(k, x)| (k.clone(), x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// One line of a THUNC_EVENTS file. Lines it doesn't understand (a newer format, a half-written
/// line) are skipped rather than stopping the dashboard.
pub fn parse_event_line(line: &str) -> Option<Ev> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    if v.get("v").and_then(Value::as_u64).unwrap_or(1) != 1 {
        return None;
    }
    let event = v.get("event")?.as_str()?;
    let t = num(&v, "t");
    let key = format!("{}:{}", v.get("pid").and_then(Value::as_i64).unwrap_or(0), count(&v, "id"));
    Some(match event {
        "call.start" => Ev::CallStart {
            key,
            t,
            function: text(&v, "function").unwrap_or_else(|| "(thunc.call)".into()),
            backend: text(&v, "backend").unwrap_or_default(),
            model: text(&v, "model").unwrap_or_default(),
            inputs: inputs(&v),
        },
        "call.attempt" => Ev::CallAttempt {
            key,
            t,
            n: count(&v, "n"),
            seconds: num(&v, "seconds"),
            ok: flag(&v, "ok"),
            problem: text(&v, "problem"),
            reply: text(&v, "reply"),
        },
        "call.end" => Ev::CallEnd {
            key,
            t,
            ok: flag(&v, "ok"),
            cached: flag(&v, "cached"),
            attempts: count(&v, "attempts"),
            seconds: num(&v, "seconds"),
            value: text(&v, "value"),
            error: text(&v, "error"),
        },
        "agent.start" => Ev::AgentStart {
            key,
            t,
            agent: text(&v, "agent").unwrap_or_else(|| "agent".into()),
            task: text(&v, "task").unwrap_or_default(),
            returns: text(&v, "returns").unwrap_or_default(),
            session: text(&v, "session"),
            inputs: inputs(&v),
        },
        "agent.reply" => Ev::AgentReply { key, t, n: count(&v, "n"), seconds: num(&v, "seconds") },
        "agent.tool" => Ev::AgentTool {
            key,
            t,
            n: count(&v, "n"),
            tool: text(&v, "tool").unwrap_or_default(),
            target: text(&v, "target").unwrap_or_default(),
        },
        "agent.step" => Ev::AgentStep {
            key,
            t,
            n: count(&v, "n"),
            tool: text(&v, "tool").unwrap_or_default(),
            target: text(&v, "target").unwrap_or_default(),
            result: text(&v, "result").unwrap_or_default(),
            seconds: num(&v, "seconds"),
            denied: flag(&v, "denied"),
        },
        "agent.end" => Ev::AgentEnd {
            key,
            t,
            ok: flag(&v, "ok"),
            steps: count(&v, "steps"),
            seconds: num(&v, "seconds"),
            files_changed: strings(&v, "files_changed"),
            value: text(&v, "value"),
            error: text(&v, "error"),
        },
        _ => return None,
    })
}

/// What a tool call acted on, from its arguments; the same rule as thunc.events.target.
pub fn target(tool: &str, args: &Value) -> String {
    let s = |k: &str| args.get(k).and_then(Value::as_str);
    if tool == "search"
        && let Some(p) = s("pattern")
    {
        return match s("path") {
            Some(path) => format!("'{p}' in {path}"),
            None => format!("'{p}'"),
        };
    }
    for k in ["command", "path", "note"] {
        if let Some(x) = s(k) {
            return x.to_string();
        }
    }
    match args.as_object() {
        Some(o) if !o.is_empty() => args.to_string(),
        _ => String::new(),
    }
}

/// Turns one session record, line by line, into agent events. Session lines have a "time" to the
/// second and no model timings, so step durations here are the time between lines.
pub struct SessionReader {
    pub key: Key,
    agent: String,
    session: String,
    prev_t: Option<f64>,
    steps: u32,
    pub ended: bool,
    pub started: bool,
}

impl SessionReader {
    pub fn new(agent: &str, session: &str) -> Self {
        SessionReader {
            key: session.to_string(),
            agent: agent.to_string(),
            session: session.to_string(),
            prev_t: None,
            steps: 0,
            ended: false,
            started: false,
        }
    }

    pub fn line(&mut self, line: &str) -> Vec<Ev> {
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            return vec![];
        };
        let Some(event) = v.get("event").and_then(Value::as_str) else {
            return vec![];
        };
        let t = v.get("time").and_then(Value::as_str).and_then(crate::fmt::parse_session_time).unwrap_or(0.0);
        let since = self.prev_t.map(|p| (t - p).max(0.0)).unwrap_or(0.0);
        self.prev_t = Some(t);
        let key = self.key.clone();
        match event {
            "start" => {
                self.started = true;
                vec![Ev::AgentStart {
                    key,
                    t,
                    agent: self.agent.clone(),
                    task: text(&v, "task").unwrap_or_default(),
                    returns: text(&v, "returns").unwrap_or_default(),
                    session: Some(self.session.clone()),
                    inputs: inputs(&v),
                }]
            }
            "step" => {
                self.steps += 1;
                let tool = text(&v, "tool").unwrap_or_else(|| "reply".into());
                let args = v.get("args").cloned().unwrap_or(Value::Null);
                vec![Ev::AgentStep {
                    key,
                    t,
                    n: count(&v, "n").max(self.steps),
                    target: target(&tool, &args),
                    tool,
                    result: text(&v, "result").unwrap_or_default(),
                    seconds: since,
                    denied: flag(&v, "denied"),
                }]
            }
            "finish" => {
                self.ended = true;
                self.steps += 1;
                let value = text(&v, "value");
                vec![
                    Ev::AgentStep {
                        key: key.clone(),
                        t,
                        n: count(&v, "n").max(self.steps),
                        tool: "finish".into(),
                        target: value.clone().unwrap_or_default(),
                        result: "accepted".into(),
                        seconds: since,
                        denied: false,
                    },
                    Ev::AgentEnd {
                        key,
                        t,
                        ok: true,
                        steps: self.steps,
                        seconds: 0.0,
                        files_changed: strings(&v, "files_changed"),
                        value,
                        error: None,
                    },
                ]
            }
            "error" => {
                self.ended = true;
                vec![Ev::AgentEnd {
                    key,
                    t,
                    ok: false,
                    steps: self.steps,
                    seconds: 0.0,
                    files_changed: strings(&v, "files_changed"),
                    value: None,
                    error: text(&v, "error"),
                }]
            }
            _ => vec![],
        }
    }

    /// The run stopped without writing its end: its process is gone.
    pub fn interrupted(&mut self, t: f64) -> Ev {
        self.ended = true;
        Ev::AgentEnd {
            key: self.key.clone(),
            t,
            ok: false,
            steps: self.steps,
            seconds: 0.0,
            files_changed: vec![],
            value: None,
            error: Some("stopped without finishing: its process ended".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_lines() {
        let e = parse_event_line(
            r#"{"v": 1, "event": "call.attempt", "t": 41.18, "pid": 7, "id": 3, "n": 1, "seconds": 2.07,
                "ok": false, "problem": "ensure rejected 7", "reply": "7"}"#,
        );
        assert_eq!(
            e,
            Some(Ev::CallAttempt {
                key: "7:3".into(),
                t: 41.18,
                n: 1,
                seconds: 2.07,
                ok: false,
                problem: Some("ensure rejected 7".into()),
                reply: Some("7".into()),
            })
        );
        assert_eq!(parse_event_line(r#"{"v": 2, "event": "call.start"}"#), None);
        assert_eq!(parse_event_line(r#"{"v": 1, "event": "call.st"#), None);
        assert_eq!(parse_event_line(r#"{"v": 1, "event": "something.new"}"#), None);
    }

    #[test]
    fn a_call_without_a_function_name_is_thunc_call() {
        let e = parse_event_line(r#"{"v":1,"event":"call.start","t":1,"pid":1,"id":1,"function":null}"#);
        assert!(matches!(e, Some(Ev::CallStart { function, .. }) if function == "(thunc.call)"));
    }

    #[test]
    fn targets() {
        let v: Value = serde_json::from_str(r#"{"pattern": "def main", "path": "src"}"#).unwrap();
        assert_eq!(target("search", &v), "'def main' in src");
        let v: Value = serde_json::from_str(r#"{"command": "pytest -q"}"#).unwrap();
        assert_eq!(target("run", &v), "pytest -q");
        assert_eq!(target("list", &Value::Null), "");
    }

    #[test]
    fn session_records_become_agent_events() {
        let mut r = SessionReader::new("repo-guide", "/x/s.jsonl");
        let start = r.line(r#"{"time": "2026-10-05T14:02:11+0000", "event": "start", "task": "tests_for", "returns": "list", "inputs": {"feature": "caching"}}"#);
        assert!(matches!(&start[..], [Ev::AgentStart { agent, task, inputs, .. }]
            if agent == "repo-guide" && task == "tests_for" && inputs == &vec![("feature".to_string(), "caching".to_string())]));
        let step = r.line(r#"{"time": "2026-10-05T14:02:14+0000", "event": "step", "n": 1, "tool": "run", "args": {"command": "pytest"}, "result": "exit 0"}"#);
        assert!(matches!(&step[..], [Ev::AgentStep { tool, target, seconds, .. }]
            if tool == "run" && target == "pytest" && *seconds == 3.0));
        let bad = r.line(r#"{"time": "2026-10-05T14:02:15+0000", "event": "step", "n": 2, "reply": "hm", "result": "error: no JSON"}"#);
        assert!(matches!(&bad[..], [Ev::AgentStep { tool, .. }] if tool == "reply"));
        let end = r.line(r#"{"time": "2026-10-05T14:02:16+0000", "event": "finish", "n": 3, "value": ["tests/a.py"], "files_changed": []}"#);
        assert!(
            matches!(&end[..], [Ev::AgentStep { tool, .. }, Ev::AgentEnd { ok: true, steps: 3, .. }] if tool == "finish")
        );
        assert!(r.ended);
    }
}
