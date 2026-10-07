//! Everything the dashboard knows, built up one event at a time.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::event::{Ev, Key};

const MAX_CALLS: usize = 4000; // finished calls kept for the call screen; totals keep counting
const MAX_LOG: usize = 500;
const RECENT: usize = 12;

#[derive(Debug, Clone)]
pub struct Attempt {
    pub n: u32,
    pub seconds: f64,
    pub ok: bool,
    pub problem: Option<String>,
    pub reply: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CallEnd {
    pub ok: bool,
    pub cached: bool,
    pub attempts: u32,
    pub seconds: f64,
    pub value: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Call {
    pub key: Key,
    pub function: String,
    pub backend: String,
    pub model: String,
    pub inputs: Vec<(String, String)>,
    pub started: f64,
    pub attempts: Vec<Attempt>,
    pub end: Option<CallEnd>,
}

impl Call {
    pub fn model_seconds(&self) -> f64 {
        self.attempts.iter().map(|a| a.seconds).sum()
    }
    pub fn retried(&self) -> bool {
        self.attempts.iter().any(|a| !a.ok)
    }
    pub fn failed(&self) -> bool {
        self.end.as_ref().is_some_and(|e| !e.ok)
    }
    pub fn seconds(&self, now: f64) -> f64 {
        match &self.end {
            Some(e) => e.seconds,
            None => now - self.started,
        }
    }
    /// The first input, for a one-line description: ticket="I was charged twice".
    pub fn input_line(&self) -> String {
        self.inputs.iter().map(|(k, v)| format!("{k}={v:?}")).collect::<Vec<_>>().join(", ")
    }
}

#[derive(Debug, Clone, Default)]
pub struct FnStats {
    pub name: String,
    pub calls: u32,
    pub cached: u32,
    pub retries: u32,
    pub failed: u32,
    pub times: Vec<f64>, // seconds of each call answered by the model
    pub total: f64,
    pub model: f64,
    pub recent: VecDeque<(f64, bool)>, // seconds, ok
}

impl FnStats {
    pub fn mean(&self) -> Option<f64> {
        (!self.times.is_empty()).then(|| self.times.iter().sum::<f64>() / self.times.len() as f64)
    }
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.times.is_empty() {
            return None;
        }
        let mut s = self.times.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let i = ((p / 100.0) * s.len() as f64).ceil() as usize;
        Some(s[i.clamp(1, s.len()) - 1])
    }
    pub fn max(&self) -> Option<f64> {
        self.times.iter().cloned().reduce(f64::max)
    }
}

#[derive(Debug, Clone)]
pub struct Step {
    pub n: u32,
    pub tool: String,
    pub target: String,
    pub result: String,
    pub seconds: f64,
    pub denied: bool,
}

#[derive(Debug, Clone)]
pub struct RunEnd {
    pub t: f64,
    pub ok: bool,
    pub steps: u32,
    pub files_changed: Vec<String>,
    pub value: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Run {
    pub key: Key,
    pub agent: String,
    pub task: String,
    pub returns: String,
    pub session: Option<String>,
    pub inputs: Vec<(String, String)>,
    pub started: f64,
    pub last_t: f64,
    pub steps: Vec<Step>,
    pub replies: u32,
    pub model: f64, // seconds waiting on the model (0 when read from a session record, which has no timings)
    pub tools: f64,
    pub on_model: bool, // waiting for the model's next reply (else carrying out a tool call)
    pub current: Option<(String, String)>, // the tool call in progress: tool, target
    pub end: Option<RunEnd>,
}

impl Run {
    pub fn running(&self) -> bool {
        self.end.is_none()
    }
    pub fn seconds(&self, now: f64) -> f64 {
        match &self.end {
            Some(e) => e.t - self.started,
            None => now - self.started,
        }
    }
    /// Seconds waiting on the model: measured when the events say, else the time not spent in tools.
    /// Whether model and tool time were measured: true for events, false for runs read from a
    /// session record, whose lines say only when each step was written.
    pub fn measured(&self) -> bool {
        self.replies > 0
    }
    /// Steps so far, counted as thunc counts them (model replies), once the run has ended.
    pub fn step_count(&self) -> usize {
        self.end.as_ref().map_or(self.steps.len(), |e| e.steps as usize)
    }
    pub fn denied(&self) -> impl Iterator<Item = &Step> {
        self.steps.iter().filter(|s| s.denied)
    }
    pub fn call_line(&self) -> String {
        let args = self.inputs.iter().map(|(k, v)| format!("{k}={v:?}")).collect::<Vec<_>>().join(", ");
        format!("{}({args})", short_name(&self.task))
    }
}

/// A Python qualname without the parts nobody reads: "test_x.<locals>.urgency" -> "urgency".
pub fn short_name(name: &str) -> &str {
    name.rsplit("<locals>.").next().unwrap_or(name)
}

/// Something a row on screen leads to when it's opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Link {
    Function(String),
    Call(Key),
    Run(Key),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogKind {
    Ok,
    Cached,
    Retry,
    Fail,
    Start,
    Step,
    Denied,
    RunOk,
    RunFail,
}

#[derive(Debug, Clone)]
pub struct LogLine {
    pub link: Option<Link>,
    pub t: f64,
    pub kind: LogKind,
    pub who: String,
    pub text: String,
    pub seconds: Option<f64>,
}

#[derive(Debug, Default)]
pub struct State {
    pub calls: Vec<Call>,
    call_index: HashMap<Key, usize>,
    pub functions: Vec<FnStats>,
    fn_index: HashMap<String, usize>,
    pub runs: Vec<Run>,
    run_index: HashMap<Key, usize>,
    pub log: VecDeque<LogLine>,
    pub activity: BTreeMap<i64, u32>, // results per second
    pub first_t: Option<f64>,
    pub last_t: f64,
    pub now: f64,
    pub dropped_calls: usize,
}

impl State {
    pub fn new(now: f64) -> Self {
        State { now, ..Default::default() }
    }

    fn call(&mut self, key: &str, t: f64) -> &mut Call {
        let i = match self.call_index.get(key) {
            Some(&i) => i,
            None => {
                self.calls.push(Call {
                    key: key.to_string(),
                    function: "(thunc.call)".into(),
                    backend: String::new(),
                    model: String::new(),
                    inputs: vec![],
                    started: t,
                    attempts: vec![],
                    end: None,
                });
                self.call_index.insert(key.to_string(), self.calls.len() - 1);
                self.calls.len() - 1
            }
        };
        &mut self.calls[i]
    }

    fn run(&mut self, key: &str, t: f64) -> &mut Run {
        let i = match self.run_index.get(key) {
            Some(&i) => i,
            None => {
                self.runs.push(Run {
                    key: key.to_string(),
                    agent: "agent".into(),
                    task: String::new(),
                    returns: String::new(),
                    session: None,
                    inputs: vec![],
                    started: t,
                    last_t: t,
                    steps: vec![],
                    replies: 0,
                    model: 0.0,
                    tools: 0.0,
                    on_model: true,
                    current: None,
                    end: None,
                });
                self.run_index.insert(key.to_string(), self.runs.len() - 1);
                self.runs.len() - 1
            }
        };
        &mut self.runs[i]
    }

    fn stats(&mut self, function: &str) -> &mut FnStats {
        let i = match self.fn_index.get(function) {
            Some(&i) => i,
            None => {
                self.functions.push(FnStats { name: function.to_string(), ..Default::default() });
                self.fn_index.insert(function.to_string(), self.functions.len() - 1);
                self.functions.len() - 1
            }
        };
        &mut self.functions[i]
    }

    fn log(&mut self, link: Link, t: f64, kind: LogKind, who: &str, text: String, seconds: Option<f64>) {
        self.log.push_back(LogLine { link: Some(link), t, kind, who: who.to_string(), text, seconds });
        while self.log.len() > MAX_LOG {
            self.log.pop_front();
        }
    }

    pub fn apply(&mut self, ev: Ev) {
        let t = ev.t();
        if t > 0.0 {
            self.first_t = Some(self.first_t.map_or(t, |f| f.min(t)));
            self.last_t = self.last_t.max(t);
        }
        match ev {
            Ev::CallStart { key, t, function, backend, model, inputs } => {
                let c = self.call(&key, t);
                c.started = t;
                c.function = function.clone();
                c.backend = backend;
                c.model = model;
                c.inputs = inputs;
                self.stats(&function);
            }
            Ev::CallAttempt { key, t, n, seconds, ok, problem, reply } => {
                let c = self.call(&key, t);
                c.attempts.push(Attempt { n, seconds, ok, problem: problem.clone(), reply });
                let who = c.function.clone();
                if !ok {
                    let why = problem.unwrap_or_else(|| "rejected".into());
                    self.log(
                        Link::Call(key.clone()),
                        t,
                        LogKind::Retry,
                        &who,
                        format!("attempt {n}: {why}"),
                        Some(seconds),
                    );
                }
            }
            Ev::CallEnd { key, t, ok, cached, attempts, seconds, value, error } => {
                let c = self.call(&key, t);
                c.end = Some(CallEnd { ok, cached, attempts, seconds, value: value.clone(), error: error.clone() });
                let (function, model) = (c.function.clone(), c.model_seconds());
                let s = self.stats(&function);
                s.calls += 1;
                s.total += seconds;
                s.model += model;
                if cached {
                    s.cached += 1;
                } else {
                    s.times.push(seconds);
                    s.retries += attempts.saturating_sub(1);
                }
                if !ok {
                    s.failed += 1;
                }
                s.recent.push_back((seconds, ok));
                while s.recent.len() > RECENT {
                    s.recent.pop_front();
                }
                *self.activity.entry(t.floor() as i64).or_default() += 1;
                let (kind, text) = match (ok, cached) {
                    (false, _) => (LogKind::Fail, error.unwrap_or_else(|| "failed".into())),
                    (true, true) => (LogKind::Cached, format!("→ {}", value.unwrap_or_default())),
                    (true, false) => (LogKind::Ok, format!("→ {}", value.unwrap_or_default())),
                };
                self.log(Link::Call(key.clone()), t, kind, &function, text, (!cached).then_some(seconds));
                self.trim_calls();
            }
            Ev::AgentStart { key, t, agent, task, returns, session, inputs } => {
                let r = self.run(&key, t);
                r.started = t;
                r.last_t = t;
                r.agent = agent.clone();
                r.task = task;
                r.returns = returns;
                r.session = session;
                r.inputs = inputs;
                let line = r.call_line();
                self.log(Link::Run(key.clone()), t, LogKind::Start, &agent, format!("started {line}"), None);
            }
            Ev::AgentReply { key, t, n, seconds } => {
                let r = self.run(&key, t);
                r.replies = r.replies.max(n);
                r.model += seconds;
                r.last_t = t;
                r.on_model = false;
            }
            Ev::AgentTool { key, t, tool, target, .. } => {
                let r = self.run(&key, t);
                r.current = Some((tool, target));
                r.on_model = false;
                r.last_t = t;
            }
            Ev::AgentStep { key, t, n, tool, target, result, seconds, denied } => {
                let r = self.run(&key, t);
                r.current = None;
                r.steps.push(Step { n, tool: tool.clone(), target: target.clone(), result, seconds, denied });
                if r.measured() {
                    r.tools += seconds;
                }
                r.last_t = t;
                r.on_model = true;
                let agent = r.agent.clone();
                let (kind, text) = if denied {
                    (LogKind::Denied, format!("{tool} {target}: denied"))
                } else {
                    (LogKind::Step, format!("{tool} {target}"))
                };
                self.log(Link::Run(key.clone()), t, kind, &agent, text, Some(seconds));
            }
            Ev::AgentEnd { key, t, ok, steps, seconds: _, files_changed, value, error } => {
                let r = self.run(&key, t);
                let steps = steps.max(r.steps.len() as u32);
                r.current = None;
                r.end = Some(RunEnd { t, ok, steps, files_changed, value: value.clone(), error: error.clone() });
                r.last_t = t;
                let (agent, secs) = (r.agent.clone(), t - r.started);
                if ok {
                    let v = value.unwrap_or_default();
                    self.log(
                        Link::Run(key.clone()),
                        t,
                        LogKind::RunOk,
                        &agent,
                        format!("finished in {steps} steps → {v}"),
                        Some(secs),
                    );
                } else {
                    let e = error.unwrap_or_else(|| "failed".into());
                    self.log(Link::Run(key.clone()), t, LogKind::RunFail, &agent, format!("failed: {e}"), Some(secs));
                }
            }
        }
    }

    fn trim_calls(&mut self) {
        if self.calls.len() <= MAX_CALLS {
            return;
        }
        let excess = self.calls.len() - MAX_CALLS * 3 / 4;
        let mut removed = 0;
        self.calls.retain(|c| {
            if removed < excess && c.end.is_some() {
                removed += 1;
                false
            } else {
                true
            }
        });
        self.dropped_calls += removed;
        self.call_index = self.calls.iter().enumerate().map(|(i, c)| (c.key.clone(), i)).collect();
    }

    pub fn in_flight(&self) -> impl Iterator<Item = &Call> {
        self.calls.iter().filter(|c| c.end.is_none())
    }

    pub fn fn_stats(&self, function: &str) -> Option<&FnStats> {
        self.fn_index.get(function).map(|&i| &self.functions[i])
    }

    /// Wall time covered so far: from the first event to now.
    pub fn wall(&self) -> f64 {
        self.first_t.map_or(0.0, |f| (self.now.max(self.last_t)) - f)
    }

    /// Results per second over the last `n` seconds, oldest first.
    pub fn activity_series(&self, n: usize) -> Vec<f64> {
        let end = self.now.floor() as i64;
        (0..n as i64).rev().map(|back| *self.activity.get(&(end - back)).unwrap_or(&0) as f64).collect()
    }

    /// Seconds in which at least one call or run was active, and the sum of their durations: their
    /// ratio is how much they overlapped (thunc.map, threads).
    pub fn busy(&self) -> (f64, f64) {
        let mut spans: Vec<(f64, f64)> = self
            .calls
            .iter()
            .map(|c| (c.started, c.started + c.seconds(self.now)))
            .chain(self.runs.iter().map(|r| (r.started, r.started + r.seconds(self.now))))
            .collect();
        let summed = spans.iter().map(|(a, b)| b - a).sum();
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut busy, mut cur): (f64, Option<(f64, f64)>) = (0.0, None);
        for (a, b) in spans {
            cur = match cur {
                Some((s, e)) if a <= e => Some((s, e.max(b))),
                Some((s, e)) => {
                    busy += e - s;
                    Some((a, b))
                }
                None => Some((a, b)),
            };
        }
        if let Some((s, e)) = cur {
            busy += e - s;
        }
        (busy, summed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::parse_event_line;

    fn feed(state: &mut State, lines: &[&str]) {
        for l in lines {
            state.apply(parse_event_line(l).expect(l));
        }
    }

    #[test]
    fn calls_add_up_per_function() {
        let mut s = State::new(0.0);
        feed(
            &mut s,
            &[
                r#"{"v":1,"event":"call.start","t":10,"pid":1,"id":1,"function":"urgency","backend":"fake","model":"m","inputs":{"ticket":"x"}}"#,
                r#"{"v":1,"event":"call.attempt","t":12,"pid":1,"id":1,"n":1,"seconds":2,"ok":false,"problem":"ensure rejected 7"}"#,
                r#"{"v":1,"event":"call.attempt","t":13,"pid":1,"id":1,"n":2,"seconds":1,"ok":true}"#,
                r#"{"v":1,"event":"call.end","t":13,"pid":1,"id":1,"ok":true,"cached":false,"attempts":2,"seconds":3,"value":"4"}"#,
                r#"{"v":1,"event":"call.start","t":11,"pid":1,"id":2,"function":"urgency","inputs":{}}"#,
                r#"{"v":1,"event":"call.end","t":11,"pid":1,"id":2,"ok":true,"cached":true,"attempts":0,"seconds":0.01,"value":"2"}"#,
                r#"{"v":1,"event":"call.start","t":12,"pid":1,"id":3,"function":"urgency","inputs":{}}"#,
            ],
        );
        let f = s.fn_stats("urgency").unwrap();
        assert_eq!((f.calls, f.cached, f.retries, f.failed), (2, 1, 1, 0));
        assert_eq!(f.times, vec![3.0]);
        assert_eq!(f.model, 3.0);
        assert_eq!(s.in_flight().count(), 1);
        assert_eq!(s.log.iter().map(|l| l.kind).collect::<Vec<_>>(), [LogKind::Retry, LogKind::Ok, LogKind::Cached]);
        assert_eq!(s.wall(), 3.0);
    }

    #[test]
    fn an_end_without_a_start_still_counts() {
        let mut s = State::new(0.0);
        feed(
            &mut s,
            &[
                r#"{"v":1,"event":"call.end","t":5,"pid":1,"id":9,"ok":false,"attempts":3,"seconds":6,"error":"No valid int"}"#,
            ],
        );
        assert_eq!(s.fn_stats("(thunc.call)").unwrap().failed, 1);
    }

    #[test]
    fn agent_runs_track_steps_and_time() {
        let mut s = State::new(30.0);
        feed(
            &mut s,
            &[
                r#"{"v":1,"event":"agent.start","t":10,"pid":1,"id":4,"agent":"repo-guide","task":"tests_for","inputs":{"feature":"caching"}}"#,
                r#"{"v":1,"event":"agent.reply","t":12,"pid":1,"id":4,"n":1,"seconds":2}"#,
                r#"{"v":1,"event":"agent.step","t":15,"pid":1,"id":4,"n":1,"tool":"run","target":"pytest","result":"exit 0","seconds":3}"#,
                r#"{"v":1,"event":"agent.reply","t":16,"pid":1,"id":4,"n":2,"seconds":1}"#,
                r#"{"v":1,"event":"agent.step","t":16,"pid":1,"id":4,"n":2,"tool":"edit","target":"README.md","result":"denied","seconds":0,"denied":true}"#,
            ],
        );
        let r = &s.runs[0];
        assert!(r.running());
        assert_eq!((r.model, r.tools, r.steps.len(), r.denied().count()), (3.0, 3.0, 2, 1));
        assert_eq!(r.call_line(), r#"tests_for(feature="caching")"#);
        feed(&mut s, &[r#"{"v":1,"event":"agent.end","t":20,"pid":1,"id":4,"ok":true,"steps":3,"value":"[]"}"#]);
        assert!(!s.runs[0].running());
        assert_eq!(s.runs[0].seconds(99.0), 10.0);
    }

    #[test]
    fn a_tool_call_in_progress_is_shown_until_its_step_arrives() {
        let mut s = State::new(0.0);
        feed(
            &mut s,
            &[
                r#"{"v":1,"event":"agent.start","t":1,"pid":1,"id":1,"agent":"a","task":"t"}"#,
                r#"{"v":1,"event":"agent.reply","t":2,"pid":1,"id":1,"n":1,"seconds":1}"#,
                r#"{"v":1,"event":"agent.tool","t":2,"pid":1,"id":1,"n":1,"tool":"run","target":"pytest"}"#,
            ],
        );
        assert_eq!(s.runs[0].current, Some(("run".into(), "pytest".into())));
        assert!(!s.runs[0].on_model);
        feed(
            &mut s,
            &[r#"{"v":1,"event":"agent.step","t":9,"pid":1,"id":1,"n":1,"tool":"run","target":"pytest","seconds":7}"#],
        );
        assert_eq!(s.runs[0].current, None);
        assert!(s.runs[0].on_model);
    }

    #[test]
    fn overlap() {
        let mut s = State::new(10.0);
        for (id, a, b) in [(1, 0.0, 4.0), (2, 1.0, 5.0), (3, 8.0, 9.0)] {
            feed(
                &mut s,
                &[
                    &format!(r#"{{"v":1,"event":"call.start","t":{a},"pid":1,"id":{id},"function":"f"}}"#),
                    &format!(
                        r#"{{"v":1,"event":"call.end","t":{b},"pid":1,"id":{id},"ok":true,"attempts":1,"seconds":{}}}"#,
                        b - a
                    ),
                ],
            );
        }
        assert_eq!(s.busy(), (6.0, 9.0));
    }

    #[test]
    fn names() {
        assert_eq!(short_name("test_x.<locals>.urgency"), "urgency");
        assert_eq!(short_name("urgency"), "urgency");
    }
}
