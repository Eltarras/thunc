//! Where events come from: a program run by thunc-watch, an events file another process writes,
//! an agents folder, or a saved file played back.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::event::{Ev, SessionReader, parse_event_line};

pub fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Follows a file that grows: each poll returns the complete lines added since the last one. A line
/// still being written is held until its newline arrives. A file that shrinks is read from the start.
pub struct Tail {
    pub path: PathBuf,
    pos: u64,
    partial: Vec<u8>,
}

impl Tail {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Tail { path: path.into(), pos: 0, partial: Vec::new() }
    }

    pub fn poll(&mut self) -> Vec<String> {
        let Ok(mut f) = File::open(&self.path) else {
            return vec![];
        };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < self.pos {
            self.pos = 0;
            self.partial.clear();
        }
        if len == self.pos || f.seek(SeekFrom::Start(self.pos)).is_err() {
            return vec![];
        }
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_err() {
            return vec![];
        }
        self.pos += buf.len() as u64;
        self.partial.extend_from_slice(&buf);
        let mut lines = Vec::new();
        while let Some(i) = self.partial.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.partial.drain(..=i).collect();
            let text = String::from_utf8_lossy(&line[..line.len() - 1]).trim_end_matches('\r').to_string();
            if !text.trim().is_empty() {
                lines.push(text);
            }
        }
        lines
    }
}

/// Program output, kept for the output screen and printed again when the dashboard closes.
#[derive(Default)]
pub struct Output {
    pub lines: VecDeque<(bool, String)>, // (from stderr, text)
    pub dropped: usize,
}

const MAX_OUTPUT: usize = 20_000;

impl Output {
    fn push(&mut self, err: bool, line: String) {
        self.lines.push_back((err, line));
        if self.lines.len() > MAX_OUTPUT {
            self.lines.pop_front();
            self.dropped += 1;
        }
    }
}

/// A program run by thunc-watch, with THUNC_EVENTS pointing at a file only it writes.
pub struct Program {
    child: Child,
    pub events: Tail,
    keep_events: bool,
    pub output: Arc<Mutex<Output>>,
    pub status: Option<ExitStatus>,
    pub started: f64,
    stops: u8,
    pub description: String,
}

pub struct Launch {
    pub argv: Vec<String>,
    pub capture: bool,
    pub save_events: Option<PathBuf>,
    pub plain: bool,
}

impl Program {
    pub fn start(launch: &Launch) -> Result<Self> {
        let (events_path, keep_events) = match &launch.save_events {
            Some(p) => (p.clone(), true),
            None => {
                let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
                let name = format!("thunc-watch-{}-{nanos}.jsonl", std::process::id());
                (std::env::temp_dir().join(name), false)
            }
        };
        // Start from an empty file, so events from an earlier run with the same --save-events path don't mix in.
        File::create(&events_path).with_context(|| format!("can't write the events file {}", events_path.display()))?;
        let events_abs = std::fs::canonicalize(&events_path).unwrap_or(events_path.clone());

        let mut cmd = Command::new(&launch.argv[0]);
        cmd.args(&launch.argv[1..]).env("THUNC_EVENTS", &events_abs).env("PYTHONUNBUFFERED", "1");
        if launch.capture {
            cmd.env("THUNC_EVENTS_CAPTURE", "1");
        }
        let output = Arc::new(Mutex::new(Output::default()));
        if !launch.plain {
            // The dashboard owns the terminal: the program gets no input and its output is collected.
            cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                cmd.process_group(0); // its own group, so a stop reaches the backends it started too
            }
        }
        let mut child = cmd.spawn().with_context(|| format!("can't start {}", launch.argv[0]))?;
        for (err, pipe) in [
            (false, child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>)),
            (true, child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>)),
        ] {
            if let Some(pipe) = pipe {
                let out = Arc::clone(&output);
                std::thread::spawn(move || {
                    for line in BufReader::new(pipe).split(b'\n') {
                        let Ok(line) = line else { break };
                        let text = String::from_utf8_lossy(&line).trim_end_matches('\r').to_string();
                        if let Ok(mut o) = out.lock() {
                            o.push(err, text);
                        }
                    }
                });
            }
        }
        Ok(Program {
            child,
            events: Tail::new(events_abs),
            keep_events,
            output,
            status: None,
            started: unix_now(),
            stops: 0,
            description: launch.argv.join(" "),
        })
    }

    pub fn poll(&mut self) -> Vec<Ev> {
        if self.status.is_none()
            && let Ok(Some(status)) = self.child.try_wait()
        {
            self.status = Some(status);
        }
        self.events.poll().iter().filter_map(|l| parse_event_line(l)).collect()
    }

    pub fn running(&self) -> bool {
        self.status.is_none()
    }

    /// Ask the program to stop, as Ctrl+C would. Asking again is more forceful.
    pub fn stop(&mut self) {
        if !self.running() {
            return;
        }
        self.stops += 1;
        #[cfg(unix)]
        {
            let pid = self.child.id() as libc::pid_t;
            let signal = match self.stops {
                1 => libc::SIGINT,
                2 => libc::SIGTERM,
                _ => libc::SIGKILL,
            };
            // SAFETY: kill only sends a signal; -pid is the program's own process group.
            unsafe {
                libc::kill(-pid, signal);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.kill();
        }
    }

    /// Wait up to `seconds` for the program to end after a stop; force it after that.
    pub fn wait_stopped(&mut self, seconds: f64) {
        let until = Instant::now() + std::time::Duration::from_secs_f64(seconds);
        while self.running() {
            if let Ok(Some(status)) = self.child.try_wait() {
                self.status = Some(status);
                break;
            }
            if Instant::now() >= until {
                self.stops = self.stops.max(2);
                self.stop();
                self.status = self.child.wait().ok();
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self.status {
            Some(s) => s.code().unwrap_or(1),
            None => 1,
        }
    }

    pub fn exit_text(&self) -> String {
        match self.status {
            None => "running".into(),
            Some(s) => match s.code() {
                Some(c) => format!("exited {c}"),
                None => "stopped by a signal".into(),
            },
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if !self.keep_events {
            let _ = std::fs::remove_file(&self.events.path);
        }
    }
}

/// Follows every agent's run records in an agents folder (.thunc_agents by default).
pub struct AgentsDir {
    pub root: PathBuf,
    sessions: HashMap<PathBuf, Watched>,
    last_scan: Option<Instant>,
    pub agents: usize,
}

struct Watched {
    tail: Tail,
    reader: SessionReader,
    agent_dir: PathBuf,
    dead_checks: u8,
}

const SESSIONS_PER_AGENT: usize = 30; // older records are left out when the dashboard opens

impl AgentsDir {
    pub fn new(root: PathBuf) -> Self {
        AgentsDir { root, sessions: HashMap::new(), last_scan: None, agents: 0 }
    }

    pub fn poll(&mut self) -> Vec<Ev> {
        let mut out = Vec::new();
        if self.last_scan.is_none_or(|t| t.elapsed().as_millis() >= 1000) {
            self.scan(&mut out);
            self.last_scan = Some(Instant::now());
        }
        let mut open: Vec<&PathBuf> = self.sessions.iter().filter(|(_, w)| !w.reader.ended).map(|(p, _)| p).collect();
        open.sort();
        let open: Vec<PathBuf> = open.into_iter().cloned().collect();
        for path in &open {
            let w = self.sessions.get_mut(path).expect("listed above");
            for line in w.tail.poll() {
                out.extend(w.reader.line(&line));
            }
        }
        self.check_alive(&open, &mut out);
        out
    }

    fn scan(&mut self, out: &mut Vec<Ev>) {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return;
        };
        let mut agents = 0;
        let mut found: Vec<(PathBuf, PathBuf, String)> = Vec::new();
        for entry in entries.flatten() {
            let dir = entry.path();
            let sessions = dir.join("sessions");
            if !sessions.is_dir() {
                continue;
            }
            agents += 1;
            let name = agent_name(&dir);
            let Ok(files) = std::fs::read_dir(&sessions) else {
                continue;
            };
            let mut files: Vec<PathBuf> =
                files.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "jsonl")).collect();
            files.sort(); // the names start with the time the run started
            let skip = files.len().saturating_sub(SESSIONS_PER_AGENT);
            for f in files.into_iter().skip(skip) {
                if !self.sessions.contains_key(&f) {
                    found.push((f, dir.clone(), name.clone()));
                }
            }
        }
        self.agents = agents;
        found.sort();
        for (path, agent_dir, name) in found {
            let mut w = Watched {
                tail: Tail::new(&path),
                reader: SessionReader::new(&name, &path.to_string_lossy()),
                agent_dir,
                dead_checks: 0,
            };
            for line in w.tail.poll() {
                out.extend(w.reader.line(&line));
            }
            self.sessions.insert(path, w);
        }
    }

    /// A record without an end belongs to a run in progress only if it's its agent's newest and the
    /// process holding the agent's lock is alive; any other is a run whose process ended mid-way.
    fn check_alive(&mut self, open: &[PathBuf], out: &mut Vec<Ev>) {
        let mut newest: HashMap<PathBuf, PathBuf> = HashMap::new();
        for (path, w) in &self.sessions {
            let e = newest.entry(w.agent_dir.clone()).or_insert_with(|| path.clone());
            if path > e {
                *e = path.clone();
            }
        }
        let now = unix_now();
        for path in open {
            let w = self.sessions.get_mut(path).expect("listed above");
            if w.reader.ended || !w.reader.started {
                continue;
            }
            let alive = newest.get(&w.agent_dir) == Some(path) && lock_holder_alive(&w.agent_dir, path);
            w.dead_checks = if alive { 0 } else { w.dead_checks + 1 };
            if w.dead_checks >= 2 {
                out.push(w.reader.interrupted(now));
            }
        }
    }
}

fn agent_name(dir: &Path) -> String {
    let folder = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    std::fs::read_to_string(dir.join("agent.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .unwrap_or(folder)
}

#[cfg(unix)]
fn lock_holder_alive(agent_dir: &Path, _session: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(agent_dir.join(".lock")) else {
        return false;
    };
    let Ok(pid) = text.trim_matches(char::from(0)).trim().parse::<libc::pid_t>() else {
        return false;
    };
    // SAFETY: signal 0 only checks that the process exists.
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn lock_holder_alive(_agent_dir: &Path, session: &Path) -> bool {
    // No process check here yet: a record that changed in the last ten minutes counts as running.
    std::fs::metadata(session)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|m| m.elapsed().ok())
        .is_some_and(|age| age.as_secs() < 600)
}

/// A file played back: an events file, or one agent's session record.
pub struct Replay {
    events: VecDeque<Ev>,
    pub speed: f64,
    origin: Option<(f64, Instant)>, // first event's time, when playback started
    pub clock: f64,
    pub total: usize,
    held_at: Option<f64>,
}

impl Replay {
    pub fn open(path: &Path, speed: f64) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("can't read {}", path.display()))?;
        let mut events: Vec<Ev> = text.lines().filter_map(parse_event_line).collect();
        if events.is_empty() {
            // Not an events file: try it as a session record.
            let agent = path.parent().and_then(Path::parent).map(agent_name).unwrap_or_else(|| "agent".into());
            let mut reader = SessionReader::new(&agent, &path.to_string_lossy());
            events = text.lines().flat_map(|l| reader.line(l)).collect();
        }
        anyhow::ensure!(!events.is_empty(), "{} has no thunc events or session records", path.display());
        events.sort_by(|a, b| a.t().total_cmp(&b.t()));
        let total = events.len();
        let clock = events[0].t();
        Ok(Replay { events: events.into(), speed, origin: None, clock, total, held_at: None })
    }

    pub fn poll(&mut self) -> Vec<Ev> {
        let Some(first) = self.events.front().map(Ev::t) else {
            return vec![];
        };
        let (t0, started) = *self.origin.get_or_insert((first, Instant::now()));
        self.clock = match self.held_at {
            Some(at) => at,
            None if self.speed <= 0.0 => f64::INFINITY,
            None => t0 + started.elapsed().as_secs_f64() * self.speed,
        };
        let mut out = Vec::new();
        while self.events.front().is_some_and(|e| e.t() <= self.clock) {
            out.extend(self.events.pop_front());
        }
        if self.held_at.is_none() && (self.speed <= 0.0 || self.events.is_empty()) {
            self.clock = out.last().map(Ev::t).unwrap_or(self.clock);
        }
        out
    }

    /// Play up to `seconds` after the first event and hold there (for --dump-screens --at).
    pub fn stop_at(&mut self, seconds: f64) {
        let t0 = self.events.front().map(Ev::t).unwrap_or(0.0);
        self.held_at = Some(t0 + seconds);
    }

    pub fn done(&self) -> bool {
        self.events.is_empty()
    }

    pub fn played(&self) -> usize {
        self.total - self.events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("thunc-watch-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tail_holds_half_written_lines() {
        let path = temp("tail").join("e.jsonl");
        let mut f = File::create(&path).unwrap();
        let mut tail = Tail::new(&path);
        write!(f, "one\ntw").unwrap();
        assert_eq!(tail.poll(), vec!["one"]);
        write!(f, "o\r\n\nthree\n").unwrap();
        assert_eq!(tail.poll(), vec!["two", "three"]);
        assert!(tail.poll().is_empty());
        std::fs::write(&path, "new\n").unwrap(); // replaced by a shorter file
        assert_eq!(tail.poll(), vec!["new"]);
    }

    #[test]
    fn agents_dir_reads_records_and_spots_dead_runs() {
        let root = temp("agents");
        let sessions = root.join("repo-guide").join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(root.join("repo-guide").join("agent.json"), r#"{"name": "Repo guide"}"#).unwrap();
        std::fs::write(
            sessions.join("2026-10-05T10-00-00.000000Z-a.jsonl"),
            "{\"time\": \"2026-10-05T10:00:00+0000\", \"event\": \"start\", \"task\": \"a\"}\n\
             {\"time\": \"2026-10-05T10:00:02+0000\", \"event\": \"finish\", \"n\": 1, \"value\": 1}\n",
        )
        .unwrap();
        // A run with no end, and a lock naming a process that doesn't exist.
        std::fs::write(
            sessions.join("2026-10-05T11-00-00.000000Z-b.jsonl"),
            "{\"time\": \"2026-10-05T11:00:00+0000\", \"event\": \"start\", \"task\": \"b\"}\n",
        )
        .unwrap();
        std::fs::write(root.join("repo-guide").join(".lock"), "999999999").unwrap();

        let mut dir = AgentsDir::new(root);
        let first = dir.poll();
        assert_eq!(dir.agents, 1);
        assert!(matches!(&first[0], Ev::AgentStart { agent, task, .. } if agent == "Repo guide" && task == "a"));
        assert!(first.iter().any(|e| matches!(e, Ev::AgentEnd { ok: true, .. })));
        assert!(!first.iter().any(|e| matches!(e, Ev::AgentEnd { ok: false, .. })), "one dead check isn't enough");
        let second = dir.poll();
        assert!(matches!(&second[..], [Ev::AgentEnd { ok: false, error: Some(e), .. }] if e.contains("process ended")));
        assert!(dir.poll().is_empty());
    }

    #[test]
    fn replay_plays_in_time_order() {
        let path = temp("replay").join("e.jsonl");
        std::fs::write(
            &path,
            r#"{"v":1,"event":"call.end","t":12,"pid":1,"id":1,"ok":true,"attempts":1,"seconds":2}
{"v":1,"event":"call.start","t":10,"pid":1,"id":1,"function":"f"}
"#,
        )
        .unwrap();
        let mut r = Replay::open(&path, 0.0).unwrap();
        let all = r.poll();
        assert_eq!(all.len(), 2);
        assert!(matches!(all[0], Ev::CallStart { .. }));
        assert!(r.done());
    }
}
