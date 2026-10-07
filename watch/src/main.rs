//! thunc-watch: a live dashboard for thunc calls and agent runs, in the terminal.

mod app;
mod event;
mod fmt;
mod plain;
mod source;
mod state;
mod summary;
mod ui;

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};
use ratatui::crossterm::event::{self as term_event, Event as TermEvent};

use app::{App, Source};
use source::{AgentsDir, Launch, Program, Replay, Tail};
use state::State;

const USAGE: &str = "\
thunc-watch: a live view of a program's thunc calls and agent runs.

Usage:
  thunc-watch [OPTIONS] SCRIPT.py [ARGS...]     run a Python script and watch it
  thunc-watch [OPTIONS] -m MODULE [ARGS...]     run a module, as python -m does
  thunc-watch [OPTIONS] -- COMMAND [ARGS...]    run any command (pytest, uv run app.py, ...)
  thunc-watch --agents [DIR]                    follow agent runs in DIR, from any process
                                                (default: THUNC_AGENTS_DIR or ./.thunc_agents)
  thunc-watch --events FILE                     follow a file written with THUNC_EVENTS=FILE
  thunc-watch --replay FILE [--speed N]         play back an events file or an agent's session record

Options:
  --capture            send whole inputs and replies, not previews (THUNC_EVENTS_CAPTURE=1);
                       they may contain personal data
  --save-events FILE   keep the program's events in FILE (default: a temporary file, deleted at the end)
  --python PATH        the Python for SCRIPT.py and -m (default: THUNC_PYTHON, the active
                       virtualenv's, or python3)
  --plain              print one line per event instead of the dashboard
  --no-mouse           leave the mouse to the terminal, for selecting text (keys still work)
  --speed N            playback speed for --replay (default 1; 0 plays it all at once)
  -h, --help           show this
  -V, --version        show the version

In the dashboard, click a row to select it and click it again to open it, or use the arrow keys
and enter. esc goes back, ? lists every key, q quits, and Ctrl+C stops the program.";

#[derive(Debug, Default, PartialEq)]
struct Args {
    program: Vec<String>,
    agents: Option<PathBuf>,
    events: Option<PathBuf>,
    replay: Option<PathBuf>,
    speed: f64,
    dump: Option<(u16, u16)>,
    at: Option<f64>,
    capture: bool,
    save_events: Option<PathBuf>,
    python: Option<String>,
    plain: bool,
    no_mouse: bool,
}

enum Parsed {
    Run(Box<Args>), // boxed: Args is much larger than the other variants
    Help,
    Version,
}

fn default_python() -> String {
    if let Ok(p) = std::env::var("THUNC_PYTHON")
        && !p.is_empty()
    {
        return p;
    }
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let bin = if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" };
        let p = PathBuf::from(venv).join(bin);
        if p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
    if cfg!(windows) { "python".into() } else { "python3".into() }
}

fn parse(argv: &[String]) -> Result<Parsed> {
    let mut a = Args { speed: 1.0, ..Default::default() };
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> Result<String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
    };
    let mut module = false;
    while i < argv.len() {
        let arg = &argv[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help),
            "-V" | "--version" => return Ok(Parsed::Version),
            "--capture" => a.capture = true,
            "--plain" => a.plain = true,
            "--no-mouse" => a.no_mouse = true,
            "--save-events" => a.save_events = Some(value(&mut i, arg)?.into()),
            "--python" => a.python = Some(value(&mut i, arg)?),
            "--events" => a.events = Some(value(&mut i, arg)?.into()),
            "--replay" => a.replay = Some(value(&mut i, arg)?.into()),
            // Undocumented, for tests and bug reports: draw every screen once, as text, and exit.
            "--dump-screens" => {
                let v = value(&mut i, arg)?;
                let (w, h) = v.split_once('x').ok_or_else(|| anyhow::anyhow!("--dump-screens takes COLSxROWS"))?;
                a.dump = Some((w.parse()?, h.parse()?));
            }
            "--at" => a.at = Some(value(&mut i, arg)?.parse()?),
            "--speed" => {
                let v = value(&mut i, arg)?;
                a.speed = v.parse().map_err(|_| anyhow::anyhow!("--speed takes a number, like 2 or 0.5, not {v:?}"))?;
            }
            "--agents" => {
                let dir = match argv.get(i + 1) {
                    Some(d) if !d.starts_with('-') => {
                        i += 1;
                        PathBuf::from(d)
                    }
                    _ => {
                        std::env::var("THUNC_AGENTS_DIR").map(PathBuf::from).unwrap_or_else(|_| ".thunc_agents".into())
                    }
                };
                a.agents = Some(dir);
            }
            "-m" => {
                module = true;
                a.program = argv[i + 1..].to_vec();
                if a.program.is_empty() {
                    bail!("-m needs a module name");
                }
                break;
            }
            "--" => {
                a.program = argv[i + 1..].to_vec();
                if a.program.is_empty() {
                    bail!("-- needs a command to run");
                }
                break;
            }
            s if s.starts_with('-') => bail!("unknown option {s} (see thunc-watch --help)"),
            _ => {
                a.program = argv[i..].to_vec();
                break;
            }
        }
        i += 1;
    }
    let python = a.python.clone().unwrap_or_else(default_python);
    if module {
        a.program.splice(0..0, [python, "-m".into()]);
    } else if a.program.first().is_some_and(|p| p.ends_with(".py")) {
        a.program.insert(0, python);
    }
    let modes = [!a.program.is_empty(), a.agents.is_some(), a.events.is_some(), a.replay.is_some()];
    match modes.iter().filter(|m| **m).count() {
        0 => {
            bail!("nothing to watch: give a script to run, or --agents, --events or --replay (see thunc-watch --help)")
        }
        1 => Ok(Parsed::Run(Box::new(a))),
        _ => bail!("give one of: a program to run, --agents, --events or --replay"),
    }
}

fn main() {
    // Exit quietly when the reader of --plain output goes away (`thunc-watch --plain ... | head`),
    // as other command-line tools do, instead of panicking on the failed write.
    #[cfg(unix)]
    // SAFETY: restores the default action for SIGPIPE before any thread starts.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let code = match parse(&argv) {
        Ok(Parsed::Help) => {
            println!("{USAGE}");
            0
        }
        Ok(Parsed::Version) => {
            println!("thunc-watch {}", env!("CARGO_PKG_VERSION"));
            0
        }
        Ok(Parsed::Run(args)) => match run(*args) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("thunc-watch: {e:#}");
                2
            }
        },
        Err(e) => {
            eprintln!("thunc-watch: {e:#}");
            2
        }
    };
    std::process::exit(code);
}

fn run(args: Args) -> Result<i32> {
    let source = if !args.program.is_empty() {
        let launch = Launch {
            argv: args.program.clone(),
            capture: args.capture,
            save_events: args.save_events.clone(),
            plain: args.plain,
        };
        Source::Program(Program::start(&launch)?)
    } else if let Some(dir) = &args.agents {
        if !dir.is_dir() {
            eprintln!("thunc-watch: {} doesn't exist yet; waiting for an agent's first run there.", dir.display());
        }
        Source::Agents(AgentsDir::new(dir.clone()))
    } else if let Some(file) = &args.events {
        Source::Events(Tail::new(file))
    } else {
        Source::Replay(Replay::open(args.replay.as_ref().expect("checked in parse"), args.speed)?)
    };
    let mut app = App::new(source);
    if let Some((w, h)) = args.dump {
        match (&mut app.source, args.at) {
            (Source::Replay(r), Some(at)) => r.stop_at(at),
            (Source::Replay(r), None) => r.speed = 0.0,
            _ => {}
        }
        app.tick();
        for screen in [app::Screen::Overview, app::Screen::Agents, app::Screen::Calls, app::Screen::Summary] {
            app.screen = screen;
            println!("{}\n", ui::render_text(&app, w, h));
        }
        app.screen = app::Screen::Overview;
        app.dialog = Some(app::Dialog::Help);
        println!("{}", ui::render_text(&app, w, h));
        return Ok(0);
    }
    if args.plain || !std::io::stdout().is_terminal() {
        return Ok(plain_loop(&mut app));
    }
    app.mouse = !args.no_mouse;
    dashboard(&mut app)?;
    Ok(after_dashboard(&mut app))
}

/// The full-screen view, until q (or Ctrl+C with nothing running).
fn dashboard(app: &mut App) -> Result<()> {
    use ratatui::crossterm::event::{DisableMouseCapture, EnableMouseCapture};
    use ratatui::crossterm::execute;

    let mut terminal = ratatui::init(); // also restores the terminal if thunc-watch panics
    if app.mouse {
        execute!(std::io::stdout(), EnableMouseCapture)?;
        let restore = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = execute!(std::io::stdout(), DisableMouseCapture);
            restore(info);
        }));
    }
    let result = (|| -> Result<()> {
        while !app.quit {
            app.tick();
            let mut hits = Vec::new();
            terminal.draw(|f| hits = ui::draw(f, app))?;
            app.hits = hits;
            if term_event::poll(Duration::from_millis(100))? {
                while term_event::poll(Duration::ZERO)? {
                    match term_event::read()? {
                        TermEvent::Key(k) => app.key(k),
                        TermEvent::Mouse(m) => app.mouse(m),
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    })();
    if app.mouse {
        let _ = execute!(std::io::stdout(), DisableMouseCapture);
    }
    ratatui::restore();
    result
}

/// After the dashboard closes: the program's output, as it would have printed it, then the report.
fn after_dashboard(app: &mut App) -> i32 {
    let code = match &mut app.source {
        Source::Program(p) => {
            if p.running() {
                p.wait_stopped(5.0);
            }
            let o = p.output.lock().map(|o| (o.lines.clone(), o.dropped)).unwrap_or_default();
            if o.1 > 0 {
                eprintln!("[thunc-watch: the first {} lines of output are not shown]", o.1);
            }
            let (mut out, mut err) = (std::io::stdout().lock(), std::io::stderr().lock());
            for (is_err, line) in o.0 {
                let _ = if is_err { writeln!(err, "{line}") } else { writeln!(out, "{line}") };
            }
            p.exit_code()
        }
        _ => 0,
    };
    app.tick();
    let status = match app.program() {
        Some(p) => format!("  ·  the program {}", p.exit_text()),
        None => String::new(),
    };
    if !app.state.calls.is_empty() || !app.state.runs.is_empty() {
        eprintln!();
        for (_, line, _) in
            summary::lines(&app.state, &status, matches!(app.source, Source::Program(_) | Source::Replay(_)))
        {
            eprintln!("{line}");
        }
    }
    code
}

/// --plain, or output that isn't a terminal: one line per event.
fn plain_loop(app: &mut App) -> i32 {
    let mut state = State::new(source::unix_now());
    let mut printed_any = false;
    loop {
        let events = match &mut app.source {
            Source::Program(p) => p.poll(),
            Source::Events(t) => t.poll().iter().filter_map(|l| event::parse_event_line(l)).collect(),
            Source::Agents(a) => a.poll(),
            Source::Replay(r) => r.poll(),
        };
        // A program's own output goes to the terminal: events go to stderr so the two don't mix in a pipe.
        let to_stderr = matches!(app.source, Source::Program(_));
        for e in events {
            state.apply(e.clone());
            if let Some(line) = plain::line(&e, &state) {
                printed_any = true;
                if to_stderr { eprintln!("thunc  {line}") } else { println!("{line}") }
            }
        }
        state.now = source::unix_now();
        let done = match &app.source {
            Source::Program(p) => !p.running(),
            Source::Replay(r) => r.done(),
            _ => false,
        };
        if done {
            // One more read, for events written just before the program ended.
            if let Source::Program(p) = &mut app.source {
                for e in p.poll() {
                    state.apply(e.clone());
                    if let Some(line) = plain::line(&e, &state) {
                        eprintln!("thunc  {line}");
                    }
                }
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    if let Source::Replay(r) = &app.source {
        state.now = r.clock;
    }
    if printed_any || !state.calls.is_empty() {
        let status = app.program().map(|p| format!("  ·  the program {}", p.exit_text())).unwrap_or_default();
        eprintln!();
        for (_, line, _) in
            summary::lines(&state, &status, matches!(app.source, Source::Program(_) | Source::Replay(_)))
        {
            eprintln!("{line}");
        }
    }
    app.program().map(|p| p.exit_code()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Args {
        let v: Vec<String> = s.split_whitespace().map(String::from).collect();
        match parse(&v).unwrap() {
            Parsed::Run(a) => *a,
            _ => panic!("not a run"),
        }
    }

    #[test]
    fn programs() {
        let a = args("--python py app.py --flag x");
        assert_eq!(a.program, ["py", "app.py", "--flag", "x"]);
        let a = args("--capture --python py -m examples.hello -v");
        assert_eq!(a.program, ["py", "-m", "examples.hello", "-v"]);
        assert!(a.capture);
        let a = args("-- uv run app.py");
        assert_eq!(a.program, ["uv", "run", "app.py"]);
        let a = args("pytest -q");
        assert_eq!(a.program, ["pytest", "-q"]);
    }

    #[test]
    fn other_modes() {
        assert_eq!(args("--agents /x").agents, Some("/x".into()));
        assert!(args("--agents --plain").plain);
        assert_eq!(args("--replay e.jsonl --speed 0").speed, 0.0);
        assert_eq!(args("--events e.jsonl").events, Some("e.jsonl".into()));
    }

    #[test]
    fn mistakes_are_explained() {
        let err = |s: &str| match parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("{s} parsed"),
        };
        assert!(err("").contains("nothing to watch"));
        assert!(err("--agents x app.py").contains("give one of"));
        assert!(err("--bogus").contains("unknown option"));
        assert!(err("--speed fast --replay x").contains("--speed takes a number"));
    }
}
