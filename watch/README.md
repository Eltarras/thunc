# thunc-watch

A live dashboard in the terminal for a program's thunc calls and agent runs. It shows:

- calls in flight
- retries, and why each reply was rejected
- timings for each function
- each agent's steps as they happen
- the `--profile` report when the program ends

It's a compiled binary in its own package, so `thunc` itself stays pure Python with no
dependencies. Install it as an extra:

```bash
pip install "thunc[watch]"
```

Then run your program through it:

```bash
thunc watch app.py                 # run a script and watch it
thunc watch -m examples.hello      # a module, as python -m does
thunc watch -- uv run app.py       # any command
thunc watch --agents               # agent runs in ./.thunc_agents, from any process
thunc watch --events events.jsonl  # a file another process writes with THUNC_EVENTS=events.jsonl
thunc watch --replay events.jsonl  # play back a saved events file, or one agent's session record
thunc watch --plain app.py         # one line per event, for CI logs and pipes
```

`thunc watch` runs scripts with the Python that thunc is installed in. The binary is also on your
`PATH` as `thunc-watch`, which takes the same arguments. `thunc watch --help` lists the options.

## Using it

Everything works with the mouse or the keyboard. Click a tab, or press `1`–`5` or `←` `→`. Click a
row to select it (or use `↑` `↓`), and click it again (or press `⏎`) to open it. `esc` goes back to
where you opened it from. `?` shows every key.

| Screen | What's on it | Opening a row shows |
|---|---|---|
| **Overview** | calls in flight, a table per function, running agents, results per second, recent events | a function's calls, one call, or an agent run |
| **Agents** | every run, and the selected run's steps, time split, changed files and denials | |
| **Calls** | recent calls, and each attempt of the selected one: the reply and why it was rejected | |
| **Summary** | the same tables as `thunc run --profile`. It opens by itself when the program ends | a function's calls, or an agent run |
| **Output** | what the program printed (while it runs, the dashboard owns the terminal) | |

The buttons at the bottom also have keys:

- **p Pause** freezes the display. The program keeps running, and its events wait until you resume.
- **f Failures** shows only retries, failures and denials. Filters show as tags at the top right;
  click a tag's ✕ to clear it.
- **? Help** lists every key.
- **q Quit** asks first if the program is still running, because quitting stops it.

`Ctrl+C` stops the program, as it would in its own terminal.

When the dashboard closes, it prints the program's output and then the report. It exits with the
program's exit code.

While the dashboard has the mouse, select text by holding Shift (Option on macOS) as you drag.
Start it with `--no-mouse` to leave the mouse to the terminal.

## Which agents it sees

| Mode | Agents shown |
|---|---|
| `thunc watch app.py` | the program's own agent runs, from its events |
| `thunc watch --agents [DIR]` | runs recorded in `DIR`, by default `THUNC_AGENTS_DIR` or `./.thunc_agents` in the directory you start it in |

Agent folders are found the same way thunc finds them, so a `configure(agents_dir=...)` in your
code isn't visible here: pass the same folder as `DIR`.

In `--agents` mode, a run is live while its session record has no end and the process holding the
agent's `.lock` is alive. A record whose process died partway through shows as **interrupted**.
Session records are written to the second and don't time the model, so in this mode a step's time
is the time between its line and the one before, and model and tool time aren't split.

## How it gets its data

thunc writes events when `THUNC_EVENTS` names a file (see
[thunc/events.py](https://github.com/Eltarras/thunc/blob/main/thunc/events.py)). There's one JSON
line per call start, model reply, call end, agent step and agent end. When thunc-watch runs a
program, it sets `THUNC_EVENTS` to a temporary file and reads that file as it grows. Each line
carries a format version, so a dashboard and a thunc of different releases notice if they don't
match.

Inputs, replies and values are cut to 120-character previews. `--capture` sends them whole. They
can contain personal data, as a trace can.

## Developing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

`thunc watch` runs the binary named by `THUNC_WATCH_BIN` when that's set, so you can try a build
from this folder with the thunc in your checkout:

```bash
cargo build && THUNC_WATCH_BIN=watch/target/debug/thunc-watch .venv/bin/python -m thunc watch app.py
```

To build the wheel that goes to PyPI: `uvx maturin build --release` (it lands in `target/wheels/`).

`demo/demo_app.py` runs the support-inbox functions and an agent against a scripted fake backend,
with delays and some bad replies. It makes no model calls. Run it from the repository root:

```bash
cargo run --manifest-path watch/Cargo.toml -- --python .venv/bin/python watch/demo/demo_app.py
```

To see a screen as text (useful in bug reports and tests), record the events and draw every screen
at a moment in the run:

```bash
watch/target/debug/thunc-watch --plain --save-events /tmp/e.jsonl --python .venv/bin/python watch/demo/demo_app.py
watch/target/debug/thunc-watch --replay /tmp/e.jsonl --dump-screens 110x34 --at 9
```

## Releasing

thunc-watch has its own version (in `Cargo.toml`) and its own releases. Publishing a GitHub release
tagged `watch-vX.Y.Z` runs `.github/workflows/publish-watch.yml`. It builds wheels for macOS, Linux
(glibc and musl) and Windows, checks that the tag matches `Cargo.toml`, and publishes them to PyPI.
thunc's own `vX.Y.Z` releases skip it. The extra in thunc's `pyproject.toml` pins the versions
that speak the same events format, so release a new thunc-watch before a thunc that needs it.

## Not in this version

- **Controls beyond watching.** Stopping a single agent run, holding new calls, re-running a call,
  and approving agent actions live (`ask:` permissions) all need changes in thunc first.
- **`--all`.** Agent runs from every project on the machine would need a registry of running runs.
- **Stopping on Windows.** Ctrl+C and quitting end the program at once there, without the
  KeyboardInterrupt it gets on macOS and Linux.
