![A thunc function returning list[Item] is called with "2 oat lattes and a croissant pls. oh, one more latte!" and returns two Item dataclasses: oat latte ×3 and croissant ×1.](https://raw.githubusercontent.com/Eltarras/thunc/main/.github/demo-function.gif)

**think + function.** Call an LLM like a typed Python function.

[![PyPI](https://img.shields.io/pypi/v/thunc)](https://pypi.org/project/thunc/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/thunc/)
[![CI](https://github.com/Eltarras/thunc/actions/workflows/ci.yml/badge.svg)](https://github.com/Eltarras/thunc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/Eltarras/thunc/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-eltarras.github.io%2Fthunc-9b481b)](https://eltarras.github.io/thunc/docs/)

```bash
pip install thunc
```

```python
import thunc

thunc.configure(backend="claude-code")  # or "codex", "anthropic", "openai"


@thunc.function
def urgency(ticket: str) -> int:
    """Rate how urgent this ticket is, from 1 (can wait) to 5 (customer is blocked)."""
    ...


urgency("I was charged twice!")  # -> 4, a checked int
```

The docstring is the prompt and the return annotation is the type. The answer is parsed into that
type; if it doesn't fit, the model is asked again, and after that `thunc.ThuncError` is raised. No
dependencies, Python 3.10+.

It also runs [agents](#agents): typed functions that can read, edit and test your code before they
answer.

The [docs](https://eltarras.github.io/thunc/docs/) cover everything below, a page per topic.

## Quickstart

No API key needed if you have Claude Code or Codex installed: thunc can use their login.

```bash
pip install thunc
THUNC_BACKEND=claude-code python3 -c 'import thunc; print(thunc.call("Say hello in five words or fewer."))'
```

Swap in the backend you have:

| You have | Set | Install |
|---|---|---|
| [Claude Code](https://claude.com/claude-code), logged in | `THUNC_BACKEND=claude-code` | `pip install thunc` |
| [Codex](https://github.com/openai/codex), logged in | `THUNC_BACKEND=codex` | `pip install thunc` |
| An Anthropic API key | `ANTHROPIC_API_KEY` | `pip install "thunc[anthropic]"` |
| An OpenAI API key | `OPENAI_API_KEY` | `pip install "thunc[openai]"` |
| A local model (LM Studio) | `OPENAI_BASE_URL` (see **Local models** below) | `pip install "thunc[openai]"` |

With an API key set, thunc picks that backend on its own, so `THUNC_BACKEND` isn't needed. In code,
`thunc.configure(backend=...)` does the same. Durable agents on Temporal add
`pip install "thunc[temporal]"`.

> **Beta (v0.2).** The API may still change. Bug reports and feedback are welcome in
> [issues](https://github.com/Eltarras/thunc/issues).

**More examples.** Clone the repo and run them from its root, with no install, through your
Claude Code login:

```bash
git clone https://github.com/Eltarras/thunc && cd thunc
python3 -m examples.hello
python3 -m examples.support_inbox
THUNC_BACKEND=codex python3 -m examples.log_triage
```

## Two ways to write a prompt

| | When | |
|---|---|---|
| `@thunc.function` | The prompt is fixed and should read like code | The docstring is the prompt, the parameters are the inputs, the return annotation is the type |
| `thunc.call(...)` | The prompt is built in code (from config, in a loop, loaded from a file) | `thunc.call(f"Translate into {lang}.", {"text": note})` |

`@thunc.function(instructions=some_string)` combines the two: a typed, reusable function whose
prompt is generated.

**Keep user data out of the instructions.** Your own text can go in the instructions string.
Anything from users, files or the web goes in the inputs:

- `@thunc.function` does this automatically.
- With `thunc.call` it's up to you. In a live test, a hostile email pasted in with an f-string
  tricked the model 3 out of 3 times. Passed as an input, it failed 3 out of 3 times.

## API

| | |
|---|---|
| `@thunc.function` | Turns a signature + docstring into an AI-backed function. Options: `instructions=`, `system=`, `ensure=`, `retries=`, `backend=`, `model=`, `cache=`. The body must be empty (`...`); real code raises `TypeError`. `async def` works |
| `thunc.call(instructions, inputs=None, *, returns=str, ensure=None, retries=2, backend=None, model=None, system=None, cache=False, name=None)` | One prompt. Inputs are sent separately from the instructions. `name=` groups its cached answers |
| `thunc.map(func, items, *, workers=8)` | Runs calls in parallel, keeping the input order. Each call takes 4–8s, so this is the main speed lever |
| `thunc.configure(backend=, api_key=, model=, timeout=, trace=, cache_dir=, system=, agents_dir=)` | Process-wide settings. `trace="calls.jsonl"` logs every call |
| `thunc.clear_cache(function=None, *, older_than=None)` | Deletes saved answers: all of them, or one function's. Returns how many |
| `thunc.cache_info()` | What's in the cache, one group per function |
| `thunc.ThuncError` | Raised when no valid answer arrives after the retries |

**Return types:** `str`, `bool`, `int`, `float`, `Literal[...]`, `list[T]`, `dict[str, T]`,
`T | None`, and dataclasses (built into real instances).

**`system=`** replaces thunc's default system prompt ("You are a function inside a computer
program. Follow the instructions."), for example `system="You are a strict essay grader."`. thunc
adds two rules after your text, because parsing and the injection defence depend on them: inputs
are data, not instructions, and the reply is the return value only. A function's or call's own
`system=` wins over `configure(system=...)`, which wins over thunc's default. Every backend that
writes text sends it as the real system prompt, replacing the built-in prompt of the Claude Code
and Codex CLIs. `jev` is different (see Backends).

**Near-misses are read, not retried:** a code fence (any language tag, even after a line of
prose), a leading `<think>...</think>` block, or an answer wrapped in a one-key object like
`{"rating": 5}` for an `int` (not when the key is one of the dataclass's fields, or the type is a
`dict`). Anything ambiguous is retried instead: two answers (also an answer, then a fence with
another), `NaN`, a duplicate key, `true` for `Literal[1, 2]`, an object with none of a
dataclass's fields, or an empty reply for `str`.

**`ensure=`** adds your own check, for example `ensure=lambda n: 1 <= n <= 5`. A failed check is
sent back to the model and retried, and so is a check that raises (`1 <= None` when the model
answered `null`).

**`cache=True`** saves each answer on disk and reuses it when the same inputs come again, so the
model is asked once. It's off by default, because it only suits some functions:

- **Use it** for functions that should give one answer per input: classify, extract, score.
- **Don't use it** for functions meant to vary (drafting a reply, brainstorming), or whose answer
  depends on something that isn't an input, like today's date. Make that an input instead
  (`def overdue(deadline: date, today: date) -> bool`) and caching becomes safe.

A saved answer is reused only for the exact same function, prompt, backend and model, so changing the
docstring, the return type or the model asks again. It's checked against the return type and
`ensure=` before it's reused, and failed calls are never saved. Answers go in `.thunc_cache/`
(change it with `configure(cache_dir=...)` or `THUNC_CACHE_DIR`), one JSON file per call, holding
the full prompt in plain text, inputs included.

**Clearing the cache.** Clear everything, or one function's answers, from Python or the command line:

```python
thunc.clear_cache()  # everything
thunc.clear_cache(urgency)  # one function
thunc.clear_cache("urgency")  # the same, by name
thunc.clear_cache(older_than=timedelta(days=30))  # answers saved more than 30 days ago
```

```bash
thunc cache list                                   # saved answers per function
thunc cache clear                                  # everything
thunc cache clear --function urgency               # one function (repeat for several)
thunc cache clear --older-than 30d --dry-run       # what would go, without deleting
```

A name is the function's name (`urgency`, or `Triage.urgency` for a method), optionally with its
module (`support_inbox.urgency`). For `thunc.call`, pass `name="..."` to group its answers the same
way; unnamed calls are cleared only with everything or by age. The function's name is part of the
cache key, so renaming a function starts its cache fresh. Ages count from when the answer was saved.
Clearing deletes only cache entries, never other files in the folder, and it's safe while another
process is using the cache. The `thunc` command (also `python -m thunc`) reads `THUNC_CACHE_DIR`,
or takes `--cache-dir`; it can't see a `configure(cache_dir=...)` in your code.

**Backends:**
- `anthropic` is the Claude API: `configure(api_key=...)` or `ANTHROPIC_API_KEY`, plus
  `pip install "thunc[anthropic]"`.
- `openai` is the OpenAI API: `configure(backend="openai", api_key=...)` or `OPENAI_API_KEY`, plus
  `pip install "thunc[openai]"`. The default model is `gpt-5.5`. `OPENAI_BASE_URL` points it at
  any server that speaks the OpenAI Responses API.
- `claude-code` and `codex` call your local CLI login, and are meant for cheap testing.
  Both run with their own tools turned off, so the model can only answer; an agent on either
  gets only its thunc tools, as native calls (see Agents). `codex` also ignores
  `~/.codex/config.toml` (your MCP servers, plugins, `notify` command and model settings); your
  login still works. Pick the model with `configure(model=...)` or `model=`.
- `jev` is TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  judgment model, through the [`jev` CLI](https://github.com/model-clis/jev). The key comes from
  `jev login` or `JEV_API_KEY`, not `configure(api_key=...)`, so Jev can be used for some
  functions alongside another backend's key. Jev doesn't write text: it answers `bool`,
  `Literal` of strings (up to 255) and `Literal` of integers (as ordered levels), with the most
  likely answer returned. Any other return type raises `ThuncError` before a request is sent. The
  inputs are sent as Jev's state and the instructions as its question; a `system=` of your own goes
  before the instructions, and thunc's default system prompt isn't sent. `model=` is ignored (the
  CLI always uses `jev-latest`), and an answer that fails `ensure=` isn't retried, since Jev would
  give the same one. It's only used when you choose it: `backend="jev"` or `THUNC_BACKEND=jev`.
  Setup (install the CLI, log in, check it works): the
  [Jev guide](https://eltarras.github.io/thunc/docs/jev.html).

**Local models:** the `openai` backend works with a local server through `OPENAI_BASE_URL`. This
has been tested with [LM Studio](https://lmstudio.ai) running `openai/gpt-oss-20b`:

```python
# OPENAI_BASE_URL=http://localhost:1234/v1  OPENAI_API_KEY=lm-studio  (any non-empty key works)
thunc.configure(backend="openai", model="openai/gpt-oss-20b")
```

Small models need the retry more often, for example when they explain the answer instead of
giving it alone.

The backend can also be set with `THUNC_BACKEND`. With none set, `ANTHROPIC_API_KEY` (or a
`configure(api_key=...)` alone) selects `anthropic`, and otherwise `OPENAI_API_KEY` selects `openai`.

**Profiling:** run your program with `thunc run --profile` to see where the time went when it ends:

```bash
thunc run --profile support_inbox.py --limit 20   # a script and its arguments
thunc run --profile -m myapp.triage               # a module, as with python -m
```

The report goes to stderr: per function, the calls, cache hits, retries and failures, the total,
mean, p95 and slowest time, and how much of it was the model and how much thunc's own work
(building the prompt, parsing, the cache). Agent runs get their steps, model time and time in each
tool. It also says what share of the program's wall time was spent in thunc, and how much calls
overlapped under `thunc.map`. Without `--profile`, `thunc run` just runs the program, and nothing
is recorded. The program's exit code is passed through.

```
CALLS
FUNCTION  CALLS  CACHED  RETRIES  FAILED  TOTAL   MEAN    P95    MAX  MODEL  LOCAL
urgency      11       1        1       0  1.70s  155ms  309ms  309ms  1.69s   12ms

In thunc:      774ms of 980ms wall time (79%); the rest was the program's own code
Model time:    1.69s, 99% of the time in calls (anthropic/default model 1.69s)
Concurrency:   calls overlapped 2.2x on average (thunc.map or threads)
Slowest:       urgency took 309ms
```

**Watching:** `thunc watch` runs your program with a live dashboard in the terminal: the calls in
flight, retries and why each reply was rejected, each agent's steps as they happen, and the same
report when it ends. Click around with the mouse, or use the keys (`?` lists them). It's a separate
compiled binary, so it's an extra:

```bash
pip install "thunc[watch]"
thunc watch support_inbox.py --limit 20   # a script and its arguments, as with thunc run
thunc watch --agents                      # agent runs in ./.thunc_agents, from any process
```

See [watch/README.md](https://github.com/Eltarras/thunc/blob/main/watch/README.md) for the screens,
the keys and `--plain` output for CI.

**Type checking:** signatures and return types are visible to mypy and Pyright. mypy reports
empty bodies; turn that off with `disable_error_code = ["empty-body"]`.

## Agents

> **New in 0.2.** Agents are new; their API may change in a later release as feedback comes in.

![An agent allowed to write src/** and run pytest is asked to run the tests and fix any problems. pytest shows 1 failure; it reads src/pricing.py, fixes one line, reruns pytest (3 passed) and returns True.](https://raw.githubusercontent.com/Eltarras/thunc/main/.github/demo-agent.gif)

An agent is a typed function that can look around before it answers. Give it a name and a working
directory, declare its tasks the way you write `@thunc.function`, and call them from Python:

```python
repo = thunc.Agent("repo-guide", workdir="~/code/myapp")


@repo.task
def request_timeout() -> int:
    """Find the HTTP request timeout this app uses, in seconds."""
    ...


request_timeout()  # -> 45, after the agent searched the code and read the file that sets it
```

An agent with a single task can be declared in one go, and a task built in code runs with
`agent.call`, the agent version of `thunc.call`:

```python
@thunc.agent("release-notes", workdir="~/code/myapp", permissions=["write:CHANGELOG.md", "run:git log"])
def changelog(since_tag: str) -> list[str]:
    """Add an entry to CHANGELOG.md for the commits since `since_tag`. Return the bullets you wrote."""
    ...


repo.call(f"Where is {setting} set?", returns=str)
```

Each call is one run. The model takes one step at a time (list a folder, search, read or edit a
file) and ends by calling `finish` with a value of the return type, which is checked like any thunc
result. It runs on the Claude and OpenAI APIs through their own tool calls (the
model can make several at once, and the fixed part of the prompt is cached). On Claude Code the
calls are native too: the agent's tools are an MCP server that one `claude -p` process per run
calls, while thunc carries out each call with its own tools, permissions and records. If Claude Code
can't start them (an older `claude` CLI, or MCP servers turned off by a policy), the run uses the
text protocol below instead, with a warning, and so do later runs in the process;
`protocol="native"` fails instead. On Codex it's the same: each `codex exec` is a turn in which
the model calls the agent's tools through the MCP server, a turn that ends without `finish` is
continued with `codex exec resume`, Codex's own tools stay off and its sandbox read-only, and the
run's Codex session is deleted when the run ends. With `protocol="text"` the model replies with
JSON actions as text instead: one at a time, or several independent ones (reading three files) as
a JSON array, which saves turns. That works on any backend, for example with a server behind
`OPENAI_BASE_URL` that has no function calling. (Durable runs on Codex use it.) A reply that wraps its
action in prose or tool-call markup, or carries on past it, is read for its first complete action,
and on Claude Code the step stops as soon as that action has arrived.
The `jev` backend only answers typed questions and cannot run agents, even for a task returning
`bool` or `Literal[...]`. An agent run using it raises `ThuncError` before creating any run files
or calling a backend. Use `@thunc.function` or `thunc.call` for Jev questions.

- **Permissions** say what the agent may do. By default it may read everything in `workdir` and
  save notes, and may not write:

  ```python
  fixer = thunc.Agent("fixer", workdir=".", permissions=["write:src/**", "run:pytest", "!read:.env*"])
  ```

  | Rule | Means |
  |---|---|
  | `write:docs/**`, `write` | create and edit matching files (all files with no path); also lets it read them |
  | `read:src/**` | read only these; any `read:` rule replaces the read-everything default |
  | `run:pytest`, `run:git log`, `run` | run commands that start with these words (`run:git log` allows `git log --oneline`, not `git push`); `run` alone allows any |
  | `shell` | run any command line in a shell (`sh -c`, or `cmd /c` on Windows), so pipes, `&&`, `cd` and redirects work; off by default, and it can't be combined with `!run:` rules |
  | `!read:.env*`, `!write:...`, `!run:git push`, `!memory` | deny; a deny always wins, and `!read` also stops writing |

  `*` stays within one folder, `**` crosses folders, and paths are relative to `workdir`. The agent
  is told its permissions, and an action they don't allow is refused with the reason, after which
  the run carries on. Bad rules fail when the agent is declared.
- **Tools:** `list`, `read` and `search` (a regular expression, optionally limited with a `glob`
  such as `*.py`); `write` (create a file, or replace one) and `edit` (replace text that appears
  exactly once, or every occurrence with `replace_all`; several changes to one file can go in one
  call as `edits`, all made or none) when a write rule allows it; `run` when a run or shell rule
  allows it; and
  `remember`. Every path must stay inside `workdir`: `..`, absolute paths and symlinks that point
  outside are refused, and the rules are checked on where a link really leads. Files the agent may
  not read are left out of `list` and `search`, and so is what git ignores, in a git repository
  (build output, caches, vendored code); a folder named explicitly is still listed and searched.
- **No blind overwrites.** A file is only replaced (`write`) after the agent read it with `read` in
  the same run, and only if it hasn't changed on disk since. An `edit` needs no read, because it
  only changes text the agent quotes exactly, but a file the agent did read must not have changed
  since. There is no undo, so run agents that write in a git repository with a clean tree, and
  review their changes with `git diff`.
- **Commands** run in `workdir`, or in a folder inside it given as `cwd`. Without the `shell`
  permission there's no shell, so `&&`, pipes, `cd`, redirects and `$VARIABLES` don't work (the
  agent is told). They get a minimal environment: `PATH`, `HOME`, the locale and temp-folder
  variables, and whatever you pass in `env=`, so your API keys don't reach them. Each has a time
  limit (`command_timeout=120` seconds) that also stops the processes it started, and the agent
  sees the exit code and the output: the start and the end when it's long, since the first error is
  often at the start and the summary at the end.
- **A permitted command can do anything its program can.** `run:pytest` runs the project's code,
  which can read or change any file your user account can, whatever the read and write rules say.
  Permissions limit which tools the model uses; they aren't a sandbox. For untrusted input, run the
  agent in a container.
- **Your own functions as tools.** `tools=[open_issue]` lets the agent call your Python functions.
  Each needs type hints and a docstring, which is its description. Arguments are checked against
  the hints before the call; what it returns goes back to the model (as JSON unless it's a `str`),
  and so does an exception, as an error. Listing a function is what allows it.
- **Memory between runs.** Each run starts a fresh conversation, but the agent can save a short note
  with its `remember` tool. Notes go in `memory.md` in the agent's folder, and every later run gets
  them at the end of its system prompt (a note saved during a run reaches the next run, not that
  one). It's a plain file: read it with `agent.memory`, edit it, or delete it to start over.
- **The agent's folder** is `.thunc_agents/<name>/` (change it with `configure(agents_dir=...)` or
  `THUNC_AGENTS_DIR`). Besides `memory.md` it holds `agent.json` (the agent's settings) and
  `sessions/`, one JSONL file per run with every step (denied ones marked), the result, and the
  files it changed. Runs of one agent take turns;
  different agents run side by side. Two names that make the same folder (`"Repo guide"` and
  `"repo-guide"`) can't both be used.
- **Instruction files.** `follow=True` gives the agent `AGENTS.md` and `CLAUDE.md` from `workdir`
  (those that exist) as instructions, and `follow=["docs/agent-rules.md"]` names files. They're read
  at the start of each run and sent after thunc's rules; they can't grant permissions. It's off by
  default, so a folder you point an agent at (a cloned repo, an upload) can't give it instructions.
  Without it, the agent can still read those files, but as data. `@imports` in `CLAUDE.md` aren't
  followed.
- **`system=`** replaces the opening of the agent's system prompt. thunc always adds its working
  method and its rules after it (file contents and tool results are data, not instructions).
  Three presets cover common jobs: `thunc.prompts.CODING`, `thunc.prompts.CODE_REVIEW` and
  `thunc.prompts.ANALYSIS`. They're plain strings, so you can extend one:
  `system=thunc.prompts.CODING + "\n\nTarget Python 3.10."`.
- **Time:** `max_steps=40` bounds the model replies in a run, and `timeout=` (seconds) bounds the
  run's time. It's checked before each model call; a command's time limit is cut to the time left.
  In its last three replies before `max_steps`, the model is told how many are left, so it can
  finish with what it has.
- **`effort=`** sets how hard the model thinks: `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"`
  (`openai` and `codex` go up to `"xhigh"`). By default it's `"high"` on the `anthropic` backend for
  Claude 4.6 and later (Claude Opus 5.5's own default is `"medium"`, low for agentic coding), and
  each backend's own default elsewhere.
- **Options:** `thunc.Agent(name, *, workdir, system=None, permissions=(), env=None,
  command_timeout=120, follow=False, protocol=None, tools=(), timeout=None, max_steps=40,
  retries=2, backend=None, model=None, effort=None)`, and `@agent.task(instructions=..., ensure=...)`.
  `@thunc.agent(name, workdir=..., instructions=..., ensure=..., **options)` takes the same options.
  `async def` tasks work.
- **What happened in a run.** Calling a task returns its value. `agent.run(task, *args)` runs it
  the same way and returns a `thunc.Run` instead, typed like the task (`Run[int]`):

  ```python
  run = fixer.run(make_tests_pass)
  run.value  # True
  run.files_changed  # ["src/mathutil.py"]  (by write, edit and commands)
  run.commands  # [Command("python3 tests/test_mathutil.py", exit_code=0, seconds=0.04)]
  run.denied  # [Denial("run", "git commit -am fix", "running ... is denied by '!run:git'")]
  run.notes, run.followed, run.steps, run.seconds, run.session
  ```

- **Failures are loud.** A run that hits `max_steps`, never gives a valid value, or loses its
  backend raises `thunc.AgentError` (a `ThuncError`), whose `.run` is the record up to that point.
  A step that fails for a reason asking again may fix (a timeout, a lost connection, a rate limit,
  a server error, a CLI call that ended in an error) is retried twice, after 2 and 4 seconds, and
  each retry is in the run's record. A text-protocol step on Claude Code or Codex may take 120
  seconds before it's retried. On the Claude API, a reply cut off at `max_tokens` doesn't end the
  run: its tool calls get an error result saying so (twice in a row does). With tracing on, each run
  is also one line with every model reply.

**How the prompt was tested.** `python -m live_tests.eval_prompts --backend anthropic` runs three
small tasks (fix a bug, review a diff, answer a question about a repo) with three versions of the
system prompt: bare (no working method), the default, and the task's preset. Five runs of each, on
the Claude API on 4 October 2026 and on Claude Code on 5 October 2026:

| | Claude API (Opus 5.5, native calls) | Claude Code (Sonnet 5.5, native calls) |
|---|---|---|
| Passed | 45/45: every task, every version | 45/45 |
| Steps (bare / default / preset) | fix 4.0 / 4.0 / 4.0, review 2.0 / 2.4 / 2.8, analysis 3.0 / 3.0 / 3.0 | fix 4.0 / 4.0 / 4.0, review 2.0 / 2.0 / 2.0, analysis 3.0 / 2.8 / 2.6 |
| Cost | $0.76 for all 45 runs (cache reads were 257,553 of 312,294 input tokens) | |

Every version passed every time, so these tasks are too easy to tell the versions apart: the result
says the prompt does no harm, not that it helps. On the API, the review preset read more of the code
before answering, and each review flagged the renamed function as a minor issue (outside code
importing the old name breaks), never as blocking. On Claude Code, only the review preset flagged
it (2 of 5 runs, as minor). On the text protocol these tasks took more steps (fix 5.2 / 6.0 / 6.0
on Claude Code before native calls). For harder tasks that do tell harnesses apart, see the tool-use
benchmark in `live_tests/bench_tooluse.py` and its report.

## Durable agents with Temporal

> **New in 0.2.1.** Install with `pip install "thunc[temporal]"`. Durable agents are new; their
> API may change in a later release as feedback comes in.

The optional `thunc.temporal` runtime records each agent model turn and tool result
in a Temporal workflow. Workers can restart and continue recorded progress. Existing
function calls and `agent.run()` remain local and need no Temporal installation.

Register tasks on a worker, then submit by stable identity:

```python
from thunc.temporal import Registry, Runtime, Worker

# Worker process; review_task is an existing @agent.task function.
registry = Registry(state_dir="/srv/thunc-state")
registry.agent_task("repo.review", review_task, version="1", workspace_id="repo")
worker = await Worker.connect("localhost:7233", task_queue="repo-v1", registry=registry)
# await worker.run() in the worker's async entry point

# Client process; a reconnectable handle survives this process exiting.
runtime = await Runtime.connect("localhost:7233", task_queue="repo-v1")
handle = await runtime.start(
    "repo.review",
    version="1",
    workspace_id="repo",
    inputs={},
    returns=str,
    request_id="review-123",
    deadline_seconds=1800,
)
run = await handle.result()
print(run.value)
```

Durable mode requires a Temporal service, a worker, and persistent storage on the
same volume. File changes and memory updates use recovery receipts. Commands, and the agent's
own `tools=` functions, with uncertain outcomes pause for operator resolution instead of blindly
running twice (`retry_safe_tools=` names functions that may run again).
Temporal does not back up your workspace or guarantee exactly-once external effects.

The [Temporal guide and runnable example](examples/temporal/README.md) cover service
setup, typed functions, composition, retries, cancellation, permissions, storage,
history replay and upgrades.

## Examples

| | |
|---|---|
| [hello.py](https://github.com/Eltarras/thunc/blob/main/examples/hello.py) | The smallest call |
| [support_inbox.py](https://github.com/Eltarras/thunc/blob/main/examples/support_inbox.py) | Docstring functions returning a `Literal`, an `int` with `ensure=`, a dataclass, and a reply; tickets processed in parallel |
| [dynamic_prompts.py](https://github.com/Eltarras/thunc/blob/main/examples/dynamic_prompts.py) | Prompts built from a style guide with `thunc.call`, and a grading function generated from a rubric |
| [log_triage.py](https://github.com/Eltarras/thunc/blob/main/examples/log_triage.py) | Plain Python and AI functions mixed, with tracing |
| [repo_guide.py](https://github.com/Eltarras/thunc/blob/main/examples/repo_guide.py) | Agents: read-only tasks over this repo returning a dataclass and lists, on the Codex backend, with each run's steps read from the trace |
| [jev_hello.py](https://github.com/Eltarras/thunc/blob/main/examples/jev_hello.py) | The smallest Jev calls: a yes/no, a label and a rating |
| [jev_inbox.py](https://github.com/Eltarras/thunc/blob/main/examples/jev_inbox.py) | A support inbox triaged on Jev: spam, team and urgency for 8 tickets in about a second |
| [jev_with_claude.py](https://github.com/Eltarras/thunc/blob/main/examples/jev_with_claude.py) | Jev decides which messages need a reply; Claude writes only those replies |

## Code

```
thunc/
  __init__.py    public API
  decorator.py   @thunc.function
  agent.py       thunc.Agent, @agent.task, @thunc.agent
  tools.py       the agent's tools: list, read, search, write, edit, run
  permissions.py the agent's permission rules
  runs.py        thunc.Run and AgentError: what a run did
  execution.py   the agent loop's decisions, shared by local and durable runs
  native.py      how a run talks to its backend: native tool calls or the text protocol
  claude_code.py native tool calls on Claude Code, through an MCP server (mcp_relay.py)
  codex.py       native tool calls on Codex, the same way
  mcp_relay.py   the MCP server the CLI starts: it forwards each tool call to the run
  relay.py       the run's end of the MCP server: the connection the calls come through
  store.py       the agent's folder: memory, settings, run records, the lock
  prompts.py     the agent's system prompt
  __main__.py    the thunc command: thunc run [--profile], thunc watch, thunc cache list / clear
  profiling.py   thunc run --profile: timing records and the report
  events.py      THUNC_EVENTS: the live events thunc watch reads
  core.py        thunc.call, thunc.map, tracing
  cache.py       the answer cache: saving, listing, clearing
  schema.py      return types: describe, parse, validate
  config.py      settings and backend selection
  backends.py    anthropic, openai, claude-code, codex, jev
  errors.py      ThuncError, and TransientError for failures worth asking again
  temporal/      durable agents on Temporal: registry, worker, client, workflows, effect journal
watch/           thunc watch, the dashboard: a Rust binary, published as thunc-watch
tests/           offline: a fake backend, never a real model
live_tests/      against a real model: hello, a yes/no decision, labels and ratings, messy text to a
                 dict, agents; also the agent prompt eval (eval_prompts.py) and the tool-use
                 benchmark (bench_tooluse.py)
examples/
```

## Limitations

- **There's no record/replay for tests yet.** `cache=True` is per function; there's no switch
  that serves every call from disk and fails on a miss.
- **`Literal` results from `thunc.call` are typed as `Any`.** `@thunc.function` has no such gap.
- **Docstrings disappear under `python -OO`.** Use `instructions=` there.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[anthropic,openai,temporal-test,dev]"
.venv/bin/pytest                    # offline tests (these run in CI)
THUNC_TEMPORAL_TESTS=1 .venv/bin/pytest -c pytest-temporal.ini tests/temporal   # a real local Temporal service; no model calls
.venv/bin/pytest live_tests         # real model calls through your Claude Code login; costs quota
THUNC_BACKEND=anthropic .venv/bin/pytest live_tests   # the same, through the Claude API (needs ANTHROPIC_API_KEY)
THUNC_BACKEND=openai .venv/bin/pytest live_tests      # the same, through the OpenAI API (needs OPENAI_API_KEY)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy --strict thunc && .venv/bin/mypy --strict --platform win32 thunc
```

## License

[MIT](https://github.com/Eltarras/thunc/blob/main/LICENSE)
