# Changelog

All notable changes to thunc. The full notes for each release are on the
[releases page](https://github.com/Eltarras/thunc/releases).

## Unreleased

### Added

- **`thunc run --profile`**: runs a script (or `-m module`) and prints a performance report to
  stderr when it ends: per function, calls, cache hits, retries, failures, total/mean/p95/max time
  and the split between model time and thunc's own; for agents, steps and time in each tool; and
  the share of wall time spent in thunc, with the overlap from `thunc.map`.

### Changed

- **The `anthropic` and `openai` backends reuse their connections.** One SDK client is shared by
  every call in the process (`thunc.map`'s threads and agent runs included), instead of a new
  client, and so a new TCP and TLS handshake, for each call. A new client is made when the API key,
  the SDK's environment variables (`ANTHROPIC_*`, `OPENAI_*`) or the process change. In a local
  benchmark with 60 ms of connection setup, 20 calls in a row went from 1.47 s to 68 ms.
- **Agents on the text protocol can act several times per reply.** On Claude Code, Codex and
  `protocol="text"`, a reply can be a JSON array of independent actions (reading three files)
  instead of one. They run in order, at most 16 per reply, and every result comes back together,
  as with native tool calls. Each turn resends the whole transcript and, on the CLI backends,
  starts the CLI, so fewer turns save both. A single JSON action works as before. On Codex, with
  `live_tests/eval_prompts.py` (default prompt, 5 runs of each task), every run batched its first
  reads: replies went from 6.0 / 4.8 / 5.0 to 5.0 / 3.0 / 3.6 (fix / review / analysis) and the
  mean time from 37 / 27 / 27 s to 29 / 19 / 21 s, with the same work done and 30/30 passing.
- **The `codex` backend returns as soon as the answer arrives.** It reads Codex's JSON events as
  they come (`codex exec --json`) instead of waiting for the process to exit and reading the answer
  from a file. Codex takes about 0.4 s to shut down after answering; that now happens in the
  background. Over 8 alternating pairs of real calls the new way was faster every time, by a median
  of 0.67 s on a call of about 4 s. Every agent turn on Codex is one call, so the saving repeats.

## 0.2.1 (beta)

**Durable agents with Temporal.** An optional `thunc[temporal]` runtime records each model turn
and tool call in a Temporal workflow, so a run survives worker restarts and can be reattached
from another process. Local thunc stays dependency-free. See the
[Temporal guide](https://github.com/Eltarras/thunc/blob/main/examples/temporal/README.md).

### Added

- **`thunc.temporal`**: `Registry`, `Worker`, `Runtime` and `Handle` to register versioned tasks
  and start, reattach to, inspect, cancel and resolve durable runs. One coordinator per
  workspace runs requests in order; a repeated request ID reattaches to the same run. Agents
  with `tools=` or `timeout=` can't be registered for durable runs yet
  ([#37](https://github.com/Eltarras/thunc/pull/37)).
- **Recoverable tool effects**: file writes and memory notes go through an intent and receipt
  journal with atomic replacement and content hashes. A command whose outcome is uncertain is
  never rerun automatically: the run waits for an operator's `resolve()`
  ([#37](https://github.com/Eltarras/thunc/pull/37)).
- **`thunc.temporal.adapters.execute_task`** composes registered tasks from native Temporal
  workflows, with a classify → agent analysis → typed summary example
  ([#38](https://github.com/Eltarras/thunc/pull/38)).

### Changed

- The agent loop's decisions moved into a shared engine (`thunc/execution.py`) that local and
  durable runs both use. A `remember` call that fails no longer appears in `Run.notes`
  ([#36](https://github.com/Eltarras/thunc/pull/36)).

## 0.2.0 (beta)

**Agents.** An agent is a typed function that can look around before it answers: it lists,
reads and searches files in a working directory, and, when its permissions allow, writes, edits
and runs commands, then returns a checked value of the task's return type.
See the [Agents section](https://github.com/Eltarras/thunc#agents) of the README.

### Added

- **`thunc.Agent(name, workdir=...)` and `@agent.task`**: declare tasks like `@thunc.function`.
  Read-only by default, with `list`, `read`, `search` and `remember` tools. Sync and async tasks
  ([#23](https://github.com/Eltarras/thunc/pull/23)).
- **Memory and run files** in `.thunc_agents/<name>/`: `memory.md` (notes kept between runs),
  `agent.json`, and one JSONL record per run. Runs of one agent take turns through an OS file
  lock ([#23](https://github.com/Eltarras/thunc/pull/23)).
- **Permission rules**: `write:`, `read:`, `run:` and `!` denies with globs, plus the `write` and
  `edit` tools. Edits need a fresh read of the file in the same run
  ([#26](https://github.com/Eltarras/thunc/pull/26)).
- **The `run` tool**: commands allowed by `run:` rules run without a shell, with a minimal
  environment and a time limit that also stops their child processes
  ([#28](https://github.com/Eltarras/thunc/pull/28)).
- **`agent.run(task, ...)`** returns a `thunc.Run` with the value, files changed, commands,
  denials, notes and steps. A failed run raises `thunc.AgentError` with the partial record
  ([#29](https://github.com/Eltarras/thunc/pull/29)).
- **`follow=`** gives the agent `AGENTS.md` / `CLAUDE.md`, or files you name, as instructions.
  Off by default ([#31](https://github.com/Eltarras/thunc/pull/31)).
- **Native tool calls** on the Claude and OpenAI APIs, with the fixed part of the prompt cached.
  Claude Code and Codex use a JSON text protocol; `protocol="text"` picks it on an API too
  ([#32](https://github.com/Eltarras/thunc/pull/32)).
- **System prompt presets**: `thunc.prompts.CODING`, `CODE_REVIEW` and `ANALYSIS`, for `system=`
  ([#34](https://github.com/Eltarras/thunc/pull/34)).
- **`tools=`**: your own typed, documented Python functions as agent tools
  ([#34](https://github.com/Eltarras/thunc/pull/34)).
- **`agent.call(...)`**, the agent version of `thunc.call`, and the **`@thunc.agent(...)`**
  shorthand for a one-task agent ([#34](https://github.com/Eltarras/thunc/pull/34)).
- **`timeout=`** bounds a run's time; command limits are cut to the time left
  ([#34](https://github.com/Eltarras/thunc/pull/34)).
- **Files changed by commands** are included in `Run.files_changed`
  ([#34](https://github.com/Eltarras/thunc/pull/34)).
- `live_tests/eval_prompts.py`, an evaluation of the agent system prompt (bare / default /
  preset). 90/90 runs passed on the Claude API and Claude Code
  ([#34](https://github.com/Eltarras/thunc/pull/34)).
- CI now also runs the offline tests on Windows ([#34](https://github.com/Eltarras/thunc/pull/34)).

### Changed

- **`thunc.agent` is the decorator** for one-task agents. `from thunc.agent import Agent` still
  works; only `import thunc.agent as m` now gives the decorator rather than the module.
- The `codex` backend leaves out Codex's own permission notes, which made it refuse allowed edits
  ([#26](https://github.com/Eltarras/thunc/pull/26)).
- Agents refuse the `jev` backend with a `ThuncError` before a run starts; it only answers typed
  questions ([#33](https://github.com/Eltarras/thunc/pull/33)).

Nothing changes for `@thunc.function` and `thunc.call`.

## 0.1.3 (beta)

- **`jev` backend** for TypeSafe's Jev judgment model: `bool` and `Literal` answers in about
  0.3 s ([#27](https://github.com/Eltarras/thunc/pull/27)).
- The **`codex` backend** runs with Codex's own tools off and ignores `~/.codex/config.toml`
  ([#25](https://github.com/Eltarras/thunc/pull/25)).
- `@thunc.function` bodies like `return 1` now raise `TypeError` at definition
  ([#24](https://github.com/Eltarras/thunc/pull/24)).

## 0.1.2 (beta)

- **`cache=True`** saves valid answers on disk; `thunc.clear_cache()`, `thunc.cache_info()` and
  the `thunc cache` command manage them ([#18](https://github.com/Eltarras/thunc/pull/18),
  [#19](https://github.com/Eltarras/thunc/pull/19)).
- **`system=`** replaces the opening of the default system prompt; Codex gets it as its
  instructions file ([#21](https://github.com/Eltarras/thunc/pull/21)).
- **Sturdier parsing**: common near-misses are read, and wrong values are retried instead of
  returned. Only `ThuncError` escapes ([#20](https://github.com/Eltarras/thunc/pull/20)).
- An empty reply is no longer a valid `str`.

## 0.1.1 (beta)

- **`openai` backend** on the Responses API, and local models through `OPENAI_BASE_URL`.
- The Claude API backend tested live; CONTRIBUTING.md and Discussions added.

## 0.1.0 (beta)

- First release: `@thunc.function`, `thunc.call`, typed and validated results with retries and
  `ensure=`, `thunc.map`, JSONL tracing; the `anthropic`, `claude-code` and `codex` backends.
