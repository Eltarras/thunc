# Changelog

All notable changes to thunc. The full notes for each release are on the
[releases page](https://github.com/Eltarras/thunc/releases).

## Unreleased (0.3)

### Added

- **`thunc watch`**, a live dashboard in the terminal for a program's thunc calls and agent runs:
  calls in flight, retries and why each reply was rejected, per-function timings, each agent's
  steps as they happen, and the `--profile` report when the program ends. Click around with the
  mouse or use the keys. `thunc watch app.py` runs a program and watches it; `thunc watch --agents`
  follows agent runs from any process; `--plain` prints one line per event for CI. It's a compiled
  binary in its own package, thunc-watch, installed with `pip install "thunc[watch]"`, so thunc
  itself stays pure Python with no dependencies.
- **`THUNC_EVENTS=FILE`** writes one JSON line per call, attempt and agent step to FILE, which
  thunc-watch reads. Inputs, replies and values are cut to short previews unless
  `THUNC_EVENTS_CAPTURE=1`. Nothing is written when it isn't set.
- **thunc write: functions that write themselves, with `@thunc.function(write=True)`.** On its first
  call the function writes its own body. Three requests start side by side: the call's answer, a
  draft of the body from the docstring and signature (with the whole file and any project types in
  the signature), and five test calls, each answered by the model on its own. The draft is linted
  and run on the call and the test calls within a time limit, and must match every answer; one that
  doesn't goes back with the failing calls, up to three drafts. A passing draft goes into the file
  in place of `...`, the decorator is removed (`import thunc` stays), the checked calls become
  doctest examples, and the call runs the new code: from then on it's plain Python. If the model
  says the task needs judgment, or no draft passes, the call returns the model's answer, the file is
  left as it was, and the reason is kept in `.thunc_write/` until the docstring or signature
  changes. Writing is refused, with a warning, in CI or with `THUNC_WRITE=0`, outside the project,
  and for read-only, installed or changed files and nested functions.
- **The `thunc write FILE::FUNCTION [--dry-run]` command** writes a `write=True` function ahead of
  its first call, or shows the change as a diff.

## 0.2.3 (beta)

**Native tool calls everywhere agents run, durable runs that keep them, and the last of 0.2.**
Agents on Codex make native tool calls through the same MCP relay as Claude Code, durable runs on
Claude Code do too and pick up after a crash mid-call, and durable agents can take `tools=`. On the
Claude API, agents stream their replies, think at effort `high`, and no longer lose a run to
`max_tokens` or a stalled reply. `edit` can replace every occurrence or make several changes at
once, and no longer needs a prior `read`. In the tool-use benchmark (`live_tests/bench_tooluse.py`,
8 tasks, Claude Sonnet 5.5, 3 runs each), every harness passed 24 of 24, the text protocol included
(20 of 24 in 0.2.2) in about half the time; through the Claude API, Sonnet 5.5 and Opus 5.5 passed
8 of 8; on Codex, native calls took 30 seconds a task against 45 for the text protocol.

### Behavior changes

- **Agents on `codex` make native tool calls**, as on Claude Code since 0.2.2 (see Added). When
  Codex can't start them, a run falls back to the text protocol with a warning; `protocol="text"`
  keeps the old way, and durable runs on Codex still use it.
- **`finish` called in the same reply as other calls is refused** (except beside `remember`): its
  value can't account for results the model hasn't seen yet. The other calls run, and the model is
  told to call `finish` on its own. In the tool-use benchmark, a run on the text protocol batched
  `[search, read, finish 0.0]` and returned the guess. Every protocol and durable runs get it.
- **Agents on the Claude API think at effort `high` by default** on Claude 4.6 and later. Claude
  Opus 5.5's own default is `medium`, which is low for agentic coding. `effort=` changes it (below).
- **`edit` no longer needs the file to have been read first.** It only changes text the agent
  quotes exactly, so it can't overwrite what the agent hasn't seen. With the `shell` permission,
  agents often read files with `cat`, and `edit` refused them until they read the file again with
  `read`: 7 times in 24 runs of the tool-use benchmark. A file the agent did read must still not
  have changed on disk since, and `write` still replaces only a file read with `read` (a file
  changed by an edit alone still counts as unread). The `run` tool's description, with `shell`, now
  says to read files with `read`.

### Added

- **Durable runs on Claude Code make native tool calls.** A run goes in segments: one activity keeps
  one `claude -p` process for many model replies, and before each reply's calls are carried out it
  saves a checkpoint of the run and of Claude Code's session. If the worker stops or the CLI dies,
  the retried activity restores the session and continues it with `--resume`; Claude Code marks the
  call that was in flight as interrupted, the model asks for it again, and it gets the journal entry
  it had, so it's replayed, waits for `resolve()`, or runs (one the model doesn't ask for again
  keeps its entry). The session is removed when the run ends. When Claude Code can't start native
  calls, the run goes on with the text protocol. Runs already in progress keep the text protocol.
- **Durable agents can have `tools=`.** Each call of one of the agent's own functions is journaled
  like a command: the intent is recorded before it runs and its result after, so a retried activity
  replays the result instead of calling it again, and a call interrupted by a worker stopping waits
  for `resolve()`. `registry.agent_task(..., retry_safe_tools=["find_issue"])` names the tools that
  may run again instead. A tool's description, arguments and retry marking are part of the task's
  fingerprint, so changing one needs a new version; tasks without tools keep their fingerprint.
- **Native calls on `codex`**: the agent's tools are an MCP server (thunc's relay, given with
  `-c mcp_servers.thunc.*`) that Codex calls; thunc carries out each call with its own tools,
  permissions and run record. Each `codex exec` is a turn: one that ends without `finish` is
  continued with `codex exec resume`, as is one that fails (twice at most). Codex's own tools and
  your `~/.codex` config stay out and its sandbox stays read-only; only thunc's server is approved
  to run without asking, with a tool timeout above `command_timeout`. Resuming needs the session
  saved, so thunc deletes the run's Codex session (`codex delete --force`) when the run ends.
- **`Agent(effort=...)`**: `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"`, on every backend
  (`output_config.effort` on the Claude API, `reasoning.effort` on OpenAI, `--effort` on Claude Code,
  `model_reasoning_effort` on Codex; OpenAI and Codex go up to `"xhigh"`). Recorded in `agent.json`
  only when set, so durable tasks registered without it keep their fingerprint.
- **The model is told when few steps are left.** In its last three replies before `max_steps`, the
  last tool result says how many replies remain, so the model can finish with what it has instead
  of being cut off. Every protocol and durable runs get it; the run record keeps each tool's output.
- **`edit` can replace every occurrence, and make several changes in one call.** With
  `"replace_all": true`, every occurrence of `old` is replaced and the result gives the count. With
  `"edits": [{"old": ..., "new": ..., "replace_all"?: ...}, ...]` (at most 50) instead of `old` and
  `new`, the changes apply in order, each to the text the ones before it left; if one fails, none is
  made, and the error names it. In the tool-use benchmark, models renamed a symbol by writing a
  throwaway script instead of making 26 separate edits, and took twice Claude Code's turns on a
  multi-spot fix. In a durable run, a multi-edit is one effect, recovered as a whole.

### Changed

- **The text protocol reads replies that aren't only the action** (finding 1 of the tool-use
  benchmark report). Models trained for native tool calls often wrap the action in prose, a code
  fence or `<invoke>` markup, or carry on past it with results they make up: on Sonnet 5.5, 25% of
  text-protocol replies were sent back as "not valid JSON" with a correct action inside. Now the
  first complete action (or array of them) in the reply is used, with literal newlines in its
  strings accepted, and a reply written only as `<invoke name="...">` markup is read as its calls,
  each argument in its tool's type (or as one JSON `args` parameter). A reply with no action in it
  is still sent back, as before. A reply read this way runs, but its results carry a note to reply
  with the JSON action alone: without it, a model that slipped into markup was never corrected, and
  on Sonnet 5.5 fell into repeating empty markup until the step timed out.
  This is the text protocol on Codex, `protocol="text"`, durable runs on Codex, and Claude
  Code's fallback from native calls.
- **A text-protocol step on Claude Code stops once its action is complete.** The reply is streamed
  (`--output-format stream-json --include-partial-messages`) and the CLI is stopped as soon as a
  complete action has arrived, rather than left to make up the tool's result until the step times
  out (7 of 8 replayed first steps on Sonnet ran past 120 seconds that way). A batch that has begun
  is waited for, and two blocks of `<invoke>` markup end the step too (a model repeating itself).
  Plain `@thunc.function` calls on Claude Code aren't streamed.
- **The Claude API agent path keeps runs going** (finding 7 of the tool-use benchmark report):
  - Replies are streamed with `max_tokens=64000` (was 16,000 without streaming), so a large write
    fits.
  - A reply cut off at `max_tokens` no longer ends the run: its tool calls aren't run and get an
    error result saying so, and the model is asked again. Two in a row end the run.
  - `pause_turn` is asked to carry on (up to 6 times in a row) instead of ending the run.
    `model_context_window_exceeded` ends it with a message that says so.
  - On Claude 4.6 and later, the API clears old tool results on long runs (context editing, beta
    `context-management-2025-06-27`).
  - Tools are `strict` (arguments guaranteed to match their schema) on the models that support it,
    when the schema allows it: the built-in tools except `edit`, `remember`, and `finish` and custom
    tools whose schema is closed.
  - A reply that sends nothing for `timeout` seconds (300 by default) is stopped and asked again. The
    SDK's read timeout doesn't catch it, as the API's keep-alive pings count as reading: in the
    tool-use benchmark, one reply on Claude Opus 5.5 sent nothing for an hour.
  - A lost connection, a rate limit or a server error on the Claude and OpenAI APIs is retried as a
    step, like the CLI backends' errors, instead of ending the run.

### Fixed

- **A large system prompt no longer stops the `claude-code` backend from starting.** It went on the
  command line, so large `follow=` files and memory could pass the operating system's limit on its
  length (128 KB for one argument on Linux, 32,767 characters for the whole line on Windows), and
  starting `claude` failed with a raw `OSError: Argument list too long`. The prompt now goes in a
  temporary file (`--system-prompt-file`), as it already did for agents' native calls and on Codex.
  This covers `@thunc.function` and `thunc.call`, agents on the text protocol, and durable runs on
  Claude Code. A command line that is still too long raises a `ThuncError` that gives its size.

## 0.2.2 (beta)

**Agents that use their tools reliably, and faster calls.** Agents on Claude Code make native tool
calls instead of writing each action as JSON text, a failed step is retried instead of ending the
run, and the agent's tools fill gaps a benchmark found. In that tool-use benchmark
(`live_tests/bench_tooluse.py`, 8 tasks, Claude Sonnet 5.5, 3 runs each), agents on Claude Code went
from 12 of 24 runs passing to 24 of 24, from 99 to 10 seconds a task, and from $0.084 to $0.022 a
task; Claude Code itself took 11 seconds and $0.069. The API backends also reuse their connections,
Codex answers return sooner, and `thunc run --profile` shows where a program's time goes.

### Behavior changes

Nothing is removed, but these defaults change:

- **Agents on `claude-code` make native tool calls** (see Added). With a `claude` CLI too old for
  them, or with MCP servers turned off by a policy, a run falls back to the text protocol with a
  warning. `protocol="text"` keeps the old way.
- **`list` and `search` leave out what git ignores** in a git repository (build output, caches,
  vendored code). A folder named explicitly is still listed and searched.
- **Long command output keeps its start and its end** (the first error and the summary), not only
  the end.
- **The `claude-code` backend loads none of your Claude Code settings** (`--setting-sources ""`):
  no `CLAUDE.md`, settings or hooks reach thunc's calls, plain function calls included, so an
  agent's `workdir` can't give it instructions unless `follow=` asks for them.

### Added

- **Native calls on `claude-code`**: the agent's tools are an MCP server that one `claude -p`
  process per run calls; thunc carries out each call with its own tools, permissions and run
  record. The CLI runs in the agent's `workdir`. `protocol="text"` keeps the old way, and durable
  runs on Claude Code still use it. When Claude Code can't start native calls (an older CLI, or MCP
  servers turned off by a policy), a run falls back to the text protocol with a warning and a
  `fallback` entry in its record; `protocol="native"` raises instead.
- **`run` takes `cwd`**, a folder inside `workdir` to run the command in.
- **The `shell` permission** runs command lines through the system shell, so pipes, `&&`, `cd`
  and redirects work. Off by default; it can't be combined with `!run:` rules.
- **`search` takes `glob`** (`*.py` by file name, `src/**/*.ts` by path) to limit the files
  searched.
- **`thunc run --profile`**: runs a script (or `-m module`) and prints a performance report to
  stderr when it ends: per function, calls, cache hits, retries, failures, total/mean/p95/max time
  and the split between model time and thunc's own; for agents, steps and time in each tool; and
  the share of wall time spent in thunc, with the overlap from `thunc.map`.

### Changed

- **A failed step is retried.** A timeout, lost connection, rate limit, server error or CLI call
  that ended in an error (`thunc.errors.TransientError`) is retried twice in an agent run, with a
  note in the run record, before the run fails. A text-protocol step on Claude Code or Codex may
  take 120 seconds before it's retried, instead of the whole `timeout`.
- **The `anthropic` and `openai` backends reuse their connections.** One SDK client is shared by
  every call in the process (`thunc.map`'s threads and agent runs included), instead of a new
  client, and so a new TCP and TLS handshake, for each call. A new client is made when the API key,
  the SDK's environment variables (`ANTHROPIC_*`, `OPENAI_*`) or the process change. In a local
  benchmark with 60 ms of connection setup, 20 calls in a row went from 1.47 s to 68 ms.
- **Agents on the text protocol can act several times per reply.** On Codex and `protocol="text"`
  (and on Claude Code when it falls back to the text protocol), a reply can be a JSON array of
  independent actions (reading three files) instead of one. They run in order, at most 16 per reply,
  and every result comes back together, as with native tool calls. Each turn resends the whole
  transcript and, on the CLI backends, starts the CLI, so fewer turns save both. A single JSON
  action works as before. On Codex, with `live_tests/eval_prompts.py` (default prompt, 5 runs of
  each task), every run batched its first reads: replies went from 6.0 / 4.8 / 5.0 to 5.0 / 3.0 /
  3.6 (fix / review / analysis) and the mean time from 37 / 27 / 27 s to 29 / 19 / 21 s, with the
  same work done and 30/30 passing.
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
