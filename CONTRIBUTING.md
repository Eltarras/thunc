# Contributing to thunc

Thanks for helping. thunc is in beta, so feedback on the API is as useful as code.

## Where to start

- Issues labelled [`good first issue`](https://github.com/Eltarras/thunc/labels/good%20first%20issue)
  are small and self-contained. [`help wanted`](https://github.com/Eltarras/thunc/labels/help%20wanted)
  issues are bigger, or need a design discussion first.
- For anything beyond a small fix, comment on the issue or start a thread in
  [Ideas](https://github.com/Eltarras/thunc/discussions/categories/ideas) before writing code, so
  we agree on the approach first.
- Questions go in [Q&A](https://github.com/Eltarras/thunc/discussions/categories/q-a).

## Setup

```bash
git clone https://github.com/Eltarras/thunc && cd thunc
python3 -m venv .venv && .venv/bin/pip install -e ".[anthropic,openai,temporal-test,dev]"
```

`temporal-test` installs the pinned Temporal SDK the durable-run tests use. Without it the tests in
`tests/temporal/` are skipped and everything else still runs. The README's
[Development](README.md#development) section lists the commands for live tests.

## Before you open a PR

CI runs these on Python 3.10 to 3.14 on Linux, and all must pass. Run each on its own and check
its exit code:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy --strict thunc
.venv/bin/mypy --strict --platform win32 thunc
```

CI also runs the offline tests on Windows (Python 3.13), since the agent lock, command handling and
paths have Windows-only code. If you touch `thunc/temporal/`, run the integration tests against a
real local Temporal service too (they download it once and make no model calls):

```bash
THUNC_TEMPORAL_TESTS=1 .venv/bin/pytest -c pytest-temporal.ini tests/temporal
```

If you touch `watch/`, the dashboard, CI also builds and tests it on Linux, macOS and Windows. It
needs a Rust toolchain; from `watch/`, run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

A change to the events in `thunc/events.py` is a change to what the dashboard reads: update
`watch/src/event.rs` in the same PR, and raise the format number if old dashboards would misread it.

## Ground rules

- **Standard library only in the core.** A provider SDK is an optional extra in `pyproject.toml`,
  imported inside its backend function, never at module level. The same goes for Temporal:
  only `thunc/temporal/` imports `temporalio`, and `thunc.call`, `@thunc.function` and
  `agent.run()` must keep working without it.
- **Tests in `tests/` never call a real model.** Use the `fake` fixture in `tests/conftest.py`,
  which returns scripted replies and records the prompts. Tests of native tool calls replace only
  the SDK client and use the SDK's real types. Real calls belong in `live_tests/`, which CI
  doesn't run.
- **Agent permissions are a safety boundary.** A change to `permissions.py`, `tools.py` or the
  effect journal in `thunc/temporal/effects.py` needs a test that fails when the rule is broken.
  Check it by breaking the rule on purpose and watching the test fail.
- **Durable runs replay.** Workflow code in `thunc/temporal/workflows.py` stays deterministic;
  model calls, tools and file access happen in activities. A file or memory change goes through
  the intent and receipt journal, and a command whose outcome is uncertain is never rerun
  automatically. Before changing orchestration or the saved state format, version it and replay
  saved histories (see the [Temporal guide](examples/temporal/README.md)).
- **User data stays out of the instructions.** Inputs are sent separately from the prompt
  (see `_build_prompt` in `thunc/core.py`). Don't add code paths that paste inputs into
  instructions.
- **Failures are loud.** When no valid answer arrives, raise `ThuncError`; never return a
  default value.
- **Update the README** when you change the public API or the supported return types, and the
  [Temporal guide](examples/temporal/README.md) when you change durable runs.
- **One change per PR**, with a description of what it does and how you tested it.

## Adding a backend

1. Write a function in `thunc/backends.py` with the same signature as the others:
   `(text, *, system, model, api_key, timeout) -> str`. It should raise `ThuncError` on
   connection errors, refusals and cut-off answers.
   A model that answers typed questions instead of writing text (like `jev`) takes
   `(instructions, inputs, returns, *, system, timeout)` instead, raises `ThuncError` for
   return types it can't answer, and returns its answer as JSON text.
2. Register it in `BACKENDS` (or `TYPED_BACKENDS`), and in the selection logic in
   `thunc/config.py` if needed.
3. Add an optional extra for its SDK in `pyproject.toml`, and add the extra to the CI install.
4. Add offline tests with a stubbed SDK module, like the existing ones in `tests/test_backends.py`.
5. Agents work on any text backend through the JSON text protocol. For native tool calls, add a
   `Conversation` for the API in `thunc/native.py`, add the backend to `native.NATIVE`, and pick
   the class where `thunc/agent.py` builds the conversation. Test it like `tests/test_native.py`.
   A CLI that can call MCP tools can get native calls the way Claude Code and Codex do: through the
   relay in `thunc/relay.py` (see `thunc/claude_code.py` and `thunc/codex.py`, tested with fake CLIs
   in `tests/test_claude_code_agent.py` and `tests/test_codex_agent.py`).
   Raise `TransientError` for failures worth asking again, so agent runs retry them.
   Agents and durable runs refuse typed backends.
6. Run `THUNC_BACKEND=<name> .venv/bin/pytest live_tests` against the real service, and say in
   the PR that you did.

## Known issues

Found while testing how thunc handles replies that don't follow the contract. Some are left alone
on purpose; the rest are open. Check here before reporting one, and comment on the issue (or open
one) before working on an open item.

**Retried rather than read, on purpose.** Each of these has a reading that could be wrong, so the
model is asked again instead of thunc guessing:

- Quoted numbers (`"4"` for an `int`), labels in the wrong case (`Bug` for `bug`), prose around
  JSON, Python-style values (`['a']`, `None`), trailing commas, curly quotes, non-ASCII digits.
- Unquoted text for `str | None`, and exclamations like `Yes!` for a `bool`.
- Several `<think>` blocks in a row, `~~~`, indented or four-backtick fences.
- A one-key object whose key is a field of the expected dataclass: `{"customer": {...}}` for an
  `Order` with a `customer` field is an `Order` with a bad customer, not a wrapper.
- A reply after the closing fence (for example a reasoning paragraph): it could be a second answer.

**Returned as they are, on purpose.** For a `str` return type, a preamble ("Sure! Here's…"),
refusal text, a code fence or JSON quotes come back as part of the answer: with plain text there's
no reliable way to tell them from a real one. Only an empty reply is retried.

**Open: behavior that might change.**

- `thunc.map` loses every result when one item fails, and returns coroutines for `async` functions.
- Backend errors (timeouts, connection failures) are never retried by `thunc.call`; only bad
  replies are. Local agent runs retry a step that failed with a `TransientError` twice, and durable
  runs retry transient provider failures.
- Several lines of prose before a code fence are read as a preamble, though `_unfence` in
  `thunc/schema.py` documents one line, and no test pins either behavior.
- A union takes the first option that accepts the reply as JSON, in the order written: `3` for
  `float | int` is `3.0`. An unquoted label is only tried after that, so `2` for
  `Literal["2"] | int` is the int `2`, not the label.
- Unknown fields in a dataclass object are ignored as long as one known field is present, and a
  dataclass with no fields accepts any object.
- A float written as a whole number is read as an `int` below 2**53, even when the float only
  approximates what was written (`3.9999999999999999` is `4`).
- The previous reply is pasted into the retry prompt as is (only lone surrogates are escaped), so a
  reply containing `</instructions>` lands in the prompt unchanged.
- A trace file that can't be written raises an error, while the cache only warns.
- The cache key doesn't include `OPENAI_BASE_URL`: two servers with the same model name share entries.
- The trace records `"model": null` when the backend's default model is used.

**Open: not about replies.**

- A dataclass that refers to itself makes `describe()` recurse forever.
- `returns=None` written literally is reported as unsupported (`-> None` works).
- `dict[int, X]` returns string keys.
- `Literal[math.inf]` is described as `Infinity`, which isn't valid JSON, so it can't be read back.
- A list of dataclasses passed as an input is shown to the model as Python reprs, not JSON.
- A string return annotation naming a class defined after the function raises `NameError` when
  the decorator runs.
- On Python 3.10 and 3.11, a `make_dataclass` class whose field types are strings naming anything
  but builtins (`"Item"`, `"Annotated[int, 'x']"`) raises `NameError`: before 3.12 its module is
  `types`, not the caller's.
- `Annotated[...]` works on dataclass fields but not as a return type.
- The Python 3.10 fallback for string annotations (`_hints` and `resolve_strings` in
  `thunc/schema.py`) can be removed when 3.10 support is dropped.

**Open: agents and durable runs.**

- On Codex, native calls count each tool call as a step: Codex's events don't say which calls came
  from the same model reply. A long task reaches `max_steps` sooner than on Claude Code (22 steps on
  `rename` in the tool-use benchmark, against `max_steps=40`).
- Tokens aren't reported for native runs on Codex: `codex exec --json` reports usage at the end of
  a turn, and a run ends inside one when the model calls `finish`.
- Durable runs on `codex` use the JSON text protocol, not native calls.
- On a fix in a long file (`deep_fix` in the tool-use benchmark), native calls take about twice
  Claude Code's steps (8 against 4.3), reading around the file in pages. A larger `read` limit is
  the next thing to measure.
- Durable runs on Claude Code save the whole session file at each checkpoint, so a long run's saved
  state grows with every reply, and counts toward the 16 MiB limit. A CLI that dies leaves Claude
  Code's own `~/.claude/sessions/<pid>.json` behind.
- The `shell` permission's own test (pipes, `cd`, redirects) is skipped on Windows; only a single
  command line runs through `cmd /c` in CI.
- Durable runs have no garbage collection: the journal, transcript artifacts and request-ID
  tombstones are kept forever. There's no context summarization either, so a long run fails once
  its saved state passes 16 MiB.
- A durable workspace can only restart on the same volume and absolute paths; nothing moves it
  between hosts.
- A hard worker kill can leave a command's subprocesses running.
- Durable runs aren't tested on Windows (the code is only type-checked for it), and there are no
  live provider tests for durable runs yet.

**Open: `thunc watch`.**

- The dashboard only watches. Stopping a single agent run, holding new calls, re-running a call and
  approving `ask:` actions live all need changes in thunc first.
- `--agents` follows one folder; agent runs from every project on the machine (`--all`) would need
  a registry of running runs.
- On Windows, Ctrl+C and quitting end the program at once, without the `KeyboardInterrupt` it gets
  on macOS and Linux.

**Open: thunc write (experimental).** It may change or be removed in a later release.

- The draft is checked against the model's own answers, so a rule the model gets wrong every time
  passes. The checked calls go into the docstring as doctests for review.
- `thunc write FILE::FUNCTION` refuses methods: there's no instance to make the test calls with.
  A method is written on its first call instead.
- It doesn't run on `jev`, which can't write code.

## Reporting bugs

Open an [issue](https://github.com/Eltarras/thunc/issues) with:

- the smallest code that shows the problem
- the backend
- your thunc and Python versions
- if you can, the relevant lines from `thunc.configure(trace="calls.jsonl")`

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
