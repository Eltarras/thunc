# Tool-use benchmark: harness issues, 4 October 2026

**Question.** When a thunc agent fails or runs slowly, how much of that comes from thunc rather
than the model? Same model, same tasks, compared with Claude Code as the reference harness.

**Answer.** Most of it. On Claude Sonnet 5.5, thunc agents passed **12 of 24** runs and Claude
Code passed **24 of 24**. Every thunc failure traces to the harness: the text protocol thunc uses
on the `claude-code` backend, and the fact that one failed model call ends the whole run. With
thunc's own tools, permissions and prompt delivered as *native* tool calls instead, the same
tasks went **24 of 24**, at about Claude Code's speed and **58% below its cost**. What still
separates the two harnesses is the tool set: thunc agents make about twice as many tool calls on
fix-and-test tasks, because `run` has no shell and no working directory.

Reproduce with `python -m live_tests.bench_tooluse` (see its docstring).

## The benchmark

Eight small repositories. Each one presses on a single part of a harness:

| Task | What it presses on | How it passes |
|---|---|---|
| `needle` | A 600-module repo with a stale, gitignored `build/` copy that comes first in tree order | The rate the code actually uses (0.137, two hops from the call site) |
| `deep_fix` | A bug in a helper near line 1,900 of a 2,400-line file | The test passes and is unchanged |
| `rename` | `get_usr` → `get_user` in 14 files, 26 places | No `get_usr` left; the tests pass |
| `write_tricky` | A new module that is mostly backslashes and quotes | Visible and hidden tests pass |
| `tabs` | Tab-indented near-duplicate functions | The test passes and the tabs are kept |
| `noisy_build` | The one real error is the first line of 4,000; the end is only warnings | The build succeeds, `build.py` unchanged |
| `subdir` | Tests that must run from `services/api` | The tests pass from there; tests and fixtures unchanged |
| `routes` | A nested structured answer gathered from 4 files, with a commented-out decoy and a test-only decoy | Exactly the 6 runtime routes |

Harnesses, each on the same model. Every thunc agent had permissions `["write", "run"]`, so no
permission rule held anything back.

- **thunc**: `thunc.Agent` on the `claude-code` backend, which uses the text protocol (one JSON
  action per reply, the whole transcript re-sent each step). This is what users get today
  without an API key.
- **thunc-fixed**: thunc with three text-protocol fixes patched in for the benchmark only.
- **thunc-mcp**: thunc's own tools, permission checks and system prompt, served to `claude -p`
  as an MCP server so the model makes native tool calls. Claude Code's built-in tools are off.
- **claude-code**: Claude Code (`claude -p` with Read, Edit, Write, Bash, Grep, Glob).

## Results

Claude Sonnet 5.5, 3 runs of each task (24 runs per harness):

| Harness | Passed | Turns per task | Tool calls per task | Seconds per task | $ per task | Cache-read share | Errors and rejected replies |
|---|---|---|---|---|---|---|---|
| thunc (today) | **12/24** | 5.4 | 3.5 | 99 | 0.084 | 34% | 32 |
| thunc-fixed (text-protocol fixes) | 24/24 | 7.8 | 5.8 | 206 | 0.112 | 35% | 43 |
| **thunc-mcp (native calls)** | **24/24** | 7.8 | 6.8 | **14** | **0.030** | 86% | **1** |
| claude-code | 24/24 | 4.0 | 3.0 | 12 | 0.070 | 92% | 1 |

Claude Opus 5.5, 1 run of each task:

| Harness | Passed | Turns per task | Seconds per task | $ per task | Cache-read share | Rejected replies |
|---|---|---|---|---|---|---|
| thunc (today) | 8/8 | 8.5 | 44 | 0.264 | 25% | 11 of 68 turns |
| claude-code | 8/8 | 4.0 | 18 | 0.127 | 91% | 0 |

Results per task (Sonnet 5.5, passed out of 3):

| Task | thunc | thunc-fixed | thunc-mcp | claude-code |
|---|---|---|---|---|
| needle | 3 | 3 | 3 | 3 |
| deep_fix | 0 | 3 | 3 | 3 |
| rename | 0 | 3 | 3 | 3 |
| write_tricky | 1 | 3 | 3 | 3 |
| tabs | 3 | 3 | 3 | 3 |
| noisy_build | 3 | 3 | 3 | 3 |
| subdir | 2 | 3 | 3 | 3 |
| routes | 0 | 3 | 3 | 3 |

On Opus, thunc passes, but at 2.1× Claude Code's cost and 2.4× its wall time. On Sonnet, it fails
half the runs. The model is the same in every column; only the harness changes.

## Findings, by impact

### 1. The text protocol works against the model's trained tool use (critical)

On the `claude-code` and `codex` backends, the model is asked to reply with one JSON object as
plain text, and the CLI runs with no tools (`--tools ""`). Current models are trained to call
tools natively, and they slip back into that habit. This failed in three distinct ways.

**a. The model doesn't stop after the action.** It writes a correct action, then keeps going:
it makes up the tool's result and the next steps, and sometimes falls into a repetition loop.
Here is the start of one first reply, abridged:

```
{"tool": "read", "args": {"path": "tests/test_escapes.py"}}

<tool_result>1	import os, sys
2	sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
...
<tool_result>
<tool_result>
<tool_result>          (repeated for over 4,000 stream events)
```

Replaying the first step of `write_tricky` 8 times, **7 of 8** calls on Sonnet and **1 of 8** on
Opus were still running after 120 seconds. On Opus, 2 more of the 8 went on to make up results
and later actions. In one of them, the reply thunc would receive was an `edit` to a file that
doesn't exist yet, built on file contents the model had invented. In the benchmark, these
became `claude timed out after 300s`, which ended 6 runs. A native tool call ends the turn by
itself. The text protocol has no stop sequence, and nothing cuts the reply off once a complete
action has arrived (`thunc/native.py:93`, `thunc/backends.py:139`).

**b. The model tries to make native calls and the CLI gives up.** When parallel calls are the
natural next step (for example, 26 edits after a `search`), the model emits native tool-call
markup. With no tools defined, the CLI ends the call with `terminal_reason:
"malformed_tool_use_exhausted"` ("The model's tool call could not be parsed (retry also
failed)"), and thunc treats that as fatal. `rename` failed this way **6 of 6** times when
reproduced, always at step 2. It caused 5 of the 12 benchmark failures.

**c. Replies with a valid action get rejected.** 25% of thunc's turns on Sonnet (32 of 130) and
16% on Opus were sent back as "not valid JSON". In almost all of them a correct action was in
the reply, wrapped in something else:

| Reply shape | Count (Sonnet) |
|---|---|
| `<invoke name="read">…</invoke>` markup, then the JSON | 19 |
| Prose, then the JSON | 7 |
| The JSON, then a stray fence or made-up results | 6 |

`action()` hands the whole reply to `parse()`, which accepts a fenced block but not raw JSON
with text around it (`thunc/native.py:117-119`). The same parser also rejects literal newlines
inside JSON strings (`json.loads` in strict mode), which is how models often write file
contents. Every rejected reply costs a full model call and counts against `max_steps`.

**Fix: native tool calls on the CLI backends.** The `thunc-mcp` prototype serves thunc's own
`Workdir` tools, permission checks and `finish` to `claude -p` over MCP (about 100 lines; see
`run_thunc_mcp` and `mcp_server` in `bench_tooluse.py`). Results: 24/24, 1 tool error in 164
calls, 14 seconds and $0.03 per task. It also removes finding 4 (cost and caching), because the
CLI runs one process per agent run and caches a conversation that grows by appending.
Producing a full `thunc.Run` record needs a bit more work: stream the CLI's events into the
session file, and have the MCP server run inside thunc so denials and changed files are
recorded as they are today. Codex supports MCP servers as well.

**Interim fix, if the text protocol stays.** `thunc-fixed` combines three changes: (1) read the
first complete action in the reply and ignore what surrounds it, with `strict=False`; (2) stream
the CLI's output and stop the process as soon as an action is complete; (3) accept a JSON array
of independent actions and say so in the prompt. That took Sonnet from 12/24 to 24/24. It is
still slower and pricier than Claude Code: 13% of CLI calls stalled for 100+ seconds before any
text arrived (the retry rescued them), 21% of replies were still pure `<invoke>` markup, and the
cache-read share stayed at 35%.

### 2. One failed model call ends the whole run (critical)

`conversation.next()` raising `ThuncError` for any reason (a CLI timeout, a CLI error, an API
500 or 529 after the SDK's 2 retries) aborts the run with `AgentError` (`thunc/agent.py:476`,
`:503`). Everything the agent did up to that point is kept on disk but can't be resumed from.
In the benchmark this turned a single bad call into a failed task 11 times (6 timeouts, 5 CLI
errors); the twelfth failure was the made-up routes answer in finding 4. The CLI timeout is
the global `timeout=300` (`thunc/config.py:16`), so each hang also wastes 5 minutes.

**Fix:** retry transient failures inside the loop (timeouts, CLI errors, 429, 5xx, connection
errors) with backoff, re-sending the same step. Use a much shorter per-call timeout for agent
steps (about 60–90 s, separate from the run's `timeout=`). On the API backends, raise the SDK's
`max_retries`.

### 3. The `claude-code` backend leaks Claude Code's own context into every call (high)

Even with `--system-prompt`, the CLI adds its environment block and system reminders. Asking the
model what it could see returned: working directory `/tmp`, Claude Code's scratchpad path, proxy
notes, the user's email address, git commit attribution rules, today's date and a token budget.
That reaches every thunc call on this backend, plain `@thunc.function` calls included.

It changes what agents do. The model believed its working directory was `/tmp` (it invented
`python: can't open file '/tmp/tests/test_ledger.py'`), and one run tried to `write`
`/tmp/claude-0/-tmp/…/scratchpad/rename.py`, Claude Code's scratchpad, which thunc correctly
refused. The cause is `cwd=tempfile.gettempdir()` in `_run_cli` (`thunc/backends.py:135`), plus
the CLI's own injection. `--bare` removes it but works only with `ANTHROPIC_API_KEY`, not a CLI
login, and `--exclude-dynamic-system-prompt-sections` is ignored when `--system-prompt` is set.

**Fix:** for agent runs, start the CLI in the agent's `workdir`, so the injected directory is at
least the right one. Use `--bare` when an API key is present. Add a line to the agent prompt
saying the environment details the CLI adds are not the agent's environment. Keep the neutral
working directory for plain function calls.

### 4. The text protocol costs more and runs slower, even when it works (high)

- **Caching.** The transcript goes out as one growing user message, so each step writes the
  whole prefix to the cache again: 34% of input tokens were cache reads, against 92% for Claude
  Code and 86% for thunc-mcp. On `noisy_build`, each step re-sent the 18,000-character build
  output: $0.27 per run, against $0.05 for Claude Code and $0.06 for thunc-mcp.
- **One action per step.** Every read, search or edit is a full model call (about 4 s of CLI
  start-up each). Claude Code reads three files and runs the tests in one turn.
- **The model's reasoning is lost between steps.** Only the JSON actions survive into the next
  step's transcript, and they come back as part of a *user* message. The routes run that
  invented three routes read `router.py` and an empty `__init__.py`, then answered without
  opening the files that define routes. With native calls (thunc-mcp), the same model went 3/3.

Native calls (finding 1) fix all three.

### 5. `search` and `list` ignore `.gitignore` (high on real repos)

In `needle`, searching for the symbol returned **100 of 100** matches from the gitignored
`build/` folder and hit the match limit before reaching any real source. `list .` returned 300
entries: **298 under `build/`, none under `shop/`** (the actual code). Agents still passed, but
only by narrowing to `path: "shop"` after a wasted search, which works only if the junk folder
is obvious from its name. `SKIPPED_DIRS` is a fixed list (`thunc/tools.py:32`). `dist/`,
`build/`, `target/`, `.tox/`, `.mypy_cache/`, coverage output and vendored code all get through.
Claude Code's Grep (ripgrep) skips gitignored files by default.

**Fix:** respect `.gitignore` (use `git ls-files` or ripgrep when present, otherwise parse the
ignore files). Give `search` a glob or include filter, a case-insensitive option and a few lines
of context. Make `list` show one level by default, with a `depth` argument. `search` is also
about 7× slower than ripgrep on a 19,000-file tree (0.48 s against 0.07 s).

### 6. `run` has no working directory, no shell, and keeps only the end of the output (medium)

- **No working directory.** In `subdir`, the model passed `"cwd": "services/api"` to `run` 3
  times (refused as an unknown argument), and got the tests running only with
  `python -c "import os; os.chdir('services/api'); exec(open('test_api.py').read())"`. Claude
  Code did `cd services/api && python test_api.py`.
- **No shell.** Pipes, `&&`, `cd` and `VAR=value` are refused (`thunc/tools.py:61`, `:218`). In
  `noisy_build`, the one useful line was the first of 4,000, and the output keeps only the last
  18,000 characters (`thunc/tools.py:293`). The model never saw the error and had to work it out
  from `build.py` (10 turns, against Claude Code's 3, which used `| tail -20`). A test runner
  that prints its first failure early and a long cascade after it has the same problem.
- **More calls.** Without compound commands, thunc-mcp needed 11 tool calls on `deep_fix`
  (Claude Code: 3), 9 on `subdir` (3.7) and 7 on `noisy_build` (2).

**Fix:** add an optional `cwd` argument to `run`, checked against `workdir`. Keep the start
*and* the end of long output, and save the full output to a file the agent can `read` or
`search`. Consider a `shell` permission (for example `run:sh`) that allows pipes for users who
accept the risk, since `run:pytest` is already not a sandbox (the README says so).

### 7. The native API path (static review; not run here) (medium)

No `ANTHROPIC_API_KEY` was available in this environment, so `AnthropicConversation` could not be
exercised live. From the code, checked against current Messages API guidance:

- **Effort isn't set.** Claude Opus 5.5 defaults to effort `medium`. For agentic coding,
  `high` or `xhigh` is recommended. Expose `effort=` on `Agent`, with a default of `high`.
- **`max_tokens=16000`, without streaming** (`thunc/native.py:171`). A `write` of a large file,
  plus thinking, can reach the cap, and `stop_reason == "max_tokens"` ends the whole run
  (`:190`). Stream with a larger cap (around 64k), and on `max_tokens` tell the model its call
  was cut off instead of aborting.
- **Other stop reasons are fatal.** `pause_turn` and `model_context_window_exceeded` raise
  (`:193`).
- **No context management.** Tool results pile up for the whole run. Enable tool-result clearing
  (`clear_tool_uses_20250919`) or compaction for long runs.
- **Tools aren't `strict: true`.** Strict mode guarantees arguments match the schema.
- **No step budget.** `max_steps=40` ends the run without warning (`thunc/execution.py:47`). Tell
  the model how many steps are left as the limit nears (a mid-conversation system message), or
  use the task-budgets beta.

### 8. Smaller issues

- **Command-line length limit.** The CLI backends pass the system prompt as a command-line
  argument. Over about 128 KB, starting `claude` fails with a raw
  `OSError: [Errno 7] Argument list too long`, not a `ThuncError`. This is reachable within
  thunc's own caps: two 50 KB `follow=` files, 25 KB of memory and the fixed prompt come to about
  131 KB. Use `--system-prompt-file`.
- **`edit` has no "replace all" or multi-edit.** For `rename`, models wrote throwaway Python
  scripts instead of making 26 single-occurrence edits.
- **`read` on minified files.** A single 160 KB line is cut at 20,000 characters, with advice to
  "ask for less, e.g. a smaller limit", which can't help for one line. Add a character offset, or
  cut long lines individually.
- **Missing tools compared with other harnesses.** Find by file name (glob), web fetch, a plan
  or to-do tool, and sub-agents. Lower priority than the items above.

## Recommended order of work

1. Native tool calls for the `claude-code` (and `codex`) backend through an MCP server (finding
   1). The prototype shows this is enough to match Claude Code on these tasks, at lower cost.
2. Retry failed model calls inside the agent loop, with a short per-call timeout (finding 2).
3. Run the CLI in the agent's `workdir`; use `--bare` when an API key is present (finding 3).
4. Make `search` and `list` respect `.gitignore`, and add a glob filter (finding 5).
5. Add `cwd=` to `run`, keep the start and end of long output, and save the full output
   (finding 6).
6. On the API path: effort, streaming with a larger `max_tokens`, recoverable stop reasons, and
   a step-budget warning (finding 7).
7. If the text protocol stays for other servers, apply the `thunc-fixed` changes as well.

## Caveats

- Small samples: 3 runs per task on Sonnet and 1 on Opus. The direction is clear; the decimals
  are not.
- `thunc-fixed` and `thunc-mcp` are benchmark-only prototypes. `thunc-mcp` doesn't write thunc's
  session record and has no memory or custom tools.
- The `claude-code` harness uses Claude Code's default system prompt, and its costs include it
  (about 125,000 input tokens per task, against about 30,000 for thunc).
- The context leak in finding 3 was observed in a Claude Code cloud session. On a laptop the
  injected block differs (no proxy notes or attribution), but the working directory, platform
  and date are still added.
- In the README, the prompt eval reports 45/45 for Claude Code (text protocol). Its tasks are
  easier, as the README itself notes, and its table doesn't name the CLI's model. These tasks
  separate the harnesses, so they should go into that eval.
