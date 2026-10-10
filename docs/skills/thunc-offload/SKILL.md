---
name: thunc-offload
description: Hand long, self-contained work to a thunc agent or typed thunc functions on a cheaper model instead of doing it turn by turn in this session. Use when a task with a checkable outcome (tests pass, a typed answer) would likely take this session more than about 10 turns, such as a wide refactor or rename, a debugging hunt without an obvious lead, or a feature with tests to write; or when the same judgment must be applied to many items (classify, extract, triage or summarize more than about 10 files, log lines, rows or tickets). Also use when the user asks to offload or delegate to thunc.
---

# Offloading to thunc

[thunc](https://eltarras.github.io/thunc/) runs typed Python functions and agents on a model you
choose. A thunc agent on the `claude-code` backend does a coding task for far less than Claude Code
does the same task: 78% less on Opus 5.5 and 73% less on Sonnet 5.5 in thunc's tool-use benchmark,
because its prompt is about a fifth of the size. The work also stays out of this session's context:
you read back a small typed result, not every file and command output.

This session still has its own part to do: writing the task, waiting for the run and checking the
result. Counting that, handing a multi-module feature to a Sonnet agent from Opus came out 21%
cheaper overall than Opus doing it alone, with the same pass rate on hidden tests. On small fixes
that part costs about what handing off saves, so do small tasks yourself.

Run scripts with `uv run --no-project --with "thunc>=0.3" python <script>`, or with a Python that
has thunc installed (`pip install thunc`). Write scripts to a scratch or temp folder, never into
the user's repo. Set `THUNC_AGENTS_DIR=~/.cache/thunc/agents` for agent runs, so their records stay
out of the repo.

## When not to offload

- The task needs this conversation's context, a design decision, or the user's input.
- It's small: one obvious fix, a few files, under about 10 items.
- The outcome can't be checked: no tests to run, no typed answer to verify.

## Agentic work

```python
import thunc
from dataclasses import dataclass


@dataclass
class Fix:
    root_causes: list[str]  # one line per bug
    files_changed: list[str]
    tests_passing: bool


agent = thunc.Agent(
    "fix-tests",  # one name per kind of job
    workdir="/abs/path/to/repo",
    permissions=["write", "run:python", "run:pytest"],  # the narrowest that works
    backend="claude-code",
    model="sonnet",
    follow=True,  # gives it the repo's CLAUDE.md / AGENTS.md
    max_steps=40,
    timeout=900,
)


@agent.task
def fix_failing_tests() -> Fix:
    """Run the test suite with `python -m pytest -q`, find why tests fail, and fix the source code
    (not the tests). Rerun until everything passes. Report each root cause in one line."""
    ...


run = agent.run(fix_failing_tests)
print(run.value)
print("changed:", run.files_changed, "denied:", run.denied, "steps:", run.steps, f"{run.seconds:.0f}s")
```

- The docstring must hold everything the agent needs, because it can't see this conversation:
  the goal, the constraints (don't touch the tests, keep the public API), and how to check the result.
- Return a dataclass with what you need to continue, not prose.
- Permissions: `write`, `write:src/**`, `run:pytest`, `run:git diff`, `shell` (pipes, `&&`),
  `!read:.env*`. Don't grant `run:git commit` or `run:git push` unless the user asked.
- Start from a clean git tree, or note what was already modified. Afterwards, read the diff and
  run the tests once yourself instead of trusting the agent's report.
- A failed run raises `thunc.AgentError`, whose `.run` holds the record. If the agent fails twice,
  do the task yourself.
- Wait for the run to finish before you end your turn: run the script in the foreground with a
  long Bash timeout (up to 10 minutes). If it may take longer, start it in the background and wait
  for it with a loop that ends when it exits. Don't end your turn and rely on being notified: a
  headless session (`claude -p`) exits when the turn ends, and the agent is cut off mid-run.

## Many items

```python
from typing import Literal


@thunc.function(cache=True, backend="claude-code", model="haiku")
def triage(line: str) -> Literal["bug", "noise", "config"]:
    """Classify this log line by what it indicates."""
    ...


results = thunc.map(triage, items, workers=8)
```

- Print only the compact result (counts, the few items that matter), not every answer.
- `cache=True` only for one-answer-per-input functions (classify, extract, score), never for drafting.
- File and user content goes in the inputs, never into an f-string of instructions.
- If the task is really a rule (parsing, formatting), use `@thunc.function(write=True)` so it
  becomes plain Python.

## Reporting

Tell the user what was offloaded, to which model, and the outcome you verified.
