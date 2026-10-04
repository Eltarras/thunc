"""Agents: a read-only guide to this repository, running on your Codex login.

Each task is one agent run: the model lists, searches and reads files in the repo, then returns a
typed value. The trace shows every step it took.

Run from the repo root:  python3 -m examples.repo_guide
Another backend:         THUNC_BACKEND=claude-code python3 -m examples.repo_guide
"""

import json
import os
import tempfile
from dataclasses import dataclass

import thunc

trace_file = os.path.join(tempfile.gettempdir(), "thunc_repo_guide.jsonl")
if os.path.exists(trace_file):
    os.remove(trace_file)
thunc.configure(backend=os.environ.get("THUNC_BACKEND", "codex"), trace=trace_file)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this checkout, wherever you run from

guide = thunc.Agent(
    "thunc-guide",
    workdir=REPO,
    system="You help new contributors find their way around a Python library. Be precise and brief.",
    max_steps=20,
)


@dataclass
class Location:
    file: str
    line: int
    summary: str  # one sentence


@guide.task
def find_definition(name: str) -> Location:
    """Find where this name is defined in the thunc package (the thunc/ folder, not the tests).
    Return the file, the line where it's defined, and one sentence on what it does."""
    ...


@guide.task
def backends() -> list[str]:
    """List the backends thunc supports, by the names a user passes to configure(backend=...)."""
    ...


@guide.task(ensure=lambda paths: bool(paths) and all(p.startswith("tests/") for p in paths))
def tests_for(feature: str) -> list[str]:
    """Find the test files that cover this feature. Return their paths, relative to the repo root."""
    ...


def steps(task_name: str) -> str:
    """How the last run of a task went, from the trace: the tools it called, in order."""
    with open(trace_file, encoding="utf-8") as f:
        runs = [json.loads(line) for line in f]
    run = [r for r in runs if r["function"].endswith(task_name)][-1]
    tools = []
    for answer in run["answers"]:
        try:
            tools.append(json.loads(answer)["tool"])
        except (ValueError, KeyError, TypeError):
            tools.append("?")  # a reply that wasn't one JSON action; thunc sent it back to be fixed
    return f"{len(tools)} steps, {run['seconds']:.0f}s: {' -> '.join(tools)}"


where = find_definition("clear_cache")
print(f"clear_cache is defined in {where.file}:{where.line}\n  {where.summary}")
print(f"  ({steps('find_definition')})\n")

print("Backends:", ", ".join(backends()))
print(f"  ({steps('backends')})\n")

print("Tests for the answer cache:", ", ".join(tests_for("the answer cache: saving, reusing and clearing answers")))
print(f"  ({steps('tests_for')})")
print(f"\nFull trace, with every model reply: {trace_file}")
