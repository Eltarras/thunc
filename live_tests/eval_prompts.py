"""Does the agent prompt earn its place? Three small tasks, three versions of the system prompt.

    python -m live_tests.eval_prompts --backend claude-code --runs 5
    python -m live_tests.eval_prompts --backend anthropic --runs 5   # costs money; prints the token bill

Arms:
- bare:    the opening and the rules only (no working method)
- default: thunc's default prompt (opening, working method, rules)
- preset:  the task's preset (CODING, CODE_REVIEW or ANALYSIS) in place of the opening

Tasks, each in a fresh copy of a small repo:
- fix:      a failing test caused by an off-by-one. Passes when the tests pass, only the helper's
            file changed, and the test file is untouched (the agent may write anywhere).
- review:   a diff with a planted SQL injection, a removed empty-list guard and a rename.
            Passes when both real bugs are found near the right line. The rename breaks outside
            code that imports the old name, so a "minor" finding on it is fair; one marked
            "blocking" counts as a false positive.
- analysis: "where is the request timeout set, and what is it?" with a stale doc and an unused
            config file as decoys. Passes when the value and file are right, there are citations,
            and nothing changed (the agent may write anywhere).

Not part of pytest (the name doesn't start with test_). Results go to a JSON file and a table.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import thunc

PY = shlex.quote(sys.executable)
agent_module = sys.modules["thunc.agent"]  # `thunc.agent` is also the @thunc.agent decorator


# --- the fixture repos --------------------------------------------------------------------------

FIX_FILES = {
    "src/textutil.py": '''def chunks(items, size):
    """Split items into lists of at most `size` items, in order."""
    return [items[i : i + size] for i in range(0, len(items) - 1, size)]


def title_case(text):
    """Capitalise each word."""
    return " ".join(word.capitalize() for word in text.split())
''',
    "tests/test_textutil.py": """import sys

sys.path.insert(0, "src")
from textutil import chunks, title_case

assert chunks([1, 2, 3], 2) == [[1, 2], [3]], chunks([1, 2, 3], 2)
assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
assert chunks([], 3) == []
assert chunks([7], 5) == [[7]], chunks([7], 5)
assert title_case("hello big world") == "Hello Big World"
print("all tests pass")
""",
}

REVIEW_FILES = {
    "app/db.py": """import sqlite3


def connect(path):
    return sqlite3.connect(path)


def find_user(conn, name):
    query = f"SELECT id, email FROM users WHERE name = '{name}'"
    return conn.execute(query).fetchone()
""",
    "app/stats.py": """def average(values):
    return sum(values) / len(values)


def spread(values):
    return max(values) - min(values) if values else 0.0
""",
    "app/format.py": """def format_name(first, last):
    return f"{last.upper()}, {first}"


def badge(user):
    return format_name(user["first"], user["last"])
""",
}
REVIEW_DIFF = """--- a/app/db.py
+++ b/app/db.py
@@ -7,5 +7,5 @@ def connect(path):
 def find_user(conn, name):
-    query = "SELECT id, email FROM users WHERE name = ?"
-    return conn.execute(query, (name,)).fetchone()
+    query = f"SELECT id, email FROM users WHERE name = '{name}'"
+    return conn.execute(query).fetchone()
--- a/app/stats.py
+++ b/app/stats.py
@@ -1,4 +1,2 @@
 def average(values):
-    if not values:
-        return 0.0
     return sum(values) / len(values)
--- a/app/format.py
+++ b/app/format.py
@@ -1,6 +1,6 @@
-def fmt_name(first, last):
+def format_name(first, last):
     return f"{last.upper()}, {first}"


 def badge(user):
-    return fmt_name(user["first"], user["last"])
+    return format_name(user["first"], user["last"])
"""
BUGS = {"app/db.py": 9, "app/stats.py": 2}  # file -> line of the planted bug

ANALYSIS_FILES = {
    "app/settings.py": """import os

# Seconds an outgoing HTTP request may take. APP_TIMEOUT overrides it.
REQUEST_TIMEOUT = int(os.environ.get("APP_TIMEOUT", "45"))
RETRIES = 3
""",
    "app/http/client.py": """import urllib.request

from app import settings


def get(url):
    return urllib.request.urlopen(url, timeout=settings.REQUEST_TIMEOUT).read()
""",
    "config/defaults.toml": """# Not loaded by anything any more; kept for the old deploy scripts.
[http]
timeout_seconds = 30
""",
    "docs/notes.md": """# Ops notes

The HTTP timeout is 10 seconds.
""",
}


def make_repo(files: dict[str, str]) -> str:
    root = tempfile.mkdtemp(prefix="thunc-eval-")
    for path, text in files.items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
    return root


# --- the tasks ----------------------------------------------------------------------------------


@dataclass
class FixReport:
    fixed: bool
    summary: str


@dataclass
class Finding:
    file: str
    line: int
    severity: Literal["blocking", "minor"]
    problem: str


@dataclass
class TimeoutAnswer:
    value_seconds: int
    file: str
    line: int
    citations: list[str]  # where each claim comes from, like "app/settings.py:4"


def run_fix(agent: thunc.Agent) -> dict[str, Any]:
    test_file = os.path.join(agent.workdir, "tests/test_textutil.py")
    with open(test_file, encoding="utf-8") as f:
        test_before = f.read()

    @agent.task
    def fix() -> FixReport:
        """The tests in tests/test_textutil.py fail. Fix the code so they pass."""
        ...

    run = agent.run(fix)
    check = subprocess.run([sys.executable, "tests/test_textutil.py"], cwd=agent.workdir, capture_output=True)
    with open(test_file, encoding="utf-8") as f:
        test_untouched = f.read() == test_before
    ok = check.returncode == 0 and test_untouched and run.files_changed == ["src/textutil.py"]
    return {"passed": ok, "run": run, "detail": {"tests_pass": check.returncode == 0, "test_untouched": test_untouched}}


def run_review(agent: thunc.Agent) -> dict[str, Any]:
    @agent.task
    def review(diff: str) -> list[Finding]:
        """Review this change to the repository. Report the problems it introduces."""
        ...

    run = agent.run(review, REVIEW_DIFF)
    found = {
        bug: any(f.file.endswith(bug) and abs(f.line - line) <= 3 for f in run.value) for bug, line in BUGS.items()
    }
    on_rename = [f for f in run.value if f.file.endswith("format.py")]
    false_positives = sum(1 for f in on_rename if f.severity == "blocking")
    return {
        "passed": all(found.values()),
        "run": run,
        "detail": {
            "found": found,
            "false_positives": false_positives,
            "minor_on_rename": len(on_rename) - false_positives,
            "findings": len(run.value),
        },
    }


def run_analysis(agent: thunc.Agent) -> dict[str, Any]:
    @agent.task
    def request_timeout() -> TimeoutAnswer:
        """Where is the HTTP request timeout set for this app, and what is it by default, in seconds?"""
        ...

    run = agent.run(request_timeout)
    answer = run.value
    ok = (
        answer.value_seconds == 45
        and answer.file.endswith("app/settings.py")
        and bool(answer.citations)
        and not run.files_changed
    )
    return {"passed": ok, "run": run, "detail": {"answer": asdict(answer)}}


TASKS = {
    "fix": (FIX_FILES, ["write", f"run:{PY} tests/test_textutil.py"], thunc.prompts.CODING, run_fix),
    "review": (REVIEW_FILES, [], thunc.prompts.CODE_REVIEW, run_review),
    "analysis": (ANALYSIS_FILES, ["write"], thunc.prompts.ANALYSIS, run_analysis),
}
ARMS = ("bare", "default", "preset")


# --- token accounting on the Claude API ---------------------------------------------------------

usage: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "requests": 0}
usage_lock = threading.Lock()
PRICE = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}  # $/M tokens, Claude Opus 5.5


def count_usage() -> None:
    """Wrap the Anthropic client so every response's usage is added up."""
    import anthropic

    original = anthropic.Anthropic

    class Counting(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for create_owner in (self.messages, self.beta.messages):
                create_owner.create = self._counted(create_owner.create)

        @staticmethod
        def _counted(create: Any) -> Any:
            def counted(**kwargs: Any) -> Any:
                response = create(**kwargs)
                u = response.usage
                with usage_lock:
                    usage["input"] += u.input_tokens or 0
                    usage["output"] += u.output_tokens or 0
                    usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                    usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                    usage["requests"] += 1
                return response

            return counted

    anthropic.Anthropic = Counting  # type: ignore[misc]


def dollars(u: dict[str, int]) -> float:
    return sum(u[k] * PRICE[k] for k in PRICE) / 1_000_000


# --- running -----------------------------------------------------------------------------------


def one(task: str, arm: str, n: int, backend: str) -> dict[str, Any]:
    files, permissions, preset, check = TASKS[task]
    repo = make_repo(files)
    agent = thunc.Agent(
        f"eval-{task}-{arm}-{n}",
        workdir=repo,
        permissions=permissions,
        system=preset if arm == "preset" else None,
        backend=backend,
        max_steps=25,
        timeout=600,
    )
    started = time.monotonic()
    try:
        outcome = check(agent)
        run = outcome.pop("run")
        return {"task": task, "arm": arm, "n": n, **outcome, "steps": run.steps, "seconds": run.seconds}
    except thunc.AgentError as exc:
        return {
            "task": task,
            "arm": arm,
            "n": n,
            "passed": False,
            "steps": exc.run.steps,
            "seconds": exc.run.seconds,
            "detail": {"error": str(exc)[:300]},
        }
    except Exception as exc:  # a crash in one run shouldn't lose the others
        return {
            "task": task,
            "arm": arm,
            "n": n,
            "passed": False,
            "steps": 0,
            "seconds": round(time.monotonic() - started, 1),
            "detail": {"error": f"{type(exc).__name__}: {exc}"[:300]},
        }
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", default=None, help="JSON file for the results")
    args = parser.parse_args()
    thunc.configure(agents_dir=tempfile.mkdtemp(prefix="thunc-eval-agents-"))
    if args.backend == "anthropic":
        count_usage()

    real_method = agent_module.method
    results = []
    for arm in args.arms.split(","):
        # The bare arm drops the working method; arms run one after another so the swap stays contained.
        agent_module.method = (lambda tools: "") if arm == "bare" else real_method
        before = dict(usage)
        jobs = [(task, arm, n) for task in args.tasks.split(",") for n in range(1, args.runs + 1)]
        arm_results = thunc.map(lambda job: one(*job, args.backend), jobs, workers=args.workers)
        results += arm_results
        spent = {k: usage[k] - before[k] for k in usage}
        print(
            f"{arm}: done ({sum(r['passed'] for r in arm_results)}/{len(arm_results)} passed"
            + (f", ${dollars(spent):.2f}" if args.backend == "anthropic" else "")
            + ")",
            flush=True,
        )
    agent_module.method = real_method

    print(f"\n{'task':9} {'arm':8} {'pass':>6} {'steps':>6} {'secs':>6}  notes")
    for task in args.tasks.split(","):
        for arm in args.arms.split(","):
            rows = [r for r in results if r["task"] == task and r["arm"] == arm]
            passed = sum(r["passed"] for r in rows)
            steps = sum(r["steps"] for r in rows) / len(rows)
            secs = sum(r["seconds"] for r in rows) / len(rows)
            notes = ""
            if task == "review":
                blocking = sum(r["detail"].get("false_positives", 0) for r in rows)
                minor = sum(r["detail"].get("minor_on_rename", 0) for r in rows)
                notes = f"rename flagged: {minor} minor, {blocking} blocking (false positive)"
            errors = sum(1 for r in rows if "error" in r["detail"])
            if errors:
                notes += f" errors {errors}"
            print(f"{task:9} {arm:8} {passed:>3}/{len(rows):<2} {steps:>6.1f} {secs:>6.0f}  {notes}")
    if args.backend == "anthropic":
        print(f"\nClaude API: {usage['requests']} requests, ${dollars(usage):.2f} at Opus 5.5 prices; tokens {usage}")
    out = args.out or os.path.join(tempfile.gettempdir(), f"thunc-eval-{args.backend}-{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"backend": args.backend, "runs": args.runs, "usage": usage, "results": results}, f, indent=2, default=str
        )
    print(f"results: {out}")


if __name__ == "__main__":
    main()
