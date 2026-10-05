"""Tool-use benchmark: where does the harness, not the model, cost an agent the task?

    python -m live_tests.bench_tooluse --harness thunc,claude-code --runs 3 --model claude-sonnet-5-5
    python -m live_tests.bench_tooluse --harness thunc-fixed --runs 3   # the prototype on its own
    python -m live_tests.bench_tooluse --report results.json

Needs a logged-in `claude` CLI. Findings from the first runs: live_tests/bench_tooluse_report.md.

Each task is a small repository built to press on one part of an agent harness: navigating a
big repo with generated junk in it, paging through a long file, writing escape-heavy code,
editing tab-indented near-duplicates, reading a command whose real error scrolls off the top,
running tests that need another working directory, renaming across many files, and returning a
nested structured result.

Harnesses, on the same model:
- thunc:       a thunc.Agent on the claude-code backend with its defaults (native tool calls
               through an MCP server), permissions ["write", "run"] so nothing is held back by rules.
- thunc-text:  the same with protocol="text" (one JSON action per reply as text).
- thunc-shell: the default, with permissions ["write", "shell"]: commands run in a shell.
- thunc-fixed: the text protocol with two fixes patched in (lenient reading of the first action,
               stopping the CLI once an action has arrived). A prototype; it
               patches the text protocol for the whole process, so run it on its own.
- claude-code: Claude Code itself (`claude -p` with Read, Edit, Write, Bash, Grep, Glob), as the
               reference harness.

Every run records whether it passed, the model turns, wall time, tokens and cost, and every tool
call that came back as an error, so failures can be traced to a harness cause. Results go to a
JSON file; `--report` prints the tables from one.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import thunc
from thunc import backends
from thunc.claude_code import ClaudeCodeConversation
from thunc.schema import describe, parse

PY = sys.executable


# --- fixture repos ------------------------------------------------------------------------------


def write_files(root: str, files: dict[str, str]) -> None:
    for path, text in files.items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full) or root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)


def git_init(root: str) -> None:
    run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, check=True)  # noqa: E731
    run("init", "-q")
    run("-c", "user.email=b@b", "-c", "user.name=bench", "add", "-A")
    run("-c", "user.email=b@b", "-c", "user.name=bench", "commit", "-qm", "init")


# 1. needle: a 600-module repo, with a stale build/ copy (gitignored) that comes first in tree order.
def needle_files() -> dict[str, str]:
    files = {".gitignore": "build/\n__pycache__/\n"}
    files["README.md"] = "# shop\n\nShipping fees: see docs/fees.md.\n"
    files["docs/fees.md"] = "International orders pay a 10% surcharge.\n"  # stale
    files["shop/__init__.py"] = ""
    files["shop/rates.py"] = (
        "from shop.pkg_33 import mod_07 as _surcharge\n\nINTL_SURCHARGE_RATE = _surcharge.BASE_RATE\n"
        "DOMESTIC_RATE = 0.0\n"
    )
    for p in range(40):
        files[f"shop/pkg_{p:02d}/__init__.py"] = ""
        for m in range(15):
            body = [f'"""Module {p}.{m}."""', "from shop import rates", ""]
            for k in range(6):
                body.append(f"def helper_{p}_{m}_{k}(x):\n    return x * {k + 1} + {p}\n")
            if (p * 15 + m) % 4 == 0:
                body.append(f"def fee_{p}_{m}(total):\n    return total * (1 + rates.INTL_SURCHARGE_RATE)\n")
            files[f"shop/pkg_{p:02d}/mod_{m:02d}.py"] = "\n".join(body)
    files["shop/pkg_33/mod_07.py"] += "\n# Agreed with finance, 2026-03.\nBASE_RATE = 0.137\n"
    files["shop/pkg_12/mod_03.py"] += (
        "\ndef compute_shipping_fee(total, international):\n"
        "    if international:\n        return round(total * (1 + rates.INTL_SURCHARGE_RATE), 2)\n"
        "    return round(total * (1 + rates.DOMESTIC_RATE), 2)\n"
    )
    for path, text in list(files.items()):  # a stale build output, ignored by git, first in tree order
        if path.startswith("shop/"):
            files["build/lib/" + path] = text.replace("BASE_RATE = 0.137", "BASE_RATE = 0.09")
    files["build/lib/shop/rates.py"] = "INTL_SURCHARGE_RATE = 0.09\nDOMESTIC_RATE = 0.0\n"
    return files


# 2. deep_fix: the bug is in a helper near line 1900 of a 2400-line file.
def deep_fix_files() -> dict[str, str]:
    lines = ['"""Ledger rules, generated."""', ""]
    for n in range(470):
        if n == 380:
            lines += [
                "def _round_cents(amount):",
                '    """Round to whole cents, half up."""',
                "    return int(amount * 100) / 100",
                "",
            ]
        lines += [f"def rule_{n:04d}(x):", f'    """Rule {n}."""', f"    return x * {1 + n / 10000:.4f}", ""]
    lines += [
        "def apply_discount(total, percent):",
        '    """Total after a percentage discount, in whole cents."""',
        "    return _round_cents(total * (100 - percent) / 100)",
        "",
    ]
    test = (
        "import sys\nsys.path.insert(0, 'src')\nfrom ledger import apply_discount, rule_0001\n\n"
        "assert rule_0001(10) == 10.001\n"
        "assert apply_discount(100, 10) == 90.0\n"
        "assert apply_discount(10.01, 5) == 9.51, apply_discount(10.01, 5)\n"
        "assert apply_discount(3.33, 33) == 2.23, apply_discount(3.33, 33)\n"
        "assert apply_discount(19.99, 0) == 19.99, apply_discount(19.99, 0)\n"
        "print('ok')\n"
    )
    return {"src/ledger.py": "\n".join(lines), "tests/test_ledger.py": test}


# 3. rename: get_usr -> get_user across 12 modules, tests and docs.
def rename_files() -> dict[str, str]:
    files = {
        "app/__init__.py": "",
        "app/users.py": (
            "USERS = {1: 'ada', 2: 'grace'}\n\n\ndef get_usr(user_id):\n"
            '    """The name of a user, or None."""\n    return USERS.get(user_id)\n'
        ),
        "README.md": "# app\n\nUse `app.users.get_usr(id)` to look a user up.\n",
    }
    for n in range(12):
        calls = "\n".join(f"    names.append(get_usr({k}))" for k in range(1 + n % 3))
        files[f"app/views_{n:02d}.py"] = (
            f"from app.users import get_usr\n\n\ndef view_{n}():\n    names = []\n{calls}\n    return names\n"
        )
    files["tests/run_tests.py"] = (
        "import sys, os\nsys.path.insert(0, os.getcwd())\nfrom app.users import get_user\n"
        "from app import " + ", ".join(f"views_{n:02d}" for n in range(12)) + "\n"
        "assert get_user(1) == 'ada'\nassert get_user(3) is None\n"
        + "".join(f"assert views_{n:02d}.view_{n}()[0] == None or True\n" for n in range(12))
        + "print('ok')\n"
    )
    return files


# 4. write_tricky: a new module whose code is mostly backslashes and quotes.
ESCAPE_TESTS = r"""import sys
sys.path.insert(0, "src")
from escapes import shell_quote, unescape_c

assert shell_quote("abc") == "'abc'"
assert shell_quote("") == "''"
assert shell_quote("it's") == "'it'\"'\"'s'", shell_quote("it's")
assert shell_quote('a "b" \\c') == "'a \"b\" \\c'"
assert unescape_c(r"a\nb") == "a\nb"
assert unescape_c(r"tab\there") == "tab\there"
assert unescape_c(r"q\"q") == 'q"q'
assert unescape_c(r"back\\slash") == "back\\slash"
assert unescape_c(r"\x41\x42") == "AB"
assert unescape_c(r"end\\") == "end\\"
print("ok")
"""
ESCAPE_HIDDEN = r"""import sys
sys.path.insert(0, "src")
from escapes import shell_quote, unescape_c
assert shell_quote("'") == "''\"'\"''"
assert unescape_c(r"\\n") == "\\n"
assert unescape_c(r"\x7e\t") == "~\t"
assert unescape_c("plain") == "plain"
print("ok")
"""


def write_tricky_files() -> dict[str, str]:
    return {"tests/test_escapes.py": ESCAPE_TESTS, "src/README.md": "Put escapes.py here.\n"}


# 5. tabs: tab-indented near-duplicate functions; the bug is in the third.
def tabs_files() -> dict[str, str]:
    code = (
        "def area_rect(w, h):\n"
        "\tif w < 0 or h < 0:\n\t\traise ValueError('negative size')\n"
        "\tresult = w * h\n\treturn result\n\n\n"
        "def area_tri(b, h):\n"
        "\tif b < 0 or h < 0:\n\t\traise ValueError('negative size')\n"
        "\tresult = b * h / 2\n\treturn result\n\n\n"
        "def area_trap(a, b, h):\n"
        "\tif a < 0 or b < 0 or h < 0:\n\t\traise ValueError('negative size')\n"
        "\tresult = (a + b) * h\n\treturn result\n"
    )
    test = (
        "import sys\nsys.path.insert(0, 'src')\nfrom geometry import *\n"
        "assert area_rect(2, 3) == 6\nassert area_tri(2, 3) == 3\n"
        "assert area_trap(2, 4, 3) == 9, area_trap(2, 4, 3)\nprint('ok')\n"
    )
    return {"src/geometry.py": code, "tests/test_geometry.py": test}


# 6. noisy_build: the one real error is the first line of 4000; the end is only cascade warnings.
BUILD_PY = """import json, sys

RULES = {
    "name": str, "version": str, "target": ("linux", "macos", "windows"), "jobs": int,
    "optimization_level": (0, 1, 2, 3), "debug": bool, "strip": bool, "lto": bool,
    "output_dir": str, "cache": bool, "sanitizers": list, "features": list,
}

def main():
    with open("config/build.json") as f:
        config = json.load(f)
    errors = []
    for key, rule in RULES.items():
        value = config.get(key)
        ok = value in rule if isinstance(rule, tuple) else isinstance(value, rule)
        if isinstance(rule, tuple) and isinstance(value, bool):
            ok = False
        if not ok:
            errors.append(f"ERROR config/build.json: {key!r} is invalid: {value!r}")
    for line in errors:
        print(line)
    if errors:
        for n in range(4000):
            print(f"warning: module_{n:04d} skipped: configuration invalid (see the first error above)")
        print(f"BUILD FAILED: {len(errors)} error(s), 4000 warning(s)")
        sys.exit(1)
    print("BUILD OK")

main()
"""


def noisy_build_files() -> dict[str, str]:
    config = {
        "name": "app", "version": "1.4.0", "target": "linux", "jobs": 8, "optimization_level": "2",
        "debug": False, "strip": True, "lto": True, "output_dir": "out", "cache": True,
        "sanitizers": [], "features": ["net", "tls"],
    }  # fmt: skip
    return {"build.py": BUILD_PY, "config/build.json": json.dumps(config, indent=2) + "\n"}


# 7. subdir: the service's tests read fixtures by a path relative to services/api.
def subdir_files() -> dict[str, str]:
    return {
        "services/api/handlers.py": (
            "def active_users(users):\n"
            '    """Names of the active users, sorted."""\n'
            "    return sorted(u['name'] for u in users if u.get('active') == 'true')\n"
        ),
        "services/api/test_api.py": (
            "import json\nfrom handlers import active_users\n\n"
            "with open('fixtures/users.json') as f:\n    users = json.load(f)\n"
            "assert active_users(users) == ['ada', 'linus'], active_users(users)\nprint('ok')\n"
        ),
        "services/api/fixtures/users.json": json.dumps(
            [
                {"name": "linus", "active": True},
                {"name": "ada", "active": True},
                {"name": "bob", "active": False},
            ]
        ),
        "README.md": "Run each service's tests from its own folder: cd services/api && python test_api.py\n",
    }


# 8. routes: a nested structured answer gathered from several files, with decoys.
def routes_files() -> dict[str, str]:
    return {
        "web/router.py": "class Router:\n    def get(self, p):\n        return lambda f: f\n"
        "    def post(self, p):\n        return lambda f: f\n    def add(self, m, p, f):\n        pass\n\n"
        "router = Router()\n",
        "web/users.py": "from web.router import router\n\n\n@router.get('/users')\ndef list_users():\n    pass\n\n\n"
        "@router.get('/users/<id>')\ndef show_user(id):\n    pass\n\n\n"
        "# @router.get('/users/<id>/delete')  (removed in 2.0)\n# def delete_user(id): ...\n",
        "web/auth.py": "from web.router import router\n\n\ndef login():\n    pass\n\n\ndef logout():\n    pass\n\n\n"
        "router.add('POST', '/login', login)\nrouter.add('POST', '/logout', logout)\n",
        "web/orders.py": "from web.router import router\n\n\n@router.post('/orders')\n"
        "def create_order():\n    pass\n\n\n@router.get(\n    '/orders/<id>'\n)\ndef show_order(id):\n    pass\n",
        "tests/test_routes.py": "from web.router import router\n\n\n@router.get('/test-only')\n"
        "def probe():\n    pass\n",
        "web/__init__.py": "",
    }


ROUTES = {
    ("GET", "/users", "list_users"),
    ("GET", "/users/<id>", "show_user"),
    ("POST", "/login", "login"),
    ("POST", "/logout", "logout"),
    ("POST", "/orders", "create_order"),
    ("GET", "/orders/<id>", "show_order"),
}


@dataclass
class Route:
    method: str
    path: str
    handler: str


# --- tasks --------------------------------------------------------------------------------------


def py(root: str, *args: str, cwd: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run([PY, *args], cwd=os.path.join(root, cwd), capture_output=True, text=True, timeout=60)


def unchanged(root: str, files: dict[str, str], path: str) -> bool:
    with open(os.path.join(root, path), encoding="utf-8") as f:
        return f.read() == files[path]


@dataclass
class Task:
    name: str
    files: Callable[[], dict[str, str]]
    instructions: str
    returns: Any
    check: Callable[[str, dict[str, str], Any], tuple[bool, str]]  # (root, files, value) -> (passed, why)
    git: bool = False


def check_needle(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    return isinstance(value, (int, float)) and abs(value - 0.137) < 1e-9, f"answered {value!r}"


def check_deep(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    ok = py(root, "tests/test_ledger.py").returncode == 0 and unchanged(root, files, "tests/test_ledger.py")
    return ok, "tests pass" if ok else "tests fail or test edited"


def check_rename(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    left = [p for p in files if "get_usr" in open(os.path.join(root, p), encoding="utf-8").read()]
    tests = py(root, "tests/run_tests.py").returncode == 0
    return not left and tests, f"get_usr left in {left}; tests {'pass' if tests else 'fail'}"


def check_tricky(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    if not os.path.exists(os.path.join(root, "src/escapes.py")):
        return False, "src/escapes.py missing"
    visible = py(root, "tests/test_escapes.py").returncode == 0
    write_files(root, {"tests/_hidden.py": ESCAPE_HIDDEN})
    hidden = py(root, "tests/_hidden.py").returncode == 0
    untouched = unchanged(root, files, "tests/test_escapes.py")
    return visible and hidden and untouched, f"visible={visible} hidden={hidden} test_untouched={untouched}"


def check_tabs(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    text = open(os.path.join(root, "src/geometry.py"), encoding="utf-8").read()
    ok = py(root, "tests/test_geometry.py").returncode == 0
    tabs_kept = "\tresult = w * h" in text and not re.search(r"^ +\S", text, re.M)
    return ok and tabs_kept, f"tests={'pass' if ok else 'fail'} tabs_kept={tabs_kept}"


def check_build(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    ok = "BUILD OK" in py(root, "build.py").stdout and unchanged(root, files, "build.py")
    return ok, "builds" if ok else "still fails or build.py edited"


def check_subdir(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    ok = py(root, "test_api.py", cwd="services/api").returncode == 0
    untouched = unchanged(root, files, "services/api/test_api.py") and unchanged(
        root, files, "services/api/fixtures/users.json"
    )
    return ok and untouched, f"tests={'pass' if ok else 'fail'} untouched={untouched}"


def check_routes(root: str, files: dict[str, str], value: Any) -> tuple[bool, str]:
    got = {(r.method.upper(), r.path, r.handler) for r in value} if isinstance(value, list) else set()
    return got == ROUTES, f"missing {sorted(ROUTES - got)} extra {sorted(got - ROUTES)}"


TASKS = {
    t.name: t
    for t in [
        Task(
            "needle",
            needle_files,
            "What international surcharge rate does compute_shipping_fee apply? Give the rate the code "
            "actually uses, as a fraction (0.25 for 25%).",
            float,
            check_needle,
            git=True,
        ),
        Task(
            "deep_fix",
            deep_fix_files,
            "tests/test_ledger.py fails (run it with `python tests/test_ledger.py`). Fix the code so it "
            "passes, without changing the test. Return a one-line summary of the fix.",
            str,
            check_deep,
        ),
        Task(
            "rename",
            rename_files,
            "Rename the function get_usr to get_user everywhere in this repository: the definition, every "
            "caller and the docs. tests/run_tests.py already expects the new name; run it with "
            "`python tests/run_tests.py` to check. Return a one-line summary.",
            str,
            check_rename,
        ),
        Task(
            "write_tricky",
            write_tricky_files,
            "Create src/escapes.py with shell_quote(s) (POSIX single-quote quoting) and unescape_c(s) "
            '(turns C escapes \\n \\t \\\\ \\" and \\xHH into the characters) so that '
            "`python tests/test_escapes.py` passes. Don't change the test. Return a one-line summary.",
            str,
            check_tricky,
        ),
        Task(
            "tabs",
            tabs_files,
            "`python tests/test_geometry.py` fails. Fix the bug in src/geometry.py without changing its "
            "indentation style or the test. Return a one-line summary.",
            str,
            check_tabs,
        ),
        Task(
            "noisy_build",
            noisy_build_files,
            "`python build.py` fails. Find out why and fix config/build.json so the build succeeds. Don't "
            "change build.py. Return a one-line summary of the cause.",
            str,
            check_build,
        ),
        Task(
            "subdir",
            subdir_files,
            "The API service's tests fail. Fix the bug in the service code (not the tests or fixtures) and "
            "check that the tests pass. Return a one-line summary.",
            str,
            check_subdir,
        ),
        Task(
            "routes",
            routes_files,
            "List every HTTP route the web app registers at runtime: method, path and the handler "
            "function's name. Leave out anything that isn't registered when the app runs.",
            list[Route],
            check_routes,
        ),
    ]
}


# --- harness: thunc -----------------------------------------------------------------------------

_local = threading.local()
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
_real_run_cli = backends._run_cli


def _counting_run_cli(args: list[str], text: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """backends._run_cli, also adding up the usage `claude -p --output-format json` reports."""
    started = time.monotonic()
    proc = _real_run_cli(args, text, timeout)
    meter = getattr(_local, "meter", None)
    if meter is not None:
        meter["calls"] += 1
        meter["call_seconds"].append(round(time.monotonic() - started, 2))
        meter["prompt_chars"].append(len(text) + _system_chars(args))
        try:
            data = json.loads(proc.stdout)
            u = data.get("usage") or {}
            meter["cost"] += data.get("total_cost_usd") or 0
            for key in TOKEN_KEYS:
                meter[key] += u.get(key) or 0
        except (ValueError, AttributeError):
            pass
    return proc


_real_cli_events = backends._cli_events


def _counting_cli_events(args: list[str], text: str, timeout: float, last: str, stop: Any = None) -> Any:
    """backends._cli_events, also adding up the usage of a streamed `claude -p` step (the text
    protocol stops it once an action is complete, so there may be no result event to read it from)."""
    started = time.monotonic()
    events, code, stderr = _real_cli_events(args, text, timeout, last, stop)
    meter = getattr(_local, "meter", None)
    if meter is not None and args[0] == "claude":
        meter["calls"] += 1
        meter["call_seconds"].append(round(time.monotonic() - started, 2))
        meter["prompt_chars"].append(len(text) + _system_chars(args))
        usage: collections.Counter[str] = collections.Counter()
        for event, _ in events:
            inner = event.get("event") or {}
            if inner.get("type") == "message_start":
                usage.update(
                    {k: v for k, v in (inner.get("message", {}).get("usage") or {}).items() if isinstance(v, int)}
                )
            elif inner.get("type") == "message_delta":
                usage["output_tokens"] += (inner.get("usage") or {}).get("output_tokens") or 0
        result = next((e for e, _ in events if e.get("type") == "result"), None)
        meter["early_stops"] += result is None and code is None
        model = args[args.index("--model") + 1] if "--model" in args else ""
        prices = _PRICES.get(model, _PRICES["claude-sonnet-5-5"])
        for key in TOKEN_KEYS:
            meter[key] += usage[key]
        meter["cost"] += sum(usage[key] * price for key, price in zip(TOKEN_KEYS, prices, strict=True)) / 1e6
    return events, code, stderr


def _system_chars(args: list[str]) -> int:
    """The length of the system prompt a `claude -p` call was given, on its command line or in a file."""
    if "--system-prompt-file" in args:  # still there: claude_code removes it when it returns
        with open(args[args.index("--system-prompt-file") + 1], encoding="utf-8") as f:
            return len(f.read())
    return len(args[args.index("--system-prompt") + 1]) if "--system-prompt" in args else 0


def run_thunc(task: Task, root: str, model: str, max_steps: int, n: int, variant: str = "thunc") -> dict[str, Any]:
    meter = collections.Counter()  # type: ignore[var-annotated]
    meter["call_seconds"], meter["prompt_chars"] = [], []  # type: ignore[assignment]
    _local.meter = meter
    protocol, permissions = THUNC_VARIANTS[variant]
    agent = thunc.Agent(
        f"bench-{task.name}-{n}-{os.getpid()}-{threading.get_ident()}",
        workdir=root,
        permissions=permissions,
        protocol=protocol,
        backend="claude-code",
        model=model,
        max_steps=max_steps,
        timeout=1200,
    )
    value, error, run = None, None, None
    try:
        run = agent.run(_retyped(agent, task))
        value = run.value
    except thunc.AgentError as exc:
        run, error = exc.run, str(exc)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _local.meter = None
    steps = _session_steps(run.session) if run else []
    return {
        "value": value,
        "error": error,
        "turns": run.steps if run else 0,
        "seconds": run.seconds if run else 0,
        "tool_calls": [s for s in steps if s.get("tool")],
        "nudges": [s for s in steps if not s.get("tool") and s.get("result", "").startswith("error")],
        "usage": {k: v for k, v in meter.items() if not isinstance(v, list)},
        "call_seconds": meter["call_seconds"],
        "prompt_chars": meter["prompt_chars"],
    }


def _counting_close(self: Any) -> None:
    """ClaudeCodeConversation.close, also adding up the cost and tokens its CLI reported."""
    meter = getattr(_local, "meter", None)
    if meter is not None:
        usage = self.usage
        prices = _PRICES.get(self.model or "", _PRICES["claude-sonnet-5-5"])
        meter["calls"] += len(self.replies)
        meter["cost"] += sum(usage[key] * price for key, price in zip(TOKEN_KEYS, prices, strict=True)) / 1e6
        for key in TOKEN_KEYS:
            meter[key] += usage[key]
    _real_close(self)


_real_close = ClaudeCodeConversation.close


def _retyped(agent: thunc.Agent, task: Task) -> Callable[..., Any]:
    """A task for this agent with the benchmark task's return type (annotations are read at decoration)."""

    def job() -> None: ...

    job.__annotations__["return"] = task.returns
    job.__doc__ = task.instructions
    return agent.task(job)


def _session_steps(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("event") == "step":
                out.append(entry)
    return out


# --- harness: thunc-fixed (a prototype of the recommended text-protocol fixes) -------------------
#
# Two changes, applied as patches for this process only, so the same tasks can be run with and
# without them. Not part of thunc: they're here to measure how much of the gap they close. (A third,
# several actions per reply as a JSON array, is in thunc's text protocol now.)
#   1. Lenient reading: the first complete action (or array of actions) in a reply is used,
#      whatever text, markup or invented "results" come before or after it.
#   2. Stop early: the CLI's output is streamed, and the process is stopped as soon as a complete
#      action has arrived, so a model that carries on (inventing the tool's result) can't run on
#      to the timeout. A failed or empty CLI call is retried once.

_ACTION_START = re.compile(r'(\[\s*)?\{\s*"tool"\s*:')
_PRICES = {  # $/M tokens: input, output, cache read, cache write (the CLI writes 1-hour entries: 2x input)
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 4.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 8.0),
}


def find_actions(text: str, final: bool = True) -> tuple[list[dict[str, Any]], int] | None:
    """The first complete action (or array of actions) in text, and where it ends; None if none yet.
    While the text is still streaming (final=False), an array that has begun is waited for."""
    decoder = json.JSONDecoder(strict=False)  # literal newlines in strings are accepted
    for match in _ACTION_START.finditer(text):
        try:
            value, end = decoder.raw_decode(text, match.start())
        except ValueError:
            if match.group(1) and not final:
                return None
            continue
        items = value if isinstance(value, list) else [value]
        if items and all(isinstance(item, dict) and isinstance(item.get("tool"), str) for item in items):
            return items, end
    return None


def _patched_next(self: Any) -> Any:
    from thunc import native

    answer = native._send(self._transcript(), self.system, self.backend, self.model)
    found = find_actions(answer)
    if found is None:
        try:
            return native.Reply(native.actions(answer, self.names), answer)  # or its error message
        except ValueError as problem:
            return native.Reply([], answer, str(problem))
    calls = []
    for item in found[0]:
        tool = item["tool"]
        args = item["args"] if "args" in item else {k: v for k, v in item.items() if k != "tool"}
        if tool not in self.names:
            return native.Reply([], answer, f"unknown tool {tool!r}; use one of {self.names}")
        calls.append(native.Call(None, tool, args if isinstance(args, dict) else {}))
    return native.Reply(calls, answer[: found[1]])


def streaming_claude_code(text: str, *, system: str, model: str | None, api_key: str | None, timeout: float) -> str:
    """The claude-code backend, streamed: returns as soon as a complete action has arrived."""
    for attempt in range(2):
        try:
            return _stream_once(text, system, model, min(timeout, 180))
        except thunc.ThuncError:
            if attempt:
                raise
    raise AssertionError("unreachable")


def _stream_once(text: str, system: str, model: str | None, timeout: float) -> str:
    args = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            "--tools", "", "--strict-mcp-config", "--system-prompt", system, "--no-session-persistence"]  # fmt: skip
    if model:
        args += ["--model", model]
    started = time.monotonic()
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, cwd=tempfile.gettempdir(), start_new_session=True)  # fmt: skip
    assert proc.stdin and proc.stdout
    proc.stdin.write(text)
    proc.stdin.close()
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(iter(proc.stdout.readline, "")), daemon=True)  # type: ignore[union-attr]
    reader.start()
    seen, streamed, result, usage = 0, "", None, collections.Counter()  # type: ignore[var-annotated]
    stopped_early = False
    while True:
        done = not reader.is_alive()
        while seen < len(lines):
            line = lines[seen]
            seen += 1
            try:
                event = json.loads(line)
            except ValueError:
                continue
            inner = event.get("event") or {}
            if inner.get("type") == "message_start":
                for key, value in (inner.get("message", {}).get("usage") or {}).items():
                    if isinstance(value, int):
                        usage[key] += value
            elif inner.get("type") == "message_delta":
                usage["output_tokens"] += (inner.get("usage") or {}).get("output_tokens") or 0
            elif inner.get("type") == "content_block_delta" and inner.get("delta", {}).get("type") == "text_delta":
                streamed += inner["delta"]["text"]
            elif event.get("type") == "result":
                result = event
        found = find_actions(streamed, final=done or result is not None)
        if found is not None and not stopped_early:
            stopped_early = True
            break
        if done or result is not None:
            break
        if time.monotonic() - started > timeout:
            break
        time.sleep(0.05)
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    meter = getattr(_local, "meter", None)
    if meter is not None:
        p = _PRICES.get(model or "", _PRICES["claude-sonnet-5-5"])
        meter["calls"] += 1
        meter["call_seconds"].append(round(time.monotonic() - started, 2))
        meter["prompt_chars"].append(len(text) + len(system))
        meter["early_stops"] += stopped_early and result is None
        for key in TOKEN_KEYS:
            meter[key] += usage[key]
        meter["cost"] += sum(usage[key] * price for key, price in zip(TOKEN_KEYS, p, strict=True)) / 1e6
    found = find_actions(streamed)
    if found is not None:
        return streamed[: found[1]]
    if result is not None and not result.get("is_error") and isinstance(result.get("result"), str):
        return str(result["result"])
    if streamed.strip():
        return streamed  # no action in it: the agent loop says so and asks again
    why = (result or {}).get("result") or f"no answer within {timeout:.0f}s"
    raise thunc.ThuncError(f"claude error: {why}")


def patch_text_protocol() -> None:
    from thunc import native

    native.TextConversation.next = _patched_next  # type: ignore[method-assign]
    backends.BACKENDS["claude-code"] = streaming_claude_code


# --- harness: Claude Code -----------------------------------------------------------------------


def run_claude_code(task: Task, root: str, model: str, max_steps: int, n: int) -> dict[str, Any]:
    prompt = task.instructions
    if task.returns is not str:
        prompt += (
            f"\n\nWhen you are done, end your reply with your answer as {describe(task.returns, True)}, "
            "alone in a ```json code block."
        )
    args = [
        "claude", "-p", "--output-format", "stream-json", "--verbose", "--model", model,
        "--allowedTools", "Read", "Edit", "Write", "Bash", "Grep", "Glob",
        "--disallowedTools", "Task", "WebFetch", "WebSearch", "TodoWrite", "Agent",
        "--strict-mcp-config", "--no-session-persistence", "--max-turns", str(max_steps),
    ]  # fmt: skip
    out = _claude_stream(args, prompt, root)
    result = out.pop("result")
    if task.returns is str:
        out["value"] = result.get("result")
    elif not out["error"]:
        text = str(result.get("result") or "")
        fence = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
        try:
            out["value"] = parse(fence[-1] if fence else text, task.returns)
        except ValueError as exc:
            out["error"] = f"unparseable answer: {exc}"
    return out


def _claude_stream(args: list[str], prompt: str, cwd: str) -> dict[str, Any]:
    """Run `claude -p --output-format stream-json` to the end; its tool calls, usage and result."""
    started = time.monotonic()
    proc = subprocess.run(args, input=prompt, cwd=cwd, capture_output=True, text=True, timeout=1500)
    seconds = round(time.monotonic() - started, 1)
    events = []
    for line in proc.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    calls: dict[str, dict[str, Any]] = {}
    for e in events:
        content = (e.get("message") or {}).get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                calls[block["id"]] = {"tool": block["name"], "args": block.get("input"), "result": ""}
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                text = block.get("content")
                if isinstance(text, list):
                    text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))
                call = calls[block["tool_use_id"]]
                call["result"] = ("error: " if block.get("is_error") else "") + str(text)[:4000]
    error = None
    if result.get("is_error") or result.get("subtype") != "success":
        error = f"{result.get('subtype')}: {str(result.get('result'))[:300]}"
    u = result.get("usage") or {}
    return {
        "value": None,
        "error": error,
        "turns": result.get("num_turns", 0),
        "seconds": seconds,
        "tool_calls": list(calls.values()),
        "nudges": [],
        "result": result,
        "usage": {
            "cost": result.get("total_cost_usd") or 0,
            "calls": result.get("num_turns", 0),
            **{k: u.get(k) or 0 for k in TOKEN_KEYS},
        },
    }


# thunc variants: (protocol, permissions)
THUNC_VARIANTS = {
    "thunc": (None, ["write", "run"]),
    "thunc-text": ("text", ["write", "run"]),
    "thunc-shell": (None, ["write", "shell"]),
    "thunc-fixed": ("text", ["write", "run"]),
}
HARNESSES = {
    **{name: functools.partial(run_thunc, variant=name) for name in THUNC_VARIANTS},
    "claude-code": run_claude_code,
}


# --- running ------------------------------------------------------------------------------------


def one(harness: str, task_name: str, n: int, model: str, max_steps: int, base: str) -> dict[str, Any]:
    task = TASKS[task_name]
    root = tempfile.mkdtemp(prefix=f"{task_name}-{harness}-{n}-", dir=base)
    files = task.files()
    write_files(root, files)
    if task.git:
        git_init(root)
    try:
        out = HARNESSES[harness](task, root, model, max_steps, n)
    except Exception as exc:  # noqa: BLE001
        out = {"value": None, "error": f"harness crash {type(exc).__name__}: {exc}", "turns": 0, "seconds": 0,
               "tool_calls": [], "nudges": [], "usage": {}}  # fmt: skip
    try:
        passed, why = task.check(root, files, out["value"])
    except Exception as exc:  # noqa: BLE001
        passed, why = False, f"check crashed: {type(exc).__name__}: {exc}"
    value = out["value"]
    if isinstance(value, list):
        value = [asdict(v) if hasattr(v, "__dataclass_fields__") else v for v in value]
    print(f"  {harness:12} {task_name:13} #{n} {'PASS' if passed else 'FAIL'} turns={out['turns']} "
          f"{out['seconds']:.0f}s ${out['usage'].get('cost', 0):.3f} {why if not passed else ''}"
          f"{' ERR ' + out['error'][:120] if out['error'] else ''}", flush=True)  # fmt: skip
    shutil.rmtree(root, ignore_errors=True)
    return {"harness": harness, "task": task_name, "n": n, "passed": passed, "why": why, **out, "value": value}


def report(data: dict[str, Any]) -> None:
    results = data["results"]
    harnesses = list(dict.fromkeys(r["harness"] for r in results))
    print(f"\nmodel {data['model']}, {data['runs']} run(s) each\n")
    print(f"{'task':13} " + " ".join(f"{h + ' pass':>16} {'turns':>6} {'secs':>6} {'$':>6}" for h in harnesses))
    for task in TASKS:
        cells = []
        for h in harnesses:
            rows = [r for r in results if r["harness"] == h and r["task"] == task]
            if not rows:
                cells.append(" " * 38)
                continue
            mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
            cells.append(
                f"{sum(r['passed'] for r in rows):>12}/{len(rows):<3} {mean([r['turns'] for r in rows]):>6.1f} "
                f"{mean([r['seconds'] for r in rows]):>6.0f} {mean([r['usage'].get('cost', 0) for r in rows]):>6.3f}"
            )
        print(f"{task:13} " + " ".join(cells))
    for h in harnesses:
        rows = [r for r in results if r["harness"] == h]
        errors = collections.Counter()  # type: ignore[var-annotated]
        for r in rows:
            for c in r["tool_calls"]:
                if str(c.get("result", "")).startswith("error"):
                    errors[f"{c['tool']}: {_category(str(c['result']))}"] += 1
            for s in r["nudges"]:
                errors[f"(no action): {_category(s['result'])}"] += 1
        calls = sum(len(r["tool_calls"]) for r in rows)
        print(f"\n{h}: {sum(r['passed'] for r in rows)}/{len(rows)} passed, {calls} tool calls, "
              f"{sum(errors.values())} errors, ${sum(r['usage'].get('cost', 0) for r in rows):.2f}")  # fmt: skip
        for what, count in errors.most_common(25):
            print(f"  {count:4}  {what}")


def _category(result: str) -> str:
    text = re.sub(r"'[^']*'", "'…'", result.removeprefix("error: "))
    text = re.sub(r"\d+", "N", text)
    return text[:110]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness", default="thunc,claude-code")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None, help="print the tables from a results file and exit")
    args = parser.parse_args()
    if args.report:
        with open(args.report, encoding="utf-8") as f:
            report(json.load(f))
        return
    backends._run_cli = _counting_run_cli
    backends._cli_events = _counting_cli_events
    ClaudeCodeConversation.close = _counting_close  # type: ignore[method-assign]
    harnesses = args.harness.split(",")
    if "thunc-fixed" in harnesses:
        if len(harnesses) > 1:
            parser.error("thunc-fixed patches the text protocol for the whole process; run it on its own")
        patch_text_protocol()
    base = tempfile.mkdtemp(prefix="thunc-bench-")
    thunc.configure(agents_dir=os.path.join(base, "agents"))
    jobs = [(h, t, n) for n in range(1, args.runs + 1) for t in args.tasks.split(",") for h in harnesses]
    print(f"{len(jobs)} runs, model {args.model}, workdirs in {base}", flush=True)
    results = thunc.map(lambda j: one(*j, args.model, args.max_steps, base), jobs, workers=args.workers)
    data = {"model": args.model, "runs": args.runs, "max_steps": args.max_steps, "results": results}
    out = args.out or os.path.join(base, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)
    report(data)
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
