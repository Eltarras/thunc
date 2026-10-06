"""The `thunc` command (also `python -m thunc`): run or watch a program, and look at and clear the answer cache.

thunc run [--profile] SCRIPT [ARG]...
thunc run [--profile] -m MODULE [ARG]...
thunc watch [OPTION]... SCRIPT [ARG]...    (the dashboard: pip install "thunc[watch]")
thunc cache list
thunc cache clear [--function NAME]... [--older-than AGE] [--dry-run] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import runpy
import shutil
import subprocess
import sys
import sysconfig
import traceback
from collections.abc import Sequence
from contextlib import nullcontext

from . import __version__, profiling
from .cache import CacheGroup, _clear, _info
from .config import cache_dir

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


WATCH_INSTALL = 'thunc watch needs the thunc-watch package. Install it with: pip install "thunc[watch]"'


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["watch"]:  # everything after it is for thunc-watch, options included
        return _watch(argv[1:])
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args.target, args.args, module=args.module, profile=args.profile)
    folder = args.cache_dir or cache_dir()
    if args.action == "list":
        return _list(folder)
    return _clear_command(folder, args.function, args.older_than, args.dry_run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thunc", description="thunc: call an LLM like a typed Python function.")
    parser.add_argument("--version", action="version", version=f"thunc {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = commands.add_parser(
        "run",
        help="run a Python program that uses thunc",
        description="Run a Python script (or a module, with -m) as `python` would. With --profile, a report of "
        "where the time went in thunc calls and agent runs is printed to stderr when it ends.",
    )
    run.add_argument("--profile", action="store_true", help="print a performance report when the program ends")
    run.add_argument("-m", dest="module", action="store_true", help="TARGET is a module name, as in python -m")
    run.add_argument("target", metavar="TARGET", help="the script to run (or the module, with -m)")
    run.add_argument("args", nargs=argparse.REMAINDER, metavar="ARG", help="arguments passed to the program")
    commands.add_parser(
        "watch",
        help="run a program with a live dashboard of its calls and agent runs (needs thunc[watch])",
        add_help=False,  # `thunc watch --help` is thunc-watch's own help
    )
    cache = commands.add_parser("cache", help="look at or clear the answers saved by cache=True")
    actions = cache.add_subparsers(dest="action", required=True, metavar="ACTION")

    folder = argparse.ArgumentParser(add_help=False)
    folder.add_argument(
        "--cache-dir",
        metavar="DIR",
        help="the cache folder (default: THUNC_CACHE_DIR, or .thunc_cache). "
        "A configure(cache_dir=...) in your code isn't seen here: pass the same folder.",
    )
    actions.add_parser("list", parents=[folder], help="show the saved answers, grouped by function")
    clear = actions.add_parser("clear", parents=[folder], help="delete saved answers (all of them by default)")
    clear.add_argument(
        "--function",
        action="append",
        metavar="NAME",
        help="only this function's answers: its name (urgency, Triage.urgency), module.name, or the "
        "name= given to thunc.call. Repeat for several.",
    )
    clear.add_argument(
        "--older-than",
        type=_age,
        metavar="AGE",
        help="only answers saved longer ago than AGE: 45s, 90m, 12h, 30d, or a number of seconds",
    )
    clear.add_argument("--dry-run", action="store_true", help="say what would be deleted, and delete nothing")
    return parser


def _run(target: str, args: list[str], *, module: bool, profile: bool) -> int:
    """Run the program in this process, like `python [-m] target args...`; its exit code is ours."""
    sys.argv = [target, *args]
    sys.path.insert(0, os.getcwd() if module else os.path.dirname(os.path.abspath(target)))
    with profiling.profiling() if profile else nullcontext() as profiler:
        try:
            if module:
                runpy.run_module(target, run_name="__main__", alter_sys=True)
            else:
                runpy.run_path(target, run_name="__main__")
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else _exit_message(exc.code))
        except KeyboardInterrupt:
            code = 130
        except BaseException:
            traceback.print_exc()
            code = 1
    if profiler is not None:
        sys.stdout.flush()
        print(f"\n{profiler.report(f'thunc profile: {target}')}", file=sys.stderr)
    return code


def _watch(args: list[str]) -> int:
    """`thunc watch`: run the thunc-watch dashboard. Scripts it runs get this Python, where thunc is."""
    binary = _watch_binary()
    if binary is None:
        print(WATCH_INSTALL, file=sys.stderr)
        return 2
    # Its own --python, given later, wins over this one.
    argv = [binary, "--python", sys.executable, *args]
    if sys.platform != "win32":
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(binary, argv)  # the dashboard takes over this process: signals and exit code are its own
    process = subprocess.Popen(argv)
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:  # the dashboard handles Ctrl+C itself
            continue


def _watch_binary() -> str | None:
    """thunc-watch from THUNC_WATCH_BIN, next to this Python (where pip puts it), or on PATH."""
    if path := os.environ.get("THUNC_WATCH_BIN"):
        return path
    name = "thunc-watch.exe" if sys.platform == "win32" else "thunc-watch"
    for folder in (sysconfig.get_path("scripts"), os.path.dirname(sys.executable)):
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("thunc-watch")


def _exit_message(message: object) -> int:
    """sys.exit("message"): print it, as Python does, and exit with 1."""
    print(message, file=sys.stderr)
    return 1


def _age(text: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"{text!r} is not an age: use 45s, 90m, 12h, 30d, or a number of seconds")
    return float(match[1]) * _UNITS.get(match[2], 1)


def _list(folder: str) -> int:
    groups = _info(folder)
    if not groups:
        print(f"No saved answers in {folder}")
        return 0
    rows = [(_label(g), str(g.entries), _size(g.size), g.newest.strftime("%Y-%m-%d %H:%M")) for g in groups]
    width = max(len("FUNCTION"), *(len(row[0]) for row in rows))
    print(f"{'FUNCTION':<{width}}  {'ENTRIES':>7}  {'SIZE':>8}  NEWEST")
    for label, entries, size, newest in rows:
        print(f"{label:<{width}}  {entries:>7}  {size:>8}  {newest}")
    print(f"\n{_entries(sum(g.entries for g in groups))} ({_size(sum(g.size for g in groups))}) in {folder}")
    return 0


def _clear_command(folder: str, names: list[str] | None, older_than: float | None, dry_run: bool) -> int:
    count, size = _clear(folder, names, older_than, dry_run=dry_run)
    print(f"{'Would remove' if dry_run else 'Removed'} {_entries(count)} ({_size(size)}) from {folder}")
    return 0


def _label(group: CacheGroup) -> str:
    if not group.readable:
        return "(unreadable)"
    if group.function is None:
        return "(thunc.call)"
    return f"{group.module}.{group.function}" if group.module else group.function


def _size(size: int) -> str:
    if size < 1000:
        return f"{size} B"
    if size < 1000**2:
        return f"{size / 1000:.0f} kB"
    if size < 1000**3:
        return f"{size / 1000**2:.1f} MB"
    return f"{size / 1000**3:.1f} GB"


def _entries(count: int) -> str:
    return f"{count} entry" if count == 1 else f"{count} entries"


if __name__ == "__main__":
    sys.exit(main())
