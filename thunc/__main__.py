"""The `thunc` command (also `python -m thunc`): look at and clear the answer cache.

thunc cache list
thunc cache clear [--function NAME]... [--older-than AGE] [--dry-run] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence

from . import __version__
from .cache import CacheGroup, _clear, _info
from .config import cache_dir

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    folder = args.cache_dir or cache_dir()
    if args.action == "list":
        return _list(folder)
    return _clear_command(folder, args.function, args.older_than, args.dry_run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thunc", description="thunc: call an LLM like a typed Python function.")
    parser.add_argument("--version", action="version", version=f"thunc {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
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
