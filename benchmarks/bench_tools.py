"""The agent's tools on a synthetic repository: list, read, search, and what `run` adds."""

from __future__ import annotations

import os
import sys

from thunc.permissions import Permissions
from thunc.tools import Workdir

from .harness import bench

MODULE = '''"""Module {d}.{f}."""

import os

TIMEOUT = {f}


def handle_{f}_handler(request):
    value = request.get("value")
    if value is None:
        return None
    return value * {f}

'''


def make_repo(root: str, folders: int, files: int, lines: int = 120) -> int:
    """`folders` x `files` Python files of about `lines` lines, plus a node_modules folder (skipped)."""
    filler = "".join(f"# filler line {n}: nothing to see here, keep scrolling\n" for n in range(lines - 12))
    for d in range(folders):
        folder = os.path.join(root, "src", f"pkg{d:03}")
        os.makedirs(folder)
        for f in range(files):
            with open(os.path.join(folder, f"mod{f:03}.py"), "w") as out:
                out.write(MODULE.format(d=d, f=f) + filler)
    os.makedirs(os.path.join(root, "node_modules", "big"))
    for f in range(200):
        with open(os.path.join(root, "node_modules", "big", f"x{f}.js"), "w") as out:
            out.write("x\n")
    with open(os.path.join(root, "README.md"), "w") as out:
        out.write("# demo\n")
    return folders * files


def _workdir(ctx, permissions: tuple[str, ...] = ()) -> Workdir:
    repo = os.path.join(ctx.tmp, "repo")
    os.makedirs(repo)
    folders, files = (3, 5) if ctx.smoke else (40, 50)
    ctx.notes["files"] = make_repo(repo, folders, files)
    return Workdir(repo, Permissions(permissions))


@bench("tools.list")
def tools_list(ctx):
    """list on the root of a 2,000-file repo: stops at 300 entries."""
    workdir = _workdir(ctx)
    yield lambda: workdir.list(".")


@bench("tools.read")
def tools_read(ctx):
    """read the first 400 lines of a file."""
    workdir = _workdir(ctx)
    yield lambda: workdir.read("src/pkg000/mod001.py")


@bench("tools.search_hit")
def tools_search_hit(ctx):
    """search a regex that matches in every file: stops at 100 matches."""
    workdir = _workdir(ctx)
    yield lambda: workdir.search(r"def handle_\d+_handler\(")


@bench("tools.search_miss")
def tools_search_miss(ctx):
    """search a string that's nowhere: reads all 2,000 files (240k lines), the worst case."""
    workdir = _workdir(ctx)
    yield lambda: workdir.search("DATABASE_URL")


@bench("tools.search_miss_with_rules")
def tools_search_miss_rules(ctx):
    """search_miss with 4 permission rules: every file is checked against them."""
    workdir = _workdir(ctx, ("read:src/**", "!read:**/secrets/**", "!read:**/*.pem", "write:docs/**"))
    yield lambda: workdir.search("DATABASE_URL")


@bench("tools.snapshot")
def tools_snapshot(ctx):
    """The scan `run` does before and after every command, to see what it changed: stats every file."""
    workdir = _workdir(ctx)
    yield workdir._snapshot


@bench("tools.run_command")
def tools_run(ctx):
    """run a command that does nothing (`python -c pass`): process start plus two snapshots."""
    workdir = _workdir(ctx, ("run",))
    command = f"{sys.executable} -S -c pass"
    yield lambda: workdir.run(command)
