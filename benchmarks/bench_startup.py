"""Start-up: importing thunc, and the `thunc` command, in a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys

from .harness import bench


def _python(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, capture_output=True)


@bench("startup.python")
def startup_python(ctx):
    """A bare interpreter: subtract this from the others."""
    yield lambda: _python("-c", "pass")


@bench("startup.import_thunc")
def startup_import(ctx):
    """python -c 'import thunc'."""
    yield lambda: _python("-c", "import thunc")


@bench("startup.cli_cache_list")
def startup_cli(ctx):
    """`thunc cache list` on an empty cache."""
    yield lambda: _python("-m", "thunc", "cache", "list", "--cache-dir", ctx.tmp)
