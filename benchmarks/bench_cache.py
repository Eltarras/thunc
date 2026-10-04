"""The answer cache: `thunc cache list` and clearing one function, over many entries."""

from __future__ import annotations

import thunc
from thunc.cache import _clear, _info
from thunc.config import cache_dir

from .fakes import Backend, use
from .harness import bench


def _filled(ctx) -> tuple[str, int]:
    """A cache with entries from 5 functions, written through thunc so they're the real format."""
    use(Backend("4"))
    entries = 50 if ctx.smoke else 5000
    for i in range(entries):
        thunc.call("Rate it.", {"n": i}, returns=int, cache=True, name=f"f{i % 5}")
    ctx.notes["entries"] = entries
    return cache_dir(), entries


@bench("cache.list")
def cache_list(ctx):
    """`thunc cache list` over 5,000 entries: reads every file."""
    folder, _ = _filled(ctx)
    yield lambda: _info(folder)


@bench("cache.clear_one_function_dry")
def cache_clear_dry(ctx):
    """`thunc cache clear --function f0 --dry-run` over 5,000 entries."""
    folder, _ = _filled(ctx)
    yield lambda: _clear(folder, {"f0"}, None, dry_run=True)
