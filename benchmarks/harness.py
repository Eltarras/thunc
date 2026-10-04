"""A small benchmark harness: no dependencies, so it runs wherever thunc does.

A benchmark is a generator function decorated with @bench. It sets up, yields the operation to time
(a no-argument callable), and tears down after the yield:

    @bench("schema.parse_int")
    def parse_int(ctx):
        yield lambda: parse("4", int)

ctx.tmp is a fresh folder, ctx.smoke is True for a quick run that only checks the benchmark works
(use it to shrink inputs), and ctx.notes takes extra measurements shown next to the timing, such
as connections opened or bytes sent.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from thunc import backends, config

Setup = Callable[["Context"], Iterator[Callable[[], Any]]]


@dataclass
class Context:
    tmp: str
    smoke: bool
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Bench:
    name: str
    setup: Setup
    about: str


@dataclass
class Result:
    name: str
    per_op: float  # seconds, median of the samples
    best: float  # seconds, fastest sample
    spread: float  # (slowest - fastest) / median of the samples: high means noisy
    ops: int  # operations timed in all
    notes: dict[str, Any]
    error: str | None = None


REGISTRY: list[Bench] = []


def bench(name: str) -> Callable[[Setup], Setup]:
    def register(setup: Setup) -> Setup:
        REGISTRY.append(Bench(name, setup, (setup.__doc__ or "").strip().split("\n")[0]))
        return setup

    return register


@contextlib.contextmanager
def isolated(tmp: str) -> Iterator[None]:
    """thunc's settings, environment and backends as they were, after the benchmark changes them."""
    settings, env, registered = dict(config._settings), dict(os.environ), dict(backends.BACKENDS)
    for var in ("THUNC_BACKEND", "THUNC_TRACE", "THUNC_CACHE_DIR", "THUNC_AGENTS_DIR"):
        os.environ.pop(var, None)
    config._settings.update(config.DEFAULTS)
    config.configure(cache_dir=os.path.join(tmp, "cache"), agents_dir=os.path.join(tmp, "agents"))
    cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(cwd)
        config._settings.clear()
        config._settings.update(settings)
        os.environ.clear()
        os.environ.update(env)
        backends.BACKENDS.clear()
        backends.BACKENDS.update(registered)


def run(b: Bench, *, smoke: bool = False, target: float = 0.2, samples: int = 5) -> Result:
    """Time b's operation: `samples` batches of n operations, n chosen so a batch takes about `target`
    seconds. A smoke run times one operation once."""
    with tempfile.TemporaryDirectory(prefix="thunc-bench-") as tmp, isolated(tmp):
        ctx = Context(os.path.realpath(tmp), smoke)
        try:
            with contextlib.contextmanager(b.setup)(ctx) as op:  # the code after the yield runs too
                first = _batch(op, 1)  # warm up: imports, caches, the first connection
                if smoke:
                    times, n = [first], 1
                else:
                    n = max(1, int(target / max(first, 1e-7)))
                    while n < 1_000_000 and (took := _batch(op, n)) < target / 2:
                        n = max(n * 2, int(n * target / max(took, 1e-9)))
                    times = [_batch(op, n) / n for _ in range(3 if first > 1.0 else samples)]
        except Skip as skip:
            return Result(b.name, 0.0, 0.0, 0.0, 0, ctx.notes, error=f"skipped: {skip}")
        median = statistics.median(times)
        return Result(b.name, median, min(times), (max(times) - min(times)) / median, n * len(times), ctx.notes)


class Skip(Exception):
    """Raised from a benchmark's setup when it can't run here (an SDK or a CLI isn't installed)."""


def _batch(op: Callable[[], Any], n: int) -> float:
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter()
        for _ in range(n):
            op()
        return time.perf_counter() - started
    finally:
        if enabled:
            gc.enable()


# Reporting


def table(results: list[Result], baseline: dict[str, Any] | None = None) -> str:
    rows = []
    for r in results:
        if r.error:
            rows.append([r.name, r.error, "", "", "", ""])
            continue
        change = ""
        if baseline and r.name in baseline and baseline[r.name].get("per_op"):
            ratio = r.per_op / baseline[r.name]["per_op"]
            change = f"{(ratio - 1) * 100:+.0f}%" if abs(ratio - 1) >= 0.005 else "="
        rows.append(
            [
                r.name,
                duration(r.per_op),
                f"{1 / r.per_op:,.0f}/s" if r.per_op else "",
                f"±{r.spread * 50:.0f}%",
                change,
                ", ".join(f"{k}={_note(v)}" for k, v in r.notes.items()),
            ]
        )
    header = ["BENCHMARK", "PER OP", "RATE", "NOISE", "VS BASE" if baseline else "", "NOTES"]
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    lines = []
    for row in [header, *rows]:
        cells = [f"{row[0]:<{widths[0]}}", *(f"{c:>{w}}" for c, w in zip(row[1:5], widths[1:5], strict=True)), row[5]]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def to_json(results: list[Result]) -> dict[str, Any]:
    return {
        r.name: {"per_op": r.per_op, "best": r.best, "spread": r.spread, "ops": r.ops, "notes": r.notes}
        for r in results
        if not r.error
    }


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data.get("results", data)


def duration(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f}ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.2f}s"


def _note(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, int) and value >= 10_000:
        return f"{value:,}"
    return str(value)
