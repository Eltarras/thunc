"""Where a program's time goes in thunc: `thunc run --profile script.py` prints a report when it ends.

While a Profiler is active, every thunc.call, @thunc.function and agent run adds one record to it:
how long it took, how much of that was spent waiting on the model, and (for agents) on each tool.
Nothing is recorded, and nothing costs anything, when no Profiler is active.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

UNNAMED = "(thunc.call)"


@dataclass
class CallRecord:
    """One thunc.call or @thunc.function call."""

    function: str
    backend: str | None
    model: str | None
    start: float
    end: float
    model_seconds: float  # waiting on the backend, over all attempts
    attempts: int  # requests sent to the model (0 when answered from the cache)
    cached: bool
    ok: bool

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class AgentRecord:
    """One agent run."""

    function: str
    backend: str | None
    model: str | None
    start: float
    end: float
    model_seconds: float
    steps: int  # model replies
    tools: dict[str, list[float]] = field(default_factory=dict)  # tool name -> seconds of each call
    ok: bool = True

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def tool_seconds(self) -> float:
        return sum(sum(times) for times in self.tools.values())


class Profiler:
    """Collects a record per call while active. Thread-safe: thunc.map runs calls on worker threads."""

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []
        self.agents: list[AgentRecord] = []
        self.started = time.monotonic()
        self.stopped: float | None = None
        self._lock = threading.Lock()

    def add(self, record: CallRecord | AgentRecord) -> None:
        with self._lock:
            if isinstance(record, CallRecord):
                self.calls.append(record)
            else:
                self.agents.append(record)

    def stop(self) -> None:
        self.stopped = time.monotonic()

    @property
    def wall(self) -> float:
        return (self.stopped or time.monotonic()) - self.started

    def report(self, title: str = "thunc profile") -> str:
        return _report(self, title)


_active: Profiler | None = None


def active() -> Profiler | None:
    return _active


@contextmanager
def profiling() -> Iterator[Profiler]:
    """Record every thunc call made inside the block; the Profiler's report() says where the time went."""
    global _active
    previous, profiler = _active, Profiler()
    _active = profiler
    try:
        yield profiler
    finally:
        profiler.stop()
        _active = previous


def timed(seconds: list[float], func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """func(*args, **kwargs), with the time it took appended to `seconds` (even when it raises)."""
    started = time.monotonic()
    try:
        return func(*args, **kwargs)
    finally:
        seconds.append(time.monotonic() - started)


# The report


def _report(profiler: Profiler, title: str) -> str:
    wall = profiler.wall
    calls, agents = list(profiler.calls), list(profiler.agents)
    lines = [f"{title}: {_secs(wall)} wall time"]
    if not calls and not agents:
        lines.append("No thunc calls were made.")
        return "\n".join(lines)

    if calls:
        lines += ["", "CALLS"]
        rows = []
        for name, group in _grouped(calls):
            times = [c.seconds for c in group]
            model = sum(c.model_seconds for c in group)
            asked = [c for c in group if not c.cached]
            rows.append(
                [
                    name,
                    str(len(group)),
                    str(sum(c.cached for c in group)),
                    str(sum(max(c.attempts - 1, 0) for c in asked)),
                    str(sum(not c.ok for c in group)),
                    _secs(sum(times)),
                    _secs(sum(times) / len(times)),
                    _secs(_percentile(times, 95)),
                    _secs(max(times)),
                    _secs(model),
                    _secs(max(sum(times) - model, 0.0)),
                ]
            )
        lines += _table(
            ["FUNCTION", "CALLS", "CACHED", "RETRIES", "FAILED", "TOTAL", "MEAN", "P95", "MAX", "MODEL", "LOCAL"], rows
        )

    if agents:
        lines += ["", "AGENT RUNS"]
        rows = []
        for name, runs in _grouped(agents):
            times = [r.seconds for r in runs]
            model = sum(r.model_seconds for r in runs)
            tool = sum(r.tool_seconds for r in runs)
            rows.append(
                [
                    name,
                    str(len(runs)),
                    str(sum(r.steps for r in runs)),
                    str(sum(not r.ok for r in runs)),
                    _secs(sum(times)),
                    _secs(sum(times) / len(times)),
                    _secs(max(times)),
                    _secs(model),
                    _secs(tool),
                    _secs(max(sum(times) - model - tool, 0.0)),
                ]
            )
        lines += _table(["TASK", "RUNS", "STEPS", "FAILED", "TOTAL", "MEAN", "MAX", "MODEL", "TOOLS", "LOCAL"], rows)

        tools: dict[str, list[float]] = {}
        for run in agents:
            for tool, times in run.tools.items():
                tools.setdefault(tool, []).extend(times)
        if tools:
            lines += ["", "AGENT TOOLS"]
            rows = [
                [tool, str(len(times)), _secs(sum(times)), _secs(sum(times) / len(times)), _secs(max(times))]
                for tool, times in sorted(tools.items(), key=lambda item: -sum(item[1]))
            ]
            lines += _table(["TOOL", "CALLS", "TOTAL", "MEAN", "MAX"], rows)

    records: list[CallRecord | AgentRecord] = [*calls, *agents]
    busy = _union([(r.start, r.end) for r in records])
    summed = sum(r.seconds for r in records)
    model = sum(r.model_seconds for r in records)
    by_model: dict[str, float] = {}
    for r in records:
        key = f"{r.backend or '?'}/{r.model or 'default model'}"
        by_model[key] = by_model.get(key, 0.0) + r.model_seconds
    split = ", ".join(f"{key} {_secs(secs)}" for key, secs in sorted(by_model.items(), key=lambda item: -item[1]))
    lines += [
        "",
        f"In thunc:      {_secs(busy)} of {_secs(wall)} wall time ({_share(busy, wall)}); "
        "the rest was the program's own code",
        f"Model time:    {_secs(model)}, {_share(model, summed)} of the time in calls ({split})",
    ]
    if busy and summed > busy * 1.05:
        lines.append(f"Concurrency:   calls overlapped {summed / busy:.1f}x on average (thunc.map or threads)")
    slowest = max(records, key=lambda r: r.seconds)
    lines.append(f"Slowest:       {slowest.function} took {_secs(slowest.seconds)}")
    return "\n".join(lines)


def _grouped(records: list[Any]) -> list[tuple[str, list[Any]]]:
    """Records grouped by function, the group with the most total time first."""
    groups: dict[str, list[Any]] = {}
    for record in records:
        groups.setdefault(record.function, []).append(record)
    return sorted(groups.items(), key=lambda item: -sum(r.seconds for r in item[1]))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        first = f"{cells[0]:<{widths[0]}}"
        return "  ".join([first, *(f"{cell:>{width}}" for cell, width in zip(cells[1:], widths[1:], strict=True))])

    return [line(header), *(line(row) for row in rows)]


def _union(intervals: list[tuple[float, float]]) -> float:
    """Total time covered by at least one interval: nested and parallel calls count once."""
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(percent / 100 * len(ordered)) - 1)]


def _secs(seconds: float) -> str:
    if seconds < 0.01:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{int(seconds // 60)}m{seconds % 60:04.1f}s"


def _share(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}%" if whole > 0 else "-"
