"""The answer cache behind cache=True: one JSON file per call, and the tools to list and clear it."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import tempfile
import time
import warnings
from collections.abc import Callable, Collection, Iterator
from typing import Any

from .config import cache_dir

# Only files with these names are ever read or deleted, so a cache_dir pointed at the wrong
# folder (".", a home directory) can't lose anything that isn't a thunc cache entry.
_ENTRY = re.compile(r"[0-9a-f]{64}\.json")
_TEMP = re.compile(r"tmp\w+\.tmp")
_TEMP_GRACE = 3600.0  # a .tmp file younger than this may be a write in progress (thunc.map)


@dataclasses.dataclass(frozen=True)
class CacheGroup:
    """The saved answers of one function, as cache_info() reports them."""

    function: str | None  # qualified name, or the name= given to thunc.call; None for unnamed calls
    module: str | None  # where the function was defined, for display (not part of the cache key)
    entries: int
    size: int  # bytes on disk
    newest: dt.datetime  # when the most recent of these answers was saved
    readable: bool = True  # False for entries that couldn't be read, so can't be attributed to a function


def clear_cache(
    function: Callable[..., Any] | str | None = None, *, older_than: float | dt.timedelta | None = None
) -> int:
    """Delete saved answers and return how many were deleted.

        thunc.clear_cache()                               # everything
        thunc.clear_cache(urgency)                        # one @thunc.function
        thunc.clear_cache("urgency")                      # the same, by name (also "module.urgency")
        thunc.clear_cache(older_than=timedelta(days=30))  # answers saved more than 30 days ago

    `function` and `older_than` combine. A name matches a function's qualified name ("urgency",
    "Triage.urgency") or the name= given to thunc.call. Uses the configured cache folder.
    """
    names = None if function is None else {_name_of(function)}
    return _clear(cache_dir(), names, _seconds(older_than))[0]


def cache_info() -> list[CacheGroup]:
    """What's in the cache, one group per function. Unnamed thunc.call answers have function=None."""
    return _info(cache_dir())


def get(key: str) -> str | None:
    """The saved answer, or None if there's none (a missing or unreadable entry is a miss)."""
    try:
        with open(os.path.join(cache_dir(), f"{key}.json"), encoding="utf-8") as f:
            answer = json.load(f)["answer"]
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return None
    return answer if isinstance(answer, str) else None


def put(where: dict[str, Any], answer: str) -> None:
    """One JSON file per call. Written to a temporary file and renamed into place, so concurrent
    calls (thunc.map) never see half an entry. A cache that can't be written warns, not fails:
    the answer is valid and already paid for."""
    folder = cache_dir()
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **where, "answer": answer}
    tmp = None
    try:
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix="tmp", suffix=".tmp")
        # backslashreplace: a lone surrogate (half an emoji, "\ud83d") is written as the JSON escape it came from.
        with os.fdopen(fd, "w", encoding="utf-8", errors="backslashreplace") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(folder, f"{where['key']}.json"))
    except OSError as exc:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
        # stacklevel: put <- _call <- thunc.call or the @thunc.function wrapper <- the user's code
        warnings.warn(f"thunc could not save to the cache in {folder!r}: {exc}", RuntimeWarning, stacklevel=4)


@dataclasses.dataclass(frozen=True)
class _Entry:
    path: str
    size: int
    saved: float  # modification time: when the answer was written
    function: str | None = None
    module: str | None = None
    readable: bool = True


def _clear(
    folder: str, names: Collection[str] | None, older_than: float | None, dry_run: bool = False
) -> tuple[int, int]:
    """Delete the matching entries (all of them if `names` is None). Returns (entries, bytes)."""
    cutoff = None if older_than is None else time.time() - older_than
    chosen = [
        entry
        for entry in _scan(folder, read=names is not None)
        if (cutoff is None or entry.saved <= cutoff) and (names is None or _matches(entry, names))
    ]
    if dry_run:
        return len(chosen), sum(entry.size for entry in chosen)
    count = size = 0
    for entry in chosen:
        try:
            os.unlink(entry.path)
        except FileNotFoundError:
            continue  # already gone: another process cleared it first
        count += 1
        size += entry.size
    if names is None:
        _remove_stale_temp(folder, older_than)
    return count, size


def _info(folder: str) -> list[CacheGroup]:
    groups: dict[tuple[bool, str | None, str | None], list[_Entry]] = {}
    for entry in _scan(folder, read=True):
        groups.setdefault((entry.readable, entry.function, entry.module), []).append(entry)
    result = [
        CacheGroup(
            function=function,
            module=module,
            entries=len(entries),
            size=sum(entry.size for entry in entries),
            newest=dt.datetime.fromtimestamp(max(entry.saved for entry in entries)),
            readable=readable,
        )
        for (readable, function, module), entries in groups.items()
    ]
    # Named functions first, by name; then unnamed thunc.call answers; unreadable entries last.
    return sorted(result, key=lambda g: (not g.readable, g.function is None, g.function or "", g.module or ""))


def _scan(folder: str, *, read: bool) -> Iterator[_Entry]:
    """Every cache entry in `folder`; with read=True, also which function saved it."""
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return
    for name in names:
        if not _ENTRY.fullmatch(name):
            continue
        path = os.path.join(folder, name)
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            continue
        entry = _Entry(path, stat.st_size, stat.st_mtime)
        if read:
            entry = _read_owner(entry)
        yield entry


def _read_owner(entry: _Entry) -> _Entry:
    try:
        with open(entry.path, encoding="utf-8") as f:
            data = json.load(f)
        function, module = data.get("function"), data.get("module")
    except (OSError, ValueError, AttributeError, RecursionError):
        return dataclasses.replace(entry, readable=False)
    if not all(value is None or isinstance(value, str) for value in (function, module)):
        return dataclasses.replace(entry, readable=False)
    return dataclasses.replace(entry, function=function, module=module)


def _matches(entry: _Entry, names: Collection[str]) -> bool:
    if not entry.readable or entry.function is None:
        return False
    return entry.function in names or (entry.module is not None and f"{entry.module}.{entry.function}" in names)


def _remove_stale_temp(folder: str, older_than: float | None) -> None:
    """Leftovers of writes that crashed between creating the temporary file and renaming it."""
    grace = max(_TEMP_GRACE, older_than or 0.0)
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return
    for name in names:
        if not _TEMP.fullmatch(name):
            continue
        path = os.path.join(folder, name)
        try:
            if time.time() - os.stat(path).st_mtime > grace:
                os.unlink(path)
        except FileNotFoundError:
            continue


def _name_of(function: Callable[..., Any] | str) -> str:
    if isinstance(function, str):
        if not function:
            raise ValueError("clear_cache: the function name is empty")
        return function
    name = getattr(getattr(function, "__func__", function), "__thunc_function__", None)
    if not isinstance(name, str):
        raise TypeError(
            f"clear_cache: {function!r} is not a @thunc.function. "
            "Pass one, its name as a string, or nothing to clear everything."
        )
    return name


def _seconds(older_than: float | dt.timedelta | None) -> float | None:
    if older_than is None:
        return None
    seconds = older_than.total_seconds() if isinstance(older_than, dt.timedelta) else float(older_than)
    if seconds < 0:
        raise ValueError(f"clear_cache: older_than can't be negative, got {older_than!r}")
    return seconds
