"""Live events for thunc-watch: one JSON line per event, appended to the file THUNC_EVENTS names.

    THUNC_EVENTS=events.jsonl python app.py       # then, elsewhere: thunc-watch --events events.jsonl

thunc-watch sets THUNC_EVENTS itself when it runs a program. Every line has "v" (the format, 1),
"event", "t" (Unix time) and "pid". The events:

    call.start     id, function, backend, model, inputs (name -> preview)
    call.attempt   id, n, seconds, ok, problem (why the reply was rejected), reply (preview)
    call.end       id, ok, cached, attempts, seconds, value (preview), error
    agent.start    id, agent, task, returns, session (path of the run's record), inputs
    agent.reply    id, n, seconds (waiting on the model)
    agent.tool     id, n, tool, target: a tool call starting (a command can take minutes)
    agent.step     id, n, tool, target, result (preview), seconds, denied
    agent.end      id, ok, steps, seconds, files_changed, value (preview), error

Inputs, replies and values are cut to a short preview. THUNC_EVENTS_CAPTURE=1 sends them whole,
which the dashboard shows on the call screen; they may contain personal data, as they do in a trace.
Nothing is written, and nothing costs more than one environment lookup, when THUNC_EVENTS isn't set.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from typing import Any

from .schema import short_repr, shorten

FORMAT = 1
PREVIEW = 120  # characters of an input, reply or value, unless capture is on

_lock = threading.Lock()
_ids = itertools.count(1)


def enabled() -> bool:
    return bool(os.environ.get("THUNC_EVENTS"))


def capture() -> bool:
    return os.environ.get("THUNC_EVENTS_CAPTURE", "") not in ("", "0")


def next_id() -> int:
    with _lock:
        return next(_ids)


def preview(value: Any) -> str:
    """A value as the dashboard shows it: strings as they are, the rest as repr(), cut short unless capturing."""
    if capture():
        return value if isinstance(value, str) else short_repr(value, 100_000)
    return shorten(value, PREVIEW) if isinstance(value, str) else short_repr(value, PREVIEW)


def emit(event: str, **fields: Any) -> None:
    """Append one event, if THUNC_EVENTS is set. Never raises: watching must not break the program."""
    path = os.environ.get("THUNC_EVENTS")
    if not path:
        return
    entry = {"v": FORMAT, "event": event, "t": round(time.time(), 3), "pid": os.getpid(), **fields}
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
    except (RecursionError, TypeError, ValueError):
        line = json.dumps({key: entry[key] for key in ("v", "event", "t", "pid")} | {"fields": short_repr(fields)})
    for separator in "\x85  ":  # valid in JSON, but they split lines for some readers
        line = line.replace(separator, f"\\u{ord(separator):04x}")
    try:
        with _lock, open(path, "a", encoding="utf-8", errors="backslashreplace") as f:
            f.write(line + "\n")
    except OSError:
        pass


def target(tool: str, args: dict[str, Any]) -> str:
    """What a tool call acted on, in one line: its path, command or pattern."""
    if tool == "search" and "pattern" in args:
        where = args.get("path")
        return f"{args['pattern']!r}" + (f" in {where}" if where else "")
    for key in ("command", "path", "note"):
        if isinstance(args.get(key), str):
            return shorten(args[key], PREVIEW)
    return short_repr(args, PREVIEW) if args else ""
