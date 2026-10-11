"""Record and replay: answers saved in a folder, so tests and CI run without asking a model.

    thunc.configure(recordings="tests/recordings")  # or THUNC_RECORDINGS=tests/recordings

With a recordings folder set, every thunc.call, @thunc.function and agent run is answered from it
and no model is asked. An answer that isn't there raises ThuncError, so CI fails at once instead of
spending money or waiting on a login. THUNC_RECORD=1 asks the model and saves every answer there;
THUNC_RECORD=missing replays what's there and asks for, and saves, only the rest.

A call is matched on what it sends: its function's name, the system prompt and the request (the
instructions, inputs and return type). The backend and model are saved with the answer but not
matched, so replaying needs no backend or API key. A recorded answer is checked against the return
type and ensure= again before it's used, and a failed call is never recorded.

An agent run is matched on its agent, task, request, tools and system=, and the model's replies are
replayed in order. Its tools run for real: files are written and commands run, as when it was
recorded. The replies don't depend on what the tools return, so a run whose tools go differently
can run out of replies, which fails the run.

Entries use the answer cache's format (cache.py), one JSON file per call or run, so
`thunc cache list --cache-dir tests/recordings` lists them and `thunc cache clear` removes them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .cache import read, write
from .config import recordings_dir
from .errors import ThuncError
from .schema import shorten

if TYPE_CHECKING:
    from .native import Call, Conversation, Reply

FORMAT = 1  # part of every key: raise it when what a key covers changes
# THUNC_RECORD's values. Unset (or 0) replays.
_RECORD: dict[str, str | None] = {"": None, "0": None, "false": None, "no": None, "off": None}
_RECORD.update({value: "all" for value in ("1", "true", "yes", "on", "all")})
_RECORD["missing"] = "missing"


def mode() -> str | None:
    """None without a recordings folder. With one: "replay" (answer from it, never ask a model),
    "all" (THUNC_RECORD=1: ask, and save every answer) or "missing" (THUNC_RECORD=missing: replay
    what's there, ask for and save the rest)."""
    value = os.environ.get("THUNC_RECORD", "")
    if value:
        value = value.strip().lower()
    if value not in _RECORD:
        raise ThuncError(
            f"THUNC_RECORD={value!r} isn't a setting: use 1 to record every answer, missing to record only "
            "the answers the recordings lack, or 0 to replay"
        )
    record = _RECORD[value]
    if recordings_dir() is None:
        if record:  # a recording that would silently go nowhere
            raise ThuncError(
                "THUNC_RECORD is set, but there's no recordings folder to save to: "
                "set THUNC_RECORDINGS or thunc.configure(recordings=...)"
            )
        return None
    return record or "replay"


def check_live(what: str) -> None:
    """Raise if replaying, where nothing may reach a model. Every way to a model checks it, so a path
    that isn't recorded (a durable run) fails here instead of spending money in CI."""
    if mode() == "replay":
        raise ThuncError(
            f"{what} would ask a model, but thunc is replaying from {recordings_dir()!r}, where no model is "
            "asked. Durable runs aren't recorded. Unset THUNC_RECORDINGS, or record with THUNC_RECORD=1."
        )


def call_key(system: str, request: str, function: str | None) -> str:
    return _key({"kind": "call", "system": system, "function": function, "request": request})


def run_key(agent: str, task: str, system: str | None, request: str, tools: Sequence[str]) -> str:
    return _key({"kind": "run", "agent": agent, "task": task, "system": system, "request": request, "tools": tools})


def answer(key: str) -> dict[str, Any] | None:
    """The recorded call, with its answer as a string, or None."""
    entry = read(_folder(), key)
    if entry is None or not isinstance(entry.get("answer"), str):
        return None
    return entry


def save_answer(
    key: str, function: str | None, module: str | None, backend: str, model: str | None, request: str, text: str
) -> None:
    entry = {"key": key, "function": function, "module": module, "backend": backend, "model": model}
    # stacklevel: _save <- save_answer <- recorded <- _call <- thunc.call or the @thunc.function wrapper <- the user
    _save(key, {**entry, "request": request, "answer": text}, stacklevel=6)


def run(key: str) -> Replay | None:
    """The recorded run, ready to replay, or None if there's none or it can't be read."""
    entry = read(_folder(), key)
    if entry is None:
        return None
    try:
        replies = [_reply(data) for data in entry["replies"]]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    backend, model = entry.get("backend"), entry.get("model")
    return Replay(
        replies,
        entry.get("function") or "the run",
        backend if isinstance(backend, str) else None,
        model if isinstance(model, str) else None,
    )


def save_run(key: str, function: str, backend: str, model: str | None, request: str, replies: Sequence[Reply]) -> None:
    entry = {"key": key, "function": function, "module": None, "backend": backend, "model": model}
    # stacklevel: _save <- save_run <- Agent._run <- the task's record <- run <- the task's wrapper <- the user
    _save(key, {**entry, "request": request, "replies": [dataclasses.asdict(reply) for reply in replies]}, stacklevel=7)


def missing(what: str) -> ThuncError:
    return ThuncError(
        f"No recorded answer for {what} in {recordings_dir()!r}, and thunc is replaying, so no model is "
        "asked. Record it with THUNC_RECORD=missing (only the answers missing) or THUNC_RECORD=1 (every answer)."
    )


def stale(what: str, problem: object) -> ThuncError:
    return ThuncError(
        f"The recorded answer for {what} in {recordings_dir()!r} no longer fits ({shorten(str(problem), 300)}). "
        "Record it again with THUNC_RECORD=1."
    )


class Replay:
    """A conversation that gives back a recorded run's replies, in order, and asks no model."""

    def __init__(self, replies: list[Reply], function: str, backend: str | None, model: str | None) -> None:
        self.replies, self.function, self.backend, self.model = replies, function, backend, model
        self.given = 0

    def next(self) -> Reply:
        if self.given == len(self.replies):
            raise ThuncError(
                f"{self.function}: the recorded run has {len(self.replies)} model replies and this run needs more, "
                "so it went differently than when it was recorded (a tool's result, the files or ensure= changed). "
                "Record it again with THUNC_RECORD=1."
            )
        self.given += 1
        return self.replies[self.given - 1]

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        pass  # the next reply is already recorded

    def nudge(self, reply: Reply) -> None:
        pass


class Recorder:
    """Passes a run's conversation through, keeping each model reply to save once the run succeeds."""

    def __init__(self, inner: Conversation) -> None:
        self.inner = inner
        self.replies: list[Reply] = []

    def next(self) -> Reply:
        reply = self.inner.next()
        self.replies.append(reply)
        return reply

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        self.inner.results(results)

    def nudge(self, reply: Reply) -> None:
        self.inner.nudge(reply)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()


def _key(what: dict[str, Any]) -> str:
    blob = json.dumps({"format": FORMAT, **what}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()


def _folder() -> str:
    folder = recordings_dir()
    if folder is None:
        raise AssertionError("recordings are off")
    return folder


def _save(key: str, entry: dict[str, Any], stacklevel: int) -> None:
    """Save an entry. One that can't be written warns, not fails: the answer is valid and paid for."""
    folder = _folder()
    try:
        write(folder, key, entry)
    except OSError as exc:
        warnings.warn(f"thunc could not save a recording in {folder!r}: {exc}", RuntimeWarning, stacklevel=stacklevel)


def _reply(data: Any) -> Reply:
    from .native import Call, Reply

    if not isinstance(data, dict) or not isinstance(data["raw"], str) or not isinstance(data["calls"], list):
        raise ValueError("not a recorded reply")
    calls = [Call(c["id"], c["tool"], c["args"], c.get("problem")) for c in data["calls"]]
    if not all(isinstance(c.tool, str) and isinstance(c.args, dict) for c in calls):
        raise ValueError("not a recorded call")
    return Reply(calls, data["raw"], data.get("problem"), bool(data.get("same_turn")))
