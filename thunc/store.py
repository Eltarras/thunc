"""An agent's folder on disk: its settings, its memory and a record of every run.

    .thunc_agents/<name>/
      agent.json     the agent's settings as of its last run
      memory.md      notes kept across runs; the agent adds to it with remember, you can edit it
      sessions/      one JSONL file per run: every step, then the result or the error
      .lock          locked while a run is active, so one run per agent at a time

The folder is under configure(agents_dir=...), THUNC_AGENTS_DIR or ./.thunc_agents. Nothing is
written until the agent's first run.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Iterator
from typing import Any, TextIO

from .config import agents_dir
from .errors import ThuncError

MEMORY_LIMIT = 25_000  # characters of memory.md put in the prompt; the newest notes are kept
NOTE_LIMIT = 500  # characters in one note
WAIT_WARNING_SECONDS = 5.0  # a run that waits this long for another run of its agent says so


def slug(name: str) -> str:
    """The folder name for an agent name: "Release notes" -> "release-notes"."""
    folder = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")
    if not folder:
        raise ValueError(f"Agent name {name!r} has no letters or digits to name its folder after")
    return folder


class Store:
    """One agent's folder."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.folder = os.path.abspath(os.path.join(agents_dir(), slug(name)))

    # --- settings ---

    def save_settings(self, settings: dict[str, Any]) -> dict[str, Any] | None:
        """Write agent.json. Returns the previous settings if they were different, else None.
        Raises ThuncError if the folder belongs to an agent with another name ("A b" and "a-b")."""
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, "agent.json")
        previous = None
        with contextlib.suppress(OSError, ValueError):
            with open(path, encoding="utf-8") as f:
                previous = json.load(f)
        if isinstance(previous, dict) and previous.get("name") not in (None, self.name):
            raise ThuncError(
                f"Agent {self.name!r} would share the folder {self.folder} with agent {previous['name']!r}; "
                "give one of them a different name"
            )
        if previous != settings:
            _write_atomic(path, json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
        return previous if isinstance(previous, dict) and previous != settings else None

    # --- memory ---

    @property
    def memory_path(self) -> str:
        return os.path.join(self.folder, "memory.md")

    def memory(self) -> str:
        """memory.md as the prompt gets it: the newest notes, up to MEMORY_LIMIT characters."""
        try:
            with open(self.memory_path, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        except FileNotFoundError:
            return ""
        if len(text) <= MEMORY_LIMIT:
            return text
        kept: list[str] = []
        size = 0
        for line in reversed(text.splitlines()):
            if size + len(line) + 1 > MEMORY_LIMIT:
                break
            kept.append(line)
            size += len(line) + 1
        if self.folder not in _warned:
            _warned.add(self.folder)
            warnings.warn(
                f"Agent {self.name!r}: {self.memory_path} is over {MEMORY_LIMIT} characters, so only its newest "
                "notes are sent. Trim the file to keep the ones that matter.",
                RuntimeWarning,
                stacklevel=4,
            )
        return "(older notes left out)\n" + "\n".join(reversed(kept))

    def remember(self, note: str) -> None:
        """Append one dated note to memory.md. Raises ValueError for an empty or overlong note."""
        note = " ".join(note.split())  # one line, so a note can't fake the start of another
        if not note:
            raise ValueError("the note is empty")
        if len(note) > NOTE_LIMIT:
            raise ValueError(f"the note is {len(note)} characters; keep it under {NOTE_LIMIT}")
        os.makedirs(self.folder, exist_ok=True)
        with _locks_lock:  # appends from threads of one process don't interleave
            ends_cleanly = True
            with contextlib.suppress(OSError):
                with open(self.memory_path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    if f.tell():
                        f.seek(-1, os.SEEK_END)
                        ends_cleanly = f.read(1) == b"\n"
            with open(self.memory_path, "a", encoding="utf-8") as f:
                f.write(("" if ends_cleanly else "\n") + f"- {time.strftime('%Y-%m-%d')}: {note}\n")

    # --- runs ---

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the agent for one run. A second run, from this process or another, waits its turn.

        The lock is the operating system's (flock on macOS and Linux, a byte-range lock on Windows),
        so it's released when its process ends in any way, even a crash or kill -9: a run never
        waits for one that no longer exists. If the wait goes on, a warning says who holds it."""
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, ".lock")
        in_process = _thread_lock(self.folder)
        if not in_process.acquire(timeout=WAIT_WARNING_SECONDS):
            self._warn_waiting(path)
            in_process.acquire()
        try:
            # The file stays: deleting it would let a waiter lock the old file while a newcomer
            # creates and locks a new one, and both would run.
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                started, warned = time.monotonic(), False
                while not _try_lock(fd):
                    if not warned and time.monotonic() - started >= WAIT_WARNING_SECONDS:
                        self._warn_waiting(path)
                        warned = True
                    time.sleep(0.1)
                try:
                    _write_pid(fd)  # for the warning another run shows while it waits
                    yield
                finally:
                    _unlock(fd)
            finally:
                os.close(fd)
        finally:
            in_process.release()

    def _warn_waiting(self, path: str) -> None:
        holder = ""
        with contextlib.suppress(OSError, UnicodeDecodeError):
            with open(path, encoding="utf-8") as f:
                holder = f.read().strip()
        who = f"process {holder}" if holder.isdigit() else "another run"
        warnings.warn(
            f"Agent {self.name!r} is waiting for {who} to finish its run: runs of one agent take turns. "
            f"It goes ahead as soon as that run ends or its process stops (lock: {path}).",
            RuntimeWarning,
            stacklevel=5,
        )

    def session(self, task: str) -> Session:
        """A new run record in sessions/, named after the time and the task."""
        folder = os.path.join(self.folder, "sessions")
        os.makedirs(folder, exist_ok=True)
        now = time.time()  # microseconds in the name, so the files sort in the order the runs started
        stem = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(now)) + f".{int(now % 1 * 1e6):06d}Z-" + slug(task)
        for n in range(1, 1000):
            path = os.path.join(folder, stem + ("" if n == 1 else f"-{n}") + ".jsonl")
            try:
                return Session(path, open(path, "x", encoding="utf-8", errors="backslashreplace"))
            except FileExistsError:
                continue
        raise ThuncError(f"Too many runs of {task!r} in one second in {folder}")


class Session:
    """One run's record: a JSON line per event."""

    def __init__(self, path: str, file: TextIO) -> None:
        self.path = path
        self._file = file

    def write(self, event: str, **fields: Any) -> None:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except (RecursionError, TypeError, ValueError):
            line = json.dumps({"time": entry["time"], "event": event, "fields": repr(fields)[:2000]})
        for separator in "\x85  ":  # valid in JSON, but they split lines for str.splitlines()
            line = line.replace(separator, f"\\u{ord(separator):04x}")
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


_warned: set[str] = set()
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _thread_lock(folder: str) -> threading.Lock:
    with _locks_lock:
        return _locks.setdefault(folder, threading.Lock())


if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # the first byte; it needn't exist
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _write_pid(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, str(os.getpid()).encode())


def _write_atomic(path: str, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
