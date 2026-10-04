"""The tools an agent uses on its working directory: list, read, search, write and edit.

Every path is resolved (symlinks included) and must land inside the working directory, so `..`,
absolute paths and links that point outside are refused. Then the agent's permissions are checked
on the real path, so a link can't lead to a file the rules deny.

A file is only replaced or edited after the agent read it in this run, and only if it hasn't
changed on disk since: the agent never overwrites what it hasn't seen.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from typing import Any

from .permissions import Denied, Permissions

# Folders that are rarely what an agent is looking for, and can be huge.
SKIPPED_DIRS = frozenset(
    {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".thunc_cache", ".thunc_agents"}
)
MAX_RESULT = 20_000  # characters of one tool result sent back to the model
MAX_LIST = 300  # entries
MAX_MATCHES = 100
MAX_SEARCH_FILE = 1_000_000  # bytes; bigger files are skipped by search


class ToolError(Exception):
    """A tool call that can't be carried out. Its message goes back to the model, which can try again."""


class NotPermitted(ToolError):
    """A tool call the agent's permissions don't allow."""


class Workdir:
    """A working directory and the tools that act on it, within the agent's permissions."""

    def __init__(self, root: str, permissions: Permissions | None = None) -> None:
        self.root = os.path.realpath(root)
        self.permissions = permissions or Permissions()
        self.seen: dict[str, str] = {}  # real path -> sha256 of the content the agent last read or wrote
        self.changed: list[str] = []  # files created or changed in this run, as the model sees them

    def path(self, relative: str) -> str:
        """The real path for `relative`, which must stay inside the working directory."""
        full = os.path.realpath(os.path.join(self.root, relative))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ToolError(f"{relative!r} is outside the working directory; use a path inside it")
        return full

    def show(self, full: str) -> str:
        """A path as the model sees it: relative to the working directory, with / separators."""
        return os.path.relpath(full, self.root).replace(os.sep, "/")

    def readable(self, full: str) -> bool:
        try:
            self.permissions.check_read(self.show(full))
        except Denied:
            return False
        return True

    def _check(self, check: Callable[[str], None], full: str) -> None:
        try:
            check(self.show(full))
        except Denied as exc:
            raise NotPermitted(f"not permitted: {exc}") from None

    def list(self, path: str = ".") -> str:
        start = self.path(path)
        if not os.path.isdir(start):
            raise ToolError(f"{path!r} is not a folder")
        entries = []
        for full, is_dir in self._walk(start):
            if not is_dir and not self.readable(full):
                continue  # files the agent may not read aren't shown at all
            entries.append(self.show(full) + ("/" if is_dir else ""))
            if len(entries) == MAX_LIST:
                entries.append(f"(stopped at {MAX_LIST} entries; list a subfolder to see more)")
                break
        return "\n".join(entries) or "(empty folder)"

    def read(self, path: str, offset: int = 1, limit: int = 400) -> str:
        full = self.path(path)
        if not os.path.isfile(full):
            raise ToolError(f"{path!r} is not a file" + ("; it is a folder, use list" if os.path.isdir(full) else ""))
        self._check(self.permissions.check_read, full)
        if offset < 1 or limit < 1:
            raise ToolError("offset and limit must be at least 1")
        with open(full, "rb") as f:
            data = f.read()
        if b"\0" in data[:8192]:
            raise ToolError(f"{path!r} is a binary file")
        self.seen[full] = _digest(data)
        lines = data.decode("utf-8", errors="replace").splitlines()
        chunk = lines[offset - 1 : offset - 1 + limit]
        if not chunk:
            return f"(no lines from {offset}; the file has {len(lines)})"
        width = len(str(offset + len(chunk) - 1))
        body = "\n".join(f"{n:>{width}}  {line}" for n, line in enumerate(chunk, start=offset))
        end = offset + len(chunk) - 1
        if end < len(lines):
            body += f"\n(lines {offset}-{end} of {len(lines)}; read again with offset={end + 1} for more)"
        return body

    def search(self, pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from None
        start = self.path(path)
        if os.path.isfile(start):
            self._check(self.permissions.check_read, start)
            files: Iterator[str] | list[str] = [start]
        else:
            files = (full for full, is_dir in self._walk(start) if not is_dir and self.readable(full))
        matches = []
        for full in files:
            for number, line in self._lines(full):
                if regex.search(line):
                    matches.append(f"{self.show(full)}:{number}: {line[:300]}")
                    if len(matches) == MAX_MATCHES:
                        return "\n".join(matches) + f"\n(stopped at {MAX_MATCHES} matches; narrow the search)"
        return "\n".join(matches) or "(no matches)"

    def write(self, path: str, content: str) -> str:
        full = self.path(path)
        self._check(self.permissions.check_write, full)
        if os.path.isdir(full):
            raise ToolError(f"{path!r} is a folder")
        exists = os.path.exists(full)
        if exists:
            self._unchanged_since_read(path, full)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        self._save(full, content)
        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return f"{'replaced' if exists else 'created'} {self.show(full)} ({lines} lines)"

    def edit(self, path: str, old: str, new: str) -> str:
        full = self.path(path)
        self._check(self.permissions.check_write, full)
        if not os.path.isfile(full):
            raise ToolError(f"{path!r} is not a file; use write to create it")
        data = self._unchanged_since_read(path, full)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{path!r} isn't UTF-8 text, so it can't be edited") from None
        if not old:
            raise ToolError("old is empty; give the exact text to replace")
        if old not in text and "\r\n" in text:  # the file uses Windows line endings; the model wrote \n
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        count = text.count(old)
        if count != 1:
            problem = "isn't in the file" if count == 0 else f"appears {count} times"
            raise ToolError(
                f"the old text {problem}; it must appear exactly once, so include more of the lines around it"
            )
        self._save(full, text.replace(old, new, 1))
        return f"edited {self.show(full)}"

    def _unchanged_since_read(self, path: str, full: str) -> bytes:
        """The file's current content, if the agent read it in this run and it hasn't changed since."""
        with open(full, "rb") as f:
            data = f.read()
        if full not in self.seen:
            raise ToolError(f"read {path!r} before changing it")
        if self.seen[full] != _digest(data):
            raise ToolError(f"{path!r} changed on disk since you read it; read it again first")
        return data

    def _save(self, full: str, text: str) -> None:
        data = text.encode("utf-8")
        with open(full, "wb") as f:
            f.write(data)
        self.seen[full] = _digest(data)  # the agent knows what it wrote, so it can edit it again
        shown = self.show(full)
        if shown not in self.changed:
            self.changed.append(shown)

    def _walk(self, start: str) -> Iterator[tuple[str, bool]]:
        """(path, is_folder) for everything under `start`, in tree order: each folder's contents
        right after it. Links are not followed into folders, so nothing outside is reached."""
        try:
            entries = sorted(os.scandir(start), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in SKIPPED_DIRS:
                    yield entry.path, True
                    yield from self._walk(entry.path)
            elif entry.is_file():
                yield entry.path, False

    def _lines(self, full: str) -> Iterator[tuple[int, str]]:
        try:
            if os.path.getsize(full) > MAX_SEARCH_FILE:
                return
            with open(full, "rb") as f:
                data = f.read()
        except OSError:
            return
        if b"\0" in data[:8192]:
            return
        yield from enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1)


# name -> (method name, {argument: (type, required)}, description for the prompt)
TOOLS: dict[str, tuple[str, dict[str, tuple[type, bool]], str]] = {
    "list": (
        "list",
        {"path": (str, False)},
        '{"path": "."}  Files and folders under a folder, recursively (folders end in /).',
    ),
    "read": (
        "read",
        {"path": (str, True), "offset": (int, False), "limit": (int, False)},
        '{"path": "src/app.py", "offset": 1, "limit": 400}  Numbered lines of a text file; offset, limit optional.',
    ),
    "search": (
        "search",
        {"pattern": (str, True), "path": (str, False)},
        '{"pattern": "def main", "path": "."}  Lines matching a Python regular expression, as file:line: text.',
    ),
    "write": (
        "write",
        {"path": (str, True), "content": (str, True)},
        '{"path": "docs/new.md", "content": "..."}  Creates a file, or replaces one you read in this run, '
        "with the full content.",
    ),
    "edit": (
        "edit",
        {"path": (str, True), "old": (str, True), "new": (str, True)},
        '{"path": "src/app.py", "old": "exact text", "new": "replacement"}  Replaces text that appears exactly '
        "once in a file you read in this run.",
    ),
}
WRITE_TOOLS = ("write", "edit")


def run(workdir: Workdir, name: str, args: dict[str, Any]) -> str:
    """Carry out one tool call and return its result as text. Raises ToolError for a bad call."""
    method, params, _ = TOOLS[name]
    unknown = sorted(set(args) - set(params))
    if unknown:
        raise ToolError(f"{name} has no argument {', '.join(map(repr, unknown))}; it takes {sorted(params)}")
    for param, (kind, required) in params.items():
        if param not in args:
            if required:
                raise ToolError(f"{name} needs {param!r}")
        elif not isinstance(args[param], kind) or isinstance(args[param], bool):
            raise ToolError(f"{name}: {param!r} must be a {kind.__name__}")
    tool: Callable[..., str] = getattr(workdir, method)
    result = tool(**args)
    if len(result) > MAX_RESULT:
        result = result[:MAX_RESULT] + f"\n(cut at {MAX_RESULT} characters; ask for less, e.g. a smaller limit)"
    return result


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
