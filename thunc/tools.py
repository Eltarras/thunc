"""The tools an agent uses on its working directory: list, read and search. Read-only for now.

Every path is resolved (symlinks included) and must land inside the working directory, so `..`,
absolute paths and links that point outside are refused.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from typing import Any

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


class Workdir:
    """A working directory and the read-only tools that act on it."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)

    def path(self, relative: str) -> str:
        """The real path for `relative`, which must stay inside the working directory."""
        full = os.path.realpath(os.path.join(self.root, relative))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ToolError(f"{relative!r} is outside the working directory; use a path inside it")
        return full

    def show(self, full: str) -> str:
        """A path as the model sees it: relative to the working directory, with / separators."""
        return os.path.relpath(full, self.root).replace(os.sep, "/")

    def list(self, path: str = ".") -> str:
        start = self.path(path)
        if not os.path.isdir(start):
            raise ToolError(f"{path!r} is not a folder")
        entries = []
        for full, is_dir in self._walk(start):
            entries.append(self.show(full) + ("/" if is_dir else ""))
            if len(entries) == MAX_LIST:
                entries.append(f"(stopped at {MAX_LIST} entries; list a subfolder to see more)")
                break
        return "\n".join(entries) or "(empty folder)"

    def read(self, path: str, offset: int = 1, limit: int = 400) -> str:
        full = self.path(path)
        if not os.path.isfile(full):
            raise ToolError(f"{path!r} is not a file" + ("; it is a folder, use list" if os.path.isdir(full) else ""))
        if offset < 1 or limit < 1:
            raise ToolError("offset and limit must be at least 1")
        with open(full, "rb") as f:
            data = f.read()
        if b"\0" in data[:8192]:
            raise ToolError(f"{path!r} is a binary file")
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
        files = [start] if os.path.isfile(start) else (full for full, is_dir in self._walk(start) if not is_dir)
        matches = []
        for full in files:
            for number, line in self._lines(full):
                if regex.search(line):
                    matches.append(f"{self.show(full)}:{number}: {line[:300]}")
                    if len(matches) == MAX_MATCHES:
                        return "\n".join(matches) + f"\n(stopped at {MAX_MATCHES} matches; narrow the search)"
        return "\n".join(matches) or "(no matches)"

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
}


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
