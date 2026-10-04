"""The tools an agent uses on its working directory: list, read, search, write, edit and run.

Every path is resolved (symlinks included) and must land inside the working directory, so `..`,
absolute paths and links that point outside are refused. Then the agent's permissions are checked
on the real path, so a link can't lead to a file the rules deny.

A file is only replaced or edited after the agent read it in this run, and only if it hasn't
changed on disk since: the agent never overwrites what it hasn't seen.

Commands run in the working directory (or a folder inside it) without a shell, unless the "shell"
permission allows one, with a minimal environment (your API keys aren't in it), a timeout that also
stops the processes they start, and the start and end of their output kept when it's long: the
first error is often at the start, the summary at the end.

In a git repository, list and search leave out what git ignores (build output, caches, vendored
code), as git ls-files sees it; a folder named explicitly is still searched.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, MutableSequence
from typing import Any

from .permissions import Denied, Permissions, join_command, split_command
from .runs import Command

# Folders that are rarely what an agent is looking for, and can be huge.
SKIPPED_DIRS = frozenset(
    {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".thunc_cache", ".thunc_agents"}
)
MAX_RESULT = 20_000  # characters of one tool result sent back to the model
MAX_LIST = 300  # entries
MAX_MATCHES = 100
MAX_SEARCH_FILE = 1_000_000  # bytes; bigger files are skipped by search
MAX_TRACKED = 20_000  # files; past this, changes made by commands aren't tracked
MAX_OUTPUT = 18_000  # characters of a command's output sent back: its start and its end
OUTPUT_HEAD = 5_000  # of those, characters from the start
# Environment variables a command gets by default: enough to find programs and run them, nothing else.
PASSED_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",  # Windows needs these to start programs
)
# Words a shell would treat specially. Without a shell they'd reach the program as plain arguments.
SHELL_OPERATORS = frozenset({"&&", "||", "|", ";", "&", ">", ">>", "<", "<<", "2>", "2>&1", "&>", "|&"})


def command_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment commands run with: PASSED_ENV from this process, plus `extra`."""
    env = {name: os.environ[name] for name in PASSED_ENV if name in os.environ}
    env.update(extra or {})
    return env


class ToolError(Exception):
    """A tool call that can't be carried out. Its message goes back to the model, which can try again."""


class NotPermitted(ToolError):
    """A tool call the agent's permissions don't allow."""


class Workdir:
    """A working directory and the tools that act on it, within the agent's permissions."""

    def __init__(
        self,
        root: str,
        permissions: Permissions | None = None,
        *,
        env: Mapping[str, str] | None = None,
        command_timeout: float = 120.0,
        deadline: float | None = None,
    ) -> None:
        self.root = os.path.realpath(root)
        self.permissions = permissions or Permissions()
        self.shell = self.permissions.may("shell")  # commands run in a shell: pipes, && and cd work
        self.env = command_env(env)
        self.command_timeout = command_timeout
        self.deadline = deadline  # time.monotonic() when the run's own time limit is up, if it has one
        self.seen: dict[str, str] = {}  # real path -> sha256 of the content the agent last read or wrote
        self.cancelled: Callable[[], bool] | None = None
        self.changed: list[str] = []  # files created or changed in this run, as the model sees them
        self.commands: list[Command] = []  # commands run in this run

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
        hidden: list[str] = []
        for full, is_dir in self._walk(start, self._ignored(start, hidden)):
            if not is_dir and not self.readable(full):
                continue  # files the agent may not read aren't shown at all
            entries.append(self.show(full) + ("/" if is_dir else ""))
            if len(entries) == MAX_LIST:
                entries.append(f"(stopped at {MAX_LIST} entries; list a subfolder to see more)")
                break
        if hidden:
            shown = ", ".join(hidden[:5]) + (", ..." if len(hidden) > 5 else "")
            entries.append(f"(left out because git ignores them: {shown}; list one by name to see inside)")
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

    def search(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from None
        wanted = _file_glob(glob) if glob else None
        start = self.path(path)
        if os.path.isfile(start):
            self._check(self.permissions.check_read, start)
            files: Iterator[str] | list[str] = [start]
        else:
            files = (
                full
                for full, is_dir in self._walk(start, self._ignored(start))
                if not is_dir and self.readable(full) and (wanted is None or wanted(self.show(full)))
            )
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

    def run(self, command: str, cwd: str = ".") -> str:
        folder = self.path(cwd)
        if not os.path.isdir(folder):
            raise ToolError(f"cwd {cwd!r} is not a folder in the working directory")
        if self.shell:  # the "shell" permission: any command line, run by the system shell
            if not command.strip():
                raise ToolError("the command is empty")
            argv = ["cmd", "/c", command] if sys.platform == "win32" else ["/bin/sh", "-c", command]
            shown = command
        else:
            try:
                argv = split_command(command)
            except ValueError as exc:
                raise ToolError(f"can't read the command: {exc}") from None
            if not argv:
                raise ToolError("the command is empty")
            operators = [word for word in argv if word in SHELL_OPERATORS]
            if operators:
                raise ToolError(
                    f"commands run without a shell, so {operators[0]!r} doesn't work. Run one command at a time "
                    "(use cwd to run it in a folder), and read files with read instead of redirecting output"
                )
            try:
                self.permissions.check_run(argv)
            except Denied as exc:
                raise NotPermitted(f"not permitted: {exc}") from None
            shown = join_command(argv)
        limit = self.command_timeout
        if self.deadline is not None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise ToolError("the run's time limit is up; call finish with what you have")
            limit = min(limit, left)
        before = self._snapshot()
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                argv,
                cwd=folder,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=sys.platform != "win32",  # its own process group, so a timeout stops all of it
            )
        except FileNotFoundError:
            raise ToolError(f"{argv[0]!r} was not found") from None
        except OSError as exc:
            raise ToolError(f"{argv[0]!r} couldn't be started: {exc.strerror or exc}") from None
        exit_code: int | None
        try:
            if self.cancelled is None:
                output, _ = process.communicate(timeout=limit)
            else:
                while True:
                    if self.cancelled():
                        raise ToolError("command cancelled")
                    remaining = limit - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, limit)
                    try:
                        output, _ = process.communicate(timeout=min(0.25, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            exit_code = process.returncode
            status = f"exit code {exit_code}"
        except subprocess.TimeoutExpired:
            exit_code = None
            _stop(process)
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:  # something it started still holds the output open
                output = b""
            status = f"stopped after {limit:g}s, the time limit"
        except BaseException:
            _stop(process)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=5)
            raise
        seconds = time.monotonic() - started
        where = "" if folder == self.root else f" (in {self.show(folder)})"
        self.commands.append(Command(shown + where, exit_code, round(seconds, 3)))
        self._note_changes(before, self._snapshot())
        text = output.decode("utf-8", errors="replace") if output else ""
        if len(text) > MAX_OUTPUT:  # keep the start (often the first error) and the end (the summary)
            tail = MAX_OUTPUT - OUTPUT_HEAD
            left_out = len(text) - MAX_OUTPUT
            text = f"{text[:OUTPUT_HEAD]}\n(... {left_out} characters left out ...)\n{text[-tail:]}"
        return f"{status} ({seconds:.1f}s)\n{text}".rstrip()

    def _snapshot(self) -> dict[str, tuple[int, int]] | None:
        """(modification time, size) of every file in the workdir, to see what a command changed.
        None for a workdir too big to scan for every command."""
        found: dict[str, tuple[int, int]] = {}
        for full, is_dir in self._walk(self.root):
            if is_dir:
                continue
            try:
                info = os.stat(full)
            except OSError:
                continue
            found[full] = (info.st_mtime_ns, info.st_size)
            if len(found) > MAX_TRACKED:
                return None
        return found

    def _note_changes(
        self, before: dict[str, tuple[int, int]] | None, after: dict[str, tuple[int, int]] | None
    ) -> None:
        """Add the files a command created, changed or deleted to the run's changed files."""
        if before is None or after is None:
            return
        for full in sorted(set(before) | set(after)):
            if before.get(full) != after.get(full):
                shown = self.show(full)
                if shown not in self.changed:
                    self.changed.append(shown)

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

    def _walk(self, start: str, ignored: Callable[[str, bool], bool] | None = None) -> Iterator[tuple[str, bool]]:
        """(path, is_folder) for everything under `start`, in tree order: each folder's contents
        right after it. Links are not followed into folders, so nothing outside is reached.
        `ignored(path, is_folder)` leaves out entries (and a folder's whole contents)."""
        try:
            entries = sorted(os.scandir(start), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in SKIPPED_DIRS and not (ignored and ignored(entry.path, True)):
                    yield entry.path, True
                    yield from self._walk(entry.path, ignored)
            elif entry.is_file() and not (ignored and ignored(entry.path, False)):
                yield entry.path, False

    def _ignored(self, start: str, hidden: MutableSequence[str] | None = None) -> Callable[[str, bool], bool] | None:
        """What git ignores under `start`, as a test for _walk; None outside a git repository, or when
        `start` is itself ignored (the agent asked for it by name). Ignored folders are added to `hidden`."""
        visible = self._git_files()
        if visible is None:
            return None
        folders = {"."}
        for file in visible:
            parts = file.split("/")[:-1]
            folders.update("/".join(parts[: i + 1]) for i in range(len(parts)))
        if self.show(start) not in folders:
            return None

        def ignored(full: str, is_dir: bool) -> bool:
            shown = self.show(full)
            if shown in (folders if is_dir else visible):
                return False
            if is_dir and hidden is not None:
                hidden.append(shown + "/")
            return True

        return ignored

    def _git_files(self) -> set[str] | None:
        """Files git doesn't ignore (tracked, or untracked and not ignored), relative to the root;
        None when the root isn't in a git repository or git isn't available."""
        try:
            done = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=self.root,
                env=self.env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        return {name for name in done.stdout.decode("utf-8", "surrogateescape").split("\0") if name}

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
        '{"path": "."}  Files and folders under a folder, recursively (folders end in /). In a git repository, '
        "what git ignores is left out.",
    ),
    "read": (
        "read",
        {"path": (str, True), "offset": (int, False), "limit": (int, False)},
        '{"path": "src/app.py", "offset": 1, "limit": 400}  Numbered lines of a text file; offset, limit optional.',
    ),
    "search": (
        "search",
        {"pattern": (str, True), "path": (str, False), "glob": (str, False)},
        '{"pattern": "def main", "path": ".", "glob": "*.py"}  Lines matching a Python regular expression '
        "(start it with (?i) to ignore case), as file:line: text. glob (optional) limits the files searched: "
        '"*.py" matches by file name, "src/**/*.ts" by path. In a git repository, what git ignores is left out.',
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
    "run": (
        "run",
        {"command": (str, True), "cwd": (str, False)},
        '{"command": "pytest -q tests/test_app.py", "cwd": "."}  Runs a command and returns its exit code and '
        "output (the start and end of long output). cwd (optional) is the folder to run it in, inside the "
        "working directory.",
    ),
}
RUN_NO_SHELL = " There is no shell: no pipes, &&, cd, redirects or $VARIABLES."
RUN_SHELL = " It runs in a shell, so pipes, && and redirects work."


def description(name: str, shell: bool = False) -> str:
    """A built-in tool's description for the prompt: an example of its arguments, then what it does."""
    text = TOOLS[name][2]
    if name == "run":
        text += RUN_SHELL if shell else RUN_NO_SHELL
    return text


# The permission each tool needs before the agent is offered it ("shell" also allows run).
NEEDS = {"write": "write", "edit": "write", "run": "run"}


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


def _file_glob(pattern: str) -> Callable[[str], bool]:
    """search's glob: a pattern without / matches the file name in any folder, one with / the whole path."""
    from .permissions import _glob

    pattern = pattern.strip().replace("\\", "/").removeprefix("./")
    if not pattern:
        raise ToolError("glob is empty; leave it out to search every file")
    regex = _glob(pattern)
    if "/" in pattern:
        return lambda path: regex.fullmatch(path) is not None
    return lambda path: regex.fullmatch(path.rsplit("/", 1)[-1]) is not None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Stop a command and everything it started."""
    with contextlib.suppress(OSError):
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
