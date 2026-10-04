"""What an agent may do, written as short rule strings.

    permissions=["write:CHANGELOG.md", "write:docs/**", "!read:.env*", "!memory"]

- read, read:<glob>     read files (list and search only show readable files). Allowed everywhere
                        by default; giving any read: rule replaces that default with your rules.
- write, write:<glob>   create and edit files. Writing a file also lets the agent read it.
- run, run:<command>    run commands that start with these words: "run:git log" allows
                        "git log --oneline" but not "git push". run alone allows any command.
- memory                save notes with remember. Allowed by default.
- !<rule>               deny. A deny always wins over an allow. !read also stops writing.

Globs match the path relative to the working directory, with / separators: * stays within one
folder, ** crosses folders, ? is one character. "docs/" means everything under docs/.

Commands run without a shell, and a command can do anything its program can: "run:pytest" runs the
project's code. Permissions limit which tools the model uses, not what a permitted command does.
"""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

KINDS = ("read", "write", "run", "memory")


@dataclass(frozen=True)
class Rule:
    deny: bool
    kind: str
    pattern: str | None  # None: every path, or every command
    source: str  # as written, for messages

    def matches(self, path: str) -> bool:
        return self.pattern is None or _glob(self.pattern).fullmatch(path) is not None

    def matches_command(self, argv: Sequence[str]) -> bool:
        if self.pattern is None:
            return True
        words = split_command(self.pattern)
        return list(argv[: len(words)]) == words


class Denied(Exception):
    """An action the permissions don't allow. The message says which rule is missing or which denies it."""


class Permissions:
    def __init__(self, rules: Iterable[str] = ()) -> None:
        if isinstance(rules, str):
            raise ValueError('permissions= takes a list of rules, like ["write:docs/**"], not a single string')
        self.rules = [_parse(rule) for rule in rules]
        if not any(r.kind == "read" and not r.deny for r in self.rules):
            self.rules.insert(0, Rule(False, "read", None, "read (default)"))
        self.rules.insert(0, Rule(False, "memory", None, "memory (default)"))

    @property
    def written(self) -> list[str]:
        """The rules as given, without the defaults."""
        return [r.source for r in self.rules if not r.source.endswith("(default)")]

    def check_read(self, path: str) -> None:
        """Raise Denied unless `path` (relative, / separators) may be read."""
        deny = self._first(path, deny=True, kinds=("read",))
        if deny:
            raise Denied(f"reading {path!r} is denied by {deny.source!r}")
        if not self._first(path, deny=False, kinds=("read", "write")):
            raise Denied(f"reading {path!r} isn't allowed: no read rule matches it")

    def check_write(self, path: str) -> None:
        """Raise Denied unless `path` may be created or changed."""
        deny = self._first(path, deny=True, kinds=("read", "write"))
        if deny:
            raise Denied(f"writing {path!r} is denied by {deny.source!r}")
        if not self._first(path, deny=False, kinds=("write",)):
            allowed = [r.source for r in self.rules if r.kind == "write" and not r.deny]
            hint = f"; this agent may write: {', '.join(allowed)}" if allowed else ""
            raise Denied(f"writing {path!r} isn't allowed: no write rule matches it{hint}")

    def check_run(self, argv: Sequence[str]) -> None:
        """Raise Denied unless this command (already split into words) may run."""
        command = shlex.join(argv)
        for rule in self.rules:
            if rule.deny and rule.kind == "run" and rule.matches_command(argv):
                raise Denied(f"running {command!r} is denied by {rule.source!r}")
        if not any(not r.deny and r.kind == "run" and r.matches_command(argv) for r in self.rules):
            allowed = [r.source for r in self.rules if r.kind == "run" and not r.deny]
            hint = f"; this agent may run: {', '.join(allowed)}" if allowed else ""
            raise Denied(f"running {command!r} isn't allowed: no run rule matches it{hint}")

    def may(self, kind: str) -> bool:
        """Whether any action of this kind could be allowed, which decides the tools an agent is offered."""
        if any(r.deny and r.kind == kind and r.pattern is None for r in self.rules):
            return False
        return any(not r.deny and r.kind == kind for r in self.rules)

    def describe(self) -> str:
        """The rules in words, for the system prompt."""
        lines = []
        for kind, verb in (("read", "Read"), ("write", "Write")):
            allowed = [r for r in self.rules if r.kind == kind and not r.deny]
            denied = [r for r in self.rules if r.deny and (r.kind == kind or (kind == "write" and r.kind == "read"))]
            if not self.may(kind):
                lines.append(f"- {verb}: nothing.")
                continue
            where = "everything" if any(r.pattern is None for r in allowed) else ", ".join(_show(r) for r in allowed)
            if kind == "read":
                where += " (and anything you may write)"
            except_ = f", except {', '.join(_show(r) for r in denied)}" if denied else ""
            lines.append(f"- {verb}: {where}{except_}.")
        if self.may("run"):
            allowed = [r for r in self.rules if r.kind == "run" and not r.deny]
            denied = [r for r in self.rules if r.kind == "run" and r.deny]
            which = (
                "any command"
                if any(r.pattern is None for r in allowed)
                else ", ".join(f"{r.pattern} ..." for r in allowed)
            )
            except_ = f", except {', '.join(f'{r.pattern} ...' for r in denied)}" if denied else ""
            lines.append(f"- Run commands: {which}{except_}.")
        else:
            lines.append("- Run commands: none.")
        lines.append("- Save notes with remember: " + ("yes." if self.may("memory") else "no."))
        return "Your permissions:\n" + "\n".join(lines)

    def _first(self, path: str, *, deny: bool, kinds: tuple[str, ...]) -> Rule | None:
        for rule in self.rules:
            if rule.deny == deny and rule.kind in kinds and rule.matches(path):
                return rule
        return None


def _parse(text: str) -> Rule:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Not a permission rule: {text!r}")
    source = text.strip()
    deny = source.startswith("!")
    body = source[1:].strip() if deny else source
    kind, colon, pattern = body.partition(":")
    kind = kind.strip()
    if kind not in KINDS:
        raise ValueError(f"Unknown permission {source!r}; rules start with {', '.join(KINDS)}")
    if not colon:
        return Rule(deny, kind, None, source)
    if kind == "memory":
        raise ValueError(f"{source!r}: memory takes no path; use 'memory' or '!memory'")
    if kind == "run":
        try:
            words = split_command(pattern)
        except ValueError as exc:
            raise ValueError(f"{source!r}: {exc}") from None
        if not words:
            raise ValueError(f"{source!r} has no command after the colon")
        return Rule(deny, kind, shlex.join(words), source)
    pattern = pattern.strip().replace("\\", "/")
    while pattern.startswith("./"):
        pattern = pattern[2:]
    if not pattern:
        raise ValueError(f"{source!r} has no path after the colon")
    if pattern.startswith("/") or re.match(r"^[A-Za-z]:", pattern) or ".." in pattern.split("/"):
        raise ValueError(f"{source!r}: paths are relative to the working directory and stay inside it")
    if pattern.endswith("/"):
        pattern += "**"
    _glob(pattern)
    return Rule(deny, kind, pattern, source)


def split_command(text: str) -> list[str]:
    """A command line as words, quoted the way a shell quotes them. On Windows a backslash is part of
    a path, not an escape, so C:\\tools\\python.exe stays whole; quotes around a word are removed."""
    if sys.platform != "win32":
        return shlex.split(text)
    words = shlex.split(text, posix=False)
    return [w[1:-1] if len(w) >= 2 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]


def _show(rule: Rule) -> str:
    return rule.pattern or "everything"


_compiled: dict[str, re.Pattern[str]] = {}


def _glob(pattern: str) -> re.Pattern[str]:
    """A glob as a regex: * within one folder, ** across folders ("**/" also matches no folder)."""
    if pattern not in _compiled:
        out, i = "", 0
        while i < len(pattern):
            if pattern.startswith("**/", i):
                out, i = out + "(?:.*/)?", i + 3
            elif pattern.startswith("**", i):
                out, i = out + ".*", i + 2
            elif pattern[i] == "*":
                out, i = out + "[^/]*", i + 1
            elif pattern[i] == "?":
                out, i = out + "[^/]", i + 1
            else:
                out, i = out + re.escape(pattern[i]), i + 1
        _compiled[pattern] = re.compile(out)
    return _compiled[pattern]
