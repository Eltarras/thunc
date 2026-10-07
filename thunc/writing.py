"""thunc write, functions that write themselves: @thunc.function(write=True) writes its own body on its
first call.

    @thunc.function(write=True)
    def minutes(duration: str) -> int:
        \"\"\"Convert a duration like '1h 30m', '90 min' or '2 hours' to whole minutes.\"\"\"
        ...

The first call that can write it:

1. checks that writing is allowed: a real, writable file in the project, unchanged since it was
   imported, outside CI (see _refusal);
2. holds a lock for the function, so of several calls at once one writes and the rest wait;
3. starts three requests side by side:
   - this call's answer, as an ordinary @thunc.function would get it;
   - a draft of the body, given the whole file and any project types in the signature, or word
     that the task can't be written as rules (prompts.WRITER);
   - five test calls (prompts.CASES), each then answered by the model on its own. They come from
     their own request, so they don't share the draft's blind spots;
4. checks the draft: lint, compile, run on this call and the test calls within a time limit, and
   compare with the model's answers. Failures go back for a new draft, up to ROUNDS drafts;
5. writes the file (source.splice): the body in place of `...`, the decorator gone, the test calls
   as doctest examples;
6. compiles the written function into the running program, and the call runs it.

If no draft passes or it can't be written, the call returns the model's answer, the file is left as
it was, and the reason is saved in .thunc_write/ so later runs don't try again until the docstring
or signature changes. After a write, the function is plain Python: the model isn't asked again.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import sys
import threading
import time
import typing
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from . import source
from .backends import TYPED_BACKENDS
from .config import resolve_backend
from .core import _call, _ensured, _plain
from .errors import ThuncError
from .prompts import CASES, WRITER
from .schema import describe, short_repr, shorten, validate
from .store import _thread_lock, _try_lock, _unlock

ROUNDS = 3  # drafts per function before it stays a model call
MIN_CASES = 3  # of the test calls, how many need a model answer to check against
CASE_SECONDS = 10.0  # how long the draft may take on one test call
FOLDER = ".thunc_write"
# The system prompt of the draft and test-call requests, in place of the program's own.
CODER = "You are an expert Python programmer working inside a computer program."
# Modules the written code may not import, though they're in the standard library.
BANNED_MODULES = frozenset(
    "subprocess socket ctypes multiprocessing http urllib ftplib smtplib telnetlib shutil signal "
    "importlib pickle marshal webbrowser".split()
)
BANNED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint", "input"})
BANNED_OS = frozenset({"system", "popen", "remove", "unlink", "rmdir", "removedirs", "rename", "replace", "kill"})


@dataclasses.dataclass
class Draft:
    """What the model returns when asked to write the body."""

    can_write: bool
    reason: str
    imports: list[str]
    body: str


@dataclasses.dataclass
class Case:
    """A test call with the model's answer for it."""

    text: str  # as written, like minutes('2 hours')
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    expected: Any


class Rejected(Exception):
    """A draft failed its checks; args[0] lists the problems, as the next draft request shows them."""


class NotWritten(Exception):
    """The function wasn't written; the message says why."""


# The call being answered while the function is written: its arguments and the model's answer.
# None when it's written ahead of any call (thunc write).
Answered = tuple[tuple[Any, ...], dict[str, Any], Any]


class _Timer:
    """How long each part of a write took, for the message at the end."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.parts: dict[str, list[float]] = {}

    def timed(self, part: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                self.parts.setdefault(part, []).append(time.monotonic() - started)

        return run

    def report(self) -> str:
        shown = [f"{part} {' + '.join(f'{s:.0f}s' for s in times)}" for part, times in self.parts.items()]
        return f"{time.monotonic() - self.started:.0f}s ({', '.join(shown)}; side by side)"


class Writer:
    """The writing state of one @thunc.function(write=True). Made when the function is decorated."""

    def __init__(self, func: Callable[..., Any], spec: Any, options: dict[str, Any], name: str) -> None:
        self.func, self.spec, self.options, self.name = func, spec, options, name
        self.short = func.__name__
        self.path = _source_file(func)
        self.digest = _digest(self.path) if self.path else None  # the file as it was imported
        self.impl: Callable[..., Any] | None = None  # the written function, once there is one
        self.off: str | None = None  # why it won't be written in this process, once known
        self._lock = threading.Lock()

    # --- the call --------------------------------------------------------------------------

    def answer(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """This call's answer from the model, after writing the function if this call can.
        Once self.impl is set the caller runs it instead, and the answer is None."""
        if self.impl is None and self.off is None:
            with self._lock:
                if self.impl is not None:
                    return None
                if self.off is None:
                    return self._first_call(args, kwargs)
        if self.impl is not None:
            return None
        return self._ask(args, kwargs)

    def _ask(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        spec, options = self.spec, self.options
        inputs = spec.inputs(args, kwargs)
        return _call(spec.instructions, inputs, spec.returns, name=self.name, module=self.func.__module__, **options)

    def _first_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        reason = self._refusal()
        if reason:
            self._stop(f"{self.short}() won't be written: {reason}. It answers through the model.")
            return self._ask(args, kwargs)
        root, key = self._root_and_key()
        with _file_lock(root, key):
            opened = self._open(root, key, use_memo=True)
            if isinstance(opened, str):
                self._stop(opened)
                return self._ask(args, kwargs)
            text, encoding, node = opened
            _say(f"writing {self.short}() in {_relative(self.path or '')} (first call)")
            timer = _Timer()
            pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="thunc-write")
            try:
                answering = pool.submit(timer.timed("answer", self._ask), args, kwargs)
                work = self._start(pool, timer, text, node, root, (args, kwargs))
                answer = answering.result()  # its error is the call's error, as without write=True
                try:
                    self._write(root, key, text, encoding, node, work, timer, (args, kwargs, answer))
                except NotWritten as exc:
                    self._stop(f"{self.short}() stays a model call: {exc}")
                except (ThuncError, source.SourceError, OSError) as exc:
                    self._stop(f"{self.short}() wasn't written: {exc}. It answers through the model for now.")
                except Exception as exc:  # a bug in writing mustn't fail a call that already has its answer
                    problem = f"{type(exc).__name__}: {shorten(str(exc), 300)}"
                    self._stop(f"{self.short}() wasn't written ({problem}). It answers through the model for now.")
                return answer
            finally:
                pool.shutdown(wait=False, cancel_futures=True)  # after a failure, nothing waits for the rest

    def write_now(self, dry_run: bool = False) -> str:
        """Write the function ahead of any call, for `thunc write`, and return the file's new text.
        With dry_run the file is left as it is and nothing is saved. Raises NotWritten saying why when
        it isn't written. A reason saved by an earlier attempt is ignored: this was asked for."""
        if self.impl is not None:
            raise NotWritten("it was written already")
        reason = self._refusal()
        if reason:
            raise NotWritten(reason)
        if self.spec.skip:
            raise NotWritten("it's a method, and thunc write has no instance to test it with; call it once instead")
        root, key = self._root_and_key()
        with self._lock, _file_lock(root, key):
            opened = self._open(root, key, use_memo=False)
            if isinstance(opened, str):
                raise NotWritten(opened)
            text, encoding, node = opened
            _say(f"writing {self.short}() in {_relative(self.path or '')}" + (" (dry run)" if dry_run else ""))
            timer = _Timer()
            pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="thunc-write")
            try:
                work = self._start(pool, timer, text, node, root, None)
                return self._write(root, key, text, encoding, node, work, timer, None, dry_run)
            except (ThuncError, source.SourceError, OSError) as exc:
                raise NotWritten(str(exc)) from exc
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

    def _root_and_key(self) -> tuple[str, str]:
        assert self.path is not None
        root = _project_root(self.path)
        assert root is not None
        return root, self._key(root)

    def _open(self, root: str, key: str, use_memo: bool) -> tuple[str, str, source.Function] | str:
        """The file's text, its encoding and the function's def, or a message saying why it can't be written."""
        assert self.path is not None
        if _digest(self.path) != self.digest:
            return (
                f"{self.short}() won't be written: {_relative(self.path)} changed since it was imported "
                "(written by another process?). Restart the program to use the file as it is now."
            )
        saved = _read_memo(root, key) if use_memo else None
        if saved is not None:
            return (
                f"{self.short}() stays a model call: {saved}. Change its docstring or signature to try again, "
                f"run `thunc write`, or delete {FOLDER}/{key}.json."
            )
        try:
            text, encoding = source.read(self.path)
            node = source.find(ast.parse(text), self.func.__qualname__, self.func.__code__.co_firstlineno)
        except (source.SourceError, OSError, SyntaxError, UnicodeDecodeError) as exc:
            return f"{self.short}() won't be written: {exc}"
        return text, encoding, node

    def _start(
        self,
        pool: ThreadPoolExecutor,
        timer: _Timer,
        text: str,
        node: source.Function,
        root: str,
        call: tuple[tuple[Any, ...], dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], Future[Draft], Future[list[Case]]]:
        """Start the draft and the test calls side by side."""
        inputs = self._draft_inputs(text, node, root, call)
        drafting = pool.submit(timer.timed("draft", self._draft), inputs, text)
        casing = pool.submit(timer.timed("test calls", self._cases), node, text, call)
        return inputs, drafting, casing

    def _stop(self, message: str) -> None:
        self.off = message
        warnings.warn(f"thunc: {message}", RuntimeWarning, 5)

    # --- writing ---------------------------------------------------------------------------

    def _write(
        self,
        root: str,
        key: str,
        text: str,
        encoding: str,
        node: source.Function,
        work: tuple[dict[str, Any], Future[Draft], Future[list[Case]]],
        timer: _Timer,
        answered: Answered | None,
        dry_run: bool = False,
    ) -> str:
        """Check drafts until one passes, then write it and load it; returns the file's new text.
        Raises NotWritten when it can't be written or no draft passes, saving the reason in .thunc_write/."""
        assert self.path is not None
        inputs, drafting, casing = work
        line = source.first_line(node)
        draft = drafting.result()
        problems: list[str] = []
        for round in range(1, ROUNDS + 1):
            if round > 1:
                inputs = {**inputs, "previous_body": draft.body, "problems": "\n".join(f"- {p}" for p in problems)}
                draft = timer.timed("draft", self._draft)(inputs, text)
            if not draft.can_write:
                reason = " ".join(draft.reason.split()) or "the model said it can't be written as rules"
                if not dry_run:
                    _write_memo(root, key, f"the model said it can't be written as code ({shorten(reason, 300)})")
                raise NotWritten(shorten(reason, 300))
            try:
                examples, checked = self._verify(text, line, draft, casing.result(), answered)
                break
            except Rejected as rejected:
                problems = list(rejected.args[0])
                _say(
                    f"draft {round} of {ROUNDS} failed {len(problems)} check(s)"
                    + ("" if round == ROUNDS else "; asking for another")
                )
        else:
            summary = "; ".join(problems[:3])
            if not dry_run:
                _write_memo(root, key, f"no draft passed its checks in {ROUNDS} tries (last: {shorten(summary, 300)})")
            raise NotWritten(f"no draft passed its checks ({shorten(summary, 300)})")
        _say(f"checked against {checked} model answers: all agree")
        note = f"Written by thunc from the docstring on {dt.date.today().isoformat()}. Review it."
        final, start = source.splice(text, self.func.__qualname__, line, draft.body, draft.imports, examples, note)
        if dry_run:
            return final
        source.write(self.path, final, encoding)
        namespace = self.func.__globals__
        source.run_imports(draft.imports, namespace)
        self.impl = source.load(final, self.path, self.func.__qualname__, start, namespace, self._owner())
        end = source.find(ast.parse(final), self.func.__qualname__, start).end_lineno
        _say(
            f"wrote {_relative(self.path)} lines {start}-{end} in {timer.report()}. "
            f"Removed @thunc.function. Review: git diff {_relative(self.path)}"
        )
        return final

    def _draft_inputs(
        self, text: str, node: source.Function, root: str, call: tuple[tuple[Any, ...], dict[str, Any]] | None
    ) -> dict[str, Any]:
        """What the draft request gets: the function, its file, and the project types it uses."""
        assert self.path is not None
        inputs: dict[str, Any] = {
            "function": source.function_source(text, node),
            "file": _relative(self.path, root),
            "module": text,
            "return_type": describe(self.spec.returns),
            "example_call": self._example(call),
        }
        types = _project_types(self.func, root, self.path)
        if types:
            inputs["types"] = types
        return inputs

    def _draft(self, inputs: dict[str, Any], text: str) -> Draft:
        """One draft from the model. Lint problems go back to it with the request's own retries."""
        draft: Draft = _call(
            WRITER,
            inputs,
            Draft,
            3,
            lambda d: self._well_formed(d, text),
            self.options.get("backend"),
            self.options.get("model"),
            False,
            f"{self.name} (draft)",
            self.func.__module__,
            CODER,
        )
        return draft

    def _cases(
        self, node: source.Function, text: str, call: tuple[tuple[Any, ...], dict[str, Any]] | None
    ) -> list[Case]:
        """Test calls from the model, each then answered by the model on its own, side by side. A call
        the model can't answer is dropped; fewer than MIN_CASES left is a ThuncError."""
        calls: list[str] = _call(
            CASES,
            {
                "function": source.function_source(text, node),
                "example_call": self._example(call),
            },
            list[str],
            3,
            self._good_cases,
            self.options.get("backend"),
            self.options.get("model"),
            False,
            f"{self.name} (test calls)",
            self.func.__module__,
            CODER,
        )
        bound_self = call[0][:1] if call and self.spec.skip else ()
        parsed = []
        for text_call in calls:
            case_args, case_kwargs = _case_arguments(text_call, self.short, self._classes())
            parsed.append((text_call, (*bound_self, *case_args), case_kwargs))
        cases: list[Case] = []
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="thunc-check") as pool:
            futures = [pool.submit(self._ask, a, k) for _, a, k in parsed]
            for (text_call, a, k), future in zip(parsed, futures, strict=True):
                try:
                    cases.append(Case(text_call, a, k, future.result()))
                except (ThuncError, TypeError, ValueError):
                    continue  # the model couldn't answer it, or it doesn't fit the signature
        if len(cases) < MIN_CASES:
            raise ThuncError(f"only {len(cases)} test calls got an answer from the model to check against")
        return cases

    def _classes(self) -> dict[str, type]:
        """The classes a test call may build its arguments with: those defined in the function's module,
        and those in its signature (a dataclass imported from elsewhere in the project)."""
        namespace = self.func.__globals__
        module = namespace.get("__name__")
        classes = {k: v for k, v in namespace.items() if isinstance(v, type) and v.__module__ == module}
        for tp in _signature_types(self.func):
            if tp.__module__ != "builtins" and namespace.get(tp.__name__) is tp:
                classes[tp.__name__] = tp
        return classes

    def _good_cases(self, calls: list[str]) -> bool:
        if len(calls) < MIN_CASES:
            raise ValueError(f"give at least {MIN_CASES} calls (five are asked for)")
        for call in calls:
            _case_arguments(call, self.short, self._classes())
        return True

    def _well_formed(self, draft: Draft, text: str) -> bool:
        """Cheap checks, sent back with the draft request's retries (the message is cut to 200 characters)."""
        if not draft.can_write:
            return True
        problems = _lint(draft, text)
        if problems:
            raise ValueError(problems[0])
        return True

    def _verify(
        self,
        text: str,
        line: int,
        draft: Draft,
        cases: list[Case],
        answered: Answered | None,
    ) -> tuple[list[tuple[str, str]], int]:
        """Check a draft; raises Rejected with the problems. Returns the doctest examples and how many
        model answers the draft was checked against."""
        problems = _lookup_table(draft.body, cases)
        if problems:
            raise Rejected(problems)
        try:
            edited, start = source.splice(text, self.func.__qualname__, line, draft.body, draft.imports)
            namespace = dict(self.func.__globals__)  # a copy: nothing reaches the module until it passes
            source.run_imports(draft.imports, namespace)
            candidate = source.load(
                edited, self.path or "<thunc>", self.func.__qualname__, start, namespace, self._owner()
            )
        except source.SourceError as exc:
            raise Rejected([str(exc)]) from None
        except Exception as exc:  # an import that fails, a name error at definition time
            raise Rejected([f"the code doesn't load: {type(exc).__name__}: {shorten(str(exc), 300)}"]) from None

        checks: list[tuple[str, str | None, tuple[Any, ...], dict[str, Any], Any]] = []
        if answered is not None:
            args, kwargs, answer = answered
            this_call = self._call_text(args, kwargs)  # None when its arguments aren't literals
            checks.append((this_call or f"{self.short}(<this call>)", this_call, args, kwargs, answer))
        checks += [(case.text, case.text, case.args, case.kwargs, case.expected) for case in cases]
        examples: list[tuple[str, str]] = []
        for label, example, a, k, expected in checks:
            outcome = _run_case(candidate, a, k, self.spec)
            if isinstance(outcome, _Failed):
                problems.append(f"{label}: {outcome}; the model answered {_shown(expected)}")
                continue
            try:
                value = _ensured(_as_type(outcome, self.spec.returns), self.options.get("ensure"))
            except ValueError as exc:
                problems.append(f"{label}: returned {short_repr(outcome)}, which fails: {shorten(str(exc), 200)}")
                continue
            if not _same(value, expected):
                problems.append(f"{label}: returned {_shown(value)}, but the model answered {_shown(expected)}")
                continue
            # Doctest can't show a method (no instance) or an awaitable.
            if example and not self.spec.skip and not self.spec.is_async and len(repr(outcome)) <= 200:
                examples.append((_normal_call(example), repr(outcome)))
        if problems:
            raise Rejected(problems)
        return examples, len(checks)

    # --- where and whether -----------------------------------------------------------------

    def _refusal(self) -> str | None:
        """Why the function can't be written here, or None."""
        path = self.path
        if path is None or not os.path.isfile(path):
            return "its source file can't be found (a REPL, a notebook cell or exec)"
        if "<locals>" in self.func.__qualname__:
            return "it's defined inside another function"
        if _installed(path):
            return f"{path} is installed code, not part of your project"
        if _project_root(path) is None:
            return f"{_relative(path)} isn't in a git repository or under the current directory"
        setting = os.environ.get("THUNC_WRITE", "").strip().lower()
        if setting in ("0", "false", "no", "off"):
            return "THUNC_WRITE is off"
        ci = os.environ.get("CI", "").strip().lower()
        if ci and ci not in ("0", "false", "no"):
            return "it's running in CI (CI is set)"
        if not os.access(path, os.W_OK):
            return f"{_relative(path)} is read-only"
        if _digest(path) != self.digest:
            return f"{_relative(path)} changed since it was imported"
        try:
            backend = resolve_backend(self.options.get("backend"))
        except ThuncError as exc:
            return str(exc)
        if backend in TYPED_BACKENDS:
            return f"the {backend} backend can't run the agent that writes it"
        return None

    def _key(self, root: str) -> str:
        """What the saved outcome is for: the file, the function, its prompt and its signature."""
        blob = json.dumps(
            {
                "file": _relative(self.path or "", root),
                "function": self.func.__qualname__,
                "instructions": self.spec.instructions,
                "signature": str(self.spec.sig),
                "returns": describe(self.spec.returns),
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _owner(self) -> type | None:
        """The class a method is defined in, found from the module's globals."""
        *classes, _ = self.func.__qualname__.split(".")
        owner: Any = None
        for i, cls in enumerate(classes):
            owner = self.func.__globals__.get(cls) if i == 0 else getattr(owner, cls, None)
        return owner if isinstance(owner, type) else None

    def _example(self, call: tuple[tuple[Any, ...], dict[str, Any]] | None) -> str:
        """The call being answered, for the draft and test-call requests."""
        if call is None:
            return "(none: the function is written before its first call)"
        return self._call_text(*call) or "(arguments that can't be written as literals)"

    def _call_text(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
        """This call as source text, like minutes('1h 30m'), if its arguments can be written as literals."""
        inputs = self.spec.inputs(args, kwargs)
        params = self.spec.sig.parameters
        shown, by_name = [], False
        for name, value in inputs.items():
            text = repr(value)
            try:
                if ast.literal_eval(text) != value:
                    return None
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                return None
            kind = params[name].kind
            if kind is inspect.Parameter.VAR_POSITIONAL or kind is inspect.Parameter.VAR_KEYWORD:
                return None
            by_name = by_name or kind is inspect.Parameter.KEYWORD_ONLY
            shown.append(f"{name}={text}" if by_name else text)
        return f"{self.short}({', '.join(shown)})"


# --- checking a draft ----------------------------------------------------------------------


def _lint(draft: Draft, text: str) -> list[str]:
    """What's wrong with a draft's code, before it runs. These catch mistakes, not attacks: the
    real boundary is that writing happens only in development, in your own files, as a diff to review."""
    try:
        body = ast.parse(draft.body)
    except SyntaxError as exc:
        return [f"the body doesn't parse: {exc.msg} (line {exc.lineno})"]
    if not body.body or all(isinstance(s, (ast.Pass, ast.Expr)) for s in body.body):
        return ["the body does nothing"]
    module = ast.parse(text)
    allowed = {_top_module(n) for n in source.top_level_imports(module)} - {None}
    problems: list[str] = []
    for statement in draft.imports:
        try:
            parsed = ast.parse(statement).body
        except SyntaxError:
            problems.append(f"{statement!r} isn't an import statement")
            continue
        if len(parsed) != 1 or not isinstance(parsed[0], (ast.Import, ast.ImportFrom)):
            problems.append(f"{statement!r} isn't one import statement")
            continue
        for name in _modules(parsed[0]):
            top = name.split(".")[0]
            if top in BANNED_MODULES or (
                top not in sys.stdlib_module_names and name not in allowed and top not in allowed
            ):
                problems.append(f"the body may not import {name}: only the standard library and the file's imports")
    for node in ast.walk(body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append("put imports in imports, not in the body")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("don't use global or nonlocal")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            problems.append(f"don't call {node.func.id}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = (
                node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
            )
            if mode is not None and not (
                isinstance(mode, ast.Constant) and isinstance(mode.value, str) and not set(mode.value) & set("wax+")
            ):
                problems.append("don't open files for writing")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
            if node.attr in BANNED_OS or node.attr.startswith(("exec", "spawn")):
                problems.append(f"don't call os.{node.attr}")
    return list(dict.fromkeys(problems))


def _lookup_table(body: str, cases: list[Case]) -> list[str]:
    """Problems if the body has the test calls' inputs written into it: it learned the cases, not the rule."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []
    literals = {c.value for c in ast.walk(tree) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    found = []
    for case in cases:
        for c in ast.walk(ast.parse(case.text, mode="eval")):
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and len(c.value) >= 4 and c.value in literals:
                found.append(f"the body contains the test input {c.value!r}: write the general rule instead")
    return list(dict.fromkeys(found))


def _signature_types(func: Callable[..., Any]) -> list[type]:
    """The classes in a function's type hints, and in the fields of dataclasses among them."""
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        return []
    found: list[type] = []

    def visit(tp: Any) -> None:
        for arg in typing.get_args(tp):
            visit(arg)
        if not isinstance(tp, type) or tp in found:
            return
        found.append(tp)
        if dataclasses.is_dataclass(tp):
            with contextlib.suppress(Exception):
                for hint in typing.get_type_hints(tp).values():
                    visit(hint)

    for hint in hints.values():
        visit(hint)
    return found


def _project_types(func: Callable[..., Any], root: str, path: str) -> str:
    """The source of the classes in the function's signature that are defined elsewhere in the project,
    such as a dataclass input from models.py, so the draft request knows their fields."""
    parts: list[str] = []
    for tp in _signature_types(func):
        try:
            where = os.path.realpath(inspect.getsourcefile(tp) or "")
        except TypeError:  # a builtin
            continue
        if where != path and where.startswith(root + os.sep) and not _installed(where):
            with contextlib.suppress(OSError, TypeError):
                parts.append(f"# {_relative(where, root)}\n{inspect.getsource(tp)}")
    return "\n".join(parts)


def _modules(statement: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(statement, ast.Import):
        return [a.name for a in statement.names]
    return ["." * statement.level + (statement.module or "")]


def _top_module(statement: ast.Import | ast.ImportFrom) -> str | None:
    names = _modules(statement)
    return names[0].split(".")[0] if names and not names[0].startswith(".") else (names[0] if names else None)


_LITERAL_NODES = (
    ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Load, ast.UnaryOp, ast.USub, ast.UAdd, ast.keyword,
)  # fmt: skip


def _case_arguments(case: str, name: str, classes: dict[str, type]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The arguments of a test call like minutes('2 hours'). Each is a literal, or a call of one of
    `classes` (a dataclass input) with literal arguments; nothing else runs."""
    try:
        tree = ast.parse(case.strip(), mode="eval")
    except SyntaxError:
        raise ValueError(f"the case {case!r} isn't a Python call") from None
    call = tree.body
    target = call.func if isinstance(call, ast.Call) else None
    called = target.attr if isinstance(target, ast.Attribute) else target.id if isinstance(target, ast.Name) else None
    if not isinstance(call, ast.Call) or called != name:
        raise ValueError(f"the case {case!r} isn't a call of {name}(...)")
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        raise ValueError(f"the case {case!r} uses * or **; write the arguments out")
    for part in [*call.args, *(k.value for k in call.keywords)]:
        for node in ast.walk(part):
            if isinstance(node, ast.Call):
                if not (isinstance(node.func, ast.Name) and node.func.id in classes):
                    raise ValueError(
                        f"the case {case!r} has an argument that isn't a literal or one of the module's classes"
                    )
            elif isinstance(node, ast.Name):
                if node.id not in classes:
                    raise ValueError(f"the case {case!r} has an argument that isn't a literal")
            elif not isinstance(node, _LITERAL_NODES):
                raise ValueError(f"the case {case!r} has an argument that isn't a literal")

    def value(expr: ast.expr) -> Any:
        try:
            return ast.literal_eval(expr)
        except ValueError:  # a class of the module, called with literals: evaluated with nothing else in reach
            return eval(compile(ast.Expression(expr), "<thunc case>", "eval"), {"__builtins__": {}, **classes})

    try:
        return tuple(value(a) for a in call.args), {k.arg: value(k.value) for k in call.keywords if k.arg}
    except Exception as exc:
        raise ValueError(f"the case {case!r} can't be evaluated: {type(exc).__name__}: {shorten(str(exc))}") from None


def _normal_call(text: str) -> str:
    """A test call as doctest shows it: parsed and unparsed, so the spacing is standard."""
    try:
        return ast.unparse(ast.parse(text.strip(), mode="eval"))
    except SyntaxError:
        return text.strip()


class _Failed(str):
    """What went wrong when a draft ran on a test call: an error or a timeout."""


def _run_case(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], spec: Any) -> Any:
    """The draft's result for one call, or _Failed saying what went wrong (an error, a timeout).
    The call runs in a thread so a hang can be abandoned; it's left running if it hangs."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            result = func(*args, **kwargs)
            box["value"] = asyncio.run(result) if spec.is_async and inspect.iscoroutine(result) else result
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=target, name="thunc-check", daemon=True)
    thread.start()
    thread.join(CASE_SECONDS)
    if thread.is_alive():
        return _Failed(f"took longer than {CASE_SECONDS:g}s (stopped waiting)")
    if "error" in box:
        exc = box["error"]
        return _Failed(f"raised {type(exc).__name__}: {shorten(str(exc), 200)}")
    return box["value"]


def _as_type(value: Any, returns: Any) -> Any:
    """The draft's result checked against the return type, as an answer from the model is."""
    if returns is int and isinstance(value, float):  # validate() takes 3.0 from a model; code should say 3
        raise ValueError("it's a float, not an int")
    try:
        plain = json.loads(json.dumps(_plain(value), default=_plain))
    except (TypeError, ValueError):
        raise ValueError(f"it isn't {describe(returns)}") from None
    return validate(plain, returns)


def _same(a: Any, b: Any) -> bool:
    """Whether two results agree: exactly, except for floats, which may differ by rounding."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) or isinstance(b, float):
        return (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        )
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b):
        return type(a) is type(b) and _same(_plain(a), _plain(b))
    return bool(a == b)


def _shown(value: Any) -> str:
    return short_repr(value)


# --- files and folders ---------------------------------------------------------------------


def _source_file(func: Callable[..., Any]) -> str | None:
    try:
        path = inspect.getsourcefile(func)
    except TypeError:
        return None
    return os.path.realpath(path) if path else None


def _digest(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _installed(path: str) -> bool:
    parts = set(path.split(os.sep))
    if parts & {"site-packages", "dist-packages"}:
        return True
    prefixes = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    root = _project_root(path)
    for prefix in prefixes:
        inside = path.startswith(prefix + os.sep)
        if inside and not (root and prefix.startswith(root + os.sep)):  # a venv inside the project is fine
            return True
    return False


def _project_root(path: str) -> str | None:
    """The git repository the file is in, else the current directory if the file is under it."""
    folder = os.path.dirname(path)
    while True:
        if os.path.exists(os.path.join(folder, ".git")):
            return folder
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    cwd = os.path.realpath(os.getcwd())
    return cwd if path.startswith(cwd + os.sep) else None


def _relative(path: str, root: str | None = None) -> str:
    try:
        return os.path.relpath(path, root or os.getcwd())
    except ValueError:  # another drive, on Windows
        return path


def _folder(root: str) -> str:
    folder = os.path.join(root, FOLDER)
    os.makedirs(folder, exist_ok=True)
    ignore = os.path.join(folder, ".gitignore")
    if not os.path.exists(ignore):
        with open(ignore, "w", encoding="utf-8") as f:
            f.write("# thunc write's notes about functions it didn't write; not for git\n*\n")
    return folder


@contextlib.contextmanager
def _file_lock(root: str, key: str) -> Iterator[None]:
    """Hold this function's lock across processes, so one of them writes it."""
    path = os.path.join(_folder(root), f"{key}.lock")
    with _thread_lock(path):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            while not _try_lock(fd):
                threading.Event().wait(0.1)
            try:
                yield
            finally:
                _unlock(fd)
        finally:
            os.close(fd)


def _read_memo(root: str, key: str) -> str | None:
    try:
        with open(os.path.join(root, FOLDER, f"{key}.json"), encoding="utf-8") as f:
            return str(json.load(f)["reason"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_memo(root: str, key: str, reason: str) -> None:
    path = os.path.join(_folder(root), f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"reason": reason, "time": dt.datetime.now().isoformat(timespec="seconds")}, f)


def _say(message: str) -> None:
    print(f"thunc: {message}", file=sys.stderr, flush=True)
