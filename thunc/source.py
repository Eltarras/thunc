"""Editing one function in its source file, for thunc write (see writing.py).

splice() returns the file's text with a written body in place of the empty one: the
@thunc.function(write=True) decorator removed, the imports the body needs added, the checked cases
added to the docstring as doctest examples. Every other line stays as it was, `import thunc` included:
other code may still use it, in the file or through it. load() compiles the function from that text, so the running
program can call it without importing its module again.
"""

from __future__ import annotations
import __future__

import ast
import os
import re
import tempfile
import tokenize
from collections.abc import Callable, Sequence
from typing import Any

Function = ast.FunctionDef | ast.AsyncFunctionDef


class SourceError(Exception):
    """The file can't be edited as asked; the message says why, for a warning."""


def read(path: str) -> tuple[str, str]:
    """The file's text, exactly (line endings kept), and its encoding."""
    with open(path, "rb") as f:
        encoding, _ = tokenize.detect_encoding(f.readline)
    with open(path, encoding=encoding, newline="") as f:
        return f.read(), encoding


def write(path: str, text: str, encoding: str) -> None:
    """Replace the file in one step, keeping its permissions: a crash leaves the old file or the new one."""
    mode = os.stat(path).st_mode
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".thunc-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def find(tree: ast.Module, qualname: str, line: int | None = None) -> Function:
    """The def for `qualname` ("name" or "Class.name"). `line` (its first line, decorators included,
    as in co_firstlineno) picks between definitions of the same name."""
    *classes, name = qualname.split(".")
    scope: list[ast.AST] = list(_statements(tree.body))
    for cls in classes:
        found = [n for n in scope if isinstance(n, ast.ClassDef) and n.name == cls]
        if len(found) != 1:
            raise SourceError(f"can't find class {cls} in the file")
        scope = list(_statements(found[0].body))
    matches = [n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(matches) > 1 and line is not None:
        matches = [n for n in matches if first_line(n) == line]
    if len(matches) != 1:
        raise SourceError(f"can't find a single definition of {qualname} in the file")
    return matches[0]


def first_line(node: Function) -> int:
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def function_source(text: str, node: Function) -> str:
    """The function's lines, decorators included."""
    return "".join(text.splitlines(keepends=True)[first_line(node) - 1 : node.end_lineno])


def splice(
    text: str,
    qualname: str,
    line: int | None,
    body: str,
    imports: Sequence[str] = (),
    examples: Sequence[tuple[str, str]] = (),
    note: str = "",
) -> tuple[str, int]:
    """The file's text with the function written, and the function's first line in it.

    body:     the statements of the body, not indented (the def line and docstring stay as they are)
    imports:  import statements the body needs; ones the file already has are skipped
    examples: (call, result) pairs, as source text, added to the docstring as doctest examples
    note:     a comment put above the body
    """
    tree = ast.parse(text)
    node = find(tree, qualname, line)
    lines = text.splitlines(keepends=True)
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    ours = _our_decorator(node)
    at = _cut(lines[ours.lineno - 1], ours.col_offset)
    after = _rest_of(lines[(ours.end_lineno or ours.lineno) - 1], ours.end_col_offset or 0).strip()
    if not re.fullmatch(r"\s*@\s*", at) or (after and not after.startswith("#")):
        raise SourceError("@thunc.function(write=True) must be on lines of its own")
    doc, rest = _docstring_and_rest(node, lines)
    indent = _cut(lines[doc.lineno - 1], doc.col_offset)

    code = body.strip("\n")
    if not code.strip():
        raise SourceError("the body is empty")
    written = [f"{indent}# {note}{newline}"] if note else []
    written += [f"{indent}{row}{newline}" if row.strip() else newline for row in code.splitlines()]

    edits: list[tuple[int, int, list[str]]] = []  # (first line, last line, replacement), 1-based, inclusive
    doc_end = doc.end_lineno or doc.lineno
    if rest:
        edits.append((rest[0].lineno, rest[-1].end_lineno or rest[-1].lineno, written))
    else:
        edits.append((doc_end + 1, doc_end, written))  # an insertion after the docstring
    if examples:
        edits.append((doc.lineno, doc_end, _with_examples(lines, doc, indent, examples, newline)))
    edits.append((ours.lineno, ours.end_lineno or ours.lineno, []))
    edits += _import_edits(tree, lines, imports, newline)

    start = first_line(node)
    for first, last, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[first - 1 : last] = replacement
        if last < start:  # an edit above the function moves it
            start += len(replacement) - (last - first + 1)
    result = "".join(lines)
    _check(result, qualname, start, code)
    return result, start


def load(
    text: str,
    path: str,
    qualname: str,
    line: int,
    namespace: dict[str, Any],
    owner: type | None = None,
) -> Callable[..., Any]:
    """Compile the function from `text` (without its decorators) against `namespace`, the module's
    globals, and return it. `owner` is the real class of a method, for super() and __class__."""
    tree = ast.parse(text)
    node = find(tree, qualname, line)
    lines = text.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno]  # from the def line: no decorators
    classes = qualname.split(".")[:-1]
    if classes:  # compiled inside a class of the same name, so self.__private names are mangled as in the file
        header = f"class {classes[-1]}:\n"
    elif lines and lines[0][:1] in (" ", "\t"):
        header = "if 1:\n"
    else:
        header = ""
    padding = "\n" * (node.lineno - 1 - (1 if header else 0))  # so tracebacks show the file's line numbers
    flags = _future_flags(tree)
    code = compile(padding + header + "".join(lines), path, "exec", flags=flags, dont_inherit=True)
    scope: dict[str, Any] = {}
    exec(code, namespace, scope)
    func: Callable[..., Any] = vars(scope[classes[-1]])[node.name] if classes else scope[node.name]
    func.__qualname__ = qualname
    if owner is not None and "__class__" in func.__code__.co_freevars and func.__closure__:
        func.__closure__[func.__code__.co_freevars.index("__class__")].cell_contents = owner
    return func


def run_imports(imports: Sequence[str], namespace: dict[str, Any]) -> None:
    """Carry out import statements in a module's namespace."""
    for statement in imports:
        exec(compile(statement, "<thunc imports>", "exec", dont_inherit=True), namespace)


def import_key(statement: ast.stmt) -> str:
    return ast.unparse(statement)


def top_level_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    return [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


# --- helpers -----------------------------------------------------------------------------------


def _statements(body: list[ast.stmt]) -> list[ast.AST]:
    """Statements at this level, including those inside if/try/with blocks, but not inside defs or classes."""
    found: list[ast.AST] = []
    for stmt in body:
        found.append(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            found += _statements(getattr(stmt, field, []) or [])
        for handler in getattr(stmt, "handlers", []) or []:
            found += _statements(handler.body)
    return found


def _our_decorator(node: Function) -> ast.expr:
    """@thunc.function(write=True), which must be the decorator nearest the def line."""
    ours = [
        d
        for d in node.decorator_list
        if isinstance(d, ast.Call)
        and any(k.arg == "write" and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords)
    ]
    if len(ours) != 1:
        raise SourceError("can't find @thunc.function(write=True) on the function (write=True must be written as is)")
    if node.decorator_list[-1] is not ours[0]:
        raise SourceError("@thunc.function(write=True) must be the decorator nearest the def line")
    return ours[0]


def _docstring_and_rest(node: Function, lines: list[str]) -> tuple[ast.Expr, list[ast.stmt]]:
    """The docstring statement and the empty body after it, each checked to sit on lines of its own."""
    first = node.body[0]
    if not (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    ):
        raise SourceError("the function needs a docstring")
    rest = node.body[1:]
    for stmt in [first, *rest]:
        before = _cut(lines[stmt.lineno - 1], stmt.col_offset)
        after = _rest_of(lines[(stmt.end_lineno or stmt.lineno) - 1], stmt.end_col_offset or 0)
        if before.strip() or (after.strip() and not after.strip().startswith("#")):
            raise SourceError("the docstring and the `...` must each be on lines of their own")
    return first, rest


def _with_examples(
    lines: list[str], doc: ast.Expr, indent: str, examples: Sequence[tuple[str, str]], newline: str
) -> list[str]:
    """The docstring's lines with doctest examples added before its closing quotes."""
    start, end = doc.lineno, doc.end_lineno or doc.lineno
    prefix = _cut(lines[start - 1], doc.col_offset)
    suffix = _rest_of(lines[end - 1], doc.end_col_offset or 0)
    segment = "".join(lines[start - 1 : end]).rstrip("\r\n")[len(prefix) :]
    segment = segment[: len(segment) - len(suffix)]
    quote = segment[-3:]
    opening = re.match(r"[rRuU]?", segment)
    raw = bool(opening and opening.group().lower() == "r")
    if quote not in ('"""', "'''"):
        return lines[start - 1 : end]  # not triple-quoted: leave it as it is
    rows = []
    for call, result in examples:
        if not raw:
            call, result = call.replace("\\", "\\\\"), result.replace("\\", "\\\\")
        if quote in call or quote in result or "\n" in call or "\n" in result:
            continue
        rows += [f"{indent}>>> {call}{newline}", f"{indent}{result}{newline}"]
    if not rows:
        return lines[start - 1 : end]
    content = segment[:-3].rstrip()
    rebuilt = f"{prefix}{content}{newline}{newline}" + "".join(rows) + f"{indent}{quote}{suffix}{newline}"
    return rebuilt.splitlines(keepends=True)


def _import_edits(
    tree: ast.Module, lines: list[str], imports: Sequence[str], newline: str
) -> list[tuple[int, int, list[str]]]:
    """The edit that adds the body's imports the file doesn't have yet, if there are any."""
    edits: list[tuple[int, int, list[str]]] = []
    existing = top_level_imports(tree)
    have = {import_key(n) for n in existing}
    new: list[str] = []
    for statement in imports:
        for parsed in ast.parse(statement).body:
            key = import_key(parsed)
            if key not in have:
                have.add(key)
                new.append(key + newline)
    if new:
        after = _import_line(tree, lines, existing)
        edits.append((after + 1, after, new))
    return edits


def _import_line(tree: ast.Module, lines: list[str], existing: list[ast.Import | ast.ImportFrom]) -> int:
    """The line after which new imports go: the last import in the file's opening block of imports,
    else after the module docstring, else after a shebang and encoding line."""
    last = 0
    for stmt in tree.body:
        is_doc = (
            stmt is tree.body[0]
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
        if isinstance(stmt, (ast.Import, ast.ImportFrom)) or is_doc:
            last = stmt.end_lineno or stmt.lineno
        else:
            break
    if last:
        return last
    head = 0
    while head < min(2, len(lines)) and lines[head].startswith("#"):
        head += 1
    return head


def _check(text: str, qualname: str, line: int, code: str) -> None:
    """The edited file parses, and the function in it has exactly the body asked for."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise SourceError(f"the edited file doesn't parse: {exc.msg} (line {exc.lineno})") from None
    node = find(tree, qualname, line)
    if ast.dump(ast.Module(body=node.body[1:], type_ignores=[])) != ast.dump(ast.parse(code)):
        raise SourceError("the body changed when it was indented (a multi-line string?)")


def _future_flags(tree: ast.Module) -> int:
    flags = 0
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            for alias in stmt.names:
                feature = getattr(__future__, alias.name, None)
                flags |= getattr(feature, "compiler_flag", 0)
    return flags


def _cut(line: str, offset: int) -> str:
    """The line up to a column offset from ast, which counts UTF-8 bytes."""
    return line.encode("utf-8")[:offset].decode("utf-8", "replace")


def _rest_of(line: str, offset: int) -> str:
    """The line from a column offset from ast to its end, without the line break."""
    return line.encode("utf-8")[offset:].decode("utf-8", "replace").rstrip("\r\n")
