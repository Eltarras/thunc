"""The public API, frozen (from 0.6 for agents, and everything else in thunc.__all__): every public
name's signature, every public method and property of its classes, and the fields of its dataclasses,
compared with tests/api.txt.

A difference fails the test, so an API change is a visible line in api.txt in review. Adding is
fine: update the file with THUNC_UPDATE_API=1 pytest tests/test_api.py. Removing or changing a name,
a parameter or a default goes through a deprecation first (CONTRIBUTING.md, "Deprecations").
"""

import dataclasses
import difflib
import inspect
import os
from pathlib import Path

import thunc
from thunc import runs, testing

SNAPSHOT = Path(__file__).parent / "api.txt"
MODULES = {"thunc": thunc, "thunc.testing": testing, "thunc.runs": runs}
PUBLIC = {
    "thunc": thunc.__all__,
    "thunc.testing": ["FakeModel", "Request", "act", "as_text", "fake_model", "finish", "isolated"],
    "thunc.runs": ["Command", "Denial"],
}


def describe(name, obj):
    """Lines for one public name."""
    if inspect.isclass(obj):
        lines = [f"class {name}{_signature(obj)}"]
        if dataclasses.is_dataclass(obj):
            lines += [f"    field {f.name}: {_annotation(f.type)}" for f in dataclasses.fields(obj)]
        for member, value in sorted(vars(obj).items()):
            if member.startswith("_"):
                continue
            if isinstance(value, property):
                lines.append(f"    property {member}")
            elif callable(value) or isinstance(value, (staticmethod, classmethod)):
                lines.append(f"    def {member}{_signature(getattr(obj, member))}")
        return lines
    if callable(obj):
        return [f"def {name}{_signature(obj)}"]
    return [f"{name}: {type(obj).__name__}"]


def _signature(obj):
    """A signature, written the same way on every Python version: thunc's annotations are strings."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    parts, marked = [], False
    params = list(signature.parameters.values())
    for i, p in enumerate(params):
        if p.kind is p.KEYWORD_ONLY and not marked:
            parts.append("*")
            marked = True
        if p.kind is p.VAR_POSITIONAL:
            marked = True
        text = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "") + p.name
        if p.annotation is not p.empty:
            text += f": {_annotation(p.annotation)}"
        if p.default is not p.empty:
            text += f" = {_default(p.default)}"
        parts.append(text)
        following = params[i + 1] if i + 1 < len(params) else None
        if p.kind is p.POSITIONAL_ONLY and (following is None or following.kind is not p.POSITIONAL_ONLY):
            parts.append("/")
    result = f"({', '.join(parts)})"
    if signature.return_annotation is not signature.empty:
        result += f" -> {_annotation(signature.return_annotation)}"
    return result


def _default(value):
    if inspect.isclass(value):
        return value.__name__
    return repr(value)


def _annotation(tp):
    return tp.replace("'", '"') if isinstance(tp, str) else getattr(tp, "__name__", repr(tp))


def snapshot():
    lines = []
    for module, names in PUBLIC.items():
        lines.append(f"# {module}")
        for name in sorted(names, key=str.lower):
            lines += describe(f"{module}.{name}", getattr(MODULES[module], name))
        lines.append("")
    return "\n".join(lines)


def test_the_public_api_is_the_one_in_api_txt():
    current = snapshot()
    if os.environ.get("THUNC_UPDATE_API"):
        SNAPSHOT.write_text(current, encoding="utf-8")
    recorded = SNAPSHOT.read_text(encoding="utf-8")
    if current != recorded:
        diff = "\n".join(
            difflib.unified_diff(recorded.splitlines(), current.splitlines(), "api.txt", "now", lineterm="")
        )
        raise AssertionError(
            "The public API changed. If that's intended (an addition, or a removal that went through a "
            f"deprecation), update tests/api.txt with THUNC_UPDATE_API=1.\n{diff}"
        )
