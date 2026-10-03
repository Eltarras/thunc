"""Return types: describe one to the model, and turn the model's text back into a checked value.

Supported: str, bool, int, float, None, Any, Literal[...], list[T], dict[str, T], unions
(T | None) and dataclasses.
"""

from __future__ import annotations

import dataclasses
import json
import re
import types
import typing
from typing import Any, Literal, Union

from .errors import ThuncError


def describe(tp: Any, nested: bool = False) -> str:
    """How the expected value is described in the prompt ("Return <description>.")."""
    if tp is str:
        return "a JSON string" if nested else "plain text"
    simple = {
        bool: "a JSON boolean (true or false)",
        int: "a JSON integer",
        float: "a JSON number",
        type(None): "null",
        Any: "any JSON value",
    }
    if tp in simple:
        return simple[tp]
    if _is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        fields = [
            f'"{f.name}": {describe(hints[f.name], True)}{"" if _required(f) else " (optional)"}'
            for f in dataclasses.fields(tp)
        ]
        return "a JSON object with these fields: {" + ", ".join(fields) + "}"
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        return "exactly one of these JSON values: " + ", ".join(json.dumps(a) for a in args)
    if origin in (Union, types.UnionType):
        return " or ".join(describe(a, True) for a in args)
    if tp is list or origin is list:
        return "a JSON array" + (f" whose items are each {describe(args[0], True)}" if args else "")
    if tp is dict or origin is dict:
        return "a JSON object" + (f" whose values are each {describe(args[1], True)}" if args else "")
    raise ThuncError(f"Unsupported return type: {tp!r}")


def parse(text: str, tp: Any) -> Any:
    """Model text -> value of type `tp`. Raises ValueError (with a reason the model can act on)."""
    if tp is str:
        return text.strip()
    cleaned = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        # Tolerate the usual near-misses: yes/no for booleans, an unquoted label for Literals.
        word = cleaned.lower().strip(".")
        if tp is bool and word in {"true", "yes", "false", "no"}:
            return word in {"true", "yes"}
        if typing.get_origin(tp) is Literal and cleaned.strip("'\"") in typing.get_args(tp):
            return cleaned.strip("'\"")
        raise ValueError(f"not valid JSON: {cleaned[:200]!r}") from None
    return validate(value, tp)


def validate(value: Any, tp: Any) -> Any:
    """Check a decoded JSON value against `tp` (coercing 3.0 -> 3, dicts -> dataclasses)."""
    if tp is Any:
        return value
    if tp is type(None):
        return _expect(value is None, value, "null")
    if tp is bool:
        return _expect(isinstance(value, bool), value, "true or false")
    if tp is int:
        whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        ok = whole and not isinstance(value, bool)
        return int(_expect(ok, value, "an integer"))
    if tp is float:
        return float(_expect(isinstance(value, (int, float)) and not isinstance(value, bool), value, "a number"))
    if tp is str:
        return _expect(isinstance(value, str), value, "a string")
    if _is_dataclass(tp):
        _expect(isinstance(value, dict), value, "an object")
        hints = typing.get_type_hints(tp)
        kwargs = {}
        for f in dataclasses.fields(tp):
            if f.name in value:
                try:
                    kwargs[f.name] = validate(value[f.name], hints[f.name])
                except ValueError as exc:
                    raise ValueError(f"field {f.name!r}: {exc}") from None
            elif _required(f):
                raise ValueError(f"missing required field {f.name!r}")
        return tp(**kwargs)

    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        return _expect(value in args, value, f"one of {list(args)}")
    if origin in (Union, types.UnionType):
        errors = []
        for option in args:
            try:
                return validate(value, option)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError("; ".join(errors))
    if tp is list or origin is list:
        _expect(isinstance(value, list), value, "an array")
        return [validate(v, args[0]) for v in value] if args else value
    if tp is dict or origin is dict:
        _expect(isinstance(value, dict), value, "an object")
        return {k: validate(v, args[1]) for k, v in value.items()} if args else value
    raise ThuncError(f"Unsupported return type: {tp!r}")


def _expect(ok: bool, value: Any, wanted: str) -> Any:
    if not ok:
        raise ValueError(f"expected {wanted}, got {value!r}")
    return value


def _is_dataclass(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _required(f: dataclasses.Field[Any]) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
