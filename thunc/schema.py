"""Return types: describe one to the model, and turn the model's text back into a checked value.

Supported: str, bool, int, float, None, Any, Literal[...], list[T], dict[str, T], unions
(T | None) and dataclasses.
"""

from __future__ import annotations

import dataclasses
import json
import math
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
    cleaned = _trim(_THINK.sub("", _trim(text), count=1))
    if tp is str:
        if not cleaned:
            raise ValueError("the reply was empty")
        return cleaned
    cleaned = _unfence(cleaned)
    try:
        value = json.loads(cleaned, object_pairs_hook=_no_duplicates, parse_constant=_no_constant, parse_float=_finite)
    except RecursionError:
        raise ValueError("the JSON is nested too deeply") from None
    except json.JSONDecodeError:
        word = _bare_word(cleaned, tp)
        if word is not _NO_WORD:
            return word
        raise ValueError(f"not valid JSON: {cleaned[:200]!r}") from None
    try:
        return validate(value, tp)
    except ValueError:
        # A label that happens to be valid JSON, like 2 for Literal["1", "2"].
        word = _bare_word(cleaned, tp)
        if word is not _NO_WORD:
            return word
        # An answer wrapped in a one-key object, like {"rating": 4} for an int: unwrap it, once.
        if isinstance(value, dict) and len(value) == 1:
            try:
                return validate(next(iter(value.values())), tp)
            except ValueError:
                pass
        raise


# A leading reasoning block, as some local models emit: <think>...</think>
_THINK = re.compile(r"\A<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# Whitespace (all of it is below U+3001), a byte-order mark and zero-width characters, which
# str.strip() keeps. Stripped with str.strip(chars): a regex here is quadratic on long blank runs.
_EDGES = "".join(c for c in map(chr, range(0x3001)) if c.isspace()) + "\ufeff\u200b\u200c\u200d\u2060"
_NO_WORD = object()


def _trim(text: str) -> str:
    return text.strip(_EDGES)


def _bare_word(text: str, tp: Any) -> Any:
    """The usual near-misses that aren't JSON of the right kind: yes/no for a bool, an unquoted
    (or single-quoted) label for a string Literal. Also inside an Optional. _NO_WORD if neither."""
    options = typing.get_args(tp) if typing.get_origin(tp) in (Union, types.UnionType) else (tp,)
    label = text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"" else text
    literals = [typing.get_args(o) for o in options if typing.get_origin(o) is Literal]
    if any(label in labels for labels in literals):  # a str only equals a str option
        return label  # an exact label first: "no" for Literal["no", "partial"] | bool is the label
    word = text.lower().strip(".")
    if bool in options and str not in options and word in {"true", "yes", "false", "no"}:
        return word in {"true", "yes"}
    return _NO_WORD


def _unfence(text: str) -> str:
    """The inside of a Markdown code fence: the whole reply, or a fence that ends the reply after a
    line of prose ("Here you go:"). The language tag is ignored, whatever it is. A reply with more
    fences, or text after the fence (a second answer?), is left as it is (and fails). Linear time:
    it only looks for fences at the start of a line, so "```" inside a JSON string is fine."""
    if not text.endswith("```") or len(text) < 6:
        return text
    if "\n" not in text:  # ```4``` or ```json [1, 2]```
        inline = re.fullmatch(r"```(?:json\b)?(.*)```", text, re.DOTALL | re.IGNORECASE)
        return inline.group(1).strip() if inline and text.startswith("```") else text
    close = len(text) - 3
    opens = [m.start() for m in re.finditer(r"^```", text, re.MULTILINE) if m.start() != close]
    if len(opens) != 1:
        return text
    body = text.find("\n", opens[0]) + 1  # the line after the opening fence and its language tag
    return text[body:close].strip() if 0 < body <= close else text


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A repeated key in the model's object is ambiguous, not "the last one wins"."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"the key {key!r} appears more than once")
        result[key] = value
    return result


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a valid number")


def _finite(digits: str) -> float:
    number = float(digits)
    if not math.isfinite(number):
        raise ValueError(f"{digits[:50]} is out of range")
    return number


def validate(value: Any, tp: Any) -> Any:
    """Check a decoded JSON value against `tp` (coercing 3.0 -> 3, dicts -> dataclasses)."""
    if tp is Any:
        return value
    if tp is type(None):
        return _expect(value is None, value, "null")
    if tp is bool:
        return _expect(isinstance(value, bool), value, "true or false")
    if tp is int:
        # 3.0 is 3, but only while a float holds the integer exactly (12345678901234567.0 doesn't).
        whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) < 2**53)
        ok = whole and not isinstance(value, bool)
        return int(_expect(ok, value, "an integer"))
    if tp is float:
        try:
            number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan
        except OverflowError:  # an integer too big for a float, like 1 followed by 400 zeros
            number = math.inf
        _expect(math.isfinite(number), value, "a finite number")
        return number
    if tp is str:
        return _expect(isinstance(value, str), value, "a string")
    if _is_dataclass(tp):
        _expect(isinstance(value, dict), value, "an object")
        hints = typing.get_type_hints(tp)
        kwargs = {}
        for f in dataclasses.fields(tp):
            if not f.init:
                continue  # computed by the class itself, never set from the answer
            if f.name in value:
                try:
                    kwargs[f.name] = validate(value[f.name], hints[f.name])
                except ValueError as exc:
                    raise ValueError(f"field {f.name!r}: {exc}") from None
            elif _required(f):
                raise ValueError(f"missing required field {f.name!r}")
        names = [f.name for f in dataclasses.fields(tp) if f.init]
        if value and names and not kwargs:
            # Not one known field, like {"filters": {...}} for Filters: not an empty Filters().
            raise ValueError(f"expected an object with the fields {names}, got {short_repr(value)}")
        try:
            return tp(**kwargs)
        except Exception as exc:  # a check in __post_init__, for example
            raise ValueError(f"{tp.__name__}(...) failed: {type(exc).__name__}: {shorten(str(exc))}") from exc

    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        # Compared by kind too, or true would match Literal[1] (True == 1 in Python).
        match = [a for a in args if _kind(a) == _kind(value) and a == value]
        _expect(bool(match), value, f"one of {list(args)}")
        return match[0]
    if origin in (Union, types.UnionType):
        errors = []
        # The option of the value's own type first, so 3 for `float | int` stays the int 3.
        for option in sorted(args, key=lambda option: option is not type(value)):
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
        raise ValueError(f"expected {wanted}, got {short_repr(value)}")
    return value


def short_repr(value: Any, limit: int = 200) -> str:
    """repr() cut to `limit` characters, so a huge wrong answer doesn't flood the retry prompt."""
    return shorten(repr(value), limit)


def shorten(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _kind(value: Any) -> str:
    """bool, number, string, ...: the JSON kind of a value, for comparing Literal options."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _is_dataclass(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _required(f: dataclasses.Field[Any]) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
