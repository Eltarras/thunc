"""Return types: describe one to the model, and turn the model's text back into a checked value.

Supported: str, bool, int, float, None, Any, Literal[...], list[T], dict[str, T], unions
(T | None) and dataclasses.
"""

from __future__ import annotations

import dataclasses
import decimal
import functools
import json
import math
import operator
import re
import sys
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
        fields = [
            f'"{name}": {describe(hint, True)}{"" if required else " (optional)"}'
            for name, hint, required in _init_fields(tp)
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
    try:
        return _parse(text, tp)
    except RecursionError:  # JSON just under json's depth limit can still be too deep to check
        raise ValueError("the JSON is nested too deeply") from None


def _parse(text: str, tp: Any) -> Any:
    cleaned = _trim(_THINK.sub("", _trim(text), count=1))
    if tp is str:
        if not cleaned:
            raise ValueError("the reply was empty")
        return cleaned
    cleaned, before = _unfence(cleaned)
    if before and _is_answer(before, tp):
        raise ValueError("two answers: one before the code fence and one inside it")
    try:
        value = json.loads(cleaned, object_pairs_hook=_no_duplicates, parse_constant=_no_constant, parse_float=_finite)
    except ValueError as problem:  # not JSON, or JSON thunc refuses (NaN, a duplicate key, ...)
        word = _bare_word(cleaned, tp)  # like NaN for Literal["NaN", "ok"]
        if word is not _NO_WORD:
            return word
        if isinstance(problem, json.JSONDecodeError):
            raise ValueError(f"not valid JSON: {cleaned[:200]!r}") from None
        raise
    try:
        return validate(value, tp)
    except ValueError:
        # A label that happens to be valid JSON, like 2 for Literal["1", "2"].
        word = _bare_word(cleaned, tp)
        if word is not _NO_WORD:
            return word
        # An answer wrapped in a one-key object, like {"rating": 4} for an int: unwrap it, once.
        if isinstance(value, dict) and len(value) == 1 and _may_unwrap(value, tp):
            try:
                return validate(next(iter(value.values())), tp)
            except ValueError:
                pass
        raise


# A leading reasoning block, as some local models emit: <think>...</think>
_THINK = re.compile(r"\A<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
# Whitespace (all of it is below U+3001), a byte-order mark, zero-width and direction marks, which
# str.strip() keeps. Stripped with str.strip(chars): a regex here is quadratic on long blank runs.
_INVISIBLE = "\ufeff\u200b\u200c\u200d\u2060"  # a byte-order mark, zero-width characters
_DIRECTION = "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"  # marks, embeddings, isolates
_EDGES = "".join(c for c in map(chr, range(0x3001)) if c.isspace()) + _INVISIBLE + _DIRECTION
_NO_WORD = object()


def _trim(text: str) -> str:
    return text.strip(_EDGES)


def _options(tp: Any) -> tuple[Any, ...]:
    return typing.get_args(tp) if typing.get_origin(tp) in (Union, types.UnionType) else (tp,)


def _may_unwrap(value: dict[str, Any], tp: Any) -> bool:
    """Whether a one-key object can be a wrapper. Not when the type takes objects of any shape
    (a dict, Any), and not when the key is a field of the expected class and the value isn't a
    whole object: {"category": null} for `Ticket | None` is a Ticket with a bad field, not None."""
    options = _options(tp)
    if any(o is Any or o is dict or typing.get_origin(o) is dict for o in options):
        return False
    (key, inner), fields = next(iter(value.items())), set()
    for option in options:
        if _is_dataclass(option):
            fields |= {name for name, _, _ in _init_fields(option)}
    return key not in fields or isinstance(inner, dict)


def _bare_word(text: str, tp: Any) -> Any:
    """The usual near-misses that aren't JSON of the right kind: yes/no for a bool, an unquoted
    (or single-quoted) label for a string Literal. Also inside an Optional. _NO_WORD if neither."""
    options = _options(tp)
    literals = [typing.get_args(o) for o in options if typing.get_origin(o) is Literal]
    unquoted = text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"" else text
    for label in (text, unquoted):  # the text as it is first: Literal["'quoted'"] has the quotes
        if any(label in labels for labels in literals):  # a str only equals a str option
            return label  # an exact label first: "no" for Literal["no", "partial"] | bool is the label
    word = text.lower().rstrip(".")
    if bool in options and str not in options and word in {"true", "yes", "false", "no"}:
        return word in {"true", "yes"}
    return _NO_WORD


def _unfence(text: str) -> tuple[str, str]:
    """(the inside of a Markdown code fence, the text before it). The fence is the whole reply, or
    ends the reply after a line of prose ("Here you go:"). The language tag is ignored, whatever it
    is. A reply with more fences, or text after the fence (a second answer?), is left as it is (and
    fails). Linear time: fences only count at the start of a line, so "```" in a JSON string is fine."""
    if not text.endswith("```") or len(text) < 6:
        return text, ""
    if "\n" not in text:  # ```4``` or ```json [1, 2]```
        inline = re.fullmatch(r"```(?:json\b)?(.*)```", text, re.DOTALL | re.IGNORECASE)
        return (inline.group(1).strip() if inline and text.startswith("```") else text), ""
    close = len(text) - 3
    opens = [m.start() for m in re.finditer(r"^```", text, re.MULTILINE) if m.start() != close]
    if len(opens) != 1:
        return text, ""
    body = text.find("\n", opens[0]) + 1  # the line after the opening fence and its language tag
    return (text[body:close].strip(), text[: opens[0]].strip()) if 0 < body <= close else (text, "")


def _is_answer(text: str, tp: Any) -> bool:
    """Whether the prose before a fence is an answer itself, like the 4 in "4\n```json\n5\n```"."""
    try:
        parse(text, tp)
    except ValueError:
        return False
    return True


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


class _Rounded(float):
    """A JSON number that only rounds to a whole float: 3.9999999999999999 is 4.0, 1e-400 is 0.0,
    12345678901234567.0 is ...568.0. The model didn't write that integer, so it's never read as one.
    Its repr is what the model wrote, so an error message doesn't say "got 4.0"."""

    digits = ""

    def __repr__(self) -> str:
        return self.digits


def _finite(digits: str) -> float:
    number = float(digits)
    if not math.isfinite(number):
        raise ValueError(f"{digits[:50]} is out of range")
    if number.is_integer():
        try:
            exact = decimal.Decimal(digits) == decimal.Decimal(number)
        except ArithmeticError:  # an exponent beyond what Decimal holds, like 0e99999999999999999999
            exact = False
        if not exact:
            rounded = _Rounded(number)
            rounded.digits = digits[:50]
            return rounded
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
        # 3.0 is 3, but not a number that only rounds to a whole float (see _Rounded).
        exact = isinstance(value, float) and value.is_integer() and not isinstance(value, _Rounded)
        whole = isinstance(value, int) or exact
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
        kwargs = {}
        for name, hint, required in _init_fields(tp):
            if name in value:
                try:
                    kwargs[name] = validate(value[name], hint)
                except ValueError as exc:
                    raise ValueError(f"field {name!r}: {exc}") from None
            elif required:
                raise ValueError(f"missing required field {name!r}")
        names = [name for name, _, _ in _init_fields(tp)]
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
    try:
        return shorten(repr(value), limit)
    except RecursionError:
        return f"<{type(value).__name__} nested too deeply to show>"


def shorten(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _kind(value: Any) -> str:
    """bool, number, string, ...: the JSON kind of a value, for comparing Literal options."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, _Rounded):
        return "rounded"  # matches no option: 0.99999999999999999 isn't Literal[1]
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _is_dataclass(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


@functools.lru_cache(maxsize=256)
def _init_fields(tp: type) -> tuple[tuple[str, Any, bool], ...]:
    """(name, type, required) for each __init__ argument of a dataclass: its fields, and InitVars
    (passed to __post_init__, never stored). Not ClassVars, nor fields the class sets itself
    (init=False). The model is asked for these, and only these are read from its answer."""
    hints = _hints(tp)
    stored = {f.name for f in dataclasses.fields(tp) if f.init}  # no ClassVars, no InitVars
    result = []
    for f in tp.__dataclass_fields__.values():  # type: ignore[attr-defined]  # fields() skips InitVars
        hint = hints.get(f.name, Any)
        if isinstance(hint, dataclasses.InitVar):
            hint = hint.type
        elif hint is dataclasses.InitVar:  # a bare InitVar
            hint = Any
        elif f.name not in stored:
            continue
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        result.append((f.name, hint, required))
    return tuple(result)


def _hints(tp: type) -> dict[str, Any]:
    """typing.get_type_hints(tp), also where Python 3.10's falls short: it can't evaluate a string
    "InitVar[int]" (from __future__ import annotations), and leaves strings inside builtin
    generics (list["Item"]). Annotations are evaluated in the class's module, as it would."""
    module = sys.modules.get(tp.__module__)
    namespace = {**(vars(module) if module else {}), tp.__name__: tp}
    try:
        hints = typing.get_type_hints(tp)
    except TypeError:
        hints = {}
        for cls in reversed(tp.__mro__):
            hints.update(cls.__dict__.get("__annotations__", {}))
    return {name: resolve_strings(hint, namespace) for name, hint in hints.items()}


def resolve_strings(tp: Any, namespace: dict[str, Any]) -> Any:
    """A type with any string (or ForwardRef) inside it evaluated, or as it is if one can't be."""
    try:
        if isinstance(tp, typing.ForwardRef):
            tp = tp.__forward_arg__
        if isinstance(tp, str):
            tp = eval(tp, namespace)  # what typing.get_type_hints does with a string annotation
    except Exception:
        return tp  # describe() then names it as unsupported
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal or not args:
        return tp
    resolved = tuple(resolve_strings(a, namespace) for a in args)
    if resolved == args:
        return tp
    if origin in (Union, types.UnionType):
        return functools.reduce(operator.or_, resolved)
    if origin in (list, dict):
        return types.GenericAlias(origin, resolved)
    return tp
