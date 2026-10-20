"""Return types: describe one to the model, and turn the model's text back into a checked value.

Supported: str, bool, int, float, None, Any, Literal[...], Enum classes, list[T], dict[str, T],
unions (T | None), dataclasses and Pydantic v2-compatible models (without requiring Pydantic).
"""

from __future__ import annotations

import copy
import dataclasses
import enum
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
    if is_enum(tp):
        return "exactly one of these JSON values: " + ", ".join(_member(m) for m in _members(tp))
    if _is_dataclass(tp):
        fields = [
            f'"{name}": {describe(hint, True)}{"" if required else " (optional)"}'
            for name, hint, required in _init_fields(tp)
        ]
        return "a JSON object with these fields: {" + ", ".join(fields) + "}"
    if _is_pydantic_model(tp):
        return _describe_schema(_pydantic_schema(tp))
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
    (a dict, Any), and not when the key is a field of the expected class: {"category": null} for
    `Ticket | None` is a Ticket with a bad field, not None."""
    options = _options(tp)
    if any(o is Any or o is dict or typing.get_origin(o) is dict for o in options):
        return False
    key, fields = next(iter(value)), set()
    for option in options:
        if _is_dataclass(option):
            fields |= {name for name, _, _ in _init_fields(option)}
        elif _is_pydantic_model(option):
            fields |= _pydantic_fields(option)
    return key not in fields


def _bare_word(text: str, tp: Any) -> Any:
    """The usual near-misses that aren't JSON of the right kind: yes/no for a bool, an unquoted
    (or single-quoted) label for a string Literal. Also inside an Optional. _NO_WORD if neither."""
    options = _options(tp)
    # (label, what it reads as): a Literal's string options, and an Enum's members with string values
    labels = [(a, a) for o in options if typing.get_origin(o) is Literal for a in typing.get_args(o)]
    labels += [(m.value, m) for o in options if is_enum(o) for m in _members(o)]
    unquoted = text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"" else text
    for label in (text, unquoted):  # the text as it is first: Literal["'quoted'"] has the quotes
        for option in (result for value, result in labels if isinstance(value, str) and value == label):
            return option  # an exact label first: "no" for Literal["no", "partial"] | bool is the label
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
    newline = text.find("\n", opens[0])
    if not 0 <= newline < close:
        return text, ""
    first, rest = text[opens[0] + 3 : newline].strip(), text[newline + 1 : close].strip()
    # The opening line holds a language tag ("json", "python"), maybe with attributes ("json title=x"),
    # when more lines follow. "json" can also be followed by the answer: ```json{"a": 1}, ```json true. Anything
    # else on it is the start of the answer: ```4, ```[1, or a label like ```not urgent.
    tag = re.match(r"[A-Za-z][\w+.#-]*", first)
    after = first[tag.end() :].lstrip() if tag else ""
    if tag and tag.group().lower() == "json" and after and (after[0] in '[{"-0123456789' or not rest):
        first = after
    elif tag and rest and (tag.end() == len(first) or first[tag.end()].isspace()):
        first = ""
    inside = f"{first}\n{rest}".strip()
    return inside, text[: opens[0]].strip()


def _is_answer(text: str, tp: Any) -> bool:
    """Whether the prose before a fence is an answer itself, like the 4 in "4\n```json\n5\n```"."""
    try:
        _parse(text, tp)
    except RecursionError:  # JSON as deep as the stack allows here: no line of prose
        return True
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
        # 3.0 is 3, while a float holds every integer exactly (12345678901234567.0 isn't one).
        exact = isinstance(value, float) and value.is_integer() and abs(value) < 2**53
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
    if is_enum(tp):
        values = [m.value for m in _members(tp)]
        match = [m for m in _members(tp) if _kind(m.value) == _kind(value) and m.value == value]
        _expect(bool(match), value, f"one of {values}")
        return match[0]
    if _is_pydantic_model(tp):
        schema = _pydantic_schema(tp)
        properties = schema.get("properties")
        fields = _pydantic_fields(tp, schema)
        if (
            isinstance(value, dict)
            and value
            and isinstance(properties, dict)
            and properties
            and fields.isdisjoint(value)
        ):
            raise ValueError(f"expected an object with the fields {sorted(fields)}, got {short_repr(value)}")
        validate_json = getattr(tp, "model_validate_json", None)
        try:
            if callable(validate_json):
                return validate_json(json.dumps(value), strict=True)
            return tp.model_validate(value)
        except Exception as exc:
            method = "model_validate_json" if callable(validate_json) else "model_validate"
            message = f"{tp.__name__}.{method}(...) failed: {type(exc).__name__}: {exc}"
            raise ValueError(shorten(message)) from exc
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        # Compared by kind too, or true would match Literal[1] (True == 1 in Python).
        match = [a for a in args if _kind(a) == _kind(value) and a == value]
        _expect(bool(match), value, f"one of {list(args)}")
        return match[0]
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
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):  # a str Enum member too: Literal[Color.RED] is "red" in JSON
        return "string"
    return type(value).__name__


def is_enum(tp: Any) -> bool:
    """An Enum class whose members are the answers. Not a Flag: its combinations aren't members."""
    return isinstance(tp, type) and issubclass(tp, enum.Enum) and not issubclass(tp, enum.Flag)


def _members(tp: type[enum.Enum]) -> list[enum.Enum]:
    """The members, without aliases. Their values must be what JSON can carry exactly."""
    members = list(tp)
    for m in members:
        exact = isinstance(m.value, (str, int, bool)) or isinstance(m.value, float) and math.isfinite(m.value)
        if not exact:
            raise ThuncError(
                f"Unsupported return type: {tp.__name__}, whose member {m.name} has the value {m.value!r}. "
                "An Enum's values must be strings, numbers or booleans."
            )
    if not members:
        raise ThuncError(f"Unsupported return type: {tp.__name__} has no members")
    return members


def _member(m: enum.Enum) -> str:
    """A member as the model sees it: its value, with its name where the value doesn't say it (3 (HIGH))."""
    shown = json.dumps(m.value, ensure_ascii=False)
    return shown if str(m.value).lower() == m.name.lower() else f"{shown} ({m.name})"


def _is_dataclass(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _is_pydantic_model(tp: Any) -> bool:
    """A Pydantic-compatible model class, without importing Pydantic."""
    return (
        isinstance(tp, type)
        and callable(getattr(tp, "model_json_schema", None))
        and callable(getattr(tp, "model_validate", None))
    )


def _pydantic_schema(tp: Any) -> dict[str, Any]:
    schema = tp.model_json_schema()
    if not isinstance(schema, dict):
        raise ThuncError(f"Unsupported return type: {tp!r} has an invalid model_json_schema()")
    return _inline_local_refs(schema)


def _pydantic_fields(tp: Any, schema: dict[str, Any] | None = None) -> set[str]:
    """Field names and aliases that must not be mistaken for response wrappers."""
    fields: set[str] = set()
    properties = (schema if schema is not None else _pydantic_schema(tp)).get("properties")
    if isinstance(properties, dict):
        fields.update(properties)

    def add_alias(alias: Any) -> None:
        if isinstance(alias, str):
            fields.add(alias)
        choices = getattr(alias, "choices", None)
        if isinstance(choices, (list, tuple)):
            for choice in choices:
                add_alias(choice)
        path = getattr(alias, "path", None)
        if isinstance(path, (list, tuple)) and path and isinstance(path[0], str):
            fields.add(path[0])

    model_fields = getattr(tp, "model_fields", None)
    if isinstance(model_fields, dict):
        for name, field in model_fields.items():
            fields.add(name)
            add_alias(getattr(field, "alias", None))
            add_alias(getattr(field, "validation_alias", None))
    return fields


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a schema and inline its local references, so it stays valid when embedded elsewhere."""
    root = copy.deepcopy(schema)

    def lookup(ref: str) -> Any:
        if ref == "#":
            return root
        if not ref.startswith("#/"):
            return None
        target: Any = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                raise ThuncError(f"Invalid local JSON Schema reference: {ref!r}")
            target = target[part]
        return target

    def shape(target: Any) -> dict[str, Any]:
        if isinstance(target, dict) and target.get("type") in {"array", "object"}:
            return {"type": target["type"]}
        return {}

    def inline(value: Any, resolving: frozenset[str] = frozenset()) -> Any:
        if isinstance(value, list):
            return [inline(item, resolving) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#"):
            target = lookup(ref)
            resolved = shape(target) if ref in resolving else inline(target, resolving | {ref})
            siblings = inline({key: item for key, item in value.items() if key != "$ref"}, resolving | {ref})
            return {"allOf": [resolved], **siblings} if siblings else resolved
        # Pydantic's discriminator mapping contains reference strings that can't point at inlined
        # branches. oneOf and each branch's const still carry the same validation semantics.
        return {key: inline(item, resolving) for key, item in value.items() if key not in {"$defs", "discriminator"}}

    result = inline(root)
    if not isinstance(result, dict):
        raise ThuncError("A model's JSON Schema must be an object")
    return result


def _describe_schema(schema: dict[str, Any]) -> str:
    """Describe the basic JSON Schema shapes emitted for Pydantic model fields."""
    if "const" in schema:
        return "exactly this JSON value: " + json.dumps(schema["const"], ensure_ascii=False)
    if "enum" in schema:
        return "exactly one of these JSON values: " + ", ".join(
            json.dumps(value, ensure_ascii=False) for value in schema["enum"]
        )
    for alternatives in ("anyOf", "oneOf"):
        options = schema.get(alternatives)
        if isinstance(options, list):
            return " or ".join(_describe_schema(option) for option in options)
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1:
        return _describe_schema(all_of[0])
    kind = schema.get("type")
    simple = {
        "string": "a JSON string",
        "boolean": "a JSON boolean (true or false)",
        "integer": "a JSON integer",
        "number": "a JSON number",
        "null": "null",
    }
    if kind in simple:
        return simple[kind]
    if kind == "array":
        items = schema.get("items")
        return "a JSON array" + (f" whose items are each {_describe_schema(items)}" if isinstance(items, dict) else "")
    if kind == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict):
            required = schema.get("required", [])
            fields = [
                f'"{name}": {_describe_schema(field)}{"" if name in required else " (optional)"}'
                for name, field in properties.items()
                if isinstance(field, dict)
            ]
            return "a JSON object with these fields: {" + ", ".join(fields) + "}"
        values = schema.get("additionalProperties")
        return "a JSON object" + (
            f" whose values are each {_describe_schema(values)}" if isinstance(values, dict) else ""
        )
    return "any JSON value"


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


def json_schema(tp: Any, _seen: frozenset[type] = frozenset()) -> dict[str, Any]:
    """A JSON Schema for `tp`, for tools on the native APIs. Where JSON Schema can't say it exactly
    it's looser, never stricter: the value is checked with validate() either way."""
    simple = {str: "string", bool: "boolean", int: "integer", float: "number", type(None): "null"}
    if tp in simple:
        return {"type": simple[tp]}
    if tp is Any:
        return {}
    if _is_dataclass(tp):
        if tp in _seen:  # a dataclass that contains itself
            return {"type": "object"}
        fields = _init_fields(tp)
        return {
            "type": "object",
            "properties": {name: json_schema(hint, _seen | {tp}) for name, hint, _ in fields},
            "required": [name for name, _, required in fields if required],
        }
    if is_enum(tp):
        return {"enum": [m.value for m in _members(tp)]}
    if _is_pydantic_model(tp):
        return _pydantic_schema(tp)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        return {"enum": [a.value if isinstance(a, enum.Enum) else a for a in args]}
    if origin in (Union, types.UnionType):
        return {"anyOf": [json_schema(a, _seen) for a in args]}
    if tp is list or origin is list:
        return {"type": "array", **({"items": json_schema(args[0], _seen)} if args else {})}
    if tp is dict or origin is dict:
        return {"type": "object", **({"additionalProperties": json_schema(args[1], _seen)} if args else {})}
    return {}
