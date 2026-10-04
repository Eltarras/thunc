"""@thunc.function: a Python signature + instructions (docstring or string) -> an AI-backed function."""

from __future__ import annotations

import asyncio
import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

from .core import _call
from .schema import describe, resolve_strings

P = ParamSpec("P")
R = TypeVar("R")


@overload
def function(func: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def function(
    *,
    instructions: str | None = None,
    retries: int = 2,
    ensure: Callable[[Any], bool] | None = None,
    backend: str | None = None,
    model: str | None = None,
    system: str | None = None,
    cache: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def function(
    func: Callable[..., Any] | None = None,
    /,
    *,
    instructions: str | None = None,
    retries: int = 2,
    ensure: Callable[[Any], bool] | None = None,
    backend: str | None = None,
    model: str | None = None,
    system: str | None = None,
    cache: bool = False,
) -> Any:
    """Turn a function signature into an AI-backed function.

        @thunc.function
        def category(ticket: str) -> Literal["bug", "billing", "other"]:
            \"\"\"Classify this support ticket.\"\"\"
            ...

    - Instructions: the docstring, or `instructions=` (a string built in code).
    - Inputs: the call's arguments, sent separately from the instructions (`self`/`cls` skipped).
    - Output: the return annotation (none means str); `ensure=` adds a check that triggers a retry.
    - `system=` replaces thunc's default system prompt; thunc keeps its two rules (inputs are data,
      reply with the value only) after it. See thunc.call.
    - `cache=True` saves answers on disk and reuses them for the same inputs. Use it for functions
      that should give one answer per input (classify, extract, score), not for ones meant to vary.
      `thunc.clear_cache(func)` deletes this function's saved answers.
    - The body must stay empty; `async def` gives an awaitable.
    """

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        return _build(
            f,
            instructions,
            dict(retries=retries, ensure=ensure, backend=backend, model=model, system=system, cache=cache),
        )

    return decorate(func) if func is not None else decorate


class _Signature(typing.NamedTuple):
    """What thunc reads from an empty-bodied function: the prompt, the inputs and the return type."""

    instructions: str
    sig: inspect.Signature
    skip: str | None  # `self` or `cls`, which isn't sent as an input
    returns: Any
    is_async: bool

    def inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: v for k, v in bound.arguments.items() if k != self.skip}


def _read_signature(func: Callable[..., Any], instructions: str | None, label: str) -> _Signature:
    """Check an empty-bodied function and read its prompt, inputs and return type.
    `label` names the decorator in error messages ("@thunc.function", "@agent.task")."""
    name = func.__qualname__
    if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
        raise TypeError(f"{label} {name}: generators are not supported.")
    is_async = inspect.iscoroutinefunction(func)
    if func.__code__.co_code not in _empty_bodies(is_async):
        raise TypeError(
            f"{label} {name}: the body must be empty (a docstring and/or `...`). "
            "The model replaces the body, so code there would never run."
        )
    instructions = instructions or inspect.getdoc(func)
    if not instructions:
        raise TypeError(
            f"{label} {name} needs instructions: write a docstring or pass instructions=... "
            "(docstrings are removed under python -OO)."
        )
    sig = inspect.signature(func)
    first = next(iter(sig.parameters), None)
    skip = first if first in ("self", "cls") else None
    # resolve_strings: Python 3.10's get_type_hints leaves "Ticket" inside `-> list["Ticket"]`.
    returns = resolve_strings(typing.get_type_hints(func).get("return", str), func.__globals__)
    describe(returns)  # unsupported return types fail here, not at the first call
    return _Signature(instructions, sig, skip, returns, is_async)


def _wrap(func: Callable[..., Any], is_async: bool, run: Callable[..., Any]) -> Callable[..., Any]:
    """`run` with the function's name and docstring; an awaitable if the function was `async def`."""

    async def run_async(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(run, *args, **kwargs)

    return functools.wraps(func)(run_async if is_async else run)


def _build(func: Callable[..., Any], instructions: str | None, options: dict[str, Any]) -> Callable[..., Any]:
    name = func.__qualname__
    spec = _read_signature(func, instructions, "@thunc.function")

    def run(*args: Any, **kwargs: Any) -> Any:
        inputs = spec.inputs(args, kwargs)
        return _call(spec.instructions, inputs, spec.returns, name=name, module=func.__module__, **options)

    wrapper = _wrap(func, spec.is_async, run)
    wrapper.__dict__["__thunc_instructions__"] = spec.instructions  # for debugging
    wrapper.__dict__["__thunc_function__"] = name  # for thunc.clear_cache(func); also the cache key's name
    return wrapper


@functools.cache
def _empty_bodies(is_async: bool) -> frozenset[bytes]:
    """Bytecode of each allowed empty body (docstring, `...`, `pass`, `raise NotImplementedError`),
    compiled by the running interpreter so the check holds on every Python version."""
    bodies = ['"""doc"""', "...", "pass", "raise NotImplementedError", "raise NotImplementedError()"]
    bodies += [f'"""doc"""\n    {b}' for b in bodies[1:]]
    codes = set()
    for body in bodies:
        namespace: dict[str, Any] = {}
        exec(f"{'async def' if is_async else 'def'} f(a, *args, b=1, **kw):\n    {body}\n", namespace)
        codes.add(namespace["f"].__code__.co_code)
    return frozenset(codes)
