"""Deprecating part of the API: it keeps working, warns where it's used, and goes in a named release.

    @deprecated(since="0.7", removed_in="0.9", instead="thunc.acall")
    def old_call(...): ...

    @renamed_parameter("retry", "retries", since="0.7", removed_in="0.9")
    def call(..., retries=2): ...

The warning is a ThuncDeprecationWarning, shown once per place in the user's code by default (Python
hides plain DeprecationWarnings outside __main__), and turned off with
warnings.filterwarnings("ignore", category=thunc.ThuncDeprecationWarning). Every deprecation is listed
in DEPRECATIONS, and a test fails once thunc's version reaches one's removed_in, so none outlives it.
See "Deprecations" in CONTRIBUTING.md for the policy.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ThuncDeprecationWarning(DeprecationWarning):
    """Part of thunc's API you use is deprecated: it still works, and is removed in the release named."""


# Shown by default, once per place it's raised from: a plain DeprecationWarning is hidden outside __main__.
warnings.filterwarnings("default", category=ThuncDeprecationWarning)


@dataclasses.dataclass(frozen=True)
class Deprecation:
    what: str  # "thunc.old_call", "thunc.call(retry=)"
    since: str  # the version that deprecated it
    removed_in: str  # the version that removes it
    instead: str | None  # what to use instead

    @property
    def message(self) -> str:
        text = f"{self.what} is deprecated since thunc {self.since} and will be removed in {self.removed_in}"
        return f"{text}: use {self.instead} instead." if self.instead else f"{text}."


DEPRECATIONS: list[Deprecation] = []  # every deprecation declared, for the test that keeps them honest


def warn(deprecation: Deprecation, stacklevel: int = 2) -> None:
    """Warn about one use. stacklevel counts from the caller of warn(), as warnings.warn does."""
    warnings.warn(deprecation.message, ThuncDeprecationWarning, stacklevel=stacklevel + 1)


def deprecated(*, since: str, removed_in: str, instead: str | None = None, what: str | None = None) -> Callable[[F], F]:
    """Mark a function, method or class as deprecated: each call warns, then runs as before."""

    def decorate(func: F) -> F:
        deprecation = _declare(what or f"thunc.{func.__qualname__}", since, removed_in, instead)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn(deprecation)
            return func(*args, **kwargs)

        wrapper.__dict__["__thunc_deprecation__"] = deprecation
        return wrapper  # type: ignore[return-value]  # wraps keeps the signature

    return decorate


def renamed_parameter(old: str, new: str, *, since: str, removed_in: str, what: str | None = None) -> Callable[[F], F]:
    """Accept a keyword argument under its old name, with a warning, and pass it on under the new one.
    Both at once is a TypeError, as passing an argument twice is."""

    def decorate(func: F) -> F:
        deprecation = _declare(f"{what or f'thunc.{func.__qualname__}'}({old}=)", since, removed_in, f"{new}=")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(f"{func.__qualname__}() got both {old}= and {new}=; {old}= is deprecated")
                warn(deprecation)
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


def _declare(what: str, since: str, removed_in: str, instead: str | None) -> Deprecation:
    if _version(removed_in) <= _version(since):
        raise ValueError(f"{what}: removed_in ({removed_in}) must come after since ({since})")
    deprecation = Deprecation(what, since, removed_in, instead)
    DEPRECATIONS.append(deprecation)
    return deprecation


def _version(text: str) -> tuple[int, ...]:
    """'0.7' or '0.7.1' as numbers, for comparing; a pre-release suffix is ignored (1.0.0rc1 is 1.0.0)."""
    parts = []
    for part in text.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits or 0))
    return tuple(parts) + (0,) * (3 - len(parts))
