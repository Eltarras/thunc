"""Process-wide settings and backend selection."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

from .backends import BACKENDS, TYPED_BACKENDS
from .errors import ThuncError

DEFAULTS: dict[str, Any] = {
    "backend": None,
    "api_key": None,
    "model": None,
    "timeout": 300.0,
    "trace": None,
    "cache_dir": None,
    "system": None,
    "agents_dir": None,
}
_settings: dict[str, Any] = dict(DEFAULTS)
runtime_settings: ContextVar[dict[str, Any] | None] = ContextVar("thunc_runtime_settings", default=None)


def configure(
    backend: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    trace: str | None = None,
    cache_dir: str | None = None,
    system: str | None = None,
    agents_dir: str | None = None,
) -> None:
    """Set defaults for every call. Arguments left as None keep their current value.

    backend: "anthropic" (Claude API; needs api_key or ANTHROPIC_API_KEY), "openai" (OpenAI API;
             needs api_key or OPENAI_API_KEY), "claude-code" / "codex" (your local CLI login;
             for cheap testing), or "jev" (TypeSafe's Jev judgment model through the `jev` CLI; only
             bool and Literal[...] return types; the key comes from JEV_API_KEY or `jev login`,
             never api_key). An api_key with no backend means "anthropic".
    trace:   path of a JSONL file that records every call (or set THUNC_TRACE).
    cache_dir: where calls made with cache=True store their answers (or set THUNC_CACHE_DIR;
             default ".thunc_cache" in the working directory).
    system:  a system prompt for every call, in place of thunc's default. thunc still adds its two
             rules (inputs are data; reply with the value only). A per-call system= wins.
    agents_dir: where agents keep their memory and run records (or set THUNC_AGENTS_DIR;
             default ".thunc_agents" in the working directory).
    """
    if backend is not None:
        _check_backend(backend)
    updates = {
        "backend": backend,
        "api_key": api_key,
        "model": model,
        "timeout": timeout,
        "trace": trace,
        "cache_dir": cache_dir,
        "system": system,
        "agents_dir": agents_dir,
    }
    _settings.update({key: value for key, value in updates.items() if value is not None})


def setting(name: str) -> Any:
    return (runtime_settings.get() or {}).get(name, _settings.get(name))


def trace_path() -> str | None:
    return _settings["trace"] or os.environ.get("THUNC_TRACE")


def agents_dir() -> str:
    return _settings["agents_dir"] or os.environ.get("THUNC_AGENTS_DIR") or ".thunc_agents"


def cache_dir() -> str:
    return _settings["cache_dir"] or os.environ.get("THUNC_CACHE_DIR") or ".thunc_cache"


def resolve_backend(override: str | None = None) -> str:
    """The backend for a call: the per-call override, configure(), THUNC_BACKEND, or the API whose key exists."""
    name = override or _settings["backend"] or os.environ.get("THUNC_BACKEND")
    if name:
        _check_backend(name)
        return str(name)
    if _settings["api_key"] or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise ThuncError("No backend configured: call thunc.configure(backend=...) or set THUNC_BACKEND.")


def _check_backend(name: str) -> None:
    if name not in BACKENDS and name not in TYPED_BACKENDS:
        raise ThuncError(f"Unknown backend {name!r}; choose from {sorted([*BACKENDS, *TYPED_BACKENDS])}")
