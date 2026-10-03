"""Process-wide settings and backend selection."""

from __future__ import annotations

import os
from typing import Any

from .backends import BACKENDS
from .errors import ThuncError

DEFAULTS: dict[str, Any] = {
    "backend": None,
    "api_key": None,
    "model": None,
    "timeout": 300.0,
    "trace": None,
    "cache_dir": None,
}
_settings: dict[str, Any] = dict(DEFAULTS)


def configure(
    backend: str | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    trace: str | None = None,
    cache_dir: str | None = None,
) -> None:
    """Set defaults for every call. Arguments left as None keep their current value.

    backend: "anthropic" (Claude API; needs api_key or ANTHROPIC_API_KEY), "openai" (OpenAI API;
             needs api_key or OPENAI_API_KEY), or "claude-code" / "codex" (your local CLI login;
             for cheap testing). An api_key with no backend means "anthropic".
    trace:   path of a JSONL file that records every call (or set THUNC_TRACE).
    cache_dir: where calls made with cache=True store their answers (or set THUNC_CACHE_DIR;
             default ".thunc_cache" in the working directory).
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
    }
    _settings.update({key: value for key, value in updates.items() if value is not None})


def setting(name: str) -> Any:
    return _settings[name]


def trace_path() -> str | None:
    return _settings["trace"] or os.environ.get("THUNC_TRACE")


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
    if name not in BACKENDS:
        raise ThuncError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}")
