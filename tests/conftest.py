"""Shared fixtures. No test ever calls a real model: `fake` is a scripted backend."""

import os
import sys

import pytest

import thunc
from thunc import backends, config


class FakeBackend:
    def __init__(self):
        self.replies: list[str] = []  # scripted answers, consumed in order
        self.prompts: list[str] = []  # what was sent
        self.systems: list[str] = []  # the system prompt sent with each

    def __call__(self, text, **kwargs):
        self.prompts.append(text)
        self.systems.append(kwargs["system"])
        if not self.replies:
            raise AssertionError("FakeBackend ran out of scripted replies")
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_settings", dict(config.DEFAULTS))
    for var in (
        "THUNC_BACKEND",
        "THUNC_TRACE",
        "THUNC_CACHE_DIR",
        "THUNC_AGENTS_DIR",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    thunc.configure(agents_dir=str(tmp_path / ".thunc_agents"))  # agents never write into the repo


@pytest.fixture(autouse=True)
def no_real_clis(monkeypatch, tmp_path):
    """`claude`, `codex` and `jev` on PATH fail loudly, so a test that isn't faking one can't reach the
    real CLI and your login. Tests with their own fake CLI put it in front of these."""
    if sys.platform == "win32":  # a shebang script isn't a command there; CI has no real CLI anyway
        return
    blocked = tmp_path / "blocked-clis"
    blocked.mkdir()
    for name in ("claude", "codex", "jev"):
        stub = blocked / name
        stub.write_text(f"#!/bin/sh\necho 'the real {name} is blocked in tests' >&2\nexit 97\n")
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{blocked}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setitem(backends.BACKENDS, "fake", backend)
    thunc.configure(backend="fake")
    return backend
