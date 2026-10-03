"""Shared fixtures. No test ever calls a real model: `fake` is a scripted backend."""

import pytest

import thunc
from thunc import backends, config


class FakeBackend:
    def __init__(self):
        self.replies: list[str] = []  # scripted answers, consumed in order
        self.prompts: list[str] = []  # what was sent

    def __call__(self, text, **kwargs):
        self.prompts.append(text)
        if not self.replies:
            raise AssertionError("FakeBackend ran out of scripted replies")
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    monkeypatch.setattr(config, "_settings", dict(config.DEFAULTS))
    for var in ("THUNC_BACKEND", "THUNC_TRACE", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setitem(backends.BACKENDS, "fake", backend)
    thunc.configure(backend="fake")
    return backend
