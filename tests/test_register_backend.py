"""The public backend interface: thunc.register_backend for text and typed backends."""

import importlib
import json
from typing import Literal

import pytest

import thunc
from thunc import backends


@pytest.fixture(autouse=True)
def forget():
    """Registrations made in a test don't outlive it."""
    registries = (backends.BACKENDS, backends.TYPED_BACKENDS, backends.DEFAULT_MODELS)
    saved = [dict(registry) for registry in registries]
    yield
    for registry, before in zip(registries, saved, strict=True):
        registry.clear()
        registry.update(before)


def test_a_text_backend_is_used_by_name_everywhere(monkeypatch):
    sent = []

    def local(text, *, system, model, api_key, timeout):
        sent.append({"text": text, "system": system, "model": model, "api_key": api_key, "timeout": timeout})
        return "4"

    thunc.register_backend("local", local, default_model="tiny-1b")
    assert thunc.call("Rate it.", {"n": 1}, returns=int, backend="local") == 4
    thunc.configure(backend="local", api_key="k", timeout=9)
    assert thunc.call("Rate it.", returns=int) == 4
    monkeypatch.setitem(thunc.config._settings, "backend", None)
    monkeypatch.setenv("THUNC_BACKEND", "local")
    assert thunc.call("Rate it.", returns=int) == 4
    assert "<n>\n1\n</n>" in sent[0]["text"] and "You are a function" in sent[0]["system"]
    assert sent[1]["api_key"] == "k" and sent[1]["timeout"] == 9 and sent[1]["model"] is None


def test_options_a_backend_does_not_take_are_not_sent(tmp_path):
    calls = []

    def plain(text, *, system, model, api_key, timeout):
        calls.append("plain")
        return json.dumps({"tool": "finish", "args": {"value": 1}})

    def keen(text, *, system, model, api_key, timeout, effort=None):
        calls.append(effort)
        return json.dumps({"tool": "finish", "args": {"value": 1}})

    thunc.register_backend("plain", plain)
    thunc.register_backend("keen", keen)
    for name in ("plain", "keen"):
        agent = thunc.Agent(f"a-{name}", workdir=tmp_path, backend=name, effort="high")
        assert agent.call("Pick 1.", returns=int) == 1  # agents run on the text protocol
    assert calls == ["plain", "high"]


def test_a_typed_backend_gets_the_question_apart():
    asked = []

    def judge(instructions, inputs, returns, *, system, timeout):
        asked.append((instructions, inputs, returns))
        return json.dumps("bug")

    thunc.register_backend("judge", judge, typed=True)
    kinds = Literal["bug", "other"]
    assert thunc.call("Label it.", {"text": "crash"}, returns=kinds, backend="judge") == "bug"
    assert asked == [("Label it.", {"text": "crash"}, kinds)]
    with pytest.raises(thunc.ThuncError, match="cannot run agents"):
        thunc.Agent("a", workdir=".", backend="judge").call("x")


def test_errors_a_backend_raises_reach_the_caller_and_transient_ones_are_retried_by_agents(tmp_path, monkeypatch):
    replies = iter([thunc.TransientError("rate limited"), json.dumps({"tool": "finish", "args": {"value": 2}})])

    def flaky(text, *, system, model, api_key, timeout):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    thunc.register_backend("flaky", flaky)
    monkeypatch.setattr(importlib.import_module("thunc.agent"), "RETRY_DELAY", 0)  # not the @thunc.agent function
    assert thunc.Agent("f", workdir=tmp_path, backend="flaky").call("x", returns=int) == 2

    def down(text, **kwargs):
        raise thunc.ThuncError("the server is down")

    thunc.register_backend("down", down)
    with pytest.raises(thunc.ThuncError, match="the server is down"):
        thunc.call("x", backend="down")


def test_a_backend_that_returns_something_else_than_text_is_an_error():
    thunc.register_backend("number", lambda text, **kwargs: 4)
    with pytest.raises(thunc.ThuncError, match="returned int, not text"):
        thunc.call("x", backend="number")


@pytest.mark.parametrize(
    "backend, problem",
    [
        (lambda text: "x", "must take the keyword arguments system, model, api_key, timeout"),
        (lambda *, system, model, api_key, timeout: "x", "must take 1 positional argument"),
        ("not callable", "must be callable"),
    ],
)
def test_a_backend_that_cant_be_called_as_thunc_calls_it_fails_at_registration(backend, problem):
    with pytest.raises(TypeError, match=problem):
        thunc.register_backend("bad", backend)
    assert "bad" not in backends.BACKENDS


def test_names_built_in_or_taken_need_care():
    for name in ("anthropic", "openai", "claude-code", "codex", "jev"):
        with pytest.raises(ValueError, match="built-in"):
            thunc.register_backend(name, lambda text, **kwargs: "x", replace=True)
    thunc.register_backend("mine", lambda text, **kwargs: "a")
    with pytest.raises(ValueError, match="registered already"):
        thunc.register_backend("mine", lambda text, **kwargs: "b")
    thunc.register_backend("mine", lambda text, **kwargs: "b", replace=True)
    assert thunc.call("x", backend="mine") == "b"
    thunc.register_backend("mine", lambda instructions, inputs, returns, **kwargs: "true", typed=True, replace=True)
    assert "mine" not in backends.BACKENDS and thunc.call("x", returns=bool, backend="mine") is True
    thunc.unregister_backend("mine")
    with pytest.raises(thunc.ThuncError, match="Unknown backend 'mine'"):
        thunc.call("x", backend="mine")
    for bad in ("", " mine", 3):
        with pytest.raises(ValueError, match="name"):
            thunc.register_backend(bad, lambda text, **kwargs: "x")
    with pytest.raises(ValueError, match="built-in"):
        thunc.unregister_backend("anthropic")


def test_the_default_model_is_the_model_in_the_cache_key_and_events(tmp_path, monkeypatch):
    thunc.register_backend("local", lambda text, **kwargs: "ok", default_model="tiny-1b")
    thunc.configure(cache_dir=str(tmp_path / "cache"))
    assert thunc.call("x", backend="local", cache=True) == "ok"
    (entry,) = [json.loads(path.read_text()) for path in (tmp_path / "cache").iterdir()]
    assert entry["backend"] == "local" and entry["model"] == "tiny-1b"
