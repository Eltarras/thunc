"""Backend selection, the CLI backends and the OpenAI backend (subprocess and the SDK are stubbed)."""

import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import thunc
from thunc import backends


def stub_cli(monkeypatch, stdout="", returncode=0, calls=None, write_file=None):
    def run(args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        if write_file is not None:
            with open(args[args.index("--output-last-message") + 1], "w") as f:
                f.write(write_file)
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "run", run)


def test_no_backend_configured():
    with pytest.raises(thunc.ThuncError, match="No backend configured"):
        thunc.call("hi")


def test_unknown_backend():
    with pytest.raises(thunc.ThuncError, match="Unknown backend"):
        thunc.configure(backend="gpt-banana")


def test_api_key_selects_anthropic(monkeypatch):
    seen = {}
    monkeypatch.setitem(backends.BACKENDS, "anthropic", lambda text, **kw: seen.update(kw) or "ok")
    thunc.configure(api_key="sk-test")
    assert thunc.call("hi") == "ok"
    assert seen["api_key"] == "sk-test"


def test_openai_key_selects_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setitem(backends.BACKENDS, "openai", lambda text, **kw: "from openai")
    assert thunc.call("hi") == "from openai"


def test_anthropic_key_wins_over_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setitem(backends.BACKENDS, "anthropic", lambda text, **kw: "from anthropic")
    assert thunc.call("hi") == "from anthropic"


def test_env_selects_backend(monkeypatch):
    monkeypatch.setenv("THUNC_BACKEND", "codex")
    monkeypatch.setitem(backends.BACKENDS, "codex", lambda text, **kw: "from codex")
    assert thunc.call("hi") == "from codex"


def test_claude_code(monkeypatch):
    calls = []
    stub_cli(monkeypatch, json.dumps({"result": "pong", "is_error": False}), calls=calls)
    thunc.configure(backend="claude-code")
    assert thunc.call("ping") == "pong"
    args, kwargs = calls[0]
    assert args[:2] == ["claude", "-p"] and args[args.index("--tools") + 1] == ""
    assert "<instructions>\nping" in kwargs["input"]


def test_claude_code_error(monkeypatch):
    stub_cli(monkeypatch, json.dumps({"result": "Not logged in", "is_error": True}), returncode=1)
    thunc.configure(backend="claude-code")
    with pytest.raises(thunc.ThuncError, match="Not logged in"):
        thunc.call("ping")


def test_codex(monkeypatch):
    stub_cli(monkeypatch, write_file="pong\n")
    thunc.configure(backend="codex")
    assert thunc.call("ping") == "pong"


def test_missing_cli(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda exe: None)
    thunc.configure(backend="codex")
    with pytest.raises(thunc.ThuncError, match="not found on PATH"):
        thunc.call("ping")


def stub_openai(monkeypatch, response, calls=None):
    """A fake `openai` module whose client returns `response` from responses.create."""

    class Client:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            return response

    module = ModuleType("openai")
    module.OpenAI = Client
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    module.APIStatusError = type("APIStatusError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", module)


def openai_response(text="", status="completed", reason=None, refusal=False):
    part = SimpleNamespace(type="refusal", refusal="no") if refusal else SimpleNamespace(type="output_text", text=text)
    return SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="message", content=[part])],
        output_text=text,
    )


def test_openai(monkeypatch):
    calls = []
    stub_openai(monkeypatch, openai_response("pong"), calls=calls)
    thunc.configure(backend="openai")
    assert thunc.call("ping") == "pong"
    assert calls[0]["model"] == backends.DEFAULT_OPENAI_MODEL
    assert "<instructions>\nping" in calls[0]["input"] and calls[0]["instructions"]


def test_openai_refusal(monkeypatch):
    stub_openai(monkeypatch, openai_response(refusal=True))
    thunc.configure(backend="openai")
    with pytest.raises(thunc.ThuncError, match="declined"):
        thunc.call("ping")


def test_openai_cut_off(monkeypatch):
    stub_openai(monkeypatch, openai_response("po", status="incomplete", reason="max_output_tokens"))
    thunc.configure(backend="openai")
    with pytest.raises(thunc.ThuncError, match="cut off"):
        thunc.call("ping")


def test_openai_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    thunc.configure(backend="openai")
    with pytest.raises(thunc.ThuncError, match=r"thunc\[openai\]"):
        thunc.call("ping")
