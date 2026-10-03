"""Backend selection and the CLI backends (subprocess is stubbed; nothing is executed)."""

import json
import subprocess

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
