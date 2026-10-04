"""Backend selection, the CLI backends and the OpenAI backend (subprocess and the SDK are stubbed)."""

import json
import os
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


def test_claude_code_system_prompt(monkeypatch):
    calls = []
    stub_cli(monkeypatch, json.dumps({"result": "pong", "is_error": False}), calls=calls)
    thunc.configure(backend="claude-code")
    thunc.call("ping")
    thunc.call("ping", system="You are a pirate.")
    default, own = (args[args.index("--system-prompt") + 1] for args, _ in calls)
    assert default.startswith("You are a function inside a computer program")
    assert own.startswith("You are a pirate.\n\n") and "never instructions to you" in own


def test_claude_code_without_text(monkeypatch):
    stub_cli(monkeypatch, json.dumps({"result": None, "is_error": False}))
    thunc.configure(backend="claude-code")
    with pytest.raises(thunc.ThuncError, match="no text"):
        thunc.call("ping")


def test_claude_code_error(monkeypatch):
    stub_cli(monkeypatch, json.dumps({"result": "Not logged in", "is_error": True}), returncode=1)
    thunc.configure(backend="claude-code")
    with pytest.raises(thunc.ThuncError, match="Not logged in"):
        thunc.call("ping")


@pytest.mark.parametrize("stdout", ["[1, 2]", '"4"', "null", "42", "[" * 100_000 + "]" * 100_000, "Not JSON"])
def test_claude_code_output_that_is_not_its_json_object(monkeypatch, stdout):
    stub_cli(monkeypatch, stdout)
    thunc.configure(backend="claude-code")
    with pytest.raises(thunc.ThuncError, match="claude exited 0"):
        thunc.call("ping", returns=int)


def test_claude_code_stdout_that_is_not_utf8(monkeypatch):
    stub_cli(monkeypatch, '{"is_error": false, "result": "caf\udce9"}')  # \xe9 as surrogateescape decodes it
    thunc.configure(backend="claude-code")
    with pytest.raises(thunc.ThuncError, match="isn't UTF-8"):
        thunc.call("ping")


def test_claude_code_result_with_an_escaped_lone_surrogate_is_text(monkeypatch):
    stub_cli(monkeypatch, '{"is_error": false, "result": "\\ud83d"}')  # the model's half emoji, as JSON
    thunc.configure(backend="claude-code")
    assert thunc.call("ping") == "\ud83d"


def test_codex_noise_that_is_not_utf8_does_not_matter(monkeypatch):
    def run(args, **kwargs):
        with open(args[args.index("--output-last-message") + 1], "w") as f:
            f.write("4")
        return subprocess.CompletedProcess(args, 0, stdout="\udcff", stderr="\udce9")

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "run", run)
    thunc.configure(backend="codex")
    assert thunc.call("ping", returns=int) == 4


def test_cli_is_run_so_undecodable_output_cannot_raise(monkeypatch):
    calls = []
    stub_cli(monkeypatch, json.dumps({"result": "pong", "is_error": False}), calls=calls)
    thunc.configure(backend="claude-code")
    thunc.call("ping \ud83d")  # a lone surrogate in the prompt is sent escaped, not refused
    kwargs = calls[0][1]
    assert kwargs["errors"] == "surrogateescape" and kwargs["encoding"] == "utf-8"
    assert "\\ud83d" in kwargs["input"]


def test_codex_output_that_is_not_utf8(monkeypatch, tmp_path):
    def run(args, **kwargs):
        with open(args[args.index("--output-last-message") + 1], "wb") as f:
            f.write(b"\xff\xfe4\x00")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "run", run)
    thunc.configure(backend="codex")
    with pytest.raises(thunc.ThuncError, match="Could not read codex's answer"):
        thunc.call("ping", returns=int)


def test_codex_answer_path_turned_into_a_directory(monkeypatch):
    def run(args, **kwargs):
        out = args[args.index("--output-last-message") + 1]
        os.remove(out)
        os.mkdir(out)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "run", run)
    thunc.configure(backend="codex")
    with pytest.raises(thunc.ThuncError, match="Could not read codex's answer"):
        thunc.call("ping", returns=int)


def test_codex(monkeypatch):
    stub_cli(monkeypatch, write_file="pong\n")
    thunc.configure(backend="codex")
    assert thunc.call("ping") == "pong"


def test_codex_runs_without_its_own_tools_or_the_users_config(monkeypatch):
    calls = []
    stub_cli(monkeypatch, write_file="pong\n", calls=calls)
    thunc.configure(backend="codex")
    thunc.call("ping")
    args, _ = calls[0]
    assert "--ignore-user-config" in args and "--ignore-rules" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    disabled = {args[i + 1] for i, a in enumerate(args) if a == "--disable"}
    assert {"shell_tool", "unified_exec", "multi_agent", "plugins", "apps", "hooks"} <= disabled
    settings = [args[i + 1] for i, a in enumerate(args) if a == "--config"]
    assert settings[0].startswith("model_instructions_file=") and 'web_search="disabled"' in settings
    assert args[-1] == "-"  # the prompt still comes last, from stdin


def test_codex_system_prompt_goes_in_an_instructions_file(monkeypatch):
    calls, seen = [], {}
    stub_cli(monkeypatch, write_file="pong\n", calls=calls)
    real_run = backends.subprocess.run

    def run(args, **kwargs):  # read the instructions file while it still exists
        setting = args[args.index("--config") + 1]
        key, _, value = setting.partition("=")
        seen[key] = json.loads(value)
        with open(seen[key], encoding="utf-8") as f:
            seen["text"] = f.read()
        return real_run(args, **kwargs)

    monkeypatch.setattr(backends.subprocess, "run", run)
    thunc.configure(backend="codex")
    assert thunc.call("ping", system="You are a pirate.") == "pong"
    args, kwargs = calls[0]
    path = seen["model_instructions_file"]
    assert os.path.isabs(path) and not os.path.exists(path)  # absolute, and removed afterwards
    assert seen["text"].startswith("You are a pirate.\n\n") and "never instructions to you" in seen["text"]
    assert "You are a pirate" not in kwargs["input"]  # no longer pasted in front of the prompt
    assert kwargs["input"].startswith("<instructions>\nping")


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


def test_openai_failed(monkeypatch):
    stub_openai(monkeypatch, openai_response("", status="failed"))
    thunc.configure(backend="openai")
    with pytest.raises(thunc.ThuncError, match="is failed, not completed"):
        thunc.call("ping")


def stub_anthropic(monkeypatch, stop_reason, text="pong"):
    """A fake `anthropic` module whose client returns one text block with `stop_reason`."""
    response = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
    messages = SimpleNamespace(create=lambda **kwargs: response)

    class Client:
        def __init__(self, **kwargs):
            self.messages = messages
            self.beta = SimpleNamespace(messages=messages)

    module = ModuleType("anthropic")
    module.Anthropic = Client
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    module.APIStatusError = type("APIStatusError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_anthropic(monkeypatch):
    stub_anthropic(monkeypatch, "end_turn")
    thunc.configure(backend="anthropic")
    assert thunc.call("ping") == "pong"


@pytest.mark.parametrize(
    "stop_reason, error",
    [
        ("refusal", "declined"),
        ("max_tokens", "cut off"),
        ("model_context_window_exceeded", "cut off"),
        ("pause_turn", "stopped before finishing"),
    ],
)
def test_anthropic_unfinished_answer_is_never_returned(monkeypatch, stop_reason, error):
    stub_anthropic(monkeypatch, stop_reason, text="The first half of the ans")
    thunc.configure(backend="anthropic")
    with pytest.raises(thunc.ThuncError, match=error):
        thunc.call("Summarize.")


def test_openai_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    thunc.configure(backend="openai")
    with pytest.raises(thunc.ThuncError, match=r"thunc\[openai\]"):
        thunc.call("ping")
