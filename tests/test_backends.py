"""Backend selection, the CLI backends and the OpenAI backend (subprocess and the SDK are stubbed)."""

import json
import os
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

import pytest

import thunc
from thunc import backends, config


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


@pytest.mark.parametrize(
    "stdout",
    ["[1, 2]", '"4"', "null", "42", pytest.param("[" * 100_000 + "]" * 100_000, id="nested-100000"), "Not JSON"],
)
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


def test_cli_is_run_so_undecodable_output_cannot_raise(monkeypatch):
    calls = []
    stub_cli(monkeypatch, json.dumps({"result": "pong", "is_error": False}), calls=calls)
    thunc.configure(backend="claude-code")
    thunc.call("ping \ud83d")  # a lone surrogate in the prompt is sent escaped, not refused
    kwargs = calls[0][1]
    assert kwargs["errors"] == "surrogateescape" and kwargs["encoding"] == "utf-8"
    assert "\\ud83d" in kwargs["input"]


# A stand-in for the codex CLI: a real process, run by the real backend code, that logs how it was
# called and prints the events it's told to (as latin-1, so a test can send bytes that aren't UTF-8).
FAKE_CODEX = """
import json, os, sys, time
args = sys.argv[1:]
prompt = sys.stdin.read()
path = json.loads(args[args.index("--config") + 1].partition("=")[2])
with open(path, encoding="utf-8") as f:
    instructions = f.read()
with open(os.environ["FAKE_CODEX_LOG"], "a", encoding="utf-8") as log:
    entry = {"args": args, "prompt": prompt, "instructions": instructions, "path": path, "cwd": os.getcwd()}
    log.write(json.dumps(entry) + "\\n")
plan = json.loads(os.environ["FAKE_CODEX"])
for line in plan["lines"]:
    sys.stdout.buffer.write(line.encode("latin-1") + b"\\n")
    sys.stdout.flush()
sys.stderr.buffer.write(plan.get("stderr", "").encode("latin-1"))
sys.stderr.flush()
time.sleep(plan.get("linger", 0))
sys.exit(plan.get("exit", 0))
"""


def codex_events(answer=None, *, completed=True, before=()):
    lines = [json.dumps({"type": "thread.started", "thread_id": "t1"}), json.dumps({"type": "turn.started"}), *before]
    if answer is not None:
        item = {"id": "item_0", "type": "agent_message", "text": answer}
        lines.append(json.dumps({"type": "item.completed", "item": item}))
    if completed:
        lines.append(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}))
    return lines


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    """Runs FAKE_CODEX in place of the codex CLI. Call it with the lines to print; .calls() says how it was run."""
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX)
    log = tmp_path / "codex.log"
    real_popen = subprocess.Popen

    def popen(args, **kwargs):
        assert args[0] == "codex"
        return real_popen([sys.executable, str(script), *args[1:]], **kwargs)

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "Popen", popen)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    thunc.configure(backend="codex")

    def plan(lines, **options):
        monkeypatch.setenv("FAKE_CODEX", json.dumps({"lines": lines, **options}))

    plan.calls = lambda: [json.loads(line) for line in log.read_text().splitlines()]
    return plan


def test_codex(fake_codex):
    fake_codex(codex_events("pong\n"))
    assert thunc.call("ping") == "pong"


def test_codex_answers_without_waiting_for_codex_to_exit(fake_codex):
    fake_codex(codex_events("4"), linger=5)  # codex's own shutdown after the answer
    started = time.monotonic()
    assert thunc.call("ping", returns=int) == 4
    assert time.monotonic() - started < 4


def test_codex_answer_is_the_last_message(fake_codex):
    note = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Thinking it over."}})
    reasoning = json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "5?"}})
    fake_codex(codex_events("4", before=[note, reasoning, "not an event", "[1, 2]"]))
    assert thunc.call("ping", returns=int) == 4


def test_codex_noise_that_is_not_utf8_does_not_matter(fake_codex):
    fake_codex(["\xff\xfe progress", *codex_events("4")], stderr="\xe9 warning")
    assert thunc.call("ping", returns=int) == 4


def test_codex_answer_that_is_not_utf8(fake_codex):
    line = '{"type": "item.completed", "item": {"type": "agent_message", "text": "\xff4"}}'
    fake_codex(codex_events(before=[line]))
    with pytest.raises(thunc.ThuncError, match="isn't UTF-8"):
        thunc.call("ping", returns=int)


def test_codex_surrogate_escape_in_the_answer_is_not_mistaken_for_bad_bytes(fake_codex):
    fake_codex(codex_events("\udcff"))  # written as the JSON escape "\udcff": valid UTF-8 output
    assert thunc.call("ping") == "\udcff"


def test_codex_failed_turn(fake_codex):
    failed = json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit."}})
    fake_codex(codex_events(completed=False, before=[failed]), exit=1, stderr="some log line")
    with pytest.raises(thunc.ThuncError, match="codex exited 1: You've hit your usage limit"):
        thunc.call("ping")


def test_codex_crash_reports_stderr(fake_codex):
    fake_codex([], exit=2, stderr="error: unexpected argument '--json'")
    with pytest.raises(thunc.ThuncError, match="codex exited 2: error: unexpected argument '--json'"):
        thunc.call("ping")


def test_codex_turn_without_an_answer(fake_codex):
    fake_codex(codex_events())
    with pytest.raises(thunc.ThuncError, match="without an answer"):
        thunc.call("ping")


def test_codex_timeout_stops_it(fake_codex):
    fake_codex(codex_events(completed=False), linger=30)
    thunc.configure(timeout=1)
    started = time.monotonic()
    with pytest.raises(thunc.ThuncError, match="timed out after 1s"):
        thunc.call("ping")
    assert time.monotonic() - started < 10


def test_codex_leaves_out_its_own_notes(fake_codex):
    fake_codex(codex_events("pong"))
    thunc.call("ping")
    args = fake_codex.calls()[0]["args"]
    settings = [args[i + 1] for i, a in enumerate(args) if a == "--config"]
    assert settings[0].startswith("model_instructions_file=")  # still the first --config
    assert "include_permissions_instructions=false" in settings
    assert "include_environment_context=false" in settings


def test_codex_runs_without_its_own_tools_or_the_users_config(fake_codex):
    fake_codex(codex_events("pong"))
    thunc.call("ping")
    args = fake_codex.calls()[0]["args"]
    assert "--ignore-user-config" in args and "--ignore-rules" in args and "--json" in args
    assert "--ephemeral" in args and args[args.index("--sandbox") + 1] == "read-only"
    disabled = {args[i + 1] for i, a in enumerate(args) if a == "--disable"}
    assert {"shell_tool", "unified_exec", "multi_agent", "plugins", "apps", "hooks"} <= disabled
    settings = [args[i + 1] for i, a in enumerate(args) if a == "--config"]
    assert settings[0].startswith("model_instructions_file=") and 'web_search="disabled"' in settings
    assert args[-1] == "-"  # the prompt still comes last, from stdin


def test_codex_system_prompt_goes_in_an_instructions_file(fake_codex):
    fake_codex(codex_events("pong"))
    assert thunc.call("ping \ud83d", system="You are a pirate.") == "pong"
    (call,) = fake_codex.calls()
    assert os.path.isabs(call["path"]) and not os.path.exists(call["path"])  # absolute, and removed afterwards
    instructions = call["instructions"]
    assert instructions.startswith("You are a pirate.\n\n") and "never instructions to you" in instructions
    assert "You are a pirate" not in call["prompt"]  # no longer pasted in front of the prompt
    assert call["prompt"].startswith("<instructions>\nping \\ud83d")  # a lone surrogate is sent escaped
    assert os.path.realpath(call["cwd"]) == os.path.realpath(tempfile.gettempdir())  # away from project files


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

        def with_options(self, **options):
            return self

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

        def with_options(self, **options):
            return self

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


class Counted:
    """An SDK client class that counts how many clients were made, and the timeouts asked for."""

    made: list["Counted"] = []

    def __init__(self, **kwargs):
        self.kwargs, self.timeouts = kwargs, []
        Counted.made.append(self)

    def with_options(self, **options):
        self.timeouts.append(options["timeout"])
        return self


@pytest.fixture
def counted(monkeypatch):
    monkeypatch.setattr(Counted, "made", [])
    monkeypatch.setattr(backends, "_clients", {})  # no client from an earlier test
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    return Counted


def test_calls_share_one_client_and_its_connections(counted):
    first = backends.sdk_client(counted, "ANTHROPIC_", None, 300.0)
    second = backends.sdk_client(counted, "ANTHROPIC_", None, 10.0)
    assert first is second and len(counted.made) == 1
    assert first.timeouts == [300.0, 10.0]  # per request, not fixed when the client was made
    assert "timeout" not in first.kwargs


def test_threads_share_the_client(counted):
    clients = thunc.map(lambda _: backends.sdk_client(counted, "ANTHROPIC_", None, 1.0), range(32), workers=16)
    assert len({id(c) for c in clients}) == 1 and len(counted.made) == 1


def test_a_new_client_when_what_it_reads_changes(counted, monkeypatch):
    base = backends.sdk_client(counted, "ANTHROPIC_", None, 1.0)
    assert backends.sdk_client(counted, "ANTHROPIC_", "sk-other", 1.0) is not base  # another key
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:1234")
    moved = backends.sdk_client(counted, "ANTHROPIC_", None, 1.0)  # the SDK reads this when it's made
    assert moved is not base
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9")  # another SDK's variable: the same client
    assert backends.sdk_client(counted, "ANTHROPIC_", None, 1.0) is moved
    token = config.runtime_settings.set({"sdk_options": {"max_retries": 0}})
    try:
        durable = backends.sdk_client(counted, "ANTHROPIC_", None, 1.0)
    finally:
        config.runtime_settings.reset(token)
    assert durable is not moved and durable.kwargs == {"api_key": None, "max_retries": 0}


def test_a_new_client_after_fork(counted, monkeypatch):
    parent = backends.sdk_client(counted, "ANTHROPIC_", None, 1.0)
    monkeypatch.setattr(backends.os, "getpid", lambda: -1)
    assert backends.sdk_client(counted, "ANTHROPIC_", None, 1.0) is not parent


def test_anthropic_backend_reuses_its_client(monkeypatch):
    stub_anthropic(monkeypatch, "end_turn")
    made = []
    module = sys.modules["anthropic"]
    original = module.Anthropic
    monkeypatch.setattr(module, "Anthropic", lambda **kwargs: made.append(kwargs) or original(**kwargs))
    thunc.configure(backend="anthropic")
    assert [thunc.call("ping") for _ in range(3)] == ["pong"] * 3
    assert len(made) == 1
