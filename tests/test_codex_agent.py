"""Agents on the codex backend: native tool calls through an MCP server (thunc/codex.py).

A fake `codex` (tests/fake_codex_mcp.py) starts thunc's real relay and talks MCP to it, so these tests
cover the CLI arguments, the relay and its connection back to the run, turns continued with
`codex exec resume`, and the agent loop, with a scripted model instead of a real one.
"""

import json
import os
import sys
import warnings

import pytest

import thunc

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="the fake codex is a script with a shebang"),
    pytest.mark.filterwarnings("error::ResourceWarning"),  # every pipe, socket and file the run opened is closed
]


@pytest.fixture(autouse=True)
def native_calls_not_ruled_out(monkeypatch):
    from thunc import relay

    monkeypatch.setattr(relay, "unavailable", {})  # what one test learns doesn't leak into the next


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Put the fake codex first on PATH. Returns a function that scripts it; .log() is what it saw."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    with open(os.path.join(os.path.dirname(__file__), "fake_codex_mcp.py"), encoding="utf-8") as f:
        code = f.read()
    exe = bin_dir / "codex"
    exe.write_text(f"#!{sys.executable}\n{code}")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    log = tmp_path / "codex-log.jsonl"
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    thunc.configure(backend="codex", timeout=20)

    def script(*turns):
        monkeypatch.setenv("FAKE_CODEX_SCRIPT", json.dumps(list(turns)))

    def entries():
        with open(log, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    script.log = entries
    script.runs = lambda: [e for e in entries() if "args" in e]
    script.results = lambda: [e["result"] for e in entries() if "result" in e]
    script.deleted = lambda: [e["delete"] for e in entries() if "delete" in e]
    return script


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.py").write_text("TIMEOUT = 30\n")
    return work


def calls(*items):
    return {"calls": [list(item) for item in items]}


def timeout_task(agent):
    @agent.task
    def timeout() -> int:
        """Find the timeout this app uses, in seconds."""
        ...

    return timeout


def config(args):
    return {args[i + 1].partition("=")[0]: args[i + 1].partition("=")[2] for i, a in enumerate(args) if a == "--config"}


def test_a_run_uses_native_calls_through_the_relay(cli, repo):
    cli([calls(("read", {"path": "config.py"})), calls(("finish", {"value": 30}))])
    agent = thunc.Agent("guide", workdir=repo, model="gpt-5.5", command_timeout=200)
    run = agent.run(timeout_task(agent))
    assert run.value == 30 and run.steps == 2
    assert "1  TIMEOUT = 30" in cli.results()[0]["text"]
    (seen,) = cli.runs()
    args, settings = seen["args"], config(seen["args"])
    assert args[0] == "exec" and args[1] != "resume"
    assert settings["sandbox_mode"] == '"read-only"'  # thunc's tools change files, not Codex's
    assert settings["mcp_servers.thunc.default_tools_approval_mode"] == '"approve"'  # only thunc's server
    assert settings["mcp_servers.thunc.tool_timeout_sec"] == "260"  # the command timeout, and a margin
    assert "--ignore-user-config" in args and args[args.index("--model") + 1] == "gpt-5.5" and args[-1] == "-"
    assert {"shell_tool", "unified_exec"} <= {args[i + 1] for i, a in enumerate(args) if a == "--disable"}
    assert os.path.realpath(seen["cwd"]) == os.path.realpath(repo)  # Codex's environment names the workdir
    assert seen["system"] == agent.system_prompt() and "Find the timeout" in seen["prompt"]
    assert {t["name"] for t in seen["tools"]} == {"list", "read", "search", "remember", "finish"}
    assert cli.deleted() == [["--force", "019a-fake-thread"]]  # its session isn't left behind
    with open(run.session, encoding="utf-8") as f:
        assert json.loads(f.readline())["protocol"] == "mcp"


def test_a_turn_without_finish_is_continued_with_a_nudge(cli, repo):
    cli([{"text": "The timeout is 30."}], [calls(("finish", {"value": 30}))])
    agent = thunc.Agent("guide", workdir=repo)
    assert agent.run(timeout_task(agent)).value == 30
    first, second = cli.runs()
    assert first["resume"] is None and second["resume"] == "019a-fake-thread"
    assert second["args"][:3] == ["exec", "resume", "019a-fake-thread"]
    assert second["prompt"] == thunc.native.NUDGE
    assert config(second["args"])["sandbox_mode"] == '"read-only"'  # resume takes no --sandbox


def test_a_failed_turn_gets_another_chance(cli, repo):
    cli([{"fail": "stream disconnected"}], [calls(("finish", {"value": 30}))])
    agent = thunc.Agent("guide", workdir=repo)
    assert agent.run(timeout_task(agent)).value == 30
    assert cli.runs()[1]["prompt"].startswith("That step failed (stream disconnected).")


def test_repeated_failed_turns_end_the_run(cli, repo):
    cli(*[[{"fail": "stream disconnected"}]] * 3)
    agent = thunc.Agent("guide", workdir=repo)
    with pytest.raises(thunc.AgentError, match="codex error: stream disconnected"):
        agent.run(timeout_task(agent))
    assert cli.deleted()  # the session is deleted when the run fails too


@pytest.mark.parametrize(
    "mode, reason", [("old", "unexpected argument"), ("no-mcp", "ended before thunc's tool server connected")]
)
def test_a_codex_that_cant_run_thunc_tools_falls_back_to_the_text_protocol(monkeypatch, cli, repo, mode, reason):
    from thunc import backends

    sent = []

    def text_backend(text, **kwargs):
        sent.append(kwargs["system"])
        return json.dumps({"tool": "finish", "args": {"value": 30}})

    monkeypatch.setitem(backends.BACKENDS, "codex", text_backend)
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    agent = thunc.Agent("guide", workdir=repo)
    with pytest.warns(RuntimeWarning, match="Codex couldn't run thunc's tools as native calls"):
        run = agent.run(timeout_task(agent))
    assert run.value == 30 and len(sent) == 1 and "reply with exactly one JSON object" in sent[0]
    with open(run.session, encoding="utf-8") as f:
        fallback = [e for e in map(json.loads, f) if e["event"] == "fallback"]
    assert fallback and reason in fallback[0]["reason"]

    # The next run in this process goes straight to the text protocol: no CLI start, no new warning.
    from thunc import codex

    monkeypatch.setattr(codex.CodexConversation, "_popen", lambda self, args: pytest.fail("codex was started again"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert agent.run(timeout_task(agent)).value == 30 and len(sent) == 2


def test_protocol_native_does_not_fall_back(monkeypatch, cli, repo):
    from thunc import backends

    monkeypatch.setitem(backends.BACKENDS, "codex", lambda text, **kwargs: pytest.fail("no fallback"))
    monkeypatch.setenv("FAKE_CODEX_MODE", "old")
    agent = thunc.Agent("guide", workdir=repo, protocol="native")
    with pytest.raises(thunc.AgentError, match="codex exited 2 before the task was done: error: unexpected argument"):
        agent.run(timeout_task(agent))


def test_failures_after_the_tools_started_do_not_fall_back(monkeypatch, cli, repo):
    from thunc import backends

    monkeypatch.setitem(backends.BACKENDS, "codex", lambda text, **kwargs: pytest.fail("no fallback"))
    cli([calls(("read", {"path": "config.py"})), {"exit": 3}])
    agent = thunc.Agent("guide", workdir=repo)
    with pytest.raises(thunc.AgentError, match="codex exited 3"):
        agent.run(timeout_task(agent))


def test_permissions_are_checked_in_the_run(cli, repo):
    cli([calls(("write", {"path": "config.py", "content": "x"})), calls(("finish", {"value": 30}))])
    agent = thunc.Agent("guide", workdir=repo, permissions=["write:docs/**"])
    run = agent.run(timeout_task(agent))
    assert "not permitted" in cli.results()[0]["text"] and cli.results()[0]["error"] is True
    assert (repo / "config.py").read_text() == "TIMEOUT = 30\n" and run.denied


def test_effort_is_passed_to_codex(cli, repo):
    cli([calls(("finish", {"value": 30}))])
    agent = thunc.Agent("guide", workdir=repo, effort="xhigh")
    agent.run(timeout_task(agent))
    assert config(cli.runs()[0]["args"])["model_reasoning_effort"] == '"xhigh"'
    with pytest.raises(thunc.ThuncError, match="codex backend takes effort low, medium, high or xhigh"):
        timeout_task(thunc.Agent("max", workdir=repo, effort="max"))()


def test_protocol_text_keeps_the_json_text_protocol(monkeypatch, cli, repo):
    from thunc import backends

    sent = []
    monkeypatch.setitem(
        backends.BACKENDS, "codex", lambda text, **kw: sent.append(text) or '{"tool": "finish", "args": {"value": 30}}'
    )
    agent = thunc.Agent("guide", workdir=repo, protocol="text")
    assert agent.run(timeout_task(agent)).value == 30 and len(sent) == 1
