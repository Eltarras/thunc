"""Agents on the claude-code backend: native tool calls through an MCP server (thunc/claude_code.py).

A fake `claude` (tests/fake_claude.py) starts thunc's real relay and talks MCP to it, so these tests
cover the CLI arguments, the relay, the local connection back to the run, and the agent loop, with
a scripted model instead of a real one.
"""

import json
import os
import sys
from dataclasses import dataclass

import pytest

import thunc

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a script with a shebang")


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Put the fake claude first on PATH. Returns a function that scripts it and reads its log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    source = os.path.join(os.path.dirname(__file__), "fake_claude.py")
    with open(source, encoding="utf-8") as f:
        code = f.read()
    exe = bin_dir / "claude"
    exe.write_text(f"#!{sys.executable}\n{code}")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    log = tmp_path / "claude-log.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    thunc.configure(backend="claude-code", timeout=20)

    class Script:
        def __call__(self, *turns):
            monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", json.dumps(list(turns)))

        @property
        def log(self):
            with open(log, encoding="utf-8") as f:
                return json.load(f)

    return Script()


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.py").write_text("TIMEOUT = 30\n")
    (work / "app.py").write_text("from config import TIMEOUT\n")
    return work


def calls(*items):
    return {"calls": [list(item) for item in items]}


def timeout_task(agent):
    @agent.task
    def timeout() -> int:
        """Find the timeout this app uses, in seconds."""
        ...

    return timeout


def test_a_run_uses_native_calls_through_the_relay(cli, repo):
    cli(
        calls(("search", {"pattern": "TIMEOUT ="})),
        calls(("read", {"path": "config.py"})),
        calls(("finish", {"value": 30})),
    )
    agent = thunc.Agent("guide", workdir=repo, model="claude-sonnet-5-5")
    run = agent.run(timeout_task(agent))
    assert run.value == 30 and run.steps == 3
    log = cli.log
    assert log["results"][0] == {"text": "config.py:1: TIMEOUT = 30", "error": False}
    assert "1  TIMEOUT = 30" in log["results"][1]["text"]
    args = log["args"]
    assert args[args.index("--tools") + 1] == "" and "--strict-mcp-config" in args
    assert args[args.index("--setting-sources") + 1] == ""  # CLAUDE.md in the workdir isn't loaded
    assert args[args.index("--model") + 1] == "claude-sonnet-5-5"
    assert "mcp__thunc__finish" in args and "mcp__thunc__write" not in args  # only the offered tools
    assert os.path.realpath(log["cwd"]) == os.path.realpath(repo)  # the CLI's environment names the workdir
    assert log["system"] == agent.system_prompt() and "call finish with your result" in log["system"]
    assert "Find the timeout" in log["messages"][0]
    assert {t["name"] for t in log["tools"]} == {"list", "read", "search", "remember", "finish"}
    with open(run.session, encoding="utf-8") as f:
        assert json.loads(f.readline())["protocol"] == "mcp"


def test_calls_of_one_model_reply_count_as_one_step(cli, repo):
    cli(calls(("read", {"path": "config.py"}), ("read", {"path": "app.py"})), calls(("finish", {"value": 30})))
    agent = thunc.Agent("guide", workdir=repo)
    run = agent.run(timeout_task(agent))
    assert run.value == 30 and run.steps == 2
    assert len(cli.log["results"]) == 2


def test_calls_of_one_model_reply_sent_one_at_a_time_count_as_one_step(cli, repo):
    reply = calls(("read", {"path": "config.py"}), ("read", {"path": "app.py"}), ("list", {}))
    cli({**reply, "serial": True}, calls(("finish", {"value": 30})))
    agent = thunc.Agent("guide", workdir=repo)
    run = agent.run(timeout_task(agent))
    assert run.value == 30 and run.steps == 2 and len(cli.log["results"]) == 3


def test_permissions_are_checked_in_the_run(cli, repo):
    cli(calls(("write", {"path": "config.py", "content": "TIMEOUT = 1\n"})), calls(("finish", {"value": 30})))
    agent = thunc.Agent("guide", workdir=repo, permissions=["write:docs/**"])
    run = agent.run(timeout_task(agent))
    result = cli.log["results"][0]
    assert result["error"] and "not permitted" in result["text"]
    assert (repo / "config.py").read_text() == "TIMEOUT = 30\n"
    assert run.denied and run.denied[0].tool == "write" and run.denied[0].target == "config.py"


def test_writes_and_commands_are_recorded(cli, repo):
    cli(
        calls(("read", {"path": "config.py"})),
        calls(("edit", {"path": "config.py", "old": "30", "new": "45"})),
        calls(("run", {"command": f"{sys.executable} -c \"print(open('config.py').read())\""})),
        calls(("finish", {"value": 45})),
    )
    agent = thunc.Agent("fixer", workdir=repo, permissions=["write", "run"])
    run = agent.run(timeout_task(agent))
    assert run.value == 45 and run.files_changed == ["config.py"]
    assert run.commands[0].exit_code == 0 and "TIMEOUT = 45" in cli.log["results"][2]["text"]


def test_an_invalid_finish_is_sent_back_to_be_fixed(cli, repo):
    cli(calls(("finish", {"value": "thirty"})), calls(("finish", {"value": 30})))
    agent = thunc.Agent("guide", workdir=repo)
    assert agent.run(timeout_task(agent)).value == 30
    result = cli.log["results"][0]
    assert result["error"] and "that value is invalid" in result["text"]


def test_a_turn_without_a_tool_call_gets_a_nudge(cli, repo):
    cli({"text": "It's 30."}, calls(("finish", {"value": 30})))
    agent = thunc.Agent("guide", workdir=repo)
    assert agent.run(timeout_task(agent)).value == 30
    assert "call finish" in cli.log["messages"][1]


def test_a_turn_that_ends_in_an_error_is_tried_again(cli, repo):
    cli({"error": "API Error: 529 overloaded"}, calls(("finish", {"value": 30})))
    agent = thunc.Agent("guide", workdir=repo)
    assert agent.run(timeout_task(agent)).value == 30
    assert "That step failed (API Error: 529 overloaded)" in cli.log["messages"][1]


def test_repeated_errors_end_the_run(cli, repo):
    cli(*[{"error": f"failure {n}"} for n in range(3)])
    agent = thunc.Agent("guide", workdir=repo)
    with pytest.raises(thunc.AgentError, match="claude error: failure 2"):
        agent.run(timeout_task(agent))


def test_a_cli_that_exits_early_ends_the_run(cli, repo):
    cli({"exit": 3})
    agent = thunc.Agent("guide", workdir=repo)
    with pytest.raises(thunc.AgentError, match="claude exited 3"):
        agent.run(timeout_task(agent))


def test_custom_tools_and_structured_results(cli, repo):
    opened = []

    def open_issue(title: str) -> int:
        """Open an issue and return its number."""
        opened.append(title)
        return 7

    @dataclass
    class Finding:
        file: str
        line: int

    found = [{"file": "config.py", "line": 1}]
    cli(calls(("open_issue", {"title": "Timeout too low"})), calls(("finish", {"value": found})))
    agent = thunc.Agent("reviewer", workdir=repo, tools=[open_issue])
    findings = agent.call("Review the config.", returns=list[Finding])
    assert findings == [Finding("config.py", 1)] and opened == ["Timeout too low"]
    assert cli.log["results"][0] == {"text": "7", "error": False}


def test_protocol_text_keeps_the_json_text_protocol(monkeypatch, cli, repo):
    from thunc import backends

    sent = []

    def text_backend(text, **kwargs):
        sent.append(text)
        return json.dumps({"tool": "finish", "args": {"value": 30}})

    monkeypatch.setitem(backends.BACKENDS, "claude-code", text_backend)
    agent = thunc.Agent("guide", workdir=repo, protocol="text")
    assert agent.run(timeout_task(agent)).value == 30 and len(sent) == 1
