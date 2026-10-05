"""The THUNC_EVENTS stream that thunc-watch reads: one JSON line per call, attempt and agent step."""

import json

import pytest

import thunc
from thunc import events


def act(tool, **args):
    return json.dumps({"tool": tool, "args": args})


@pytest.fixture
def stream(monkeypatch, tmp_path):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("THUNC_EVENTS", str(path))
    monkeypatch.delenv("THUNC_EVENTS_CAPTURE", raising=False)

    def read():
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return read


def test_nothing_is_written_without_thunc_events(fake, monkeypatch, tmp_path):
    monkeypatch.delenv("THUNC_EVENTS", raising=False)
    folder = tmp_path / "program"
    folder.mkdir()
    monkeypatch.chdir(folder)
    fake.replies = ["4"]
    assert thunc.call("Rate it.", {"ticket": "x"}, returns=int) == 4
    assert list(folder.iterdir()) == []


def test_a_call_that_retries_reports_each_attempt(fake, stream):
    @thunc.function(ensure=lambda n: 1 <= n <= 5)
    def urgency(ticket: str) -> int:
        """Rate how urgent this ticket is, from 1 to 5."""
        ...

    fake.replies = ["high", "7", "4"]
    assert urgency("I was charged twice") == 4

    start, *attempts, end = stream()
    assert start["event"] == "call.start" and start["function"].endswith("urgency") and start["backend"] == "fake"
    assert start["inputs"] == {"ticket": "I was charged twice"}
    assert all(e["v"] == 1 and e["id"] == start["id"] and isinstance(e["t"], float) for e in stream())
    assert [(a["event"], a["n"], a["ok"]) for a in attempts] == [
        ("call.attempt", 1, False),
        ("call.attempt", 2, False),
        ("call.attempt", 3, True),
    ]
    assert attempts[0]["reply"] == "high" and "not valid JSON" in attempts[0]["problem"]
    assert "validation check" in attempts[1]["problem"]
    assert end["event"] == "call.end" and end["ok"] and end["attempts"] == 3 and end["value"] == "4"
    assert end["error"] is None and not end["cached"]


def test_a_failed_call_ends_with_its_error(fake, stream):
    fake.replies = ["no", "nope", "still no"]
    with pytest.raises(thunc.ThuncError):
        thunc.call("Count it.", returns=int)
    end = stream()[-1]
    assert end["event"] == "call.end" and not end["ok"] and end["attempts"] == 3
    assert end["value"] is None and "No valid" in end["error"]


def test_a_cached_answer_is_an_end_without_attempts(fake, stream, monkeypatch, tmp_path):
    thunc.configure(cache_dir=str(tmp_path / "cache"))
    fake.replies = ["4"]
    thunc.call("Rate it.", {"ticket": "x"}, returns=int, cache=True)
    thunc.call("Rate it.", {"ticket": "x"}, returns=int, cache=True)
    second = [e for e in stream() if e["id"] == stream()[-1]["id"]]
    assert [e["event"] for e in second] == ["call.start", "call.end"]
    assert second[-1]["cached"] and second[-1]["attempts"] == 0


def test_previews_are_short_unless_capture_is_on(fake, stream, monkeypatch):
    long = "x" * 500
    fake.replies = ["1", "1"]
    thunc.call("Rate it.", {"ticket": long}, returns=int)
    assert len(stream()[0]["inputs"]["ticket"]) < 200

    monkeypatch.setenv("THUNC_EVENTS_CAPTURE", "1")
    thunc.call("Rate it.", {"ticket": long}, returns=int)
    assert stream()[-3]["inputs"]["ticket"] == long


def test_an_agent_run_reports_replies_steps_and_its_end(fake, stream, tmp_path):
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    agent = thunc.Agent("repo-guide", workdir=tmp_path, permissions=["!write"])

    @agent.task
    def timeout() -> int:
        """Find the timeout."""
        ...

    fake.replies = [
        act("search", pattern="TIMEOUT"),
        act("write", path="notes.md", content="hi"),
        act("finish", value=30),
    ]
    assert timeout() == 30

    seen = stream()
    assert [e["event"] for e in seen] == [
        "agent.start",
        "agent.reply",
        "agent.tool",
        "agent.step",
        "agent.reply",
        "agent.tool",
        "agent.step",
        "agent.reply",
        "agent.end",
    ]
    tool = next(e for e in seen if e["event"] == "agent.tool")
    assert (tool["n"], tool["tool"], tool["target"]) == (1, "search", "'TIMEOUT'")
    start, steps, end = seen[0], [e for e in seen if e["event"] == "agent.step"], seen[-1]
    assert start["agent"] == "repo-guide" and start["task"].endswith("timeout") and start["returns"] == "int"
    assert start["session"].endswith(".jsonl") and "repo-guide" in start["session"]
    assert (steps[0]["tool"], steps[0]["target"], steps[0]["denied"]) == ("search", "'TIMEOUT'", False)
    assert "config.py:1" in steps[0]["result"]
    assert (steps[1]["tool"], steps[1]["target"], steps[1]["denied"]) == ("write", "notes.md", True)
    assert end["ok"] and end["steps"] == 3 and end["value"] == "30" and end["files_changed"] == []


def test_a_failed_agent_run_ends_with_its_error(fake, stream, tmp_path):
    agent = thunc.Agent("a", workdir=tmp_path, max_steps=1)
    fake.replies = [act("list"), act("list")]
    with pytest.raises(thunc.AgentError):
        agent.call("List the files.", returns=int)
    end = stream()[-1]
    assert end["event"] == "agent.end" and not end["ok"] and end["error"]


def test_a_broken_events_path_never_breaks_the_program(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("THUNC_EVENTS", str(tmp_path / "missing" / "events.jsonl"))
    fake.replies = ["4"]
    assert thunc.call("Rate it.", returns=int) == 4


def test_targets_name_what_a_tool_acted_on():
    assert events.target("run", {"command": "pytest -q"}) == "pytest -q"
    assert events.target("read", {"path": "a.py", "offset": 3}) == "a.py"
    assert events.target("search", {"pattern": "def main", "path": "src"}) == "'def main' in src"
    assert events.target("list", {}) == ""
