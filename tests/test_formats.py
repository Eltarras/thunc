"""The formats other programs read: an agent run's record (sessions/*.jsonl) and the THUNC_EVENTS
stream. Each line's fields are checked against the lists below, which match the documentation in
thunc/store.py, thunc/events.py and docs/docs/api.html#formats. A change here is a change to a format:
adding an event or a field is fine within a format; removing or redefining one needs a new format
number (see store.py)."""

import importlib
import json
from pathlib import Path

import pytest

import thunc
from thunc import events
from thunc.errors import TransientError
from thunc.store import RECORD_FORMAT

agent_module = importlib.import_module("thunc.agent")  # thunc.agent is the @thunc.agent decorator

# event -> (fields always there, fields sometimes there)
RECORD = {
    "start": (
        {
            "time",
            "event",
            "format",
            "task",
            "instructions",
            "inputs",
            "returns",
            "memory_characters",
            "followed",
            "protocol",
        },
        {"settings_changed_from"},
    ),  # fmt: skip
    "fallback": ({"time", "event", "protocol", "reason"}, set()),
    "retry": ({"time", "event", "attempt", "error"}, set()),
    "step": ({"time", "event", "n", "result"}, {"tool", "args", "denied", "reply"}),
    "finish": ({"time", "event", "n", "value", "files_changed"}, set()),
    "error": ({"time", "event", "error", "files_changed"}, set()),
}

EVENTS = {
    "call.start": {"id", "function", "backend", "model", "inputs"},
    "call.attempt": ({"id", "n", "seconds", "ok", "reply"}, {"problem"}),
    "call.end": {"id", "ok", "cached", "replayed", "attempts", "seconds", "value", "error"},
    "agent.start": {"id", "agent", "task", "returns", "session", "replayed", "inputs"},
    "agent.reply": {"id", "n", "seconds"},
    "agent.tool": {"id", "n", "tool", "target"},
    "agent.step": {"id", "n", "tool", "target", "result", "seconds", "denied"},
    "agent.end": {"id", "ok", "steps", "seconds", "files_changed", "value", "error"},
}


def act(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def check(line, spec, kind):
    required, optional = spec if isinstance(spec, tuple) else (spec, set())
    keys = set(line)
    assert required <= keys, f"{kind} is missing {required - keys}: {line}"
    assert keys <= required | optional, f"{kind} has undocumented fields {keys - required - optional}: {line}"


@pytest.fixture
def stream(monkeypatch, tmp_path):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("THUNC_EVENTS", str(path))
    monkeypatch.delenv("THUNC_EVENTS_CAPTURE", raising=False)
    return lambda: [json.loads(line) for line in path.read_text().splitlines()]


def runs(fake, tmp_path, monkeypatch):
    """Runs that between them write every kind of record line and event."""
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    agent = thunc.Agent("formats", workdir=tmp_path, permissions=["write:notes.md"], protocol="text", max_steps=6)

    @agent.task
    def timeout() -> int:
        """Find the timeout."""
        ...

    flaky = iter([TransientError("overloaded")])
    original = fake.__call__

    def sometimes_down(text, **kwargs):
        error = next(flaky, None)
        if error is not None:
            raise error
        return original(text, **kwargs)

    monkeypatch.setattr(agent_module, "RETRY_DELAY", 0)
    monkeypatch.setitem(thunc.backends.BACKENDS, "fake", sometimes_down)
    fake.replies = [
        "no JSON here",  # a reply with no usable call
        act("read", path="config.py"),
        act("write", path="config.py", content="x"),  # denied
        act("finish", value=30),
    ]
    assert agent.run(timeout).value == 30
    fake.replies = [act("read", path="config.py")] * 6
    with pytest.raises(thunc.AgentError):
        timeout()  # runs out of steps: an error line
    fake.replies = ["high", "4"]
    assert thunc.call("Rate it.", {"n": 1}, returns=int) == 4
    return agent


def test_every_record_line_has_its_documented_fields(fake, tmp_path, monkeypatch):
    agent = runs(fake, tmp_path, monkeypatch)
    seen = set()
    for path in sorted((Path(agent.folder) / "sessions").iterdir()):
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert lines[0]["event"] == "start" and lines[0]["format"] == RECORD_FORMAT == 1
        for line in lines:
            assert line["event"] in RECORD, f"undocumented record line {line['event']!r}"
            check(line, RECORD[line["event"]], line["event"])
            seen.add(line["event"])
    assert seen >= {"start", "retry", "step", "finish", "error"}  # fallback needs a CLI: see test_claude_code_agent


def test_every_event_has_its_documented_fields(fake, tmp_path, monkeypatch, stream):
    runs(fake, tmp_path, monkeypatch)
    seen = set()
    for line in stream():
        assert line["v"] == events.FORMAT == 1
        kind = line["event"]
        assert kind in EVENTS, f"undocumented event {kind!r}"
        check({k: v for k, v in line.items() if k not in ("v", "event", "t", "pid")}, EVENTS[kind], kind)
        seen.add(kind)
    assert seen == set(EVENTS)


def test_the_documentation_names_every_line_and_event():
    import thunc.events
    import thunc.store

    api = (Path(__file__).parents[1] / "docs" / "docs" / "api.html").read_text()
    for kind, spec in RECORD.items():
        required, optional = spec
        for field in (required | optional) - {"time", "event"}:
            assert field in thunc.store.__doc__, f"store.py doesn't document {kind}.{field}"
        assert f"<code>{kind}</code>" in api, f"api.html doesn't document the record's {kind} line"
    for kind, spec in EVENTS.items():
        fields = spec[0] | spec[1] if isinstance(spec, tuple) else spec
        line = next(line for line in thunc.events.__doc__.splitlines() if line.strip().startswith(kind + " "))
        for field in fields:
            assert field in line, f"events.py doesn't document {kind}.{field}"
        assert f"<code>{kind}</code>" in api, f"api.html doesn't document the {kind} event"
