"""Durable runs on Claude Code's native calls, in segments (thunc/temporal/activities.py).

The segment step is driven here as Temporal would: prepare, then segment, then the same segment again
when an attempt fails (a crash), with a fake `claude` (tests/fake_claude.py) that keeps a session
file and continues it with --resume. The checks are the durable ones: the session is restored, the
call the model asks for again gets the journal entry it had, and no effect happens twice.
"""

import json
import os
import sys

import pytest
from temporalio.exceptions import ApplicationError

import thunc
from thunc.schema import json_schema
from thunc.temporal import Registry
from thunc.temporal.activities import RESUMED, Activities, session_path
from thunc.temporal.effects import resolve

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a script with a shebang")


class WorkerStopped(BaseException):
    """A worker stopping in the middle of a tool call: not an error the tool reports."""


@pytest.fixture
def cli(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    with open(os.path.join(os.path.dirname(__file__), "..", "fake_claude.py"), encoding="utf-8") as f:
        code = f.read()
    exe = bin_dir / "claude"
    exe.write_text(f"#!{sys.executable}\n{code}")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))  # never the real ~/.claude
    log = tmp_path / "claude-log.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    thunc.configure(timeout=20)

    def script(*turns):
        monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", json.dumps(list(turns)))

    script.log = lambda: json.loads(log.read_text())
    return script


def calls(*items):
    return {"calls": [list(item) for item in items]}


@pytest.fixture
def durable(tmp_path):
    """An agent on claude-code with one tool of its own, registered, and its first segment prepared."""
    root = tmp_path / "workspace"
    root.mkdir()
    opened, crash = [], []

    def open_issue(title: str) -> int:
        """Open an issue in the tracker. Returns its number."""
        if crash:
            crash.pop()
            raise WorkerStopped
        opened.append(title)
        return 40 + len(opened)

    agent = thunc.Agent("issues", workdir=root, tools=[open_issue], backend="claude-code")

    @agent.task
    def report() -> int:
        """Open an issue for the flaky test and finish with its number."""
        ...

    registry = Registry(state_dir=tmp_path / "state")
    registry.agent_task("report", report, version="1", workspace_id="issues")
    definition = registry.get({"task": "report", "version": "1", "workspace_id": "issues"})
    request = {"id": "run-1", "task": "report", "version": "1", "workspace_id": "issues", "inputs": {},
               "returns": json_schema(int), "deadline_seconds": 600, "queue": "q"}  # fmt: skip
    activities = Activities(registry)
    progress = activities.prepare(definition, request)

    class Durable:
        pass

    run = Durable()
    run.activities, run.definition, run.request, run.progress = activities, definition, request, progress
    run.storage, run.opened, run.crash, run.root = registry.storage, opened, crash, root
    run.segment = lambda: activities.segment(definition, registry.storage.get(progress["ref"]))
    return run


def test_a_run_on_claude_code_is_one_segment_of_native_calls(cli, durable):
    assert durable.progress["next"] == "segment"
    cli(calls(("open_issue", {"title": "Flaky test"})), calls(("finish", {"value": 41})))
    done = durable.segment()
    assert done["next"] == "finish" and done["result"]["value"] == 41 and durable.opened == ["Flaky test"]
    args = cli.log()["args"]
    assert "--session-id" in args and "--no-session-persistence" not in args and "--resume" not in args
    session = args[args.index("--session-id") + 1]
    assert not session_path(str(durable.root), session).exists()  # removed when the run finished


def test_a_crash_after_a_call_resumes_and_replays_the_call(cli, durable):
    cli(calls(("open_issue", {"title": "Flaky test"})), {"exit": 1})  # the CLI dies after the call's result
    with pytest.raises(ApplicationError, match="Claude Code segment interrupted"):
        durable.segment()
    assert durable.opened == ["Flaky test"]
    cli(calls(("open_issue", {"title": "Flaky test"})), calls(("finish", {"value": 41})))  # asked for again
    done = durable.segment()  # Temporal's retry: the same step
    assert done["next"] == "finish" and done["result"]["value"] == 41
    assert durable.opened == ["Flaky test"]  # the journal's result, not a second issue
    log = cli.log()
    assert "--resume" in log["args"] and log["resumed_lines"] >= 2 and log["messages"] == [RESUMED]


def test_a_call_interrupted_mid_effect_waits_for_resolve(cli, durable):
    durable.crash.append(True)  # the worker stops inside open_issue: its outcome is unknown
    cli(calls(("open_issue", {"title": "Flaky test"})), calls(("finish", {"value": 41})))
    with pytest.raises(WorkerStopped):
        durable.segment()
    cli(calls(("open_issue", {"title": "Flaky test"})), calls(("finish", {"value": 41})))
    waiting = durable.segment()
    assert waiting["next"] == "attention" and waiting["resume"] == "segment"
    assert "'open_issue' may or may not have run" in waiting["reason"] and durable.opened == []
    resolve(durable.storage, "issues", waiting["operation_id"], "retry", "checked the tracker: no issue")
    done = durable.activities.segment(durable.definition, durable.storage.get(waiting["ref"]))
    assert done["next"] == "finish" and durable.opened == ["Flaky test"]


def test_a_different_call_after_a_resume_gets_its_own_journal_entry(cli, durable):
    cli(calls(("open_issue", {"title": "Flaky test"})), {"exit": 1})
    with pytest.raises(ApplicationError):
        durable.segment()
    cli(calls(("open_issue", {"title": "Slow build"})), calls(("finish", {"value": 42})))
    done = durable.segment()
    assert done["result"]["value"] == 42 and durable.opened == ["Flaky test", "Slow build"]


def test_a_claude_code_that_cant_run_thunc_tools_goes_on_with_the_text_protocol(cli, durable, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "old")
    switched = durable.segment()
    assert switched["next"] == "turn"
    state = durable.storage.get(switched["ref"])
    assert state["mode"] == "text" and "reply with exactly one JSON object" in state["fixed"]
    assert "unknown option" in state["startup_error"]


def test_the_session_is_removed_when_a_failed_run_is_recorded(cli, durable):
    cli(calls(("open_issue", {"title": "Flaky test"})), {"exit": 1})
    with pytest.raises(ApplicationError):
        durable.segment()
    pointer = durable.storage.effect("run-1/segment")[1]
    assert os.path.exists(pointer["path"])
    durable.activities._step(
        {"action": "record", "request": durable.request, "result": {"id": "run-1", "status": "failed"}}
    )
    assert not os.path.exists(pointer["path"])
