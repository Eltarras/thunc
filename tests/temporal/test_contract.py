import subprocess
import sys
from pathlib import Path

import pytest

import thunc
from thunc.temporal import Registry
from thunc.temporal.activities import provider_error
from thunc.temporal.effects import perform, resolve
from thunc.temporal.models import plain
from thunc.temporal.storage import Attention


def test_core_import_does_not_import_temporal():
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.startswith('temporalio'):
            raise ImportError('temporal absent')
sys.meta_path.insert(0,Block())
import thunc
assert 'temporalio' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[2])


def definition(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    agent = thunc.Agent("repo", workdir=root, permissions=["write:note", "run:echo"], backend="codex")

    @agent.task
    def task() -> str:
        """Do a task."""
        ...

    registry = Registry(state_dir=tmp_path / "state")
    registry.agent_task("task", task, version="1", workspace_id="repo")
    item = registry.get({"task": "task", "version": "1", "workspace_id": "repo"})
    state = {"id": "run", "seen": {}, "permissions": agent.permissions.written, "offered": agent.tools()}
    return registry, item, state, root


def test_permissions_and_read_hashes_survive_tool_restart(tmp_path):
    registry, item, state, root = definition(tmp_path)
    (root / "note").write_text("before")
    read = perform(registry.storage, item, state, {"tool": "read", "args": {"path": "note"}}, "run/1/0")
    state["seen"] = read["seen"]
    item.agent.permissions = __import__("thunc.permissions", fromlist=["Permissions"]).Permissions(["!write"])
    result = perform(
        registry.storage,
        item,
        state,
        {"tool": "edit", "args": {"path": "note", "old": "before", "new": "after"}},
        "run/2/0",
    )
    assert result["denied"] and (root / "note").read_text() == "before"


def test_stale_read_hash_cannot_overwrite(tmp_path):
    registry, item, state, root = definition(tmp_path)
    (root / "note").write_text("before")
    read = perform(registry.storage, item, state, {"tool": "read", "args": {"path": "note"}}, "run/1/0")
    state["seen"] = read["seen"]
    (root / "note").write_text("external")
    result = perform(
        registry.storage,
        item,
        state,
        {"tool": "edit", "args": {"path": "note", "old": "before", "new": "after"}},
        "run/2/0",
    )
    assert result["output"].startswith("error:") and (root / "note").read_text() == "external"


def test_command_receipt_blocks_retry_and_resolution_delivery_is_idempotent(tmp_path):
    registry, item, state, root = definition(tmp_path)
    call = {"tool": "run", "args": {"command": "echo must-not-run"}}
    registry.storage.save_effect("run/1/0", "run", "started", {"call": call})
    with pytest.raises(Attention):
        perform(registry.storage, item, state, call, "run/1/0")
    resolve(registry.storage, "repo", "run/1/0", "retry", "verified process has stopped")
    resolve(registry.storage, "repo", "run/1/0", "retry", "verified process has stopped")
    assert registry.storage.effect("run/1/0") is None


def test_note_saved_once_even_if_activity_completion_was_lost(tmp_path):
    registry, item, state, root = definition(tmp_path)
    call = {"tool": "remember", "args": {"note": "one note"}}
    first = perform(registry.storage, item, state, call, "run/1/0")
    second = perform(registry.storage, item, state, call, "run/1/0")
    assert first == second
    assert (Path(item.store_folder) / "memory.md").read_text().count("one note") == 1


def test_transient_provider_errors_keep_classification_without_secret_text():
    class APIConnectionError(Exception):
        pass

    cause = APIConnectionError("secret")
    wrapper = thunc.ThuncError("also secret")
    wrapper.__cause__ = cause
    converted = provider_error(wrapper)
    assert not converted.non_retryable and "secret" not in str(converted)
    assert provider_error(ValueError("invalid")).non_retryable


def test_wire_values_and_ownership_conflicts(tmp_path):
    with pytest.raises(ValueError):
        plain(float("nan"))
    with pytest.raises(ValueError):
        plain({1: "bad"})
    registry, item, state, root = definition(tmp_path)
    registry.claim("first")
    with pytest.raises(ValueError):
        registry.claim("second")
