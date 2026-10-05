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


def test_several_edits_are_one_effect_applied_once(tmp_path):
    registry, item, state, root = definition(tmp_path)
    (root / "note").write_text("a = 1\nb = 2\nb = 2\n")
    read = perform(registry.storage, item, state, {"tool": "read", "args": {"path": "note"}}, "run/1/0")
    state["seen"] = read["seen"]
    edits = [{"old": "a = 1", "new": "a = 9"}, {"old": "b = 2", "new": "b = 3", "replace_all": True}]
    call = {"tool": "edit", "args": {"path": "note", "edits": edits}}
    first = perform(registry.storage, item, state, call, "run/2/0")
    second = perform(registry.storage, item, state, call, "run/2/0")  # the activity's completion was lost
    assert first == second and first["output"] == "edited note (2 edits, 3 replacements)"
    assert (root / "note").read_text() == "a = 9\nb = 3\nb = 3\n"


def test_several_edits_after_a_stale_read_change_nothing(tmp_path):
    registry, item, state, root = definition(tmp_path)
    (root / "note").write_text("a = 1\n")
    read = perform(registry.storage, item, state, {"tool": "read", "args": {"path": "note"}}, "run/1/0")
    state["seen"] = read["seen"]
    (root / "note").write_text("a = 1\nexternal\n")
    call = {"tool": "edit", "args": {"path": "note", "edits": [{"old": "a = 1", "new": "a = 2"}]}}
    result = perform(registry.storage, item, state, call, "run/2/0")
    assert result["output"].startswith("error:") and (root / "note").read_text() == "a = 1\nexternal\n"


def test_an_edit_without_a_read_keeps_the_file_unread_for_a_later_write(tmp_path):
    registry, item, state, root = definition(tmp_path)
    (root / "note").write_text("a = 1\nsecret = 2\n")
    edit = {"tool": "edit", "args": {"path": "note", "old": "a = 1", "new": "a = 9"}}
    result = perform(registry.storage, item, state, edit, "run/1/0")
    assert result["output"] == "edited note" and "note" not in result["seen"]
    state["seen"] = result["seen"]
    write = {"tool": "write", "args": {"path": "note", "content": "overwritten\n"}}
    result = perform(registry.storage, item, state, write, "run/2/0")
    assert result["output"].startswith("error: read 'note' with read before replacing it")
    assert (root / "note").read_text() == "a = 9\nsecret = 2\n"


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
