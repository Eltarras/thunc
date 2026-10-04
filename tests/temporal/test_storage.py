import pytest

from thunc.temporal.effects import make_plan
from thunc.temporal.storage import Attention, Storage, atomic


def test_effect_reconciles_crash_after_replace_without_rewriting(tmp_path, monkeypatch):
    store = Storage(tmp_path / "state")
    path = tmp_path / "file"
    path.write_text("before")
    body = {"plan": make_plan(path, "after"), "result": {"output": "edited"}}
    store.save_effect("run/1/0", "run", "prepared", body)
    atomic(path, b"after")  # worker died before recording completion

    def no_write(*args):
        raise AssertionError("completed effect must not run again")

    monkeypatch.setattr("thunc.temporal.storage.atomic", no_write)
    assert store.apply_file("run/1/0", "run", body) == {"output": "edited"}
    assert store.unresolved("run") == []


def test_conflict_never_overwrites_external_edit(tmp_path):
    store = Storage(tmp_path / "state")
    path = tmp_path / "file"
    path.write_text("before")
    body = {"plan": make_plan(path, "after"), "result": {}}
    path.write_text("human edit")
    with pytest.raises(Attention):
        store.apply_file("run/1/0", "run", body)
    assert path.read_text() == "human edit"


def test_retry_before_write_and_artifact_integrity(tmp_path):
    store = Storage(tmp_path / "state")
    path = tmp_path / "file"
    body = {"plan": make_plan(path, "after"), "result": {}}
    store.save_effect("run/1/0", "run", "prepared", body)
    store.apply_file("run/1/0", "run", body)
    assert path.read_text() == "after"
    ref = store.put({"opaque": "signature", "large": "x" * 300_000})
    assert store.get(ref)["opaque"] == "signature"
    (store.root / "artifacts" / ref["sha256"]).write_text("corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        store.get(ref)
    with pytest.raises(ValueError):
        store.get({"sha256": "../secret"})


def test_submission_identity_is_permanent_and_payload_checked(tmp_path):
    store = Storage(tmp_path)
    request = {"id": "run", "inputs": {"x": 1}, "queue": "v1"}
    assert store.admit(request)
    assert not store.admit({**request, "queue": "v2"})
    with pytest.raises(ValueError, match="different"):
        store.admit({**request, "inputs": {"x": 2}})
    store.finish("run", {"value": 42})
    assert store.result("run") == {"value": 42}
