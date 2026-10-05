"""Tool effects guarded by persistent intentions and a volume-local lock."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from temporalio import activity

from thunc import tools
from thunc.agent import _use
from thunc.permissions import Permissions
from thunc.store import NOTE_LIMIT, Store

from .models import digest, encoded
from .registry import Definition
from .storage import Attention, Storage, file_hash


class Restricted(Permissions):
    def __init__(self, original: list[str], current: list[str]) -> None:
        super().__init__([*original, "!read:.thunc-temporal-owner"])
        self.current = Permissions([*current, "!read:.thunc-temporal-owner"])

    def check_read(self, path: str) -> None:
        super().check_read(path)
        self.current.check_read(path)

    def check_write(self, path: str) -> None:
        super().check_write(path)
        self.current.check_write(path)

    def check_run(self, argv: Any) -> None:
        super().check_run(argv)
        self.current.check_run(argv)

    def may(self, kind: str) -> bool:
        return super().may(kind) and self.current.may(kind)


class PlannedWorkdir(tools.Workdir):
    plan: dict[str, Any] | None = None

    def _save(self, full: str, text: str) -> None:
        self.plan = make_plan(Path(full), text)
        self.seen[full] = self.plan["after"]
        if self.show(full) not in self.changed:
            self.changed.append(self.show(full))


def make_plan(path: Path, text: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "before": file_hash(path),
        "after": hashlib.sha256(text.encode()).hexdigest(),
        "content": text,
    }


def perform(
    storage: Storage, definition: Definition, state: dict[str, Any], call: dict[str, Any], operation: str
) -> dict[str, Any]:
    agent = definition.agent
    assert agent is not None
    permissions = Restricted(state["permissions"], agent.permissions.written)
    workdir = PlannedWorkdir(agent.workdir, permissions, env=agent.env, command_timeout=agent.command_timeout)
    if activity.in_activity():
        workdir.cancelled = activity.is_cancelled
    workdir.seen = {workdir.path(k): v for k, v in state["seen"].items()}
    store = Store(agent.name)
    store.folder = str(definition.store_folder)
    tool, args = call["tool"], call["args"]
    with storage.lock(definition.workspace):
        previous = storage.effect(operation)
        if previous:
            status, body = previous
            # A revoked permission must not become a retry that executes an effect.
            if status == "done":
                return cast(dict[str, Any], body["result"])
            if body.get("plan"):
                path = Path(body["plan"]["path"])
                if tool != "remember":
                    if str(path) != workdir.path(str(args.get("path", ""))):
                        raise Attention("File path changed since effect intent")
                    workdir._check(permissions.check_write, str(path))
                elif not permissions.may("memory"):
                    raise Attention("Memory permission revoked with a pending operation")
                return storage.apply_file(operation, state["id"], body)
            started = (body.get("call") or {}).get("tool")
            if started and started != "run" and agent.custom.get(started):
                raise Attention(f"{started!r} may or may not have run; check what it did before resolving")
            raise Attention("Command outcome is uncertain; inspect the process and effects before resolving")
        plan = None
        if tool == "remember" and permissions.may("memory"):
            note = args.get("note")
            if set(args) != {"note"} or not isinstance(note, str):
                output, denied = 'error: remember takes {"note": "..."}', False
            else:
                note = " ".join(note.split())
                if not note or len(note) > NOTE_LIMIT:
                    output, denied = f"error: note must be 1–{NOTE_LIMIT} characters", False
                else:
                    path = Path(store.memory_path)
                    old = path.read_text() if path.exists() else ""
                    plan = make_plan(
                        path,
                        old
                        + ("\n" if old and not old.endswith("\n") else "")
                        + f"- {time.strftime('%Y-%m-%d')}: {note}\n",
                    )
                    output, denied = "saved. Later runs of this agent will see this note in their memory.", False
        else:
            offered = [name for name in state["offered"] if name != "remember" or permissions.may("memory")]
            custom = agent.custom.get(tool) if tool in offered else None
            if (tool == "run" and tool in offered) or (custom and tool not in definition.retry_safe):
                # Intent is committed before the effect starts, so a crash is conservatively ambiguous:
                # a command, or one of the agent's own tools, that may have run isn't run again unasked.
                storage.save_effect(operation, state["id"], "started", {"call": call})
            if custom is not None:
                output, denied = custom.call(args), False  # an exception in it is an error result
            else:
                output, denied = _use(tool, args, workdir, store, offered)
            plan = workdir.plan
        result = {
            "output": output,
            "denied": denied,
            "seen": {workdir.show(k): v for k, v in workdir.seen.items()},
            "changed": workdir.changed,
            "commands": [asdict(c) for c in workdir.commands],
        }
        body = {"call": call, "result": result, "plan": plan}
        if plan:
            storage.save_effect(operation, state["id"], "prepared", body)
            return storage.apply_file(operation, state["id"], body)
        storage.save_effect(operation, state["id"], "done", body)
        return result


def resolve(storage: Storage, workspace: str, operation: str, decision: str, evidence: str, output: str = "") -> None:
    if decision not in {"complete", "abort", "retry"} or not evidence.strip():
        raise ValueError("Resolution needs complete/abort/retry and operator evidence")
    with storage.lock(workspace):
        receipt = {"operation": operation, "decision": decision, "evidence": evidence, "output": output}
        receipt_id = digest(receipt)
        with storage.db() as db:
            if db.execute("SELECT 1 FROM resolutions WHERE id=?", (receipt_id,)).fetchone():
                return
        row = storage.effect(operation)
        if not row:
            raise ValueError("Unknown operation")
        status, body = row
        if status == "done":
            if body.get("resolution") == {"decision": decision, "evidence": evidence, "output": output}:
                return
            raise ValueError("Operation is already completed")
        if decision == "retry" and body.get("plan"):
            raise ValueError("File conflict: restore the expected content before retrying the run")
        body["resolution"] = {"decision": decision, "evidence": evidence, "output": output}
        if decision == "retry":
            # Keep audit evidence in a separate immutable artifact before resetting this attempt.
            storage.put({"operation": operation, **body})
            with storage.db() as db:
                db.execute("DELETE FROM effects WHERE id=?", (operation,))
                db.execute("INSERT INTO resolutions VALUES (?,?)", (receipt_id, encoded(receipt).decode()))
            return
        body["result"] = body.get("result", {"seen": {}, "changed": [], "commands": [], "denied": False})
        body["result"]["output"] = output if decision == "complete" else "error: operator aborted this effect"
        # Run identity is encoded before the final /turn/tool suffix.
        storage.save_effect(operation, operation.split("/")[0], "done", body)
        with storage.db() as db:
            db.execute("INSERT OR IGNORE INTO resolutions VALUES (?,?)", (receipt_id, encoded(receipt).decode()))
