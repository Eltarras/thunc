"""Typed, reconnectable client API. Client disconnect never cancels a run."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Generic, TypeVar, cast

from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.service import RPCError, RPCStatusCode

from thunc.schema import json_schema, validate

from .models import INLINE_LIMIT, DurableError, DurableRun, digest, encoded, identifiers, plain

T = TypeVar("T")


class Runtime:
    def __init__(self, client: Client, *, task_queue: str) -> None:
        self.client, self.task_queue = client, task_queue

    @classmethod
    async def connect(cls, address: str, *, task_queue: str, namespace: str = "default", **options: Any) -> Runtime:
        return cls(await Client.connect(address, namespace=namespace, **options), task_queue=task_queue)

    async def start(
        self,
        task: str,
        *,
        version: str,
        workspace_id: str,
        inputs: dict[str, Any],
        returns: type[T],
        request_id: str,
        deadline_seconds: float = 1800,
    ) -> Handle[T]:
        if not all(isinstance(x, str) and 0 < len(x) <= 200 for x in (task, version, workspace_id, request_id)):
            raise ValueError("Task/version/workspace/request IDs must be 1–200 characters")
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("Deadline must be finite and positive")
        coordinator, identity = identifiers(workspace_id, task, version, request_id)
        request = {
            "id": identity,
            "task": task,
            "version": version,
            "workspace_id": workspace_id,
            "inputs": plain(inputs),
            "returns": json_schema(returns),
            "queue": self.task_queue,
            "deadline_seconds": deadline_seconds,
        }
        if len(encoded(request)) > INLINE_LIMIT:
            raise ValueError("Submission exceeds the 256 KiB inline budget")
        await self._submit(request, coordinator)
        return Handle(self, identity, returns, request)

    async def _submit(self, request: dict[str, Any], coordinator: str) -> None:
        owner = await self.client.start_workflow(
            "thunc.workspace.v1",
            {"workspace_id": request["workspace_id"]},
            id=coordinator,
            task_queue=self.task_queue,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
            id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
        )
        await owner.execute_update("submit", request)

    def get(self, identity: str, *, returns: type[T]) -> Handle[T]:
        return Handle(self, identity, returns)


class Handle(Generic[T]):
    def __init__(
        self, runtime: Runtime, identity: str, returns: type[T], request: dict[str, Any] | None = None
    ) -> None:
        parts = identity.split("-")
        if len(parts) != 4 or parts[:2] != ["thunc", "run"] or len(parts[2]) != 32 or len(parts[3]) != 64:
            raise ValueError("Invalid durable run ID")
        self.runtime, self.id, self.returns, self.request = runtime, identity, returns, request
        self.owner = runtime.client.get_workflow_handle("thunc-workspace-" + parts[2])
        self.child = runtime.client.get_workflow_handle(identity)

    async def status(self) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], await self.child.query("status"))
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise
            return cast(dict[str, Any], await self.owner.query("status", self.id))

    async def result(self) -> DurableRun[T]:
        # Wait for the child to be started without writing polling events to history.
        while True:
            try:
                record = await self.child.result()
                break
            except RPCError as exc:
                if exc.status != RPCStatusCode.NOT_FOUND:
                    raise
                status = await self.owner.query("status", self.id)
                if status["status"] == "recorded":
                    if self.request is None:
                        raise DurableError(
                            "History expired; reattach with the original start request to read its receipt",
                            {"id": self.id, "status": "expired"},
                        ) from exc
                    record = await self.owner.execute_update("fetch", self.request)
                    break
                await asyncio.sleep(0.25)
        if record["status"] != "completed":
            raise DurableError(record.get("error", record["status"]), record)
        fields = {k: v for k, v in record.items() if k not in {"value", "error"}}
        return DurableRun(value=validate(record["value"], self.returns), **fields)

    async def cancel(self) -> None:
        await self.owner.execute_update("cancel", self.id)

    async def resolve(self, operation_id: str, decision: str, evidence: str, *, output: str = "") -> None:
        resolution = {"operation_id": operation_id, "decision": decision, "evidence": evidence, "output": output}
        resolution["resolution_id"] = digest(resolution)
        if not operation_id.startswith(self.id + "/"):
            raise ValueError("Operation belongs to another run")
        try:
            await self.child.execute_update("resolve", resolution)
        except RPCError as exc:
            if exc.status not in {RPCStatusCode.NOT_FOUND, RPCStatusCode.FAILED_PRECONDITION}:
                raise
            await self.owner.execute_update("resolve", resolution)
