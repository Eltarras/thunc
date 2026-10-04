"""Deterministic orchestration only. No filesystem, provider SDKs or user callbacks."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, ChildWorkflowError


async def step(request: dict[str, Any], action: str, *, attempt_seconds: float = 150, **data: Any) -> dict[str, Any]:
    result: dict[str, Any] = await workflow.execute_activity(
        "thunc.step.v1",
        {"request": request, "action": action, **data},
        task_queue=request["queue"],
        start_to_close_timeout=timedelta(seconds=attempt_seconds),
        schedule_to_close_timeout=timedelta(seconds=attempt_seconds * 3 + 60),
        heartbeat_timeout=timedelta(seconds=15),
        retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=3,
        ),
        cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
    )
    return result


@workflow.defn(name="thunc.agent.v1")
class AgentWorkflow:
    @workflow.init
    def __init__(self, data: dict[str, Any]) -> None:
        self.state: dict[str, Any] = {"status": "queued"}
        self.request: dict[str, Any] = data["request"]
        self.cancelled = data.get("cancelled", False) or self.request.get("cancelled", False)
        self.resolving = False
        self.resolved = False
        self.resolutions: dict[str, dict[str, Any]] = {}
        self.active: asyncio.Task[dict[str, Any]] | None = None

    @workflow.query
    def status(self) -> dict[str, Any]:
        return self.state

    @workflow.update
    async def cancel(self) -> None:
        self.request_cancel()

    @workflow.signal
    def request_cancel(self) -> None:
        self.cancelled = True
        if self.active:
            self.active.cancel()

    @workflow.update
    async def resolve(self, resolution: dict[str, Any]) -> None:
        operation = str(resolution.get("operation_id", ""))
        resolution_id = str(resolution.get("resolution_id", ""))
        if resolution_id in self.resolutions:
            if self.resolutions[resolution_id] != resolution:
                raise ApplicationError("Conflicting resolution", non_retryable=True)
            return
        if self.state.get("operation_id") != operation or self.state["status"] != "needs_attention" or self.resolving:
            raise ApplicationError("Operation is not awaiting resolution", non_retryable=True)
        self.resolving = True
        try:
            await step(self.request, "resolve", resolution=resolution)
            self.resolutions[resolution_id] = resolution
            if resolution["decision"] == "abort":
                self.cancelled = True
            self.resolved = True
        finally:
            self.resolving = False

    async def attention(self, operation: str, reason: str) -> None:
        self.resolved = False
        self.state = {"status": "needs_attention", "operation_id": operation, "reason": reason}
        await workflow.wait_condition(lambda: self.resolved and not self.resolving)

    @workflow.run
    async def run(self, data: dict[str, Any]) -> dict[str, Any]:
        self.request = data["request"]
        deadline = data.get(
            "deadline", self.request.get("deadline_at", workflow.time() + self.request["deadline_seconds"])
        )
        progress: dict[str, Any] = data.get("progress") or {}
        self.resolutions = data.get("resolutions", {})
        result: dict[str, Any] | None = None
        turns_this_history = 0
        try:
            if not progress:
                progress = await step(self.request, "prepare")
            while True:
                if self.cancelled or workflow.time() >= deadline:
                    result = {"id": self.request["id"], "status": "cancelled" if self.cancelled else "timed_out"}
                    break
                if progress["next"] == "finish":
                    result = progress["result"]
                    break
                if progress["next"] == "attention":
                    await self.attention(progress["operation_id"], progress["reason"])
                    progress["next"] = "tool"
                    continue
                self.state = {"status": "running", "steps": progress.get("turns", 0)}
                self.active = asyncio.create_task(
                    step(
                        self.request,
                        progress["next"],
                        ref=progress["ref"],
                        attempt_seconds=progress.get("attempt_seconds", 150),
                    )
                )
                try:
                    progress = await asyncio.wait_for(self.active, timeout=max(0.001, deadline - workflow.time()))
                except (ActivityError, asyncio.TimeoutError):
                    pending = (await step(self.request, "reconcile"))["pending"]
                    if pending:
                        await self.attention(pending[0], "Activity failed with an unresolved effect")
                        continue
                    if self.cancelled:  # the step ended because we cancelled it; report a cancellation
                        continue
                    raise
                finally:
                    self.active = None
                turns_this_history += 1
                if (turns_this_history >= 100 or workflow.info().is_continue_as_new_suggested()) and not self.cancelled:
                    await workflow.wait_condition(workflow.all_handlers_finished)
                    workflow.continue_as_new(
                        {
                            "request": self.request,
                            "deadline": deadline,
                            "progress": progress,
                            "resolutions": self.resolutions,
                        }
                    )
        except asyncio.CancelledError:
            self.cancelled = True
            result = {"id": self.request["id"], "status": "cancelled"}
        except (ActivityError, asyncio.TimeoutError):
            result = {
                "id": self.request["id"],
                "status": "failed",
                "error": "Durable step failed; inspect Activity history",
            }
        # Never release a lane while an effect might still be executing.
        pending = (await step(self.request, "reconcile"))["pending"]
        for operation in pending:
            await self.attention(operation, "Reconcile pending effects before releasing this workspace")
        assert result is not None
        await workflow.wait_condition(workflow.all_handlers_finished)
        self.state = {"status": result["status"], "steps": result.get("steps", 0)}
        return await step(self.request, "record", result=result)


@workflow.defn(name="thunc.workspace.v1")
class WorkspaceWorkflow:
    @workflow.init
    def __init__(self, state: dict[str, Any]) -> None:
        self.queue: list[dict[str, Any]] = state.get("queue", [])
        self.workspace = state["workspace_id"]
        self.pending: list[str] = []
        self.resolving = False
        self.active_request: dict[str, Any] | None = None
        self.child: workflow.ChildWorkflowHandle[Any, Any] | None = None

    @workflow.update
    async def submit(self, request: dict[str, Any]) -> str:
        if request["workspace_id"] != self.workspace or len(self.queue) >= 100:
            raise ApplicationError("Workspace mismatch or queue full", non_retryable=True)
        admission = await step(request, "admit")
        if not admission["complete"] and not any(r["id"] == request["id"] for r in self.queue):
            if len(self.queue) >= 100:
                raise ApplicationError("Workspace queue full; retry submission later", non_retryable=True)
            self.queue.append({**request, "deadline_at": workflow.time() + request["deadline_seconds"]})
        return str(request["id"])

    @workflow.update
    async def cancel(self, identity: str) -> None:
        for request in self.queue:
            if request["id"] == identity:
                request["cancelled"] = True
                if self.active_request is request and self.child is not None:
                    await self.child.signal("request_cancel")
                return

    @workflow.query
    def status(self, identity: str) -> dict[str, Any]:
        if self.active_request and identity == self.active_request["id"] and self.pending:
            return {"status": "needs_attention", "operation_id": self.pending[0]}
        return {"status": "queued" if any(r["id"] == identity for r in self.queue) else "recorded"}

    @workflow.update
    async def fetch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["workspace_id"] != self.workspace:
            raise ApplicationError("Workspace mismatch", non_retryable=True)
        return await step(request, "result")

    @workflow.update
    async def resolve(self, resolution: dict[str, Any]) -> None:
        if not self.active_request or resolution["operation_id"] not in self.pending or self.resolving:
            raise ApplicationError("No matching unresolved effect", non_retryable=True)
        self.resolving = True
        try:
            await step(self.active_request, "resolve", resolution=resolution)
            self.pending.remove(resolution["operation_id"])
        finally:
            self.resolving = False

    @workflow.run
    async def run(self, state: dict[str, Any]) -> None:
        count = 0
        while True:
            await workflow.wait_condition(lambda: bool(self.queue))
            request = self.queue[0]
            self.active_request = request
            try:
                self.child = await workflow.start_child_workflow(
                    "thunc.agent.v1",
                    {"request": request},
                    id=request["id"],
                    task_queue=request["queue"],
                    parent_close_policy=workflow.ParentClosePolicy.ABANDON,
                )
                if request.get("cancelled"):
                    await self.child.signal("request_cancel")
                await self.child
            except ChildWorkflowError:
                self.pending = (await step(request, "reconcile"))["pending"]
                await workflow.wait_condition(lambda: not self.pending and not self.resolving)
                await step(
                    request,
                    "record",
                    result={"id": request["id"], "status": "failed", "error": "Child execution failed or terminated"},
                )
            self.queue.pop(0)
            self.active_request = None
            self.child = None
            count += 1
            if count >= 50 or workflow.info().is_continue_as_new_suggested():
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new({"workspace_id": self.workspace, "queue": self.queue})
