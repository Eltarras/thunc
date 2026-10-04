"""Call registered thunc tasks from a native Temporal Workflow.

The coordinator owns the agent execution; the parent orchestrates typed tasks.
The invoke Activity only submits/awaits a stable run, never executes an agent loop.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, TypeVar, cast

from temporalio import workflow
from temporalio.common import RetryPolicy

T = TypeVar("T")


async def execute_task(
    task: str,
    *,
    version: str,
    workspace_id: str,
    inputs: dict[str, Any],
    returns: type[T],
    request_id: str,
    task_queue: str,
    deadline_seconds: float = 1800,
) -> T:
    with workflow.unsafe.imports_passed_through():
        from thunc.schema import json_schema, validate
    record: dict[str, Any] = await workflow.execute_activity(
        "thunc.invoke.v1",
        {
            "task": task,
            "version": version,
            "workspace_id": workspace_id,
            "inputs": inputs,
            "returns": json_schema(returns),
            "request_id": request_id,
            "deadline_seconds": deadline_seconds,
        },
        task_queue=task_queue + "-clients",
        start_to_close_timeout=timedelta(days=7),
        heartbeat_timeout=timedelta(seconds=20),
        retry_policy=RetryPolicy(maximum_attempts=3),
        cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
    )
    return cast(T, validate(record["value"], returns))
