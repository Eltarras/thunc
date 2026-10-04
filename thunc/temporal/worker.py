"""Worker lifecycle, with a bounded thread pool for existing synchronous providers."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

from temporalio.client import Client
from temporalio.worker import Worker as TemporalWorker

from .activities import Activities
from .registry import Registry
from .workflows import AgentWorkflow, WorkspaceWorkflow


class Worker:
    def __init__(self, client: Client, *, task_queue: str, registry: Registry, concurrency: int = 8) -> None:
        registry.claim(client.service_client.config.target_host + "/" + client.namespace)
        self.executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="thunc-activity")
        self.activities = Activities(registry)
        self.activities.client, self.activities.task_queue = client, task_queue
        self.worker = TemporalWorker(
            client,
            task_queue=task_queue,
            workflows=[AgentWorkflow, WorkspaceWorkflow],
            activities=[self.activities.step],
            activity_executor=self.executor,
            max_concurrent_activities=concurrency,
            graceful_shutdown_timeout=timedelta(seconds=10),
            max_heartbeat_throttle_interval=timedelta(seconds=3),
        )
        # Awaiting runs must not occupy the same Activity slots needed to execute them.
        self.clients = TemporalWorker(
            client,
            task_queue=task_queue + "-clients",
            activities=[self.activities.invoke],
            max_concurrent_activities=128,
            graceful_shutdown_timeout=timedelta(seconds=10),
        )

    @classmethod
    async def connect(
        cls,
        address: str,
        *,
        task_queue: str,
        registry: Registry,
        namespace: str = "default",
        concurrency: int = 8,
        **options: Any,
    ) -> Worker:
        return cls(
            await Client.connect(address, namespace=namespace, **options),
            task_queue=task_queue,
            registry=registry,
            concurrency=concurrency,
        )

    async def run(self) -> None:
        try:
            async with self.clients:
                await self.worker.run()
        finally:
            self.executor.shutdown(wait=True)

    async def shutdown(self) -> None:
        await asyncio.gather(self.worker.shutdown(), self.clients.shutdown())

    async def __aenter__(self) -> Worker:
        await self.clients.__aenter__()
        try:
            await self.worker.__aenter__()
        except BaseException:
            await self.clients.__aexit__(None, None, None)
            self.executor.shutdown(wait=True)
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        try:
            await self.worker.__aexit__(*args)
        finally:
            await self.clients.__aexit__(*args)
            self.executor.shutdown(wait=True)
