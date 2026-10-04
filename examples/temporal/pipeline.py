"""Native Temporal composition; run this Workflow on a separate pipeline worker.

The application task worker from worker.py must also be running. Each invocation
reattaches to its stable request ID if this parent or its waiting Activity restarts.
"""

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from thunc.temporal.adapters import execute_task


@workflow.defn
class ReviewPipeline:
    @workflow.run
    async def run(self, question: str) -> str:
        identity = workflow.info().workflow_id
        category = await execute_task(
            "question.classify",
            version="1",
            workspace_id="myapp",
            inputs={"question": question},
            returns=str,
            request_id=f"{identity}:classify",
            task_queue="thunc-myapp-v1",
        )
        analysis = await execute_task(
            "repo.analyze",
            version="1",
            workspace_id="myapp",
            inputs={"question": f"{category}: {question}"},
            returns=str,
            request_id=f"{identity}:analyze",
            task_queue="thunc-myapp-v1",
        )
        return await execute_task(
            "analysis.summarize",
            version="1",
            workspace_id="myapp",
            inputs={"analysis": analysis},
            returns=str,
            request_id=f"{identity}:summarize",
            task_queue="thunc-myapp-v1",
        )


async def main() -> None:
    import os

    from temporalio.client import Client
    from temporalio.worker import Worker

    client = await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    async with Worker(client, task_queue="thunc-pipeline-v1", workflows=[ReviewPipeline]):
        result = await client.execute_workflow(
            ReviewPipeline.run,
            "Where is the request timeout configured?",
            id="timeout-pipeline-1",
            task_queue="thunc-pipeline-v1",
        )
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
