"""Run from this directory after installing thunc[temporal,openai]."""

import asyncio
import os

from application import analyze, classify, summarize

from thunc.temporal import Registry, Worker


async def main() -> None:
    registry = Registry(state_dir=os.environ.get("THUNC_TEMPORAL_STATE", "./temporal-state"))
    registry.agent_task("repo.analyze", analyze, version="1", workspace_id="myapp")
    registry.function("question.classify", classify, version="1", workspace_id="myapp")
    registry.function("analysis.summarize", summarize, version="1", workspace_id="myapp")
    worker = await Worker.connect(
        os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"), task_queue="thunc-myapp-v1", registry=registry
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
