import asyncio
import os

from thunc.temporal import Runtime


async def main() -> None:
    runtime = await Runtime.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"), task_queue="thunc-myapp-v1")
    handle = await runtime.start(
        "repo.analyze",
        version="1",
        workspace_id="myapp",
        inputs={"question": "Where is the request timeout configured?"},
        returns=str,
        request_id="timeout-review-1",
    )
    print("Reconnect with:", handle.id)
    print((await handle.result()).value)


if __name__ == "__main__":
    asyncio.run(main())
