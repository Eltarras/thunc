"""Subprocess fixture for real worker termination tests; never calls a provider."""

import asyncio
import json
import sys
import time
from pathlib import Path

import thunc
from thunc.backends import BACKENDS
from thunc.temporal import Registry, Worker
from thunc.temporal.activities import Activities


async def main():
    address, folder, mode = sys.argv[1:]
    base = Path(folder)
    thunc.configure(agents_dir=str(base / "agents"))
    replies = [
        ("read", {"path": "note"}),
        ("edit", {"path": "note", "old": "before", "new": "after"}),
        ("remember", {"note": "once"}),
        ("finish", {"value": "ok"}),
    ]

    def model(text, **kwargs):
        path = base / "model-calls"
        count = len(path.read_text().splitlines()) if path.exists() else 0
        with path.open("a") as stream:
            stream.write("call\n")
            stream.flush()
        tool, args = replies[count]
        return json.dumps({"tool": tool, "args": args})

    BACKENDS["scripted"] = model
    thunc.configure(backend="scripted")
    agent = thunc.Agent("crash", workdir=base / "workspace", permissions=["write:note"])

    @agent.task
    def change() -> str:
        """Edit note and remember once."""
        ...

    registry = Registry(state_dir=base / "state")
    registry.agent_task("change", change, version="1", workspace_id="crash")
    if mode == "pause":
        original = Activities.tool

        def stop_before_tool(self, definition, state):
            (base / "ready").write_text("model turn recorded")
            time.sleep(90)
            return original(self, definition, state)

        Activities.tool = stop_before_tool
    worker = await Worker.connect(address, task_queue="crash", registry=registry)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
