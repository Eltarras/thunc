import asyncio
import json
import os
from pathlib import Path

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer

import thunc
from thunc.temporal import Registry, Runtime, Worker
from thunc.temporal.workflows import AgentWorkflow

pytest_asyncio = pytest.importorskip("pytest_asyncio")
pytestmark = [
    pytest.mark.temporal_integration,
    pytest.mark.skipif(os.environ.get("THUNC_TEMPORAL_TESTS") != "1", reason="Temporal service tests are opt-in"),
]


@pytest_asyncio.fixture
async def service(tmp_path):
    if os.environ.get("THUNC_TEMPORAL_TESTS") != "1":
        pytest.skip("set THUNC_TEMPORAL_TESTS=1 to start the local Temporal service")
    binary = os.environ.get("THUNC_TEMPORAL_CLI")
    async with await WorkflowEnvironment.start_local(
        dev_server_existing_path=binary,
        download_dest_dir=os.environ.get("THUNC_TEMPORAL_DOWNLOAD_DIR", str(tmp_path)),
        dev_server_database_filename=str(tmp_path / "server.sqlite"),
    ) as environment:
        yield environment


@pytest.mark.asyncio
async def test_agent_write_dedup_reconnect_and_replay(service, tmp_path, fake):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note").write_text("before")
    agent = thunc.Agent("test", workdir=root, permissions=["write:note"])

    @agent.task
    def change() -> str:
        """Read note, edit it, remember and finish."""
        ...

    fake.replies = [
        json.dumps({"tool": t, "args": a})
        for t, a in [
            ("read", {"path": "note"}),
            ("edit", {"path": "note", "old": "before", "new": "after"}),
            ("remember", {"note": "done"}),
            ("finish", {"value": "ok"}),
        ]
    ]
    registry = Registry(state_dir=tmp_path / "state")
    registry.agent_task("change", change, version="1", workspace_id="repo")
    runtime = Runtime(service.client, task_queue="test")
    async with Worker(service.client, task_queue="test", registry=registry):
        handle = await runtime.start(
            "change", version="1", workspace_id="repo", inputs={}, returns=str, request_id="one"
        )
        run = await asyncio.wait_for(handle.result(), timeout=30)
        assert run.value == "ok" and run.steps == 4
        assert run.files_changed == ["note"] and (root / "note").read_text() == "after"
        assert agent.memory.count("done") == 1
        duplicate = await runtime.start(
            "change", version="1", workspace_id="repo", inputs={}, returns=str, request_id="one"
        )
        assert (await duplicate.result()).value == "ok"
        assert (await runtime.get(handle.id, returns=str).result()).value == "ok"
        assert len(fake.prompts) == 4
        with pytest.raises(thunc.ThuncError, match="owned by Temporal"):
            change()
        history = await service.client.get_workflow_handle(handle.id).fetch_history()
        await Replayer(workflows=[AgentWorkflow]).replay_workflow(history)


@pytest.mark.asyncio
async def test_durable_and_local_runs_make_the_same_decisions(service, tmp_path, fake):
    replies = [
        json.dumps({"tool": t, "args": a})
        for t, a in [
            ("teleport", {}),
            ("remember", {"note": "checked"}),
            ("finish", {"value": "many"}),
            ("finish", {"value": 3}),
        ]
    ]
    runs, prompts = [], []
    for mode in ("local", "durable"):
        root = tmp_path / mode
        root.mkdir()
        agent = thunc.Agent(f"parity-{mode}", workdir=root, retries=1)

        @agent.task
        def count() -> int:
            """Count the things."""
            ...

        fake.replies, fake.prompts[:] = list(replies), []
        if mode == "local":
            runs.append(agent.run(count))
        else:
            registry = Registry(state_dir=tmp_path / "state")
            registry.agent_task("count", count, version="1", workspace_id="parity")
            runtime = Runtime(service.client, task_queue="parity")
            async with Worker(service.client, task_queue="parity", registry=registry):
                handle = await runtime.start(
                    "count", version="1", workspace_id="parity", inputs={}, returns=int, request_id="one"
                )
                runs.append(await asyncio.wait_for(handle.result(), timeout=30))
        prompts.append(fake.prompts[-1])
    local, durable = runs
    assert local.value == durable.value == 3 and local.steps == durable.steps == 4
    assert local.denied == durable.denied == [] and local.notes == durable.notes == ["checked"]
    for prompt in prompts:  # what the model was told along the way
        assert "unknown tool 'teleport'" in prompt and "that value is invalid" in prompt


@pytest.mark.asyncio
async def test_functions_repair_each_turn_and_reject_mismatched_request(service, tmp_path, fake):
    from temporalio.client import WorkflowUpdateFailedError

    @thunc.function
    def number(text: str) -> int:
        """Return the integer in text."""
        ...

    fake.replies = ["bad", "42"]
    registry = Registry(state_dir=tmp_path / "state")
    registry.function("number", number, version="1", workspace_id="functions")
    runtime = Runtime(service.client, task_queue="functions")
    async with Worker(service.client, task_queue="functions", registry=registry):
        handle = await runtime.start(
            "number", version="1", workspace_id="functions", inputs={"text": "42"}, returns=int, request_id="one"
        )
        run = await asyncio.wait_for(handle.result(), 30)
        assert run.value == 42 and run.steps == 2 and len(fake.prompts) == 2
        with pytest.raises(WorkflowUpdateFailedError):
            await runtime.start(
                "number", version="1", workspace_id="functions", inputs={"text": "7"}, returns=int, request_id="one"
            )
        history = await handle.child.fetch_history()
        await Replayer(workflows=[AgentWorkflow]).replay_workflow(history)


@pytest.mark.asyncio
async def test_unresolved_command_pauses_lane_and_resolution_is_idempotent(service, tmp_path, fake, monkeypatch):
    from thunc.temporal.activities import Activities

    root = tmp_path / "workspace"
    root.mkdir()
    agent = thunc.Agent("commands", workdir=root, permissions=["run:echo"])

    @agent.task
    def command() -> str:
        """Run echo hello and finish."""
        ...

    fake.replies = [
        json.dumps({"tool": "run", "args": {"command": "echo hello"}}),
        json.dumps({"tool": "finish", "args": {"value": "recovered"}}),
        json.dumps({"tool": "finish", "args": {"value": "second"}}),
    ]
    registry = Registry(state_dir=tmp_path / "state")
    registry.agent_task("command", command, version="1", workspace_id="commands")
    original = Activities.tool

    def uncertain(self, definition, state):
        if state["agent"]["turns"] == 1 and state["request"]["id"].endswith(first_id[0]) and not injected[0]:
            injected[0] = True
            self.storage.save_effect(state["id"] + "/1/0", state["id"], "started", {"call": {}})
        return original(self, definition, state)

    injected, first_id = [False], [""]
    monkeypatch.setattr(Activities, "tool", uncertain)
    runtime = Runtime(service.client, task_queue="commands")
    async with Worker(service.client, task_queue="commands", registry=registry):
        first = await runtime.start(
            "command", version="1", workspace_id="commands", inputs={}, returns=str, request_id="one"
        )
        first_id[0] = first.id
        second = await runtime.start(
            "command", version="1", workspace_id="commands", inputs={}, returns=str, request_id="two"
        )

        async def attention():
            while (await first.status())["status"] != "needs_attention":
                await asyncio.sleep(0.05)

        await asyncio.wait_for(attention(), 20)
        assert (await second.status())["status"] == "queued"
        assert len(fake.prompts) == 1
        operation = (await first.status())["operation_id"]
        await first.resolve(operation, "complete", "Verified process stopped and its output", output="hello")
        assert (await asyncio.wait_for(first.result(), 20)).value == "recovered"
        assert (await asyncio.wait_for(second.result(), 20)).value == "second"
        assert registry.storage.unresolved(first.id) == []


@pytest.mark.asyncio
async def test_continue_as_new_preserves_progress(service, tmp_path, fake):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note").write_text("value")
    agent = thunc.Agent("long", workdir=root, max_steps=60)

    @agent.task
    def long_run() -> str:
        """Read repeatedly and finish."""
        ...

    fake.replies = [json.dumps({"tool": "read", "args": {"path": "note"}})] * 52 + [
        json.dumps({"tool": "finish", "args": {"value": "ok"}})
    ]
    registry = Registry(state_dir=tmp_path / "state")
    registry.agent_task("long", long_run, version="1", workspace_id="long")
    runtime = Runtime(service.client, task_queue="long")
    async with Worker(service.client, task_queue="long", registry=registry):
        handle = await runtime.start("long", version="1", workspace_id="long", inputs={}, returns=str, request_id="one")
        run = await asyncio.wait_for(handle.result(), 40)
        assert run.steps == 53 and len(fake.prompts) == 53
        history = await handle.child.fetch_history()
        assert history.events[0].workflow_execution_started_event_attributes.continued_execution_run_id
        await Replayer(workflows=[AgentWorkflow]).replay_workflow(history)


@pytest.mark.asyncio
async def test_cancel_queued_run_never_calls_model(service, tmp_path, fake, monkeypatch):
    import threading

    from thunc.temporal import DurableError
    from thunc.temporal.activities import Activities

    root = tmp_path / "workspace"
    root.mkdir()
    agent = thunc.Agent("cancel", workdir=root)

    @agent.task
    def task() -> str:
        """Finish."""
        ...

    fake.replies = [json.dumps({"tool": "finish", "args": {"value": "first"}})]
    registry = Registry(state_dir=tmp_path / "state")
    registry.agent_task("task", task, version="1", workspace_id="cancel")
    gate = threading.Event()
    started = threading.Event()
    original = Activities.turn

    def paused(self, definition, state):
        started.set()
        gate.wait(20)
        return original(self, definition, state)

    monkeypatch.setattr(Activities, "turn", paused)
    runtime = Runtime(service.client, task_queue="cancel")
    async with Worker(service.client, task_queue="cancel", registry=registry):
        first = await runtime.start(
            "task", version="1", workspace_id="cancel", inputs={}, returns=str, request_id="first"
        )
        await asyncio.to_thread(started.wait, 10)
        second = await runtime.start(
            "task", version="1", workspace_id="cancel", inputs={}, returns=str, request_id="second"
        )
        await second.cancel()
        gate.set()
        assert (await first.result()).value == "first"
        with pytest.raises(DurableError, match="cancelled"):
            await asyncio.wait_for(second.result(), 20)
        assert len(fake.prompts) == 1


@pytest.mark.asyncio
async def test_killed_worker_resumes_recorded_turn(service, tmp_path):
    import subprocess
    import sys

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note").write_text("before")
    script = Path(__file__).with_name("process_worker.py")
    address = service.client.service_client.config.target_host
    processes = []
    log = (tmp_path / "workers.log").open("w")

    def start(mode):
        process = subprocess.Popen(
            [sys.executable, str(script), address, str(tmp_path), mode],
            stdout=log,
            stderr=log,
            env={**os.environ, "PYTHONPATH": str(Path.cwd())},
        )
        processes.append(process)
        return process

    first = start("pause")
    runtime = Runtime(service.client, task_queue="crash")
    try:
        handle = await asyncio.wait_for(
            runtime.start("change", version="1", workspace_id="crash", inputs={}, returns=str, request_id="one"), 20
        )

        async def ready():
            while not (tmp_path / "ready").exists():
                if first.poll() is not None:
                    pytest.fail((tmp_path / "workers.log").read_text())
                await asyncio.sleep(0.05)

        await asyncio.wait_for(ready(), 20)
        first.kill()
        first.wait(timeout=5)
        start("resume")
        run = await asyncio.wait_for(handle.result(), 45)
        assert run.value == "ok" and run.steps == 4
        assert (tmp_path / "model-calls").read_text().splitlines() == ["call"] * 4
        assert (root / "note").read_text() == "after"
        assert (tmp_path / "agents" / "crash" / "memory.md").read_text().count("once") == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        log.close()


@pytest.mark.asyncio
async def test_transport_retries_are_bounded_and_counted(service, tmp_path, fake, monkeypatch):
    from thunc.backends import BACKENDS

    attempts = []

    def unreliable(text, **kwargs):
        attempts.append(text)
        if len(attempts) < 3:
            raise TimeoutError("transient")
        return "7"

    monkeypatch.setitem(BACKENDS, "fake", unreliable)

    @thunc.function
    def task() -> int:
        """Return seven."""
        ...

    registry = Registry(state_dir=tmp_path / "state")
    registry.function("task", task, version="1", workspace_id="retry")
    runtime = Runtime(service.client, task_queue="retry")
    async with Worker(service.client, task_queue="retry", registry=registry):
        handle = await runtime.start(
            "task", version="1", workspace_id="retry", inputs={}, returns=int, request_id="one"
        )
        run = await asyncio.wait_for(handle.result(), 20)
        assert run.value == 7 and run.steps == 1 and run.retries == 2 and len(attempts) == 3


@pytest.mark.asyncio
async def test_cancel_running_command_stops_before_its_effect(service, tmp_path, fake):
    import shlex
    import sys

    from thunc.temporal import DurableError

    root = tmp_path / "workspace"
    root.mkdir()
    code = (
        "import os,time; from pathlib import Path; Path('started').write_text(str(os.getpid())); "
        "time.sleep(15); Path('should-not-exist').touch()"
    )
    command = shlex.join([sys.executable, "-c", code])
    agent = thunc.Agent("cancel-command", workdir=root, permissions=["run:" + shlex.quote(sys.executable)])

    @agent.task
    def task() -> str:
        """Run the command."""
        ...

    fake.replies = [json.dumps({"tool": "run", "args": {"command": command}})]
    registry = Registry(state_dir=tmp_path / "state")
    registry.agent_task("task", task, version="1", workspace_id="cancel-command")
    runtime = Runtime(service.client, task_queue="cancel-command")
    async with Worker(service.client, task_queue="cancel-command", registry=registry):
        handle = await runtime.start(
            "task", version="1", workspace_id="cancel-command", inputs={}, returns=str, request_id="one"
        )

        async def started():
            while not (root / "started").exists():
                await asyncio.sleep(0.05)

        await asyncio.wait_for(started(), 10)
        await handle.cancel()

        # SDK thread cancellation may conservatively leave an unresolved intent;
        # if so, the test has verified process cleanup before acknowledging it.
        async def cancelled_or_attention():
            while (await handle.status())["status"] not in {"cancelled", "needs_attention"}:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(cancelled_or_attention(), 15)
        status = await handle.status()
        with pytest.raises(ProcessLookupError):
            os.kill(int((root / "started").read_text()), 0)
        if status["status"] == "needs_attention":
            await handle.resolve(status["operation_id"], "abort", "Test command process was stopped by cancellation")
        with pytest.raises(DurableError, match="cancelled"):
            await asyncio.wait_for(handle.result(), 10)
        assert not (root / "should-not-exist").exists()


@pytest.mark.asyncio
async def test_native_workflow_composes_registered_tasks_without_slot_deadlock(service, tmp_path, fake):
    from temporalio.worker import Worker as SDKWorker

    from examples.temporal.pipeline import ReviewPipeline

    @thunc.function
    def classify(question: str) -> str:
        """Classify."""
        ...

    @thunc.function
    def analyze(question: str) -> str:
        """Analyze."""
        ...

    @thunc.function
    def summarize(analysis: str) -> str:
        """Summarize."""
        ...

    registry = Registry(state_dir=tmp_path / "state")
    registry.function("question.classify", classify, version="1", workspace_id="myapp")
    registry.function("repo.analyze", analyze, version="1", workspace_id="myapp")
    registry.function("analysis.summarize", summarize, version="1", workspace_id="myapp")
    fake.replies = ["bug", "analysis", "summary"]
    async with Worker(service.client, task_queue="thunc-myapp-v1", registry=registry, concurrency=1):
        async with SDKWorker(service.client, task_queue="pipeline", workflows=[ReviewPipeline]):
            result = await asyncio.wait_for(
                service.client.execute_workflow(
                    ReviewPipeline.run, "question", id="pipeline-one", task_queue="pipeline"
                ),
                30,
            )
            assert result == "summary" and len(fake.prompts) == 3
