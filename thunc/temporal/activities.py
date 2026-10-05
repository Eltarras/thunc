"""All nondeterministic work lives here, outside Workflow replay."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import os
import threading
from dataclasses import asdict
from datetime import timedelta
from typing import Any, cast

import temporalio.exceptions
from temporalio import activity
from temporalio.client import Client
from temporalio.exceptions import ApplicationError

from thunc import native
from thunc.agent import _memory_section, _request
from thunc.config import runtime_settings
from thunc.core import _build_prompt, _check, _send, _sendable, system_prompt
from thunc.execution import AgentState
from thunc.schema import describe, shorten
from thunc.store import Store

from .effects import Restricted, perform, resolve
from .models import INLINE_LIMIT, DurableError, encoded, identifiers, plain
from .registry import Definition, Registry
from .storage import Attention


@contextlib.contextmanager
def heartbeat() -> Any:
    stop = threading.Event()
    context = contextvars.copy_context()

    def pulse() -> None:
        while not stop.wait(2):
            activity.heartbeat()

    thread = threading.Thread(target=lambda: context.run(pulse), daemon=True)
    thread.start()
    try:
        activity.heartbeat()
        yield
    finally:
        stop.set()
        thread.join(timeout=3)


def provider_error(exc: Exception) -> ApplicationError:
    cause: BaseException | None = exc
    transient = False
    delay = None
    while cause is not None:
        status = getattr(cause, "status_code", None)
        if status in {408, 429, 500, 502, 503, 504} or type(cause).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ReadTimeout",
            "TimeoutError",
        }:
            transient = True
            response = getattr(cause, "response", None)
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                retry_after = getattr(response, "headers", {}).get("retry-after")
                if retry_after is not None:
                    delay = timedelta(seconds=max(0, min(300, float(retry_after))))
        cause = cause.__cause__
    # Avoid transmitting provider exception text (which can contain credentials/input).
    return ApplicationError(
        "Transient provider failure" if transient else f"{type(exc).__name__}: durable step failed",
        type="TransientProviderError" if transient else "PermanentStepError",
        non_retryable=not transient,
        next_retry_delay=delay,
    )


class Activities:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.storage = registry.storage

    client: Client | None = None
    task_queue: str = ""

    @activity.defn(name="thunc.invoke.v1")
    async def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        from .client import Handle, Runtime

        assert self.client is not None
        coordinator, identity = identifiers(
            request["workspace_id"], request["task"], request["version"], request["request_id"]
        )
        submission = {k: v for k, v in request.items() if k != "request_id"}
        submission.update(id=identity, queue=self.task_queue)
        if len(encoded(submission)) > INLINE_LIMIT:
            raise ApplicationError("Submission exceeds 256 KiB", non_retryable=True)
        runtime = Runtime(self.client, task_queue=self.task_queue)
        await runtime._submit(submission, coordinator)
        handle = Handle(runtime, identity, Any, submission)
        waiting = asyncio.create_task(handle.result())
        try:
            while not waiting.done():
                activity.heartbeat(identity)
                await asyncio.wait({waiting}, timeout=2)
            return plain(asdict(await waiting))  # type: ignore[no-any-return]
        except DurableError as exc:
            raise ApplicationError(str(exc), non_retryable=True) from None
        except asyncio.CancelledError:
            await handle.cancel()
            raise
        finally:
            waiting.cancel()

    @activity.defn(name="thunc.step.v1")
    def step(self, message: dict[str, Any]) -> dict[str, Any]:
        token = runtime_settings.set({"timeout": 120.0, "sdk_options": {"max_retries": 0}})
        try:
            with heartbeat():
                response = self._step(message)
                activity.logger.info(
                    "thunc durable step completed",
                    extra={
                        "thunc_run_id": message["request"]["id"],
                        "thunc_action": message["action"],
                        "thunc_attempt": activity.info().attempt,
                        "thunc_next": response.get("next"),
                    },
                )
                return response
        # Temporal's CancelledError is an Exception; report it as a cancellation, not a step failure.
        except (ApplicationError, temporalio.exceptions.CancelledError):
            raise
        except Exception as exc:
            raise provider_error(exc) from None
        finally:
            runtime_settings.reset(token)

    def _step(self, message: dict[str, Any]) -> dict[str, Any]:
        action = message["action"]
        request = message["request"]
        definition = self.registry.get(request)
        if action == "admit":
            self.registry.validate_request(request)
            self.storage.admit(request)
            return {"complete": self.storage.result(request["id"]) is not None}
        if action == "result":
            return self.storage.result(request["id"]) or {"status": "queued"}
        if action == "reconcile":
            return {"pending": self.storage.unresolved(request["id"])}
        if action == "resolve":
            resolution = message["resolution"]
            operation = resolution["operation_id"]
            if not operation.startswith(request["id"] + "/"):
                raise ValueError("Resolution is for a different run")
            resolve(
                self.storage,
                definition.workspace,
                operation,
                resolution["decision"],
                resolution["evidence"],
                resolution.get("output", ""),
            )
            return {"resolved": operation}
        if action == "record":
            if len(encoded(message["result"])) > INLINE_LIMIT:
                raise ValueError("Final result exceeds 256 KiB; return a small artifact reference instead")
            self.storage.finish(request["id"], message["result"])
            return cast(dict[str, Any], message["result"])
        if action == "prepare":
            return self.prepare(definition, request)
        state: dict[str, Any] = self.storage.get(message["ref"])
        if state["format"] != 2 or state["fingerprint"] != definition.fingerprint:
            raise ValueError("Task definition/state version changed; restore its original worker")
        state["retries"] += activity.info().attempt - 1
        if action == "turn":
            return self.turn(definition, state)
        if action == "tool":
            return self.tool(definition, state)
        if action == "validate":
            return self.validate(definition, state)
        raise ValueError("Unknown durable action")

    def save(self, state: dict[str, Any], action: str, **extra: Any) -> dict[str, Any]:
        return {
            "ref": self.storage.put(state),
            "next": action,
            "turns": state["agent"]["turns"],
            "attempt_seconds": max(150, state.get("command_timeout", 120) + 30) if action == "tool" else 150,
            **extra,
        }

    def prepare(self, definition: Definition, request: dict[str, Any]) -> dict[str, Any]:
        # A prepare retry uses the same immutable initial snapshot.
        key = request["id"] + "/prepare"
        with self.storage.lock(definition.workspace):
            previous = self.storage.effect(key)
            if previous:
                return cast(dict[str, Any], previous[1]["result"])
            state: dict[str, Any] = {
                "format": 2,
                "id": request["id"],
                "request": request,
                "fingerprint": definition.fingerprint,
                "agent": AgentState().to_json(),
                "seen": {},
                "files_changed": [],
                "commands": [],
                "retries": 0,
                "inputs": plain(definition.inputs(request["inputs"])),
            }
            agent = definition.agent
            if agent:
                store = Store(agent.name)
                store.folder = str(definition.store_folder)
                with store.lock():
                    memory = store.memory()
                    followed = agent._read_followed()
                state.update(
                    command_timeout=agent.command_timeout,
                    fixed=agent._fixed_prompt(followed, agent._native()),
                    memory=_memory_section(memory),
                    native=agent._native(),
                    permissions=agent.permissions.written,
                    offered=agent.tools(),
                    followed=[path for path, _ in followed],
                    max_steps=agent.max_steps,
                    repairs=agent.retries,
                )
                conversation = self.conversation(definition, state, restore=False)
                state["conversation"] = native.snapshot(conversation)
            if not agent:
                state["text"] = _build_prompt(definition.spec.instructions, state["inputs"], definition.spec.returns)
                state["original_text"] = state["text"]
                state["system"] = system_prompt(definition.options.get("system"))
                state["max_steps"] = definition.options["retries"] + 1
            result = self.save(state, "turn")
            self.storage.save_effect(key, request["id"], "done", {"result": result})
            return result

    def conversation(
        self, definition: Definition, state: dict[str, Any], *, restore: bool = True
    ) -> native.Conversation:
        agent = copy.copy(definition.agent)
        assert agent is not None
        agent.permissions = Restricted(state["permissions"], agent.permissions.written)
        endpoint = (
            os.environ.get("OPENAI_BASE_URL") if agent.backend == "openai" else os.environ.get("ANTHROPIC_BASE_URL")
        )
        if endpoint != definition.endpoint:
            raise ValueError("Provider endpoint changed; restart a correctly configured versioned worker")
        conversation = agent._conversation(
            state["native"],
            state["fixed"],
            state["memory"],
            _request(definition.spec.instructions, state["inputs"], definition.spec.returns),
            definition.spec.returns,
        )
        if restore:
            native.restore(conversation, state["conversation"])
        return conversation

    def turn(self, definition: Definition, state: dict[str, Any]) -> dict[str, Any]:
        engine = AgentState.from_json(state["agent"])
        engine.begin_turn(state["max_steps"], definition.name)
        if definition.agent is None:
            state["agent"] = engine.to_json()
            state["answer"] = _send(
                state["text"], state["system"], definition.options["backend"], definition.options["model"]
            )
            return self.save(state, "validate")
        conversation = self.conversation(definition, state)
        reply = conversation.next()
        if len(reply.calls) > 64:
            raise ValueError("Model reply exceeds 64 tool calls")
        if not engine.receive(reply):
            conversation.nudge(reply)
        elif engine.pending() is None:
            conversation.results(engine.results_to_send(state["max_steps"]))
        state["agent"] = engine.to_json()
        state["conversation"] = native.snapshot(conversation)
        return self.save(state, "tool" if engine.pending() else "turn")

    def validate(self, definition: Definition, state: dict[str, Any]) -> dict[str, Any]:
        try:
            value = _check(state["answer"], definition.spec.returns, definition.options.get("ensure"))
        except ValueError as exc:
            state["text"] = (
                state["original_text"]
                + "\n\nYour previous reply was:\n"
                + _sendable(state["answer"][:1000])
                + "\nThat is invalid ("
                + _sendable(shorten(str(exc), 1000))
                + "). Reply with only "
                + describe(definition.spec.returns)
            )
            return self.save(state, "turn")
        state["value"] = plain(value)
        return self.save(state, "finish", result=self.record(state))

    def tool(self, definition: Definition, state: dict[str, Any]) -> dict[str, Any]:
        """Settle one pending call. Decisions come from the shared engine; effects go through the journal."""
        engine = AgentState.from_json(state["agent"])
        call = engine.pending()
        if call is None:
            raise ValueError("No pending tool call")
        operation = f"{state['id']}/{engine.turns}/{engine.index}"
        ensure = definition.function.__dict__["__thunc_ensure__"]
        outcome = engine.check(call, definition.spec.returns, ensure, state["repairs"], definition.name)
        if outcome.finished:
            state["agent"] = engine.to_json()
            state["value"] = plain(outcome.value)
            return self.save(state, "finish", result=self.record(state))
        if outcome.error:
            raise outcome.error
        if outcome.output is None:
            try:
                # Defer asynchronous thread interruption until the effect receipt is safe.
                # Commands still observe is_cancelled cooperatively and stop their process group.
                with activity.shield_thread_cancel_exception():
                    result = perform(self.storage, definition, state, asdict(call), operation)
            except Attention as exc:
                return self.save(state, "attention", operation_id=operation, reason=str(exc))
            state["seen"].update(result.get("seen", {}))
            state["files_changed"] = list(dict.fromkeys([*state["files_changed"], *result.get("changed", [])]))
            state["commands"].extend(result.get("commands", []))
            output, denied = result["output"], result["denied"]
        else:
            output, denied = outcome.output, False
        engine.done(call, output, denied)
        if engine.operations > 4096:
            raise ValueError("Agent exceeded 4096 tool operations")
        state["agent"] = engine.to_json()
        if engine.pending() is None:
            conversation = self.conversation(definition, state)
            conversation.results(engine.results_to_send(state["max_steps"]))
            state["conversation"] = native.snapshot(conversation)
            return self.save(state, "turn")
        return self.save(state, "tool")

    def record(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": state["id"],
            "status": "completed",
            "value": state["value"],
            "steps": state["agent"]["turns"],
            "retries": state["retries"],
            "files_changed": state["files_changed"],
            "commands": state["commands"],
            "denied": state["agent"]["denied"],
            "notes": state["agent"]["notes"],
            "artifacts": [self.storage.put(state)["sha256"]],
        }
