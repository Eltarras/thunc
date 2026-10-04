"""How a run talks to its backend, one model reply at a time.

- TextConversation works on text backends: the model replies with JSON actions as text (one, or
  an array of independent ones), and the whole transcript is sent again each turn (the CLI
  backends keep no conversation), so every action saved by batching saves a resend.
- AnthropicConversation and OpenAIConversation use the APIs' own tool calls: tools are declared
  with JSON Schemas, the model can call several at once, and the conversation grows by appending.

Each gives the agent loop the same three things: next() for the model's reply, results() to send
back what its calls did, and nudge() when a reply had no usable call.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .backends import _FALLBACK_MODELS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL, sdk_client
from .config import resolve_backend, setting
from .core import _send, _sendable
from .errors import ThuncError
from .schema import parse, shorten

NUDGE = "Reply by calling one of your tools. When you're done, call finish with your result."
CLI_BACKENDS = frozenset({"claude-code", "codex"})
CLI_STEP_TIMEOUT = 120.0  # seconds one text-protocol step on a CLI may take before it's retried


@dataclass
class Call:
    """One tool call from the model."""

    id: str | None  # the API's id for it; None on the text protocol
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    problem: str | None = None  # why it can't be carried out as given (arguments that aren't JSON, ...)


@dataclass
class Reply:
    calls: list[Call]
    raw: str  # for the trace: the reply's text, or its calls as JSON
    problem: str | None = None  # why the reply has no usable call
    same_turn: bool = False  # more calls of the model reply the last Reply came from (see claude_code.py)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema of its arguments


class Conversation(Protocol):
    def next(self) -> Reply: ...
    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None: ...  # (call, output, failed)
    def nudge(self, reply: Reply) -> None: ...


def snapshot(conversation: Conversation) -> dict[str, Any]:
    """Lossless provider wire data; never serialize clients or Python SDK objects."""
    if isinstance(conversation, TextConversation):
        data = dict(kind="text", steps=conversation.steps)
    elif isinstance(conversation, AnthropicConversation):
        data = dict(kind="anthropic", messages=conversation.messages)
    elif isinstance(conversation, OpenAIConversation):
        data = dict(kind="openai", input=conversation.input)
    else:
        raise TypeError("This conversation does not support snapshots")
    return typing.cast(dict[str, Any], json.loads(json.dumps(data, default=lambda x: x.model_dump(mode="json"))))


def restore(conversation: Conversation, data: dict[str, Any]) -> None:
    """Restore wire messages, including opaque reasoning/signature blocks."""
    if isinstance(conversation, TextConversation) and data["kind"] == "text":
        conversation.steps = data["steps"]
    elif isinstance(conversation, AnthropicConversation) and data["kind"] == "anthropic":
        conversation.messages = data["messages"]
    elif isinstance(conversation, OpenAIConversation) and data["kind"] == "openai":
        conversation.input = data["input"]
    else:
        raise ValueError("Conversation snapshot protocol mismatch")


# --- the text protocol --------------------------------------------------------------------------


class TextConversation:
    def __init__(self, system: str, request: str, names: Sequence[str], backend: str | None, model: str | None):
        self.system, self.request, self.names = system, request, list(names)
        self.backend, self.model = backend, model
        self.steps: list[str] = []

    def next(self) -> Reply:
        timeout = setting("timeout")
        if resolve_backend(self.backend) in CLI_BACKENDS:  # a step that hangs is retried sooner (see agent.py)
            timeout = min(timeout, CLI_STEP_TIMEOUT)
        answer = _send(self._transcript(), self.system, self.backend, self.model, timeout)
        try:
            calls = actions(answer, self.names)
        except ValueError as problem:
            return Reply([], answer, str(problem))
        return Reply(calls, answer)

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        for call, output, _ in results:
            shown = json.dumps({"tool": call.tool, "args": call.args}, ensure_ascii=False)
            self.steps.append(_step(len(self.steps) + 1, shown, output))

    def nudge(self, reply: Reply) -> None:
        self.steps.append(
            _step(len(self.steps) + 1, _sendable(shorten(reply.raw.strip(), 1000)), f"error: {reply.problem}")
        )

    def _transcript(self) -> str:
        ask = "Reply with your {} action as one JSON object, or several independent ones as a JSON array."
        if not self.steps:
            return f"{self.request}\n\n{ask.format('first')}"
        return f"{self.request}\n\n" + "\n\n".join(self.steps) + f"\n\n{ask.format('next')}"


MAX_ACTIONS = 16  # in one text-protocol reply; Temporal's limit per reply is higher (64)


def actions(answer: str, names: Sequence[str]) -> list[Call]:
    """A text-protocol reply as its calls: one JSON action, or a JSON array of them, carried out in
    order. Raises ValueError, with a reason the model can act on, for a reply with no usable call. In
    an array, an action that isn't valid becomes a call with a problem: it fails, the others run."""
    value = parse(answer, dict[str, Any] | list[Any])  # reads code fences and <think> blocks like any answer
    if isinstance(value, dict):
        return [Call(None, *_action(value, names))]
    if not value:
        raise ValueError("the array is empty; send at least one action")
    if len(value) > MAX_ACTIONS:
        raise ValueError(f"{len(value)} actions in one reply; send at most {MAX_ACTIONS}")
    calls = []
    for n, item in enumerate(value, start=1):
        try:
            calls.append(Call(None, *_action(item, names)))
        except ValueError as problem:
            tool = item.get("tool") if isinstance(item, dict) else None
            args = item.get("args") if isinstance(item, dict) else None
            calls.append(
                Call(
                    None,
                    tool if isinstance(tool, str) else "?",
                    args if isinstance(args, dict) else {},
                    f"action {n}: {problem}",
                )
            )
    return calls


def _action(obj: Any, names: Sequence[str]) -> tuple[str, dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError(f'expected an action like {{"tool": ..., "args": {{...}}}}, got {shorten(json.dumps(obj))}')
    tool = obj.get("tool")
    if not isinstance(tool, str) or tool not in names:
        raise ValueError(
            f'unknown tool {tool!r}; reply with {{"tool": ..., "args": {{...}}}} using one of {list(names)}'
        )
    args = obj["args"] if "args" in obj else {k: v for k, v in obj.items() if k != "tool"}
    if not isinstance(args, dict):
        raise ValueError('"args" must be a JSON object')
    return tool, args


def _step(number: int, action_text: str, output: str) -> str:
    output = output.replace("</result>", "<\\/result>")  # a file can't end its own result early
    return f'<step n="{number}">\n<action>{action_text}</action>\n<result>\n{_sendable(output)}\n</result>\n</step>'


def _calls_text(calls: Sequence[Call]) -> str:
    return json.dumps([{"tool": c.tool, "args": c.args} for c in calls], ensure_ascii=False)


# --- the Claude API -----------------------------------------------------------------------------


class AnthropicConversation:
    """Messages API tool use. The fixed system prompt is one block marked for caching and the
    memory a second block after it, so a changed memory doesn't undo the cache; automatic caching
    covers the growing conversation. Replies are appended unchanged, thinking blocks included."""

    def __init__(self, fixed: str, memory: str, request: str, tools: Sequence[Tool], model: str | None):
        try:
            import anthropic
        except ImportError as exc:
            raise ThuncError("The anthropic backend needs the SDK: pip install 'thunc[anthropic]'") from exc
        self._anthropic = anthropic
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = sdk_client(anthropic.Anthropic, "ANTHROPIC_", setting("api_key"), setting("timeout"))
        self.model = model or setting("model") or DEFAULT_ANTHROPIC_MODEL
        self.system: list[Any] = [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}]
        if memory:
            self.system.append({"type": "text", "text": memory})
        self.tools: list[Any] = [
            {"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools
        ]
        self.messages: list[Any] = [{"role": "user", "content": request}]

    def next(self) -> Reply:
        anthropic = self._anthropic
        request: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            cache_control={"type": "ephemeral"},
        )
        try:
            if self.model in _FALLBACK_MODELS:
                response = self.client.beta.messages.create(
                    **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                response = self.client.messages.create(**request)
        except anthropic.APIConnectionError as exc:
            raise ThuncError(f"Could not reach the Claude API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ThuncError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        if response.stop_reason == "refusal":
            raise ThuncError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise ThuncError("The answer was cut off at max_tokens.")
        if response.stop_reason not in ("tool_use", "end_turn", "stop_sequence"):
            raise ThuncError(f"The Claude API stopped with {response.stop_reason!r}.")
        self.messages.append({"role": "assistant", "content": response.content})  # unchanged, thinking blocks too
        calls = [
            Call(block.id, block.name, block.input if isinstance(block.input, dict) else {})
            for block in response.content
            if block.type == "tool_use"
        ]
        text = "".join(block.text for block in response.content if block.type == "text")
        if not calls:
            return Reply([], text, "no tool call")
        return Reply(calls, _calls_text(calls))

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        # Every result in one message: split up, it teaches the model to stop calling tools in parallel.
        content = [
            {"type": "tool_result", "tool_use_id": call.id, "content": output, **({"is_error": True} if failed else {})}
            for call, output, failed in results
        ]
        self.messages.append({"role": "user", "content": content})

    def nudge(self, reply: Reply) -> None:
        self.messages.append({"role": "user", "content": NUDGE})


# --- the OpenAI Responses API -------------------------------------------------------------------


class OpenAIConversation:
    """Responses API function calling. With store=False nothing is kept on OpenAI's side, so every
    request carries the whole conversation, reasoning items included (encrypted)."""

    def __init__(self, system: str, request: str, tools: Sequence[Tool], model: str | None):
        try:
            import openai
        except ImportError as exc:
            raise ThuncError("The openai backend needs the SDK: pip install 'thunc[openai]'") from exc
        self._openai = openai
        # api_key=None lets the SDK resolve OPENAI_API_KEY (and OPENAI_BASE_URL for compatible servers).
        self.client = sdk_client(openai.OpenAI, "OPENAI_", setting("api_key"), setting("timeout"))
        self.model = model or setting("model") or DEFAULT_OPENAI_MODEL
        self.system = system
        self.tools: list[Any] = [
            {"type": "function", "name": t.name, "description": t.description, "parameters": t.schema, "strict": False}
            for t in tools
        ]
        self.input: list[Any] = [{"role": "user", "content": request}]

    def next(self) -> Reply:
        openai = self._openai
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.system,
                input=self.input,
                tools=self.tools,
                store=False,
                include=["reasoning.encrypted_content"],
            )
        except openai.APIConnectionError as exc:
            raise ThuncError(f"Could not reach the OpenAI API: {exc}") from exc
        except openai.APIStatusError as exc:
            raise ThuncError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            raise ThuncError(f"The OpenAI response is incomplete ({reason or 'no reason given'}).")
        if response.status in ("failed", "cancelled"):
            raise ThuncError(f"The OpenAI response {response.status}.")
        calls = []
        for item in response.output:
            if item.type == "message" and any(part.type == "refusal" for part in item.content):
                raise ThuncError("The model declined this request.")
            self.input.append(_as_input(item))
            if item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except ValueError:
                    calls.append(Call(item.call_id, item.name, {}, "the arguments weren't valid JSON"))
                    continue
                if not isinstance(args, dict):
                    calls.append(Call(item.call_id, item.name, {}, "the arguments must be a JSON object"))
                    continue
                calls.append(Call(item.call_id, item.name, args))
        if not calls:
            return Reply([], response.output_text or "", "no tool call")
        return Reply(calls, _calls_text(calls))

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        for call, output, _ in results:
            self.input.append({"type": "function_call_output", "call_id": call.id, "output": output})

    def nudge(self, reply: Reply) -> None:
        self.input.append({"role": "user", "content": NUDGE})


def _as_input(item: Any) -> dict[str, Any]:
    """An output item as the next request's input: the fields an input item of its type takes."""
    data: dict[str, Any] = item.model_dump(exclude_none=True, by_alias=True)
    allowed = _input_fields().get(data.get("type", ""))
    return {k: v for k, v in data.items() if k in allowed} if allowed else data


_INPUT_FIELDS: dict[str, frozenset[str]] = {}


def _input_fields() -> dict[str, frozenset[str]]:
    if not _INPUT_FIELDS:
        from openai.types import responses as r

        for kind, param in (
            ("message", r.ResponseOutputMessageParam),
            ("function_call", r.ResponseFunctionToolCallParam),
            ("reasoning", r.ResponseReasoningItemParam),
        ):
            _INPUT_FIELDS[kind] = frozenset(typing.get_type_hints(param))
    return _INPUT_FIELDS


# backend name -> whether it has native tool calls (the others use the text protocol)
NATIVE = frozenset({"anthropic", "openai"})
