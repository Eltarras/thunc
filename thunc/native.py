"""How a run talks to its backend, one model reply at a time.

- TextConversation works on every backend: the model replies with one JSON action as text, and
  the whole transcript is sent again each step (the CLI backends keep no conversation).
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

from .backends import _FALLBACK_MODELS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL
from .config import setting
from .core import _send, _sendable
from .errors import ThuncError
from .schema import parse, shorten

NUDGE = "Reply by calling one of your tools. When you're done, call finish with your result."


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


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema of its arguments


class Conversation(Protocol):
    def next(self) -> Reply: ...
    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None: ...  # (call, output, failed)
    def nudge(self, reply: Reply) -> None: ...


# --- the text protocol --------------------------------------------------------------------------


class TextConversation:
    def __init__(self, system: str, request: str, names: Sequence[str], backend: str | None, model: str | None):
        self.system, self.request, self.names = system, request, list(names)
        self.backend, self.model = backend, model
        self.steps: list[str] = []

    def next(self) -> Reply:
        answer = _send(self._transcript(), self.system, self.backend, self.model)
        try:
            tool, args = action(answer, self.names)
        except ValueError as problem:
            return Reply([], answer, str(problem))
        return Reply([Call(None, tool, args)], answer)

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        for call, output, _ in results:
            shown = json.dumps({"tool": call.tool, "args": call.args}, ensure_ascii=False)
            self.steps.append(_step(len(self.steps) + 1, shown, output))

    def nudge(self, reply: Reply) -> None:
        self.steps.append(
            _step(len(self.steps) + 1, _sendable(shorten(reply.raw.strip(), 1000)), f"error: {reply.problem}")
        )

    def _transcript(self) -> str:
        if not self.steps:
            return f"{self.request}\n\nReply with your first action as one JSON object."
        return f"{self.request}\n\n" + "\n\n".join(self.steps) + "\n\nReply with your next action as one JSON object."


def action(answer: str, names: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """A text-protocol reply as (tool, args). Raises ValueError with a reason the model can act on."""
    obj = parse(answer, dict[str, Any])  # reads code fences and <think> blocks like any answer
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
        self.client = anthropic.Anthropic(api_key=setting("api_key"), timeout=setting("timeout"))
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
        self.client = openai.OpenAI(api_key=setting("api_key"), timeout=setting("timeout"))
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
