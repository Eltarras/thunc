"""How a run talks to its backend, one model reply at a time.

- TextConversation works on text backends: the model replies with JSON actions as text (one, or
  an array of independent ones), and the whole transcript is sent again each turn (the CLI
  backends keep no conversation), so every action saved by batching saves a resend. Models trained
  for native tool calls often wrap the action in prose or tool-call markup, or carry on past it with
  results they make up; the first complete action is read, and on Claude Code the step stops as
  soon as it has arrived.
- AnthropicConversation and OpenAIConversation use the APIs' own tool calls: tools are declared
  with JSON Schemas, the model can call several at once, and the conversation grows by appending.

Each gives the agent loop the same three things: next() for the model's reply, results() to send
back what its calls did, and nudge() when a reply had no usable call.
"""

from __future__ import annotations

import json
import re
import threading
import time
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import tools as builtin_tools
from .backends import _FALLBACK_MODELS, DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL, sdk_client
from .config import resolve_backend, setting
from .core import _send, _sendable
from .errors import ThuncError, TransientError, transient_claude_error, transient_status
from .schema import parse, shorten

NUDGE = "Reply by calling one of your tools. When you're done, call finish with your result."
EFFORTS = ("low", "medium", "high", "xhigh", "max")  # Agent(effort=...): how hard the model thinks
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
    def __init__(
        self,
        system: str,
        request: str,
        names: Sequence[str],
        backend: str | None,
        model: str | None,
        effort: str | None = None,
    ):
        self.system, self.request, self.names = system, request, list(names)
        self.backend, self.model, self.effort = backend, model, effort
        self.steps: list[str] = []
        self.note: str | None = None  # for the model with the results of a reply read leniently

    def next(self) -> Reply:
        timeout = setting("timeout")
        if resolve_backend(self.backend) in CLI_BACKENDS:  # a step that hangs is retried sooner (see agent.py)
            timeout = min(timeout, CLI_STEP_TIMEOUT)
        # On Claude Code the reply is streamed and the step ends once an action is complete.
        until = _complete if resolve_backend(self.backend) == "claude-code" else None
        answer = _send(self._transcript(), self.system, self.backend, self.model, timeout, self.effort, until)
        try:
            calls, self.note = read_actions(answer, self.names)
        except ValueError as problem:
            return Reply([], answer, str(problem))
        return Reply(calls, answer)

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        for n, (call, output, _) in enumerate(results, start=1):
            shown = json.dumps({"tool": call.tool, "args": call.args}, ensure_ascii=False)
            if n == len(results) and self.note:  # the run goes on, but the model is told to keep to JSON
                output = f"{output}\n\n(Note: {self.note})"
            self.steps.append(_step(len(self.steps) + 1, shown, output))
        self.note = None

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


# Told to the model with the results of a reply that wasn't only its action. Without it, a model
# that slips into tool-call markup is never corrected, and can fall into repeating empty markup.
EXTRA_TEXT = "Your reply had text besides the action, and only its first action was used. Reply with the action alone."
MARKUP = (
    "Your reply was tool-call markup, which this program reads only as well as it can. Write each action "
    'as JSON instead, like {"tool": "read", "args": {"path": "README.md"}}, and nothing else.'
)


def actions(answer: str, names: Sequence[str]) -> list[Call]:
    """A text-protocol reply as its calls: one JSON action, or a JSON array of them, carried out in
    order. Raises ValueError, with a reason the model can act on, for a reply with no usable call. In
    an array, an action that isn't valid becomes a call with a problem: it fails, the others run."""
    return read_actions(answer, names)[0]


def read_actions(answer: str, names: Sequence[str]) -> tuple[list[Call], str | None]:
    """actions(), and a note for the model when the reply wasn't only the action but was read anyway:
    the first complete action in it, whatever prose, code fence or made-up results surround it
    (literal newlines in its strings accepted), or else tool calls written as <invoke> markup."""
    note = None
    try:
        value = parse(answer, dict[str, Any] | list[Any])  # reads code fences and <think> blocks like any answer
    except ValueError as problem:
        found = first_action(answer)
        invoked = _invoked(answer) if found is None else None
        if found is not None:
            value, note = found[0], EXTRA_TEXT
        elif invoked:
            value, note = (invoked[0] if len(invoked) == 1 else invoked), MARKUP
        else:
            raise problem from None
    return _calls(value, names), note


def _calls(value: Any, names: Sequence[str]) -> list[Call]:
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


_ACTION_START = re.compile(r'(\[\s*)?\{\s*"tool"\s*:')
_INVOKE = re.compile(r'<invoke name="([^"<>]+)">(.*?)</invoke>', re.DOTALL)
_PARAMETER = re.compile(r'<parameter name="([^"<>]+)">(.*?)</parameter>', re.DOTALL)


def first_action(text: str, final: bool = True) -> tuple[Any, int] | None:
    """The first complete action (an object with a "tool"), or array of them, in `text`, and where
    it ends; None if there is none. Literal newlines in its strings are accepted, as models write
    file contents. While the reply is still arriving (final=False), an array that has begun is
    waited for rather than read as its first action."""
    decoder = json.JSONDecoder(strict=False)
    for match in _ACTION_START.finditer(text):
        try:
            value, end = decoder.raw_decode(text, match.start())
        except (ValueError, RecursionError):
            if match.group(1) and not final:
                return None
            continue
        items = value if isinstance(value, list) else [value]
        if items and all(isinstance(item, dict) and isinstance(item.get("tool"), str) for item in items):
            return value, end
    return None


def _complete(reply: str) -> bool:
    """Whether a reply still arriving holds a complete action, so the step can stop there: a JSON
    action, or two blocks of tool-call markup (one may be followed by the JSON action; more are
    usually the model repeating itself, which it would do until the step times out)."""
    return first_action(reply, final=False) is not None or len(_INVOKE.findall(reply)) >= 2


def _invoked(text: str) -> list[dict[str, Any]]:
    """Tool calls written as markup, <invoke name="read"><parameter name="path">a.py</parameter>
    </invoke>, as actions. A built-in tool's text arguments are taken as they are; its other
    arguments, and a custom tool's, are read as JSON where they parse (5, true, a list)."""
    found = []
    for tool, body in _INVOKE.findall(text):
        kinds = builtin_tools.TOOLS[tool][1] if tool in builtin_tools.TOOLS else {}
        args: dict[str, Any] = {}
        for name, raw in _PARAMETER.findall(body):
            if tool == "finish" or kinds.get(name, (None,))[0] is str:
                args[name] = raw  # finish reads a value written as text against the task's type itself
                continue
            try:
                args[name] = json.loads(raw)
            except ValueError:
                args[name] = raw
        if set(args) == {"args"} and isinstance(args["args"], dict) and "args" not in kinds:
            args = args["args"]  # all its arguments as one JSON object, as in a text-protocol action
        found.append({"tool": tool, "args": args})
    return found


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


MAX_TOKENS = 64_000  # per Claude API reply; streamed, so a long write isn't cut off by an HTTP timeout
CONTINUATIONS = 5  # pause_turn replies in a row before the run fails
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CLEARING_BETA = "context-management-2025-06-27"
CUT_OFF = (
    "Your reply was cut off at max_tokens, so its tool call wasn't run: its arguments were incomplete. "
    "Write a large file in parts: write the start, then add the rest with edit."
)
# Claude 4.6 and later: they take effort, which thunc sets to high for agents (Claude Opus 5.5 defaults
# to medium), and clearing old tool results on long runs. Other models get the API's defaults unless
# effort= is given.
_CURRENT = (
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-4-6", "claude-sonnet-5",
    "claude-fable-", "claude-mythos-",
)  # fmt: skip
# Models that take strict tool schemas (arguments guaranteed to match).
_STRICT = (
    "claude-opus-4-1", "claude-opus-4-5", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5",
    "claude-fable-5", "claude-mythos-5",
)  # fmt: skip
# Schema keywords strict mode doesn't take (numeric, string and array constraints).
_NOT_STRICT = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength"}
    | {"pattern", "minItems", "maxItems", "uniqueItems"}
)


def strict_schema(schema: Any) -> bool:
    """Whether a tool's input schema can be strict: every object closed (additionalProperties false),
    no constraint strict mode leaves out, and no recursion ($ref)."""
    if isinstance(schema, list):
        return all(strict_schema(item) for item in schema)
    if not isinstance(schema, dict):
        return True
    if _NOT_STRICT & set(schema) or "$ref" in schema:
        return False
    if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
        return False
    return all(strict_schema(value) for key, value in schema.items() if key not in ("enum", "const"))


class AnthropicConversation:
    """Messages API tool use. The fixed system prompt is one block marked for caching and the
    memory a second block after it, so a changed memory doesn't undo the cache; automatic caching
    covers the growing conversation. Replies are appended unchanged, thinking blocks included, and
    nothing earlier is ever edited, so the history stays valid for preserved thinking.

    Replies are streamed, with room for a large write. A reply cut off at max_tokens isn't the end
    of the run: its calls are answered with an error saying so, once in a row. pause_turn is asked
    to carry on. On current models, effort defaults to high and old tool results are cleared by the
    API on long runs (context editing), and the tools are strict where their schemas allow."""

    def __init__(
        self, fixed: str, memory: str, request: str, tools: Sequence[Tool], model: str | None, effort: str | None = None
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise ThuncError("The anthropic backend needs the SDK: pip install 'thunc[anthropic]'") from exc
        self._anthropic = anthropic
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = sdk_client(anthropic.Anthropic, "ANTHROPIC_", setting("api_key"), setting("timeout"))
        self.model = model or setting("model") or DEFAULT_ANTHROPIC_MODEL
        current = self.model.startswith(_CURRENT)
        self.effort = effort or ("high" if current else None)
        self.clearing = current
        strict = self.model.startswith(_STRICT)
        self.system: list[Any] = [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}]
        if memory:
            self.system.append({"type": "text", "text": memory})
        self.tools: list[Any] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema,
                **({"strict": True} if strict and strict_schema(t.schema) else {}),
            }
            for t in tools
        ]
        self.messages: list[Any] = [{"role": "user", "content": request}]
        self.cut_off: list[str] | None = None  # the calls of a reply cut off at max_tokens, until answered
        self.cut_offs = 0  # replies cut off in a row

    def next(self) -> Reply:
        for _ in range(CONTINUATIONS + 1):
            response = self._create()
            if response.stop_reason == "refusal":
                raise ThuncError("The model declined this request.")
            if response.stop_reason == "model_context_window_exceeded":
                raise ThuncError("The run outgrew the model's context window.")
            if response.stop_reason not in ("tool_use", "end_turn", "stop_sequence", "max_tokens", "pause_turn"):
                raise ThuncError(f"The Claude API stopped with {response.stop_reason!r}.")
            self.messages.append({"role": "assistant", "content": response.content})  # unchanged, thinking too
            if response.stop_reason != "pause_turn":  # paused: asked again with no new message, it carries on
                break
        else:
            raise ThuncError(f"The Claude API paused the turn {CONTINUATIONS + 1} times in a row.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "max_tokens":
            self.cut_offs += 1
            if self.cut_offs > 1:
                raise ThuncError(f"Two replies in a row were cut off at max_tokens={MAX_TOKENS}.")
            self.cut_off = [block.id for block in response.content if block.type == "tool_use"]
            return Reply([], text, "the reply was cut off at max_tokens")
        self.cut_offs = 0
        calls = [
            Call(block.id, block.name, block.input if isinstance(block.input, dict) else {})
            for block in response.content
            if block.type == "tool_use"
        ]
        if not calls:
            return Reply([], text, "no tool call")
        return Reply(calls, _calls_text(calls))

    def _create(self) -> Any:
        anthropic = self._anthropic
        request: dict[str, Any] = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            cache_control={"type": "ephemeral"},
        )
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        betas = []
        if self.model in _FALLBACK_MODELS:
            betas.append(FALLBACK_BETA)
            request["fallbacks"] = "default"
        if self.clearing:
            betas.append(CLEARING_BETA)
            request["context_management"] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        try:
            if betas:
                return _streamed(self.client.beta.messages.stream(**request, betas=betas), setting("timeout"))
            return _streamed(self.client.messages.stream(**request), setting("timeout"))
        except anthropic.APIConnectionError as exc:
            raise TransientError(f"Could not reach the Claude API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            error = TransientError if transient_claude_error(exc.status_code, exc.body) else ThuncError
            raise error(f"Claude API error {exc.status_code}: {exc.message}") from exc

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        # Every result in one message: split up, it teaches the model to stop calling tools in parallel.
        content = [
            {"type": "tool_result", "tool_use_id": call.id, "content": output, **({"is_error": True} if failed else {})}
            for call, output, failed in results
        ]
        self.messages.append({"role": "user", "content": content})

    def nudge(self, reply: Reply) -> None:
        if self.cut_off is None:
            self.messages.append({"role": "user", "content": NUDGE})
            return
        # Each call of the cut-off reply needs its result, in the message right after it.
        content: list[Any] = [
            {"type": "tool_result", "tool_use_id": ident, "content": CUT_OFF, "is_error": True}
            for ident in self.cut_off
        ]
        content.append(
            {"type": "text", "text": NUDGE if self.cut_off else f"Your reply was cut off at max_tokens. {NUDGE}"}
        )
        self.messages.append({"role": "user", "content": content})
        self.cut_off = None


def _streamed(manager: Any, quiet: float) -> Any:
    """The final message of a streamed reply. A reply may stream for minutes (a large write), but one
    that sends nothing for `quiet` seconds is stopped and raises TransientError, to be asked again.
    The SDK's own read timeout doesn't catch it: the API's keep-alive pings count as reading. In the
    tool-use benchmark, one reply on Claude Opus 5.5 sent nothing for an hour."""
    with manager as stream:
        last = [time.monotonic()]
        box: dict[str, Any] = {}

        def read() -> None:
            try:
                for _ in stream:  # pings aren't events: only the reply itself counts
                    last[0] = time.monotonic()
                box["message"] = stream.get_final_message()
            except BaseException as exc:  # raised again below, in the run's thread
                box["error"] = exc

        reader = threading.Thread(target=read, name="thunc-claude-stream", daemon=True)
        reader.start()
        while reader.is_alive():
            reader.join(min(1.0, quiet))
            if reader.is_alive() and time.monotonic() - last[0] > quiet:
                stream.close()  # ends the read; the thread's error goes nowhere
                raise TransientError(f"The Claude API's reply stalled: nothing arrived for {quiet:g}s.")
        if "error" in box:
            raise box["error"]
        return box["message"]


# --- the OpenAI Responses API -------------------------------------------------------------------


class OpenAIConversation:
    """Responses API function calling. With store=False nothing is kept on OpenAI's side, so every
    request carries the whole conversation, reasoning items included (encrypted)."""

    def __init__(self, system: str, request: str, tools: Sequence[Tool], model: str | None, effort: str | None = None):
        if effort == "max":
            raise ThuncError("The openai backend takes effort low, medium, high or xhigh, not max.")
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
        self.reasoning: Any = {"effort": effort} if effort else openai.omit

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
                reasoning=self.reasoning,
            )
        except openai.APIConnectionError as exc:
            raise TransientError(f"Could not reach the OpenAI API: {exc}") from exc
        except openai.APIStatusError as exc:
            error = TransientError if transient_status(exc.status_code) else ThuncError
            raise error(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
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
