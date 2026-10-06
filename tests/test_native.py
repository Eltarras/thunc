"""Agents on the Claude and OpenAI APIs' own tool calls. The SDKs are real; only their clients are
replaced by scripted ones that record each request, so the requests and replies have the real shapes."""

import json
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import anthropic
import openai
import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem

import thunc
from thunc import native


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "config.py").write_text("NAME = 'demo'\nTIMEOUT = 30\n")
    return tmp_path


# --- Claude API -----------------------------------------------------------------------------


def message(*blocks, stop="tool_use"):
    return anthropic.types.Message.model_validate(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": list(blocks),
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def use(tool, id, **args):
    return {"type": "tool_use", "id": id, "name": tool, "input": args}


THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}


@pytest.fixture
def claude(monkeypatch):
    """A scripted Messages API. Append replies to .replies; each request is in .requests."""
    script = SimpleNamespace(replies=[], requests=[])

    class Stream:  # what messages.stream() returns: a context manager with the final message
        def __init__(self, reply):
            self.reply = reply
            self.closed = threading.Event()

        def __enter__(self):
            if isinstance(self.reply, Exception):
                raise self.reply
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            if self.reply == "stall":  # nothing arrives until the stream is closed
                self.closed.wait(30)
                raise anthropic.APIConnectionError(request=None)
            return iter(())

        def close(self):
            self.closed.set()

        def get_final_message(self):
            return self.reply

    class Client:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self.create, stream=self.stream)
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create, stream=self.stream))

        def with_options(self, **options):
            return self

        def create(self, **kwargs):
            script.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.model_dump())))
            return script.replies.pop(0)

        def stream(self, **kwargs):
            script.requests.append({**json.loads(json.dumps(kwargs, default=lambda o: o.model_dump())), "stream": True})
            return Stream(script.replies.pop(0))

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    thunc.configure(backend="anthropic")
    return script


def test_claude_tool_calls_end_to_end(claude, repo):
    @dataclass
    class Setting:
        name: str
        seconds: int

    agent = thunc.Agent("n", workdir=repo, permissions=["write:notes.md"])

    @agent.task
    def timeout() -> Setting:
        """Find the timeout."""
        ...

    claude.replies = [
        message(
            THINKING, {"type": "text", "text": "Reading both."}, use("read", "t1", path="config.py"), use("list", "t2")
        ),
        message(use("write", "t3", path="config.py", content="x"), use("write", "t4", path="notes.md", content="ok")),
        message(use("finish", "t5", value={"name": "TIMEOUT", "seconds": 30})),
    ]
    run = agent.run(timeout)
    assert run.value == Setting("TIMEOUT", 30) and run.steps == 3
    assert run.files_changed == ["notes.md"] and run.denied[0].target == "config.py"

    first, second, third = claude.requests
    # The fixed prompt is one cached block; the tools are the API's own definitions.
    (system,) = first["system"]
    assert system["cache_control"] == {"type": "ephemeral"} and first["cache_control"] == {"type": "ephemeral"}
    assert "How to use a tool: reply with exactly one JSON object" not in system["text"]
    assert "Work through the task with your tools" in system["text"] and "- Write: notes.md." in system["text"]
    tools = {t["name"]: t for t in first["tools"]}
    assert set(tools) == {"list", "read", "search", "write", "edit", "remember", "finish"}  # no run rule: no run
    assert tools["read"]["input_schema"]["required"] == ["path"]
    assert tools["finish"]["input_schema"]["properties"]["value"]["required"] == ["name", "seconds"]
    assert not tools["read"]["description"].startswith("{")  # the text-protocol example is left out
    # Strict where the schema allows it: finish's dataclass object isn't closed, so it isn't strict.
    assert tools["read"]["strict"] is True and tools["remember"]["strict"] is True and "strict" not in tools["finish"]
    assert "strict" not in tools["edit"]  # maxItems on edits is a constraint strict mode leaves out
    # Opus 5.5 gets the server-side refusal fallback and old tool results cleared on long runs (both
    # beta), effort high (its own default is medium), and a streamed reply with room for a large write.
    assert first["betas"] == ["server-side-fallback-2026-07-01", "context-management-2025-06-27"]
    assert first["fallbacks"] == "default"
    assert first["context_management"] == {"edits": [{"type": "clear_tool_uses_20250919"}]}
    assert first["output_config"] == {"effort": "high"}
    assert first["max_tokens"] == 64_000 and first["stream"] is True
    assert "tool_choice" not in first  # forced tool choice is a 400 on current models

    # The reply goes back unchanged, thinking block included; both results in one message.
    assistant, results = second["messages"][1], second["messages"][2]
    assert [b["type"] for b in assistant["content"]] == ["thinking", "text", "tool_use", "tool_use"]
    assert [r["tool_use_id"] for r in results["content"]] == ["t1", "t2"]
    assert "TIMEOUT = 30" in results["content"][0]["content"] and "is_error" not in results["content"][0]
    denied, ok = third["messages"][4]["content"]
    assert denied["is_error"] is True and "not permitted" in denied["content"] and "is_error" not in ok


def test_claude_memory_is_a_second_system_block_after_the_cached_one(claude, repo):
    agent = thunc.Agent("m", workdir=repo)
    task_reply = [
        message(use("remember", "t1", note="Timeout is in config.py.")),
        message(use("finish", "t2", value="a")),
    ]
    claude.replies = [*task_reply, message(use("finish", "t3", value="b"))]

    @agent.task
    def anything() -> str:
        """Do it."""
        ...

    anything(), anything()
    first_run, second_run = claude.requests[0], claude.requests[2]
    assert len(first_run["system"]) == 1
    fixed, memory = second_run["system"]
    assert fixed == first_run["system"][0]  # the cached part is unchanged by the new note
    assert "cache_control" not in memory and "Timeout is in config.py." in memory["text"]


def test_claude_reply_without_a_tool_call_is_nudged(claude, repo):
    claude.replies = [
        message({"type": "text", "text": "The timeout is 30."}, stop="end_turn"),
        message(use("finish", "t1", value=30)),
    ]

    @thunc.Agent("x", workdir=repo).task
    def timeout() -> int:
        """Find the timeout."""
        ...

    assert timeout() == 30
    assert claude.requests[1]["messages"][-1] == {"role": "user", "content": native.NUDGE}


def test_claude_invalid_finish_is_an_error_result(claude, repo):
    claude.replies = [message(use("finish", "t1", value="thirty")), message(use("finish", "t2", value=30))]

    @thunc.Agent("x", workdir=repo).task
    def timeout() -> int:
        """Find the timeout."""
        ...

    assert timeout() == 30
    (result,) = claude.requests[1]["messages"][2]["content"]
    assert result["is_error"] is True and "that value is invalid" in result["content"]


@pytest.mark.parametrize(
    "stop, problem",
    [
        ("refusal", "declined"),
        ("model_context_window_exceeded", "outgrew the model's context window"),
    ],
)
def test_claude_stops_that_end_the_run(claude, repo, stop, problem):
    claude.replies = [message({"type": "text", "text": "..."}, stop=stop)]

    @thunc.Agent("x", workdir=repo).task
    def timeout() -> int:
        """Find the timeout."""
        ...

    with pytest.raises(thunc.AgentError, match=problem):
        timeout()


def test_claude_models_without_the_fallback_use_the_plain_endpoint(claude, repo):
    claude.replies = [message(use("finish", "t1", value="ok"))]
    assert _task(thunc.Agent("x", workdir=repo, model="claude-haiku-4-5"))() == "ok"
    (request,) = claude.requests
    assert "betas" not in request and request["model"] == "claude-haiku-4-5" and request["stream"] is True
    # Before Claude 4.6: no effort or clearing unless asked for. Haiku 4.5 does take strict tools.
    assert "output_config" not in request and "context_management" not in request
    assert {t["name"]: t.get("strict") for t in request["tools"]}["read"] is True


def test_claude_effort_given_is_sent_to_any_model(claude, repo):
    claude.replies = [message(use("finish", "t1", value="ok")), message(use("finish", "t2", value="ok"))]
    _task(thunc.Agent("x", workdir=repo, effort="max"))()
    _task(thunc.Agent("y", workdir=repo, model="claude-opus-4-5", effort="low"))()
    assert claude.requests[0]["output_config"] == {"effort": "max"}
    assert claude.requests[1]["output_config"] == {"effort": "low"} and "betas" not in claude.requests[1]
    with pytest.raises(ValueError, match="effort= is one of low, medium, high, xhigh, max"):
        thunc.Agent("z", workdir=repo, effort="extreme")


def test_claude_a_call_cut_off_at_max_tokens_gets_an_error_result(claude, repo):
    claude.replies = [
        message(use("write", "t1", path="big.md", content="half of it"), stop="max_tokens"),
        message(use("finish", "t2", value="ok")),
    ]
    assert _task(thunc.Agent("x", workdir=repo, permissions=["write"]))() == "ok"
    assert not (repo / "big.md").exists()  # the cut-off call isn't run
    sent = claude.requests[1]["messages"]
    assert sent[1]["content"][0]["type"] == "tool_use"  # the cut-off reply goes back unchanged
    result, text = sent[2]["content"]
    assert result == {"type": "tool_result", "tool_use_id": "t1", "content": native.CUT_OFF, "is_error": True}
    assert text == {"type": "text", "text": native.NUDGE}


def test_claude_a_text_reply_cut_off_at_max_tokens_is_nudged(claude, repo):
    claude.replies = [
        message({"type": "text", "text": "Long..."}, stop="max_tokens"),
        message(use("finish", "t1", value="ok")),
    ]
    assert _task(thunc.Agent("x", workdir=repo))() == "ok"
    (text,) = claude.requests[1]["messages"][2]["content"]
    assert text["text"].startswith("Your reply was cut off at max_tokens.")


def test_claude_two_replies_cut_off_in_a_row_end_the_run(claude, repo):
    cut = message({"type": "text", "text": "Long..."}, stop="max_tokens")
    claude.replies = [cut, cut]
    with pytest.raises(thunc.AgentError, match="Two replies in a row were cut off at max_tokens"):
        _task(thunc.Agent("x", workdir=repo))()


def test_claude_a_paused_turn_is_asked_to_carry_on(claude, repo):
    claude.replies = [
        message({"type": "text", "text": "Working"}, stop="pause_turn"),
        message(use("finish", "t1", value="ok")),
    ]
    agent = thunc.Agent("x", workdir=repo)
    run = agent.run(_task(agent))
    assert run.value == "ok" and run.steps == 1  # one reply, paused once on the way
    first, second = claude.requests
    assert second["messages"][-1]["role"] == "assistant"  # no new message: the API carries on from the pause


def test_claude_pausing_again_and_again_ends_the_run(claude, repo):
    claude.replies = [message({"type": "text", "text": "Working"}, stop="pause_turn")] * 6
    with pytest.raises(thunc.AgentError, match="paused the turn 6 times in a row"):
        _task(thunc.Agent("x", workdir=repo))()


def test_claude_overloaded_and_unreachable_are_retried(claude, repo, monkeypatch):
    import httpx2

    monkeypatch.setattr(sys.modules["thunc.agent"], "RETRY_DELAY", 0)  # thunc.agent is also a function
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    overloaded = anthropic.APIStatusError("Overloaded", response=httpx2.Response(529, request=request), body=None)
    claude.replies = [
        overloaded,
        anthropic.APIConnectionError(request=request),
        message(use("finish", "t1", value="ok")),
    ]
    agent = thunc.Agent("x", workdir=repo)
    run = agent.run(_task(agent))
    assert run.value == "ok"
    with open(run.session, encoding="utf-8") as f:
        retries = [json.loads(line) for line in f if '"retry"' in line]
    assert [r["error"][:20] for r in retries] == ["Claude API error 529", "Could not reach the "]


def test_claude_a_bad_request_is_not_retried(claude, repo):
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    bad = anthropic.APIStatusError("Bad request", response=httpx2.Response(400, request=request), body=None)
    claude.replies = [bad]
    with pytest.raises(thunc.AgentError, match="Claude API error 400"):
        _task(thunc.Agent("x", workdir=repo))()


@pytest.mark.parametrize(
    "schema, strict",
    [
        ({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}, True),
        ({"type": "object", "properties": {"a": {"type": "string"}}}, False),  # not closed
        (
            {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "additionalProperties": False},
            False,
        ),
        (
            {
                "type": "object",
                "properties": {"v": {"type": "array", "items": {"type": "object"}}},
                "additionalProperties": False,
            },
            False,
        ),
        ({"type": "object", "properties": {"v": {"enum": [{"type": "object"}]}}, "additionalProperties": False}, True),
    ],
)
def test_which_tool_schemas_are_strict(schema, strict):
    assert native.strict_schema(schema) is strict


def _task(agent):
    @agent.task
    def anything() -> str:
        """Do it."""
        ...

    return anything


# --- OpenAI Responses API -------------------------------------------------------------------


def call(tool, id, arguments):
    return ResponseFunctionToolCall.model_validate(
        {
            "type": "function_call",
            "call_id": id,
            "name": tool,
            "arguments": arguments,
            "id": f"fc_{id}",
            "status": "completed",
        }
    )


REASONING = ResponseReasoningItem.model_validate(
    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"}
)


def text_item(text):
    return ResponseOutputMessage.model_validate(
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    )


def response(*items, text=""):
    return SimpleNamespace(status="completed", incomplete_details=None, output=list(items), output_text=text)


@pytest.fixture
def gpt(monkeypatch):
    script = SimpleNamespace(replies=[], requests=[])

    class Client:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)

        def with_options(self, **options):
            return self

        def create(self, **kwargs):
            sent = {k: v for k, v in kwargs.items() if v is not openai.omit}  # as the SDK leaves them out
            script.requests.append(json.loads(json.dumps(sent)))
            return script.replies.pop(0)

    monkeypatch.setattr(openai, "OpenAI", Client)
    thunc.configure(backend="openai")
    return script


def test_openai_function_calls_end_to_end(gpt, repo):
    gpt.replies = [
        response(REASONING, call("read", "c1", '{"path": "config.py"}'), call("search", "c2", "not json")),
        response(text_item("Done."), call("finish", "c3", '{"value": 30}')),
    ]

    @thunc.Agent("o", workdir=repo).task
    def timeout() -> int:
        """Find the timeout."""
        ...

    assert timeout() == 30
    first, second = gpt.requests
    assert first["store"] is False and first["include"] == ["reasoning.encrypted_content"]
    assert "How to use a tool" not in first["instructions"] and "Work through the task" in first["instructions"]
    tools = {t["name"]: t for t in first["tools"]}
    assert tools["read"]["type"] == "function" and tools["read"]["parameters"]["required"] == ["path"]
    assert tools["finish"]["parameters"]["properties"]["value"] == {"type": "integer"}
    # The whole conversation goes back: the reasoning item (encrypted), the calls, then both outputs.
    kinds = [item.get("type", item.get("role")) for item in second["input"]]
    assert kinds == [
        "user",
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert second["input"][1]["encrypted_content"] == "enc"
    read_out, bad_out = second["input"][4:]
    assert read_out["call_id"] == "c1" and "TIMEOUT = 30" in read_out["output"]
    assert bad_out["call_id"] == "c2" and "arguments weren't valid JSON" in bad_out["output"]


def test_openai_reply_without_a_call_is_nudged(gpt, repo):
    gpt.replies = [response(text_item("It's 30."), text="It's 30."), response(call("finish", "c1", '{"value": 30}'))]

    @thunc.Agent("o", workdir=repo).task
    def timeout() -> int:
        """Find the timeout."""
        ...

    assert timeout() == 30
    assert gpt.requests[1]["input"][-1] == {"role": "user", "content": native.NUDGE}


# --- choosing the protocol ------------------------------------------------------------------


def test_protocol_text_on_an_api_backend(gpt, repo, monkeypatch):
    from thunc import backends

    sent = []
    monkeypatch.setitem(
        backends.BACKENDS,
        "openai",
        lambda text, **kw: sent.append(kw["system"]) or '{"tool": "finish", "args": {"value": "ok"}}',
    )
    assert _task(thunc.Agent("o", workdir=repo, protocol="text"))() == "ok"
    assert "How to use a tool: reply with exactly one JSON object" in sent[0] and gpt.requests == []


def test_protocol_native_needs_an_api_backend(fake, repo):
    with pytest.raises(thunc.ThuncError, match="the fake backend has no native tool calls"):
        _task(thunc.Agent("x", workdir=repo, protocol="native"))()
    with pytest.raises(ValueError, match="protocol= is 'native', 'text' or None"):
        thunc.Agent("x", workdir=repo, protocol="json")


def lookup_order(order_id: str) -> dict[str, float]:
    """Look up an order's totals."""
    return {"total": 49.0}


def test_claude_custom_tool(claude, repo):
    agent = thunc.Agent("c", workdir=repo, tools=[lookup_order])
    claude.replies = [message(use("lookup_order", "t1", order_id="A-1")), message(use("finish", "t2", value=49.0))]

    @agent.task
    def total() -> float:
        """Find the total of order A-1."""
        ...

    assert total() == 49.0
    tool = next(t for t in claude.requests[0]["tools"] if t["name"] == "lookup_order")
    assert tool["description"] == "Look up an order's totals."
    assert tool["input_schema"] == {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    }
    (result,) = claude.requests[1]["messages"][2]["content"]
    assert result["content"] == '{"total": 49.0}' and "is_error" not in result


def test_anthropic_snapshot_preserves_thinking_and_tool_ids(claude):
    claude.replies = [message(THINKING, use("read", "t1", path="config.py")), message(use("finish", "t2", value="ok"))]
    original = native.AnthropicConversation("fixed", "memory", "request", [], None)
    reply = original.next()
    snapshot = native.snapshot(original)
    restored = native.AnthropicConversation("fixed", "memory", "request", [], None)
    native.restore(restored, json.loads(json.dumps(snapshot)))
    restored.results([(reply.calls[0], "file contents", False)])
    restored.next()
    assistant = claude.requests[1]["messages"][1]
    assert assistant["content"][0] == THINKING
    assert claude.requests[1]["messages"][2]["content"][0]["tool_use_id"] == "t1"


def test_openai_snapshot_preserves_encrypted_reasoning(gpt):
    gpt.replies = [
        response(REASONING, call("read", "c1", '{"path":"config.py"}')),
        response(call("finish", "c2", '{"value":"ok"}')),
    ]
    original = native.OpenAIConversation("fixed", "request", [], None)
    reply = original.next()
    restored = native.OpenAIConversation("fixed", "request", [], None)
    native.restore(restored, json.loads(json.dumps(native.snapshot(original))))
    restored.results([(reply.calls[0], "contents", False)])
    restored.next()
    assert gpt.requests[1]["input"][1]["encrypted_content"] == "enc"
    assert gpt.requests[1]["input"][-1]["call_id"] == "c1"


def test_openai_effort(gpt, repo):
    gpt.replies = [response(call("finish", "c1", '{"value": "ok"}')), response(call("finish", "c2", '{"value": "ok"}'))]
    _task(thunc.Agent("o", workdir=repo, effort="xhigh"))()
    _task(thunc.Agent("p", workdir=repo))()
    assert gpt.requests[0]["reasoning"] == {"effort": "xhigh"} and "reasoning" not in gpt.requests[1]
    with pytest.raises(thunc.ThuncError, match="openai backend takes effort low, medium, high or xhigh"):
        _task(thunc.Agent("q", workdir=repo, effort="max"))()


def test_claude_a_reply_that_stalls_is_stopped_and_asked_again(claude, repo, monkeypatch):
    monkeypatch.setattr(sys.modules["thunc.agent"], "RETRY_DELAY", 0)
    thunc.configure(timeout=1)
    claude.replies = ["stall", message(use("finish", "t1", value="ok"))]
    agent = thunc.Agent("x", workdir=repo)
    started = time.monotonic()
    run = agent.run(_task(agent))
    assert run.value == "ok" and time.monotonic() - started < 10
    with open(run.session, encoding="utf-8") as f:
        (retry,) = [json.loads(line) for line in f if '"retry"' in line]
    assert retry["error"] == "The Claude API's reply stalled: nothing arrived for 1s."
