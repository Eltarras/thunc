"""Agents on the Claude and OpenAI APIs' own tool calls. The SDKs are real; only their clients are
replaced by scripted ones that record each request, so the requests and replies have the real shapes."""

import json
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

    class Client:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self.create)
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            script.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.model_dump())))
            return script.replies.pop(0)

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
    # Opus 5.5 gets the server-side refusal fallback (the beta endpoint), as single calls do.
    assert first["betas"] == ["server-side-fallback-2026-07-01"] and first["fallbacks"] == "default"
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
    "stop, problem", [("refusal", "declined"), ("max_tokens", "cut off"), ("pause_turn", "stopped with")]
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
    assert "betas" not in claude.requests[0] and claude.requests[0]["model"] == "claude-haiku-4-5"


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

        def create(self, **kwargs):
            script.requests.append(json.loads(json.dumps(kwargs)))
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
