"""agent.conversation(): several tasks in one conversation with the model (experimental)."""

import json
import sys
import threading
import time

import pytest
from test_claude_code_agent import calls, cli, native_calls_not_ruled_out, repo  # noqa: F401 - fixtures

import thunc
from thunc import native
from thunc.testing import act, fake_model, finish


def test_the_next_task_continues_the_conversation(tmp_path):
    agent = thunc.Agent("helper", workdir=tmp_path)
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    with fake_model([act("read", path="config.py"), finish(30), finish(60)]) as model:
        with agent.conversation(returns=int) as chat:
            assert not chat.going_on
            first = chat.send("Find the timeout, in seconds.")
            assert chat.going_on
            second = chat.send("Double it.", {"unit": "seconds"})
    assert (first.value, second.value) == (30, 60)
    assert [r.value for r in chat.runs] == [30, 60]
    last = model.calls[-1].prompt  # the text protocol sends the whole conversation each step
    assert "Find the timeout" in last and "TIMEOUT = 30" in last  # the first task and what it read
    assert native.NEXT_TASK in last and "Double it." in last and "<unit>" in last
    assert native.FINISHED in last  # the first finish was answered before the next task
    with open(second.session, encoding="utf-8") as f:
        start = json.loads(f.readline())
    assert start["continues"] == first.session and start["task"] == "conversation"


def test_a_failed_task_ends_the_conversation(tmp_path):
    agent = thunc.Agent("helper", workdir=tmp_path, retries=0)
    with fake_model([finish(1), finish("not a number"), finish(3)]) as model:
        chat = agent.conversation(returns=int)
        assert chat.send("One.").value == 1
        with pytest.raises(thunc.AgentError) as failed:
            chat.send("Two.")
        assert not chat.going_on and failed.value.run.error and len(chat.runs) == 2
        third = chat.send("Three.")  # a new conversation: the earlier tasks aren't in it
        chat.close()
    assert third.value == 3
    assert "One." not in model.calls[-1].prompt and "Three." in model.calls[-1].prompt
    with pytest.raises(thunc.ThuncError, match="closed"):
        chat.send("Four.")


def test_one_task_at_a_time(tmp_path):
    agent = thunc.Agent("helper", workdir=tmp_path)
    started, release = threading.Event(), threading.Event()

    def slow(request):
        started.set()
        release.wait(10)
        return finish("done")

    with fake_model(slow):
        chat = agent.conversation()
        worker = threading.Thread(target=chat.send, args=("Slow.",))
        worker.start()
        assert started.wait(10)
        with pytest.raises(thunc.ThuncError, match="one task at a time"):
            chat.send("Another.")
        release.set()
        worker.join(10)
        chat.close()


def test_not_with_recordings(tmp_path, monkeypatch):
    monkeypatch.setenv("THUNC_RECORDINGS", str(tmp_path / "rec"))
    chat = thunc.Agent("helper", workdir=tmp_path).conversation()
    with pytest.raises(thunc.ThuncError, match="recordings"):
        chat.send("Hi.")


def test_an_unsupported_return_type_fails_at_once(tmp_path):
    with pytest.raises((TypeError, ValueError, thunc.ThuncError)):
        thunc.Agent("helper", workdir=tmp_path).conversation(returns=object)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a script with a shebang")
def test_on_claude_code_the_same_cli_process_takes_the_next_task(cli, repo):  # noqa: F811
    cli(
        calls(("read", {"path": "config.py"})),
        calls(("finish", {"value": 30})),
        {"text": "Done."},  # the turn that the answer to finish ends
        calls(("finish", {"value": 60})),
        {"text": "Done."},
    )
    agent = thunc.Agent("guide", workdir=repo)
    with agent.conversation(returns=int) as chat:
        first = chat.send("Find the timeout, in seconds.")
        second = chat.send("Double it.")
        log = settled(cli)  # read before close() stops the CLI, which may be rewriting its log
    assert (first.value, second.value) == (30, 60)
    # one process: both tasks reached the same CLI
    assert len(log["messages"]) == 2
    assert "Find the timeout" in log["messages"][0]
    assert log["messages"][1].startswith(native.NEXT_TASK) and "Double it." in log["messages"][1]
    assert native.FINISHED in [r["text"] for r in log["results"]]


def settled(fake, seconds=10):
    """The fake CLI's log once it has both answers to finish."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            log = fake.log
            if [r["text"] for r in log["results"]].count(native.FINISHED) == 2:
                return log
        except (ValueError, OSError):
            pass  # being written
        assert time.monotonic() < deadline, "the CLI didn't get both answers to finish"
        time.sleep(0.05)


def test_anthropic_follow_up_answers_the_last_reply_with_the_next_task(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setattr(native, "sdk_client", lambda *args, **kwargs: object())
    conversation = native.AnthropicConversation("fixed", "", "first task", [], "claude-opus-5-5")
    conversation.messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": "t1"}]})
    conversation.finished([(native.Call("t1", "finish", {"value": 1}), native.FINISHED, False)])
    assert len(conversation.messages) == 2  # nothing is sent until the next task
    conversation.follow_up("second task")
    last = conversation.messages[-1]
    assert last["role"] == "user" and last["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": native.FINISHED,
    }
    assert last["content"][1]["type"] == "text" and last["content"][1]["text"].endswith("second task")


def test_openai_follow_up(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setattr(native, "sdk_client", lambda *args, **kwargs: object())
    conversation = native.OpenAIConversation("system", "first task", [], "gpt-5.5")
    conversation.finished([(native.Call("c1", "finish", {"value": 1}), native.FINISHED, False)])
    conversation.follow_up("second task")
    assert conversation.input[-2] == {"type": "function_call_output", "call_id": "c1", "output": native.FINISHED}
    assert conversation.input[-1]["role"] == "user" and conversation.input[-1]["content"].endswith("second task")
