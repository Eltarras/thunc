"""thunc.testing: the fake model, the agent helpers, isolated(), and the pytest fixtures."""

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pytest

import thunc
from thunc import testing


class Priority(Enum):
    LOW = 1
    HIGH = 3


@dataclass
class Item:
    name: str
    quantity: int


@thunc.function
def urgency(ticket: str) -> int:
    """Rate how urgent this is, from 1 to 5."""
    ...


@thunc.function
def category(ticket: str) -> Literal["bug", "billing", "other"]:
    """Classify this ticket."""
    ...


@thunc.function
def priority(ticket: str) -> Priority:
    """How urgent?"""
    ...


@thunc.function
def order(text: str) -> list[Item]:
    """Read the order."""
    ...


def test_replies_by_function_and_values_as_json():
    with testing.fake_model({"urgency": 4, "category": "bug", "priority": Priority.HIGH}) as model:
        model.reply([Item("latte", 3)], function="order")
        assert category("charged twice") == "bug"
        assert urgency("site down") == 4
        assert priority("site down") is Priority.HIGH
        assert order("3 lattes") == [Item("latte", 3)]
    first = model.calls[0]
    assert first.function == "category" and first.inputs == {"ticket": "charged twice"} and first.n == 1
    assert first.returns == Literal["bug", "billing", "other"] and "<ticket>" in first.prompt
    assert "You are a function" in first.system and first.agent is False


def test_replies_in_order_a_bad_one_is_retried_like_a_real_one():
    with testing.fake_model(["high", "4", "Paris"]) as model:
        assert urgency("x") == 4
        assert thunc.call("Capital of France?") == "Paris"
    first, retry, _ = model.calls
    assert (first.n, retry.n) == (1, 2) and "invalid" in retry.prompt


def test_a_function_reply_sees_the_request():
    with testing.fake_model(lambda request: len(request.inputs["ticket"])):
        assert thunc.map(urgency, ["a", "abc"]) == [1, 3]
    with testing.fake_model({"urgency": lambda request: 5}) as model:
        assert urgency("x") == 5 and urgency("y") == 5
    assert [r.inputs for r in model.requests_for("urgency")] == [{"ticket": "x"}, {"ticket": "y"}]


def test_running_out_of_replies_fails_with_what_was_asked():
    with testing.fake_model(), pytest.raises(AssertionError, match=r"no reply for urgency \(request 1"):
        urgency("x")


def test_every_backend_and_the_recordings_give_way_to_it(monkeypatch, tmp_path):
    thunc.configure(recordings=str(tmp_path / "recordings"))  # replaying, which would refuse a miss
    with testing.fake_model({"spam": True}):
        assert thunc.call("Spam?", returns=bool, backend="anthropic", name="spam") is True
    assert not (tmp_path / "recordings").exists()
    with pytest.raises(thunc.ThuncError, match="No recorded answer"):  # back to replaying after
        thunc.call("Spam?", returns=bool, name="spam")


def test_async_calls_and_amap_use_it():
    with testing.fake_model({"urgency": [1, 2, 3]}):
        assert asyncio.run(thunc.acall("x", {"t": 1}, returns=int, name="urgency")) == 1
        assert sorted(asyncio.run(thunc.amap(urgency, ["a", "b"]))) == [2, 3]


def test_an_agent_is_scripted_with_act_and_finish(tmp_path):
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    agent = thunc.Agent("fixer", workdir=tmp_path, permissions=["write:config.py"], protocol="native")

    @agent.task
    def double_timeout() -> int:
        """Double TIMEOUT in config.py and return the new value."""
        ...

    steps = [
        testing.act("read", path="config.py"),
        testing.act("edit", path="config.py", old="TIMEOUT = 30", new="TIMEOUT = 60"),
        testing.finish(60),
    ]
    with testing.fake_model({"fixer.double_timeout": steps}) as model:
        run = agent.run(double_timeout)
    assert run.value == 60 and run.files_changed == ["config.py"]
    assert (tmp_path / "config.py").read_text() == "TIMEOUT = 60\n"
    assert [r.n for r in model.calls] == [1, 2, 3] and all(r.agent for r in model.calls)
    assert "TIMEOUT = 30" in model.calls[1].prompt  # the read's result went back to the model


def test_fake_models_dont_nest_and_the_backend_comes_back():
    thunc.configure(backend="openai")
    with testing.fake_model():
        with pytest.raises(RuntimeError, match="already active"), testing.fake_model():
            pass
    assert thunc.config.resolve_backend() == "openai"
    assert testing.FAKE not in thunc.backends.BACKENDS


def test_as_text():
    assert testing.as_text("4") == "4" and testing.as_text(4) == "4" and testing.as_text(True) == "true"
    assert json.loads(testing.as_text({"items": [Item("a", 1)], "p": Priority.LOW})) == {
        "items": [{"name": "a", "quantity": 1}],
        "p": 1,
    }


def test_isolated_puts_settings_and_environment_back(monkeypatch):
    monkeypatch.delenv("THUNC_RECORD", raising=False)
    thunc.configure(model="before")
    with testing.isolated():
        thunc.configure(model="inside", backend="openai")
        os.environ["THUNC_RECORD"] = "1"
    assert thunc.config.setting("model") == "before" and thunc.config._settings["backend"] is None
    assert "THUNC_RECORD" not in os.environ


USER_TESTS = """
import thunc


@thunc.function
def urgency(ticket: str) -> int:
    \"\"\"Rate it.\"\"\"
    ...


def test_a_fake_model_for_the_test(thunc_model):
    thunc_model.reply({"urgency": 4})
    assert urgency("site down") == 4


def test_settings_changed(thunc_isolated):
    thunc.configure(model="only-in-this-test")


def test_nothing_leaked():
    assert thunc.config.setting("model") is None
    assert thunc.config.forced_backend is None
"""


def test_the_fixtures_load_with_thunc_and_clean_up(tmp_path):
    """In a separate pytest, as a user's: the fixtures come from thunc's pytest11 entry point."""
    import subprocess
    import sys

    (tmp_path / "test_user.py").write_text(USER_TESTS)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_user.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("THUNC_")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
