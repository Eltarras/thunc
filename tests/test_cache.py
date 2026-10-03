"""cache=True: answers saved on disk and reused, against the fake backend."""

import asyncio
import json
from typing import Literal

import pytest

import thunc


@pytest.fixture
def cache(tmp_path):
    folder = tmp_path / "cache"
    thunc.configure(cache_dir=str(folder))
    return folder


def test_second_call_is_answered_from_disk(fake, cache):
    fake.replies = ["4"]  # only one: a second model call would fail
    assert thunc.call("Rate 1-5.", {"ticket": "charged twice"}, returns=int, cache=True) == 4
    assert thunc.call("Rate 1-5.", {"ticket": "charged twice"}, returns=int, cache=True) == 4
    assert len(fake.prompts) == 1
    (entry,) = [json.loads(p.read_text()) for p in cache.glob("*.json")]
    assert entry["answer"] == "4" and entry["backend"] == "fake" and "charged twice" in entry["request"]


def test_off_by_default(fake, cache):
    fake.replies = ["a", "b"]
    assert thunc.call("Name a letter.") == "a"
    assert thunc.call("Name a letter.") == "b"
    assert not cache.exists()


def test_anything_the_model_would_see_differently_is_a_miss(fake, cache):
    fake.replies = ["1", "2", "3", "4", "5"]
    assert thunc.call("Rate 1-5.", {"ticket": "a"}, returns=int, cache=True) == 1
    assert thunc.call("Rate 1-5.", {"ticket": "b"}, returns=int, cache=True) == 2  # inputs
    assert thunc.call("Rate 1-10.", {"ticket": "a"}, returns=int, cache=True) == 3  # instructions
    assert thunc.call("Rate 1-5.", {"ticket": "a"}, returns=float, cache=True) == 4.0  # return type
    assert thunc.call("Rate 1-5.", {"ticket": "a"}, returns=int, model="other", cache=True) == 5  # model
    assert len(list(cache.glob("*.json"))) == 5


def test_only_the_valid_answer_is_saved_and_failures_are_not(fake, cache):
    fake.replies = ["four", "4"]
    assert thunc.call("Rate 1-5.", returns=int, cache=True) == 4
    (entry,) = [json.loads(p.read_text()) for p in cache.glob("*.json")]
    assert entry["answer"] == "4"

    fake.replies = ["?"] * 3
    with pytest.raises(thunc.ThuncError):
        thunc.call("Is it spam?", returns=bool, cache=True)
    assert len(list(cache.glob("*.json"))) == 1


def test_saved_answer_is_checked_again(fake, cache):
    fake.replies = ["4", "2"]
    assert thunc.call("Rate 1-5.", returns=int, cache=True) == 4
    # A stricter ensure= rejects the saved 4, so the model is asked and the new answer replaces it.
    assert thunc.call("Rate 1-5.", returns=int, ensure=lambda n: n <= 3, cache=True) == 2
    assert "rejected" not in fake.prompts[1]  # the stale answer isn't sent back as a mistake
    assert thunc.call("Rate 1-5.", returns=int, cache=True) == 2


def test_unreadable_entry_is_a_miss(fake, cache):
    fake.replies = ["4", "5"]
    thunc.call("Rate 1-5.", returns=int, cache=True)
    (path,) = cache.glob("*.json")
    path.write_text("{not json")
    assert thunc.call("Rate 1-5.", returns=int, cache=True) == 5


def test_function_and_async_function(fake, cache):
    @thunc.function(cache=True)
    def category(ticket: str) -> Literal["bug", "billing"]:
        """Classify this support ticket."""
        ...

    @thunc.function(cache=True)
    async def is_spam(email: str) -> bool:
        """Is this spam?"""
        ...

    fake.replies = ['"billing"', "false"]
    assert category("charged twice") == category("charged twice") == "billing"
    assert asyncio.run(is_spam("hello")) is asyncio.run(is_spam("hello")) is False
    assert len(fake.prompts) == 2


def test_cache_dir_from_environment(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("THUNC_CACHE_DIR", str(tmp_path / "env"))
    fake.replies = ["ok"]
    thunc.call("Say ok.", cache=True)
    assert len(list((tmp_path / "env").glob("*.json"))) == 1


def test_unwritable_cache_warns_but_returns(fake, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    thunc.configure(cache_dir=str(blocker / "cache"))  # a folder can't be made inside a file
    fake.replies = ["ok"]
    with pytest.warns(RuntimeWarning, match="could not save to the cache"):
        assert thunc.call("Say ok.", cache=True) == "ok"


def test_trace_marks_cached_calls(fake, cache, tmp_path):
    path = tmp_path / "calls.jsonl"
    thunc.configure(trace=str(path))
    fake.replies = ["true"]
    thunc.call("Is it spam?", {"email": "x"}, returns=bool, cache=True)
    thunc.call("Is it spam?", {"email": "x"}, returns=bool, cache=True)
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert not first["cached"] and first["attempts"] == 1
    assert second["cached"] and second["attempts"] == 0 and second["value"] is True
