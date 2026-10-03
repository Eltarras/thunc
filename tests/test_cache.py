"""cache=True: answers saved on disk and reused, against the fake backend."""

import asyncio
import json
import os
import time
from datetime import timedelta
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


# Clearing: thunc.clear_cache() and thunc.cache_info()


def _age(path, seconds):
    """Make a cache file look `seconds` old."""
    then = time.time() - seconds
    os.utime(path, (then, then))


@thunc.function(cache=True)
def urgency(ticket: str) -> int:
    """Rate how urgent this ticket is, from 1 to 5."""
    ...


@thunc.function(cache=True)
def category(ticket: str) -> Literal["bug", "billing"]:
    """Classify this support ticket."""
    ...


def test_clear_everything(fake, cache):
    fake.replies = ["4", '"bug"', "x", "5"]
    urgency("down"), category("down"), thunc.call("Say x.", cache=True)
    assert thunc.clear_cache() == 3
    assert not list(cache.glob("*.json"))
    assert urgency("down") == 5  # asks the model again
    assert len(fake.prompts) == 4


def test_clear_one_function_by_object_or_name(fake, cache):
    fake.replies = ["4", "2", '"bug"', "5", "1"]
    urgency("down"), urgency("typo"), category("down")
    assert thunc.clear_cache(urgency) == 2
    assert category("down") == "bug"  # still answered from disk
    assert urgency("down") == 5
    assert len(fake.prompts) == 4

    assert thunc.clear_cache("urgency") == 1
    assert thunc.clear_cache(f"{__name__}.urgency") == 0  # module-qualified works too; nothing left
    assert urgency("down") == 1


def test_clear_by_module_qualified_name(fake, cache):
    fake.replies = ["4"]
    urgency("down")
    assert thunc.clear_cache("other_module.urgency") == 0
    assert thunc.clear_cache(f"{__name__}.urgency") == 1


def test_clear_async_function_and_method(fake, cache):
    @thunc.function(cache=True)
    async def is_spam(email: str) -> bool:
        """Is this spam?"""
        ...

    class Triage:
        @thunc.function(cache=True)
        def score(self, ticket: str) -> int:
            """Score this ticket from 1 to 5."""
            ...

    fake.replies = ["true", "3"]
    asyncio.run(is_spam("win money"))
    triage = Triage()
    triage.score("down")
    assert thunc.clear_cache(is_spam) == 1
    assert thunc.clear_cache(triage.score) == 1  # a bound method works
    assert not list(cache.glob("*.json"))


def test_named_calls_can_be_cleared(fake, cache, tmp_path):
    thunc.configure(trace=str(tmp_path / "calls.jsonl"))
    fake.replies = ["Hola", "Hello"]
    assert thunc.call("Translate into Spanish.", {"text": "Hello"}, cache=True, name="translate") == "Hola"
    thunc.call("Say hello.", cache=True)
    assert thunc.clear_cache("translate") == 1
    (left,) = [json.loads(p.read_text()) for p in cache.glob("*.json")]
    assert left["function"] is None and left["request"].startswith("<instructions>\nSay hello.")
    first, _ = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert first["function"] == "translate"


def test_same_prompt_from_two_functions_is_two_entries(fake, cache):
    fake.replies = ["a", "b", "c"]
    assert thunc.call("Name a letter.", cache=True, name="one") == "a"
    assert thunc.call("Name a letter.", cache=True, name="two") == "b"
    assert thunc.clear_cache("one") == 1
    assert thunc.call("Name a letter.", cache=True, name="two") == "b"  # untouched
    assert thunc.call("Name a letter.", cache=True, name="one") == "c"  # really cleared


def test_entry_records_function_and_module(fake, cache):
    fake.replies = ["4"]
    urgency("down")
    (entry,) = [json.loads(p.read_text()) for p in cache.glob("*.json")]
    assert entry["function"] == "urgency" and entry["module"] == __name__


def test_older_than(fake, cache):
    fake.replies = ["1", "2"]
    urgency("old"), urgency("new")
    (old,) = [p for p in cache.glob("*.json") if "<ticket>\nold\n" in json.loads(p.read_text())["request"]]
    (new,) = [p for p in cache.glob("*.json") if p != old]
    _age(old, 40 * 86400)
    _age(new, 3600)
    assert thunc.clear_cache(older_than=timedelta(days=30)) == 1
    assert [p.name for p in cache.glob("*.json")] == [new.name]
    assert thunc.clear_cache(urgency, older_than=7200) == 0
    assert thunc.clear_cache(urgency, older_than=60) == 1


def test_only_cache_files_are_removed(fake, cache):
    fake.replies = ["4"]
    urgency("down")
    (cache / "notes.json").write_text("{}")
    (cache / "README").write_text("mine")
    (cache / "nested").mkdir()
    fresh_tmp, stale_tmp = cache / "tmpfresh01.tmp", cache / "tmpstale01.tmp"
    fresh_tmp.write_text(""), stale_tmp.write_text("")
    _age(stale_tmp, 2 * 3600)
    assert thunc.clear_cache() == 1
    assert sorted(p.name for p in cache.iterdir()) == ["README", "nested", "notes.json", "tmpfresh01.tmp"]


def test_unreadable_entries(fake, cache):
    fake.replies = ["4"]
    urgency("down")
    (cache / ("a" * 64 + ".json")).write_text("{not json")
    (cache / ("b" * 64 + ".json")).write_text('["a list"]')
    groups = thunc.cache_info()
    assert [(g.function, g.readable, g.entries) for g in groups] == [("urgency", True, 1), (None, False, 2)]
    assert thunc.clear_cache("urgency") == 1  # can't tell who owns the unreadable ones: left alone
    assert thunc.clear_cache() == 2


def test_cache_info(fake, cache):
    assert thunc.cache_info() == []  # no folder yet
    fake.replies = ["4", "2", '"bug"', "x"]
    urgency("down"), urgency("typo"), category("down"), thunc.call("Say x.", cache=True)
    groups = thunc.cache_info()
    assert [(g.function, g.module, g.entries) for g in groups] == [
        ("category", __name__, 1),
        ("urgency", __name__, 2),
        (None, None, 1),
    ]
    assert all(g.readable and g.size > 0 for g in groups)
    assert groups[1].size == sum(p.stat().st_size for p in cache.glob("*.json") if '"urgency"' in p.read_text())


def test_clear_missing_folder_and_bad_arguments(cache):
    assert thunc.clear_cache() == 0
    assert thunc.clear_cache("urgency") == 0
    with pytest.raises(TypeError, match="not a @thunc.function"):
        thunc.clear_cache(len)
    with pytest.raises(ValueError, match="negative"):
        thunc.clear_cache(older_than=-1)
    with pytest.raises(ValueError, match="empty"):
        thunc.clear_cache("")


def test_a_different_system_prompt_is_a_miss(fake, cache):
    fake.replies = ["4", "2"]
    assert thunc.call("Rate 1-5.", {"t": "x"}, returns=int, cache=True) == 4
    assert thunc.call("Rate 1-5.", {"t": "x"}, returns=int, cache=True, system="You are harsh.") == 2
    assert thunc.call("Rate 1-5.", {"t": "x"}, returns=int, cache=True, system="You are harsh.") == 2
    assert len(fake.prompts) == 2
