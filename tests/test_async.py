"""The async API: thunc.acall, thunc.amap, async def functions, and thunc.map over them."""

import asyncio
import threading
import time
from typing import Literal

import pytest

import thunc


def test_acall_is_thunc_call_awaited(fake):
    fake.replies = ["high", "4"]
    value = asyncio.run(thunc.acall("Rate it.", {"ticket": "down"}, returns=int, ensure=lambda n: n <= 5))
    assert value == 4 and len(fake.prompts) == 2  # retried like thunc.call
    fake.replies = ["Paris"]
    assert asyncio.run(thunc.acall("Capital of France?")) == "Paris"


def test_acall_raises_thunc_error_and_keeps_the_loop_free(fake, monkeypatch):
    from thunc import backends

    started = threading.Event()

    def slow(text, **kwargs):
        started.set()
        time.sleep(0.2)
        return "nope"

    monkeypatch.setitem(backends.BACKENDS, "fake", slow)

    async def main():
        call = asyncio.ensure_future(thunc.acall("Rate it.", returns=int, retries=0))
        ticks = 0
        while not call.done():  # the loop keeps running while the model answers
            ticks += 1
            await asyncio.sleep(0.01)
        assert ticks > 5
        return await call

    with pytest.raises(thunc.ThuncError, match="No valid"):
        asyncio.run(main())


def concurrent_backend(monkeypatch, delay=0.05):
    """A backend that answers each prompt's <n> doubled, and records how many ran at once."""
    from thunc import backends

    state = {"now": 0, "most": 0}
    lock = threading.Lock()

    def double(text, **kwargs):
        with lock:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(delay)
        with lock:
            state["now"] -= 1
        n = int(text.split("<n>\n")[1].split("\n")[0])
        return str(n * 2)

    monkeypatch.setitem(backends.BACKENDS, "fake", double)
    return state


@thunc.function
def double(n: int) -> int:
    """Double it."""
    ...


@thunc.function
async def adouble(n: int) -> int:
    """Double it."""
    ...


@pytest.mark.parametrize("func", [double, adouble], ids=["plain", "async"])
def test_amap_keeps_order_and_runs_at_most_workers_at_once(fake, monkeypatch, func):
    state = concurrent_backend(monkeypatch)
    started = time.monotonic()
    assert asyncio.run(thunc.amap(func, range(8), workers=4)) == [0, 2, 4, 6, 8, 10, 12, 14]
    assert state["most"] == 4
    assert time.monotonic() - started < 0.05 * 8 * 0.75  # concurrent, not one after another


def test_amap_takes_any_async_function(fake):
    async def shout(word: str) -> str:
        await asyncio.sleep(0)
        return word.upper()

    assert asyncio.run(thunc.amap(shout, ["a", "b"])) == ["A", "B"]
    assert asyncio.run(thunc.amap(lambda n: n + 1, [1, 2])) == [2, 3]
    assert asyncio.run(thunc.amap(shout, [])) == []


def test_amap_raises_the_first_error_and_cancels_what_has_not_started(fake):
    started = []

    async def work(n):
        started.append(n)
        await asyncio.sleep(0.01 if n else 0)
        if n == 0:
            raise ValueError("bad item")
        await asyncio.sleep(1)
        return n

    with pytest.raises(ValueError, match="bad item"):
        asyncio.run(thunc.amap(work, range(10), workers=2))
    assert len(started) < 10


def test_amap_needs_at_least_one_worker(fake):
    with pytest.raises(ValueError, match="workers"):
        asyncio.run(thunc.amap(double, [1], workers=0))


def test_map_runs_async_functions_to_completion(fake, monkeypatch):
    concurrent_backend(monkeypatch, delay=0)
    assert thunc.map(adouble, [1, 2, 3]) == [2, 4, 6]


def test_map_of_async_functions_works_from_inside_an_event_loop(fake, monkeypatch):
    concurrent_backend(monkeypatch, delay=0)

    async def main():
        return thunc.map(adouble, [5])  # blocks the loop, but each item runs in its own thread

    assert asyncio.run(main()) == [10]


def test_async_function_returning_a_literal(fake):
    @thunc.function
    async def label(text: str) -> Literal["bug", "other"]:
        """Label it."""
        ...

    fake.replies = ["bug"]
    assert asyncio.run(label("crash")) == "bug"
