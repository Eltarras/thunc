"""thunc.call and @thunc.function: thunc's own cost per call, with a model that answers at once."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Literal

import thunc

from .fakes import Backend, use
from .harness import bench

TICKET = "The checkout page returns a 500 error for every customer since the 14:02 deploy."


@dataclass
class Ticket:
    category: Literal["bug", "billing", "question"]
    urgency: int
    tags: list[str]
    summary: str


TICKET_JSON = '{"category": "bug", "urgency": 5, "tags": ["checkout", "outage"], "summary": "Checkout is down."}'


@bench("call.str")
def call_str(ctx):
    """thunc.call returning plain text: the floor for one call."""
    use(Backend("Checkout is down since the deploy."))
    yield lambda: thunc.call("Summarize this ticket in one line.", {"ticket": TICKET})


@bench("call.int")
def call_int(ctx):
    """@thunc.function -> int: signature binding, prompt, parse."""
    use(Backend("4"))

    @thunc.function
    def urgency(ticket: str) -> int:
        """Rate how urgent this ticket is, from 1 to 5."""
        ...

    yield lambda: urgency(TICKET)


@bench("call.dataclass")
def call_dataclass(ctx):
    """@thunc.function -> a dataclass with a Literal, a list and a str."""
    use(Backend(TICKET_JSON))

    @thunc.function
    def triage(ticket: str) -> Ticket:
        """Triage this support ticket."""
        ...

    yield lambda: triage(TICKET)


@bench("call.retry")
def call_retry(ctx):
    """One invalid answer, then a valid one: the retry path (two requests)."""
    replies = itertools.cycle(["Probably a 4.", "4"])
    use(Backend(lambda _: next(replies)))
    yield lambda: thunc.call("Rate how urgent this is, from 1 to 5.", {"ticket": TICKET}, returns=int)


@bench("call.traced")
def call_traced(ctx):
    """call.str with trace= on: one JSON line appended per call."""
    use(Backend("Checkout is down since the deploy."))
    thunc.configure(trace=os.path.join(ctx.tmp, "trace.jsonl"))
    yield lambda: thunc.call("Summarize this ticket in one line.", {"ticket": TICKET})


@bench("call.cache_hit")
def call_cache_hit(ctx):
    """cache=True, answered from disk: what a hit costs instead of a model call."""
    backend = use(Backend("4"))
    yield lambda: thunc.call("Rate it from 1 to 5.", {"ticket": TICKET}, returns=int, cache=True)
    ctx.notes["model_requests"] = backend.requests


@bench("call.cache_miss")
def call_cache_miss(ctx):
    """cache=True with a new input each time: lookup, model, save."""
    use(Backend("4"))
    counter = itertools.count()
    yield lambda: thunc.call("Rate it from 1 to 5.", {"ticket": f"{TICKET} #{next(counter)}"}, returns=int, cache=True)
    ctx.notes["entries"] = len(os.listdir(os.path.join(ctx.tmp, "cache")))


@bench("call.big_inputs")
def call_big_inputs(ctx):
    """1,000 dataclass rows as an input: rendering inputs to JSON in the prompt."""
    use(Backend("bug"))
    rows = [Ticket("bug", i % 5 + 1, ["a", "b"], f"Ticket number {i}") for i in range(50 if ctx.smoke else 1000)]
    yield lambda: thunc.call("Which category is most common?", {"tickets": rows})


def _map(ctx, workers: int):
    latency, items = 0.02, 8 if ctx.smoke else 32
    use(Backend("4", latency=latency))
    tickets = [f"{TICKET} #{i}" for i in range(items)]

    @thunc.function
    def urgency(ticket: str) -> int:
        """Rate how urgent this ticket is, from 1 to 5."""
        ...

    ideal = latency * -(-items // workers)
    ctx.notes.update(items=items, latency="20ms", ideal=f"{ideal * 1000:.0f}ms")
    yield lambda: thunc.map(urgency, tickets, workers=workers)


@bench("map.32x20ms.workers8")
def map_8(ctx):
    """thunc.map: 32 calls to a 20 ms model, 8 at a time. Ideal: 4 rounds, 80 ms."""
    yield from _map(ctx, 8)


@bench("map.32x20ms.workers32")
def map_32(ctx):
    """thunc.map: 32 calls to a 20 ms model, all at once. Ideal: 20 ms."""
    yield from _map(ctx, 32)
