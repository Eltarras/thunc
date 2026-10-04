"""Return types: describing them in the prompt, and parsing the model's reply back into a value."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from thunc.schema import describe, parse

from .bench_calls import TICKET_JSON, Ticket
from .harness import bench


@dataclass
class Line:
    sku: str
    quantity: int
    price: float | None


@dataclass
class Order:
    id: str
    status: Literal["open", "paid", "shipped", "cancelled"]
    lines: list[Line]
    notes: dict[str, str]


@bench("parse.int")
def parse_int(ctx):
    """'4' -> int: the common case."""
    yield lambda: parse("4", int)


@bench("parse.literal_bare_word")
def parse_literal(ctx):
    """An unquoted label for a Literal (not JSON): the fallback path."""
    tp = Literal["bug", "billing", "question", "not urgent"]
    yield lambda: parse("not urgent", tp)


@bench("parse.dataclass_fenced")
def parse_fenced(ctx):
    """A dataclass answer in a ```json fence after a line of prose."""
    text = f"Here is the triage:\n```json\n{TICKET_JSON}\n```"
    yield lambda: parse(text, Ticket)


@bench("parse.think_block")
def parse_think(ctx):
    """A 20 kB <think> block before the answer, as local reasoning models send."""
    text = "<think>" + "Let me weigh the evidence carefully. " * (50 if ctx.smoke else 550) + "</think>\n4"
    yield lambda: parse(text, int)


@bench("parse.long_text")
def parse_long_text(ctx):
    """A 200 kB plain-text answer (str return type)."""
    text = "  " + "The quick brown fox jumps over the lazy dog. " * (100 if ctx.smoke else 4500) + "\n\n"
    yield lambda: parse(text, str)


@bench("parse.orders_10k_lines")
def parse_orders(ctx):
    """list[Order], 100 orders x 100 lines: validating a large nested answer."""
    orders, lines = (5, 5) if ctx.smoke else (100, 100)
    data = [
        {
            "id": f"o{i}",
            "status": "paid",
            "lines": [{"sku": f"s{j}", "quantity": j, "price": 9.5} for j in range(lines)],
            "notes": {"gift": "no"},
        }
        for i in range(orders)
    ]
    text = json.dumps(data)
    ctx.notes["kB"] = len(text) // 1000
    yield lambda: parse(text, list[Order])


@bench("describe.nested_dataclass")
def describe_nested(ctx):
    """describe(list[Order]): the 'Return ...' line, built on every call."""
    yield lambda: describe(list[Order])
