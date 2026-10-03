"""Docstring prompts: triage a support inbox.

Run from the repo root:  python3 -m examples.support_inbox
"""

import os
from dataclasses import dataclass
from typing import Literal

import thunc

thunc.configure(backend=os.environ.get("THUNC_BACKEND", "claude-code"))


@dataclass
class Order:
    order_id: str
    product: str | None = None
    amount_eur: float | None = None


@thunc.function
def category(ticket: str) -> Literal["bug", "billing", "feature-request", "other"]:
    """Classify this customer support ticket."""
    ...


@thunc.function(ensure=lambda n: 1 <= n <= 5)
def urgency(ticket: str) -> int:
    """Rate how urgent this ticket is, from 1 (can wait a week) to 5 (customer is blocked right now)."""
    ...


@thunc.function
def find_order(ticket: str) -> Order | None:
    """Extract the order this ticket is about, or null if it doesn't mention one."""
    ...


@thunc.function
def draft_reply(ticket: str, tone: str = "friendly and concise") -> str:
    """Write a short first reply to this ticket in the given tone. Never promise a refund."""
    ...


tickets = [
    "I was charged twice for order #A-1042 (Pro plan, €49). Please fix this today!",
    "It would be great if the dashboard had a dark mode.",
    "Export to CSV crashes with 'unexpected token' since this morning's update.",
]


def triage(ticket: str) -> tuple[str, int, Order | None]:
    return category(ticket), urgency(ticket), find_order(ticket)


# The tickets are triaged in parallel; each ticket's three calls run in sequence.
for ticket, (kind, level, order) in zip(tickets, thunc.map(triage, tickets), strict=True):
    print(f"[{kind:15}] urgency {level}  {ticket}")
    if order:
        print(f"{'':19}order: {order}")

print("\nDraft reply to the first ticket:\n" + draft_reply(tickets[0]))
