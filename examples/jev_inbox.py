"""Triage a support inbox with Jev: spam check, team and urgency for every ticket, in parallel.

The same docstring functions as support_inbox.py, with the return types Jev can answer.
Needs the jev CLI, logged in (see docs/jev.html).  Run from the repo root:  python3 -m examples.jev_inbox
"""

import time
from typing import Literal

import thunc

thunc.configure(backend="jev")


@thunc.function
def is_spam(ticket: str) -> bool:
    """Is this message spam or a scam rather than a real customer request?"""
    ...


@thunc.function
def team(ticket: str) -> Literal["bug", "billing", "feature-request", "account", "other"]:
    """Which team should handle this customer support ticket?"""
    ...


@thunc.function
def urgency(ticket: str) -> Literal[1, 2, 3, 4, 5]:
    """Rate how urgent this ticket is, from 1 (can wait a week) to 5 (customer is blocked right now)."""
    ...


tickets = [
    "I was charged twice for order #A-1042 (Pro plan, €49). Please fix this today!",
    "It would be great if the dashboard had a dark mode.",
    "Export to CSV crashes with 'unexpected token' since this morning's update.",
    "CONGRATULATIONS!!! You have been selected for a $1000 gift card, reply with your bank details.",
    "How do I change the email address on my account?",
    "Our whole team gets a 500 error on login. We have a demo with a client in an hour.",
    "Can you send me last year's invoices as one PDF? Our accountant needs them by Friday.",
    "The search box ignores accented letters: 'café' finds nothing, 'cafe' works.",
]


def triage(ticket: str) -> tuple[bool, str, int]:
    return is_spam(ticket), team(ticket), urgency(ticket)


start = time.monotonic()
results = thunc.map(triage, tickets)
print(f"{len(tickets)} tickets, {3 * len(tickets)} calls in {time.monotonic() - start:.1f}s\n")

# Most urgent first; spam last.
for (spam, kind, level), ticket in sorted(zip(results, tickets, strict=True), key=lambda row: (row[0][0], -row[0][2])):
    label = "spam" if spam else kind
    print(f"[{label:15}] urgency {level}  {ticket[:70]}")
