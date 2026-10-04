"""Jev and Claude together: Jev makes the quick decisions, Claude writes text only where it's needed.

Jev can't write text, so a reply draft comes from another backend. Each function picks its own.
Needs the jev CLI, logged in (see docs/jev.html), and Claude Code (or set THUNC_BACKEND).
Run from the repo root:  python3 -m examples.jev_with_claude
"""

import os
import textwrap
from typing import Literal

import thunc

thunc.configure(backend=os.environ.get("THUNC_BACKEND", "claude-code"))  # the default, for text


@thunc.function(backend="jev")
def needs_reply(message: str) -> bool:
    """Does this message ask a question or report a problem that a person on our team should answer?"""
    ...


@thunc.function(backend="jev")
def tone(message: str) -> Literal["calm", "frustrated", "angry"]:
    """How does the customer sound?"""
    ...


@thunc.function  # the configured backend, since Jev can't write text
def draft_reply(message: str, tone: str) -> str:
    """Write a short first reply to this message. Match the customer's tone: the angrier they are,
    the more direct the apology. Never promise a refund."""
    ...


messages = [
    "Thanks, that fixed it!",
    "I've asked three times now why my invoice shows the wrong VAT number. Is anyone reading these?",
    "Out of office until Monday.",
    "Quick question: can I add a second admin to our workspace?",
]

for message in messages:
    if not needs_reply(message):  # most messages stop here, after one fast Jev call
        print(f"skip   {message}")
        continue
    mood = tone(message)
    print(f"reply  {message}  ({mood})")
    print(textwrap.indent(draft_reply(message, mood), "       > ") + "\n")
