"""A program to try thunc-watch on without a model: support-inbox functions and an agent, answered
by a scripted fake backend that takes a few seconds per reply and sometimes answers badly.

    cargo run --manifest-path watch/Cargo.toml -- --python .venv/bin/python watch/demo/demo_app.py

Run it from the repository root, so `import thunc` finds this checkout.
"""

from __future__ import annotations

import json
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import thunc  # noqa: E402
from thunc import backends  # noqa: E402

TICKETS = [
    "I was charged twice for March",
    "App crashes when I upload a PDF",
    "Can you add dark mode to the dashboard?",
    "My invoice PDF won't open and payroll runs tomorrow",
    "Where is order A-1043? It was due Friday",
    "Password reset email never arrives",
    "Any plans for a public API?",
    "Refund still not showing after 10 days",
    "Export to CSV drops the last row",
    "Order B-0217 arrived damaged",
]

rng = random.Random(7)


def agent_reply(text: str) -> str:
    """The agent's next action, from how many tool results it has seen so far."""
    seen = text.count("<step ")
    script = [
        {"tool": "list", "args": {"path": "."}},
        {"tool": "read", "args": {"path": "cache.py"}},
        {"tool": "search", "args": {"pattern": "cache=True", "path": "tests"}},
        {"tool": "run", "args": {"command": f"{sys.executable} -c \"import time; time.sleep(2); print('3 passed')\""}},
        {"tool": "write", "args": {"path": "README.md", "content": "notes"}},
        {"tool": "finish", "args": {"value": ["tests/test_cache.py"]}},
    ]
    return json.dumps(script[min(seen, len(script) - 1)])


def fake(text: str, system: str = "", **_: object) -> str:
    if system.startswith("You are an agent"):
        time.sleep(rng.uniform(1.0, 2.5))
        return agent_reply(text)
    time.sleep(rng.uniform(0.6, 3.5))
    if "Rate how urgent" in text:
        return rng.choice(["high", "7", "4", "2", "5", "3", "1"]) if rng.random() < 0.3 else rng.choice("12345")
    if "Classify" in text:
        return json.dumps(rng.choice(["bug", "billing", "feature-request", "other"]))
    if "order number" in text:
        found = re.search(r"\b([AB]-\d{4})\b", text)
        return json.dumps({"id": found.group(1)} if found else None)
    if "Draft a reply" in text:
        time.sleep(rng.uniform(1.0, 3.0))
        replies = ["Thanks for flagging this, we're on it.", "Sorry about that! A fix ships Thursday."]
        return json.dumps(rng.choice(replies))
    return "null"


backends.BACKENDS["demo"] = fake
thunc.configure(backend="demo")


@dataclass
class Order:
    id: str


@thunc.function
def category(ticket: str) -> Literal["bug", "billing", "feature-request", "other"]:
    """Classify this support ticket."""
    ...


@thunc.function(ensure=lambda n: 1 <= n <= 5)
def urgency(ticket: str) -> int:
    """Rate how urgent this ticket is, from 1 (can wait) to 5 (customer is blocked)."""
    ...


@thunc.function
def find_order(ticket: str) -> Order | None:
    """The order number this ticket mentions, if any."""
    ...


@thunc.function
def draft_reply(ticket: str) -> str:
    """Draft a reply to this ticket: friendly and concise."""
    ...


def triage(ticket: str) -> str:
    kind = category(ticket)
    try:
        level = urgency(ticket)
    except thunc.ThuncError:
        level = 0
    order = find_order(ticket)
    reply = draft_reply(ticket) if level >= 3 else ""
    return f"{kind:16} urgency {level}  order {order.id if order else '-':7}  {reply}"


def main() -> None:
    repo = Path(tempfile.mkdtemp(prefix="thunc-watch-demo-"))
    (repo / "cache.py").write_text("def get(key):\n    return None\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_cache.py").write_text("def test_cache_true():\n    assert True  # cache=True\n")
    guide = thunc.Agent("repo-guide", workdir=repo, permissions=["run", "!write"], protocol="text")

    @guide.task
    def tests_for(feature: str) -> list[str]:
        """Find the tests that cover this feature."""
        ...

    import threading

    agent_thread = threading.Thread(target=lambda: print("agent:", tests_for("caching")))
    agent_thread.start()
    for line in thunc.map(triage, TICKETS * 2, workers=4):
        print(line)
    agent_thread.join()


if __name__ == "__main__":
    main()
