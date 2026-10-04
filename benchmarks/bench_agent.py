"""Whole agent runs on the text protocol, with a model that answers at once: the loop's own cost, and
how the prompt grows with the number of steps."""

from __future__ import annotations

import json
import os

import thunc

from .bench_tools import make_repo
from .fakes import Backend, use
from .harness import bench


def _agent_run(ctx, steps: int, per_reply: int = 1):
    repo = os.path.join(ctx.tmp, "repo")
    os.makedirs(repo)
    make_repo(repo, *((2, 5) if ctx.smoke else (10, 20)))
    actions = [
        {"tool": "list", "args": {"path": "src/pkg000"}},
        {"tool": "read", "args": {"path": "src/pkg000/mod001.py", "limit": 40}},
        {"tool": "search", "args": {"pattern": "TIMEOUT = 1\\b", "path": "src/pkg001"}},
    ]

    def reply(prompt: str) -> str:
        done = prompt.count('<step n="')  # actions carried out so far
        if done >= steps - 1:
            return json.dumps({"tool": "finish", "args": {"value": 1}})
        batch = [actions[n % len(actions)] for n in range(done, min(done + per_reply, steps - 1))]
        return json.dumps(batch[0] if per_reply == 1 else batch)

    backend = use(Backend(reply))
    agent = thunc.Agent("bench", workdir=repo, max_steps=steps + 1)

    @agent.task
    def timeout() -> int:
        """Find the request timeout this app uses, in seconds."""
        ...

    replies = -(-(steps - 1) // per_reply) + 1  # the actions, then finish
    yield timeout
    runs = max(backend.requests // replies, 1)
    ctx.notes.update(actions=steps, replies=replies, sent_per_run=f"{backend.bytes_sent // runs // 1000} kB")


@bench("agent.text_10_steps")
def agent_10(ctx):
    """A 10-step run: list, read and search, then finish."""
    yield from _agent_run(ctx, 10)


@bench("agent.text_30_steps")
def agent_30(ctx):
    """A 30-step run. Compare sent_per_run with the 10-step run: the whole transcript goes every turn."""
    yield from _agent_run(ctx, 30)


@bench("agent.text_30_steps_batched3")
def agent_30_batched(ctx):
    """The same 30 actions, sent 3 to a reply: a third of the turns, so far less resent."""
    yield from _agent_run(ctx, 30, per_reply=3)
