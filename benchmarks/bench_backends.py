"""The API backends against a local stand-in for the Claude API.

The local server has no TLS and no network distance, so a new connection costs almost nothing
here. Against the real API it's a TCP and a TLS handshake: two round trips, typically 50-300 ms.
The *_handshake60ms benchmarks add 60 ms to each new connection to stand in for that (two round
trips at 30 ms). The `connections` note is what carries over exactly: how many connections the
requests opened.
"""

from __future__ import annotations

import os

import thunc

from .fakes import MessagesAPI
from .harness import Skip, bench

CALLS = 20


def _anthropic(ctx, handshake: float = 0.0):
    try:
        import anthropic
    except ImportError:
        raise Skip("the anthropic SDK isn't installed: uv sync --extra anthropic") from None
    api = MessagesAPI(latency=0.002, handshake=handshake)
    os.environ["ANTHROPIC_BASE_URL"] = api.url
    os.environ["ANTHROPIC_API_KEY"] = "bench"
    thunc.configure(backend="anthropic", model="claude-bench")
    return anthropic, api


def _through_thunc(ctx, handshake: float = 0.0):
    _, api = _anthropic(ctx, handshake)
    count = 2 if ctx.smoke else CALLS
    with api.running():

        def calls() -> None:
            for _ in range(count):
                thunc.call("Rate it.", {"ticket": "down"}, returns=int)

        yield calls
        ctx.notes["connections"] = f"{api.connections} for {api.requests} requests"


def _one_client(ctx, handshake: float = 0.0):
    anthropic, api = _anthropic(ctx, handshake)
    with api.running():
        client = anthropic.Anthropic()
        count = 2 if ctx.smoke else CALLS

        def calls() -> None:
            for _ in range(count):
                client.messages.create(
                    model="claude-bench", max_tokens=16000, system="s", messages=[{"role": "user", "content": "x"}]
                )

        yield calls
        ctx.notes["connections"] = f"{api.connections} for {api.requests} requests"


@bench("backend.anthropic_20_calls")
def anthropic_calls(ctx):
    """20 thunc.call through the anthropic backend, to a local API that answers in 2 ms."""
    yield from _through_thunc(ctx)


@bench("backend.anthropic_20_calls_one_client")
def anthropic_one_client(ctx):
    """For reference: the same 20 requests through one reused SDK client, as thunc could send them."""
    yield from _one_client(ctx)


@bench("backend.anthropic_20_calls_handshake60ms")
def anthropic_calls_handshake(ctx):
    """backend.anthropic_20_calls, with 60 ms of simulated TCP+TLS setup per new connection."""
    yield from _through_thunc(ctx, 0.06)


@bench("backend.anthropic_20_calls_one_client_handshake60ms")
def anthropic_one_client_handshake(ctx):
    """The one-client reference, with the same simulated connection setup."""
    yield from _one_client(ctx, 0.06)
