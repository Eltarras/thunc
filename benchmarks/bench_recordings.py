"""Record and replay: what a replayed call or agent run costs, and what recording adds to one.

Compare record.call_replayed with call.cache_hit (both read one file instead of asking the model), and
record.call_recording with call.str (the same call, plus saving its answer). call.str itself measures
what checking for a recordings folder costs every call when there's none.
"""

from __future__ import annotations

import itertools
import os

import thunc
from thunc import config

from .bench_agent import setup_run
from .bench_calls import TICKET
from .fakes import Backend, use
from .harness import bench


def _recording(ctx) -> str:
    folder = os.path.join(ctx.tmp, "recordings")
    thunc.configure(recordings=folder)
    os.environ["THUNC_RECORD"] = "1"  # the harness puts the environment back afterwards
    return folder


def _replaying() -> None:
    """Stop recording and unset the backend, as in CI: nothing could reach a model now."""
    os.environ.pop("THUNC_RECORD", None)
    config._settings["backend"] = None


@bench("record.call_replayed")
def call_replayed(ctx):
    """A call answered from the recordings, with no backend configured."""
    _recording(ctx)
    backend = use(Backend("4"))
    thunc.call("Rate it from 1 to 5.", {"ticket": TICKET}, returns=int)
    _replaying()
    yield lambda: thunc.call("Rate it from 1 to 5.", {"ticket": TICKET}, returns=int)
    ctx.notes["model_requests"] = backend.requests


@bench("record.call_recording")
def call_recording(ctx):
    """THUNC_RECORD=1 with a new input each time: ask the model, then save the answer."""
    folder = _recording(ctx)
    use(Backend("4"))
    counter = itertools.count()
    yield lambda: thunc.call("Rate it from 1 to 5.", {"ticket": f"{TICKET} #{next(counter)}"}, returns=int)
    ctx.notes["entries"] = len(os.listdir(folder))


@bench("record.agent_10_steps_replayed")
def agent_replayed(ctx):
    """agent.text_10_steps answered from a recording: the tools still run, no model is asked."""
    _recording(ctx)
    timeout, backend, replies = setup_run(ctx, 10)
    timeout()
    _replaying()
    yield timeout
    ctx.notes.update(replies=replies, model_requests=backend.requests)


@bench("record.agent_10_steps_recording")
def agent_recording(ctx):
    """agent.text_10_steps with THUNC_RECORD=1: the run, plus saving its replies once it finishes."""
    _recording(ctx)
    timeout, _, replies = setup_run(ctx, 10)
    yield timeout
    ctx.notes["replies"] = replies
