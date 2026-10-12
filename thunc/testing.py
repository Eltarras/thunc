"""Testing code that uses thunc, without a model: a scripted fake model, and helpers for agents.

    from thunc import testing

    def test_triage():
        with testing.fake_model({"urgency": 4, "category": "bug"}) as model:
            assert triage("I was charged twice") == ("bug", 4)
        assert model.calls[0].inputs == {"ticket": "I was charged twice"}

While fake_model() is active, every call and agent run is answered by it, whatever backend= it
names, and the recordings (THUNC_RECORDINGS) are left alone. A reply can be:

- a list of replies, given out in order to whatever asks;
- a dict from a function's name ("urgency", "Triage.urgency", an agent task's "repo.fix", or a
  thunc.call's name=) to its reply, a list of replies, or a function of the Request. A function
  defined inside another (in a test) is named without it: "urgency", not "test_x.<locals>.urgency";
- a function of the Request, which returns the reply.

A reply that's a string is sent as the model's text, as it is: "4", or "high" to see the call retry.
Anything else is sent as JSON, so 4, True, Priority.HIGH, a dataclass or a list of them work as
written. An agent's replies are its tool calls, written with act() and finish().

With pytest, the `thunc_model` fixture is a fake_model() for the test, and `thunc_isolated` puts
thunc's settings back after it. Both come with thunc; nothing to configure.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import json
import os
import re
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from . import backends, config, core
from .core import CallInfo

FAKE = "thunc-testing-fake"  # the backend name the fake model is registered under while active
NO_REPLY = object()


@dataclasses.dataclass(frozen=True)
class Request:
    """One request the fake model got: the call or agent run it belongs to, and what was sent."""

    function: str | None  # the function's name, "agent.task" for an agent run, or thunc.call's name=
    instructions: str
    inputs: dict[str, Any]
    returns: Any
    prompt: str  # the text sent, as a real backend gets it
    system: str
    agent: bool  # a step of an agent run, rather than a call
    n: int  # 1 for the first request of this call or run, 2 for its first retry or next step, ...


Reply = Any  # a string, sent as the model's text as it is, or a value, sent as JSON
Replies = Sequence[Reply] | Mapping[str, Any] | Callable[[Request], Reply]


class FakeModel:
    """A scripted model. Use it through fake_model(), or the thunc_model fixture.

    .reply(...) adds replies; .calls lists every request, in order; .requests_for(name) those of one
    function. Running out of replies fails the test with what was asked."""

    def __init__(self, replies: Replies | None = None) -> None:
        self._lock = threading.Lock()
        self._queue: list[Reply] = []
        self._by_function: dict[str, list[Reply] | Callable[[Request], Reply]] = {}
        self._answer: Callable[[Request], Reply] | None = None
        self.calls: list[Request] = []
        self._counts: dict[int, int] = {}  # requests per call or run, by the id of its CallInfo
        if replies is not None:
            self.reply(replies)

    def reply(self, *replies: Any, function: str | None = None) -> FakeModel:
        """Add replies: in order (`reply("4", "2")`), for one function (`reply(4, function="urgency")`),
        or from a list, a dict or a function, as fake_model() takes them."""
        with self._lock:
            if function is not None:
                kept = self._by_function.setdefault(function, [])
                if not isinstance(kept, list):
                    raise ValueError(f"{function!r} already answers with a function")
                kept.extend(replies)
                return self
            for given in replies:
                if callable(given) and not isinstance(given, type):
                    self._answer = given
                elif isinstance(given, Mapping):
                    for name, value in given.items():
                        if callable(value) and not isinstance(value, (type, enum.Enum)):
                            self._by_function[name] = value
                        else:
                            listed = list(value) if isinstance(value, list) else [value]
                            current = self._by_function.get(name)
                            self._by_function[name] = [*current, *listed] if isinstance(current, list) else listed
                elif isinstance(given, (list, tuple)):
                    self._queue.extend(given)
                else:
                    self._queue.append(given)
        return self

    def requests_for(self, function: str) -> list[Request]:
        """The requests made for one function (or agent task), in order."""
        return [request for request in self.calls if function in _names(request.function)]

    def __call__(self, text: str, *, system: str, **_: Any) -> str:
        info = core.current_call.get() or CallInfo(None, text, {}, str)
        with self._lock:
            n = self._counts[id(info)] = self._counts.get(id(info), 0) + 1
            request = Request(info.function, info.instructions, info.inputs, info.returns, text, system, info.agent, n)
            self.calls.append(request)
            reply = self._next(request)
        if reply is NO_REPLY:
            raise AssertionError(
                f"The fake model has no reply for {request.function or 'a thunc.call'} "
                f"(request {n}, instructions {request.instructions[:80]!r}). Add one with reply()."
            )
        if callable(reply) and not isinstance(reply, (type, enum.Enum)):
            reply = reply(request)
        return as_text(reply)

    def _next(self, request: Request) -> Any:
        named = next((self._by_function[n] for n in _names(request.function) if n in self._by_function), None)
        if callable(named):
            return named
        if isinstance(named, list) and named:
            return named.pop(0)
        if self._queue:
            return self._queue.pop(0)
        if self._answer is not None:
            return self._answer
        return NO_REPLY


def _names(function: str | None) -> tuple[str, ...]:
    """A function's names to match replies on: its own, and without the functions it's defined in
    ("fixer.double_timeout" for a task defined in a test function, "fixer.test_x.<locals>.double_timeout")."""
    if function is None:
        return ()
    short = re.sub(r"[^.<>]+\.<locals>\.", "", function)
    return (function, short) if short != function else (function,)


def as_text(reply: Any) -> str:
    """A reply as the model's text: a string as it is, anything else as JSON."""
    if isinstance(reply, str) and not isinstance(reply, enum.Enum):
        return reply
    return json.dumps(_plain(reply), ensure_ascii=False)


def _plain(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


@contextlib.contextmanager
def fake_model(replies: Replies | None = None) -> Iterator[FakeModel]:
    """Answer every call and agent run with a FakeModel until the block ends; see the module docs.
    Agents use the JSON text protocol on it: script their steps with act() and finish()."""
    if config.forced_backend is not None:
        raise RuntimeError("fake_model() is already active; nest the replies in one instead")
    model = FakeModel(replies)
    backends.BACKENDS[FAKE] = model
    config.forced_backend = FAKE
    try:
        yield model
    finally:
        config.forced_backend = None
        backends.BACKENDS.pop(FAKE, None)


def act(tool: str, **args: Any) -> str:
    """An agent's tool call, as a reply: act("read", path="config.py"), act("run", command="pytest -q")."""
    return json.dumps({"tool": tool, "args": _plain(args)}, ensure_ascii=False)


def finish(value: Any) -> str:
    """An agent's last reply: finish with this value (a dataclass, an Enum and so on are written as JSON)."""
    return act("finish", value=value)


_ENVIRONMENT = (
    "THUNC_BACKEND",
    "THUNC_TRACE",
    "THUNC_CACHE_DIR",
    "THUNC_AGENTS_DIR",
    "THUNC_RECORDINGS",
    "THUNC_RECORD",
    "THUNC_EVENTS",
)


@contextlib.contextmanager
def isolated() -> Iterator[None]:
    """thunc's settings (configure()) and THUNC_* environment variables as they were, after the block."""
    settings = dict(config._settings)
    environment = {name: os.environ.get(name) for name in _ENVIRONMENT}
    try:
        yield
    finally:
        config._settings.clear()
        config._settings.update(settings)
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
