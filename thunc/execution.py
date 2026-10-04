"""The agent loop's decisions, shared by local and durable execution.

AgentState holds everything the loop decides from, in a form that serializes to JSON. An
executor asks the model for a reply, hands it to receive(), then works through pending() calls:
check() settles the ones that need no tool (bad calls, finish) and done() records each outcome.
Carrying out a tool is the executor's part: agent.run() uses it directly, the Temporal worker
through its effect journal.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import asdict, dataclass, field
from typing import Any

from . import tools
from .core import _ensured
from .errors import ThuncError
from .native import Call, Reply
from .runs import Denial
from .schema import describe, parse, shorten, validate

KNOWN = frozenset({*tools.TOOLS, "remember", "finish"})


@dataclass
class Outcome:
    """What check() made of a call."""

    output: str | None = None  # what the model is told, when no tool is needed; None: carry it out
    finished: bool = False  # finish with a valid value
    value: Any = None
    error: ThuncError | None = None  # the run fails, once the output is recorded


@dataclass
class AgentState:
    turns: int = 0  # replies asked for
    operations: int = 0  # calls settled
    bad_finishes: int = 0
    calls: list[Call] = field(default_factory=list)  # the current reply's calls
    index: int = 0  # the next of them to settle
    results: list[tuple[Call, str, bool]] = field(default_factory=list)  # (call, output, failed) for the reply
    denied: list[Denial] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def begin_turn(self, limit: int, name: str = "") -> None:
        if self.turns >= limit:
            raise ThuncError(f"{name + ': ' if name else ''}the agent didn't finish within max_steps={limit}")
        self.turns += 1

    def receive(self, reply: Reply) -> bool:
        """Take a reply's calls. False when the reply has none to carry out: nudge the model instead."""
        self.calls = [] if reply.problem else list(reply.calls)
        self.index = 0
        self.results = []
        return reply.problem is None

    def pending(self) -> Call | None:
        return self.calls[self.index] if self.index < len(self.calls) else None

    def check(
        self,
        call: Call,
        returns: Any,
        ensure: Callable[[Any], bool] | None,
        repairs: int,
        name: str,
        known: Collection[str] = KNOWN,
    ) -> Outcome:
        """Settle a call that needs no tool: a malformed or unknown one, or finish."""
        if call.problem or call.tool not in known:
            return Outcome(f"error: {call.problem or f'unknown tool {call.tool!r}'}")
        if call.tool != "finish":
            return Outcome()
        try:
            return Outcome(finished=True, value=_finished(call.args, returns, ensure))
        except ValueError as problem:
            self.bad_finishes += 1
            output = (
                f"error: that value is invalid ({shorten(str(problem), 1000)}). "
                f"Call finish again with {describe(returns, True)}."
            )
            if self.bad_finishes <= repairs:
                return Outcome(output)
            error = ThuncError(
                f"{name}: no valid {describe(returns)} after {self.bad_finishes} finish attempt(s); "
                f"last error: {problem}"
            )
            error.__cause__ = problem
            return Outcome(output, error=error)

    def done(self, call: Call, output: str, denied: bool = False) -> None:
        """Record what the model is told about the call, and move on to the next."""
        if denied:
            target = call.args.get("path") or call.args.get("command") or ""
            reason = output.removeprefix("error: ").removeprefix("not permitted: ")
            self.denied.append(Denial(call.tool, target if isinstance(target, str) else repr(target), reason))
        elif call.tool == "remember" and not output.startswith("error: "):
            self.notes.append(" ".join(str(call.args.get("note", "")).split()))
        self.results.append((call, output, output.startswith("error: ")))
        self.index += 1
        self.operations += 1

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AgentState:
        return cls(
            **{k: v for k, v in data.items() if k not in {"calls", "results", "denied"}},
            calls=[Call(**c) for c in data["calls"]],
            results=[(Call(**c), output, failed) for c, output, failed in data["results"]],
            denied=[Denial(**d) for d in data["denied"]],
        )


def _finished(args: dict[str, Any], returns: Any, ensure: Callable[[Any], bool] | None) -> Any:
    """The value given to finish, checked against the return type and ensure=."""
    if "value" not in args:
        raise ValueError('finish takes {"value": ...}')
    value = args["value"]
    try:
        value = validate(value, returns)
    except ValueError:
        if not isinstance(value, str) or returns is str:
            raise
        value = parse(value, returns)  # the value sent as JSON text, like "4" for an int
    if returns is str and not value.strip():
        raise ValueError("the value was empty")
    return _ensured(value, ensure)
