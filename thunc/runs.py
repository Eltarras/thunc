"""What an agent did in one run: returned by agent.run(task, ...) and carried by AgentError."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .errors import ThuncError

T = TypeVar("T")


@dataclass(frozen=True)
class Command:
    """A command the agent ran."""

    command: str
    exit_code: int | None  # None: stopped at the time limit
    seconds: float


@dataclass(frozen=True)
class Denial:
    """An action the permissions refused."""

    tool: str  # read, write, edit, run, remember, ...
    target: str  # the path or command it was for
    reason: str


@dataclass(frozen=True)
class Run(Generic[T]):
    """One run of a task.

    value:         the result, checked against the task's return type (None if the run failed)
    steps:         model replies in the run
    files_changed: files the agent created or changed with write and edit (not by running commands)
    commands:      every command it ran, with its exit code
    denied:        actions the permissions refused
    notes:         notes it saved to memory with remember
    followed:      instruction files it was given with follow=
    session:       the run's record on disk, a JSONL file with every step
    error:         why the run failed, or None
    """

    task: str
    value: T
    steps: int
    seconds: float
    session: str
    files_changed: list[str] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    denied: list[Denial] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    followed: list[str] = field(default_factory=list)
    error: str | None = None


class AgentError(ThuncError):
    """A run that ended without a valid result. `.run` says what the agent did before it stopped."""

    def __init__(self, message: str, run: Run[None]) -> None:
        super().__init__(message)
        self.run = run
