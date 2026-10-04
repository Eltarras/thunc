"""Wire formats. JSON only, versioned independently of task definitions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from thunc.errors import ThuncError

T = TypeVar("T")
INLINE_LIMIT = 256 * 1024
STATE_LIMIT = 16 * 1024 * 1024


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("Durable JSON objects require string keys")
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported durable value: {type(value).__name__}")


def encoded(value: Any) -> bytes:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def identifiers(workspace: str, task: str, version: str, request: str) -> tuple[str, str]:
    scope = digest(workspace)[:32]
    return f"thunc-workspace-{scope}", f"thunc-run-{scope}-{digest([task, version, request])}"


@dataclass(frozen=True)
class DurableRun(Generic[T]):
    id: str
    value: T
    status: str
    steps: int
    retries: int
    files_changed: list[str] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)
    denied: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


class DurableError(ThuncError):
    """An unsuccessful durable run; its persisted record remains available."""

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record
