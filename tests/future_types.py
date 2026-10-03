"""Dataclasses with string annotations, as `from __future__ import annotations` makes them."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field


@dataclass
class Item:
    sku: str


@dataclass
class Holder:
    items: list["Item"]  # noqa: UP037  # a string inside a builtin generic, which Python 3.10 leaves as is
    table: dict[str, "Item"] = field(default_factory=dict)  # noqa: UP037


@dataclass
class Scaled:
    a: int
    scale: InitVar[int] = 1

    def __post_init__(self, scale: int) -> None:
        self.a *= scale
