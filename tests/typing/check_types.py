"""Checked by tests/test_typing.py with mypy and basedpyright, never run: what the public API's
signatures make a type checker infer. Each assert_type fails the check if the inference changes."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from typing_extensions import assert_type

import thunc


@dataclass
class Item:
    name: str
    quantity: int


class Priority(Enum):
    LOW = 1
    HIGH = 3


@thunc.function
def urgency(ticket: str) -> int:
    """Rate it."""
    ...


@thunc.function
async def label(text: str) -> Literal["bug", "other"]:
    """Label it."""
    ...


async def check_async() -> None:
    assert_type(await thunc.acall("x"), str)
    assert_type(await thunc.acall("x", returns=int), int)
    assert_type(await thunc.acall("x", returns=Literal["a", "b"]), Literal["a", "b"])
    assert_type(await thunc.amap(urgency, ["a"]), list[int])
    assert_type(await thunc.amap(label, ["a"]), list[Literal["bug", "other"]])
    assert_type(await label("x"), Literal["bug", "other"])


def check_sync() -> None:
    assert_type(thunc.call("x"), str)
    assert_type(thunc.call("x", returns=int), int)
    assert_type(thunc.call("x", returns=Item), Item)
    assert_type(thunc.call("x", returns=Priority), Priority)
    assert_type(thunc.call("x", returns=Literal["a", "b"]), Literal["a", "b"])
    assert_type(thunc.call("x", returns=list[Item]), list[Item])
    assert_type(thunc.call("x", returns=int | None), int | None)
    assert_type(thunc.call("x", returns=dict[str, float]), dict[str, float])
    thunc.call("x", returns=int, ensure=lambda n: n > 0)  # ensure takes the return type
    assert_type(thunc.map(urgency, ["a"]), list[int])
    assert_type(thunc.map(label, ["a"]), list[Literal["bug", "other"]])
    assert_type(urgency("x"), int)


def check_agent(agent: thunc.Agent) -> None:
    assert_type(agent.call("x"), str)
    assert_type(agent.call("x", returns=Literal["yes", "no"]), Literal["yes", "no"])
    assert_type(agent.call("x", returns=list[str]), list[str])
