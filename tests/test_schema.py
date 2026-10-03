"""Return types: parsing model text into checked values."""

from dataclasses import dataclass
from typing import Literal

import pytest

from thunc.schema import describe, parse


@dataclass
class Order:
    order_id: str
    amount: float | None = None


@pytest.mark.parametrize(
    "text, tp, expected",
    [
        ("true", bool, True),
        ("No.", bool, False),
        ("42", int, 42),
        ("3.0", int, 3),
        ("0.75", float, 0.75),
        ('"billing"', Literal["bug", "billing"], "billing"),
        ("billing", Literal["bug", "billing"], "billing"),
        ('```json\n["a", "b"]\n```', list[str], ["a", "b"]),
        ('{"total": 12.5}', dict[str, float], {"total": 12.5}),
        ("null", int | None, None),
        ('{"order_id": "A-1", "amount": 49}', Order, Order("A-1", 49.0)),
        ('{"order_id": "A-1"}', Order, Order("A-1")),
        ("  hello  ", str, "hello"),
    ],
)
def test_accepts(text, tp, expected):
    assert parse(text, tp) == expected


@pytest.mark.parametrize(
    "text, tp",
    [
        ("maybe", bool),
        ("true", int),
        ("4.5", int),
        ('"spam"', Literal["bug"]),
        ('[1, "x"]', list[int]),
        ('{"amount": 3}', Order),  # missing required field
        ('{"order_id": 7}', Order),  # wrong field type
    ],
)
def test_rejects(text, tp):
    with pytest.raises(ValueError):
        parse(text, tp)


def test_describe_dataclass_marks_optional_fields():
    assert describe(Order) == (
        'a JSON object with these fields: {"order_id": a JSON string, "amount": a JSON number or null (optional)}'
    )
