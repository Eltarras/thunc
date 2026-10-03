"""Return types: parsing model text into checked values."""

import time
from dataclasses import dataclass, field
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


@dataclass
class Ticket:
    category: str
    urgency: int


@dataclass
class Slugged:
    name: str
    slug: str = field(init=False, default="")


@dataclass
class Positive:
    n: int

    def __post_init__(self):
        assert self.n > 0, "n must be positive"


@dataclass
class Filters:  # every field has a default
    city: str | None = None
    max_price: float | None = None


@pytest.mark.parametrize(
    "text, tp, expected",
    [
        # A leading reasoning block, as local models emit
        ("<think>charged twice, so urgent</think>\n4", int, 4),
        ("<THINKING>\nhmm\n</THINKING>\n\ntrue", bool, True),
        ("<think>translate it</think>Bonjour", str, "Bonjour"),
        # Code fences: any language tag, any case, after a line of prose
        ("```JSON\n4\n```", int, 4),
        ("```python\n[1, 2]\n```", list[int], [1, 2]),
        ("```json [1, 2]```", list[int], [1, 2]),
        ("```4```", int, 4),
        ('Here you go:\n```json\n{"category": "bug", "urgency": 4}\n```', Ticket, Ticket("bug", 4)),
        # Invisible characters around the answer
        ("\ufeff4", int, 4),
        ("\u200btrue\u200b", bool, True),
        # An answer wrapped in a one-key object
        ('{"rating": 4}', int, 4),
        ('{"answer": true}', bool, True),
        ('{"category": "bug"}', Literal["bug", "billing"], "bug"),
        ('{"items": ["a", "b"]}', list[str], ["a", "b"]),
        ('{"ticket": {"category": "bug", "urgency": 4}}', Ticket, Ticket("bug", 4)),
        # Literal options keep their own type
        ("3.0", Literal[1, 2, 3], 3),
        ("'bug'", Literal["bug", "billing"], "bug"),
        # A union keeps the value's own type when it's one of the options
        ("3", float | int, 3),
        ("3.5", int | float, 3.5),
        # Fields the class sets itself (init=False) are ignored, not passed to __init__
        ('{"name": "A", "slug": "a"}', Slugged, Slugged("A")),
        # A wrapped object for a class whose fields all have defaults isn't read as empty
        ('{"filters": {"city": "Paris", "max_price": 100}}', Filters, Filters("Paris", 100.0)),
        # Labels that happen to be valid JSON, and near-misses inside an Optional
        ("2", Literal["1", "2", "3"], "2"),
        ("bug", Literal["bug", "billing"] | None, "bug"),
        ("yes", bool | None, True),
        # Invisible characters after a reasoning block
        ("<think>easy</think>\n\u200b4", int, 4),
        # The same wrapper inside an Optional, and an exact label before yes/no
        ('{"filters": {"city": "Paris"}}', Filters | None, Filters("Paris")),
        ("no", Literal["no", "partial"] | bool, "no"),
        # One fence whose JSON mentions a fence mid-line; CRLF line endings; a closing fence mid-line
        ('```json\n{"md": "Run:\\n```bash\\nls\\n```"}\n```', dict[str, str], {"md": "Run:\n```bash\nls\n```"}),
        ("```json\r\n[1]\r\n```", list[int], [1]),
        ("```json\n4```", int, 4),
        # A label that looks like a JSON constant; direction marks around the reply
        ("NaN", Literal["NaN", "ok"], "NaN"),
        ("\u200e4\u200f", int, 4),
        # Whole numbers keep their exact value
        ("9007199254740994.0", int, 9007199254740994),
        ("1e2", int, 100),
    ],
)
def test_recovers_near_misses(text, tp, expected):
    value = parse(text, tp)
    assert value == expected and type(value) is type(expected)


@pytest.mark.parametrize(
    "text, tp, reason",
    [
        ("true", Literal[1, 2, 3], "one of"),  # True == 1 in Python, but not here
        ("false", Literal[0, 1], "one of"),
        ("[1, true]", list[Literal[1, 2]], "one of"),
        ("1", Literal[True, False], "one of"),
        ("NaN", float, "NaN"),
        ("-Infinity", float, "Infinity"),
        ("1e999", float, "out of range"),
        ("[1.5, NaN]", list[float], "NaN"),
        ('{"a": 1, "a": 2}', dict[str, int], "more than once"),
        ('{"category": "bug", "urgency": 1, "urgency": 5}', Ticket, "more than once"),
        ('"bug', Literal["bug", "billing"], "not valid JSON"),  # an unbalanced quote
        ("", str, "empty"),
        ("  \n\u200b ", str, "empty"),
        ("<think>still thinking</think>", str, "empty"),
        pytest.param("[" * 100_000 + "]" * 100_000, int, "nested too deeply", id="deep-nesting"),
        ('{"n": -1}', Positive, "AssertionError"),  # __post_init__ raising anything is a retry
        ('{"rating": "4"}', int, "expected an integer"),  # unwrapped, but still the wrong type
        ('{"a": 4, "b": 5}', int, "expected an integer"),  # two keys: not a wrapper
        ("```json\n1\n```\n```json\n2\n```", int, "not valid JSON"),  # two fences: no guessing
        ("```json\n4\n```\n5", int, "not valid JSON"),  # a second answer after the fence
        pytest.param("1" + "0" * 400, float, "finite number", id="huge-int-for-float"),
        pytest.param('{"order_id": "A", "amount": 1' + "0" * 400 + "}", Order, "finite number", id="huge-field"),
        ('{"filters": "Paris"}', Filters, "expected an object"),  # not silently Filters()
        ('{"confidence": 0.9}', Filters, "expected an object with the fields"),
        ("12345678901234567.0", int, "expected an integer"),  # not ...568, which is what a float holds
        ("3.9999999999999999", int, "expected an integer"),  # not 4: it only rounds to a whole float
        ("1e-400", int, "expected an integer"),  # not 0
        ("0.99999999999999999", Literal[0, 1], "one of"),
        ('{"category": null}', Ticket | None, "field 'category'"),  # a Ticket with a bad field, not None
        ('{"alice": null}', dict[str, int] | None, "expected an integer"),  # a dict can't be a wrapper
    ],
)
def test_rejects_with_a_reason(text, tp, reason):
    with pytest.raises(ValueError, match=reason):
        parse(text, tp)


@pytest.mark.parametrize(
    "text",
    ["4" + "\n" * 200_000 + ".", "```\n" * 50_000 + "done", "```json\n" + "[\n" * 50_000, "<think>" + "x" * 500_000],
    ids=["blank-lines", "fence-lines", "open-fence", "unclosed-think"],
)
def test_long_replies_are_handled_in_linear_time(text):
    started = time.monotonic()
    with pytest.raises(ValueError):
        parse(text, int)
    assert time.monotonic() - started < 1


def test_error_message_shortens_a_huge_answer():
    with pytest.raises(ValueError) as caught:
        parse(str(list(range(10_000))), dict[str, int])
    assert len(str(caught.value)) < 300


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


def test_describe_leaves_out_fields_the_class_sets_itself():
    assert describe(Slugged) == 'a JSON object with these fields: {"name": a JSON string}'


def test_describe_dataclass_marks_optional_fields():
    assert describe(Order) == (
        'a JSON object with these fields: {"order_id": a JSON string, "amount": a JSON number or null (optional)}'
    )
