"""Return types: parsing model text into checked values."""

import copy
import json
import time
from dataclasses import InitVar, dataclass, field
from enum import Enum, IntEnum
from typing import Annotated, Any, ClassVar, Literal

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel, create_model, field_validator

import thunc
from thunc.schema import describe, json_schema, parse


@dataclass
class Order:
    order_id: str
    amount: float | None = None


class Invoice(BaseModel):
    vendor: str
    total_eur: float
    note: str | None = None


class Address(BaseModel):
    city: str


class CustomerModel(BaseModel):
    name: str
    address: Address = Field(description="The customer's address")


class Node(BaseModel):
    name: str
    children: list["Node"] = Field(default_factory=list)


class Cat(BaseModel):
    kind: Literal["cat"]
    lives: int


class Dog(BaseModel):
    kind: Literal["dog"]
    bark: bool


class PetOwner(BaseModel):
    pet: Annotated[Cat | Dog, Field(discriminator="kind")]


class FlexibleAliases(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: int = Field(validation_alias=AliasChoices("first", AliasPath("payload", "value")))


class PydanticFilters(BaseModel):
    tag: str | None = None


class StrictValues(BaseModel):
    count: int
    enabled: bool


class BrokenValidation(BaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def fail(cls, value):
        raise TypeError("broken validator")


class NumberRoot(RootModel[int]):
    pass


class NamesRoot(RootModel[list[str]]):
    pass


class CompatibleModel:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, CompatibleModel) and self.value == other.value

    @classmethod
    def model_json_schema(cls):
        return {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}

    @classmethod
    def model_validate(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get("value"), int):
            raise ValueError("value must be an integer")
        return cls(value["value"])


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
        ('{"vendor": "Acme", "total_eur": 49}', Invoice, Invoice(vendor="Acme", total_eur=49)),
        (
            '{"name": "Ana", "address": {"city": "Bratislava"}}',
            CustomerModel,
            CustomerModel(name="Ana", address=Address(city="Bratislava")),
        ),
        ('{"home": {"city": "Bratislava"}}', dict[str, Address], {"home": Address(city="Bratislava")}),
        ("4", NumberRoot, NumberRoot(4)),
        ('["a", "b"]', NamesRoot, NamesRoot(["a", "b"])),
        ("  hello  ", str, "hello"),
    ],
)
def test_accepts(text, tp, expected):
    assert parse(text, tp) == expected


@pytest.mark.parametrize(
    "tp, expected",
    [
        (
            Invoice,
            'a JSON object with these fields: {"vendor": a JSON string, "total_eur": a JSON number, '
            '"note": a JSON string or null (optional)}',
        ),
        (
            CustomerModel,
            'a JSON object with these fields: {"name": a JSON string, "address": '
            'a JSON object with these fields: {"city": a JSON string}}',
        ),
        (NumberRoot, "a JSON integer"),
        (NamesRoot, "a JSON array whose items are each a JSON string"),
    ],
)
def test_describes_pydantic_models(tp, expected):
    assert describe(tp) == expected


def test_pydantic_validation_errors_are_short():
    with pytest.raises(ValueError) as caught:
        parse('{"vendor": "Acme", "total_eur": "bad"}', Invoice)
    assert "Invoice" in str(caught.value) and "total_eur" in str(caught.value)
    assert len(str(caught.value)) <= 203


def test_pydantic_validation_is_strict_and_validator_errors_are_retried():
    with pytest.raises(ValueError):
        parse('{"count": "3", "enabled": true}', StrictValues)
    with pytest.raises(ValueError):
        parse('{"count": 3, "enabled": "yes"}', StrictValues)
    with pytest.raises(ValueError, match="TypeError.*broken validator"):
        parse('{"value": 1}', BrokenValidation)


def test_wrapped_pydantic_model_with_no_required_fields_is_not_lost():
    assert parse('{"filters": {"tag": "x"}}', PydanticFilters) == PydanticFilters(tag="x")
    with pytest.raises(ValueError, match="fields"):
        parse('{"unknown": 1, "also_unknown": 2}', PydanticFilters)


def test_pydantic_schema_is_self_contained_when_embedded():
    @dataclass
    class Result:
        customers: list[CustomerModel] | None

    tool_schema = {"type": "object", "properties": {"value": json_schema(Result)}, "required": ["value"]}
    shown = repr(tool_schema)
    address = tool_schema["properties"]["value"]["properties"]["customers"]["anyOf"][0]["items"]["properties"][
        "address"
    ]

    assert "$defs" not in shown and "$ref" not in shown
    assert address["description"] == "The customer's address"
    assert address["allOf"][0]["properties"]["city"]["type"] == "string"
    assert json_schema(dict[str, Address])["additionalProperties"]["properties"]["city"]["type"] == "string"


def test_inlining_copies_schemas_and_avoids_definition_collisions():
    original = {
        "$defs": {"Value": {"type": "integer"}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/Value"}},
    }

    class ReusedSchema:
        @classmethod
        def model_json_schema(cls):
            return original

        @classmethod
        def model_validate(cls, value):
            return value

    before = copy.deepcopy(original)
    assert json_schema(ReusedSchema)["properties"]["value"] == {"type": "integer"}
    assert original == before

    left = create_model("Detail", left=(int, ...))
    right = create_model("Detail", right=(str, ...))
    schema = json_schema(create_model("First", detail=(left, ...)) | create_model("Second", detail=(right, ...)))
    assert set(schema["anyOf"][0]["properties"]["detail"]["properties"]) == {"left"}
    assert set(schema["anyOf"][1]["properties"]["detail"]["properties"]) == {"right"}


def test_recursive_and_discriminated_pydantic_models():
    recursive = json_schema(Node)
    assert recursive["properties"]["children"]["items"] == {"type": "object"}
    assert "$ref" not in repr(recursive)

    discriminated = json_schema(PetOwner)
    shown = repr(discriminated)
    choices = discriminated["properties"]["pet"]["oneOf"]
    assert "$ref" not in shown and "discriminator" not in shown
    assert [choice["properties"]["kind"]["const"] for choice in choices] == ["cat", "dog"]
    assert "exactly this JSON value" in describe(PetOwner)


@pytest.mark.parametrize("field", ["value", "first", "payload"])
def test_pydantic_field_names_and_aliases_are_not_unwrapped(field):
    with pytest.raises(ValueError):
        parse(json.dumps({field: {"first": 3}}), FlexibleAliases)
    assert parse('{"result": {"first": 3}}', FlexibleAliases) == FlexibleAliases(first=3)


def test_duck_typed_models_and_existing_types_keep_their_behavior():
    assert parse('{"value": 3}', CompatibleModel) == CompatibleModel(3)
    assert describe(CompatibleModel) == 'a JSON object with these fields: {"value": a JSON integer}'

    @dataclass
    class CompatibleDataclass:
        value: int

        @classmethod
        def model_json_schema(cls):
            return {"type": "string"}

        @classmethod
        def model_validate(cls, value):
            return "wrong path"

    class CompatibleEnum(str, Enum):
        VALUE = "value"

        @classmethod
        def model_json_schema(cls):
            return {"type": "object"}

        @classmethod
        def model_validate(cls, value):
            return "wrong path"

    assert parse('{"value": 3}', CompatibleDataclass) == CompatibleDataclass(3)
    assert json_schema(CompatibleDataclass)["properties"]["value"] == {"type": "integer"}
    assert parse('"value"', CompatibleEnum) is CompatibleEnum.VALUE
    assert json_schema(CompatibleEnum) == {"enum": ["value"]}


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


@dataclass
class Scaled:
    name: str
    scale: InitVar[int]

    def __post_init__(self, scale):
        self.size = len(self.name) * scale


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class Level(IntEnum):
    ZERO = 0
    ONE = 1


COLOR = Literal[Color.RED, Color.BLUE]


@pytest.mark.parametrize(
    "text, tp, expected",
    [
        ('["red", "blue"]', list[COLOR], [Color.RED, Color.BLUE]),
        ('"blue"', COLOR, Color.BLUE),
        ("red", COLOR, Color.RED),
        ("1", Literal[Level.ZERO, Level.ONE], Level.ONE),
    ],
)
def test_enum_members_as_literal_options_come_back_as_members(text, tp, expected):
    value = parse(text, tp)
    assert value == expected and repr(value) == repr(expected)


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
        # A union takes the first option that accepts the value, in the order written
        ("3", float | int, 3.0),
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
        # The answer on the opening line of a fence, alone or after a tag; a tag with attributes
        ("```4\n```", int, 4),
        ("```[1,\n2]\n```", list[int], [1, 2]),
        ('```json {"order_id": "A-1"}\n```', Order, Order("A-1")),
        ("```true\n```", bool, True),
        ("```json title=answer.json\n[1]\n```", list[int], [1]),
        # A label of several words on the opening line is the label, not a tag and the rest of it
        ("```not urgent\n```", Literal["urgent", "not urgent"], "not urgent"),
        ("```top 10\n```", Literal["top 10", "top 5"], "top 10"),
        # "json" running straight into the answer, as main read it
        ('```json{"a": 1}\n```', dict[str, int], {"a": 1}),
        ('```json"bug"\n```', Literal["bug", "billing"], "bug"),
        # A label that looks like a JSON constant; direction marks around the reply
        ("NaN", Literal["NaN", "ok"], "NaN"),
        ("\u200e4\u200f", int, 4),
        ("1e2", int, 100),
        # A float option that no integer text equals exactly, asked for as describe() writes it
        ("1e+23", Literal[1e23, "none"], 1e23),
        # Direction marks, embeddings and isolates around the reply
        ("\u20664\u2069", int, 4),
        ("\u202a4\u202c", int, 4),
        ("\u061c4\u061c", int, 4),
        # A label that has quotes of its own, sent exactly
        ("'quoted'", Literal["'quoted'", "other"], "'quoted'"),
        ('"x"', Literal['"x"', "other"], '"x"'),
        # Exponents beyond what Decimal holds are still numbers
        ("0e99999999999999999999", float, 0.0),
        ("[1e-99999999999999999999]", list[float], [0.0]),
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
        (".yes", bool, "not valid JSON"),
        # An answer, then a fence with another answer: two answers
        ("4\n```json\n5\n```", int, "two answers"),
        ("true\n```json\nfalse\n```", bool, "two answers"),
        ("bug\n```\nbilling\n```", Literal["bug", "billing"], "two answers"),
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


def test_very_deep_json_is_a_valueerror():
    # Python 3.14 limits recursion by the real stack size, so with a big stack json.loads can read
    # this, and the value is then the wrong type; with a small one it's too deep. A retry either way.
    with pytest.raises(ValueError, match="nested too deeply|expected an integer"):
        parse("[" * 100_000 + "]" * 100_000, int)


@pytest.mark.parametrize("tp", [int, list[int], dict[str, int], int | None, Literal["a"]])
def test_json_just_under_the_depth_limit_is_a_valueerror(tp):
    # Somewhere below json's own depth limit (where exactly depends on the stack in use), json.loads
    # still works but checking the value, or showing it in the error, is too deep.
    for depth in range(2, 1001):  # "[]" is a valid list[int]
        with pytest.raises(ValueError):
            parse("[" * depth + "]" * depth, tp)


def test_deep_answer_before_a_fence_is_still_a_second_answer():
    for depth in range(2, 1001):
        before = "[" * depth + "]" * depth
        try:
            parse(before, Any)
        except ValueError:
            continue  # too deep to be an answer on its own
        with pytest.raises(ValueError, match="two answers"):
            parse(before + "\n```json\n4\n```", Any)


@dataclass
class Customer:
    id: str
    age: int


@dataclass
class Purchase:
    id: str = ""
    customer: Customer | None = None


@pytest.mark.parametrize("text", ['{"customer": {"id": "c1", "age": "old"}}', '{"customer": {"id": "c1"}}'])
def test_a_bad_nested_object_is_not_read_as_the_outer_class(text):
    # {"customer": {...}} is a Purchase with a bad customer, not a wrapper around a Purchase with id "c1".
    with pytest.raises(ValueError, match="field 'customer'"):
        parse(text, Purchase)


def test_a_union_tries_its_options_in_order():
    assert parse('{"order_id": "A-1", "amount": 3}', Order | dict) == Order("A-1", 3.0)
    assert parse('[{"order_id": "A-1"}]', list[Order] | list) == [Order("A-1")]


@pytest.mark.parametrize("tp", [Any, list, dict[str, Any], list[Any]])
def test_unchecked_values_hold_plain_floats(tp):
    text = '{"x": [1e23, 3.9999999999999999]}' if tp in (Any, dict[str, Any]) else "[[1e23, 3.9999999999999999]]"
    value = parse(text, tp)
    numbers = value["x"] if isinstance(value, dict) else value[0]
    assert [type(n) for n in numbers] == [float, float] and repr(numbers) == "[1e+23, 4.0]"


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


def test_initvar_is_asked_for_and_passed_to_post_init():
    assert describe(Scaled) == 'a JSON object with these fields: {"name": a JSON string, "scale": a JSON integer}'
    assert parse('{"name": "ab", "scale": 3}', Scaled).size == 6


@dataclass
class Counter:
    n: int
    registry: ClassVar = {}  # a bare ClassVar


def test_classvars_are_not_asked_for_nor_read():
    assert describe(Counter) == 'a JSON object with these fields: {"n": a JSON integer}'
    assert parse('{"n": 3, "registry": 5}', Counter) == Counter(3)


def test_string_annotations_in_a_future_annotations_module():
    from future_types import Holder, Item, Scaled

    assert (
        describe(Scaled) == 'a JSON object with these fields: {"a": a JSON integer, "scale": a JSON integer (optional)}'
    )
    assert parse('{"a": 2, "scale": 3}', Scaled).a == 6
    assert parse('{"items": [{"sku": "A"}], "table": {"b": {"sku": "B"}}}', Holder) == Holder(
        [Item("A")], {"b": Item("B")}
    )


def test_describe_leaves_out_fields_the_class_sets_itself():
    assert describe(Slugged) == 'a JSON object with these fields: {"name": a JSON string}'


def test_describe_dataclass_marks_optional_fields():
    assert describe(Order) == (
        'a JSON object with these fields: {"order_id": a JSON string, "amount": a JSON number or null (optional)}'
    )


# --- Enum classes as return types -----------------------------------------------------------


class Priority(Enum):
    LOW = 1
    HIGH = 3


class Mood(Enum):
    happy = "happy"
    sad = "sad"
    MEH = "meh"
    SO_SO = "meh"  # an alias: the same member as MEH


@pytest.mark.parametrize(
    "text, tp, expected",
    [
        ('"red"', Color, Color.RED),
        ("blue", Color, Color.BLUE),  # an unquoted label, as for a Literal
        ("'blue'", Color, Color.BLUE),
        ("3", Priority, Priority.HIGH),
        ("3.0", Priority, Priority.HIGH),
        ('"meh"', Mood, Mood.MEH),
        ("1", Level, Level.ONE),
        ('["red", "blue"]', list[Color], [Color.RED, Color.BLUE]),
        ("null", Color | None, None),
        ('{"mood": "sad"}', Mood, Mood.sad),  # an answer in a one-key object, unwrapped
    ],
)
def test_enum_classes_come_back_as_members(text, tp, expected):
    value = parse(text, tp)
    assert value == expected and repr(value) == repr(expected)


@pytest.mark.parametrize(
    "text, tp",
    [
        ('"RED"', Color),  # a name, not a value: asked again rather than guessed
        ("HIGH", Priority),
        ('"3"', Priority),  # a string, not the number
        ("true", Level),  # true isn't 1
        ("2", Priority),
        ('"purple"', Color),
    ],
)
def test_enum_answers_that_are_not_a_value_are_retried(text, tp):
    with pytest.raises(ValueError):
        parse(text, tp)


def test_an_enum_is_described_by_its_values_with_names_where_they_add_meaning():
    assert describe(Color) == 'exactly one of these JSON values: "red", "blue"'
    assert describe(Priority) == "exactly one of these JSON values: 1 (LOW), 3 (HIGH)"
    assert describe(Mood) == 'exactly one of these JSON values: "happy", "sad", "meh"'  # no alias
    assert json_schema(Priority) == {"enum": [1, 3]}
    assert json_schema(list[Color]) == {"type": "array", "items": {"enum": ["red", "blue"]}}


def test_enums_json_cant_carry_and_flags_are_unsupported():
    import enum

    class Point(Enum):
        ORIGIN = (0, 0)

    class Empty(Enum):
        pass

    class Perm(enum.Flag):
        R = 1
        W = 2

    for tp in (Point, Empty, Perm):
        with pytest.raises(thunc.ThuncError, match="Unsupported return type"):
            describe(tp)


def test_a_function_returning_an_enum(fake):
    @thunc.function
    def priority(ticket: str) -> Priority:
        """How urgent is this?"""
        ...

    fake.replies = ["HIGH", "3"]  # the name is retried with the values listed
    assert priority("Site down") is Priority.HIGH
    assert "1 (LOW), 3 (HIGH)" in fake.prompts[0] and "invalid" in fake.prompts[1]
