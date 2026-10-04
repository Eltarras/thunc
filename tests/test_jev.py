"""The jev backend: type mapping, the request it sends, reading the answer (subprocess is stubbed)."""

import dataclasses
import datetime
import json
import subprocess
from typing import Literal

import pytest

import thunc
from thunc import backends


def stub_jev(monkeypatch, answer=None, *, stdout=None, stderr="", returncode=0):
    """Stub `jev ask`. `answer` is the answer to the one question; the calls are returned."""
    calls = []
    if stdout is None:
        stdout = json.dumps({"model": "jev-latest", "answers": {"answer": answer}, "usage": {"input_tokens": 9}})

    def run(args, **kwargs):
        calls.append({"args": args, "request": json.loads(kwargs["input"]), "env": kwargs.get("env")})
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "run", run)
    thunc.configure(backend="jev")
    return calls


# Answers in the shape jev 2026.919.0 prints (jev-1.13.0).
def chose(label, probabilities):
    return {
        "choice": label,
        "confidence": max(probabilities.values()),
        "probabilities": probabilities,
        "type": "choice",
    }


def scored(probabilities):
    """A score answer; `probabilities` lists one per level, lowest first. "score" is the expected position."""
    expected = sum(position * p for position, p in enumerate(probabilities))
    return {
        "confidence": max(probabilities),
        "legend": {str(position): "?" for position in range(len(probabilities))},
        "probabilities": {str(position): p for position, p in enumerate(probabilities)},
        "score": round(expected, 2),
        "type": "score",
    }


@pytest.mark.parametrize(("probability", "expected"), [(0.62, True), (0.5, True), (0.41, False), (0, False)])
def test_bool_is_a_noul_question(monkeypatch, probability, expected):
    calls = stub_jev(monkeypatch, {"noul": probability})
    assert thunc.call("Is this spam?", {"email": "WIN A PRIZE"}, returns=bool) is expected
    assert calls[0]["args"] == ["jev", "ask", "-"]
    assert calls[0]["request"] == {
        "state": {"email": "WIN A PRIZE"},
        "questions": {"answer": {"type": "noul", "instructions": "Is this spam?"}},
    }


def test_string_literal_is_a_choice_question(monkeypatch):
    calls = stub_jev(monkeypatch, chose("billing", {"bug": 0.2, "billing": 0.7, "other": 0.1}))
    assert thunc.call(
        "Route this ticket.", {"ticket": "charged twice"}, returns=Literal["bug", "billing", "other"]
    ) == ("billing")
    question = calls[0]["request"]["questions"]["answer"]
    assert question["type"] == "choice"
    assert question["criteria"] == {"bug": "bug", "billing": "billing", "other": "other"}


def test_integer_literal_is_a_score_question_with_levels_in_order(monkeypatch):
    calls = stub_jev(monkeypatch, scored([0.1, 0.6, 0.3]))
    value = thunc.call("How urgent is this?", {"ticket": "site down"}, returns=Literal[5, 1, 3])
    assert value == 3 and type(value) is int
    question = calls[0]["request"]["questions"]["answer"]
    assert question["type"] == "score"
    assert question["criteria"] == ["1", "3", "5"]


def test_score_is_the_most_likely_level_not_the_expected_position(monkeypatch):
    # Expected position 2.0 would be the middle level, which Jev thinks least likely.
    stub_jev(monkeypatch, scored([0.0, 0.5, 0.0, 0.5 - 0.01, 0.01]))
    assert thunc.call("How urgent?", returns=Literal[1, 2, 3, 4, 5]) == 2


@dataclasses.dataclass
class Ticket:
    title: str
    opened: datetime.date


@dataclasses.dataclass
class Verdict:
    spam: bool
    reason: str


@pytest.mark.parametrize(
    "returns",
    [
        str,
        int,
        float,
        list[str],
        dict[str, int],
        Verdict,
        bool | None,
        Literal["a", 1],
        Literal[True, False],
        Literal[tuple(f"label{i}" for i in range(256))],
    ],
)
def test_types_jev_cannot_answer_fail_before_any_request(monkeypatch, returns):
    calls = stub_jev(monkeypatch, {"noul": 0.9})
    with pytest.raises(thunc.ThuncError, match="The jev backend answers bool, Literal of strings"):
        thunc.call("Anything?", returns=returns)
    assert calls == []


def test_255_choices_are_allowed(monkeypatch):
    labels = tuple(f"label{i}" for i in range(255))
    stub_jev(monkeypatch, chose("label254", {"label254": 0.9, "label0": 0.1}))
    assert thunc.call("Pick one.", returns=Literal[labels]) == "label254"


def test_inputs_are_the_state(monkeypatch):
    calls = stub_jev(monkeypatch, {"noul": 0.9})
    thunc.call("Overdue?", {"ticket": Ticket("Login fails", datetime.date(2026, 10, 1)), "days": 3}, returns=bool)
    assert calls[0]["request"]["state"] == {"ticket": {"title": "Login fails", "opened": "2026-10-01"}, "days": 3}


def test_no_inputs_is_an_empty_state(monkeypatch):
    calls = stub_jev(monkeypatch, {"noul": 0.1})
    assert thunc.call("Is 1 = 3?", returns=bool) is False
    assert calls[0]["request"]["state"] == {}


def test_a_custom_system_prompt_goes_before_the_instructions(monkeypatch):
    calls = stub_jev(monkeypatch, {"noul": 0.9})
    thunc.call("Is this spam?", {"email": "hi"}, returns=bool, system="You triage a shop's inbox.")
    thunc.configure(system="You moderate a forum.")
    thunc.call("Is this spam?", {"email": "hi"}, returns=bool)
    instructions = [call["request"]["questions"]["answer"]["instructions"] for call in calls]
    assert instructions == [
        "You triage a shop's inbox.\n\nIs this spam?",
        "You moderate a forum.\n\nIs this spam?",
    ]


def test_thuncs_default_system_prompt_is_not_sent(monkeypatch):
    calls = stub_jev(monkeypatch, {"noul": 0.9})
    thunc.call("Is this spam?", {"email": "hi"}, returns=bool)
    assert "function inside a computer program" not in json.dumps(calls[0]["request"])


def test_function_docstring_is_the_question(monkeypatch):
    calls = stub_jev(monkeypatch, {"noul": 0.8})

    @thunc.function(backend="jev")
    def is_true(statement: str) -> bool:
        """Is this statement true?"""
        ...

    assert is_true("2 + 2 = 4") is True
    assert calls[0]["request"] == {
        "state": {"statement": "2 + 2 = 4"},
        "questions": {"answer": {"type": "noul", "instructions": "Is this statement true?"}},
    }


def test_another_backends_api_key_is_not_sent_to_jev(monkeypatch):
    # The usual setup: Claude for most functions, Jev for the yes/no and label ones.
    calls = stub_jev(monkeypatch, {"noul": 0.9})
    thunc.configure(backend="anthropic", api_key="sk-ant-test")
    thunc.call("Spam?", returns=bool, backend="jev")
    assert calls[0]["env"] is None  # inherited: the CLI finds JEV_API_KEY or its stored login


def test_an_ensure_rejection_is_not_retried(monkeypatch):
    calls = stub_jev(monkeypatch, {"noul": 0.2})
    with pytest.raises(thunc.ThuncError, match="after 1 attempt"):
        thunc.call("Spam?", returns=bool, ensure=lambda value: value is True, retries=5)
    assert len(calls) == 1


def test_cached_answer_skips_the_cli(monkeypatch, tmp_path):
    calls = stub_jev(monkeypatch, chose("bug", {"bug": 0.9, "billing": 0.1}))
    thunc.configure(cache_dir=str(tmp_path / "cache"))
    for _ in range(2):
        assert thunc.call("Route it.", {"t": "crash"}, returns=Literal["bug", "billing"], cache=True) == "bug"
    assert len(calls) == 1


def test_a_configured_model_is_not_used_for_jev(monkeypatch, tmp_path):
    stub_jev(monkeypatch, {"noul": 0.9})
    thunc.configure(model="gpt-5.5", trace=str(tmp_path / "calls.jsonl"))
    thunc.call("Spam?", returns=bool, model="claude-opus-5-5")
    entry = json.loads((tmp_path / "calls.jsonl").read_text())
    assert (entry["backend"], entry["model"], entry["answers"]) == ("jev", "jev-latest", ["true"])
    assert entry["system"] is None  # thunc's default prompt isn't sent to Jev, so it isn't traced


def test_env_selects_jev(monkeypatch):
    stub_jev(monkeypatch, {"noul": 0.9})
    monkeypatch.setattr(thunc.config, "_settings", dict(thunc.config.DEFAULTS))  # undo the stub's configure()
    monkeypatch.setenv("THUNC_BACKEND", "jev")
    assert thunc.call("Spam?", returns=bool) is True


def test_jev_key_alone_does_not_select_jev(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "jev-test-key")
    with pytest.raises(thunc.ThuncError, match="No backend configured"):
        thunc.call("Spam?", returns=bool)


def test_cli_error(monkeypatch):
    stub_jev(monkeypatch, stdout="", stderr="error: not logged in; run `jev login`", returncode=1)
    with pytest.raises(thunc.ThuncError, match="jev exited 1: error: not logged in"):
        thunc.call("Spam?", returns=bool)


@pytest.mark.parametrize(
    "stdout", ["", "not json", "[]", '{"answers": {}}', '{"answers": {"answer": {"choice": "x"}}}']
)
def test_output_without_an_answer(monkeypatch, stdout):
    stub_jev(monkeypatch, stdout=stdout)
    with pytest.raises(thunc.ThuncError, match="jev returned no answer"):
        thunc.call("Spam?", returns=bool)


@pytest.mark.parametrize("probability", ["0.9", True, 1.5, -0.1, None, float("nan")])
def test_noul_that_is_not_a_probability(monkeypatch, probability):
    stub_jev(monkeypatch, {"noul": probability})
    with pytest.raises(thunc.ThuncError, match="not a probability"):
        thunc.call("Spam?", returns=bool)


@pytest.mark.parametrize("choice", ["feature", "Bug", "", None, 1, ["bug"], {"bug": 1.0}])
def test_choice_that_is_not_one_of_the_labels(monkeypatch, choice):
    stub_jev(monkeypatch, {"choice": choice, "type": "choice"})
    with pytest.raises(thunc.ThuncError, match=r"not one of \['bug', 'billing'\]"):
        thunc.call("Route it.", returns=Literal["bug", "billing"])


@pytest.mark.parametrize(
    "probabilities",
    [{}, {"0": 0.5, "1": 0.5}, {"0": 0.2, "1": 0.3, "3": 0.5}, {"0": 0.2, "1": 0.3, "2": "0.5"}, [0.2, 0.3, 0.5]],
)
def test_score_without_a_probability_for_each_level(monkeypatch, probabilities):
    stub_jev(monkeypatch, {"probabilities": probabilities, "score": 1.3, "type": "score"})
    with pytest.raises(thunc.ThuncError, match="not probabilities for the positions"):
        thunc.call("How urgent?", returns=Literal[1, 2, 3])


def test_score_without_probabilities(monkeypatch):
    stub_jev(monkeypatch, {"score": 1.3, "type": "score"})
    with pytest.raises(thunc.ThuncError, match="jev returned no answer"):
        thunc.call("How urgent?", returns=Literal[1, 2, 3])


def test_missing_cli(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda exe: None)
    thunc.configure(backend="jev")
    with pytest.raises(thunc.ThuncError, match="`jev` was not found on PATH"):
        thunc.call("Spam?", returns=bool)
