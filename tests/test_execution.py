"""The shared agent engine: the decisions agent.run() and durable runs both make."""

import json

import pytest

from thunc.errors import ThuncError
from thunc.execution import AgentState
from thunc.native import Call, Reply


def through_json(state: AgentState) -> AgentState:
    return AgentState.from_json(json.loads(json.dumps(state.to_json())))


def test_unknown_and_malformed_calls_are_errors_not_denials():
    state = AgentState()
    for call in (Call(None, "teleport"), Call("c1", "read", problem="arguments aren't JSON")):
        outcome = state.check(call, str, None, 2, "t")
        assert outcome.output and outcome.output.startswith("error: ") and not outcome.finished
        state.done(call, outcome.output)
    assert state.denied == [] and [failed for _, _, failed in state.results] == [True, True]
    assert "unknown tool 'teleport'" in state.results[0][1]


def test_invalid_finish_is_repaired_then_fails_the_run():
    state = AgentState()
    bad = Call(None, "finish", {"value": "nope"})
    first = state.check(bad, int, None, 1, "count")
    assert first.output and "that value is invalid" in first.output and "Call finish again" in first.output
    assert first.error is None
    state = through_json(state)  # the count survives a restart
    second = state.check(bad, int, None, 1, "count")
    assert isinstance(second.error, ThuncError) and "after 2 finish attempt(s)" in str(second.error)
    good = state.check(Call(None, "finish", {"value": "4"}), int, None, 1, "count")
    assert good.finished and good.value == 4


def test_tools_are_left_to_the_executor():
    outcome = AgentState().check(Call(None, "read", {"path": "a"}), str, None, 2, "t")
    assert outcome.output is None and not outcome.finished


def test_denials_and_notes():
    state = AgentState()
    state.done(Call(None, "write", {"path": "x"}), "error: not permitted: may not write x", denied=True)
    state.done(Call(None, "remember", {"note": "  keep\n it "}), "saved.")
    state.done(Call(None, "remember", {"note": 3}), 'error: remember takes {"note": "..."}')
    assert [(d.tool, d.target, d.reason) for d in state.denied] == [("write", "x", "may not write x")]
    assert state.notes == ["keep it"]  # a failed remember saves nothing


def test_a_reply_resumes_mid_way_through_its_calls():
    calls = [Call("a", "read", {"path": "1"}), Call("b", "read", {"path": "2"}), Call("c", "read", {"path": "3"})]

    def settle(state: AgentState, call: Call) -> None:
        state.done(call, f"contents of {call.args['path']}")

    straight = AgentState()
    straight.begin_turn(5)
    assert straight.receive(Reply(calls, raw="..."))
    while (call := straight.pending()) is not None:
        settle(straight, call)

    resumed = AgentState()
    resumed.begin_turn(5)
    resumed.receive(Reply(calls, raw="..."))
    settle(resumed, resumed.pending())  # type: ignore[arg-type]
    resumed = through_json(resumed)  # the worker restarts between two calls of one reply
    assert resumed.pending() == calls[1]
    while (call := resumed.pending()) is not None:
        settle(resumed, call)
    assert resumed == straight and resumed.operations == 3


def test_a_reply_without_calls_asks_again_and_turns_are_bounded():
    state = AgentState()
    state.begin_turn(1)
    assert not state.receive(Reply([Call(None, "read")], raw="?", problem="no tool call"))
    assert state.pending() is None
    with pytest.raises(ThuncError, match="t: the agent didn't finish within max_steps=1"):
        state.begin_turn(1, "t")


def test_the_last_replies_before_max_steps_say_how_many_are_left():
    state = AgentState()
    sent = []
    for _ in range(5):
        state.begin_turn(5)
        state.receive(Reply([Call(None, "read", {"path": "a"}), Call(None, "list")], "raw"))
        state.done(state.calls[0], "first")
        state.done(state.calls[1], "second")
        sent.append(state.results_to_send(5))
    notes = [results[-1][1].partition("\n\n")[2] for results in sent]
    soon = "(You have {} replies left before the run's step limit: call finish soon with what you have.)"
    last = "(This is your last reply before the run's step limit: call finish now with what you have.)"
    # After reply 1, 4 are left: nothing yet. Then 3, 2 and 1. After reply 5 none are: the run ends.
    assert notes == ["", soon.format(3), soon.format(2), last, ""]
    assert all(results[0][1] == "first" for results in sent)  # only the last result carries it
    assert state.results[-1][1] == "second"  # what's recorded is unchanged


def test_a_reply_with_no_results_gets_no_countdown():
    state = AgentState()
    state.begin_turn(1)
    assert state.results_to_send(1) == []


def test_finish_with_other_calls_in_its_reply_is_refused_except_beside_a_note():
    state = AgentState()
    state.receive(Reply([Call(None, "read", {"path": "a"}), Call(None, "finish", {"value": 1})], "raw"))
    refused = state.check(state.calls[1], int, None, 2, "t")
    assert not refused.finished and refused.output and refused.output.startswith("error: finish wasn't run")
    assert state.bad_finishes == 0  # not an invalid value: no repair is used up
    state.receive(Reply([Call(None, "remember", {"note": "n"}), Call(None, "finish", {"value": 1})], "raw"))
    assert state.check(state.calls[1], int, None, 2, "t").finished
