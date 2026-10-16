"""Record and replay (thunc/recordings.py): THUNC_RECORD=1 saves every answer to the recordings folder,
and with only the folder set every call and agent run is answered from it, with no model asked."""

import asyncio
import enum
import importlib
import json
import warnings
from dataclasses import dataclass
from typing import Literal

import pytest

import thunc
from thunc import config, core, events, recordings
from thunc.cache import _info

agent_module = importlib.import_module("thunc.agent")  # thunc.agent is also the @thunc.agent decorator


def act(tool, **args):
    return json.dumps({"tool": tool, "args": args})


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / "recordings"
    thunc.configure(recordings=str(path))
    return path


@pytest.fixture
def recording(monkeypatch, folder):
    """Switch between recording (THUNC_RECORD=1 or missing) and replaying (unset) in one test."""

    def switch(value):
        if value is None:
            monkeypatch.delenv("THUNC_RECORD", raising=False)
        else:
            monkeypatch.setenv("THUNC_RECORD", value)

    return switch


def no_backend(monkeypatch):
    """As in CI: no backend configured, no API key, so asking a model would fail."""
    monkeypatch.setitem(config._settings, "backend", None)


@thunc.function
def urgency(ticket: str) -> int:
    """Rate how urgent this ticket is, from 1 to 5."""
    ...


# --- calls ---------------------------------------------------------------------------------


def test_recorded_answers_replay_without_a_model_or_a_backend(fake, recording, folder, monkeypatch):
    recording("1")
    fake.replies = ["4", "2", "yes"]
    assert urgency("I was charged twice!") == 4
    assert urgency("Typo on the pricing page") == 2
    assert thunc.call("Is this spam?", {"text": "WIN NOW"}, returns=bool) is True
    assert len(fake.prompts) == 3 and len(list(folder.iterdir())) == 3

    recording(None)
    no_backend(monkeypatch)
    assert urgency("Typo on the pricing page") == 2
    assert urgency("I was charged twice!") == 4
    assert thunc.call("Is this spam?", {"text": "WIN NOW"}, returns=bool) is True
    assert len(fake.prompts) == 3  # nothing more reached the model


def test_the_environment_alone_turns_replaying_on(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("THUNC_RECORDINGS", str(tmp_path / "rec"))
    monkeypatch.setenv("THUNC_RECORD", "1")
    fake.replies = ["3"]
    assert urgency("Slow page") == 3
    monkeypatch.delenv("THUNC_RECORD")
    assert urgency("Slow page") == 3 and len(fake.prompts) == 1


def test_an_answer_that_was_not_recorded_fails_loudly_and_asks_no_model(fake, folder):
    with pytest.raises(thunc.ThuncError, match=r"No recorded answer for urgency\(\).*THUNC_RECORD=missing"):
        urgency("Never seen")
    with pytest.raises(thunc.ThuncError, match=r"No recorded answer for the call 'Summarize the notes\.'"):
        thunc.call("Summarize the notes.", {"notes": "..."})
    assert fake.prompts == []


def test_a_call_is_matched_on_what_the_program_wrote_so_changing_it_needs_a_new_recording(fake, recording):
    recording("1")
    fake.replies = ["4"]
    assert urgency("Down for everyone") == 4
    recording(None)

    with pytest.raises(thunc.ThuncError, match="No recorded answer"):
        urgency("Down for everyone!")  # another input
    with pytest.raises(thunc.ThuncError, match="No recorded answer"):
        thunc.call("Rate how urgent this ticket is, from 1 to 5.", {"ticket": "Down for everyone"}, returns=str)
    with pytest.raises(thunc.ThuncError, match="No recorded answer"):
        thunc.call(
            "Rate how urgent this ticket is, from 1 to 5.",
            {"ticket": "Down for everyone"},
            returns=int,
            system="You triage support tickets.",
        )
    # The function's name is part of the match, as in the cache: the same prompt from thunc.call isn't it.
    with pytest.raises(thunc.ThuncError, match="No recorded answer"):
        thunc.call("Rate how urgent this ticket is, from 1 to 5.", {"ticket": "Down for everyone"}, returns=int)
    assert thunc.call(
        "Rate how urgent this ticket is, from 1 to 5.", {"ticket": "Down for everyone"}, returns=int, name="urgency"
    )


def test_the_backend_and_model_are_saved_but_not_matched(fake, recording, folder):
    recording("1")
    fake.replies = ["4"]
    thunc.configure(model="small-model")
    assert urgency("Login broken") == 4
    (entry,) = [json.loads(path.read_text()) for path in folder.iterdir()]
    assert entry["backend"] == "fake" and entry["model"] == "small-model" and entry["function"] == "urgency"
    assert entry["answer"] == "4" and "<ticket>\nLogin broken\n</ticket>" in entry["request"]
    assert entry["module"] == __name__

    recording(None)
    thunc.configure(model="another-model")
    assert urgency("Login broken") == 4


def test_a_recorded_answer_is_checked_again_and_a_stale_one_fails(fake, recording):
    recording("1")
    fake.replies = ["9"]

    @thunc.function
    def score(text: str) -> int:
        """Score it."""
        ...

    assert score("x") == 9
    recording(None)

    @thunc.function(ensure=lambda n: n <= 5)
    def score(text: str) -> int:  # noqa: F811 - the same function, now with a check
        """Score it."""
        ...

    stale = r"recorded answer for [\w.<>]*score\(\).*no longer fits.*THUNC_RECORD=1"
    with pytest.raises(thunc.ThuncError, match=stale):
        score("x")
    assert len(fake.prompts) == 1


def test_recording_only_the_missing_answers_replays_the_rest(fake, recording, folder):
    recording("1")
    fake.replies = ["4"]
    assert urgency("Old ticket") == 4

    recording("missing")
    fake.replies = ["1"]
    assert urgency("Old ticket") == 4  # replayed, not asked
    assert urgency("New ticket") == 1  # asked, and saved
    assert len(fake.prompts) == 2

    recording(None)
    assert urgency("New ticket") == 1 and len(list(folder.iterdir())) == 2


def test_recording_missing_answers_asks_again_for_a_stale_one(fake, recording):
    recording("1")
    fake.replies = ["nine"]
    assert thunc.call("Spell it.", returns=str) == "nine"
    recording("missing")
    fake.replies = ["9"]
    assert thunc.call("Spell it.", returns=str, ensure=lambda s: s.isdigit()) == "9"
    recording(None)
    assert thunc.call("Spell it.", returns=str, ensure=lambda s: s.isdigit()) == "9"


def test_recording_every_answer_asks_again_and_replaces_the_old_one(fake, recording):
    recording("1")
    fake.replies = ["2", "5"]
    assert urgency("Flaky") == 2
    assert urgency("Flaky") == 5
    recording(None)
    assert urgency("Flaky") == 5


def test_only_the_answer_that_passed_is_recorded_and_failures_never_are(fake, recording, folder):
    recording("1")
    fake.replies = ["high", "4"]  # the first is retried
    assert urgency("Retried") == 4
    fake.replies = ["no", "no", "no"]
    with pytest.raises(thunc.ThuncError):
        urgency("Failed")
    (entry,) = [json.loads(path.read_text()) for path in folder.iterdir()]
    assert entry["answer"] == "4"

    recording(None)
    assert urgency("Retried") == 4
    with pytest.raises(thunc.ThuncError, match="No recorded answer"):
        urgency("Failed")


def test_the_cache_is_left_alone_while_replaying_and_a_cache_hit_is_recorded(fake, monkeypatch, tmp_path):
    thunc.configure(cache_dir=str(tmp_path / "cache"))
    fake.replies = ["3"]
    assert thunc.call("Rate it.", {"n": 1}, returns=int, cache=True) == 3  # cached, before recording

    monkeypatch.setenv("THUNC_RECORDINGS", str(tmp_path / "recordings"))
    monkeypatch.setenv("THUNC_RECORD", "1")
    assert thunc.call("Rate it.", {"n": 1}, returns=int, cache=True) == 3  # a cache hit, recorded
    assert len(fake.prompts) == 1

    monkeypatch.delenv("THUNC_RECORD")
    thunc.clear_cache()
    assert thunc.call("Rate it.", {"n": 1}, returns=int, cache=True) == 3
    assert thunc.cache_info() == []  # replaying never writes the cache


def test_map_and_async_functions_record_and_replay(fake, recording, monkeypatch):
    @thunc.function
    async def label(text: str) -> Literal["bug", "other"]:
        """Label it."""
        ...

    recording("1")
    fake.replies = ["bug"]
    assert asyncio.run(label("crash")) == "bug"
    fake.replies = ["1", "1", "1"]
    assert thunc.map(urgency, ["a", "b", "c"]) == [1, 1, 1]

    recording(None)
    no_backend(monkeypatch)
    assert asyncio.run(label("crash")) == "bug"
    assert thunc.map(urgency, ["c", "b", "a"]) == [1, 1, 1]


def test_dataclass_answers_replay_as_dataclasses(fake, recording):
    @dataclass
    class Item:
        name: str
        quantity: int

    recording("1")
    fake.replies = ['[{"name": "latte", "quantity": 3}]']
    assert thunc.call("Read the order.", {"text": "3 lattes"}, returns=list[Item]) == [Item("latte", 3)]
    recording(None)
    assert thunc.call("Read the order.", {"text": "3 lattes"}, returns=list[Item]) == [Item("latte", 3)]


def test_typed_backends_record_and_replay(recording, monkeypatch):
    from thunc import backends

    asked = []

    def jev(instructions, inputs, returns, *, system, timeout):
        asked.append(instructions)
        return "true"

    monkeypatch.setitem(backends.TYPED_BACKENDS, "jev", jev)
    thunc.configure(backend="jev")
    recording("1")
    assert thunc.call("Is it urgent?", {"ticket": "down"}, returns=bool) is True
    recording(None)
    no_backend(monkeypatch)
    assert thunc.call("Is it urgent?", {"ticket": "down"}, returns=bool) is True
    assert len(asked) == 1


# --- settings ------------------------------------------------------------------------------


def test_thunc_record_without_a_folder_is_an_error_not_a_silent_no_op(fake, monkeypatch):
    monkeypatch.setenv("THUNC_RECORD", "1")
    fake.replies = ["4"]
    with pytest.raises(thunc.ThuncError, match="no recordings folder"):
        urgency("x")
    assert fake.prompts == []


@pytest.mark.parametrize("value", ["2", "everything", "replay"])
def test_an_unknown_thunc_record_value_is_an_error(fake, folder, monkeypatch, value):
    monkeypatch.setenv("THUNC_RECORD", value)
    with pytest.raises(thunc.ThuncError, match="isn't a setting"):
        urgency("x")


@pytest.mark.parametrize("value", ["", "0", "false", "off"])
def test_thunc_record_off_replays(fake, folder, monkeypatch, value):
    monkeypatch.setenv("THUNC_RECORD", value)
    with pytest.raises(thunc.ThuncError, match="No recorded answer"):
        urgency("x")


def test_configure_wins_over_the_environment(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("THUNC_RECORDINGS", str(tmp_path / "from-env"))
    monkeypatch.setenv("THUNC_RECORD", "1")
    thunc.configure(recordings=str(tmp_path / "from-configure"))
    fake.replies = ["4"]
    urgency("x")
    assert (tmp_path / "from-configure").is_dir() and not (tmp_path / "from-env").exists()


def test_without_recordings_nothing_changes(fake, tmp_path, monkeypatch):
    program = tmp_path / "program"
    program.mkdir()
    monkeypatch.chdir(program)
    fake.replies = ["4"]
    assert urgency("x") == 4
    assert list(program.iterdir()) == []


def test_a_recording_that_cant_be_saved_warns_and_the_call_returns_its_value(fake, recording, tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    thunc.configure(recordings=str(blocked / "recordings"))
    recording("1")
    fake.replies = ["4"]
    with pytest.warns(RuntimeWarning, match="could not save a recording") as caught:
        assert urgency("x") == 4
    assert caught[0].filename == __file__  # pointed at the caller


def test_while_replaying_every_way_to_a_model_is_closed(folder):
    # Durable runs and anything else not recorded go through these: none may reach a model.
    with pytest.raises(thunc.ThuncError, match="replaying"):
        core._send("hello", "system", "fake", None)
    from thunc import backends

    with pytest.raises(thunc.ThuncError, match="replaying"):
        backends.sdk_client(dict, "ANTHROPIC_", None, 1.0)
    from thunc import claude_code, codex

    for cli in (claude_code.ClaudeCodeConversation, codex.CodexConversation):
        with pytest.raises(thunc.ThuncError, match="Starting .* would ask a model"):
            cli._popen(None, ["true"])


def test_recordings_are_listed_and_cleared_like_the_cache(fake, recording, folder):
    recording("1")
    fake.replies = ["4", "yes"]
    urgency("x")
    thunc.call("Spam?", returns=bool, name="spam")
    groups = {group.function: group.entries for group in _info(str(folder))}
    assert groups == {"urgency": 1, "spam": 1}


# --- trace and events ----------------------------------------------------------------------


def test_the_trace_and_events_say_a_call_was_replayed(fake, recording, monkeypatch, tmp_path):
    trace = tmp_path / "trace.jsonl"
    thunc.configure(trace=str(trace))
    recording("1")
    fake.replies = ["4"]
    urgency("x")
    recording(None)
    no_backend(monkeypatch)
    stream = tmp_path / "events.jsonl"
    monkeypatch.setenv("THUNC_EVENTS", str(stream))
    monkeypatch.delenv("THUNC_EVENTS_CAPTURE", raising=False)
    assert events.enabled()
    urgency("x")

    asked, replayed = [json.loads(line) for line in trace.read_text().splitlines()]
    assert asked["replayed"] is False and asked["cached"] is False and asked["answers"] == ["4"]
    assert replayed["replayed"] is True and replayed["cached"] is True and replayed["answers"] == []
    assert replayed["backend"] == "fake" and replayed["value"] == 4
    start, end = [e for e in map(json.loads, stream.read_text().splitlines()) if e["event"].startswith("call.")]
    assert start["backend"] == "fake"
    assert end["replayed"] is True and end["cached"] is True and end["attempts"] == 0


# --- agents --------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.py").write_text("TIMEOUT = 30\n")
    return work


def fixer(repo, ensure=None, **options):
    """The same agent and task each time: a run is matched on the task's name, and a task defined in a
    test function would have that function in its name."""
    options = {"permissions": ["write:*.py", "run:python"], "protocol": "text", **options}
    agent = thunc.Agent("fixer", workdir=repo, **options)

    @agent.task(ensure=ensure)
    def raise_timeout(seconds: int) -> int:
        """Set TIMEOUT in config.py to the given seconds, and return the old value."""
        ...

    return agent, raise_timeout


RUN = [
    act("read", path="config.py"),
    act("edit", path="config.py", old="TIMEOUT = 30", new="TIMEOUT = 60"),
    act("run", command='python -c "import config; print(config.TIMEOUT)"'),
    act("finish", value=30),
]


def test_an_agent_run_replays_its_replies_and_its_tools_really_run(fake, recording, repo, monkeypatch, tmp_path):
    agent, raise_timeout = fixer(repo)
    recording("1")
    fake.replies = list(RUN)
    first = agent.run(raise_timeout, 60)
    assert first.value == 30 and (repo / "config.py").read_text() == "TIMEOUT = 60\n"
    asked = len(fake.prompts)
    assert asked == 4

    (repo / "config.py").write_text("TIMEOUT = 30\n")  # a fresh checkout, as in CI
    recording(None)
    no_backend(monkeypatch)
    again = agent.run(raise_timeout, 60)
    assert len(fake.prompts) == asked
    assert again.value == 30 and again.steps == first.steps
    assert (repo / "config.py").read_text() == "TIMEOUT = 60\n"  # the edit was made again
    assert again.files_changed == ["config.py"] and [c.command for c in again.commands] == [
        c.command for c in first.commands
    ]
    session = [json.loads(line) for line in open(again.session, encoding="utf-8")]
    assert session[0]["protocol"] == "replay"


def test_an_agent_run_that_was_not_recorded_fails_before_it_starts(fake, folder, repo):
    agent, raise_timeout = fixer(repo)
    with pytest.raises(thunc.ThuncError, match=r"No recorded answer for fixer\.fixer\.<locals>\.raise_timeout\(\)"):
        raise_timeout(60)
    assert fake.prompts == [] and (repo / "config.py").read_text() == "TIMEOUT = 30\n"


def test_an_agent_run_is_matched_on_its_inputs_tools_and_system(fake, recording, repo):
    agent, raise_timeout = fixer(repo)
    recording("1")
    fake.replies = [act("finish", value=30)]
    assert raise_timeout(60) == 30
    recording(None)
    assert raise_timeout(60) == 30
    with pytest.raises(thunc.ThuncError, match="No recorded answer"):
        raise_timeout(90)
    _, same = fixer(repo)
    assert same(60) == 30
    for changed in ({"system": "You are careful."}, {"permissions": ["write:*.py"]}):  # no run: other tools
        _, other = fixer(repo, **changed)
        with pytest.raises(thunc.ThuncError, match="No recorded answer"):
            other(60)


def test_a_replayed_run_that_goes_differently_runs_out_of_replies(fake, recording, repo):
    _, raise_timeout = fixer(repo, retries=1)
    recording("1")
    fake.replies = [act("finish", value=30)]
    assert raise_timeout(60) == 30
    recording(None)
    _, raise_timeout = fixer(repo, retries=1, ensure=lambda old: old > 100)  # now with a check it fails
    with pytest.raises(thunc.AgentError, match="recorded run has 1 model replies.*THUNC_RECORD=1"):
        raise_timeout(60)


def test_a_failed_agent_run_is_not_recorded(fake, recording, repo, folder):
    agent, raise_timeout = fixer(repo, max_steps=1)
    recording("1")
    fake.replies = [act("read", path="config.py")]
    with pytest.raises(thunc.AgentError):
        raise_timeout(60)
    assert not folder.exists() or list(folder.iterdir()) == []


def test_agent_call_and_its_trace_record_and_replay(fake, recording, repo, monkeypatch, tmp_path):
    trace = tmp_path / "trace.jsonl"
    thunc.configure(trace=str(trace))
    agent, _ = fixer(repo)
    recording("1")
    fake.replies = [act("read", path="config.py"), act("finish", value="TIMEOUT")]
    assert agent.call("Which setting is in config.py?") == "TIMEOUT"
    recording(None)
    no_backend(monkeypatch)
    assert agent.call("Which setting is in config.py?") == "TIMEOUT"
    asked, replayed = [json.loads(line) for line in trace.read_text().splitlines()]
    assert asked["replayed"] is False and replayed["replayed"] is True and replayed["backend"] == "fake"
    assert replayed["answers"] == asked["answers"]


def test_a_native_run_on_the_claude_api_records_and_replays(recording, repo, monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    requests = []
    replies = [
        [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "config.py"}}],
        [{"type": "tool_use", "id": "t2", "name": "finish", "input": {"value": 30}}],
    ]

    class Stream:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return anthropic.types.Message.model_validate(
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": self.content,
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    class Client:
        def __init__(self, **kwargs):
            stream = self.stream
            self.messages = type("M", (), {"stream": staticmethod(stream)})()
            self.beta = type("B", (), {"messages": self.messages})()

        def with_options(self, **options):
            return self

        def stream(self, **kwargs):
            requests.append(kwargs)
            return Stream(replies.pop(0))

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    thunc.configure(backend="anthropic", api_key="test")
    agent = thunc.Agent("native", workdir=repo)

    @agent.task
    def timeout() -> int:
        """Find the timeout."""
        ...

    recording("1")
    assert timeout() == 30 and len(requests) == 2
    recording(None)
    no_backend(monkeypatch)
    monkeypatch.setitem(config._settings, "api_key", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert timeout() == 30
    assert len(requests) == 2


# --- keys across thunc releases ------------------------------------------------------------


class Priority(enum.Enum):
    LOW = 1
    HIGH = 3


@dataclass
class Finding:
    path: str
    line: int


# function, instructions, inputs, returns, system=
CALLS = {
    "plain": (
        "triage.urgency",
        "Rate how urgent this ticket is, from 1 to 5.",
        {"ticket": "Down for everyone"},
        int,
        None,
    ),
    "dataclasses": (
        "review.findings",
        "List the problems in this diff.\n",
        {"diff": "- a\n+ b", "seen": [Finding("a.py", 3)]},
        list[Finding],
        "You review code.",
    ),
    "literal": (
        None,
        "Is this urgent?",
        {"priority": Priority.HIGH, "tags": ("db", "prod")},
        Literal["yes", "no"],
        None,
    ),
    "enum": (
        "triage.priority",
        "How urgent is this?",
        {"ticket": "Slow page", "meta": {"b": 2, "a": 1.5}},
        Priority,
        None,
    ),
}
# agent, task, system=, instructions, inputs, returns, tools offered
RUNS = {
    "plain": (
        "fixer",
        "tasks.raise_timeout",
        None,
        "Set TIMEOUT in config.py to the given seconds, and return the old value.",
        {"seconds": 60},
        int,
        ["list", "read", "search", "write", "edit", "run", "remember", "finish"],
    ),
    "custom": (
        "reviewer",
        "tasks.review",
        "You are careful.",
        "Review the change.",
        {"diff": "- a"},
        list[Finding],
        ["list", "read", "search", "open_issue", "finish"],
    ),
}
BUILTIN = ["list", "read", "search", "write", "edit", "run"]
# name (format 2), legacy (format 1, as thunc 0.4 to 0.6 named them)
PINNED = {
    ("call", "plain"): (
        "4ee8c70c2f2803500a3f7c67936fcf7b890c41fac9080f0af07f7105c5e2b8bd",
        "3097b6fb93d84220663fa4fa663da6dae924811541c783160e1f1dd56f751527",
    ),
    ("call", "dataclasses"): (
        "1530640f3a89762ef0cb1f4f2a4f800956b01a28400034c61a0ab29f8d339c93",
        "3e48d0de8080aea32fdac4d3a0c492b916ba711c64940608220f36170f618d34",
    ),
    ("call", "literal"): (
        "7c66c1736f4761ef06a55c50f567529ad8a178677878f3730b20774f54dc9c82",
        "7d4ab83fa6b3ccb42c730916b59591a0550fcb14316a8e06ffb642077f9bc5e0",
    ),
    ("call", "enum"): (
        "67342ee082701c837fb935235dac61d460aeada5543f2ada620c41b5abafcafa",
        "408c02b28095b293c5eb4734d3eccf082a5b08f6b885832b3d8666914feb2298",
    ),
    ("run", "plain"): (
        "582506d055e8bcd4db8ba77fde6c8ddef086a1c8f62089c89f0cbe87e8dfd94f",
        "7d90b159b5ef32da9ec1fbd5e0e0c1f3243d04791daf9285829d40a84a0cc694",
    ),
    ("run", "custom"): (
        "78f4f0cbb249976b1ff16dbd80892d273e84d85fd80b2e6ef14b5d63c31db14c",
        "0affcf99af44653626d37e7360f89c16aedbd8d7234facbbf8a4c40c60a20302",
    ),
}


def call_key(function, instructions, inputs, returns, system):
    sent = (core.system_prompt(system), core._build_prompt(instructions, inputs, returns))
    return recordings.call_key(function, instructions, inputs, returns, system, sent=sent)


def run_key(agent, task, system, instructions, inputs, returns, tools, builtin=BUILTIN):
    request = agent_module._request(instructions, inputs, returns)
    return recordings.run_key(agent, task, system, instructions, inputs, returns, tools, builtin, request=request)


def keys():
    for name, shape in CALLS.items():
        yield ("call", name), call_key(*shape)
    for name, shape in RUNS.items():
        yield ("run", name), run_key(*shape)


def test_recording_keys_dont_change_with_thunc():
    """A key that changes loses every recording made under it, and CI fails with "No recorded answer".
    If this fails, the change renames recordings: put back what the key covers, or raise
    recordings.FORMAT and keep matching the keys pinned here. The legacy keys are the names of
    recordings made with 0.4 to 0.6, from the text thunc sent then: changing that text (the system
    prompt, how a type is described) changes them, so keep a copy of the old text for them."""
    assert {kind: (key.name, key.legacy) for kind, key in keys()} == PINNED


def test_rewording_thuncs_prompts_keeps_the_recordings(fake, recording, monkeypatch):
    recording("1")
    fake.replies = ["4"]
    assert urgency("Down for everyone") == 4
    recording(None)
    no_backend(monkeypatch)
    monkeypatch.setattr(core, "SYSTEM", "You are a function that has been reworded.")
    monkeypatch.setattr(core, "describe", lambda returns, nested=False: "a whole number, described anew")
    assert urgency("Down for everyone") == 4
    assert len(fake.prompts) == 1


def test_a_builtin_tool_added_later_leaves_run_keys_alone():
    agent, task, system, instructions, inputs, returns, tools = RUNS["plain"]
    before = run_key(agent, task, system, instructions, inputs, returns, tools)
    later = run_key(agent, task, system, instructions, inputs, returns, ["glob", *tools], ["glob", *BUILTIN])
    assert later.name == before.name
    custom = run_key(agent, task, system, instructions, inputs, returns, ["glob", *tools])  # glob of the program's own
    assert custom.name != before.name


def test_keys_dont_depend_on_the_order_of_a_set_or_a_blank_system_prompt():
    first = call_key("f", "Tag these.", {"tags": {"db", "prod", "api", "web"}}, list[str], None)
    again = call_key("f", "Tag these.", {"tags": {"web", "api", "prod", "db"}}, list[str], "  ")
    assert first.name == again.name


def keys_used(monkeypatch):
    """The keys thunc looks recordings up by, as they're made."""
    used = []
    for kind in ("call_key", "run_key"):
        make = getattr(recordings, kind)
        monkeypatch.setattr(recordings, kind, lambda *a, _make=make, **k: used.append(_make(*a, **k)) or used[-1])
    return used


def as_if_recorded_by_0_6(folder, key):
    """Rename a recording to the name thunc 0.4 to 0.6 gave it."""
    entry = json.loads((folder / f"{key.name}.json").read_text())
    (folder / f"{key.legacy}.json").write_text(json.dumps({**entry, "key": key.legacy}))
    (folder / f"{key.name}.json").unlink()


def test_a_call_recorded_before_0_7_replays_and_moves_when_recorded_again(fake, recording, folder, monkeypatch):
    used = keys_used(monkeypatch)
    recording("1")
    fake.replies = ["4"]
    assert urgency("Down for everyone") == 4
    key = used[-1]
    as_if_recorded_by_0_6(folder, key)

    recording(None)
    assert urgency("Down for everyone") == 4
    recording("missing")
    assert urgency("Down for everyone") == 4
    assert len(fake.prompts) == 1  # found under its old name: not asked again
    assert [p.name for p in folder.iterdir()] == [f"{key.legacy}.json"]

    recording("1")
    fake.replies = ["5"]
    assert urgency("Down for everyone") == 5
    assert [p.name for p in folder.iterdir()] == [f"{key.name}.json"]  # the old file is gone
    recording(None)
    assert urgency("Down for everyone") == 5


def test_an_agent_run_recorded_before_0_7_replays(fake, recording, repo, folder, monkeypatch):
    used = keys_used(monkeypatch)
    agent, raise_timeout = fixer(repo)
    recording("1")
    fake.replies = [act("finish", value=30)]
    assert raise_timeout(60) == 30
    key = used[-1]
    as_if_recorded_by_0_6(folder, key)

    recording(None)
    no_backend(monkeypatch)
    assert raise_timeout(60) == 30
    assert len(fake.prompts) == 1
