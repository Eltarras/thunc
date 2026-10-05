"""The text protocol: reading actions out of replies that aren't only the action, and the Claude Code
step that stops once an action is complete. The reply shapes are the ones the tool-use benchmark
found (live_tests/bench_tooluse_report.md, finding 1c)."""

import json
import subprocess
import sys
import time

import pytest

import thunc
from thunc import backends, native

NAMES = ["list", "read", "search", "write", "edit", "run", "remember", "finish", "open_issue"]
READ = '{"tool": "read", "args": {"path": "config.py"}}'


def invoke(tool, **params):
    """A tool call written as markup, as models trained for native calls sometimes reply."""
    body = "".join(f'<parameter name="{name}">{value}</parameter>\n' for name, value in params.items())
    return f'<invoke name="{tool}">\n{body}</invoke>'


def tools_of(answer):
    return [(call.tool, call.args, call.problem) for call in native.actions(answer, NAMES)]


# --- reading actions ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(READ, id="only-the-action"),
        pytest.param(f"I'll read the config first.\n\n{READ}", id="prose-then-json"),
        pytest.param(
            f'<invoke name="read">\n<parameter name="path">config.py</parameter>\n</invoke>\n{READ}',
            id="markup-then-json",
        ),
        pytest.param(f"{READ}\n```", id="json-then-stray-fence"),
        pytest.param(
            f'{READ}\n\n<tool_result>1  TIMEOUT = 31\n</tool_result>\n{{"tool": "finish"',
            id="json-then-made-up-results",
        ),
        pytest.param(f"Reading it:\n```json\n{READ}\n```\nThen I'll decide.", id="fence-with-prose-after"),
    ],
)
def test_the_first_complete_action_is_read(answer):
    assert tools_of(answer) == [("read", {"path": "config.py"}, None)]


def test_literal_newlines_in_a_string_are_accepted():
    answer = '{"tool": "write", "args": {"path": "a.py", "content": "x = 1\ny = 2\n"}}'  # real newlines, not \\n
    assert tools_of(answer) == [("write", {"path": "a.py", "content": "x = 1\ny = 2\n"}, None)]


def test_an_array_in_prose_is_read_as_a_batch():
    answer = 'Both at once:\n[{"tool": "read", "args": {"path": "a.py"}}, {"tool": "list", "args": {}}] and then'
    assert tools_of(answer) == [("read", {"path": "a.py"}, None), ("list", {}, None)]


def test_tool_call_markup_alone_is_read_with_each_arguments_type():
    answer = "\n".join(
        [
            invoke("read", path="notes.json", offset="5"),
            invoke("write", path="n.json", content='{"a": 1}'),
            invoke("edit", path="a.py", edits='[{"old": "a", "new": "b"}]'),
            invoke("open_issue", title="Flaky test", priority="2"),
        ]
    )
    assert tools_of(answer) == [
        ("read", {"path": "notes.json", "offset": 5}, None),
        ("write", {"path": "n.json", "content": '{"a": 1}'}, None),  # a text argument stays text, JSON or not
        ("edit", {"path": "a.py", "edits": [{"old": "a", "new": "b"}]}, None),
        ("open_issue", {"title": "Flaky test", "priority": 2}, None),  # a custom tool's: JSON where it parses
    ]


def test_finish_in_markup_keeps_its_value_as_text_for_the_task_type_to_read():
    answer = '<invoke name="finish">\n<parameter name="value">30</parameter>\n</invoke>'
    assert tools_of(answer) == [("finish", {"value": "30"}, None)]


@pytest.mark.parametrize(
    "answer, problem",
    [
        ("Let me look around first.", "not valid JSON"),
        ('Here: {"path": "config.py"} is the file.', "not valid JSON"),  # JSON, but no action in it
        ('Sure: {"tool": "delete", "args": {"path": "a"}}', "unknown tool 'delete'"),
    ],
)
def test_a_reply_without_a_usable_action_is_still_an_error(answer, problem):
    with pytest.raises(ValueError, match=problem):
        native.actions(answer, NAMES)


def test_while_a_reply_arrives_a_started_array_is_waited_for():
    assert native.first_action('{"tool": "read", "args": {"path": "a"}}', final=False) is not None
    assert native.first_action('[{"tool": "read", "args": {"path": "a"}}, {"tool": "li', final=False) is None
    # A finished reply whose batch was cut off has no action: a batch isn't run in part.
    assert native.first_action('[{"tool": "read", "args": {"path": "a"}}, {"tool": "li', final=True) is None
    assert native.first_action('{"tool": "write", "args": {"content": "}', final=False) is None
    # Every action of the batch is complete but the batch isn't: still waited for, not cut to its last one.
    batch = '[{"tool": "read", "args": {"path": "a"}}, {"tool": "list", "args": {}}'
    assert native.first_action(batch, final=False) is None


# --- the Claude Code step stops once the action is complete ---------------------------------


FAKE_CLAUDE_STREAM = """
import json, os, sys, time
args = sys.argv[1:]
sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\\n")
plan = json.loads(os.environ["FAKE_CLAUDE_STREAM"])
for text in plan["deltas"]:
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    print(json.dumps({"type": "stream_event", "event": delta}), flush=True)
    time.sleep(plan.get("pause", 0))
if "result" in plan:
    print(json.dumps({"type": "result", **plan["result"]}), flush=True)
time.sleep(plan.get("linger", 0))
"""


@pytest.fixture
def streaming_claude(tmp_path, monkeypatch):
    """Runs FAKE_CLAUDE_STREAM in place of `claude -p`. Call it with the plan; .args() is how it was run."""
    script = tmp_path / "fake_claude_stream.py"
    script.write_text(FAKE_CLAUDE_STREAM)
    log = tmp_path / "claude.log"
    real_popen = subprocess.Popen

    def popen(args, **kwargs):
        assert args[0] == "claude"
        return real_popen([sys.executable, str(script), *args[1:]], **kwargs)

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "Popen", popen)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))

    def plan(**options):
        monkeypatch.setenv("FAKE_CLAUDE_STREAM", json.dumps(options))

    plan.args = lambda: [json.loads(line) for line in log.read_text().splitlines()]
    return plan


def step(until=native._complete, timeout=20):
    return backends.claude_code("transcript", system="rules", model=None, api_key=None, timeout=timeout, until=until)


def test_the_step_stops_once_the_action_is_complete(streaming_claude):
    # The model writes its action, then carries on making up the result: it's stopped, not waited for.
    streaming_claude(
        deltas=['I will read it. {"tool": "read", ', '"args": {"path": "a"}}', "\n<tool_result>made up"], linger=30
    )
    started = time.monotonic()
    answer = step()
    assert time.monotonic() - started < 10
    assert answer == 'I will read it. {"tool": "read", "args": {"path": "a"}}'
    (args,) = streaming_claude.args()
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in args and "--include-partial-messages" in args and "--tools" in args


def test_a_started_array_is_waited_for(streaming_claude):
    deltas = ['[{"tool": "read", "args": {"path": "a"}}', ', {"tool": "list", "args": {}}]', " more"]
    streaming_claude(deltas=deltas, linger=30)
    assert step() == '[{"tool": "read", "args": {"path": "a"}}, {"tool": "list", "args": {}}]'


def test_a_reply_with_no_action_ends_with_the_cli(streaming_claude):
    streaming_claude(deltas=["Let me ", "think."], result={"is_error": False, "result": "Let me think."})
    assert step() == "Let me think."


def test_a_cli_error_can_be_retried(streaming_claude):
    streaming_claude(deltas=[], result={"is_error": True, "result": "The model's tool call could not be parsed"})
    with pytest.raises(thunc.errors.TransientError, match="could not be parsed"):
        step()


def test_a_step_that_never_completes_times_out(streaming_claude):
    streaming_claude(deltas=['{"tool": "write", "args": {"content": "'], linger=30)
    with pytest.raises(thunc.errors.TransientError, match="timed out after 1s"):
        step(timeout=1)


def test_an_agent_on_the_text_protocol_finishes_through_the_streamed_step(streaming_claude, tmp_path):
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    streaming_claude(deltas=["Done: ", '{"tool": "finish", "args": {"value": 30}}', "\n\nAnything else?"], linger=30)
    thunc.configure(backend="claude-code")
    agent = thunc.Agent("t", workdir=tmp_path, protocol="text")

    @agent.task
    def timeout() -> int:
        """Find the timeout."""
        ...

    started = time.monotonic()
    assert timeout() == 30 and time.monotonic() - started < 10


def test_plain_calls_on_claude_code_are_not_streamed(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"result": "pong", "is_error": False}), stderr="")

    monkeypatch.setattr(backends.shutil, "which", lambda exe: "/usr/bin/" + exe)
    monkeypatch.setattr(backends.subprocess, "run", run)
    thunc.configure(backend="claude-code")
    assert thunc.call("ping") == "pong"
    assert calls[0][calls[0].index("--output-format") + 1] == "json"


# --- keeping the model to JSON --------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, note",
    [
        (f"Let me read it.\n{READ}", native.EXTRA_TEXT),
        (invoke("read", path="config.py"), native.MARKUP),
        (READ, None),
    ],
)
def test_a_reply_read_leniently_gets_a_note_with_its_results(fake, tmp_path, reply, note):
    # Without the note, a model that slips into markup is never corrected; in the benchmark it then
    # repeated empty markup until the step timed out.
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    fake.replies = [reply, '{"tool": "finish", "args": {"value": 30}}']
    agent = thunc.Agent("n", workdir=tmp_path)

    @agent.task
    def timeout() -> int:
        """Find the timeout."""
        ...

    assert timeout() == 30
    assert "1  TIMEOUT = 30" in fake.prompts[1]
    if note:
        assert f"(Note: {note})" in fake.prompts[1]
    else:
        assert "(Note:" not in fake.prompts[1]


def test_repeated_markup_ends_the_step():
    one = invoke("read")
    assert not native._complete(one)  # it may be followed by the JSON action
    assert native._complete(f"{one}\n\n{one}")  # repeating itself: stop


def test_a_step_repeating_empty_markup_is_stopped(streaming_claude):
    streaming_claude(deltas=['<invoke name="read">\n</invoke>\n\n'] * 200, pause=0.01, linger=30)
    started = time.monotonic()
    answer = step()
    assert time.monotonic() - started < 10 and answer.count("</invoke>") == 2
    assert native.read_actions(answer, NAMES) == (
        [native.Call(None, "read", {}), native.Call(None, "read", {})],
        native.MARKUP,
    )


def test_markup_with_all_its_arguments_in_one_json_parameter():
    answer = invoke("edit", args='{"path": "a.py", "old": "x = 1", "new": "x = 2"}')
    assert tools_of(answer) == [("edit", {"path": "a.py", "old": "x = 1", "new": "x = 2"}, None)]
