"""Profiling: the records calls and agent runs leave, the report, and `thunc run --profile`."""

import json
import textwrap

import pytest

import thunc
from thunc import profiling
from thunc.__main__ import main


def act(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def test_nothing_is_recorded_without_a_profiler(fake):
    fake.replies = ["4"]
    assert thunc.call("Rate it.", returns=int) == 4
    assert profiling.active() is None


def test_calls_record_attempts_cache_hits_and_failures(fake, tmp_path):
    thunc.configure(cache_dir=str(tmp_path / "cache"))

    @thunc.function(cache=True)
    def urgency(ticket: str) -> int:
        """Rate how urgent this ticket is, from 1 to 5."""
        ...

    fake.replies = ["soon", "4", "x", "x", "x"]
    with profiling.profiling() as profiler:
        urgency("down")  # a retry, then a valid answer
        urgency("down")  # from the cache
        with pytest.raises(thunc.ThuncError):
            thunc.call("Rate it.", returns=int)
    assert profiling.active() is None

    first, cached, failed = profiler.calls
    assert first.function.endswith("urgency") and first.attempts == 2 and not first.cached and first.ok
    assert first.backend == "fake" and 0 <= first.model_seconds <= first.seconds
    assert cached.attempts == 0 and cached.cached and cached.model_seconds == 0
    assert failed.function == profiling.UNNAMED and failed.attempts == 3 and not failed.ok

    report = profiler.report()
    header, *rows = report.split("CALLS\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert header.split() == [
        "FUNCTION", "CALLS", "CACHED", "RETRIES", "FAILED", "TOTAL", "MEAN", "P95", "MAX", "MODEL", "LOCAL"
    ]  # fmt: skip
    assert {row.split()[0]: row.split()[1:5] for row in rows} == {
        first.function: ["2", "1", "1", "0"],
        "(thunc.call)": ["1", "0", "2", "1"],
    }
    assert "In thunc:" in report and "Model time:" in report and "fake/default model" in report


def test_agent_runs_record_steps_and_tool_times(fake, tmp_path):
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    agent = thunc.Agent("repo-guide", workdir=tmp_path)

    @agent.task
    def timeout() -> int:
        """Find the request timeout."""
        ...

    fake.replies = [act("list"), act("read", path="config.py"), act("read", path="config.py"), act("finish", value=30)]
    with profiling.profiling() as profiler:
        assert timeout() == 30

    (run,) = profiler.agents
    assert run.function.startswith("repo-guide.") and run.function.endswith(".timeout") and run.steps == 4 and run.ok
    assert {tool: len(times) for tool, times in run.tools.items()} == {"list": 1, "read": 2}
    assert run.model_seconds + run.tool_seconds <= run.seconds
    report = profiler.report()
    assert "AGENT RUNS" in report and "AGENT TOOLS" in report
    assert next(line for line in report.splitlines() if line.startswith("read")).split()[1] == "2"


def test_overlapping_calls_count_once_in_the_busy_time():
    assert profiling._union([(0, 2), (1, 3), (1.5, 2), (5, 6)]) == 4
    assert profiling._union([]) == 0


def test_run_profile_prints_a_report_after_the_program(fake, tmp_path, capsys):
    script = tmp_path / "triage.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import thunc

            print("args", sys.argv[1:], thunc.call("Rate it.", returns=int))
            sys.exit(3)
            """
        )
    )
    fake.replies = ["4"]
    code = main(["run", "--profile", str(script), "--fast", "x"])
    out, err = capsys.readouterr()
    assert code == 3
    assert out == "args ['--fast', 'x'] 4\n"
    assert f"thunc profile: {script}:" in err and "(thunc.call)" in err


def test_run_without_profile_prints_no_report(fake, tmp_path, capsys):
    script = tmp_path / "hello.py"
    script.write_text("import thunc\nprint(thunc.call('Say hi.'))\n")
    fake.replies = ["hi"]
    assert main(["run", str(script)]) == 0
    out, err = capsys.readouterr()
    assert out == "hi\n" and err == ""


def test_run_reports_even_when_the_program_crashes(tmp_path, capsys):
    script = tmp_path / "broken.py"
    script.write_text("raise RuntimeError('boom')\n")
    assert main(["run", "--profile", str(script)]) == 1
    err = capsys.readouterr().err
    assert "RuntimeError: boom" in err and "No thunc calls were made." in err
