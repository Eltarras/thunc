"""Agents against a real model: native tool calls on the anthropic and openai backends, the text
protocol on the CLIs. Each test builds a small repo in a temp folder.

THUNC_BACKEND=anthropic pytest live_tests/test_agent.py   # needs ANTHROPIC_API_KEY
THUNC_BACKEND=openai pytest live_tests/test_agent.py      # needs OPENAI_API_KEY
"""

import subprocess
import sys

import pytest

import thunc


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "mathutil.py").write_text(
        "def last_n(items, n):\n    return items[len(items) - n - 1:]\n\n\n"
        "def mean(values):\n    return sum(values) / len(values) if values else None\n"
    )
    (tmp_path / "tests" / "test_mathutil.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\nfrom mathutil import last_n, mean\n\n"
        "assert last_n([1, 2, 3, 4], 2) == [3, 4]\nassert mean([2, 4]) == 3\nassert mean([]) == 0\nprint('ok')\n"
    )
    (tmp_path / "AGENTS.md").write_text("This project's code name is Bluebird.\n")
    thunc.configure(agents_dir=str(tmp_path / ".agents"))
    return tmp_path


def test_reads_edits_runs_and_finishes(repo):
    python = f"run:{sys.executable} tests/test_mathutil.py"
    agent = thunc.Agent("fixer", workdir=repo, permissions=["write:src/**", python, "!run:git"])

    @agent.task
    def make_tests_pass() -> bool:
        """Make the tests pass by fixing src/ only. Run them with the command your permissions allow.
        Return whether they pass after your last change."""
        ...

    run = agent.run(make_tests_pass)
    assert run.value is True
    assert run.files_changed == ["src/mathutil.py"]
    assert run.commands and run.commands[-1].exit_code == 0
    check = subprocess.run([sys.executable, "tests/test_mathutil.py"], cwd=repo, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_denied_writes_are_reported_and_the_run_finishes(repo):
    agent = thunc.Agent("scribe", workdir=repo, permissions=["write:NOTES.md"])

    @agent.task
    def note_and_rename() -> str:
        """Create NOTES.md with one line saying the tests cover mathutil, and also rename src/mathutil.py
        to src/maths.py. Report what you did and what you couldn't."""
        ...

    run = agent.run(note_and_rename)
    assert run.value.strip() and run.files_changed == ["NOTES.md"]
    assert (repo / "NOTES.md").exists() and (repo / "src" / "mathutil.py").exists()
    assert not (repo / "src" / "maths.py").exists()


def test_follow_and_a_dataclass_result(repo):
    from dataclasses import dataclass

    @dataclass
    class Answer:
        code_name: str
        source: str  # the file it came from, or "instructions"

    agent = thunc.Agent("guide", workdir=repo, follow=True)

    @agent.task
    def code_name() -> Answer:
        """What is this project's code name, and where did you learn it?"""
        ...

    answer = code_name()
    assert answer.code_name == "Bluebird"
