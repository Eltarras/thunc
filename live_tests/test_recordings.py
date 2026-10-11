"""Record and replay against a real model: record a call and an agent run, then replay both with no
backend configured, so nothing could reach a model, and get the same answers."""

import pytest

import thunc
from thunc import config


@pytest.fixture
def recordings(tmp_path, monkeypatch):
    thunc.configure(recordings=str(tmp_path / "recordings"), agents_dir=str(tmp_path / ".agents"))
    monkeypatch.setenv("THUNC_RECORD", "1")

    def replay():
        monkeypatch.delenv("THUNC_RECORD")
        monkeypatch.setitem(config._settings, "backend", None)
        monkeypatch.delenv("THUNC_BACKEND", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    return replay


@thunc.function
def capital(country: str) -> str:
    """The capital city of this country, in English."""
    ...


def test_a_recorded_call_replays_without_a_model(recordings):
    answer = capital("France")
    assert "paris" in answer.lower()
    recordings()
    assert capital("France") == answer


def test_a_recorded_agent_run_replays_with_its_tools(recordings, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.py").write_text("TIMEOUT = 30\n")
    agent = thunc.Agent("fixer", workdir=work, permissions=["write:config.py"])

    @agent.task
    def double_timeout() -> int:
        """Double TIMEOUT in config.py, and return the new value."""
        ...

    first = agent.run(double_timeout)
    assert first.value == 60 and "TIMEOUT = 60" in (work / "config.py").read_text()

    (work / "config.py").write_text("TIMEOUT = 30\n")
    recordings()
    again = agent.run(double_timeout)
    assert again.value == 60 and again.steps == first.steps
    assert "TIMEOUT = 60" in (work / "config.py").read_text()  # the edit was made again, from the recording
