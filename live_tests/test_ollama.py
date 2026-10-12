"""thunc on a local Ollama server, through the openai backend and Ollama's OpenAI-compatible Responses
API (/v1/responses, Ollama 0.13.3 or later). Skipped unless OLLAMA_MODEL names a pulled model:

    ollama pull qwen3:8b
    OLLAMA_MODEL=qwen3:8b THUNC_BACKEND=openai OPENAI_API_KEY=ollama \\
        OPENAI_BASE_URL=http://localhost:11434/v1 pytest live_tests/test_ollama.py

Small models get the retry more often; these tests check that thunc gets a valid answer, not that a
small model always knows it.
"""

import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pytest

import thunc

MODEL = os.environ.get("OLLAMA_MODEL")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")


def _version() -> tuple[int, ...] | None:
    """Ollama's version, from its own API next to /v1, or None if no Ollama answers there."""
    root = BASE_URL.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/version", timeout=5) as response:
            text = json.load(response)["version"]
    except (OSError, ValueError, KeyError):
        return None
    return tuple(int(part) for part in text.split("-")[0].split(".")[:3] if part.isdigit())


@pytest.fixture(autouse=True)
def ollama(monkeypatch):
    if not MODEL:
        pytest.skip("OLLAMA_MODEL is not set")
    version = _version()
    if version is None:
        pytest.skip(f"no Ollama server answers at {BASE_URL}")
    if version < (0, 13, 3):
        pytest.skip(f"Ollama {'.'.join(map(str, version))} has no /v1/responses (0.13.3 or later)")
    monkeypatch.setenv("OPENAI_BASE_URL", BASE_URL)
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "ollama")
    thunc.configure(backend="openai", model=MODEL, timeout=600)


class Mood(Enum):
    HAPPY = "happy"
    ANGRY = "angry"


@dataclass
class Order:
    item: str
    quantity: int


def test_text():
    reply = thunc.call("Say hello in five words or fewer.")
    assert isinstance(reply, str) and reply.strip()


def test_typed_answers():
    assert thunc.call("Is 17 a prime number?", returns=bool) is True
    assert thunc.call("How many legs does a spider have?", returns=int) == 8
    label = thunc.call("Classify this ticket.", {"ticket": "I was charged twice"}, returns=Literal["bug", "billing"])
    assert label == "billing"
    mood = thunc.call("What's the mood of this message?", {"text": "THIS IS UNACCEPTABLE!!"}, returns=Mood)
    assert mood is Mood.ANGRY
    assert thunc.call("Read the order.", {"text": "three lattes please"}, returns=Order) == Order("latte", 3)


@pytest.mark.parametrize("protocol", ["native", "text"])
def test_an_agent_reads_edits_and_runs(tmp_path, protocol):
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    thunc.configure(agents_dir=str(tmp_path / ".agents"))
    check = f"run:{sys.executable} -c"
    agent = thunc.Agent("ollama", workdir=tmp_path, permissions=["write:config.py", check], protocol=protocol)

    @agent.task
    def double_timeout() -> int:
        """Double TIMEOUT in config.py, then return the new value."""
        ...

    run = agent.run(double_timeout)
    assert run.value == 60
    command = [sys.executable, "-c", "import config; print(config.TIMEOUT)"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.stdout.strip() == "60"
