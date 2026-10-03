"""Live tests call a real model, so they cost quota and answers can vary. Plain `pytest` skips them.

pytest live_tests                          # through your local Claude Code login
THUNC_BACKEND=codex pytest live_tests
THUNC_BACKEND=anthropic pytest live_tests  # needs ANTHROPIC_API_KEY and `pip install anthropic`
THUNC_BACKEND=openai pytest live_tests     # needs OPENAI_API_KEY and `pip install openai`
"""

import os
import shutil

import pytest

import thunc

BACKEND = os.environ.get("THUNC_BACKEND", "claude-code")
CLI = {"claude-code": "claude", "codex": "codex"}


@pytest.fixture(autouse=True, scope="session")
def live_backend():
    if BACKEND in CLI and shutil.which(CLI[BACKEND]) is None:
        pytest.skip(f"the `{CLI[BACKEND]}` CLI is not installed")
    if BACKEND == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    if BACKEND == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    thunc.configure(backend=BACKEND)
