"""thunc's pytest fixtures, loaded by pytest through the pytest11 entry point (see pyproject.toml).

    def test_triage(thunc_model):
        thunc_model.reply({"urgency": 4})
        assert urgency("Site down") == 4

thunc_model:    a thunc.testing.FakeModel answering every call and agent run of the test.
thunc_isolated: thunc's settings and THUNC_* environment variables put back after the test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from . import testing


@pytest.fixture
def thunc_model() -> Iterator[testing.FakeModel]:
    with testing.isolated(), testing.fake_model() as model:
        yield model


@pytest.fixture
def thunc_isolated() -> Iterator[None]:
    with testing.isolated():
        yield
