"""@thunc.function(write=True) against a real model: the function writes itself into a scratch project."""

import doctest
import importlib.util
import sys

import pytest

MODULE = '''\
import thunc


@thunc.function(write=True)
def minutes(duration: str) -> int:
    """Convert a duration like '1h 30m', '90 min' or '2 hours' to whole minutes."""
    ...
'''


def test_a_function_writes_itself_and_then_runs_without_the_model(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("THUNC_WRITE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    path = tmp_path / "durations.py"
    path.write_text(MODULE)
    spec = importlib.util.spec_from_file_location("live_durations", path)
    durations = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "live_durations", durations)
    spec.loader.exec_module(durations)

    assert durations.minutes("1h 30m") == 90
    err = capsys.readouterr().err
    if "stays a model call" in err or "wasn't written" in err:
        pytest.fail(f"not written:\n{err}")
    written = path.read_text()
    assert "@thunc.function" not in written and "import thunc" in written
    assert "# Written by thunc from the docstring" in written

    writer = durations.minutes.__thunc_writer__
    assert writer.impl is not None
    assert durations.minutes("2 hours") == 120  # the written code
    assert durations.minutes("45 min") == 45

    spec = importlib.util.spec_from_file_location("live_durations_written", path)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "live_durations_written", fresh)
    spec.loader.exec_module(fresh)
    assert doctest.testmod(fresh).failed == 0
