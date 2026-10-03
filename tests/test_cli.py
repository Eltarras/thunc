"""The `thunc` command: `thunc cache list` and `thunc cache clear`."""

import os
import subprocess
import sys
import time

import pytest

import thunc
from thunc import config
from thunc.__main__ import main


@pytest.fixture
def cache(fake, tmp_path):
    """A cache folder with 2 urgency answers, 1 category answer and 1 unnamed thunc.call answer."""
    folder = tmp_path / "cache"
    thunc.configure(cache_dir=str(folder))

    @thunc.function(cache=True)
    def urgency(ticket: str) -> int:
        """Rate how urgent this ticket is, from 1 to 5."""
        ...

    @thunc.function(cache=True)
    def category(ticket: str) -> str:
        """Classify this support ticket."""
        ...

    fake.replies = ["4", "2", "bug", "x"]
    urgency("down"), urgency("typo"), category("down"), thunc.call("Say x.", cache=True)
    return folder


def run(capsys, *args):
    code = main(list(args))
    return code, capsys.readouterr().out


def test_list(cache, capsys):
    code, out = run(capsys, "cache", "list", "--cache-dir", str(cache))
    lines = out.splitlines()
    assert code == 0
    assert lines[0].split() == ["FUNCTION", "ENTRIES", "SIZE", "NEWEST"]
    assert lines[1].split()[:2] == [f"{__name__}.cache.<locals>.category", "1"]
    assert lines[2].split()[:2] == [f"{__name__}.cache.<locals>.urgency", "2"]
    assert lines[3].split()[:2] == ["(thunc.call)", "1"]
    assert lines[-1].startswith("4 entries (") and lines[-1].endswith(f"in {cache}")


def test_list_empty(tmp_path, capsys):
    code, out = run(capsys, "cache", "list", "--cache-dir", str(tmp_path / "none"))
    assert code == 0 and out.startswith("No saved answers in")


def test_clear_everything(cache, capsys):
    code, out = run(capsys, "cache", "clear", "--cache-dir", str(cache))
    assert code == 0 and out.startswith("Removed 4 entries (") and out.rstrip().endswith(f"from {cache}")
    assert not list(cache.glob("*.json"))


def test_clear_functions_by_name(cache, capsys):
    name = "cache.<locals>.urgency"  # the qualified name of a function defined inside the fixture
    code, out = run(capsys, "cache", "clear", "--function", name, "--cache-dir", str(cache))
    assert out.startswith("Removed 2 entries")
    code, out = run(
        capsys,
        "cache",
        "clear",
        "--function",
        "nothing",
        "--function",
        "cache.<locals>.category",
        "--cache-dir",
        str(cache),
    )
    assert code == 0 and out.startswith("Removed 1 entry (")
    assert len(list(cache.glob("*.json"))) == 1  # the thunc.call answer


def test_clear_older_than_and_dry_run(cache, capsys):
    (old, *_) = sorted(cache.glob("*.json"))
    then = time.time() - 3 * 86400
    os.utime(old, (then, then))
    code, out = run(capsys, "cache", "clear", "--older-than", "2d", "--dry-run", "--cache-dir", str(cache))
    assert code == 0 and out.startswith("Would remove 1 entry (")
    assert len(list(cache.glob("*.json"))) == 4
    code, out = run(capsys, "cache", "clear", "--older-than", "2d", "--cache-dir", str(cache))
    assert out.startswith("Removed 1 entry (") and len(list(cache.glob("*.json"))) == 3
    code, out = run(capsys, "cache", "clear", "--older-than", "3600", "--cache-dir", str(cache))
    assert out.startswith("Removed 0 entries")


def test_cache_dir_from_environment(cache, capsys, monkeypatch):
    monkeypatch.setitem(config._settings, "cache_dir", None)  # a new process: no configure() call
    monkeypatch.setenv("THUNC_CACHE_DIR", str(cache))
    code, out = run(capsys, "cache", "clear")
    assert out.startswith("Removed 4 entries")


@pytest.mark.parametrize("args", [[], ["cache"], ["cache", "clear", "--older-than", "soon"], ["cache", "wipe"]])
def test_bad_arguments_exit_2(args, capsys):
    with pytest.raises(SystemExit) as exit:
        main(args)
    assert exit.value.code == 2


def test_python_dash_m(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "thunc", "cache", "list", "--cache-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and result.stdout.startswith("No saved answers")
