"""The `thunc` command: `thunc cache list` and `thunc cache clear`."""

import contextlib
import os
import subprocess
import sys
import time

import pytest

import thunc
from thunc import config
from thunc.__main__ import _watch_binary, main


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


# --- thunc watch ----------------------------------------------------------------------------


@pytest.fixture
def no_watch_binary(monkeypatch, tmp_path):
    monkeypatch.delenv("THUNC_WATCH_BIN", raising=False)
    monkeypatch.setattr("sysconfig.get_path", lambda name: str(tmp_path / "empty"))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "empty" / "python"))
    monkeypatch.setattr("shutil.which", lambda name: None)


def test_watch_without_the_package_says_how_to_install_it(no_watch_binary, capsys):
    assert main(["watch", "app.py"]) == 2
    assert 'pip install "thunc[watch]"' in capsys.readouterr().err


def test_watch_hands_every_argument_to_the_dashboard(monkeypatch):
    started = []

    def record(binary, argv):
        started.append(argv)
        raise SystemExit(0)  # execv doesn't return

    class Popen:
        def __init__(self, argv):
            started.append(argv)

        def wait(self):
            return 0

    monkeypatch.setenv("THUNC_WATCH_BIN", "/opt/thunc-watch")
    monkeypatch.setattr(os, "execv", record)
    monkeypatch.setattr(subprocess, "Popen", Popen)
    with pytest.raises(SystemExit) if sys.platform != "win32" else contextlib.nullcontext():
        main(["watch", "--agents", "--plain", "app.py", "--flag"])
    # Scripts run on this Python, unless thunc-watch is given its own --python, which comes later.
    assert started == [["/opt/thunc-watch", "--python", sys.executable, "--agents", "--plain", "app.py", "--flag"]]


def test_watch_finds_the_binary_pip_installed_next_to_python(monkeypatch, tmp_path):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    binary = scripts / ("thunc-watch.exe" if sys.platform == "win32" else "thunc-watch")
    binary.write_text("")
    monkeypatch.delenv("THUNC_WATCH_BIN", raising=False)
    monkeypatch.setattr("sysconfig.get_path", lambda name: str(scripts))
    assert _watch_binary() == str(binary)


@pytest.mark.skipif(sys.platform == "win32", reason="runs a shell script as the dashboard")
def test_thunc_watch_runs_the_dashboard_in_place_of_itself(tmp_path):
    fake = tmp_path / "thunc-watch"
    fake.write_text('#!/bin/sh\necho "dashboard $*"\nexit 3\n')
    fake.chmod(0o755)
    env = {**os.environ, "THUNC_WATCH_BIN": str(fake)}
    done = subprocess.run(
        [sys.executable, "-m", "thunc", "watch", "--replay", "e.jsonl"], capture_output=True, text=True, env=env
    )
    assert done.returncode == 3
    word, flag, python, *rest = done.stdout.split()
    assert (word, flag, rest) == ("dashboard", "--python", ["--replay", "e.jsonl"])
    assert os.path.realpath(python) == os.path.realpath(sys.executable)


def test_help_lists_watch(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "watch" in capsys.readouterr().out
