"""What a type checker infers from thunc's public API: tests/typing/check_types.py under mypy and
basedpyright (the dev extra installs both; a missing one is skipped)."""

import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
CHECKED = os.path.join(HERE, "typing", "check_types.py")
ROOT = os.path.dirname(HERE)


def test_mypy_infers_the_documented_types():
    api = pytest.importorskip("mypy.api")
    out, err, status = api.run(["--strict", "--no-incremental", "--python-version", "3.10", CHECKED])
    assert status == 0, out + err


def test_pyright_infers_the_documented_types():
    pyright = shutil.which("basedpyright", path=os.path.dirname(sys.executable)) or shutil.which("basedpyright")
    if pyright is None:
        pytest.skip("basedpyright is not installed")
    result = subprocess.run(
        # Its own settings (tests/typing/pyrightconfig.json): standard rules, and only errors fail, so a
        # newer basedpyright's stricter defaults can't fail it. A wrong assert_type is an error.
        [pyright, "--project", os.path.join(HERE, "typing"), "--pythonpath", sys.executable, CHECKED],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
