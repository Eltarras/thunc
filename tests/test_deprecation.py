"""The deprecation mechanism (thunc/deprecation.py), and the rule that no deprecation outlives its release."""

import subprocess
import sys
import warnings

import pytest

import thunc
from thunc import deprecation
from thunc.deprecation import DEPRECATIONS, ThuncDeprecationWarning, _version, deprecated, renamed_parameter


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Declarations made in a test don't stay in the real list."""
    monkeypatch.setattr(deprecation, "DEPRECATIONS", list(DEPRECATIONS))


def test_a_deprecated_function_warns_at_the_callers_line_and_still_works():
    @deprecated(since="0.7", removed_in="0.9", instead="thunc.new")
    def old(a, b=1):
        """Old."""
        return a + b

    with pytest.warns(ThuncDeprecationWarning) as caught:
        assert old(1, b=2) == 3
    (warning,) = caught
    assert str(warning.message) == (
        "thunc.test_a_deprecated_function_warns_at_the_callers_line_and_still_works.<locals>.old is deprecated "
        "since thunc 0.7 and will be removed in 0.9: use thunc.new instead."
    )
    assert warning.filename == __file__
    assert old.__doc__ == "Old." and old.__name__ == "old"


def test_a_renamed_parameter_is_passed_on_under_its_new_name():
    @renamed_parameter("retry", "retries", since="0.7", removed_in="0.9", what="thunc.call")
    def call(text, *, retries=2):
        return retries

    assert call("x") == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert call("x", retries=5) == 5  # the new name doesn't warn
    with pytest.warns(ThuncDeprecationWarning, match=r"thunc.call\(retry=\) is deprecated .* use retries= instead"):
        assert call("x", retry=4) == 4
    with pytest.raises(TypeError, match="both retry= and retries="):
        call("x", retry=1, retries=2)


def test_removed_in_must_come_after_since():
    with pytest.raises(ValueError, match="must come after"):
        deprecated(since="0.9", removed_in="0.8")(lambda: None)


def test_the_warning_is_a_deprecation_warning_shown_by_default():
    assert issubclass(thunc.ThuncDeprecationWarning, DeprecationWarning)
    # Outside __main__ Python hides DeprecationWarning; thunc's is shown, once per place.
    code = (
        "import thunc, warnings\n"
        "from thunc.deprecation import deprecated\n"
        "f = deprecated(since='0.7', removed_in='0.9')(lambda: 1)\n"
        "import types\n"
        "m = types.ModuleType('user'); exec('def g(f):\\n    for _ in range(3): f()', m.__dict__)\n"
        "m.g(f)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stderr.count("ThuncDeprecationWarning") == 1


@pytest.mark.parametrize(
    "older, newer", [("0.7", "0.9"), ("0.9", "0.10"), ("0.9.1", "1.0"), ("1.0.0rc1", "1.0.1"), ("1", "1.0.1")]
)
def test_versions_compare_as_numbers(older, newer):
    assert _version(older) < _version(newer)


def test_no_deprecation_outlives_the_release_that_removes_it():
    current = _version(thunc.__version__)
    overdue = [d.what for d in DEPRECATIONS if _version(d.removed_in) <= current]
    assert not overdue, f"thunc {thunc.__version__} should have removed: {overdue}"
