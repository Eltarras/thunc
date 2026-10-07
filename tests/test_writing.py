"""@thunc.function(write=True), end to end against a scripted backend: the function's answers come
from an oracle, and the draft and test-call requests get scripted replies."""

import asyncio
import doctest
import importlib.util
import json
import re
import sys
import textwrap
import threading
import time

import pytest

import thunc
from thunc import backends, writing

UNITS = {"h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60, "m": 1, "min": 1, "minute": 1, "minutes": 1}
GOOD = textwrap.dedent("""\
    units = {"h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60, "m": 1, "min": 1, "minute": 1, "minutes": 1}
    parts = re.findall(r"(\\d+(?:\\.\\d+)?)\\s*([a-z]+)", duration.lower())
    if not parts or any(unit not in units for _, unit in parts):
        raise ValueError(f"not a duration: {duration!r}")
    return round(sum(float(n) * units[unit] for n, unit in parts))
""")
CASES = ["minutes('2 hours')", "minutes('90 min')", "minutes('1.5 hrs')", "minutes('45m')", "minutes('3h')"]
MODULE = '''\
import thunc


@thunc.function(write=True)
def minutes(duration: str) -> int:
    """Convert a duration like '1h 30m', '90 min' or '2 hours' to whole minutes."""
    ...
'''


def truth(duration):
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", duration.lower())
    return round(sum(float(n) * UNITS[u] for n, u in parts))


def draft(body=GOOD, imports=("import re",), can_write=True, reason=""):
    return json.dumps({"can_write": can_write, "reason": reason, "imports": list(imports), "body": body})


class Scripted:
    """The model: answers the function's calls with an oracle; the draft and test-call requests get
    scripted replies, in order (the last test-call reply repeats)."""

    def __init__(self, oracle=truth, field="duration"):
        self.oracle, self.field = oracle, field
        self.drafts: list[str] = []
        self.cases: list[str] = [json.dumps(CASES)]
        self.draft_prompts: list[str] = []
        self.case_prompts: list[str] = []
        self.answers = 0
        self.delay = 0.0
        self.lock = threading.Lock()

    def __call__(self, text, **kwargs):
        time.sleep(self.delay)
        with self.lock:
            if text.startswith("<instructions>\nWrite the body"):
                self.draft_prompts.append(text)
                if not self.drafts:
                    raise AssertionError("ran out of scripted drafts")
                return self.drafts.pop(0)
            if text.startswith("<instructions>\nSuggest five test calls"):
                self.case_prompts.append(text)
                return self.cases.pop(0) if len(self.cases) > 1 else self.cases[0]
            self.answers += 1
        value = re.search(rf"<{self.field}>\n(.*?)\n</{self.field}>", text, re.S).group(1)
        return json.dumps(self.oracle(value))


@pytest.fixture
def model(monkeypatch):
    backend = Scripted()
    monkeypatch.setitem(backends.BACKENDS, "scripted", backend)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("THUNC_WRITE", raising=False)
    thunc.configure(backend="scripted")
    return backend


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


_count = 0


def module(path, text=MODULE):
    """Write `text` to `path` and import it under a fresh name."""
    global _count
    _count += 1
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"writing_demo_{_count}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_first_call_writes_the_file_and_runs_the_code(model, project, capsys):
    app = module(project / "app.py")
    model.drafts = [draft()]
    assert app.minutes("1h 30m") == 90
    written = (project / "app.py").read_text()
    assert written.startswith("import thunc\nimport re\n\n\ndef minutes(duration: str) -> int:\n")
    assert "@thunc" not in written
    assert ">>> minutes('1h 30m')\n    90\n    >>> minutes('2 hours')\n    120\n" in written
    assert "    # Written by thunc from the docstring on " in written
    assert "return round(sum(" in written
    err = capsys.readouterr().err
    assert "thunc: writing minutes() in app.py (first call)" in err
    assert "thunc: checked against 6 model answers: all agree" in err
    timing = r"in \d+s \(answer \d+s, draft \d+s, test calls \d+s; side by side\)"
    assert re.search(rf"thunc: wrote app.py lines 5-\d+ {timing}", err)
    assert "Removed @thunc.function. Review: git diff app.py" in err

    answers = model.answers  # one for the call, five for the test calls
    assert answers == 6
    assert app.minutes("2h 15m") == 135  # plain Python now
    assert model.answers == answers
    with pytest.raises(ValueError, match="not a duration"):
        app.minutes("soon")  # the written rule, not the model

    fresh = module(project / "app2.py", written)  # the file as written, imported again
    assert doctest.testmod(fresh).failed == 0
    assert fresh.minutes("3 hours") == 180


def test_the_requests_get_the_function_its_file_and_the_call(model, project):
    app = module(project / "app.py")
    model.drafts = [draft()]
    app.minutes("1h 30m")
    prompt = model.draft_prompts[0]
    assert "<function>\n@thunc.function(write=True)\ndef minutes(duration: str) -> int:" in prompt
    assert "<file>\napp.py\n</file>" in prompt
    assert "<module>\nimport thunc\n\n\n@thunc.function(write=True)" in prompt
    assert "<example_call>\nminutes('1h 30m')\n</example_call>" in prompt
    assert "<problems>\n" not in prompt and "<types>\n" not in prompt
    cases = model.case_prompts[0]
    assert "<function>\n@thunc.function(write=True)" in cases and "<example_call>\nminutes('1h 30m')" in cases
    assert "<module>" not in cases  # the test calls come from the docstring, not the code around it


def test_the_draft_test_calls_and_answer_run_side_by_side(model, project):
    app = module(project / "app.py")
    model.drafts = [draft()]
    model.delay = 0.4  # one after another: answer, draft, test calls, their answers = 1.6s
    started = time.monotonic()
    assert app.minutes("1h 30m") == 90
    assert time.monotonic() - started < 1.2  # side by side: the test calls, then their answers = 0.8s


def test_project_types_in_the_signature_are_sent_with_the_draft(model, project):
    (project / "models.py").write_text(
        "from dataclasses import dataclass\n\n\n@dataclass\nclass Span:\n    start: int\n    end: int\n"
    )
    sys.path.insert(0, str(project))
    try:
        text = MODULE.replace("import thunc\n", "import thunc\nfrom models import Span\n")
        text = text.replace("def minutes(duration: str) -> int:", "def minutes(duration: str, span: Span) -> int:")
        app = module(project / "app.py", text)
        model.drafts = [draft()]
        model.cases = [json.dumps([c.replace("')", "', Span(0, 1))") for c in CASES])]
        assert app.minutes("1h 30m", app.Span(0, 1)) == 90
    finally:
        sys.path.remove(str(project))
        sys.modules.pop("models", None)
    assert "<types>\n# models.py\n@dataclass\nclass Span:\n    start: int\n    end: int\n" in model.draft_prompts[0]
    assert ">>> minutes('2 hours', Span(0, 1))" in (project / "app.py").read_text()


def test_a_draft_that_disagrees_goes_back_with_the_failing_cases(model, project, capsys):
    app = module(project / "app.py")
    wrong = GOOD.replace('"hours": 60', '"hours": 6')
    model.drafts = [draft(wrong), draft()]
    assert app.minutes("1h 30m") == 90
    retry = model.draft_prompts[1]
    assert "<problems>\n- minutes('2 hours'): returned 12, but the model answered 120\n</problems>" in retry
    assert "<previous_body>" in retry and '"hours": 6' in retry
    assert len(model.case_prompts) == 1  # the test calls and their answers are reused
    err = capsys.readouterr().err
    assert "draft 1 of 3 failed 1 check(s); asking for another" in err
    assert re.search(r"draft \d+s \+ \d+s", err)
    assert '"hours": 60' in (project / "app.py").read_text()


def test_lint_problems_go_back_with_the_requests_retries(model, project):
    app = module(project / "app.py")
    model.drafts = [draft(imports=["import subprocess"]), draft(body="eval(duration)\nreturn 0\n"), draft()]
    model.cases = [json.dumps(["minutes(open('x'))", *CASES]), json.dumps(CASES)]
    assert app.minutes("1h 30m") == 90
    assert "may not import subprocess" in model.draft_prompts[1]
    assert "don't call eval" in model.draft_prompts[2]
    assert "isn't a literal" in model.case_prompts[1]


def test_a_body_that_has_the_test_inputs_written_in_is_rejected(model, project):
    app = module(project / "app.py")
    model.drafts = [draft(body="if duration == '2 hours':\n    return 120\nreturn 90\n"), draft()]
    assert app.minutes("1h 30m") == 90
    assert "contains the test input '2 hours': write the general rule instead" in model.draft_prompts[1]


def test_when_it_cant_be_written_the_file_stays_and_the_reason_is_kept(model, project):
    app = module(project / "app.py")
    model.drafts = [draft(can_write=False, reason="Needs judgment.", body="", imports=[])]
    with pytest.warns(RuntimeWarning, match=r"minutes\(\) stays a model call: Needs judgment"):
        assert app.minutes("1h 30m") == 90
    assert (project / "app.py").read_text() == MODULE
    assert app.minutes("2h") == 120  # through the model, without a new draft
    assert len(model.draft_prompts) == 1

    again = module(project / "app.py")  # the next run of the program
    with pytest.warns(RuntimeWarning, match="Change its docstring or signature to try again"):
        assert again.minutes("1h") == 60
    assert len(model.draft_prompts) == 1
    assert (project / ".thunc_write" / ".gitignore").read_text().endswith("*\n")

    changed = module(project / "app.py", MODULE.replace("whole minutes", "minutes"))
    model.drafts = [draft()]
    assert changed.minutes("1h 30m") == 90  # a new docstring tries again
    assert "def minutes" in (project / "app.py").read_text() and "@thunc" not in (project / "app.py").read_text()


def test_after_three_failed_drafts_it_stays_a_model_call(model, project):
    app = module(project / "app.py")
    wrong = GOOD.replace("round(", "int(")
    model.drafts = [draft(wrong)] * 3
    with pytest.warns(RuntimeWarning, match="no draft passed its checks"):
        assert app.minutes("1.51 hrs") == 91
    assert (project / "app.py").read_text() == MODULE
    memo = next((project / ".thunc_write").glob("*.json")).read_text()
    assert "no draft passed its checks in 3 tries" in memo


def test_too_few_answerable_test_calls_stops_without_keeping_a_reason(model, project):
    app = module(project / "app.py")
    model.oracle = lambda d: truth(d) if d == "1h 30m" else "not a number"
    model.drafts = [draft()]
    with pytest.warns(RuntimeWarning, match="only 0 test calls got an answer"):
        assert app.minutes("1h 30m") == 90
    assert (project / "app.py").read_text() == MODULE
    assert not list((project / ".thunc_write").glob("*.json"))  # the next run tries again


def test_an_error_from_the_answer_is_the_calls_error(model, project):
    app = module(project / "app.py")
    model.oracle = lambda d: "not a number"
    model.drafts = [draft()]
    with pytest.raises(thunc.ThuncError, match="No valid a JSON integer"):
        app.minutes("1h 30m")
    assert (project / "app.py").read_text() == MODULE


def test_a_draft_that_hangs_is_stopped(model, project, monkeypatch):
    monkeypatch.setattr(writing, "CASE_SECONDS", 0.2)
    app = module(project / "app.py")
    model.drafts = [draft("time.sleep(2)\nreturn 0\n", imports=["import time"]), draft()]
    assert app.minutes("1h 30m") == 90
    assert "took longer than 0.2s" in model.draft_prompts[1]


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (lambda mp, path: mp.setenv("CI", "true"), "running in CI"),
        (lambda mp, path: mp.setenv("THUNC_WRITE", "0"), "THUNC_WRITE is off"),
        (lambda mp, path: path.write_text(MODULE + "\n# edited\n"), "changed since it was imported"),
        (lambda mp, path: path.chmod(0o444), "read-only"),
    ],
)
def test_where_it_wont_write_it_answers_through_the_model(model, project, monkeypatch, setup, reason):
    path = project / "app.py"
    app = module(path)
    before = path.read_text()
    setup(monkeypatch, path)
    with pytest.warns(RuntimeWarning, match=f"won't be written: .*{reason}"):
        assert app.minutes("1h 30m") == 90
    assert app.minutes("2h") == 120
    assert model.draft_prompts == [] and model.case_prompts == []
    path.chmod(0o644)
    assert path.read_text().startswith(before)


def test_outside_a_project_it_answers_through_the_model(model, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    app = module(elsewhere / "app.py")
    with pytest.warns(RuntimeWarning, match="isn't in a git repository or under the current directory"):
        assert app.minutes("1h") == 60


def test_nested_functions_and_instructions_are_refused(model, project):
    @thunc.function(write=True)
    def inner(duration: str) -> int:
        """Minutes."""
        ...

    with pytest.warns(RuntimeWarning, match="defined inside another function"):
        assert inner("1h") == 60
    with pytest.raises(TypeError, match="instructions= can't be used"):
        thunc.function(write=True, instructions="Minutes.")(lambda duration: None)


def test_concurrent_first_calls_write_once(model, project):
    app = module(project / "app.py")
    model.drafts = [draft()]
    items = ["1h 30m", "2h", "45m", "3 hours", "10 min", "1.5 hrs", "2h 5m", "7m"]
    assert thunc.map(app.minutes, items) == [truth(i) for i in items]
    assert len(model.draft_prompts) == 1
    assert model.answers == 6  # the first call and its five test calls; the others waited and ran the code


def test_async_functions_are_written_and_awaited(model, project):
    app = module(project / "app.py", MODULE.replace("def minutes", "async def minutes"))
    model.drafts = [draft()]
    assert asyncio.run(app.minutes("1h 30m")) == 90
    written = (project / "app.py").read_text()
    assert "async def minutes(duration: str) -> int:" in written
    assert ">>>" not in written  # doctest can't await
    assert asyncio.run(app.minutes("2h")) == 120


def test_methods_are_written_and_keep_self(model, project):
    text = textwrap.dedent('''\
        import thunc


        class Clock:
            factor = 1

            @thunc.function(write=True)
            def minutes(self, duration: str) -> int:
                """Convert a duration like '1h 30m', '90 min' or '2 hours' to whole minutes."""
                ...
    ''')
    app = module(project / "app.py", text)
    model.drafts = [draft(GOOD.replace("return round(", "return self.factor * round("))]
    clock = app.Clock()
    assert clock.minutes("1h 30m") == 90
    written = (project / "app.py").read_text()
    assert "    def minutes(self, duration: str) -> int:\n" in written
    assert ">>>" not in written  # no instance for doctest
    clock.factor = 2
    assert clock.minutes("1h") == 120  # the written code, using self


def test_a_test_call_the_model_cant_answer_is_dropped(model, project):
    app = module(project / "app.py")
    model.oracle = lambda d: "not a number" if d == "3h" else truth(d)
    model.drafts = [draft()]
    assert app.minutes("1h 30m") == 90
    written = (project / "app.py").read_text()
    assert ">>> minutes('45m')" in written and ">>> minutes('3h')" not in written


def test_the_float_kind_matters_for_int_functions(model, project):
    app = module(project / "app.py")
    model.drafts = [draft(GOOD.replace("return round(", "return float(round(").rstrip() + ")\n"), draft()]
    assert app.minutes("1h 30m") == 90
    assert "returned 120.0, which fails: it's a float, not an int\n" in model.draft_prompts[1]


# --- thunc write ---------------------------------------------------------------------------


@pytest.fixture
def cli(project, monkeypatch):
    """Run `thunc write` in the project, with sys.path and sys.modules as they were afterwards."""
    from thunc.__main__ import main

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    sys.modules.pop("app", None)
    return lambda *args: main(["write", *args])


def test_thunc_write_writes_a_function_before_any_call(model, project, cli, capsys):
    (project / "app.py").write_text(MODULE + '\n\nif __name__ == "__main__":\n    raise SystemExit("ran main")\n')
    model.drafts = [draft()]
    assert cli("app.py::minutes") == 0
    written = (project / "app.py").read_text()
    assert "def minutes(duration: str) -> int:" in written and "@thunc.function" not in written
    assert ">>> minutes('2 hours')\n    120\n" in written  # the test calls, as there's no call
    assert "<example_call>\n(none: the function is written before its first call)" in model.draft_prompts[0]
    assert model.answers == 5  # the test calls only
    err = capsys.readouterr().err
    assert "thunc: writing minutes() in app.py\n" in err and "checked against 5 model answers" in err


def test_thunc_write_dry_run_shows_a_diff_and_leaves_the_file(model, project, cli, capsys):
    (project / "app.py").write_text(MODULE)
    model.drafts = [draft()]
    assert cli("--dry-run", "app.py::minutes") == 0
    assert (project / "app.py").read_text() == MODULE
    out, err = capsys.readouterr()
    assert out.startswith("--- a/app.py\n+++ b/app.py\n")
    assert "-@thunc.function(write=True)\n" in out and "+import re\n" in out and "-    ...\n" in out
    assert "(dry run: app.py is unchanged)" in err


def test_thunc_write_says_why_it_didnt_write(model, project, cli, capsys):
    (project / "app.py").write_text(MODULE + "\n\n@thunc.function\ndef plain(x: str) -> str:\n    '''Plain.'''\n")
    assert cli("app.py") == 1
    assert cli("app.py::nothing") == 1
    assert cli("app.py::plain") == 1
    model.drafts = [draft(can_write=False, reason="Needs judgment.", body="", imports=[])]
    assert cli("--dry-run", "app.py::minutes") == 1
    err = capsys.readouterr().err
    assert "thunc write: 'app.py' isn't FILE::FUNCTION" in err
    assert "thunc write: nothing isn't in the file" in err
    assert "thunc write: plain isn't a @thunc.function(write=True)" in err
    assert "thunc write: minutes() wasn't written: Needs judgment." in err
    assert not (project / ".thunc_write").exists() or not list((project / ".thunc_write").glob("*.json"))


def test_thunc_write_ignores_a_saved_reason_and_refuses_methods(model, project, cli, capsys):
    app = module(project / "app.py")
    model.drafts = [draft(can_write=False, reason="Needs judgment.", body="", imports=[])]
    with pytest.warns(RuntimeWarning):
        app.minutes("1h")  # saves the reason
    model.drafts = [draft()]
    assert cli("app.py::minutes") == 0  # asked for on purpose
    assert "@thunc" not in (project / "app.py").read_text()

    (project / "clock.py").write_text(
        "import thunc\n\n\nclass Clock:\n    @thunc.function(write=True)\n    def minutes(self, d: str) -> int:\n"
        '        """Minutes."""\n'
    )
    assert cli("clock.py::Clock.minutes") == 1
    assert "it's a method, and thunc write has no instance to test it with" in capsys.readouterr().err
