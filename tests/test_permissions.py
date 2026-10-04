"""Permission rules on their own: parsing, matching, deny wins, the defaults."""

import pytest

from thunc.permissions import Denied, Permissions


def allowed(check, path):
    try:
        check(path)
    except Denied:
        return False
    return True


def test_defaults_read_everything_save_notes_write_nothing():
    p = Permissions()
    assert allowed(p.check_read, "src/app.py") and allowed(p.check_read, ".env")
    assert not allowed(p.check_write, "src/app.py")
    assert p.may("memory") and not p.may("write")


@pytest.mark.parametrize(
    "rule, path, ok",
    [
        ("write:CHANGELOG.md", "CHANGELOG.md", True),
        ("write:CHANGELOG.md", "docs/CHANGELOG.md", False),
        ("write:*.md", "README.md", True),
        ("write:*.md", "docs/guide.md", False),  # * stays within one folder
        ("write:**/*.md", "docs/guide.md", True),
        ("write:**/*.md", "README.md", True),  # **/ also matches no folder
        ("write:docs/**", "docs/a/b/c.txt", True),
        ("write:docs/", "docs/a.txt", True),  # a trailing / means everything under it
        ("write:docs/**", "docsx/a.txt", False),
        ("write:src/?.py", "src/a.py", True),
        ("write:src/?.py", "src/ab.py", False),
        ("write:./notes.txt", "notes.txt", True),
        ("write:a[1].txt", "a[1].txt", True),  # brackets are literal, not a character class
        ("write", "anything/at/all.py", True),
    ],
)
def test_globs(rule, path, ok):
    assert allowed(Permissions([rule]).check_write, path) is ok


def test_deny_wins_and_read_deny_also_stops_writing():
    p = Permissions(["write", "!write:secrets/**", "!read:.env*"])
    assert allowed(p.check_write, "src/app.py")
    assert not allowed(p.check_write, "secrets/key.pem")
    assert allowed(p.check_read, "secrets/key.pem")  # !write leaves reading alone
    assert not allowed(p.check_read, ".env.local") and not allowed(p.check_write, ".env.local")
    with pytest.raises(Denied, match=r"denied by '!read:\.env\*'"):
        p.check_write(".env")


def test_read_rules_replace_the_default():
    p = Permissions(["read:src/**"])
    assert allowed(p.check_read, "src/app.py") and not allowed(p.check_read, "README.md")
    p = Permissions(["write:docs/**"])  # a write rule alone keeps reading everything
    assert allowed(p.check_read, "README.md")


def test_write_implies_read():
    p = Permissions(["read:src/**", "write:CHANGELOG.md"])
    assert allowed(p.check_read, "CHANGELOG.md") and not allowed(p.check_read, "README.md")


def test_memory_and_whole_kind_denies():
    assert not Permissions(["!memory"]).may("memory")
    assert not Permissions(["write", "!write"]).may("write")
    assert Permissions(["write:docs/**", "!write:docs/private/**"]).may("write")
    p = Permissions(["!read"])
    assert not allowed(p.check_read, "a.txt") and not p.may("read")


@pytest.mark.parametrize(
    "rule, problem",
    [
        ("run:pytest", "running commands isn't supported yet"),
        ("delete:x", "Unknown permission"),
        ("memory:notes", "memory takes no path"),
        ("write:", "no path after the colon"),
        ("write:/etc/passwd", "stay inside it"),
        ("write:../other/x", "stay inside it"),
        ("write:docs/../../x", "stay inside it"),
        ("", "Not a permission rule"),
    ],
)
def test_bad_rules_fail_when_the_agent_is_declared(rule, problem):
    with pytest.raises(ValueError, match=problem):
        Permissions([rule])


def test_a_single_string_is_refused():
    with pytest.raises(ValueError, match="takes a list"):
        Permissions("write:docs/**")


def test_denial_messages_name_the_rules():
    p = Permissions(["write:CHANGELOG.md", "write:docs/**"])
    with pytest.raises(
        Denied, match=r"no write rule matches it; this agent may write: write:CHANGELOG\.md, write:docs/\*\*"
    ):
        p.check_write("pyproject.toml")


def test_describe():
    text = Permissions(["write:CHANGELOG.md", "write:docs/**", "!read:.env*", "!memory"]).describe()
    assert text == (
        "Your permissions:\n"
        "- Read: everything (and anything you may write), except .env*.\n"
        "- Write: CHANGELOG.md, docs/**, except .env*.\n"
        "- Save notes with remember: no."
    )
