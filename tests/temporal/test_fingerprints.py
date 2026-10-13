"""Durable task fingerprints don't change with thunc (docs/docs/api.html#fingerprints).

A worker refuses to continue a run whose task's fingerprint changed, so a thunc release that changed
the fingerprint of a task registered the same way would strand every run in flight. These pin what
goes into the fingerprints of a set of task definitions, with the workdir written as <workdir>, and
the digest of each. If one fails, the change alters fingerprints: make the new input join only when
it's set, or say so in the changelog's Behavior changes (see the docs above)."""

import json
from dataclasses import dataclass
from typing import Literal

import pytest

pytest.importorskip("temporalio")

import thunc  # noqa: E402
from thunc.temporal import Registry  # noqa: E402
from thunc.temporal.models import digest  # noqa: E402


@dataclass
class Finding:
    path: str
    line: int


def open_issue(title: str) -> int:
    """Open an issue and return its number."""
    return 1


def definitions(tmp_path):
    """name -> Definition, for the task shapes a fingerprint has to cover."""
    root = tmp_path / "repo"
    root.mkdir()
    registry = Registry(state_dir=tmp_path / "state")
    plain = thunc.Agent("plain", workdir=root, backend="anthropic", model="claude-opus-5-5")
    rules = thunc.Agent(
        "rules",
        workdir=root,
        backend="openai",
        model="gpt-5.5",
        permissions=["write:docs/**", "run:pytest", "!read:.env*"],
        env={"CI": "1"},
        command_timeout=60,
        follow=True,
        protocol="text",
        max_steps=12,
        retries=1,
        system="You review code.",
        effort="low",
    )
    tooled = thunc.Agent("tooled", workdir=root, backend="anthropic", model="claude-opus-5-5", tools=[open_issue])

    @plain.task
    def summary(topic: str) -> str:
        """Summarize what the repo says about this topic."""
        ...

    @rules.task
    def findings(limit: int = 3) -> list[Finding]:
        """Find up to `limit` places where errors are swallowed."""
        ...

    @tooled.task
    def report(problem: str) -> int:
        """Open an issue for this problem and return its number."""
        ...

    @thunc.function(backend="anthropic", model="claude-opus-5-5", system="You triage tickets.", retries=3)
    def label(ticket: str) -> Literal["bug", "billing", "other"]:
        """Classify this ticket."""
        ...

    registry.agent_task("summary", summary, version="1", workspace_id="w")
    registry.agent_task("findings", findings, version="1", workspace_id="w")
    registry.agent_task("report", report, version="1", workspace_id="w", retry_safe_tools=["open_issue"])
    registry.function("label", label, version="1", workspace_id="w")
    return {name: registry.get({"task": name, "version": "1", "workspace_id": "w"}) for name in
            ("summary", "findings", "report", "label")}, str(root)  # fmt: skip


def normalized(value, root):
    return json.loads(json.dumps(value).replace(json.dumps(root)[1:-1], "<workdir>"))


EXPECTED: dict[str, tuple[dict, str]] = {
    "summary": (
        {
            "backend": "anthropic",
            "endpoint": None,
            "instructions": "Summarize what the repo says about this topic.",
            "model": "claude-opus-5-5",
            "returns": {"type": "string"},
            "settings": {
                "backend": "anthropic",
                "command_timeout": 120.0,
                "env": [],
                "follow": [],
                "max_steps": 40,
                "model": "claude-opus-5-5",
                "name": "plain",
                "protocol": None,
                "retries": 2,
                "system": None,
                "timeout": None,
                "tools": [],
                "workdir": "<workdir>",
            },
        },
        "a0a6eb9801acaf292b2966fa844eb2bf63c795b86711966dcb6e55a77d108c1c",
    ),
    "findings": (
        {
            "backend": "openai",
            "endpoint": None,
            "instructions": "Find up to `limit` places where errors are swallowed.",
            "model": "gpt-5.5",
            "returns": {
                "items": {
                    "properties": {"line": {"type": "integer"}, "path": {"type": "string"}},
                    "required": ["path", "line"],
                    "type": "object",
                },
                "type": "array",
            },
            "settings": {
                "backend": "openai",
                "command_timeout": 60,
                "effort": "low",
                "env": ["CI"],
                "follow": ["AGENTS.md", "CLAUDE.md"],
                "max_steps": 12,
                "model": "gpt-5.5",
                "name": "rules",
                "protocol": "text",
                "retries": 1,
                "system": "You review code.",
                "timeout": None,
                "tools": [],
                "workdir": "<workdir>",
            },
        },
        "b2d7df07382a6913318d5ced9dfb99a12479fa204d4834aab371ad98365b4c48",
    ),
    "report": (
        {
            "backend": "anthropic",
            "endpoint": None,
            "instructions": "Open an issue for this problem and return its number.",
            "model": "claude-opus-5-5",
            "returns": {"type": "integer"},
            "settings": {
                "backend": "anthropic",
                "command_timeout": 120.0,
                "env": [],
                "follow": [],
                "max_steps": 40,
                "model": "claude-opus-5-5",
                "name": "tooled",
                "protocol": None,
                "retries": 2,
                "system": None,
                "timeout": None,
                "tools": ["open_issue"],
                "workdir": "<workdir>",
            },
            "tools": {
                "open_issue": {
                    "description": "Open an issue and return its number.",
                    "retry_safe": True,
                    "schema": {
                        "additionalProperties": False,
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                        "type": "object",
                    },
                }
            },
        },
        "a00faa08079ea45a9859e65cabab90f8dc04309342464b067a526a04dc4c8f3c",
    ),
    "label": (
        {
            "backend": "anthropic",
            "endpoint": None,
            "instructions": "Classify this ticket.",
            "model": "claude-opus-5-5",
            "returns": {"enum": ["bug", "billing", "other"]},
            "settings": {
                "backend": "anthropic",
                "model": "claude-opus-5-5",
                "retries": 3,
                "system": "You triage tickets.",
            },
        },
        "a21c86b5e619b400c3ccbf049f22a9ea51dc1932dc2a70d8f48a81dc5070e115",
    ),
}


def test_fingerprints_are_the_digest_of_what_they_cover(tmp_path, monkeypatch):
    for var in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    found, root = definitions(tmp_path)
    for definition in found.values():
        assert definition.fingerprint == digest(definition.fingerprinted)


@pytest.mark.parametrize("name", ["summary", "findings", "report", "label"])
def test_a_task_registered_the_same_way_keeps_its_fingerprint(tmp_path, monkeypatch, name):
    for var in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    found, root = definitions(tmp_path)
    covered = normalized(found[name].fingerprinted, root)
    expected, expected_digest = EXPECTED[name]
    assert covered == expected
    assert digest(covered) == expected_digest
