"""Explicit worker-local task registration: no remote import paths or pickling."""

from __future__ import annotations

import copy
import json
import math
import os
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thunc.agent import Agent
from thunc.backends import DEFAULT_MODELS, TYPED_BACKENDS
from thunc.config import resolve_backend, setting
from thunc.decorator import _Signature
from thunc.schema import json_schema, validate
from thunc.store import Store

from .models import digest, plain
from .storage import Storage, atomic


@dataclass
class Definition:
    name: str
    version: str
    workspace: str
    function: Callable[..., Any]
    spec: _Signature
    agent: Agent | None
    options: dict[str, Any]
    fingerprint: str
    store_folder: str | None
    endpoint: str | None

    def inputs(self, values: dict[str, Any]) -> dict[str, Any]:
        bound = self.spec.sig.bind(**values)
        bound.apply_defaults()
        hints = typing.get_type_hints(self.function)
        return {k: validate(v, hints.get(k, Any)) for k, v in bound.arguments.items()}


class Registry:
    def __init__(self, *, state_dir: str | Path = ".thunc_temporal") -> None:
        self.storage = Storage(state_dir)
        self.definitions: dict[tuple[str, str], Definition] = {}
        self.roots: dict[str, str] = {}

    def agent_task(self, name: str, task: Callable[..., Any], *, version: str, workspace_id: str) -> None:
        agent = getattr(task, "__thunc_agent__", None)
        if not isinstance(agent, Agent):
            raise ValueError("Expected an @agent.task function")
        self._register(name, task, version, workspace_id, agent)

    def function(self, name: str, function: Callable[..., Any], *, version: str, workspace_id: str) -> None:
        if not hasattr(function, "__thunc_function__"):
            raise ValueError("Expected an @thunc.function function")
        self._register(name, function, version, workspace_id, None)

    def _register(self, name: str, fn: Callable[..., Any], version: str, workspace: str, agent: Agent | None) -> None:
        if not all(isinstance(x, str) and 0 < len(x) <= 200 for x in (name, version, workspace)):
            raise ValueError("Task, version and workspace IDs must be 1–200 characters")
        if (name, version) in self.definitions:
            raise ValueError("Duplicate task/version registration")
        spec: _Signature = fn.__dict__["__thunc_spec__"]
        for parameter in spec.sig.parameters.values():
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise ValueError("Durable tasks require named, fixed parameters")
        if spec.skip:
            raise ValueError("Register a plain task function, not a bound method")
        options = dict(fn.__dict__.get("__thunc_options__", {}))
        options.pop("cache", None)
        backend = resolve_backend(agent.backend if agent else options.get("backend"))
        if backend in TYPED_BACKENDS:
            raise ValueError("Typed-only backends are not supported by the durable adapter")
        model = (agent.model if agent else options.get("model")) or setting("model") or DEFAULT_MODELS.get(backend)
        options.update(backend=backend, model=model)
        if not agent and options.get("system") is None:
            options["system"] = setting("system")
        if agent:
            if agent.custom:
                raise ValueError("Durable agents don't support tools= yet: their effects can't be journaled")
            if agent.timeout is not None:
                raise ValueError("Durable agents take a run deadline: use deadline_seconds= instead of timeout=")
            agent = copy.copy(agent)
            agent.backend, agent.model = backend, model
            if not math.isfinite(agent.command_timeout) or agent.command_timeout > 86400:
                raise ValueError("Durable command_timeout must be finite and at most one day")
            if not isinstance(agent.max_steps, int) or agent.max_steps > 4096:
                raise ValueError("Durable max_steps must be an integer no greater than 4096")
            root = agent.workdir
            for other, path in self.roots.items():
                if workspace != other and os.path.commonpath([root, path]) in (root, path):
                    raise ValueError("Durable workspace roots must not overlap")
            if workspace in self.roots and self.roots[workspace] != root:
                raise ValueError("A workspace ID must identify exactly one root")
            if self.storage.root == Path(root) or Path(root) in self.storage.root.parents:
                raise ValueError("state_dir must be outside the agent workspace")
            self.roots[workspace] = root
        endpoint = os.environ.get("OPENAI_BASE_URL") if backend == "openai" else os.environ.get("ANTHROPIC_BASE_URL")
        fingerprint = digest(
            {
                "instructions": spec.instructions,
                "returns": json_schema(spec.returns),
                "backend": backend,
                "model": model,
                "endpoint": endpoint,
                "settings": {k: v for k, v in agent._settings().items() if k != "permissions"}
                if agent
                else {k: v for k, v in options.items() if k != "ensure"},
            }
        )
        self.definitions[name, version] = Definition(
            name,
            version,
            workspace,
            fn,
            spec,
            agent,
            options,
            fingerprint,
            Store(agent.name).folder if agent else None,
            endpoint,
        )

    def get(self, request: dict[str, Any]) -> Definition:
        definition = self.definitions.get((request["task"], request["version"]))
        if definition is None or definition.workspace != request["workspace_id"]:
            raise ValueError("Task/version/workspace is not registered on this worker")
        return definition

    def claim(self, namespace: str) -> None:
        with self.storage.lock("registry"):
            for workspace, root in self.roots.items():
                with self.storage.db() as db:
                    for old_id, path, old_namespace in db.execute("SELECT id,path,namespace FROM workspaces"):
                        if old_id == workspace:
                            if path != root or old_namespace != namespace:
                                raise ValueError("Workspace ownership cannot change namespace or root")
                        elif os.path.commonpath([root, path]) in (root, path):
                            raise ValueError("Workspace overlaps an existing durable workspace")
                    db.execute("INSERT OR IGNORE INTO workspaces VALUES (?,?,?)", (workspace, root, namespace))
                owner = {"workspace_id": workspace, "namespace": namespace, "state_dir": str(self.storage.root)}
                for definition in self.definitions.values():
                    if definition.workspace != workspace or not definition.agent:
                        continue
                    store = Store(definition.agent.name)
                    store.folder = str(definition.store_folder)
                    with store.lock():
                        for location in (Path(root), Path(store.folder)):
                            marker = location / ".thunc-temporal-owner"
                            if marker.exists():
                                if json.loads(marker.read_text()) != owner:
                                    raise ValueError("Workspace or agent memory already has a different durable owner")
                            atomic(marker, json.dumps(owner).encode())

    def validate_request(self, request: dict[str, Any]) -> None:
        definition = self.get(request)
        definition.inputs(request["inputs"])
        if request["returns"] != json_schema(definition.spec.returns):
            raise ValueError("Client and registered return schemas do not match")
        plain(request)
