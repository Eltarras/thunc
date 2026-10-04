"""Persistent artifacts and effect receipts on a worker-owned local volume.

SQLite records intentions before effects. It is not a workflow scheduler. Temporal
owns orchestration; this journal closes the filesystem/completion crash window.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from thunc.store import Store

from .models import STATE_LIMIT, encoded


class Attention(Exception):
    """An effect requires explicit reconciliation, never a blind retry."""


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".thunc-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class Storage:
    def __init__(self, folder: str | Path):
        self.root = Path(folder).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS effects (
                    id TEXT PRIMARY KEY, run TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT);
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, namespace TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS resolutions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
            """)

    @contextlib.contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.root / "journal.sqlite3", timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def lock(self, key: str) -> Any:
        store = Store("temporal-lock")
        store.folder = str(self.root / "locks" / hashlib.sha256(key.encode()).hexdigest())
        return store.lock()

    def put(self, value: Any) -> dict[str, str]:
        data = encoded(value)
        if len(data) > STATE_LIMIT:
            raise ValueError("Durable state exceeded 16 MiB; reduce input/context size")
        key = hashlib.sha256(data).hexdigest()
        path = self.root / "artifacts" / key
        if not path.exists():
            atomic(path, data)
        return {"sha256": key}

    def get(self, ref: dict[str, str]) -> Any:
        key = ref["sha256"]
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact reference")
        data = (self.root / "artifacts" / key).read_bytes()
        if hashlib.sha256(data).hexdigest() != key:
            raise ValueError("Corrupt durable artifact")
        return json.loads(data)

    def admit(self, request: dict[str, Any]) -> bool:
        # Result retention may expire, but ID tombstones never silently expire.
        identity = request["id"]
        fingerprint = encoded({k: v for k, v in request.items() if k not in {"queue"}}).decode()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT fingerprint FROM requests WHERE id=?", (identity,)).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise ValueError("Request ID already used with different inputs or options")
                return False
            db.execute("INSERT INTO requests VALUES (?, ?, NULL)", (identity, fingerprint))
            return True

    def finish(self, identity: str, result: dict[str, Any]) -> None:
        ref = self.put(result)
        with self.db() as db:
            db.execute("UPDATE requests SET result=? WHERE id=?", (json.dumps(ref), identity))
        # A diagnostic projection only: failure never invalidates a completed run.
        with contextlib.suppress(OSError):
            atomic(self.root / "runs" / (hashlib.sha256(identity.encode()).hexdigest() + ".json"), encoded(result))

    def result(self, identity: str) -> dict[str, Any] | None:
        with self.db() as db:
            row = db.execute("SELECT result FROM requests WHERE id=?", (identity,)).fetchone()
        return self.get(json.loads(row[0])) if row and row[0] else None

    def effect(self, identity: str) -> tuple[str, dict[str, Any]] | None:
        with self.db() as db:
            row = db.execute("SELECT status, body FROM effects WHERE id=?", (identity,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def save_effect(self, identity: str, run: str, status: str, body: dict[str, Any]) -> None:
        with self.db() as db:
            db.execute(
                "INSERT INTO effects VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "status=excluded.status, body=excluded.body",
                (identity, run, status, encoded(body).decode()),
            )

    def unresolved(self, run: str) -> list[str]:
        with self.db() as db:
            return [r[0] for r in db.execute("SELECT id FROM effects WHERE run=? AND status!='done'", (run,))]

    def apply_file(self, identity: str, run: str, body: dict[str, Any]) -> dict[str, Any]:
        plan = body["plan"]
        path = Path(plan["path"])
        current = file_hash(path)
        if current == plan["after"]:
            pass
        elif current == plan["before"]:
            atomic(path, plan["content"].encode("utf-8"))
        else:
            raise Attention("File changed outside this operation; reconcile before continuing")
        self.save_effect(identity, run, "done", body)
        return cast(dict[str, Any], body["result"])
