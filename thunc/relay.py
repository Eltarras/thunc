"""The run's end of the MCP relay: how a CLI's model makes native calls to an agent's tools.

The CLI (Claude Code or Codex) starts mcp_relay.py as an MCP server. The relay connects back to the
run over 127.0.0.1 with a one-time token and forwards each tool call; the run carries it out with
its own tools, permissions and records, and sends the result back. Codex starts a new relay for
each turn of a run (each `codex exec resume`), so connections are accepted for as long as the run
lasts, and the newest is the one calls come from and results go to.

When a CLI can't start native calls (an older CLI, MCP servers turned off by a policy), WithFallback
switches the run to the text protocol, and later runs in the process with the same backend start
there.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import secrets
import socket
import sys
import threading
from collections.abc import Callable, Sequence
from typing import IO, Any, Protocol

from .errors import ThuncError
from .native import Conversation, Reply, Tool

RELAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_relay.py")
GATHER_SECONDS = 0.05  # how long to wait for more calls of the same model reply
unavailable: dict[str, str] = {}  # backend -> why native calls couldn't start in this process, once they couldn't

Events = queue.Queue[tuple[str, Any]]


class StartupError(ThuncError):
    """The CLI couldn't start a run with thunc's tools: an older CLI that doesn't know the options,
    MCP servers turned off by a policy, or a tool server that never connected."""


class Relay:
    """A listener the relay connects back to. Calls it forwards go on `events` as ("call", message);
    if none connects within `timeout` seconds of the start, ("relay", why) does."""

    def __init__(self, folder: str, tools: Sequence[Tool], events: Events, timeout: float) -> None:
        tools_file = os.path.join(folder, "tools.json")
        with open(tools_file, "w", encoding="utf-8") as f:
            json.dump([{"name": t.name, "description": t.description, "inputSchema": t.schema} for t in tools], f)
        self.listener = socket.create_server(("127.0.0.1", 0))
        token = secrets.token_hex(16)
        port = self.listener.getsockname()[1]
        # How the CLI starts the relay: an MCP server entry.
        self.server: dict[str, Any] = {
            "command": sys.executable,
            "args": [RELAY],
            "env": {"THUNC_RELAY": f"{port} {token}", "THUNC_RELAY_TOOLS": tools_file},
        }
        self.connected = threading.Event()  # a relay is connected: thunc's tools are available
        self.out: IO[str] | None = None
        self.opened: list[Any] = []  # connections and their files, closed with the relay
        self.ids: dict[str, Any] = {}  # call id as thunc knows it -> the relay's JSON-RPC id
        self._write = threading.Lock()
        threading.Thread(target=self._accept, args=(token, events, timeout), daemon=True).start()

    def _accept(self, token: str, events: Events, timeout: float) -> None:
        """Take each relay's connection (checking its token), and queue the calls it forwards."""
        self.listener.settimeout(timeout)
        while True:
            try:
                connection, _ = self.listener.accept()
            except OSError as exc:  # a timeout, or the listener closed at the end of the run
                if not self.connected.is_set():
                    events.put(("relay", f"no connection from the tool server ({exc})"))
                return
            reader = connection.makefile("r", encoding="utf-8")
            self.opened += [connection, reader]
            hello = reader.readline()
            ok = False
            with contextlib.suppress(ValueError, AttributeError):
                ok = secrets.compare_digest(str(json.loads(hello).get("token")), token)
            if not ok:
                connection.close()  # not our relay
                continue
            connection.settimeout(None)
            with self._write:
                self.out = connection.makefile("w", encoding="utf-8")
                self.opened.append(self.out)
            self.connected.set()
            self.listener.settimeout(None)  # later relays (Codex's next turns) come when they come
            threading.Thread(target=self._read, args=(reader, events), daemon=True).start()

    @staticmethod
    def _read(reader: IO[str], events: Events) -> None:
        with contextlib.suppress(OSError, ValueError):  # the relay closed at the end of the run
            for line in reader:
                with contextlib.suppress(ValueError):
                    events.put(("call", json.loads(line)))

    def calls(self, first: dict[str, Any], events: Events) -> list[tuple[str, str, Any]]:
        """The calls of one model reply, as (id, tool, arguments): the first, and any that arrive
        with it. Something else that arrives is put back for the next step."""
        forwarded = [first]
        while True:
            try:
                kind, event = events.get(timeout=GATHER_SECONDS)
            except queue.Empty:
                break
            if kind != "call":
                events.put((kind, event))  # for the next step (rare: an end right after calls)
                break
            forwarded.append(event)
        found = []
        for item in forwarded:
            ident = json.dumps(item.get("id"))
            self.ids[ident] = item.get("id")
            found.append((ident, str(item.get("name") or ""), item.get("arguments")))
        return found

    def send(self, ident: str, text: str, error: bool) -> None:
        """A call's result, to the relay that forwarded it."""
        with self._write:
            if self.out is None:
                raise ThuncError("the tool server isn't connected")
            try:
                self.out.write(
                    json.dumps({"id": self.ids.pop(ident), "text": text, "error": error}, ensure_ascii=False) + "\n"
                )
                self.out.flush()
            except OSError as exc:
                raise ThuncError(f"lost the tool server: {exc}") from None

    def close(self) -> None:
        for closable in (*self.opened, self.listener):
            with contextlib.suppress(OSError):
                closable.close()


class Native(Conversation, Protocol):
    """A conversation through a CLI and the relay: closed at the end of the run."""

    def close(self) -> None: ...


class WithFallback:
    """The native conversation, or the text protocol when the CLI can't start one. Decided at the
    first step: a StartupError there switches this run to `text()` (and later runs in this process
    on the same backend start with the text protocol); any failure after that is a failure."""

    def __init__(
        self, backend: str, native: Native, text: Callable[[], Conversation], switched: Callable[[str], None]
    ) -> None:
        self.backend, self.native, self.text, self.switched = backend, native, text, switched
        self.current: Conversation = native
        self.decided = False

    def next(self) -> Reply:
        if self.decided:
            return self.current.next()
        try:
            reply = self.native.next()
        except StartupError as exc:
            self.native.close()
            unavailable[self.backend] = str(exc)
            self.current = self.text()
            self.decided = True
            self.switched(str(exc))
            return self.current.next()
        self.decided = True
        return reply

    def results(self, results: Sequence[tuple[Any, str, bool]]) -> None:
        self.current.results(results)

    def nudge(self, reply: Reply) -> None:
        self.current.nudge(reply)

    def close(self) -> None:
        self.native.close()
