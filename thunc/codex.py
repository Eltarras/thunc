"""Native tool calls on the codex backend: the agent's tools as an MCP server for Codex.

As on Claude Code (see claude_code.py), the model gets real tool calls instead of writing each
action as JSON text: Codex starts thunc's relay (mcp_relay.py) as an MCP server, given with
`-c mcp_servers.thunc.*`, and the relay forwards each call to the run (see relay.py).

- One `codex exec` is one turn: the model calls tools until it ends the turn. A turn that ends
  without finish gets a nudge, and a turn that fails gets a second chance, each as a new turn of the
  same session (`codex exec resume <thread id>`), which starts a new relay.
- Codex's own tools and the user's Codex setup stay out, as for plain calls (backends._CODEX_LOCKDOWN),
  and its read-only sandbox stays on: the agent changes files only through thunc's tools. Only
  thunc's server is approved to run without asking (default_tools_approval_mode), as `codex exec`
  has no one to ask.
- Resuming needs the session saved, so a run's session is deleted when the run ends.

Codex runs in the agent's working directory, so the environment details it adds name the folder the
tools work in.
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from typing import Any

from .backends import _CODEX_LOCKDOWN, _CODEX_QUIET, _codex_problem
from .errors import ThuncError
from .native import NUDGE, Call, Reply, Tool, _calls_text
from .relay import Relay, StartupError

TURN_ERRORS = 2  # turns that may fail before the run does
TOOL_MARGIN = 60.0  # seconds a tool may take beyond the command timeout before Codex gives up on it


class CodexConversation:
    """A Conversation (see native.py) whose model replies and tool calls go through Codex."""

    def __init__(
        self,
        system: str,
        request: str,
        tools: Sequence[Tool],
        model: str | None,
        workdir: str,
        timeout: float,
        effort: str | None = None,
        tool_timeout: float = 120.0,
    ) -> None:
        if shutil.which("codex") is None:
            raise ThuncError("`codex` was not found on PATH.")
        if effort == "max":
            raise ThuncError("The codex backend takes effort low, medium, high or xhigh, not max.")
        self.system, self.request, self.tools, self.model = system, request, list(tools), model
        self.effort = effort
        self.workdir = workdir
        self.timeout = timeout  # seconds to wait for the model's next step
        self.tool_timeout = tool_timeout + TOOL_MARGIN  # how long Codex waits for one of thunc's tools
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.turn = 0  # which turn's process events come from; older ones' are stale
        self.folder: str | None = None
        self.relay: Relay | None = None
        self.thread: str | None = None  # Codex's session id, to continue it and delete it
        self.text = ""  # the last message of the turn
        self.problem: str | None = None  # the last error Codex reported in the turn
        self.turn_errors = 0
        self.stderr: collections.deque[str] = collections.deque(maxlen=50)
        self.usage: collections.Counter[str] = collections.Counter()  # tokens, added up over the turns

    @property
    def connected(self) -> bool:
        """Whether a relay has connected: thunc's tools are available to the model."""
        return self.relay is not None and self.relay.connected.is_set()

    # --- the Conversation protocol ---

    def next(self) -> Reply:
        if self.process is None:
            self.folder = tempfile.mkdtemp(prefix="thunc-codex-")
            with open(os.path.join(self.folder, "system.md"), "w", encoding="utf-8") as f:
                f.write(self.system)
            self.relay = Relay(self.folder, self.tools, self.events, self.timeout)
            self._turn(self.request)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                kind, event = self.events.get(timeout=max(deadline - time.monotonic(), 0.001))
            except queue.Empty:
                if not self.connected:
                    raise StartupError(f"thunc's tool server didn't connect within {self.timeout:g}s") from None
                raise ThuncError(f"codex took no step within {self.timeout:g}s") from None
            if kind == "call":
                return self._calls(event)
            if kind == "relay":
                raise StartupError(f"codex couldn't start thunc's tools: {event}")
            turn, event = event
            if turn != self.turn:
                continue  # from a turn already over
            if kind == "event":
                ended = self._event(event)
                if ended == "completed":
                    if not self.connected:  # a turn without thunc's tools: they never started
                        raise StartupError("the model's turn ended before thunc's tool server connected")
                    return Reply([], self.text, "no tool call")
                if ended == "failed":
                    self.turn_errors += 1
                    problem = (self.problem or "an error")[:500]
                    if self.turn_errors > TURN_ERRORS:
                        raise ThuncError(f"codex error: {problem}")
                    self._turn(f"That step failed ({problem}). Carry on with the task with your tools.")
                    deadline = time.monotonic() + self.timeout
            elif kind == "exit":
                time.sleep(0.1)  # what it printed to stderr on the way out says why
                tail = (self.problem or "".join(self.stderr)).strip()[-500:]
                message = f"codex exited {event} before the task was done" + (f": {tail}" if tail else "")
                raise ThuncError(message) if self.connected else StartupError(message)

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        assert self.relay is not None
        for call, output, failed in results:
            assert call.id is not None
            self.relay.send(call.id, output, failed)

    def nudge(self, reply: Reply) -> None:
        self._turn(NUDGE)

    def close(self) -> None:
        """Stop Codex (and the relay it started), delete its session and remove the run's files."""
        self._stop()
        if self.relay is not None:
            self.relay.close()
        if self.thread is not None:  # saved so a turn could continue it; the run's record is thunc's own
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(
                    ["codex", "delete", "--force", self.thread],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=30,
                )
        if self.folder:
            shutil.rmtree(self.folder, ignore_errors=True)

    # --- turns ---

    def _turn(self, prompt: str) -> None:
        """Start a turn: the first one, or the next of the same session with `prompt` as its message."""
        if self.process is not None and self.thread is None:
            raise ThuncError("codex didn't say which session it started, so the run can't go on")
        self._stop()
        self.turn += 1
        self.text, self.problem = "", None
        try:
            self.process = self._popen(self._args())
        except OSError as exc:
            raise StartupError(f"`codex` couldn't be started: {exc.strerror or exc}") from None
        turn, process = self.turn, self.process
        threading.Thread(target=self._read_stdout, args=(turn, process), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(process,), daemon=True).start()
        try:
            assert process.stdin is not None
            with process.stdin:
                process.stdin.write(prompt.encode("utf-8", "backslashreplace").decode("utf-8"))
        except OSError:
            pass  # it stopped before reading the prompt: its exit says why

    def _args(self) -> list[str]:
        assert self.relay is not None and self.folder is not None
        server = self.relay.server
        env = ", ".join(f"{name} = {json.dumps(value)}" for name, value in server["env"].items())
        args = ["codex", "exec"]
        if self.thread is not None:
            args += ["resume", self.thread]
        args += [
            "--json",
            "--skip-git-repo-check",
            # As config, not --sandbox: `codex exec resume` doesn't take that option.
            "--config", 'sandbox_mode="read-only"',
            "--config", f"model_instructions_file={json.dumps(os.path.join(self.folder, 'system.md'))}",
            *_CODEX_QUIET,
            *_CODEX_LOCKDOWN,
            "--config", f"mcp_servers.thunc.command={json.dumps(server['command'])}",
            "--config", f"mcp_servers.thunc.args={json.dumps(server['args'])}",
            "--config", f"mcp_servers.thunc.env={{{env}}}",
            "--config", f"mcp_servers.thunc.tool_timeout_sec={math.ceil(self.tool_timeout)}",
            # Only thunc's tools, which check the agent's own permissions, run without asking.
            "--config", 'mcp_servers.thunc.default_tools_approval_mode="approve"',
        ]  # fmt: skip
        if self.model:
            args += ["--model", self.model]
        if self.effort:
            args += ["--config", f"model_reasoning_effort={json.dumps(self.effort)}"]
        args.append("-")  # the prompt, from stdin
        return args

    def _popen(self, args: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(
            args,
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            start_new_session=sys.platform != "win32",  # its own process group, so stopping it stops the relay
        )

    def _stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        with contextlib.suppress(OSError):
            if sys.platform == "win32":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)

    def _read_stdout(self, turn: int, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    self.events.put(("event", (turn, event)))
        self.events.put(("exit", (turn, process.wait())))

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        with process.stderr:
            for line in process.stderr:
                self.stderr.append(line)

    def _event(self, event: dict[str, Any]) -> str | None:
        """Take in one of Codex's events. Returns "completed" or "failed" when the turn ended."""
        kind = event.get("type")
        if kind == "thread.started" and isinstance(event.get("thread_id"), str):
            self.thread = event["thread_id"]
        elif kind == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                self.text = item["text"]
        elif kind == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                self.usage.update({k: v for k, v in usage.items() if isinstance(v, int)})
            return "completed"
        elif kind in ("turn.failed", "error"):
            self.problem = _codex_problem(event) or self.problem
            if kind == "turn.failed":
                return "failed"
        return None

    def _calls(self, first: dict[str, Any]) -> Reply:
        """The calls of one model reply: the first, and any that arrive with it."""
        assert self.relay is not None
        calls = [
            Call(ident, tool, arguments if isinstance(arguments, dict) else {})
            for ident, tool, arguments in self.relay.calls(first, self.events)
        ]
        return Reply(calls, _calls_text(calls))
