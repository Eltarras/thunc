"""Native tool calls on the claude-code backend: the agent's tools as an MCP server for Claude Code.

The text protocol asks the model to write each action as JSON in its reply. Current models are
trained to call tools natively, so on the CLI they drift back to it: they invent a tool's result
and carry on, emit native tool-call markup the CLI then refuses, or wrap the JSON in prose. Here the
model gets real tool calls instead:

- One `claude -p` process per run, in stream-json mode, with Claude Code's own tools turned off and
  thunc's system prompt (from a file, so its size doesn't hit the command-line limit).
- Its only tools are an MCP server, mcp_relay.py, which Claude Code starts. The relay connects back
  to this process (see relay.py) and forwards each tool call, so the agent loop carries it out with
  its usual tools, permissions and records, and sends the result back.
- The CLI keeps the conversation, its caching and its retries of the API. A turn that ends without
  a tool call gets a nudge, and a turn that ends in an error gets a second chance, as a new user
  message to the same process.

The CLI runs in the agent's working directory, so the environment details it adds to the prompt
name the folder the tools work in. It loads no setting sources (--setting-sources ""): otherwise
Claude Code would read CLAUDE.md from that folder as instructions, and the user's own settings and
hooks, while thunc only follows instruction files the program asks for with follow=.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from .errors import ThuncError
from .native import NUDGE, Call, Reply, Tool, _calls_text
from .relay import Relay, StartupError

PREFIX = "mcp__thunc__"  # how Claude Code names the relay's tools
TURN_ERRORS = 2  # CLI turns that may end in an error before the run fails


class ClaudeCodeConversation:
    """A Conversation (see native.py) whose model replies and tool calls go through Claude Code."""

    def __init__(
        self,
        system: str,
        request: str,
        tools: Sequence[Tool],
        model: str | None,
        workdir: str,
        timeout: float,
        effort: str | None = None,
        *,
        session: str | None = None,
        resume: bool = False,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """`session` keeps the conversation in Claude Code's saved session with this id (a durable run
        restores it after a crash), continued with --resume when `resume`; then `request` is the
        message that starts the next turn. `cancelled()` true stops waiting for the model."""
        if shutil.which("claude") is None:
            raise ThuncError("`claude` was not found on PATH.")
        self.system, self.request, self.tools, self.model = system, request, list(tools), model
        self.effort = effort
        self.session, self.resume, self.cancelled = session, resume, cancelled
        self.workdir = workdir
        self.timeout = timeout  # seconds to wait for the model's next step
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.folder: str | None = None
        self.relay: Relay | None = None
        self.tool_uses: list[tuple[str, str, Any]] = []  # (message id, tool, input) not yet matched to a call
        self.last_message: str | None = None
        self.turn_errors = 0
        self.stderr: collections.deque[str] = collections.deque(maxlen=50)
        self.stderr_reader: threading.Thread | None = None
        self.replies: dict[str, dict[str, int]] = {}  # model reply id -> its token usage, as the CLI printed it
        self._write = threading.Lock()

    @property
    def connected(self) -> bool:
        """Whether the relay is connected: thunc's tools are available to the model."""
        return self.relay is not None and self.relay.connected.is_set()

    # --- the Conversation protocol ---

    def next(self) -> Reply:
        if self.process is None:
            self._start()
        deadline = time.monotonic() + self.timeout
        while True:
            left = deadline - time.monotonic()
            try:
                kind, event = self.events.get(timeout=max(min(left, 1.0), 0.001))
            except queue.Empty:
                if self.cancelled is not None and self.cancelled():
                    raise ThuncError("the run was cancelled") from None
                if time.monotonic() < deadline:
                    continue
                if not self.connected:
                    raise StartupError(f"thunc's tool server didn't connect within {self.timeout:g}s") from None
                raise ThuncError(f"claude took no step within {self.timeout:g}s") from None
            if kind == "call":
                return self._calls(event)
            if kind == "result":
                if not self.connected:  # a turn without thunc's tools: they never started
                    raise StartupError("the model's turn ended before thunc's tool server connected")
                if event.get("is_error") or event.get("subtype") != "success":
                    self.turn_errors += 1
                    problem = str(event.get("result") or event.get("subtype") or "an error")[:500]
                    if self.turn_errors > TURN_ERRORS:
                        raise ThuncError(f"claude error: {problem}")
                    self._say(f"That step failed ({problem}). Carry on with the task with your tools.")
                    deadline = time.monotonic() + self.timeout
                    continue
                return Reply([], str(event.get("result") or ""), "no tool call")
            if kind == "exit":
                if self.stderr_reader is not None:
                    self.stderr_reader.join(timeout=2)  # what it printed before exiting says why
                tail = "".join(self.stderr).strip()[-500:]
                message = f"claude exited {event} before the task was done" + (f": {tail}" if tail else "")
                raise ThuncError(message) if self.connected else StartupError(message)
            if kind == "relay":
                raise StartupError(f"claude couldn't start thunc's tools: {event}")

    def results(self, results: Sequence[tuple[Call, str, bool]]) -> None:
        assert self.relay is not None
        for call, output, failed in results:
            assert call.id is not None
            self.relay.send(call.id, output, failed)

    def nudge(self, reply: Reply) -> None:
        self._say(NUDGE)

    def close(self) -> None:
        """Stop the CLI (and the relay it started) and remove the run's files."""
        if self.process is not None and self.process.poll() is None:
            with contextlib.suppress(OSError):
                if sys.platform == "win32":
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=10)
        if self.process is not None and self.process.stdin is not None:
            with contextlib.suppress(OSError), self._write:
                self.process.stdin.close()  # the CLI read each message from it
        if self.relay is not None:
            self.relay.close()
        if self.folder:
            shutil.rmtree(self.folder, ignore_errors=True)

    # --- starting the CLI ---

    def _start(self) -> None:
        self.folder = tempfile.mkdtemp(prefix="thunc-claude-")
        system_file = os.path.join(self.folder, "system.md")
        with open(system_file, "w", encoding="utf-8") as f:
            f.write(self.system)
        self.relay = Relay(self.folder, self.tools, self.events, self.timeout)
        relay = self.relay.server
        args = [
            "claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--system-prompt-file", system_file, "--tools", "",
            "--mcp-config", json.dumps({"mcpServers": {"thunc": relay}}), "--strict-mcp-config",
            "--allowedTools", *(PREFIX + t.name for t in self.tools),
            "--setting-sources", "",  # no CLAUDE.md, settings or hooks: the workdir can't instruct the agent
        ]  # fmt: skip
        if self.session is None:
            args.append("--no-session-persistence")
        else:  # saved, so a durable run can continue it after a crash
            args += ["--resume" if self.resume else "--session-id", self.session]
        if self.model:
            args += ["--model", self.model]
        if self.effort:
            args += ["--effort", self.effort]
        try:
            self.process = self._popen(args)
        except OSError as exc:
            raise StartupError(f"`claude` couldn't be started: {exc.strerror or exc}") from None
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self.stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_reader.start()
        self._say(self.request)

    def _popen(self, args: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(
            args,
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            start_new_session=sys.platform != "win32",  # its own process group, so close() stops the relay too
        )

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        with self.process.stdout:
            self._read_events(self.process.stdout)
        self.events.put(("exit", self.process.wait()))

    def _read_events(self, stdout: Any) -> None:
        for line in stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "assistant":
                message = event.get("message") or {}
                usage = message.get("usage")
                if isinstance(usage, dict):  # printed with each part of a reply; the last is complete
                    self.replies[str(message.get("id"))] = {k: v for k, v in usage.items() if isinstance(v, int)}
                for block in message.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        self.tool_uses.append((str(message.get("id")), str(block.get("name")), block.get("input")))
            elif kind == "result":
                self.events.put(("result", event))
            elif kind == "system" and event.get("subtype") == "init":
                listed = [s for s in event.get("mcp_servers") or [] if isinstance(s, dict)]
                servers = {s.get("name"): s.get("status") for s in listed}
                if "thunc" not in servers:
                    self.events.put(("relay", "Claude Code didn't load it (are MCP servers turned off by a policy?)"))
                elif servers["thunc"] not in ("connected", "pending"):
                    self.events.put(("relay", f"the tool server is {servers['thunc']}"))

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        with self.process.stderr:
            for line in self.process.stderr:
                self.stderr.append(line)

    # --- helpers ---

    def _calls(self, first: dict[str, Any]) -> Reply:
        """The calls of one model reply, and whether they continue the reply the last ones came from."""
        assert self.relay is not None
        calls, messages = [], set()
        for ident, tool, arguments in self.relay.calls(first, self.events):
            name = tool.removeprefix(PREFIX)
            calls.append(Call(ident, name, arguments if isinstance(arguments, dict) else {}))
            messages.add(self._message_of(name, arguments))
        same_turn = self.last_message is not None and messages == {self.last_message}
        if len(messages) == 1 and None not in messages:
            self.last_message = next(iter(messages))
        else:
            self.last_message = None
        return Reply(calls, _calls_text(calls), same_turn=same_turn)

    def _message_of(self, name: str, arguments: Any) -> str | None:
        """Which model reply a call came from, matched against the tool uses the CLI printed."""
        for _ in range(20):  # the CLI may print the reply a moment after sending its calls
            for i, (message, tool, given) in enumerate(self.tool_uses):
                if tool == PREFIX + name and given == arguments:
                    del self.tool_uses[i]
                    return message
            time.sleep(0.01)
        return None

    @property
    def usage(self) -> collections.Counter[str]:
        """Tokens the run's model replies used, added up (input, output, cache reads and writes)."""
        total: collections.Counter[str] = collections.Counter()
        for usage in self.replies.values():
            total.update(usage)
        return total

    def _say(self, text: str) -> None:
        """A user message to the CLI, which starts its next turn."""
        assert self.process is not None and self.process.stdin is not None
        message = {"type": "user", "message": {"role": "user", "content": text}}
        try:
            with self._write:
                self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
        except OSError as exc:
            raise ThuncError(f"claude stopped accepting messages: {exc}") from None
