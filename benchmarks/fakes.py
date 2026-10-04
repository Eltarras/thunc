"""Stand-ins for the model: a backend with scripted replies, and a local Claude Messages API."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import thunc
from thunc import backends


class Backend:
    """A backend that answers with `reply(prompt)`, after `latency` seconds. Counts what it's sent."""

    def __init__(self, reply: Callable[[str], str] | str, latency: float = 0.0) -> None:
        self.reply = reply if callable(reply) else (lambda _: reply)
        self.latency = latency
        self.requests = 0
        self.bytes_sent = 0  # prompt + system prompt characters, as a proxy for input tokens
        self._lock = threading.Lock()

    def __call__(self, text: str, **kwargs: Any) -> str:
        with self._lock:
            self.requests += 1
            self.bytes_sent += len(text) + len(kwargs.get("system") or "")
        if self.latency:
            time.sleep(self.latency)
        return self.reply(text)


def use(backend: Backend, name: str = "bench") -> Backend:
    backends.BACKENDS[name] = backend
    thunc.configure(backend=name)
    return backend


class MessagesAPI:
    """A local HTTP server that answers POST /v1/messages like the Claude API, after `latency` seconds.
    `connections` counts the TCP connections clients opened: a client that keeps its connection
    alive opens one; a client per call opens one per call. `handshake` delays each new connection's
    first answer, to stand in for the TCP and TLS round trips a real connection costs."""

    def __init__(self, answer: str = "4", latency: float = 0.0, handshake: float = 0.0) -> None:
        self.answer, self.latency, self.handshake = answer, latency, handshake
        self.connections = 0
        self.requests = 0
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, as the real API

            def setup(self) -> None:
                super().setup()
                api.connections += 1
                if api.handshake:
                    time.sleep(api.handshake)

            def do_POST(self) -> None:  # noqa: N802 (http.server's naming)
                self.rfile.read(int(self.headers.get("content-length", 0)))
                api.requests += 1
                if api.latency:
                    time.sleep(api.latency)
                body = json.dumps(
                    {
                        "id": "msg_bench",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-bench",
                        "content": [{"type": "text", "text": api.answer}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    @contextlib.contextmanager
    def running(self) -> Iterator[MessagesAPI]:
        thread = threading.Thread(target=self.server.serve_forever, args=(0.02,), daemon=True)
        thread.start()
        try:
            yield self
        finally:
            self.server.shutdown()
            self.server.server_close()
