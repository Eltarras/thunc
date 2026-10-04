"""The MCP server Claude Code starts for an agent run: it relays tool calls to the thunc process.

Run as a script by the claude CLI (see claude_code.py), never imported by thunc. It speaks MCP
(JSON-RPC, one message per line) on stdin and stdout, and forwards each tools/call over a local
TCP connection to the agent run, which carries the call out with its own tools and permissions and
sends back the result. Standard library only, and no thunc imports, so it runs from any directory.

Environment: THUNC_RELAY is "<port> <token>": the run's listening port on 127.0.0.1 and the token
it expects first. THUNC_RELAY_TOOLS is a JSON file with the tools to list.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading

_write_lock = threading.Lock()


def _send(stream: object, message: dict[str, object]) -> None:
    line = json.dumps(message, ensure_ascii=False) + "\n"
    with _write_lock:
        stream.write(line)  # type: ignore[attr-defined]
        stream.flush()  # type: ignore[attr-defined]


def main() -> None:
    port, token = os.environ["THUNC_RELAY"].split()
    with open(os.environ["THUNC_RELAY_TOOLS"], encoding="utf-8") as f:
        tools = json.load(f)
    connection = socket.create_connection(("127.0.0.1", int(port)))
    run_in = connection.makefile("r", encoding="utf-8")
    run_out = connection.makefile("w", encoding="utf-8")
    _send(run_out, {"token": token})
    out = sys.stdout

    def answers() -> None:  # results from the run, as MCP responses
        for line in run_in:
            reply = json.loads(line)
            result = {"content": [{"type": "text", "text": reply["text"]}], "isError": bool(reply["error"])}
            _send(out, {"jsonrpc": "2.0", "id": reply["id"], "result": result})
        os._exit(0)  # the run is over

    threading.Thread(target=answers, daemon=True).start()
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method, ident = message.get("method"), message.get("id")
        if ident is None:
            continue  # a notification
        if method == "initialize":
            version = (message.get("params") or {}).get("protocolVersion") or "2025-06-18"
            info = {"name": "thunc", "version": "1"}
            _send(out, {"jsonrpc": "2.0", "id": ident, "result": {
                "protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": info,
            }})  # fmt: skip
        elif method == "tools/list":
            _send(out, {"jsonrpc": "2.0", "id": ident, "result": {"tools": tools}})
        elif method == "tools/call":
            params = message.get("params") or {}
            call = {"id": ident, "name": params.get("name", ""), "arguments": params.get("arguments") or {}}
            _send(run_out, call)  # carried out by the run; its answer comes back in answers()
        elif method == "ping":
            _send(out, {"jsonrpc": "2.0", "id": ident, "result": {}})
        else:
            error = {"code": -32601, "message": f"method not found: {method}"}
            _send(out, {"jsonrpc": "2.0", "id": ident, "error": error})
    os._exit(0)  # the CLI closed stdin


if __name__ == "__main__":
    main()
