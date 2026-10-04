"""A stand-in for the `claude` CLI in stream-json mode, for tests/test_claude_code_agent.py.

It starts the MCP server named in --mcp-config (thunc's relay) and talks MCP to it like Claude Code
does, then plays a script instead of asking a model. FAKE_CLAUDE_SCRIPT is a JSON list of turns:

    {"calls": [["read", {"path": "a.py"}], ...]}   one model reply with these tool calls (sent at once,
                                                    or one after another with "serial": true)
    {"text": "..."}                                 a turn that ends with text and no tool call
    {"error": "..."}                                a turn that ends in an error
    {"exit": 3}                                     the CLI exits

A turn after a text or error turn waits for the next user message, as the real CLI does. What it
saw (arguments, system prompt, tools, user messages, tool results) goes to FAKE_CLAUDE_LOG as JSON.

FAKE_CLAUDE_MODE plays a CLI that can't run thunc's tools: "old" rejects an option and exits, as a
CLI too old for the options would; "no-mcp" starts without loading any MCP server, as when a policy
turns them off, and ends its turn with text.
"""

import json
import os
import subprocess
import sys
import threading
import time


def main():
    args = sys.argv[1:]
    mode = os.environ.get("FAKE_CLAUDE_MODE")
    if mode == "old":
        sys.stderr.write("error: unknown option '--setting-sources'\n")
        sys.exit(1)
    if mode == "no-mcp":
        sys.stdout.write(json.dumps({"type": "system", "subtype": "init", "mcp_servers": []}) + "\n")
        sys.stdout.flush()
        sys.stdin.readline()
        result = {"type": "result", "subtype": "success", "is_error": False, "result": "I have no tools."}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
        time.sleep(60)
    log = {"args": args, "cwd": os.getcwd(), "messages": [], "results": []}
    path = os.environ["FAKE_CLAUDE_LOG"]

    def save():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f)

    with open(args[args.index("--system-prompt-file") + 1], encoding="utf-8") as f:
        log["system"] = f.read()
    config = json.loads(args[args.index("--mcp-config") + 1])["mcpServers"]["thunc"]
    server = subprocess.Popen(
        [config["command"], *config["args"]],
        env={**os.environ, **config.get("env", {})},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    answers, lock = {}, threading.Condition()

    def rpc(ident, method, params=None):
        message = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}}
        with lock:
            server.stdin.write(json.dumps(message) + "\n")
            server.stdin.flush()

    def read_server():
        for line in server.stdout:
            reply = json.loads(line)
            with lock:
                answers[reply["id"]] = reply
                lock.notify_all()

    def wait(ident):
        with lock:
            while ident not in answers:
                lock.wait()
            return answers.pop(ident)

    threading.Thread(target=read_server, daemon=True).start()
    rpc(1, "initialize", {"protocolVersion": "2025-06-18"})
    wait(1)
    server.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    server.stdin.flush()
    rpc(2, "tools/list")
    log["tools"] = wait(2)["result"]["tools"]
    save()

    def emit(event):
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

    emit({"type": "system", "subtype": "init", "mcp_servers": [{"name": "thunc", "status": "connected"}]})
    log["messages"].append(json.loads(sys.stdin.readline())["message"]["content"])
    save()
    ident = 10
    for n, turn in enumerate(json.loads(os.environ["FAKE_CLAUDE_SCRIPT"])):
        if "exit" in turn:
            save()
            os._exit(turn["exit"])
        if "calls" in turn:
            blocks = [{"type": "tool_use", "id": f"t{n}-{i}", "name": "mcp__thunc__" + name, "input": arguments}
                      for i, (name, arguments) in enumerate(turn["calls"])]  # fmt: skip
            emit({"type": "assistant", "message": {"id": f"m{n}", "content": blocks}})
            idents = []
            for name, arguments in turn["calls"]:
                ident += 1
                idents.append(ident)
                rpc(ident, "tools/call", {"name": "mcp__thunc__" + name, "arguments": arguments})
                if turn.get("serial"):
                    wait_for = [idents.pop()]
                else:
                    continue
                for one in wait_for:
                    result = wait(one)["result"]
                    log["results"].append({"text": result["content"][0]["text"], "error": result["isError"]})
                    save()
            for one in idents:
                result = wait(one)["result"]
                log["results"].append({"text": result["content"][0]["text"], "error": result["isError"]})
                save()
            continue
        if "text" in turn:
            emit({"type": "result", "subtype": "success", "is_error": False, "result": turn["text"],
                  "total_cost_usd": 0.01, "usage": {"input_tokens": 5, "output_tokens": 2}})  # fmt: skip
        else:
            emit({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": turn["error"]})
        log["messages"].append(json.loads(sys.stdin.readline())["message"]["content"])
        save()
    save()
    time.sleep(60)  # the script is over: wait to be stopped, like a CLI waiting for its next message


if __name__ == "__main__":
    main()
