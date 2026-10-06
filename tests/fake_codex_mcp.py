"""A stand-in for the `codex` CLI with thunc's relay as its MCP server, for tests/test_codex_agent.py.

`codex exec` starts the MCP server given with -c mcp_servers.thunc.* (thunc's relay), talks MCP to
it like Codex does, and plays a script instead of asking a model, printing Codex's JSON events.
Each `codex exec` (and `codex exec resume <id>`) plays the next turn of FAKE_CODEX_SCRIPT, a JSON
list of turns, each a list of steps:

    {"calls": [["read", {"path": "a.py"}], ...]}   one model reply with these tool calls
    {"text": "..."}                                 the turn ends with this message and no tool call
    {"fail": "..."}                                 the turn fails with this error
    {"exit": 3}                                     the CLI exits

After its last step a turn waits to be stopped, as when the model called finish. `codex delete`
is logged. What each invocation saw goes to FAKE_CODEX_LOG, one JSON object per line.

FAKE_CODEX_MODE plays a Codex that can't run thunc's tools: "old" rejects an option and exits;
"no-mcp" starts no MCP server, as when it can't load one, and ends its turn with text.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time


def log(entry):
    with open(os.environ["FAKE_CODEX_LOG"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def settings(args):
    found = {}
    for i, arg in enumerate(args):
        if arg == "--config":
            key, _, value = args[i + 1].partition("=")
            found[key] = value
    return found


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]
    if args[0] == "delete":
        log({"delete": args[1:]})
        return
    mode = os.environ.get("FAKE_CODEX_MODE")
    if mode == "old":
        sys.stderr.write("error: unexpected argument 'mcp_servers.thunc.default_tools_approval_mode' found\n")
        sys.exit(2)
    prompt = sys.stdin.read()
    config = settings(args)
    resume = args[2] if args[1] == "resume" else None
    with open(json.loads(config["model_instructions_file"]), encoding="utf-8") as f:
        system = f.read()
    state_file = os.environ["FAKE_CODEX_LOG"] + ".turn"
    turn = int(open(state_file).read()) if os.path.exists(state_file) else 0
    open(state_file, "w").write(str(turn + 1))
    entry = {"args": args, "cwd": os.getcwd(), "prompt": prompt, "system": system, "resume": resume, "results": []}
    thread = resume or "019a-fake-thread"
    emit({"type": "thread.started", "thread_id": thread})
    emit({"type": "turn.started"})
    if mode == "no-mcp":
        log(entry)
        emit({"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": "I have no tools."}})
        emit({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}})
        return
    env = {
        name: json.loads(value)
        for name, value in re.findall(r'(\w+) = ("(?:[^"\\]|\\.)*")', config["mcp_servers.thunc.env"])
    }
    server = subprocess.Popen(
        [json.loads(config["mcp_servers.thunc.command"]), *json.loads(config["mcp_servers.thunc.args"])],
        env={**os.environ, **env},
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
    rpc(2, "tools/list")
    entry["tools"] = wait(2)["result"]["tools"]
    log(entry)
    script = json.loads(os.environ["FAKE_CODEX_SCRIPT"])
    ident = 10
    for step in script[turn] if turn < len(script) else []:
        if "exit" in step:
            os._exit(step["exit"])
        if "calls" in step:
            idents = []
            for name, arguments in step["calls"]:
                ident += 1
                idents.append((ident, name, arguments))
                rpc(ident, "tools/call", {"name": name, "arguments": arguments})
            for one, name, arguments in idents:
                result = wait(one)["result"]
                text = result["content"][0]["text"]
                log({"turn": turn, "result": {"text": text, "error": result["isError"]}})
                item = {"id": f"c{one}", "type": "mcp_tool_call", "server": "thunc", "tool": name,
                        "arguments": arguments, "status": "completed"}  # fmt: skip
                emit({"type": "item.completed", "item": item})
        elif "text" in step:
            emit({"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": step["text"]}})
            emit({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 5}})
            return
        elif "fail" in step:
            emit({"type": "error", "message": step["fail"]})
            emit({"type": "turn.failed", "error": {"message": step["fail"]}})
            return
    time.sleep(60)  # the model called finish: wait to be stopped


if __name__ == "__main__":
    main()
