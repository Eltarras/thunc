"""Backends: each takes (text, *, system, model, api_key, timeout) and returns the answer text.

- anthropic:   Claude API via the official SDK and an API key (production path).
- openai:      OpenAI Responses API via the official SDK and an API key.
- claude-code: headless `claude -p` using the local Claude Code login (cheap testing).
- codex:       headless `codex exec` using the local Codex login (cheap testing).

The API backends send `system` as the API's system prompt. The CLIs replace their own built-in
prompt with it: `claude -p --system-prompt`, and Codex's `model_instructions_file` setting.

Typed backends answer a typed question instead of writing text, so they take the instructions,
inputs and return type separately: (instructions, inputs, returns, *, system, timeout).
They return the answer as JSON text, so parsing, ensure=, the cache and the trace work as usual.

- jev:         TypeSafe's Jev judgment model via `jev ask`. Answers bool and Literal[...] only.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast, get_args, get_origin

from .errors import ThuncError, TransientError, transient_status
from .schema import describe

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
# The model a backend uses when none is given, where thunc decides it (the claude and codex CLIs
# pick their own). Jev's can't be changed: the CLI always uses jev-latest.
DEFAULT_MODELS = {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL, "jev": "jev-latest"}
# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_JEV_MAX_CHOICES = 255  # the most options a Jev choice question takes


_clients: dict[tuple[Any, ...], Any] = {}
_clients_lock = threading.Lock()
_MAX_CLIENTS = 16  # different keys, sdk_options or environments in one process; past this, start over


C = TypeVar("C")


def sdk_client(factory: Callable[..., C], env_prefix: str, api_key: str | None, timeout: float) -> C:
    """The SDK client for the Claude or OpenAI API, shared by every call in this process, so calls
    reuse its connections instead of opening a new one (a TCP and TLS handshake) each time. The SDKs'
    clients are thread-safe, so thunc.map's threads share it too.

    A client is kept per everything it reads when it's made: the api_key, sdk_options, and the
    environment variables starting with `env_prefix` (the API key, the base URL, ...), so a change
    to any of them gets a new client. And per process, as a connection can't be shared across fork.
    The timeout is applied per request with with_options(), which keeps the same connections."""
    from .config import setting

    options = setting("sdk_options") or {}
    environment = tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith(env_prefix)))
    key = (factory, os.getpid(), api_key, repr(sorted(options.items())), environment)
    with _clients_lock:
        client: Any = _clients.get(key)
        if client is None:
            if len(_clients) >= _MAX_CLIENTS:
                _clients.clear()
            client = _clients[key] = factory(api_key=api_key, **options)
    return cast(C, client.with_options(timeout=timeout))


def anthropic_api(
    text: str, *, system: str, model: str | None, api_key: str | None, timeout: float, effort: str | None = None
) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise ThuncError("The anthropic backend needs the SDK: pip install 'thunc[anthropic]'") from exc

    model = model or DEFAULT_ANTHROPIC_MODEL
    # api_key=None lets the SDK resolve ANTHROPIC_API_KEY or an `ant auth login` profile.
    client = sdk_client(anthropic.Anthropic, "ANTHROPIC_", api_key, timeout)
    output_config: Any = {"effort": effort} if effort else anthropic.omit
    try:
        if model in _FALLBACK_MODELS:
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": text}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config=output_config,
            )
        else:
            response = client.messages.create(  # type: ignore[assignment]  # Message vs BetaMessage: same fields used below
                model=model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": text}],
                output_config=output_config,
            )
    except anthropic.APIConnectionError as exc:
        raise TransientError(f"Could not reach the Claude API: {exc}") from exc
    except anthropic.APIStatusError as exc:
        error = TransientError if transient_status(exc.status_code) else ThuncError
        raise error(f"Claude API error {exc.status_code}: {exc.message}") from exc

    if response.stop_reason == "refusal":
        raise ThuncError("The model declined this request.")
    if response.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        raise ThuncError(f"The answer was cut off ({response.stop_reason}).")
    if response.stop_reason not in ("end_turn", "stop_sequence", None):
        raise ThuncError(f"The model stopped before finishing its answer ({response.stop_reason}).")
    return "".join(block.text for block in response.content if block.type == "text")


def openai_api(
    text: str, *, system: str, model: str | None, api_key: str | None, timeout: float, effort: str | None = None
) -> str:
    try:
        import openai
    except ImportError as exc:
        raise ThuncError("The openai backend needs the SDK: pip install 'thunc[openai]'") from exc

    if effort == "max":
        raise ThuncError("The openai backend takes effort low, medium, high or xhigh, not max.")
    reasoning: Any = {"effort": effort} if effort else openai.omit
    # api_key=None lets the SDK resolve OPENAI_API_KEY (and OPENAI_BASE_URL for compatible servers).
    client = sdk_client(openai.OpenAI, "OPENAI_", api_key, timeout)
    try:
        response = client.responses.create(
            model=model or DEFAULT_OPENAI_MODEL,
            instructions=system,
            input=text,
            store=False,
            reasoning=reasoning,
        )
    except openai.APIConnectionError as exc:
        raise TransientError(f"Could not reach the OpenAI API: {exc}") from exc
    except openai.APIStatusError as exc:
        error = TransientError if transient_status(exc.status_code) else ThuncError
        raise error(f"OpenAI API error {exc.status_code}: {exc.message}") from exc

    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        if reason == "max_output_tokens":
            raise ThuncError("The answer was cut off at max_output_tokens.")
        raise ThuncError(f"The OpenAI response is incomplete ({reason or 'no reason given'}).")
    if response.status not in ("completed", None):  # failed, cancelled, ...: no finished answer
        raise ThuncError(f"The OpenAI response is {response.status}, not completed.")
    for item in response.output:
        if item.type == "message" and any(part.type == "refusal" for part in item.content):
            raise ThuncError("The model declined this request.")
    return response.output_text


def _run_cli(args: list[str], text: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a CLI with `text` on stdin. Output bytes that aren't UTF-8 come back as lone surrogates
    (surrogateescape) rather than failing here: see _not_utf8."""
    exe = args[0]
    if shutil.which(exe) is None:
        raise ThuncError(f"`{exe}` was not found on PATH.")
    try:
        # A neutral cwd keeps the CLI from picking up project files (CLAUDE.md, AGENTS.md, ...).
        return subprocess.run(
            args,
            input=text.encode("utf-8", "backslashreplace").decode("utf-8"),  # no lone surrogates on stdin
            capture_output=True,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"`{exe}` timed out after {timeout:.0f}s.") from exc


Events = list[tuple[dict[str, Any], str]]  # (event, the line it was printed as)


def _cli_events(
    args: list[str], text: str, timeout: float, last: str, stop: Callable[[dict[str, Any]], bool] | None = None
) -> tuple[Events, int | None, str]:
    """Run a CLI that prints one JSON event per line, with `text` on stdin, and return its events
    (each with the line it came from) up to the first of type `last`, as soon as that one arrives.
    What the CLI does after it (codex takes about 0.4 s to shut down) finishes in the background: a
    thread reads the rest of its output and collects the process. If the CLI ends without a `last`
    event, returns all its events with its exit code and stderr; the exit code is None otherwise.
    `stop(event)` true ends it sooner: the CLI is stopped, and the events so far are returned."""
    exe = args[0]
    if shutil.which(exe) is None:
        raise ThuncError(f"`{exe}` was not found on PATH.")
    stderr = tempfile.TemporaryFile()  # a file, not a pipe: the CLI never blocks on a full stderr
    try:
        # A neutral cwd keeps the CLI from picking up project files (CLAUDE.md, AGENTS.md, ...).
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            encoding="utf-8",
            errors="surrogateescape",
            cwd=tempfile.gettempdir(),
        )
    except BaseException:
        stderr.close()
        raise
    lines: queue.Queue[str | None] = queue.Queue()

    def read() -> None:
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                lines.put(line)
        lines.put(None)
        process.wait()  # collected even when nobody is waiting for the answer any more

    threading.Thread(target=read, name=f"thunc-{exe}", daemon=True).start()
    events: Events = []
    try:
        try:
            assert process.stdin is not None
            with process.stdin:
                process.stdin.write(text.encode("utf-8", "backslashreplace").decode("utf-8"))  # no lone surrogates
        except BrokenPipeError:
            pass  # it stopped before reading the prompt: its exit code and stderr say why
        deadline = time.monotonic() + timeout
        while (line := lines.get(timeout=max(deadline - time.monotonic(), 0))) is not None:
            try:
                event = json.loads(line)
            except (ValueError, RecursionError):
                continue  # not an event: progress text, a warning
            if isinstance(event, dict):
                events.append((event, line))
                if event.get("type") == last:
                    return events, None, ""
                if stop is not None and stop(event):
                    process.kill()  # what it would say next isn't wanted, nor paid for
                    return events, None, ""
        code = process.wait()
        stderr.seek(0)
        return events, code, stderr.read().decode("utf-8", "surrogateescape")
    except queue.Empty:
        process.kill()
        raise TransientError(f"`{exe}` timed out after {timeout:.0f}s.") from None
    except BaseException:  # Ctrl-C too: don't leave the CLI running
        process.kill()
        raise
    finally:
        stderr.close()  # the CLI keeps its own handle until it exits


def claude_code(
    text: str,
    *,
    system: str,
    model: str | None,
    api_key: str | None,
    timeout: float,
    effort: str | None = None,
    until: Callable[[str], bool] | None = None,
) -> str:
    """`claude -p` with no tools. With `until` (an agent's text-protocol step), the reply is streamed
    and the CLI is stopped as soon as `until(reply so far)` is true: a model that carries on past
    its action, making up the tool's result, can't run on to the timeout."""
    args = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",  # plain answer only; no file or shell access
        "--strict-mcp-config",  # no MCP servers
        "--setting-sources",
        "",  # no CLAUDE.md, settings or hooks from the user's Claude Code setup
        "--system-prompt",
        system,  # replaces the large default coding prompt
        "--no-session-persistence",
    ]
    if model:
        args += ["--model", model]
    if effort:
        args += ["--effort", effort]
    if until is not None:
        return _claude_streamed(args, text, timeout, until)
    proc = _run_cli(args, text, timeout)
    if _not_utf8(proc.stdout):
        raise ThuncError(f"claude printed output that isn't UTF-8: {_printable(proc.stdout)[-500:]}")
    try:
        data = json.loads(proc.stdout)
    except (ValueError, RecursionError):
        data = None
    if not isinstance(data, dict):  # not the JSON object `--output-format json` prints
        raise ThuncError(f"claude exited {proc.returncode}: {_printable(proc.stderr or proc.stdout).strip()[-500:]}")
    if data.get("is_error") or proc.returncode != 0:  # e.g. a tool call it couldn't parse: asking again may work
        raise TransientError(f"claude error: {data.get('result') or _printable(proc.stderr).strip()[-500:]}")
    if not isinstance(data.get("result"), str):
        raise ThuncError(f"claude returned no text: {proc.stdout.strip()[-500:]}")
    return str(data["result"])


def _claude_streamed(args: list[str], text: str, timeout: float, until: Callable[[str], bool]) -> str:
    at = args.index("--output-format")
    args = [*args[: at + 1], "stream-json", "--verbose", "--include-partial-messages", *args[at + 2 :]]
    reply: list[str] = []

    def done(event: dict[str, Any]) -> bool:
        inner = event.get("event") if event.get("type") == "stream_event" else None
        delta = inner.get("delta") if isinstance(inner, dict) and inner.get("type") == "content_block_delta" else None
        if not isinstance(delta, dict) or delta.get("type") != "text_delta" or not isinstance(delta.get("text"), str):
            return False
        reply.append(delta["text"])
        # Only a closing brace or bracket can complete an action, so the reply is checked then.
        return ("}" in delta["text"] or "]" in delta["text"]) and until("".join(reply))

    events, code, stderr = _cli_events(args, text, timeout, "result", stop=done)
    bad = next((line for _, line in events if _not_utf8(line)), None)
    if bad is not None:
        raise ThuncError(f"claude printed output that isn't UTF-8: {_printable(bad)[-500:]}")
    result = next((event for event, _ in reversed(events) if event.get("type") == "result"), None)
    if result is None:
        if code is None:  # stopped once the action was complete
            return "".join(reply)
        raise ThuncError(f"claude exited {code}: {_printable(stderr).strip()[-500:]}")
    if result.get("is_error"):  # e.g. a tool call it couldn't parse: asking again may work
        raise TransientError(f"claude error: {result.get('result') or _printable(stderr).strip()[-500:]}")
    if not isinstance(result.get("result"), str):
        raise ThuncError(f"claude returned no text: {_printable(json.dumps(result))[-500:]}")
    return str(result["result"])


# Notes Codex adds to every request besides thunc's prompt. The permissions note says the sandbox is
# read-only, which led agents to refuse edits they were allowed to make with thunc's own write tools;
# the others describe its working directory (a temp folder) and its collaboration and apps features.
_CODEX_QUIET = tuple(
    arg
    for setting in (
        "include_permissions_instructions",
        "include_environment_context",
        "include_collaboration_mode_instructions",
        "include_apps_instructions",
    )
    for arg in ("--config", f"{setting}=false")
)


# Codex's own tools and the user's Codex setup stay out of thunc's calls, as `--tools ""` and
# --strict-mcp-config do for claude: no shell (exec_command, write_stdin), no images, plugins, apps,
# hooks, sub-agents or web search, and no MCP servers, model settings or notify command from
# ~/.codex/config.toml (login still comes from CODEX_HOME). What remains is apply_patch, which the
# read-only sandbox stops from writing, request_user_input and a clock; Codex can't turn those off.
_CODEX_LOCKDOWN = (
    "--ignore-user-config",
    "--ignore-rules",
    *(
        flag
        for feature in (
            "shell_tool",
            "unified_exec",
            "view_image",
            "multi_agent",
            "apps",
            "plugins",
            "skill_search",
            "hooks",
            "browser_use",
            "computer_use",
            "image_generation",
        )
        for flag in ("--disable", feature)
    ),
    "--config",
    'web_search="disabled"',
)


def codex(
    text: str, *, system: str, model: str | None, api_key: str | None, timeout: float, effort: str | None = None
) -> str:
    if effort == "max":
        raise ThuncError("The codex backend takes effort low, medium, high or xhigh, not max.")
    # codex exec has no system-prompt flag; the model_instructions_file setting replaces Codex's
    # built-in instructions with the file's text. The path is absolute because the CLI runs in a temp dir.
    fd, system_path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(system)
    try:
        args = [
            "codex",
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
            "--json",  # events as they happen, so the answer is read without waiting for codex to exit
            "--config",
            f"model_instructions_file={json.dumps(system_path)}",  # a TOML string (JSON escapes are valid TOML)
            *_CODEX_QUIET,
            *_CODEX_LOCKDOWN,
        ]
        if model:
            args += ["--model", model]
        if effort:
            args += ["--config", f"model_reasoning_effort={json.dumps(effort)}"]
        args.append("-")  # read the prompt from stdin
        events, code, stderr = _cli_events(args, text, timeout, "turn.completed")
    finally:
        with contextlib.suppress(OSError):  # gone, or not a file any more: nothing to clean up
            os.unlink(system_path)  # read when codex started
    if code is not None:  # it ended without finishing the turn
        problems = [_codex_problem(event) for event, _ in events]
        problem = next((p for p in reversed(problems) if p), None) or stderr
        raise TransientError(f"codex exited {code}: {_printable(problem).strip()[-500:]}")
    messages = [
        (event["item"]["text"], line)
        for event, line in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "agent_message"
        and isinstance(event["item"].get("text"), str)
    ]
    if not messages:
        raise ThuncError("codex finished its turn without an answer.")
    answer, line = messages[-1]  # the last message is the answer, as with --output-last-message
    if _not_utf8(line):
        raise ThuncError(f"codex printed an answer that isn't UTF-8: {_printable(answer)[-500:]}")
    return str(answer).strip()


def _codex_problem(event: dict[str, Any]) -> str | None:
    """The message of a codex `error` or `turn.failed` event."""
    if event.get("type") == "error" and isinstance(event.get("message"), str):
        return str(event["message"])
    error = event.get("error")
    if event.get("type") == "turn.failed" and isinstance(error, dict) and isinstance(error.get("message"), str):
        return str(error["message"])
    return None


def jev(instructions: str, inputs: dict[str, Any], returns: Any, *, system: str | None, timeout: float) -> str:
    """One Jev question: the inputs are its state, the instructions its question. The answer comes
    back as JSON text: true when a yes is at least as likely as not, the chosen label, or the most
    likely level. The CLI always uses jev-latest, and finds the key itself: JEV_API_KEY or `jev login`.
    configure(api_key=...) isn't passed on, since it's usually the key of another backend."""
    kind, labels = _jev_question(returns)
    if system and system.strip():  # context from the program; thunc's own default is written for text models
        instructions = f"{system.strip()}\n\n{instructions}"
    question: dict[str, Any] = {"type": kind, "instructions": instructions}
    if kind == "choice":  # option -> description
        question["criteria"] = {label: label for label in labels}
    elif kind == "score":  # level descriptions, lowest first
        question["criteria"] = [str(level) for level in labels]
    request = json.dumps({"state": inputs, "questions": {"answer": question}}, ensure_ascii=False, default=str)
    proc = _run_cli(["jev", "ask", "-"], request, timeout)
    if proc.returncode != 0:
        raise ThuncError(f"jev exited {proc.returncode}: {_printable(proc.stderr or proc.stdout).strip()[-500:]}")
    try:
        answer = json.loads(proc.stdout)["answers"]["answer"]
        value = answer["probabilities"] if kind == "score" else answer[kind]
    except (ValueError, RecursionError, KeyError, TypeError) as exc:
        raise ThuncError(f"jev returned no answer: {_printable(proc.stdout).strip()[-500:]}") from exc
    shown = _printable(json.dumps(value))[:200]
    if kind == "noul":  # {"noul": 0.98}: the probability of a yes
        if not _probability(value):
            raise ThuncError(f"jev returned {shown}, not a probability.")
        return json.dumps(value >= 0.5)
    if kind == "choice":  # {"choice": "billing", "probabilities": {...}}
        if value not in labels:
            raise ThuncError(f"jev chose {shown}, not one of {list(labels)}.")
        return json.dumps(value, ensure_ascii=False)
    # {"score": 3.99, "probabilities": {"0": 0.0, ..., "4": 1.0}}: "score" is the expected position,
    # so it can fall between two likely levels; the most likely level comes from the probabilities.
    positions = [str(position) for position in range(len(labels))]
    if not isinstance(value, dict) or sorted(value) != sorted(positions) or not all(map(_probability, value.values())):
        raise ThuncError(f"jev returned {shown}, not probabilities for the positions {positions}.")
    return json.dumps(labels[int(max(positions, key=value.__getitem__))])


def _jev_question(returns: Any) -> tuple[str, tuple[Any, ...]]:
    """The Jev question for a return type: noul for bool, choice for string literals, score for
    integer literals (levels in ascending order). Jev can't write text, so nothing else."""
    if returns is bool:
        return "noul", ()
    if get_origin(returns) is Literal:
        options = get_args(returns)
        if all(isinstance(option, str) for option in options) and len(options) <= _JEV_MAX_CHOICES:
            return "choice", options
        if all(isinstance(option, int) and not isinstance(option, bool) for option in options):
            return "score", tuple(sorted(options))
    raise ThuncError(
        f"The jev backend answers bool, Literal of strings (up to {_JEV_MAX_CHOICES}) or Literal of "
        f"integers, not {describe(returns)}."
    )


def _probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _not_utf8(output: str) -> bool:
    """Whether CLI output had bytes that aren't UTF-8 (surrogateescape turns them into U+DC80-DCFF).
    A surrogate the CLI wrote as a JSON escape ("\\ud83d") is plain ASCII here, so it doesn't count."""
    return re.search("[\udc80-\udcff]", output) is not None


def _printable(output: str) -> str:
    return output.encode("utf-8", "backslashreplace").decode("utf-8")


BACKENDS: dict[str, Callable[..., str]] = {
    "anthropic": anthropic_api,
    "openai": openai_api,
    "claude-code": claude_code,
    "codex": codex,
}

# Typed backends: see the module docstring.
TYPED_BACKENDS: dict[str, Callable[..., str]] = {
    "jev": jev,
}
