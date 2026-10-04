"""Backends: each takes (text, *, system, model, api_key, timeout) and returns the answer text.

- anthropic:   Claude API via the official SDK and an API key (production path).
- openai:      OpenAI Responses API via the official SDK and an API key.
- claude-code: headless `claude -p` using the local Claude Code login (cheap testing).
- codex:       headless `codex exec` using the local Codex login (cheap testing).

The API backends send `system` as the API's system prompt. The CLIs replace their own built-in
prompt with it: `claude -p --system-prompt`, and Codex's `model_instructions_file` setting.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable

from .errors import ThuncError

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
# The model a backend uses when none is given, where thunc decides it (the CLIs pick their own).
DEFAULT_MODELS = {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL}
# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def anthropic_api(text: str, *, system: str, model: str | None, api_key: str | None, timeout: float) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise ThuncError("The anthropic backend needs the SDK: pip install 'thunc[anthropic]'") from exc

    model = model or DEFAULT_ANTHROPIC_MODEL
    # api_key=None lets the SDK resolve ANTHROPIC_API_KEY or an `ant auth login` profile.
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    try:
        if model in _FALLBACK_MODELS:
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": text}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        else:
            response = client.messages.create(  # type: ignore[assignment]  # Message vs BetaMessage: same fields used below
                model=model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
    except anthropic.APIConnectionError as exc:
        raise ThuncError(f"Could not reach the Claude API: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise ThuncError(f"Claude API error {exc.status_code}: {exc.message}") from exc

    if response.stop_reason == "refusal":
        raise ThuncError("The model declined this request.")
    if response.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        raise ThuncError(f"The answer was cut off ({response.stop_reason}).")
    if response.stop_reason not in ("end_turn", "stop_sequence", None):
        raise ThuncError(f"The model stopped before finishing its answer ({response.stop_reason}).")
    return "".join(block.text for block in response.content if block.type == "text")


def openai_api(text: str, *, system: str, model: str | None, api_key: str | None, timeout: float) -> str:
    try:
        import openai
    except ImportError as exc:
        raise ThuncError("The openai backend needs the SDK: pip install 'thunc[openai]'") from exc

    # api_key=None lets the SDK resolve OPENAI_API_KEY (and OPENAI_BASE_URL for compatible servers).
    client = openai.OpenAI(api_key=api_key, timeout=timeout)
    try:
        response = client.responses.create(
            model=model or DEFAULT_OPENAI_MODEL,
            instructions=system,
            input=text,
            store=False,
        )
    except openai.APIConnectionError as exc:
        raise ThuncError(f"Could not reach the OpenAI API: {exc}") from exc
    except openai.APIStatusError as exc:
        raise ThuncError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc

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
        raise ThuncError(f"`{exe}` timed out after {timeout:.0f}s.") from exc


def claude_code(text: str, *, system: str, model: str | None, api_key: str | None, timeout: float) -> str:
    args = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",  # plain answer only; no file or shell access
        "--strict-mcp-config",  # no MCP servers
        "--system-prompt",
        system,  # replaces the large default coding prompt
        "--no-session-persistence",
    ]
    if model:
        args += ["--model", model]
    proc = _run_cli(args, text, timeout)
    if _not_utf8(proc.stdout):
        raise ThuncError(f"claude printed output that isn't UTF-8: {_printable(proc.stdout)[-500:]}")
    try:
        data = json.loads(proc.stdout)
    except (ValueError, RecursionError):
        data = None
    if not isinstance(data, dict):  # not the JSON object `--output-format json` prints
        raise ThuncError(f"claude exited {proc.returncode}: {_printable(proc.stderr or proc.stdout).strip()[-500:]}")
    if data.get("is_error") or proc.returncode != 0:
        raise ThuncError(f"claude error: {data.get('result') or _printable(proc.stderr).strip()[-500:]}")
    if not isinstance(data.get("result"), str):
        raise ThuncError(f"claude returned no text: {proc.stdout.strip()[-500:]}")
    return str(data["result"])


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


def codex(text: str, *, system: str, model: str | None, api_key: str | None, timeout: float) -> str:
    # codex exec has no system-prompt flag; the model_instructions_file setting replaces Codex's
    # built-in instructions with the file's text. The path is absolute because the CLI runs in a temp dir.
    fd, system_path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(system)
    fd, out_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
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
            "--output-last-message",
            out_path,
            "--config",
            f"model_instructions_file={json.dumps(system_path)}",  # a TOML string (JSON escapes are valid TOML)
            *_CODEX_QUIET,
        ]
        if model:
            args += ["--model", model]
        args.append("-")  # read the prompt from stdin
        proc = _run_cli(args, text, timeout)
        if proc.returncode != 0:
            raise ThuncError(f"codex exited {proc.returncode}: {_printable(proc.stderr).strip()[-500:]}")
        try:
            with open(out_path, encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ThuncError(f"Could not read codex's answer: {exc}") from exc
    finally:
        for path in (out_path, system_path):
            with contextlib.suppress(OSError):  # gone, or not a file any more: nothing to clean up
                os.unlink(path)


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
