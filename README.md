# thunc

[![CI](https://github.com/Eltarras/thunc/actions/workflows/ci.yml/badge.svg)](https://github.com/Eltarras/thunc/actions/workflows/ci.yml)

**think + function.** Call an LLM like a typed Python function.

> **Status: beta (v0.1).** Expect bugs; the API may change. Feedback and issues are welcome.

```python
import thunc

thunc.configure(backend="claude-code")


@thunc.function
def urgency(ticket: str) -> int:
    """Rate how urgent this ticket is, from 1 (can wait) to 5 (customer is blocked)."""
    ...


urgency("I was charged twice!")  # -> 4, a checked int
```

The answer is parsed into the declared type. If it doesn't fit, the model is asked again, and
after that `thunc.ThuncError` is raised. The library uses the standard library only and needs
Python 3.10+.

## Install

```bash
pip install thunc               # standard library only
pip install "thunc[anthropic]"  # adds the Claude API backend
```

## Try it

Clone the repo and run the examples from its root. No install is needed; the examples run through
your local [Claude Code](https://claude.com/claude-code) login:

```bash
git clone https://github.com/Eltarras/thunc && cd thunc
python3 -m examples.hello
python3 -m examples.support_inbox
THUNC_BACKEND=codex python3 -m examples.log_triage
```

## Two ways to write a prompt

| | When | |
|---|---|---|
| `@thunc.function` | The prompt is fixed and should read like code | The docstring is the prompt, the parameters are the inputs, the return annotation is the type |
| `thunc.call(...)` | The prompt is built in code (from config, in a loop, loaded from a file) | `thunc.call(f"Translate into {lang}.", {"text": note})` |

`@thunc.function(instructions=some_string)` combines the two: a typed, reusable function whose
prompt is generated.

**Keep user data out of the instructions.** Your own text can go in the instructions string.
Anything from users, files or the web goes in the inputs:

- `@thunc.function` does this automatically.
- With `thunc.call` it's up to you. In a live test, a hostile email pasted in with an f-string
  tricked the model 3 out of 3 times. Passed as an input, it failed 3 out of 3 times.

## API

| | |
|---|---|
| `@thunc.function` | Turns a signature + docstring into an AI-backed function. Options: `instructions=`, `ensure=`, `retries=`, `backend=`, `model=`. The body must be empty (`...`); real code raises `TypeError`. `async def` works |
| `thunc.call(instructions, inputs=None, *, returns=str, ensure=None, retries=2, backend=None, model=None)` | One prompt. Inputs are sent separately from the instructions |
| `thunc.map(func, items, workers=8)` | Runs calls in parallel, keeping the input order. Each call takes 4–8s, so this is the main speed lever |
| `thunc.configure(backend=, api_key=, model=, timeout=, trace=)` | Process-wide settings. `trace="calls.jsonl"` logs every call |
| `thunc.ThuncError` | Raised when no valid answer arrives after the retries |

**Return types:** `str`, `bool`, `int`, `float`, `Literal[...]`, `list[T]`, `dict[str, T]`,
`T | None`, and dataclasses (built into real instances).

**`ensure=`** adds your own check, for example `ensure=lambda n: 1 <= n <= 5`. A failed check is
sent back to the model and retried.

**Backends:**
- `anthropic` is the Claude API: `configure(api_key=...)` or `ANTHROPIC_API_KEY`, plus
  `pip install "thunc[anthropic]"`.
- `claude-code` and `codex` call your local CLI login, and are meant for cheap testing.

The backend can also be set with `THUNC_BACKEND`.

**Type checking:** signatures and return types are visible to mypy and Pyright. mypy reports
empty bodies; turn that off with `disable_error_code = ["empty-body"]`.

## Examples

| | |
|---|---|
| [hello.py](https://github.com/Eltarras/thunc/blob/main/examples/hello.py) | The smallest call |
| [support_inbox.py](https://github.com/Eltarras/thunc/blob/main/examples/support_inbox.py) | Docstring functions returning a `Literal`, an `int` with `ensure=`, a dataclass, and a reply; tickets processed in parallel |
| [dynamic_prompts.py](https://github.com/Eltarras/thunc/blob/main/examples/dynamic_prompts.py) | Prompts built from a style guide with `thunc.call`, and a grading function generated from a rubric |
| [log_triage.py](https://github.com/Eltarras/thunc/blob/main/examples/log_triage.py) | Plain Python and AI functions mixed, with tracing |

## Code

```
thunc/
  __init__.py    public API
  decorator.py   @thunc.function
  core.py        thunc.call, thunc.map, tracing
  schema.py      return types: describe, parse, validate
  config.py      settings and backend selection
  backends.py    anthropic, claude-code, codex
  errors.py      ThuncError
tests/           offline: a fake backend, never a real model
live_tests/      against a real model: hello, a yes/no decision, messy text to a dict
examples/
```

## Limitations

- **There's no caching and no record/replay yet**, so repeated calls cost again.
- **The API-key backend hasn't been run live yet.** It's only checked against the SDK's types.
- **`Literal` results from `thunc.call` are typed as `Any`.** `@thunc.function` has no such gap.
- **Docstrings disappear under `python -OO`.** Use `instructions=` there.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[anthropic,dev]"
.venv/bin/pytest                    # offline tests (these run in CI)
.venv/bin/pytest live_tests         # real model calls through your Claude Code login; costs quota
.venv/bin/ruff check . && .venv/bin/mypy --strict thunc
```

## License

[MIT](https://github.com/Eltarras/thunc/blob/main/LICENSE)
