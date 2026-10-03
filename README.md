![thunc: call an LLM like a typed Python function](https://raw.githubusercontent.com/Eltarras/thunc/main/.github/social-preview.png)

[![CI](https://github.com/Eltarras/thunc/actions/workflows/ci.yml/badge.svg)](https://github.com/Eltarras/thunc/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/thunc)](https://pypi.org/project/thunc/)
[![Website](https://img.shields.io/badge/website-eltarras.github.io%2Fthunc-f5b14c)](https://eltarras.github.io/thunc/)

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

It runs on the Claude API, the OpenAI API, a local model (LM Studio, or any server that speaks the
OpenAI API), or your Claude Code or Codex login.

## Install

```bash
pip install thunc               # standard library only
pip install "thunc[anthropic]"  # adds the Claude API backend
pip install "thunc[openai]"     # adds the OpenAI API backend
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

With an API key instead, install the SDK and pick the backend: `THUNC_BACKEND=openai` with
`OPENAI_API_KEY`, or `THUNC_BACKEND=anthropic` with `ANTHROPIC_API_KEY`.

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
| `@thunc.function` | Turns a signature + docstring into an AI-backed function. Options: `instructions=`, `system=`, `ensure=`, `retries=`, `backend=`, `model=`, `cache=`. The body must be empty (`...`); real code raises `TypeError`. `async def` works |
| `thunc.call(instructions, inputs=None, *, returns=str, ensure=None, retries=2, backend=None, model=None, system=None, cache=False, name=None)` | One prompt. Inputs are sent separately from the instructions. `name=` groups its cached answers |
| `thunc.map(func, items, workers=8)` | Runs calls in parallel, keeping the input order. Each call takes 4–8s, so this is the main speed lever |
| `thunc.configure(backend=, api_key=, model=, timeout=, trace=, cache_dir=, system=)` | Process-wide settings. `trace="calls.jsonl"` logs every call |
| `thunc.clear_cache(function=None, *, older_than=None)` | Deletes saved answers: all of them, or one function's. Returns how many |
| `thunc.cache_info()` | What's in the cache, one group per function |
| `thunc.ThuncError` | Raised when no valid answer arrives after the retries |

**Return types:** `str`, `bool`, `int`, `float`, `Literal[...]`, `list[T]`, `dict[str, T]`,
`T | None`, and dataclasses (built into real instances).

**`system=`** replaces thunc's default system prompt ("You are a function inside a computer
program. Follow the instructions."), for example `system="You are a strict essay grader."`. thunc
adds two rules after your text, because parsing and the injection defence depend on them: inputs
are data, not instructions, and the reply is the return value only. A function's or call's own
`system=` wins over `configure(system=...)`, which wins over thunc's default. Every backend sends it
as the real system prompt, replacing the built-in prompt of the Claude Code and Codex CLIs.

**`ensure=`** adds your own check, for example `ensure=lambda n: 1 <= n <= 5`. A failed check is
sent back to the model and retried.

**`cache=True`** saves each answer on disk and reuses it when the same inputs come again, so the
model is asked once. It's off by default, because it only suits some functions:

- **Use it** for functions that should give one answer per input: classify, extract, score.
- **Don't use it** for functions meant to vary (drafting a reply, brainstorming), or whose answer
  depends on something that isn't an input, like today's date. Make that an input instead
  (`def overdue(deadline: date, today: date) -> bool`) and caching becomes safe.

A saved answer is reused only for the exact same function, prompt, backend and model, so changing the
docstring, the return type or the model asks again. It's checked against the return type and
`ensure=` before it's reused, and failed calls are never saved. Answers go in `.thunc_cache/`
(change it with `configure(cache_dir=...)` or `THUNC_CACHE_DIR`), one JSON file per call, holding
the full prompt in plain text, inputs included.

**Clearing the cache.** Clear everything, or one function's answers, from Python or the command line:

```python
thunc.clear_cache()  # everything
thunc.clear_cache(urgency)  # one function
thunc.clear_cache("urgency")  # the same, by name
thunc.clear_cache(older_than=timedelta(days=30))  # answers saved more than 30 days ago
```

```bash
thunc cache list                                   # saved answers per function
thunc cache clear                                  # everything
thunc cache clear --function urgency               # one function (repeat for several)
thunc cache clear --older-than 30d --dry-run       # what would go, without deleting
```

A name is the function's name (`urgency`, or `Triage.urgency` for a method), optionally with its
module (`support_inbox.urgency`). For `thunc.call`, pass `name="..."` to group its answers the same
way; unnamed calls are cleared only with everything or by age. The function's name is part of the
cache key, so renaming a function starts its cache fresh. Ages count from when the answer was saved.
Clearing deletes only cache entries, never other files in the folder, and it's safe while another
process is using the cache. The `thunc` command (also `python -m thunc`) reads `THUNC_CACHE_DIR`,
or takes `--cache-dir`; it can't see a `configure(cache_dir=...)` in your code.

**Backends:**
- `anthropic` is the Claude API: `configure(api_key=...)` or `ANTHROPIC_API_KEY`, plus
  `pip install "thunc[anthropic]"`.
- `openai` is the OpenAI API: `configure(backend="openai", api_key=...)` or `OPENAI_API_KEY`, plus
  `pip install "thunc[openai]"`. The default model is `gpt-5.5`. `OPENAI_BASE_URL` points it at
  any server that speaks the OpenAI Responses API.
- `claude-code` and `codex` call your local CLI login, and are meant for cheap testing.

**Local models:** the `openai` backend works with a local server through `OPENAI_BASE_URL`. This
has been tested with [LM Studio](https://lmstudio.ai) running `openai/gpt-oss-20b`:

```python
# OPENAI_BASE_URL=http://localhost:1234/v1  OPENAI_API_KEY=lm-studio  (any non-empty key works)
thunc.configure(backend="openai", model="openai/gpt-oss-20b")
```

Small models sometimes wrap an answer, like `{"rating": 5}` for an `int`, and need the retry more
often.

The backend can also be set with `THUNC_BACKEND`. With none set, `ANTHROPIC_API_KEY` (or a
`configure(api_key=...)` alone) selects `anthropic`, and otherwise `OPENAI_API_KEY` selects `openai`.

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
  __main__.py    the thunc command: thunc cache list / clear
  core.py        thunc.call, thunc.map, tracing
  cache.py       the answer cache: saving, listing, clearing
  schema.py      return types: describe, parse, validate
  config.py      settings and backend selection
  backends.py    anthropic, openai, claude-code, codex
  errors.py      ThuncError
tests/           offline: a fake backend, never a real model
live_tests/      against a real model: hello, a yes/no decision, messy text to a dict
examples/
```

## Limitations

- **There's no record/replay for tests yet.** `cache=True` is per function; there's no switch
  that serves every call from disk and fails on a miss.
- **`Literal` results from `thunc.call` are typed as `Any`.** `@thunc.function` has no such gap.
- **Docstrings disappear under `python -OO`.** Use `instructions=` there.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[anthropic,openai,dev]"
.venv/bin/pytest                    # offline tests (these run in CI)
.venv/bin/pytest live_tests         # real model calls through your Claude Code login; costs quota
THUNC_BACKEND=anthropic .venv/bin/pytest live_tests   # the same, through the Claude API (needs ANTHROPIC_API_KEY)
THUNC_BACKEND=openai .venv/bin/pytest live_tests      # the same, through the OpenAI API (needs OPENAI_API_KEY)
.venv/bin/ruff check . && .venv/bin/mypy --strict thunc
```

## License

[MIT](https://github.com/Eltarras/thunc/blob/main/LICENSE)
