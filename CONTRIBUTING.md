# Contributing to thunc

Thanks for helping. thunc is in beta, so feedback on the API is as useful as code.

## Where to start

- Issues labelled [`good first issue`](https://github.com/Eltarras/thunc/labels/good%20first%20issue)
  are small and self-contained. [`help wanted`](https://github.com/Eltarras/thunc/labels/help%20wanted)
  issues are bigger, or need a design discussion first.
- For anything beyond a small fix, comment on the issue or start a thread in
  [Ideas](https://github.com/Eltarras/thunc/discussions/categories/ideas) before writing code, so
  we agree on the approach first.
- Questions go in [Q&A](https://github.com/Eltarras/thunc/discussions/categories/q-a).

## Setup

```bash
git clone https://github.com/Eltarras/thunc && cd thunc
python3 -m venv .venv && .venv/bin/pip install -e ".[anthropic,openai,dev]"
```

The README's [Development](README.md#development) section lists the commands for tests and checks.

## Before you open a PR

CI runs these on Python 3.10 to 3.14, and all must pass:

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy --strict thunc
```

## Ground rules

- **Standard library only in the core.** A provider SDK is an optional extra in `pyproject.toml`,
  imported inside its backend function, never at module level.
- **Tests in `tests/` never call a real model.** Use the `fake` fixture in `tests/conftest.py`,
  which returns scripted replies and records the prompts. Real calls belong in `live_tests/`,
  which CI doesn't run.
- **User data stays out of the instructions.** Inputs are sent separately from the prompt
  (see `_build_prompt` in `thunc/core.py`). Don't add code paths that paste inputs into
  instructions.
- **Failures are loud.** When no valid answer arrives, raise `ThuncError`; never return a
  default value.
- **Update the README** when you change the public API or the supported return types.
- **One change per PR**, with a description of what it does and how you tested it.

## Adding a backend

1. Write a function in `thunc/backends.py` with the same signature as the others:
   `(text, *, system, model, api_key, timeout) -> str`. It should raise `ThuncError` on
   connection errors, refusals and cut-off answers.
2. Register it in `BACKENDS`, and in the selection logic in `thunc/config.py` if needed.
3. Add an optional extra for its SDK in `pyproject.toml`, and add the extra to the CI install.
4. Add offline tests with a stubbed SDK module, like the existing ones in `tests/test_backends.py`.
5. Run `THUNC_BACKEND=<name> .venv/bin/pytest live_tests` against the real service, and say in
   the PR that you did.

## Known issues

Found while testing how thunc handles replies that don't follow the contract. Some are left alone
on purpose; the rest are open. Check here before reporting one, and comment on the issue (or open
one) before working on an open item.

**Retried rather than read, on purpose.** Each of these has a reading that could be wrong, so the
model is asked again instead of thunc guessing:

- Quoted numbers (`"4"` for an `int`), labels in the wrong case (`Bug` for `bug`), prose around
  JSON, Python-style values (`['a']`, `None`), trailing commas, curly quotes, non-ASCII digits.
- Unquoted text for `str | None`, and exclamations like `Yes!` for a `bool`.
- Several `<think>` blocks in a row, several lines of prose before a fence, `~~~`, indented or
  four-backtick fences.
- A one-key object whose key is a field of the expected dataclass: `{"customer": {...}}` for an
  `Order` with a `customer` field is an `Order` with a bad customer, not a wrapper.
- A reply after the closing fence (for example a reasoning paragraph): it could be a second answer.

**Returned as they are, on purpose.** For a `str` return type, a preamble ("Sure! Here's…"),
refusal text, a code fence or JSON quotes come back as part of the answer: with plain text there's
no reliable way to tell them from a real one. Only an empty reply is retried.

**Open: behavior that might change.**

- `thunc.map` loses every result when one item fails, and returns coroutines for `async` functions.
- Backend errors (timeouts, connection failures) are never retried; only bad replies are.
- A union takes the first option that accepts the reply as JSON, in the order written: `3` for
  `float | int` is `3.0`. An unquoted label is only tried after that, so `2` for
  `Literal["2"] | int` is the int `2`, not the label.
- Unknown fields in a dataclass object are ignored as long as one known field is present, and a
  dataclass with no fields accepts any object.
- A float written as a whole number is read as an `int` below 2**53, even when the float only
  approximates what was written (`3.9999999999999999` is `4`).
- The previous reply is pasted into the retry prompt as is (only lone surrogates are escaped), so a
  reply containing `</instructions>` lands in the prompt unchanged.
- A trace file that can't be written raises an error, while the cache only warns.
- The cache key doesn't include `OPENAI_BASE_URL`: two servers with the same model name share entries.
- The trace records `"model": null` when the backend's default model is used.

**Open: not about replies.**

- A dataclass that refers to itself makes `describe()` recurse forever.
- `returns=None` written literally is reported as unsupported (`-> None` works).
- `dict[int, X]` returns string keys.
- `Literal[math.inf]` is described as `Infinity`, which isn't valid JSON, so it can't be read back.
- A list of dataclasses passed as an input is shown to the model as Python reprs, not JSON.
- A string return annotation naming a class defined after the function raises `NameError` when
  the decorator runs.
- `make_dataclass` classes, and `Annotated` fields on Python 3.10, aren't supported.
- The Python 3.10 fallback for string annotations (`_hints` and `resolve_strings` in
  `thunc/schema.py`) can be removed when 3.10 support is dropped.

## Reporting bugs

Open an [issue](https://github.com/Eltarras/thunc/issues) with:

- the smallest code that shows the problem
- the backend
- your thunc and Python versions
- if you can, the relevant lines from `thunc.configure(trace="calls.jsonl")`

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
