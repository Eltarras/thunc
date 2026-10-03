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

## Reporting bugs

Open an [issue](https://github.com/Eltarras/thunc/issues) with:

- the smallest code that shows the problem
- the backend
- your thunc and Python versions
- if you can, the relevant lines from `thunc.configure(trace="calls.jsonl")`

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
