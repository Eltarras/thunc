# Benchmarks

thunc's own cost, measured with stand-in models that answer at once (or after a fixed delay), so
the numbers are thunc's work, not a model's. No dependencies beyond thunc; the backend benchmarks
use the `anthropic` SDK when it's installed and are skipped otherwise.

```bash
uv run python -m benchmarks                        # everything, about a minute
uv run python -m benchmarks -k tools -k agent      # only names containing "tools" or "agent"
uv run python -m benchmarks --list                 # what each one measures
uv run python -m benchmarks --save before.json     # keep the results...
uv run python -m benchmarks --compare before.json  # ...and show the change after a fix
uv run python -m benchmarks --smoke                # each once on small inputs (tests/test_benchmarks.py)
```

Each benchmark is timed in batches of about 0.2 s, five times; PER OP is the median and NOISE is
half the spread between the fastest and slowest batch. Compare runs on the same machine, plugged
in, with little else running: differences under about 10% are usually noise.

| Group | What it covers |
|---|---|
| `call.*` | `thunc.call` and `@thunc.function`: prompt, parse, retry, trace, cache hit and miss, big inputs |
| `map.*` | `thunc.map` over a model with 20 ms of latency, against the ideal time |
| `parse.*`, `describe.*` | turning replies into values, and return types into prompt text |
| `cache.*` | `thunc cache list` and `clear --function` over 5,000 entries |
| `tools.*` | the agent's list, read, search and run on a 2,000-file repository |
| `agent.*` | whole text-protocol runs; `sent_per_run` is the prompt text sent to the model per run |
| `backend.*` | the anthropic backend against a local stand-in for the Claude API; `connections` is how many TCP connections the requests opened. The `_handshake60ms` variants add 60 ms per new connection to stand in for TCP and TLS setup, which localhost doesn't have |
| `startup.*` | `import thunc` and the `thunc` command in a fresh interpreter |

To add one, write a generator in a `bench_*.py` module: set up, `yield` the operation to time,
tear down after the yield. See `harness.py`.
