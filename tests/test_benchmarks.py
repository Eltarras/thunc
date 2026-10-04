"""The benchmark suite still runs: each benchmark once, on small inputs (python -m benchmarks --smoke)."""

import pytest

from benchmarks.__main__ import REGISTRY
from benchmarks.harness import run


@pytest.mark.parametrize("bench", REGISTRY, ids=lambda b: b.name)
def test_benchmark_runs(bench):
    result = run(bench, smoke=True)
    assert result.error is None or result.error.startswith("skipped: ")
    assert result.error or result.per_op > 0
