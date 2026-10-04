"""Temporal tests are optional; ordinary tests remain dependency-free at runtime."""

import pytest

pytest.importorskip("temporalio")
