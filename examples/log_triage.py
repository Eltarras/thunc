"""Plain Python and AI functions together: triage the errors in a service log, with tracing on.

Run from the repo root:  python3 -m examples.log_triage
"""

import json
import os
import re
import tempfile
from collections import Counter
from typing import Literal

import thunc

trace_file = os.path.join(tempfile.gettempdir(), "thunc_log_triage.jsonl")
if os.path.exists(trace_file):
    os.remove(trace_file)
thunc.configure(backend=os.environ.get("THUNC_BACKEND", "claude-code"), trace=trace_file)

LOG = """\
2026-10-03 09:12:01 INFO  api    request GET /health 200
2026-10-03 09:12:04 ERROR api    psycopg.OperationalError: connection to server at "10.0.3.7", port 5432 failed: timeout
2026-10-03 09:12:09 ERROR api    psycopg.OperationalError: connection to server at "10.0.3.7", port 5432 failed: timeout
2026-10-03 09:13:30 ERROR worker KeyError: 'customer_id' in invoices/render.py line 88
2026-10-03 09:15:47 ERROR api    ssl.SSLCertVerificationError: certificate has expired (hostname='payments.example.com')
2026-10-03 09:16:12 ERROR worker KeyError: 'customer_id' in invoices/render.py line 88
"""


@thunc.function
def owner(error: str) -> Literal["database", "application-code", "certificates", "network", "unknown"]:
    """Which area most likely needs to act on this production error?"""
    ...


@thunc.function
def first_step(error: str, owner: str) -> str:
    """Suggest the single most useful first debugging step for this error, in one sentence."""
    ...


# Plain Python does the cheap part: find and count the unique errors.
errors = Counter(m[1] for m in re.finditer(r"ERROR \w+\s+(.*)", LOG))


def analyse(error: str) -> tuple[str, str]:
    area = owner(error)
    return area, first_step(error, area)


for error, (area, step) in zip(errors, thunc.map(analyse, list(errors)), strict=True):
    print(f"{errors[error]}x [{area}] {error[:70]}\n     → {step}")

# The trace has one JSON line per call: inputs, raw answers, attempts, timing.
calls = [json.loads(line) for line in open(trace_file, encoding="utf-8")]
retried = sum(c["attempts"] > 1 for c in calls)
print(
    f"\n{len(calls)} calls traced to {trace_file}; {retried} needed a retry; "
    f"slowest {max(c['seconds'] for c in calls):.1f}s"
)
