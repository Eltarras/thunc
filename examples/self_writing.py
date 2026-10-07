"""A function that writes itself.  Run from the repo root:  python3 -m examples.self_writing

The function in DURATIONS is saved as examples/scratch/durations.py (a folder git ignores) and
called there. On its first call, the model writes its body into that file, thunc checks it against
the model's own answers, and the call runs the new code; the program then prints the file. Each run
starts from a fresh copy. Your own functions are written where they are: add write=True and call one.
"""

import importlib
import os
import sys
import time

import thunc

thunc.configure(backend=os.environ.get("THUNC_BACKEND", "claude-code"))

DURATIONS = '''\
"""Written by examples/self_writing.py, and then by thunc."""

import thunc


@thunc.function(write=True)
def minutes(duration: str) -> int:
    """Convert a duration like '1h 30m', '90 min' or '2 hours' to whole minutes."""
    ...
'''

scratch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch")
os.makedirs(scratch, exist_ok=True)
with open(os.path.join(scratch, ".gitignore"), "w") as f:
    f.write("*\n")
path = os.path.join(scratch, "durations.py")
with open(path, "w") as f:
    f.write(DURATIONS)
sys.path.insert(0, scratch)
durations = importlib.import_module("durations")

started = time.monotonic()
print(f"minutes('1h 30m') = {durations.minutes('1h 30m')}  ({time.monotonic() - started:.1f}s: written, then run)")
started = time.monotonic()
print(f"minutes('2 hours') = {durations.minutes('2 hours')}  ({time.monotonic() - started:.4f}s: plain Python)")
print(f"\n--- {os.path.relpath(path)} ---")
with open(path) as f:
    print(f.read())
