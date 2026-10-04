"""Run the benchmarks: uv run python -m benchmarks [-k FILTER]... [--save FILE] [--compare FILE] [--smoke]

Every model is a stand-in that answers at once (or after a fixed delay), so what's measured is
thunc's own work. See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from . import (  # noqa: F401
    bench_agent,
    bench_backends,
    bench_cache,
    bench_calls,
    bench_schema,
    bench_startup,
    bench_tools,
)
from .harness import REGISTRY, Result, load, run, table, to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description="thunc's benchmark suite")
    parser.add_argument("-k", dest="filters", action="append", metavar="TEXT", help="only names containing TEXT")
    parser.add_argument("--save", metavar="FILE", help="write the results as JSON, to compare later")
    parser.add_argument("--compare", metavar="FILE", help="show the change against results saved with --save")
    parser.add_argument("--smoke", action="store_true", help="run each benchmark once on small inputs")
    parser.add_argument("--list", action="store_true", help="list the benchmarks and what they measure")
    args = parser.parse_args(argv)

    chosen = [b for b in REGISTRY if not args.filters or any(f in b.name for f in args.filters)]
    if args.list:
        width = max(len(b.name) for b in chosen)
        for b in chosen:
            print(f"{b.name:<{width}}  {b.about}")
        return 0

    baseline = load(args.compare) if args.compare else None
    results: list[Result] = []
    for b in chosen:
        print(f"  {b.name} ...", end="\r", file=sys.stderr, flush=True)
        results.append(run(b, smoke=args.smoke))
        print(" " * (len(b.name) + 6), end="\r", file=sys.stderr)
    print(table(results, baseline))

    if args.save:
        meta = {"python": platform.python_version(), "machine": platform.platform(), "time": time.ctime()}
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": to_json(results)}, f, indent=2)
        print(f"\nSaved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
