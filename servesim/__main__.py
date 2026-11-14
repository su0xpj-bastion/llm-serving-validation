"""servesim CLI.

  python -m servesim qualify --version 1.5.0-rc2      # invariant scenarios; exit 1 on any violation
  python -m servesim experiments [--md out.md]        # performance-behaviour table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ENCODING = "utf-8"


def cmd_qualify(a: argparse.Namespace) -> int:
    from servesim.scenarios import qualify
    from servesim.versions import bugs_for
    bugs_for(a.version)
    results = qualify(a.version)
    bad = 0
    for name, violations in results.items():
        status = "ok" if not violations else "FAIL"
        kinds = sorted({v.kind for v in violations})
        print(f"{status:4}  {name:34} {', '.join(kinds)}")
        for v in violations[: a.show]:
            print(f"        {v.kind}: {v.detail}")
        bad += bool(violations)
    verdict = "GO" if not bad else "NO-GO"
    print(f"\n{a.version}: {verdict} ({bad}/{len(results)} scenarios violated invariants)")
    return 0 if not bad else 1


def cmd_experiments(a: argparse.Namespace) -> int:
    from servesim.experiments import EXPERIMENTS, markdown
    names = a.only or list(EXPERIMENTS)
    table = markdown({n: EXPERIMENTS[n]() for n in names})
    print(table)
    if a.md:
        with Path(a.md).open("w", encoding=ENCODING) as fh:
            fh.write("### Serving-engine simulator: performance behaviour\n\n" + table + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="servesim")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualify", help="run the invariant scenarios against an engine version")
    q.add_argument("--version", default="1.5.0")
    q.add_argument("--show", type=int, default=2, help="violations to print per scenario")
    e = sub.add_parser("experiments", help="print the performance-behaviour table")
    e.add_argument("--only", nargs="*")
    e.add_argument("--md")
    a = p.parse_args(argv)
    return {"qualify": cmd_qualify, "experiments": cmd_experiments}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
