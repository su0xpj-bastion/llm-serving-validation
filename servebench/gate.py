"""SLO gate: compare a benchmark summary with an SLO file.

SLO file (JSON):
  {
    "goodput_slo": {"ttft_s": 0.5, "tpot_s": 0.05},       # per-request SLO for goodput
    "max": {"ttft_s.p99": 1.0, "itl_s.p99": 0.1, "error_rate": 0.01},
    "min": {"output_throughput_tps": 100, "slo_attainment": 0.95}
  }
Keys under max/min are dotted paths into the summary dict.
"""
from __future__ import annotations

import json
from pathlib import Path

from servebench.metrics import GoodputSLO

ENCODING = "utf-8"


def load_slo(path: Path) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        slo = json.load(fh)
    unknown = set(slo) - {"goodput_slo", "max", "min", "description"}
    if unknown:
        raise ValueError(f"unknown SLO sections: {sorted(unknown)}")
    return slo


def goodput_slo(slo: dict) -> GoodputSLO | None:
    g = slo.get("goodput_slo")
    return GoodputSLO(g["ttft_s"], g["tpot_s"]) if g else None


def lookup(summary: dict, dotted: str) -> float:
    cur = summary
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"SLO key {dotted!r} not in summary")
        cur = cur[part]
    return float(cur)


def check(summary: dict, slo: dict) -> list[str]:
    """Breaches as human-readable strings. Empty list == pass."""
    out = []
    for key, limit in slo.get("max", {}).items():
        v = lookup(summary, key)
        if v > limit:
            out.append(f"{key} = {v:.4g} > max {limit:.4g}")
    for key, limit in slo.get("min", {}).items():
        v = lookup(summary, key)
        if v < limit:
            out.append(f"{key} = {v:.4g} < min {limit:.4g}")
    return out


def markdown(summary: dict, slo: dict | None, breaches: list[str]) -> str:
    rows = [
        ("requests (ok / total)", f"{summary['completed']} / {summary['requests']}"),
        ("error rate", f"{summary['error_rate']:.2%}"),
        ("request throughput", f"{summary['request_throughput_rps']:.2f} req/s"),
        ("output throughput", f"{summary['output_throughput_tps']:.0f} tok/s"),
    ]
    if "goodput_rps" in summary:
        rows.append(("goodput", f"{summary['goodput_rps']:.2f} req/s ({summary['slo_attainment']:.1%} met SLO)"))
    for name in ("ttft_s", "itl_s", "tpot_s", "e2e_s"):
        d = summary[name]
        rows.append((name.replace("_s", "").upper() + " p50 / p90 / p99",
                     f"{d['p50'] * 1e3:.1f} / {d['p90'] * 1e3:.1f} / {d['p99'] * 1e3:.1f} ms"))
    if "usage_mismatch" in summary:
        rows.append(("usage.completion_tokens != requested", str(summary["usage_mismatch"])))
    lines = ["| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    if slo is not None:
        verdict = "PASS" if not breaches else "FAIL"
        lines += ["", f"**SLO gate: {verdict}**"] + [f"- {b}" for b in breaches]
    return "\n".join(lines)
