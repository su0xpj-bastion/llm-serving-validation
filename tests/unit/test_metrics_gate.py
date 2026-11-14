from __future__ import annotations

import json

import pytest

from servebench import gate
from servebench.metrics import GoodputSLO, RequestRecord, aggregate, percentile


def test_percentile_nearest_rank():
    xs = [float(i) for i in range(1, 101)]
    assert percentile(xs, 50) == 50 and percentile(xs, 99) == 99 and percentile(xs, 100) == 100
    assert percentile([], 99) == 0.0
    with pytest.raises(ValueError):
        percentile(xs, 0)


def test_request_metrics_definitions():
    r = RequestRecord("a", arrival=1.0, token_times=[1.2, 1.25, 1.35, 1.40], output_tokens=4)
    assert r.ttft == pytest.approx(0.2)
    assert r.itls == pytest.approx([0.05, 0.10, 0.05])
    assert r.tpot == pytest.approx(0.2 / 3)
    assert r.e2e == pytest.approx(0.4)


def test_tpot_uses_usage_tokens_not_chunks():
    """Servers may pack several tokens per SSE chunk; TPOT divides by usage tokens."""
    r = RequestRecord("a", 0.0, token_times=[0.1, 0.2, 0.3], output_tokens=9)
    assert r.tpot == pytest.approx(0.2 / 8)


def test_goodput_counts_only_requests_meeting_both_slos():
    recs = [
        RequestRecord("ok", 0.0, [0.1, 0.12, 0.14], 3),
        RequestRecord("slow_ttft", 0.0, [0.9, 0.92], 2),
        RequestRecord("slow_tpot", 0.0, [0.1, 0.5], 2),
        RequestRecord("err", 0.0, [], 0, ok=False, error="http_500"),
    ]
    s = aggregate(recs, GoodputSLO(ttft_s=0.5, tpot_s=0.05))
    assert s["completed"] == 3 and s["errors"] == 1 and s["error_rate"] == 0.25
    assert s["slo_attainment"] == 0.25
    assert s["goodput_rps"] == pytest.approx(1 / 0.92)


def test_gate_max_min_and_unknown_keys(tmp_path):
    summary = {"ttft_s": {"p99": 0.8}, "error_rate": 0.0, "output_throughput_tps": 50.0}
    slo = {"max": {"ttft_s.p99": 0.5, "error_rate": 0.01}, "min": {"output_throughput_tps": 100}}
    breaches = gate.check(summary, slo)
    assert len(breaches) == 2 and "ttft_s.p99" in breaches[0] and "output_throughput_tps" in breaches[1]
    with pytest.raises(KeyError):
        gate.check(summary, {"max": {"itl_s.p99": 1}})
    p = tmp_path / "slo.json"
    p.write_text(json.dumps({"maxx": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        gate.load_slo(p)
