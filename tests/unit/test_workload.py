from __future__ import annotations

import math
import random

import pytest

from servebench import workload
from servebench.workload import TraceRequest, generate, prompt_ids, prompt_text
from servesim.model import golden


def _cv(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / m


def _gaps(times: list[float]) -> list[float]:
    return [b - a for a, b in zip([0.0] + times, times)]


def test_poisson_rate_and_cv():
    """Poisson arrivals: mean gap ~ 1/rate, coefficient of variation ~ 1."""
    gaps = _gaps(workload.poisson_arrivals(5000, 10.0, random.Random(1)))
    assert sum(gaps) / len(gaps) == pytest.approx(0.1, rel=0.05)
    assert _cv(gaps) == pytest.approx(1.0, abs=0.08)


def test_bursty_same_rate_higher_cv():
    """Bursty arrivals keep the mean rate but clump: CV = 1/sqrt(shape) = 2 at shape 0.25."""
    gaps = _gaps(workload.bursty_arrivals(5000, 10.0, random.Random(1), burstiness=0.25))
    assert sum(gaps) / len(gaps) == pytest.approx(0.1, rel=0.1)
    assert _cv(gaps) == pytest.approx(2.0, rel=0.15)


def test_lengths_are_sharegpt_like_and_clipped():
    t = generate(2000, 10, seed=3, max_prompt=1024, max_output=256)
    plens = sorted(r.prompt_tokens for r in t)
    olens = sorted(r.output_tokens for r in t)
    assert 150 < plens[len(plens) // 2] < 450          # median a few hundred
    assert plens[-1] <= 1024 and olens[-1] <= 256
    assert plens[int(0.95 * len(plens))] > 2 * plens[len(plens) // 2]   # long tail


def test_shared_prefix_ratio_knob():
    t = generate(2000, 10, seed=5, shared_prefix_ratio=0.6, prefix_groups=3, prefix_tokens=128)
    grouped = [r for r in t if r.prefix_group is not None]
    assert len(grouped) / len(t) == pytest.approx(0.6, abs=0.04)
    assert {r.prefix_group for r in grouped} == {0, 1, 2}
    assert all(r.prompt_tokens > 128 for r in grouped)


def test_same_group_shares_exact_prefix_tokens_and_text():
    a = TraceRequest("a", 0, 300, 5, prefix_group=1, prefix_tokens=128)
    b = TraceRequest("b", 0, 200, 5, prefix_group=1, prefix_tokens=128)
    assert prompt_ids(a)[:128] == prompt_ids(b)[:128]
    assert prompt_ids(a)[128:140] != prompt_ids(b)[128:140]
    assert len(prompt_ids(a)) == 300 and len(prompt_text(a).split()) == 300
    assert prompt_text(a).split()[:128] == prompt_text(b).split()[:128]


def test_generation_is_deterministic_and_round_trips(tmp_path):
    t1 = generate(50, 5, seed=9, arrival="bursty", shared_prefix_ratio=0.3)
    assert t1 == generate(50, 5, seed=9, arrival="bursty", shared_prefix_ratio=0.3)
    workload.save(t1, tmp_path / "t.jsonl")
    assert workload.load(tmp_path / "t.jsonl") == t1


def test_bad_knobs_rejected():
    with pytest.raises(ValueError):
        generate(5, 1, shared_prefix_ratio=1.5)
    with pytest.raises(ValueError):
        generate(5, 1, arrival="uniform")


def test_golden_model_depends_on_every_prefix_token():
    """The output must change if any earlier token changes, or KV corruption would go unseen."""
    p = list(range(100))
    base = golden(p, 8)
    assert golden(p, 8) == base
    for i in (0, 50, 99):
        q = list(p)
        q[i] += 1
        assert golden(q, 8) != base
