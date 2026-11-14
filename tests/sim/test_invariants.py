"""Engine invariants on the clean release, across the scenario matrix."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from servesim.cluster import Cluster, LinkModel, SimRequest
from servesim.engine import EngineConfig
from servesim.invariants import check_run
from servesim.model import golden
from servesim.scenarios import SCENARIOS, rag_reordered, shared_prefix_pressure
from servesim.versions import LATEST


@pytest.fixture(scope="module")
def runs():
    return {s.name: s.run(LATEST) for s in SCENARIOS}


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_scenario_holds_every_invariant(runs, name):
    """No leaks, ref counts consistent every step, budgets respected, outputs == golden."""
    assert check_run(runs[name]) == []


def test_pressure_scenarios_actually_preempt(runs):
    """Guard against a vacuous pass: the pressure scenarios must exercise preemption."""
    for name, r in runs.items():
        if name.startswith("pressure/") or name.startswith("disagg/"):
            assert r.metrics()["preemptions"] > 10, name
    swap = runs["pressure/swap/chunked"]
    assert sum(i.swap.swapped_out for i in swap.instances) > 100
    tiny = runs["pressure/swap/tiny-swap-space"]
    # swap space too small for some victims: the recompute fallback runs too
    assert any(s.preemptions for s in tiny.sequences)


def test_prefix_cache_actually_hits(runs):
    for name in ("pressure/recompute/chunked", "rag-reordered/prefix-cache"):
        assert runs[name].metrics()["prefix_hit_tokens"] > 1000, name


def test_prefix_hit_returns_same_output_as_cold_run():
    """A cache hit must be indistinguishable from recomputing the prefix."""
    reqs = shared_prefix_pressure()
    cold = Cluster.colocated(EngineConfig(enable_prefix_caching=False)).run(reqs)
    warm = Cluster.colocated(EngineConfig(enable_prefix_caching=True)).run(reqs)
    assert warm.metrics()["prefix_hit_tokens"] > 0 and cold.metrics()["prefix_hit_tokens"] == 0
    assert {s.rid: s.outputs for s in warm.sequences} == {s.rid: s.outputs for s in cold.sequences}


def test_preemption_never_loses_or_duplicates_tokens():
    """Recompute and swap must both resume a preempted sequence exactly where it stopped."""
    reqs = shared_prefix_pressure()
    for mode in ("recompute", "swap"):
        cfg = EngineConfig(num_gpu_blocks=200, max_num_batched_tokens=512, preemption_mode=mode,
                           num_cpu_blocks=2000)
        r = Cluster.colocated(cfg).run(reqs)
        preempted = [s for s in r.sequences if s.preemptions]
        assert len(preempted) > 5, mode
        for s in preempted:
            assert len(s.outputs) == s.max_new_tokens
            assert len(s.token_times) == s.max_new_tokens
            assert s.outputs == golden(s.prompt, s.max_new_tokens)


def test_step_budget_and_batch_cap_never_exceeded(runs):
    for name, r in runs.items():
        for st in r.steps:
            cfg = r.instances[st.instance].cfg
            assert st.num_batched_tokens <= cfg.max_num_batched_tokens, name
            assert st.num_seqs <= cfg.max_num_seqs, name
            assert st.used_blocks <= cfg.num_gpu_blocks, name


def test_aborts_cover_queue_states_and_free_blocks(runs):
    """Aborted requests free their blocks wherever the abort lands."""
    states = Counter()
    for name in ("pressure/recompute/aborts", "pressure/swap/aborts", "disagg/aborts"):
        r = runs[name]
        states.update(s.aborted_in for s in r.sequences if s.state == "aborted")
        for inst in r.instances:
            assert inst.alloc.num_used() == 0 and not inst.swap.slots
    assert {"waiting", "running", "swapped", "handoff"} <= set(states)


def test_abort_during_kv_transfer_frees_both_sides():
    """An abort that lands while KV is on the wire frees prefill and decode blocks."""
    reqs = [SimRequest(f"x{i}", 0.01 * i, list(range(i, i + 900)), 32) for i in range(6)]
    cfg = EngineConfig(num_gpu_blocks=400)
    slow = LinkModel(bandwidth_bytes_per_s=2e9)
    probe = Cluster.disaggregated(cfg, cfg, link=slow).run(reqs)
    tr = next(t for t in probe.transfers if t.seq.rid == "x3")
    mid = (tr.start + tr.end) / 2
    r = Cluster.disaggregated(cfg, cfg, link=slow).run(reqs, aborts={"x3": mid}, audit=True)
    x3 = next(s for s in r.sequences if s.rid == "x3")
    assert x3.state == "aborted" and x3.aborted_in == "transferring"
    assert check_run(r) == []


def test_disaggregated_outputs_match_golden_after_transfer():
    """KV copied prefill -> decode must decode to the same tokens as a colocated run."""
    reqs = rag_reordered()
    cfg = EngineConfig()
    co = Cluster.colocated(cfg).run(reqs)
    dis = Cluster.disaggregated(cfg, cfg).run(reqs)
    assert len(dis.transfers) == len(reqs)
    assert {s.rid: s.outputs for s in dis.sequences} == {s.rid: s.outputs for s in co.sequences}


def test_decode_backpressure_holds_prefill_blocks_not_drops_requests():
    """A decode pool that is full delays transfers; nothing is lost and nothing leaks."""
    reqs = shared_prefix_pressure()
    pre = EngineConfig(num_gpu_blocks=2000, max_num_batched_tokens=1024)
    dec = replace(pre, num_gpu_blocks=150)
    r = Cluster.disaggregated(pre, dec).run(reqs, audit=True)
    assert check_run(r) == []
    assert len(r.transfers) == len(reqs)
    assert max(t.start - t.queued_at for t in r.transfers) > 0.05   # transfers waited for decode blocks


def test_simulation_is_deterministic():
    reqs = shared_prefix_pressure()
    cfg = EngineConfig(num_gpu_blocks=200, max_num_batched_tokens=512)
    a, b = Cluster.colocated(cfg).run(reqs), Cluster.colocated(cfg).run(reqs)
    assert [s.token_times for s in a.sequences] == [s.token_times for s in b.sequences]
