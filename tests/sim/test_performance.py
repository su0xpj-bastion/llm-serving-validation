"""Performance behaviour with explicit expectations. Each threshold is a claim
about how a serving engine should behave, not a snapshot of today's number;
the margins are wide enough that only a behaviour change trips them."""
from __future__ import annotations

import pytest

from servesim import experiments


@pytest.fixture(scope="module")
def disagg():
    return experiments.disaggregation()


def test_disaggregation_cuts_itl_p99_under_prefill_heavy_load(disagg):
    """Colocated decode steps carry prefill chunks, so ITL p99 tracks prefill size.
    A dedicated decode worker only ever runs decode steps."""
    co, dis = disagg["colocated_2x"], disagg["disagg_1p1d"]
    assert dis["itl_s"]["p99"] * 3 < co["itl_s"]["p99"]
    assert dis["itl_s"]["max"] < co["itl_s"]["p99"]


def test_disaggregation_is_not_free_ttft_pays_for_transfer_and_one_prefill_gpu(disagg):
    """Honest trade: 1 prefill GPU instead of 2 data-parallel ones, plus the KV hop."""
    co, dis = disagg["colocated_2x"], disagg["disagg_1p1d"]
    assert dis["ttft_s"]["p50"] > co["ttft_s"]["p50"]
    assert dis["kv_transfer_s"] > 0
    assert dis["output_throughput_tps"] == pytest.approx(co["output_throughput_tps"], rel=0.05)


def test_slow_kv_link_makes_transfer_the_ttft_bottleneck():
    r = experiments.disaggregation_slow_link()
    fast, slow = r["link_50GBps"], r["link_1GBps"]
    assert slow["ttft_s"]["p50"] > 10 * fast["ttft_s"]["p50"]
    assert slow["kv_transfer_s"] > 20 * fast["kv_transfer_s"]


def test_prefix_caching_cuts_ttft_on_shared_system_prompts():
    r = experiments.prefix_caching()
    off, on = r["prefix_cache_off"], r["prefix_cache_on"]
    assert on["prefix_hit_tokens"] > 0.5 * 150 * 1024
    assert on["ttft_s"]["p50"] < 0.5 * off["ttft_s"]["p50"]
    assert on["ttft_s"]["p99"] < 0.5 * off["ttft_s"]["p99"]


@pytest.fixture(scope="module")
def hol():
    return experiments.head_of_line()


def test_long_prompt_without_chunked_prefill_stalls_every_decode(hol):
    """Head-of-line blocking: an 8k prefill in one step freezes all running decodes."""
    stall = hol["no_chunked_prefill"]["itl_s"]["max"]
    assert stall > 0.2                                   # ~8000 tokens * 30 us
    assert hol["chunked_prefill_512"]["itl_s"]["max"] < stall / 5


def test_chunked_prefill_trades_a_higher_itl_floor_for_no_stalls(hol):
    """The cost of chunking is visible too: many steps carry a 512-token chunk, so ITL p99
    rises even though the worst stall disappears. Both numbers belong in a release report."""
    assert hol["chunked_prefill_512"]["itl_s"]["p99"] > hol["no_chunked_prefill"]["itl_s"]["p99"]


@pytest.fixture(scope="module")
def kv():
    return experiments.kv_pressure()


def test_kv_pressure_shows_up_as_a_preemption_storm(kv):
    roomy, tight = kv["kv_4096_blocks"], kv["kv_512_blocks"]
    assert roomy["preemptions"] == 0
    assert tight["preemptions"] > 40
    assert tight["itl_s"]["max"] > 10 * roomy["itl_s"]["max"]          # recompute stalls
    assert tight["output_throughput_tps"] < 0.8 * roomy["output_throughput_tps"]


def test_admission_watermark_damps_the_storm_at_a_ttft_cost(kv):
    tight, wm = kv["kv_512_blocks"], kv["kv_512_blocks_wm20"]
    assert wm["preemptions"] < tight["preemptions"] / 4
    assert wm["ttft_s"]["p50"] > tight["ttft_s"]["p50"]


def test_continuous_batching_scales_throughput_with_batch_size():
    """Decode is memory-bound: one weight read serves the whole batch."""
    r = experiments.batching()
    t1, t8, t64 = (r[f"max_num_seqs_{n}"]["output_throughput_tps"] for n in (1, 8, 64))
    assert t8 > 5 * t1 and t64 > 3 * t8
    assert r["max_num_seqs_64"]["itl_s"]["p50"] > r["max_num_seqs_1"]["itl_s"]["p50"]   # per-token cost of batching


def test_experiment_table_renders():
    md = experiments.markdown({"batching": experiments.batching()})
    assert md.count("\n") == 4 and "max_num_seqs_64" in md
