"""Named performance experiments. The perf tests assert on these and the CLI
prints them, so the numbers in the README come from the code the tests run.

Each experiment returns {variant_name: metrics_dict}.
"""
from __future__ import annotations

from servebench.workload import TraceRequest, generate
from servesim import materialize
from servesim.cluster import Cluster, LinkModel, SimRequest
from servesim.engine import EngineConfig

ENGINE_SEED = 7


def _run(cluster: Cluster, reqs: list[SimRequest]) -> dict:
    return cluster.run(reqs).metrics()


def prefill_heavy_trace(n: int = 120, rate: float = 6.0, seed: int = ENGINE_SEED) -> list[TraceRequest]:
    """Long prompts, short answers: RAG / summarisation traffic."""
    return generate(n, rate, seed=seed, prompt_dist=(7.3, 0.5), output_dist=(4.6, 0.4),
                    max_prompt=4096, max_output=256)


def disaggregation(bandwidth: float = 50e9) -> dict[str, dict]:
    """2 GPUs either way: colocated data-parallel x2 vs 1 prefill + 1 decode."""
    reqs = materialize(prefill_heavy_trace())
    cfg = EngineConfig(num_gpu_blocks=4096, max_num_batched_tokens=2048, max_num_seqs=128)
    return {
        "colocated_2x": _run(Cluster.colocated(cfg, n=2), reqs),
        "disagg_1p1d": _run(Cluster.disaggregated(cfg, cfg, link=LinkModel(bandwidth)), reqs),
    }


def disaggregation_slow_link() -> dict[str, dict]:
    """Same split over a 1 GB/s link: the transfer now dominates TTFT."""
    reqs = materialize(prefill_heavy_trace())
    cfg = EngineConfig(num_gpu_blocks=4096, max_num_batched_tokens=2048, max_num_seqs=128)
    return {
        "link_50GBps": _run(Cluster.disaggregated(cfg, cfg, link=LinkModel(50e9)), reqs),
        "link_1GBps": _run(Cluster.disaggregated(cfg, cfg, link=LinkModel(1e9)), reqs),
    }


def prefix_caching() -> dict[str, dict]:
    """Chat traffic where 90% of requests carry one of 4 shared 1024-token system prompts."""
    trace = generate(150, 8.0, seed=ENGINE_SEED, shared_prefix_ratio=0.9, prefix_groups=4,
                     prefix_tokens=1024, prompt_dist=(4.5, 0.6), max_prompt=2048, max_output=128)
    reqs = materialize(trace)
    out = {}
    for name, on in (("prefix_cache_off", False), ("prefix_cache_on", True)):
        cfg = EngineConfig(num_gpu_blocks=4096, max_num_batched_tokens=2048, enable_prefix_caching=on)
        out[name] = _run(Cluster.colocated(cfg), reqs)
    return out


def head_of_line() -> dict[str, dict]:
    """Steady decode traffic plus a few 8k-token prompts.

    Without chunked prefill the step budget must cover the longest prompt
    (8192), so each long prompt is prefilled in one step and every running
    decode waits for it. With chunked prefill (512-token budget) the long
    prompt is spread over many steps that still carry the decodes.
    """
    base = generate(80, 10.0, seed=ENGINE_SEED, prompt_dist=(5.0, 0.4), output_dist=(5.3, 0.3),
                    max_prompt=512, max_output=256)
    longs = [TraceRequest(f"long{i}", 2.0 + 2.0 * i, 8000, 16) for i in range(3)]
    reqs = materialize(sorted(base + longs, key=lambda r: r.arrival_s))
    out = {}
    for name, chunked, budget in (("no_chunked_prefill", False, 8192), ("chunked_prefill_512", True, 512)):
        cfg = EngineConfig(num_gpu_blocks=4096, max_num_batched_tokens=budget, chunked_prefill=chunked)
        out[name] = _run(Cluster.colocated(cfg), reqs)
    return out


def kv_pressure() -> dict[str, dict]:
    """Same load, shrinking KV pool. Admission is optimistic (prompt blocks
    only), so decode growth runs out of blocks and preempts; a larger
    admission watermark keeps headroom for that growth."""
    trace = generate(120, 12.0, seed=ENGINE_SEED, prompt_dist=(5.5, 0.5), output_dist=(5.8, 0.4),
                     max_prompt=1024, max_output=512)
    reqs = materialize(trace)
    out = {}
    for name, blocks, wm in (("kv_4096_blocks", 4096, 0.01), ("kv_512_blocks", 512, 0.01),
                             ("kv_512_blocks_wm20", 512, 0.20)):
        cfg = EngineConfig(num_gpu_blocks=blocks, max_num_batched_tokens=2048, watermark=wm)
        out[name] = _run(Cluster.colocated(cfg), reqs)
    return out


def batching() -> dict[str, dict]:
    """Continuous batching: throughput as the batch cap grows (decode is memory-bound,
    so the weight read is amortised over the batch)."""
    trace = generate(64, 1000.0, seed=ENGINE_SEED, prompt_dist=(5.0, 0.3), output_dist=(5.0, 0.3),
                     max_prompt=512, max_output=256)
    reqs = materialize(trace)
    out = {}
    for cap in (1, 8, 64):
        cfg = EngineConfig(num_gpu_blocks=4096, max_num_batched_tokens=2048, max_num_seqs=cap)
        out[f"max_num_seqs_{cap}"] = _run(Cluster.colocated(cfg), reqs)
    return out


EXPERIMENTS = {
    "disaggregation": disaggregation,
    "disaggregation_slow_link": disaggregation_slow_link,
    "prefix_caching": prefix_caching,
    "head_of_line": head_of_line,
    "kv_pressure": kv_pressure,
    "batching": batching,
}

COLUMNS = (
    ("TTFT p50 ms", lambda m: m["ttft_s"]["p50"] * 1e3),
    ("TTFT p99 ms", lambda m: m["ttft_s"]["p99"] * 1e3),
    ("ITL p50 ms", lambda m: m["itl_s"]["p50"] * 1e3),
    ("ITL p99 ms", lambda m: m["itl_s"]["p99"] * 1e3),
    ("ITL max ms", lambda m: m["itl_s"]["max"] * 1e3),
    ("out tok/s", lambda m: m["output_throughput_tps"]),
    ("preempt", lambda m: m["preemptions"]),
)


def markdown(results: dict[str, dict[str, dict]]) -> str:
    head = "| experiment | variant | " + " | ".join(c for c, _ in COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 2)]
    for exp, variants in results.items():
        for var, m in variants.items():
            cells = []
            for _, f in COLUMNS:
                v = f(m)
                cells.append(f"{v:.0f}" if isinstance(v, int) or v >= 100 else f"{v:.1f}")
            lines.append(f"| {exp} | {var} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
