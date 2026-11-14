"""Invariant scenarios: workloads built to reach the code paths that break.

`qualify(version)` runs every scenario against one engine version and
returns the violations per scenario. The invariant tests and the
`python -m servesim qualify` command both go through here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable

from servebench.workload import generate
from servesim import materialize
from servesim.cluster import Cluster, LinkModel, RunResult, SimRequest
from servesim.engine import EngineConfig
from servesim.invariants import Violation, check_run


def shared_prefix_pressure(seed: int = 2) -> list[SimRequest]:
    """Shared system prompts + a KV pool too small for the load: forces
    preemption of sequences that share prefix blocks with running ones."""
    return materialize(generate(80, 20.0, seed=seed, shared_prefix_ratio=0.8, prefix_tokens=256,
                                prefix_groups=2, max_prompt=768, max_output=256), seed)


def rag_reordered(seed: int = 3) -> list[SimRequest]:
    """Multi-tenant RAG: the same retrieved document appears after a system
    prompt in some requests and at position 0 in others. Its blocks have
    identical tokens but different KV, because KV depends on what precedes."""
    r = random.Random(seed)
    system = [r.randrange(32000) for _ in range(64)]
    doc = [r.randrange(32000) for _ in range(256)]
    out = []
    for i in range(20):
        q = [r.randrange(32000) for _ in range(20)]
        out.append(SimRequest(f"rag{i:02d}", 0.05 * i, (system + doc + q) if i % 2 == 0 else (doc + q), 16))
    return out


def long_prompts(seed: int = 4) -> list[SimRequest]:
    """Prompts longer than the step budget: every one is chunked."""
    return materialize(generate(30, 5.0, seed=seed, max_output=32), seed)


@dataclass(frozen=True)
class Scenario:
    name: str
    workload: Callable[[], list[SimRequest]]
    cfg: EngineConfig
    topology: str = "colocated"         # or "disagg"
    abort_every: int = 0                # abort every k-th request mid-flight

    def run(self, version: str) -> RunResult:
        cfg = replace(self.cfg, version=version)
        reqs = self.workload()
        aborts = {}
        if self.abort_every:
            for i, r in enumerate(reqs):
                if i % self.abort_every == 1:
                    aborts[r.rid] = r.arrival + 0.01 + 0.15 * (i % 23)   # spread: queued .. mid-decode
        if self.topology == "disagg":
            cluster = Cluster.disaggregated(cfg, cfg, link=LinkModel(20e9))
        else:
            cluster = Cluster.colocated(cfg, n=1)
        return cluster.run(reqs, aborts=aborts, audit=True)


_TIGHT = EngineConfig(num_gpu_blocks=200, max_num_batched_tokens=512, num_cpu_blocks=2000)

SCENARIOS: list[Scenario] = [
    Scenario("pressure/recompute/chunked", shared_prefix_pressure, _TIGHT),
    Scenario("pressure/recompute/no-chunk", shared_prefix_pressure,
             replace(_TIGHT, chunked_prefill=False, max_num_batched_tokens=1024)),
    Scenario("pressure/swap/chunked", shared_prefix_pressure, replace(_TIGHT, preemption_mode="swap")),
    Scenario("pressure/swap/no-prefix-cache", shared_prefix_pressure,
             replace(_TIGHT, preemption_mode="swap", enable_prefix_caching=False)),
    Scenario("pressure/swap/tiny-swap-space", shared_prefix_pressure,
             replace(_TIGHT, preemption_mode="swap", num_cpu_blocks=24)),
    Scenario("pressure/recompute/aborts", shared_prefix_pressure, _TIGHT, abort_every=4),
    Scenario("pressure/swap/aborts", shared_prefix_pressure, replace(_TIGHT, preemption_mode="swap"),
             abort_every=4),
    Scenario("rag-reordered/prefix-cache", rag_reordered, EngineConfig()),
    Scenario("long-prompts/chunk-256", long_prompts, EngineConfig(max_num_batched_tokens=256)),
    Scenario("disagg/pressure", shared_prefix_pressure, replace(_TIGHT, num_gpu_blocks=260), "disagg"),
    Scenario("disagg/aborts", shared_prefix_pressure, replace(_TIGHT, num_gpu_blocks=260), "disagg",
             abort_every=3),
]


def qualify(version: str) -> dict[str, list[Violation]]:
    return {s.name: check_run(s.run(version)) for s in SCENARIOS}
