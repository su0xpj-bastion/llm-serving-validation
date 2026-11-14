"""servesim: a deterministic discrete-event model of an LLM serving engine."""
from __future__ import annotations

from servebench.workload import TraceRequest, prompt_ids
from servesim.cluster import SimRequest


def materialize(trace: list[TraceRequest], seed: int = 0) -> list[SimRequest]:
    """Turn a trace into token-level requests the simulator can run."""
    return [SimRequest(r.rid, r.arrival_s, prompt_ids(r, seed), r.output_tokens) for r in trace]
