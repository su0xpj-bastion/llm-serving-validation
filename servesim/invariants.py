"""Invariants the engine must hold, checked per step and after the run.

Per step (`audit_step`, enabled with `Cluster.run(audit=True)`):
  budget_exceeded     a step scheduled more than max_num_batched_tokens
  max_seqs_exceeded   more than max_num_seqs sequences running
  refcount_mismatch   a block's ref count != number of block tables that hold it
                      (catches a reference dropped or taken twice, the step it happens)
  refcount_negative / alloc_referenced   recorded by the allocator itself

After the run (`check_run`):
  block_leak          blocks still referenced once every request finished or aborted
  swap_leak           CPU swap slots never returned
  output_mismatch     generated tokens differ from the engine-free golden model
                      (catches wrong KV: prefix collisions, use-after-free, lost chunks)
  token_count         a finished request produced != max_new_tokens tokens
                      (lost or duplicated tokens across preemption)
  non_monotonic       token timestamps go backwards
  unfinished          a non-aborted request never completed (stall / lost request)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from servesim.cluster import Cluster, RunResult
from servesim.engine import ABORTED, FINISHED, Instance, Sequence
from servesim.model import golden


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str


def audit_step(cluster: Cluster, inst: Instance, seqs: Iterable[Sequence]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if inst.steps:
        st = inst.steps[-1]
        if st.num_batched_tokens > inst.cfg.max_num_batched_tokens:
            out.append(("budget_exceeded",
                        f"instance {inst.iid} step@{st.start:.4f}: {st.num_batched_tokens} > "
                        f"{inst.cfg.max_num_batched_tokens}"))
    if len(inst.running) > inst.cfg.max_num_seqs:
        out.append(("max_seqs_exceeded", f"instance {inst.iid}: {len(inst.running)} running"))
    expected: Counter[int] = Counter()
    for s in seqs:
        if s.instance == inst.iid and s.state not in (FINISHED, ABORTED):
            expected.update(s.block_table)
            expected.update(ident for kind, ident in s.swap_slots if kind == "gpu")
        if s.xfer_dst == inst.iid and s.xfer_table:
            expected.update(s.xfer_table)
    for blk in inst.alloc.blocks:
        if blk.ref != expected.get(blk.bid, 0):
            out.append(("refcount_mismatch",
                        f"instance {inst.iid} block {blk.bid}: ref={blk.ref}, "
                        f"held by {expected.get(blk.bid, 0)} tables"))
            break                                   # one per step is enough signal
    return out


def check_run(result: RunResult, requests_by_rid: dict | None = None) -> list[Violation]:
    """Every invariant, over the whole run. Empty list == pass."""
    out = [Violation(k, d) for k, d in dict.fromkeys(result.audit)]   # audit runs twice per event
    for inst in result.instances:
        out += [Violation(k, f"instance {inst.iid}: {d}") for k, d in inst.alloc.violations]
        out += [Violation(k, f"instance {inst.iid}: {d}") for k, d in inst.alloc.quiescent_problems()]
        if inst.swap.slots:
            out.append(Violation("swap_leak", f"instance {inst.iid}: {len(inst.swap.slots)} slots"))
    for s in result.sequences:
        if s.state == ABORTED:
            continue
        if s.state != FINISHED:
            out.append(Violation("unfinished", f"{s.rid} ended in state {s.state}"))
            continue
        if len(s.outputs) != s.max_new_tokens:
            out.append(Violation("token_count", f"{s.rid}: {len(s.outputs)} != {s.max_new_tokens}"))
        if s.outputs != golden(s.prompt, s.max_new_tokens):
            first = next((i for i, (a, b) in enumerate(zip(s.outputs, golden(s.prompt, s.max_new_tokens)))
                          if a != b), len(s.outputs))
            out.append(Violation("output_mismatch",
                                 f"{s.rid}: diverges from golden at output token {first} "
                                 f"(prefix_hit={s.prefix_hit_tokens}, preemptions={s.preemptions})"))
        if any(b < a for a, b in zip(s.token_times, s.token_times[1:])):
            out.append(Violation("non_monotonic", f"{s.rid}: token timestamps go backwards"))
    if result.stalled:
        out.append(Violation("unfinished", "event queue drained with live requests (scheduler stall)"))
    return out


def kinds(violations: list[Violation]) -> set[str]:
    return {v.kind for v in violations}
