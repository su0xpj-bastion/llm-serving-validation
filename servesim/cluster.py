"""Discrete-event driver: instances, router, KV-transfer links, aborts.

Two topologies over the same `Instance` code:

* colocated: N instances, each runs prefill and decode in one batch
  (data parallel, least-loaded routing).
* disaggregated: P prefill instances and D decode instances. After prefill
  the KV blocks are copied to a decode instance over a link that costs
  `latency_s + bytes / bandwidth`. Each prefill instance has one outbound
  link and transfers are FIFO on it. The decode side reserves blocks before
  the transfer starts, so a full decode pool pushes back on prefill (the
  transfer waits; the prefill blocks stay held).

The first output token is sampled on the prefill instance but reaches the
client from the decode instance, so its timestamp is the transfer finish.
That is how vLLM's disaggregated proxy and Dynamo behave from the client's
side: TTFT includes the KV transfer.

Everything is deterministic: same inputs, same event order, same floats.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field

from servebench.metrics import GoodputSLO, RequestRecord, aggregate
from servesim.engine import (ABORTED, FINISHED, HANDOFF, TRANSFERRING, CostModel, EngineConfig,
                             Instance, Sequence, StepRecord)

_ARRIVAL, _STEP_DONE, _XFER_DONE, _ABORT = 0, 1, 2, 3


@dataclass(frozen=True)
class LinkModel:
    """KV transfer link, e.g. NVLink/RDMA between prefill and decode GPUs."""

    bandwidth_bytes_per_s: float = 50e9
    latency_s: float = 0.0005

    def transfer_time(self, nbytes: int) -> float:
        return self.latency_s + nbytes / self.bandwidth_bytes_per_s


@dataclass(frozen=True)
class SimRequest:
    rid: str
    arrival: float
    prompt: list[int]
    max_new_tokens: int


@dataclass
class Transfer:
    seq: Sequence
    src: int
    dst: int
    queued_at: float = 0.0
    start: float = 0.0
    end: float = 0.0


@dataclass
class RunResult:
    sequences: list[Sequence]
    steps: list[StepRecord]
    instances: list[Instance]
    transfers: list[Transfer]
    audit: list[tuple[str, str]] = field(default_factory=list)
    stalled: bool = False

    def records(self) -> list[RequestRecord]:
        out = []
        for s in self.sequences:
            out.append(RequestRecord(rid=s.rid, arrival=s.arrival, token_times=list(s.token_times),
                                     output_tokens=len(s.outputs), prompt_tokens=len(s.prompt),
                                     ok=s.state == FINISHED, error="" if s.state == FINISHED else s.state))
        return out

    def metrics(self, slo: GoodputSLO | None = None) -> dict:
        m = aggregate([r for r in self.records() if r.error != ABORTED], slo)
        m["preemptions"] = sum(i.num_preemptions for i in self.instances)
        m["prefix_hit_tokens"] = sum(s.prefix_hit_tokens for s in self.sequences)
        m["evictions"] = sum(i.alloc.evictions for i in self.instances)
        m["steps"] = len(self.steps)
        m["max_step_tokens"] = max((s.num_batched_tokens for s in self.steps), default=0)
        m["kv_transfer_s"] = sum(t.end - t.start for t in self.transfers)
        return m


class Cluster:
    def __init__(self, instances: list[Instance], link: LinkModel | None = None) -> None:
        self.instances = instances
        self.prefill = [i for i in instances if i.role in ("colocated", "prefill")]
        self.decode = [i for i in instances if i.role == "decode"]
        if self.decode and not link:
            raise ValueError("disaggregated cluster needs a LinkModel")
        self.link = link
        self.xfer_queue: dict[int, deque[Transfer]] = {i.iid: deque() for i in self.prefill}
        self.link_busy: dict[int, bool] = {i.iid: False for i in self.prefill}
        self.transfers: list[Transfer] = []

    @classmethod
    def colocated(cls, cfg: EngineConfig, n: int = 1, cost: CostModel | None = None) -> Cluster:
        cost = cost or CostModel()
        return cls([Instance(i, cfg, cost) for i in range(n)])

    @classmethod
    def disaggregated(cls, prefill_cfg: EngineConfig, decode_cfg: EngineConfig, n_prefill: int = 1,
                      n_decode: int = 1, link: LinkModel | None = None,
                      cost: CostModel | None = None) -> Cluster:
        cost = cost or CostModel()
        inst = [Instance(i, prefill_cfg, cost, "prefill") for i in range(n_prefill)]
        inst += [Instance(n_prefill + i, decode_cfg, cost, "decode") for i in range(n_decode)]
        return cls(inst, link or LinkModel())

    # -- event loop -----------------------------------------------------------
    def run(self, requests: list[SimRequest], aborts: dict[str, float] | None = None,
            audit: bool = False, max_events: int = 5_000_000) -> RunResult:
        from servesim.invariants import audit_step   # local: invariants imports this module

        seqs = {r.rid: Sequence(r.rid, r.arrival, list(r.prompt), r.max_new_tokens) for r in requests}
        heap: list[tuple[float, int, int, object]] = []
        counter = itertools.count()

        def push(t: float, kind: int, data: object) -> None:
            heapq.heappush(heap, (t, next(counter), kind, data))

        for r in requests:
            push(r.arrival, _ARRIVAL, seqs[r.rid])
        for rid, t in (aborts or {}).items():
            push(t, _ABORT, seqs[rid])

        audit_log: list[tuple[str, str]] = []
        events = 0
        while heap:
            events += 1
            if events > max_events:
                raise RuntimeError("event budget exceeded; scheduler livelock?")
            now, _, kind, data = heapq.heappop(heap)
            touched: set[int] = set()
            if kind == _ARRIVAL:
                inst = min(self.prefill, key=lambda i: (i.load(), i.iid))
                inst.add_request(data)
                touched.add(inst.iid)
            elif kind == _ABORT:
                seq = data
                if seq.state not in (FINISHED, ABORTED) and seq.instance >= 0:
                    self.instances[seq.instance].request_abort(seq)
                    touched.add(seq.instance)
                    if seq.state == HANDOFF:
                        self._drop_queued_transfer(seq, now)
            elif kind == _STEP_DONE:
                inst = data
                for seq in inst.finish_step(now):
                    # finish_step processes pending aborts last, so a handoff can
                    # already be aborted here; it must not enter the link queue.
                    if seq.state == HANDOFF:
                        self.xfer_queue[inst.iid].append(Transfer(seq, inst.iid, -1, queued_at=now))
                touched.add(inst.iid)
                if audit:
                    audit_log += audit_step(self, inst, seqs.values())
            elif kind == _XFER_DONE:
                tr = data
                self._finish_transfer(tr, now)
                touched.update((tr.src, tr.dst))
            for src in self.xfer_queue:
                self._start_transfers(src, now, push)
            for inst in self.instances:
                if not inst.busy and inst.has_work():
                    dur = inst.schedule(now)
                    if dur is not None:
                        push(now + dur, _STEP_DONE, inst)
                    if audit:
                        audit_log += audit_step(self, inst, seqs.values())

        stalled = any(s.state not in (FINISHED, ABORTED) for s in seqs.values())
        steps = sorted((st for i in self.instances for st in i.steps), key=lambda s: (s.start, s.instance))
        return RunResult(list(seqs.values()), steps, self.instances, self.transfers, audit_log, stalled)

    # -- KV transfer ------------------------------------------------------------
    def _start_transfers(self, src: int, now: float, push) -> None:
        q = self.xfer_queue[src]
        while q and not self.link_busy[src]:
            tr = q[0]
            dst = max(self.decode, key=lambda d: (d.alloc.num_free(), -d.iid))
            if not dst.reserve_incoming(tr.seq, len(tr.seq.prompt)):
                return                                   # decode pool full: back-pressure
            q.popleft()
            tr.dst, tr.start = dst.iid, now
            nblocks = len(tr.seq.xfer_table)
            nbytes = nblocks * dst.cfg.block_size * dst.cost.kv_bytes_per_token
            tr.end = now + self.link.transfer_time(nbytes)
            tr.seq.state = TRANSFERRING
            self.instances[src].handoff.remove(tr.seq)
            self.link_busy[src] = True
            self.transfers.append(tr)
            push(tr.end, _XFER_DONE, tr)

    def _finish_transfer(self, tr: Transfer, now: float) -> None:
        src, dst, seq = self.instances[tr.src], self.instances[tr.dst], tr.seq
        self.link_busy[tr.src] = False
        # Copy block contents prefill -> decode, block by block through both tables.
        for sb, db in zip(seq.block_table, seq.xfer_table):
            s, d = src.alloc.blocks[sb], dst.alloc.blocks[db]
            d.tokens, d.kv = list(s.tokens), s.kv
        src.release_handoff(seq)
        if seq.abort_requested:
            for bid in seq.xfer_table:
                dst.alloc.free(bid)
            seq.xfer_table = []
            dst.incoming -= 1
            seq.aborted_in = TRANSFERRING
            seq.state, seq.finish_time = ABORTED, now
            dst.finished.append(seq)
            return
        seq.token_times[0] = now                      # the client sees token 0 from the decode side
        dst.accept_transfer(seq)

    def _drop_queued_transfer(self, seq: Sequence, now: float) -> None:
        q = self.xfer_queue.get(seq.instance)
        if q is None:
            return
        for tr in list(q):
            if tr.seq is seq:
                q.remove(tr)
