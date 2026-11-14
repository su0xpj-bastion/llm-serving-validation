"""One engine instance: a continuous-batching scheduler over a paged KV cache.

The scheduler follows vLLM v1's shape. Every step it hands each sequence a
number of tokens to compute (`num_tokens - num_computed`, capped by the
step's token budget), so prefill chunks and decode tokens share one batch:

  1. RUNNING sequences first, in arrival order. A sequence that needs a new
     block and finds none preempts the newest running sequence (recompute:
     drop its KV and re-prefill later; swap: copy its KV to CPU).
  2. If nothing was preempted: SWAPPED sequences come back in.
  3. Then WAITING requests are admitted while the token budget, max_num_seqs
     and the free-block watermark (admission control) allow. Without chunked
     prefill a prompt that does not fit the remaining budget blocks the queue.

A sequence whose tokens are all computed samples one token at the end of the
step. Roles: "colocated" does everything; "prefill" stops after the first
token and hands the sequence (and its KV blocks) to the cluster for transfer;
"decode" receives sequences whose prompt KV arrived over the link.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from servesim import versions
from servesim.kv_cache import NONE_HASH, BlockAllocator, SwapSpace, block_hash
from servesim.model import SEED_STATE, mix, sample

WAITING, RUNNING, SWAPPED, HANDOFF, TRANSFERRING, FINISHED, ABORTED = (
    "waiting", "running", "swapped", "handoff", "transferring", "finished", "aborted")


@dataclass
class EngineConfig:
    num_gpu_blocks: int = 2048
    block_size: int = 16
    max_num_batched_tokens: int = 2048     # per-step token budget
    max_num_seqs: int = 64
    chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    preemption_mode: str = "recompute"     # or "swap"
    num_cpu_blocks: int = 0                # swap space
    watermark: float = 0.01                # fraction of blocks admission keeps free
    version: str = versions.LATEST

    def __post_init__(self) -> None:
        if self.preemption_mode not in ("recompute", "swap"):
            raise ValueError(f"preemption_mode must be recompute|swap, got {self.preemption_mode}")
        self.bugs = versions.bugs_for(self.version)


@dataclass(frozen=True)
class CostModel:
    """Step latency, loosely calibrated to an 8B model on one H100-class GPU.

    Decode is memory-bound: every step reads the weights (the fixed overhead)
    plus the KV of every sequence in the batch. Prefill is compute-bound:
    cost grows with tokens. Defaults: ~5 ms weight read, ~33k prefill tok/s,
    128 KiB of KV per token at ~3 TB/s.
    """

    step_overhead_s: float = 0.005
    prefill_s_per_token: float = 0.00003
    decode_s_per_seq: float = 0.00002
    kv_read_s_per_token: float = 0.00000004
    kv_bytes_per_token: int = 128 * 1024

    def step_time(self, prefill_tokens: int, decode_seqs: int, context_tokens: int) -> float:
        return (self.step_overhead_s + self.prefill_s_per_token * prefill_tokens
                + self.decode_s_per_seq * decode_seqs + self.kv_read_s_per_token * context_tokens)


@dataclass
class Sequence:
    rid: str
    arrival: float
    prompt: list[int]
    max_new_tokens: int
    outputs: list[int] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)
    state: str = WAITING
    num_computed: int = 0
    block_table: list[int] = field(default_factory=list)
    swap_slots: list[tuple[str, int]] = field(default_factory=list)   # ("gpu"|"cpu", id)
    xfer_table: list[int] = field(default_factory=list)               # decode-side blocks in flight
    instance: int = -1
    xfer_dst: int = -1
    preemptions: int = 0
    prefix_hit_tokens: int = 0
    finish_time: float | None = None
    abort_requested: bool = False
    aborted_in: str = ""              # state the request was in when the abort landed
    _hashes: list[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.outputs)

    def token(self, i: int) -> int:
        p = len(self.prompt)
        return self.prompt[i] if i < p else self.outputs[i - p]

    def full_block_hashes(self, upto: int, block_size: int, ignore_parent: bool) -> list[int]:
        """Hashes of the full blocks covering tokens[:upto] (cached, append-only)."""
        nfull = upto // block_size
        while len(self._hashes) < nfull:
            k = len(self._hashes)
            parent = NONE_HASH if (k == 0 or ignore_parent) else self._hashes[k - 1]
            toks = [self.token(i) for i in range(k * block_size, (k + 1) * block_size)]
            self._hashes.append(block_hash(parent, toks))
        return self._hashes[:nfull]


@dataclass
class StepRecord:
    instance: int
    start: float
    end: float
    num_batched_tokens: int
    prefill_tokens: int
    decode_seqs: int
    num_seqs: int
    used_blocks: int
    preemptions: int


class Instance:
    def __init__(self, iid: int, cfg: EngineConfig, cost: CostModel, role: str = "colocated") -> None:
        if role not in ("colocated", "prefill", "decode"):
            raise ValueError(role)
        self.iid, self.cfg, self.cost, self.role = iid, cfg, cost, role
        self.alloc = BlockAllocator(cfg.num_gpu_blocks, cfg.block_size)
        self.swap = SwapSpace(cfg.num_cpu_blocks)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.swapped: deque[Sequence] = deque()
        self.handoff: list[Sequence] = []
        self.busy = False
        self.inflight: list[tuple[Sequence, int]] = []
        self.num_preemptions = 0
        self.steps: list[StepRecord] = []
        self.finished: list[Sequence] = []
        self.pending_aborts: list[Sequence] = []
        self.incoming = 0     # transfers reserved against this decode instance
        self.watermark_blocks = int(cfg.num_gpu_blocks * cfg.watermark)

    # -- queue ops ----------------------------------------------------------
    def add_request(self, seq: Sequence) -> None:
        seq.instance, seq.state = self.iid, WAITING
        self.waiting.append(seq)

    def load(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + self.incoming

    def has_work(self) -> bool:
        return bool(self.running or self.waiting or self.swapped or self.pending_aborts)

    def live_sequences(self) -> list[Sequence]:
        return [*self.waiting, *self.running, *self.swapped, *self.handoff]

    # -- block helpers ------------------------------------------------------
    def _blocks_for(self, ntokens: int) -> int:
        return math.ceil(ntokens / self.cfg.block_size)

    def _ensure_blocks(self, seq: Sequence, upto: int) -> bool:
        need = self._blocks_for(upto) - len(seq.block_table)
        if need <= 0:
            return True
        if self.alloc.num_free() < need:
            return False
        for _ in range(need):
            seq.block_table.append(self.alloc.allocate())
        return True

    def _free_table(self, seq: Sequence) -> None:
        for bid in reversed(seq.block_table):   # tail first: deepest blocks evicted first
            self.alloc.free(bid)
        seq.block_table = []

    def _chunk(self, need: int, budget: int) -> int:
        if versions.CHUNK_BUDGET_OFF_BY_ONE in self.cfg.bugs and need > 1:
            return min(need, budget + 1)          # seeded: "inclusive" remaining budget
        return min(need, budget)

    # -- preemption ---------------------------------------------------------
    def _preempt(self, seq: Sequence) -> None:
        seq.preemptions += 1
        self.num_preemptions += 1
        self._step_preemptions += 1
        if self.cfg.preemption_mode == "swap":
            private = sum(1 for b in seq.block_table if self.alloc.blocks[b].ref == 1)
            if private <= self.swap.num_free():
                self._swap_out(seq)
                return
        self._free_table(seq)
        seq.num_computed = 0
        seq.state = WAITING
        self.waiting.appendleft(seq)

    def _swap_out(self, seq: Sequence) -> None:
        slots = []
        for bid in seq.block_table:
            blk = self.alloc.blocks[bid]
            if blk.ref > 1:
                # Shared prefix block: another sequence keeps it resident. We keep
                # our reference so it cannot be evicted while we are swapped out.
                slots.append(("gpu", bid))
                if versions.SWAP_FREES_SHARED_BLOCK in self.cfg.bugs:
                    self.alloc.free(bid)          # seeded: "the other holder owns it now"
            else:
                slots.append(("cpu", self.swap.store(blk.tokens, blk.kv)))
                self.alloc.free(bid)
        seq.block_table, seq.swap_slots, seq.state = [], slots, SWAPPED
        self.swapped.append(seq)

    def _swap_in(self, seq: Sequence) -> bool:
        need = sum(1 for kind, _ in seq.swap_slots if kind == "cpu")
        if self.alloc.num_free() - need < self.watermark_blocks:
            return False
        table = []
        for kind, ident in seq.swap_slots:
            if kind == "gpu":
                table.append(ident)
            else:
                bid = self.alloc.allocate()
                blk = self.alloc.blocks[bid]
                blk.tokens, blk.kv = self.swap.load(ident)
                table.append(bid)
        seq.block_table, seq.swap_slots, seq.state = table, [], RUNNING
        return True

    # -- abort --------------------------------------------------------------
    def request_abort(self, seq: Sequence) -> None:
        seq.abort_requested = True
        self.pending_aborts.append(seq)

    def _process_aborts(self, now: float) -> None:
        for seq in self.pending_aborts:
            if seq.state in (FINISHED, ABORTED):
                continue
            if seq.state != TRANSFERRING:
                seq.aborted_in = seq.state
            if seq in self.running:
                self.running.remove(seq)
            elif seq in self.waiting:
                self.waiting.remove(seq)
            elif seq in self.swapped:
                self.swapped.remove(seq)
                for kind, ident in seq.swap_slots:
                    if kind == "gpu":
                        self.alloc.free(ident)
                    else:
                        self.swap.load(ident)
                seq.swap_slots = []
            elif seq in self.handoff:
                self.handoff.remove(seq)
            elif seq.state == TRANSFERRING:
                continue          # the cluster frees both sides when the link finishes
            self._free_table(seq)
            seq.state, seq.finish_time = ABORTED, now
            self.finished.append(seq)
        self.pending_aborts = [s for s in self.pending_aborts if s.state == TRANSFERRING]

    # -- one step -----------------------------------------------------------
    def schedule(self, now: float) -> float | None:
        """Pick the batch, write its KV, and return the step duration.

        Returns None when there is nothing runnable (e.g. the queue head
        does not fit in memory and nothing is running to free blocks).
        """
        self._process_aborts(now)
        cfg, bs = self.cfg, self.cfg.block_size
        budget = cfg.max_num_batched_tokens
        self._step_preemptions = 0
        scheduled: list[tuple[Sequence, int]] = []

        # 1. running
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            need = seq.num_tokens - seq.num_computed
            n = self._chunk(need, budget)
            if n <= 0 or (not cfg.chunked_prefill and n < need):
                i += 1
                continue
            while not self._ensure_blocks(seq, seq.num_computed + n):
                victim = self.running.pop()
                self._preempt(victim)
                if victim is seq:
                    break
            else:
                scheduled.append((seq, n))
                budget -= n
                i += 1
                continue
            break

        # 2. swapped back in, 3. new admissions: only on a step with no preemption
        if self._step_preemptions == 0:
            while self.swapped and len(self.running) < cfg.max_num_seqs:
                seq = self.swapped[0]
                if not self._swap_in(seq):
                    break
                self.swapped.popleft()
                self.running.append(seq)
                n = self._chunk(seq.num_tokens - seq.num_computed, budget)
                if n > 0 and (cfg.chunked_prefill or n == seq.num_tokens - seq.num_computed) \
                        and self._ensure_blocks(seq, seq.num_computed + n):
                    scheduled.append((seq, n))
                    budget -= n
        if self._step_preemptions == 0 and not self.swapped:
            while self.waiting and len(self.running) < cfg.max_num_seqs and budget > 0:
                seq = self.waiting[0]
                hits: list[int] = []
                if cfg.enable_prefix_caching and seq.num_computed == 0:
                    # Leave at least one token to compute so the step yields logits.
                    hashes = seq.full_block_hashes(seq.num_tokens - 1, bs,
                                                   versions.PREFIX_HASH_IGNORES_PARENT in cfg.bugs)
                    hits = self.alloc.lookup(hashes)
                start = len(hits) * bs
                need = seq.num_tokens - start
                n = self._chunk(need, budget)
                if not cfg.chunked_prefill and n < need:
                    break                                    # head-of-line: wait for a roomier step
                new_blocks = self._blocks_for(start + n) - len(hits)
                reclaim = sum(1 for b in hits if self.alloc.blocks[b].ref == 0)
                if self.alloc.num_free() - reclaim - new_blocks < self.watermark_blocks:
                    break                                    # admission control
                for b in hits:
                    self.alloc.acquire(b)
                self.alloc.hits += len(hits)
                seq.block_table = list(hits)
                seq.num_computed = start
                seq.prefix_hit_tokens = max(seq.prefix_hit_tokens, start)
                self._ensure_blocks(seq, start + n)
                self.waiting.popleft()
                seq.state = RUNNING
                self.running.append(seq)
                scheduled.append((seq, n))
                budget -= n

        if not scheduled:
            return None

        prefill_tokens = decode_seqs = context = 0
        for seq, n in scheduled:
            self._write_kv(seq, n)
            context += seq.num_computed
            if n == 1 and seq.outputs and seq.num_computed == seq.num_tokens:
                decode_seqs += 1
            else:
                prefill_tokens += n
        self.inflight = scheduled
        dur = self.cost.step_time(prefill_tokens, decode_seqs, context)
        self.steps.append(StepRecord(
            self.iid, now, now + dur, sum(n for _, n in scheduled), prefill_tokens, decode_seqs,
            len(scheduled), self.alloc.num_used(), self._step_preemptions))
        self.busy = True
        return dur

    def _write_kv(self, seq: Sequence, n: int) -> None:
        """Compute n tokens: each slot's state is chained from the previous
        slot *as read through this sequence's block table*."""
        bs, blocks = self.cfg.block_size, self.alloc.blocks
        for pos in range(seq.num_computed, seq.num_computed + n):
            k, off = divmod(pos, bs)
            blk = blocks[seq.block_table[k]]
            if off == 0:
                prev = SEED_STATE if k == 0 else blocks[seq.block_table[k - 1]].kv
                blk.tokens = []
            else:
                prev = blk.kv
            blk.tokens.append(seq.token(pos))
            blk.kv = mix(prev, seq.token(pos))
            if off == bs - 1 and self.cfg.enable_prefix_caching:
                hashes = seq.full_block_hashes(pos + 1, bs, versions.PREFIX_HASH_IGNORES_PARENT in self.cfg.bugs)
                self.alloc.register(seq.block_table[k], hashes[k])
        seq.num_computed += n

    def finish_step(self, now: float) -> list[Sequence]:
        """Sample for sequences that caught up. Returns sequences to hand off."""
        handoffs = []
        for seq, _ in self.inflight:
            if seq.state != RUNNING or seq.num_computed != seq.num_tokens:
                continue
            last = self.alloc.blocks[seq.block_table[(seq.num_computed - 1) // self.cfg.block_size]]
            seq.outputs.append(sample(last.kv))
            seq.token_times.append(now)
            if len(seq.outputs) >= seq.max_new_tokens:
                self.running.remove(seq)
                self._free_table(seq)
                seq.state, seq.finish_time = FINISHED, now
                self.finished.append(seq)
            elif self.role == "prefill":
                self.running.remove(seq)
                seq.state = HANDOFF
                self.handoff.append(seq)
                handoffs.append(seq)
        self.inflight = []
        self.busy = False
        self._process_aborts(now)
        return handoffs

    # -- disaggregation hooks ------------------------------------------------
    def reserve_incoming(self, seq: Sequence, ntokens: int) -> bool:
        """Decode side: reserve blocks for a transfer before it starts."""
        need = self._blocks_for(ntokens)
        if (len(self.running) + self.incoming >= self.cfg.max_num_seqs
                or self.alloc.num_free() - need < self.watermark_blocks):
            return False
        seq.xfer_table = [self.alloc.allocate() for _ in range(need)]
        seq.xfer_dst = self.iid
        self.incoming += 1
        return True

    def accept_transfer(self, seq: Sequence) -> None:
        self.incoming -= 1
        seq.block_table, seq.xfer_table = seq.xfer_table, []
        seq.instance, seq.state = self.iid, RUNNING
        seq.num_computed = len(seq.prompt)
        self.running.append(seq)

    def release_handoff(self, seq: Sequence) -> None:
        """Prefill side: the KV has been copied out; drop our references."""
        if seq in self.handoff:
            self.handoff.remove(seq)
        self._free_table(seq)
