"""Paged KV cache: fixed-size blocks, ref counts, prefix cache, LRU eviction.

Modelled on vLLM's block manager:

* GPU memory is `num_blocks` blocks of `block_size` token slots.
* A sequence owns a block table (logical block i -> physical block id).
* A full block gets a hash chained from its parent block's hash, so the hash
  identifies the whole prefix up to and including that block. Blocks with the
  same hash are interchangeable, which is what prefix caching relies on.
* A block whose ref count drops to 0 keeps its contents. If it is hashed it
  goes on the evictable LRU list and can still be hit by a later request.
  It is reclaimed only when the free list is empty.

The allocator records violations (double free, negative ref count) instead
of raising, so a seeded bug runs to completion and the invariant checker
can report every symptom it caused.
"""
from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict, deque
from dataclasses import dataclass, field

NONE_HASH = 0  # parent hash of the first block


def block_hash(parent: int, tokens: list[int] | tuple[int, ...]) -> int:
    """Chained block hash: identifies the entire prefix, not just this block."""
    h = hashlib.blake2b(digest_size=8)
    h.update(struct.pack("<Q", parent))
    h.update(struct.pack(f"<{len(tokens)}I", *tokens))
    return int.from_bytes(h.digest(), "little") or 1


@dataclass
class Block:
    bid: int
    ref: int = 0
    tokens: list[int] = field(default_factory=list)
    kv: int = 0                    # model state after the last token written here
    hash: int | None = None        # set once the block is full and registered


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_list: deque[int] = deque(range(num_blocks))
        self.evictable: OrderedDict[int, None] = OrderedDict()   # LRU first
        self.cached: dict[int, int] = {}                          # hash -> bid
        self.violations: list[tuple[str, str]] = []
        self.evictions = 0
        self.hits = 0

    # -- capacity ---------------------------------------------------------
    def num_free(self) -> int:
        """Blocks an allocation can take: truly free plus evictable cached."""
        return len(self.free_list) + len(self.evictable)

    def num_used(self) -> int:
        return self.num_blocks - self.num_free()

    # -- allocation -------------------------------------------------------
    def allocate(self) -> int | None:
        if self.free_list:
            bid = self.free_list.popleft()
        elif self.evictable:
            bid, _ = self.evictable.popitem(last=False)          # least recently used
            blk = self.blocks[bid]
            if blk.hash is not None and self.cached.get(blk.hash) == bid:
                del self.cached[blk.hash]
            self.evictions += 1
        else:
            return None
        blk = self.blocks[bid]
        if blk.ref != 0:
            self.violations.append(("alloc_referenced", f"block {bid} handed out with ref={blk.ref}"))
        blk.ref, blk.tokens, blk.kv, blk.hash = 1, [], 0, None
        return bid

    def acquire(self, bid: int) -> None:
        """Take another reference to a block (prefix hit or fork)."""
        blk = self.blocks[bid]
        if blk.ref == 0:
            self.evictable.pop(bid, None)
        blk.ref += 1

    def free(self, bid: int) -> None:
        blk = self.blocks[bid]
        blk.ref -= 1
        if blk.ref < 0:
            self.violations.append(("refcount_negative", f"block {bid} ref={blk.ref}"))
            return
        if blk.ref == 0:
            if blk.hash is not None and self.cached.get(blk.hash) == bid:
                self.evictable[bid] = None
                self.evictable.move_to_end(bid)
            else:
                blk.hash = None
                self.free_list.append(bid)

    # -- prefix cache -----------------------------------------------------
    def lookup(self, hashes: list[int]) -> list[int]:
        """Longest cached prefix. Does not take references (see acquire)."""
        hit = []
        for h in hashes:
            bid = self.cached.get(h)
            if bid is None:
                break
            hit.append(bid)
        return hit

    def register(self, bid: int, h: int) -> None:
        """Publish a full block under its hash; first writer wins."""
        if h in self.cached:
            return
        self.blocks[bid].hash = h
        self.cached[h] = bid

    # -- audit ------------------------------------------------------------
    def quiescent_problems(self) -> list[tuple[str, str]]:
        """With no live sequences every block must be reclaimable."""
        out = []
        held = [b.bid for b in self.blocks if b.ref != 0]
        if held:
            out.append(("block_leak", f"{len(held)} blocks still referenced: {held[:8]}"))
        pool = list(self.free_list) + list(self.evictable)
        if len(pool) != len(set(pool)):
            out.append(("free_list_duplicate", "a block is on the free/evictable lists twice"))
        if len(set(pool)) != self.num_blocks and not held:
            out.append(("block_lost", f"{self.num_blocks - len(set(pool))} blocks on no list"))
        return out


class SwapSpace:
    """CPU-side block storage for swap preemption."""

    def __init__(self, num_blocks: int) -> None:
        self.capacity = num_blocks
        self.slots: dict[int, tuple[list[int], int]] = {}
        self._next = 0
        self.swapped_out = 0
        self.swapped_in = 0

    def num_free(self) -> int:
        return self.capacity - len(self.slots)

    def store(self, tokens: list[int], kv: int) -> int:
        if self.num_free() <= 0:
            raise RuntimeError("swap space exhausted; caller must check num_free()")
        sid = self._next
        self._next += 1
        self.slots[sid] = (list(tokens), kv)
        self.swapped_out += 1
        return sid

    def load(self, sid: int) -> tuple[list[int], int]:
        self.swapped_in += 1
        return self.slots.pop(sid)
