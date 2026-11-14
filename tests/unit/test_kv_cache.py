from __future__ import annotations

from servesim.kv_cache import NONE_HASH, BlockAllocator, SwapSpace, block_hash


def test_allocate_until_exhausted_then_none():
    """Memory budget: the allocator never hands out more blocks than exist."""
    a = BlockAllocator(4, 16)
    got = [a.allocate() for _ in range(4)]
    assert sorted(got) == [0, 1, 2, 3]
    assert a.allocate() is None
    assert a.num_free() == 0 and a.num_used() == 4


def test_free_returns_block_and_quiescent_is_clean():
    """Leak check: after every reference is dropped, nothing is held."""
    a = BlockAllocator(4, 16)
    b = a.allocate()
    a.acquire(b)
    a.free(b)
    assert a.num_used() == 1
    a.free(b)
    assert a.num_used() == 0
    assert a.quiescent_problems() == []


def test_double_free_is_recorded_as_negative_refcount():
    """Ref counts never go negative; if they would, the allocator records it."""
    a = BlockAllocator(2, 16)
    b = a.allocate()
    a.free(b)
    a.free(b)
    assert ("refcount_negative" in {k for k, _ in a.violations})


def test_leak_is_reported():
    a = BlockAllocator(3, 16)
    a.allocate()
    assert {k for k, _ in a.quiescent_problems()} == {"block_leak"}


def test_block_hash_is_chained_on_parent():
    """Same tokens under a different parent must hash differently: the hash names the whole prefix."""
    toks = list(range(16))
    assert block_hash(NONE_HASH, toks) == block_hash(NONE_HASH, toks)
    assert block_hash(NONE_HASH, toks) != block_hash(block_hash(NONE_HASH, [9] * 16), toks)


def test_prefix_lookup_stops_at_first_miss():
    a = BlockAllocator(8, 4)
    h1, h2, h3 = 11, 22, 33
    for h in (h1, h2):
        b = a.allocate()
        a.register(b, h)
    assert len(a.lookup([h1, h2, h3])) == 2
    assert a.lookup([h3, h1]) == []


def test_freed_hashed_block_stays_cached_until_evicted():
    """Prefix cache: a ref-0 hashed block is evictable but still hittable."""
    a = BlockAllocator(2, 4)
    b = a.allocate()
    a.register(b, 42)
    a.free(b)
    assert a.num_free() == 2
    assert a.lookup([42]) == [b]
    a.acquire(b)
    assert a.num_free() == 1 and a.blocks[b].ref == 1


def test_lru_eviction_order_and_hash_removed():
    """LRU: the block freed longest ago is reclaimed first, and its hash leaves the cache."""
    a = BlockAllocator(3, 4)
    bids = []
    for h in (1, 2, 3):
        b = a.allocate()
        a.register(b, h)
        bids.append(b)
    for b in bids:                 # freed in order 1, 2, 3 -> 1 is LRU
        a.free(b)
    a.acquire(a.lookup([2])[0])    # touch 2
    a.free(a.lookup([2])[0])       # 2 becomes MRU
    first = a.allocate()
    assert first == bids[0] and a.lookup([1]) == [] and a.evictions == 1
    second = a.allocate()
    assert second == bids[2]       # 3 was freed before 2 was touched again


def test_first_registration_wins():
    a = BlockAllocator(2, 4)
    x, y = a.allocate(), a.allocate()
    a.register(x, 7)
    a.register(y, 7)
    assert a.lookup([7]) == [x]
    a.free(y)
    assert y in a.free_list          # unhashed duplicate goes straight to the free list


def test_swap_space_round_trip_and_capacity():
    s = SwapSpace(1)
    sid = s.store([1, 2], 99)
    assert s.num_free() == 0
    assert s.load(sid) == ([1, 2], 99)
    assert s.num_free() == 1
