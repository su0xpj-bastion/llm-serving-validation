"""Engine versions and the defects seeded into them.

Each seeded defect is a realistic engine bug that a basic "does it serve
requests" smoke test passes. `tests/sim/test_seeded_bugs.py` shows which
invariant catches each one, and that the clean releases pass.
"""
from __future__ import annotations

PREFIX_HASH_IGNORES_PARENT = "prefix_hash_ignores_parent"
SWAP_FREES_SHARED_BLOCK = "swap_frees_shared_block"
CHUNK_BUDGET_OFF_BY_ONE = "chunk_budget_off_by_one"

BUG_DESCRIPTIONS = {
    PREFIX_HASH_IGNORES_PARENT:
        "block hash covers only the block's own tokens, not the parent hash, so an "
        "identical 16-token block at a different position reuses KV computed under "
        "a different prefix",
    SWAP_FREES_SHARED_BLOCK:
        "swap-out leaves shared prefix blocks resident (correct) but also drops the "
        "preempted sequence's reference to them, so the block can be evicted and "
        "reused while the swapped sequence still points at it",
    CHUNK_BUDGET_OFF_BY_ONE:
        "chunked-prefill chunk size is computed as remaining budget + 1, so a step "
        "can schedule one token more than max_num_batched_tokens",
}

VERSIONS: dict[str, frozenset[str]] = {
    "1.4.0": frozenset(),
    "1.5.0-rc1": frozenset({PREFIX_HASH_IGNORES_PARENT}),
    "1.5.0-rc2": frozenset({SWAP_FREES_SHARED_BLOCK}),
    "1.5.0-rc3": frozenset({CHUNK_BUDGET_OFF_BY_ONE}),
    "1.5.0": frozenset(),
}
LATEST = "1.5.0"


def bugs_for(version: str) -> frozenset[str]:
    try:
        return VERSIONS[version]
    except KeyError:
        raise ValueError(f"unknown engine version {version!r}; known: {sorted(VERSIONS)}") from None
