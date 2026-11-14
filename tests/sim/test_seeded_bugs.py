"""Each seeded engine bug is caught, by the invariant you would expect,
and slips past the check you might have written instead."""
from __future__ import annotations

import pytest

from servesim.cluster import Cluster
from servesim.engine import EngineConfig
from servesim.invariants import check_run, kinds
from servesim.scenarios import SCENARIOS, qualify, rag_reordered, shared_prefix_pressure
from servesim.versions import (CHUNK_BUDGET_OFF_BY_ONE, PREFIX_HASH_IGNORES_PARENT, SWAP_FREES_SHARED_BLOCK,
                               VERSIONS, bugs_for)

BY_NAME = {s.name: s for s in SCENARIOS}

# version -> (scenario that exposes it, violation kinds it must produce)
EXPECTED = {
    "1.5.0-rc1": ("rag-reordered/prefix-cache", {"output_mismatch"}),
    "1.5.0-rc2": ("pressure/swap/chunked",
                  {"refcount_mismatch", "refcount_negative", "output_mismatch", "block_leak"}),
    "1.5.0-rc3": ("long-prompts/chunk-256", {"budget_exceeded"}),
}


@pytest.mark.parametrize("version", sorted(EXPECTED))
def test_seeded_bug_is_caught(version):
    scenario, must = EXPECTED[version]
    found = kinds(check_run(BY_NAME[scenario].run(version)))
    assert must <= found, f"{version}: expected {must}, got {found}"


@pytest.mark.parametrize("version", ["1.4.0", "1.5.0"])
def test_clean_releases_pass_every_scenario(version):
    assert {k: v for k, v in qualify(version).items() if v} == {}


def test_versions_carry_exactly_one_seeded_bug_each():
    assert bugs_for("1.5.0-rc1") == {PREFIX_HASH_IGNORES_PARENT}
    assert bugs_for("1.5.0-rc2") == {SWAP_FREES_SHARED_BLOCK}
    assert bugs_for("1.5.0-rc3") == {CHUNK_BUDGET_OFF_BY_ONE}
    with pytest.raises(ValueError):
        bugs_for("9.9.9")


def test_prefix_hash_bug_passes_ordinary_shared_prefix_traffic():
    """Shared system prompts always start at position 0, so a token-only hash never
    collides there. Only a reordered-document workload exposes it."""
    r = Cluster.colocated(EngineConfig(version="1.5.0-rc1")).run(shared_prefix_pressure(), audit=True)
    assert r.metrics()["prefix_hit_tokens"] > 0
    assert check_run(r) == []


def test_prefix_hash_bug_is_a_silent_wrong_answer():
    """No crash, no leak, right token count: only the golden comparison sees it."""
    r = Cluster.colocated(EngineConfig(version="1.5.0-rc1")).run(rag_reordered(), audit=True)
    v = check_run(r)
    assert kinds(v) == {"output_mismatch"}
    assert all(len(s.outputs) == s.max_new_tokens for s in r.sequences)
    bad = {x.detail.split(":")[0] for x in v}
    # only requests where the document sits at position 0 are corrupted
    assert bad and all(int(rid[3:]) % 2 == 1 for rid in bad)


def test_swap_bug_needs_both_sharing_and_swap():
    """Recompute preemption or no prefix sharing: the bug path is never taken."""
    assert check_run(BY_NAME["pressure/recompute/chunked"].run("1.5.0-rc2")) == []
    assert check_run(BY_NAME["pressure/swap/no-prefix-cache"].run("1.5.0-rc2")) == []


def test_swap_bug_caught_at_the_step_it_happens():
    """The per-step ref-count audit flags it before any output is corrupted."""
    r = BY_NAME["pressure/swap/chunked"].run("1.5.0-rc2")
    first = r.audit[0]
    assert first[0] == "refcount_mismatch"


def test_chunk_off_by_one_produces_correct_outputs():
    """Outputs are right; the step is just one token over budget. In a real engine that
    overflows a buffer sized for max_num_batched_tokens or misses the CUDA-graph bucket."""
    r = BY_NAME["long-prompts/chunk-256"].run("1.5.0-rc3")
    v = check_run(r)
    assert kinds(v) == {"budget_exceeded"}
    assert r.metrics()["max_step_tokens"] == 257


def test_every_known_version_is_qualified_somewhere():
    assert set(EXPECTED) | {"1.4.0", "1.5.0"} == set(VERSIONS)
