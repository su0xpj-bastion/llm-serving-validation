"""A stand-in "model" whose output depends on the KV cache it actually reads.

Each KV slot holds a 64-bit state chained from the previous slot, the way
real K/V at position i depends on every token before it:

    kv[i] = mix(kv[i-1], token[i])       next_token = kv[last] % VOCAB

The engine writes these states into physical blocks and reads them back
through the sequence's block table. So if the engine hands a sequence the
wrong block (a prefix-hash collision, a block reused after it was freed, a
dropped chunk), the generated tokens diverge from `golden()`, which never
touches the engine. That is the property that makes "a prefix-cache hit
returns the same output as a cold run" a real test, not a tautology.
"""
from __future__ import annotations

VOCAB = 32000
SEED_STATE = 0x6A09E667F3BCC908
_M = (1 << 64) - 1


def mix(state: int, token: int) -> int:
    """splitmix64 finaliser over (state, token): cheap and well-distributed."""
    x = (state ^ ((token + 1) * 0x9E3779B97F4A7C15)) & _M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M
    return x ^ (x >> 31)


def sample(state: int) -> int:
    """Greedy decode: a deterministic function of the last KV state."""
    return state % VOCAB


def golden(prompt: list[int], max_new_tokens: int) -> list[int]:
    """Reference output computed straight through, with no engine involved."""
    state = SEED_STATE
    for t in prompt:
        state = mix(state, t)
    out = []
    for _ in range(max_new_tokens):
        tok = sample(state)
        out.append(tok)
        state = mix(state, tok)
    return out
