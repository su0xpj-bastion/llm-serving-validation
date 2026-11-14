"""Workload traces: arrivals, length distributions, shared-prefix groups.

A trace is JSONL, one request per line:
  {"rid": "r0007", "arrival_s": 0.913, "prompt_tokens": 412, "output_tokens": 96,
   "prefix_group": 2, "prefix_tokens": 256}

The same trace drives the simulator (materialised as token ids) and the
HTTP replay client (materialised as text whose word count is the token
count). Requests in one prefix group share their first `prefix_tokens`
tokens exactly, which is what a prefix cache keys on.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

ENCODING = "utf-8"
VOCAB = 32000

# ShareGPT-like lengths. ShareGPT V3 (as used by vLLM's benchmark_serving)
# has a long-tailed prompt length with a median of a few hundred tokens and
# an output median of roughly 200. A clipped lognormal reproduces that shape.
SHAREGPT_PROMPT = (5.6, 0.9)    # lognormal (mu, sigma): median ~270
SHAREGPT_OUTPUT = (5.2, 0.8)    # median ~180


@dataclass(frozen=True)
class TraceRequest:
    rid: str
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    prefix_group: int | None = None
    prefix_tokens: int = 0


def poisson_arrivals(n: int, rate: float, rng: random.Random) -> list[float]:
    """Exponential inter-arrivals: the memoryless baseline (CV = 1)."""
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def bursty_arrivals(n: int, rate: float, rng: random.Random, burstiness: float = 0.25) -> list[float]:
    """Gamma inter-arrivals with the same mean rate as Poisson.

    shape = burstiness; shape < 1 gives CV = 1/sqrt(shape) > 1, so requests
    clump into bursts with long gaps between them (vLLM's --burstiness).
    """
    if burstiness <= 0:
        raise ValueError("burstiness must be > 0")
    scale = 1.0 / (rate * burstiness)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.gammavariate(burstiness, scale)
        out.append(t)
    return out


def lognormal_len(rng: random.Random, mu_sigma: tuple[float, float], lo: int, hi: int) -> int:
    mu, sigma = mu_sigma
    return max(lo, min(hi, int(round(math.exp(rng.gauss(mu, sigma))))))


def generate(
    n: int,
    rate: float,
    seed: int = 0,
    arrival: str = "poisson",
    burstiness: float = 0.25,
    prompt_dist: tuple[float, float] = SHAREGPT_PROMPT,
    output_dist: tuple[float, float] = SHAREGPT_OUTPUT,
    max_prompt: int = 2048,
    max_output: int = 512,
    shared_prefix_ratio: float = 0.0,
    prefix_groups: int = 4,
    prefix_tokens: int = 256,
) -> list[TraceRequest]:
    """Build a trace. `shared_prefix_ratio` is the fraction of requests that
    carry one of `prefix_groups` shared prefixes of `prefix_tokens` tokens
    (system prompts, few-shot templates); the prompt length includes it."""
    if not 0.0 <= shared_prefix_ratio <= 1.0:
        raise ValueError("shared_prefix_ratio must be in [0, 1]")
    rng = random.Random(seed)
    if arrival == "poisson":
        times = poisson_arrivals(n, rate, rng)
    elif arrival == "bursty":
        times = bursty_arrivals(n, rate, rng, burstiness)
    else:
        raise ValueError(f"unknown arrival process {arrival!r}")
    out = []
    for i, t in enumerate(times):
        plen = lognormal_len(rng, prompt_dist, 4, max_prompt)
        olen = lognormal_len(rng, output_dist, 1, max_output)
        group, ptoks = None, 0
        if rng.random() < shared_prefix_ratio:
            group = rng.randrange(prefix_groups)
            ptoks = prefix_tokens
            plen = max(plen, prefix_tokens + 1)
        out.append(TraceRequest(f"r{i:05d}", round(t, 6), plen, olen, group, ptoks))
    return out


def prompt_ids(req: TraceRequest, seed: int = 0) -> list[int]:
    """Deterministic token ids. Same group -> identical first prefix_tokens."""
    ids: list[int] = []
    if req.prefix_group is not None and req.prefix_tokens:
        g = random.Random(f"prefix-{seed}-{req.prefix_group}")
        ids = [g.randrange(VOCAB) for _ in range(req.prefix_tokens)]
    r = random.Random(f"body-{seed}-{req.rid}")
    ids += [r.randrange(VOCAB) for _ in range(req.prompt_tokens - len(ids))]
    return ids


_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
          "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa")


def prompt_text(req: TraceRequest, seed: int = 0) -> str:
    """Text whose whitespace word count equals prompt_tokens. On a real
    tokenizer the token count drifts (usually 1-1.3 tokens per word); the
    replay client reports the server's usage.prompt_tokens next to it."""
    return " ".join(_WORDS[t % len(_WORDS)] + str(t % 97) for t in prompt_ids(req, seed))


def save(trace: list[TraceRequest], path: Path) -> None:
    with path.open("w", encoding=ENCODING) as fh:
        for r in trace:
            fh.write(json.dumps(asdict(r)) + "\n")


def load(path: Path) -> list[TraceRequest]:
    with path.open("r", encoding=ENCODING) as fh:
        return [TraceRequest(**json.loads(line)) for line in fh if line.strip()]
