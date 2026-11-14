"""Latency math shared by the simulator and the HTTP benchmark client.

No I/O. Both sides produce `RequestRecord`s and call `aggregate()`, so a
metric means the same thing whether it came from the simulator or from
a real vLLM / SGLang / TRT-LLM endpoint.

Definitions (the ones vLLM's and SGLang's benchmark scripts use):
  TTFT  arrival -> first content token
  ITL   gap between consecutive content chunks, pooled across requests
  TPOT  (last token - first token) / (output_tokens - 1), per request
  E2E   arrival -> last token
  goodput  completed requests per second that met BOTH the TTFT and the
           TPOT SLO (DistServe's definition). Throughput counts everything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

PERCENTILES = (50, 90, 99)


@dataclass
class RequestRecord:
    """Client-visible timing of one request. Times are seconds on one clock."""

    rid: str
    arrival: float
    token_times: list[float] = field(default_factory=list)
    output_tokens: int = 0            # from usage when available, else chunk count
    prompt_tokens: int = 0
    ok: bool = True
    error: str = ""

    @property
    def ttft(self) -> float | None:
        return self.token_times[0] - self.arrival if self.token_times else None

    @property
    def itls(self) -> list[float]:
        t = self.token_times
        return [b - a for a, b in zip(t, t[1:])]

    @property
    def tpot(self) -> float | None:
        if len(self.token_times) < 2 or self.output_tokens < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (self.output_tokens - 1)

    @property
    def e2e(self) -> float | None:
        return self.token_times[-1] - self.arrival if self.token_times else None


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: always returns a value that was observed."""
    if not values:
        return 0.0
    if not 0 < pct <= 100:
        raise ValueError(f"pct must be in (0, 100], got {pct}")
    ordered = sorted(values)
    return ordered[math.ceil(pct / 100 * len(ordered)) - 1]


def distribution(values: list[float]) -> dict[str, float]:
    out = {f"p{p}": percentile(values, p) for p in PERCENTILES}
    out["mean"] = sum(values) / len(values) if values else 0.0
    out["max"] = max(values) if values else 0.0
    return out


@dataclass(frozen=True)
class GoodputSLO:
    ttft_s: float
    tpot_s: float


def meets_slo(rec: RequestRecord, slo: GoodputSLO) -> bool:
    if not rec.ok or rec.ttft is None:
        return False
    tpot = rec.tpot
    return rec.ttft <= slo.ttft_s and (tpot is None or tpot <= slo.tpot_s)


def aggregate(records: list[RequestRecord], slo: GoodputSLO | None = None) -> dict:
    """Roll request records up into the metrics an SLO file gates on."""
    done = [r for r in records if r.ok and r.token_times]
    if done:
        start = min(r.arrival for r in records)
        end = max(r.token_times[-1] for r in done)
        duration = max(end - start, 1e-9)
    else:
        duration = 0.0
    out_tokens = sum(r.output_tokens for r in done)
    summary = {
        "requests": len(records),
        "completed": len(done),
        "errors": len(records) - len(done),
        "error_rate": (len(records) - len(done)) / len(records) if records else 0.0,
        "duration_s": duration,
        "output_tokens": out_tokens,
        "request_throughput_rps": len(done) / duration if duration else 0.0,
        "output_throughput_tps": out_tokens / duration if duration else 0.0,
        "ttft_s": distribution([r.ttft for r in done if r.ttft is not None]),
        "itl_s": distribution([g for r in done for g in r.itls]),
        "tpot_s": distribution([r.tpot for r in done if r.tpot is not None]),
        "e2e_s": distribution([r.e2e for r in done if r.e2e is not None]),
    }
    if slo is not None:
        good = sum(1 for r in done if meets_slo(r, slo))
        summary["goodput_rps"] = good / duration if duration else 0.0
        summary["slo_attainment"] = good / len(records) if records else 0.0
    return summary
