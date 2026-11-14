"""Open-loop trace replay against any OpenAI-compatible endpoint.

Requests are sent at their trace arrival times whether or not earlier ones
have finished (open loop: a slow server builds a queue instead of slowing
the client down, which is what production traffic does). Every request
streams; the client timestamps each content chunk.

Works against vLLM, SGLang and TensorRT-LLM (trtllm-serve), which all serve
/v1/chat/completions with `stream_options.include_usage`. `--ignore-eos`
sends `ignore_eos: true` so the server generates exactly `output_tokens`
(vLLM and SGLang accept it; TRT-LLM's OpenAI server does too).

TTFT is measured from the *scheduled* arrival, so client-side queueing and
429 back-off count against the server, as the user would experience it.
"""
from __future__ import annotations

import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import openai

from servebench.metrics import GoodputSLO, RequestRecord, aggregate
from servebench.workload import TraceRequest, prompt_text

MODEL = "mock-llm"
MAX_ATTEMPTS = 3
MAX_CONCURRENCY = 256


def stream_one(client: openai.OpenAI, model: str, req: TraceRequest, text: str, arrival: float,
               ignore_eos: bool, finish_counts: Counter, lock: threading.Lock) -> RequestRecord:
    rec = RequestRecord(rid=req.rid, arrival=arrival)
    usage = None
    finish_reason: str | None = None
    extra = {"ignore_eos": True} if ignore_eos else None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
                max_tokens=req.output_tokens,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra,
            ) as stream:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        rec.token_times.append(time.perf_counter())
                    if chunk.choices and chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                    if chunk.usage:
                        usage = chunk.usage
            break
        except openai.RateLimitError:
            if attempt == MAX_ATTEMPTS - 1:
                rec.ok, rec.error = False, "rate_limited"
                return rec
            time.sleep(2 ** attempt + random.random())
        except openai.APIStatusError as exc:
            rec.ok, rec.error = False, f"http_{exc.status_code}"
            return rec
        except openai.APIError as exc:
            rec.ok, rec.error = False, type(exc).__name__
            return rec
    with lock:
        finish_counts[finish_reason or "none"] += 1
    if finish_reason is None:
        rec.ok, rec.error = False, "no_finish_reason"
    if usage is not None:
        rec.output_tokens = usage.completion_tokens
        rec.prompt_tokens = usage.prompt_tokens
    else:
        rec.output_tokens = len(rec.token_times)
        rec.error = rec.error or "no_usage_chunk"
    return rec


def replay(client: openai.OpenAI, trace: list[TraceRequest], model: str = MODEL, seed: int = 0,
           time_scale: float = 1.0, ignore_eos: bool = True,
           max_concurrency: int = MAX_CONCURRENCY) -> tuple[list[RequestRecord], Counter]:
    """Send every request at arrival_s * time_scale. Returns records + finish_reason counts."""
    texts = {r.rid: prompt_text(r, seed) for r in trace}       # build prompts before the clock starts
    finish_counts: Counter = Counter()
    lock = threading.Lock()
    t0 = time.perf_counter() + 0.05
    futures = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for r in sorted(trace, key=lambda x: x.arrival_s):
            at = t0 + r.arrival_s * time_scale
            delay = at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(stream_one, client, model, r, texts[r.rid], at, ignore_eos,
                                       finish_counts, lock))
    records = [f.result() for f in futures]
    non_stop = {k: v for k, v in finish_counts.items() if k != "stop"}
    if non_stop:
        print(f"[WARN] stream ended with finish_reason={non_stop} "
              f"(length is expected when replaying fixed output lengths)", file=sys.stderr, flush=True)
    return records, finish_counts


def usage_mismatches(records: list[RequestRecord], trace: list[TraceRequest]) -> int:
    """Requests whose server-reported completion tokens != what the trace asked for."""
    want = {r.rid: r.output_tokens for r in trace}
    return sum(1 for r in records if r.ok and r.output_tokens != want[r.rid])


def summarize(records: list[RequestRecord], trace: list[TraceRequest], slo: GoodputSLO | None) -> dict:
    s = aggregate(records, slo)
    s["usage_mismatch"] = usage_mismatches(records, trace)
    s["error_kinds"] = dict(Counter(r.error for r in records if not r.ok))
    return s
