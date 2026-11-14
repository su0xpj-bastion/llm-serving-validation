from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import Counter

import pytest

from servebench import gate, replay
from servebench.metrics import GoodputSLO
from servebench.workload import TraceRequest, generate, prompt_text


def _sse(port: int, body: dict) -> list[str]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return [ln for ln in resp.read().decode().split("\n\n") if ln]


def test_mock_speaks_the_openai_stream_contract(mock):
    """Role chunk, content chunks, finish chunk, usage chunk with choices=[], [DONE]."""
    server, _ = mock
    events = _sse(server.server_port, {"model": "m", "stream": True, "max_tokens": 4,
                                       "stream_options": {"include_usage": True},
                                       "messages": [{"role": "user", "content": "a b c"}]})
    assert events[-1] == "data: [DONE]"
    objs = [json.loads(e[len("data: "):]) for e in events[:-1]]
    assert objs[0]["choices"][0]["delta"]["role"] == "assistant"
    assert sum(1 for o in objs if o["choices"] and o["choices"][0]["delta"].get("content")) == 4
    assert objs[-2]["choices"][0]["finish_reason"] == "length"
    assert objs[-1]["choices"] == [] and objs[-1]["usage"]["completion_tokens"] == 4
    assert objs[-1]["usage"]["prompt_tokens"] == 3


def test_replay_measures_every_request_and_usage_matches(client):
    trace = generate(30, 60.0, seed=1, max_prompt=256, max_output=24)
    recs, finishes = replay.replay(client, trace)
    s = replay.summarize(recs, trace, None)
    assert s["completed"] == 30 and s["error_rate"] == 0
    assert s["usage_mismatch"] == 0
    assert finishes == Counter({"length": 30})
    assert all(len(r.token_times) == r.output_tokens for r in recs)
    # mock: TTFT >= 20 ms base; ITL ~ 2 ms
    assert s["ttft_s"]["p50"] >= 0.02
    assert 0.0015 < s["itl_s"]["p50"] < 0.02


def test_open_loop_sends_on_schedule_not_after_previous_finishes(client):
    """Two requests 50 ms apart must overlap when each takes longer than 50 ms."""
    trace = [TraceRequest("a", 0.0, 10, 60), TraceRequest("b", 0.05, 10, 60)]
    recs, _ = replay.replay(client, trace)
    a, b = recs
    assert b.token_times[0] < a.token_times[-1]
    assert b.arrival - a.arrival == pytest.approx(0.05, abs=1e-6)


def test_shared_prefix_trace_shows_lower_ttft_on_cached_requests(client):
    """Same text prefix -> the mock's prefix cache skips it, as vLLM's would."""
    first = TraceRequest("p0", 0.0, 1100, 4, prefix_group=0, prefix_tokens=1024)
    later = [TraceRequest(f"p{i}", 0.3 + 0.05 * i, 1100, 4, prefix_group=0, prefix_tokens=1024)
             for i in range(1, 6)]
    cold = [TraceRequest(f"c{i}", 0.3 + 0.05 * i, 1100, 4) for i in range(1, 6)]
    recs, _ = replay.replay(client, [first] + later + cold)
    ttft = {r.rid: r.ttft for r in recs}
    warm_avg = sum(ttft[f"p{i}"] for i in range(1, 6)) / 5
    cold_avg = sum(ttft[f"c{i}"] for i in range(1, 6)) / 5
    assert warm_avg < 0.6 * cold_avg


def test_429_is_retried_then_reported(client, mock):
    _, state = mock
    c = client.with_options(default_headers={"x-fault": "429"})
    r = replay.stream_one(c, "m", TraceRequest("x", 0, 3, 3), "a b c", time.perf_counter(), True,
                          Counter(), threading.Lock())
    assert not r.ok and r.error == "rate_limited"
    assert state.faults_served == replay.MAX_ATTEMPTS


def test_5xx_is_not_retried(client, mock):
    _, state = mock
    c = client.with_options(default_headers={"x-fault": "500"})
    r = replay.stream_one(c, "m", TraceRequest("x", 0, 3, 3), "a b c", time.perf_counter(), True,
                          Counter(), threading.Lock())
    assert not r.ok and r.error == "http_500" and state.faults_served == 1


def test_midstream_drop_is_an_error_not_a_short_success(client):
    """A stream that ends without finish_reason must not count as a completed request."""
    c = client.with_options(default_headers={"x-fault": "drop-midstream"})
    r = replay.stream_one(c, "m", TraceRequest("x", 0, 3, 10), "a b c", time.perf_counter(), True,
                          Counter(), threading.Lock())
    assert not r.ok and r.error == "no_finish_reason" and len(r.token_times) == 3


def test_slo_gate_passes_loose_and_fails_tight(client):
    trace = generate(20, 50.0, seed=2, max_prompt=128, max_output=16)
    recs, _ = replay.replay(client, trace)
    s = replay.summarize(recs, trace, GoodputSLO(ttft_s=1.0, tpot_s=0.1))
    assert s["slo_attainment"] == 1.0
    loose = {"max": {"ttft_s.p99": 2.0, "error_rate": 0.0}, "min": {"slo_attainment": 0.9}}
    tight = {"max": {"ttft_s.p99": 0.001, "itl_s.p99": 0.0001}}
    assert gate.check(s, loose) == []
    assert len(gate.check(s, tight)) == 2
    assert "SLO gate: FAIL" in gate.markdown(s, tight, gate.check(s, tight))


def test_prompt_text_word_count_matches_trace(mock):
    server, _ = mock
    r = TraceRequest("w", 0, 77, 2)
    events = _sse(server.server_port, {"model": "m", "stream": True, "max_tokens": 2,
                                       "stream_options": {"include_usage": True},
                                       "messages": [{"role": "user", "content": prompt_text(r)}]})
    assert json.loads(events[-2][len("data: "):])["usage"]["prompt_tokens"] == 77
