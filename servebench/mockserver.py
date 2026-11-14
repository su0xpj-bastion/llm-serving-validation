"""A small OpenAI-compatible streaming server for testing the replay client.

It is a timing fixture, not a performance model: each request gets
    TTFT = ttft_base_s + prefill_s_per_token * uncached_prompt_tokens
    ITL  = itl_s
independent of load. "Uncached" uses a toy prefix cache keyed on chained
hashes of 16-word blocks, so shared-prefix traces show a TTFT drop on the
mock the same way they would on vLLM with --enable-prefix-caching.
Tokens are whitespace words, one per SSE chunk.

Faults (request header `x-fault`): `429` (rate limited), `500`,
`drop-midstream` (close the socket after a few tokens, no [DONE]).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENCODING = "utf-8"
BLOCK_WORDS = 16
MODEL = "mock-llm"


@dataclass
class MockConfig:
    ttft_base_s: float = 0.005
    prefill_s_per_token: float = 0.00002
    itl_s: float = 0.002
    prefix_cache_blocks: int = 4096


class MockState:
    def __init__(self, cfg: MockConfig) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.cache: OrderedDict[bytes, None] = OrderedDict()
        self.requests = 0
        self.faults_served = 0

    def uncached_tokens(self, words: list[str]) -> int:
        """Longest cached block-prefix; insert the rest. Returns tokens to prefill."""
        parent = b""
        hit = 0
        keys = []
        for i in range(0, len(words) - len(words) % BLOCK_WORDS, BLOCK_WORDS):
            parent = hashlib.blake2b(parent + " ".join(words[i:i + BLOCK_WORDS]).encode(ENCODING),
                                     digest_size=8).digest()
            keys.append(parent)
        with self.lock:
            for k in keys:
                if k not in self.cache:
                    break
                self.cache.move_to_end(k)
                hit += 1
            for k in keys[hit:]:
                self.cache[k] = None
            while len(self.cache) > self.cfg.prefix_cache_blocks:
                self.cache.popitem(last=False)
        return len(words) - hit * BLOCK_WORDS


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: MockState

    def log_message(self, fmt: str, *args) -> None:   # keep test output clean
        return

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode(ENCODING)
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        if code == 429:
            self.send_header("retry-after", "0")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "mock"}]})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        st = self.state
        with st.lock:
            st.requests += 1
        fault = self.headers.get("x-fault", "")
        if fault in ("429", "500"):
            with st.lock:
                st.faults_served += 1
            kind = "rate_limit_exceeded" if fault == "429" else "server_error"
            self._json(int(fault), {"error": {"message": f"injected {fault}", "type": kind}})
            return
        words = " ".join(m.get("content", "") for m in body.get("messages", [])).split()
        n_out = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16)
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        ttft = st.cfg.ttft_base_s + st.cfg.prefill_s_per_token * st.uncached_tokens(words)

        if not body.get("stream"):
            time.sleep(ttft + st.cfg.itl_s * (n_out - 1))
            self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": MODEL,
                             "choices": [{"index": 0, "finish_reason": "length",
                                          "message": {"role": "assistant", "content": " ".join(["tok"] * n_out)}}],
                             "usage": {"prompt_tokens": len(words), "completion_tokens": n_out,
                                       "total_tokens": len(words) + n_out}})
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        def chunk(delta: dict, finish: str | None = None, usage: dict | None = None, choices: bool = True) -> None:
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []}
            if usage is not None:
                obj["usage"] = usage
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode(ENCODING))
            self.wfile.flush()

        try:
            chunk({"role": "assistant", "content": ""})
            time.sleep(ttft)
            for i in range(n_out):
                if i:
                    time.sleep(st.cfg.itl_s)
                if fault == "drop-midstream" and i == 3:
                    self.close_connection = True
                    return
                chunk({"content": f" tok{i}" if i else "tok0"})
            chunk({}, finish="length")
            if include_usage:
                chunk({}, usage={"prompt_tokens": len(words), "completion_tokens": n_out,
                                 "total_tokens": len(words) + n_out}, choices=False)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def serve(host: str = "127.0.0.1", port: int = 0, cfg: MockConfig | None = None) -> tuple[ThreadingHTTPServer, MockState]:
    """Start in a daemon thread. Returns (server, state); server.server_port is the bound port."""
    state = MockState(cfg or MockConfig())
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state
