"""servebench CLI.

  python -m servebench gen  --n 200 --rate 8 --arrival bursty --shared-prefix-ratio 0.5 --out trace.jsonl
  python -m servebench mock --port 8000
  python -m servebench run  --base-url http://localhost:8000/v1 --model M --trace trace.jsonl \
                            --slo slo/mock.json --out results.json --summary-md summary.md

`run` exits 1 when the SLO gate fails, 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ENCODING = "utf-8"


def cmd_gen(a: argparse.Namespace) -> int:
    from servebench import workload
    trace = workload.generate(a.n, a.rate, seed=a.seed, arrival=a.arrival, burstiness=a.burstiness,
                              max_prompt=a.max_prompt, max_output=a.max_output,
                              shared_prefix_ratio=a.shared_prefix_ratio, prefix_groups=a.prefix_groups,
                              prefix_tokens=a.prefix_tokens)
    workload.save(trace, Path(a.out))
    print(f"wrote {len(trace)} requests over {trace[-1].arrival_s:.1f}s to {a.out}")
    return 0


def cmd_mock(a: argparse.Namespace) -> int:
    from servebench.mockserver import MockConfig, serve
    server, _ = serve(a.host, a.port, MockConfig(ttft_base_s=a.ttft_ms / 1e3, itl_s=a.itl_ms / 1e3))
    print(f"mock OpenAI server on http://{a.host}:{server.server_port}/v1", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    import openai

    from servebench import gate, replay, workload
    trace = workload.load(Path(a.trace))
    if a.max_requests:
        trace = trace[: a.max_requests]
    slo = gate.load_slo(Path(a.slo)) if a.slo else None
    client = openai.OpenAI(base_url=a.base_url, api_key=os.environ.get(a.api_key_env, "EMPTY"),
                           max_retries=0, timeout=a.timeout)
    records, finishes = replay.replay(client, trace, model=a.model, seed=a.seed, time_scale=a.time_scale,
                                      ignore_eos=not a.no_ignore_eos)
    summary = replay.summarize(records, trace, gate.goodput_slo(slo) if slo else None)
    summary["finish_reasons"] = dict(finishes)
    breaches = gate.check(summary, slo) if slo else []
    md = gate.markdown(summary, slo, breaches)
    print(md)
    for target in (a.out, a.summary_md):
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
    if a.out:
        with Path(a.out).open("w", encoding=ENCODING) as fh:
            json.dump({"summary": summary, "breaches": breaches,
                       "requests": [asdict(r) for r in records]}, fh, indent=1)
    if a.summary_md:
        with Path(a.summary_md).open("w", encoding=ENCODING) as fh:
            fh.write(f"### Replay: `{a.trace}` against `{a.base_url}` ({a.model})\n\n{md}\n")
    return 1 if breaches else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="servebench")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a workload trace (JSONL)")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--rate", type=float, default=8.0, help="mean requests/s")
    g.add_argument("--arrival", choices=("poisson", "bursty"), default="poisson")
    g.add_argument("--burstiness", type=float, default=0.25, help="gamma shape for bursty (<1 = burstier)")
    g.add_argument("--max-prompt", type=int, default=2048)
    g.add_argument("--max-output", type=int, default=512)
    g.add_argument("--shared-prefix-ratio", type=float, default=0.0)
    g.add_argument("--prefix-groups", type=int, default=4)
    g.add_argument("--prefix-tokens", type=int, default=256)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)

    m = sub.add_parser("mock", help="run the mock OpenAI-compatible server")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8000)
    m.add_argument("--ttft-ms", type=float, default=5.0)
    m.add_argument("--itl-ms", type=float, default=2.0)

    r = sub.add_parser("run", help="replay a trace against an endpoint and gate on an SLO file")
    r.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    r.add_argument("--model", required=True)
    r.add_argument("--trace", required=True)
    r.add_argument("--slo")
    r.add_argument("--out")
    r.add_argument("--summary-md")
    r.add_argument("--seed", type=int, default=0, help="must match the seed the prompts were built with")
    r.add_argument("--time-scale", type=float, default=1.0, help="<1 compresses arrivals (more load)")
    r.add_argument("--max-requests", type=int, default=0)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--api-key-env", default="OPENAI_API_KEY")
    r.add_argument("--no-ignore-eos", action="store_true",
                   help="do not send ignore_eos (for servers that reject unknown fields)")

    a = p.parse_args(argv)
    handlers = {"gen": cmd_gen, "mock": cmd_mock, "run": cmd_run}
    return handlers[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
