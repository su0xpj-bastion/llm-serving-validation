from __future__ import annotations

import json
from pathlib import Path

from servebench.__main__ import main as bench
from servesim.__main__ import main as sim

ROOT = Path(__file__).resolve().parents[2]


def test_gen_run_and_gate_end_to_end(tmp_path, mock):
    server, _ = mock
    trace = tmp_path / "t.jsonl"
    assert bench(["gen", "--n", "25", "--rate", "50", "--max-prompt", "200", "--max-output", "16",
                  "--shared-prefix-ratio", "0.5", "--prefix-tokens", "64", "--out", str(trace)]) == 0
    base = ["run", "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--model", "mock-llm",
            "--trace", str(trace)]
    out, md = tmp_path / "r.json", tmp_path / "s.md"
    assert bench(base + ["--slo", str(ROOT / "slo/mock.json"), "--out", str(out), "--summary-md", str(md)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["completed"] == 25 and data["breaches"] == []
    assert "SLO gate: PASS" in md.read_text(encoding="utf-8")
    tight = tmp_path / "tight.json"
    tight.write_text(json.dumps({"max": {"ttft_s.p50": 0.0001}}), encoding="utf-8")
    assert bench(base + ["--slo", str(tight)]) == 1


def test_qualify_cli_exit_codes(capsys):
    assert sim(["qualify", "--version", "1.5.0"]) == 0
    assert "GO (0/" in capsys.readouterr().out
    assert sim(["qualify", "--version", "1.5.0-rc3"]) == 1
    assert "NO-GO" in capsys.readouterr().out


def test_checked_in_traces_load():
    from servebench.workload import load
    for p in sorted((ROOT / "traces").glob("*.jsonl")):
        assert load(p), p
