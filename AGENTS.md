# llm-serving-validation — agent guardrails

## Scope
- `servesim/` is the system under test. The seeded bugs in `versions.py` are intentional; never "fix" them in the rc versions.
  A new seeded bug gets a new version plus a test in `tests/sim/test_seeded_bugs.py` naming the invariant that catches it.
- The golden (`servesim/model.py::golden`) must never import engine, cache or scheduler code.
- Performance tests assert behaviour with margins and a stated trade-off, never exact snapshots.
  If an experiment changes, update the README table from `python -m servesim experiments`.

## OpenAI-compatible client (`servebench/replay.py`)
- Chat Completions only: `client.chat.completions.create()`. Never Assistants, Responses, Batch or `.stream()`.
- `stream=True` + `stream_options={"include_usage": True}`, consumed inside `with ... as stream:`.
- Guard `chunk.choices` before `chunk.choices[0]`. Three separate if-blocks: content / finish_reason / usage. No hasattr/getattr/isinstance on chunks.
- Exceptions in order: `openai.RateLimitError` (retry, `2 ** attempt + random.random()`) → `openai.APIStatusError` → `openai.APIError`.
- `MAX_ATTEMPTS`, never `MAX_RETRIES`. Client built with `max_retries=0`, in `main()` or fixtures, never at module level.

## Hygiene
- `from __future__ import annotations` on line 1 of every module. Module-level `ENCODING`, `MODEL`.
- JSON/CSV via `Path.open("r", encoding=ENCODING)`. Bare `raise`. No pandas/numpy.

## Done means
`python -m pytest` green; `python -m servesim qualify` is GO for 1.4.0 and 1.5.0 and NO-GO for every rc.
Never claim real-GPU validation: everything here is simulated or run against the mock.
