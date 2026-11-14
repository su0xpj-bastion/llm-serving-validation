from __future__ import annotations

import openai
import pytest

from servebench.mockserver import MockConfig, serve


@pytest.fixture()
def mock():
    server, state = serve(cfg=MockConfig(ttft_base_s=0.02, prefill_s_per_token=0.0001, itl_s=0.002))
    yield server, state
    server.shutdown()


@pytest.fixture()
def client(mock):
    server, _ = mock
    c = openai.OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test", max_retries=0)
    yield c
    c.close()
