"""Tests for scripts/mock_vllm_server.py — the /reset_cache hook (P1.1 / P4.5).

Skipped unless the optional `mock` extra (fastapi) is installed.
"""
import importlib.util
import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mock_module():
    path = os.path.join(ROOT, "scripts", "mock_vllm_server.py")
    spec = importlib.util.spec_from_file_location("mock_vllm_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chat(client):
    return client.post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "summarizer"},
                     {"role": "user", "content": "hello"}],
        "stream": False,
    })


class TestMockResetCache:
    def test_reset_clears_counters_and_seen_prefixes(self):
        mod = _load_mock_module()
        client = TestClient(mod.app)

        # Prime: one chat call populates counters + the seen-prefix set.
        assert _chat(client).status_code == 200
        assert mod.STATE["prompt_tokens_total"] > 0
        assert len(mod._SEEN_PREFIXES) > 0

        r = client.post("/reset_cache")
        assert r.status_code == 200
        assert r.json()["status"] == "reset"

        # State fully cleared so the offline path doesn't carry cross-run warming.
        assert mod.STATE["prompt_tokens_total"] == 0.0
        assert mod.STATE["prefix_cache_hits_total"] == 0.0
        assert mod._SEEN_PREFIXES == set()

    def test_prefix_cache_warms_then_resets(self):
        """Second identical call is billed as cached; after reset it is cold again."""
        mod = _load_mock_module()
        client = TestClient(mod.app)

        _chat(client)                       # cold: no cached tokens
        second = _chat(client).json()
        assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0

        client.post("/reset_cache")
        after_reset = _chat(client).json()
        assert after_reset["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
