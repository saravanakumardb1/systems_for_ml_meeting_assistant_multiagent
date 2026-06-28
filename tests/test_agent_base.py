"""Tests for agents/base.py — error handling and response robustness."""
import pytest
from unittest.mock import AsyncMock

from agents import base, config


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _FakePostClient:
    """Minimal stand-in for httpx.AsyncClient.post (non-stream path)."""
    def __init__(self, payload):
        self._p = payload
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        return _FakeResp(self._p)


# ------------------------------------------------------------------ P3.1
class TestMalformedResponseRetry:
    @pytest.mark.asyncio
    async def test_empty_choices_is_retried_then_errored(self, monkeypatch):
        """obj['choices'][0] on [] raises IndexError — must be caught + retried,
        not propagated, and end as an errored CallResult."""
        monkeypatch.setattr(base.asyncio, "sleep", AsyncMock())
        monkeypatch.setattr(base.config, "MAX_RETRIES", 2)
        client = _FakePostClient({"choices": []})
        agent = base.Agent(config.AGENT_SPECS["summarizer"], client, stream=False)

        res = await agent.call("hello")

        assert res.error is not None
        assert "IndexError" in res.error
        assert res.text == ""
        assert client.calls == 2  # retried up to MAX_RETRIES

    @pytest.mark.asyncio
    async def test_missing_message_key_is_retried(self, monkeypatch):
        """Missing 'message' key raises KeyError — already retryable."""
        monkeypatch.setattr(base.asyncio, "sleep", AsyncMock())
        monkeypatch.setattr(base.config, "MAX_RETRIES", 2)
        client = _FakePostClient({"choices": [{}]})  # no 'message'
        agent = base.Agent(config.AGENT_SPECS["extractor"], client, stream=False)

        res = await agent.call("hello")

        assert res.error is not None
        assert client.calls == 2

    @pytest.mark.asyncio
    async def test_well_formed_response_succeeds(self, monkeypatch):
        """Sanity: a valid non-stream response yields text + usage with no error."""
        monkeypatch.setattr(base.asyncio, "sleep", AsyncMock())
        payload = {
            "choices": [{"message": {"content": "hi there"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                      "total_tokens": 13,
                      "prompt_tokens_details": {"cached_tokens": 4}},
        }
        client = _FakePostClient(payload)
        agent = base.Agent(config.AGENT_SPECS["drafter"], client, stream=False)

        res = await agent.call("hello")

        assert res.error is None
        assert res.text == "hi there"
        assert res.prompt_tokens == 10
        assert res.cached_tokens == 4
        assert client.calls == 1
