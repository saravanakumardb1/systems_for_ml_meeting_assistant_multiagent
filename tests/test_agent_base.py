"""Tests for agents/base.py — error handling and response robustness."""
import pytest
from unittest.mock import AsyncMock

from agents import base, config, worker_util
from agents.drafter import Drafter
from agents.extractor import Extractor
from agents.summarizer import Summarizer


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


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln


class _FakeStreamClient:
    """Stand-in for httpx.AsyncClient.stream (SSE path)."""
    def __init__(self, lines):
        self._lines = lines
        self.calls = 0

    def stream(self, *args, **kwargs):
        self.calls += 1
        return _FakeStream(self._lines)


# ------------------------------------------------------------------ P4.2 streaming
class TestStreamingPath:
    @pytest.mark.asyncio
    async def test_stream_with_usage_populates_tokens_and_ttft(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,'
            '"total_tokens":14,"prompt_tokens_details":{"cached_tokens":5}}}',
            "data: [DONE]",
        ]
        agent = base.Agent(config.AGENT_SPECS["summarizer"],
                           _FakeStreamClient(lines), stream=True)
        res = await agent.call("x")
        assert res.error is None
        assert res.text == "Hello world"
        assert res.prompt_tokens == 12
        assert res.completion_tokens == 2
        assert res.cached_tokens == 5
        assert res.ttft_s is not None  # first content chunk set TTFT

    @pytest.mark.asyncio
    async def test_stream_without_usage_degrades_gracefully(self):
        """Some servers omit the usage chunk — text still parsed, tokens default 0."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            "data: [DONE]",
        ]
        agent = base.Agent(config.AGENT_SPECS["drafter"],
                           _FakeStreamClient(lines), stream=True)
        res = await agent.call("x")
        assert res.error is None
        assert res.text == "Hi"
        assert res.prompt_tokens == 0
        assert res.total_tokens == 0
        assert res.ttft_s is not None

    @pytest.mark.asyncio
    async def test_stream_skips_non_data_and_blank_lines(self):
        lines = [
            "",                       # blank
            ": keep-alive comment",   # non-data line
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        agent = base.Agent(config.AGENT_SPECS["extractor"],
                           _FakeStreamClient(lines), stream=True)
        res = await agent.call("x")
        assert res.text == "ok"
        assert res.error is None


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


# ------------------------------------------------------------------ P1.3 shared-prefix layout
class TestSharedPrefixLayout:
    def test_default_layout_brief_first_no_task(self):
        body = worker_util.compose_worker_input("BRIEF", "TRANSCRIPT", None)
        assert body.startswith("COORDINATION BRIEF:")
        assert "TASK:" not in body

    def test_shared_layout_transcript_first_task_last(self):
        body = worker_util.compose_worker_input(
            "BRIEF", "TRANSCRIPT", None,
            role_instruction="do the thing", shared_prefix=True)
        assert body.startswith("TRANSCRIPT:")
        assert "TASK:" in body
        assert body.rstrip().endswith("do the thing")  # role instruction at the END

    def test_default_construction_keeps_role_system_prompt(self, monkeypatch):
        monkeypatch.setattr(worker_util.config, "SHARED_PREFIX_LAYOUT", False)
        s = Summarizer(object())
        assert s.shared_prefix is False
        assert s.system_prompt == s.role_instruction  # no swap in default mode

    def test_shared_construction_swaps_to_common_preamble(self, monkeypatch):
        monkeypatch.setattr(worker_util.config, "SHARED_PREFIX_LAYOUT", True)
        s, e, d = Summarizer(object()), Extractor(object()), Drafter(object())
        # all three workers now share an IDENTICAL system preamble ...
        assert s.system_prompt == e.system_prompt == d.system_prompt
        # ... while each retains its distinct role instruction
        assert len({s.role_instruction, e.role_instruction, d.role_instruction}) == 3
        # composed bodies share an identical (transcript+brief) leading prefix
        prefixes = [
            worker_util.compose_worker_input(
                "BRIEF", "TRANSCRIPT", None,
                role_instruction=a.role_instruction, shared_prefix=True
            ).split("TASK:")[0]
            for a in (s, e, d)
        ]
        assert prefixes[0] == prefixes[1] == prefixes[2]
