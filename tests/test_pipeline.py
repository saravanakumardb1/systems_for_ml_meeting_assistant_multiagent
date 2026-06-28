"""Tests for pipeline revision loop logic.

Uses a lightweight mock agent to verify:
  - revision loop terminates at MAX_REVISIONS
  - selective re-execution only re-runs flagged workers
  - sequential and parallel modes produce consistent revision counts
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import config
from agents.base import CallResult
from agents.reviewer import Verdict


# ------------------------------------------------------------------ helpers
def _make_call(agent: str, text: str = "ok", error: str | None = None) -> CallResult:
    return CallResult(agent=agent, text=text, prompt_tokens=10,
                      completion_tokens=5, total_tokens=15, e2e_s=0.1,
                      start_ts=0.0, end_ts=0.1, error=error)


def _pass_verdict() -> tuple[CallResult, Verdict]:
    r = _make_call("reviewer", json.dumps(
        {"status": "pass", "issues": {"summarizer": "", "extractor": "", "drafter": ""}}))
    return r, Verdict(status="pass", issues={}, raw=r.text)


def _revise_verdict(who: str) -> tuple[CallResult, Verdict]:
    issues = {"summarizer": "", "extractor": "", "drafter": ""}
    issues[who] = f"Fix {who} output"
    r = _make_call("reviewer", json.dumps({"status": "revise", "issues": issues}))
    return r, Verdict(status="revise", issues=issues, raw=r.text)


# ------------------------------------------------------------------ sequential
class TestSequentialRevisionLoop:
    @pytest.mark.asyncio
    async def test_no_revision_when_reviewer_passes(self):
        from pipeline.sequential import run_sequential
        with patch("pipeline.sequential.AgentBundle") as MockBundle:
            bundle = MagicMock()
            MockBundle.return_value = bundle
            bundle.coordinator.run = AsyncMock(return_value=_make_call("coordinator", "brief"))
            bundle.summarizer.run  = AsyncMock(return_value=_make_call("summarizer"))
            bundle.extractor.run   = AsyncMock(return_value=_make_call("extractor"))
            bundle.drafter.run     = AsyncMock(return_value=_make_call("drafter"))
            bundle.reviewer.run    = AsyncMock(return_value=_pass_verdict())

            from bench.trace import Tracer
            import httpx
            async with httpx.AsyncClient() as client:
                result = await run_sequential("transcript", "small", client,
                                              Tracer(), stream=False)

        assert result.revisions == 0
        assert bundle.summarizer.run.call_count == 1

    @pytest.mark.asyncio
    async def test_revision_loop_terminates_at_max(self):
        """Reviewer always says revise → loop must stop at MAX_REVISIONS."""
        from pipeline.sequential import run_sequential
        with patch("pipeline.sequential.AgentBundle") as MockBundle:
            bundle = MagicMock()
            MockBundle.return_value = bundle
            bundle.coordinator.run = AsyncMock(return_value=_make_call("coordinator", "brief"))
            bundle.summarizer.run  = AsyncMock(return_value=_make_call("summarizer"))
            bundle.extractor.run   = AsyncMock(return_value=_make_call("extractor"))
            bundle.drafter.run     = AsyncMock(return_value=_make_call("drafter"))
            # Always return revise for summarizer
            bundle.reviewer.run = AsyncMock(
                return_value=_revise_verdict("summarizer"))

            from bench.trace import Tracer
            import httpx
            async with httpx.AsyncClient() as client:
                result = await run_sequential("transcript", "small", client,
                                              Tracer(), stream=False)

        assert result.revisions == config.MAX_REVISIONS
        # workers run: 1 initial + MAX_REVISIONS revision rounds
        assert bundle.summarizer.run.call_count == 1 + config.MAX_REVISIONS


# ------------------------------------------------------------------ parallel selective re-execution
class TestSelectiveReExecution:
    @pytest.mark.asyncio
    async def test_only_flagged_worker_rerun(self):
        """When only extractor is flagged, summarizer and drafter must NOT be re-run."""
        from pipeline.parallel import run_parallel
        with patch("pipeline.parallel.AgentBundle") as MockBundle:
            bundle = MagicMock()
            MockBundle.return_value = bundle
            bundle.coordinator.run = AsyncMock(return_value=_make_call("coordinator", "brief"))
            bundle.summarizer.run  = AsyncMock(return_value=_make_call("summarizer"))
            bundle.drafter.run     = AsyncMock(return_value=_make_call("drafter"))
            bundle.extractor.run   = AsyncMock(return_value=_make_call("extractor"))
            # first review: revise extractor only; second review: pass
            bundle.reviewer.run = AsyncMock(side_effect=[
                _revise_verdict("extractor"),
                _pass_verdict(),
            ])

            from bench.trace import Tracer
            import httpx
            async with httpx.AsyncClient() as client:
                await run_parallel("transcript", "small", client,
                                   Tracer(), stream=False)

        # summarizer and drafter called once (initial only — not on revision)
        assert bundle.summarizer.run.call_count == 1
        assert bundle.drafter.run.call_count    == 1
        # extractor called twice (initial + 1 revision)
        assert bundle.extractor.run.call_count  == 2


# ------------------------------------------------------------------ sequential selective (P2.1)
class TestSequentialSelectiveReExecution:
    @pytest.mark.asyncio
    async def test_only_flagged_worker_rerun_and_no_double_count(self):
        """Sequential must re-run ONLY the flagged worker (parity with parallel),
        and carried-over workers must not be double-counted in result.calls (P2.1)."""
        from pipeline.sequential import run_sequential
        with patch("pipeline.sequential.AgentBundle") as MockBundle:
            bundle = MagicMock()
            MockBundle.return_value = bundle
            bundle.coordinator.run = AsyncMock(return_value=_make_call("coordinator", "brief"))
            bundle.summarizer.run  = AsyncMock(return_value=_make_call("summarizer"))
            bundle.extractor.run   = AsyncMock(return_value=_make_call("extractor"))
            bundle.drafter.run     = AsyncMock(return_value=_make_call("drafter"))
            bundle.reviewer.run = AsyncMock(side_effect=[
                _revise_verdict("extractor"),   # flag extractor only
                _pass_verdict(),
            ])

            from bench.trace import Tracer
            import httpx
            async with httpx.AsyncClient() as client:
                result = await run_sequential("transcript", "small", client,
                                              Tracer(), stream=False)

        assert bundle.summarizer.run.call_count == 1   # not re-run
        assert bundle.drafter.run.call_count    == 1   # not re-run
        assert bundle.extractor.run.call_count  == 2   # initial + 1 revision
        # calls = coordinator(1) + initial workers(3) + reviewer(1)
        #         + rerun extractor(1) + reviewer(1) = 7  (no carried-over double count)
        assert len(result.calls) == 7


# ------------------------------------------------------------------ langgraph revision parity
class TestLangGraphRevisionLoop:
    @pytest.mark.asyncio
    async def test_revisions_match_sequential_at_cap(self):
        """Reviewer always says revise → langgraph must report the SAME revision
        count as sequential/parallel (== MAX_REVISIONS), not MAX_REVISIONS+1 (P2.2)."""
        pytest.importorskip("langgraph")
        from pipeline.langgraph_pipeline import run_langgraph
        with patch("pipeline.langgraph_pipeline.AgentBundle") as MockBundle:
            bundle = MagicMock()
            MockBundle.return_value = bundle
            bundle.coordinator.run = AsyncMock(return_value=_make_call("coordinator", "brief"))
            bundle.summarizer.run  = AsyncMock(return_value=_make_call("summarizer"))
            bundle.extractor.run   = AsyncMock(return_value=_make_call("extractor"))
            bundle.drafter.run     = AsyncMock(return_value=_make_call("drafter"))
            bundle.reviewer.run    = AsyncMock(return_value=_revise_verdict("summarizer"))

            from bench.trace import Tracer
            import httpx
            async with httpx.AsyncClient() as client:
                result = await run_langgraph("transcript", "small", client,
                                             Tracer(), stream=False)

        assert result.revisions == config.MAX_REVISIONS
        # langgraph re-runs all workers each round: 1 initial + MAX_REVISIONS reruns
        assert bundle.summarizer.run.call_count == 1 + config.MAX_REVISIONS


# ------------------------------------------------------------------ errored run guard
class TestErroredRunNotPass:
    @pytest.mark.asyncio
    async def test_errored_worker_downgrades_pass_to_error(self):
        """A worker that errored must NOT be reported as a clean 'pass' even if the
        reviewer returns pass over the empty artifact (P3.2)."""
        from pipeline.sequential import run_sequential
        with patch("pipeline.sequential.AgentBundle") as MockBundle:
            bundle = MagicMock()
            MockBundle.return_value = bundle
            bundle.coordinator.run = AsyncMock(return_value=_make_call("coordinator", "brief"))
            # summarizer errored (empty text + error set)
            bundle.summarizer.run  = AsyncMock(
                return_value=_make_call("summarizer", text="", error="timeout"))
            bundle.extractor.run   = AsyncMock(return_value=_make_call("extractor"))
            bundle.drafter.run     = AsyncMock(return_value=_make_call("drafter"))
            bundle.reviewer.run    = AsyncMock(return_value=_pass_verdict())

            from bench.trace import Tracer
            import httpx
            async with httpx.AsyncClient() as client:
                result = await run_sequential("transcript", "small", client,
                                              Tracer(), stream=False)

        assert result.errored is True
        assert result.verdict_status == "error"  # not "pass"

    @pytest.mark.asyncio
    async def test_clean_run_keeps_pass(self):
        """No errors → finalize() leaves a genuine 'pass' untouched."""
        from pipeline.sequential import run_sequential
        with patch("pipeline.sequential.AgentBundle") as MockBundle:
            bundle = MagicMock()
            MockBundle.return_value = bundle
            bundle.coordinator.run = AsyncMock(return_value=_make_call("coordinator", "brief"))
            bundle.summarizer.run  = AsyncMock(return_value=_make_call("summarizer"))
            bundle.extractor.run   = AsyncMock(return_value=_make_call("extractor"))
            bundle.drafter.run     = AsyncMock(return_value=_make_call("drafter"))
            bundle.reviewer.run    = AsyncMock(return_value=_pass_verdict())

            from bench.trace import Tracer
            import httpx
            async with httpx.AsyncClient() as client:
                result = await run_sequential("transcript", "small", client,
                                              Tracer(), stream=False)

        assert result.errored is False
        assert result.verdict_status == "pass"
