"""Sequential baseline: coordinator -> summarizer -> extractor -> drafter ->
reviewer, all serial. Reflection loop re-runs flagged workers (still serially)
up to MAX_REVISIONS times.
"""
from __future__ import annotations

import httpx

from agents import config
from bench.trace import Tracer
from pipeline.base import AgentBundle, PipelineResult


async def run_sequential(transcript: str, transcript_size: str,
                         client: httpx.AsyncClient, tracer: Tracer,
                         stream: bool = True) -> PipelineResult:
    agents = AgentBundle(client, stream=stream)
    result = PipelineResult(mode="sequential", transcript_size=transcript_size)

    with tracer.span("coordinator", kind="phase"):
        coord = await agents.coordinator.run(transcript)
    result.add(coord)
    tracer.add_span(coord.to_span())
    brief = coord.text

    async def run_workers(critiques: dict[str, str] | None, rev: int):
        critiques = critiques or {}
        with tracer.span("summarizer", kind="phase"):
            s = await agents.summarizer.run(brief, transcript,
                                            critiques.get("summarizer"), revision=rev)
        with tracer.span("extractor", kind="phase"):
            e = await agents.extractor.run(brief, transcript,
                                           critiques.get("extractor"), revision=rev)
        with tracer.span("drafter", kind="phase"):
            d = await agents.drafter.run(brief, transcript,
                                         critiques.get("drafter"), revision=rev)
        for c in (s, e, d):
            tracer.add_span(c.to_span())
        return s, e, d

    summ, extr, draft = await run_workers(None, 0)
    result.add(summ, extr, draft)

    rev = 0
    while True:
        with tracer.span("reviewer", kind="phase"):
            rev_res, verdict = await agents.reviewer.run(
                transcript, summ.text, extr.text, draft.text, revision=rev)
        result.add(rev_res)
        tracer.add_span(rev_res.to_span())
        result.verdict_status = verdict.status

        if not verdict.needs_revision or rev >= config.MAX_REVISIONS:
            break
        rev += 1
        result.revisions = rev
        summ, extr, draft = await run_workers(verdict.issues, rev)
        result.add(summ, extr, draft)

    result.artifacts = {"summary": summ.text, "action_items": extr.text,
                        "followups": draft.text, "brief": brief}
    result.finalize()
    return result
