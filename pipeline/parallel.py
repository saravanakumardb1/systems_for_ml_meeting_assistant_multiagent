"""Parallel pipeline: the three workers run concurrently via asyncio.gather so
end-to-end worker latency is the slowest of the three rather than the sum. This
is the configuration where shared-transcript KV-cache pressure (and possible
server-side serialisation) is expected to show up.
"""
from __future__ import annotations

import asyncio

import httpx

from agents import config
from bench.trace import Tracer
from pipeline.base import AgentBundle, PipelineResult


async def run_parallel(transcript: str, transcript_size: str,
                       client: httpx.AsyncClient, tracer: Tracer,
                       stream: bool = True) -> PipelineResult:
    agents = AgentBundle(client, stream=stream)
    result = PipelineResult(mode="parallel", transcript_size=transcript_size)

    with tracer.span("coordinator", kind="phase"):
        coord = await agents.coordinator.run(transcript)
    result.add(coord)
    tracer.add_span(coord.to_span())
    brief = coord.text

    async def run_workers(critiques: dict[str, str] | None, rev: int,
                          prev: tuple | None = None):
        critiques = critiques or {}
        prev_s, prev_e, prev_d = prev if prev else (None, None, None)

        # On initial call (no prev), run all workers. On revision rounds, only
        # re-run workers whose critique is non-empty (selective re-dispatch).
        specs: dict[str, tuple] = {
            "summarizer": (agents.summarizer, prev_s),
            "extractor":  (agents.extractor,  prev_e),
            "drafter":    (agents.drafter,    prev_d),
        }
        to_run = {
            name: agent.run(brief, transcript, critiques.get(name), revision=rev)
            for name, (agent, prev_result) in specs.items()
            if prev_result is None or critiques.get(name)
        }

        with tracer.span("workers_parallel", kind="phase"):
            gathered = await asyncio.gather(*to_run.values()) if to_run else []
        run_results = dict(zip(to_run.keys(), gathered))

        s = run_results.get("summarizer", prev_s)
        e = run_results.get("extractor",  prev_e)
        d = run_results.get("drafter",    prev_d)
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
        summ, extr, draft = await run_workers(verdict.issues, rev,
                                              prev=(summ, extr, draft))
        result.add(summ, extr, draft)

    result.artifacts = {"summary": summ.text, "action_items": extr.text,
                        "followups": draft.text, "brief": brief}
    result.finalize()
    return result
