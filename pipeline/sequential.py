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

    async def run_workers(critiques: dict[str, str] | None, rev: int,
                          prev: tuple | None = None):
        critiques = critiques or {}
        prev_s, prev_e, prev_d = prev if prev else (None, None, None)

        # Selective re-dispatch (P2.1): on the initial pass (no prev) run all three;
        # on revision rounds re-run ONLY workers the reviewer flagged, carrying the
        # rest forward unchanged. This matches the parallel/langgraph policy so the
        # three modes do the same amount of work per revision.
        specs: dict[str, tuple] = {
            "summarizer": (agents.summarizer, prev_s),
            "extractor":  (agents.extractor,  prev_e),
            "drafter":    (agents.drafter,    prev_d),
        }
        run_results: dict[str, object] = {}
        for name, (agent, prev_result) in specs.items():
            if prev_result is None or critiques.get(name):
                with tracer.span(name, kind="phase"):
                    run_results[name] = await agent.run(
                        brief, transcript, critiques.get(name), revision=rev)

        # Count + trace ONLY freshly-run workers so carried-over results aren't
        # double-counted in result.calls across revision rounds.
        result.add(*run_results.values())
        for c in run_results.values():
            tracer.add_span(c.to_span())

        s = run_results.get("summarizer", prev_s)
        e = run_results.get("extractor",  prev_e)
        d = run_results.get("drafter",    prev_d)
        return s, e, d

    summ, extr, draft = await run_workers(None, 0)

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

    result.artifacts = {"summary": summ.text, "action_items": extr.text,
                        "followups": draft.text, "brief": brief}
    result.finalize()
    return result
