"""LangGraph pipeline: the same five-agent topology expressed as a StateGraph.

Provided as a framework comparison against the hand-written asyncio pipelines.
Topology: coordinator fans out to the three workers (parallel super-step), which
fan in to the reviewer; a conditional edge loops back to the workers on a
"revise" verdict, up to MAX_REVISIONS.

Requires `langgraph` (install via the project's optional extra). If it is not
installed, calling run_langgraph raises a clear ImportError.
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

import httpx

from agents import config
from bench.trace import Tracer
from pipeline.base import AgentBundle, PipelineResult


class _State(TypedDict, total=False):
    transcript: str
    brief: str
    summary: str
    action_items: str
    followups: str
    calls: Annotated[list, operator.add]   # reducer: concat across parallel nodes
    verdict_status: str
    revision: int
    critiques: dict


def _build_graph(agents: AgentBundle, tracer: Tracer):
    from langgraph.graph import StateGraph, END  # imported lazily

    async def coordinator_node(state: _State) -> dict:
        with tracer.span("coordinator", kind="phase"):
            c = await agents.coordinator.run(state["transcript"])
        tracer.add_span(c.to_span())
        return {"brief": c.text, "calls": [c]}

    def _critique(state: _State, who: str) -> Optional[str]:
        return (state.get("critiques") or {}).get(who)

    async def summarizer_node(state: _State) -> dict:
        rev = state.get("revision", 0)
        with tracer.span("summarizer", kind="phase"):
            r = await agents.summarizer.run(state["brief"], state["transcript"],
                                            _critique(state, "summarizer"), revision=rev)
        tracer.add_span(r.to_span())
        return {"summary": r.text, "calls": [r]}

    async def extractor_node(state: _State) -> dict:
        rev = state.get("revision", 0)
        with tracer.span("extractor", kind="phase"):
            r = await agents.extractor.run(state["brief"], state["transcript"],
                                           _critique(state, "extractor"), revision=rev)
        tracer.add_span(r.to_span())
        return {"action_items": r.text, "calls": [r]}

    async def drafter_node(state: _State) -> dict:
        rev = state.get("revision", 0)
        with tracer.span("drafter", kind="phase"):
            r = await agents.drafter.run(state["brief"], state["transcript"],
                                         _critique(state, "drafter"), revision=rev)
        tracer.add_span(r.to_span())
        return {"followups": r.text, "calls": [r]}

    async def reviewer_node(state: _State) -> dict:
        rev = state.get("revision", 0)
        with tracer.span("reviewer", kind="phase"):
            res, verdict = await agents.reviewer.run(
                state["transcript"], state.get("summary", ""),
                state.get("action_items", ""), state.get("followups", ""), revision=rev)
        tracer.add_span(res.to_span())
        # Only count a revision when a rerun will ACTUALLY happen (budget enforced
        # here, not in route). This keeps `revision` == performed rerun rounds, so
        # the reported `revisions` matches the sequential/parallel pipelines (P2.2).
        will_revise = verdict.needs_revision and rev < config.MAX_REVISIONS
        return {"verdict_status": verdict.status, "calls": [res],
                "critiques": verdict.issues if will_revise else {},
                "revision": rev + (1 if will_revise else 0)}

    def route(state: _State):
        # Budget is already enforced in reviewer_node: non-empty critiques means a
        # rerun is both wanted and within budget.
        if state.get("critiques"):
            return ["summarizer", "extractor", "drafter"]
        return END

    g = StateGraph(_State)
    for name, fn in [("coordinator", coordinator_node), ("summarizer", summarizer_node),
                     ("extractor", extractor_node), ("drafter", drafter_node),
                     ("reviewer", reviewer_node)]:
        g.add_node(name, fn)
    g.set_entry_point("coordinator")
    for w in ("summarizer", "extractor", "drafter"):
        g.add_edge("coordinator", w)
        g.add_edge(w, "reviewer")
    g.add_conditional_edges("reviewer", route,
                            {"summarizer": "summarizer", "extractor": "extractor",
                             "drafter": "drafter", END: END})
    return g.compile()


async def run_langgraph(transcript: str, transcript_size: str,
                        client: httpx.AsyncClient, tracer: Tracer,
                        stream: bool = True) -> PipelineResult:
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for the 'langgraph' pipeline mode. "
            "Install it with: uv pip install langgraph"
        ) from exc

    agents = AgentBundle(client, stream=stream)
    graph = _build_graph(agents, tracer)

    # recursion_limit must cover coordinator + (workers+reviewer) * (MAX_REVISIONS+1)
    final = await graph.ainvoke(
        {"transcript": transcript, "revision": 0, "calls": []},
        config={"recursion_limit": 4 * (config.MAX_REVISIONS + 1) + 4},
    )

    result = PipelineResult(mode="langgraph", transcript_size=transcript_size)
    result.calls = list(final.get("calls", []))
    result.verdict_status = final.get("verdict_status", "pass")
    result.revisions = final.get("revision", 0)
    result.artifacts = {
        "summary": final.get("summary", ""),
        "action_items": final.get("action_items", ""),
        "followups": final.get("followups", ""),
        "brief": final.get("brief", ""),
    }
    result.finalize()
    return result
