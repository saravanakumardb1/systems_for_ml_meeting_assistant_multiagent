"""Mock vLLM OpenAI-compatible server for offline smoke tests / CI.

Implements just enough of vLLM's surface to exercise the full pipeline + metric
collection WITHOUT a TPU/GPU or real model:

  POST /v1/chat/completions
    - streaming (SSE) and non-streaming
    - returns `usage` including prompt_tokens_details.cached_tokens, simulating
      prefix-cache reuse: identical prompt prefixes seen before are billed as
      cached tokens.
  GET /metrics
    - Prometheus text exposition with the KV-cache / queue / prefix-cache /
      token counters that bench/vllm_metrics.py knows how to parse.

This is NOT a model. Responses are canned, shaped to satisfy each agent's
output contract (JSON for extractor/reviewer, Markdown for the rest) so the
reflection loop and parsers run for real.

Run:
    python scripts/mock_vllm_server.py --port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
import uvicorn

app = FastAPI()

# crude global counters to back /metrics
STATE = {
    "prompt_tokens_total": 0.0,
    "generation_tokens_total": 0.0,
    "prefix_cache_queries_total": 0.0,
    "prefix_cache_hits_total": 0.0,
    "requests_total": 0.0,
    "running": 0,
}
_SEEN_PREFIXES: set[str] = set()


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def _canned_response(system: str, user: str) -> str:
    s = system.lower()
    if "extractor" in s or "action-item" in s:
        return json.dumps([
            {"task": "Update the runbook", "owner": "Alice", "deadline": "Friday",
             "evidence": "Action item for me: update the runbook"},
            {"task": "Share roadmap doc", "owner": "Bob", "deadline": "none",
             "evidence": "I'll take the first pass and share a doc by Friday"},
        ])
    if "quality reviewer" in s or "critic" in s:
        # pass on the first look to keep the smoke test bounded
        return json.dumps({"status": "pass",
                           "issues": {"summarizer": "", "extractor": "", "drafter": ""}})
    if "coordinator" in s:
        return ("PARTICIPANTS: Alice (lead), Bob, Carol.\nSCOPE: Plan Q3 work.\n"
                "DECISION CUES: ship limited beta; update runbook.\n"
                "CAUTIONS: deadline tentative.")
    if "summarizer" in s:
        return ("## Overview\nThe team planned Q3 work.\n\n## Key Discussion Points\n"
                "- Roadmap\n- Migration\n\n## Decisions Made\n- Ship limited beta first.\n\n"
                "## Open Questions\n- Final deadline.")
    if "drafter" in s:
        return ("## Email\nSubject: Meeting recap\nWe agreed to ship the limited beta.\n\n"
                "## Slack\nBeta-first decided. Alice owns the runbook update.")
    return "OK."


def _usage(prompt_text: str, completion_text: str):
    pt = _approx_tokens(prompt_text)
    ct = _approx_tokens(completion_text)
    # simulate prefix cache: hash a stable prefix (first ~200 chars)
    prefix_key = hashlib.sha1(prompt_text[:2000].encode()).hexdigest()
    STATE["prefix_cache_queries_total"] += 1
    cached = 0
    if prefix_key in _SEEN_PREFIXES:
        cached = int(pt * 0.85)            # most of the prefix served from cache
        STATE["prefix_cache_hits_total"] += 1
    else:
        _SEEN_PREFIXES.add(prefix_key)
    STATE["prompt_tokens_total"] += pt
    STATE["generation_tokens_total"] += ct
    STATE["requests_total"] += 1
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    text = _canned_response(system, user)
    usage = _usage(system + "\n" + user, text)
    stream = body.get("stream", False)
    model = body.get("model", "mock-8b")

    STATE["running"] += 1
    # tiny simulated latency proportional to prompt size
    await asyncio.sleep(min(0.05 + usage["prompt_tokens"] / 20000.0, 0.5))

    if not stream:
        STATE["running"] -= 1
        return JSONResponse({
            "id": "mock", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage,
        })

    async def gen():
        # first token after a small prefill delay -> drives TTFT
        await asyncio.sleep(0.02)
        words = text.split(" ")
        for i, w in enumerate(words):
            piece = (w if i == 0 else " " + w)
            chunk = {"id": "mock", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": {"content": piece},
                                  "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.002)
        final = {"id": "mock", "object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": usage}
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
        STATE["running"] -= 1

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/reset_cache")
async def reset_cache():
    """Clear the simulated prefix cache + counters (mock-only).

    Lets tests / a cold-control benchmark arm reset state between modes so the
    offline path doesn't carry the same cross-run warming artifact as a real,
    never-restarted vLLM server (P1.1 mock parity).
    """
    _SEEN_PREFIXES.clear()
    for k in STATE:
        STATE[k] = 0 if k == "running" else 0.0
    return JSONResponse({"status": "reset"})


@app.get("/metrics")
async def metrics():
    # cache usage proxy: more in-flight -> higher occupancy
    cache_usage = min(0.05 + STATE["running"] * 0.2, 0.95)
    lines = [
        "# HELP vllm:gpu_cache_usage_perc KV-cache usage.",
        "# TYPE vllm:gpu_cache_usage_perc gauge",
        f'vllm:gpu_cache_usage_perc{{model_name="mock"}} {cache_usage}',
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{model_name="mock"}} {STATE["running"]}',
        "# TYPE vllm:num_requests_waiting gauge",
        'vllm:num_requests_waiting{model_name="mock"} 0',
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{model_name="mock"}} {STATE["prompt_tokens_total"]}',
        "# TYPE vllm:generation_tokens_total counter",
        f'vllm:generation_tokens_total{{model_name="mock"}} {STATE["generation_tokens_total"]}',
        "# TYPE vllm:prefix_cache_queries_total counter",
        f'vllm:prefix_cache_queries_total{{model_name="mock"}} {STATE["prefix_cache_queries_total"]}',
        "# TYPE vllm:prefix_cache_hits_total counter",
        f'vllm:prefix_cache_hits_total{{model_name="mock"}} {STATE["prefix_cache_hits_total"]}',
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
