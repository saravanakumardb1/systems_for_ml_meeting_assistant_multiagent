"""Base agent: prompt loading, async vLLM call, retry, and per-call tracing.

Every agent issues OpenAI-compatible chat completions against the single 8B
vLLM server. Streaming is used so we can measure time-to-first-token (TTFT) and
separate prefill from decode. Token usage -- including prefix-cache hits
(`prompt_tokens_details.cached_tokens`) -- is captured from the final stream
chunk via `stream_options.include_usage`.
"""
from __future__ import annotations

import os
import time
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from agents import config
from bench.trace import Span


@dataclass
class CallResult:
    """Result of a single agent call, including scale/system token metrics."""
    agent: str
    text: str
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0          # prompt tokens served from the KV prefix cache
    # Latency
    ttft_s: Optional[float] = None  # time to first token (prefill proxy)
    decode_s: Optional[float] = None
    e2e_s: float = 0.0
    # Bookkeeping
    start_ts: float = 0.0
    end_ts: float = 0.0
    revision: int = 0
    attempts: int = 1
    error: Optional[str] = None

    @property
    def prefix_cache_hit_ratio(self) -> float:
        return (self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0.0

    @property
    def decode_tokens_per_s(self) -> float:
        if self.decode_s and self.decode_s > 0:
            return self.completion_tokens / self.decode_s
        return 0.0

    def to_span(self) -> Span:
        return Span(
            name=self.agent,
            kind="agent",
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            duration_s=self.e2e_s,
            attrs={
                "revision": self.revision,
                "attempts": self.attempts,
                "error": self.error,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cached_tokens": self.cached_tokens,
                "prefix_cache_hit_ratio": self.prefix_cache_hit_ratio,
                "ttft_s": self.ttft_s,
                "decode_s": self.decode_s,
                "decode_tokens_per_s": self.decode_tokens_per_s,
            },
        )


_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt(prompt_file: str) -> str:
    if prompt_file not in _PROMPT_CACHE:
        path = os.path.join(config.PROMPTS_DIR, prompt_file)
        with open(path, "r", encoding="utf-8") as fh:
            _PROMPT_CACHE[prompt_file] = fh.read()
    return _PROMPT_CACHE[prompt_file]


class Agent:
    """A single specialised agent backed by the shared 8B vLLM server."""

    def __init__(self, spec: config.AgentSpec, client: httpx.AsyncClient,
                 model: str = config.MODEL_NAME, stream: bool = True) -> None:
        self.spec = spec
        self.name = spec.name
        self.client = client
        self.model = model
        self.stream = stream
        self.system_prompt = _load_prompt(spec.prompt_file)

    # ------------------------------------------------------------------ #
    async def call(self, user_content: str, *, revision: int = 0) -> CallResult:
        """Call the model with retry. Returns a fully-populated CallResult."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        last_err: Optional[str] = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                res = await (self._call_stream(messages) if self.stream
                             else self._call_once(messages))
                res.revision = revision
                res.attempts = attempt
                return res
            # IndexError/TypeError cover malformed responses with empty/missing
            # `choices` (e.g. obj["choices"][0] on []); these are retryable too.
            except (httpx.HTTPError, json.JSONDecodeError, KeyError,
                    IndexError, TypeError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < config.MAX_RETRIES:
                    await asyncio.sleep(0.5 * attempt)
        return CallResult(agent=self.name, text="", revision=revision,
                          attempts=config.MAX_RETRIES, error=last_err,
                          start_ts=time.time(), end_ts=time.time())

    # ------------------------------------------------------------------ #
    async def _call_stream(self, messages: list[dict]) -> CallResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start_perf = time.perf_counter()
        start_wall = time.time()
        ttft: Optional[float] = None
        chunks: list[str] = []
        usage: dict[str, Any] = {}

        url = f"{config.VLLM_BASE_URL}/v1/chat/completions"
        async with self.client.stream("POST", url, json=payload,
                                      timeout=config.REQUEST_TIMEOUT_S) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start_perf
                        chunks.append(piece)

        end_perf = time.perf_counter()
        e2e = end_perf - start_perf
        return self._build_result("".join(chunks), usage, start_wall, e2e, ttft)

    async def _call_once(self, messages: list[dict]) -> CallResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
            "stream": False,
        }
        start_perf = time.perf_counter()
        start_wall = time.time()
        url = f"{config.VLLM_BASE_URL}/v1/chat/completions"
        resp = await self.client.post(url, json=payload,
                                      timeout=config.REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        obj = resp.json()
        text = obj["choices"][0]["message"]["content"]
        e2e = time.perf_counter() - start_perf
        return self._build_result(text, obj.get("usage", {}), start_wall, e2e, None)

    # ------------------------------------------------------------------ #
    def _build_result(self, text: str, usage: dict, start_wall: float,
                       e2e: float, ttft: Optional[float]) -> CallResult:
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        tt = int(usage.get("total_tokens", pt + ct))
        cached = 0
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens", 0) or 0)
        decode_s = (e2e - ttft) if (ttft is not None) else None
        return CallResult(
            agent=self.name,
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            cached_tokens=cached,
            ttft_s=ttft,
            decode_s=decode_s,
            e2e_s=e2e,
            start_ts=start_wall,
            end_ts=start_wall + e2e,
        )
