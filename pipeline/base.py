"""Shared pipeline scaffolding.

A pipeline takes a transcript, runs the coordinator + three workers + reviewer
(with a bounded reflection loop), and returns a PipelineResult bundling every
agent CallResult plus the final artifacts. The benchmark layer reads the
CallResults for token/latency metrics; system metrics are sampled separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from agents import (Coordinator, Summarizer, Extractor, Drafter, Reviewer,
                    CallResult)


@dataclass
class PipelineResult:
    mode: str
    transcript_size: str
    calls: list[CallResult] = field(default_factory=list)
    revisions: int = 0
    verdict_status: str = "pass"
    artifacts: dict[str, str] = field(default_factory=dict)

    def add(self, *results: CallResult) -> None:
        self.calls.extend(results)

    # ---- aggregate token/latency metrics over all calls ----
    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self.calls)

    @property
    def aggregate_prefix_cache_hit_ratio(self) -> float:
        pt = self.total_prompt_tokens
        return (self.total_cached_tokens / pt) if pt else 0.0

    @property
    def errored(self) -> bool:
        return any(c.error for c in self.calls)

    def finalize(self) -> None:
        """Post-run consistency guard (P3.2).

        A failed agent call returns empty text, and the reviewer (or its
        fallback parser) can then emit a spurious 'pass' over empty artifacts.
        If any call errored, never report a clean 'pass' — downgrade to 'error'
        so consumers don't treat a broken run as good. Call once at pipeline end.
        """
        if self.errored and self.verdict_status == "pass":
            self.verdict_status = "error"


class AgentBundle:
    """Instantiates the five agents against one shared async client."""

    def __init__(self, client: httpx.AsyncClient, stream: bool = True) -> None:
        self.coordinator = Coordinator(client, stream=stream)
        self.summarizer = Summarizer(client, stream=stream)
        self.extractor = Extractor(client, stream=stream)
        self.drafter = Drafter(client, stream=stream)
        self.reviewer = Reviewer(client, stream=stream)
