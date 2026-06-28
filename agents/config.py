"""Central configuration for the Automated Meeting Assistant.

Single-model setup: every agent (coordinator, workers, reviewer) targets the
same Llama-3.1-8B vLLM OpenAI-compatible server. The experiment sweeps
transcript *size* (small / medium / large) across three pipeline topologies
(sequential, parallel, langgraph) and records system + scale metrics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Server / model
# --------------------------------------------------------------------------- #
# The 8B server. Override with env vars when pointing at a real vLLM host.
VLLM_BASE_URL: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
VLLM_METRICS_URL: str = os.environ.get("VLLM_METRICS_URL", f"{VLLM_BASE_URL}/metrics")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
# TODO-1: API_KEY is read here but never sent as an Authorization header by the
# client (agents/base.py). Wire `Authorization: Bearer {API_KEY}` into requests if
# we ever target a secured vLLM endpoint; harmless to leave unsent for local/mock.
API_KEY: str = os.environ.get("VLLM_API_KEY", "EMPTY")  # vLLM ignores the value

# Networking
REQUEST_TIMEOUT_S: float = float(os.environ.get("REQUEST_TIMEOUT_S", "600"))
MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))

# Reflection / self-correction
MAX_REVISIONS: int = int(os.environ.get("MAX_REVISIONS", "2"))

# P1.3: shared-prefix prompt layout. When enabled, the three workers use an
# IDENTICAL leading prompt (common system preamble + transcript + brief) and move
# their role-specific instruction to a trailing TASK block, so vLLM can reuse the
# long shared KV prefix across the concurrent workers. Default OFF preserves the
# baseline (role-specific system prompt; transcript in the middle). See TODO-2 for
# the open question on whether trailing instructions affect output quality.
SHARED_PREFIX_LAYOUT: bool = os.environ.get(
    "SHARED_PREFIX_LAYOUT", "0").lower() in ("1", "true", "yes", "on")
WORKER_SHARED_PROMPT_FILE: str = "worker_shared.md"


@dataclass(frozen=True)
class AgentSpec:
    """Per-agent generation settings. All agents share MODEL_NAME (8B)."""
    name: str
    prompt_file: str
    temperature: float = 0.2
    max_tokens: int = 1024


AGENT_SPECS: dict[str, AgentSpec] = {
    "coordinator": AgentSpec("coordinator", "coordinator.md", temperature=0.1, max_tokens=512),
    "summarizer":  AgentSpec("summarizer",  "summarizer.md",  temperature=0.3, max_tokens=1536),
    "extractor":   AgentSpec("extractor",   "extractor.md",   temperature=0.0, max_tokens=1024),
    "drafter":     AgentSpec("drafter",     "drafter.md",     temperature=0.4, max_tokens=1024),
    "reviewer":    AgentSpec("reviewer",    "reviewer.md",    temperature=0.0, max_tokens=768),
}


@dataclass(frozen=True)
class TranscriptSize:
    """A transcript bucket. `target_tokens` is approximate prompt length."""
    label: str
    target_tokens: int
    n_speakers: int
    n_topics: int


# Approximate prompt token budgets for the SYNTHETIC SMOKE generator
# (scripts/generate_transcripts.py), used for offline/CI runs. These are
# intentionally small and are NOT the sizes of the committed benchmark corpus:
# the six named files in transcripts/ are produced by
# scripts/make_synthetic_transcripts.py at larger, domain-specific word targets
# (small_ami ~5k tok ... large_meetingbank ~66k tok). See README "Transcripts".
TRANSCRIPT_SIZES: dict[str, TranscriptSize] = {
    "small":  TranscriptSize("small",  target_tokens=1200,  n_speakers=3, n_topics=2),
    "medium": TranscriptSize("medium", target_tokens=5000,  n_speakers=5, n_topics=4),
    "large":  TranscriptSize("large",  target_tokens=13000, n_speakers=8, n_topics=7),
}

PIPELINE_MODES: tuple[str, ...] = ("sequential", "parallel", "langgraph")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(ROOT, "prompts")
DATA_DIR = os.path.join(ROOT, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
NORMALIZED_DIR = os.path.join(DATA_DIR, "normalized")
RESULTS_DIR = os.path.join(ROOT, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
