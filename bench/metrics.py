"""Metric assembly and aggregation.

Ground-truth / action-item-recall scoring has been intentionally removed. This
module produces purely SYSTEM + SCALE measurements per run:

  Token / scale:
    - total prompt / completion / cached tokens
    - aggregate prefix KV-cache hit ratio (per-request cached_tokens / prompt)
    - tokens per agent call
  Latency:
    - end-to-end wall time, mean/max per-call TTFT (prefill proxy)
    - decode throughput (output tokens / s)
  System (sampled):
    - host CPU %, orchestrator RSS MB, host memory used/percent
    - vLLM KV-cache usage %, running/waiting request depth
  Server counters (delta over the run):
    - prompt/generation tokens, prefix-cache hits/queries -> server hit rate

`build_run_record` returns one flat dict; `aggregate` flattens many run JSONs
into a tidy pandas DataFrame for plotting / the report.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Optional

import pandas as pd

from pipeline.base import PipelineResult
from bench.system_sampler import ResourceSamples


def build_run_record(result: PipelineResult, *, e2e_wall_s: float,
                     samples: ResourceSamples,
                     counter_delta: dict[str, Optional[float]],
                     server_available: bool, repeat: int) -> dict[str, Any]:
    calls = result.calls
    ttfts = [c.ttft_s for c in calls if c.ttft_s is not None]
    decode_tps = [c.decode_tokens_per_s for c in calls if c.decode_tokens_per_s]

    rec: dict[str, Any] = {
        # identity
        "mode": result.mode,
        "transcript_size": result.transcript_size,
        "repeat": repeat,
        "revisions": result.revisions,
        "verdict_status": result.verdict_status,
        "errored": result.errored,
        "n_agent_calls": len(calls),
        # latency
        "e2e_wall_s": e2e_wall_s,
        "ttft_mean_s": _mean(ttfts),
        "ttft_max_s": max(ttfts) if ttfts else None,
        "decode_tps_mean": _mean(decode_tps),
        # token / scale
        "prompt_tokens": result.total_prompt_tokens,
        "completion_tokens": result.total_completion_tokens,
        "total_tokens": result.total_prompt_tokens + result.total_completion_tokens,
        "cached_tokens": result.total_cached_tokens,
        "prefix_cache_hit_ratio": result.aggregate_prefix_cache_hit_ratio,
        "throughput_tok_per_s": (
            (result.total_prompt_tokens + result.total_completion_tokens) / e2e_wall_s
            if e2e_wall_s > 0 else None),
        # server-side / system
        "server_metrics_available": server_available,
    }

    # flatten sampled system metrics
    for key, agg in samples.summary().items():
        rec[f"{key}_mean"] = agg["mean"]
        rec[f"{key}_max"] = agg["max"]

    # flatten server counter deltas
    for k, v in counter_delta.items():
        rec[f"delta_{k}"] = v

    rec["artifacts"] = result.artifacts

    # per-agent token breakdown
    rec["per_agent"] = [
        {
            "agent": c.agent, "revision": c.revision,
            "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
            "cached_tokens": c.cached_tokens, "ttft_s": c.ttft_s,
            "e2e_s": c.e2e_s, "decode_tps": c.decode_tokens_per_s,
            "error": c.error, "text": c.text,
        }
        for c in calls
    ]
    return rec


def _mean(xs: list[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


# --------------------------------------------------------------------------- #
def load_runs(runs_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def aggregate(runs_dir: str) -> pd.DataFrame:
    """One row per run (drops the nested per_agent / span detail)."""
    rows = []
    for run in load_runs(runs_dir):
        rec = {k: v for k, v in run.items() if k not in ("per_agent", "spans")}
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, group size, and 95% CI half-width over repeats, grouped by
    (transcript_size, mode).

    Adds `n_runs` and a `<metric>_ci95` column per metric (P5.1) so reports can
    show uncertainty rather than bare means — important given the small repeat
    counts and the visible run-to-run variance.
    """
    if df.empty:
        return df
    # `run_index` is a sweep-ordering counter, not a measurement — don't aggregate it.
    metric_cols = [c for c in df.columns
                   if df[c].dtype.kind in "fi" and c not in ("repeat", "run_index")]
    grouped_by = df.groupby(["transcript_size", "mode"])
    mean_df = grouped_by[metric_cols].mean(numeric_only=True).reset_index()
    std_df = (grouped_by[metric_cols]
              .std(numeric_only=True)
              .rename(columns={c: f"{c}_std" for c in metric_cols})
              .reset_index()
              .drop(columns=["transcript_size", "mode"]))
    n_runs = grouped_by.size().reset_index(name="n_runs")["n_runs"]

    grouped = pd.concat([mean_df, std_df], axis=1)
    grouped["n_runs"] = n_runs.values
    # 95% CI half-width = 1.96 * std / sqrt(n). NaN when n < 2 (std undefined).
    for c in metric_cols:
        grouped[f"{c}_ci95"] = 1.96 * grouped[f"{c}_std"] / (grouped["n_runs"] ** 0.5)
    grouped["transcript_size"] = grouped["transcript_size"].astype(object)
    grouped["mode"] = grouped["mode"].astype(object)
    return grouped
