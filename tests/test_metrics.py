"""Tests for bench/metrics.py — build_run_record and summarize."""
import pytest
import pandas as pd

from bench.metrics import build_run_record, summarize
from bench.system_sampler import ResourceSamples
from pipeline.base import PipelineResult
from agents.base import CallResult


# ------------------------------------------------------------------ fixtures
def _call(agent="summarizer", prompt=100, completion=50, cached=40,
          ttft=1.2, e2e=3.0, error=None):
    return CallResult(
        agent=agent,
        text="dummy output",
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cached_tokens=cached,
        ttft_s=ttft,
        decode_s=e2e - ttft if ttft else None,
        e2e_s=e2e,
        start_ts=0.0,
        end_ts=e2e,
        error=error,
    )


def _pipeline_result(mode="sequential", size="small_ami"):
    r = PipelineResult(mode=mode, transcript_size=size)
    r.add(_call("coordinator", prompt=50,  completion=30,  cached=0,  ttft=0.3, e2e=1.0))
    r.add(_call("summarizer",  prompt=100, completion=80,  cached=60, ttft=1.0, e2e=4.0))
    r.add(_call("extractor",   prompt=100, completion=120, cached=60, ttft=1.1, e2e=6.0))
    r.add(_call("drafter",     prompt=100, completion=60,  cached=60, ttft=0.9, e2e=3.5))
    r.add(_call("reviewer",    prompt=120, completion=40,  cached=80, ttft=0.5, e2e=2.0))
    r.verdict_status = "pass"
    r.revisions = 0
    return r


# ------------------------------------------------------------------ tests
class TestBuildRunRecord:
    def test_basic_fields_present(self):
        result = _pipeline_result()
        samples = ResourceSamples()
        rec = build_run_record(result, e2e_wall_s=10.0, samples=samples,
                               counter_delta={}, server_available=False, repeat=0)
        assert rec["mode"] == "sequential"
        assert rec["transcript_size"] == "small_ami"
        assert rec["repeat"] == 0
        assert rec["e2e_wall_s"] == pytest.approx(10.0)

    def test_token_totals(self):
        result = _pipeline_result()
        samples = ResourceSamples()
        rec = build_run_record(result, e2e_wall_s=10.0, samples=samples,
                               counter_delta={}, server_available=False, repeat=0)
        # coordinator:50+30  summarizer:100+80  extractor:100+120  drafter:100+60  reviewer:120+40
        assert rec["prompt_tokens"]     == 50 + 100 + 100 + 100 + 120
        assert rec["completion_tokens"] == 30 +  80 + 120 +  60 +  40
        assert rec["total_tokens"] == rec["prompt_tokens"] + rec["completion_tokens"]

    def test_cached_tokens_and_hit_ratio(self):
        result = _pipeline_result()
        samples = ResourceSamples()
        rec = build_run_record(result, e2e_wall_s=10.0, samples=samples,
                               counter_delta={}, server_available=False, repeat=0)
        # coordinator:0  summarizer:60  extractor:60  drafter:60  reviewer:80  → 260
        assert rec["cached_tokens"] == 0 + 60 + 60 + 60 + 80
        expected_ratio = rec["cached_tokens"] / rec["prompt_tokens"]
        assert rec["prefix_cache_hit_ratio"] == pytest.approx(expected_ratio)

    def test_ttft_mean_and_max(self):
        result = _pipeline_result()
        samples = ResourceSamples()
        rec = build_run_record(result, e2e_wall_s=10.0, samples=samples,
                               counter_delta={}, server_available=False, repeat=0)
        ttfts = [0.3, 1.0, 1.1, 0.9, 0.5]
        assert rec["ttft_mean_s"] == pytest.approx(sum(ttfts) / len(ttfts))
        assert rec["ttft_max_s"]  == pytest.approx(max(ttfts))

    def test_throughput(self):
        result = _pipeline_result()
        samples = ResourceSamples()
        rec = build_run_record(result, e2e_wall_s=10.0, samples=samples,
                               counter_delta={}, server_available=False, repeat=0)
        assert rec["throughput_tok_per_s"] == pytest.approx(
            rec["total_tokens"] / 10.0)

    def test_errored_flag(self):
        result = _pipeline_result()
        result.add(_call("summarizer", error="timeout"))
        samples = ResourceSamples()
        rec = build_run_record(result, e2e_wall_s=5.0, samples=samples,
                               counter_delta={}, server_available=False, repeat=0)
        assert rec["errored"] is True

    def test_per_agent_list(self):
        result = _pipeline_result()
        samples = ResourceSamples()
        rec = build_run_record(result, e2e_wall_s=10.0, samples=samples,
                               counter_delta={}, server_available=False, repeat=0)
        agents = [r["agent"] for r in rec["per_agent"]]
        assert "coordinator" in agents
        assert "reviewer" in agents


# ------------------------------------------------------------------ summarize
class TestSummarize:
    def _make_df(self):
        rows = []
        for rep in range(3):
            rows.append({"transcript_size": "small_ami", "mode": "sequential",
                         "repeat": rep, "e2e_wall_s": 20.0 + rep,
                         "total_tokens": 1000})
            rows.append({"transcript_size": "small_ami", "mode": "parallel",
                         "repeat": rep, "e2e_wall_s": 10.0 + rep,
                         "total_tokens": 1000})
        return pd.DataFrame(rows)

    def test_mean_correct(self):
        df = self._make_df()
        s = summarize(df)
        seq = s[(s["transcript_size"] == "small_ami") & (s["mode"] == "sequential")]
        assert seq["e2e_wall_s"].values[0] == pytest.approx(21.0)

    def test_std_column_present(self):
        df = self._make_df()
        s = summarize(df)
        assert "e2e_wall_s_std" in s.columns

    def test_std_correct(self):
        df = self._make_df()
        s = summarize(df)
        seq = s[(s["transcript_size"] == "small_ami") & (s["mode"] == "sequential")]
        # values are 20, 21, 22 → std = 1.0
        assert seq["e2e_wall_s_std"].values[0] == pytest.approx(1.0)

    def test_n_runs_and_ci95(self):
        df = self._make_df()
        s = summarize(df)
        seq = s[(s["transcript_size"] == "small_ami") & (s["mode"] == "sequential")]
        assert seq["n_runs"].values[0] == 3
        # CI95 half-width = 1.96 * std / sqrt(n) = 1.96 * 1.0 / sqrt(3)
        assert seq["e2e_wall_s_ci95"].values[0] == pytest.approx(1.96 / (3 ** 0.5))
