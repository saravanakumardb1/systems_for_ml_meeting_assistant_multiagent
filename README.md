# Automated Meeting Assistant — Multi-Agent System (vLLM + Cloud TPU)

A five-agent pipeline that turns a meeting transcript into a post-meeting packet
(summary, action items, follow-up email/Slack), instrumented for **system and
scale** benchmarking on a single **Llama-3.1-8B** vLLM server.

## What changed vs. the original proposal

This revision implements four requested changes:

1. **Ground truth removed.** The transcript generator no longer emits reference
   summaries/action-item labels, and there is no `data/ground_truth/` directory.
   The Quality Reviewer judges artifacts against the transcript only (internal
   consistency / hallucination / structure), and `bench/metrics.py` contains **no
   action-item recall scoring** — only system + scale measurements.
2. **More metrics.** Each run records memory, compute/throughput, token usage,
   KV-cache prefix-hit rate, queue depth, KV-cache occupancy, and latency/TTFT
   (see below).
3. **Router mode removed.** Pipeline modes are `sequential`, `parallel`, and
   `langgraph` only.
4. **Single model, size sweep.** Everything runs on Llama-3.1-8B; the experiment
   sweeps transcript size `small` / `medium` / `large`.

## Agents

`coordinator` (triage + brief) → `summarizer`, `extractor`, `drafter` (workers,
share the transcript prefix) → `reviewer` (critic; bounded reflection loop,
`MAX_REVISIONS`). All five target the same 8B server.

## Pipeline modes

| mode         | workers run        | notes                                         |
|--------------|--------------------|-----------------------------------------------|
| `sequential` | one after another  | baseline                                      |
| `parallel`   | `asyncio.gather`   | exposes shared-prefix KV-cache pressure       |
| `langgraph`  | StateGraph fan-out | framework comparison (requires `langgraph`)   |

## Metrics collected (per run → `results/runs/*.json`)

- **Token / scale:** prompt, completion, cached tokens; aggregate **prefix
  KV-cache hit ratio** (per-request `prompt_tokens_details.cached_tokens`).
- **Latency:** end-to-end wall time, mean/max **TTFT** (prefill proxy), decode
  throughput (tokens/s), overall throughput.
- **System (sampled at 4 Hz):** host CPU %, orchestrator RSS (MB), host memory;
  vLLM **KV-cache occupancy %**, **running/waiting request depth** (reveals
  whether "parallel" requests are serialised by the server).
- **Server counters (delta over the run):** prompt/generation tokens,
  prefix-cache hits/queries → **server-side prefix-cache hit rate**.

> Note: TPU/GPU device utilisation is not exposed via the OpenAI API. The
> scrapeable proxies for device pressure are KV-cache occupancy and queue depth
> from vLLM `/metrics`; host CPU/RAM come from `psutil`.
>
> **vLLM version note (P1.2/P5.4):** the per-request prefix-cache metric depends on
> the server populating `usage.prompt_tokens_details.cached_tokens`. Some vLLM
> builds never do (then `prefix_cache_hit_ratio` is uniformly 0). The runner emits
> a loud warning at the end of a sweep when this happens; in that case rely on the
> server-side `delta_server_prefix_cache_hit_rate` from `/metrics`. Pin a vLLM
> version known to emit per-request cached tokens if you need that metric.

## Results

> **⚠️ Methodology caveat (results pre-date the measurement fixes).** The committed
> figures below were produced *before* the benchmark-correctness fixes in
> `docs/IMPROVEMENT_ROADMAP.md`:
> - The sweep ran modes in fixed order against a never-reset server, so the
>   prefix-cache hit rate correlated with **execution order, not topology**
>   (P1.1). Result #3's monotonic seq→par→lg rise is largely a cache-warming
>   artifact; once warm, all three modes are comparable.
> - The per-request `prefix_cache_hit_ratio` was **inert** (the server never
>   populated `cached_tokens`); only the server-side counter carried signal (P1.2).
> - Re-dispatch policy differed across modes, so revision-round token/latency
>   counts weren't directly comparable (P2.1).
>
> The runner now defaults to `--warmup 1` + shuffled order and records
> `run_index`. **Re-run the sweep on real hardware to regenerate these figures**
> before citing Results #2–#4.

Measured on **Llama-3.1-8B** (vLLM, Cloud TPU v5e) across a 54-run sweep:
6 transcripts spanning three size tiers (`small_ami`, `small_swe`,
`medium_finance`, `medium_welsh`, `large_finance`, `large_meetingbank`) ×
3 modes × 3 repeats. Raw per-run JSON is in `results/runs/`; aggregates in
`results/figures/summary.csv` and the figure PDFs/PNGs.

> **Transcript provenance.** All transcripts shipped in `transcripts/` are
> **synthetic**, produced by the seeded generator
> `scripts/make_synthetic_transcripts.py` (no real recordings or personal
> data). The committed `results/` were measured on real hardware with these
> inputs, so the figures below remain representative of the systems behaviour
> rather than of any specific meeting content.
>
> **Two transcript generators (don't confuse them):**
> - `scripts/make_synthetic_transcripts.py` produces the six **named benchmark
>   files** in `transcripts/` (domain-flavoured, word-targeted:
>   `small_ami` ~5k tok … `large_meetingbank` ~66k tok). This is the canonical
>   corpus for the published sweep.
> - `scripts/generate_transcripts.py` + `config.TRANSCRIPT_SIZES` produce small
>   token-targeted `small/medium/large.txt` for **offline smoke tests / CI**
>   (1.2k / 5k / 13k tok). These are intentionally smaller and are *not* the
>   benchmark corpus.

1. **Concurrent dispatch cuts end-to-end latency 29–40% on medium/large
   transcripts.** Parallel vs. sequential: `large_meetingbank` 63.1 s → 37.9 s
   (−40%), `medium_finance` 49.8 s → 31.0 s (−38%), `large_finance` 67.5 s →
   47.9 s (−29%). On the smallest transcripts the gain shrinks (6–17%) as
   fixed orchestration overhead dominates the worker latency it overlaps.

2. **The fan-out is real at the engine, not just the client.** Server-side
   `num_requests_running` rises from **1** (sequential) to **3** (parallel /
   langgraph), and peak KV-cache occupancy grows ~3× — e.g. `large_finance`
   **20% → 59%**. The orchestration topology measurably changes inference-engine
   pressure, which is invisible from application-layer latency alone.

3. **The shared transcript prefix pays off.** Server-side prefix-cache hit rate
   averages **0.47 → 0.58 → 0.60** (sequential → parallel → langgraph) and spans
   **26–76%** across transcripts: concurrent workers that share the identical
   `(brief + transcript)` prefix let vLLM reuse cached KV blocks.

4. **Concurrency trades per-request decode bandwidth for wall-clock.** Mean
   per-request decode throughput drops **124 → ~105 tok/s** under parallel /
   langgraph fan-out even as end-to-end time improves — the three workers
   contend for the same decode budget.

## Setup

```bash
uv pip install -e .                 # core deps
uv pip install -e ".[langgraph]"    # for the langgraph mode
uv pip install -e ".[mock]"         # for the offline mock server (testing)
```

## Run on real hardware (Cloud TPU v5e, Llama-3.1-8B)

```bash
# 1. Start the 8B server (prefix caching + Prometheus /metrics on :8001)
bash scripts/launch_vllm_8b.sh

# 2. (Optional) generate additional transcripts — pre-committed ones are in transcripts/
python scripts/generate_transcripts.py --sizes small medium large

# 3. Benchmark the sweep using the pre-committed transcripts
python -m bench.runner \
    --sizes small_ami small_swe medium_finance medium_welsh large_finance large_meetingbank \
    --transcript-dir transcripts \
    --modes sequential parallel langgraph --repeats 3

# 4. Figures + summary.csv
python -m bench.plot
```

Point the client at the server with env vars if not on localhost:
`VLLM_BASE_URL`, `VLLM_METRICS_URL`, `MODEL_NAME`.

## Run offline (no TPU/GPU) — smoke test the whole stack

```bash
export VLLM_BASE_URL=http://127.0.0.1:8001
export VLLM_METRICS_URL=http://127.0.0.1:8001/metrics
python scripts/mock_vllm_server.py --port 8001 &      # canned model + /metrics
python scripts/generate_transcripts.py --offline      # local transcripts
python -m bench.runner --repeats 2
python -m bench.plot
```

The mock returns contract-correct outputs (JSON for extractor/reviewer, Markdown
elsewhere) and simulated `usage` with `cached_tokens`, so the reflection loop,
parsers, and every metric path execute for real.

## Run locally on Ollama (real model, no TPU/GPU server)

[Ollama](https://ollama.com) exposes an OpenAI-compatible API, so the agents can
target it directly. A convenience wrapper handles env + transcripts + summary:

```bash
ollama pull llama3.1:8b
scripts/run_ollama.sh                          # small, 3 modes, 3 repeats, warmup 1
SIZES="small medium" REPEATS=2 scripts/run_ollama.sh
OLLAMA_MODEL=qwen3.5:9b scripts/run_ollama.sh
```

Or run the pieces manually:

```bash
export VLLM_BASE_URL=http://localhost:11434 MODEL_NAME=llama3.1:8b
export NO_PROXY=localhost,127.0.0.1            # bypass any corporate proxy for localhost
python scripts/generate_transcripts.py --offline --sizes small
python -m bench.runner --sizes small --no-server-metrics --runs-dir results/ollama/runs
```

Caveats specific to Ollama:

- **No vLLM `/metrics`** → pass `--no-server-metrics` (host CPU/RAM still sampled).
- **No per-request `cached_tokens`** → `prefix_cache_hit_ratio` is 0 and the runner
  prints a loud P1.2 warning. This is expected; use a vLLM host to exercise prefix
  caching.
- **Single model instance serializes requests** → `parallel` ≈ `sequential`
  wall-clock (no decode batching), unlike a batching server like vLLM.
- Output goes to `results/ollama/runs/` (git-ignored) so a local sweep never
  clobbers the canonical TPU corpus in `results/runs/`.

## Layout

```
meeting-assistant/
├── agents/        config, base Agent, coordinator, summarizer, extractor, drafter, reviewer
├── pipeline/      sequential.py, parallel.py, langgraph_pipeline.py
├── bench/         runner.py, metrics.py, trace.py, system_sampler.py, vllm_metrics.py, plot.py
├── prompts/       one .md per agent + transcript_generator.md
├── scripts/       launch_vllm_8b.sh, run_ollama.sh, generate_transcripts.py,
│                  make_synthetic_transcripts.py, mock_vllm_server.py
├── transcripts/   synthetic transcripts (small_ami, small_swe, medium_finance,
│                  medium_welsh, large_finance, large_meetingbank)
├── tests/         pytest suite (test_metrics, test_pipeline, test_reviewer, test_vllm_metrics)
├── paper/         related_work.tex, references.bib
└── results/       runs/ (per-run JSON), figures/ (PNGs + summary.csv)
```

## License

Released under the [MIT License](LICENSE) © 2026 Umamaheshwari Devarajan.
