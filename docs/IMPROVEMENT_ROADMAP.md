# Improvement Roadmap & Implementation Checklist

> Single source of truth for fixing the gaps found in the code/measurement review
> of the multi-agent meeting-assistant benchmark. Each item is a checkbox; severity
> and rationale are inline. Work top-down — phases are ordered by impact × safety.
>
> Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Baseline (verified at review time)

- [x] `pytest tests/` — 35 passed
- [x] `ruff check agents/ bench/ pipeline/ scripts/ tests/` — clean
- [x] Offline stack runs end-to-end (mock server → sequential/parallel/langgraph → metrics → plot)

These must stay green after every phase below.

---

## Phase 0 — Documentation truth (low risk, high clarity)

The repo currently ships docs that describe a removed architecture. Fix before anyone
relies on them.

- [ ] **P0.1 Rewrite `.env.example`.** It documents `MA_USE_MOCK`, `MA_SMALL_URL`,
  `MA_LARGE_URL`, `MA_SMALL_MODEL`, `MA_LARGE_MODEL` (8B + 70B), `MA_GUIDED_JSON`,
  `MA_PARSE_RETRIES`, `MA_TEMPERATURE`, etc. — **none are read by any code**. Replace
  with the vars actually consumed by `agents/config.py`:
  `VLLM_BASE_URL`, `VLLM_METRICS_URL`, `MODEL_NAME`, `VLLM_API_KEY`,
  `REQUEST_TIMEOUT_S`, `MAX_RETRIES`, `MAX_REVISIONS`.
- [ ] **P0.2 Reconcile transcript-size docs.** `config.TRANSCRIPT_SIZES` says
  `large.target_tokens=13000` and README says "~8k–15k tokens", but committed
  `large_finance` runs show ~66k prompt tokens/agent. Either re-document the real
  tiers or note that committed transcripts intentionally exceed the synthetic targets.
- [ ] **P0.2a Reconcile the two transcript generators.** There are **two disjoint
  sizing systems**: `scripts/make_synthetic_transcripts.py` builds the six *named*
  files actually used in the README sweep using per-domain `target_words`
  (small_ami ~4.0k words/~5.3k tok … large_meetingbank ~50k words/~66k tok), while
  `scripts/generate_transcripts.py` + `config.TRANSCRIPT_SIZES` use token-targeted
  `small/medium/large` (1.2k/5k/13k tok). The named files are thus ~4–5× the config
  tiers and the two paths never agree. Document which generator is canonical, and
  either align `config.TRANSCRIPT_SIZES` to the named files' real sizes or clearly
  separate "synthetic smoke" (generate_transcripts) from "benchmark corpus"
  (make_synthetic_transcripts).
- [ ] **P0.3 Fix `.gitignore` contradiction.** Comment says "keep committed results"
  but patterns ignore `results/runs/*.json` and `results/figures/*.png` (only tracked
  via prior `git add -f`). Either un-ignore the committed artifacts or change the
  comment + document the `-f` workflow so `bench.plot` re-runs are reproducible.
- [ ] **P0.4 Clean stale references.** `setup.sh` installs `vllm-tpu` and launches
  `Qwen/Qwen3-4B` (unrelated to the documented Llama-3.1-8B in
  `scripts/launch_vllm_8b.sh`); `.gitignore` still lists `data/ground_truth/*.json`
  though ground truth was removed.

---

## Phase 1 — Measurement validity (the core scientific fixes)

These are the findings that undermine the headline results. Highest priority for a
benchmark whose entire purpose is measurement.

- [ ] **P1.1 Eliminate the cache-warming confound.** *(Critical)*
  The sweep always runs modes in fixed order `sequential → parallel → langgraph`
  against one long-lived server whose prefix cache is never reset. The first repeat of
  the first mode is cold; everything after is warm. Result: server-side prefix-cache
  hit rate rises monotonically *with execution order, not topology* — and once warm,
  all three modes sit at ~0.66–0.76. Evidence (`medium_finance`):
  `seq[0.61,0.75,0.76] par[0.75,0.76,0.76] lg[0.76,0.76,0.75]`.
  - [ ] Interleave/randomize `(size, mode, repeat)` execution order (`bench/runner.py`).
  - [ ] Reset cache state between modes by **restarting the server** per mode, or run a
        dedicated control arm with `--no-enable-prefix-caching` (stable vLLM exposes no
        runtime cache-reset endpoint — do not assume one).
  - [ ] Add an explicit warmup run per `(size)` that is excluded from aggregation.
  - [ ] Record `run_index`/`cache_state` in each run record so warmth is auditable.
  - [ ] **Mock parity:** `scripts/mock_vllm_server.py` simulates the prefix cache with a
        process-global `_SEEN_PREFIXES` set that also persists across runs, so the
        offline reproduction / CI exhibits the *same* warming artifact (first run cold,
        rest warm). Apply the warmup/reset logic to the offline path too, or expose a
        reset hook on the mock, so offline numbers are representative.
- [ ] **P1.2 Make the per-request prefix-cache metric real or remove it.** *(Critical)*
  In every committed run, every agent reports `cached_tokens=0`, so
  `prefix_cache_hit_ratio` (a headline metric) is uniformly `0.000`. vLLM never
  populated `prompt_tokens_details.cached_tokens`.
  - [ ] At startup, probe the server and **warn loudly** if `cached_tokens` is absent.
  - [ ] Document the vLLM version/flags required to populate it.
  - [ ] If unavailable on target hardware, drop the metric and rely on the server-side
        counter (and update README Result #3 accordingly).
- [ ] **P1.3 Reorder worker prompts so the shared prefix is actually shared.** *(High)*
  Workers send `[system=role-specific, user=brief+transcript]`. vLLM prefix caching
  matches from token 0, but the system prompt differs per worker, so the big
  `brief+transcript` block sits in the *middle* and is never reused across workers —
  defeating the experiment's premise.
  - [ ] Restructure so the shared `transcript (+brief)` is the leading block and the
        role-specific instruction comes last (same `system` preamble across workers).
  - [ ] Re-measure; the cross-worker prefix-cache hit rate should now be non-trivial.
  - [ ] Keep an A/B toggle (`SHARED_PREFIX_LAYOUT=0|1`) to quantify the delta.

---

## Phase 2 — Cross-mode comparability (fairness of the benchmark)

The three modes currently do different *amounts of work* on revision, contaminating
latency/token comparisons.

- [ ] **P2.1 Unify the re-dispatch policy.** Sequential re-runs all 3 workers,
  parallel re-runs only flagged workers, langgraph re-runs all 3
  (`pipeline/langgraph_pipeline.py:82-87`). Pick one policy (recommend selective) and
  apply it across all three modes, or document the difference and exclude it from
  comparisons.
- [ ] **P2.2 Fix the `revisions` off-by-one.** For identical work, langgraph reports
  `revisions=3` while sequential/parallel report `2` (`reviewer_node` increments even
  on the budget-capped pass). Make `result.revisions` count *performed* rerun rounds
  consistently across modes.

---

## Phase 3 — Robustness & reliability

- [ ] **P3.1 Catch all malformed-response errors.** `Agent.call` retries only on
  `httpx.HTTPError, json.JSONDecodeError, KeyError`, but `_call_once` can raise
  `IndexError`/`TypeError` on empty/malformed `choices` (`agents/base.py:180`). Add
  these to the retry set.
- [ ] **P3.2 Stop silent failure swallowing.** A failed worker returns `text=""`;
  `parse_verdict("")` then defaults to `pass`, so an errored run can yield a "pass"
  verdict with empty artifacts. Surface worker errors into the verdict / mark the run
  invalid rather than silently passing.
- [ ] **P3.3 Log dropped reviewer verdicts.** When `parse_verdict` falls back to `pass`
  on unparseable JSON, emit a warning so quality regressions aren't hidden.
- [ ] **P3.4 Sampler nit.** `system_sampler` primes *process* CPU but samples *system*
  CPU un-primed; align priming with what is sampled (or drop the priming call).

---

## Phase 4 — Test coverage (lock in the fixes)

Add tests that would have caught the bugs above.

- [ ] **P4.1** Assert langgraph revision count == sequential/parallel for the same
  reviewer behavior (covers P2.2).
- [ ] **P4.2** Test `_call_once` and streaming error paths (empty `choices`,
  malformed JSON, missing `usage`) — covers P3.1.
- [ ] **P4.3** Test that an errored worker does **not** produce a spurious `pass`
  verdict / is flagged invalid — covers P3.2.
- [ ] **P4.4** Test re-dispatch policy parity across modes — covers P2.1.
- [ ] **P4.5** Test the warmup/cache-reset bookkeeping in run records — covers P1.1.

---

## Phase 5 — Analysis & reporting quality

- [ ] **P5.1 Statistical rigor.** With 3 repeats and visible bimodality
  (e.g. `parallel large_finance [0.66,0.39,0.66]`), report confidence intervals and
  increase repeats; current means hide variance.
- [ ] **P5.2 Cache-cold control arm.** Run each mode once with prefix caching disabled
  to isolate the topology effect from cache warming.
- [ ] **P5.3 Refresh README results** once P1/P2 land — Results #2/#3 currently
  describe confounded numbers.
- [ ] **P5.4 Pin vLLM version** in docs and fail-loud on missing metrics (ties to P1.2).

---

## Suggested execution order

1. **Phase 0** (docs) — minutes, zero code risk, stops active misinformation.
2. **P3.1 + P2.2 + P3.2** — small, well-scoped code fixes with new tests.
3. **P1.1** (interleave + warmup + cache reset) — the single most important
   correctness fix for the benchmark's conclusions.
4. **P1.3** (prompt reorder, behind a toggle) — makes the headline claim real.
5. **P1.2, P2.1, Phase 4–5** — finalize metrics, parity, tests, and rewrite results.

## Definition of done

- [ ] All phases checked, `pytest` + `ruff` green (both CI Python versions + the
      langgraph-extra job).
- [ ] README Results section regenerated from a re-run sweep with the confound removed.
- [ ] `.env.example` matches `agents/config.py` exactly.
- [ ] No headline metric is silently zero on target hardware.
- [ ] A single canonical transcript generator is documented (P0.2a).

---

## Doc review log

- All file/line claims re-verified against source at review time.
- Coverage gap closed: `scripts/make_synthetic_transcripts.py` reviewed → added P0.2a
  (dual-generator sizing mismatch).
- Corrected P1.1: removed the assumption of a vLLM runtime cache-reset endpoint;
  added mock-server order-dependence (offline/CI reproduces the warming artifact).
- `CONTRIBUTIONS.md` ("54 experiments", v5litepod-4) is consistent with the 54
  committed run JSONs and the README — no action needed.
