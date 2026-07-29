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

- [x] **P0.1 Rewrite `.env.example`.** ✅ [`162511d`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/162511d) It documents `MA_USE_MOCK`, `MA_SMALL_URL`,
  `MA_LARGE_URL`, `MA_SMALL_MODEL`, `MA_LARGE_MODEL` (8B + 70B), `MA_GUIDED_JSON`,
  `MA_PARSE_RETRIES`, `MA_TEMPERATURE`, etc. — **none are read by any code**. Replace
  with the vars actually consumed by `agents/config.py`:
  `VLLM_BASE_URL`, `VLLM_METRICS_URL`, `MODEL_NAME`, `VLLM_API_KEY`,
  `REQUEST_TIMEOUT_S`, `MAX_RETRIES`, `MAX_REVISIONS`.
- [x] **P0.2 Reconcile transcript-size docs.** ✅ [`1cf6c83`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/1cf6c83) `config.TRANSCRIPT_SIZES` says
  `large.target_tokens=13000` and README says "~8k–15k tokens", but committed
  `large_finance` runs show ~66k prompt tokens/agent. Either re-document the real
  tiers or note that committed transcripts intentionally exceed the synthetic targets.
- [x] **P0.2a Reconcile the two transcript generators.** ✅ [`1cf6c83`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/1cf6c83) There are **two disjoint
  sizing systems**: `scripts/make_synthetic_transcripts.py` builds the six *named*
  files actually used in the README sweep using per-domain `target_words`
  (small_ami ~4.0k words/~5.3k tok … large_meetingbank ~50k words/~66k tok), while
  `scripts/generate_transcripts.py` + `config.TRANSCRIPT_SIZES` use token-targeted
  `small/medium/large` (1.2k/5k/13k tok). The named files are thus ~4–5× the config
  tiers and the two paths never agree. Document which generator is canonical, and
  either align `config.TRANSCRIPT_SIZES` to the named files' real sizes or clearly
  separate "synthetic smoke" (generate_transcripts) from "benchmark corpus"
  (make_synthetic_transcripts).
- [x] **P0.3 Fix `.gitignore` contradiction.** ✅ [`f5b8d3e`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/f5b8d3e) Comment says "keep committed results"
  but patterns ignore `results/runs/*.json` and `results/figures/*.png` (only tracked
  via prior `git add -f`). Either un-ignore the committed artifacts or change the
  comment + document the `-f` workflow so `bench.plot` re-runs are reproducible.
- [x] **P0.4 Clean stale references.** ✅ [`f5b8d3e`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/f5b8d3e) `setup.sh` installs `vllm-tpu` and launches
  `Qwen/Qwen3-4B` (unrelated to the documented Llama-3.1-8B in
  `scripts/launch_vllm_8b.sh`); `.gitignore` still lists `data/ground_truth/*.json`
  though ground truth was removed.

---

## Phase 1 — Measurement validity (the core scientific fixes)

These are the findings that undermine the headline results. Highest priority for a
benchmark whose entire purpose is measurement.

- [x] **P1.1 Eliminate the cache-warming confound.** *(Critical)* ✅ [`8ed2ba1`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/8ed2ba1)
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
- [x] **P1.2 Make the per-request prefix-cache metric real or remove it.** *(Critical)* ✅ [`083d1ed`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/083d1ed)
  In every committed run, every agent reports `cached_tokens=0`, so
  `prefix_cache_hit_ratio` (a headline metric) is uniformly `0.000`. vLLM never
  populated `prompt_tokens_details.cached_tokens`.
  - [ ] At startup, probe the server and **warn loudly** if `cached_tokens` is absent.
  - [ ] Document the vLLM version/flags required to populate it.
  - [ ] If unavailable on target hardware, drop the metric and rely on the server-side
        counter (and update README Result #3 accordingly).
- [x] **P1.3 Reorder worker prompts so the shared prefix is actually shared.** *(High)* ✅ [`e43b617`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/e43b617) — behind `SHARED_PREFIX_LAYOUT` (default OFF); output-quality validation is **TODO-2**.
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

- [x] **P2.1 Unify the re-dispatch policy.** ✅ [`17a9c44`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/17a9c44) — all modes now selective; also fixed a
  carried-over double-count in `result.calls`. Previously sequential re-ran all 3
  workers, parallel re-ran only flagged, langgraph re-ran all 3.
- [x] **P2.2 Fix the `revisions` off-by-one.** ✅ [`518d12a`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/518d12a) For identical work, langgraph reports
  `revisions=3` while sequential/parallel report `2` (`reviewer_node` increments even
  on the budget-capped pass). Make `result.revisions` count *performed* rerun rounds
  consistently across modes.

---

## Phase 3 — Robustness & reliability

- [x] **P3.1 Catch all malformed-response errors.** ✅ [`1d7572e`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/1d7572e) `Agent.call` retries only on
  `httpx.HTTPError, json.JSONDecodeError, KeyError`, but `_call_once` can raise
  `IndexError`/`TypeError` on empty/malformed `choices` (`agents/base.py:180`). Add
  these to the retry set.
- [x] **P3.2 Stop silent failure swallowing.** ✅ [`076654e`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/076654e) A failed worker returns `text=""`;
  `parse_verdict("")` then defaults to `pass`, so an errored run can yield a "pass"
  verdict with empty artifacts. Surface worker errors into the verdict / mark the run
  invalid rather than silently passing.
- [x] **P3.3 Log dropped reviewer verdicts.** ✅ [`8e9ef8b`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/8e9ef8b) When `parse_verdict` falls back to `pass`
  on unparseable JSON, emit a warning so quality regressions aren't hidden.
- [x] **P3.4 Sampler nit.** ✅ [`ff49068`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/ff49068) `system_sampler` primes *process* CPU but samples *system*
  CPU un-primed; align priming with what is sampled (or drop the priming call).

---

## Phase 4 — Test coverage (lock in the fixes)

Add tests that would have caught the bugs above.

- [x] **P4.1** Assert langgraph revision count == sequential/parallel for the same
  reviewer behavior (covers P2.2). ✅ [`518d12a`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/518d12a)
- [x] **P4.2** Test `_call_once` and streaming error paths (empty `choices`,
  malformed JSON, missing `usage`) — covers P3.1. Non-stream [`1d7572e`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/1d7572e) + streaming [`b05d139`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/b05d139).
- [x] **P4.3** Test that an errored worker does **not** produce a spurious `pass`
  verdict / is flagged invalid — covers P3.2. ✅ [`076654e`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/076654e)
- [x] **P4.4** Test re-dispatch policy parity across modes — covers P2.1. ✅ [`17a9c44`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/17a9c44)
- [x] **P4.5** Test the warmup/cache-reset bookkeeping — mock `/reset_cache` + warm-then-reset. ✅ [`b05d139`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/b05d139)

---

## Phase 5 — Analysis & reporting quality

- [x] **P5.1 Statistical rigor.** ✅ [`9815a11`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/9815a11) `summarize` now emits `n_runs` + `<metric>_ci95`.
  Increasing repeat counts on real hardware remains the author's call.
- [ ] **P5.2 Cache-cold control arm.** *(open — TODO-3)* Run each mode once with prefix
  caching disabled (or server restarted between modes) to isolate the topology effect
  from cache warming. Needs a real-hardware reset strategy (no vLLM runtime endpoint).
- [x] **P5.3 Refresh README results caveat** ✅ [`82ff812`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/82ff812) — added a methodology caveat;
  **regenerating the figures on real hardware remains open (TODO-4).**
- [x] **P5.4 Pin vLLM version / fail-loud on missing metrics.** ✅ [`82ff812`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/82ff812) (doc note) + [`083d1ed`](https://github.com/saravanakumardb1/systems_for_ml_meeting_assistant_multiagent/commit/083d1ed) (runtime warning).

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

## Open TODOs — sorted by what unblocks them

Flagged inline as `TODO-N` code comments so they can't slip.

**Reviewed 2026-07-28.** All four were filed under *"for the author to decide"*.
Re-read against the source, **only two are decisions.** The heading was doing
double duty: *deferred* and *needs a ruling* are different states, and conflating
them means a two-line change waits on a judgement nobody owes it, while the real
questions sit in the same undifferentiated list.

| | Item | Actually blocked on | Ready? |
|---|---|---|---|
| TODO-1 | `Authorization` header never sent | **Nothing** — no ruling needed | **Do it** |
| TODO-2 | Shared-prefix layout default | Owner, after a real-model A/B | No — measurement missing |
| TODO-3 | Cold-control reset strategy | Owner | Yes |
| TODO-4 | Regenerate committed figures | Real hardware | No — needs a GPU host |

### TODO-1 — `Authorization` header is never sent · **not a decision**

`agents/config.py:24` reads `VLLM_API_KEY`, and `agents/base.py` never sends it.
Confirmed rather than assumed: both `_call_stream` (`:144`) and `_call_once`
(`:179`) post to `/v1/chat/completions` with `json=payload` and
`timeout=config.REQUEST_TIMEOUT_S` and **no `headers=` argument at all**.

There is nothing to choose here. Adding the header is ~2 lines, and it is
**harmless when unused** — local vLLM ignores the value (hence the `"EMPTY"`
default) and the mock server does not inspect headers. The condition in the
current note, *"wire it only if a secured vLLM endpoint is needed"*, defers work
that costs less than the deferral: today the failure mode is that someone points
this at a secured endpoint, gets 401s, and has to discover why, with a config var
that reads as though it is wired.

**Do:** pass `headers={"Authorization": f"Bearer {config.API_KEY}"}` in both call
paths. **Done when:** `pytest` + `ruff` stay green against the mock.

### TODO-2 — Shared-prefix layout default · **decision, prereq not met**

`SHARED_PREFIX_LAYOUT` (`agents/config.py:39`) is implemented and defaults to `0`.
`agents/worker_util.py:31` puts `TRANSCRIPT` first and the role instruction last
when enabled — the layout that makes prefix caching actually shareable across
agents, which is the headline claim.

**Question:** does moving the role instruction to the end of a ~66k-token
transcript degrade output quality enough to matter?

| | Option | Cost |
|---|---|---|
| **A** | Flip the default to `1` | Free; makes the headline claim real by default |
| **B** | Leave `0`, keep the toggle | Free; the claim stays opt-in and mostly untested |

**Before this can be answered:** the A/B has never been run on a real model.
Answering now means guessing whether instruction-at-the-end costs quality — a
known-sensitive prompt-ordering question. Gather first: run both arms on real
hardware and compare output quality, then decide.

### TODO-3 — Cold-control reset strategy · **decision, answerable now**

`bench/runner.py:89-90` names the fork in a comment and leaves it open. The mock
already exposes `POST /reset_cache`; the question is only what the *real-hardware*
arm does.

**Question:** how does the cold-control arm clear the prefix cache between modes?

| | Option | Cost | Trade |
|---|---|---|---|
| **A** | Restart the server per mode | Slow sweeps; more orchestration | Truly cold, closest to a clean-room measurement |
| **B** | `--no-enable-prefix-caching` | Cheap; one flag | Measures *caching disabled*, not *cache cold* — subtly different claim |
| **C** | An HTTP reset endpoint, as the mock has | Fast | Depends on the real server exposing one; vLLM may not |

Worth deciding before the sweep is re-run for the results rewrite, since it
changes what the cold numbers mean.

### TODO-4 — Regenerate committed figures · **blocked on hardware, not a ruling**

The nine PNGs plus `summary.csv` in `results/figures/` were produced by the
pre-`--warmup`/pre-shuffle runner, so they encode the confound P1.1 removed.
`--warmup` (`bench/runner.py:170`) now exists and defaults to `1`.

No decision — it needs a GPU host and a re-run. Filed here only so the stale
figures are not mistaken for current. Ties to the Definition of done item
*"README Results section regenerated from a re-run sweep with the confound removed."*

---

## Doc review log

- All file/line claims re-verified against source at review time.
- Coverage gap closed: `scripts/make_synthetic_transcripts.py` reviewed → added P0.2a
  (dual-generator sizing mismatch).
- Corrected P1.1: removed the assumption of a vLLM runtime cache-reset endpoint;
  added mock-server order-dependence (offline/CI reproduces the warming artifact).
- `CONTRIBUTIONS.md` ("54 experiments", v5litepod-4) is consistent with the 54
  committed run JSONs and the README — no action needed.
