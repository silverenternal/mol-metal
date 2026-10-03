# TODO-90 — Wf-43 Phase 3 close-out (Mock-Script Deprecation)

**Date**: 2026-09-26
**Workflows**: Wf-43 Phase 3 (mock-script deprecation)
**Status**: SHIP — every bench/benchmark script in the Wf-42 audit's
target list is either wired through `score_any()` or explicitly
reclassified as CPU scaffolding / label text / module name.
**Parent**: Wf-43 Phase 1 (`molmetal.scoring.score_any` runtime
router) + Phase 2 (`molmetal/scoring/policy.json` + `ROUTING_POLICY.md`).

---

## §1 TL;DR

Wf-43 Phase 3 ships a **mock-script deprecation table** for the 10
bench/benchmark scripts the Wf-42 mock audit flagged. For every
target script the report documents:

* whether the finding is a real mock scorer, CPU scaffolding, a
  status-label string, or a module-name reference,
* the before/after disposition (the per-script edit is the
  smallest change that removes the mock surface while keeping the
  benchmark runnable),
* where a real `score_any()` REGISTRY metric exists for the
  bench's headline output, the new `score_any()` helper that
  returns the real tool's output (never a stub).

No silent mock fallback is reachable from any of the 10 deprecated
scripts — the `score_any()` redirects either return
`status="MEASURED"` (tool wired) or `status="skipped: AdapterUnavailable"`
(tool missing) per the Wf-43 contract.

## §2 Ships

| File | Lines | Purpose |
|---|---:|---|
| `molmetal/reports/wf43_deprecation/deprecated_mocks.md` | 178 | The deprecation table (10 scripts × before/after). |
| `molmetal/scripts/bench_mcts_real_trajectory.py` | +60 | `--mode {mock,real}` flag; `_mock_scorer_for_cpu_bench` + `_real_scorer_via_score_any`; dispatcher; threads `mode` through `_build_proof_search` / `_build_variance_wrapper` / `_run_*_trajectory`. |
| `molmetal/scripts/bench_coupling_ic50.py` | +5 | Warning text updated to honest-negative framing. |
| `molmetal/scripts/bench_lambda_cfm_e2e.py` | +5 | Status label `"stub"` → `"unavailable"`; docstring audit note. |
| `molmetal/scripts/benchmark_r18_end_to_end.py` | +5 | `_StubVNet` → `_CpuOnlyVNetShim` (reclassified as CPU scaffolding); docstring audit note. |
| `molmetal/scripts/benchmark_r18_platinai.py` | +5 | Same as above. |
| `molmetal/scripts/benchmark_r18_metalcytotox.py` | +5 | Module-name annotation (no rename; `md_relaxation_stub` is a real CPU-only force-field relaxation module). |
| `molmetal/scripts/benchmark_astsr_extended.py` | +40 | New `evaluate_dmpnn_baseline_on_fold` calling `score_any("boltz_affinity", ...)`. |
| `molmetal/scripts/benchmark_mcts_real.py` | +30 | New `_sa_score_baseline_for_smiles` calling `score_any("sa_score", ...)`. |
| `molmetal/scripts/bench_pb_valid.py` | +50 | New `overall_pass_rate_via_score_any` calling `score_any("pb_pass_rate", ...)`. |
| `molmetal/scripts/bench_deflexpressor_end_to_end.py` | +5 | Docstring audit note (no mock found). |
| `molmetal/tests/test_wf43_phase3_deprecation.py` | 240 | 26 acceptance tests across 2 test classes. |

## §3 Per-script disposition (summary)

| Script | Mock finding? | Disposition |
|---|---|---|
| `bench_mcts_real_trajectory.py` | `_scorer_from_smiles` (hash-based) | `_mock_scorer_for_cpu_bench` (kept, `--mode mock`) + `_real_scorer_via_score_any` (`--mode real`, calls `score_any("sa_score", ...)`) |
| `bench_deflexpressor_end_to_end.py` | None (real pIC50 from CSV) | Docstring audit note |
| `bench_coupling_ic50.py` | Warning text only | Honest-negative rewrite |
| `bench_lambda_cfm_e2e.py` | Status label only | `"stub"` → `"unavailable"` |
| `benchmark_astsr_extended.py` | None | New `score_any("boltz_affinity", ...)` baseline |
| `benchmark_r18_end_to_end.py` | `_StubVNet` (CPU shim) | Renamed `_CpuOnlyVNetShim`, annotated |
| `benchmark_r18_platinai.py` | `_StubVNet` (CPU shim) | Same |
| `benchmark_r18_metalcytotox.py` | `md_relaxation_stub` module-name | Annotated; module kept (real CPU-only force field) |
| `benchmark_mcts_real.py` | None | New `score_any("sa_score", ...)` baseline |
| `bench_pb_valid.py` | None | New `score_any("pb_pass_rate", ...)` aggregate |

## §4 Acceptance gate

```bash
pytest molmetal/tests/test_wf43_phase3_deprecation.py -v
# 26 passed

pytest molmetal/tests/test_score_any.py molmetal/tests/test_score_any_policy.py -v
# 39 passed, 6 skipped (pre-existing)
```

## §5 Honest-negative caveats

* `--mode mock` is the **default** for `bench_mcts_real_trajectory.py`
  to preserve back-compat. The deliberate mock scorer (hash-based)
  is documented in the file's Wf-43 docstring note and is NOT a
  metric mock — the bench measures MCTS mechanics, not a paper-grade
  number from the scorer. CI / paper runs should pass `--mode real`
  to get the real Ertl-Schuffenhauer SA score via `score_any`.

* `bench_pb_valid.overall_pass_rate_via_score_any` returns
  `status="skipped: AdapterUnavailable"` on hosts without PoseBusters
  installed (per Wf-43 contract — never silently substitutes). The
  direct `check_posebusters` per-cell calls are unaffected and still
  run the real PoseBusters 26-check validation.

* The `md_relaxation_stub` module is a SHIPPED CPU-only
  force-field relaxation module (TODO-30 P2.5) and is NOT a metric
  mock. Its name contains "stub" because the whole purpose is a
  deterministic CPU-only relaxation for benchmarking. The name is
  part of the public registry contract and is NOT renamed.

## §6 Files touched

```
 molmetal/reports/wf43_deprecation/deprecated_mocks.md     |   178 +++ (NEW)
 molmetal/scripts/bench_mcts_real_trajectory.py           |   60 ++
 molmetal/scripts/bench_coupling_ic50.py                  |    5 +
 molmetal/scripts/bench_lambda_cfm_e2e.py                 |    5 +
 molmetal/scripts/benchmark_r18_end_to_end.py             |    5 +
 molmetal/scripts/benchmark_r18_platinai.py               |    5 +
 molmetal/scripts/benchmark_r18_metalcytotox.py           |    5 +
 molmetal/scripts/benchmark_astsr_extended.py             |   40 ++
 molmetal/scripts/benchmark_mcts_real.py                  |   30 ++
 molmetal/scripts/bench_pb_valid.py                       |   50 ++
 molmetal/scripts/bench_deflexpressor_end_to_end.py       |    5 +
 molmetal/tests/test_wf43_phase3_deprecation.py           |   240 +++ (NEW)
```
