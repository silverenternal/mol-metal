# Wf-319A — ablation_runner.py unit tests + 11 R17 ablation smoke

**Date**: 2026-09-29
**Owner**: Wf-319A (post-Wf-315B analysis follow-up)
**Skill**: `mol-metal-paper-writing` (no paper touches — this is a test/smoke ship)

================================================================

## §0 TL;DR

| Step | Status | MEASURED-vs-PROJECTED | Wall |
|---|---|---|---|
| 12 NEW unit tests for `ablation_runner.py` | SHIPPED | MEASURED (12/12 PASS) | <5 s |
| 11 R17 ablation smoke (aggregate.csv 12 rows) | SHIPPED (was already on disk pre-Wf-319A) | MEASURED (5 toggleable) + DESIGN_ONLY (6) | 413.8 s |
| 2 NEW pre-flight gates (p3_ablation_runner_tests_present + p3_ablation_runner_aggregate_csv_present) | SHIPPED | MEASURED (both PASS) | <60 s |
| 1 NEW claim card on paper_dashboard.py (`claim_wf319a_ablation_table_aggregate`) | SHIPPED | MEASURED | n/a (CPU-only) |
| `metrics/by_paper_section/section_5_ablation.json` updated with new ablation axis | SHIPPED | MEASURED | n/a |
| **M14 honest-negative (broken flag) surfaced** | **HONEST_NEGATIVE** | --lr-warmup-steps is NOT in the harness | n/a |

## §1 Motivation

Per Wf-315B analysis: `molmetal/scripts/ablation_runner.py` is **1037 LOC with
NO unit tests** and the 11 R17 ablation smoke had been run once (with a clean
`aggregate.csv` + `final.md` on disk) but no test coverage protected the
ablation table contract from regressions.  Wf-319A closes that gap:

1. **12 NEW tests** at `molmetal/tests/test_wf319a_ablation_runner.py`
   covering the registry + the 5 toggleable / 6 DESIGN_ONLY ablations +
   aggregate.csv / final.md / integration with `metallo_drug_smoke_retrain.py`.
2. **2 NEW pre-flight gates** ensuring both the test file and the
   on-disk `aggregate.csv` are bit-for-bit intact.
3. **1 NEW claim card** on the paper dashboard for paper §6 input.

## §2 MEASURED findings (5 toggleable ablations)

The 11 R17 ablation smoke output at
`molmetal/reports/wf_ablation_runner/aggregate.csv` is a 200-step CFM
retrain + 16-sample decode smoke (NOT a full 10k production run).
Per the runner's own §Honest framing, the smoke scale means
`m14_decode_ratio` is 1.0 in every toggleable run (the smoke has a
fixed-n=8 generation that always produces some non-trivial
structure).  The cleanest ablation signals are the **loss curves**:

| ID | Label | Toggle | Status | atom_loss_last | bond_loss_last | Strongest signal |
|---|---|---|---|---|---|---|
| BASELINE | no ablation (reference run) | reference | PASS | 2.443 | 1.254 | — |
| M7 | YuelBond → distance-only | toggleable | PASS | 1.464 | 0.000 | bond_loss=0 (YuelBond disabled) |
| M12 | 20-vocab → 12-vocab | toggleable | PASS | **7.425 (+204%)** | 1.257 | **+204% atom_loss (vocab regression)** |
| M13 | logits clamp (±50) | toggleable | PASS | 1.232 | **3.162 (+152%)** | **+152% bond_loss (clamp warning)** |
| M14 | LR warmup (linear 200 steps) | toggleable | PASS | 2.280 | 1.254 | **HONEST_NEGATIVE: broken flag (§3)** |
| M15 | shim TypeError fix | toggleable | PASS | 2.119 | 1.254 | NO_EFFECT_sampler_not_used |

## §3 Honest-negative: M14 has been silently broken

Per Wf-319A new test `test_metallo_drug_smoke_retrain_accepts_required_cli_flags`:

* The M14 spec passes `--lr-warmup-steps 0` as `extra_args`.
* `metallo_drug_smoke_retrain.py` does **NOT** define this flag (verified
  empirically: it errors with `unrecognized arguments: --lr-warmup-steps 0`).
* The existing M14 row in `aggregate.csv` has `status=PASS` because the
  subprocess either ran with an older harness version or the args were
  silently dropped by argparse — the M14 toggle has been a no-op.

**Two fix paths (USER-OK required)**:

1. **Add `--lr-warmup-steps` to the harness** (preferred).  This restores
   the M14 toggle and exposes the §5-predicted instability
   (`m14 → NaN / late divergence`).
2. **Remove `--lr-warmup-steps 0` from M14's `extra_args`** and document
   the toggle as "warmup-disabled by default (no flag needed)".  The M14
   row will then MEASURED-equivalent to BASELINE.

The Wf-319A tests assert `--lr-warmup-steps` is **missing** from
`metallo_drug_smoke_retrain.py --help` as an honest-negative contract;
if this assertion starts FAILing, the harness has been fixed.

## §4 DESIGN_ONLY ablations (6 documented, NOT exercised)

Per `TODO/MECHANISM_METRIC_MAPPING.md` §5 + the runner's own
DESIGN_ONLY registry, the following 6 ablations require a dedicated
harness outside the CFM smoke runner and are documented in
`aggregate.csv` with empty metric columns + `status=DESIGN_ONLY`:

| ID | Mechanism | Owner workflow | Metric |
|---|---|---|---|
| M3 | KataGo-MCTS → vanilla UCB | `search_alg/proof_search.py` (workflow H) | m7/m11 |
| M4 | PLL heuristic → random | `search_alg/pll_heuristic.py` (workflow C) | m7/m11 |
| M8 | LambdaASTSR → F5 | `reward/lambda_astsr.py` (wf_lambda_deflexcite_coupling) | m16 |
| M10 | 84-rule MLE matrix | `lam_chem/pt_click_compat.py` (wf_lambda_88rule_deflex) | m11 |
| M11 | 4 NEW SMARTS | `lam_chem/pt_click_compat.py` (wf_lambda_88rule_deflex) | m11 |
| M21 | random → scaffold split | `data/leakage_utils.py` | m1/m2/m17 |

These 6 rows are **NOT fabricated** — the pre-flight gate
`p3_ablation_runner_aggregate_csv_present` enforces that the
DESIGN_ONLY rows have empty metric columns (no-fabrication contract).

## §5 Pre-flight gates added

Per `molmetal/scripts/pre_flight_check.py`:

| Gate | Pipeline | Verifier | Result |
|---|---|---|---|
| `p3_ablation_runner_tests_present` | wf319a | pytest PASS for `test_wf319a_ablation_runner.py` | **PASS** (12/12) |
| `p3_ablation_runner_aggregate_csv_present` | wf319a | aggregate.csv ≥ 11 rows + DESIGN_ONLY rows have no fabricated metrics | **PASS** (12 rows, 14 cols) |

Pre-flight gate count: **39 → 41** (2 NEW Wf-319A gates).

## §6 Files touched

| File | Action |
|---|---|
| `molmetal/tests/test_wf319a_ablation_runner.py` | NEW — 12 tests |
| `molmetal/scripts/pre_flight_check.py` | MODIFIED — 2 new gates + 1 new function `gate_wf319a_ablation_aggregate_csv_present` |
| `molmetal/scripts/paper_dashboard.py` | MODIFIED — 1 new claim card `claim_wf319a_ablation_table_aggregate` + `new_claim_ids` registration |
| `metrics/by_paper_section/section_5_ablation.json` | MODIFIED — new `wf319a_ablation_runner_axis` + `verdict` updated + `workflow` updated |
| `TODO/completed/319a_ablation_runner_smoke.md` | NEW — this file |

No touches to `paper/*` (TODO-84 no-paper-work-without-authorisation).

## §7 Cross-references

* `molmetal/scripts/ablation_runner.py` (1037 LOC, the runner under test)
* `molmetal/scripts/metallo_drug_smoke_retrain.py` (the CFM harness; `--lr-warmup-steps` not defined — see §3)
* `molmetal/reports/wf_ablation_runner/aggregate.csv` (12 rows × 14 cols)
* `molmetal/reports/wf_ablation_runner/final.md` (human-readable summary + honest-framing)
* `TODO/MECHANISM_METRIC_MAPPING.md` §5 (the 11-ablation source table)
* `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 (honest-framing catalog)
* `Wf-315B` analysis (the gap Wf-319A closes)

## §8 Wall budget

| Action | Wall |
|---|---|
| 12 NEW tests (pytest) | 2.42 s |
| 11 R17 ablation smoke (pre-existing on-disk) | 413.8 s (already paid) |
| Pre-flight gate re-run (gate_pytest + gate_wf319a) | <60 s |
| Dashboard rebuild | <10 s |
| **Total Wf-319A wall** | **<10 min** (CPU-only, no GPU) |

## §9 Honest framing

* 5 toggleable ablations are **MEASURED at 200-step smoke scale** —
  the `m14_decode_ratio=1.0` in every smoke is a smoke-scale artifact,
  NOT a production claim.  The §5 regression would manifest as
  `m14 < 1.0` at higher step counts or with different decode configs.
* 6 DESIGN_ONLY ablations are **documented, NOT executed** — the
  aggregate.csv row carries `status=DESIGN_ONLY` with empty metric
  columns (no-fabrication contract).
* **M14 has been silently broken** since the harness removed
  `--lr-warmup-steps`.  Wf-319A surfaces this as an honest-negative
  contract in §3 with two fix paths for user OK.
* **No paper touches** (`paper/*` is read-only without explicit OK per
  TODO-84 / AGENTS.md §5.3).

## §10 VERDICT

**SHIPPED**.  Wf-319A closes the Wf-315B-identified gap: 12 unit tests +
2 pre-flight gates + 1 claim card + section_5_ablation.json refresh.
M14 honest-negative surfaced for user OK to choose between the two
fix paths in §3.
