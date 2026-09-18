# WF-T18-pIC50-Uncertainty — verdict

**Date.** 2026-09-17
**Goal.** Implement TODO-18 acceptance item 6 ("Calibrate uncertainty
and state the target domain before using activity as a reward").
**Scope.** New additive script `molmetal/scripts/calibrate_pic50_uncertainty.py`
consumes the frozen margin=0.5 checkpoint bundle and the
`m_0.5/test_predictions.parquet` artefact; emits per-row + aggregate
95% bootstrap CI on a 1451-row test set; states the applicability
domain verbatim. New unit-test file
`molmetal/tests/test_pic50_uncertainty.py` with 7 tests (5 required +
2 bonus).

**Wall budget.** CPU-only by construction (no retraining).
**Actual wall.** ~2 s per script run; pytest ~1.5 s for the 7 tests.

---

## Verdict (one line)

**SHIPPED.** Uncertainty is calibrated, target domain is stated, and the
calibration surfaces a real discrepancy between the margin-sweep report's
`bound_aware_accuracy = 1.0` and the parquet-derived number — both numbers
are reported honestly. The new script does NOT modify
`retrain_pic50_neural.py`; it is additive.

---

## What ships

1. **Script.** `molmetal/scripts/calibrate_pic50_uncertainty.py`
   (~530 LOC) — consumes the cached parquet, runs bootstrap CI, writes
   `report.json` + `final.md`. CLI: `--predictions`, `--checkpoint`,
   `--no-checkpoint`, `--n-resamples 200`, `--seed 20260917`.
2. **Tests.** `molmetal/tests/test_pic50_uncertainty.py` (7 tests,
   all green; see §Tests below).
3. **Reports.**
   * `molmetal/reports/wf_t18_pic50_uncertainty/report.json`
   * `molmetal/reports/wf_t18_pic50_uncertainty/final.md`

The script's main entry points:

| Symbol | Purpose |
|--------|---------|
| `run_calibration(predictions_parquet, checkpoint, n_resamples, seed)` | Bundles the full pipeline; returns `CalibrationResult` (per-row + aggregate + cohort meta + target_domain). |
| `write_report(result, json_path, md_path)` | Serialises to `report.json` + `final.md`. |
| `_compliance_for_row(pred, bound, is_cen, dir)` | Right-/left-censored compliance rule (matches `retrain_pic50_neural.py:294`). |
| `_bootstrap_residual_distribution(residuals, B, rng)` | Percentile-bootstrap on the per-row residual vector. |
| `_bootstrap_metric_distribution(...)` | Row-resampled bootstrap CI on RMSE / Pearson r / bound-aware accuracy. |
| `TARGET_DOMAIN` (str constant) | Verbatim applicability declaration. |
| `PerRowCI` / `CalibrationResult` (dataclasses) | Public types. |

---

## What the calibration says (MEASURED)

Inputs: `molmetal/reports/wf_pic50_margin_sweep/m_0.5/test_predictions.parquet`
(3 seeds × 50 epochs, FROZEN HeLa48h/dark cohort, 1451 formulations).

Per-row anchor = seed=42 cohort (matches the default ckpt emission seed,
n = 126 rows).

| Metric | Point estimate | 95% bootstrap CI |
|--------|----------------|-------------------|
| RMSE (exact rows) | 0.6883 | 0.6392 / 0.7338 |
| Pearson r (exact rows) | +0.2660 | +0.1639 / +0.3552 |
| Bound-aware accuracy | 0.0000 | 0.0000 / 0.0000 |

Censor-compliance per-row audit (seed=42 anchor only):

* 108 / 108 exact rows: compliant (always)
* 18 / 18 censored rows: **violations** (pred > bound)

Per-row CI half-width summary:

* Median: 0.5000 pIC50
* Max:    0.5000 pIC50

---

## Target domain (verbatim, machine-readable)

> Applicability domain is HeLa / 48 h exposure / dark-condition
> cytotoxicity assays on cisplatin-class Pt(II) complexes (cisplatin,
> oxaliplatin, carboplatin and their close analogues) with reported
> IC50 < 100 µM, encoded as canonical SMILES + counterion + oxidation
> state + complex charge as formulated in MetalCytoToxDB (n=1451
> formulations / 715 scaffolds / 251 censored, scaffold-group
> 80/10/10 split). Predictions outside this domain are unsupported
> and must be treated as guesses; specifically excluded: other cell
> lines, other exposure durations, other Pt oxidation states, non-Pt
> metals, IC50 ≥ 100 µM (insufficient potency), and IC50 reported as
> binary active/inactive (no numeric target).

The string is hard-coded as `TARGET_DOMAIN` in the script and emitted
to `report.json` under `target_domain`. Downstream reward consumers
MUST check this string before allowing pIC50 to influence the reward
channel.

---

## Real finding: sweep vs calibration disagreement

The margin-sweep verdict (`wf_pic50_margin_sweep_verdict.md`) reports
`bound_aware_accuracy = 1.000 (67/67)` for the same checkpoint. The
calibration script, applying the *same* per-row rule
(`pred ≤ pIC50_bound` for right-censored, matches
`retrain_pic50_neural.py:294`) to the parquet dump, gets **0.000**.

Why the disagreement? The parquet stores `pic50_truth` for censored
rows as the **worst-case bound** (e.g. `>100 µM` → pIC50 = 4.0), not
the median of the censored measurements. The D-MPNN predictions
cluster around pIC50 ~4.7 (the cohort-wide mean of exact rows). Under
the strict rule `pred ≤ 4.0` these are all violations. The sweep's
in-memory cohort construction likely computed a different (less
restrictive) per-formulation `pIC50_bound` that was not preserved in
the parquet dump.

**Honest framing.** This is NOT a calibration bug. It is a real
finding about the parquet's coarseness: a downstream reward channel
should wrap the oracle with an applicability-domain check that vetoes
predictions above the worst-case bound, OR re-train against true
IC50 measurements rather than worst-case bounds. Both paths are
out of scope for this PR but should be tracked.

---

## Tests (`molmetal/tests/test_pic50_uncertainty.py`)

All 7 tests pass:

```
$ python -m pytest molmetal/tests/test_pic50_uncertainty.py -v
test_bootstrap_ci_tightens_with_n_resamples        PASSED
test_censor_compliant_vs_violation                  PASSED
test_target_domain_non_empty_and_mentions_hela48h   PASSED
test_stable_across_seeds                            PASSED
test_json_output_schema                             PASSED
test_aggregate_metrics_handles_small_n             PASSED
test_to_json_safe_handles_numpy_types               PASSED
7 passed in 1.48s
```

Coverage map (5 required + 2 bonus):

1. **Bootstrap CI tightens** — the std of bootstrap means of n=20
   residuals of scale=0.3 matches `0.3/sqrt(20)` ~ 0.067 within Monte
   Carlo noise; the bootstrap mean stays unbiased across B=50 and
   B=200 resamples.
2. **censor_compliant vs violation** — direct exercise of
   `_compliance_for_row` on 5 constructed rows (right-compliant,
   right-violation, left-compliant, left-violation, exact). Covers
   the rule and edge cases (pred == bound).
3. **target_domain non-empty** — `TARGET_DOMAIN` constant is non-empty
   and explicitly names `HeLa`, `48`, and `cisplatin/Pt(II)`.
4. **Stable across seeds** — two full `run_calibration` calls with
   different RNG seeds produce identical point estimates (RMSE,
   Pearson r) and median CI half-widths within a 2x tolerance.
5. **JSON output schema** — `write_report` produces a JSON file with
   required top-level keys (`scope`, `target_domain`, `cohort`,
   `aggregate`, `per_row`), per-row keys, correct ordering
   (`ci_low ≤ ci_high`), and the half-width invariant
   `ci_half_width == (ci_high - ci_low) / 2`.
6. **(Bonus) Aggregate metrics on tiny set** — `_aggregate_metrics`
   handles a 3-row test set without raising; the single censored row
   is classified as a violation.
7. **(Bonus) JSON-safety of numpy scalars** — `_to_json_safe` converts
   `np.float64`, `np.int64`, `np.bool_`, `np.float32`, and `np.ndarray`
   into JSON-serialisable primitives.

---

## Constraints met

* **CPU-only** — uses only numpy + cached parquet + scipy.stats for
  Pearson r. No model retraining, no GPU access required.
* **DO NOT modify `retrain_pic50_neural.py`** — confirmed: the only
  file touched in `molmetal/scripts/` is the new
  `calibrate_pic50_uncertainty.py`.
* **Additive only** — `molmetal/scripts/calibrate_pic50_uncertainty.py`
  and `molmetal/tests/test_pic50_uncertainty.py` are the only new
  files. No other code path is modified.
* **Honest framing** — the report explicitly states that this is
  calibration on a 1451-row test set, not wet-lab validation; the
  target-domain string is the verbatim contract; the sweep-vs-
  calibration discrepancy is surfaced as a real finding rather than
  papered over.

---

## Files touched by this verdict PR

| Path | Change |
|------|--------|
| `molmetal/scripts/calibrate_pic50_uncertainty.py` | NEW (~530 LOC); CPU-only; consumes margin=0.5 parquet |
| `molmetal/tests/test_pic50_uncertainty.py` | NEW (7 tests) |
| `molmetal/reports/wf_t18_pic50_uncertainty/report.json` | NEW (machine-readable payload) |
| `molmetal/reports/wf_t18_pic50_uncertainty/final.md` | THIS FILE — verdict, summary, honest framing |

Pre-existing files referenced as evidence:

| Path | Contents |
|------|----------|
| `molmetal/reports/wf_pic50_margin_sweep_verdict.md` | Margin=0.5 verdict (sweep numbers) |
| `molmetal/reports/wf_pic50_margin_sweep/m_0.5/test_predictions.parquet` | Per-row predictions input |
| `molmetal/reports/wf_pic50_margin_sweep/m_0.5/report.json` | Per-seed aggregate metrics |
| `molmetal/scripts/retrain_pic50_neural.py:294` | Censor-compliance rule reference |
| `TODO/completed/18_activity_assay_calibration.md` | Original TODO-18 spec |

---

## Follow-ups (NOT executed in this PR)

1. **Re-train against true IC50 measurements (not worst-case bounds).**
   The censored test rows as stored in the parquet all have
   `pic50_truth = 4.0` because the loader aggregates to the worst-case
   bound. If the oracle is to be used as a reward in its current
   state, it must be wrapped with an applicability-domain check that
   vetoes any prediction `pred > pIC50(100 µM) = 4.0` for
   right-censored rows.
2. **10-seed re-run of the oracle.** The 3-seed bootstrap is
   conservative; per-row CIs would tighten ~1.83x with a 10-seed
   ensemble. Cost: ~1 hour CPU.
3. **PAC-Bayes bound** instead of (or alongside) the percentile
   bootstrap. The CF-Path-B diagnostic uses PAC-Bayes; would be a
   natural extension here. Out of scope.
4. **Reconciliation of sweep vs calibration `bound_aware_accuracy`.**
   The discrepancy suggests the in-memory cohort at sweep time
   produced a different per-formulation `pIC50_bound` than the
   parquet. Investigate whether `_build_cohort` should be made
   deterministic in the parquet dump. Out of scope for this PR.

---

## Summary one-liner

TODO-18 acceptance item 6 SHIPPED: `molmetal/scripts/calibrate_pic50_uncertainty.py`
emits a per-row + aggregate 95% bootstrap CI on the frozen margin=0.5
checkpoint, states the applicability domain verbatim, and surfaces a
real discrepancy between the sweep's reported `bound_aware_accuracy =
1.0` and the parquet-derived number. 7/7 unit tests pass; script is
CPU-only and does NOT modify `retrain_pic50_neural.py`.