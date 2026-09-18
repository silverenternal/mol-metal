# WF-Pending-Code-Batch3 — MASTER (TODO-06, TODO-18, TODO-25)

**Date.** 2026-09-17
**Branch.** metallodrug-de-novo-vertical (worktree-local)
**Scope.** Close three pending code tasks in one batch: TODO-18 pIC50
uncertainty calibration (acceptance item 6), TODO-25 metallodrug-specific
proxies, TODO-06 LightGBM + LOMO + temporal holdout baselines.

---

## TL;DR

3 code tasks SHIPPED, all CPU-only: pIC50 uncertainty calibration
(95% bootstrap CI + verbatim target domain), 4 metallodrug proxies
(reduction_potential / trans_effect / LFSE / pt_dna_crosslink) wired
into the Lambda cell report, and LightGBM + LOMO + temporal holdout
baselines for MetalCytoToxDB. 0 GPU retrain claimed.

---

## Outcomes

| Workflow | Verdict | New LOC | Tests (new/total) | Files |
|----------|---------|---------|-------------------|-------|
| T18 pIC50 uncertainty | SHIPPED | ~530 | 7 / 7 | calibrate_pic50_uncertainty.py + test_pic50_uncertainty.py |
| T25 metallodrug proxies | SHIPPED | ~640 (340 module + 150 CLI + 152 tests) | 34 / 34 new + 48 / 48 existing = 82 / 82 | metallodrug_property_proxies.py + test_metallodrug_proxies.py + r4_lambda_only_run.py CLI |
| T06 baselines (LightGBM / LOMO / temporal) | SHIPPED | ~880 (250 + 330 + 290 + 12 tests) | 12 / 12 new + 7 / 7 existing = 19 / 19 | morgan_lightgbm.py + baseline_lomo_cv.py + baseline_temporal_holdout.py + test_baselines_lightgbm_lomo_temporal.py |

All three pass on CPU-only, deterministic pytest runs.

---

## MEASURED deltas

### T18 pIC50 uncertainty calibration
- 95% bootstrap CI on frozen margin=0.5 checkpoint (3 seeds × 50 epochs, 1451-row test set, 80/10/10 scaffold split).
- **RMSE (exact rows)** = 0.6883 [95% CI 0.6392 / 0.7338].
- **Pearson r (exact rows)** = +0.2660 [95% CI +0.1639 / +0.3552].
- **Bound-aware accuracy** = 0.0000 [95% CI 0.0000 / 0.0000] (vs sweep verdict's 1.000 — discrepancy surfaced honestly).
- Per-row CI median / max half-width = 0.5000 pIC50.
- **Target domain (verbatim, machine-readable)**: HeLa / 48 h / dark-condition cytotoxicity assays on cisplatin-class Pt(II) complexes (cisplatin, oxaliplatin, carboplatin + close analogues), IC50 < 100 µM, encoded as canonical SMILES + counterion + oxidation state + complex charge per MetalCytoToxDB (n=1451 / 715 scaffolds / 251 censored, scaffold-group 80/10/10 split). Excluded: other cell lines, other exposure durations, other Pt oxidation states, non-Pt metals, IC50 ≥ 100 µM, binary active/inactive reports.

### T25 metallodrug proxies (4 ship)
- `reduction_potential_proxy(smiles, metal="Pt")` — Pt(II)/Pt(IV) redox liability from inner-sphere ligand-field strength (Shriver & Atkins Table 17.7; Reedijk 1996).
- `trans_effect_proxy(smiles, metal="Pt")` — count of high-trans-effect ligands (CN-, CO, NO2-, PR3, C2H4) in inner sphere, normalised to [0, 1] by 4-coordinate Pt(II) (Appleton 1997, Coord. Chem. Rev. 166).
- `lfse_proxy(smiles, metal="Pt", d_electron_count=8)` — ligand-field stabilisation energy proxy for d8 square-planar Pt(II); 0.0 for d0 / d10 (Miessler, Fischer & Tarr 2014).
- `pt_dna_crosslink_proxy(smiles)` — Pt-DNA covalent crosslink propensity = labile Pt-Cl/Pt-O bonds × logP membrane scaling; 0.0 for non-Pt (Wang & Lippard 2005; Cohen 2007).
- All four return values in `[0, 1]`; non-Pt / invalid SMILES return 0.0 gracefully.
- Smoke on cisplatin (`[H]N...Pt(Cl)(Cl)(NH3)(NH3)`): reduction_potential=0.4133, trans_effect=0.0, lfse=1.0000 (d8 + 4 strong-field ligands), pt_dna_crosslink=0.5344 (2 labile Pt-Cl + logP ~ -2.4).
- `--metallodrug-proxies` CLI flag on `r4_lambda_only_run.py` (default OFF, bit-exact backward compat); 4 new columns added to CellResult: `metallodrug_reduction_potential_mean`, `metallodrug_trans_effect_mean`, `metallodrug_lfse_mean`, `metallodrug_pt_dna_crosslink_mean`.

### T06 baselines (LightGBM + LOMO + temporal holdout)
- **LightGBM parallel to XGBoost** shipped — same kwargs (metal / morgan_radius / morgan_nbits / seed / splitter), JSON-shape parity, splitter dispatch identical. Import-guarded with `lightgbm_available()`.
- **Ru test ROC-AUC**: LightGBM 0.9143 vs XGBoost 0.9206 (Δ |XGB − LGB| ≤ 0.0063 AUC on 367-row test set — faithful parallel GBDT).
- **Leave-One-Metal-Out CV** script ships: train on all metals except held-out, eval on held-out, mean ± std Pearson r. Smoke run on Ru/Ir (200 rows/metal): mean Pearson r = +0.0127 ± 0.0609, mean ROC-AUC = 0.4671 ± 0.1133. Cross-metal transfer is hard at 200-row scale — full-data run deferred to F1.
- **Temporal holdout** script ships (cutoff=2023 default, configurable): post-2023 Ru ROC-AUC = **0.5530** vs random-split baseline 0.9206 (Δ ≈ 0.37 AUC) — the **temporal leak the random split hides** (93% SMILES seen in train, per leakage_diagnosis.md). **Headline finding** for paper §6 limitations anchor.

---

## What remains BLOCKED

1. **PlatinAI oracle accuracy** — T18 calibration surfaces a real discrepancy between sweep's `bound_aware_accuracy = 1.0` and parquet-derived 0.000; reconciliation requires investigating `_build_cohort` determinism in the parquet dump.
2. **10-seed pIC50 re-run** — 3-seed bootstrap is conservative; per-row CIs would tighten ~1.83x with a 10-seed ensemble (~1 h CPU).
3. **Wet-lab ground-truth calibration of metallodrug proxies** — CV for reduction_potential, Kb assays for pt_dna_crosslink; out of scope per existing user constraint.

---

## Honest framing

- **Proxies, not measurements.** The metallodrug reduction_potential / trans_effect / LFSE / pt_dna_crosslink numbers are screening heuristics built from published spectrochemical / kinetic series. Useful for relative ranking in a reward channel; NOT a substitute for cyclic voltammetry (true E1/2), DNA-mobility-shift Kb assays (true crosslink rate constants), or spectroscopic ligand-field splitting measurements. Each value is clipped to `[0, 1]` so it folds into the MCTS aggregator without dominating Vina / SA / QED.
- **Sweep-vs-calibration disagreement is a real finding.** The parquet stores `pic50_truth` for censored rows as the worst-case bound (e.g. `>100 µM` → pIC50 = 4.0), not the median. D-MPNN predictions cluster around pIC50 ~4.7 → all are strict-rule violations. The sweep's in-memory `pIC50_bound` was less restrictive than the parquet dump.
- **LOMO smoke at 200 rows/metal is near zero by design.** Cross-metal transfer of an activity-prediction classifier is hard; production needs all rows × 3 seeds (~30 min).
- **Temporal holdout is the honest anchor for §6.** Ru random-split AUC = 0.92 vs post-2023 Ru AUC = 0.55 — the random-split test rows have seen SMILES in train 93% of the time. This closes the leakage door that the random-split baseline hid.
- **0 GPU retrain claimed.** All three workflows are CPU-only by construction; no model retraining, no GPU access required.

---

## Files touched (all paths absolute)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/calibrate_pic50_uncertainty.py` (NEW, ~530 LOC)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_pic50_uncertainty.py` (NEW, 7 tests)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metallodrug_property_proxies.py` (NEW, 340 LOC)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_metallodrug_proxies.py` (NEW, 152 LOC, 34 tests)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py` (EDIT, +~150 LOC for CLI flag + 4 wrappers + 4 CellResult fields + aggregate/serialise/summary_md)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/baselines/morgan_lightgbm.py` (NEW, ~250 LOC)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/baselines.py` (EDIT, +14 lines for `--model lightgbm`)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/baseline_lomo_cv.py` (NEW, ~330 LOC)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/baseline_temporal_holdout.py` (NEW, ~290 LOC)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_baselines_lightgbm_lomo_temporal.py` (NEW, 12 tests)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/baseline_lomo_xgb.json` (NEW, 2-metal smoke)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/baseline_temporal_2023_xgb.json` (NEW, Ru/Ir smoke)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_t18_pic50_uncertainty/{report.json,final.md}` (NEW)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_t25_metallo_proxies/final.md` (NEW)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_t06_baselines_lightgbm/final.md` (NEW)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pending_code_batch3_done/MASTER.md` (THIS FILE)

---

## Cross-references

- `molmetal/reports/wf_t18_pic50_uncertainty/final.md` — T18 verdict
- `molmetal/reports/wf_t25_metallo_proxies/final.md` — T25 verdict
- `molmetal/reports/wf_t06_baselines_lightgbm/final.md` — T06 verdict
- `molmetal/reports/wf_pic50_margin_sweep_verdict.md` — pre-existing margin sweep that T18 reconciles against
- `molmetal/reports/leakage_diagnosis.md` — temporal leak evidence for T06 §6 anchor
- `molmetal/reports/wf_3_citeonly_sota.tex` — SOTA citation table (T06 LightGBM not yet wired — F4)
- `TODO/pending/22_data_gap_alignment_plan.md` — TargetDiff 25-metric gap (T25 closes 4/25)
- `TODO/pending/25_round14_lit_grounded_plan.md` — lit-grounded plan that motivated T25 proxies
- `TODO/completed/18_activity_assay_calibration.md` — original TODO-18 spec
- `TODO/06_milestones/milestones.md` — TODO-06 Phase 4 cross-metal generalisation spec

---

## Follow-ups (NOT executed in this batch)

- **T18 follow-ups:** (1) re-train oracle against true IC50 measurements (not worst-case bounds); (2) PAC-Bayes bound alongside percentile bootstrap; (3) reconcile sweep vs calibration `bound_aware_accuracy`.
- **T25 follow-ups:** wet-lab ground-truth calibration (CV for reduction_potential, Kb assays for pt_dna_crosslink) per existing user constraint.
- **T06 follow-ups:** F1 full LOMO on 5 metals × 3 seeds; F2 `wf_baseline_lomo_full.json`; F3 `--cutoff-year` grid runner {2020..2024}; F4 wire LightGBM into `wf_3_citeonly_sota.tex`.
