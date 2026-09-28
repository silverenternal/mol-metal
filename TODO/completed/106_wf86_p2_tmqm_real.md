# TODO-106 — Wf-86 Phase 2 close-out (tmQM d-block real R²)

**Date**: 2026-09-26
**Workflow**: Wf-86 (Phase 2 of `49778e6`) — REAL cross-dataset R² on tmQM d-block 5-element
**Status**: ✅ **SHIP (honest-negative)** — Script + 25 tests PASS; MEASURED aggregate R² = −0.1974 across 5 d-block elements (target `coord_number`)
**Priority**: **MEDIUM** (closes Wf-49 PARTIAL Phase 4 cross-dataset gap on tmQM; honest anchor)
**Owner**: Hugo Lee
**Parent**: `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 PARTIAL SHIP — Phase 4 tmQM QUEUED) + `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 follow-up) + Wf-86 Phase 1 (`TODO/completed/106_wf86_p1_metalcytotox.md` for sibling ship)
**Constraint**: `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate) — code + bench only

---

## §0 TL;DR

Wf-86 Phase 2 implements the **Wf-49 PARTIAL Phase 4 cross-dataset R² close-out on tmQM d-block** with the same Hilbert+ASTSR combined harness as Wf-86 Phase 1 (MetalCytoToxDB 5-metal).  The script `molmetal/scripts/bench_hilbert_astr_real_tmqm.py` runs **5 seeds × 50 epochs × strict-scaffold 5-fold CV** on the **5 largest first-row d-block elements** (Cu / Fe / Ni / Zn / Co — total 1 501 rows out of 5 000 in the tmQM d-block subset), targeting the **`coord_number`** regression column.

The **MEASURED** result is **−0.1974** (5-element aggregate combined R²) — the Hilbert+ASTSR blend **does not beat predict-the-mean** for the `coord_number` regression task on this dataset.  This is the **honest-negative anchor** per `party-twenty-spirit §3.5`: we report the negative result and explain why instead of fabricating positive numbers.

Three observations explain the negative result:
1. **ASTSR dominates the blend** — `alpha_mean ≈ 0.10` across all 25 fits.
2. **Hilbert overfits dramatically** — Hilbert-only R² = **−1.9 to −10.2** (N >> F failure mode).
3. **ASTSR is at-baseline** — ASTSR-only R² ranges from **−0.08 to −0.44**, clustered near 0 because `coord_number` is a **structural property of the metal centre**, not a property of the ligand envelope — no 12-d RDKit 2D feature strongly correlates with it.

| Deliverable | Result | Path |
|---|---|---|
| Wf-86 P2 script | ✅ SHIP | `molmetal/scripts/bench_hilbert_astr_real_tmqm.py` (~900 LOC) |
| Wf-86 P2 tests | ✅ SHIP | `molmetal/tests/test_hilbert_astr_real_tmqm.py` (25/25 tests PASS) |
| Wf-86 P2 per-element metrics | ✅ MEASURED | `molmetal/reports/wf86_p2_tmqm_real/{Cu,Fe,Ni,Zn,Co}/per_seed_metrics.json` |
| Wf-86 P2 synthesis | ✅ SHIP | `molmetal/reports/wf86_p2_tmqm_real/synthesis.md` (honest-negative framing) |
| Wf-86 P2 verify gate | ❌ FAIL (honest) | aggregate combined R² = −0.1974 < 0.0 |

---

## §1 Per-deliverable detail

### §1.1 Wf-86 P2 script

**File**: `molmetal/scripts/bench_hilbert_astr_real_tmqm.py` (~900 LOC)

Same architecture as Wf-86 P1 (`bench_hilbert_astr_real_metalcytotox.py`), adapted for tmQM d-block:

* **Data loader** — `load_element_cohort(element, target_col=...)` reads
  `molmetal/data/tmqm_dblock_subset.csv`, filters by `metal_center`, drops
  rows with NaN/unparseable SMILES, returns `(X, y, smiles, scaffold_ids)`.
* **Two-tier scaffold fallback** — Bemis-Murcko returns "" for the many
  open-chain ligands in tmQM d-block; the loader falls back to
  canonicalised ligand SMILES (with metal atom removed) when Bemis-Murcko
  is empty, ensuring every row has a non-empty scaffold key for the
  strict-scaffold 5-fold split.
* **Combined Hilbert + ASTSR branch** — identical to Wf-86 P1: 12-d RDKit
  2D descriptor → 64-d Hilbert feature map (linear + bias + variance-ranked
  pairwise interactions) → `LambdaASTSRHilbert` PD optimisation; ASTSR
  uses 24-population × 15-generation × depth-4 evolutionary search
  (slightly leaner than P1's 32 × 20 for the smaller per-element cohorts).
* **Per-fold alpha selection** — 90/10 inner validation, grid over
  `alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`; the chosen `alpha` is reported per fold.
* **CLI flags** — `--dataset tmqm_dblock`, `--elements Cu Fe Ni Zn Co`
  (or any subset), `--target coord_number` (or `logp`, `qed_score`,
  `sa_score`, `n_atoms`, `mw`, `n_rings`, `formal_charge`), `--seeds 0 1 2 3 4`,
  `--epochs 50`, `--n-folds 5`, `--strict-scaffold true`,
  `--device cpu|cuda`, `--out-dir`.

### §1.2 Wf-86 P2 tests

**File**: `molmetal/tests/test_hilbert_astr_real_tmqm.py` (~340 LOC, **25 tests**)

Test inventory:
1. `test_feature_columns_constant` — 12-d F5_RDKIT_2D schema matches Wf-49 conventions.
2. `test_default_elements_constant` — default 5-element cohort is `("Cu", "Fe", "Ni", "Zn", "Co")`.
3. `test_supported_targets_constant` — supported target columns stable.
4. `test_tmqmcsv_candidates_constant` — CSV path discovery has at least one candidate path.
5. `test_counterion_fragments_constant` — counter-ion fragment list stable.
6. `test_extract_ligand_smiles_drops_counterions` — `L1.[Cl-].L2` → `L1.L2`.
7. `test_extract_ligand_smiles_keeps_largest` — 3 fragments → largest kept.
8. `test_extract_ligand_smiles_returns_none_for_empty` — empty SMILES → `None`.
9. `test_compute_r2_zero_for_perfect` — perfect predictor ⇒ R² = 1.
10. `test_compute_r2_zero_for_constant` — constant predictor ⇒ R² = 0.
11. `test_compute_r2_negative` — bad predictor ⇒ R² < 0.
12. `test_collapse_detector` — pred_std / y_std < 0.1 ⇒ True.
13. `test_blend_alphas_constant` — BLEND_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0).
14. `test_strict_scaffold_disjoint` — each scaffold in at most one fold's test set.
15. `test_strict_scaffold_covers_all` — union of test folds = full set.
16. `test_strict_scaffold_deterministic` — same seed ⇒ identical splits.
17. `test_random_kfold_no_overlap` — random fallback: train / test never overlap.
18. `test_kfold_indices_dispatches_to_strict_when_enabled` — strict path produces disjoint scaffolds.
19. `test_hilbert_feature_map_shape` — output shape (N, 64) for train, (M, 64) for test.
20. `test_hilbert_feature_map_no_test` — output shape (N, 64) when no test set.
21. `test_predict_hilbert_clips` — predictions clipped to ±PRED_CLIP.
22. `test_load_element_cohort_cu_smoke` — Cu cohort: X (n, 12), y finite, scaffolds non-empty.
23. `test_load_element_cohort_unsupported_target` — bad target column raises clear error.
24. `test_write_synthesis_creates_file` — `synthesis.md` written with required sections.
25. `test_combined_baseline_smoke` — CLI end-to-end smoke (`--elements Cu --seeds 0 --epochs 5 --n-folds 2`).

### §1.3 Wf-86 P2 per-element metrics

`molmetal/reports/wf86_p2_tmqm_real/{element}/per_seed_metrics.json` for each of:
* `Cu` (174 rows) — combined R² = −0.1101 ± 0.0273
* `Fe` (261 rows) — combined R² = −0.4446 ± 0.5048 (high variance due to seed 1-2 outlier)
* `Ni` (522 rows, largest cohort) — combined R² = −0.0874 ± 0.0169 (most stable)
* `Zn` (365 rows) — combined R² = −0.1156 ± 0.0353
* `Co` (179 rows) — combined R² = −0.2293 ± 0.1519 (worst, smallest cohort)

**5-element aggregate combined R²** = **−0.1974 ± 0.1331** (mean of 5 element means).

Per-seed JSON shape mirrors Wf-86 P1: `combined_r2_overall`, `hilbert_r2_overall`, `astr_r2_overall`, `combined_r2_per_fold_mean`, `alpha_mean`, `per_fold[]` (each with `hilbert_r2`, `astr_r2`, `combined_r2`, `alpha_chosen`, `collapse_detected`).

### §1.4 Wf-86 P2 synthesis

**File**: `molmetal/reports/wf86_p2_tmqm_real/synthesis.md` (~6.7 KB)

Sections:
* Header with target / constraint / hardware / split / inputs.
* Cross-element summary table (5 elements × 5-seed × 5-fold CV).
* Verify gate verdict.
* **Interpretation** — three observations explaining the negative result
  (ASTSR dominates / Hilbert overfits / ASTSR at-baseline).
* Per-element per-seed R² detail (5 tables).
* Honest framing — 7 bullets including scaffold fallback explanation,
  per-element cohort sizes, alpha selection, collapse detector, CPU
  smoke, no Multi-PH + PLL, and the production anchor.

---

## §2 MEASURED vs PROJECTED

### §2.1 MEASURED (this build, CPU-only, real data)

* **Aggregate combined R² (5 elements × 5 seeds × 5 folds)** = **−0.1974 ± 0.1331**
  — the Hilbert+ASTSR blend does **not** beat predict-the-mean for
  `coord_number` regression on tmQM d-block 5-element.
* **Per-element breakdown**:
  * Ni (largest, 522 rows): −0.0874 ± 0.0169 — closest to baseline
  * Cu: −0.1101 ± 0.0273
  * Zn: −0.1156 ± 0.0353
  * Co (smallest, 179 rows): −0.2293 ± 0.1519 — worst
  * Fe: −0.4446 ± 0.5048 — highest variance, driven by seeds 1-2 outliers
* **Hilbert-only R²** = **−1.9 to −10.2** — Hilbert branch overfits the
  64-d feature map on 100-500 train rows.
* **ASTSR-only R²** = **−0.08 to −0.44** — ASTSR branch clusters very
  close to baseline because `coord_number` is a metal-centre structural
  property, not a property of the ligand envelope.
* **alpha_mean** = **~0.10** across all 25 fits — the inner-validation
  grid consistently down-weights Hilbert.

### §2.2 PROJECTED (path to R² ≥ 0)

The negative result is the **MEASURED honest anchor** for the
Hilbert+ASTSR combined approach on this task.  Path to R² ≥ 0:

1. **Per-element structural features** — dative-bond topology
   (count donor atoms, types of donor atoms, geometric
   descriptor of coordination geometry).
2. **D-MPNN embeddings of the full metal-coordination complex** —
   pretrained on tmQM (Wipff / 42_tmqm_pretrain_v4_encoder.md ship),
   not just the ligand envelope.
3. **Multi-PH + PLL augmentation** — Wf-79 P2 workstream
   (TODO/pending/94_r20_p3_multiph_pll_feature.md).
4. **Constrained-primal-dual Hilbert** — explicit regularization or
   rank-truncation for the N << F underdetermined regime.

---

## §3 Cross-references

* `molmetal/scripts/bench_hilbert_astr_real_tmqm.py` — Phase 2 script
* `molmetal/tests/test_hilbert_astr_real_tmqm.py` — 25 tests PASS
* `molmetal/reports/wf86_p2_tmqm_real/{Cu,Fe,Ni,Zn,Co}/per_seed_metrics.json` — per-element per-seed JSON
* `molmetal/reports/wf86_p2_tmqm_real/synthesis.md` — synthesis
* `molmetal/reports/wf86_p2_tmqm_real/run.log` — clean summary log
* `molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py` — Wf-86 P1 sibling (MetalCytoToxDB 5-metal)
* `molmetal/data/tmqm_dblock_subset.csv` — 5 000-row tmQM d-block subset
* `TODO/completed/91_wf49_r2_root_fix.md` — Wf-49 PARTIAL SHIP (parent)
* `TODO/completed/91b_wf49_p3p4_followup.md` — Wf-69 follow-up close-out
* `TODO/completed/105_wf85_p1_hilbert_only_real.md` — Wf-85 P1 (PlatinAI 4-cell Hilbert-only)
* `TODO/pending/84_no_paper_task_without_authorization.md` — paper-gate

---

## §4 Honest framing (party-twenty-spirit §3.5)

Wf-86 Phase 2 closes the **Wf-49 PARTIAL Phase 4 cross-dataset R²
synthesis gap on tmQM d-block** with the **MEASURED** honest-negative
anchor.  The aggregate combined R² = **−0.1974** is reported as-is —
the Hilbert+ASTSR blend does **not** beat predict-the-mean for
`coord_number` regression on this dataset, and we explain why
(ASTR dominates / Hilbert overfits / ASTSR at-baseline) instead of
fabricating positive numbers.

The 25-test pytest suite passes; the script supports 8 regression
targets (`coord_number`, `logp`, `qed_score`, `sa_score`, `n_atoms`,
`mw`, `n_rings`, `formal_charge`) for any subset of the 30 d-block
metals in the tmQM d-block subset.  No Multi-PH + PLL (separate
workstream); CPU-only (GPU-1 RTX 5090 optional but not required for
the Wf-86 P2 production anchor).
