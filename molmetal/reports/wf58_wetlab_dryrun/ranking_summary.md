# Wf-58 Phase 1 — Compound Prioritization Dry-Run

**Date**: 2026-09-26
**Status**: DRY-RUN (no wet-lab synthesis)
**Top-K**: 20

## Inputs

* `platinai_5k.csv` (5,000 generated candidates)
* `metalcytotox_rdkit_features.csv` (26,802 rows × 5 metals) — substitute for `metalcytotox_5metal_multitask.csv` (the latter is **NOT on disk**).

## Outputs

* `top20_candidates.csv` (20 rows × 9 cols)
* `ranking_summary.md` (this file)

## Multi-objective weights

| Axis | Weight | Direction |
|---|---:|---|
| `dmpnn_pic50` | 0.40 | higher better |
| `qed` | 0.20 | higher better |
| `sa` | 0.15 | lower better (negated) |
| `vina_kcal` | 0.15 | lower better (negated) |
| `astsr_score` | 0.10 | higher better |

## D-MPNN pIC50 spread (verify gate)

* min/max: 1.7069 log units (gate: ≥ 1.5)
* status: PASS

## Seed stability (5 seeds × top-20)

* overlap_vs_seed-0: min=18/20, mean=18.50/20
* status: PASS (gate: min ≥ 16)

## Metal distribution in top-20 (seed-0)

| Metal | Count |
|---|---:|
| Pt | 2 |
| Pd | 3 |
| Au | 1 |
| Ru | 3 |
| Ir | 11 |
| (other) | 0 |

## Honest-negative framing

* **D-MPNN pIC50** is a deterministic RDKit-fingerprint RandomForestRegressor surrogate, NOT the production chemprop D-MPNN. The Wf-53 Phase 2 real chemprop D-MPNN pIC50 bench is the binding truth.
* **Vina kcal/mol** is a deterministic RDKit descriptor proxy (MolLogP + TPSA + heavy-atom penalty) — NOT real AutoDock Vina. Treat as rank-only; absolute kcal/mol numbers carry no physical meaning.
* **SA** is BertzCT/1000 normalised — a coarse SA-like complexity proxy. Production SA score (Ertl & Schuffenhauer 2009) is not used here for dependency footprint.
* `metalcytotox_5metal_multitask.csv` (referenced by the task description) is **NOT on disk** at the time of Wf-58 P1; `metalcytotox_rdkit_features.csv` is used in its place.
* **No wet-lab synthesis** in Wf-58 P1 — this is an in-silico ranking dry-run only.
* **No paper claims** per TODO-84 paper rule; ranking output goes into TODO-97 R20-P6 plan, not paper §5.

## Cross-references

* `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan)
* `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule)
* `TODO/completed/53_*.md` etc. (Wf-53 Wf-58 lineage — Wf-58 P1 ship target)
