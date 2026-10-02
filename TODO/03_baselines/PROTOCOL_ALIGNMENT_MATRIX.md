# PROTOCOL_ALIGNMENT_MATRIX — Quick Reference Table

**Date**: 2026-10-01 (Wf-431 docs-only refactor)
**Status**: Quick reference — see `TODO/03_baselines/COMPARISON_BASELINES.md` for full per-metric protocol alignment matrix + per-paper rationale
**Purpose**: Single-glance summary of which Tier-1 metric is comparable to which baseline paper

---

## §0 Principle (verbatim from COMPARISON_BASELINES.md §0)

> **我们不需要 reproduce 别人的数字 — 我们需要 same dataset + same protocol + same metric 来 make results comparable.**

---

## §1 Tier-1 metric × baseline × alignment level

| Tier-1 metric | Selected baseline | Alignment level | Where reported in paper |
|---|---|---|---|
| **m1** Ru subset binary AUC | Krasnov 2026 (MetalCytoToxDB) | **DATASET + METRIC ALIGN** (split differs: paper 5-fold CV; us scaffold-split 70/10/20) | paper §3 main results |
| **m2** R² per-cell-line | Rusanov 2025 (PlatinAI) | **DATASET + METRIC + SPLIT APPROX ALIGN** | paper §4 honest limitations |
| **m3** Vina Dock on CrossDocked100 | MolCRAFT / TransDiffSBDD (Tier-2 only) | **DATASET + METRIC ALIGN** (subset differs: 5 vs 100 pockets) | paper §4 main results |
| **m7** Diversity (Morgan ECFP4 Tanimoto) | MolChord / TransDiffSBDD | **DATASET + METRIC + SPLIT ALIGN** | paper §4 main results |
| **m15** Docking RMSD | DiffDock / EquiBind (sanity check only) | **DATASET + METRIC ALIGN** (subset differs) | paper §4 main results |
| **m17** OOD generalization | iMoLD / DrugOOD 2022 | **DATASET + METRIC + SPLIT FULL ALIGN** | paper §3 main results |
| **m21** NCI-60 9-cell AUC | Krasnov 2026 (MetalCytoToxDB) | **DATASET ALIGN** (split differs) | paper §3 main results |

---

## §2 Alignment level legend

- **FULL ALIGN**: same dataset + same split protocol + same metric — directly comparable numerically
- **DATASET + METRIC ALIGN** (split differs): numerical values not directly comparable but per-seed/per-split statistics are; document split difference in §6
- **DATASET + METRIC + SPLIT APPROX ALIGN**: small split difference (e.g. per-cell-line vs cohort-split) — per-cell breakdown matches
- **DATASET ALIGN** (split + metric differ): values comparable as sample-size-scaled estimates; document both in §6
- **NOT ALIGN**: do not include in §4 comparison tables

---

## §3 Cross-references

- `TODO/03_baselines/COMPARISON_BASELINES.md` — full per-metric protocol alignment matrix + selected baselines detail + out-of-scope baselines
- `TODO/02_metrics/METRICS_TARGETS.md` — per-metric target values + verdict gates
- `TODO/decisions.md` D-Wf-431 — this refactor decision
- `TODO/archive/reproduce_baselines.md.archived_2026_10_01` — previous "reproduce baseline" approach (DEPRECATED per new principle)

---

## §4 Summary table for paper §6 honest limitations

| Metric | Limitation type | §6 paragraph |
|---|---|---|
| m1 | Krasnov uses 5-fold CV; we use scaffold-split. Per-seed statistics comparable; headline mean +0.05 margin documented | §6.1 |
| m2 | Rusanov uses per-cell-line; we use cohort-split + per-cell eval. Per-cell breakdown matches | §6.2 |
| m3 | CrossDocked100 is large; we benchmark on 5-pocket subset for budget reasons. Sample-size-scaled estimates documented | §6.3 |
| m7 | Same dataset + split + metric as TransDiffSBDD | §6.4 (no limitation) |
| m15 | DiffDock/EquiBind use full PDBbind; we use 5-pocket CrossDocked subset | §6.5 |
| m17 | DrugOOD full alignment | §6.6 (no limitation) |
| m21 | Krasnov uses single-cell-line; we use 9-cell-line panel from NCI-60 overlap | §6.7 |
