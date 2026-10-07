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
| **m1** Ru subset binary AUC | Krasnov 2026 (MetalCytoToxDB v2) | **DATASET + METRIC + SPLIT ALIGN — cohort honest-negative** (2026-10-04: paper criteria adopted — filters dropped+recorded; DOI-split 10-fold CV replicated (DOI-disjoint); release v2 md5-verified; MEASURED D-MPNN 0.7717 ± 0.0380 vs paper LightGBM 0.810 ± 0.024; paper cohort 11,294 / 4,399 / 23 not replicated — ours 15,799 / 4,793 / 591; see COMPARISON_BASELINES.md §3 + §6) | paper §3 main results (cross-cohort caveat; see §6) |
| **m2** R² per-cell-line | Rusanov 2025 (PlatinAI) | **DATASET + METRIC + SPLIT APPROX ALIGN** | paper §4 honest limitations |
| **m3** Vina Dock on CrossDocked100 | — none | **NO FORMAL PAPER COMPARISON — Tier-2 supplementary** | paper §4 supplementary only (no comparison table) |
| **m7** Diversity (Morgan ECFP4 Tanimoto) | — none | **NO FORMAL PAPER COMPARISON — Tier-2 supplementary** | paper §4 supplementary only (no comparison table) |
| **m15** Docking RMSD | — none | **NO FORMAL PAPER COMPARISON — Tier-2 supplementary** | paper §4 supplementary only (no comparison table) |
| **m17** OOD generalization | iMoLD / DrugOOD 2022 | **NOT ALIGN — fallback cohort** (5,207-row `metacytotox_fallback`, not the official DrugOOD IC50 splits; official CSVs on disk at `.data/molmetal/drugood/ic50/csv/`, regen pending wf-R34-B2) | NOT a comparison table — absolute measured value + §6.6 caveat |
| **m21** NCI-60 9-cell AUC | Krasnov 2026 (MetalCytoToxDB) | **DATASET ALIGN** (split differs) | paper §3 main results |

---

## §2 Alignment level legend

- **FULL ALIGN**: same dataset + same split protocol + same metric — directly comparable numerically
- **DATASET + METRIC ALIGN** (split differs): numerical values not directly comparable but per-seed/per-split statistics are; document split difference in §6
- **DATASET + METRIC + SPLIT APPROX ALIGN**: small split difference (e.g. per-cell-line vs cohort-split) — per-cell breakdown matches
- **DATASET ALIGN** (split + metric differ): values comparable as sample-size-scaled estimates; document both in §6
- **NOT ALIGN**: do not include in §4 comparison tables
- **NOT ALIGN — fallback cohort**: only the metric definition matches; the dataset is a locally-derived stand-in for the paper's dataset (m17: `metacytotox_fallback`, 5,207 MetalCytoToxDB-derived rows, in place of the official DrugOOD IC50 splits). Numeric gates on it are internal sanity checks, not head-to-head comparisons; report standalone + §6 limitation, and regenerate on the official cohort before any comparison table (m17: pending wf-R34-B2)
- **NO FORMAL PAPER COMPARISON — Tier-2 supplementary**: no external baseline exists; the value is reported standalone in paper §4 supplementary and never placed in a comparison table
- **DATASET + METRIC + SPLIT ALIGN — cohort honest-negative**: dataset + metric + split protocol aligned, but the cohort scope differs from the paper's — the numeric comparison is a cross-cohort estimate (see `COMPARISON_BASELINES.md` §6)

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
| m1 | **Cohort honest-negative** — paper criteria adopted 2026-10-04 (filters dropped + recorded; DOI-split 10-fold CV replicated; release v2 md5-verified) but the paper's 23-cell-line cohort is not reproduced (ours 15,799 rows / 591 cell lines vs paper 11,294 / 23; D-MPNN 0.7717 ± 0.0380 vs paper LightGBM 0.810 ± 0.024). The shipped 0.7613 record (scaffold split, unrecorded filter) remains untraceable but is superseded as the m1 comparison number by the additive new entry. Full state: `COMPARISON_BASELINES.md` §6 | §6.1 |
| m2 | Rusanov uses per-cell-line; we use cohort-split + per-cell eval. Per-cell breakdown matches | §6.2 |
| m3 | **No external paper baseline exists** — reported as an absolute median kcal/mol on our 5-pocket CrossDocked100 subset. There is nothing to compare against, so no comparison claim is made | §6.3 |
| m7 | **No external paper baseline exists** — Morgan ECFP4 Tanimoto reported as an absolute diversity value | §6.4 |
| m15 | **No external paper baseline exists** — median Å reported as an absolute value on our 5-pocket CrossDocked100 subset | §6.5 |
| m17 | **Fallback cohort — NOT a DrugOOD comparison.** Our shipped m17 ran on `metacytotox_fallback` (MetalCytoToxDB-derived, 5,207 rows) with DrugOOD-*style* assay/scaffold/size splits, not the official DrugOOD IC50 splits. The metric (binary AUC) matches; the cohort does not, so the 3/3 gate pass is an internal sanity check only. Official DrugOOD CSVs are on disk at `.data/molmetal/drugood/ic50/csv/`; regeneration is **pending wf-R34-B2**. Full state: `COMPARISON_BASELINES.md` §6.6 | §6.6 |
| m21 | Krasnov uses single-cell-line; we use 9-cell-line panel from NCI-60 overlap | §6.7 |
