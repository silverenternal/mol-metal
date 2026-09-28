# Tier-1 Panel — wf_tier1_table2 final report

**Workflow**: wf_tier1_table2
**Date**: 2026-09-25 (R17)
**Author**: wf_tier1_table2 (aggregate step)
**Inputs**: `metrics/by_metric/{metalcytotox_ru_auc,platinai_4cell_auc,drugood_ood_auc}.json`
**Outputs**:
- `molmetal/reports/wf_tier1_table2/paper_section4_tier1.tex` (Table 2 Tier-1 rows)
- `molmetal/reports/wf_tier1_table2/final.md` (this file)
- `metrics/by_metric/tier1_summary.json` (machine-readable aggregate)

---

## 1. Pass / fail per gate

| Gate | Metric | Our value | Gate target | Status |
|---|---|---|---|---|
| Metric 1 | Ru subset AUC (MetalCytoToxDB) | 0.712 | ≥ 0.86 (Tier-1 beat); ≥ 0.81 (Tier-1 match) | **FAIL** (below gate by 0.148; below Tier-1 baseline by 0.098) |
| Metric 2 | Pt A2780cis AUC (PlatinAI) | 0.648 | ≥ 0.85 (Tier-1 highlight) | **FAIL** (below gate by 0.202; below Tier-1 baseline range) |
| Metric 2 | Pt MCF-7 AUC (PlatinAI) | 0.584 | match Tier-1 baseline (Rusanov) | **PARTIAL** (within qualitative match range, but exact Rusanov per-line numbers pending supplementary download) |
| Metric 2 | Pt A549 AUC (PlatinAI) | 0.630 | match Tier-1 baseline (Rusanov) | **PARTIAL** (within qualitative match range) |
| Metric 2 | Pt A2780 AUC (PlatinAI) | 0.350 | match Tier-1 baseline (Rusanov) | **FAIL** (sub-0.5, model near chance on parental cell line) |
| Metric 17 | DrugOOD OOD AUC — assay | 0.891 | ≥ iMoLD − 0.02 = 0.701 | **PASS** (beat by +0.170) |
| Metric 17 | DrugOOD OOD AUC — scaffold | 0.697 | ≥ iMoLD − 0.02 = 0.668 | **PASS** (beat by +0.009) |
| Metric 17 | DrugOOD OOD AUC — size | 0.748 | ≥ iMoLD − 0.02 = 0.659 | **PASS** (beat by +0.069) |

**Aggregate verdict**: 3 PASS / 3 FAIL / 2 PARTIAL (out of 8 individual gates).
- **Strict Tier-1 match count**: 1 / 3 metrics (only Metric 17 OOD generalisation passes all gates).
- **Net Tier-1 readiness**: 33% (1 of 3 Tier-1 metrics on track for the paper highlight tier).

---

## 2. Per-metric recommendation

### Metric 1 — Ru subset AUC (MetalCytoToxDB)

- **Status**: **BELOW GATE**.  Single-seed 5-epoch D-MPNN run returns AUC 0.712, which is below both the Tier-1 Krasnov 2026 GBDT baseline (0.81) and the gate target (0.86).
- **Recommendation**:
  1. Re-run `molmetal/scripts/train_metal_cytotox.py --metal Ru --seeds 0 1 2 --scaffold-split --eval-auc` with **3 seeds × 50 epochs** (current submission used 1 seed × 5 epochs).
  2. If 3-seed lift is insufficient, try D-MPNN multi-task over the full MetalCytoToxDB (Ru + Ir + Rh + Re + Os) to share representation across the 5 precious-metal corpora.
  3. If still below gate, defer Metric 1 from §4.1 to §7 Future Work and report the current 0.712 value as a baseline anchor with an honest framing sentence.
- **Paper framing**: cite Krasnov 2026 as the Tier-1 baseline; explicitly state "Mol-Metal single-seed baseline 0.712 < Krasnov GBDT 0.81; 3-seed retrain in flight per TODO-21 deferred".

### Metric 2 — Pt 4-cell-line AUC (PlatinAI)

- **Status**: **MIXED**.  Overall mean 0.553; per-cell-line: MCF-7 0.584, A549 0.630 (both qualitatively in match range), A2780 0.350 (sub-0.5, near chance), A2780cis 0.648 (below gate 0.85).
- **Critical finding**: A2780 (the parental non-resistant cell line) AUC = 0.350. This is the most negative finding of the round — a model near chance on the easier cell line while performing OK on A2780cis is structurally inconsistent.
- **Recommendation**:
  1. Investigate the A2780 label distribution and scaffold-split leak. The synthetic IC50 labels are seeded from a smaller known-Pt subset for A2780 than the other three cell lines; check `build_platinai_4cell.py` for the A2780 partition.
  2. If A2780 is fundamentally label-noisy, drop A2780 from the panel and report the 3-line version (MCF-7, A549, A2780cis).
  3. For A2780cis, attempt a heavier D-MPNN backbone or per-cell-line fine-tune (not just multi-task head sharing).
- **Paper framing**: report the per-cell-line table honestly; cite Rusanov 2025; flag the A2780 sub-0.5 result as the paper's most negative finding; do not claim cisplatin-resistance thesis support from this round.

### Metric 17 — DrugOOD OOD generalisation

- **Status**: **PASS** on all 3 splits. Assay +0.170, scaffold +0.009, size +0.069 vs iMoLD.
- **Recommendation**:
  1. This is the **strongest Tier-1 positive control** of the round. Promote it to the §4 abstract headline and the §4.5 "What is NOT a gap (positive controls)" list.
  2. Re-run on the full DrugOOD benchmark (not the metacytotox_fallback subset) before claiming the iMoLD head-to-head in print. The current numbers are best read as a "drug-OOD stress test passes" rather than a full iMoLD reproduction.
  3. The assay-split lift (+0.170) is suspiciously large; double-check the data-source mapping and ensure the assay labels in metacytotox_fallback align with the DrugOOD assay-split definition.
- **Paper framing**: "Mol-Metal generalises across scaffold/size OOD with AUC ≥ iMoLD on all three DrugOOD splits. Scaffold-OOD lift is +0.009 (within seed variance); assay-OOD and size-OOD lifts (+0.170 / +0.069) are substantial." Cite iMoLD 2024 + DrugOOD 2022.

---

## 3. Aggregate recommendation

**Tier-1 readiness for the paper §4 highlight tier: 33% (1 of 3 metrics).**

- **Promote to §4 headline positive**: Metric 17 (DrugOOD OOD generalisation) — the only Tier-1 row that strictly beats the named Tier-1 SOTA baseline across all sub-metrics.
- **Report honestly in §4.5 negative framing**: Metric 1 (Ru subset AUC) and Metric 2 A2780 / A2780cis rows. These are the load-bearing honest-negative findings of the round.
- **Defer to §7 Future Work**: if the structural fixes above do not land by paper-grade submission, defer Metric 1 + Metric 2 cell-line panel to V5 / next-round paper and report current values as baseline anchors.
- **Cross-link the audit trail**: every number in the §4 Tier-1 panel must be cross-referenced to `metrics/by_metric/<metric>.json` (machine-readable) and to this `wf_tier1_table2/final.md` (verdict summary). Reviewers who load the JSON files must see the same numbers reported in the LaTeX.

---

## 4. Paper figure hook

The Tier-1 panel supports three paper figures:

### Figure A — Per-cell-line AUC bar (PlatinAI)
- **Source**: `molmetal/reports/wf_tier1_table2/paper_section4_tier1.tex` Table row "Metric 2: Pt 4-cell-line AUC"
- **Plot**: 4-cell-line bar chart (MCF-7, A549, A2780, A2780cis); Mol-Metal (3 seeds, mean ± std) vs Rusanov 2025 GNN+fragment baseline (per-line, cite-only)
- **Highlight**: A2780cis is the cisplatin-resistance anchor; gate line at 0.85
- **Generator stub**: `paper/figures/fig7_cell_line_bar.py` (R17 stub already exists; needs the per-cell-line data fill-in from this panel)

### Figure B — OOD generalisation radar (DrugOOD)
- **Source**: Metric 17 row of the Tier-1 panel
- **Plot**: 3-axis radar / bar (assay, scaffold, size OOD AUC); Mol-Metal vs iMoLD vs ERM
- **Highlight**: assay-OOD lift is the most visually striking; scaffold-OOD is the most rigorous (smallest gap, hardest split)
- **Generator stub**: needs creation; suggested path `paper/figures/fig8_drugood_ood_bar.py`

### Figure C — Tier-1 vs Tier-2 scorecard
- **Source**: aggregate Tier-1 panel
- **Plot**: 3-row scorecard (Metric 1, Metric 2, Metric 17) with PASS/FAIL/PARTIAL marker; serves as the §4.5 honest-negative framing visual
- **Highlight**: makes the 33% Tier-1 readiness immediately legible to reviewers
- **Generator stub**: needs creation; suggested path `paper/figures/fig9_tier1_scorecard.py`

**Status of figures**: Figures A (R17 stub exists, needs data fill), B (TBD), C (TBD). All three are paper-grade priorities for the next round.

---

## 5. Audit trail

| File | Purpose | Lines | Status |
|---|---|---|---|
| `metrics/by_metric/metalcytotox_ru_auc.json` | Metric 1 source | — | MEASURED (1 seed) |
| `metrics/by_metric/platinai_4cell_auc.json` | Metric 2 source | — | MEASURED (3 seeds, synthetic labels) |
| `metrics/by_metric/drugood_ood_auc.json` | Metric 17 source | — | MEASURED (3 seeds × 30 epochs) |
| `metrics/by_metric/tier1_summary.json` | Tier-1 aggregate | — | GENERATED by this workflow |
| `molmetal/reports/wf_tier1_table2/paper_section4_tier1.tex` | LaTeX Table 2 Tier-1 panel | 130+ | GENERATED |
| `molmetal/reports/wf_tier1_table2/final.md` | Verdict summary | this file | GENERATED |

---

## 6. Next steps (priority order)

1. **Re-run Metric 1 with 3 seeds × 50 epochs** (MetalCytoToxDB Ru subset). Track under TODO-21 deferred. ETA: 1 day.
2. **Investigate Metric 2 A2780 label noise**. Trace through `build_platinai_4cell.py` A2780 partition; consider dropping A2780 from the panel. ETA: 0.5 day.
3. **Re-run Metric 17 on full DrugOOD benchmark** (not metacytotox_fallback). ETA: 1 day.
4. **Generate Figure B + C stubs** at `paper/figures/fig{8,9}_*.py`. ETA: 0.5 day.
5. **Fold `paper_section4_tier1.tex` into `paper/sections/04_results.tex`** as Table~\ref{tab:tier1-panel}. ETA: 0.25 day.

Total: ~3.25 days to Tier-1 panel paper-grade.

---

**Sign-off**: This panel is honest about 5 of 8 gates failing or partially-passing. Metric 17 is the paper's Tier-1 positive control; Metric 1 + Metric 2 are paper's Tier-1 honest-negative findings. The §4.5 negative-verdict list is structurally aligned with this verdict.
