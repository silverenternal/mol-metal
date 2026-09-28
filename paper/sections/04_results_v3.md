# Paper §4 — R18 Results Narrative (v3 TEMPLATE)

**Date**: 2026-09-26
**Round**: R18 (post-R17 close-out, post-R18 5-seed Pt retrain)
**Owner**: Hugo Lee + ultracode agents
**Companion**: `paper/sections/04_results_table2_v3.tex` (Table 2 v3 template)
**Sibling**: `paper/sections/04_results_table2.tex` (v1, R17 1-seed) +
             `paper/sections/04_results_table2_v2.tex` (v2, R17 5-seed Pt)

---

## §1. Purpose

The v3 table is a **SHELL**: every R18 MEASURED column 3 cell is a
`\MEASURED{TBD Wf-NN}` placeholder. The table exists so:

1. Paper §4 has the R18 layout ready when Wf-30/31/32 ship their GPU-retrain JSONs
2. Auditors can grep `MEASURED{TBD` and see exactly which metrics are pending
3. The MEASURED-vs-PROJECTED split is structurally enforced (no silent fabrication)

## §2. Layout (4-col vs 6-col)

| Column | v2 (Wf-22) | v3 (Wf-33 template) |
|---|---|---|
| 1 | Metric | Metric (unchanged) |
| 2 | R17 (MEASURED) | R17 baseline (MEASURED) |
| 3 | Tier-1 baseline | **R18 MEASURED (MEASURED{TBD})** ← new |
| 4 | Direction | **R18 target (PROJECTED)** ← from METRICS_TARGETS.md §2 |
| 5 | R18 lift direction (PROJECTED) | dropped (mechanism ptr moves to caption) |
| 6 | Honest verdict | Honest verdict (template) |

The v3 layout makes the R18 MEASURED column the centrepiece; the v2 "R18 lift
direction" mechanism pointer is preserved in the caption text and in the
per-row responsible-mechanism cell when filled.

## §3. Placeholder taxonomy (strict)

| Tag | Count | Means |
|---|---|---|
| `\MEASURED{TBD Wf-31}` | 6 (m1, m2×4, m7, m16) + m19, m3 = 9 | TODO-82 §1.2 R18-WEAK; Wf-31 GPU retrain will MEASURE |
| `\MEASURED{TBD}` | 3 (m4, m8, m10) | Cascade from m3 Wf-30 / m8 Wf-32; no standalone Wf |
| `same as R17 (carry-over)` | 11 (m11/m12/m13/m14/m17×3/m21/m18/Deflexpressor/MoleculeNet) | STRONG metrics; no R18 GPU retrain scheduled |

The `\MEASURED{TBD` syntax is **deliberately broken LaTeX-rendering-safe**:
when the template ships, pdflatex renders "MEASURED TBD Wf-31" as visible
text in the column, so the placeholder is impossible to miss visually.

## §4. Per-Wf attribution (TODO-82 §2)

| Workflow | Phase | Metrics | Wall-time | Output dir |
|---|---|---|---|---|
| **Wf-30** | B (Vina Dock GPU retrain) | m3, m4 (cascade), m10 (cascade) | ~2 days | `molmetal/reports/wf30_*` |
| **Wf-31** | C (m1 + m2 + m7 GPU retrain) | m1, m2×4, m7, m19 (LOMO) | ~2 days | `molmetal/reports/wf31_*` |
| **Wf-32** | D (m16 + m19 + m8 bench) | m16, m8, m19 (LOMO alt) | ~2 days | `molmetal/reports/wf32_*` |

Note: the task spec wrote `m19` under `TBD Wf-31` but the gap-filling plan §2
Phase C/D assigns m19 to Wf-31 LOMO retrain (Phase C) with Wf-32 (Phase D)
as the affinity/PB-Valid bench; we follow the **task spec literally** and tag
m19 as `TBD Wf-31` while flagging the Wf-32 fallback here. m3 is tagged
`TBD Wf-31` per task spec; TODO-82 §2 Phase B assigns m3 to Wf-30.

## §5. Honest framing (party-twenty-spirit §3.5)

- Every TBD placeholder is **explicitly visible** in the rendered PDF.
- When Wf-30/31/32 ship, the row is rewritten with MEASURED value +
  (n_seeds × epochs) provenance string + measured-vs-projected verdict.
- If MEASURED < PROJECTED → row is **honest-negative** (no "we beat SOTA").
- If MEASURED ≥ PROJECTED → row is **honest-positive** (cite the SOTA baseline
  and the lift).
- No row is silently filled with a PROJECTED value masquerading as MEASURED.

## §6. When this template ships

| Trigger | Action |
|---|---|
| Wf-30 ships | Replace m3/m4/m10 placeholders with Vina Dock GPU-retrain JSON values |
| Wf-31 ships | Replace m1/m2/m7/m16 placeholders with 5-seed × 50-epoch JSON values |
| Wf-32 ships | Replace m19 placeholder with LOMO JSON values; m8 with PB-Valid prod JSON |
| All 3 ship | Rename `tab:main_v3_template` → `tab:main_v3_r18` and remove "TEMPLATE" wording |

Until then, the file `paper/sections/04_results_table2_v3.tex` carries the
`(TEMPLATE)` flag in its title and caption, so the auditor / reviewer can
immediately see that the v3 numbers are placeholders.

## §7. Cross-reference

- `TODO/pending/82_metric_gap_filling_plan.md` §2 Phases B/C/D — the Wf-30/31/32
  execution plan that fills the placeholders.
- `TODO/02_metrics/METRICS_TARGETS.md` §2 — the per-metric PROJECTED ceiling
  that the MEASURED column must meet or exceed.
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §1.2 — the 8 WEAK metrics + R18 phases.
- `paper/sections/04_results_table2.tex` — v1 (R17 1-seed) preserved.
- `paper/sections/04_results_table2_v2.tex` — v2 (R17 5-seed Pt retrain) preserved.
- `paper/main.tex:104-110` — `\MEASURED{}` / `\PROJECTED{}` macro definitions.
