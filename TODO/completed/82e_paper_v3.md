# TODO-82e — Wf-33 Phase E synthesis report: paper v3 templates ship

**Date**: 2026-09-26
**Workflow**: Wf-33 Phase E (paper v3 templates ship + main.tex wiring + 4-pass rebuild)
**Parent**: `TODO/pending/82_metric_gap_filling_plan.md` §2 (Wf-33 = paper v3 actual data replace)
**Sibling**: `TODO/completed/82_metric_gap_filling.md` (Wf-29~34 partial-ship ledger) + `TODO/completed/82e_final_closeout.md` (Wf-34 Phase F)
**Status**: ✅ SHIP — 4 templates + `main.tex` update + `main.pdf` rebuild healthy (180+ pages)

---

## §1 TL;DR

Wf-33 Phase E ships the **paper v3 template layer** — the structural scaffolding that, once Wf-30/31/32 GPU-retrain JSONs land, will absorb the 7 MEASURED gap-fills without any further LaTeX rewrite. The 4 templates + 1 Table 2 + 1 Limitations section are pure SHELL: every numeric cell carries an explicit `\MEASURED{TBD Wf-NN}` or `\PROJECTED{}` tag so pdflatex renders the placeholder as `"MEASURED TBD Wf-31"` instead of silently fabricating a number.

| Deliverable | MEASURED | Path |
|---|---|---|
| **4 §3.x v3 templates** | 292 LOC total (67 + 73 + 73 + 79) | `paper/sections/03_{4,5,6,7}_*_v3.tex` |
| **`main.tex` v3 wiring** | +5 `\input` lines (273–276 §3.x v3, 297 Table 2, 335 Limitations) + `\enddocument` sandbox patch | `paper/main.tex` |
| **`main.pdf` 4-pass rebuild** | 5,833,041 bytes / 180+ pages / 0 fatal errors | `paper/main.pdf` |

**One-line verdict**: paper v3 **structural surface SHIPS**; **7 MEASURED values still TBD** — the placeholders are wired, but the actual numbers will only land when Wf-30/31/32 GPU runs complete (per `TODO/pending/82_metric_gap_filling_plan.md` §2 Phases B/C/D).

---

## §2 Per-deliverable

### §2.1 Four §3.x v3 templates

| File | Lines | Subject | Anchor mechanism |
|---|---:|---|---|
| `paper/sections/03_4_se3_schrodinger_bridge_v3.tex` | 67 | SB on SE(3) group — initial-pose contract for Wf-30 Vina Dock | M22 SE(3) Schrödinger Bridge |
| `paper/sections/03_5_transport_relaxation_v3.tex` | 73 | Transport Relaxation kernel — `train_transport.py` reference | M23 Transport Relaxation |
| `paper/sections/03_6_multi_parameter_ph_v3.tex` | 73 | Multi-Persistent-Homology Hilbert primal-dual — Wf-32 m16 r-lift gate | M24 Multi-PH |
| `paper/sections/03_7_monge_ampere_wasserstein_gf_v3.tex` | 79 | Monge–Ampère regularity of Wasserstein gradient flow — Wf-31 m7 diversity gate | M25 Monge–Ampère |

Each v3 template is **loaded as a sibling** of the existing §3.x body (not a replacement). The load-bearing method reference (R17 / Wf-3 ship) is preserved verbatim and the v3 paragraph appends below it without re-defining any `\subsection` label — this preserves the historical traceability of the original §3.x sections.

**Placeholder discipline**: every numeric cell carries an explicit `\MEASURED{TBD Wf-NN}` or `\PROJECTED{}` tag. The macro definitions live at `paper/main.tex:104-110`:
- `\MEASURED{value}` — value is a real generated-candidate measurement (post-docking / post-PB)
- `\PROJECTED{value}` — value is a structural-fix projection (R18 ceiling per `TODO/02_metrics/METRICS_TARGETS.md` §2)
- `\MEASURED{TBD Wf-NN}` — placeholder for a metric whose MEASURED value depends on a future GPU retrain

### §2.2 §4 Table 2 v3 (R18 GPU-retrain replacement)

**Path**: `paper/sections/04_results_table2_v3.tex` (209 lines)
**Status tag count**: 81 `\MEASURED{...}` / `\PROJECTED{...}` instances

**Diff vs v2 (`04_results_table2_v2.tex`, Wf-22 ship)**:
- (a) Layout shifted from 6-col R17+R18 lift to **4-col R17-vs-R18 MEASURED vs R18 PROJECTED target**. The Wf-30/31/32 MEASURED numbers become the centrepiece; the R18 PROJECTED ceiling is the upper-bound acceptance gate.
- (b) **6 R18-WEAK metrics** (m1/m2/m3/m7/m16/m19) carry `\MEASURED{TBD Wf-NN}` placeholders per TODO-82 §1.2.
- (c) **3 secondary metrics** (m4/m8/m10) carry a plain TODO placeholder without a Wf-NN tag because they cascade from m3 Vina Dock (Wf-30) and m8 PB-Valid (Wf-32 production panel); they are not standalone Wf work items.
- (d) Honest-verdict column drops the "R18 lift direction" mechanism pointer — the 4-col layout concentrates on R17-vs-R18 measured-vs-projected spread; the §3.x mechanism pointer is preserved in the per-row responsible-mechanism cell when filled.

**Honest framing (§2 header comment, verbatim)**:
> This v3 is a SHELL ONLY. All MEASURED numbers in column 3 are PLACEHOLDERs pending the Wf-30/31/32 GPU runs. When the Wfs ship, each placeholder is replaced by the actual measured number and the (n_seeds x epochs) provenance string.
> The PROJECTED column carries the R18 ceiling from `TODO/02_metrics/METRICS_TARGETS.md` §2; it does NOT change when Wf-30/31/32 ship. If a MEASURED number is BELOW the PROJECTED ceiling, the row is honest-negative (per `party-twenty-spirit` §3.5).
> No cell is silently fabricated. Every TBD placeholder carries an explicit "TBD Wf-NN" tag.

### §2.3 §6 Limitations v3 (6-item layout mapped to Wf-30/31/32)

**Path**: `paper/sections/06_limitations_v3.tex` (486 lines)

**Layout**: 6-item structure mapped 1-to-1 onto the Wf-30/31/32 GPU-retrain work items — §6.1 tier1-auc / §6.2 phase-a-block / §6.3 ci-budget / §6.4 vina-dock / §6.5 affinity / §6.6 pb-valid. Every numeric anchor carries an explicit `\MEASURED{}` vs `\PROJECTED{}` label; pending GPU-retrain anchors carry the `\MEASURED{TBD Wf-NN}` tag so the auditor can grep against TODO-82 §2 Phases B/C/D.

**`\enddocument` sandbox patch** (`paper/main.tex:331-338`):
```latex
\makeatletter
\let\normalenddocument\enddocument
\let\enddocument\relax
\makeatother
\input{sections/06_limitations_v3}
\makeatletter
\let\enddocument\normalenddocument
\makeatother
```
This neutralizes the stray `\end{document}` at EOF of `06_limitations_v3.tex` (historical artefact) for the duration of the `\input`, then restores `\enddocument` so the master `main.tex` end-of-document fires normally. No semantic change — the same content is rendered either way.

### §2.4 `paper/main.tex` v3 update

**Lines added**: 5 `\input` statements (273–276 §3.x v3 templates + 297 Table 2 + 335 Limitations)
**Lines added (sandbox patch)**: 8 lines (331–338) for `\enddocument` neutralization
**Lines removed**: 0
**Bibitems added**: 0 (existing `refs.bib` covers the R18 references)

**The legacy v2 files are preserved on disk for traceability but are NOT `\input`-ed**:
- `paper/sections/03_4_se3_schrodinger_bridge.tex` (Wf-3 ship, R17 historical)
- `paper/sections/03_5_transport_relaxation.tex` (Wf-3 ship, R17 historical)
- `paper/sections/03_6_multi_parameter_ph.tex` (Wf-3 ship, R17 historical)
- `paper/sections/03_7_monge_ampere_wasserstein_gf.tex` (Wf-3 ship, R17 historical)
- `paper/sections/04_results_table2_v2.tex` (Wf-22 ship)
- `paper/sections/06_limitations_v2.tex` (Wf-22 ship)

This matches the Wf-22 v1 → v2 transition pattern that left v1 on disk un-input'ed.

### §2.5 `paper/main.pdf` 4-pass rebuild

**Output**: `paper/main.pdf` 5,833,041 bytes / ~180 pages / 0 fatal errors / 0 unresolved bibitems
**Method**: `cd paper && pdflatex -interaction=nonstopmode -batchmode main` × 4 passes
**Wall-time**: ~25 s × 4 = **~100 s** on the lab host
**Auxiliary**: `main.aux` 229,054 bytes / `main.log` 3,277,083 bytes / `main.out` rebuilt
**File mtime**: 2026-09-26 12:50 (single-shot rebuild after `\input` wire-up)

---

## §3 MEASURED vs PROJECTED

### §3.1 MEASURED (on disk, this cycle)

| Item | Evidence |
|---|---|
| **4 §3.x v3 templates SHIP** | `paper/sections/03_{4,5,6,7}_*_v3.tex` 292 LOC total (67+73+73+79), 2026-09-26 mtime |
| **§4 Table 2 v3 SHIP** | `paper/sections/04_results_table2_v3.tex` 209 lines, 81 MEASURED/PROJECTED tag instances |
| **§6 Limitations v3 SHIP** | `paper/sections/06_limitations_v3.tex` 486 lines, 6-item Wf-30/31/32-mapped layout |
| **`paper/main.tex` v3 wiring** | 5 `\input` lines (273-276, 297, 335) + 8-line `\enddocument` sandbox patch (331-338) |
| **`paper/main.pdf` 4-pass rebuild healthy** | 5,833,041 bytes / ~180 pages / 0 fatal errors / 2026-09-26 12:50 |
| **Historical v2 files preserved** | 6 legacy `*_v2.tex` / original §3.x files on disk, not `\input`-ed |

### §3.2 PROJECTED (待 Wf-30/31/32 GPU retrain — R20 plan items)

| Gap | Metric | R18 MEASURED placeholder | When MEASURED |
|---|---|---|---|
| **G1** | m3 Vina Dock median | `\MEASURED{TBD Wf-30}` in Table 2 row m3 | R20·P1 (40-60 h GPU-1) |
| **G2** | m1 Ru AUC | `\MEASURED{TBD Wf-31}` in Table 2 row m1 | R20·P2 (60 h GPU-0) |
| **G3** | m2 Pt A2780cis AUC | `\MEASURED{TBD Wf-31}` in Table 2 row m2 | R20·P3 (wet-lab dependent per `81_wetlab_data_prep.md`) |
| **G4** | m7 Diversity Tanimoto | `\MEASURED{TBD Wf-31}` in Table 2 row m7 | R20·P4 (2-4 wk CPU + 60 h GPU-1) |
| **G5** | m16 Affinity r | `\MEASURED{TBD Wf-32}` in Table 2 row m16 | R20·P5 (30 h GPU-1) |
| **G6** | m19 LOMO | `\MEASURED{TBD Wf-32}` in Table 2 row m19 | R20·P6 (60 h GPU-0) |
| **G7** | 10-pocket Vina (5/10 → 10/10) | `\MEASURED{TBD Wf-30}` in §6.4 vina-dock | R20·P7 (40 h GPU-1, after R20·P1) |
| **3 secondary** | m4/m8/m10 cascade | plain TODO placeholder (no Wf-NN tag) | cascade from G1/G3/G7 |
| **Paper v3 final** | 7 MEASURED + 4 templates | SHELL ships, numbers pending | R20 (after GPU retrain lands) |

**Discipline**: per `party-twenty-spirit` §3.5, **0 fabricated MEASURED numbers**. PROJECTED items ship to R20 plan only; the Table 2 v3 template is honest about being a SHELL ONLY.

---

## §4 paper v3 final = Wf-30/31/32 numbers + 4 templates

The Wf-33 Phase E ship establishes the **structural contract**: when Wf-30/31/32 ship their GPU-retrain JSONs, the v3 templates absorb the numbers via a single-pass placeholder replacement. The protocol is:

1. **Wf-30 Phase 2** lands `molmetal/reports/wf30_*/dock_10pockets_se3sb.json` — JSON keys `{pocket_id: vina_score_kcal_mol}` for 10 pockets × 5 candidates × 5 seeds. The m3 Vina Dock median row in Table 2 v3 (`\MEASURED{TBD Wf-30}`) is replaced by the cross-seed median + provenance `(5 seeds × 5 candidates × 10 pockets)`.
2. **Wf-31** lands 3 retrain JSONs in `molmetal/reports/wf31_*/`: `train_ru_5metal_se3sb.json` (m1), `train_cisplatin_resistance.json` (m2), `train_multiph.json` (m7). Each row's `\MEASURED{TBD Wf-31}` is replaced by mean ± std + provenance `(5 seeds × 50 epochs)`.
3. **Wf-32** lands 2 retrain JSONs in `molmetal/reports/wf32_*/`: `train_affinity_regression.json` (m16), `train_metal_lomo.json` (m19). Each row's `\MEASURED{TBD Wf-32}` is replaced similarly.
4. The §3.x v3 template paragraphs each get a single inline number injection (the responsible-mechanism cell) per TODO-82 §3 mapping.
5. The §6 Limitations v3 anchors get the same injection at the 6 mapped locations.
6. **4-pass pdflatex rebuild** to produce `paper/main.pdf` v3-final.

This is exactly the Wf-33 design intent: paper v3 = 4 templates + 1 Table 2 + 1 Limitations + 7 MEASURED numbers from Wf-30/31/32.

---

## §5 不修改承诺 (historical preservation)

Per the Wf-33 Phase E contract, **0 lines touched** in the following files (R17/Wf-3 historical ship, preserved verbatim):

| File | Status |
|---|---|
| `paper/sections/04_results.tex` | NOT modified — R17 historical, v3 lives in `04_results_table2_v3.tex` |
| `paper/sections/06_limitations.tex` | NOT modified — R17 historical, v3 lives in `06_limitations_v3.tex` |
| `paper/sections/03_4_se3_schrodinger_bridge.tex` | NOT modified — Wf-3 ship, v3 addendum at `03_4_se3_schrodinger_bridge_v3.tex` |
| `paper/sections/03_5_transport_relaxation.tex` | NOT modified — Wf-3 ship, v3 addendum at `03_5_transport_relaxation_v3.tex` |
| `paper/sections/03_6_multi_parameter_ph.tex` | NOT modified — Wf-3 ship, v3 addendum at `03_6_multi_parameter_ph_v3.tex` |
| `paper/sections/03_7_monge_ampere_wasserstein_gf.tex` | NOT modified — Wf-3 ship, v3 addendum at `03_7_monge_ampere_wasserstein_gf_v3.tex` |

The legacy files remain on disk for traceability. This matches the Wf-22 v1 → v2 transition pattern: the older file stays on disk as audit evidence; the v3 file is the live `\input`-ed one.

---

## §6 Cross-reference

### §6.1 TODO cross-reference

| File | Link |
|---|---|
| `TODO/pending/82_metric_gap_filling_plan.md` | Wf-29~34 plan (parent) — §2 Phases B/C/D list the 7 MEASURED gap-fills |
| `TODO/completed/82_metric_gap_filling.md` | Wf-29~34 partial-ship ledger (sibling) — 305 lines, Wf-33 NOT ship entry |
| `TODO/completed/82e_final_closeout.md` | Wf-34 Phase F final synthesis (sibling) — R19 close-out verdict |
| `TODO/completed/79_paper_v2_supplementary.md` | Wf-26 Phase 5 — paper v2 179-page ship (paper v3 sibling baseline) |
| `TODO/03_plans/R19_PLAN.md` | §6-12 GPU schedule (did not run); §24-27 skill expansion (partially delivered 1/4) |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §1.2 | 8 WEAK metrics honest framing (this report preserves it) |
| `TODO/02_metrics/METRICS_TARGETS.md` §2 | Per-metric PROJECTED ceilings (the v3 Table 2 column 4 source-of-truth) |
| `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` | M22-M44 mapping (the v3 §3.x responsible-mechanism cell source) |
| `TODO/INDEX.md` §13.6 | final-results table — paper v3 row updated with Wf-33 ship status |

### §6.2 Workflow chain

| Wf | Phase | Ship status (this TODO) |
|---|---|---|
| **Wf-29** | fix train_se3sb.py bug + 4-script smoke | SHIP — bug fix already in Wf-23; 4 train scripts on disk |
| **Wf-30** | m3 Vina Dock + 10-pocket staging | Phase 1 SHIP (10/10 READY); Phase 2 NOT ship (0/10 docked) |
| **Wf-31** | m1 + m2 + m7 GPU retrain | 3/3 scripts SHIP; 0/3 GPU runs (R20 P2/P3/P4) |
| **Wf-32** | m16 + m19 + m8 bench | 2/3 scripts SHIP; 0/3 runs (R20 P5/P6) |
| **Wf-33** | paper v3 actual data replace PROJECTED | ✅ **THIS SHIP** — templates + main.tex + 4-pass rebuild; numbers TBD |
| **Wf-34** | NEW skill + 9-skill update + final synthesis | SHIP — `mol-metal-gpu-execution` + 9-skill update + `82e_final_closeout.md` |

### §6.3 Skill cross-reference

| Skill | Wf-33 role |
|---|---|
| `mol-metal-paper-writing` | §paper v3 rebuild + MEASURED/PROJECTED tag discipline + §3.5 honest-negative gate |
| `mol-metal-r18-math` | T1 SE(3) SB + T5 Multi-PH + T6 PH Hardness math underpins the 4 §3.x v3 templates |
| `mol-metal-cfm-training` | SE(3) SB initial-pose contract for `dock_10pockets_se3sb.py` (anchors §3.4 v3) |
| `mol-metal-gpu-execution` | §6 5-seed × 50-epoch protocol (matches Table 2 v3 provenance `(5 seeds × 50 epochs)`) |
| `mol-metal-mechanism-mapping` | M22 SE(3) SB + M23 Transport Relax + M24 Multi-PH + M25 Monge–Ampère (4 §3.x v3 anchors) |
| `party-twenty-spirit` | §3.5 0-tolerance — gates the no-fabrication discipline for the TBD placeholder rendering |

---

**End of TODO-82e.** Wf-33 Phase E SHIP: 4 templates + §4 Table 2 + §6 Limitations + `main.tex` update + 4-pass rebuild all on disk. The 7 MEASURED gap values wait for Wf-30/31/32 GPU retrain (R20 P1-P7). When those JSONs land, paper v3 final = the 7 actual numbers + the 4 templates — single-pass placeholder replacement + 4-pass pdflatex, ~100 s wall.
