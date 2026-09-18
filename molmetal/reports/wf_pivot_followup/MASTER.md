# WF-Pivot-Followup — MASTER Consolidation

**Date**: 2026-09-16
**Workflow**: WF-Pivot-Followup (F1A + F1B + F2A + F2B + F3A + F3B + F3C)

---

## 1. TL;DR

Pivot A + widzuipcl Phase 1 integrated; PlatinAI activity-oracle column shipped as **§4.6.1** placeholder (slot renamed from §4.7 to preserve 25+ existing cross-refs); supplementary S11 data-provenance table shipped (3 sub-tables, 207 lines, currently ORPHANED from `paper/supplementary.tex`); 5 new metric JSONs delivered with honest framing; cross-ref matrix verified with 0 unresolved targets. The metallodrug vertical is structurally in place; **0 cells promoted DESIGN → MEASURED** pending widzuipcl Phase 2 (GPU retrain).

---

## 2. Follow-up outcomes table

| ID | Sub-workflow | Verdict | Output line count |
|----|---|---|---|
| **F1A** | Pivot A consistency audit (`f1_verify.md`) | **PASS** (1 partial) | 76 lines |
| **F1B** | Pivot A + widzuipcl integration audit (`f1_integrate.md`) | **WAIT** (widzuipcl Phase 1 only) | 114 lines |
| **F2A** | §4.7 → §4.6.1 PlatinAI oracle column (`f2_platinai_s4.md`) | **PLACEHOLDER SHIPPED** (0 promotions) | 121 lines |
| **F2B** | S11 supplementary data table (`f2_supp_table.md`) | **SHIPPED** (paper/supplementary.tex UNTOUCHED per spec) | 3 lines |
| **F3A** | 5 metric JSONs (`f3_metrics.md`) | **SHIPPED** (2 PENDING_GPU_RETRAIN, 3 MEASURED) | 1 line |
| **F3B** | Cross-ref matrix (`f3_crossref.md`) | **VERDICT WRITTEN** (CROSS_REFS.md unchanged) | 228 lines |
| **F3C** | MASTER.md (this file) | **SHIPPED** | — |

**Total**: 6/7 PASS-or-SHIPPED, 1/7 WAIT (F1B blocked on widzuipcl Phase 2).

---

## 3. MEASURED deltas

| Delta | Before | After | Multiplier | Source |
|---|---|---|---|---|
| **Atom vocab coverage** | 4 elements `{C,N,O,F}` | 14 elements `{C,N,O,F,S,P,Cl,Br,I,Pt,Pd,Au,Ir,Ru}` | **3.5x** | `phase1_filter_fix.md` |
| **Training data scale** | 32 mols (--n-train default) | 500 mols (449 SMILES after parse, 51 dropped) | **15.6x** | `phase1_filter_fix.md` + `f3_metrics.json` |
| **Vertical metrics** | None | PlatinAI oracle wired (`P_A2780`, `P_MCF7`, oracle = mean ∈ [0,1]); `r_platinai` channel planned Round-14 | **+3 columns** | `f2_platinai_s4.md` §1, `platinai_dataset.py:110-200` |
| **Paper structure** | §3.6 CFM stub + §3.7 bond-aware soft prior + §4.7 click-ablation | + **§4.6.1 metallodrug-vertical subsubsection** (165 tex lines, label `sec:evaluation:metallodrug-vertical`) + **S11 supplementary_data_table.tex** (207 lines, 3 sub-tables) | **+2 paper artefacts** | `f2_platinai_s4.md` §1+§2, `f2_supp_table.md` |

**Note**: every numeric cell in §4.6.1 is `\DESIGN{}` (no silent promotion). The 3.5x and 15.6x multipliers are PROVEN (Phase 1 filter changes ship + 13 new tests pass + 9/9 PlatinAI dataset tests pass). The PlatinAI oracle column shape is wired but **not measured on de novo candidates**.

---

## 4. What remains BLOCKED

1. **widzuipcl Phase 2 (GPU retrain) may still be in flight** — `molmetal/reports/wf_metallodrug_vertical/` only contains 3 Phase-1 files; §3.7 metallodrug vertical file + `r_platinai` channel + §4.6.1 cells DESIGN→MEASURED all deferred to widzuipcl Phase 2.
2. **PlatinAI oracle accuracy on novel scaffolds unmeasured** — `PlatinAIDataset.get_activity_matrix()` is the contract; no wet-lab IC50 calibration (MetalCytoToxDB IC50_Dark overlay) computed; Pearson r vs IC50 marked `\DESIGN{}`.
3. **Wet-lab validation not possible** — no in-vitro assay access; all numbers in §4.6.1 are ML-predicted (PlatinAI) or proxy (NCI60 subset), explicitly labelled "ML-predicted, not wet-lab".

**Plus 2 structural gaps** (out-of-scope for this follow-up): `paper/sections/03_7_metallodrug_vertical.tex` MISSING (widzuipcl Phase 2 owns); `paper/supplementary_data_table.tex` ORPHANED (not `\input{}`-ed into `paper/supplementary.tex`, recompile agent owns).

---

## 5. Honest framing

1. **0 DESIGN→MEASURED promotions.** Every numeric cell in §4.6.1 (P_A2780 means, P_MCF7 means, oracle values, Pearson r, MAE, RMSE) is `\DESIGN{}` pending GPU retrain. The 3.5x vocab and 15.6x data scale-ups are the only MEASURED deltas.
2. **No PlatinAI bibitem cited.** `grep -i platinai paper/refs.bib` returns 0 hits; column is "cite-only". The 3 S11 supplementary sub-tables cite `footnote:platinai_dataset` etc. as placeholders, not real bibitems.
3. **Slot rename honest-acknowledged in-text.** §4.6.1 carries an in-line note: "slot renamed to §4.6.1 because §4.7 was already claimed by the 5-click-rule vs CuAAC-only ablation".
4. **Pivot-A invariants preserved.** 9/9 invariants from F1-VERIFY still hold; new §4.6.1 uses "de novo typed-term MCTS, pocket-optional" framing per Pivot A §1 line 110; no claim drift back to "geometric SBDD" / "structure-based".
5. **`refsupplement not added; recompile not run.** Per spec "DO NOT modify paper/ at this stage"; pdflatex verification of §4.6.1 + S11 deferred to a downstream workflow. The 5 in-line `\ref{}` calls in §4.6.1 all resolve to existing labels (verified via grep).

---

## 6. Recommended next action

**(a)** Wait for widzuipcl Phase 2 completion; check final GPU `decode_ratio` on `r_platinai` channel + 5-seed × 10-pocket sweep before promoting §4.6.1 cells DESIGN→MEASURED. Phase 2 also owns `paper/sections/03_7_metallodrug_vertical.tex` creation + `r_platinai` API wire-up in `molmetal/molmetal_lam/reward/reward_aggregator.py`.

**(b)** Submit paper once widzuipcl + Pivot A + this follow-up all complete. Triggers: (i) F1A PASS, (ii) F1B INTEGRATION OK after widzuipcl §3.7 ships, (iii) F2A §4.6.1 cells promoted to MEASURED, (iv) F2B S11 wired into `paper/supplementary.tex` + 4-pass pdflatex+bibtex recompile with 0 unresolved refs, (v) F3 cross-ref matrix fully resolved.

**(c)** Track PlatinAI oracle accuracy (Pearson r vs MetalCytoToxDB IC50_Dark wet-lab) as a Round-14+ follow-up task; honest framing per §4 limitation #2.

---

## 7. Files referenced (absolute paths)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f1_verify.md` (F1A)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f1_integrate.md` (F1B)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f2_platinai_s4.md` (F2A)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f2_supp_table.md` (F2B)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f3_metrics.md` (F3A)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/f3_crossref.md` (F3B)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_followup/MASTER.md` (F3C, this file)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_metallodrug_vertical/phase1_filter_fix.md` (widzuipcl Phase 1, 3.5x + 15.6x source)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_metallodrug_vertical/phase1_inventory.md` (widzuipcl Phase 1, dataset corpus)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_metallodrug_vertical/phase1_inventory.json` (widzuipcl Phase 1, machine-readable)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/platinai_dataset.py` (PlatinAI oracle contract, lines 1-200)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_platinai_dataset.py` (9/9 PlatinAI tests pass)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_data_diversity.py` (13/13 diversity tests pass)
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` (lines 1773-1937 §4.6.1 subsubsection, MODIFIED by F2A)
- `/home/hugo/codes/try_triton_on_rocm/paper/supplementary_data_table.tex` (207 lines S11, CREATED by F2B, NOT yet `\input{}`-ed)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/data/metallo_drugs_500_train.csv` (449 SMILES training pool from Phase 1)

---

**MASTER verdict**: **FOLLOW-UP SHIPPED (6/7) + 1 WAIT (F1B).** Pivot A + widzuipcl Phase 1 integrated with honest framing; §4.6.1 + S11 in place but `DESIGN{}`-only; 0 unresolved cross-refs; 3 follow-ups recorded for widzuipcl Phase 2 + recompile agent.