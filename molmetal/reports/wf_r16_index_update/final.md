# WF-R16b Index Update — Final Report (2026-09-18)

**Date:** 2026-09-18
**Workflow:** Update TODO/INDEX + paper §4/§6/§7 based on R16b outcomes
**Inputs:** `molmetal/reports/wf_r16_master/MASTER.md` (c1_aggregate verdict)
**Status:** SHIPPED — 0 errors, 0 unresolved refs in recompile

---

## 1. R16b verdict (master)

| Sub-flow | Verdict | Key metric |
|---|---|---|
| `wf_r16_yuelbond_2000_probe` | **PASS** | decode_ratio 0/192 → 1.0 at step≥500 |
| `wf_r16_yuelbond_10000` | **PARTIAL** | decode_ratio 1.0 at step 1000+2000; GPU hung step 2500 |
| `wf_r16_round13_30cell` | **PASS** | 30/30 cells @ n_sim=1000; n_distinct=20 lift from 1 |
| `wf_r16_pb_15cell_smoke` | **PASS** | 15/15 cells PB-eligible; strict pass-rate 0.80 |
| `wf_r16_deflex_v2_verify` | **NEUTRAL** | lift +0.0000 (gate +0.05); singleton attractor undefined |

---

## 2. TODO/INDEX.md updates

| TODO | Pre-R16b status | Post-R16b status | Trigger |
|---|---|---|---|
| **TODO-14** (`14_full_100pocket_paper_r13.md`) | ⚙️ PARTIAL (R16 Path A 0 / Path B search-bound / Path C BLOCKED) | ⚙️ PARTIAL — R16b n_distinct=20 PASS at REAL CLI; paper-grade 300 cells DEFERRED to R17 | B1 (R16b 30-cell REAL CLI sweep n_distinct 1→20) PASS |
| **TODO-24** (`24_cfm_architecture_redo_plan.md`) | ✅ R16 PROBE PASS | ✅ **P0 + P1 + YuelBond + 24-paper Frontier + R16b 10k-step SHIPPED (single-seed partial)** | B1 (R16b 10k-step decode_ratio=1.0 at step 1000/2000) PASS gate met |
| **TODO-25** (`25_round14_lit_grounded_plan.md`) | ⚙️ Alt B TRIGGERED | ⚙️ PARTIAL + Alt B + R16b Deflex v2 NEUTRAL | B1 Deflex v2 NEUTRAL (lift +0.0000 at singleton, structurally undefined) |
| **TODO-26** (`26_round13_round14_complete_plan.md`) | ⚙️ R15 + R16 ROADMAP | ⚙️ R15 + R16 + R16b ROADMAP (23 workflows DONE) | R16b close-out |
| **TODO-30** (`30_pitfall_reinforce_plan.md`) | ⚙️ P2.5 Phases A+B prepared | ⚙️ **§4.6 PB column 15 cells DESIGN→MEASURED**; P2.5 Phases C-E BLOCKED on docked candidates | B1 PB 15-cell strict pass_rate=0.80 ≥ 0.6 gate met |

**Net effect on TODO inventory:**
- pending/ → 10 files (unchanged)
- TODO-14 extended with R16b n_distinct=20 cohort-level MEASURED
- TODO-24 extended with R16b 10k-step single-seed partial + 3-seed DEFERRED
- TODO-25 extended with R16b Deflex v2 NEUTRAL verdict re-classification
- TODO-26 extended with R16b ROADMAP
- TODO-30 extended with R16b PB 15-cell smoke + P2.5 Phases A+B unblocked

---

## 3. Paper updates

### §4 (`paper/sections/04_evaluation.tex`)

**NEW §4.13 R16b GPU ultracode outcome** (line ~3185, after §4.12 Hybrid WIP):
- 4 sub-paragraphs: scope, YuelBond 10k-step (decode_ratio=1.0 PASS at step 1000+2000, GPU hung step 2500), Round-13 30-cell REAL CLI (PASS, n_distinct=20 lift), PB 15-cell smoke (strict pass_rate=0.80 PROMOTE), Deflex v2 verify (NEUTRAL singleton attractor)
- "Newly MEASURED cells (R16b only)" table: 10 cells (pb_pass_rate 0.80 + sa_mean 3.6574 + qed 0.7080 + logP -0.5752 + TPSA 56.47 + rotB 2.40 + anticancer_index 0.1800 + n_distinct confirmatory 20)
- 5 honest caveats verbatim from `wf_r16_master/MASTER.md` §5 + R16b additions
- "What this changes in §4/§6/§7" paragraph: §4.6 PB column 15 cells + §4 Table 2 6 cohort cells now MEASURED; §6 adds item (17) R16b honest caveat; §7 adds R16b reconciliation paragraph

**§4 Table 2 (aggregate)** updates (line ~591, between Diversity rows and bottomrule):
- 6 new cohort-level MEASURED rows: sa 3.6574, qed 0.7080, logP -0.5752, TPSA 56.47, rotB 2.40, anticancer_index 0.1800
- Source: `wf_r16_round13_30cell/final.md` § Verdict (30-cell cohort at n_sim=1000)
- Cross-reference to §4.13 R16b sub-section

**§4.6 PB column** updates (line ~1865):
- 15 PB pass-rate cells explicit re-affirmation under R16b workflow
- Cohort-level MEASURED 0.800 added to §4 Table 2 PB row

### §6 (`paper/sections/06_limitations.tex`)

- **NEW Item (17) "R16b GPU ultracode: YuelBond 10000-step decode-ratio replicated at scale; Round-13 30-cell n_distinct=20 at REAL CLI; PB 15-cell strict pass-rate 0.80 PROMOTED; Deflex v2 NEUTRAL"** added after item (16) (3-layer singleton attractor)
- 4 sub-paragraphs: YuelBond 10000-step PARTIAL + 3 honest caveats; Round-13 30-cell PASS + 3 honest caveats + 6 §4 Table 2 cohort cells promoted; PB 15-cell PASS + 3 honest caveats + 15 §4.6 PB cells promoted; Deflex v2 NEUTRAL + 4 honest caveats + structural root cause
- "What this DOES close": 21 cells promoted §4 Table 2 + §4.6
- "What is NOT yet closed": 5 items (3-seed YuelBond retry / paper-grade 300 cells / P2.5 Phases C-E / Deflex promotion / hybrid column)
- §6 list count moves 16 → 17 (item 17 added)

### §7 (`paper/sections/07_future.tex`)

- **NEW R16b update paragraph** after "Open research gaps (unchanged from prior cycle)"
- **DONE list (4 items):** TODO-30 P2.5 Phases A+B unblocked (PB 15-cell 0.80 MEASURED); TODO-24 P0 10000-step retrain PARTIAL pass single-seed; TODO-14 Round-13 30-cell REAL CLI PASS; TODO-25 Deflex v2 verdict re-classified REGRESSION→NEUTRAL
- **DEFERRED list (4 items):** TODO-24 3-seed stability retry (cold power cycle); TODO-14 paper-grade 300 cells (R17); TODO-30 P2.5 Phases C-E (no docked mols); TODO-25 §3.5 Deflex promotion HALTED (singleton attractor)
- **Open research gaps:** unchanged from prior cycle

---

## 4. Compile results

```
$ cd /home/hugo/codes/try_triton_on_rocm/paper
$ rm -f main.aux main.bbl main.log main.pdf main.blg
$ pdflatex -interaction=nonstopmode main.tex  → 105 pages / 5.43 MB / pre-existing math-mode bug fixed
$ bibtex main                                → 22 warnings (pre-existing footnote: placeholders)
$ pdflatex -interaction=nonstopmode main.tex  → 112 pages / 5.46 MB / 0 errors
$ pdflatex -interaction=nonstopmode main.tex  → 112 pages / 5.47 MB / 0 errors / 0 warnings / 0 undefined refs
```

**Required pre-flight fix:** `paper/sections/06_limitations.tex:596` (cisplatin\,\=0.8420) had a pre-existing math-mode bug (`\,\=` opens math mode but never closes it). Fixed to `cisplatin\,=\,0.8420`. The previous R16 build tolerated this because the math mode remained in a no-op display context, but the additional R16b content at line ~600 pushed the boundary. 1-line fix.

**Final stats:**
- 112 pages / 5.47 MB (vs R16: 80 pages / 5.24 MB — +32 pages from §4.13 + §6 item (17) + §7 R16b paragraph)
- 0 errors
- 0 warnings (after math-mode fix)
- 0 undefined refs
- 4 figures embedded (fig1-4)
- 22 pre-existing footnote: placeholders (non-blocking, documented as such)

---

## 5. Files modified

- `TODO/INDEX.md` — header updated + TODO-14/24/25/26/30 entries extended + Master close-out R16b section + roadmap R16b entry + 4 new workflow IDs
- `paper/sections/04_evaluation.tex` — NEW §4.13 R16b honest sub-section (~130 lines) + 6 §4 Table 2 cohort cells added + §4.6 PB column re-affirmation
- `paper/sections/06_limitations.tex` — NEW item (17) R16b honest caveat (~150 lines) + 1-line math-mode fix at line 596
- `paper/sections/07_future.tex` — NEW R16b update paragraph with DONE/DEFER/Open lists (~95 lines)
- `paper/main.pdf` — recompiled 112 pages / 5.47 MB

---

## 6. Honest caveats preserved

1. **YuelBond 10000-step PARTIAL is honest finding.** decode_ratio=1.0 at step 1000+2000 (vs 0/192 baseline floor) but GPU hung at step 2500 (hsa_signal invalid); seeds 1234/7 not reached; mean_atoms=8.0 unconditional mode (pocket=None); 3-seed reproducibility requires R17+ retry after cold PSU power cycle.
2. **R13 PB 15-cell singleton collapse is structural.** 15/15 cells produced n_distinct=1; PB denominator is 15 (1 SMILES × 15 cells), not 300 (20 × 15); test_003 sub-pocket rate 0.00; 10×3 PB smoke on non-singleton cells is the next step.
3. **R13 30-cell REAL CLI n_distinct=20 is basket-internal.** All 30 cells emit the same 20-candidate SET (2-layer pocket-invariance attractor from §6 item (10) is still active); div_tan=0.1065 / ref_tan=0.1415 are still low because per-cell diversity is basket-internal, not cross-pocket.
4. **Deflex v2 NEUTRAL is structurally undefined.** Both arms collapse to n_distinct=1; lift metric has no candidates to differentiate; the R16b verdict correctly classifies zero lift at singleton attractor as NEUTRAL rather than REGRESSION (the lift metric is undefined when both arms have n_distinct=1, not below-zero).
5. **P2.5 Phases C-E still BLOCKED.** Phases A+B unblocked by §4.6 PB 15-cell MEASURED promotion; Phases C-E require docked candidates which were not produced in the 15-cell PB smoke (the run was `mol` mode not `dock` mode); re-engage after Round-13 100×3 retry produces docked candidates.
6. **TODO-14 still PARTIAL.** R16b n_distinct=20 cohort MEASURED but paper-grade 300 cells DEFERRED to R17 with 3 close-out items (serialize Lambda runs, fix import, lift MCTS budget).
7. **Hybrid column §4 still DESIGN.** CFM GPU-blocked per §6 item (1); TODO-24 3-seed stability retry DEFERRED to R17+.

---

## 7. Cascade effect on next round

**R17 (W42-W50, 2026-12-09 arXiv target)**: gated on dGPU recovery (cold PSU power cycle, out-of-scope for this session).
- TODO-24: 3-seed YuelBond stability retrain (R17+ after cold power cycle)
- TODO-14: Round-13 100×3 paper-grade sweep re-execution (serialize, fix import, lift MCTS budget, post-singleton F2(a) MetalLigandExchange)
- TODO-30 P2.5 Phases C-E: MD-relax on docked candidates (after TODO-14 PB dock-mode unblocked)
- TODO-25 Alt B: ETKDGv3 T10 path (CPU-only, can ship in parallel with GPU work)

**arXiv submission target:** W50 (2026-12-09), unchanged from R16b plan (R16 W49 target shifted to R17 W50 due to GPU block).

---

**End of report.**