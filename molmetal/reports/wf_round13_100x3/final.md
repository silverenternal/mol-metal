# WF-Round13-100x3-Sweep — Phase 3 Final Report (paper integration, honest verdict)

**Date:** 2026-09-15
**Phase:** 3 (paper integration) — **partial completion**
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Spec target:** 100 pockets × 3 seeds = 300 evaluations, matching TargetDiff Guan ICLR 2023 standard (100 test pockets).

> **Honest-framing mandatory.** The Round-13 sweep was specified to
> deliver a paper-grade 300-cell aggregate across three paths. The
> `paper_grade_data_ready` flag is `false`. The Phase 3
> paper-integration refuses to silently promote `\DESIGN{}` →
> `\MEASURED{}` for cells that were not actually measured. This
> final report records exactly what shipped, what was blocked, and
> what the paper shows.

---

## 0. Relation to Phase 2 final report

The previous Phase-2 final.md (executed 2026-09-15, before this
Phase-3 update) is preserved verbatim upstream; this Phase-3 final
supplements it with the paper-integration verdict, the schema
machine-readable block, and the honest-deviation table that
records every spec'd deliverable that was NOT shipped. The Phase-2
report is reachable at `molmetal/reports/wf_round13_100x3/final.md`
revision @ 2026-09-15 15:55 HKT.

---

## 1. TL;DR — schema summary

| Field | Value |
|---|---|
| `n_sections_updated` | **3** (`paper/sections/04_evaluation.tex` header + status block + NEW §4.11 `sec:evaluation:round13-honest`; `paper/sections/06_limitations.tex` NEW item (8) + count 8→9; `paper/sections/CROSS_REFS.md` NEW §4.11 entry + §6 count 8→9) |
| `n_cells_design_to_measured` | **0** (integration refused silent promotion; `paper_grade_data_ready = false`) |
| `paper_grade_vina_mean` | **null** (no 100-pocket Vina aggregate; cite-only TargetDiff `$-8.45$` per Guan ICLR 2023) |
| `paper_grade_diversity_intdiv1` | **null** (no 100-pocket IntDiv$_1$ aggregate; cite-only TargetDiff `0.860`) |
| `paper_grade_pb_pass_rate` | **null** (PB path returned 30/30 search-bound, 0 PB-eligible docked candidates; cite-only Uni-Mol-v2 `75%+`) |
| `paper_grade_sa_mean` | **null** (no 100-pocket SA aggregate; cite-only TargetDiff `2.65–2.86`) |
| `comparison_vs_targetdiff` | **DESIGN** (cite-only SOTA context columns remain in §4.4; live Mol-Metal vs TargetDiff comparison cells NOT produced) |
| `comparison_vs_uni_mol_v2` | **DESIGN** (cite-only SOTA context columns remain; live Mol-Metal vs Uni-Mol-v2 comparison cells NOT produced) |
| `honest_framing_per_lift` | **NO lift claimed** (every cell that could have been promoted `\DESIGN{}`→`\MEASURED{}` remains `\DESIGN{}`; the integration refused to fabricate measurements) |

---

## 2. What Round-13 promised vs delivered

| Promised (spec) | Delivered (actual) | Why |
|---|---|---|
| 100 × 3 = 300 Lambda evaluations @ n_sim=1000 | 0 final aggregate cells on disk + 1 partial run killed at ~55% + 1 concurrent n_sim=1000 background process terminated at ~12% | Two concurrent CPU-bound sweeps competed; both killed before persisting |
| 100 × 3 = 300 PB evaluations | 30 cells (10 pockets × 3 seeds @ n_sim=200) | Per-cell search-bound (15 no_candidates + 15 seed_only); pb_eligible=0; pb_pass_rate_aggregate=null by construction |
| CFM 300 evaluations | 0 | BLOCKED: `r10_cfg_real_crossdocked.py` ModuleNotFoundError + dGPU HSA userland outage |
| 300 cells DESIGN→MEASURED in §4.2 / §4.3 / §4.6 | **0** (integration refused silent promotion) | `paper_grade_data_ready = false` |
| Paper-grade Vina mean vs TargetDiff `$-8.45$` | **null** (no 100-pocket Vina aggregate) | Path B did not produce docked candidates |
| Paper-grade PB pass rate vs Uni-Mol-v2 `75%+` | **null** (pb_pass_rate_aggregate = null) | Path B search-bound at n_sim≤200/depth=3 across all 10 pockets |
| Paper-grade Diversity (IntDiv$_1$) vs TargetDiff `0.860` | **null** (no 100-pocket diversity aggregate) | Path A did not persist |
| Paper-grade SA vs TargetDiff `2.65–2.86` | **null** (no 100-pocket SA aggregate) | Path A did not persist |
| Lift §6 weak #3 / #4 / #5 (pIC$_{50}$ regression / QVina binary / homotype H1) | NOT LIFTED (the Round-13 sweep did not provide the data to do so) | Honest-framing: §6 items (3)/(4)/(5) stand as reported; §6 item (8) NEW captures the Round-13 partial-completion limitation |

---

## 3. What this Phase 3 actually shipped

### 3.1 Paper §4 Evaluation — `paper/sections/04_evaluation.tex`

- Updated header comment block (lines 26–38): records that `paper_grade_data_ready = false` and that no `\DESIGN{}`→`\MEASURED{}` promotion was made in this update.
- Updated `\paragraph{Status (WF-Lambda-Only-Paper-Path, 2026-09-15)}` (lines 44–60): flags the partial Round-13 outcome before the headline $\Lambda$-only column narrative.
- **NEW §4.11 sub-section** `sec:evaluation:round13-honest` (inserted before the §4.12 Hybrid WIP sub-section): spec → result table (3 paths) → reasons each path failed → 4-item follow-up to close Round-13 → paper-grade comparison values NOT MEASURED, deferred to Round-13 re-execution.

### 3.2 Paper §6 Limitations — `paper/sections/06_limitations.tex`

- **NEW §6 item (8)**: explicit prose documenting the Round-13 100×3 partial-completion limitation — Path A killed (0 final aggregate), Path B 30/30 search-bound (pb_pass_rate=null), Path C BLOCKED (import + GPU outage). The §6 list count moves 8 → 9.
- Updated the user-gated decisions paragraph to reference items (1)–(8) instead of (1)–(7).

### 3.3 Paper §4 / §6 Cross-references — `paper/sections/CROSS_REFS.md`

- NEW entry for §4.11 (`sec:evaluation:round13-honest`) recording the partial-completion verdict + the 4-item close-Round-13 follow-up.
- §6 entry count updated 8 → 9 honest-framed limitations.

---

## 4. What this Phase 3 explicitly did NOT ship (and why)

| Not shipped | Why not |
|---|---|
| 9 900 cells `\DESIGN{}`→`\MEASURED{}` in §4.2 Table 1 | `paper_grade_data_ready = false`; integration refused silent promotion |
| §4.3 Table 2 (Lambda + CFM + hybrid) mean±std | Lambda path: 0 final aggregate; CFM path: BLOCKED |
| §4.6 PB + SA + Diversity panel with full 100×3 numbers | PB path: 30/30 search-bound (pb_pass_rate=null); SA / Diversity not measured at 100×3 |
| §4 paper-grade Vina mean comparison vs TargetDiff `$-8.45$` | Not measured; cite-only context preserved |
| §4 paper-grade PB pass rate comparison vs Uni-Mol-v2 `75%+` | Not measured; cite-only context preserved |
| §4 paper-grade IntDiv$_1$ comparison vs TargetDiff `0.860` | Not measured; cite-only context preserved |
| §4 paper-grade SA comparison vs TargetDiff `2.65–2.86` | Not measured; cite-only context preserved |
| §6 weak #3 lifted (pIC$_{50}$ regression-to-mean) | Round-13 did not produce data; item stands |
| §6 weak #4 lifted (QVina2 binary mismatch) | Round-13 did not produce data; item stands |
| §6 weak #5 lifted (homotype H1 strict) | Round-13 did not produce data; item stands |
| `TODO/pending/14_full_100pocket_paper_r13.md` SHIPPED status | Honest framing: this update is PARTIAL; full SHIPPED requires the 4-item close-Round-13 follow-up to land |
| `TODO/pending/25_round14_lit_grounded_plan.md` Round-13 100×3 vs projected Path B lift summary | The Round-13 100×3 was partial, not a clean comparison point; the projection-vs-measurement comparison would be misleading without data |

---

## 5. Follow-up plan to actually close Round-13

For the Round-13 sweep to deliver the 300-cell TargetDiff-grade Lambda panel that the spec demanded:

1. **Serialize Lambda runs.** Kill any concurrent n_sim=1000 process and launch a single n_sim=200 sweep with no competing CPU contention. ETA ~110 min for 300 cells.
2. **Fix `r10_cfg_real_crossdocked.py` import.** Either add `sys.path.insert(0, project_root)` at the top of the file, or set `PYTHONPATH` env var in the uv invocation. ETA 5 min.
3. **Recover the dGPU** (out-of-scope for this session). Per `wf_gpu_diag/diagnosis.md`, the only known fix is a cold PSU power cycle.
4. **Lift MCTS search budget** for PB path before re-running PB at scale. The 30-cell result shows the MCTS search is the bottleneck, not Vina / PB. Per-cell cost dominated by search not by Vina/PB; full 100 pockets would just multiply the failure mode.

After items (1)–(4) close, re-run Round-13 and re-invoke Phase 3 paper-integration. At that point the §4.2 / §4.3 / §4.6 cells become eligible for `\DESIGN{}`→`\MEASURED{}` promotion, §6 item (8) can be retired, and `TODO/pending/14 + 25` can carry the full 100×3 aggregate.

---

## 6. Artifacts produced this Phase 3

| Path | Description |
|---|---|
| `paper/sections/04_evaluation.tex` | Header comment block updated + status paragraph updated + NEW §4.11 `sec:evaluation:round13-honest` sub-section |
| `paper/sections/06_limitations.tex` | NEW §6 item (8) Round-13 partial-completion; item count 8 → 9; user-decisions paragraph (1)–(7) → (1)–(8) |
| `paper/sections/CROSS_REFS.md` | NEW §4.11 entry; §6 count 8 → 9 honest-framed limitations |
| `molmetal/reports/wf_round13_100x3/final.md` | THIS report (schema with `n_sections_updated=3`, `n_cells_design_to_measured=0`, all `paper_grade_*` = `null`, honest framing) |
| `molmetal/reports/wf_round13_100x3/aggregate.json` | Pre-existing Phase 2 aggregate (unchanged) — `paper_grade_data_ready=false` |

---

## 7. Honest deviation from spec — summary

The spec asked for paper-grade scale-up to 100 pockets × 3 seeds = 300 evaluations matching the TargetDiff standard, and for the Phase 3 paper-integration to promote 9 900 cells `\DESIGN{}`→`\MEASURED{}`, compute paper-grade comparisons vs TargetDiff / Uni-Mol-v2, and lift §6 weak items #3 / #4 / #5.

**Delivered honestly:**

- 3 paper sections updated (§4 status + §4.11 + §6 item (8) + CROSS_REFS).
- 0 cells promoted `\DESIGN{}`→`\MEASURED{}` (refused silent promotion).
- 4 headline paper-grade comparison values (Vina / PB / Diversity / SA) recorded as `null` with the cite-only context preserved.
- 4 §6 weak items (#3 / #4 / #5 + new #8) honestly recorded as NOT lifted by this round.

**The integration refused to fabricate measurements.** The honest
path forward is to close the 4-item follow-up list and re-run Round-13
to actually deliver the 300-cell aggregate.

---

## 8. Schema response (machine-readable)

```json
{
  "phase": "WF-Round13-100x3-Sweep_Phase3_paper_integration",
  "n_sections_updated": 3,
  "n_cells_design_to_measured": 0,
  "paper_grade_vina_mean": null,
  "paper_grade_diversity_intdiv1": null,
  "paper_grade_pb_pass_rate": null,
  "paper_grade_sa_mean": null,
  "comparison_vs_targetdiff": "DESIGN",
  "comparison_vs_uni_mol_v2": "DESIGN",
  "honest_framing_per_lift": "NO lift claimed; integration refused silent DESIGN->MEASURED promotion because paper_grade_data_ready was false; the 4-item close-Round-13 follow-up plan is the path to actual paper-grade scale-up",
  "paper_grade_data_ready": false,
  "r13_sweep_status": "partial",
  "r13_path_a_lambda_final_cells": 0,
  "r13_path_b_pb_final_cells": 30,
  "r13_path_b_pb_eligible": 0,
  "r13_path_b_pb_pass_rate": null,
  "r13_path_c_cfm_status": "BLOCKED",
  "r13_total_evaluations_actually_measured": 30,
  "r13_total_evaluations_spec": 300,
  "r13_follow_up_close_actions": [
    "serialize Lambda runs (single n_sim=200, ~110 min)",
    "fix r10_cfg_real_crossdocked.py import (one-line sys.path.insert or PYTHONPATH)",
    "recover dGPU via cold PSU power cycle (out-of-scope)",
    "lift MCTS search budget before re-running PB at scale"
  ],
  "paper_sections_touched": [
    "paper/sections/04_evaluation.tex",
    "paper/sections/06_limitations.tex",
    "paper/sections/CROSS_REFS.md"
  ],
  "section_06_item_count": 9,
  "todo_summary_targets": [
    "TODO/pending/14_full_100pocket_paper_r13.md",
    "TODO/pending/25_round14_lit_grounded_plan.md"
  ],
  "todo_summary_appended": false,
  "todo_summary_appended_reason": "The Round-13 100x3 was partial, not a clean comparison point; appending SHIPPED-style summaries to TODO-14 and TODO-25 would be misleading. Honest framing: Phase 3 deliverable is the paper-integrity audit trail (§4.11 + §6 item 8 + CROSS_REFS), not a SHIPPED status flag for the un-shipped full 100x3 aggregate."
}
```