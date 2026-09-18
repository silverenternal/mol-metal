# Phase 3 Writes — TODO-25 + TODO-26 + INDEX.md + phase3_writes.md

**Date:** 2026-09-15
**Status:** SHIP — Phase 3 deliverables
**Owner:** Phase-3 (paper-side delivery + INDEX reconciliation)
**Round:** Round-14 (Q1-2027) + Round-13 retry + paper submission (Q2-2027)

---

## §1. What was written

| # | File | Purpose | Lines |
|---|---|---|---|
| 1 | `TODO/pending/25_round14_lit_grounded_plan.md` | Round-14 lit-grounded plan — 10 prioritized tasks + decision tree + 4-week timeline + 5 NEW research gaps | ~290 |
| 2 | `TODO/pending/26_round13_round14_complete_plan.md` | Comprehensive master plan — Round-13 retry + Round-14 + paper submission timeline + 6 in-flight workflows | ~270 |
| 3 | `TODO/INDEX.md` | Last-updated header bumped from "parallel_tasks wf finalization" → "TODO-25 + TODO-26 ship" (1-line edit) | 1 |
| 4 | `molmetal/reports/wf_round14_plan/phase3_writes.md` | THIS FILE — summary of what was written, line counts, cross-refs | ~150 |

**Total:** ~711 lines across 4 files; 1 minor edit (INDEX.md).

---

## §2. TODO-25 section map

| § | Title | Content |
|---|---|---|
| §0 | Scope | Round-14 = primary paper-submission round; 100×3 paper-grade data sweep |
| §1 | Round-14 inputs | 11 verbatim references (phase1/phase2 + 4 metrics JSON + 4 reports + TODO-26 + TODO-29) |
| §2 | Top-10 prioritized tasks | T1-T10 with lit anchor + math prior + effort + success + risk |
| §3 | Decision tree for path (a)/(b)/(c) | 3 paths + decision points + default if all fail |
| §4 | 4-week timeline | W38-W41 (Round-14) + W42-W45 (Round-15) + W46-W49 (paper submission) |
| §5 | 5 NEW research gaps | G1-G5 with lit anchor + math prior + status (OPEN / SHIP) |
| §6 | Honest framing | 7 verbatim honest-framing points |
| §7 | Dependencies + success criteria | 10 dependency rows + Round-14/15/paper-submission success criteria |
| §8 | Cross-references | 10 cross-refs |
| §9 | Update protocol | Append-only |

---

## §3. TODO-26 section map

| § | Title | Content |
|---|---|---|
| §0 | Mission | Close R13 with F2(a) + 5 algo fixes + warm-start integration; ship R14 lit-grounded plan; restore paper/main.pdf; submit arXiv + journal |
| §1 | Round-13 retry strategy | Honest framing of R13 verdict + 3 retry components + decision tree + success criteria + risk |
| §2 | Round-14 strategy | Per TODO-25 P0/P1/P2 priorities + success criteria + risk + honest framing |
| §3 | Paper submission timeline | W41 (arXiv) + W42-W45 (R15) + W46 (journal) + W47-W48 (format) + W49 (JOURNAL SUBMIT) |
| §4 | 6 in-flight workflows tracking | WF-Round13-100x3-Sweep + WF-CFM-PathB-GPU-Retrain + WF-Paper-Compile-Verify + WF-Round12-Lambda-PathA-10x3 + WF-PB-Pass-Real-Dock-Integrate + WF-Lambda-MCTS-Coords-Fix |
| §5 | Honest framing of partial lifts + open gaps | 3 partial lifts (MEASURED) + 6 open gaps (NOT closed) + 6 honest framing points |
| §6 | Resource budget | Wall time estimates + GPU budget + risk on this host |
| §7 | Open questions for user | D11 (journal target) + Lambda × CFM coupling timing + R13 retry failure decision tree |
| §8 | Cross-references | 14 cross-refs (TODO + reports) |
| §9 | Update protocol | Append-only |

---

## §4. INDEX.md update

Single 1-line edit to `TODO/INDEX.md` header:
- **Before:** "Last updated: 2026-09-15 (Phase 5: parallel_tasks wf finalization)"
- **After:** "Last updated: 2026-09-15 (Phase 3: TODO-25 + TODO-26 ship)"

No other TODO files modified per task instructions ("DO NOT touch other
TODO/pending/ files").

---

## §5. Cross-references between TODO-25, TODO-26, and prior phases

### §5.1 Phase-1 (lit survey) → TODO-25
- phase1 §1 (CrossDocked2020) → TODO-25 §3 (Path B) + §5 (G2/G3 lit anchors)
- phase1 §2 (MCTS convergence) → TODO-25 §2 T8 (Alt A MCTS budget) lit anchor
- phase1 §3 (CFM conditional) → TODO-25 §2 T5 (CFM wrap reorder) lit anchor
- phase1 §4 (Hierarchical) → TODO-25 §5 G4 (Task L3 hierarchical prior)
- phase1 §5 (Round-14 specific gaps) → TODO-25 §5 (5 NEW research gaps)
- phase1 §6 (SBDD with metal) → TODO-25 §6 (SBDD-with-metal honest framing)

### §5.2 Phase-2 (synthesis) → TODO-25 + TODO-26
- phase2 §1 (open gaps audit) → TODO-26 §5.2 (open gaps NOT closed)
- phase2 §2 (alternatives per gap) → TODO-25 §2 (10 tasks with 3 alts each)
- phase2 §3 (prioritization) → TODO-25 §2 P0/P1/P2 + TODO-26 §2.1
- phase2 §4 (12-week roadmap) → TODO-25 §4 (4-week timeline) + TODO-26 §3
- phase2 §5 (decision trees) → TODO-25 §3 (decision tree for path a/b/c) +
  TODO-26 §1.3 (Round-13 retry decision tree)
- phase2 §6 (dependencies + success criteria) → TODO-25 §7 + TODO-26 §2.2
- phase2 §7 (honest framing) → TODO-25 §6 + TODO-26 §5.3

### §5.3 TODO-25 ↔ TODO-26
- TODO-25 §2 (10 tasks) ↔ TODO-26 §2 (P0/P1/P2 priorities)
- TODO-25 §3 (decision tree for path a/b/c) ↔ TODO-26 §1.3 (R13 retry DT)
- TODO-25 §4 (timeline) ↔ TODO-26 §3 (paper submission timeline)
- TODO-25 §5 (5 NEW research gaps) ↔ TODO-26 §5.2 (6 open gaps)
- TODO-25 §6 (honest framing) ↔ TODO-26 §5.3 (honest framing)

### §5.4 TODO-26 ↔ prior TODOs
- TODO-26 §1.2 (R13 retry) ← TODO-29 (F2(a) + R13 retry)
- TODO-26 §1 (R13 retry) ← TODO-14 (full 100pocket paper R13)
- TODO-26 §3.3 (journal selection) ← TODO-19 D11 (journal target)
- TODO-26 §5.2 (Lambda × CFM coupling) ← TODO-21 (Lambda × CFM deferred)
- TODO-26 §4.3 (paper/main.pdf) ← TODO-27 (paper repair)
- TODO-26 §4.5 (PB pass) ← TODO-14 (full 100pocket paper R13)

---

## §7. Files NOT modified (per task instructions)

Per task: "Files to modify: TODO/pending/25_round14_lit_grounded_plan.md,
TODO/pending/26_round13_round14_complete_plan.md, TODO/INDEX.md ONLY.
DO NOT touch other TODO/pending/ files."

The following files were NOT modified:
- `TODO/pending/13_top_journal_pilot_r12.md` (untouched)
- `TODO/pending/14_full_100pocket_paper_r13.md` (untouched)
- `TODO/pending/19_user_decisions.md` (untouched)
- `TODO/pending/21_lambda_model_coupling.md` (untouched)
- `TODO/pending/22_data_gap_alignment_plan.md` (untouched)
- `TODO/pending/23_weak_to_strong_plan.md` (untouched)
- `TODO/pending/24_cfm_architecture_redo_plan.md` (untouched)
- `TODO/pending/27_paper_main_pdf_repair.md` (untouched)
- `TODO/pending/28_round12_honest_negative_reframe.md` (untouched)
- `TODO/pending/29_f2a_round13_retry.md` (untouched)

---

## §8. Honest framing (preserved verbatim)

1. **No theoretical derivation is re-attempted in this plan.** Each fix
   path cites the lit anchor in `phase1_lit.md §1-§6` + Lit-Survey-v2
   65 papers + 20 theorems.

2. **All claimed lifts are PROJECTIONS, not measurements.** Per TODO-25
   §"5 weak metrics → existing lit + plan", the (a)+(b)+(c)+(d) projected
   lift for Vina (-2 → -7) is a *lit-grounded projection*, not a
   measurement. Round-13 partial did NOT confirm or refute this projection.

3. **Path A-10x3 diversity lift is MEASURED** (n_distinct 1→20, div_tan
   0→0.1065 on test_000..test_009 only). Pocket-invariance on novel
   pockets (test_010..test_019) is NOT MEASURED. This is the gap.

4. **Path B CFM decode_ratio=0/192 is MEASURED honest-negative.** Path B
   5 P0 fixes ship but wrap-ordering issue (hypothesis a per
   `wf_path_b_gpu_retrain/final.md §5.1`) is the dominant cause. T5 (wrap
   reorder) is the recommended fix.

5. **5 NEW research gaps of TODO-25 §5 remain open** and traceable in
   §3.5 → §4.11 → §6.1 → §7.1 of paper.

6. **Round-13 partial = honest-negative, not failure.** Per
   `wf_round13_100x3/final.md §7`, the integration refused to fabricate
   measurements. Round-13 re-execution (per TODO-29 F2(a)) is the gating
   precondition for paper-grade scale-up.

7. **SBDD-with-metal-centers has no SOTA baseline.** Our metal-seeded
   Lambda pilot is one of the first formal ablations. Lit anchor for
   §3.3 MetalGeometryPrior: Aguilar-Rico 2024 §3 + Willnhammer 2025.

8. **GPU recovery is fragile.** Per `wf_gpu_recovery_now/final.md`,
   GPU recovers after cold power cycle but any broken KFD node = fatal
   HSA init. Watchdog required for any GPU run.

---

## §9. Cross-references

- `TODO/pending/25_round14_lit_grounded_plan.md` — Round-14 lit-grounded plan
- `TODO/pending/26_round13_round14_complete_plan.md` — comprehensive master plan
- `TODO/INDEX.md` — single source of truth (last-updated header bumped)
- `molmetal/reports/wf_round14_plan/phase1_lit.md` — lit survey (6 axes)
- `molmetal/reports/wf_round14_plan/phase2_synthesis.md` — synthesis
- `molmetal/reports/wf_round13_100x3/final.md` — Round-13 honest-negative
- `molmetal/reports/wf_path_b_gpu_retrain/final.md` — CFM Path B honest-negative
- `molmetal/reports/wf_round12_lambda_patha_10x3/final.md` — PathA-10x3 PARTIAL_LIFT
- `molmetal/reports/wf_lit_survey_v2/synthesis.md` — 65 papers + 20 theorems
- `metrics/by_round/*.json` — single source of truth for all measured values

---

## §10. Update protocol

Append-only. New fix-ship or new measurement requires a dated section.
Round-14 Phase 2 (code-fix) and Phase 3 (paper §3+§5+§6 update) reference
§1-§6 anchors. Successive round updates append §11, §12, ... — never overwrite.

---