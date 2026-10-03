# TODO Close-out Synthesis — pre-R17 SUPERSEDED + R18 skill refresh

**Date**: 2026-09-26
**Round**: R17 close-out → R18 pivot
**Parent**: `TODO/completed/41_r17_final_closeout.md` + `TODO/completed/55_r18_integration_adapters.md` + `TODO/completed/58_weak_to_strong.md`
**Workflows aggregated**: Wf-1 (math skeleton) + Wf-2 (adapter) + Wf-3 (paper writing) + Wf-4 (TODO-21 coupling, in-flight) + Wf-5 (TODO-23 weak→strong, done) + Wf-6 (TODO-24/25, in-flight) + Wf-7 (TODO-30 pitfall, just launched) + Wf-8 (TODO-53 reality check, done)
**Skill set refreshed**: 4 skills × 1 new R18-ship-reference section each

---

## 1. TL;DR

10 pre-R17 TODO entries marked **SUPERSEDED 2026-09-25/26** and archived to `TODO/04_history/` (block prefix `_*_superseded_2026-09-{25,26}.md`). The 6 still-active pending TODOs split into 3 finished, 2 in-flight, 1 just-launched. 4 skills under `.claude/skills/` received a new "R18 ship reference" section naming the Wf-1..Wf-8 close-out anchors so future paper/training runs land on the post-R17 surface, not the legacy R12/R15 surface.

---

## 2. Per-deliverable: SUPERSEDED file path + status

| # | Was (in `pending/`) | Now (in `04_history/`) | Status | Triggered by |
|---|---|---|---|---|
| 1 | `14_full_100pocket_paper_r13.md` | `14_..._superseded_2026-09-26.md` | **SUPERSEDED** | Wf-5 (58) + Wf-7 (30) close R13 paper-grade scope to R18 follow-up |
| 2 | `19_user_decisions.md` | `19_..._superseded_2026-09-25.md` | **SUPERSEDED** (hand-ship) | D6/D7/test_005/MW/journal all closed by R17 final state |
| 3 | `22_data_gap_alignment_plan.md` | `22_..._superseded_2026-09-26.md` | **SUPERSEDED** | R17 close-out (4561625) + paper §4.5 + §6 (wt8 AC4 + AC2) |
| 4 | `26_round13_round14_complete_plan.md` | `26_..._superseded_2026-09-26.md` | **SUPERSEDED** | R12 complete + R15 structural + R16 YuelBond + R18 plan takes over |
| 5 | `28_round12_honest_negative_reframe.md` | `28_..._superseded_2026-09-26.md` | **SUPERSEDED** | §4.5 (731 lines) + §6 absorbed all 5/7 items + 2 cosmetic deferred to v2 |
| 6 | `31_r17_ultracode_dispatch.md` | `31_..._superseded_2026-09-26.md` | **SUPERSEDED** | All 8 dispatchable workflows ship via 41/55/58 |
| 7 | `31_ultracode_trackA_flow_matching.md` | `31_..._superseded_2026-09-26.md` | **SUPERSEDED** | A1+A2+A3+A4 all green; folded into 37 |
| 8 | `32_ultracode_trackB_lambda_calculus.md` | `32_..._superseded_2026-09-26.md` | **SUPERSEDED** | B0..B5 all ship via 32 + 41 |
| 9 | `33_r17_fixes_blackwell_loader_paths.md` | `33_..._superseded_2026-09-26.md` | **SUPERSEDED** | P1..P7 all closed; shim fixes snapshot in `flow_matching_lipman/` |
| 10 | `34_parallel_ultracode_dispatch.md` | `34_..._superseded_2026-09-26.md` | **SUPERSEDED** | 6 parallel tracks (UC-A/B/L/M/C/F) all ship via 41 + 55 |

**Verbatim status banner** (preserved at top of each file):
> `STATUS: SUPERSEDED 2026-09-25/26 by R17 close-out (commit 4561625) + Wf-N …`

---

## 3. Pending directory status (after close-out)

| TODO | Title | Workflow | Status | Next action |
|---|---|---|---|---|
| **21** | Lambda × 模型侧联合训练 | **Wf-4 in-flight** | Strategy 1 ships (LambdaASTSR default), Strategy 3 ships (CPU CascadedRefiner), Strategy 2 DEFERRED to R18 Phase 1-3 | GPU Blackwell unblock; 32-d → 64-d bridge adapter at `models/coupling_adapter_platinai5k.npz` |
| **23** | 弱项 → 强项 转换规划 | **Wf-5 done** | 8 LEAD claims + 8-9 honest negative + 6 limitations locked; 15 credibility tests gate §4 paper-ready | Done; consumed by `mol-metal-paper-writing` skill §"R18 ship reference" |
| **24** | CFM Architecture Redo Plan | **Wf-6 in-flight** | P0+P1+YuelBond + Frontier 24 papers ship; GPU retrain deferred | R16 GPU retrain on Blackwell (sm_120) |
| **25** | Round-14 Lit-grounded + Math-prior + Code-fix | **Wf-6 in-flight** | D2/D4 lit-grounded fixes ship; M3+CFM gate in flight | Aligns with Wf-6 CFM frontier |
| **30** | Pitfall Reinforce Plan | **Wf-7 just launched** | Tier-1 5/5 closed; Tier-2 (Rank 6-16) deferred | Launch Tier-2 once Wf-6 lands |
| **53** | Phase A Reality Check | **Wf-8 done** | HONEST_NEGATIVE — 4-layer diagnosis confirms Phase A BLOCK; pivot to R18 SE(3) Schrödinger Bridge | Done; feeds `mol-metal-r18-math` §14 |
| **19** | (was archived above) | **hand-ship** | SUPERSEDED 2026-09-25 | Closed |

**All 10 superseded files now live in `TODO/04_history/`** (block-prefix `_superseded_2026-09-{25,26}.md`). `pending/` retains the 6 active TODOs above.

---

## 4. Skill update list — 4 skills × 1 new section each

| # | Skill (`.claude/skills/`) | New section | Anchor |
|---|---|---|---|
| **1** | `mol-metal-r18-math/SKILL.md` | **§14. R18 ship reference (2026-09-26 Wf-1 + Wf-2 close-out)** | 9-定理 backlog → shipped W2.1/W2.2/W2.3 trio; T1/T2/T5 P1→shipped; T3/T4/T7 P1 deferred; T6 guardrail; T8/T9 paper/deferred |
| **2** | `mol-metal-cfm-training/SKILL.md` | **R18 ship reference — Wf-1 math + Wf-2 adapter + M3/M4/M5 mechanism anchors** | W2.1 SE(3) SB (`se3_sb_integration.py`) + W2.2 Transport Relaxation (`transport_relaxation_integration.py`) + W2.3 Multi-PH (`multi_ph_integration.py`); `integrate_with_cfm()` signature-compatible with `ODESolver` |
| **3** | `mol-metal-lambda-calculus/SKILL.md` | **R18 ship reference — TODO-21 Lambda × CFM integration (Wf-4 + Wf-6 + Wf-7)** | Strategy 1 LambdaASTSR default + PlatinAI 5K coupling adapter + Strategy 3 CascadedRefiner; 30-cell R²=1.0000; ablation gate +0.0676 ≥ +0.05 PASS |
| **4** | `mol-metal-paper-writing/SKILL.md` | **R18 ship reference — Wf-5 weak→strong decision + §4 LEAD-claim 8** | m11/m12/m13/m14/m17/m2(PC3+HL60+SKOV3)/m21/Deflexpressor = 8 LEAD claims; 8-9 honest-negative + 6 limitations; tag discipline `\textbf{MEASURED}` / `\textbf{PROJECTED @ R18}` / `\textbf{ARCHITECTURAL}` |

**Discipline invariant**: every new R18-ship-reference section cross-references the matching `completed/` synthesis report and explicitly states **which R17 surface is preserved** (no regression to `flow_matching_lipman/` close-out anchors, `lambda_combinators.py` 42-entry `PRIMITIVE_ATOMS`, `METRICS_TARGETS.md` 20-row ledger).

---

## 5. Cross-reference

| Skill | Companion synthesis report | Cross to |
|---|---|---|
| `mol-metal-r18-math` | `TODO/completed/55_r18_integration_adapters.md` (W2.1+W2.2+W2.3 trio) | `mol-metal-cfm-training` (CFM-side integration point) |
| `mol-metal-cfm-training` | `TODO/completed/55_r18_integration_adapters.md` + `TODO/completed/41_r17_final_closeout.md` §2 (wf #5 M-R17-A1+A2+A3) | `mol-metal-lambda-calculus` (β-NF → CFM refine) |
| `mol-metal-lambda-calculus` | `TODO/completed/57_lambda_cfm_coupling.md` + `TODO/completed/35_l_r17_b3_astsr_swap_done.md` | `mol-metal-cfm-training` (refine target) + `mol-metal-click-chemistry` (18 rules) |
| `mol-metal-paper-writing` | `TODO/completed/58_weak_to_strong.md` + `TODO/completed/41_r17_final_closeout.md` §3 (5 outstanding) | `mol-metal-mechanism-mapping` (M-row anchor per §4 metric) |

**One-line per skill to remember when loading**:
- `mol-metal-r18-math`: 9 theorem backlog → 3 ship / 3 P1 deferred / 1 guardrail / 2 cite-only.
- `mol-metal-cfm-training`: R17 close-out = 30/30 decode + 14/14 evaluators + 6 shim fixes; W2.x adapters sit alongside (signature-compatible).
- `mol-metal-lambda-calculus`: 42-entry `PRIMITIVE_ATOMS` + 101-rule MLE matrix + 18 click rules + LambdaASTSR default reward.
- `mol-metal-paper-writing`: 8 LEAD §4 claims + 8-9 honest-negative + 6 limitations; tag discipline mandatory.

---

## 6. Process notes (for next close-out cycle)

- All 10 superseded files use the same banner template at the top — future close-outs should preserve verbatim so `grep "STATUS: SUPERSEDED"` returns the full block.
- The `04_history/` directory is the canonical archive; do NOT add new pending files to `04_history/` directly — always go through `pending/ → 04_history/` with the `_superseded_YYYY-MM-DD.md` suffix.
- Skills should be refreshed **once per round close-out** (R17 → R18) and never mid-sprint; mid-sprint skill edits create drift between skill docs and `molmetal/reports/wf_*/final.md`.
- `TODO/INDEX.md` and `TODO/README.md` already reflect the 2026-09-25 R17 banner; the next INDEX bump should mention this file (`61_todo_closeout.md`) once Wf-6 + Wf-7 land.

Last updated: 2026-09-26.
