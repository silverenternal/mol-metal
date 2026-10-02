# Wf-303A — TODO Audit: Planned-but-never-done code tasks

**Date**: 2026-09-29
**Owner**: Hugo Lee (audit)
**Trigger**: User request "现在看一下todo目录里面还有没有说明之前规划过的但是从来没做过的任务(代码类),抓紧时间做起来"

---

## §0 Audit method

Scanned all TODO subdirs:
- `TODO/pending/` (18 files)
- `TODO/03_plans/` (10 files — IMPROVEMENT_ENUM_2026_09_27.md / IMPROVEMENT_QUEUE_2026_09_27.md enumerate 4 explicit P0 tasks)
- `TODO/archive/` (7 files — all SUPERSEDED, historical, excluded)
- `TODO/04_history/` (24 files — all `_superseded_2026-09-26` history markers, excluded)

Filter for CODE tasks (not doc-only / not wet-lab-coordination / not process-rule):
- EXCLUDE: "sign-off", "audit", "report", "doc", "TEMPLATE", "DRAFT", "rule memo"
- INCLUDE: "implement", "build", "create", "fix", "wire", "develop", "verify M1+M2", "Phase 2 retrain", "redesign", "ultracode"

---

## §1 Headline counts

| Bucket | Count |
|---|---|
| n_pending_total | **18** |
| n_wetlab_lab_or_loi_or_sop | **6** (wetlab_ic50_analysis_pipeline + 3 SOPs + 2 LOIs, exclude per `TODO/pending/84_no_paper_task_without_authorization.md` gate) |
| n_process_rule | **1** (84 — paper rule memo, not a task) |
| n_completed_but_in_pending_folder | **1** (219 — Wf-219 Krasnov-aligned, marked ✅ P1 SHIP, needs move to `completed/`) |
| n_code_tasks_planned_not_done | **11** (177, 178, 179, 180, 181, 182, 183-routing + 234 + 3 from IMPROVEMENT_ENUM) |

---

## §2 Code tasks in `TODO/pending/` (8 distinct)

| ID | Title | Scope (LOC est) | Resource | Blocker |
|---|---|---|---|---|
| **TODO-177** | Wf-123 P3 Gap: Verify M1+M2 actually lifts downstream strengthen metrics | composite metric lift across 6 downstream consumers; ~6 weeks staggered | Blackwell + RTX 5090 dual | depends on G3/G4/G5/G6 (sub-tasks 179-182) |
| **TODO-178** | Wf-123 P3 Gap: metalcytotox 5-metal CSV on disk | add missing `pic50` + `cell_line` + `source` cols; FAIL-on-missing + audit JSON; 0.5-1 day | CPU-only | none — start NOW |
| **TODO-179** | Wf-123 P3 Gap: W₂ alignment on PlatinAI 4-cell probe | W₂ pIC50 alignment training; ~4-5 hr wall + 1 day overhead | RTX 5090 | > 1 hr GPU launch needs fresh user OK (per rule 84) |
| **TODO-180** | Wf-124 P3 Gap: 14/15 (metal, seed) pairs PENDING | 14 (metal, seed) × 30-epoch multi-task D-MPNN retrain; ~1.8 hr wall + 0.5 day overhead | RTX 5090 GPU-1 | > 1 hr GPU launch needs fresh user OK |
| **TODO-181** | Wf-124 P3 Gap: 50-epoch full sweep | rerun at 50 epochs (was 30); ~5 hr wall + 0.5 day overhead | RTX 5090 GPU-1 | > 1 hr GPU launch needs fresh user OK |
| **TODO-182** | Wf-124 P3 Gap: Per-innovation ablation | 4 configs × 5 metals × 3 seeds = 60 runs; ~7.5 hr wall + 1 day overhead | RTX 5090 GPU-1 | > 1 hr GPU launch needs fresh user OK |
| **TODO-183** | Wf-123 P3 + Wf-124 P3 Gap Routing Plan | PARENT planning doc, ranks 177-182; not a task itself | none | n/a — read-only routing |
| **TODO-234** | Wf-234 Frontier-Math Audit + Integration Plan | ultracode: wire 4 frontier math modules (W2.1 SE(3) SB trio + W2.3 multi-PH) into `molmetal/adapters/flow_matching_lipman/` main CFM pipeline; ~1-2 weeks | CPU + GPU | needs explicit user OK to launch ultracode (already planned) |

---

## §3 Code tasks from `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` (3 distinct P0)

These are referenced from the enum doc but their standalone `pending/13X_*.md` / `pending/138_*.md` files are MISSING from disk (only the enum doc carries them now).

| ID | Title | Scope | Resource | ETA | Blocker |
|---|---|---|---|---|---|
| **Wf-134 / TODO-134** | m15 Docking RMSD redesign | 3 phases: (1) ref SMILES extraction, (2) Vina redocking 5pocket × 3seed × n_poses=5 = 75 poses, (3) metric JSON + HONEST_NEGATIVE_REDESIGNED; ~280 LOC script + ≥12 tests | CPU + GPU-0 | 5-8 days | Wf-126 P1 5/5 pocket READY (no blocker) — start NOW |
| **Wf-135 / TODO-135 Phase 2** | m19 LOMO 5-metal × 5-seed × 50-epoch re-bench | 1250 runs; Rh fold AUC ≥ 0.30 recovery target; ≥16 tests | GPU-1 or CPU fallback | 2 weeks | DEPENDS on Wf-139 P7 Rh balanced cohort (DONE 2026-09-27 per IMPROVEMENT_ENUM §2 Step 1); > 1 hr GPU launch needs fresh user OK |
| **TODO-138** | Paper v2 + JCIM submission package | compile paper + cover letter + 8 lead claims + 2 partial + 8 honest-negative + figures + supp S1-S8 + submission_package.zip | CPU-only docs | 5-7 days | **BLOCKED** per rule `TODO/pending/84_no_paper_task_without_authorization.md` (paper task forbidden without user authorization) |

---

## §4 Per-task recommendation (sorted by lowest-hanging fruit)

| Rank | ID | Reason |
|---:|---|---|
| 1 | **TODO-178** | CPU-only, no auth gate, 0.5-1 day — DO NOW |
| 2 | **Wf-134 / TODO-134** | CPU + GPU-0 ≤ 1 hr segments possible (Phase 1 SMILES extract = CPU-only), 5-8 days — start Phase 1 NOW |
| 3 | **TODO-219** | Move to `completed/` (already SHIP ✅) |
| 4 | **TODO-179** | GPU 4-5 hr — request user OK |
| 5 | **TODO-180** | GPU 1.8 hr — request user OK (smallest GPU lift) |
| 6 | **TODO-181** | GPU 5 hr — request user OK |
| 7 | **TODO-182** | GPU 7.5 hr — request user OK (largest GPU lift) |
| 8 | **TODO-177** | Composite (depends on 179-182) — deferred |
| 9 | **Wf-135 / TODO-135 Phase 2** | GPU 2 weeks — request user OK + depends on Phase 1 DONE |
| 10 | **TODO-234** | ultracode wire-in — request user OK to launch ultracode |
| 11 | **TODO-138** | Paper — BLOCKED per rule 84 |

---

## §5 Immediate-action list (no auth gate, no GPU > 1 hr)

- [ ] **TODO-178** — Wf-123 P3 Gap: metalcytotox CSV hygiene fix (CPU, 0.5-1 day)
- [ ] **Wf-134 / TODO-134 Phase 1** — ref SMILES extraction from 5/5 pockets (CPU, 1-2 days)
- [ ] **TODO-219** — Move file to `TODO/completed/` (mechanical, 1 min)
- [ ] **TODO-183** — Re-read-only audit (already DONE; just close)

## §6 Pending user authorization (GPU > 1 hr OR ultracode launch)

- [ ] **TODO-179** W₂ PlatinAI 4-cell alignment — 4-5 hr GPU
- [ ] **TODO-180** 14/15 (metal, seed) PENDING — 1.8 hr GPU (smallest)
- [ ] **TODO-181** 50-epoch full sweep — 5 hr GPU
- [ ] **TODO-182** per-innovation ablation — 7.5 hr GPU
- [ ] **Wf-135 / TODO-135 Phase 2** LOMO 5×5×50 — 2 weeks GPU
- [ ] **TODO-234** ultracode frontier-math wire-in — 1-2 weeks CPU+GPU

## §7 Blocked by rule 84 (paper task)

- [ ] **TODO-138** — Paper v2 + JCIM submission package

---

## §8 Conclusion

- **n_pending_total**: 18
- **n_code_tasks_planned_not_done**: **11** (8 in `pending/` + 3 in `IMPROVEMENT_ENUM_2026_09_27.md`)
- **3 immediate-action tasks** (no auth gate): TODO-178, Wf-134 Phase 1, TODO-219 move
- **6 tasks need fresh user OK for GPU > 1 hr** (179, 180, 181, 182, Wf-135 Phase 2, 234)
- **1 task BLOCKED** by rule 84 (TODO-138 paper)
- **6 wet-lab templates** (3 SOPs + 2 LOIs + 1 pipeline) deferred per rule 84
- **1 process rule** (84) — read-only, not a task