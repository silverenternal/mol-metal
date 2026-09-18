# Phase 2 — Task Graph & Parallel-Safety Verification

**Date:** 2026-09-15
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Env:** uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64
**Reference:** phase1_scope.json (11 code-actionable items, 35h estimated, 3 disjoint agent lanes)

---

## 1. Scope (from Phase 1)

Phase 1 validated that 11 items across TODO-21..TODO-25 are code-actionable, totaling 35h.
Three natural agent lanes emerged:

- **A (Lambda metrics):** TODO-22.B-3/4/5/7 + TODO-23.W1 → all in `r4_lambda_only_run.py`
- **B (Triton wire-in):** TODO-24.K1/K2/K3 + TODO-25.P2.2 → model internals
- **C (PB pipeline):** TODO-23.C5 + TODO-25.P2.4/P2.5 → posebusters_adapter + r4_c_full_sweep

Phase 2 **decomposes B and C further** and **adds 4 new tasks** (D sub-pocket diversity,
F click-rule effect-size, G metal coord probe) to fully utilize 7 parallel agents. Task A
(combining TODO-22.B-* + TODO-23.W1) is replaced by the 7-task split below.

---

## 2. The 7 Parallel Tasks

| # | task_id | label | files_touched | est h |
|---|---------|-------|---------------|-------|
| 1 | A | F2(a) MetalLigandExchange click rule | `beta_reductions.py`, `pt_click_compat.py`, `test_lambda_mcts_singleton.py` | 6.0 |
| 2 | B | 8 anticancer metrics module (NEW) | `molmetal_lam/sbdd_env/metrics_v2.py` (NEW), tests | 8.0 |
| 3 | C | PB 30-cell production harness (NEW) | `molmetal/scripts/run_pb_production.py` (NEW), tests | 5.0 |
| 4 | D | Per-residue sub-pocket diversity (NEW) | `molmetal_lam/sbdd_env/per_residue_diversity.py` (NEW), tests | 4.0 |
| 5 | E | Triton fused_residual_add wire-in | `flow_matching_lipman/__init__.py`, `egnn_rocm.py`, tests | 2.5 |
| 6 | F | Click-rule effect-size study (5x5 ablation) | `molmetal/scripts/click_rule_effect_size_study.py` (NEW) | 3.0 |
| 7 | G | Metal coordination probe (5 d-block states) | `molmetal_lam/sbdd_env/metal_coord_probe.py` (NEW), tests | 2.5 |

**Total wall-clock (parallel ceiling):** 8.0h (Task B is the longest). **Total CPU-time:** 31.0h.

---

## 3. File-Ownership Matrix (no two tasks WRITE the same file)

```
WRITE:                                 READ-ONLY:
A: beta_reductions.py, pt_click_compat.py   F: pt_click_compat.py (API surface)
B: metrics_v2.py (NEW) + tests NEW          All tasks: r4_lambda_only_run.py
C: run_pb_production.py (NEW) + tests NEW   C: r4_c_full_sweep.py, posebusters_adapter
D: per_residue_diversity.py (NEW) + tests   B,D,F: r4_lambda_only_run.py (PHASE 4 owns)
E: flow_matching_lipman/__init__.py,        E: triton_kernels/fused_residual_add.py
   egnn_rocm.py, tests NEW                  A: r4_lambda_only_run.py
F: click_rule_effect_size_study.py (NEW)    F: beta_reductions.py, pt_click_compat.py
G: metal_coord_probe.py (NEW) + tests NEW   G: pt_click_compat.py, metal_generator_adapter.py
```

**Conflict count: 0.** No file is written by 2+ agents.

---

## 4. Phase-3 / Phase-4 Boundary

**Phase 3 (7 parallel agents):**
- Each agent WRITES its own NEW file (where applicable) + its own NEW test file
- Agents A and E are the only ones modifying pre-existing files; their targets are disjoint
  (A → `beta_reductions.py` + `pt_click_compat.py`; E → `flow_matching_lipman/__init__.py` + `egnn_rocm.py`)
- All 7 agents only **READ** `molmetal/scripts/r4_lambda_only_run.py` (2935 lines)

**Phase 4 (single integration agent, after all 7 ship):**
- The sole writer of `r4_lambda_only_run.py` for this round
- Wires: (a) 8 metrics from `metrics_v2.py` into Lambda result aggregate;
  (b) PB 30-cell harness into `r4_c_full_sweep.py` workflow;
  (c) `--div-weight` and `--coord-weight` CLI flags from new modules;
  (d) optional `--click-effect-size` study trigger

This boundary eliminates 7-way conflict on a single 2935-line file.

---

## 5. Serialization Constraints

**Only one** dependency: Task A's reaction engine touches files that F and G READ.
Recommended execution:

- **Wave 1 parallel (B, C, D, E, F, G)** — 6 agents start immediately
- **Task A stub first (10 min)** — A ships 5x5 compat matrix update to `pt_click_compat.py` so F can read stable API
- **Wave 1 continues; A finishes MetalLigandExchange rule in `beta_reductions.py`** — A re-ships; F and G are unaffected since they only consume stable API
- **Wave 2 — Phase 4 single integration agent**

Alternative: A runs serially first (6h), then F + G run (5.5h total, since they are
independent after A ships). Saves 10 min of stub coordination; loses 0.5h of
parallelism. Not recommended unless stub coordination proves brittle.

---

## 6. GPU / Honest-Framing Notes

- **All 7 tasks are CPU-only** (RDKit, Triton CPU fallback, paper arithmetic). No path-(a) gate.
- Task E's triton kernels have CPU fallbacks; GPU measurement is NOT claimed.
- Task F's effect-size study IS a measured experimental study (5x5x3 = 75 cells);
  honest reporting requires reporting all 75 cells, not pre-selecting.
- Task G's coord probe is a DESIGNED-FOR diagnostic, not a pre-measured lift claim.
- Task A's MetalLigandExchange effect on singleton-attractor count is a PROJECTED
  improvement (-30 to -70 pp), not pre-measured. Phase 4 verify step is REQUIRED.
- Task B's metric parity vs TargetDiff is PROJECTED from SOTA baselines, not measured
  on Mol-Metal pocket data.

---

## 7. Lit-anchored & Math-prior Coverage

Every task includes both:

- **lit_anchors**: 1+ published theorems/papers (page + theorem verifiable)
  e.g. Polykovskiy 2020 MOSES, Guan 2023 TargetDiff, Halgren 1996 MMFF94,
       Karczewski 2024 ICML spectral-norm bound, Himo 2005 JACS CuAAC,
       Hartwig 2010 Organotransition Metal Chem
- **math_prior**: explicit algebraic / optimization formulation with formula
  e.g. IntDiv1 = (1/|G|^2) sum (1 - tanimoto), JSD = 0.5*KL(P||M) + 0.5*KL(Q||M),
       trans-effect ordering rule, chi_squared vs ideal angle

No empirical claim is made without either (a) a measured pilot OR (b) a literature-
grounded projection. Both are explicit in `phase2_task_graph.json` `honest_framing_caveats`.

---

## 8. Exit Criteria for Phase 2

- [x] Task graph JSON written to `molmetal/reports/wf_parallel_tasks/phase2_task_graph.json`
- [x] This summary written to `molmetal/reports/wf_parallel_tasks/phase2_summary.md`
- [x] 7 tasks identified with disjoint file-write sets
- [x] Phase 3 / Phase 4 boundary drawn (Phase 4 owns `r4_lambda_only_run.py`)
- [x] Lit-anchors + math-prior recorded for every task
- [x] GPU-blocking status: all 7 tasks CPU-only
- [x] Honest-framing caveats explicit for every projected lift

**Ready for Phase 3 fan-out.**
