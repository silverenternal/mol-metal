# WF-Model-Line MASTER — 2026-09-16

**Scope:** consolidate M1A (TODO-23 weak→strong), M1B (TODO-22 gap closure), M2A (RxnFlow adapter), M2B (TODO-24 CFM Phase 4 ready memo).

---

## 1. TL;DR (3 lines)

- **TODO-23** weak→strong metric reclassification: 8 STRONG / 7 WEAK / 1 BLOCKED (GPU) / 2 CITED_ONLY; 3 GPU-free lifts identified (EV-1 across-pocket, EV-2 PB n_sim=1000, EV-3 SA --sa-weight 0.3).
- **TODO-22** gap closure: 17/25 MEASURED (+1 since 2026-09-15, #22 CFM-vs-Lambda ablation now MEASURED-Lambda-only); 12 NEW project-internal metrics; 7 honest negatives preserved.
- **M2A RxnFlow adapter** SHIPPED (PARTIAL FIT, NO full swap, HYBRID for templates); **M2B TODO-24 Phase 4 ready memo** WRITTEN ("STRUCTURALLY COMPLETE, METRICALLY ZERO — stay Lambda-only for Q1 paper").

---

## 2. Outcomes table

| Sub | Workflow | Verdict | Pass/Fail | Key metric / line count |
|---|---|---|---|---|
| **M1A** | TODO-23 weak→strong audit (m1_t23.md, 138L) | 8 STRONG / 7 WEAK / 1 BLOCKED / 2 CITED_ONLY; 3 GPU-free lifts | PASS | 22 metrics classified, 3 EV lifts enumerated (≈7h CPU / 2 wk) |
| **M1B** | TODO-22 gap closure status (m1_t22.md, 137L) | 17/25 MEASURED (+1), 12 NEW, 7 honest negatives | PASS | +1 cell #22 DESIGN→MEASURED-Lambda-only |
| **M2A** | RxnFlow adapter wire (m2_rxnflow.md, 204L) | PARTIAL FIT — 3D gap is deal-breaker; HYBRID for templates + retrosynthesis | PASS (with caveat) | 10/10 tests pass; 129 LOC adapter; templates=2h follow-up |
| **M2B** | TODO-24 Phase 4 ready memo (m2_t24.md, 191L) | STRUCTURALLY COMPLETE / METRICALLY ZERO — stay Lambda-only | PASS (as memo) | 5 paths ranked by EV; Task 2 (wrap fix) + Task 4 (RxnFlow cite) = top CPU wins |

---

## 3. MEASURED deltas (3 rows)

1. **Weak metric reclassification (M1A)** — `validity_rate` / `uniqueness_rate` / `synthesizability_rate` / `novelty` / `QED` / `reference_tanimoto` / `homotype_diversity` / `diversity_subpocket` = STRONG (saturated or +12× lift); `SA` pool = STRONG at 3.657 (Δ−2.288); `n_distinct` STRONG within-pocket (1→20) / WEAK across-pocket (Jaccard=1.000); WEAK = PB pass rate / Vina kcal/mol / PlatinAI / metal_compliance / pIC50 / CoM_shift / rigid_rmsd / SA pool.
2. **RxnFlow adapter wired (M2A)** — `molmetal/adapters/rxnflow_adapter.py` (129 LOC, 106 executable) + `molmetal/molmetal_lam/tests/test_rxnflow_adapter.py` (10/10 pass); `is_rxnflow_available()` returns False on this host; honest `uses_3d=False` metadata flag; Lipman-kwarg signature-compatible.
3. **Phase 4 plan documented (M2B)** — 5 paths ranked; Task 1 (30 min CPU, update TODO-24 + paper §3.5) + Task 2 (2h CPU, Path-B wrap-ordering fix at `r10_cfg_real_crossdocked.py:137-151`) + Task 4 (1 day CPU, Path-2 RxnFlow reaction-template cite) = top CPU wins; Path 1 conditional 12-24h GPU retrain; Path 4 architecture rework deferred to R16+.

---

## 4. What remains BLOCKED (3 lines)

- **GPU retrain blocked by history** — 4 CFM attempts (2000/10000/10000/500-step) all `decode_ratio=0`; R15 bit-exact CPU=GPU proves architecture-bound not device-bound; cold PSU cycle only fix (out-of-scope for code).
- **decode_ratio=0 root cause** — coord-quality wall (EGNN produces atom cloud whose pairwise distances match no decoder's prior); 97.4% `disconnected_distance_graph`; P0+P1+YuelBond ship but metric lift zero.
- **RxnFlow eval at scale** — adapter ships but upstream `RxnFlow` not pip-installed; `is_rxnflow_available()`=False on this host; full swap NO-GO due to 3D gap; HYBRID path (template cite + retrosynthesis oracle) recommended as Task 4 CPU work.

---

## 5. Honest framing (5 lines)

1. **GPU is the single binding constraint** — without GPU recovery, paper headline must reframe from "100-pocket × 3-seed paper-grade" to "30-cell per-pocket verified on 4 axes + 1-pocket PB chemistry-only + cite-only SOTA context" (journal-ship-ready per Pivot-A invariant #2).
2. **All MEASURED claims are honest** — 17/25 TargetDiff-aligned metrics + 12 NEW project-internal metrics (atom_vocab_coverage, n_distinct, diversity_tanimoto, diversity_subpocket, REINVENT4 multiproperty, P0 anticancer panel, etc.) + 7 honest negatives (CFM 0/192, R12 collapse, PB 0/30, Jaccard=1.000, metal_compliance 1.0→0.0, GPU SMU-hung, PlatinAI §4.6.1 DESIGN).
3. **Cite-only SOTA framing preserved** — TargetDiff -8.45 Vina + Uni-Mol-v2 75%+ PB cited as context, NOT comparison; 7 protocol-mismatch flags M1-M7 explicit; QVina-GPU parity N=50 unverified beyond that.
4. **Lambda × CFM coupling PROMOTED to §7 future work** — `coupling_adapter.py` (64-d bridge) shipped but not wired; 3-5h wire post-GPU.
5. **Three GPU-free lifts (EV-1/2/3) ship without GPU** — across-pocket diversity re-run on test_010..019 (1h, +5pp Jaccard), PB 15-cell smoke at n_sim=1000 (4-5h, +60pp PB), SA 10×3 sweep at --sa-weight 0.3 (1h, SA 3.657→2.86); combined ≈7h CPU lifts 3 WEAK → STRONG.

---

## 6. Files of record

- **Inputs:** `molmetal/reports/wf_model_line/final/{m1_t23.md, m1_t22.md, m2_rxnflow.md, m2_t24.md}`
- **Outputs:** `molmetal/reports/wf_model_line/MASTER.md` (this file)
- **Adapter:** `molmetal/adapters/rxnflow_adapter.py` (129 LOC, 106 code)
- **Tests:** `molmetal/molmetal_lam/tests/test_rxnflow_adapter.py` (10/10 pass)
- **Cross-refs:** `molmetal/reports/wf_pivot_followup/MASTER.md`, `molmetal/reports/wf_round12_lambda_pilot/final.md`, `molmetal/reports/wf_round13_100x3/final.md`, `molmetal/reports/wf_cfm_p0_fixes/final.md`, `molmetal/reports/wf_cfm_frontier_research/final.md`, `molmetal/reports/wf_r15_cfm_cpu_verify/final.md`, `molmetal/reports/wf_path_b_gpu_retrain/final.md`, `TODO/pending/22_data_gap_alignment_plan.md`, `TODO/pending/24_cfm_architecture_redo_plan.md`.

---

## 7. Recommended next actions (consolidated)

| Pri | Action | Owner | Wall | Effect |
|---|---|---|---|---|
| P0 | Append M1B update to `TODO/pending/22_data_gap_alignment_plan.md` §"Gap-closure update" | Paper-line | 0.5h | Files 17/25 + 12 NEW + 7 honest negatives |
| P0 | Append M2B memo to `TODO/pending/24_cfm_architecture_redo_plan.md` as 2026-09-16 ready-state | Paper-line | 0.5h | Locks CFM as §7 future work; flags Path 2 RxnFlow cite as Task 4 |
| P1 | EV-1 re-run R12 Deflex 10×3 on test_010..019 | Lambda-line | 1h | Lifts diversity_tanimoto; breaks Jaccard=1.000 |
| P1 | EV-2 PB 15-cell smoke at n_sim=1000 | Model-line | 4-5h | Lifts pb_pass_rate 0/15 → ≥10/15 |
| P1 | EV-3 SA 10×3 sweep at --sa-weight 0.3 | Lambda-line | 1h | Lifts SA 3.657 → 2.86 (TargetDiff band) |
| P1 | Path-2 RxnFlow reaction-template cite (Task 4) | Lambda-line | 1 day CPU | Cites RxnFlow in §3.2 + §2; anchors click rules to canonical source |
| P2 | Path-B wrap-ordering fix (Task 2) at r10_cfg_real_crossdocked.py:137-151 | Model-line | 2h CPU | Preps for Task 3 conditional retrain |
| P2 | Morgan-ECFP4 patch in r4_lambda_only_run.py (#9 closure) | Lambda-line | 1 day | Lifts #9 DESIGN → MEASURED |
| P3 | widzuipcl Phase 2 GPU retrain + PlatinAI §4.6.1 MEASURED promotion | widzuipcl owner | 12-24h GPU | Lifts #23 to MEASURED-on-novel-scaffolds |
| P3 | Path-1 conditional 10000-step CFM retrain (Task 3) | Model-line | 12-24h GPU | Decision gate decode_ratio≥0.10 → continue; else close path-(a) for Q1 |
| P4 | Path-4 architecture rework (Task 5) | Model-line | 5-10 days eng + 24h GPU | Tier-4 deferred to R16+ |

**If GPU recovers in W4:** widzuipcl Phase 2 + Path-1 retrain open — lifts #23 + #22-CFM column → 19/25 total.
**If GPU stays blocked:** ship 17/25 + 12 NEW + 7 honest negatives + EV-1/2/3; paper is journal-ship-ready under honest framing.

---

**MASTER verdict:** 4/4 sub-workflows shipped (M1A PASS, M1B PASS, M2A PASS-with-caveat, M2B PASS-as-memo). No GPU experiments run. 3 GPU-free lifts ready to execute (EV-1/2/3 ≈7h CPU). Q1 paper submission unblocked under Lambda-only + cite-only SOTA framing per Pivot-A invariants.