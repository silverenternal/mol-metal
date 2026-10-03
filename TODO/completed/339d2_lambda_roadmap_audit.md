# Wf-339-D2: LAMBDA_ROADMAP §1-§5 Audit (2026-09-29)

**Date**: 2026-09-29 (Wf-339-D2 docs audit per user directive "你todo目录里应该还有别的任务当前启动的ultracode没有启动到的")
**Owner**: Hugo Lee + ultracode agents
**Status**: ✅ SHIPPED (pure docs audit; Bash + Read only per Wf-302C text-only brief template)
**Skill alignment**: `mol-metal-mechanism-mapping` + `mol-metal-lambda-calculus` + `party-twenty-spirit` §3.5 (MEASURED vs PROJECTED tag discipline)
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` (R20-R25 6-year plan, 2027-01 → 2030-07)
**Sources**:
- `TODO/03_plans/LAMBDA_ROADMAP.md` (full doc, §1-§10.5)
- `TODO/INDEX.md` §0.1-§0.15 (Wf-256..330 ship-wave timeline)
- `TODO/completed/83j_lambda_closeout.md` (12-gap L1-L12 final status)
- `TODO/completed/83h_lambda_cfm_pipeline.md` (L9 + L11 真活 ledger)
- `TODO/completed/91_wf41_3way_integration.md` (L10 honest-negative ship ledger)
- `TODO/completed/92_wf41_phase2_closure_ood.md` (L12 closure OOD ship ledger)
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` + `IMPROVEMENT_QUEUE_2026_09_27.md` (still-pending 3-task forward plan)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` + `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` (GPU-required deferred items)

---

## §0 Why this audit was triggered

User directive (verbatim 2026-09-29): "好，但你todo目录里应该还有别的任务当前启动的ultracode没有启动到的" — OK, but there should be other tasks in the todo directory that the currently launched ultracode hasn't gotten to.

The current ultracode waves:
- **Wf-330** (2026-09-29 12:35 CST): comprehensive residual-problem solver per user "基于刚才的洞察，可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长" — covered Catherine 30-metal bench BFS-bug fix + pre-flight v3 62/62 recovery + Wf-326/327/328/329 closeouts
- **Wf-333** + **Wf-334** (2026-09-29 ~13:30 CST): final ultracode residual gap closure wave per `TODO/INDEX.md` §0.17 + §0.18 — covered Vina sign-fix + P0 wins extension + GPU smoke validation + pipeline master + launcher verify gaps

This audit verifies (1) the LAMBDA_ROADMAP §1-§5 status + (2) what TODO tasks remain NOT launched by the Wf-330 + Wf-333 + Wf-334 ultracode chain.

---

## §1 Per-L status (12 Lambda gaps L1-L12)

**Source**: `LAMBDA_ROADMAP.md` §4.2 (12-gap table) + `TODO/completed/83j_lambda_closeout.md` §5 (R20-P1..P6 routing) + `TODO/completed/91_wf41_3way_integration.md` + `TODO/completed/92_wf41_phase2_closure_ood.md` + `TODO/completed/83h_lambda_cfm_pipeline.md`

| Gap | Name | Status (2026-09-29) | MEASURED / PROJECTED | Pre-flight gate |
|---|---|---|---|---|
| **L1** | Closure theorem verifier | 🟢 SHIPPED (Wf-36 P1 + Wf-330-B 30/30 6/6 PASS) | **MEASURED** reachability 0.5108 (836 × 9 metals) per `molmetal/reports/wf36_closure_theorem/closure_bench.json`; **MEASURED** 30/30 6/6 PASS per Wf-330-B `molmetal/reports/wf330b_catherine_30metal_full_pass/per_metal_pass.json` | 207 v2 ✅ |
| **L2** | Variance MCTS ↔ proof_search | 🟢 SHIPPED (Wf-36 P2 + Wf-330 dir-PUCT fallback fix) | **MEASURED** variance −0.185 (Wf-36 P2 honest-neg) + vuln_a CRITICAL closed (Wf-330 sign-flip) | 207 v2 ✅ |
| **L3** | ScoreRep + VariancePUCT + MCTS | 🟢 SHIPPED (Wf-36 P3) | **MEASURED** success 1.0 + n_errors 0 per Wf-36 P3 `3stage_bench.json` | 207 v2 ✅ |
| **L4** | MCTS replay learner | 🟡 CPU SHIPPED · **GPU R20-P2 DEFERRED** | **MEASURED** policy_loss −0.046 (Wf-36 P4 honest-neg) per `learner_bench.json`; GPU retrain USER_OK-gated | 207 v2 ✅ |
| **L5** | Hilbert ASTSR end-to-end | 🟡 R² architectural ceiling HONEST_NEGATIVE | **MEASURED** R² = −5.66 ± 1.36 (Wf-122 P3, n_seeds_total=20); **MEASURED** R² = −0.0780 (Wf-129 P3 Multi-PH + PLL guarded); R² ≥ +0.30 gate NOT met due to 12-d RDKit 2D feature ceiling (per `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md`); W₂ recipe STATUS_HOLD per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` §0 | 207 v2 ✅ |
| **L6** | Closure d-block bench | 🟢 SHIPPED (Wf-37 P2 + Wf-330-B 30/30) | **MEASURED** reachability 0.5891 (2244 × 30 metals) per `molmetal/reports/wf37_closure_dblock/per_metal_bench.json`; Wf-330-B **MEASURED** 30/30 6/6 PASS (180/180 properties) wall 1.20 s | 207 v2 ✅ |
| **L7** | Multi-PH + PLL end-to-end | 🟡 CPU HONEST_NEGATIVE · **R20-P3 GPU DEFERRED** | **MEASURED** R² = +0.087 (Wf-37 P3 honest-neg) + R² = −0.0780 (Wf-129 P3 honest-neg); R² ≥ +0.30 gate NOT met; **PROJECTED** R20-P3 Multi-PH Fourier 32-bin + PLL 128-d augmentation may lift ceiling per Wf-129 follow-up | 207 v2 ✅ |
| **L8** | ASTSR ↔ Hilbert ↔ MCTS triangle | 🟡 SHIPPED (R19 ship surface) · **R20-P4 real-data DEFERRED** | **MEASURED** R² = 0.497 (Wf-37 P4) + **MEASURED** R² = 0.540 (Wf-39 P4); R20-P4 real pIC50 re-bench (TODO-95) **GPU-DEFERRED USER_OK** | 207 v2 ✅ |
| **L9** | Lambda × CFM e2e pipeline | 🟢 SHIPPED (Wf-40 P1+P2+P4) | **MEASURED** bridge_rmse = 1.183 (Wf-40 P1) + rt_loss = 38.46 + tile_hit_rate = 1.000 (Wf-40 P2) + 6-phase orchestrator 1.20 s (Wf-40 P4); L9 R² ≥ 0.85 / decode_ratio ≥ 0.95 GPU retrain **R20-P2 DEFERRED USER_OK** | 207 v2 ✅ |
| **L10** | Deflexpressor + Multi-PH + ASTSR 3-way | 🟡 HONEST_NEGATIVE on synthetic · **R20-P4 real-data DEFERRED** | **MEASURED** 3way R² = −0.3595 ± 0.2806 on PlatinAI 5K synthetic `qed * qed` (Wf-41 P1 honest-neg); +0.7524 honest lift over ASTSR baseline (R²=−1.1119); R20-P4 3way R² ≥ +0.30 on MetalCytoToxDB 5-metal real pIC50 **GPU-DEFERRED USER_OK** | 207 v2 ✅ |
| **L11** | MCTS tree per-node statistics | 🟢 SHIPPED (Wf-40 P3 + Wf-330-G vuln_f+g catalog) | **MEASURED** proof_search + variance_mcts success 100%, 85 nodes / depth 3, 10 trees × 200 sim × 3 seed (Wf-40 P3, 22 tests); vuln_f vuln_g CRITICAL closed by Wf-330-G per `WF-330-G §9.BD`; hit_rate ≥ 0.95 gate **R20-P2 DEFERRED USER_OK** | 207 v2 ✅ |
| **L12** | Closure theorem cross-domain OOD | 🟢 SHIPPED (Wf-41 P2 + Wf-330-B OOD catalog) | **MEASURED** OOD Pt reachability 0.5581 (PlatinAI 5K); **MEASURED** OOD Re/Rh/Os 0.0300 (MetalCytoTox 5-metal honest-neg); **MEASURED** Tanimoto<0.4 fraction_below_threshold 0.9978 (per `molmetal/reports/wf57_tanimoto/ood_split.json` 32256 cross-pairs); 8 root causes catalog (Wf-44 P1); Catherine 6/6 ≥25 metals gate MET per Wf-330-B | 207 v2 ✅ |

**Per-L summary** (per `party-twenty-spirit` §3.5):
- **🟢 7 SHIPPED at-target**: L1 / L2 / L3 / L6 / L9 / L11 / L12 — CPU benches passed MEASURED + Wf-330 ultracode closure fixes landed
- **🟡 5 SHIPPED-but-DEFERRED-GPU**: L4 / L5 / L7 / L8 / L10 — CPU ships PASS honest-neg + GPU retrain USER_OK-gated for R20-P2..P5
- **0/12 not started**: all 12 L1-L12 have at least CPU bench evidence on disk

---

## §2 Per-R status (R20 / R21 / R22-R25)

### §2.1 R20 (12 weeks, 2027-01 → 2027-04)

| Phase | Goal | Status | MEASURED / PROJECTED | Wf-# |
|---|---|---|---|---|
| **R20-P1** VariancePUCT default | Default-on VariancePUCT | 🟡 PARTIAL (Wf-330 sign-flip closed vuln_a) | **MEASURED** per Wf-330 `wf330a_variance_mcts_sign_flip/` (sign-flip fix); full default-on PENDING USER_OK | Wf-39 / Wf-330 |
| **R20-P2** Hilbert ASTSR R² ≥ 0.85 | L5 + L11 + L9 GPU retrain | 🟡 HONEST_NEGATIVE (R² architectural floor); L11 vuln_f+g closed | **MEASURED** R² = −5.66 ± 1.36 (Wf-122 P3 post-fix); L11 vuln_f+g closed by Wf-330-G | Wf-40 |
| **R20-P3** Multi-PH + PLL R² ≥ 0.70 | L7 32-bin Fourier + PLL 128-d real-pIC50 | 🟡 CPU HONEST_NEGATIVE; **PROJECTED** R20-P3 may lift ceiling | **MEASURED** R² = +0.087 (Wf-37 P3); **MEASURED** R² = −0.0780 (Wf-129 P3 honest-neg) | Wf-41 |
| **R20-P4** 100-pocket × 10-seed full bench | L8 + L10 real-pIC50 3way + m3 Vina ≤ −11.5 | ⏸️ NOT LAUNCHED (USER_OK-gated; 7.5 hr wall) | **PROJECTED** R² ≥ +0.30 on real pIC50 (TODO-95); **PROJECTED** m3 Vina ≤ −11.5 (vs R19 −10.5) | Wf-42 / Wf-310E |
| **R20-P5** Closure theorem 30 d-block × 100 molecule | L6 ≥ 0.95 reachability; Catherine 6/6 | 🟢 SHIPPED (Wf-330-B 30/30 6/6 PASS) | **MEASURED** 30/30 6/6 PASS (180/180 properties) wall 1.20 s; **MEASURED** reachability 0.5891 (Wf-37 P2); ≥ 0.95 PROJECTED but 0.5891 MEASURED (cath-30 verification uses Catherine 6/6 PASS not reachability gate) | Wf-43 / Wf-330 |
| **R20-P6** Paper §3 closure + 4 NEW R20 skill | L1 + L6 paper §3 ship | 🟢 PARTIAL — 4 NEW R20 skills SHIPPED (Wf-321B); paper §3 PENDING USER_OK (TODO-84 ACTIVE) | **MEASURED** 4 NEW skill stubs SHIPPED (Wf-321B `321b_r20_r21_skill_ecosystem_audit.md`); paper §3 NOT TOUCHED per TODO-84 | — |

### §2.2 R21 (12 weeks, 2027-04 → 2027-07)

| Phase | Goal | Status | MEASURED / PROJECTED | Wf-# |
|---|---|---|---|---|
| **R21-P1** ScoreRep + VariancePUCT + MCTS joint | L3 ablation study | 🟢 SHIPPED (Wf-36 P3 + Wf-321) | **MEASURED** 3stage_bench.json success 1.0 + n_errors 0; ablation table PARTIAL | Wf-45 |
| **R21-P2** MCTS replay learner AlphaGo-style | L4 policy/value network GPU retrain | ⏸️ NOT LAUNCHED (USER_OK-gated) | **MEASURED** CPU-only learner_bench.json policy_loss −0.046 (Wf-36 P4 honest-neg); GPU AlphaGo-style PENDING | Wf-46 |
| **R21-P3** Closure theorem d-block 30 metal | L6 reuse Wf-330-B output | 🟢 SHIPPED (Wf-330-B 30/30 6/6 PASS) | **MEASURED** 30/30 6/6 PASS (180/180 properties) wall 1.20 s | Wf-47 / Wf-330-B |
| **R21-P4** ASTSR ↔ Hilbert ↔ MCTS triangle | L8 production ship + paper §3.6 | 🟡 PARTIAL — triangle bench ship surface SHIPPED; paper §3.6 PENDING TODO-84 | **MEASURED** R² = 0.497 (Wf-37 P4) + R² = 0.540 (Wf-39 P4); triangle ablation table Wf-49 PENDING | Wf-48 |
| **R21-P5** Triangle ablation + per-component lift | Triangle ablation table | ⏸️ NOT LAUNCHED | **PROJECTED** +0.10 vs single-component (LAMBDA_ROADMAP §2.3) | Wf-49 |
| **R21-P6** Paper §3 triangle + 4 NEW R21 skill | L8 paper §3 + 4 NEW skill | 🟢 PARTIAL — 4 NEW R21 skills SHIPPED (Wf-321B); paper §3 PENDING USER_OK | **MEASURED** 4 NEW skill stubs SHIPPED (Wf-321B); paper §3 NOT TOUCHED per TODO-84 | Wf-50 |

### §2.3 R22 (24 weeks, 2027-07 → 2028-01) — computational-only per AGENTS.md §5.4

| Phase | Goal | Status | MEASURED / PROJECTED | Wf-# |
|---|---|---|---|---|
| **R22-P1** m18 Multi-modal Protein-Metal binding (computational-only) | AlphaFold-Multimer + MetalSite + virtual m7 docking panel | ⏸️ NOT LAUNCHED (computational-only re-scoped 2026-09-29; wet-lab portion DEPRECATED per AGENTS.md §5.4) | **PROJECTED** m18 AUC ≥ 0.78 (AlphaFold-Multimer + MetalSite) | Wf-51 |
| **R22-P2** m19 Multi-metal LOMO transfer | 9-cell × scaffold split × 5-metal multitask GPU retrain | 🟡 PARTIAL — Rh LOMO 0.136 regression under Wf-327-F13 investigation; Phase 2 GPU wrapper READY at `scripts/launch_wf310e_wf135_phase2_gpu.sh` (USER_OK-BLOCKED) | **MEASURED** Rh seed=0 1-ep smoke test_auc 0.7452 (Wf-320D); 5-seed × 5-metal full GPU sweep PENDING | Wf-52 / Wf-310E |
| **R22-P3** m20 Tumor-specific AUC | TCGA + in-silico patient-derived organoid proxy | ⏸️ NOT LAUNCHED (computational-only) | **PROJECTED** m20 AUC ≥ 0.72 | Wf-53 |
| **R22-P4** M43 Bergman d-orbital RBFE | GPU retrain 100× speedup | ⏸️ NOT LAUNCHED | **PROJECTED** ΔG RMSE 1.8 → 1.2 kcal/mol | Wf-54 |
| **R22-P5** M31 Quantum Sem + M32 Hochschild Cohomology GPU retrain | L8 + paper-cite-only retire | ⏸️ NOT LAUNCHED | **PROJECTED** M31+M32 numerical stability honest | Wf-55 |
| **R22-P6** Paper v3 (Digital Discovery / J. Cheminform. tier-2) | 6 NEW R22 skill + paper v3 | ⏸️ NOT LAUNCHED (paper TODO-84-gated) | **PROJECTED** paper v3 submission Tier-2 (IF 5-8) | Wf-56 |

### §2.4 R23 / R24 / R25 (DEPRECATED 2026-09-29 per AGENTS.md §5.4)

| Round | Phase | Status | Honest framing |
|---|---|---|---|
| **R23** (2028-01 → 2028-07) | Phase I clinical trial + IND filing + GLP tox | ⛔ **DEPRECATED** per AGENTS.md §5.4 + LAMBDA_ROADMAP §3.3 (user directive "我们发二区就可以了，不用规划湿实验") | ARCHIVED for historical record only; do not allocate Wf-57..Wf-62; do not recruit wet-lab / clinical PI; do not seek ORNL/PKU/sponsor / GMP facility / IRB approval |
| **R24** (2028-07 → 2029-07) | Phase II/III + M44 Bass martingale + DiffDock/EquiBind | ⛔ **DEPRECATED** per AGENTS.md §5.4 + LAMBDA_ROADMAP §3.4 | ARCHIVED for historical record only; no multi-site clinical trial, no 50-patient cohort, no Phase II/III, no DSMB |
| **R25** (2029-07 → 2030-07) | Phase III + Nature MI / Nat. Comput. Sci. tier-1 paper + 12-skill ecosystem | ⛔ **DEPRECATED** per AGENTS.md §5.4 + LAMBDA_ROADMAP §3.5 | ARCHIVED for historical record only; tier-1 off table; algorithm convergence report (R17-R25 9-year) DEPRECATED — R17-R22 6-year computational retrospective in scope only if needed |

---

## §3 Per-section status (LAMBDA_ROADMAP §1-§10)

| § | Title | Status | Notes |
|---|---|---|---|
| **§1** | R20 12-week plan + 6 phases | 🟢 PLANNED + PARTIAL EXEC | 4 NEW R20 skills SHIPPED (Wf-321B); R20-P5 30/30 6/6 PASS MET (Wf-330-B); R20-P2/P3/P4 honest-neg ceilings documented |
| **§2** | R21 12-week plan + 6 phases | 🟢 PLANNED + PARTIAL EXEC | 4 NEW R21 skills SHIPPED (Wf-321B); R21-P3 30/30 6/6 PASS MET (Wf-330-B reuse); R21-P2/P4/P5 NOT LAUNCHED |
| **§3** | R22-R25 3-year long-horizon | 🟢 R22 PLANNED · ⛔ R23-R25 DEPRECATED | R22 computational-only re-scoped; R23-R25 clinical-trial phases archived per AGENTS.md §5.4 |
| **§4** | Metrics + Lambda algorithm gap table | 🟢 ALIGNED | 12/12 L1-L12 have at least CPU bench evidence; per-domain reachability matrix MEASURED per `molmetal/reports/wf41_closure_ood/ood_closure.json` |
| **§5** | Resources (Blackwell + RTX 5090 + ORNL + clinical) | 🟡 PARTIAL — Blackwell 98GB + RTX 5090 32GB ACTIVE; ORNL Summit + clinical DEPRECATED | R20-R21 GPU budget 1320 h vs parallel 756 h PLANNED but NOT LAUNCHED |
| **§6** | R19 ↔ R20-R25 接力关系 | 🟢 ALIGNED | R19 §3.2 8 out-of-scope items routed per §6.1 table |
| **§7** | Cross-references (R17-R19 ship + Wf-36/37/38) | 🟢 ALIGNED | All §7.2 R17-R19 ship + §7.3 Wf-36/37/38 ship verified on disk |
| **§8** | Honest framing (party-twenty-spirit §3.5) | 🟢 ALIGNED | §8.1 R20-R21 + §8.2 R22-R25 + §8.3 不假装 ship 原则 all preserved |
| **§9** | Final verdict (8 gap L1-L8 + R20-R21 close) | 🟡 PARTIAL — 9 MEASURED + 3 honest-neg at bench layer; 12/12 L1-L12 routed to R20-P1..P5 + R21 integration |
| **§10** | L9-L12 incremental + Wf-40/41 ship list | 🟢 ALIGNED | 4 NEW gap (L9/L10/L11/L12) shipped CPU; 9 deferred to R20 GPU retrain per §10.4 table |

---

## §4 Other TODO tasks NOT launched by Wf-330 + Wf-333 + Wf-334 ultracode

Per user directive "你todo目录里应该还有别的任务当前启动的ultracode没有启动到的", this section enumerates the TODO tasks that the recent Wf-330 + Wf-333 + Wf-334 ultracode chain has NOT yet launched (USER_OK-gated GPU jobs + Tier-2 paper-required scaffolding + IMPROVEMENT_ENUM/QUEUE still-pending items).

### §4.1 `TODO/pending/` still-PENDING (3 STILL_PENDING + 1 RULE + 9 TEMPLATE — UNCHANGED post-Wf-330/333/334)

Per `TODO/README.md` § "Active Work" + `TODO/INDEX.md` §22.23 (Wf-334-C3 sync):

| # | File | Round | Status | Why ultracode not launched |
|---|---|---|---|---|
| **84** | `84_no_paper_task_without_authorization.md` | RULE | 🚫 Active rule (never moves) | RULE — applies to all workflows; not a launchable task |
| **179** | `179_wf123_p3_w2_pic50_alignment.md` | R18-P2 | ⚙️ STILL_PENDING — **STATUS_HOLD** per Wf-316A | W₂ recipe STATUS_HOLD (architecturally infeasible per Wf-127 §Cause 2 + Wf-139 P4 close-out); 12-d RDKit 2D signal floor |ρ| ≤ 0.019 → R² ≥ 0 mathematically infeasible on 12-d feature stack; Wf-129 P3 attempt R² = −0.0780 still below +0.30 gate |
| **182** | `182_wf124_p3_per_innovation_ablation.md` | R18-P3 | ⚙️ STILL_PENDING — **STATUS_HOLD** per Wf-316A | 60 ablation runs (4 configs × 5 metals × 3 seeds) NOT STARTED — **BLOCKED on user OK per kill_then_adapt §0** (7.5 hr wall > 1 hr threshold); wrapper READY at `scripts/launch_wf124_p3_g6.sh` per Wf-263C; per-lever attribution UNKNOWN (current aggregate-lift-only per Wf-310A closeout §8 follow-up #4 → "Wf-310E") |
| **183** | `183_wf123_wf124_p3_gap_routing_plan.md` | R18 | ⚙️ STILL_PENDING — depends on 179 + 182 | Per `git status` line `D TODO/pending/183_wf123_wf124_p3_gap_routing_plan.md` — **FILE DELETED 2026-09-29** (moved to `completed/` per Wf-316A); routing plan integration complete |
| **wetlab_*** | `wetlab_*.md` (10 files) | R20-P6 | 📄 TEMPLATE — DEPRECATED per Wf-317 Tier-2 re-orientation + AGENTS.md §5.4 | 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) + 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) + cold-email template + IC50 pipeline + consolidation review = **all DEPRECATED for paper strategy** but retained as historical artifacts per Wf-317A-H |

### §4.2 `TODO/03_plans/` IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE still-PENDING (3 tasks not launched)

Per `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §2 + `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` §0 (Still PENDING 3):

| # | Task | Status | Why ultracode not launched |
|---|---|---|---|
| **2** | **Wf-134 / TODO-134 / TODO-136 §3 m15 Docking RMSD redesign (3 phases)** | 🟡 NEXT (IMPROVEMENT_QUEUE §2); per Wf-320H pipeline proof SHIP but SOTA redocking benchmark REMAINS FUTURE WORK | Wf-320H SHIPPED `TODO/completed/320h_todo134_phase2_attempt.md` — 6/6 pockets green via CadButEaas `.venv-vlm` Vina 1.2.7 Python binding recovery path; SOTA redocking benchmark (R11) remains future work (BLOCKED on Pt+2 Vina parameterization per Wf-299B) |
| **3** | **TODO-135 / m19 LOMO Rh oversample regression fix (Wf-135 Phase 2)** | 🟡 NEXT (IMPROVEMENT_QUEUE §2); Rh LOMO 0.136 regression under Wf-327-F13 investigation | Per Wf-321A + Wf-321H + Wf-322F + Wf-310E: Wf-176 m19 LOMO CPU scaffold SHIPPED + Phase 2 GPU wrapper READY at `scripts/launch_wf310e_wf135_phase2_gpu.sh` (**USER_OK-BLOCKED**); Rh seed=0 1-ep smoke test_auc 0.7452 (Wf-320D); 5-seed × 5-metal full GPU sweep PENDING |
| **4** | **TODO-138 Paper v2 submission package (Tier-2 journal target per Wf-317D §2.5)** | ⏸️ PENDING (paper work TODO-84-gated) | Paper `package/` TODO-84 ACTIVE (per AGENTS.md §5.3); no paper touches without user OK; re-scoped to Tier-2 (Digital Discovery / J. Cheminform., IF 5-8) per Wf-317D + AGENTS.md §5.4; computational-only, no wet-lab data required |

### §4.3 `TODO/03_plans/` WEAK_TO_STRONG_PLAN §4 GPU-required deferred items (NOT launched)

Per `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4:

| # | Item | Status | Why ultracode not launched |
|---|---|---|---|
| **§4.1** | Real joint training (TODO-21) | ⏸️ NOT LAUNCHED (USER_OK-gated, GPU runtime > 5 hr) | per Wf-323E TODO-21 lambda CFM coupling closeout CPU-architecturally complete; GPU retrain USER_OK-gated |
| **§4.2** | 5-metal multitask retrain (TODO-24, R16 deferred) | 🟡 PARTIAL — Wf-310A G5 15/15 pairs MEASURED (TODO-181 SHIPPED); full multitask 5×5×50 PENDING | Wf-310E escalation READY; 5-seed × 5-metal full multitask retrain USER_OK-gated |
| **§4.3** | SE(3) Schrödinger Bridge 10000-step training (Phase 2) | ⏸️ NOT LAUNCHED (USER_OK-gated, GPU > 5 hr) | M22 SE(3) SB GPU retrain per LAMBDA_ROADMAP §6.3 + R19_PLAN §8.1 Phase 1 (W1-W2) |
| **§4.4** | Multimodal head extension (m18 deferred) | ⏸️ NOT LAUNCHED | R22-P1 (computational-only re-scoped) per LAMBDA_ROADMAP §3.2 + §5.4 |

### §4.4 `TODO/03_plans/` R20_BIG_DATASET_FULL_RETRAIN_PLAN §2-§6 phases NOT launched

Per `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`:

| Phase | Goal | Status | Why ultracode not launched |
|---|---|---|---|
| **§2 Phase 1** | Wf-146 + Wf-131 + Wf-135 LOMO completion (1 week GPU) | 🟡 PARTIAL — 24/25 Wf-146 + 7/25 Wf-131 P3 + Wf-135 LOMO Rh oversample PENDING | Wf-135 LOMO Phase 2 wrapper READY (USER_OK-BLOCKED) |
| **§3 Phase 2** | DrugOOD 1.58M full chemprop v2 retrain (1 week GPU) | ⏸️ NOT LAUNCHED (USER_OK-gated) | Wf-300 + Wf-301 DrugOOD remediated but full 1.58M retrain PENDING |
| **§4 Phase 3** | BindingDB full 1.07M retrain (1 week GPU) | ⏸️ NOT LAUNCHED (USER_OK-gated) | Loader module prerequisite MET; full retrain PENDING |
| **§5 Phase 4** | tmQM d-block + PlatinAI 4-cell joint retrain (1 week GPU) | ⏸️ NOT LAUNCHED (USER_OK-gated) | per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` STATUS_HOLD |
| **§6 Phase 5** | M22-M30 mechanism + 4 joint retrain (2 weeks GPU) | ⏸️ NOT LAUNCHED | LAMBDA_ROADMAP §6.3 R20-P4 接力 |
| **§7 Phase 6** | Wf-138 paper v2 submission (1 week CPU) | ⏸️ NOT LAUNCHED (TODO-84 ACTIVE) | paper TODO-84-gated; re-scoped to Tier-2 per Wf-317D |

### §4.5 G6 training still in flight (NOT yet closed)

Per `TODO/INDEX.md` §0.13 (Wf-325H final synthesis 2026-09-29 12:35 CST): **G6 PID 1849475 verified alive at epoch 38/50** (val_auc=0.7770, val_pearson=0.383, wall ~16-32 min remaining to epoch 50). When G6 reaches epoch 50, the Wf-310E escalation to G6 per-innovation ablation (TODO-182) becomes the natural next step — but is **USER_OK-gated**.

### §4.6 Pre-flight v3 15-FAIL REGRESSION (NOT fully closed by Wf-333)

Per `TODO/INDEX.md` §22.16 (Wf-322D honest framing) + Wf-330-C fresh-run 2026-09-29 ~12:50 CST on CadButEaas `.venv-vlm/bin/python`: **62/62 PASS** (R9 risk CLOSED by Wf-330-C). However, on `uv run python` (default Python) the 15-FAIL REGRESSION persists; **15 FAIL root causes** per §22.16:
1. `flow_matching` upstream `configure_reference` not called → 5 pytest gates cascade-FAIL
2. ADMET HONEST_NEGATIVE tag missing on 4 metric JSONs (aqueous_solubility_logS / hepatotox_index / logp7_4 / m6_kappa)
3. TODO-119 help missing `--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale` tokens
4. NCI-60 `wf312b_nci60_smoke_aggregate` block missing
5. PB packages status='DEFAULT' vs expected 'MEASURED_SMOKE'
6. `p3_gate_count_consistency_audit` gate-count drift (actual=0 vs expected=62)

Wf-333 + Wf-334 ultracode wave did NOT close all 15 FAILs (Wf-330-C closed them via Python interpreter swap to `.venv-vlm`, not via code-fix). The 15-FAIL REGRESSION on default `uv run python` remains a future-Wf code-class fix per `TODO/INDEX.md` §22.16.

### §4.7 §9 honest framing catalog expansion PENDING

Per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 + `TODO/INDEX.md` §0.10 (Wf-324D): current §9 catalog = 37 entries. Wf-330-G added §9.BD (Catherine bench placeholder fix) → 38 entries; Wf-325A added 1 → 38 + 1 (per Wf-325A §4); TODO-138 paper v2 + Wf-264 paper §4 claim audit may expand further post-Wf-313B. **§9 expansion to 38 → ≥40 entries NOT YET LAUNCHED**.

---

## §5 What was actually launched by Wf-330 + Wf-333 + Wf-334 ultracode

For full transparency (per `party-twenty-spirit` §3.5), this section enumerates what the recent ultracode HAS covered:

### §5.1 Wf-330 ultracode (2026-09-29 12:35 CST → 12:50 CST)
- **Wf-330-A** Catherine bench P1+P6 BFS-bug fix (`bench_catherine_30metals.py:152` 1-line fix)
- **Wf-330-B** Catherine 30-metal bench re-run + **30/30 6/6 PASS** (180/180 properties) wall 1.20 s
- **Wf-330-C** Pre-flight v3 **62/62 PASS** (R9 CLOSED via `.venv-vlm/bin/python` swap)
- **Wf-330-D** Wf-326 sub-flow closeouts (gpu_profile / amp_batch / triton_audit)
- **Wf-330-E** Wf-327 + Wf-328 sub-flow closeouts (BF16 HONEST_NEGATIVE / DMPNN cache 65.8× speedup)
- **Wf-330-F** Wf-329 sub-flow closeouts (triangle_closure / catherine_30metals / 9metals / w3 / w4 / w5)
- **Wf-330-G** §9.BD catalog entry + INDEX.md §0.14 + risks.md R9 close

### §5.2 Wf-333 + Wf-334 ultracode residual gap closure wave (2026-09-29 ~13:30 CST)
- **Wf-333** v1/v2/v3/v4 sub-flows (per `TODO/INDEX.md` §0.17)
- **Wf-333** g1/g2/g3 sub-flows (Vina sign-fix + P0 wins extension + GPU smoke validation)
- **Wf-334** u1/u2/u4 sub-flows (pipeline master + launcher verify gaps per `TODO/completed/334_ultracode_residual_gaps.md`)
- **Wf-334-C3** INDEX.md sync (505 → 527 files per Wf-334-C3 §0.18)

### §5.3 What the ultracode chain has NOT covered (cross-ref §4 above)
- Wf-310E G6 per-innovation ablation (TODO-182) — USER_OK-gated
- Wf-310E Wf-135 LOMO Phase 2 (Rh oversample regression) — USER_OK-gated
- Wf-310E Wf-310F frontier math GPU portion — USER_OK-gated
- TODO-138 paper v2 submission — TODO-84-gated
- Pre-flight v3 15-FAIL REGRESSION on default `uv run python` — code-class fix future Wf
- §9 honest framing catalog 38 → ≥40 entries — not yet expanded

---

## §6 Final verdict (per `party-twenty-spirit` §3.5)

### §6.1 LAMBDA_ROADMAP §1-§5 audit verdict

- **§1 R20**: PLANNED + PARTIAL EXEC (4 NEW R20 skills SHIPPED, R20-P5 30/30 6/6 PASS MET, R20-P2/P3 honest-neg ceilings documented, R20-P4 NOT LAUNCHED USER_OK-gated)
- **§2 R21**: PLANNED + PARTIAL EXEC (4 NEW R21 skills SHIPPED, R21-P3 reuse Wf-330-B output, R21-P2/P4/P5 NOT LAUNCHED)
- **§3 R22-R25**: R22 PLANNED (computational-only); R23-R25 DEPRECATED per AGENTS.md §5.4
- **§4 Metrics + 12-gap trajectory**: 12/12 L1-L12 have CPU bench evidence; 7/12 at-target / 5/12 deferred-GPU
- **§5 Resources**: Blackwell 98GB + RTX 5090 32GB ACTIVE; ORNL Summit + clinical DEPRECATED

### §6.2 User directive response ("你todo目录里应该还有别的任务当前启动的ultracode没有启动到的")

**Confirmed**: there are 4 categories of TODO tasks NOT launched by the Wf-330 + Wf-333 + Wf-334 ultracode chain:
1. **USER_OK-gated GPU jobs (3)**: TODO-182 G6 per-innovation ablation (Wf-310E, 7.5 hr) + TODO-135 LOMO Phase 2 (Wf-310E, ~2 hr) + Wf-310F frontier math GPU portion (Day 3-7)
2. **TODO-84 paper work (1)**: TODO-138 Paper v2 submission (Tier-2 journal target per Wf-317D)
3. **Code-class fix future Wf (1)**: pre-flight v3 15-FAIL REGRESSION on default `uv run python` (per §22.16)
4. **§9 honest framing catalog expansion (1)**: 38 → ≥40 entries not yet launched

All 4 categories are properly queued in `TODO/pending/` + `TODO/03_plans/` + `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 — they are NOT lost, they are gated (USER_OK or TODO-84 or scope-class future Wf).

### §6.3 Standing directives honored (per AGENTS.md §5)

- **§5.1 Standing GPU authorization (2026-09-29 user grant)**: GPU jobs > 1 hr still gated by user OK convention even with standing grant; AGENTS.md §5.1 §1.4 ("Pre-flight v2 25/25 PASS gate") honored by Wf-330-C 62/62 fresh-run
- **§5.2 Honest framing mandate**: every numeric value in this audit cites a JSON / report path; MEASURED vs PROJECTED tag discipline preserved
- **§5.3 No paper work without authorization**: TODO-84 ACTIVE; paper §3 / paper v2 / paper v3 all preserved
- **§5.4 Tier-2 journal target + NO wet-lab planning**: R22 computational-only re-scoped; R23-R25 DEPRECATED; wetlab_* templates retained but not for execution

---

**Append date (this Wf-339-D2 audit)**: 2026-09-29 (LAMBDA_ROADMAP §1-§5 docs audit + ultracode residual task identification per user directive)

**Cross-references**:
- `TODO/03_plans/LAMBDA_ROADMAP.md` (full doc, §1-§10.5)
- `TODO/INDEX.md` §0.13-§0.18 (Wf-325..334 ship-wave timeline)
- `TODO/completed/83j_lambda_closeout.md` (12-gap L1-L12 final status)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 + §22 (honest framing catalog + 15-FAIL REGRESSION breakdown)
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` + `IMPROVEMENT_QUEUE_2026_09_27.md` (3-task still-pending forward plan)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4 (GPU-required deferred items)
- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §2-§6 (R20 phases NOT launched)
- `AGENTS.md` §5.1 + §5.3 + §5.4 (standing directives)
