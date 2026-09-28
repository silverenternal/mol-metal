# TODO-107 — Wf-82 Combined MCTS Variance (VariancePUCT + Score-Rep MC) Integration Close-out

**Date**: 2026-09-26
**Workflow**: Wf-82 (CPU-only docs — VariancePUCT + Score-Rep MC pipeline integration ledger for L2 + L3 + L7 + L11 cross-dataset variance reduction)
**Status**: ⚠️ **HONEST PARTIAL SHIP** — per-cohort variance_reduction JSON SHIP (Wf-65 P1 sign-flip audit 3 configs + Wf-66 slot reservation + Wf-64 cross-dataset scaffold); **0 GPU retrain on real PlatinAI 4-cell / MetalCytoToxDB 5-metal / tmQM d-block** — combined Wf-82 cross-cohort `variance_reduction ≥ 0` is **PROJECTED**, not MEASURED.
**Priority**: MEDIUM (extends Wf-65 (R20-P1 L2) + Wf-66 (R20-P3 L7+L3) close-out with combined-pipeline ledger; gates R20-P1 + R20-P3 GPU retrain verification + Wf-73 cross-dataset MCTS lift)
**Owner**: Hugo Lee + Wf-82 ultracode agent (this turn)
**Parent**:
- `TODO/completed/92_r20_p1_variance_puct_lazy_rdkit.md` (Wf-65 R20-P1 close-out, Wf-84 P1.4)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-66 R20-P3 close-out)
- `molmetal/reports/wf65_p1_sign_audit/sign_flip_audit.json` (Wf-65 P1 per-cohort 3-config audit)
- `molmetal/reports/wf64_m8_cross/synthesis.md` (Wf-64 P3 cross-dataset scaffold: PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block)
- `molmetal/reports/wf36_variance_mcts/variance_bench.json` (Wf-36 P2 mock CCO baseline, `variance_reduction = -0.185`)
- `molmetal/reports/wf36_3stage_mcts/3stage_bench.json` (Wf-36 P3 joint ScoreRep + VariancePUCT + MCTS, 16 runs × 4 stages, `success_rate=1.0`, `mean_leaf_var=0.00109`)
- `TODO/completed/101_wf73_p2_ood_permetal.md` (Wf-73 P2 per-metal OOD on MetalCytoToxDB 5-metal — Ru 0.20 only)
- `TODO/completed/102_wf73_p3_ood_perelement.md` (Wf-73 P3 per-element OOD on tmQM d-block 30 elements, 20/30 PASS)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L2 / L3 / L7 / L11 columns — 12-row trajectory)
- `TODO/INDEX.md` §17.6 (L13-L16 candidate row + party-twenty-spirit §3.5 cross-ref)

**Constraint**: CPU-only docs — 0 GPU retrain + 0 wet-lab execution; **paper-gate honored** per `pending/84_no_paper_task_without_authorization.md`; **honest framing per `party-twenty-spirit §3.5`**.
**Skill alignment**: `party-twenty-spirit` §3.5 (0-tolerance honest framing) + `mol-metal-lambda-calculus` (L2+L3+L11 anchor) + `mol-metal-r18-math` (T5 Multi-PH + M26 Score-Rep) + `mol-metal-multiph-pll` (Wf-55 S3 ship, 518 LOC).

---

## §1 TL;DR

Wf-82 ships the **VariancePUCT + Score-Rep MC pipeline integration** close-out as a CPU-only docs ledger that consolidates Wf-65 (VariancePUCT-only, R20-P1 L2) + Wf-66 (Score-Rep MC-only, R20-P3 L7+L3) + Wf-64 (cross-dataset scaffold) into a single combined-pipeline frame for R20-P1 + R20-P3 verification + Wf-73 cross-cohort MCTS lift:

| Aspect | Detail |
|---|---|
| **Combined pipeline** | VariancePUCT (`molmetal/molmetal_lam/search_alg/variance_puct_integration.py`, 917 LOC) + Score-Rep MC (`molmetal/molmetal_lam/search_alg/score_repellent_mc.py`, 857 LOC) + joint 3-stage MCTS (`molmetal/scripts/joint_score_repellent_variance_puct_mcts.py`, 1035 LOC, Wf-36 P3 SHIP) |
| **Per-cohort JSON SHIP** | Wf-65 P1 `sign_flip_audit.json` (3 configs: canonical / pre_fix / post_fix) — MEASURED on bench; Wf-64 P3 `m8_cross_summary.json` — MEASURED but m8 PB pass_rate (cross-dataset, NOT variance); Wf-36 P2 `variance_bench.json` mock CCO `variance_reduction = -0.185` (honest-negative) |
| **Cross-cohort PROJECTED** | combined `variance_reduction ≥ 0` across PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block — **NOT MEASURED** today; conditioned on RTX 5090 GPU-1 wall-clock (next-week idle slot per `mol-metal-gpu-execution` §5 kanzi contention policy) |
| **L-column status (per LAMBDA_ROADMAP §4.2)** | L2 (variance_mcts ↔ proof_search): Wf-36 P2 + Wf-39 P1 baseline SHIP + Wf-65 P1 sign-flip audit SHIP; L3 (ScoreRep + VariancePUCT + MCTS joint): Wf-36 P3 joint SHIP on mock, Wf-66 P3.3 PROJECTED on real; L7 (Multi-PH + PLL end-to-end): Wf-37 P3 +0.087 honest-negative + Wf-66 P3.1+P3.2 PROJECTED; L11 (MCTS tree per-node statistics): Wf-40 P3 SHIP success 100% |
| **0 paper work touched** | Per `pending/84_no_paper_task_without_authorization.md` — no `paper/sections/*.tex`, no `paper/main.tex` |

**One-liner (per `party-twenty-spirit §3.5`)**: Wf-82 SHIPs the **combined VariancePUCT + Score-Rep MC pipeline integration ledger** as CPU-only docs — per-cohort variance_reduction JSON on bench (3 Wf-65 configs) + per-dataset m8 PB pass_rate scaffold (Wf-64 P3, NOT variance) + Wf-36 P3 joint 3-stage SHIP — but **0 MEASURED cross-cohort `variance_reduction ≥ 0`** on PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block today; all 3 cohorts PROJECTED on RTX 5090 GPU-1 next-week slot.

---

## §2 Per-deliverable

### §2.1 Pipeline integration (Wf-82 ★ primary)

The combined VariancePUCT + Score-Rep MC pipeline is composed of 3 modules that already ship independently in R17-R18 + Wf-9/11/14:

| Module | LOC | Tests | Source workflow | Status |
|---|---:|---:|---|:---:|
| `molmetal/molmetal_lam/search_alg/score_repellent_mc.py` | 857 | 23/23 PASS | Wf-9 (R18 ship, Burch-Bardsley 2022 + Hu et al. 2026 ICML) | ✅ SHIP |
| `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` | 917 | 20/20 PASS | Wf-11 (R18 ship, Rosin 2011 PUCT composition) — sign-flip fix in Wf-51 | ✅ SHIP (no Wf-82 change) |
| `molmetal/scripts/joint_score_repellent_variance_puct_mcts.py` | 1035 | 15/15 PASS | Wf-36 P3 (3-stage joint) | ✅ SHIP |
| `molmetal/reports/wf36_3stage_mcts/3stage_bench.json` | — | n/a | Wf-36 P3 16 runs × 4 stages | ✅ SHIP (mock CCO) |

**Pipeline composition** (no new code; Wf-82 is a docs ledger, NOT an implementation cycle):
1. **Stage 1** (`score_repellent_mc.Sampler`): reparameterized score-repellent Langevin MCMC (Burch-Bardsley 2022); reduces joint Hilbert regression estimator variance by 1.4-2.0× over Dirichlet per Hu et al. 2026 ICML §3.2.
2. **Stage 2** (`variance_puct_integration.VarianceControlledPUCT`): PUCT bonus subtracts `c_var * sigma` (post Wf-51 fix — drop `(1 + N_child)` divisor per Burch & Bardsley 2022 §4.4); persistent penalty vs decaying.
3. **Stage 3** (`proof_search.MCTSVarianceWrapper`): joint 3-stage MCTS trajectory with replay buffer 48/64 fill per Wf-36 P3 SHIP.

**MEASURED on joint mock CCO** (per `wf36_3stage_mcts/3stage_bench.json`):

| Metric | Value | Source |
|---|---:|---|
| n_runs | 16 | aggregate.n_runs |
| success_rate | **1.0** | aggregate.success_rate |
| mean_best_score | 2.0 | aggregate.mean_best_score |
| mean_leaf_value_var | **0.00109** | aggregate.mean_leaf_value_var |
| n_replay_entries_added | 48 | aggregate.n_replay_entries_added (buffer_size=64) |
| stage_wall_time split | stage1=0.17% / stage2=0.011% / stage3=99.67% / stage4=0.15% | aggregate.stage_wall_time_pct |
| n_errors | 0 | aggregate.n_errors |

**Honest framing**: Wf-36 P3 joint is **SHIPPED on mock CCO** (`start_smiles="CCO"`, `max_depth=3`, `n_simulations=8`). Per-cohort real-data verification on PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block is **NOT MEASURED**; combined `variance_reduction ≥ 0` is PROJECTED on RTX 5090 GPU-1.

### §2.2 PlatinAI 4-cell cohort (L3 ★secondary)

**PlatinAI 4-cell** = Pt-only 4-cell-line synthetic pIC50 dataset (HOMO/LUMO/Gap/Polarizability) per `build_platinai_4cell.py`. Wf-82 cross-cohort verification on this cohort is **NOT MEASURED today**:

| Item | Wf-65 P1 status | Wf-66 P3.3 status | Wf-82 combined status |
|---|:---:|:---:|:---:|
| **VariancePUCT scoring rule audit** | ✅ SHIP (`sign_flip_audit.json` post_fix `variance_reduction = 0.4` on bench) | n/a | ✅ SHIP (mock) |
| **Score-Rep MC variance reduction on real** | n/a | ❌ DEFERRED (slot reservation + PLACEHOLDER per `wf66_r20_p3/score_rep_mc_variance.json`) | ❌ DEFERRED |
| **Combined pipeline on PlatinAI 4-cell** | ❌ DEFERRED (P1.3 real-data trajectory benchmark) | ❌ DEFERRED | ❌ DEFERRED |
| **Target PROJECTED** | `variance_reduction ≥ 0.4` | `variance_reduction ≥ 0.2` (stretch) / `≥ 0.10` (realistic, matching Wf-14 P5 mock α=0.185) | combined `variance_reduction ≥ 0` |

**Carry-forward MEASURED on PlatinAI 4-cell** (cross-cohort context only — NOT variance):
- `Wf-71 P3 F5_RDKIT_2D cross-dataset macro R² = +0.1248` (m24 NEW METRIC, STRONG_REAL per `TODO/completed/100_wf71_p3_wf49_cross_round2.md`)
- `Wf-49 P2 F5_RDKIT_2D best Pearson r = +0.204 on MetalCytoToxDB (MolLogP)` (per `molmetal/reports/wf49_astr_rdkit/`)
- `Wf-41 P2 OOD reachability PlatinAI 5K = 0.5581` (per `molmetal/reports/wf41_3way_integration/`)
- `Wf-32 m8 P2 baseline = 0.846` (1-pocket × 26-candidate × 26-check, MMFF94s relaxation, single connected candidate per cell)

**Honest framing**: PlatinAI 4-cell `variance_reduction` is **NOT MEASURED today**. The P1.3 trajectory benchmark (`trajectory_platinai_5k.json` + `trajectory_metalcytotox.json` per-rule score + per-arm visits) requires RTX 5090 GPU-0 wall-clock (1 week per Wf-65 plan §3). Under Wf-82 CPU-only docs context, this is **out of scope**.

### §2.3 MetalCytoToxDB 5-metal cohort (L2 ★secondary + L12 cross-ref)

**MetalCytoToxDB 5-metal** = Ru / Ir / Os / Rh / Re pIC50 dataset for metal-coordinating anticancer compounds. Wf-82 cross-cohort verification on this cohort is **NOT MEASURED today**:

| Item | Wf-65 P1 status | Wf-66 P3.3 status | Wf-82 combined status | Wf-73 P2 cross-ref |
|---|:---:|:---:|:---:|:---:|
| **VariancePUCT scoring rule audit** | ✅ SHIP (3 configs MEASURED on bench) | n/a | ✅ SHIP (mock) | n/a |
| **Score-Rep MC variance reduction on real** | n/a | ❌ DEFERRED | ❌ DEFERRED | n/a |
| **Combined pipeline on MetalCytoToxDB 5-metal** | ❌ DEFERRED | ❌ DEFERRED | ❌ DEFERRED | Wf-73 P2 Ru 0.20 reachability MEASURED (per-metal OOD) |
| **Target PROJECTED** | `variance_reduction ≥ 0.4` | `variance_reduction ≥ 0.2` | combined `variance_reduction ≥ 0` | closure reachability Ru only (variance cross-axis) |

**Carry-forward MEASURED on MetalCytoToxDB 5-metal** (cross-cohort context only — NOT variance):
- `Wf-73 P2 per-metal OOD reachability`: Ru **0.20** only (60 samples/metal × 2 seeds = 300 SMILES, 243.7 s CPU wall); Ir 0.00, Os 0.00, Rh 0.00, Re 0.00 (per `TODO/completed/101_wf73_p2_ood_permetal.md`)
- Rh bottleneck on regression axis (m19 LOMO Rh 0.368) is **orthogonal** to closure-axis — Rh bottleneck is a `F5_RDKIT_2D` regression artifact, not a closure-theorem reachability artifact
- `Wf-49 P2 F5_RDKIT_2D MolLogP |ρ| = +0.204` on MetalCytoToxDB (per `molmetal/reports/wf49_astr_rdkit/`)
- `Wf-41 P2 OOD reachability MetalCytoToxDB = 0.0300` (Tanimoto<0.4 cohort)

**Honest framing**: MetalCytoToxDB 5-metal `variance_reduction` is **NOT MEASURED today**. The cross-axis (closure OOD vs variance reduction) is independent — Wf-73 P2 Ru 0.20 is on closure axis, not on variance axis. Combined Wf-82 verification requires RTX 5090 GPU-1.

### §2.4 tmQM d-block cohort (L11 + L12 ★secondary)

**tmQM d-block** = transition-metal quantum mechanics dataset, 30 d-block elements × ≥ 60 samples/element. Wf-82 cross-cohort verification on this cohort is **NOT MEASURED today**:

| Item | Wf-65 P1 status | Wf-66 P3.3 status | Wf-82 combined status | Wf-73 P3 cross-ref |
|---|:---:|:---:|:---:|:---:|
| **VariancePUCT scoring rule audit** | ✅ SHIP (3 configs MEASURED on bench) | n/a | ✅ SHIP (mock) | n/a |
| **Score-Rep MC variance reduction on real** | n/a | ❌ DEFERRED | ❌ DEFERRED | n/a |
| **Combined pipeline on tmQM d-block** | ❌ DEFERRED | ❌ DEFERRED | ❌ DEFERRED | Wf-73 P3 20/30 PASS per-element (3d=9/10, 4d=7/10, 5d=4/10) |
| **Target PROJECTED** | `variance_reduction ≥ 0.4` | `variance_reduction ≥ 0.2` | combined `variance_reduction ≥ 0` | closure reachability ≥ 0.40 (variance cross-axis) |

**Carry-forward MEASURED on tmQM d-block** (cross-cohort context only — NOT variance):
- `Wf-73 P3 per-element OOD reachability`: 20/30 PASS (66.67% pass-rate); 3d-block 9/10 = 90.00% (V only fail); 4d-block 7/10 = 70.00% (Mo/Pd/Tc fail); 5d+La 4/10 = 40.00% (Os/Ir/Pt/Au/Re fail); bench 103.78 s CPU wall (per `TODO/completed/102_wf73_p3_ood_perelement.md`)
- `Wf-28 tmQM d-block training pool SHIP` (Wf-28 ship, R17 baseline)
- `Wf-57 P1 Tanimoto<0.4 OOD protocol` (32 256 cross-pairs verified split)

**Honest framing**: tmQM d-block `variance_reduction` is **NOT MEASURED today**. The cross-axis (closure OOD vs variance reduction) is independent — Wf-73 P3 20/30 PASS is on closure axis, not on variance axis. Combined Wf-82 verification requires RTX 5090 GPU-1.

---

## §3 MEASURED vs PROJECTED (per `party-twenty-spirit §3.5`)

### §3.1 MEASURED (on bench today — CPU-only)

| Item | Value | Source |
|---|---:|---|
| **Wf-65 P1 sign-flip audit `canonical` `variance_reduction`** | 0.0 | `sign_flip_audit.json` per_config.canonical |
| **Wf-65 P1 sign-flip audit `pre_fix` `variance_reduction`** | 0.019 | `sign_flip_audit.json` per_config.pre_fix |
| **Wf-65 P1 sign-flip audit `post_fix` `variance_reduction`** | **0.4** | `sign_flip_audit.json` per_config.post_fix (live code matches this) |
| **Wf-65 P1 sign-flip audit divergence `pre_fix_vs_post_fix_high_sigma_score_diff`** | 0.190 | `sign_flip_audit.json` divergence field |
| **Wf-36 P2 Score-Rep MC on mock CCO `variance_reduction`** | **−0.185** (honest-negative) | `wf36_variance_mcts/variance_bench.json` variance_reduction.score_repellent_vs_dirichlet |
| **Wf-36 P3 joint 3-stage mock CCO `success_rate`** | **1.0** | `wf36_3stage_mcts/3stage_bench.json` aggregate |
| **Wf-36 P3 joint 3-stage mock CCO `mean_leaf_value_var`** | **0.00109** | `wf36_3stage_mcts/3stage_bench.json` aggregate |
| **Wf-36 P3 joint 3-stage mock CCO `n_replay_entries_added`** | **48/64** | `wf36_3stage_mcts/3stage_bench.json` aggregate |
| **Wf-65 P1 NEW test count** | **27** (12 + 15) | `test_variance_puct_sign.py` (12) + `test_variance_puct_sign_fix.py` (15) |
| **Wf-66 NEW code LOC (3 retrain + 3 verify scripts)** | **0** (slot reservation only; GPU runs deferred) | `TODO/completed/94_r20_p3_multiph_pll_feature.md` §2.1 |

### §3.2 PROJECTED (combined cross-cohort verification — NOT EXECUTED today)

| Item | Stretch target | Realistic landed | Gate | ETA |
|---|---:|---:|---|---|
| **L2 VariancePUCT default + lazy RDKit (PlatinAI 4-cell + MetalCytoToxDB 5-metal)** | `variance_reduction ≥ 0.4` (matching Wf-65 P1 post_fix bench) | `variance_reduction ≥ 0.10` (matching Wf-14 P5 mock α=0.185 honest-negative lift) | n_min_visits=20, c_var=0.5 | RTX 5090 GPU-0 × 1 wk |
| **L3 Score-Rep MC variance reduction (PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block)** | `variance_reduction ≥ 0.2` | `variance_reduction ≥ 0.10` | n_target ≥ 1000 + Hu et al. 2026 reparameterization landing | RTX 5090 GPU-1 × 1 wk |
| **L7 Multi-PH + PLL R² (PlatinAI 4-cell + MetalCytoToxDB 5-metal)** | `≥ 0.70` | `≥ 0.30` | M22 + M23 mechanism cross-lift | RTX 5090 GPU-1 × 1 wk |
| **Combined Wf-82 cross-cohort (PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block)** | `variance_reduction ≥ 0` on all 3 cohorts | `variance_reduction ≥ 0` on PlatinAI 4-cell + MetalCytoToxDB 5-metal only (tmQM d-block excluded due to closure axis independence) | Wf-82 P2 MEASURED gate | RTX 5090 GPU-1 × 3 wk (parallel Wf-65 P1.3 + Wf-66 P3.3 + Wf-82 combined) |
| **L11 MCTS tree per-node statistics hit_rate** | `≥ 0.95` | `≥ 0.80` | Wf-25 vuln_f + vuln_g retrain (TODO-93 P2) | RTX 5090 GPU-0 × 2 wk |

**Cumulative wall-time** (if serialized on GPU-0/GPU-1): ~7 weeks GPU + ~3 weeks CPU.
**Cumulative wall-time** (if parallelized): ~3 weeks GPU + ~3 weeks CPU.

### §3.3 MEASURED-PROJECTED distinction (party-twenty-spirit §3.5)

| Item | Status | Reason |
|---|:---:|---|
| Wf-65 P1 sign-flip audit 3 configs (canonical / pre_fix / post_fix) | ✅ MEASURED on bench | Audit script deterministic CPU-only; `sign_flip_audit.json` reproducible |
| Wf-65 P1 `live_code_matches_post_fix` verdict | ✅ MEASURED on bench | Source-code probe via `audit_variance_puct_sign.py` line 27 (`has_divisor_on_penalty=False`) |
| Wf-36 P2 mock CCO `variance_reduction = -0.185` | ✅ MEASURED on bench | 60 trajectories × 3 strategies (dirichlet / score_repellent / variance) on `start_smiles="CCO"` |
| Wf-36 P3 joint 3-stage `success_rate = 1.0` on mock | ✅ MEASURED on bench | 16 runs × 4 stages (`wf36_3stage_mcts/3stage_bench.json`) |
| Wf-66 P3.3 score_rep_mc_variance.json on PlatinAI 4-cell + MetalCytoToxDB 5-metal | 🟡 PROJECTED (PLACEHOLDER per Wf-66 §1.3) | GPU retrain not executed; RTX 5090 GPU-1 30-min CI budget blocked |
| Wf-65 P1.3 real-data trajectory benchmark | 🟡 PROJECTED (DEFERRED to R19 GPU-0) | RTX 5090 GPU-0 wall-clock (1 wk per plan §3 P1.3) |
| Wf-82 combined cross-cohort verification | 🟠 PROJECTED on RTX 5090 GPU-1 × 3 wk | Out of Wf-82 CPU-only docs context |
| L11 MCTS tree per-node statistics hit_rate ≥ 0.95 | 🟠 DEFERRED to R20-P2 GPU retrain | Per `LAMBDA_ROADMAP §1.3` R20-P2 W3-W4 |

---

## §4 Before vs After (Wf-65 + Wf-66 + Wf-64 + Wf-73 → Wf-82 combined)

| Aspect | Before (Wf-65 VariancePUCT only + Wf-66 Score-Rep MC only + Wf-64 cross-dataset scaffold + Wf-73 per-cohort OOD) | After (Wf-82 combined pipeline ledger) |
|---|---|---|
| **VariancePUCT scoring rule audit** | Wf-65 P1 SHIP 3 configs (`canonical / pre_fix / post_fix`) on bench; `live_code_matches_post_fix=true`; 27 NEW tests + audit JSON | **Same** — Wf-82 does not redo Wf-65 P1 audit (audit is read-only deterministic) |
| **Score-Rep MC variance reduction on real data** | Wf-66 P3.3 slot reservation only; `wf66_r20_p3/score_rep_mc_variance.json` PLACEHOLDER; GPU retrain DEFERRED | **Same** — Wf-82 does not redo Wf-66 P3.3 (CPU-only docs context); PROJECTED ETA RTX 5090 GPU-1 × 1 wk |
| **Joint 3-stage ScoreRep + VariancePUCT + MCTS** | Wf-36 P3 SHIP on mock CCO; `success_rate=1.0`, `mean_leaf_value_var=0.00109`, replay_buffer 48/64; 16 runs × 4 stages | **Same** — Wf-82 does not redo Wf-36 P3 (joint pipeline is Wf-36 P3 ship surface) |
| **Cross-dataset scaffold** | Wf-64 P3 SHIP per-dataset m8 PB pass_rate (`platinai_4cell / metalcytotox_5metal / tmqm_dblock` × 5 seeds × 25 samples); all 3 cohorts pass_rate=0.0 (HONEST-NEGATIVE — disconnected fragments) | **Same** — Wf-64 P3 is m8 PB cross-dataset, NOT variance; Wf-82 cross-references without re-running |
| **Per-cohort OOD reachability** | Wf-73 P2 (MetalCytoToxDB 5-metal) Ru 0.20 + Ir/Os/Rh/Re 0.00; Wf-73 P3 (tmQM d-block 30 elements) 20/30 PASS | **Same** — Wf-73 P2+P3 are closure-axis OOD, NOT variance-axis; Wf-82 cross-references without re-running |
| **Combined cross-cohort variance_reduction JSON** | ❌ NOT EXISTED before Wf-82 (no per-cohort real-data variance_reduction JSON on PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block) | **🟡 PROJECTED in Wf-82** — `molmetal/reports/wf82_combined_variance/{platinai_4cell,metalcytotox_5metal,tmqm_dblock}/variance_reduction.json` planned (slot reservation; not yet on disk) |
| **LAMBDA_ROADMAP §4.2 L2 / L3 / L7 / L11 cross-ref** | 12-row trajectory; L2/L3/L7/L11 columns ship per `83j_lambda_closeout.md` §5 | **Same** — Wf-82 cross-references L2 + L3 + L7 + L11 columns without patching §4.2 |
| **INDEX §17 5-cat audit row** | §17.3 R20 plan items P1 (TODO-92) + P3 (TODO-94) ship per Wf-70 ledger; §17.6 L13-L16 candidates + Wf-82 §17.6 patch ship | **Same** — Wf-82 cross-references §17 + §17.6 without new audit row |
| **Honest negative carry-forward** | Wf-65 P1.2 lazy RDKit HONEST PARTIAL; Wf-66 P3 PROJECTED; Wf-64 P3 HONEST-NEGATIVE m8=0.0; Wf-73 P2 Rh 0.00 honest-negative | **Wf-82 honest framing preserved** — per-cohort variance_reduction JSON SHIP only for bench (Wf-65 P1 3 configs); per-cohort real-data variance_reduction JSON PROJECTED on RTX 5090 GPU-1 |

**Wf-82 verdict (one-liner)**: Wf-82 SHIPs a **combined pipeline ledger** for VariancePUCT + Score-Rep MC integration that:
1. Cross-references Wf-65 P1 sign-flip audit (3 configs MEASURED) + Wf-66 P3.3 slot reservation (PROJECTED) + Wf-36 P3 joint 3-stage (mock MEASURED) + Wf-64 P3 cross-dataset scaffold (m8 PB, NOT variance) + Wf-73 P2/P3 per-cohort OOD (closure-axis, NOT variance-axis)
2. **0 MEASURED cross-cohort variance_reduction on PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block today**
3. **0 paper work touched** per TODO-84 paper-gate
4. **0 LAMBDA_ROADMAP §4.2 patch executed** (12-row → 16-row extension proposed in §17.6 not adopted)

---

## §5 Honest framing per `party-twenty-spirit §3.5` (REQUIRED)

| Honest finding | Evidence |
|---|---|
| **Combined Wf-82 cross-cohort `variance_reduction ≥ 0` is PROJECTED, not MEASURED** | Per §2.2 + §2.3 + §2.4 + §3.2; requires RTX 5090 GPU-1 × 3 wk (parallel Wf-65 P1.3 + Wf-66 P3.3 + Wf-82 combined); out of Wf-82 CPU-only docs context |
| **Per-cohort real-data variance_reduction JSON SHIP only for bench** | Wf-65 P1 sign-flip audit 3 configs MEASURED on bench; Wf-36 P2 mock CCO `variance_reduction = -0.185` MEASURED on bench; no PlatinAI 4-cell / MetalCytoToxDB 5-metal / tmQM d-block JSON SHIP |
| **Wf-66 P3.3 GPU retrain is the bottleneck for L3 Score-Rep MC verification** | Per `TODO/completed/94_r20_p3_multiph_pll_feature.md` §1.3: RTX 5090 GPU-1 30-min CI budget blocked; fallback RTX PRO 6000 Blackwell kanzi contention policy per `mol-metal-gpu-execution` §5 |
| **Wf-65 P1.3 real-data trajectory benchmark is the bottleneck for L2 VariancePUCT verification** | Per `TODO/completed/92_r20_p1_variance_puct_lazy_rdkit.md` §2.3: RTX 5090 GPU-0 wall-clock (1 wk per plan §3); R19 GPU retrain slot reserved per `r20_status_ledger_2026_09_26.md` §3.1 |
| **Cross-axis independence: closure OOD ≠ variance reduction** | Wf-73 P2 Ru 0.20 (closure axis) ≠ Wf-65 P1 post_fix 0.4 (variance axis); Wf-73 P3 20/30 PASS (closure axis) ≠ Wf-36 P3 joint (variance axis). Rh bottleneck is regression axis (m19 LOMO 0.368) which is independent of both closure axis and variance axis |
| **Wf-64 P3 m8 PB pass_rate = 0.0 is NOT variance cross-cohort** | Per `molmetal/reports/wf64_m8_cross/synthesis.md` §Failure mode: structural finding (disconnected fragments, counterions) on PB `mol` config; NOT a variance measurement; Wf-82 cross-references without re-running |
| **0 paper work touched** | Per `pending/84_no_paper_task_without_authorization.md` — no `paper/sections/*.tex`, no `paper/main.tex`, no reviewer rebuttal |
| **0 LAMBDA_ROADMAP §4.2 patch executed today** | §17.6.3 proposed 12 → 16 row extension (L13-L16); Wf-72 Phase 4 (next cycle) applies the patch; Wf-82 does not patch §4.2 |
| **Wf-36 P2 mock CCO `variance_reduction = -0.185` is HONEST-NEGATIVE** | The pre-fix `variance_reduction` is a *baseline*, not current state; fix already in place from Wf-51; Wf-65 P1 verifies `live_code_matches_post_fix=true` |
| **L11 MCTS tree per-node statistics hit_rate ≥ 0.95 DEFERRED to R20-P2** | Per `LAMBDA_ROADMAP §1.3` R20-P2 W3-W4; Wf-40 P3 success 100% / 85 nodes / depth 3 is bench, NOT hit_rate on real data |
| **m18/m19/m20 tier-1 metric UNDEFINED** | Per `INDEX §17.1` — m18 (Protein-Metal AUC) / m19 (LOMO AUC) / m20 (Tumor-specific AUC) all UNDEFINED or WEAK_REAL_NEEDS_MORE_DATA; Wf-82 does not invent tier-1 metric |

**Conclusion (party-twenty-spirit §3.5)**: This Wf-82 ships **honest framing** — no MEASURED numbers fabricated, no PROJECTED numbers labeled as MEASURED, no GPU/wet-lab phases silently claimed ship. The 3 cross-cohort real-data `variance_reduction` JSONs (PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block) are PROJECTED on RTX 5090 GPU-1 × 3 wk parallel, NOT measured today. The 3 bench `variance_reduction` JSONs (Wf-65 P1 3 configs + Wf-36 P2 mock + Wf-36 P3 joint mock) ARE measured and SHIP.

---

## §6 Cross-references

Wf-82 sits at the **L2 + L3 + L7 + L11 pipeline integration** of the LAMBDA_ROADMAP §4.2 trajectory. Cross-link:

- **Wf-65 R20-P1** (`TODO/completed/92_r20_p1_variance_puct_lazy_rdkit.md`) — VariancePUCT default + lazy RDKit import (L2); sign-flip audit SHIP; P1.3 trajectory benchmark DEFERRED
- **Wf-65 P1 audit JSON** (`molmetal/reports/wf65_p1_sign_audit/sign_flip_audit.json`) — 3 configs MEASURED (canonical / pre_fix / post_fix); `live_code_matches_post_fix=true`
- **Wf-65 P1 audit script** (`molmetal/scripts/audit_variance_puct_sign.py`) — 457 LOC NEW
- **Wf-65 P1 NEW tests** (`molmetal/tests/test_variance_puct_sign.py` 12 tests + `molmetal/tests/test_variance_puct_sign_fix.py` 15 tests) — 27/27 PASS
- **Wf-66 R20-P3** (`TODO/completed/94_r20_p3_multiph_pll_feature.md`) — Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep MC (L7+L3); close-out + skill SHIP; GPU retrain DEFERRED
- **Wf-66 P3.3 PLACEHOLDER** (`molmetal/scripts/wf66_r20_p3/score_rep_mc_full_platinai.py` + `score_rep_variance_reduction.py` + `score_rep_mc_variance.json`) — slot reserved; GPU runs deferred
- **Wf-64 P3 cross-dataset scaffold** (`molmetal/reports/wf64_m8_cross/synthesis.md` + `m8_cross_summary.json`) — PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block × 5 seeds × 25 samples; m8 PB pass_rate = 0.0 HONEST-NEGATIVE
- **Wf-73 P2 per-metal OOD** (`TODO/completed/101_wf73_p2_ood_permetal.md`) — MetalCytoToxDB 5-metal per-metal; Ru 0.20 + others 0.00; `RH_HONEST_PARTIAL` verdict
- **Wf-73 P3 per-element OOD** (`TODO/completed/102_wf73_p3_ood_perelement.md`) — tmQM d-block 30 elements; 20/30 PASS (66.67% pass-rate); `PER_ELEMENT_HONEST_PARTIAL` verdict
- **Wf-36 P2 mock CCO** (`molmetal/reports/wf36_variance_mcts/variance_bench.json`) — `variance_reduction = -0.185` honest-negative on `start_smiles="CCO"`; 60 trajectories × 3 strategies
- **Wf-36 P3 joint 3-stage** (`molmetal/reports/wf36_3stage_mcts/3stage_bench.json`) — 16 runs × 4 stages; `success_rate = 1.0`, `mean_leaf_value_var = 0.00109`, `n_replay_entries_added = 48/64`
- **Wf-36 P3 joint script** (`molmetal/scripts/joint_score_repellent_variance_puct_mcts.py`) — 1035 LOC NEW
- **Wf-9 Hilbert + Score-Rep** (`TODO/completed/62_hilbert_score_repellent.md`) — Wf-9 ship: `score_repellent_mc.py` (857 LOC, 23/23 tests PASS, Burch-Bardsley 2022 + Hu et al. 2026 ICML)
- **Wf-11 R18 close-out** (`TODO/completed/64_r18_closeout.md`) — Wf-11 ship: `variance_puct_integration.py` (917 LOC, 20/20 tests PASS, Rosin 2011 PUCT composition)
- **Wf-51 sign audit** (`molmetal/reports/wf51_sign_audit/sign_flip_diff.md`) — pre-Wf-65 sign-flip fix commit-equivalent ship
- **`TODO/completed/r20_status_ledger_2026_09_26.md`** §1.1 (R20-P1 ledger) + §1.3 (R20-P3 ledger) + §3.1 (GPU retrain ETA) + §2.2 (carry-forward Wf-36 + Wf-37 baselines)
- **`TODO/completed/100_r20_r21_consolidation.md`** — Wf-70 Phase 2 L1-L8 → L1-L12 expansion + INDEX §17 sweep + 7 close-out files
- **`TODO/INDEX.md`** §17.3 (R20 plan table) + §17.5 (TODO 100-107 R22 plan items) + §17.6 (L13-L16 candidate row + party-twenty-spirit cross-ref)
- **`TODO/03_plans/LAMBDA_ROADMAP.md`** §4.2 (L2 / L3 / L7 / L11 columns in 12-row L1-L12 trajectory) + §1.2 (R20 6 phases) + §1.3 (R20 key metric targets including MCTS variance reduction −35% / −60%) + §3 (R22-R25 long-term)
- **`TODO/pending/84_no_paper_task_without_authorization.md`** — paper-gate honored (0 paper work touched today)
- **`.claude/skills/party-twenty-spirit/SKILL.md`** §3.5 (0-tolerance honest framing) — applied throughout this file
- **`.claude/skills/mol-metal-lambda-calculus/SKILL.md`** (L2 / L3 / L11 anchor)
- **`.claude/skills/mol-metal-r18-math/SKILL.md`** (T5 Multi-PH + M26 Score-Rep)
- **`.claude/skills/mol-metal-multiph-pll/SKILL.md`** (Wf-55 S3 ship, 518 LOC SKILL.md — covers L7 ★ + L3 ★)
- **`mol-metal-gpu-execution`** §5 kanzi contention policy (GPU fallback RTX PRO 6000 Blackwell)

---

## §7 One-line verdict

Wf-82 SHIPs the **combined VariancePUCT + Score-Rep MC pipeline integration ledger** as CPU-only docs — per-cohort variance_reduction JSON SHIP only on bench (Wf-65 P1 3 configs MEASURED `0.0 / 0.019 / 0.4` + Wf-36 P2 mock CCO `−0.185` honest-negative + Wf-36 P3 joint mock `success_rate=1.0 / mean_leaf_var=0.00109`) + per-dataset m8 PB scaffold (Wf-64 P3, NOT variance) + per-cohort OOD reachability (Wf-73 P2/P3, closure-axis, NOT variance-axis) — **0 MEASURED cross-cohort `variance_reduction ≥ 0` on PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block today**; combined Wf-82 cross-cohort verification PROJECTED on RTX 5090 GPU-1 × 3 wk parallel (Wf-65 P1.3 + Wf-66 P3.3 + Wf-82 combined) — 0 paper work touched, 0 LAMBDA_ROADMAP §4.2 patch executed, honest framing per `party-twenty-spirit §3.5` preserved throughout.

**Date**: 2026-09-26 (Wf-82 VariancePUCT + Score-Rep MC integration close-out synthesis, post-Wf-65/Wf-66/Wf-64/Wf-73 ship).