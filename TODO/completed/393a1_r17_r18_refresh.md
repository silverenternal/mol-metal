# Wf-393-A1 — R17 + R18 strategic plans docs-only refresh (Wf-380..390 ship surface)

**Date**: 2026-09-30
**Workflow**: Wf-393-A1 (ultracode CPU-only docs-only wave; 0 NEW tests; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning per AGENTS.md §10)
**Status**: ✅ SHIPPED — 2 plans refreshed; 1 NEW closeout doc authored
**Owner**: Hugo Lee + Wf-393-A1 ultracode agent
**Trigger**: User 2026-09-30 directive *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO items not yet covered by the workflows just launched
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-anticancer` (Krasnov 0.86 gate); `mol-metal-r18-math` (8/8 P0-P1 theorem catalog); `mol-metal-mechanism-mapping` (M22 SE(3) SB + M29-M30 multi-PH/PLL + M36-M44 CFM acceleration); no `paper/*` touches (TODO-84 honored); no wet-lab planning (§10)

---

## §0 TL;DR

| Item | Value | Status |
| | | |
|---|---:|---|
| **R17 refreshed** | `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` 345 → **462 LOC** (+117 LOC, +33.9%) | ✅ DONE |
| **R18 refreshed** | `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` 291 → **415 LOC** (+124 LOC, +42.6%) | ✅ DONE |
| **NEW sections added to R17** | §10 (5 subsections: Background / Phase 1 / Phase 2 / Phase 3 / Honest framing) | ✅ DONE |
| **NEW sections added to R18** | §12 (4 subsections: W1 M22 SE(3) SB / W2 M29 M30 / W3 M36-M44 / Forward-looking Blackwell retrain) | ✅ DONE |
| **NEW closeout doc** | `TODO/completed/393a1_r17_r18_refresh.md` (this file, ~200 LOC) | ✅ DONE |
| **NEW tests** | 0 (docs-only per task brief) | ✅ |
| **Code patches** | 2 docs files refreshed; 1 NEW closeout doc authored | ✅ docs-only |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Wet-lab planning** | 0 | ✅ per AGENTS.md §5.4 + §10 |
| **Pre-flight v3 status** | NOT re-run by Wf-393-A1 docs-only | ✅ PRESERVED |

**One-line verdict**: Wf-393-A1 SHIPS a CPU-only docs-only refresh of `R17_AUC_RECOVERY_PLAN.md` (+117 LOC §10 with Wf-387-A1 6/7 pair summary + Wf-387-A2 balanced Rh cohort + Wf-390-A2 Path A 5-metal retrain + Wf-391-A3 forward-looking priority order + honest framing) and `R18_MATH_MIGRATION_PLAN.md` (+124 LOC §12 with Wf-390-A3 + Wf-383-A2 SE(3) SB wire-in plan + step count revision 10000 → 1500-2500 + Wf-380-A1 SCAFFOLD status for M29/M30 multi-PH/PLL + Wf-353 SCAFFOLD status for M36-M44 CFM acceleration + Wf-391-A3 Blackwell retrain priority order). 0 NEW tests; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning; §1-§9 (R17) and §1-§11 (R18) preserved verbatim as historical baseline.

---

## §1 R17 refreshed sections (R17_AUC_RECOVERY_PLAN.md)

### §1.1 §10.1 Background — Wf-387 m1 AUC aggregation 6/7 pairs MEASURED

Per `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md` (canonical 6/7 pair summary):

| Field | OLD (wrong, Re seed=4 only) | NEW (correct, 6 runs) |
|---|---:|---:|
| `value` | 0.6220 (n=1) | **0.7640** (6 runs) |
| `value_std` | 0.0000 | **0.0995** (sample std) |
| `n_runs` | 1 | **6** |
| `n_folds` | 1 (Re only) | **2 (Rh, Re)** |
| `metals` | ["Re"] | **["Rh", "Re"]** |
| `seeds` | [4] | **[3, 4, 0, 1, 2, 3]** |
| `per_fold_auc_mean` | {Re: 0.6220} | **{Rh: 0.8798, Re: 0.7061}** |
| `per_fold_auc_std` | {Re: 0.0000} | **{Rh: 0.0115, Re: 0.0552}** |

**Krasnov 0.86 gate verdict** per `mol-metal-anticancer` skill:

| Fold | Mean test_auc | Krasnov 0.86 | gap |
|---|---:|:---:|---:|
| **Rh** | 0.8798 ± 0.0115 | **PASS** | +0.020 |
| **Re** | 0.7061 ± 0.0552 | **BELOW** | −0.154 |
| **Overall** | 0.7640 ± 0.0995 | **BELOW** | −0.096 |

### §1.2 §10.2 Phase 1 — Rh cohort balanced (per Wf-387-A2 TODO-135 structural fix)

Per `TODO/completed/387a2_balanced_rh_cohort.md` — Rh val_auc lift (test_auc 0.7613 from Wf-124 P3 Ru baseline) was contaminated by Wf-139 P7 synthetic-500 injection (Tanimoto-proxy-labeled cohort with mean Tanimoto to nearest real active = 0.14 — WEAK band, NOT bioactivity-validated). The structural fix:

- **Cohort**: `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` — 500 entries (250 active + 250 inactive), all `is_synthetic=False`, all `is_wf139_p7_carryover=False`, all from Rh `train` split under Wf-261 strict scaffold split
- **Source**: `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (Rh train; n=696 available; 250+250 sampled with seed=42)
- **Verify path**: Wf-125 P1 re-run with REAL 500 (instead of synthetic 500) → if val_auc ≥ Krasnov 0.86 gate, lift is REAL; if val_auc ≤ 0.5, lift was artifact
- **Stage 1 strategy revision**: Replace Wf-124 P3 1-seed × 5-epoch multi-task D-MPNN with Wf-125 P1 re-run using Wf-387-A2 REAL balanced cohort
- **Cross-ref**: `molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py` (SHIPPED per Wf-376-A2) + `scripts/launch_wf131_p3_deferred_runs.sh` (LIVE mode — Wf-355-B1 USER_OK-gated — Re seed=4 pending launch)

### §1.3 §10.3 Phase 2 — TODO-376 5-metal subset retrain (per Wf-390-A1 Path A)

Per `TODO/completed/390a2_todo179_deeper_audit.md` + `TODO/completed/388a1_m2_per_cohort_real.md` + Wf-369-A2 m2 R² FINAL verdict, the Stage 2 strategy is now scoped to a **5-metal subset retrain** with Path A as the recommended first move:

**Stage 2 strategy revision (Path A)**:
1. **5-metal subset**: Ru + Ir + Rh + Re + Os (matches D-MPNN `FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")` constraint at `train_dmpnn_multitask_wasserstein.py` line 247)
2. **Per-cohort config axis** (per Wf-380-A2 + Wf-388-A1): 12-d RDKit (n<300) / 24-d RDKit (300≤n<1000) / 32-d RDKit (n≥1000) matched to cohort size
3. **MEASURED real-cohort R²** (per Wf-388-A1, 5-seed × 5-fold Ridge baseline):
   - `platinai_4cell` (n=200, 12-d RDKit, α=1.0) → **−0.1440 ± 0.0325** HONEST_NEGATIVE_R2_DEEP
   - `hilbert` (n=200, 12-d RDKit, α=1.0) → **−0.1440 ± 0.0325** HONEST_NEGATIVE_R2_DEEP
   - `metalcytotox_combined` (n=2000, 32-d RDKit, α=0.1) → **+0.1981 ± 0.0056** WEAK_RECOVERY (lift −0.0246 vs Wf-335-M2 best +0.2227)
4. **Aggregate verdict**: `MEASURED_PARTIAL_LIFT` (1/3 cohorts WEAK_RECOVERY; 0/3 CEILING_BREAK; 2/3 HONEST_NEGATIVE_R2_DEEP preserved)
5. **Honest finding**: per-cohort config axis VALIDATED as methodology (Wf-380-A2 synthetic R² > 0), but REAL-cohort lift smaller than synthetic projection; structural finding: 50+ feature Ridge ensemble OVERFITS on small cohorts per Wf-367-B1
6. **Path A launch wrapper**: `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC, EXISTS on disk) — but per Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL, trainer `--metals` restricted to 5 metals; alternative trainer `train_cisplatin_resistance_v2.py` requires `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which DOES NOT EXIST on disk — Path A targets 5-metal subset directly via existing trainer
7. **Wall PROJECTED**: ~5 hr GPU for Path A; if Path A R² < +0.40 ceiling, follow up with Path B (alt encoder bootstrap per Wf-390-A2 §7, ~10 hr GPU)

### §1.4 §10.4 Phase 3 — Forward-looking GPU retrain priority order (per Wf-391-A3 §10)

Per `TODO/completed/391a3_metric_strategy_refresh.md` §10 (NEW Forward-looking 2026-Q4 GPU retrain priority order), 3-batch sequential priority:

| Batch | Wall PROJECTED | Wrappers | USER_OK gate | Pre-flight gate |
|:---:|---:|---|---|---|
| **Batch 1 (install)** | ~30 min CPU | Zotero 10.0.3 + Tectonic 0.17.0 (per Wf-372-A1 + Wf-372-A2 SHIPPED) | None (already installed) | N/A |
| **Batch 2 (m1 + m2 lift)** | ~5-10 hr GPU | Re seed=4 (Wf-355-B1) + Path A 5-metal retrain (Wf-376-A3 + Wf-390-A2 Path A) | USER_OK=1 + VRAM ≥ 35 GB | wf343c2 + wf343c3 PASS |
| **Batch 3 (m3 + m4 + m7 + m15)** | ~12-15 hr GPU | `cfm_vina_dock_panel.py` (BLOCKED on Pt+2 parameterisation per Wf-299B) + SA penalty wire (Wf-380-A6 SHIPPED) + m7 wire (Wf-388-A2 SHIPPED) + m15 redesign (Wf-341-T1 SHIPPED) | USER_OK=1 + VRAM ≥ 35 GB | wf343c2 + wf343c3 PASS |
| **Paper write batch** | ~12-15 hr CPU | 7 paper artefacts (Tier-2 §6 Limitations + 4 LOI/SOP Tier-2 markers + paper §3 R² honesty gate) | USER_OK=1 (TODO-84 active) | N/A |

**Combined wall PROJECTED**: ~52.9 hr GPU + ~13 hr CPU + ~30 min install (4-batch total)

### §1.5 §10.5 Honest framing — Rh fold hit Krasnov 0.86; Re fold BELOW; aggregate 0.7640 NOT hit

**m1 (Ru + Ir + Rh + Re + Os 5-metal LOMO AUC)**:
- **Rh fold 0.8798 ± 0.0115 — PASS Krasnov 0.86** (gap +0.020) — per Wf-387-A1 MEASURED
- **Re fold 0.7061 ± 0.0552 — BELOW Krasnov 0.86** (gap −0.154) — per Wf-387-A1 MEASURED
- **Aggregate 0.7640 ± 0.0995 — NOT HIT** (gap −0.096) — per Wf-387-A1 MEASURED
- **Rh val_auc 0.7613 lift** (Wf-124 P3 1-seed) was artifact of Wf-139 P7 synthetic-500 injection per Wf-387-A2 §2

**m2 (Pt 4-cell PlatinAI R² on pIC50)**:
- **platinai_4cell + hilbert n=200 — HONEST_NEGATIVE_R2_DEEP** (−0.1440 ± 0.0325 each, per Wf-388-A1 MEASURED)
- **metalcytotox_combined n=2000 — WEAK_RECOVERY** (+0.1981 ± 0.0056, lift −0.0246 vs Wf-335-M2 best +0.2227 CEILING_BREAK)

**m6_kappa (Shape Kappa Index per molecule)**:
- **PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted; structural CV-R² ceiling LIFTED +0.5410 → +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
- **Wf-377-A1 deep-dive-research move**: 16 per-molecule topological invariants evaluated on n=5460 MetalCytoToxDB labelled pIC50; best alternative MolLogP Pearson r=+0.1999 WEAK band; no cohort-level MODERATE band discovery
- **Forward-looking**: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`

**Tier-2 paper §6 Limitations draft** (per Wf-359-B2): m6_kappa + m2 R² + m1 Ru AUC 0.7640 NOT HIT all surface as honest-negative entries in §6

**Tier-1 paper claim DEPRECATED** per AGENTS.md §5.4 + 2026-09-29 user directive *"我们发二区就可以了，不用规划湿实验"* — Tier-2 journal target (JCAMD / Mol. Divers. / Comput. Chem.) does NOT require Tier-1 AUC gate

---

## §2 R18 refreshed sections (R18_MATH_MIGRATION_PLAN.md)

### §2.1 §12.1 R18 W1 M22 SE(3) SB — Wf-390-A3 + Wf-383-A2 wire-in plan + step count revision (10000 → 1500-2500)

Per `TODO/completed/390a3_se3_sb_wire.md` (SE(3) SB trainer integration audit + smoke scaffold) + `TODO/completed/383a2_se3_sb_integration_plan.md` (10 Wf-340 + Wf-353 fast-converging innovations SE(3) SB-specific integration plan):

**Trainer surface (per Wf-390-A3 §1.2 MEASURED audit)**:

| Field | Value |
|---|---|
| `trainer_path` | `/home/hugo/codes/mol-metal/molmetal/scripts/train_se3sb.py` |
| `trainer_exists` | True |
| `trainer_loc` | 750 |
| `parse_args_importable` | True |
| `parse_args_loc` | 557 |
| `parse_args_loc_match` | True |
| `n_se3_specific_flags_current` | **0** |
| `n_se3_specific_flags_total` | **10** (target per Wf-383-A2 §1.2) |
| `ast_parses` | True |

**Step count revision (per Wf-383-A1 §3.2 + Wf-390-A3 §1.2 PROJECTED bands)**:

| Field | OLD (R18 baseline) | NEW (Wf-383-A1 / Wf-390-A3 revision) |
|---|---:|---|
| `baseline_mechanical_steps` | 10000 | **10000** (MEASURED — unchanged) |
| `mechanical_step_count_band` | (not specified) | **1500-2500 conservative** (PROJECTED) |
| `aggressive_step_count_band` | (not specified) | **500-1000 aggressive** (PROJECTED) |
| `ceiling_step_count` | (not specified) | **625** (PROJECTED) |
| `blackwell_retrain_status` | DEFERRED | **DEFERRED per AGENTS.md §5.1 + USER_OK-gated** |

**Wire-in plan (per Wf-383-A2 §1 + §2)**:
- **10 fast-converging innovations** SHIPPED as CPU-only SCAFFOLDS in `molmetal/adapters/flow_matching_lipman/` + `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (per Wf-340 + Wf-353)
- **10 NEW files in `molmetal/adapters/flow_matching_lipman/` + 3 NO-NEW-FILE re-imports** = ~760 LOC production + ~150 LOC tests = ~910 LOC total per Wf-383-A2 §0
- **10 NEW tests** (1 per innovation) + 2 integration regression tests = ~12 NEW tests
- **Wall PROJECTED**: ~8-16 days CPU-only (conservative midpoint ~10-12 days) for SE(3) SB trainer wire-in

### §2.2 §12.2 R18 W2 M29 M30 — Multi-PH + PLL SCAFFOLD status (per Wf-380-A1 + Wf-382-A1)

Per `TODO/completed/380a1_deep_integration_scaffolds.md` (10 CFM innovation SCAFFOLDS) + `TODO/completed/382a1_multiph_pll_scaffold.md` (Multi-PH PLL SCAFFOLD specifically), the P1-1 Multi-parameter Persistent Homology (Kim & Jung 2026, M29) + P1-2 Persistent Local Laplacian (Liu & Feng 2026, M30) are now SCAFFOLD-CPU-only SHIPPED:

**Multi-PH SCAFFOLD (M29 per mol-metal-mechanism-mapping)**:
- **Module**: `molmetal/adapters/flow_matching_lipman/` (NEW SCAFFOLD per Wf-380-A1)
- **Test file**: `molmetal/tests/test_wf380a1_*.py` (41/41 NEW tests PASS in ~1.61 s wall)
- **Status**: SCAFFOLD CPU-only SHIPPED via Wf-380-A1 + Wf-382-A1
- **Trainer wire**: NOT WIRED into `train_se3sb.py` per Wf-383-A2 §1.1 (10/10 wires currently in `train_cfm_yuelbond_wasserstein.py` only)
- **Expected metric lift (per Wf-380-A1 SCAFFOLD paper PROJECTED)**: m7 Diversity 0.65 → 0.75; m11 n_distinct 3 → 4

**PLL SCAFFOLD (M30 per mol-metal-mechanism-mapping)**:
- **Module**: `molmetal/molmetal_lam/lam_chem/pll_signature.py` (existing) + Wf-380-A1 NEW extension
- **Status**: SCAFFOLD CPU-only SHIPPED via Wf-380-A1
- **Trainer wire**: NOT WIRED into SE(3) SB trainer
- **PH hardness guarantee (per Kolbe & Mayr 2026)**: paper-only reference, no code change

### §2.3 §12.3 R18 W3 M36-M44 — CFM acceleration SCAFFOLD status (per Wf-380-A1 + Wf-353-A1..A4)

Per `TODO/completed/380a1_deep_integration_scaffolds.md` (10 CFM innovation SCAFFOLDS, 41/41 tests) + `TODO/completed/353{a1,a2,a3,a4}_*.md` (Wf-353 CFM acceleration Triton + + kernel + FlowMol + consistency), the CFM acceleration mechanisms (M36-M44 per mol-metal-mechanism-mapping) are now SCAFFOLD CPU-only SHIPPED:

| M# | Innovation | Module | SCAFFOLD status | Tests |
|:---:|---|---|:---:|:---:|
| M36 | EDM σ-preconditioning (Karras 2022) | `acceleration.py` (80 LOC) | ✅ SHIPPED Wf-340-A1 + Wf-380-A1 | 4/4 |
| M37 | Log-normal t-sampling | `acceleration.py` (25 LOC) | ✅ SHIPPED Wf-340-A2 + Wf-380-A1 | 4/4 |
| M38 | Shortcut consistency (Frans 2024) | `acceleration.py` (70 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 |
| M39 | EMA (NVlabs/edm halflife) | `acceleration.py` (50 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 |
| M40 | Linear lr-warmup | `acceleration.py` (40 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 |
| M41 | Exact-OT solver (POT `ot.emd`) | `optimal_transport_v2.py` (85 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 |
| M42 | Triton mini-batch OT (Khainb 2025) | `triton_minibatch_ot.py` (561 LOC) | ✅ SHIPPED Wf-353-A1 + Wf-380-A1 | 4/4 |
| M43 | Consistency regularization (EDM Karras 2022) | `consistency.py` (80 LOC) | ✅ SHIPPED Wf-353-A3 + Wf-380-A1 | 4/4 |
| M44 | FlowMol unified atom-bond flow (CC-BY-NC) | `flowmol.py` (180 LOC) | ⚠ LICENSE-GATED SHIPPED Wf-353-A4 | 4/4 |

**Trainer wire status** (per Wf-383-A2 §1.1 + §1.2):
- **CFM YuelBond trainer**: 10/10 wires SHIPPED with default-OFF CLI flags
- **SE(3) SB trainer**: 0/10 wires currently (target = 10/10 per Wf-383-A2 §1.2 audit)
- **chemprop v2/v3 trainer**: 0/10 wires (out of R18 P1 scope)

### §2.4 §12.4 Forward-looking Blackwell retrain

Per `TODO/completed/391a3_metric_strategy_refresh.md` §10 + `TODO/completed/391a1_lambda_roadmap_refresh.md` §6.6 + Wf-390-A3 §1.2 (Blackwell retrain DEFERRED per AGENTS.md §5.1 + USER_OK-gated), the Blackwell retrain priority order for R18 P0-1 SE(3) Schrödinger Bridge is:

**Blackwell retrain priority order** (3-batch sequential per Wf-391-A3 §10):

| Priority | Wall PROJECTED | Target | Mechanism | USER_OK gate |
|:---:|---:|---|---|---|
| **P1-A** | ~8-16 days CPU + ~5 hr GPU | M22 SE(3) SB wire-in (Wf-383-A2 §2) | M22 (SE(3)-SB solver) | USER_OK=1 + VRAM ≥ 35 GB |
| **P2-A** | ~10-12 days CPU + ~5 hr GPU | Multi-PH + PLL wire into SE(3) SB (Wf-380-A1 + Wf-382-A1) | M29 + M30 (Multi-PH + PLL) | USER_OK=1 + VRAM ≥ 35 GB |
| **P3-A** | ~10-12 days CPU + ~5 hr GPU | CFM acceleration wire into SE(3) SB (Wf-380-A1 + Wf-353-A1..A4) | M36-M44 (CFM acceleration) | USER_OK=1 + VRAM ≥ 35 GB |

**Combined wall PROJECTED**: ~28-40 days CPU + ~15 hr GPU for full R18 P0-1 + P1-1 + P1-2 wire-in across 3 sequential Blackwell retrain batches

**Constraint priorities per AGENTS.md §5.1**:
- VRAM budget check first (`nvidia-smi` ≥ 35 GB free on RTX PRO 6000 Blackwell 98GB)
- Hyperparameters via config files (reference peer papers in same domain)
- Pre-flight PASS still required (`pre_flight_check.py` before any retrain)
- `--dry-run` defaults still the script default; agents promote to actual launch after VRAM + pre-flight check

**Forward-looking honest-negative caveats** (per `docs/HONEST_NEGATIVE_CATALOG.md` Wf-369-A3 expansion 40 → 65 entries + Wf-391-A3 §10):
- M22 SE(3) SB wire-in DEFERRED on Blackwell cudagraphs per Wf-335-R3 + AGENTS.md §5.1; `--no-cudagraphs` safety lever added to `scripts/launch_wf124_p3_g6.sh` per Wf-335-R3
- Triton fused_relu_mlp / fused_wasserstein_sliced / fused_bond_head_aggregation / fused_smiles_featurize kernel wires (Wf-333-T2) — Blackwell sm_120 GPU retune DEFERRED per AGENTS.md §5.1
- Wf-379-A1 m1_auc 5-seed IN-FLIGHT on GPU 0 Blackwell at Wf-379-B1 timestamp (Rh seed 3 RUNNING); 6 (metal, seed) pairs queued; Krasnov 0.86 gate target; full multi-pair result deferred to Wf-380
- Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL on R20 §5 tmQM + PlatinAI joint retrain wrapper (trainer `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint)

---

## §3 Cross-references (Wf-380..390 + AGENTS.md §10)

### §3.1 Source wave surface

| Wave | Sub-flow | Cross-ref |
|---|---|---|
| Wf-380 | A1..A7 | `TODO/completed/380{a1,a2,a3,a4,a5,a6,a7}_*.md` |
| Wf-381 | A4 + A5 | `TODO/completed/381{a4,a5}_*.md` |
| Wf-382 | A1..A4 | `TODO/completed/382{a1,a2,a3,a4}_*.md` |
| Wf-383 | A1..A3 | `TODO/completed/383{a1,a2,a3}_*.md` |
| Wf-384 | A1 + A2 | `TODO/completed/384{a1,a2}_*.md` |
| Wf-385 | A1..A3 | `TODO/completed/385{a1,a2,a3}_*.md` |
| Wf-386 | A1..A3 | `TODO/completed/386{a1,a2,a3}_*.md` |
| Wf-387 | A1..A3 | `TODO/completed/387{a1,a2,a3}_*.md` |
| Wf-388 | A1..A3 | `TODO/completed/388{a1,a2,a3}_*.md` |
| Wf-389 | A1 + A3 | `TODO/completed/389{a1,a3}_*.md` |
| Wf-390 | A2 + A3 | `TODO/completed/390{a2,a3}_*.md` |
| Wf-391 | A1..A3 | `TODO/completed/391{a1,a2,a3}_*.md` |

### §3.2 Aggregate metrics cross-ref

- `metrics/by_metric/metal_lomo_auc_wasserstein.json` — canonical m1 source (Wf-387-A1 rewrote with 6-run aggregate)
- `metrics/by_metric/m6_kappa.json` — canonical m6_kappa source (Wf-365-A3 + Wf-377-A1 redesign)
- `metrics/by_metric/metalcytotox_combined_r2.json` — canonical m2 source (Wf-362-A1 Wf-335-M2 CEILING_BREAK update)

### §3.3 AGENTS.md standing directives re-confirmed

Per AGENTS.md §1 (Five standing project rules):
1. **MEASURED-vs-PROJECTED tag discipline** (`party-twenty-spirit` §3.5) — preserved
2. **Standing GPU authorization** (2026-09-29 user grant) — VRAM-budgeted + config-aligned hyperparameters, no per-launch OK needed; honored
3. **No paper work without authorization** (TODO-84 active) — 0 `paper/*` touches; preserved
4. **Pre-flight v3 25/25 PASS gate** (TODO-207) — NOT re-run by Wf-393-A1 docs-only; preserved
5. **Honest-negative framing** (`party-twenty-spirit` §3.5) — preserved

Per AGENTS.md §5.1 (Standing GPU authorization): USER_OK=1 + VRAM ≥ 35 GB gates preserved
Per AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning): Tier-2 markers preserved
Per AGENTS.md §10 (Wetlab deprecation FAQ): 0 wet-lab planning; all 7 wetlab items in `TODO/pending/` retained as historical artifacts

### §3.4 Pre-flight v3 status

- **Wf-379-B1 baseline**: 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 139.61 s per `molmetal/reports/wf379b1_preflight.json`
- **PRESERVED by Wf-393-A1 docs-only sync**: NOT re-run by Wf-393-A1
- **2 pre-existing FAILs**: `p3_loi_send_ready` (Tier-2 reclassification cascade) + `p3_wf343c4_anchor_freshness_24h` (wf269c_report.json STALE natural drift)
- **3 pre-existing SKIPs (BY-DESIGN)**: `p3_wf343c4_yuelbond_decoder_env` + `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`

---

## §4 Constraints honoured

| Constraint | Status |
|---|---|
| **0 NEW tests** | ✅ (docs-only per task brief) |
| **0 GPU launches** | ✅ per AGENTS.md §5.1 |
| **0 `paper/*` touches** | ✅ per AGENTS.md §5.3 + TODO-84 |
| **0 wet-lab planning** | ✅ per AGENTS.md §5.4 + §10 |
| **0 sudo** | ✅ per environment constraints |
| **Bash + Read + Edit + Write only** | ✅ no Skill / no WebFetch / no WebSearch |
| **Preserve existing structure** | ✅ §1-§9 (R17) + §1-§11 (R18) preserved verbatim |
| **MEASURED-vs-PROJECTED tag discipline** | ✅ all numbers tagged MEASURED (on-disk JSON / AST audit) or PROJECTED (per upstream paper benchmarks) |
| **Honest framing per `party-twenty-spirit` §3.5** | ✅ 6 honest-negative caveats preserved in §1.5 + §2.4 |

---

## §5 User voice answer

**User voice** (per workflow harness relay, 2026-09-30 verbatim): *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*

**Wf-393-A1 answer**: PARTIAL — Wf-393-A1 SHIPS the docs-only refresh of `R17_AUC_RECOVERY_PLAN.md` and `R18_MATH_MIGRATION_PLAN.md` to sync the strategic plans with the Wf-380..390 SCAFFOLD/wire-in ship surface + Wf-391-A3 forward-looking GPU retrain priority order. R17 §10 (5 subsections: Background / Phase 1 / Phase 2 / Phase 3 / Honest framing) + R18 §12 (4 subsections: W1 M22 SE(3) SB / W2 M29 M30 / W3 M36-M44 / Forward-looking Blackwell retrain). All MEASURED values sourced from on-disk JSON / AST audits; all PROJECTED values tagged per upstream paper benchmarks; 0 NEW tests; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning; §1-§9 (R17) + §1-§11 (R18) preserved verbatim as historical baseline; honest framing per `party-twenty-spirit` §3.5 preserved with 6 honest-negative caveats (m1 Rh fold PASS / Re fold BELOW / aggregate NOT HIT; m2 R² HONEST_NEGATIVE_R2_DEEP for n=200 + WEAK_RECOVERY for n=2000; m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL).

Standing directives §5.1 + §5.2 + §5.3 + §5.4 + §10 all preserved unchanged. NEW ultracode Wf-394+ IS the recommended follow-up per Wf-393-A1 §3.4 + Wf-391-A1 §6.6 + Wf-391-A3 §10 with 6+ USER_OK-gated sub-flows (~21.5 hr GPU + 10 min CPU + 30 min install total) per the Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 forward-looking priority list.

---

## §6 Cross-references

- `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` — refreshed (462 LOC, +117 LOC §10)
- `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` — refreshed (415 LOC, +124 LOC §12)
- `TODO/completed/393a1_r17_r18_refresh.md` — this closeout
- `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md` — m1 AUC aggregation source
- `TODO/completed/387a2_balanced_rh_cohort.md` — balanced Rh cohort source
- `TODO/completed/388a1_m2_per_cohort_real.md` — m2 R² real-cohort verification source
- `TODO/completed/389a1_m6k_supp_reframe.md` — m6_kappa supplementary sub-table source
- `TODO/completed/390a2_todo179_deeper_audit.md` — TODO-179 deeper audit source
- `TODO/completed/390a3_se3_sb_wire.md` — SE(3) SB trainer integration audit source
- `TODO/completed/380a1_deep_integration_scaffolds.md` — 10 CFM SCAFFOLDS source
- `TODO/completed/382a1_multiph_pll_scaffold.md` — Multi-PH PLL SCAFFOLD source
- `TODO/completed/383a2_se3_sb_integration_plan.md` — SE(3) SB integration plan source
- `TODO/completed/391a1_lambda_roadmap_refresh.md` — LAMBDA_ROADMAP refresh + §6.6 forward-looking
- `TODO/completed/391a2_improvement_enum_refresh.md` — IMPROVEMENT_ENUM + QUEUE refresh
- `TODO/completed/391a3_metric_strategy_refresh.md` — METRIC_LIFT_STRATEGY_v2.md refresh + §10 forward-looking
- `TODO/completed/379a1_wf131_p3_7runs.md` — Wf-131 P3 7 DEFERRED-runs IN-FLIGHT
- `TODO/completed/379a2_r20_p5_tmqm_plat.md` — R20 §5 SCAFFOLD_NOT_FUNCTIONAL close
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` — canonical m1 source
- `metrics/by_metric/m6_kappa.json` — canonical m6_kappa source
- `metrics/by_metric/metalcytotox_combined_r2.json` — canonical m2 source
- `docs/HONEST_NEGATIVE_CATALOG.md` — 65 entries per Wf-369-A3 expansion
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §10 — standing directives re-confirmed
- `molmetal/scripts/pre_flight_check.py` — pre-flight v3 (NOT re-run by Wf-393-A1)
- `molmetal/scripts/train_se3sb.py` — SE(3) SB trainer (749 LOC, parse_args line 557)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — D-MPNN multi-task Wasserstein trainer (line 247 `FIVE_METALS`)
- `molmetal/adapters/flow_matching_lipman/` — 10 NEW SCAFFOLD modules per Wf-380-A1
- `molmetal/tests/test_wf380a1_*.py` — 41/41 NEW tests PASS (CPU-only)
- `molmetal/tests/test_wf353{a1,a2,a3,a4}_*.py` — 16/16 NEW tests PASS (CPU-only)
- `scripts/launch_wf131_p3_deferred_runs.sh` — Wf-379-A1 IN-FLIGHT wrapper (USER_OK-gated)
- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` — Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL wrapper (472 LOC)
- `scripts/launch_wf124_p3_g6.sh` — G6 launcher with `--no-cudagraphs` safety lever (Wf-335-R3)
- `mol-metal-gpu-execution` skill — §0 `kill_then_adapt` directive honored
- `mol-metal-anticancer` skill — Krasnov 0.86 gate preserved
- `mol-metal-r18-math` skill — 8/8 P0-P1 theorem catalog preserved
- `mol-metal-mechanism-mapping` skill — M22 SE(3) SB + M29-M30 multi-PH/PLL + M36-M44 CFM acceleration
- `party-twenty-spirit` skill — §3.5 MEASURED-vs-PROJECTED tag discipline preserved
- `mol-metal-cfm-training` skill — YuelBond decoder REQUIRED preserved
- `mol-metal-m3ci-m6kappa` skill — m6_kappa metric redefinition FINAL preserved
