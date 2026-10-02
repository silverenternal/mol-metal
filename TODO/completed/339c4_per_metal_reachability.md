# Wf-339-C4 — Per-metal reachability matrix update (M25 + 30 metal + OOD)

**Date**: 2026-09-29 (Wf-339-C4 ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ **SHIPPED** (4 NEW tests + 4 bonus invariants + closeout doc; 8/8 PASS)
**Skill alignment**: `mol-metal-closure-theorem` + `mol-metal-mechanism-mapping` §Per-metal reachability matrix + `party-twenty-spirit` §3.5
**Parent**: `TODO/completed/207_unified_pre_flight_checklist_v2.md` + `molmetal/reports/wf37_closure_dblock/per_metal_bench.md` (Wf-37 P2 / L6) + `molmetal/reports/wf41_closure_ood/ood_closure.md` (Wf-41 P2 / L12)
**Sibling**: `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 (Wf-313G HONEST_NEGATIVE for TODO-122 L5) + `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (Wf-94 R20-P2 architectural fix)

---

## §0 TL;DR

Wf-339-C4 ships a **per-metal reachability matrix update** for the closure theorem d-block + cross-domain OOD benches, recording MEASURED per-metal baselines (L6 + L12) + PROJECTED M25 Hilbert-ASTSR attribution (per honest framing §3.5). The matrix is now testable: 8 tests in `molmetal/tests/test_wf339c4_per_metal_reachability.py` (4 NEW + 4 bonus invariants) PASS in 0.14 s wall on CPU-only.

**Honest framing (party-twenty-spirit §3.5)**:
- **MEASURED today**: per-metal L6 (30 d-block metals, Wf-37 P2) + per-metal L12 (PlatinAI 5K + MetalCytoToxDB 5-metal, Wf-41 P2) + per-periodic-block split (3d / 4d / 5d_La).
- **PROJECTED**: M25 Hilbert-ASTSR attribution (L5 R² ≥ 0.85 NOT YET MEASURED — 12-d RDKit 2D signal floor per Wf-313G HONEST_NEGATIVE; Wf-94 R20-P2 Multi-PH + PLL is the architectural fix path).

---

## §1 Per-metal MEASURED baseline (L6 d-block)

**Source**: `molmetal/reports/wf37_closure_dblock/per_metal_bench.md` §Per-metal reachability rate (Wf-37 P2, 2026-09-26)

| Metal | n_samples | n_closure_ok | closure_reachability_rate | metal_coord_rate |
|---|---:|---:|---:|---:|
| Pt | 100 | 47 | **0.4700** | 1.0000 |
| Pd | 100 | 45 | **0.4500** | 1.0000 |
| Ru | 100 | 72 | **0.7200** | 1.0000 |
| Rh | 100 | 55 | **0.5500** | 1.0000 |
| Ir | 100 | 38 | **0.3800** | 1.0000 |
| Os | 55  | 22 | **0.4000** | 1.0000 |
| Au | 100 | 44 | **0.4400** | 1.0000 |
| Ag | 81  | 50 | **0.6173** | 1.0000 |
| Re | 100 | 54 | **0.5400** | 1.0000 |

**Test pinning**: `test_per_metal_l6_baseline` reads `per_metal_bench.json` + `per_metal.csv` and asserts each of the 9 task-spec metals' value within 1e-3 tolerance + bounded in [0, 1].

---

## §2 Per-(domain, metal) MEASURED baseline (L12 OOD)

**Source**: `molmetal/reports/wf41_closure_ood/ood_closure.md` §Per-(domain, metal) reachability rate (Wf-41 P2, 2026-09-26)

### §2.1 PlatinAI 5K (Tanimoto<0.4 OOD filter against tmQM)

| Metal | n_input | n_ood | closure_ok | reachability_rate | ood_rate |
|---|---:|---:|---:|---:|---:|
| Pt | 20 | 12 | 7 | **0.5833** | 0.6000 |
| Pd | 20 | 9  | 3 | **0.3333** | 0.4500 |
| Ru | 20 | 10 | 7 | **0.7000** | 0.5000 |
| Rh | 20 | 17 | 11 | **0.6471** | 0.8500 |
| Ir | 20 | 10 | 6 | **0.6000** | 0.5000 |
| Os | 20 | 17 | 9 | **0.5294** | 0.8500 |
| Au | 20 | 11 | 5 | **0.4545** | 0.5500 |

### §2.2 MetalCytoToxDB 5-metal (Tanimoto<0.4 OOD filter against tmQM)

| Metal | n_input | n_ood | closure_ok | reachability_rate | ood_rate |
|---|---:|---:|---:|---:|---:|
| Ru | 20 | 20 | 3 | **0.1500** | 1.0000 |
| Ir | 20 | 20 | 0 | **0.0000** | 1.0000 |
| Rh | 20 | 20 | 0 | **0.0000** | 1.0000 |
| Os | 20 | 20 | 0 | **0.0000** | 1.0000 |
| Re | 20 | 20 | 0 | **0.0000** | 1.0000 |

**Test pinning**: `test_per_metal_ood_baseline` reads `ood_closure.json` and asserts each (domain, metal) value within 1e-3 + bounded in [0, 1] + MetalCytoToxDB 5-metal Ir/Rh/Os/Re zero rates are documented honest-negative (NOT closure theorem bug; per `ood_closure.md` §Honest framing, valence-BNF rejection on `[M+]<-` notation that MLC ledger treats as under-saturated).

---

## §3 M25 Hilbert primal-dual attribution — PROJECTED

**Per party-twenty-spirit §3.5**, the M25 Hilbert-ASTSR attribution is **PROJECTED** (NOT YET MEASURED). This section deliberately does NOT fabricate Hilbert-augmented reachability numbers.

**Architectural block (per `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 + `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md`)**:
- **12-d RDKit 2D feature ceiling**: |ρ| ≤ 0.019 on PlatinAI pIC50 → linear R² upper bound = (0.019)² · 12 = 0.0043
- Hence **R² ≥ 0 is mathematically infeasible** on the 12-d feature stack — M25 Hilbert attribution alone cannot lift reachability above this ceiling
- **MEASURED today** (Wf-122 P3 + Wf-129 P3):
  - L5 Hilbert R² 4-cell: -5.66 ± 1.36 (TODO-122 P3 post-fix combined anchor; n_seeds_total=20; PlatinAI 4-cell strict-scaffold)
  - Wf-129 P3 Multi-PH + PLL guarded: -0.0780 ± 0.0148 (better than -5.66 but still below LAMBDA_ROADMAP §4.2 L7 + R20-P3 gate R² ≥ +0.30)
- **Architectural fix path**: **Wf-94 R20-P2 Multi-PH + PLL** (TODO-93) — 32-bin Fourier + 128-d PLL signature, NOT zero-padded; the only path that lifts R² above the 12-d signal floor
- **Test pinning**: `test_m25_hilbert_attribution` asserts status == "PROJECTED" (NOT "MEASURED") + target R² == 0.85 + note cites Wf-313G dashboard + Wf-94 R20-P2 + 12-d signal floor + Multi-PH + PLL

---

## §4 Per-periodic-block MEASURED split (L6 d-block)

**Source**: `molmetal/reports/wf37_closure_dblock/per_metal_bench.md` §Per-periodic-block aggregate (Wf-37 P2, 2026-09-26)

| Block | n_samples | n_closure_ok | closure_reachability_rate | mean_wall_s |
|---|---:|---:|---:|---:|
| 3d | 892 | 596 | **0.6682** | 0.089581 |
| 4d | 683 | 397 | **0.5813** | 0.077274 |
| 5d_La | 669 | 329 | **0.4918** | 0.060771 |
| **Total** | 2244 | 1322 | **0.5891** | — |

**Trend (MEASURED, NOT PROJECTED)**: reachability decreases with period — 3d (0.6682) > 4d (0.5813) > 5d_La (0.4918). Consistent with the valence-BNF rejection pattern documented in `per_metal_bench.md` §Honest framing.

**Test pinning**: `test_per_periodic_block_split` reads `per_metal_bench.json` §per_block, asserts each block value within 1e-3 + partition into 2244 total samples + 3d > 4d > 5d_La trend invariant.

---

## §5 Per-metal reachability matrix (testable)

The 4 NEW tests + 4 bonus invariants form a self-documenting matrix:

| Test | What it pins | Source artefact | Honest framing |
|---|---|---|---|
| `test_per_metal_l6_baseline` | 9 task-spec metals × 30 d-block bench | `per_metal_bench.md` + `per_metal.csv` + `per_metal_bench.json` | **MEASURED** (Wf-37 P2) |
| `test_per_metal_ood_baseline` | 7 PlatinAI metals + 5 MetalCytoToxDB metals × OOD bench | `ood_closure.md` + `ood_closure.json` | **MEASURED** (Wf-41 P2) |
| `test_m25_hilbert_attribution` | M25 Hilbert-ASTSR target R² ≥ 0.85 + PROJECTED status | `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 | **PROJECTED** (per §3.5) |
| `test_per_periodic_block_split` | 3d / 4d / 5d_La partition + 2244 total + monotonic trend | `per_metal_bench.md` §Per-periodic-block aggregate | **MEASURED** (Wf-37 P2) |
| `test_per_metal_csv_consistent_with_json` | CSV ↔ JSON round-trip sanity | both ship artefacts | **MEASURED** |
| `test_md_artefacts_present` | Markdown summaries on disk | `per_metal_bench.md` + `ood_closure.md` | **MEASURED** |
| `test_l6_l12_metal_coverage_disjoint` | L12 (12 metals) ⊂ L6 (30 metals) | both ship artefacts | **MEASURED** |
| `test_ood_metalcyto_zero_rates_documented` | MetalCytoToxDB 5-metal Ir/Rh/Os/Re 0.0 honest-negative | `ood_closure.md` §Honest framing | **MEASURED** (NOT closure bug) |

**Run with**:
```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
/home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
    molmetal/tests/test_wf339c4_per_metal_reachability.py -v
```

**Result**: 8 passed in 0.14 s wall on CPU-only (2026-09-29 13:55 CST).

---

## §6 Standing rules honored

1. **MEASURED vs PROJECTED tag discipline** (§3.5): every numeric value in the matrix traces to a JSON / report path on disk; L6 + L12 + per-periodic-block = MEASURED; M25 Hilbert attribution = PROJECTED (explicitly flagged).
2. **Standing GPU authorization** (§5.1): N/A — this Wf is CPU-only documentation + tests; no GPU launches.
3. **No paper work without authorization** (§5.3 + TODO-84): no `paper/*` touches; this is a test + closeout doc only.
4. **Pre-flight v2 gate** (TODO-207): N/A for this Wf (no retrain / panel / ablation launch); pre-flight v3 baseline 42 PASS / 15 FAIL / 57 TOTAL HONEST-NEGATIVE REGRESSION preserved (per AGENTS.md §6 Wf-324D 2026-09-29).
5. **Honest-negative framing** (§3.5): M25 Hilbert attribution flagged PROJECTED with explicit citation of Wf-313G HONEST_NEGATIVE + Wf-94 R20-P2 fix path; MetalCytoToxDB 5-metal zero rates documented as valence-BNF rejection (NOT closure theorem bug).

---

## §7 Cross-references

- `molmetal/reports/wf37_closure_dblock/per_metal_bench.md` (Wf-37 P2 / L6 source — 30 d-block metals)
- `molmetal/reports/wf37_closure_dblock/per_metal_bench.json` (Wf-37 P2 / L6 source JSON)
- `molmetal/reports/wf37_closure_dblock/per_metal.csv` (Wf-37 P2 / L6 CSV)
- `molmetal/reports/wf41_closure_ood/ood_closure.md` (Wf-41 P2 / L12 source — PlatinAI 5K + MetalCytoToxDB 5-metal)
- `molmetal/reports/wf41_closure_ood/ood_closure.json` (Wf-41 P2 / L12 source JSON)
- `molmetal/reports/wf41_closure_ood/per_domain.csv` (Wf-41 P2 / L12 CSV)
- `molmetal/tests/test_closure_dblock.py` (Wf-37 P2 / L6 test surface — 20 tests)
- `molmetal/tests/test_closure_ood.py` (Wf-41 P2 / L12 test surface — 15 tests)
- `molmetal/tests/test_closure_ood_30metals.py` (Wf-130 P3 OOD 30-metals test surface)
- `molmetal/tests/test_wf339c4_per_metal_reachability.py` (Wf-339-C4 NEW test surface — 8 tests)
- `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4 (Wf-313G HONEST_NEGATIVE for TODO-122 L5)
- `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (Wf-139 P4 architectural ceiling close-out)
- `TODO/completed/129_wf129_p3_multiph_pll_guarded.md` (Wf-129 P3 Multi-PH + PLL guarded retrain)
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (Wf-94 R20-P2 — architectural fix path)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` (Wf-94 R20-P3 Multi-PH + PLL feature)
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §Per-metal reachability matrix
- `mol-metal-closure-theorem` skill (closure theorem main result + Catherine 6/6 properties + per-metal reachability ≥ 0.95)
- `mol-metal-mechanism-mapping` skill §Per-metal reachability matrix
- `party-twenty-spirit` §3.5 (MEASURED vs PROJECTED tag discipline)

---

## §8 User feedback addressed: other pending TODO tasks NOT started by ultracode

Per user 2026-09-29 side-channel directive: "好，但你todo目录里应该还有别的任务当前启动的ultracode没有启动到的" (trans: "OK, but in your TODO directory there should still be other tasks that the currently-launched ultracode hasn't started on").

### §8.1 PENDING tasks in `TODO/pending/` not yet actioned (after Wf-339-C4)

| File | Status | Reason NOT started |
|---|---|---|
| `179_wf123_p3_w2_pic50_alignment.md` | ⏸️ STATUS_HOLD | Per Wf-316A audit 2026-09-29: W₂ recipe is **architecturally infeasible** due to 12-d RDKit 2D signal floor (Wf-313G HONEST_NEGATIVE captured in `TODO/completed/177_wf313g_downstream_lift_dashboard.md` §3.4). Blocked on Wf-94 R20-P3 architectural fix. |
| `182_wf124_p3_per_innovation_ablation.md` | ⏸️ STATUS_HOLD | Per Wf-316A audit 2026-09-29: 60-run ablation matrix = 7.5 hr wall > 1 hr kill_then_adapt threshold → requires explicit fresh user OK. Defer to GPU window per `mol-metal-gpu-execution` §0. |
| `84_no_paper_task_without_authorization.md` | ✅ Standing rule (160 LOC) | Active — no action needed; prohibits paper work without user authorization. |
| `wetlab_*.md` × 10 | 🟡 DEPRECATED per §5.4 (Tier-2 strategy) | Per AGENTS.md §5.4 (user directive 2026-09-29 "我们发二区就可以了，不用规划湿实验"): 4 LOI drafts + 3 SOPs + cold-email + IC50 pipeline + LOI consolidation review are `TEMPLATE_NOT_NEEDED_FOR_TIER2` (pure computational Tier-2 acceptance; no wet-lab data required). |

### §8.2 PENDING P0 tasks in `TODO/03_plans/` not yet actioned

Per `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §0 + `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` §0:

| Task | Source | Priority | Wall | Status | Honest framing |
|---|---|---|---:|---|---|
| **TODO-134** m15 Docking RMSD redesign (3 phases) | `TODO/pending/134_m15_docking_rmsd_redesign.md` | **P0** | 5-8 days | ⚪ QUEUED | 5/5 pocket × 3 seed × 5 poses = 75 docked poses; per-pocket median best RMSD ≤ 2.5 Å SOTA gate |
| **TODO-135 Phase 2** m19 LOMO 5-metal × 5-seed × 50-epoch re-bench | `TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md` §2 | **P0** | 2 weeks | ⚪ QUEUED | Rh fold AUC ≥ 0.30 gate; LOMO mean ≥ 0.60; depends on Rh balanced cohort from Wf-139 P7 (DONE 2026-09-27); BLOCKED on user OK (GPU) |
| **TODO-138** Paper v2 → Tier-2 journal submission package | `TODO/pending/138_paper_v2_submission_package.md` | **P0** | 5-7 days | ⚪ QUEUED | paper builds clean + cover letter + highlights + suggested reviewers + `submission_package.zip`; per AGENTS.md §5.4 + Wf-317D (2026-09-29), re-scoped from JCIM Tier-1 to Tier-2 journal target (JCAMD / WIREs Comput. Mol. Sci / Mol. Divers. / J. Mol. Graph. Model.) |

### §8.3 PENDING tasks in `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` not yet actioned

Per Wf-310A closeout + Wf-310F plan:

| Phase | Wall | Status | Notes |
|---|---:|---|---|
| Phase 0 P0 blockers (3 loader modules + BindingDB/PlatinAI download + CPU SMILES featurization fix) | 1 week CPU | ⚪ QUEUED | `molmetal/data/drugood_loader.py` + `binding_db.py` + `nci60.py` not yet wired |
| Phase 1 Wf-146 LOMO completion (24/25 remaining) | 1 week GPU | ⚪ QUEUED | 1/25 done (Rh seed=0 = 0.5256); 24/25 remaining; BLOCKED on user OK (GPU) |
| Phase 1 Wf-131 P3 D-MPNN LOMO completion (7/25 remaining) | 92 min GPU | ⚪ QUEUED | 18/25 done; Re 0/5; BLOCKED on user OK (GPU) |
| Phase 2 DrugOOD 1.58M full chemprop v2 retrain | 1 week GPU | ⚪ QUEUED | BLOCKED on user OK (GPU) |
| Phase 3 BindingDB 1.07M metal-filtered chemprop v2 | 1 week GPU | ⚪ QUEUED | BLOCKED on user OK (GPU) |
| Phase 4 tmQM 100K + PlatinAI 500 joint retrain | 1 week GPU | ⚪ QUEUED | BLOCKED on user OK (GPU) |
| Phase 5 M22-M30 mechanisms + 4 joint retrain | 2 weeks GPU | ⚪ QUEUED | BLOCKED on user OK (GPU) |
| Phase 6 Wf-138 paper v2 submission package | 1 week CPU | ⚪ QUEUED | Same as TODO-138 above |

### §8.4 PENDING tasks in `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` not yet actioned

Per Wf-310A + Wf-139 P4 + Wf-139 P5/P6 + Wf-310H closeouts:

| Metric | Status | Honest framing |
|---|---|---|
| **m1 Ru AUC** | Wf-310A 50-epoch 5-metal × 3-seed MEASURED 0.7227 ± 0.0417 | Wf-310A closeout `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md` |
| **m3 Vina Dock median** | ⏳ PROJECTED (R19 GPU retrain) | R19 ACTUAL ≤ -10.5; Wf-42 R20-P4 next escalation |
| **m4 Success Rate** | cascades from m3 | ⏳ PROJECTED (R19) |
| **m7 Diversity** | Wf-139 P6 closure-bias SA generation MEASURED (aggregate SA = 2.463) | Wf-139 P6 closeout `TODO/completed/142_wf139_p6_sa_closure_bias_generation.md` |
| **m8-prod PB-Valid production** | UNDEFINED | ⏳ Phase 1 F2(a) + Phase 2 MMFF94 (R18 GPU retrain) |
| **m10 Vina Min kcal/mol** | ⏳ PROJECTED (R19 GPU retrain) | R19 ACTUAL ≤ -10.5; Wf-42 R20-P4 |
| **m16 Affinity Pearson r** | ⏳ PROJECTED (Wf-94 R20-P2 + L5) | L5 R² ≥ 0.85 NOT MEASURED; Wf-94 R20-P2 architectural fix path |
| **m19 Multi-metal LOMO** | Wf-131 P3 partial + Wf-135 Phase 1 cohort DONE; Wf-135 Phase 2 ⏸️ QUEUED (GPU) | BLOCKED on user OK (2 weeks GPU) |

### §8.5 Recommendation (per `party-twenty-spirit` §3.5)

Per AGENTS.md §5.1 (standing GPU authorization 2026-09-29), no per-launch user OK required for >1 hr GPU launches provided:
- **VRAM budget check first** (`nvidia-smi`)
- **Pre-flight v2 PASS** (`pre_flight_check.py`)
- **Hyperparameters via config files** (reference peer papers)

**Recommend launching** (in priority order, per `IMPROVEMENT_QUEUE_2026_09_27.md` §2 串行顺序):
1. **TODO-138** Paper v2 → Tier-2 submission package (1 week CPU; user-friendly Tier-2 path; no GPU required; aligns with AGENTS.md §5.4 Tier-2 strategy)
2. **TODO-134** m15 Docking RMSD redesign (5-8 days; partial CPU + partial GPU-0; not blocking)
3. **TODO-135 Phase 2** m19 LOMO 5-metal × 5-seed × 50-epoch (2 weeks GPU; BLOCKED on user OK)

OR alternatively, per R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §0 6-phase plan, run **Phase 0 (1 week CPU)** first to unblock BindingDB / PlatinAI download gaps + 3 loader modules.

---

**Honest framing disclaimer (party-twenty-spirit §3.5)**: This Wf-339-C4 ship records per-metal reachability matrix as MEASURED (L6 d-block + L12 OOD + per-periodic-block) + M25 Hilbert-ASTSR attribution as PROJECTED. Every numeric value traces to a JSON / report path on disk. Wf-94 R20-P2 / TODO-93 is the architectural fix path for L5 R² ≥ 0.85; until that ships, do NOT cite Hilbert-augmented reachability as MEASURED.
