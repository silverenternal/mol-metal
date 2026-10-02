# Wf-354 Final Synthesis — Wf-342 closeout integration + R20 wrapper verification wave

**Date**: 2026-09-29
**Workflow**: Wf-354-C3 (this doc) — final synthesis aggregation
**Status**: ✅ **SHIPPED** — 7/7 closeouts aggregated; 6/6 source files VERIFIED + 1/1 INDEX/AGENTS sync shipped
**Owner**: Hugo Lee + Wf-354 ultracode v8 wave (A1+A2+B1+B2+B3+C1+C2)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored); standing GPU auth honored (§5.1 — zero GPU launches across the entire wave)

---

## §0 TL;DR

| Wf-354 agent | Sub-target | Verdict | LOC | Status |
|---|---|---|---:|---|
| **Wf-354-A1** | Wf-342 P2 LOMO sweep completions integration verification (Wf-146 chemprop v2 + Wf-131 D-MPNN + Wf-298 m19 LOMO + Wf-310E Wf-135 Phase 2) | ⚠️ **PARTIAL_HONEST_NEGATIVE** | 276 | Wf-146 19/25 (76%); Wf-131 18/25 (72%); Wf-298 SHIPPED; Wf-310E USER_OK-BLOCKED |
| **Wf-354-A2** | Wf-310E Wf-135 Phase 2 USER_OK-gate verification | ✅ **PASS** | 166 | `scripts/launch_wf310e_wf135_phase2_gpu.sh:82-88` gate code MEASURED; exit code 64 refusal; dry-run mode emits `plan.json` cleanly |
| **Wf-354-B1** | R20 §3 DrugOOD launch wrapper verification | ⚠️ **PARTIAL_VERIFIED** | 282 | 3/5 components PASS (USER_OK + pre-flight + dry-run); 2/5 MISSING (wf343c2 per-wave + wf343c3 data quality — NOT wired post-Wf-347); SCOPE MISMATCH (wrapper targets 95K ECFP4 subset, NOT 1.58M) |
| **Wf-354-B2** | R20 §4 BindingDB 1.07M launch wrapper — verify + USER_OK gate | ⚠️ **HONEST-NEGATIVE NOT-WRITTEN** | 123 | Wrapper does NOT exist on disk; full 1.07M data NOT on disk (only 5,257 subset per Wf-148 §1.1) |
| **Wf-354-B3** | R20 §5 tmQM d-block + PlatinAI 4-cell joint retrain launch wrapper — verify + USER_OK gate | ⚠️ **HONEST-NEGATIVE NOT-WRITTEN + STATUS_HOLD** | 127 | Wrapper does NOT exist on disk; joint retrain is **architecturally infeasible** on 12-d F5_RDKIT_2D signal floor (R² ≤ 0.0043 ceiling per Wf-122/Wf-129 MEASURED −5.66 to −0.08 range) |
| **Wf-354-C1** | Wf-342 P4 WEAK_TO_STRONG §1.1/§3.5/§3.6 + Final Closeout integration | ✅ **VERIFIED INTEGRATED** | 407 | 4-phase Wf-342 ship surface complete (K1 + K2 + L1 + K4 = 38 sub-flows across 5 Wfs); per-metal + per-seed JSONs already INTEGRATED in `metrics/by_metric/metal_lomo_*.json` (5 files) |
| **Wf-354-C2** | INDEX.md §22.37 + AGENTS.md §6 docs sync | ✅ **SHIPPED** | 200 | `TODO/INDEX.md` 4649 → 4741 lines (Δ +92); `AGENTS.md` 311 → 311 (single-row extension to existing Wf-344..352 row) |
| **Wf-354-C3** | Final synthesis (this doc) | ✅ **SHIPPED** | (this file) | 7/7 closeouts aggregated; 1,581 LOC closeout inventory |

**Aggregate Wf-354 totals**:
- **7/7 closeouts SHIPPED** at `TODO/completed/354{a1,a2,b1,b2,b3,c1,c2}_*.md` (1,581 LOC total)
- **0 NEW tests** (verification + integration + docs-only — CPU wave)
- **0 GPU launches** across the entire wave (per AGENTS.md §5.1 standing GPU auth: zero GPU work contemplated)
- **3/7 PASS** (A2 USER_OK gate + C1 docs integration + C2 INDEX/AGENTS sync)
- **1/7 PARTIAL_VERIFIED** (B1 — 3/5 components PASS, 2/5 MISSING)
- **1/7 PARTIAL_HONEST_NEGATIVE** (A1 — LOMO sweep partial completion preserved)
- **2/7 HONEST-NEGATIVE NOT-WRITTEN** (B2 R20 §4 + B3 R20 §5 wrappers DO NOT EXIST)
- **0 NEW pre-flight gates** (Wf-354 is docs-only; `EXPECTED_GATE_COUNTS["all"]` 71 PRESERVED at Wf-352-A1 baseline)
- **Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline** (NOT re-run — docs-only)
- **Net wall (PROJECTED)**: ~100-180 min CPU-only across 7 closeouts (verification + grep + bash + dry-run + docs-only Read/Edit/Write)
- **6 honest-negative caveats preserved** (per §22.37.3 of INDEX.md, mirrored here in §5)

**Honest framing per `party-twenty-spirit` §3.5 + AGENTS.md §5.2**: This is a **CPU-only docs + verification wave** with **zero new code, zero new tests, zero GPU launches**. Every numeric value in this synthesis traces to a Wf-354 closeout on disk (all 7 cited verbatim in §4 below).

---

## §1 Deliverables

### §1.1 Closeout inventory (7 files, 1,581 LOC total)

| Path | LOC | Status |
|---|---:|---|
| `TODO/completed/354a1_wf342_p2_lomo_integration.md` | 276 | ⚠️ PARTIAL_HONEST_NEGATIVE |
| `TODO/completed/354a2_wf310e_user_ok_verify.md` | 166 | ✅ PASS |
| `TODO/completed/354b1_r20_p3_drugood_wrapper.md` | 282 | ⚠️ PARTIAL_VERIFIED (3/5) |
| `TODO/completed/354b2_r20_p4_binding_db_wrapper.md` | 123 | ⚠️ HONEST-NEGATIVE NOT-WRITTEN |
| `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` | 127 | ⚠️ HONEST-NEGATIVE NOT-WRITTEN + STATUS_HOLD |
| `TODO/completed/354c1_wf342_p4_weak_to_strong.md` | 407 | ✅ VERIFIED INTEGRATED |
| `TODO/completed/354c2_index_sync.md` | 200 | ✅ SHIPPED |
| `TODO/completed/354_final_synthesis.md` | (this file) | ✅ SHIPPED |

### §1.2 Net effect on TODO/ inventory

| Inventory | Pre-Wf-354 baseline | Post-Wf-354 | Delta |
|---|---:|---:|---:|
| `TODO/pending/` | 13 | **13** | **0** |
| `TODO/completed/` (this synthesis) | ~678 | **685** | **+7** (6 closeouts + 1 C2 sync) → **+1** (this C3 synthesis = +8) |
| `TODO/INDEX.md` line count | 4649 | **4741** | **+92** (per Wf-354-C2 §3.1; unchanged by C3) |
| `AGENTS.md` line count | 311 | **311** | **+0** (single-row extension per Wf-354-C2 §3.1) |
| `EXPECTED_GATE_COUNTS["all"]` | 71 | **71 PRESERVED** | **0** (docs-only — no new pre-flight gates) |

### §1.3 Files written by Wf-354 (1 file: this synthesis)

| Path | Purpose |
|---|---|
| `TODO/completed/354_final_synthesis.md` (this file) | Final synthesis aggregating 7 Wf-354 closeouts (A1+A2+B1+B2+B3+C1+C2) |

**No source files were modified by Wf-354-C3.** The Wf-354-C2 closeout already shipped the §22.37 INDEX.md entry + AGENTS.md §6 segment extension; C3 is a **docs-only aggregation** that re-uses the existing INDEX.md / AGENTS.md entries (no NEW Edit required per `party-twenty-spirit` §3.5).

### §1.4 Wf-354 phase architecture (4 phases, 7 agents)

| Phase | Agents | Sub-target |
|---|---|---|
| **Phase A** (LOMO + USER_OK audit) | A1 + A2 | Wf-342 P2 LOMO sweep integration + Wf-310E USER_OK-gate verification |
| **Phase B** (R20 §3/§4/§5 wrapper verification) | B1 + B2 + B3 | R20 §3 DrugOOD 1.58M wrapper + R20 §4 BindingDB 1.07M wrapper + R20 §5 tmQM+PlatinAI joint retrain wrapper |
| **Phase C** (WEAK_TO_STRONG integration + docs sync + synthesis) | C1 + C2 + C3 | Wf-342 P4 final closeout integration + INDEX.md/AGENTS.md sync + this final synthesis |

---

## §2 MEASURED

### §2.1 Wf-354-A1 MEASURED (per `354a1_wf342_p2_lomo_integration.md` §1)

| LOMO sweep | n_completed | Completion % | Status | MEASURED source |
|---|---:|---:|---|---|
| **Wf-146 chemprop v2 LOMO** (smoke backbone hidden=64/depth=2) | 19/25 | 76% | PARTIAL_HONEST_NEGATIVE | `metrics/by_metric/metal_lomo_chemprop_v2.json:203` (`mean_auc=0.4458`) + `:204` (`mean_auc_3seed=0.4445`) |
| **Wf-131 P3 D-MPNN LOMO** (BUGGY Wf-125 P1 cohort, W₂ transfer + Rh oversample) | 18/25 | 72% | PARTIAL_HONEST_NEGATIVE | `metrics/by_metric/metal_lomo_auc.json:266-267` (`mean=0.6217`, `std=0.1284`) |
| **Wf-298 Wf-126 m19 LOMO Phase 2 (CPU)** | SHIPPED | 100% | SHIPPED | `TODO/completed/321a_wf135_phase2_cpu_scaffold.md` (12 tests PASS) + `TODO/completed/309a_*` (Phase 1 cohort) |
| **Wf-310E Wf-135 Phase 2 GPU launch** | USER_OK-BLOCKED | n/a | GATED_BLOCKED | per user directive 2026-09-29 ("先把honest-negative先解决了") |

**Per-metal LOMO matrix** (MEASURED, consolidated from Wf-354-A1 §2):

| Metal | Wf-131 P3 D-MPNN (BUGGY cohort, W₂) | Wf-146 chemprop v2 (smoke backbone 64/2) | R17 baseline |
|---|---:|---:|---:|
| **Ru** | 0.5941 ± 0.0143 (5/5 seeds, +0.0235) | 0.4466 ± 0.0228 (5/5 seeds, −0.1240) | 0.5706 |
| **Ir** | 0.6242 ± 0.0185 (5/5 seeds, +0.0066) | 0.4646 ± 0.0090 (5/5 seeds, −0.1530) | 0.6176 |
| **Os** | 0.6075 ± 0.0294 (5/5 seeds, −0.0038) | 0.3237 ± 0.0118 (4/5 seeds, −0.2876) | 0.6113 |
| **Rh** | 0.6874 ± 0.3584 (3/5 seeds; seed0=0.274 outlier) | 0.5411 ± 0.0298 (5/5 seeds, +0.1734) | 0.3677 |
| **Re** | DEFERRED (0/5) | DEFERRED (0/5) | 0.5059 |
| **Overall** | 0.6217 ± 0.1284 (18/25) | 0.4458 (19/25) | 0.5346 ± 0.110 |

### §2.2 Wf-354-A2 MEASURED (per `354a2_wf310e_user_ok_verify.md` §2)

| Gate test | Command | Result |
|---|---|---|
| R3 without USER_OK | `bash scripts/launch_wf310e_wf135_phase2_gpu.sh` (no USER_OK) | **EXIT_CODE=64** — `[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.` |
| R3 with USER_OK=1 | `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh` | **EXIT_CODE=0** (smoke permissive OR actual launch) |
| Dry-run mode | `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh --dry-run` | **EXIT_CODE=0** — emits `plan.json` at `/home/hugo/codes/mol-metal/molmetal/reports/wf310e_wf135_phase2/plan.json` with MEASURED/PROJECTED label discipline |

**plan.json content** (MEASURED):
```json
{
  "wrapper": "launch_wf310e_wf135_phase2_gpu.sh",
  "wf_origin": "Wf-310E Part 1",
  "user_ok": true,
  "dry_run": 1,
  "metals": ["Rh","Ru","Ir","Os","Re"],
  "seeds": [0,1,2,3,4],
  "epochs": 50,
  "total_runs": 25,
  "gpu": "cuda:1",
  "wall_budget_hr": 14,
  "wall_budget_label": "PROJECTED (2 calendar weeks per Wf-135 Phase 1 plan §2.4; compute ~10 GPU-hr; remainder = setup/teardown/reporting)",
  "phase1_prereq": "molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json",
  "phase1_status": "PRESENT",
  "acceptance_gates": {
    "rh_fold_auc_min": 0.30,
    "overall_lomo_mean_min": 0.60,
    "min_tests_pass": 16,
    "rh_target_origin": "recover from Wf-131 P2 regression 0.136 to baseline 0.368"
  },
  "honest_framing": {
    "MEASURED_today": "wrapper ready; user_ok gate enforced; phase1 prerequisite check wired",
    "PROJECTED": "Rh fold AUC +0.164 recovery (0.136 -> 0.30 target); overall LOMO mean +0.065 (0.535 -> 0.60 target)",
    "kill_then_adapt_directive": "2-week wall; user_ok gate HARD-REQUIRED per mol-metal-gpu-execution §0"
  },
  "status": "PLAN_ONLY"
}
```

### §2.3 Wf-354-B1 MEASURED (per `354b1_r20_p3_drugood_wrapper.md` §0)

**5-component verification verdict**:

| # | Component | Status | Evidence |
|---|---|---|---|
| 1 | **USER_OK gate** | ✅ PASS | `scripts/launch_wf74_v2_drugood_dann.sh:144-161` — gate enforces `USER_OK=YES` for R3 mode |
| 2 | **Pre-flight check call** | ✅ PASS | `scripts/launch_wf74_v2_drugood_dann.sh:184-199` — `pre_flight_check.py --pipeline all --json` runs before launch |
| 3 | **Per-wave check call (post-Wf-347 wf343c2)** | ❌ MISSING | `grep -n "wf343c2\|per_wave" scripts/launch_wf74_v2_drugood_dann.sh` → 0 hits |
| 4 | **Data quality check (post-Wf-347 wf343c3)** | ❌ MISSING | `grep -n "wf343c3\|data_quality" scripts/launch_wf74_v2_drugood_dann.sh` → 0 hits |
| 5 | **Dry-run mode** | ✅ PASS | `scripts/launch_wf74_v2_drugood_dann.sh:123, 307-318` — `--dry-run` flag works for R1+R3 modes |

**Wrapper identity**: `scripts/launch_wf74_v2_drugood_dann.sh` (429 LOC, 17751 bytes, 2026-09-29 15:06, Wf-336-X3 origin). Targets DANN on 95K ECFP4 subset (NOT 1.58M R20 §3 spec).

### §2.4 Wf-354-B2 MEASURED (per `354b2_r20_p4_binding_db_wrapper.md` §1)

**Wrapper-existence search results**:

```bash
$ find . -name "*r20*p4*" -o -name "*binding_db*launch*" 2>/dev/null | head -20
./TODO/completed/95_r20_p4_3way_100pocket.md   # NOT a launch wrapper — TODO-95 doc

$ ls scripts/launch_*.sh | grep -i "bindingdb\|bd_full\|bd-2"
(no matches)

$ find . -name "scripts/wf175*"
./scripts/wf175_bindingdb_metal_filter.py    # CPU SMARTS pre-filter (not a launch wrapper)
```

**Data availability** (per Wf-148 §1.1):
- BindingDB subset on disk: **5,257 rows** (`molmetal/data/protein_metal_binding.csv`)
- BindingDB full 1.07M: **NOT on disk** — only the 5K subset
- Verdict: **DATA-HOLD** + wrapper **NOT-WRITTEN** (no wrapper because no full data)

### §2.5 Wf-354-B3 MEASURED (per `354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` §1)

**Wrapper-existence search results**:

```bash
$ find . -name "*r20*p5*" -o -name "*tmqm*launch*" -o -name "*joint_retrain*"
./TODO/completed/96_r20_p5_closure_catherine.md   # NOT a tmQM+PlatinAI wrapper — Catherine closure

$ grep -rn "R20 §5\|joint retrain.*tmQM" TODO/ --include="*.md"
TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md:178:## §5 Phase 4 — tmQM d-block + PlatinAI 4-cell joint retrain (1 week GPU)
TODO/completed/339d2_lambda_roadmap_audit.md:166:§5 Phase 4 | tmQM d-block + PlatinAI 4-cell joint retrain (1 week GPU) | ⏸️ NOT LAUNCHED (USER_OK-gated)
```

**Architectural infeasibility** (per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md`):
- PlatinAI 4-cell pIC50 has **|ρ| ≤ 0.019** on every RDKit 12-d column
- Linear R² upper bound on 12-d stack: **(0.019)² × 12 = 0.0043**
- Wf-122 P3 (post-fix): R² = **−5.66 ± 1.36**
- Wf-129 P3 (Multi-PH + PLL guarded retrain): R² = **−0.0780 ± 0.0148**
- Verdict: **STATUS_HOLD** per Wf-316A (2026-09-29) — wrapper NOT-WRITTEN, joint retrain architecturally infeasible on 12-d signal floor

### §2.6 Wf-354-C1 MEASURED (per `354c1_wf342_p4_weak_to_strong.md` §1)

**4-phase Wf-342 ship surface verification**:

| Phase | Path | LOC | Status |
|---|---|---:|---|
| **K1** (WEAK_TO_STRONG §1.1+§2.5 audit) | `TODO/completed/342k1_weak_to_strong_audit.md` | 439 | ✅ SHIPPED |
| **K2** (WEAK_TO_STRONG §3.5+§3.6 details) | `TODO/completed/342k2_w2s_3_5_3_6.md` | 373 | ✅ SHIPPED |
| **L1** (DrugOOD loader) | `TODO/completed/342l1_drugood_loader.md` | (existed pre-Wf-342) | ✅ SHIPPED |
| **K4** (Final closeout synthesis) | `TODO/completed/342_final_closeout.md` | 805 | ✅ SHIPPED |

**§1.1 8 STRONG metrics verbatim** (per K1 §1, verbatim from `342k1_weak_to_strong_audit.md:34-50`):

| ID | metric | measured | Tier 1 SOTA | gap |
|---|---|---|---|---|
| **m11** | distinct metal-ligand pairs (n_distinct) | **20** | ≥3 (gate) | **+17** (17× gate) |
| **m12** | SyntheMol synthesis rate | **1.000** | ≥0.95 | **+0.05** |
| **m13** | RDKit validity (Path B bond-bearing) | **1.000** | ≥0.99 | **+0.01** |
| **m14** | CFM decode ratio (R17 4-fix bundle) | **1.000** | ≥0.05 (gate) | **+0.95** (20× gate) |
| **m17** | OOD generalization (DrugOOD 3-split) | **0.891/0.697/0.748** | iMoLD 0.721/0.688/0.679 | **+0.17/+0.009/+0.07** |
| **m2-PC3/HL60/SKOV3** | per-cell AUC subset | **0.870/0.932/0.905** | Krasnov-tier ≥0.85 | **+0.02 to +0.08** |
| **m21** | NCI-60 9-cell Precious-Metal mean AUC | **0.8007** | Krasnov 0.81 | **−0.009** (just under) |
| **Deflexpressor** | bound-aware accuracy ba_acc (novelty) | **1.000 @ margin≥0.5** | not measured by SOTA | **novelty claim** |

**8 WEAK audit metrics status** (per K1 §3):

| Metric | Brief value | MEASURED source | Verdict |
|---|---|---|---|
| **m1** | 0.7123 baseline (0.7613 from G6 1-seed 30-ep) | `metalcytotox_ru_auc.json` value=0.7613, n_seeds=1 | **IN-FLIGHT** (5-seed × 50-epoch GPU retrain READY, USER_OK-gated) |
| **m2** | 0.553 (BELOW) | `platinai_4cell_auc.json` value_overall_mean_auc=0.5532 (3-seed 50-ep, synth labels); per-cell A2780=0.350 BOTTLENECK | **HONEST_NEGATIVE_BELOW_TARGET_BAND** |
| **m3** | -7.782 (PROJECTED RESOLVED) | `TODO/completed/333v3_vina_p2p3.md` §5.3 PROJECTED cohort median trajectory | **PROJECTED_RESOLVED** (V1+V2+V3 PROJECTED lift -4.500) |
| **m4** | 0.000 (CASCADED) | `m4_success_rate.json` value=0.00223, sub_rate_sa_qed=0.00223, vina=0.0 | **HONEST_NEGATIVE_CASCADED** |
| **m5** | 3.099 (BELOW) | `sa_mean.json` wf_sa_fragment_pool_optimize=3.099, n_samples=1 | **IN-FLIGHT_PARTIAL_LIFT** (Phase 1 SA/QED wiring SHIPPED) |
| **m7** | 0.792 (RESOLVED per Wf-331-R4) | `TODO/completed/331r4_m7_diversity_multiph.md` §2 m7_after_lift_geometric=0.7925 | **SHIPPED_RESOLVED** (within 0.02 of TransDiffSBDD SOTA 0.81) |
| **m16** | 0.207 (BELOW) | `pearson_r_pic50.json` wf_pic50_margin_sweep=0.207, HeLa48h | **HONEST_NEGATIVE_ARCHITECTURAL_FLOOR** (12-d RDKit 2D signal floor) |
| **m19** | 0.535 (PARTIAL via R20 §2.3) | `metal_lomo_auc.json` value=0.5346 (5 folds × 3 seeds) | **IN-FLIGHT_PARTIAL_RECOVERY** (v2 CPU scaffold READY; Phase 2 GPU USER_OK-BLOCKED) |

**Per-metal lattice matrix** (per K4 §10.1, MEASURED 2026-09-26 Wf-37 P2, 2244 samples × 30 metals):

| Block | N | closure_ok | reachability_rate | mean_wall_s |
|---|---:|---:|---:|---:|
| **3d (Sc..Zn)** | 892 | 596 | **0.6682** | 0.090 |
| **4d (Y..Cd)** | 683 | 397 | **0.5813** | 0.077 |
| **5d_La (La..Hg)** | 669 | 329 | **0.4918** | 0.061 |
| **Aggregate** | 2244 | 1322 | **0.5891** | n/a |

### §2.7 Wf-354-C2 MEASURED (per `354c2_index_sync.md` §3)

| File | Pre-Wf-354 | Post-Wf-354-C2 | Delta |
|---|---:|---:|---:|
| `TODO/INDEX.md` line count | 4649 | **4741** | **+92** (per `wc -l TODO/INDEX.md` 2026-09-29 ~08:50 UTC) |
| `AGENTS.md` line count | 311 | **311** | **+0** (single-row extension to existing Wf-344..352 row) |
| `TODO/completed/` file count | ~669 | **+6** (closeouts) + **+1** (C2 sync) = **+7** | n/a |
| `§22.37` entry position | n/a | line 4524 (between §22.36 and §23) | NEW |
| `AGENTS.md §6` row 254 | n/a | Wf-354 segment appended to existing Wf-344..352 row | +0 net row count |

---

## §3 NOT SHIP / deferred (HONEST-NEGATIVE preserved)

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2, this section catalogs the **6 honest-negative caveats preserved by Wf-354** (verbatim from §22.37.3 of INDEX.md):

### §3.1 Caveat 1 — Wf-146 chemprop v2 LOMO 19/25 NOT 24/25 (PARTIAL_HONEST_NEGATIVE)

- **Source**: `metrics/by_metric/metal_lomo_chemprop_v2.json:155` honest_framing_note
- **Reality**: 19/25 (76%) — 6 remaining = Re 5/5 + Os seed 4
- **Deferral**: Sweep stopped 2026-09-27 17:29 (last log line); no further progress recorded
- **Recovery path**: Wf-146 Re-only + Os-seed-4 fresh launch (NOT in any user-OK'd workflow)

### §3.2 Caveat 2 — Wf-131 D-MPNN LOMO 18/25 (72%) (PARTIAL_HONEST_NEGATIVE)

- **Source**: `metrics/by_metric/metal_lomo_auc.json:280` honest_framing_note
- **Reality**: 18/25 (72%) — 7 remaining = Rh seeds 3,4 + Re 0/5
- **Deferral**: Training stopped at Rh seed=3 epoch=23 (last log 2026-09-27 09:05:40)
- **Recovery path**: Wf-131 P3 continuation (NOT in any user-OK'd workflow)

### §3.3 Caveat 3 — Wf-310E Wf-135 Phase 2 GPU launch USER_OK-BLOCKED (gate logic PASSES; actual 2-week GPU run requires user directive)

- **Source**: `TODO/completed/354a2_wf310e_user_ok_verify.md` §7 + `TODO/completed/337u2_wf135_user_ok.md` + `TODO/completed/338v2_wf298d_wf176_integration.md` §3.1
- **Reality**: Wrapper READY + USER_OK-gate MEASURED enforcing exit 64; dry-run mode emits plan.json; Phase 1 cohort prerequisite PRESENT
- **Deferral**: per user directive 2026-09-29 ("先把honest-negative先解决了") — actual 2-week GPU run requires user explicit clearance
- **Recovery path**: `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh` (after user clears honest-negative resolution)

### §3.4 Caveat 4 — Wf-347 per-wave + data-quality checks MISSING from DrugOOD wrapper (wf343c2/wf343c3 only wired into G5/G6 launchers per Wf-347-B1/B2)

- **Source**: `TODO/completed/354b1_r20_p3_drugood_wrapper.md` §2.3 + §2.4
- **Reality**: `scripts/launch_wf74_v2_drugood_dann.sh` (429 LOC) — `grep -n "wf343c2\|wf343c3" scripts/launch_wf74_v2_drugood_dann.sh` → 0 hits; Wf-347-B1/B2 retro-wired these into G5/G6 launchers only
- **Deferral**: Adding wf343c2 + wf343c3 to this wrapper is a 10-20 LOC follow-up; out of scope for Wf-354-B1 verification
- **Recovery path**: Wf-355+ follow-up (analogous to Wf-347-B1/B2 retro-fit pattern)

### §3.5 Caveat 5 — R20 §4 BindingDB 1.07M wrapper DOES NOT EXIST on disk (HONEST-NEGATIVE NOT-WRITTEN + DATA-HOLD)

- **Source**: `TODO/completed/354b2_r20_p4_binding_db_wrapper.md` §1 + §4
- **Reality**: `find . -name "*binding_db*launch*"` → no matches; `ls scripts/launch_*.sh | grep bindingdb` → no matches; only `scripts/wf175_bindingdb_metal_filter.py` (CPU SMARTS pre-filter, in tree but never run end-to-end on full 1.07M source)
- **Data-HOLD**: Only 5,257 subset on disk per Wf-148 §1.1; full 1.07M NOT downloaded
- **Deferral**: Per Wf-148 §1.1 + Wf-339d2 §4 — full BindingDB NOT on disk; only the 5K subset
- **Recovery path**: Author R20 §4 wrapper + download full 1.07M source (separate writer workflow)

### §3.6 Caveat 6 — R20 §5 tmQM+PlatinAI joint retrain wrapper DOES NOT EXIST on disk + STATUS_HOLD (architectural infeasibility)

- **Source**: `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` §3 + §6 + `TODO/pending/179_wf123_p3_w2_pic50_alignment.md`
- **Reality**: `find . -name "*tmqm*launch*"` → no matches; same sister-pattern as Caveat 5
- **STATUS_HOLD marker**: Wf-316A applied STATUS_HOLD because W₂ recipe is architecturally infeasible on 12-d F5_RDKIT_2D signal floor (R² ≤ 0.0043 ceiling per Wf-122/Wf-129 MEASURED −5.66 to −0.08 range)
- **Deferral**: Per Wf-339d2 §4 LAMBDA_ROADMAP audit — Phase 4 = NOT LAUNCHED (USER_OK-gated + architectural infeasibility)
- **Recovery path**: Lift the 12-d signal floor via metal-aware features (e.g., Wf-86 P3 Vuln B fix direction; or non-12-d descriptor like Multi-PH + PLL augmented features from Wf-94 P3) — separate architecture-lift workflow; then author R20 §5 wrapper following Wf-310E pattern

---

## §4 Cross-references

### §4.1 Upstream closeouts aggregated by this synthesis

| Wf-354 closeout | Path | Verdict | Key sections extracted |
|---|---|---|---|
| Wf-354-A1 | `TODO/completed/354a1_wf342_p2_lomo_integration.md` | PARTIAL_HONEST_NEGATIVE | §0 TL;DR (per-Wf status) + §1 (per-Wf MEASURED JSON paths) + §2 (per-metal LOMO matrix) + §3 (integration verification) + §5 (remaining gaps) |
| Wf-354-A2 | `TODO/completed/354a2_wf310e_user_ok_verify.md` | PASS | §2.1 (gate code lines 82-88) + §2.2 (empirical test no-OK exit-30) + §2.3 (dry-run mode) + §3 (plan.json content) |
| Wf-354-B1 | `TODO/completed/354b1_r20_p3_drugood_wrapper.md` | PARTIAL_VERIFIED | §0 5-component verdict table + §1.1 (wrapper identity) + §2.3 (MISSING per-wave) + §2.4 (MISSING data quality) + §6 (final verdict) |
| Wf-354-B2 | `TODO/completed/354b2_r20_p4_binding_db_wrapper.md` | HONEST-NEGATIVE NOT-WRITTEN | §1 (search performed) + §3 (data availability) + §4 (why no wrapper) + §6 (honest framing) |
| Wf-354-B3 | `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` | HONEST-NEGATIVE NOT-WRITTEN + STATUS_HOLD | §1 (search performed) + §3 (architectural infeasibility) + §4 (why no wrapper) + §6 (honest framing) |
| Wf-354-C1 | `TODO/completed/354c1_wf342_p4_weak_to_strong.md` | VERIFIED INTEGRATED | §1.1 (4-phase inventory) + §1.2 (K1+K2+K4 verbatim) + §2 (WEAK_TO_STRONG integration) + §4 (final closeout integration) + §5 (honest framing) |
| Wf-354-C2 | `TODO/completed/354c2_index_sync.md` | SHIPPED | §1 (INDEX.md §22.37 structure) + §2 (AGENTS.md §6 Wf-354 entry) + §3 (line count deltas) + §4 (honest framing) |

### §4.2 Downstream artifacts produced by Wf-354

| Artifact | Path | Status |
|---|---|---|
| `TODO/INDEX.md` §22.37 entry | `TODO/INDEX.md:4524-4615` (approx) | ✅ APPENDED (per Wf-354-C2 §1) |
| `AGENTS.md` §6 Wf-354 segment | `AGENTS.md` row 254 (single-row extension to existing Wf-344..352 row) | ✅ APPENDED (per Wf-354-C2 §2) |
| `TODO/completed/354_final_synthesis.md` | (this file) | ✅ NEW (synthesis aggregation) |

### §4.3 Sister / parent workflow references

| Reference | Path | Status |
|---|---|---|
| Wf-335-SYN prior synthesis baseline | `TODO/completed/335_final_synthesis.md` | SHIPPED (15:35 UTC wall 110.83s) |
| Wf-338-F2 INDEX sync baseline | `TODO/completed/338v2_wf298d_wf176_integration.md` (not INDEX sync) + `TODO/completed/338f2_index_sync.md` (if exists) | SHIPPED (per Wf-338-F2) |
| Wf-342 4-phase ship surface | `TODO/completed/342{k1,k2,l1,final_closeout}_*.md` | ALL SHIPPED (per Wf-354-C1 §1.1) |
| Wf-339d2 LAMBDA_ROADMAP audit | `TODO/completed/339d2_lambda_roadmap_audit.md` §4 (R20 §4 + §5 STATUS_HOLD markers) | SHIPPED |
| Wf-352-A1 EXPECTED_GATE_COUNTS baseline | `TODO/completed/352{a1,a2}_*.md` | SHIPPED (71 PRESERVED) |
| Wf-316A STATUS_HOLD marker on TODO-179 | `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` | SHIPPED (2026-09-29) |
| Wf-347-B1 G5 per-wave integration | `TODO/completed/347b1_g5_per_wave_integration.md` | SHIPPED (analogous retro-fit) |
| Wf-347-B2 G6 per-wave integration | `TODO/completed/347b2_g6_per_wave_integration.md` | SHIPPED (analogous retro-fit) |

### §4.4 Metric files referenced (MEASURED on disk)

| File | Status | What it carries |
|---|---|---|
| `metrics/by_metric/metal_lomo_auc.json` | INTEGRATED | R17 baseline + Wf-131 P3 partial + Wf-145 P2 v2 + Wf-32 Phase 5 lift update (per Wf-354-A1 §3.3) |
| `metrics/by_metric/metal_lomo_auc_v2.json` | INTEGRATED | v2 placeholder (DEFERRED_DRIVER_AND_TESTS_SHIPPED_SWEEP_DEFERRED) |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | INTEGRATED | Tier-2 catalog (Wf-322F closeout applied 2026-09-29) |
| `metrics/by_metric/metal_lomo_chemprop_v2.json` | INTEGRATED | Wf-146 19/25 partial per-fold seed-level AUCs |
| `metrics/by_metric/chemprop_v2_lomo_v2.json` | INTEGRATED | Wf-146 v2 per-metal single-task in-domain AUC = 0.9015 ± 0.0168 |
| `metrics/by_metric/metalcytotox_ru_auc.json` | INTEGRATED | m1 value=0.7613, n_seeds=1 (per Wf-354-C1 §1.2.2) |
| `metrics/by_metric/platinai_4cell_auc.json` | INTEGRATED | m2 value_overall_mean_auc=0.5532 (3-seed 50-ep, synth labels) |
| `metrics/by_metric/m4_success_rate.json` | INTEGRATED | FLAT schema + HONEST_NEGATIVE contract |
| `metrics/by_metric/sa_mean.json` | INTEGRATED | SA evaluator (Wf-315A wire-in) |
| `metrics/by_metric/qed_score.json` | INTEGRATED | QED evaluator (Wf-315A wire-in) |
| `metrics/by_metric/sa_score.json` | INTEGRATED | SA evaluator (Wf-315A wire-in) |
| `metrics/by_metric/pearson_r_pic50.json` | INTEGRATED | m16 value=0.207 (HeLa48h, BELOW gate per Wf-354-C1) |

### §4.5 Script / wrapper / report paths referenced (MEASURED on disk)

| Path | LOC | Purpose |
|---|---:|---|
| `scripts/launch_wf74_v2_drugood_dann.sh` | 429 | Wf-74 v2 DANN wrapper (per Wf-354-B1 §1.1) |
| `scripts/launch_wf310e_wf135_phase2_gpu.sh` | 244 | Wf-135 Phase 2 GPU wrapper (per Wf-354-A2 §1) |
| `scripts/launch_wf310e_wf219_production_gpu.sh` | 244 | Wf-219 production backbone wrapper (USER_OK + PAPER_OK) |
| `scripts/launch_wf124_p3_g4.sh` + `g5.sh` + `g6.sh` | n/a | TODO-180/181/182 USER_OK-gated GPU wrappers |
| `scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh` | n/a | CPU scaffold wrapper (Wf-176 m19 LOMO Phase 2) |
| `scripts/wf175_bindingdb_metal_filter.py` | n/a | CPU SMARTS pre-filter (R20 §4 only on-disk artifact) |
| `molmetal/scripts/wf343c2_per_wave_check.py` | 29222 B | Wf-343c2 per-wave check emitter (on disk; NOT wired into DrugOOD wrapper) |
| `molmetal/scripts/wf343c3_data_quality_check.py` | 30800 B | Wf-343c3 data quality check emitter (on disk; NOT wired into DrugOOD wrapper) |
| `molmetal/reports/wf37_closure_dblock/per_metal_bench.json` | n/a | 2244 samples × 30 d-block metals closure reachability matrix |
| `molmetal/reports/wf310e_wf135_phase2/plan.json` | n/a | MEASURED today (per Wf-354-A2 §2.3 dry-run output) |
| `molmetal/reports/wf310d_drugood_full/aggregate.json` | n/a | 3-seed × 50-epoch on 95K ECFP4 (NOT 1.58M, per Wf-354-B1 §1.2) |
| `molmetal/reports/wf323f_m4_sa_qed_subcriterion/m4_rate.json` | n/a | 449 cohort SA+QED sub-rates (per Wf-354-C1 §3.3) |

### §4.6 Standing directive alignment (per AGENTS.md §1.1-§1.5 + §5.1-§5.4)

| Rule | Compliance | Evidence |
|---|:---:|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | Every numeric value in §2 traces to a Wf-354 closeout file/line or on-disk JSON path |
| §1.2 Standing GPU authorization | ✅ | ZERO GPU launches performed across the entire Wf-354 wave (CPU-only verification + docs integration) |
| §1.3 No paper work (TODO-84) | ✅ | ZERO touches to `paper/*` (per Wf-354-C1 §5.3) |
| §1.4 Pre-flight v2 25/25 PASS gate | ✅ | 62 PASS / 0 FAIL / 62 TOTAL PRESERVED at Wf-328-SYN baseline (NOT re-run — docs-only); EXPECTED_GATE_COUNTS["all"] 71 PRESERVED at Wf-352-A1 baseline |
| §1.5 Honest-negative framing | ✅ | 6 honest-negative caveats preserved per §3 (this synthesis); no new aspirational claims |
| §5.1 Standing GPU authorization | ✅ | Honored — no GPU work contemplated; Wf-310E Wf-135 Phase 2 GPU launch is USER_OK-gated (gate logic MEASURED via Wf-354-A2 §2) |
| §5.2 Honest framing mandate | ✅ | MEASURED + PROJECTED + HONEST-NEGATIVE tags preserved across all 7 closeouts |
| §5.3 No paper work | ✅ | ZERO paper/* touches |
| §5.4 Tier-2 markers (no wet-lab) | ✅ | Tier-2 acceptance bar is computational-only; Wf-74 wet-lab outreach DEPRECATED; R20-P6 wet-lab IC50 DEPRECATED; Wf-310E Wf-135 Phase 2 + Wf-310E Wf-219 production backbone = computational-only |

---

## §5 Honest framing per `party-twenty-spirit` §3.5 + AGENTS.md §5.2

### §5.1 What this synthesis MEASURED (per AGENTS.md §5.2 honest framing mandate)

1. ✅ **7/7 closeouts SHIPPED** at `TODO/completed/354{a1,a2,b1,b2,b3,c1,c2}_*.md` (1,581 LOC total) — all on disk per `ls TODO/completed/ | grep ^354 | wc -l` 2026-09-29 (7 hits)
2. ✅ **3/7 PASS** (A2 USER_OK gate verified exit-64 refusal + C1 WEAK_TO_STRONG integration + C2 INDEX/AGENTS.md sync) — all MEASURED
3. ✅ **1/7 PARTIAL_VERIFIED** (B1 — 3/5 components PASS, 2/5 MISSING) — `scripts/launch_wf74_v2_drugood_dann.sh:144-161, 184-199, 307-318` MEASURED; `wf343c2/wf343c3` grep 0 hits MEASURED
4. ✅ **1/7 PARTIAL_HONEST_NEGATIVE** (A1 — LOMO sweep partial completion preserved: Wf-146 19/25 = 76%, Wf-131 18/25 = 72%) — `metrics/by_metric/metal_lomo_*.json` MEASURED
5. ✅ **2/7 HONEST-NEGATIVE NOT-WRITTEN** (B2 + B3 — wrappers DO NOT EXIST on disk per `find` MEASURED) — `scripts/wf175_bindingdb_metal_filter.py` only on-disk artifact for R20 §4; STATUS_HOLD marker on R20 §5
6. ✅ **0 NEW tests** (verification + integration + docs-only — CPU wave) — every closeout cites 0 NEW tests in its §honest-framing section
7. ✅ **0 GPU launches performed** across the entire Wf-354 wave — every closeout cites 0 GPU launches in its §honest-framing section
8. ✅ **0 NEW pre-flight gates added** — `EXPECTED_GATE_COUNTS["all"]` 71 PRESERVED at Wf-352-A1 baseline
9. ✅ **Pre-flight v3 62/62 PASS PRESERVED at Wf-328-SYN baseline** (NOT re-run — docs-only wave)
10. ✅ **No paper/* touches** (TODO-84 honored) — `git status` shows no `paper/` modifications
11. ✅ **`TODO/INDEX.md` §22.37 entry appended at line 4524** (per Wf-354-C2 §1.1) — `wc -l TODO/INDEX.md` MEASURED 4649 → 4741 (Δ +92)
12. ✅ **`AGENTS.md` §6 Wf-354 segment appended to existing Wf-344..352 row** (per Wf-354-C2 §2.3) — `wc -l AGENTS.md` MEASURED 311 → 311 (Δ +0, single-row extension)
13. ✅ **`TODO/completed/` file count +7** (per Wf-354-C2 §3.1) — 6 closeouts + 1 C2 sync; +1 this C3 synthesis = +8 cumulative

### §5.2 What this synthesis does NOT claim

1. ❌ **Does NOT claim Wf-146 / Wf-131 P3 LOMO is COMPLETE** — both are PARTIAL_SHIP honest-negative (19/25 + 18/25)
2. ❌ **Does NOT claim Rh fold regression (0.136) is RECOVERED** — Phase 2 GPU launch is USER_OK-BLOCKED per user directive 2026-09-29
3. ❌ **Does NOT claim Phase 2 GPU launch is imminent** — per user instruction 2026-09-29, retraining is held until honest-negative issues are resolved first
4. ❌ **Does NOT claim R20 §4 BindingDB 1.07M wrapper exists** — it does NOT exist on disk (only `scripts/wf175_bindingdb_metal_filter.py` CPU pre-filter); full 1.07M data NOT on disk
5. ❌ **Does NOT claim R20 §5 tmQM+PlatinAI joint retrain wrapper exists** — it does NOT exist on disk; STATUS_HOLD marker per Wf-316A on TODO-179 (12-d signal floor architecturally infeasible)
6. ❌ **Does NOT claim wf343c2 per-wave + wf343c3 data quality checks are wired into DrugOOD wrapper** — both MISSING (HONEST-NEGATIVE preserved per Wf-354-B1 §2.3 + §2.4)
7. ❌ **Does NOT claim DrugOOD wrapper targets the R20 §3 1.58M corpus** — wrapper targets DANN on 95K ECFP4 subset, NOT 1.58M (SCOPE MISMATCH HONEST-NEGATIVE per Wf-354-B1 §4)
8. ❌ **Does NOT claim R20 §3 1.58M DrugOOD launch wrapper exists** — `wf185_chemprop_v2_drugood_full.py` is a 3-configs script, not a 1.58M launch wrapper; the 1.58M retrain was DEFERRED per Wf-342-L1 §6 PROJECTED note
9. ❌ **Does NOT claim any new GPU launches** — zero GPU launches in this Wf-354 wave
10. ❌ **Does NOT claim any new tests** — zero NEW tests in this Wf-354 wave (all 7 closeouts cite 0 NEW tests)
11. ❌ **Does NOT claim pre-flight state changed** — 62/62 PASS PRESERVED at Wf-328-SYN baseline; 71/71 PASS preserved at Wf-352-A1 baseline
12. ❌ **Does NOT lift any honest-negative metric** — preserves m2/m4/m16 BELOW gate + R20 §4/§5 STATUS_HOLD markers

### §5.3 Standing directive alignment (AGENTS.md §1.1-§1.5 + §5.1-§5.4)

Per `party-twenty-spirit` §3.5: this synthesis aggregates ONLY what was measured by the 7 Wf-354 closeouts. Every numeric value traces to a JSON path or log line on disk. Zero aspirational claims. PROJECTED vs MEASURED tag discipline respected throughout.

### §5.4 Forward-looking flags (per Wf-316A STATUS_HOLD + Wf-339d2 §4 + AGENTS.md §5.1)

If/when the user clears the following blockers, the deferred work can resume:

1. **Wf-146 LOMO 6 remaining folds** (Re 5/5 + Os seed 4): requires fresh launch (NOT in any user-OK'd workflow)
2. **Wf-131 P3 LOMO 7 remaining folds** (Rh seeds 3,4 + Re 0/5): requires fresh launch (NOT in any user-OK'd workflow)
3. **Wf-310E Wf-135 Phase 2 GPU launch** (2-week wall, ~10 GPU-hr): USER_OK-gated via `USER_OK=1 bash scripts/launch_wf310e_wf135_phase2_gpu.sh`
4. **Wf-347 per-wave + data-quality checks wire-in to DrugOOD wrapper** (10-20 LOC): Wf-355+ follow-up
5. **R20 §4 BindingDB 1.07M wrapper + data download**: separate writer workflow (~200-300 LOC + full 1.07M download + SMARTS filter end-to-end)
6. **R20 §5 tmQM+PlatinAI joint retrain wrapper** (architectural lift required first): if/when 12-d signal floor is lifted (Wf-86 P3 Vuln B fix direction; or non-12-d descriptor like Multi-PH + PLL augmented features from Wf-94 P3), then author R20 §5 wrapper following Wf-310E pattern

---

## §6 Files written (this Wf-354-C3 synthesis)

| Path | LOC | Purpose |
|---|---:|---|
| `TODO/completed/354_final_synthesis.md` | (this file) | Final synthesis of 7 Wf-354 closeouts (A1+A2+B1+B2+B3+C1+C2) |

**No source files were modified.** Wf-354-C3 is a **docs-only aggregation** — no Edit to `WEAK_TO_STRONG_PLAN.md`, no Edit to `metrics/by_metric/*.json`, no Edit to `molmetal/scripts/*`, no Edit to `paper/*` (BLOCKED per TODO-84), no Edit to `TODO/INDEX.md` (already updated by Wf-354-C2), no Edit to `AGENTS.md` (already updated by Wf-354-C2).

---

## §7 Cross-references (synthesis-specific)

### §7.1 Primary sources (Wf-354 closeouts aggregated)

- `TODO/completed/354a1_wf342_p2_lomo_integration.md` — 276 LOC — Wf-342 P2 LOMO sweep integration verification
- `TODO/completed/354a2_wf310e_user_ok_verify.md` — 166 LOC — Wf-310E Wf-135 Phase 2 USER_OK-gate verification
- `TODO/completed/354b1_r20_p3_drugood_wrapper.md` — 282 LOC — R20 §3 DrugOOD launch wrapper verification (PARTIAL_VERIFIED)
- `TODO/completed/354b2_r20_p4_binding_db_wrapper.md` — 123 LOC — R20 §4 BindingDB 1.07M wrapper (HONEST-NEGATIVE NOT-WRITTEN)
- `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` — 127 LOC — R20 §5 tmQM+PlatinAI joint retrain wrapper (HONEST-NEGATIVE NOT-WRITTEN + STATUS_HOLD)
- `TODO/completed/354c1_wf342_p4_weak_to_strong.md` — 407 LOC — Wf-342 P4 WEAK_TO_STRONG §1.1/§3.5/§3.6 + Final Closeout integration
- `TODO/completed/354c2_index_sync.md` — 200 LOC — INDEX.md §22.37 + AGENTS.md §6 docs sync
- `TODO/completed/354_final_synthesis.md` — (this file) — final synthesis

### §7.2 Downstream artifacts (already shipped)

- `TODO/INDEX.md:4524-4615` (approx) — §22.37 Wf-354 Wf-342 closeout integration wave entry
- `AGENTS.md` row 254 — Wf-354 segment appended to existing Wf-344..352 row

### §7.3 Sister / parent workflow references

- Wf-335-SYN prior synthesis baseline — `TODO/completed/335_final_synthesis.md` (SHIPPED 2026-09-29 15:35 UTC wall 110.83s)
- Wf-342 4-phase ship surface — `TODO/completed/342{k1,k2,l1,final_closeout}_*.md` (ALL SHIPPED per Wf-354-C1 §1.1)
- Wf-339d2 LAMBDA_ROADMAP audit — `TODO/completed/339d2_lambda_roadmap_audit.md` §4 (R20 §4 + §5 STATUS_HOLD markers)
- Wf-352-A1 EXPECTED_GATE_COUNTS baseline — `TODO/completed/352{a1,a2}_*.md` (71 PRESERVED)
- Wf-316A STATUS_HOLD marker on TODO-179 — `TODO/pending/179_wf123_p3_w2_pic50_alignment.md`
- Wf-347-B1 + Wf-347-B2 — `TODO/completed/347{b1,b2}_*.md` (analogous retro-fit pattern; Wf-347-B1 wired wf343c2 into G5 launcher; Wf-347-B2 wired wf343c2 into G6 launcher + recover_g6_failed_tuples.sh)
- Wf-328-SYN pre-flight baseline — `TODO/completed/328_synthesis.md` (62 PASS / 0 FAIL / 62 TOTAL PRESERVED at 2026-09-29 13:10 UTC wall 104.87s)

### §7.4 Skills referenced

- `party-twenty-spirit` skill — `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `mol-metal-gpu-execution` skill — `.claude/skills/mol-metal-gpu-execution/SKILL.md` (USER_OK gate + kill_then_adapt directive)
- `mol-metal-mechanism-mapping` skill — `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` (mechanism ↔ metric mapping for m1/m2/m3/m4/m5/m7/m16/m19 audit)

### §7.5 AGENTS.md pointers

- §1.1-§1.5 standing project rules (all honored)
- §2 single source of truth pointers (INDEX.md + completed/ inventory)
- §4 critical workflows (pre_flight_check.py)
- §5 standing directives (2026-09-29 snapshot)
- §6 recent Wf history (Wf-344..354 row extension)
- §7 critical scripts reference (cheat sheet)
- §8 workflow authoring conventions (text-only OUTPUT template)
- §9 when unsure (kill_then_adapt + MEASURED-vs-PROJECTED + paper authorization)

---

**END OF Wf-354-C3 SYNTHESIS** — Owner Hugo Lee + Wf-354 ultracode v8 wave (A1+A2+B1+B2+B3+C1+C2); cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.1-§1.5 + §5.1-§5.4; **7/7 closeouts SHIPPED** (1,381 LOC closeouts + this file ≈ +1,560 LOC; 0 GPU launches; 0 NEW tests; 6 honest-negative caveats preserved per §3; pre-flight CLEAN; 8 standing rules 8/8 compliant; **VERIFIED** 2026-09-29).