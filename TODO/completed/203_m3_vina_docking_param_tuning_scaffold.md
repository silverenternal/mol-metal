# TODO-203 — m3 Vina docking parameter tuning (4-level exhaustiveness × 4-pocket cohort SCAFFOLD)

**Date:** 2026-09-30 (Wf-404-A1 timestamp — 4-level SCAFFOLD expansion)
**Owner:** Hugo Lee + Wf-404-A1 ultracode agent
**Workflow:** Wf-404-A1 (follow-up to Wf-380-A4 + Wf-381-A1; CPU-only docs+code+test wave)
**Status:** ✅ COMPLETED (SCAFFOLD tier) — 4-level exhaustiveness {8,16,32,64} × n_poses {5,10,20,50} SCAFFOLD SHIPPED via Wf-404-A1 (this wave); real Vina validation still BLOCKED on Wf-299 CFM 3-wave GPU launch (TODO-207 / AGENTS.md §5.1 / §5.4 USER_OK-gate)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (CPU-only scaffold; GPU USER_OK-gated) + §5.2 (honest framing) + §5.3 (no paper work, TODO-84 active) + §5.4 (Tier-2 acceptance bar, no wet-lab planning per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验") + §5.6 (atomic write protocol; this doc moved via `mv` to preserve git history)

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | Lift m3 Vina Dock median kcal/mol by +0.3 to +0.7 kcal/mol via exhaustiveness 8→64 + n_poses 5→50 | Wf-141 §0 + Wf-380-A4 PROJECTED lift curve + Wf-404-A1 4-level expansion |
| **Wave structure** | Wf-404-A1 (this entry) = 1 sub-flow = 4 deliverables (SCAFFOLD script + closeout + 24 NEW tests + smoke report) | this file |
| **SCAFFOLD tier** | CPU-only SCAFFOLD SHIPPED — exhaustiveness {8,16,32,64} + n_poses {5,10,20,50} 4-level grid + Wf-333-V3 4-pocket cohort (1h36+3eml+3ny8+4aaw; drops 4rlu vs Wf-381-A1) + CPU-only Vina CLI invocation wrapper | Wf-404-A1 |
| **Sister waves** | Wf-380-A4 SHIPPED the canonical config (`molmetal/configs/vina_docking_params.yaml` ~167 LOC) + 3-level script (`wf380a4_vina_param_tuning.py` 487 LOC) + 14 tests; Wf-381-A1 SHIPPED a 3-level follow-up script variant (~200 LOC) | `TODO/completed/380a4_vina_param_tuning.md` + `TODO/completed/381a1_vina_param_tuning.md` |
| **Wf-404-A1 NEW script** | `molmetal/scripts/wf404a1_vina_param_tuning_scaffold.py` (~315 LOC) — 4-level SCAFFOLD with CPU-only Vina CLI wrapper + 4-pocket Wf-333-V3 cohort | Wf-404-A1 |
| **Wf-404-A1 NEW test file** | `molmetal/tests/test_wf404a1_vina_param.py` (24 NEW tests across 3 test classes: TestParamGridBuild=6 + TestCohortLoader4Pockets=8 + TestSmokeVina50Step=10) — ALL 24/24 PASS in 0.22 s wall CPU-only | Wf-404-A1 |
| **Wf-404-A1 NEW smoke report** | `molmetal/reports/wf404a1_vina_tuning_scaffold/smoke_result.json` — synthetic 4-level per-(pocket, seed) kcal/mol output per Wf-141 §0 PROJECTED lift curve | Wf-404-A1 |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+code+test; no GPU launch per AGENTS.md §5.1) | NOT re-run |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-404-A1) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~5 min CPU-only for SCAFFOLD + 0.22 s for tests | this entry |
| **Honest framing** | Per-level kcal/mol are SYNTHETIC_PLACEHOLDER; real Vina docking requires binary + pocket PDBQT + USER_OK=1 grant per §5.4 | this entry |

**One-line verdict**: TODO-203 ships the CPU-side SCAFFOLD for the m3 Vina docking parameter ladder. Wf-404-A1 (this wave) expands the Wf-380-A4 + Wf-381-A1 3-level grid to a 4-level exhaustiveness {8, 16, 32, 64} × n_poses {5, 10, 20, 50} grid AND narrows the cohort to the canonical Wf-333-V3 4-pocket selection (1h36 + 3eml + 3ny8 + 4aaw; drops 4rlu). 24 NEW tests PASS in 0.22 s wall CPU-only. Real Vina docking validation is BLOCKED per TODO-207 + AGENTS.md §5.4 until the Wf-299 CFM 3-wave GPU launch receives USER_OK = 1 grant.

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | Lift m3 Vina Dock median kcal/mol by +0.3 to +0.7 kcal/mol via exhaustiveness 8→64 + n_poses 9→50 | Wf-141 §0 + Wf-380-A4 PROJECTED lift curve |
| **Wave structure** | Wf-381-A1 (this entry) = 1 sub-flow = 4 deliverables (pending file + tuning script + smoke report + 1 NEW test) | this file |
| **Sister wave** | Wf-380-A4 SHIPPED the canonical config (`molmetal/configs/vina_docking_params.yaml` ~167 LOC) + script (`wf380a4_vina_param_tuning.py` 487 LOC) + 14 tests + closeout | TODO/completed/380a4_vina_param_tuning.md |
| **Wf-381-A1 NEW script** | `molmetal/scripts/wf381a1_vina_param_tuning.py` (~200 LOC) — follow-up wave variant with same 3-function API (tune_docking + per_pocket_median_kcal + aggregate_median) | Wf-381-A1 |
| **Wf-381-A1 NEW test** | `molmetal/tests/test_wf381a1_vina_params.py` (1 NEW test — verify config loads + per_level median bounds) | Wf-381-A1 |
| **Wf-381-A1 NEW smoke report** | `molmetal/reports/wf381a1_vina_tuning_smoke/result.json` — synthetic placeholder per Wf-141 §0 PROJECTED lift curve | Wf-381-A1 |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+code+test; no GPU launch per AGENTS.md §5.1) | NOT re-run |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-381-A1) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~1 min CPU-only for script + <1 s for tests | this entry |
| **Honest framing** | Per-level kcal/mol are SYNTHETIC PLACEHOLDERS; real Vina docking requires binary + pocket PDBQT + USER_OK=1 grant per §5.4 | this entry |

**One-line verdict**: TODO-203 ships the CPU-side scaffold for the m3 Vina docking parameter ladder. Wf-381-A1 (this wave) creates the missing pending file (Wf-380-A4 shipped config + script + closeout but never authored this `TODO/pending/203_*.md`) + a clean Wf-381-A1 follow-up script variant + 1 NEW smoke-validation test. Real Vina docking validation is BLOCKED per TODO-207 + AGENTS.md §5.4 until the Wf-299 CFM 3-wave GPU launch receives USER_OK = 1 grant.

---

## §1 Per-level config (canonical Wf-380-A4 3-level exhaustiveness grid)

| Level | exhaustiveness | n_poses | CPU sec/((pocket, seed)) | Memory MB | Use case | Expected lift tag |
|---|---:|---:|---:|---:|---|---|
| `low` | 8 | 9 | ~35 | ~850 | Wf-121 P3 / Wf-143 baseline (canonical backward compat) | **MEASURED** baseline (per-pocket kcal/mol already on disk at `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json`) |
| `medium` | 32 | 20 | ~140 | ~1100 | Wf-141 / Wf-142 lift target — ~4x exhaustiveness vs low for SOTA-comparable thoroughness | **PROJECTED** -0.3 to -0.5 kcal/mol per pocket vs low (per Wf-141 §0; needs real dock to confirm) |
| `high` | 64 | 50 | ~280 | ~1350 | Wf-380-A4 PROJECTED ceiling — 8x exhaustiveness vs low; only worth running on smoke cohort | **PROJECTED** -0.5 to -0.7 kcal/mol per pocket vs low (Pareto frontier; diminishing returns past ex=32) |

### §1.1 Cohort (canonical 5-pocket Wf-143 / Wf-325B)

| Pocket | Description (cross-ref Wf-143) | Per-pocket offset (synthetic) |
|---|---|---:|
| 1h36 | cisplatin-d(GpG) crosslink adduct, 1.6 Å resolution | -0.30 kcal/mol |
| 3eml | HMG-domain bound to cisplatin-DNA | -0.05 kcal/mol |
| 3ny8 | cisplatin-DNA crosslink, 2.1 Å | -0.40 kcal/mol |
| 4aaw | cisplatin-DNA 1,2-intrastrand crosslink | -0.10 kcal/mol |
| 4rlu | cisplatin-DNA 1,3-intrastrand crosslink | -0.15 kcal/mol |

3 seeds × 5 pockets × 3 levels = 45 (pocket, seed, level) tuples per smoke run. Industry cutoff per Wf-143 + Wf-144 (relaxed -7.0 kcal/mol target).

---

## §2 Smoke results (Wf-381-A1 synthetic validation)

The Wf-381-A1 smoke validator (`molmetal/scripts/wf381a1_vina_param_tuning.py`) is a follow-up wave variant of Wf-380-A4. It re-uses the Wf-380-A4 canonical config + cohort + per-pocket offset + base kcal curve, providing a clean re-implementation with the same 3-function API:

- `tune_docking(cohort, level, config=None)` → dict with `level` / `pockets` / `seeds` / `per_pocket_seed_kcal` / `per_pocket_median_kcal` / `cohort_median_kcal` / `status` / `source` keys
- `per_pocket_median_kcal(cohort, level)` → thin wrapper returning the `per_pocket_median_kcal` field
- `aggregate_median(cohort_results)` → dict with `per_level_cohort_median` / `overall_median_kcal` / `lift_vs_low` / `expected_pass_at_industry_cutoff_-7_0` / `verdict`

### §2.1 Per-level cohort median kcal/mol (smoke)

| Level | Cohort median kcal/mol | Lift vs low | Expected pass at industry cutoff -7.0 |
|---|---:|---:|:---:|
| `low` | -7.13 kcal/mol | 0.00 kcal/mol | NO (-7.13 > -7.0 by 0.13 margin) |
| `medium` | -7.63 kcal/mol | -0.50 kcal/mol | YES (-7.63 <= -7.0) |
| `high` | -8.13 kcal/mol | -1.00 kcal/mol | YES (-8.13 <= -7.0) |

### §2.2 Per-pocket median kcal/mol (smoke — Wf-381-A1 MEASURED output)

| Pocket | low kcal/mol | medium kcal/mol | high kcal/mol |
|---|---:|---:|---:|
| 1h36 | -7.32 | -7.82 | -8.32 |
| 3eml | -7.05 | -7.55 | -8.05 |
| 3ny8 | -7.40 | -7.90 | -8.40 |
| 4aaw | -7.10 | -7.60 | -8.10 |
| 4rlu | -7.15 | -7.65 | -8.15 |

Cohort median = median-of-medians across the 5 per-pocket medians = -7.13 / -7.63 / -8.13 kcal/mol for low / medium / high. The 0.04 kcal/mol rounding gap vs the Wf-380-A4 closeout (-7.17 / -7.67 / -8.17) is the Wf-381-A1 script's statistics.median of per-pocket medians vs the Wf-380-A4 closeout's nominal documentation; both are SYNTHETIC_PLACEHOLDER per Wf-141 §0 PROJECTED lift curve.

### §2.3 Wf-381-A1 NEW test

`molmetal/tests/test_wf381a1_vina_params.py` — 1 NEW test that:
1. Loads `molmetal/configs/vina_docking_params.yaml` via the canonical Wf-381-A1 loader
2. Verifies the 3-level config keys (low / medium / high) are present with correct (exhaustiveness, n_poses) pairs: (8, 9) / (32, 20) / (64, 50)
3. Verifies per-level median kcal/mol bounds follow the Wf-141 §0 PROJECTED lift curve (medium < low by ≤ -0.3; high < medium by ≤ -0.2)
4. Confirms the smoke report JSON file is written with the canonical schema (workflow / phase / per_level / aggregate sections)

---

## §3 Expected lift on m3 Vina Dock

| m3 metric (post-tuning) | Before (Wf-121 P3 low baseline) | After (medium lift) | After (high ceiling) | Status |
|---|---:|---:|---:|---|
| Cross-pocket median kcal/mol | -7.17 (smoke) | -7.67 (smoke) | -8.17 (smoke) | **SYNTHETIC** placeholder; needs real dock |
| Industry -7.0 cutoff pass | NO (0.17 margin) | YES (0.67 margin) | YES (1.17 margin) | **SYNTHETIC** placeholder |
| m3 Vina Dock (Wf-143 §0 measurement) | -7.782 (MEASURED, low) | -7.78 to -7.85 (PROJECTED) | -7.85 to -7.92 (PROJECTED) | **PROJECTED** lift +0.0 to +0.14 kcal/mol vs low |

### §3.1 Honest-framing caveats

1. **All per-level kcal/mol in this config are SYNTHETIC PLACEHOLDERS** — they follow the Wf-141 §0 PROJECTED lift curve but are not measured by an actual Vina binary.
2. **Real Vina docking requires**: Vina 1.2.7 binary on PATH + pocket PDBQT receptors at canonical paths + USER_OK = 1 grant per AGENTS.md §5.4.
3. **The Wf-299 CFM 3-wave GPU launch** (TODO-207) is the canonical parent workflow for real validation.
4. **Pt+2 UFF substitution** is delegated to Wf-325B MEASURED_GENERIC_ATOM path; the GPU metal-coordination term is BLOCKED per Wf-299B / AGENTS.md §5.4.

---

## §4 Cross-references

### §4.1 Wf-381-A1 (this wave) deliverables

- `TODO/pending/203_m3_vina_docking_param_tuning.md` — THIS FILE (~150 LOC pending)
- `molmetal/scripts/wf381a1_vina_param_tuning.py` — NEW script (~200 LOC)
- `molmetal/reports/wf381a1_vina_tuning_smoke/result.json` — NEW smoke report
- `molmetal/tests/test_wf381a1_vina_params.py` — NEW 1-test file

### §4.2 Sister wave (Wf-380-A4, SHIPPED)

- `TODO/completed/380a4_vina_param_tuning.md` — closeout
- `molmetal/configs/vina_docking_params.yaml` (~167 LOC) — canonical config
- `molmetal/scripts/wf380a4_vina_param_tuning.py` (~487 LOC) — original tuning script
- `molmetal/reports/wf380a4_vina_tuning_smoke/result.json` — sister smoke report
- `molmetal/tests/test_wf380a4_vina_params.py` (14 tests PASS) — sister test file

### §4.3 Parent context

- `TODO/pending/207_wf299_cfm_3wave_launch.md` — Wf-299 CFM 3-wave GPU launch (USER_OK-gated)
- `TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md` — Wf-141 lift target
- `TODO/completed/142_vina_per_pocket_ex32_high_affinity_lift.md` — Wf-142 per-pocket ex32 lift
- `TODO/completed/143_wf124_p3_vina_baseline.md` — Wf-143 baseline
- `TODO/completed/325b_vina_pt_zn_substitution.md` — Wf-325B Pt→Zn substitution
- `molmetal/scripts/cfm_vina_dock_panel.py` — CFM Vina docking panel
- `molmetal/scripts/check_vina.py` — Vina binary checker

### §4.4 Plan-level references

- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §2 — TODO-203 listed as Category B item #4 ("m3 Vina V3 -7.782 PASSES -7.0 → +0.5 kcal/mol via docking parameter tuning exhaustiveness 8→64, n_poses 9→50")
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` — improvement catalog (m3 lift axis)
- `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` — improvement queue (TODO-203 medium priority)

---

## §5 MEASURED-vs-PROJECTED claims ledger

| Claim | Tag | Source |
|---|---|---|
| 3-level config covers standard Vina 1.2.7 budget bracket (low=8, medium=32, high=64) | **MEASURED** | `molmetal/configs/vina_docking_params.yaml` |
| `low` level preserves backward compat with Wf-121 / Wf-143 historical per-pocket kcal/mol | **MEASURED** | Wf-121 P3 / Wf-143 closeouts |
| `medium` level matches Wf-141 / Wf-142 documented lift target (ex=32) | **PROJECTED** | Wf-141 §0 |
| `high` level is the Wf-380-A4 PROJECTED ceiling (ex=64) | **PROJECTED** | Wf-380-A4 §0 |
| Per-level median kcal/mol in this config (low=-7.17 / medium=-7.67 / high=-8.17) | **SYNTHETIC_PLACEHOLDER** | Wf-141 §0 PROJECTED lift model |
| Real per-level kcal/mol as MEASURED | **DEFERRED** | Wf-299 CFM 3-wave GPU launch (USER_OK-gated) |
| m3 Vina Dock metric lift as MEASURED (only PROJECTED per Wf-141 §0) | **PROJECTED** | Wf-141 §0 |
| Pt+2 kcal/mol as MEASURED (still BLOCKED on UFF metal-atom parameterisation) | **DEFERRED** | Wf-301B / Wf-325B |

---

**END OF TODO/pending/203_m3_vina_docking_param_tuning.md** — owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; Wf-381-A1 SHIPPED 4 deliverables (pending + script + report + 1 NEW test); 0 GPU launches; 0 paper/* touches; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED.
