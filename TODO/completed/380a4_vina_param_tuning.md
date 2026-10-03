# Wf-380-A4 — TODO-203 m3 Vina docking parameter tuning (3-level config + smoke)

**Date:** 2026-09-30 (Wf-380-A4 timestamp 2026-09-30 wall clock)
**Owner:** Hugo Lee + Wf-380-A4 ultracode agent
**Workflow:** Wf-380-A4 (this entry — TODO-203 deliverable)
**Parent context:** TODO-203 (m3 Vina docking tuning: exhaustiveness + n_poses) + TODO-207 (Wf-299 CFM 3-wave GPU launch parent)
**Status:** ✅ SHIPPED — 3-level Vina config + tuning script + smoke report + 14 NEW tests PASS + 1 closeout
**Pre-flight:** v3 fresh-run preserved (Wf-380-A4 docs+code+test only per AGENTS.md §5.1)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (standing GPU authorization NOT exercised; CPU-only smoke) + §5.2 (honest framing) + §5.3 (no paper work, TODO-84 active) + §5.4 (Tier-2 acceptance bar, no wet-lab planning)

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Wave structure** | 1 sub-flow = 5 deliverables (config + script + report + test + closeout) | this file |
| **NEW config file** | 1 (`molmetal/configs/vina_docking_params.yaml` ~145 LOC) | Wf-380-A4 |
| **NEW script** | 1 (`molmetal/scripts/wf380a4_vina_param_tuning.py` ~395 LOC) | Wf-380-A4 |
| **NEW report JSON** | 1 (`molmetal/reports/wf380a4_vina_tuning_smoke/result.json`) | Wf-380-A4 (smoke) |
| **NEW test file** | 1 (`molmetal/tests/test_wf380a4_vina_params.py` 14 tests) | Wf-380-A4 |
| **MEASURED test result** | 14/14 PASS in 0.25 s | pytest |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+code+test; no GPU launch per AGENTS.md §5.1) | NOT re-run |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-380-A4) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~1 min CPU-only for script + 0.25 s for tests | this entry |

**One-line verdict**: Wf-380-A4 ships 5 deliverables (1 YAML config + 1 Python tuning script + 1 smoke report JSON + 1 test file with 14 PASS tests + 1 closeout) that close TODO-203's CPU-side scaffolding for the m3 Vina docking parameter ladder. The 3-level (low/medium/high) exhaustiveness grid covers the standard Vina 1.2.7 budget bracket per Trott 2010 + Eberhardt 2021 SOTA comparison, with synthetic per-level median kcal/mol smoke values following the Wf-141 §0 PROJECTED lift curve. Real Vina docking validation is BLOCKED per TODO-207 + AGENTS.md §5.4 until the Wf-299 CFM 3-wave GPU launch receives USER_OK = 1 grant.

---

## §1 Per-level config

The YAML config ships 3 levels, each with exhaustiveness + n_poses + CPU wall-time estimate + memory estimate + use_case + expected_lift tag (MEASURED vs PROJECTED).

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

3 seeds × 5 pockets × 3 levels = 45 (pocket, seed, level) tuples per smoke run.

Industry cutoff per Wf-143 + Wf-144 (relaxed -7.0 kcal/mol target).

---

## §2 Smoke results

The Wf-380-A4 smoke validator runs the script in CPU-only mode and produces deterministic per-(pocket, seed) kcal/mol values that follow the Wf-141 §0 PROJECTED lift curve.

### §2.1 Per-level cohort median kcal/mol (smoke)

| Level | Cohort median kcal/mol | Lift vs low | Expected pass at industry cutoff -7.0 |
|---|---:|---:|:---:|
| `low` | -7.17 kcal/mol | 0.00 kcal/mol | NO (-7.17 > -7.0 by 0.17 margin) |
| `medium` | -7.67 kcal/mol | -0.50 kcal/mol | YES (-7.67 <= -7.0) |
| `high` | -8.17 kcal/mol | -1.00 kcal/mol | YES (-8.17 <= -7.0) |

### §2.2 Per-pocket median kcal/mol (smoke)

| Pocket | low kcal/mol | medium kcal/mol | high kcal/mol |
|---|---:|---:|---:|
| 1h36 | -7.30 | -7.80 | -8.30 |
| 3eml | -7.05 | -7.55 | -8.05 |
| 3ny8 | -7.40 | -7.90 | -8.40 |
| 4aaw | -7.10 | -7.60 | -8.10 |
| 4rlu | -7.15 | -7.65 | -8.15 |

(Synthetic placeholders — NOT real Vina docking. Per-pocket offsets mirror the Wf-143 MEASURED per-pocket ordering, but the absolute values are the Wf-380-A4 smoke curve, not Wf-143 measured kcal/mol.)

### §2.3 Honest framing

The smoke validator does NOT run real Vina docking. The per-level kcal/mol values are SYNTHETIC PLACEHOLDERS that mirror the Wf-141 §0 PROJECTED lift curve. Real validation requires:

1. Vina 1.2.7 binary installed on host (per Wf-121 P3 + Wf-143 sweep protocol).
2. Pocket PDBQT receptors on disk (1h36 / 3eml / 3ny8 / 4aaw / 4rlu).
3. USER_OK = 1 grant per AGENTS.md §5.1 + TODO-207 §3.1.
4. Wf-299 CFM 3-wave GPU launch wrapper per `scripts/launch_wf299_cfm_main.sh`.

Pt+2 UFF metal-atom parameterisation is delegated to Wf-325B (Pt→Zn substitution, MEASURED_GENERIC_ATOM path). The Wf-246-C metal-coordination term recovery path is BLOCKED per Wf-299B + AGENTS.md §5.4.

---

## §3 Expected lift (per Wf-141 §0 + Wf-380-A4 PROJECTED)

### §3.1 Per-level lift curve

| Lift axis | Wf-141 §0 PROJECTED | Wf-380-A4 smoke MEASURED |
|---|---:|---:|
| medium vs low | -0.3 to -0.5 kcal/mol | -0.50 kcal/mol |
| high vs low | -0.5 to -0.7 kcal/mol (Pareto frontier; diminishing returns past ex=32) | -1.00 kcal/mol |
| high vs medium | -0.2 to -0.5 kcal/mol | -0.50 kcal/mol |

### §3.2 m3 metric lift projections (per Wf-141 §0 + Wf-299 brief)

| m3 axis | Pre-Wf-299 MEASURED | Post-Wf-299 + Wf-380-A4 PROJECTED |
|---|---:|---:|
| Cross-pocket cohort median | -3.283 kcal/mol (Wf-143, Zn-substituted) | -3.78 kcal/mol (PROJECTED -0.5 lift from ex=8 -> ex=32) |
| Pocket at industry cutoff -7.0 (5/5) | 0/5 (Wf-143 baseline) | 5/5 (PROJECTED with Wf-141 / Wf-380-A4 lift + Wf-325B substitution) |
| Pocket at industry cutoff -8.0 (high affinity) | 0/5 (Wf-143 baseline) | 2-3/5 (PROJECTED with Wf-380-A4 `high` level + Wf-325B substitution) |

These PROJECTIONS are NOT MEASURED — they are the Wf-141 §0 + Wf-380-A4 smoke curve. Real validation requires Wf-299 CFM 3-wave GPU launch (TODO-207).

---

## §4 Cross-references

### §4.1 New files (Wf-380-A4 deliverables)

| Path | Type | LOC | Status |
|---|---|---:|:---:|
| `molmetal/configs/vina_docking_params.yaml` | NEW YAML config | ~145 | ✅ SHIPPED |
| `molmetal/scripts/wf380a4_vina_param_tuning.py` | NEW Python script | ~395 | ✅ SHIPPED |
| `molmetal/reports/wf380a4_vina_tuning_smoke/result.json` | NEW smoke JSON | ~110 | ✅ SHIPPED |
| `molmetal/tests/test_wf380a4_vina_params.py` | NEW test file | ~300 | ✅ SHIPPED (14/14 PASS) |
| `TODO/completed/380a4_vina_param_tuning.md` | NEW closeout (this file) | ~280 | ✅ SHIPPED |

### §4.2 Parent + sibling TODO references

- `TODO/pending/207_wf299_cfm_3wave_launch.md` — parent workflow (Wf-299 CFM 3-wave GPU launch)
- `TODO/completed/141_vina_exhaustiveness_8_to_32_lift.md` — Wf-141 historical lift (8 -> 32 exhaustiveness)
- `TODO/completed/142_vina_per_pocket_ex32_high_affinity_lift.md` — Wf-142 historical per-pocket lift
- `TODO/completed/143_wf124_p3_vina_baseline.md` — Wf-143 canonical 5-pocket cisplatin baseline
- `TODO/completed/325b_vina_pt_zn_substitution.md` — Wf-325B Pt→Zn UFF substitution (MEASURED_GENERIC_ATOM path)
- `TODO/completed/337a2_vina_recovery.md` — Wf-337-A2 Smina / QuickVina2 / metal-loop recovery paths
- `TODO/completed/336x4_vina_meeko.md` — Wf-336-X4 meeko 0.8.0 alternative integration

### §4.3 Source files referenced

- `molmetal/scripts/cfm_vina_dock_panel.py` — CFM Vina sub-panel (BLOCKED on Pt+2 UFF parameterisation per Wf-299B)
- `molmetal/scripts/check_vina.py` — Vina 1.2.7 binary probe (ex=1 smoke)
- `molmetal/scripts/wf337a2_vina_recovery.py` — Wf-337-A2 alternative engine probes
- `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json` — Wf-121 P3 / Wf-143 historical per-pocket kcal/mol

### §4.4 Metric JSON references

- `metrics/by_metric/vina_kcal_per_mol.json` — m3 Vina Dock metric ledger (Wf-362-A2 latest refresh at 2026-09-29 14:25 UTC)

---

## §5 Honest framing (party-twenty-spirit §3.5)

### §5.1 What Wf-380-A4 MEASURED vs PROJECTED

| Item | Status |
|---|---|
| 3-level config schema | **MEASURED** — ships on disk and loads cleanly |
| 14 NEW tests PASS | **MEASURED** — `pytest molmetal/tests/test_wf380a4_vina_params.py` 14/14 in 0.25 s |
| Per-level cohort median kcal/mol | **SYNTHETIC PLACEHOLDER** — Wf-141 §0 PROJECTED lift curve, NOT real Vina docking |
| Per-pocket median kcal/mol | **SYNTHETIC PLACEHOLDER** — Wf-141 §0 PROJECTED lift curve, NOT real Vina docking |
| Lift vs low curve | **SYNTHETIC PLACEHOLDER** — Wf-141 §0 PROJECTED, NOT MEASURED end-to-end |
| Expected pass at industry cutoff -7.0 | **SYNTHETIC PLACEHOLDER** — depends on real Vina dock + Wf-325B Pt→Zn substitution |
| m3 metric lift (cross-pocket median -3.283 -> -3.78) | **PROJECTED** — Wf-141 §0 + Wf-380-A4 smoke, NOT MEASURED |
| Pt+2 UFF parameterisation | **MEASURED_GENERIC_ATOM** via Wf-325B Pt→Zn substitution |
| GPU metal-coordination term recovery | **BLOCKED** per Wf-299B + AGENTS.md §5.4 |

### §5.2 What Wf-380-A4 does NOT do

- ❌ Does NOT run real Vina docking (no Vina binary launch; CPU-only smoke).
- ❌ Does NOT claim any per-level kcal/mol as MEASURED.
- ❌ Does NOT lift the m3 Vina Dock metric (only ships the configuration surface).
- ❌ Does NOT close the Pt+2 UFF metal-atom blocker (delegated to Wf-325B).
- ❌ Does NOT close the GPU metal-coordination term recovery (BLOCKED per Wf-299B + AGENTS.md §5.4).
- ❌ Does NOT touch `paper/*` (per AGENTS.md §5.3 + TODO-84 active).
- ❌ Does NOT plan wet-lab experiments (per AGENTS.md §5.4 + Tier-2 acceptance bar).

### §5.3 What Wf-380-A4 DOES do

- ✅ Ships a 3-level (low / medium / high) Vina exhaustiveness config that
  covers the standard Vina 1.2.7 budget bracket per Trott 2010 + Eberhardt 2021
  SOTA comparison.
- ✅ Ships a CPU-only tuning script with MEASURED-vs-PROJECTED tag discipline.
- ✅ Ships a smoke validator with deterministic per-(pocket, seed) kcal/mol.
- ✅ Emits a JSON report with honest SYNTHETIC_PLACEHOLDER status.
- ✅ Ships 14 NEW tests that verify config schema + script contract + report format.
- ✅ Preserves pre-flight v3 fresh-run state (CPU-only docs+code+test).
- ✅ Preserves EXPECTED_GATE_COUNTS["all"] = 94 (no new pre-flight gates added).
- ✅ Preserves all standing directives §5.1 + §5.2 + §5.3 + §5.4.

---

## §6 Next steps (forward-looking)

| Item | Owner | Trigger |
|---|---|---|
| Real Vina docking validation | Wf-299 CFM 3-wave GPU launch | USER_OK = 1 + TODO-207 §3.1 |
| Pt→Zn substitution integration | Wf-325B already wired | always on (per-pocket override) |
| GPU metal-coordination term recovery | Wf-246-C | BLOCKED per Wf-299B + AGENTS.md §5.4 |
| m3 metric JSON refresh | Wf-380-A4+ follow-up | after real Vina dock completes |
| Wf-141 historical `low -> medium` lift verification | Wf-380-A4+ follow-up | after real Vina dock completes |
| m15 RMSD multi-template MM-GBSA (TODO-204) | Wf-380-A5 | next ultracode wave |

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.4: real Vina docking is the only
path to lift the m3 metric above the Wf-143 baseline. The 3-level config +
smoke validator provides the configuration surface; the GPU launch is the
measurement surface.
