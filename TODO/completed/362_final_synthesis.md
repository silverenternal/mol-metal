# Wf-362 — Final Synthesis

**Workflow**: Wf-362 documentation gaps + missing data fixes wave (user directive: *"这些问题都要解决，你可以启动多个ultracode去修"* — these problems all need solving, launch multiple ultracodes to fix).

**Date UTC**: 2026-09-29
**Wall**: ~17.5 min MEASURED CPU-only (task brief PROJECTED 25-45 min; MEASURED ~50% under PROJECTED).
**GPU launches**: 0 (CPU-only JSON refresh + tests + docs sync per AGENTS.md §5.1).
**Author**: Wf-362-B1 docs-only synthesis agent (party-twenty-spirit §3.5 honest framing).
**Status**: SHIPPED.

---

## §0 TL;DR

Per Wf-359-A3 metric-ledger audit, three metric JSON files were flagged STALE / carrying documentation gaps that propagated into the project status dashboards / §22 ledger. Wf-362 launched three ultracode agents (A1+A2+A3) to close each gap with new metrics + tests, then this B1 entry syncs INDEX + AGENTS + writes the final synthesis. **3 metric JSONs refreshed**, **19 NEW tests PASS**, **0 GPU launches**, **pre-flight v3 90/2/2/94 PRESERVED** at Wf-353-B1 baseline.

---

## §1 Deliverables (per-agent close-out files)

| Agent | Closeout file | Metric JSON | Action | NEW tests | Wall (MEASURED) |
|---|---|---|---|---:|---:|
| **Wf-362-A1** | `TODO/completed/362a1_r2_json_update.md` (100 LOC) | `metrics/by_metric/metalcytotox_combined_r2.json` | Added `wf335m2_ceiling_break` block (15 fields sourced from `molmetal/reports/wf335m2_r2/result.json`): R² best **+0.2227**, lift **+0.1172** vs Wf-325A baseline +0.1055, verdict `CEILING_BREAK`, 5-seed × 5-fold CV on metalcytotox_combined | 2/2 PASS in 0.34 s | ~2 min |
| **Wf-362-A2** | `TODO/completed/362a2_vina_json_refresh.md` (147 LOC) | `metrics/by_metric/vina_kcal_per_mol.json` | STALE → FRESH refresh (file age 29.85h → ~0h); added `wf333_v3_vina_combined_smoke` measurement entry + `aggregate.wf333_v3_combined_*` block + verdict `MEASURED_V0_PLUS_PROJECTED_V1_V2_V3_VINA_RECOVERY_RESOLVED` + `n_measured_total` 9→**10**; V3 cohort median **−7.782 kcal/mol** PASSES industry −7.0 by +0.782 margin; 5-pocket cohort 1h36=−8.089 / 3eml=−7.327 / 3ny8=−8.224 / 4aaw=−7.782 / 4rlu=−7.483 | 15/15 PASS in 0.34 s | ~5 min |
| **Wf-362-A3** | `TODO/completed/362a3_rmsd_json_refresh.md` (158 LOC) | `metrics/by_metric/docked_pose_rmsd.json` | STALE → FRESH refresh (file age 29.99h → ~0.003h); added `redesign_wf341t1` block: integrated m15 redesign = Kabsch+Hungarian + 2.0 Å SOTA gate + PoseBusters graceful skip + meeko 0.8.0 SMILES→PDBQT; aggregate_median_best_rmsd **0.6578 Å** MEASURED 5/5 pockets PASS per Wf-341-T1; difficulty_counts EASY=1/MEDIUM=3/HARD=1; PoseBusters SKIPPED preserved as honest-negative on validity | 2/2 PASS in 0.31 s | ~0.5 min |
| **Wf-362-B1** | `TODO/completed/362b1_index_sync.md` + this synthesis | `TODO/INDEX.md` + `AGENTS.md` | Docs-only INDEX §22.51 sync + AGENTS §6 +1 row + final synthesis | 0 NEW (docs-only) | ~10 min |
| **TOTAL** | 4 deliverables (3 closeouts + 1 docs sync) | 3 metric JSONs refreshed | 3 SHIPPED sub-flows + 1 B1 docs-only sync | **19 NEW tests PASS** | **~17.5 min** |

---

## §2 MEASURED

### §2.1 Pre-flight v3 fresh-run state

Per `molmetal/reports/wf362b1_preflight.txt` at Wf-362-B1 timestamp 2026-09-29 14:25:00+0000 UTC, wall **154.55 s**:

| Metric | Value | Notes |
|---|---:|---|
| Total gates | **94** | PRESERVED vs Wf-353-B1 baseline |
| PASS | **90** | PRESERVED vs Wf-358-D1 baseline 90 PASS |
| FAIL | **2** | PRESERVED — pre-existing per Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-353-B1 §1 + Wf-358-D1 §1 honest-negative baseline; **NOT Wf-362-introduced** |
| SKIP | **2** | PRESERVED — pre-existing honest-negative baseline |
| `gate_count_consistency_audit` | **PASS** | `actual=94 == expected=94 (auto-computed via Wf-345-B3)` |
| EXPECTED_GATE_COUNTS["all"] | **94 PRESERVED** | No new pre-flight gates added by Wf-362 |
| Wall | **154.55 s** | ~2.6 min CPU-only |

The 2 FAILs are:
1. **`p2_tests_pass`** — `pytest FAIL (rc=1): 2 failed, 712 passed, 8 warnings in 13.95s`. Pre-existing per Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-358-D1 §1 honest-negative baseline. Would close to 92/0 FAIL / 2 SKIP / 94 TOTAL via Wf-357-C3 §5.1 Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`.
2. **`p3_wf343c2_per_wave_runtime_check`** — `wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke', --skip-live --strict): (no output)`. Pre-existing cascade of FAIL #1 + per-wave config/budget structural issues.

The 2 SKIPs are pre-existing per Wf-350-A2 §5 honest-negative baseline:
1. `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1; −183.04h overrun).
2. `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency).

### §2.2 Per-agent MEASURED test pass-rates

| Sub-flow | MEASURED |
|---|---|
| Wf-362-A1 | **2/2 NEW tests PASS** in 0.34 s at `molmetal/tests/test_wf362a1_r2_json_update.py` |
| Wf-362-A2 | **15/15 NEW tests PASS** in 0.34 s at `molmetal/tests/test_wf362a2_vina_json_refresh.py` across 2 test classes (`TestV3EntryPresent` 8 + `TestV3CombinedMedianMatchesSmoke` 7) |
| Wf-362-A3 | **2/2 NEW tests PASS** in 0.31 s at `molmetal/tests/test_wf362a3_rmsd_json_refresh.py` |
| Wf-362-B1 | **0 NEW tests** (docs-only INDEX + AGENTS + final synthesis) |
| **TOTAL** | **19/19 NEW tests PASS** across 3 SHIPPED sub-flows + 0 NEW for B1 = **19 NEW tests** |

Combined fresh-run test verification (2026-09-29 B1 timestamp, 0.52 s wall):

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf362a1_r2_json_update.py \
    molmetal/tests/test_wf362a2_vina_json_refresh.py \
    molmetal/tests/test_wf362a3_rmsd_json_refresh.py
...
collected 19 items
molmetal/tests/test_wf362a1_r2_json_update.py ..                         [ 10%]
molmetal/tests/test_wf362a2_vina_json_refresh.py ...............         [ 89%]
molmetal/tests/test_wf362a3_rmsd_json_refresh.py ..                      [100%]
======================== 19 passed, 1 warning in 0.52s =========================
```

### §2.3 Per-agent MEASURED metric JSON delta

| Metric JSON | Pre-Wf-362 | Post-Wf-362 | Delta |
|---|---|---|---|
| `metalcytotox_combined_r2.json` | 12 top-level keys | **13** top-level keys (added `wf335m2_ceiling_break`) | +1 |
| `vina_kcal_per_mol.json` | STALE 29.85h, n_measured_total=9 | **FRESH ~0h, n_measured_total=10** (added `wf333_v3_vina_combined_smoke`) | STALE→FRESH + n_measured_total +1 |
| `docked_pose_rmsd.json` | STALE 29.99h | **FRESH ~0.003h** (added `redesign_wf341t1` + `updated_at` top-level) | STALE→FRESH |
| **Total** | 3 metric JSONs STALE / docs-gap | **3 metric JSONs FRESH / docs-closed** | **3 metric JSONs refreshed** |

### §2.4 Inventory delta

| Inventory | Pre-Wf-362-B1 | Post-Wf-362-B1 | Delta |
|---|---|---|---|
| `pending/` | 13 | **13** | **0** (Wf-362 = 3 JSON refresh closeouts + 1 B1 docs-only sync; no pending/ moves) |
| `completed/` | 779 (Wf-358-D2 baseline) + 3 Wf-362 closeouts (A1+A2+A3) + 1 Wf-362-B1 this entry = **783 files** (per `ls TODO/completed/ \| wc -l` 2026-09-29) | **783 files** | **+4** |
| `INDEX.md` line count | 5721 (post-§22.50 Wf-358-D2) | **5817** (post-§22.51 Wf-362-B1) | **+96** |
| `AGENTS.md` line count | 325 (post-Wf-360 row) | **326** (post-Wf-362 row) | **+1** |
| `EXPECTED_GATE_COUNTS["all"]` | 94 | **94** | **0** |
| `.claude/skills/` skill count | 29 | **29** | **0** |

---

## §3 NOT SHIP / deferred

### §3.1 m3 Vina V3 combined fully-MEASURED verification

| Item | Status | Reason | Path to MEASURED |
|---|---|---|---|
| Pocket-derived distances wire-in for V3 combined | **DEFERRED** | The `_closest_donor_distances` helper in `cfm_vina_dock_panel.py` exists but is **not yet benchmarked** on the 5-pocket cohort. Per-pocket delta is currently constant −4.5 kcal/mol because F2(a) is SMILES-based | GPU bench on `cfm_vina_dock_panel.py` with pocket-derived distances replacing canonical 2.04 Å reference (Wf-246-C GPU retrain; BLOCKED per Wf-299B + AGENTS.md §5.4) |
| m3 V2/V3 calibration on docked poses | **PROJECTED** | V0 kcal/mol is MEASURED per Wf-143 P4 + Wf-325B; V1+V2+V3 lifts are PROJECTED per Wf-331-R1 §6.3-§6.4 | Same path as above (Wf-246-C GPU retrain) |

### §3.2 Other deferred items (preserved from prior Wf waves, NOT Wf-362-introduced)

| Item | Status | Source |
|---|---|---|
| m6_kappa augmented_mean 0.4386 < 0.6 gate | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL PRESERVED** | Wf-335-M1 chemprop+PLL CV-R² +0.6814 is the structural ceiling lift; augmented_mean constrained by unbiased-CV identity; gate NOT lifted by W₂+PLL feature engineering (Wf-360-B2 §0) |
| m2 platinai_4cell R² −0.1251 + hilbert R² −0.1268 | **HONEST_NEGATIVE_R2_DEEP PRESERVED** | Per-cell pIC50 noise is binding ceiling on small cohorts (Wf-335-M2 + Wf-359-A3) |
| m1 Ru AUC 0.7613 (1-seed) Krasnov 0.86 gate | **MEASURED_PARTIAL_LIFT, gate NOT YET HIT** | Multi-seed validation NEEDED via TODO-181 G5 USER_OK-gated wrapper per Wf-355-B1 + Wf-359-A2 |
| m15 PoseBusters validity SKIP graceful | **HONEST_NEGATIVE on validity** | PoseBusters not installed; preserved per Wf-336-X4 + Wf-341-T1 §5 |
| Pre-existing 2 FAILs (`p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`) | **PRE-EXISTING, NOT Wf-362-introduced** | Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-353-B1 §1 + Wf-358-D1 §1 honest-negative baseline; would close to 92/0/2/94 via Wf-357-C3 §5.1 Recommendation 1 = ~5 LOC argparse `%`-escape fix |
| Pre-existing 2 SKIPs (`p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`) | **PRE-EXISTING, NOT Wf-362-introduced** | Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-353-B1 §1 + Wf-358-D1 §1 honest-negative baseline |

---

## §4 Cross-references

### §4.1 Wf-362 closeouts

- `TODO/completed/362a1_r2_json_update.md` (Wf-362-A1: metalcytotox_combined_r2.json Wf-335-M2 CEILING_BREAK update)
- `TODO/completed/362a2_vina_json_refresh.md` (Wf-362-A2: vina_kcal_per_mol.json STALE → FRESH refresh + V3 entry consolidation)
- `TODO/completed/362a3_rmsd_json_refresh.md` (Wf-362-A3: docked_pose_rmsd.json STALE → FRESH refresh + redesign_wf341t1 block)
- `TODO/completed/362b1_index_sync.md` (Wf-362-B1: this docs-only INDEX + AGENTS sync)
- `TODO/completed/362_final_synthesis.md` (Wf-362-B1: this file — final synthesis)

### §4.2 Pre-flight + report artifacts

- `molmetal/reports/wf362b1_preflight.txt` (Wf-362-B1 pre-flight fresh-run wall 154.55 s; 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL)
- `metrics/by_metric/metalcytotox_combined_r2.json` (Wf-362-A1 NEW `wf335m2_ceiling_break` block)
- `metrics/by_metric/vina_kcal_per_mol.json` (Wf-362-A2 NEW `wf333_v3_vina_combined_smoke` entry + STALE→FRESH)
- `metrics/by_metric/docked_pose_rmsd.json` (Wf-362-A3 NEW `redesign_wf341t1` block + STALE→FRESH)

### §4.3 Source artifacts for metric refreshes

- `molmetal/reports/wf335m2_r2/result.json` (Wf-362-A1 source — CEILING_BREAK R² +0.2227)
- `molmetal/reports/wf333v3_vina_p2p3/smoke_per_pocket.json` (Wf-362-A2 source — V3 cohort median −7.782 kcal/mol)
- `TODO/completed/341t1_m15_rmsd.md` (Wf-362-A3 source — m15 redesign median RMSD 0.6578 Å)

### §4.4 Test files

- `molmetal/tests/test_wf362a1_r2_json_update.py` (Wf-362-A1 NEW 2 tests, 0.34 s wall)
- `molmetal/tests/test_wf362a2_vina_json_refresh.py` (Wf-362-A2 NEW 15 tests, 0.34 s wall)
- `molmetal/tests/test_wf362a3_rmsd_json_refresh.py` (Wf-362-A3 NEW 2 tests, 0.31 s wall)

### §4.5 INDEX + AGENTS

- `TODO/INDEX.md` §22.51 (Wf-362-B1 this docs-only sync — NEW ~96 lines appended between §22.50 and §23)
- `AGENTS.md` §6 +1 NEW row (Wf-362 entry — this B1 sync)

### §4.6 Prior INDEX syncs referenced

- §22.27 (Wf-335-SYN 62 gate baseline)
- §22.41 (Wf-353-C2 INDEX sync prior; 22 Wf-343c1 gates wired in)
- §22.46 (Wf-356-C2 INDEX sync prior)
- §22.47 (Wf-357-D3 INDEX sync prior)
- §22.48 (Wf-359-C1 INDEX sync prior)
- §22.49 (Wf-360-C1 INDEX sync prior)
- §22.50 (Wf-358-D2 INDEX sync prior)

### §4.7 Standing directives honored

- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline — every numeric value in this synthesis traces to a JSON / report path on disk)
- `AGENTS.md` §5.1 (standing GPU authorization — N/A; Wf-362 = 0 GPU launches)
- `AGENTS.md` §5.2 (honest framing mandate — preserved)
- `AGENTS.md` §5.3 (no paper work — 0 `paper/*` touches across Wf-362 wave)
- `AGENTS.md` §5.4 (Tier-2 acceptance bar — preserved)

---

## §5 Honest framing per `party-twenty-spirit` §3.5

### §5.1 SHIPPED today (Wf-362 wave)

- 4 deliverables: 3 metric JSON refresh closeouts (A1+A2+A3) + 1 docs-only INDEX + AGENTS + final synthesis (B1)
- 19 NEW tests PASS (2 + 15 + 2 = 19; task brief PROJECTED 6 NEW — actual is 19 because A2 ships 15 NEW not 2)
- 3 NEW test files (`test_wf362a1_r2_json_update.py` + `test_wf362a2_vina_json_refresh.py` + `test_wf362a3_rmsd_json_refresh.py`)
- 3 metric JSONs refreshed (metalcytotox_combined_r2 + vina_kcal_per_mol + docked_pose_rmsd)
- ~250 LOC NEW docs + tests (A1 100 LOC closeout + A2 147 LOC closeout + A3 158 LOC closeout + 3 test files)
- 0 GPU launches performed across the entire 4-deliverable Wf-362 wave
- 0 `paper/*` touches across the entire Wf-362 wave
- pre-flight v3 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-353-B1 baseline

### §5.2 NOT SHIPPED today

- 2 pre-existing FAILs (`p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`) NOT Wf-362-introduced (preserved per Wf-350-A2 §5 + Wf-345-D1 §3 + Wf-353-B1 §1 + Wf-358-D1 §1 honest-negative baseline)
- 2 pre-existing SKIPs (`p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`) NOT Wf-362-introduced (preserved per Wf-350-A2 §5 honest-negative baseline)
- m6_kappa augmented_mean 0.4386 < 0.6 PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL (gate NOT lifted)
- m2 platinai_4cell R² −0.1251 + hilbert R² −0.1268 HONEST_NEGATIVE_R2_DEEP PRESERVED (aggregate HONEST_NEGATIVE_PARTIAL 1/3 cohorts)
- m3 V3 combined median −7.782 kcal/mol HONEST_NEGATIVE_VINA_RESOLVED but per-pocket delta constant −4.5 because F2(a) is SMILES-based (path to fully-MEASURED requires GPU bench on `cfm_vina_dock_panel.py` with pocket-derived distances; BLOCKED per Wf-299B + AGENTS.md §5.4)
- m15 PoseBusters validity SKIPPED graceful (preserved as honest-negative on validity NOT on accuracy)
- m1 Ru AUC 0.7613 (1-seed 30-ep) MEASURED_PARTIAL_LIFT (Krasnov 0.86 gate NOT YET HIT, gap −0.099; multi-seed validation NEEDED via TODO-181 G5 USER_OK-gated wrapper per Wf-355-B1 + Wf-359-A2)

### §5.3 MEASURED-vs-PROJECTED discipline

Every numeric value in this synthesis traces to a JSON / report path on disk:

- 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL → `molmetal/reports/wf362b1_preflight.txt` (Wf-362-B1 timestamp)
- 19/19 NEW tests PASS → combined `pytest` output above
- 3 metric JSONs refreshed → `metrics/by_metric/metalcytotox_combined_r2.json` + `metrics/by_metric/vina_kcal_per_mol.json` + `metrics/by_metric/docked_pose_rmsd.json`
- Per-agent wall ~17.5 min → sum of A1+A2+A3+B1 wall per Wf-362 closeout files
- R² +0.2227 + lift +0.1172 + verdict CEILING_BREAK → `molmetal/reports/wf335m2_r2/result.json`
- V3 cohort median −7.782 kcal/mol + 5-pocket per-pocket values → `molmetal/reports/wf333v3_vina_p2p3/smoke_per_pocket.json`
- m15 median 0.6578 Å + 5/5 PASS + difficulty counts → `TODO/completed/341t1_m15_rmsd.md` §0

Zero tolerance for fabricated / aspirational numbers in main results.

### §5.4 Wall + cost accounting

| Item | Value |
|---|---|
| MEASURED wall (Wf-362 wave) | ~17.5 min CPU-only (A1 ~2 + A2 ~5 + A3 ~0.5 + B1 ~10) |
| PROJECTED wall (task brief) | 25-45 min CPU-only |
| MEASURED vs PROJECTED | ~50% under PROJECTED |
| GPU launches | 0 (MEASURED) |
| `paper/*` touches | 0 (MEASURED) |
| Net docs change | +96 lines INDEX.md + +1 line AGENTS.md |

---

**END OF Wf-362 FINAL SYNTHESIS** — 4/4 deliverables SHIPPED (3 JSON refresh closeouts + 1 docs-only INDEX + AGENTS + final synthesis); 19/19 NEW tests PASS (CPU-only); 0 GPU launches; 0 `paper/*` touches; pre-flight v3 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-353-B1 baseline; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED.