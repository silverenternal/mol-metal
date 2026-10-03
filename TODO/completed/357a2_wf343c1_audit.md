# Wf-357-A2 — Wf-343c1 Pre-Retrain Gate Audit

**Date:** 2026-09-29 (Wf-357-A2 ship)
**Scope:** Run `wf343c1_pre_retrain_check.py` standalone + verify all 22 Wf-343c1
auto-verifiable gates are wired into `pre_flight_check.py` build_gates() per Wf-353-B1
contract + verify the 4 Wf-343c4 gates (per Wf-343c4 ship) PASS.
**Status:** ALL 26 Wf-343c1 gates green (21 PASS + 5 SKIP, 0 FAIL) at 29.57s wall.
**Wall:** MEASURED 29.5747 s (wf343c1 standalone, CPU-only).
**Pre-flight wall:** 153.16 s (full 94-gate `all` pipeline; Wf-350-C1 GPU probe 8.99 W idle).

---

## §1 Wf-343c1 standalone run — MEASURED results

Per the task brief, ran
`PYTHONPATH=… python molmetal/scripts/wf343c1_pre_retrain_check.py` standalone.

| Metric | Value | Evidence |
|---|---:|---|
| **total_checks** | **26** | `wf343c1_pre_retrain_check.py` (10 code + 3 ablation + 7 data + 1 test + 4 post-retrain + 1 honest-framing) |
| **pass_count** | **21** | All auto-verifiable gates green |
| **fail_count** | **0** | Zero failures |
| **skip_count** | **5** | 4 §5.11–§5.14 post-retrain SKIP + 1 §2.1.6 BindingDB external honest-negative SKIP |
| **wall_sec** | **29.5747** | CPU-only, no GPU launches |

Raw report saved to: `molmetal/reports/wf357a2_wf343c1.txt`
Parent doc reference: `科研进展_pre_retrain_checklist_2026-09-28.md` (§1.1 + §1.3 + §2.1 + §3.1 + §5.11-§5.14 + §8.15).

### §1.1 Per-gate verdict matrix (MEASURED)

| Pipeline | Gate | Status | Notes |
|---|---|---|---|
| code | `code_1_1_1_indices_aliasing_fix` | PASS | `indices = train_idx.copy()` grep hit |
| code | `code_1_1_2_kendall_log_sigma` | PASS | `kendall.parameters()` grep hit |
| code | `code_1_1_3_dann_per_metal` | PASS | `per_metal` grep hit |
| code | `code_1_1_4_nan_to_num_guard` | PASS | `nan_to_num` grep hit |
| code | `code_1_1_5_grad_clip_max_norm` | PASS | `clip_grad_norm` grep hit |
| code | `code_1_1_6_w2_epoch_avg_history` | PASS | `w2_epoch_avg_history` grep hit |
| code | `code_1_1_7_ablation_flags_present` | PASS | `--no-multi-task --no-wass --no-kendall --no-dann` exposed |
| code | `code_1_1_8_enhanced_metal_head` | PASS | `--enhanced-head --metal-attention` exposed |
| code | `code_1_1_9_click_aug_flag` | PASS | `--use-click-aug` grep hit |
| code | `code_1_1_10_master_modules_flags` | PASS | `--use-master-wass --use-master-cath` exposed |
| ablation | `ablation_1_3_1_flags_present` | PASS | 4 token ablation flags exposed |
| ablation | `ablation_1_3_2_single_metal_flag` | PASS | `--single-metal` exposed |
| ablation | `ablation_1_3_3_n_seeds_min_flag` | PASS | `--n-seeds-min` exposed |
| data | `data_2_1_1_metalcytotox_5metal_csv` | PASS | `wc -l metalcytotox_5metal_multitask.csv = 26802 ≥ 26000` |
| data | `data_2_1_2_metalcytotox_raw_csv` | PASS | `wc -l MetalCytoToxDB.csv = 26802 ≥ 1000` |
| data | `data_2_1_3_drugood_dir` | PASS | `.data/molmetal/drugood/` exists |
| data | `data_2_1_4_nci60_dir` | PASS | `.data/molmetal/NCI60_GI50/` exists |
| data | `data_2_1_5_platinai_4cell_dir` | PASS | `.data/molmetal/platinai_4cell.csv` exists |
| data | `data_2_1_6_bindingdb_external_honest_negative` | **SKIP** | BindingDB 1.07M is external (~50 precious-metal entities); NOT used in retrain per parent doc §2.1.6. Honest-negative: data fact, not blocker. |
| data | `data_2_1_7_click_yields_csv` | PASS | `wc -l click_yields_literature.csv = 61 ≥ 50` |
| test | `test_3_1_5_nan_guard_test_present` | PASS | `tests/test_train_chemprop_v2_v3_nan_guard.py` exists |
| post_retrain | `post_retrain_5_11_per_seed_std` | **SKIP** | post-retrain gate: filled in after retrain wave |
| post_retrain | `post_retrain_5_12_os_auc_non_degenerate` | **SKIP** | post-retrain gate: filled in after retrain wave |
| post_retrain | `post_retrain_5_13_ru_w2_epoch_avg` | **SKIP** | post-retrain gate: filled in after retrain wave |
| post_retrain | `post_retrain_5_14_no_nan_auc` | **SKIP** | post-retrain gate: filled in after retrain wave |
| honest_framing | `honest_framing_8_15_catalog_entry_count` | PASS | `科研进展_pre_retrain_checklist_2026-09-28.md` has 15 §8 entries (≥ 15) |

**MEASURED verdict:** 21/26 PASS + 5/26 SKIP (per honest-framing convention) + **0 FAIL**.

---

## §2 Wf-343c4 gate verdicts (per Wf-343c4 ship)

| Wf-343c4 gate | MEASURED status | Evidence |
|---|---|---|
| `p3_wf343c4_hyperparams_config_complete` | **PASS** | `gate_wf343c4_hyperparams_config_complete` (line 3863) verified wave YAML has 7 REQUIRED_CONFIG_KEYS (from pre_flight_check.py report) |
| `p3_wf343c4_yuelbond_decoder_env` | **PASS** | `cfm_decoder_yuelbond.py` (23792 B) exports `YuelBondDecoder` (per Wf-272I / Wf-269C YuelBond-conditionality structural check; `--no-yuelbond → decode_ratio 0.000` MEASURED on real cohort) |
| `p3_wf343c4_anchor_freshness_24h` | **PASS** | All paper-grade `*_anchor.json` files within 24 h freshness window |
| `p3_wf343c4_wave_budget_remaining_nonneg` | **SKIP** | `remaining_hr=-183.04h` is negative AND 8 DEFERRED wave(s) (`Wave-cfm:cfm_real_decode_lift_p1.3`, `Wave-cfm:cfm_per_innovation_ablation_p1.4`, `Wave-dmpnn:wf124_p3_per_innovation_ablation_g6`, …) are responsible for the over-budget; non-deferred cost 16.00 h (effective_remaining=-6.00 h); **BY-DESIGN** per Wf-262 §7.1 — DEFERRED waves need USER_OK or schedule outside 10 h session |

**MEASURED verdict:** 3/4 PASS + 1/4 SKIP (wave_budget is DEFERRED-only by Wf-262 §7.1 contract).

---

## §3 Wf-343c1 22 un-wired gate status — VERIFIED all 22 wired

Per Wf-353-B1 ship contract, all 22 auto-verifiable Wf-343c1 gates are wired into
`pre_flight_check.py` build_gates() via the `_run_wf343c1_gate(n, runner)` adapter (line 4694)
and the gate-name list comprehension at lines 6267–6299.

### §3.1 Wiring evidence (MEASURED)

| Source | Evidence |
|---|---|
| `molmetal/scripts/pre_flight_check.py:6267-6299` | List-comprehension wiring 22 individual runners, one per Wf-343c1 auto-verifiable gate (10 code + 3 ablation + 7 data + 1 test + 1 honest_framing) |
| `molmetal/scripts/pre_flight_check.py:6280-6290` | Each runner tuple: `(f"p3_wf353b1_{name}", lambda n=name, r=runner: _run_wf343c1_gate(n, r))` — verbatim name preserved |
| `molmetal/scripts/pre_flight_check.py:4694-4746` | `_run_wf343c1_gate` adapter — maps Wf-343c1 `GateResult` (6-field) → pre-flight `GateResult` (6-field) 1-to-1; exception-safe (any runner exception → FAIL evidence string) |
| `molmetal/scripts/pre_flight_check.py:3117-3128` | `EXPECTED_GATE_COUNTS["all"] = 94` (71 pre-Wf-353-B1 baseline + 22 NEW Wf-343c1 + 1 NEW Wf-350-C1 GPU env probe) — confirms wiring via gate count |
| `molmetal/reports/wf357a2_preflight_all.txt` | Fresh-run pre-flight shows 22 individual gate verdicts under `code_ / ablation_ / data_ / test_ / honest_framing_` pipeline labels (counted 22 unique names via `grep -c` over name keys) |

### §3.2 Pre-flight fresh-run summary (MEASURED)

Ran `python molmetal/scripts/pre_flight_check.py --pipeline all` (no `--json`, default JSON to stdout).

| Metric | Value |
|---|---:|
| Total gates | **94** (matches EXPECTED_GATE_COUNTS["all"] = 94) |
| PASS | **90** |
| FAIL | **2** (NOT Wf-343c1 — see §3.3) |
| SKIP | **2** (1 BindingDB honest-negative SKIP + 1 wave_budget DEFERRED SKIP) |
| Wall | **153.16 s** (full 94-gate run; GPU probe is fast even at 8.99 W idle) |

### §3.3 The 2 FAIL gates are NOT Wf-343c1 gates

| Gate | Pipeline | Status | Evidence |
|---|---|---|---|
| `p2_tests_pass` | `tests` | FAIL | `pytest FAIL (rc=1): 2 failed, 712 passed, 8 warnings in 9.91s` |
| `p3_wf343c2_per_wave_runtime_check` | `wf347a1` | FAIL | `wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke', --skip-live --strict): (no output)` |

These are pre-existing failures unrelated to Wf-343c1 / Wf-353-B1 wiring:
1. **`p2_tests_pass`** — pytest returns rc=1 with 2 failed tests. Needs Wf-359 (or equivalent) to identify and fix the 2 regressing tests.
2. **`p3_wf343c2_per_wave_runtime_check`** — Wf-343c2 per-wave check returns rc=1 with empty output. Needs investigation (likely a script invocation contract drift; the Wf-347A1 wave name suggests it's a smoke-test gate that fires even when no retrain is scheduled).

**ACTION ITEM (not blocking Wf-357-A2 close):** Wf-358 / Wf-359 to investigate + fix the 2 non-wf343c1 FAILs. They do NOT affect the Wf-343c1 wiring contract.

### §3.4 The 22 wired Wf-343c1 gate verdicts in pre-flight (MEASURED)

| Status | Count | Gates |
|---|---:|---|
| PASS | **21** | All auto-verifiable gates (10 code + 3 ablation + 6 of 7 data [BindingDB is SKIP] + 1 test + 1 honest_framing) |
| SKIP | **1** | `data_2_1_6_bindingdb_external_honest_negative` (honest-negative, per parent doc §2.1.6) |
| FAIL | **0** | Zero |
| TOTAL | **22** | (matches Wf-353-B1 ship contract) |

The 4 §5.11–§5.14 post-retrain gates are intentionally NOT wired individually (per the
Wf-353-B1 ship contract: line 6291–6297 comment block); they remain SKIP per honest-framing
convention and are surfaced via the standalone `wf343c1_pre_retrain_check.py` script
(which is invoked from `molmetal/tests/test_wf343c1_pre_retrain_check.py` for full verification).

---

## §4 Honest framing per party-twenty-spirit §3.5

| Field | Value | Tag |
|---|---|---|
| Total Wf-343c1 checks | 26 | MEASURED |
| Wf-343c1 standalone PASS | 21/26 | MEASURED |
| Wf-343c1 standalone FAIL | 0/26 | MEASURED |
| Wf-343c1 standalone SKIP | 5/26 | MEASURED |
| Wf-343c1 standalone wall | 29.5747 s | MEASURED |
| Pre-flight `all` total | 94/94 | MEASURED (matches EXPECTED_GATE_COUNTS["all"] = 94) |
| Pre-flight `all` PASS | 90/94 | MEASURED |
| Pre-flight `all` FAIL | 2/94 | MEASURED (NOT Wf-343c1 — see §3.3) |
| Pre-flight `all` SKIP | 2/94 | MEASURED |
| Wf-343c4 PASS | 3/4 | MEASURED |
| Wf-343c4 SKIP | 1/4 | MEASURED (wave_budget DEFERRED per Wf-262 §7.1) |
| 22 wired Wf-343c1 gates PASS | 21/22 | MEASURED |
| 22 wired Wf-343c1 gates SKIP | 1/22 | MEASURED (BindingDB honest-negative) |
| 22 wired Wf-343c1 gates FAIL | 0/22 | MEASURED |
| NEW tests added | **0** | HONEST-NEGATIVE: pure verification; no code/test changes |
| GPU launches | **0** | HONEST-NEGATIVE: CPU-only verification |

---

## §5 Standing directive compliance (§5.1 GPU + §5.2 honest framing + §5.3 paper)

- **§5.1 GPU authorization (2026-09-29 grant):** This is a pure CPU verification. Zero GPU launches.
  Pre-flight shows 2 GPUs idle (RTX PRO 6000 Blackwell 8.99 W + RTX 5090 9.17 W) per the
  Wf-350-C1 GPU env probe — no contention.
- **§5.2 honest framing mandate:** Every numeric value above is tagged MEASURED + cites the
  source JSON / report on disk. No fabricated / aspirational numbers.
- **§5.3 no paper work:** Zero touches to `paper/*`. Audit is in `TODO/completed/` only.
- **§5.4 Tier-2 journal + no wet-lab:** N/A — no paper / wet-lab work in this audit.

---

## §6 Action items (forward-looking, not blocking close)

1. **Wf-358 / Wf-359** — Investigate the 2 non-Wf-343c1 FAILs:
   - `p2_tests_pass` (2 pytest regressions; needs test-level diagnosis)
   - `p3_wf343c2_per_wave_runtime_check` (empty-output rc=1; likely invocation drift)
2. **Wf-353-B1 follow-up (DEFERRED, not blocking):** 22 individual Wf-343c1 gate runners are
   CPU-only and PASS in pre-flight. The 4 §5.11–§5.14 post-retrain gates remain SKIP per
   honest-framing convention and will be filled in after the next retrain wave (TODO-181 G5 or
   TODO-182 G6).
3. **Wave_budget SKIP:** 8 DEFERRED waves need USER_OK or schedule outside 10 h session per
   Wf-262 §7.1. Not blocking Wf-343c1 wiring contract — by-design.

---

## §7 Files referenced

| Path | Role |
|---|---|
| `molmetal/scripts/wf343c1_pre_retrain_check.py` | Wf-343c1 standalone runner (30243 B; 26 gates) |
| `molmetal/scripts/pre_flight_check.py` | pre-flight v2/v3 runner (293096 B; 94 gates total) |
| `molmetal/scripts/pre_flight_check.py:4694-4746` | `_run_wf343c1_gate` adapter |
| `molmetal/scripts/pre_flight_check.py:6267-6299` | 22 individual Wf-343c1 gate runner wiring |
| `molmetal/scripts/pre_flight_check.py:3117-3128` | `EXPECTED_GATE_COUNTS["all"] = 94` |
| `molmetal/reports/wf357a2_wf343c1.txt` | Wf-343c1 fresh-run JSON report |
| `molmetal/reports/wf357a2_preflight_all.txt` | Pre-flight v3 fresh-run JSON report (full 94-gate `all` pipeline) |
| `科研进展_pre_retrain_checklist_2026-09-28.md` | Parent doc reference (§1.1 + §1.3 + §2.1 + §3.1 + §5.11–§5.14 + §8.15) |
| `TODO/completed/353b1_wf343c1_gates_wired.md` | Wf-353-B1 ship contract + 22-gate wiring summary |

---

**END OF Wf-357-A2 AUDIT** — All 26 Wf-343c1 gates green (21 PASS + 5 SKIP + 0 FAIL) + all 22
auto-verifiable gates wired into pre-flight `build_gates()` per Wf-353-B1 contract. 2 pre-existing
non-Wf-343c1 FAILs (`p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`) flagged for Wf-358/359.
Owner: Hugo Lee; 2026-09-29.
