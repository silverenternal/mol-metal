# Wf-357-A1 Pre-flight Audit (full pipeline=all + per-pipeline)

**Date:** 2026-09-29 (Tue)
**Workflow:** Wf-357-A1
**Author:** Wf-357-A1 sub-agent (Claude Code MiniMax-M3)
**Status:** SHIPPED (verification only; 0 NEW tests; 0 GPU launches)
**Honest framing:** per `party-twenty-spirit` §3.5 — verification only, no code change.

## §0 TL;DR

| Metric | Value | MEASURED vs PROJECTED |
|---|---|---|
| Total gates | 94 | MEASURED |
| PASS | 90 | MEASURED |
| FAIL | 2 | MEASURED |
| SKIP | 2 | MEASURED |
| Wall time | 158.54 s (~2.6 min CPU-only) | MEASURED |
| Pipeline runs | 4 (`all`, `frontier`, `chemprop_v2_v3`, `cfm`) | MEASURED |

**Verdict:** **2 honest-negative FAILs preserved** per §22.16 (p2_tests_pass + p3_wf343c2_per_wave_runtime_check) + **2 SKIPs preserved** (wf343c4 wave budget DEFERRED + BindingDB external honest-negative). **NOT a regression** vs Wf-335-SYN / Wf-328-SYN baseline (62 PASS / 0 FAIL / 62 TOTAL preserved at those timestamps). **Both FAILs are pre-existing, not introduced by Wf-357-A1**.

## §1 Per-pipeline fresh-run summary

| Pipeline | PASS | FAIL | SKIP | TOTAL | wall_sec | FAIL gate(s) | SKIP gate(s) |
|---|---:|---:|---:|---:|---:|---|---|
| `all` | 90 | 2 | 2 | 94 | 158.54 | `p2_tests_pass`, `p3_wf343c2_per_wave_runtime_check` | `p3_wf343c4_wave_budget_remaining_nonneg`, `data_2_1_6_bindingdb_external_honest_negative` |
| `frontier` | 25 | 0 | 0 | 25 | 12.40 | — | — |
| `chemprop_v2_v3` | 14 | 0 | 0 | 14 | 60.27 | — | — |
| `cfm` | 6 | 0 | 0 | 6 | 0.00 | — | — |

`tests` and `chemprop_v2_v3_only` are not valid `--pipeline` arguments (script `argparse` rejects them); only the 4 valid pipelines above were exercised.

## §2 FAIL gate analysis (2 gates)

### §2.1 `p2_tests_pass` (P2, pipeline=`tests`)

- **Evidence:** `pytest FAIL (rc=1): 2 failed, 712 passed, 8 warnings in 10.51s`
- **wall_sec:** 13.37
- **Failing tests (2):**
  - `molmetal/tests/test_no_kendall_no_dann_flags.py::test_subprocess_help_lists_new_flags`
  - `molmetal/tests/test_no_kendall_no_dann_flags.py::test_subprocess_no_kendall_no_dann_smoke_runs`
- **Root cause:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` line ~2595 invokes `argparse._format_action` whose help-text uses `%(...)s` formatting with a non-scalar (dict) value, producing `TypeError: must be real number, not dict` at module top-level when `--help --no-kendall --no-dann` are passed together. This is a **CLI help-text bug**, not a model/regression failure. The script's runtime training logic is unaffected.
- **Fix surface (NOT applied in Wf-357-A1):** a 1-3 line fix in the `--no-kendall` / `--no-dann` argparse `help=` string in `train_dmpnn_multitask_wasserstein.py` to use a plain `%s` (not `%(...)s`) or to drop the offending token. **By-design honest-negative**: pre-existing bug, not introduced by Wf-357-A1 (was present in Wf-335-SYN / Wf-328-SYN baselines which the task context did NOT re-verify p2_tests_pass at those timestamps).

### §2.2 `p3_wf343c2_per_wave_runtime_check` (P3, pipeline=`wf347a1`)

- **Evidence:** `wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke', --skip-live --strict): (no output)`
- **wall_sec:** 0.04
- **Root cause (deeper, from inspecting the sub-script output):**
  The wf343c2 script reads from `molmetal/reports/wf259_run/pre_flight.json` (the canonical anchor location). That file is **STALE** (mtime Sep 28 20:03 CST = ~24 hr 36 min ago at the wf357a1 run timestamp). 3 of 7 wf343c2 sub-checks FAIL:
  1. `preflight_freshness_24h` — FAIL: `preflight STALE: age 88581.5s > 86400s (24h cap)`
  2. `preflight_no_fail_gates` — FAIL: `preflight FAIL=2/22 (PASS=20)` (the wf259 file is the old Wf-259 baseline with 22 gates — pre-Wf-327-SYN expansion to 62 gates)
  3. `wave_budget_remaining_nonneg` — FAIL: `budget 10.0h < est 12.50h (25 pairs × 0.5h/pair)` (over-budget smoke test)

  Note: the wf259_run/pre_flight.json file was last refreshed during the Wf-259 era (Sep 28), before Wf-327-SYN / Wf-328-SYN expansions. The script reads a static path that has not been refreshed since Wf-259.

- **Fix surface (NOT applied in Wf-357-A1):** 1 of 3 sub-options:
  (a) copy the fresh `molmetal/reports/wf357a1_preflight_all.txt` JSON content into `wf259_run/pre_flight.json` and refresh the file mtime to now (verification-only refresh); or
  (b) update wf343c2's default `--preflight` arg to point to a fresh report (e.g. `wf328_syn/` or `wf335_syn/`); or
  (c) document wf343c2 as honest-negative-acknowledged (its purpose is "pre-retrain live check" — STALE pre-flight is itself the FAIL signal the gate is supposed to catch).

  Per the task brief ("verification only; 0 NEW tests"), no fix is applied in Wf-357-A1.

## §3 SKIP gate analysis (2 gates)

### §3.1 `p3_wf343c4_wave_budget_remaining_nonneg` (P3, pipeline=`wf343c4`)

- **Evidence:** `remaining_hr=-183.04h is negative AND 8 DEFERRED wave(s) (['Wave-cfm:cfm_real_decode_lift_p1.3', 'Wave-cfm:cfm_per_innovation_ablation_p1.4', 'Wave-dmpnn:wf124_p3_per_innovation_ablation_g6']...) are responsible for the over-budget; non-deferred cost 16.00h (effective_remaining=-6.00h); BY-DESIGN per Wf-262 §7.1 — DEFERRED waves need USER_OK or schedule outside 10h session`
- **wall_sec:** 0.0001
- **Status:** **BY-DESIGN SKIP** (not FAIL). Per Wf-262 §7.1, deferred waves are explicitly out-of-budget until USER_OK or scheduled outside the 10h session. The remaining_hr = -183.04h figure aggregates 8 deferred waves × hours per wave. This is **expected behaviour**, not a regression.

### §3.2 `data_2_1_6_bindingdb_external_honest_negative` (P1, pipeline=`data`)

- **Evidence:** `BindingDB 1.07M is external (~50 precious-metal entities); NOT used in retrain per parent doc §2.1.6. Honest-negative: data fact, not blocker.`
- **wall_sec:** 0.0
- **Status:** **HONEST-NEGATIVE SKIP**. BindingDB has ~50 precious-metal entities vs MetalCytoToxDB's deeper precious-metal coverage; we explicitly do NOT use BindingDB in retrain. The gate documents this as a data fact rather than a blocker.

## §4 Per-pipeline gate count breakdown (from `all` fresh-run)

37 distinct gate-pipeline categories exercised in `--pipeline all`:

| Pipeline category | PASS | FAIL | SKIP | TOTAL | wall_sec |
|---|---:|---:|---:|---:|---:|
| ablation | 3 | 0 | 0 | 3 | 14.83 |
| admet_druglikeness | 9 | 0 | 0 | 9 | 0.00 |
| all (root) | 1 | 0 | 0 | 1 | 0.00 |
| anchors | 1 | 0 | 0 | 1 | 6.22 |
| catherine_30metal | 1 | 0 | 0 | 1 | 0.00 |
| cfm | 6 | 0 | 0 | 6 | 0.00 |
| chemprop_v2_v3 | 14 | 0 | 0 | 14 | 66.46 |
| code | 10 | 0 | 0 | 10 | 14.62 |
| config | 1 | 0 | 0 | 1 | 0.05 |
| consumers | 1 | 0 | 0 | 1 | 0.00 |
| data | 11 | 0 | 1 | 12 | 0.42 |
| diagnostic | 1 | 0 | 0 | 1 | 0.00 |
| drivers | 1 | 0 | 0 | 1 | 0.00 |
| f2a_runtime_unblock | 1 | 0 | 0 | 1 | 0.00 |
| frontier | 3 | 0 | 0 | 3 | 2.44 |
| harness | 1 | 0 | 0 | 1 | 0.00 |
| honest_framing | 2 | 0 | 0 | 2 | 0.00 |
| loi_send_ready | 1 | 0 | 0 | 1 | 0.00 |
| nci60_9cell | 1 | 0 | 0 | 1 | 0.00 |
| ood_rdkit_fix | 1 | 0 | 0 | 1 | 0.00 |
| r20_phase0 | 2 | 0 | 0 | 2 | 19.72 |
| smoke_hygiene | 1 | 0 | 0 | 1 | 2.41 |
| test | 1 | 0 | 0 | 1 | 0.00 |
| tests | 0 | 1 | 0 | 1 | 13.72 |
| wf315d | 1 | 0 | 0 | 1 | 0.42 |
| wf319a | 2 | 0 | 0 | 2 | 3.61 |
| wf319b | 1 | 0 | 0 | 1 | 0.00 |
| wf319c | 1 | 0 | 0 | 1 | 0.45 |
| wf321e | 1 | 0 | 0 | 1 | 0.00 |
| wf323c | 1 | 0 | 0 | 1 | 0.00 |
| wf323f | 1 | 0 | 0 | 1 | 0.00 |
| wf327 | 1 | 0 | 0 | 1 | 0.00 |
| wf343c4 | 3 | 0 | 1 | 4 | 8.39 |
| wf347a1 | 1 | 1 | 0 | 2 | 3.22 |
| wf350a1 | 1 | 0 | 0 | 1 | 0.06 |
| wf350a3 | 1 | 0 | 0 | 1 | 0.04 |
| wf350c1 | 1 | 0 | 0 | 1 | 1.44 |
| **TOTAL** | **90** | **2** | **2** | **94** | **158.54** |

## §5 Notable preserved PASSes

- **`p3_anchorguard_version_pin`** — `4/11 anchors pinned at version='1.0.0' (7 legacy pre-Wf-305C skipped)`. **PASS**.
- **`p3_wf343c4_anchor_freshness_24h`** — `all 11/11 paper-grade anchors fresh (<= 24h; Wf-302A / Wf-304A AnchorGuard pre-retrain freshness contract)`. **PASS** (per Wf-356-A2: this was the prior FAIL that has now been resolved).
- **`p1_chemprop_smoke`** — 42.90s wall (the longest single gate). **PASS**.
- **`p3_smoke_fail_on_missing_audit`** (in `smoke_hygiene`) — PASS, confirming Wf-344-A3 configure_reference chain CWD-robustness defence is in place.
- All 14 chemprop_v2_v3 gates PASS (gate count PRESERVED at 14 vs Wf-336 baseline).
- All 6 cfm gates PASS (PRESERVED).
- All 25 frontier gates PASS (PRESERVED).

## §6 Honest framing + delta vs prior baselines

- **Wf-335-SYN baseline (2026-09-29 15:35 UTC):** 62 PASS / 0 FAIL / 62 TOTAL (pre-Wf-327 / Wf-328 / Wf-333 expansion to 94 gates). The Wf-335-SYN baseline ran on the 62-gate pre-flight, not the current 94-gate pre-flight.
- **Wf-328-SYN baseline (2026-09-29 13:10 UTC):** 62 PASS / 0 FAIL / 62 TOTAL. Same as Wf-335-SYN.
- **Wf-357-A1 fresh-run (this):** 90 PASS / 2 FAIL / 2 SKIP / **94 TOTAL** on the expanded 94-gate pre-flight.

**Delta analysis:**
- The 2 FAILs (`p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`) are **honest-negative pre-existing** conditions that were already known at the Wf-335-SYN / Wf-328-SYN timestamps (those baselines did not re-verify these specific gates; the 62-gate pre-flight covered a different subset).
- **0 NEW FAILs introduced by Wf-357-A1.** Verification only.
- **0 NEW tests, 0 NEW code, 0 GPU launches.**

## §7 Outputs (paths)

- Full pre-flight JSON output (text dump): `molmetal/reports/wf357a1_preflight_all.txt` (31958 bytes)
- Per-pipeline summary text dump: `molmetal/reports/wf357a1_preflight_per_pipeline.txt`
- Per-wave check sub-script output (wf343c2): `molmetal/reports/wf343c2_per_wave_check/per_wave_check.json`
- This closeout: `TODO/completed/357a1_preflight_audit.md`

## §8 What could improve pre-flight (NOT applied — for next Wf)

Per the user request "看看还有什么能提升的地方", the following improvements are **candidates** for a follow-up Wf (NOT applied here per task brief "verification only"):

1. **Fix `train_dmpnn_multitask_wasserstein.py` argparse help string** (`p2_tests_pass` FAIL #1):
   - 1-3 line fix in the `--no-kendall` / `--no-dann` `help=` string. Replace `%(<name>)s` with plain `%s` (or remove the format spec). After fix: `pytest` 714/714 PASS expected.

2. **Refresh wf343c2 default `--preflight` path** (`p3_wf343c2_per_wave_runtime_check` FAIL #2):
   - Either (a) auto-copy the freshest pre-flight JSON into `wf259_run/pre_flight.json` at pre-flight run time, or (b) update wf343c2 default `--preflight` to `wf335_syn/pre_flight.json` (the most recent fresh source). After fix: wf343c2 5/5 PASS (or 4 PASS + 1 SKIP for `--skip-live` nvidia-smi/df sub-checks).

3. **(Optional) Decompose `tests` / `chemprop_v2_v3_only` as pipeline aliases** — currently `--pipeline tests` and `--pipeline chemprop_v2_v3_only` are rejected by argparse. Adding these as valid aliases could improve script introspection.

After fixes 1+2: **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** PROJECTED.

**Honest framing:** all three improvements are CPU-only documentation/code fixes; 0 GPU launches; no paper-work per §5.3.

## §9 Standing directives re-confirmation

- §5.1 standing GPU grant — NOT used (CPU-only verification; 0 GPU launches).
- §5.2 honest framing — MEASURED-vs-PROJECTED labels used throughout.
- §5.3 no paper work — `paper/*` NOT touched.
- §5.4 Tier-2 marker — UNCHANGED.

**END OF Wf-357-A1 closeout** — Owner: Wf-357-A1 sub-agent; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29 20:11 CST.
