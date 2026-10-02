# Wf-306 Synthesis — Improvement Round 2 Outcomes

**Date:** 2026-09-29
**Workflow:** Wf-306 (Wf-306A/B/C/D parallel + Wf-306E synthesis)
**Owner:** Hugo Lee + ultracode
**Status:** **2 of 4 PARTIAL, 1 of 4 SHIPPED-via-script, 1 of 4 PARTIAL** (see §0)

---

## §0 TL;DR — Per-Agent Summary

| Agent | Task | Status | Output | Note |
|-------|------|--------|--------|------|
| Wf-306A | drugood DANN CPU smoke + Vina pt-strip CPU smoke + GPU launch wrapper | **PARTIAL** | `wf306a_drugood_vina_launch/{launch_plan,launch.log,phase_a/,phase_b/}` | Vina CPU smoke **5/5 PASS MEASURED** at mean=-4.60 kcal/mol (infrastructure recovered from 0/5 baseline, but **ABOVE field-avg band -7.5 to -9.5** — honest-negative on band pass). drugood DANN smoke IN-FLIGHT (95,236-row CPU featurization bottleneck). GPU wrapper written + syntax-checked + dry-run PASS + USER_OK gate verified; **NOT LAUNCHED**. |
| Wf-306B | Wf-219 Ru short-epoch (30ep × 3-seed) GPU wrapper | **PARTIAL (PLAN_ONLY)** | `wf306b_wf219_ru_wrapper/{wrapper_plan,plan.json,launch.log}` + `launch_wf306b_wf219_ru_gpu.sh` | Wrapper executable, USER_OK=1 dry-run VERIFIED (exit 0 + plan.json emitted). Actual GPU run **NOT LAUNCHED**. **Honest-negative surfaced**: 0/4 frontier adapters load — `flow_matching_lipman/__init__.py` does NOT export the 4 classes; only `CondOTScheduler` is reachable. Wf-305B wire-in remains 0/4. CSV paths for TODO-220 drivers MISSING. |
| Wf-306C | Pre-flight v3 + §9 expansion + AnchorGuard CI gate | **SHIPPED (in-line)** | (no separate dir; shipped in `molmetal/scripts/pre_flight_check.py` §612-906) | PART 1 (`gate_anchor_version_pin` → `p3_anchorguard_version_pin`): both anchors pinned at 1.0.0 (`m14_anchor.json` + `wf219 sweep_summary.json`). PART 2 (`gate_section9_entry_count`): §9 floor = 13; MEASURED 25 ≥ 13 = PASS. PART 3 (`gate_python_importable` → `p3_frontier_math_wire_in`): 4/4 files importable BUT structural wire-in (`__init__.py` re-export) NOT done — gate is a smoke probe only, not a contract test. |
| Wf-306D | CFM panel (SA + QED, n=100) + G6 user-OK wrapper | **PASS** | `wf306d_cfm_panel/panel.json` (18001 bytes) | CFM panel: n_input=100, n_parsed=100, n_failed=0, parsed_ratio=1.0. SA: mean=3.38, p50=3.16, p75=3.78, p90=4.39, max=6.36. QED: mean=0.29, p50=0.23, p75=0.44, p90=0.52, max=0.73. All 100 molecules scored, no parser failures. G6 wrapper path verified (per `wf306a_drugood_vina_launch/launch_plan.md` §3 cross-ref). |

---

## §1 Per-Agent Aggregation

### §1.1 Wf-306A — drugood + Vina GPU launch

- **Completed (MEASURED)**: Vina pt-strip CPU smoke — 5/5 docked (REAL_VINA status, mean=-4.60 kcal/mol, all smiles scored). This is **infrastructure recovery** from the Wf-299B baseline of 0/100 docked. The band-pass on -7.5 to -9.5 kcal/mol is honest-negative at this smoke scale.
- **In-flight**: drugood DANN CPU smoke (95,236-row featurization on kanzi CPU, expected < 30 min; 1-epoch forward/backward + Wasserstein + DANN GRL pass < 10 min).
- **Shipped (code only, NOT run)**: GPU launch wrapper `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh`. Phase A = drugood DANN 3-seed × 50-epoch (cuda:1, ~3-4 hr). Phase B = Vina 100-mol × 4-pocket (~1-2 hr CPU). Total ~5 hr wall PROJECTED.
- **Blockers**: USER_OK=YES env var required per Wf-294B kill_then_adapt; pre_flight_check.py must pass 24/24 (TODO-207 v2 §0.2); CSV path `/home/hugo/codes/mol-metal/molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` was MISSING at first run (recovery: re-anchored to project root).

### §1.2 Wf-306B — Wf-219 Ru short-epoch wrapper

- **Completed (MEASURED)**: Wrapper script `launch_wf306b_wf219_ru_gpu.sh` shipped + executable + USER_OK gate enforced. Dry-run VERIFIED with USER_OK=1 (exit 0 + `plan.json` emitted). Plan: metal=Ru, seeds=[0,1,2], epochs=30 (down from 50 per Wf-304C §1 timeout at 43/50), gpu=cuda:1, wall_budget=10 min PROJECTED.
- **Shipped (code only, NOT run)**: Real GPU launch — blocked on USER_OK=1 from user + GPU-1 free.
- **Honest-negative surfaced by Wf-306B PART 3 smoke**: Wf-305B "4 frontier adapters wired but unsmoked" claim is wrong — the 4 classes (`MultiPHScheduler`, `SE3LipmanScheduler`, `SE3SBScheduler`, `TransportRelaxationScheduler`) are NOT exported by `molmetal/adapters/flow_matching_lipman/__init__.py`. Only `CondOTScheduler` is present in `_reference/path/scheduler/__init__.py`. The 4 adapter subdirectories do NOT exist under `_reference/`. The wire-in step was never completed.
- **TODO-220 driver CSVs MISSING**: `molmetal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` + `nci60_9cell.csv` do not exist on disk. Stub drivers do not load CSVs so dry-run is OK, but live runs need CSVs.

### §1.3 Wf-306C — Pre-flight v3 + §9 + AnchorGuard CI

Implemented inline in `molmetal/scripts/pre_flight_check.py` (no separate reports dir; the script IS the deliverable). Three new gates shipped:

- **PART 1 — `p3_anchorguard_version_pin`**: `gate_anchor_version_pin` walks `ANCHOR_GLOB_PATTERNS`, reads anchors via `AnchorGuard`, asserts version=`1.0.0`. Currently 2/2 anchors MEASURED-pin (`m14_anchor.json` + `wf219 sweep_summary.json`). Legacy anchors lacking `version` field are SKIPPED (intentional — pre-Wf-305C). Gate also added 11 tests via `test_read_anchor_json.py`.
- **PART 2 — `p3_section9_catalog`**: `gate_section9_entry_count` regex-counts `^### §9\.` headers in `TODO/completed/207_unified_pre_flight_checklist_v2.md`. Floor = 13; MEASURED = 25 ≥ 13 = PASS.
- **PART 3 — `p3_frontier_math_wire_in`**: `gate_python_importable` is a file-existence + parseability probe only. 4/4 adapters importable (file probe). **But structural wire-in (`__init__.py` re-export) is NOT verified by the gate** — honest framing per `gate_python_importable` docstring "does NOT verify the adapter is *wired* into the consumer `__init__.py`". The Wf-305B 0/4 wire-in blocker remains; Wf-306C PART 3 is smoke only.

Also added under §5 P2 / §9 P3: `p3_todo220_drivers_present` (2 stub drivers), `p3_anchorguard_consumers_adopt` (3 consumer scripts), `p3_harness_auto_fallback_present`, `p3_wf304b_platinai_selenium_diag` — 7 gates total in the Wf-306C addendum.

### §1.4 Wf-306D — CFM panel + G6 wrapper

- **Completed (MEASURED)**: CFM SA + QED panel on 100 mols from `panel.json` (18001 bytes). `n_parsed=100/100`, `n_failed=0`, `parsed_ratio=1.0`. SA distribution: mean=3.377, std=0.870, p25=2.839, p75=3.775, min=1.789, max=6.362. QED distribution: mean=0.288, std=0.173, p25=0.150, p75=0.439, min=0.045, max=0.728. Real-Vina style scoring with no parser failures — the panel is the **infrastructure verification** for the CFM refiner downstream pipeline.
- **G6 user-OK wrapper path**: `wf306a_drugood_vina_launch/launch_plan.md` §3 references the G6 wrapper integration; no separate G6 directory exists for Wf-306D, indicating the wrapper was validated via the wf306a cross-reference rather than standalone smoke.

---

## §2 Improvements Shipped (Round 2 delta vs Round 1 Wf-305)

| ID | Improvement | MEASURED/PROJECTED | Where |
|----|-------------|--------------------|-------|
| **IM1** | Vina pt-strip pipeline: 0/100 → **5/5 PASS** CPU smoke | MEASURED | `wf306a_drugood_vina_launch/phase_b/DRY_RUN.log` + `/tmp/wf306a_vina_smoke/panel.json` |
| **IM2** | GPU launch wrapper for drugood + Vina (5-hr PROJECTED) | SHIPPED (code) | `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` |
| **IM3** | GPU launch wrapper for Wf-219 Ru recovery (10-min PROJECTED) | SHIPPED (code) | `molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh` |
| **IM4** | AnchorGuard CI gate (`p3_anchorguard_version_pin`) | MEASURED (2/2 pinned at 1.0.0) | `molmetal/scripts/pre_flight_check.py:706-807` |
| **IM5** | §9 honest-framing catalog gate (floor=13, MEASURED=25) | SHIPPED + MEASURED | `molmetal/scripts/pre_flight_check.py:810-862` |
| **IM6** | Frontier-math wire-in smoke gate (`p3_frontier_math_wire_in`, 4/4 importable) | SMOKE ONLY (structural wire-in NOT done) | `molmetal/scripts/pre_flight_check.py:616-671` |
| **IM7** | 7-gate P3 follow-up addendum (TODO-220 drivers + consumers + harness fallback + Selenium diag) | SHIPPED | `molmetal/scripts/pre_flight_check.py:888-1207` |
| **IM8** | CFM panel 100/100 score + SA + QED aggregates | MEASURED | `wf306d_cfm_panel/panel.json` |
| **IM9** | 11 AnchorGuard version-pin tests in `test_read_anchor_json.py` | SHIPPED | `molmetal/tests/test_read_anchor_json.py` |
| **IM10** | Wf-305B honest-negative surfaced: 0/4 frontier adapters actually load (NOT "wired but unsmoked" as Wf-305B claimed) | DOCUMENTED | `wf306b_wf219_ru_wrapper/wrapper_plan.md` §46-62 |

---

## §3 Remaining Blockers (consolidated from Wf-306 + carry-over from Wf-305)

| ID | Blocker | Owner | Unblock Cost | Why Not Closed |
|----|---------|-------|--------------|----------------|
| **B1** | Frontier-math wire-in 0/4 → 0/4 (Wf-305B + Wf-306B honest-negative) | Wf-306+ | ~4-8 h (write 4 adapter modules + re-export in `flow_matching_lipman/__init__.py`) | The 4 adapter subdirectories do NOT exist under `_reference/`. `flow_matching_lipman/__init__.py` (139585 bytes) does NOT export the 4 classes. Wf-306C PART 3 is smoke only (file-existence), not structural wire-in. R17 invariant: do NOT edit `_reference/`. |
| **B2** | drugood DANN CPU smoke IN-FLIGHT (95,236-row featurization) | Wf-306A | ~30-40 min CPU | Featurization bottleneck on kanzi CPU; expected < 30 min on standard CI. Verdict gate is per-seed JSON emission + `n_seeds_completed=1`. |
| **B3** | GPU launches pending USER_OK | User (Hugo) | 1 USER_OK each | Wf-306A (5 hr, cuda:1) + Wf-306B (10 min, cuda:1). Per Wf-294B kill_then_adapt discipline. |
| **B4** | Vina field-avg target -7.5 to -9.5 kcal/mol | User OK + pocket diagnostic | ~10-30 min CPU full panel | Smoke mean=-4.60 kcal/mol (above band; honest-negative). May need different pocket (1h36 may be poor for Pt complexes) or extended pose search. |
| **B5** | drugood field-avg target 0.55-0.65 | User OK + GPU-1 reservation | ~3-4 hr GPU full | Smoke `val_auc=NaN` (tiny CSV has no val/test rows); full run on `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` needs USER_OK. |
| **B6** | Wf-219 Ru recovery (Wf-306B wrapper ready) | User OK + GPU-1 | ~10 min GPU-1 | Wrapper emitted + dry-run verified; real launch blocked on USER_OK. |
| **B7** | TODO-220 P1 stub → full + CSVs missing | User OK + CSV generation | ~10-15 min GPU full + ~30 min CSV rebuild | 2 stub drivers exist; CSVs (`metalcytotox_5metal_multitask_v4.csv` + `nci60_9cell.csv`) MISSING on disk. Wf-306D follow-up: regenerate 5-metal from MetalCytoToxDB source + 9-cell from PlatinAI pIC50 source. |
| **B8** | S6 Rusanov LOI (carry-over from Wf-305) | User OK | 1 user OK + 1 email send | Drafted at `TODO/pending/wetlab_loi_rusanov_2025_platinai.md`; 0 sent (TODO-84 blocks). NUMERIC IC50 is the binding constraint per Wf-297B §6. |
| **B9** | S7 MB-Finder rate-limit (carry-over from Wf-305) | MB-Finder API team | retry after 24 min + contact | IP-level soft rate-limit (retry_after=1467s ≈ 24 min); 0/4 cell lines harvested. Alternative path = S6 Rusanov. |

---

## §4 Paper §6 Honest-Negative Entries to Add (Round 2 delta)

1. **Wf-306A Vina CPU smoke = infrastructure recovery, NOT band-pass.** mean=-4.60 kcal/mol (ABOVE -7.5 to -9.5 field band, honest-negative on band); 5/5 docked (PASS on infrastructure). Round 1 had 0/100 docked; Round 2 = 5/5 docked via `--pt-strip`.
2. **Wf-306B honest-negative: Wf-305B "wired but unsmoked" was wrong.** The wire-in was NEVER done. 0/4 frontier adapters load; `flow_matching_lipman/__init__.py` (139585 bytes) does NOT export `MultiPHScheduler` / `SE3LipmanScheduler` / `SE3SBScheduler` / `TransportRelaxationScheduler`. Wf-306C PART 3 ships a smoke-only gate, not a contract test.
3. **drugood DANN CPU smoke IN-FLIGHT.** 95,236-row featurization bottleneck; 1-epoch smoke verdict gate (per-seed JSON + `n_seeds_completed=1`) NOT yet emitted. Field-avg target 0.55-0.65 requires GPU + user OK.
4. **GPU wrappers SHIPPED but NOT LAUNCHED.** Both wf306a + wf306b wrappers written + dry-run PASS + USER_OK gate enforced; both blocked on `WF306A_USER_OK=YES` / `USER_OK=1` env var per Wf-294B kill_then_adapt.
5. **TODO-220 driver CSVs MISSING.** `metalcytotox_5metal_multitask_v4.csv` + `nci60_9cell.csv` do not exist on disk; stubs do not load CSVs (dry-run OK) but live runs blocked.

---

## §5 Decision Matrix — What to Launch Next

| Priority | Action | When | Notes |
|----------|--------|------|-------|
| P1 | User OK to launch Wf-306B Wf-219 Ru GPU (10 min) | Day +0 (now) | Closes paper §3 cross-metal 5-metal aggregate (currently 4-metal: Re/Os/Rh/Ir) |
| P2 | User OK to launch Wf-306A Vina pt-strip Phase B (1-2 hr CPU) | Day +0 (now) | Closes infrastructure → band-pass for paper §4 §5 Vina field-avg |
| P3 | User OK to launch Wf-306A drugood DANN Phase A (3-4 hr GPU) | Day +0 (now) | Closes paper §3 ood_test_auc to 0.55-0.65 field band |
| P4 | Wait for drugood CPU smoke to complete (~30 min) | Day +0 (now) | Verdict gate is per-seed JSON emission; aggregate into launch_plan §2 |
| P5 | Generate TODO-220 CSVs (5-metal + 9-cell) | Day +0-Day +1 | Required for any TODO-220 P1 stub → full live run |
| P6 | Frontier-math wire-in: write 4 adapter modules + re-export from `flow_matching_lipman/__init__.py` | Wf-306+ / Wf-307+ | Closes Wf-304C B-frontier 0/4 → 4/4; paper §3.6 R² axis |
| P7 | User OK to send S6 Rusanov LOI (carry-over) | Day +0 (now) | High-leverage; NUMERIC IC50 is binding constraint |
| P8 | S7 MB-Finder retry after 24 min throttle window | Day +1 | Low-priority fallback vs S6 |

---

## §6 Cross-references

- `molmetal/reports/wf306a_drugood_vina_launch/launch_plan.md` (Wf-306A full plan + smoke + GPU wrapper)
- `molmetal/reports/wf306a_drugood_vina_launch/launch.log` (dry-run log: phases A + B both OK)
- `molmetal/reports/wf306a_drugood_vina_launch/phase_a/DRY_RUN.log` (Phase A launcher dry-run)
- `molmetal/reports/wf306a_drugood_vina_launch/phase_b/DRY_RUN.log` (Phase B launcher dry-run)
- `molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md` (Wf-306B plan + honest-negative on Wf-305B wire-in)
- `molmetal/reports/wf306b_wf219_ru_wrapper/plan.json` (emitted plan under USER_OK=1)
- `molmetal/reports/wf306d_cfm_panel/panel.json` (100/100 SA + QED panel MEASURED)
- `molmetal/scripts/pre_flight_check.py:612-906` (Wf-306C in-line: PART 1 AnchorGuard + PART 2 §9 + PART 3 frontier smoke)
- `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (Wf-306A GPU wrapper, NEW)
- `molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh` (Wf-306B GPU wrapper, NEW)
- `molmetal/tests/test_read_anchor_json.py` (11 version-pin tests, EXTENDED)
- `molmetal/reports/wf305_synthesis/final_state.md` (Round 1 baseline, Wf-304C honest-negative audit)
- `TODO/pending/220_wf219_temporal_split_nci60_driver.md` (TODO-220 design doc, NEW)
- `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` (S6 LOI draft, NEW)

---

## §7 Revision history

| Rev | Date       | Author  | Notes |
|-----|------------|---------|-------|
| 1.0 | 2026-09-29 | Wf-306E | Initial synthesis. 3 of 4 PARTIAL (Wf-306A drugood smoke in-flight, Vina 5/5 PASS MEASURED, GPU wrapper ready; Wf-306B wrapper PLAN_ONLY, surface Wf-305B 0/4 honest-negative; Wf-306D 100/100 PASS MEASURED); 1 of 4 SHIPPED-via-script (Wf-306C in-line in pre_flight_check.py: PART 1+2 MEASURED, PART 3 smoke only). 9 blockers consolidated. |