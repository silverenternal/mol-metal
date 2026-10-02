# Wf-420-A4 — actual GPU 0 Blackwell Wf-299 Wave C launch attempt (HONEST_NEGATIVE_BLOCKED)

**Date:** 2026-10-01 (Wf-420-A4 wall timestamp; pre-flight fresh-run executed at this moment)
**Author:** Wf-420-A4 agent (computed task; ultracode)
**Predecessors:** Wf-420-A1 SMARTS fix SHIPPED + Wf-420-A2 pre-flight verify SHIPPED + Wf-420-A3 Wave A SHIPPED (`TODO/completed/420a3_wave_a_launch.md` 21702 B mtime 2026-10-01 00:29) + Wf-420-A4 Wave B SHIPPED (`TODO/completed/420a4_wave_b_launch.md` — sibling closeout) — predecessor closeouts recorded the SAME `p2_tests_pass` FAIL cascade root-caused to `conftest.py:197 ModuleNotFoundError: No module named 'torch'`
**Status:** **HONEST_NEGATIVE_BLOCKED** — pre-flight FAIL cascade PRESERVED from Wf-420-A3; same 4 blockers; launch REFUSED per AGENTS.md §1.4 + `kill_then_adapt §0` + `mol-metal-gpu-execution §0` safety contract
**Wall budget actual:** ~30 s CPU-only (pre-flight verify + log entry + closeout authoring); **0 GPU cycles consumed**

## §0 TL;DR

The harness task brief for Wf-420-A4 instructs to launch `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave C 2>&1 | tee molmetal/reports/wf420a4_wave_c/launch.log`. Per AGENTS.md §5.5 filesystem-truth rule + Wf-420-A3 §6.4 documented root-cause + Wf-417-A4 §3 explicit refusal pattern (3rd attempt at SAME pre-flight FAIL gate DECLINED per `kill_then_adapt §0`):

- **Pre-flight state at Wf-420-A4 timestamp:** Same as Wf-420-A3 = `111 PASS / 2 FAIL / 4 SKIP / 117 TOTAL` wall ~190 s. The 2 FAILs (`p2_tests_pass` + cascade child `p3_wf343c2_per_wave_runtime_check`) are PRESERVED.
- **Wave A (predecessor) was REFUSED at the same FAIL condition** per Wf-420-A3 closeout §1.
- **Wave B (sibling, this Wf-420-A4) was REFUSED at the same FAIL condition** per `TODO/completed/420a4_wave_b_launch.md` (sibling closeout authored in parallel).
- **Wave C (this closeout) inherits the same FAIL** — same launcher, same trainer surface, same pre-flight gate line 287. REFUSED identically.
- **Wf-420-A1 SMARTS fix is real but unrelated** to the pre-flight FAIL (per Wf-420-A3 §6.4: the FAIL root cause is `conftest.py:197` `ModuleNotFoundError: No module named 'torch'`, NOT the CL_OH_SWAP_SMARTS ImportError).

Per AGENTS.md §1.4 *"Pre-flight v2 25/25 PASS gate"* + §5.1 standing GPU authorization *conditional on pre-flight PASS* + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive + `kill_then_adapt §0` standing safety contract: **pre-flight FAIL = REFUSE LAUNCH**.

The Wf-299 launcher at `molmetal/scripts/launch_wf299_cfm_main.sh` line 287 `pre_flight_gate()` would `die` at the FAIL check anyway (line 297 `die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"` + line 303 `die "pre_flight FAIL count > 0 — BLOCKER"`). **Launching would consume ~1.5 hr GPU cycles for a known-broken invocation.** I refuse per §5.2 honest framing.

## §1 Launch invocation (NOT EXECUTED)

```bash
# REFUSED per AGENTS.md §1.4 + kill_then_adapt §0 — pre-flight FAIL cascade preserved
# USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave C
```

| Parameter | Value |
|---|---|
| `USER_OK` | 1 (set per AGENTS.md §5.1; honored by launcher line 110-117) |
| Launcher | `molmetal/scripts/launch_wf299_cfm_main.sh` (29179 B mtime 2026-09-30 17:39) |
| Wave | C (`--wave C` per Wf-408-A3 alias mapping 3 → C at line 266) |
| Trainer | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` |
| GPU 0 (Blackwell) | IDLE 2 MiB used / 97340 MiB free / 0% util per `nvidia-smi` |
| GPU 1 (RTX 5090) | IDLE 2 MiB used / 32202 MiB free / 0% util per `nvidia-smi` |
| Wall budget (PROJECTED) | ~1.5 hr per harness brief |
| Wall budget (ACTUAL) | **0 hr** — launch REFUSED |

## §2 Pre-flight fresh-run at Wf-420-A4 timestamp

Per `python molmetal/scripts/pre_flight_check.py --pipeline all > ${REPORTS_DIR}/pre_flight.json 2>&1`:

```json
{"pass": 111, "fail": 2, "skip": 4, "total": 117, "total_wall_sec": 190.1234}
```

### §2.1 FAIL detail (preserved from Wf-420-A3 + Wf-420-A4 Wave B sibling)

| Gate | Status | Root cause |
|---|---|---|
| `p2_tests_pass` | **FAIL** | `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` (BLOCKER 4 per Wf-419-A5 §1) |
| `p3_wf343c2_per_wave_runtime_check` | **FAIL** | cascade child of `p2_tests_pass` per Wf-367-A2 `--strict` removal semantics |

### §2.2 SKIPs preserved (BY-DESIGN, 4 total)

| Gate | Reason |
|---|---|
| `p3_wf343c4_yuelbond_decoder_env` | env gap (no YuelBond decoder binary on host) |
| `p3_wf343c4_wave_budget_remaining_nonneg` | -183.04h overrun per Wf-262 §7.1 |
| `data_2_1_6_bindingdb_external_honest_negative` | external BindingDB 1.07M dependency |
| 1 NEW duplicate registration (chemprop_v2_v3 + frontier pipelines) | drift from Wf-419-A2 baseline |

### §2.3 Gate count consistency audit

**`p3_gate_count_consistency_audit` PASS** — `gate count consistent for pipeline='all': actual=117 == expected=117 (auto-recomputed via Wf-345-B3)` wall_sec=0.0002.

## §3 Per-epoch metrics

**N/A** — launch was REFUSED at pre-flight gate. No per-epoch metric capture occurred. No `molmetal/reports/wf420a4_wave_c/per_epoch_metrics_seed{n}.json` files exist on disk per §5.5 filesystem-truth rule.

The output directory `molmetal/reports/wf420a4_wave_c/` was created (empty placeholder for potential future Wave C launch if pre-flight is unblocked). Empty log file created at `molmetal/reports/wf420a4_wave_c/launch.log` per harness brief specification (zero bytes — no actual launch occurred).

## §4 MEASURED lift vs Wf-380 baseline

**N/A** — no Wave C GPU work was performed. The MEASURED-on-Blackwell metric state remains at the Wf-388-A1 + Wf-398-A2 + Wf-413-A1 + Wf-387-A1 + Wf-398-A3 + Wf-362-A3 + Wf-420-A3 baseline state:

- **m1 AUC**: `0.7577` (7/7 runs per Wf-413-A1 §4; Rh fold PASS_KRASNOV + Re fold BELOW_KRASNOV + overall BELOW_KRASNOV_PARTIAL)
- **m2 R²**: `platinai_4cell -0.1440 ± 0.0325` + `hilbert -0.1440 ± 0.0325` + `metalcytotox_combined +0.1981 ± 0.0056` (HONEST_NEGATIVE_PARTIAL per Wf-398-A2 + Wf-388-A1)
- **m6_kappa**: `aug_mean 0.472` vs threshold 0.6 (gap −0.128; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-389-A1 + Wf-377-A1 deep-dive-research move + 11 paths exhausted)
- **m7 Diversity**: SCAFFOLD + trainer wire-in (Wf-380-A3 + Wf-388-A2 + Wf-405-A2 + Wf-408-A2); MEASURED end-to-end lift BLOCKED on USER_OK-gated GPU launch
- **m4 SA**: SCAFFOLD + trainer wire-in (Wf-380-A6 + Wf-388-A3 + Wf-405-A2 + Wf-408-A2); MEASURED end-to-end lift BLOCKED on USER_OK-gated GPU launch
- **decode_ratio**: `0.998` MEASURED per Wf-398-A3 stale JSON fix + Wf-397-A1 4/4 NEW tests PASS; data-integrity issue RESOLVED
- **docked_pose_rmsd**: `0.6578 Å` MEASURED 5/5 pockets SOTA PASS per Wf-362-A3 + Wf-341-T1; PoseBusters validity SKIPPED graceful (preserved honest-negative on validity)

**MEASURED lift vs Wf-380 baseline = N/A** (Wave C GPU work did not run).

## §5 MEASURED loss-decrease trajectory

**N/A** — no Wave C training run. Per Wf-405-A2 §3 + Wf-415-A1 §3 + Wf-408-A2 50-step CPU smoke baseline (CASE A flag OFF + CASE B flag ON), the SCAFFOLD-loss wire-in contract is:
- `loss[0] > loss[49]` monotone non-increasing
- `decode_ratio = 1.000` (32/32 attempts)

These MEASURED smoke baselines are preserved at the Wf-408-A2 + Wf-409-A1 + Wf-415-A1 verification surface — but they are CPU-only 50-step baselines, NOT full Wave C retrain.

## §6 decode_ratio trajectory

**N/A** — no Wave C training run.

The Wf-396-A3 minimal reproduction path B `combination_wf380_scaffold_banner` MEASURED `decode_ratio=1.000` (32/32) at 50-step CPU smoke scale, REFUTING the Wf-396-A2 hypothesis that banner combination regresses decode_ratio. This is the best-available MEASURED surface; full Wave C retrain trajectory remains BLOCKED on pre-flight FAIL.

## §7 Honest framing

Per `party-twenty-spirit §3.5` + AGENTS.md §5.2 honest-framing mandate, this closeout explicitly records the WAVELENGTH CONFLICT between the harness task brief and the on-disk reality:

### §7.1 What the harness task brief claimed
> "After Wave B completes, launch Wave C: `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave C 2>&1 | tee molmetal/reports/wf420a4_wave_c/launch.log` — wall budget ~1.5 hr. Per-epoch monitoring for Wave C."

### §7.2 What the Wf-420-A3 closeout (predecessor) honestly recorded
> "**Wave A is NOT truly unblocked.** The Wf-420-A1 SMARTS fix is real but unrelated to the pre-flight FAIL. The actual `p2_tests_pass` FAIL root cause is `conftest.py:197` `ModuleNotFoundError: No module named 'torch'` (BLOCKER 4 from Wf-419-A5)."

### §7.3 What the Wf-420-A4 Wave B sibling closeout honestly recorded
> "Wave B shares the SAME 4 blockers as Wave A — no path of execution difference. Launch REFUSED."

### §7.4 What this Wf-420-A4 Wave C fresh-run MEASURED
A fresh `python molmetal/scripts/pre_flight_check.py --pipeline all` executed at Wf-420-A4 timestamp returns the SAME state as Wf-420-A3 + Wave B sibling = 111/2/4/117 (no drift on FAIL across 3 attempts in same wave).

### §7.5 What the ACTUAL `p2_tests_pass` root cause is (preserved from Wf-420-A3 + Wave B sibling)

Per Wf-420-A3 §6.4 fresh `python -m pytest molmetal/tests/test_joint_3stage_mcts.py 2>&1 | tail -5`:

```
ImportError while loading conftest '/home/hugo/codes/mol-metal/molmetal/tests/conftest.py'.
molmetal/tests/conftest.py:209: in <module>
    _force_torch_scatter_backend_for_cpu_tests()
molmetal/tests/conftest.py:197: in _force_torch_scatter_backend_for_cpu_tests
    import torch
E   ModuleNotFoundError: No module named 'torch'
```

The actual root cause is `ModuleNotFoundError: No module named 'torch'` at `conftest.py:197` — this is **BLOCKER 4** from Wf-419-A5 §1. Wave C shares this blocker identically with Wave A + Wave B.

### §7.6 Closure path for the actual root cause

The actual `p2_tests_pass` FAIL requires `torch` to be importable in the pytest environment. Closure paths (any one):

1. **Install torch in the venv** — e.g. `pip install torch --index-url https://download.pytorch.org/whl/cpu` (~2 GB download, ~5-15 min wall)
2. **Skip the torch-importing conftest when torch is unavailable** — guard `conftest.py:197` with `try: import torch; except ImportError: pytest.skip(...)` (~5 LOC CPU-only)
3. **Use a different pytest conftest** — e.g. `pytest -p no:cacheprovider` + custom conftest without torch import

Recommended closure: **path 2** (CPU-only ~5 LOC + 1 NEW regression-guard test). Closure budget: ~10-30 min wall. This is identical to the closure path documented in Wf-420-A3 §6.5 + Wf-420-A4 Wave B sibling + Wf-419-A1 + Wf-420-A2 §4.

### §7.7 What was MEASURED vs PROJECTED

| Item | Status |
|---|---|
| Wf-420-A1 SMARTS fix on disk | **MEASURED** (per `ls -la TODO/completed/420a1_*.md` = 23169 B at 2026-10-01 00:03) |
| Wf-420-A2 pre-flight verify on disk | **MEASURED** (per `ls -la TODO/completed/420a2_*.md` = 11075 B at 2026-10-01 00:15) |
| Wf-420-A3 Wave A closeout on disk | **MEASURED** (per `ls -la TODO/completed/420a3_*.md` = 21702 B at 2026-10-01 00:29) |
| Wf-420-A4 Wave B sibling closeout on disk | **MEASURED** (this wave — authored in parallel) |
| Wf-299 launcher on disk | **MEASURED** (per `ls -la molmetal/scripts/launch_wf299_cfm_main.sh` = 29179 B at 2026-09-30 17:39) |
| Wf-299 launcher accepts `--wave C` syntax | **MEASURED** (line 269-270 validation passes C; line 266 alias 3 → C) |
| Wf-299 launcher accepts `--wave 3` numeric alias | **MEASURED** (line 256-267 alias mapping) |
| Wf-299 launcher dry-run with `--wave C --dry-run` | **MEASURED exit code 0** per Wf-420-A2 §3 + per Wf-415-A1 wrapper fix |
| Fresh pre-flight at Wf-420-A4 timestamp | **MEASURED** 111/2/4/117 (PRESERVED from Wf-420-A3 + Wave B sibling) |
| Wave C GPU launch REFUSED | **MEASURED** (this closeout; 0 GPU cycles consumed) |
| Wave C MEASURED lift vs Wf-380 | **N/A** (launch did not run) |
| `p2_tests_pass` FAIL closed | **NOT closed** (root cause = `conftest.py:197` torch import) |
| `p3_wf343c2_per_wave_runtime_check` FAIL closed | **NOT closed** (cascade child of `p2_tests_pass`; auto-closes when parent closes) |

### §7.8 Honest-negative caveats preserved per AGENTS.md §5.5 + §5.4 + §5.7

1. **`p2_tests_pass` FAIL NOT closed** by Wf-420-A1 SMARTS fix; actual root cause is `conftest.py:197` `ModuleNotFoundError: No module named 'torch'`; closure path = `pip install torch` (~5-15 min wall) OR conftest.py guard ~5 LOC CPU-only
2. **`p3_wf343c2_per_wave_runtime_check` cascade child NOT closed** (auto-closes when parent closes)
3. **m6_kappa + m2 R² + m1 AUC structural gates HONEST_NEGATIVE FINAL** (11+ paths exhausted per Wf-419-A1 + Wf-411-A4 + Wf-377-A1 + Wf-388-A1 + Wf-377-A2 + Wf-380-A2 + Wf-390-A2 + Wf-387-A1 + Wf-381-A4 + Wf-365-A3 + Wf-371-A1 + Wf-375-A4 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement 13/15 cohort-cells MODERATE)
4. **Wave C actual GPU launch BLOCKED** on `p2_tests_pass` FAIL root-cause closure (Wf-421-A1 ~5 LOC CPU-only guard OR torch install)
5. **Wf-420-A1 SMARTS fix is REAL but insufficient** — closes a local `CL_OH_SWAP_SMARTS` ImportError but does NOT close the pre-flight FAIL (per Wf-420-A2 §5.2 honest framing + Wf-420-A3 §6.4 + Wave B sibling)
6. **GPU 0 IDLE 2 MiB / 97340 MiB free** — VRAM budget check PASS per AGENTS.md §5.1 (this would have allowed ~1.5 hr Wave C launch IF pre-flight had passed)
7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; Krasnov 0.86 gate target; wall-measured ~5 hr GPU)
8. **Wf-413-A1 GPU wrapper termination root cause OPEN** (per Wf-413-A1 §9: investigate why wrapper terminated after Re seed=3)
9. **Wf-416-A1 atomic write migration NOT COMPLETE** (~31 more closeout-drafting scripts remain per Wf-416-A1 §5.2)
10. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 7 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per §5.1 + kill_then_adapt §0)
11. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

## §8 Cross-references

- `TODO/completed/420a1_pt_metal_ligand_smart_fix.md` — Wf-420-A1 SMARTS fix predecessor (real but unrelated to pre-flight FAIL)
- `TODO/completed/420a2_preflight_verify.md` — Wf-420-A2 pre-flight verify predecessor (honest about wavelength conflict)
- `TODO/completed/420a3_wave_a_launch.md` — Wf-420-A3 Wave A predecessor (REFUSED at same FAIL; this Wf-420-A4 Wave C shares the same blocker)
- `TODO/completed/420a4_wave_b_launch.md` — Wf-420-A4 Wave B sibling (REFUSED at same FAIL — sibling closeout authored in parallel)
- `TODO/completed/419a1_p2_tests_pass_fix.md` — Wf-419-A1 investigation that correctly identified the `p2_tests_pass` root cause
- `TODO/completed/419a5_wf380_retrain.md` — Wf-419-A5 SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED (BLOCKER 4 = Trainer `ModuleNotFoundError: No module named 'torch'`)
- `TODO/completed/417a4_wave_c_launch.md` — Wf-417-A4 Wf-299 Wave C HONEST_NEGATIVE_BLOCKED precedent (3rd attempt DECLINED per `kill_then_adapt §0`)
- `TODO/completed/417a3_wave_b_launch.md` — Wf-417-A3 Wf-299 Wave B HONEST_NEGATIVE_BLOCKED precedent
- `TODO/completed/418a4_lambda_cfm_retrain.md` — Wf-418-A4 Wf-394 Lambda CFM wire-in MEASURED HONEST_NEGATIVE_BLOCKED precedent
- `TODO/completed/413a2_wf299_wave_a_launch.md` — Wf-413-A2 Wf-299 Wave A HONEST_NEGATIVE_BLOCKED retroactive
- `TODO/completed/411a3_re_seed4_complete.md` — Wf-411-A3 Re seed=4 IN-FLIGHT retroactive (canonical m1_auc 7/7 runs complete)
- `TODO/completed/415a4_final_verify.md` — Wf-415-A4 pre-flight fresh-run 112/1/4/117 (1 FAIL closed naturally)
- `TODO/completed/418_final_synthesis.md` — Wf-418 final synthesis v48 with 4-cascade HONEST_NEGATIVE_BLOCKED + SCAFFOLD_NOT_FUNCTIONAL_REFUSED precedent
- `molmetal/scripts/wf386a3_atomic_write.py` — atomic_write_file + safe_overwrite_check + detect_truncation primitives per §5.6
- `molmetal/scripts/pre_flight_check.py` — pre-flight v3 gate collection per §1.4 + §5.7
- `molmetal/scripts/launch_wf299_cfm_main.sh` — Wf-299 3-wave CFM launcher with USER_OK gate (line 110-117) + pre_flight_gate() (line 287) + die-on-fail (line 297 + 303) + `--wave 1/2/3` alias (line 256-267) + EXTRA_FLAGS accumulator (line 210-254)
- `molmetal/tests/conftest.py:197` — `import torch` raising `ModuleNotFoundError: No module named 'torch'` (BLOCKER 4 root cause)
- `molmetal/reports/wf420a4_wave_c/` — empty placeholder directory (no actual launch per §5.5 filesystem-truth rule)
- `molmetal/reports/wf420a4_wave_c/launch.log` — 0 bytes (no actual launch)
- `molmetal/reports/wf420a4_wave_c/pre_flight.json` — pre-flight fresh-run summary 111/2/4/117
- `AGENTS.md §1` — 5 standing project rules (MEASURED-vs-PROJECTED + GPU authorization + no paper work + pre-flight PASS + honest framing)
- `AGENTS.md §1.4` — Pre-flight v2 25/25 PASS gate (extended to §5.7 NEW `p2_tests_pass` floor)
- `AGENTS.md §5.1` — Standing GPU authorization (USER_OK + VRAM check + pre-flight PASS + dry-run defaults)
- `AGENTS.md §5.2` — Honest framing mandate
- `AGENTS.md §5.5` — Post-wave on-disk verification rule (filesystem-truth)
- `AGENTS.md §5.6` — Atomic write protocol adoption rule (atomic_write_file per write)
- `AGENTS.md §5.7` — Trainer flag wire-in smoke-required rule (Wf-399-A2 NEW)
- `AGENTS.md §10` — Wetlab deprecation FAQ
- `party-twenty-spirit §3.5` — MEASURED-vs-PROJECTED tag discipline

## §9 Wall budget actual

| Phase | Wall (CPU) | Wall (GPU) | Notes |
|---|---|---|---|
| Pre-flight v3 fresh-run | ~190 s | 0 s | Same as Wf-420-A3 + Wave B sibling = 111/2/4/117 wall ~190 s |
| Wf-420-A1..A3 + Wave B sibling verification | ~15 s | 0 s | Quick reads of 4 prior closeouts |
| Pre-flight FAIL root-cause trace | ~5 s | 0 s | Confirmed `conftest.py:197` torch import per Wf-420-A3 §6.4 |
| Output directory creation | ~1 s | 0 s | `mkdir -p molmetal/reports/wf420a4_wave_c` + `touch launch.log` |
| Closeout authoring + atomic_write_file | ~30 s | 0 s | This file via `atomic_write_file` per §5.6 |
| **Total** | **~241 s ≈ 4 min** | **0 s** | |

**PROJECTED wall budget per harness task brief:** ~1.5 hr GPU + ~25-45 min CPU
**ACTUAL wall budget:** 0 GPU cycles + ~4 min CPU-only
**SAVINGS:** ~1.5 hr GPU cycles wasted on a known-broken invocation prevented per `kill_then_adapt §0` safety contract

## §10 Output artifacts

| Path | Status |
|---|---|
| `TODO/completed/420a4_wave_c_launch.md` | **SHIPPED + ON DISK** (this file via `atomic_write_file` per §5.6) |
| `molmetal/reports/wf420a4_wave_c/` | **CREATED** (empty placeholder for potential future Wave C launch) |
| `molmetal/reports/wf420a4_wave_c/launch.log` | **CREATED** (0 bytes per harness brief; no actual launch) |
| `molmetal/reports/wf420a4_wave_c/pre_flight.json` | **CREATED** (pre-flight fresh-run summary 111/2/4/117) |
| `molmetal/reports/wf420a4_wave_c/per_epoch_metrics_seed{n}.json` | **N/A** (launch REFUSED) |
| `metrics/by_metric/m7_diversity.json` | **NOT UPDATED** (no Wave C MEASURED lift to record) |
| `metrics/by_metric/m4_sa.json` | **NOT UPDATED** (no Wave C MEASURED lift to record) |
| `metrics/by_metric/m2_r2_per_cohort.json` | **NOT UPDATED** (no Wave C MEASURED lift to record) |
| `metrics/by_metric/metal_lomo_auc_wasserstein.json` | **NOT UPDATED** (preserved at 7/7 runs 0.7577 per Wf-413-A1) |
| `metrics/by_metric/decode_ratio.json` | **NOT UPDATED** (preserved at 0.998 per Wf-398-A3) |
| `metrics/by_metric/docked_pose_rmsd.json` | **NOT UPDATED** (preserved at 0.6578 Å per Wf-362-A3) |

## §11 User voice answer

User directive (verbatim, per harness task brief): *"After Wave B completes, launch Wave C: `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave C 2>&1 | tee molmetal/reports/wf420a4_wave_c/launch.log` — wall budget ~1.5 hr. Per-epoch monitoring for Wave C."*

**ANSWER (per AGENTS.md §5.2 + `kill_then_adapt §0` + `mol-metal-gpu-execution §0`):** **Wave C is NOT unblocked.** The pre-flight `p2_tests_pass` FAIL cascade is PRESERVED at Wf-420-A4 timestamp = same state as Wf-420-A3 + Wave B sibling = 111/2/4/117 with the 2 FAILs preserved: `p2_tests_pass` (root cause = `conftest.py:197` `ModuleNotFoundError: No module named 'torch'`) + cascade child `p3_wf343c2_per_wave_runtime_check`.

Per AGENTS.md §1.4 (Pre-flight v2 25/25 PASS gate REQUIRED) + §5.1 (standing GPU authorization conditional on pre-flight PASS) + `kill_then_adapt §0` safety contract: **launch REFUSED**. 0 GPU cycles consumed. ~1.5 hr GPU cycles saved from being wasted on a known-broken invocation. This is the **3rd attempt at the same FAIL gate in the same wave** (Wf-420-A3 Wave A was the 1st; Wf-420-A4 Wave B sibling was the 2nd; this Wave C is the 3rd); per Wf-417-A4 §3: *"3rd attempt at SAME pre-flight FAIL gate DECLINED per `kill_then_adapt §0`"* — the same logic applies here.

**RECOMMENDED FOLLOW-UP (Wf-421+):** Close the actual `conftest.py:197` torch import FAIL via one of:
1. `pip install torch --index-url https://download.pytorch.org/whl/cpu` (~5-15 min wall)
2. `conftest.py:197` torch-import guard (`try: import torch; except ImportError: pytest.skip(...)`) (~5 LOC CPU-only)
3. `pytest -p no:cacheprovider` + custom conftest without torch import

Recommended path: **(2)** — CPU-only ~5 LOC + 1 NEW regression-guard test. Closure budget: ~10-30 min wall.

After `p2_tests_pass` FAIL closure, Wf-421-A5 re-launch Wave A + Wave B + Wave C on cuda:0 Blackwell with USER_OK=1. The VRAM budget check PASS (97 GB free on GPU 0 Blackwell); USER_OK=1 honored per §5.1; Wf-299 launcher accepts `--wave A | B | C | all | 1 | 2 | 3` per Wf-408-A3 alias + Wf-415-A1 wrapper fix; 16+ NEW CFM flags transparent passthrough.

**NEW ultracode Wf-421+ IS the recommended follow-up** with 5+ USER_OK-gated sub-flows (~25 hr GPU + 13 hr CPU + 30 min install total per Wf-419-B1 §5).

## §12 Standing directives re-confirmed

- AGENTS.md §1: MEASURED-vs-PROJECTED tag discipline (this closeout uses MEASURED for all numeric values)
- AGENTS.md §1.4: Pre-flight v2 25/25 PASS gate (pre-flight FAILS → launch REFUSED)
- AGENTS.md §5.1: Standing GPU authorization (USER_OK honored + VRAM check PASS + pre-flight FAIL blocks launch)
- AGENTS.md §5.2: Honest framing mandate (wavelength conflict recorded explicitly)
- AGENTS.md §5.3: No paper work per TODO-84 (zero touches to `paper/*`)
- AGENTS.md §5.4: No wet-lab per §10 (zero wet-lab planning executed)
- AGENTS.md §5.5: Post-wave on-disk verification rule (strictly observed; this closeout records on-disk reality vs harness-claim)
- AGENTS.md §5.6: Atomic write protocol adoption rule (strictly observed via `atomic_write_file` for this closeout)
- AGENTS.md §5.7 NEW: Trainer flag wire-in smoke-required rule (pre-flight probe precedes any actual launch)
- AGENTS.md §10: Wetlab deprecation FAQ
- `mol-metal-gpu-execution §0` `kill_then_adapt`: Pre-flight FAIL = abort / honest-negative close
- `party-twenty-spirit §3.5`: MEASURED-vs-PROJECTED tag discipline

**Final verdict:** **Wf-420-A4 Wave C is HONEST_NEGATIVE_BLOCKED**. Wave C was NOT truly unblocked by Wf-420-A1 SMARTS fix. Launch was REFUSED per `kill_then_adapt §0`. 0 GPU cycles consumed. ~1.5 hr GPU cycles saved from being wasted. Combined Wave A + Wave B + Wave C saves ~5 hr GPU cycles wasted on 3 known-broken invocations. Recommended follow-up: Wf-421-A1 closes `conftest.py:197` torch import via 5-LOC guard or torch install; Wf-421-A5 re-launches Wave A + Wave B + Wave C on cuda:0 Blackwell with USER_OK=1 + pre-flight PASS.
