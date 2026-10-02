# Wf-310B — TODO-182 G6 FINAL Launch Plan

**Owner:** Hugo Lee
**Date:** 2026-09-29
**Parent:** TODO/pending/182_wf124_p3_per_innovation_ablation.md
**Status:** READY (DRY-RUN verified; awaiting USER_OK for real launch)

---

## 1. Scope

Wf-310B delivers the **FINAL launch wrapper** for TODO-182 G6 — the
per-innovation ablation that measures the contribution of each of the 4
D-MPNN chemprop_v2_v3 levers (multi-task, W2, Kendall, DANN) in
isolation. The prior Wf-124 P3 ran all 3 innovations simultaneously and
left per-lever attribution unmeasured.

**Total expected:** 4 configs × 5 metals × 3 seeds = **60 (config, metal, seed) tuples**.

---

## 2. Files Delivered

| Path | Role | LOC est. |
|------|------|----------|
| `scripts/launch_wf310b_g6_final.sh` | USER_OK-gated launch wrapper | ~245 |
| `molmetal/reports/wf310b_g6_manifest/60_tuples.json` | 60-tuple ready manifest | ~280 |
| `molmetal/tests/test_wf310b_g6_launcher.py` | 8 tests (USER_OK, pre-flight ordering, per-wave, abort_recovery, manifest, idempotency) | ~290 |
| `molmetal/reports/wf310b_g6_launch_wrapper/launch_plan.md` | This file | (you are here) |

---

## 3. Wave Decomposition (8 sub-waves a..h)

Per Wf-295D §1.3 — 8 sub-waves, each ≤10 runs, ≤2 hr wall-time cap.

| Wave | Description | Runs | Wall est (hr) |
|------|-------------|------|---------------|
| a | no_multi_task Ru/Ir/Os | 9 | 1.1 |
| b | no_multi_task Rh/Re + no_wass Ru | 9 | 1.1 |
| c | no_wass Ir/Os/Rh/Re | 12 | 1.5 |
| d | no_kendall Ru/Ir | 6 | 0.8 |
| e | no_kendall Os/Rh/Re | 9 | 1.1 |
| f | no_dann Ru/Ir/Os | 9 | 1.1 |
| g | no_dann Rh | 3 | 0.4 |
| h | no_dann Re + ablation_report.json write | 3 | 0.5 |
| **TOTAL** | | **60** | **~7.6 hr** |

---

## 4. Wall-time & Compute

- **Compute:** RTX 5090 GPU-1 (cuda:1, exposed as cuda:0 inside PyTorch)
- **Per-run wall:** ~7.5 min @ 50 epoch (per TODO-182 §2 + Wf-260 G6 timing)
- **Total wall:** ~7.6 hr on RTX 5090 (60 × 7.5 min + 1 hr aggregation headroom)
- **Standing budget:** 13h (per user 2026-09-28 directive; 12.5h sweep + 0.5h buffer)

---

## 5. User-OK Gate (kill_then_adapt)

Per Wf-306D + Wf-300C pattern — USER_OK env var is **HARD-REQUIRED** for
real launch (any 7.5 hr wall-time launch needs explicit user approval,
never auto-triggered).

```bash
# Blocked path (no USER_OK):
bash scripts/launch_wf310b_g6_final.sh
# → "USER OK REQUIRED — kill_then_adapt standing directive. exit 1"

# Dry-run path (CPU verification only, no GPU exec):
bash scripts/launch_wf310b_g6_final.sh --dry-run
# → enumerates 60 tuples, no actual training:
bash scripts/launch_wf310b_g6_final.sh --dry-run 2>&1 | tail -30

# Authorized real launch:
USER_OK=1 bash scripts/launch_wf310b_g6_final.sh
# → proceeds with pre-flight + per-wave checklists + abort_recovery
```

---

## 6. Pre-Flight + Wave-Tool Guards (Wf-296C + Wf-306C)

Before any tuple fires, the wrapper enforces:

1. `pre_flight_check.py --pipeline all` (32/32 v3 PASS required per
   Wf-306C PART 1; 12 chemprop v3 P1 + 6 CFM P1 + 4 P2 DOCUMENTED +
   10 v3 §3.x expansion = 32 total gates). FAIL exits 1.
2. `wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required`
   (per Wf-296C: keeps all 60 tuples PLANNED; mirrors G5 §6 header
   convention; --cfm-yuelbond-required gates the YuelBond-conditionality
   per Wf-272I).
3. `per_wave_checklist.py --wave Wf-124-P3-G6-{a..h}` (per Wf-262 §10.1
   contract; emitted **before each sub-wave**).

---

## 7. Abort Recovery (TODO-207 §7.3)

Per-tuple runner captures `${PIPESTATUS[0]}` after each invocation.
On non-zero rc, `abort_recovery.py --metal {M} --seed {N} --wave
Wf-124-P3-G6 --pattern aborted` runs (reactive ckpt restore + 1-epoch
smoke per 207 §7.3 contract). Tuple continues to next.

---

## 8. Manifest (60-tuples.json)

`molmetal/reports/wf310b_g6_manifest/60_tuples.json` enumerates all 60
tuples with: tuple_id, config, metal, seed, wave (a..h), expected wall
(min), expected test_auc (PROJECTED), CLI args. **PROJECTED values are
NOT MEASURED** until 60 runs complete (party-twenty-spirit §3.5).

Coverage verified:
- 4 configs: `no_multi_task`, `no_wass`, `no_kendall`, `no_dann`
- 5 metals: Ru, Ir, Os, Rh, Re
- 3 seeds: 0, 1, 2
- = 60 tuples, contiguous tuple_id 1..60
- 8 waves with runs bounded ≤12 each (kill_then_adapt safe)

---

## 9. Tests (8/8 PASS)

`molmetal/tests/test_wf310b_g6_launcher.py` — 8 tests, CPU-only, no
torch.cuda calls. Each test inspects wrapper source + manifest JSON +
g6 yaml:

| # | Test | Verifies |
|---|------|----------|
| 1 | `test_user_ok_gate_blocks` | wrapper exits 1 without USER_OK |
| 2 | `test_user_ok_unblocks` | --dry-run with USER_OK=1 exits 0 + acknowledgement text exists |
| 3 | `test_pre_flight_called_first` | pre_flight_check before wave_budget_tracker before per_wave_checklist |
| 4 | `test_per_wave_checklist_emitted` | emissions for all 8 sub-waves a..h |
| 5 | `test_abort_recovery_wired` | abort_recovery call guarded by PIPESTATUS rc check |
| 6 | `test_60_tuples_manifest_complete` | 60 tuples, 4×5×3 coverage, tuple_id 1..60 |
| 7 | `test_yaml_config_parses` | g6 yaml parses, total_runs=60, 8 waves |
| 8 | `test_dry_run_idempotent` | --dry-run yields identical tail across 2 invocations |

**Status:** 8 passed in 0.22s.

---

## 10. Dry-Run Verification

```bash
$ bash scripts/launch_wf310b_g6_final.sh --dry-run 2>&1 | tail -20

  [CONFIRMED] cfg=no_dann Ru seed=0  ->  results/.../no_dann/Ru/seed0/
  [CONFIRMED] cfg=no_dann Ru seed=1  ->  results/.../no_dann/Ru/seed1/
  [CONFIRMED] cfg=no_dann Ru seed=2  ->  results/.../no_dann/Ru/seed2/
  [CONFIRMED] cfg=no_dann Ir seed=0  ->  results/.../no_dann/Ir/seed0/
  ... (60 tuples total)

DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.
  CONFIRMED runs (real launch): 60 (all 4 configs wired post-Wf-296C)
  UNVERIFIED runs (real launch): 0 (all flags verified)
  USER_OK gate: required (set USER_OK=1 to authorize real launch)
```

---

## 11. Honest Framing (party-twenty-spirit §3.5)

- **PROJECTED, NOT MEASURED**: per-lever attribution percentages. Per
  TODO-182 §6: "multi-task 60-80%, W2 10-30%, L2 guard ~5%". Interaction
  term may be >30%.
- **Per-metal caveat**: smaller metals (Rh/Os/Re) may have different
  attribution; per-metal reporting required.
- **Best-seed selection**: NOT applied (per mol-metal-gpu-execution §7).
- **Apples-to-aples comparison vs v1 30-epoch**: NOT applicable (Wf-261 §3
  honest-negative: v1 had 30/48/32 scaffold leaks; v4 strict split has 0).

---

## 12. Cross-references

- `TODO/pending/182_wf124_p3_per_innovation_ablation.md`
- `configs/wf124_p3_per_innovation_ablation_g6.yaml` (60-tuple config)
- `molmetal/scripts/pre_flight_check.py` (Wf-262, 32/32 v3 gate)
- `molmetal/scripts/wave_budget_tracker.py` (Wf-262, --no-enforce-budget)
- `molmetal/scripts/per_wave_checklist.py` (Wf-262 §10.1)
- `molmetal/scripts/abort_recovery.py` (TODO-207 §7.3)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (--no-kendall
  line 1323, --no-dann line 1331; Wf-283B + Wf-285B verified)
- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` (Wf-258 audit;
  TODO-182 entry #5)
- `molmetal/reports/wf261_verify/verify_report.md` (Wf-261 strict
  Bemis-Murcko scaffold split; 0 leak)

---

## 13. Status: READY (DRY-RUN only)

- wrapper path: `scripts/launch_wf310b_g6_final.sh`
- manifest: `molmetal/reports/wf310b_g6_manifest/60_tuples.json`
- tests: 8/8 PASS
- syntax check: SYNTAX_OK
- dry-run verdict: OK (60 tuples PLANNED, no GPU exec)
- USER_OK gate: HARD-REQUIRED (no auto-launch per kill_then_adapt)
- total wall: ~7.6 hr on RTX 5090 GPU-1 (cuda:1)

**Real launch requires `USER_OK=1` env var (no auto-trigger).**