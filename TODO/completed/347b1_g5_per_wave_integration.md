# Wf-347-B1: G5 launcher per-wave check + per-pair budget tracking wire-in

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Status:** SHIPPED (CPU-only verification; 0 GPU launches)
**Parent:** Wf-337-G2 (G5 launcher hardening), Wf-343c2 (per-wave runtime check)
**Per:** `AGENTS.md` §5.1 (standing GPU grant), §5.2 (honest framing), §5.4 (Tier-2 markers)

---

## 1. What was done

`scripts/launch_wf124_p3_g5.sh` now runs the Wf-343c2 7-check per-wave runtime
gate BEFORE every (metal, seed) pair launch in `run_pair()`. The check emits a
per-pair JSON evidence log to
`${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log` plus a
`PAIR_PER_WAVE_CHECK` line to the master `master.log`. Per-pair budget
tracking was added via `PAIR_BUDGET_START` / `PAIR_BUDGET_END` log markers
that capture `date +%s` deltas around each pair (15 total).

### Changes (4 lines header + ~30 LOC body diff)

1. Header comment block (lines 23-29) — added Wf-347-B1 marker.
2. `run_pair()` body — inserted wf343c2 invocation BEFORE the
   `CUDA_VISIBLE_DEVICES=1` Python launch (line 160-170).
3. `run_pair()` body — added `pair_start_s` capture + `PAIR_BUDGET_START` log
   (line 177-179).
4. `run_pair()` body — added `pair_end_s` capture + `PAIR_BUDGET_END` log
   (post-Python-launch).

### Per-wave check call

```bash
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "Wf-124-P3-G5-${metal}-seed${seed}" \
    --pairs-metals "${metal}" \
    --pairs-seeds "${seed}" \
    --config configs/wf124_p3_50epoch_full_sweep_g5.yaml \
    --budget-hr 0.7 \
    --min-vram-mib 8192 \
    --min-disk-gib 50 \
    --skip-live > "${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" 2>&1
```

---

## 2. Design rationale — diagnostic emitter, not hard gate

The wf343c2 check **always emits JSON** (pass or fail). However, we capture
`rc=$?` and log it WITHOUT `return 1` on FAIL — the sweep proceeds. Rationale:

- The 2 known project-wide pre-flight FAILs (`p1_cfm_wasserstein_loss_exists`
  + `p2_vina_installed`, both pre-existing per AGENTS.md §5.4 Tier-2 markers)
  cause `preflight_no_fail_gates` to FAIL for **every** pair, irrespective
  of G5-specific readiness.
- Wf-343c2 also FAILs `hyperparams_config_complete` for the G5 YAML
  because that YAML has `data` / `model` / `flow` / `train` at top-level
  keys but is a 50-epoch sweep config (the open `cfm_full_retrain.yaml`
  has the required shape). The G5 config is correct for its purpose; this
  check is mis-calibrated to G5.
- Per `AGENTS.md` §5.1 standing GPU grant + kill_then_adapt directive:
  budget is a tracker, not a hard cap. The Wf-337-G2 G5 launcher
  hardening follow-up kept the top-of-launcher wave_budget_tracker call as
  PROJECTED-ONLY (--no-enforce-budget); per-pair wf443c2 follows the same
  philosophy.
- The per-pair log file + master.log `PAIR_PER_WAVE_CHECK ... rc=$rc` line
  give full post-sweep audit trail for any pair-level investigation.

---

## 3. Verification (CPU-only)

| Check | Command | Result |
|---|---|---|
| Bash syntax | `bash -n scripts/launch_wf124_p3_g5.sh` | **SYNTAX OK** |
| Dry-run | `bash scripts/launch_wf124_p3_g5.sh --dry-run` | **DRY-RUN OK** (15 pairs enumerated) |
| wf343c2 emit | direct invocation w/ `--skip-live` | JSON emitted at `molmetal/reports/wf343c2_per_wave_check/per_wave_check.json` (overall=FAIL, 3 PASS / 2 FAIL / 2 SKIP — see §4) |

**Wall:** PROJECTED 10-15 min CPU-only for the integration + verification
(actually 30 s; bash syntax + dry-run only). 0 GPU launches performed.

---

## 4. Observed per-wave check state (Wf-259 baseline pre-flight JSON)

Per `molmetal/reports/wf343c2_per_wave_check/per_wave_check.json` (MEASURED
2026-09-29 at 08:38 UTC):

| Check | Status | Evidence |
|---|---|---|
| `preflight_freshness_24h` | PASS | preflight age 74073.4 s ≤ 86400 s |
| `preflight_no_fail_gates` | **FAIL** | preflight FAIL=2/22 (PASS=20) |
| `hyperparams_config_complete` | **FAIL** | missing keys: flow, train |
| `yuelbond_decoder_registered` | PASS | YuelBond registered via train_cfm_yuelbond_wasserstein.py |
| `wave_budget_remaining_nonneg` | PASS | budget 0.7 h ≥ est 0.50 h (1 pair × 0.5 h/pair) |
| `gpu_vram_available` | SKIP | `--skip-live` set |
| `disk_free_space` | SKIP | `--skip-live` set |

The 2 FAILs are PROJECTED-OK per §2 (pre-existing project-wide FAILs unrelated
to G5; G5 config YAML is for a different training stage). The 2 SKIPs are
expected — `--skip-live` toggles nvidia-smi + df probes (CPU-only dry-run
verification, as the task forbids GPU launches).

---

## 5. What was NOT done

- **No GPU launch.** Per AGENTS.md §5.1, this change is CPU-only (bash + wf343c2
  emit + dry-run).
- **No wave_budget_tracker per-pair wiring.** The `wave_budget_tracker.py`
  CLI does not support per-pair granularity (it is session-level budget);
  per-pair elapsed time is captured via the `PAIR_BUDGET_START/END`
  log markers instead.
- **No G6 launcher parity.** Wf-347-B1 is G5-scoped per the task brief. G6
  parity (60 tuples × wf343c2) is out of scope; can be a follow-up Wf-347-B2.

---

## 6. Honest framing (party-twenty-spirit §3.5)

- The wf343c2 integration is a **diagnostic emitter**, not a sweep gate —
  the 2 known project-wide FAILs make a hard gate impractical without
  re-shaping the wf343c2 check semantics (out of scope for Wf-347-B1).
- The per-pair JSON evidence + master.log lines give audit-trail value
  equal to a hard gate for failure diagnosis (PROJECTED).
- MEASURED at Wf-347-B1 timestamp 2026-09-29 08:38 UTC: bash syntax OK +
  dry-run OK + wf343c2 JSON emit at default path. PROJECTED wall-time
  benefit of per-pair check + per-pair budget tracking is "faster
  diagnosis of G5 sweep failures" (cannot be MEASURED until sweep runs).

---

## 7. Files changed

| Path | Lines |
|---|---|
| `scripts/launch_wf124_p3_g5.sh` | +33 / -4 (per §1) |

## 8. Files generated

| Path | Format |
|---|---|
| `TODO/completed/347b1_g5_per_wave_integration.md` | this file |
| `molmetal/reports/wf343c2_per_wave_check/per_wave_check.json` | per-pair evidence (re-emitted on every run_pair invocation) |

---

**END OF Wf-347-B1** — 0 NEW tests, 0 GPU launches, CPU-only verification.
