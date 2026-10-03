# TODO-321H — TODO-119 P1.4 chemprop v3 LOMO CPU scaffold (Wf-321H)

**Date**: 2026-09-29
**Status**: 🚀 **CPU scaffold VERIFIED (12/12 tests PASS + pre-flight gate PASS); GPU portion BLOCKED on user OK per kill_then_adapt §0**
**Owner**: Hugo Lee + ultracode agent
**Parent**: `WEAK_TO_STRONG §3.4` + `TODO-119 §3 P1.4` + `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (P1.4 deferred to TODO-119)
**GPU pipeline**: Wf-300C Wave-A (4 ablations × 1 seed × 3 ep smoke, ~2 hr on PRO 6000 cuda:0)

---

## §0 Why this file exists

Per WEAK_TO_STRONG §3.4: TODO-119 P1.4 chemprop v3 LOMO is a 4-wave sweep (4
ablations × 1 seed × 3 ep smoke → 5×5×50 full). Currently in GPU pipeline
(Wf-300C Wave-A). Need CPU scaffold verification BEFORE requesting fresh
user OK for the GPU launch — pre-flight v2 25/25 PASS gate (TODO-207).

This Wf-321H delivers the CPU scaffold:
- chemprop v2 vs v3 mode flag (`--chemprop-mode`) verified
- LOMO scaffold levers (`--chemprop-mode=v3-fix` + `--chemprop-min-epochs`
  + `--chemprop-lomo-init-scale`) verified
- New pre-flight gate `p3_todo_119_p14_cpu_smoke_ready` wired + PASS
- Wf-300C launcher verified present and `--dry-run` wired

---

## §1 MEASURED results (CPU scaffold)

| Item | Status | Evidence |
|---|---|---|
| New test file | ✅ 12/12 PASS | `molmetal/tests/test_wf321h_todo_119_p14_cpu_scaffold.py` |
| chemprop-mode CLI flag | ✅ VALIDATED | `--chemprop-mode {v2,v3,v3-fallback,v3-fix,shared-encoder}` present (default `v3-fallback`) |
| LOMO scaffold levers | ✅ VALIDATED | `--chemprop-min-epochs=10` + `--chemprop-lomo-init-scale=1e-3` defaults |
| Pre-flight gate | ✅ PASS | `p3_todo_119_p14_cpu_smoke_ready` PASS in `chemprop_v2_v3` pipeline (wall 13.7s) |
| Wf-300C launcher | ✅ PRESENT | `scripts/launch_wf300_cfm_main.sh` exists with `--dry-run` mode |

**Pre-flight output** (2026-09-29 10:50:51):
```
{
  "name": "p3_todo_119_p14_cpu_smoke_ready",
  "priority": "P3",
  "pipeline": "chemprop_v2_v3",
  "status": "PASS",
  "evidence": "help exposes all 4 token(s) of '--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale' (rc=0)",
  "wall_sec": 13.7437
}
```

---

## §2 Honest framing (per `party-twenty-spirit` §3.5)

- **MEASURED**: 12/12 tests PASS + pre-flight gate PASS on CPU scaffold
- **PROJECTED**: P1.4 full sweep (4 ablations × 1 seed × 3 ep smoke, then 5×5×50 full) is in GPU pipeline (Wf-300C Wave-A); NOT executed on CPU scaffold
- **BLOCKED**: Chemprop v3 LOMO needs chemprop v2 featurization fix (Wf-320D) before GPU portion can run end-to-end — this is a known blocker NOT addressed by the CPU scaffold
- **Flag mismatch**: User-facing shorthand `--chemprop-version {v2,v3}` is implemented at CLI as `--chemprop-mode` (per Wf-263A-Implement). The P1.4 ablation can select v2 or v3 explicitly via `--chemprop-mode v2` or `--chemprop-mode v3`.

---

## §3 Files added / modified

| Path | Action | Description |
|---|---|---|
| `molmetal/tests/test_wf321h_todo_119_p14_cpu_scaffold.py` | NEW | 12 tests across 4 buckets (chemprop-mode flag + LOMO scaffold + pre-flight + GPU portion) |
| `molmetal/scripts/pre_flight_check.py` | EDIT | Added `p3_todo_119_p14_cpu_smoke_ready` gate to chemprop_v2_v3 pipeline (13th gate) |

---

## §4 Test breakdown (12 tests, 4 buckets)

### §4.1 (i) chemprop v2 vs v3 flag correctness (3 tests)
- `test_argparse_chemprop_mode_accepts_v2_and_v3` — CLI accepts `{v2,v3,v3-fallback,v3-fix,shared-encoder}`
- `test_argparse_chemprop_mode_default_is_v3_fallback` — default is `v3-fallback` (Wf-263A contract)
- `test_argparse_chemprop_mode_mutually_excludes_v3_with_enhanced_head` — `--enhanced-head` opt-in documented

### §4.2 (ii) LOMO scaffold-split integrity (3 tests)
- `test_chemprop_min_epochs_floor_default_is_10` — `--chemprop-min-epochs=10` floor
- `test_chemprop_lomo_init_scale_default_is_1e_3` — `--chemprop-lomo-init-scale=1e-3` anti-bias
- `test_lomo_scaffold_shipped_v3_fix_combines_min_epochs_and_init_scale` — LOMO scaffold is shipped

### §4.3 (iii) pre-flight gate validation (3 tests)
- `test_pre_flight_script_imports_and_parses_argv` — `pre_flight_check.py --help` works
- `test_pre_flight_chemprop_v2_v3_pipeline_has_chemprop_mode_gate` — existing pre-flight gates intact
- `test_pre_flight_gate_p3_todo_119_p14_cpu_smoke_ready_wired` — new gate is wired

### §4.4 (iv) GPU portion integration readiness (3 tests)
- `test_wf300c_launch_script_exists` — `scripts/launch_wf300_cfm_main.sh` exists
- `test_wf300c_launcher_references_todo119_p14_wave_a` — references Wave-A + 4 ablations
- `test_wf300c_launcher_dry_run_mode_wired_no_gpu_work` — `--dry-run` mode wired (CPU scaffold safe)

---

## §5 What is NOT addressed (honest)

- **Chemprop v2 featurization fix (Wf-320D)** — the GPU portion's chemprop v2
  featurization is a known blocker; this CPU scaffold does NOT address it.
- **GPU portion (Wf-300C Wave-A)** — NOT executed. CPU scaffold only validates
  the script structure + LOMO levers + pre-flight gate.
- **5×5×50 full sweep** — entirely GPU-blocked per kill_then_adapt §0 (9.5 hr).

---

## §6 Next step (ultracode agent action required)

To execute the GPU portion, the user must grant fresh user OK per
`mol-metal-gpu-execution` §0 (kill_then_adapt directive). Until then,
CPU scaffold remains the verifiable artifact.

**Status**: ⏸️ CPU scaffold VERIFIED, GPU portion BLOCKED on user OK.
