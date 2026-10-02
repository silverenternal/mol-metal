# Wf-335-P2 — sm_120 Triton retune GPU wrapper (CLI scaffold)

**Wf-id:** Wf-335-P2 (2026-09-29)
**Status:** SHIPPED (CPU-only CLI scaffold; no GPU launch)
**Operator:** Claude (Sonnet 4.6)
**Pre-flight:** N/A (CPU-only scaffold per AGENTS.md §5.1; no retrain / panel / ablation launch)
**Wave:** Wf-335-P2 follows Wf-333-T1 (per-kernel autotune extension) + Wf-331-X3 (sm_120 CPU scaffold)

---

## Summary

Provides a **single-entry CLI wrapper** that combines the Wf-331-X3
sm_120 scaffold (`molmetal/scripts/wf331x3_sm120_retune.py`, 12-config
generic grid) with the Wf-333-T1 per-kernel presets
(`molmetal/scripts/wf333t1_sm120_autotune_extended.py`, 117 per-kernel
configs across 4 categories) into one launch surface for the eventual
GPU retune on RTX 5090 (Blackwell, sm_120).

The wrapper itself is **CPU-only** — it does not perform an actual
GPU launch by default.  The full retune remains **DEFERRED** per
AGENTS.md §5.1 (standing GPU authorization + VRAM budget + pre-flight
PASS + config-aligned hyperparameters required before any launch
> 1 hr).  Per Wf-335 context the wrapper provides the CLI surface +
launch-plan JSON manifest; the actual GPU execution is wired into
the follow-up wave (per `mol-metal-gpu-execution` §3 venv setup +
§6 5-seed protocol).

**Honest framing (per AGENTS.md §5.2 / party-twenty-spirit §3.5):**
The launch plan is a **PROJECTED @ R19** artifact, NOT a MEASURED
number — the actual retune timings will be MEASURED only once the
GPU launch completes.

---

## What ships

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py` | CLI wrapper + `LaunchPlan` dataclass + `build_launch_plan` + `dump_launch_plan_json` + `launch_gate_check` + `cpu_smoke_gpu_wrapper` + `main()` |
| `molmetal/tests/test_wf335p2_sm120_gpu_wrapper.py` | 15 NEW tests (CPU-only; 9 init + 6 CLI) |
| `molmetal/reports/wf335p2_sm120_triton_retune_gpu/launch_plan_per_kernel.json` | per-kernel launch plan (24 kernels × 117 configs) |
| `molmetal/reports/wf335p2_sm120_triton_retune_gpu/launch_plan_generic.json` | generic-grid launch plan (24 kernels × 288 configs) |

---

## Wrapper API surface

| Name | Type | Purpose |
|---|---|---|
| `LaunchPlan` | `@dataclasses.dataclass(frozen=True)` | Frozen plan: use_per_kernel + selected_kernels + per_kernel_configs + total_configs_to_sweep + estimated_kernel_launches + audit_path + audit_n_kernels + 3 prereqs |
| `build_launch_plan(audit_path, *, use_per_kernel, kernels)` | function | Compose a `LaunchPlan` from the Wf-326E audit + Wf-333-T1 per-kernel registry. `use_per_kernel=True` (RECOMMENDED per Wf-326E-B) uses the Wf-333-T1 presets; `False` uses the Wf-331-X3 generic grid. `kernels` is an optional subset (audit-ordered) |
| `dump_launch_plan_json(plan, output_path)` | function | Emit the plan as JSON with a top-level metadata block (`extends=[wf331x3_sm120_retune.py, wf333t1_sm120_autotune_extended.py]`, `gpu_retune_deferred=True`) |
| `launch_gate_check()` | function | CPU stub that returns `gate_passed=False` + 3 prereqs + per-prereq `how_to_resolve` commands. Does NOT auto-probe VRAM / pre-flight — operator must run them manually |
| `cpu_smoke_gpu_wrapper(output_dir)` | function | CPU-only correctness test: plan composition, JSON dump, ~3x Wf-326E-B ratio band, launch gate refusal |
| `main(argv)` | function | argparse entry-point (5 flags: `--smoke` / `--dump-launch-plan-only` / `--per-kernel` / `--kernels <csv>` / `--launch` + `--launch-prereqs-confirmed`) |

### CLI flag matrix

| Flag | Effect | Exit |
|---|---|---|
| `--smoke` | CPU smoke + write JSON dumps to `--output-dir` | 0 (PASS) / 1 (FAIL) |
| `--dump-launch-plan-only` | Emit the launch plan JSON + summary | 0 (PASS) / 1 (ERROR) |
| `--per-kernel` | Use the Wf-333-T1 per-kernel presets (default behavior; explicit flag for clarity) | (with above) |
| `--kernels <csv>` | Restrict the plan to a comma-separated subset of audit kernel names; preserves audit order | (with above) |
| `--launch` | Promote to GPU launch (REQUIRES `--launch-prereqs-confirmed`; without it prints the launch-gate checklist) | 0 (accepted) / 2 (refused, no prereq) |
| `--launch-prereqs-confirmed` | Operator's explicit ack that VRAM + pre-flight + config-aligned are done. Does NOT itself perform those checks. | (with `--launch`) |
| `--output-dir <path>` | Where to write the launch plan JSON (default: `molmetal/reports/wf335p2_sm120_triton_retune_gpu`) | (with all) |

### Plan composition (default + subset)

| Plan | use_per_kernel | Kernels | Configs each | Total | Estimated launches |
|---|---|---:|---:|---:|---:|
| Default (24 kernels) | True | 24 | varies (per-kernel) | 117 | 585 (× 5 noise avg) |
| Generic grid (24 kernels) | False | 24 | 12 (Wf-331-X3) | 288 | 1440 (× 5 noise avg) |
| Subset `fused_silu_mlp,matmul_fn` | True | 2 | 6 each (compute_bound) | 12 | 60 |

Per-kernel / generic speedup ratio: **288 / 117 = 2.46×**, within
the Wf-326E-B projection band [2.0, 3.5] — agrees with the "~3x"
follow-up recommendation.

### Launch-gate refusal (CPU stub)

`launch_gate_check()` always returns:

```json
{
  "gate_passed": false,
  "prereqs_required": [
    "nvidia-smi shows >= 16 GB free VRAM on RTX 5090 ... check_blackwell_env.py",
    "python molmetal/scripts/pre_flight_check.py ... Pre-flight v2 25/25 PASS gate per TODO-207",
    "Hyperparameters ... encoded in config files citing peer papers"
  ],
  "how_to_resolve": {
    "vram_check": "python molmetal/scripts/check_blackwell_env.py molmetal/reports/wf335p2_sm120_triton_retune_gpu/gpu_env_check.json",
    "pre_flight": "python molmetal/scripts/pre_flight_check.py --strict",
    "config_aligned": "Verify num_warps/num_stages/batch_size/hidden/depth match peer-paper config in molmetal/scripts/train_*.py argparse defaults."
  },
  "note": "Wrapper does NOT auto-probe VRAM or run pre-flight. Operator must complete all 3 prereqs manually before re-running with --launch (the wrapper will refuse to launch without --launch-prereqs-confirmed)."
}
```

This refusal is the **CPU-only honest-negative framing** for the
wrapper: it documents the 3 prereqs but does NOT silently bypass
them.  The actual GPU launch only happens after the operator runs
the 3 prereqs + re-invokes with `--launch --launch-prereqs-confirmed`.

---

## CLI usage

```bash
# Default: dump per-kernel launch plan JSON (no GPU launch).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \
    --output-dir molmetal/reports/wf335p2_sm120_triton_retune_gpu

# CPU smoke (validates the wrapper — no GPU launch).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py --smoke

# Restrict to a subset of kernels (comma-separated names).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \
    --per-kernel --kernels fused_silu_mlp,matmul_fn \
    --dump-launch-plan-only

# Emit the launch plan JSON only (skip the smoke).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \
    --dump-launch-plan-only

# Promote to an actual GPU launch — REQUIRES 3 prereqs:
#   1. nvidia-smi VRAM-budget check (per AGENTS.md §5.1)
#   2. python molmetal/scripts/pre_flight_check.py PASS
#   3. config-aligned hyperparameters (cite peer papers)
# Without --launch-prereqs-confirmed the wrapper prints the gate
# checklist and exits 2 (refuses to launch).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py \
    --launch --launch-prereqs-confirmed
```

Sample `--smoke` output (2026-09-29):

```json
{
  "smoke_passed": true,
  "n_kernels_audit": 24,
  "n_selected_kernels": 24,
  "per_kernel_total_configs": 117,
  "generic_total_configs": 288,
  "per_kernel_over_generic_speedup": 2.4615,
  "subset_total_configs": 12,
  "launch_gate_passed": false,
  "launch_plan_per_kernel_path": "/home/hugo/codes/mol-metal/molmetal/reports/wf335p2_sm120_triton_retune_gpu/launch_plan_per_kernel.json",
  "launch_plan_generic_path": "/home/hugo/codes/mol-metal/molmetal/reports/wf335p2_sm120_triton_retune_gpu/launch_plan_generic.json",
  "cli_results": {
    "dump_launch_plan_only": {"argv": ["--dump-launch-plan-only"], "returncode": 0},
    "per_kernel": {"argv": ["--per-kernel", "--dump-launch-plan-only"], "returncode": 0},
    "kernels_subset": {"argv": ["--per-kernel", "--kernels", "fused_silu_mlp,matmul_fn", "--dump-launch-plan-only"], "returncode": 0}
  },
  "elapsed_seconds": 0.0051
}
```

---

## Tests (15 NEW)

| # | Group | Test | Verifies |
|---|---|---|---|
| 1 | A | `TestWrapperInitialization::test_01_default_plan_uses_24_kernels_and_per_kernel_presets` | audit=24, selected=24, use_per_kernel=True, gpu_retune_deferred=True, len(prereqs)==3 |
| 2 | A | `test_02_per_kernel_total_configs_is_117` | per-kernel plan sweeps 117 configs (5*3 + 13*6 + 3*6 + 3*2); estimated_kernel_launches == 117*5 |
| 3 | A | `test_03_generic_plan_sweeps_24_times_12_configs` | generic plan sweeps 24*12=288 configs; estimated_kernel_launches == 288*5 |
| 4 | A | `test_04_per_kernel_strictly_smaller_than_generic_with_3x_band` | per-kernel < generic; speedup ratio in [2.0, 3.5] band (Wf-326E-B projection) |
| 5 | A | `test_05_kernel_subset_preserves_audit_order` | subset plan preserves audit order (not random) |
| 6 | A | `test_06_unknown_kernel_raises_value_error` | unknown kernel name raises ValueError (not silent skip) |
| 7 | A | `test_07_every_per_kernel_config_is_sm120_safe` | every per-kernel (num_warps, num_stages) passes `is_sm120_config_safe` |
| 8 | A | `test_08_plan_dump_emits_metadata_and_per_kernel_configs` | JSON dump has metadata.extends + plan fields + per-kernel registry |
| 9 | A | `test_09_launch_gate_always_refuses_with_3_prereqs` | launch_gate_check returns gate_passed=False + 3 non-trivial prereq strings |
| 10 | B | `TestWrapperCliArgs::test_10_smoke_flag_returns_zero_on_pass` | `--smoke` exits 0 + emits smoke JSON (smoke_passed=True, per_kernel_total_configs=117) |
| 11 | B | `test_11_dump_launch_plan_only_emits_per_kernel_plan` | `--dump-launch-plan-only` exits 0 + emits plan JSON (use_per_kernel=True, 117 configs, 3 prereqs) |
| 12 | B | `test_12_per_kernel_flag_is_default` | `--per-kernel --kernels fused_silu_mlp,matmul_fn` restricts to 2 kernels × 6 configs each = 12 |
| 13 | B | `test_13_launch_without_prereq_confirmed_refuses_with_exit_2` | `--launch` (no prereq) exits 2 + emits gate JSON to stderr |
| 14 | B | `test_14_launch_with_prereq_confirmed_emits_launch_accepted` | `--launch --launch-prereqs-confirmed` exits 0 + launch_accepted=True (note defers actual retune to follow-up) |
| 15 | B | `test_15_unknown_kernel_via_cli_returns_exit_1` | `--kernels not_a_real_kernel_xyz` exits 1 (CLI-side validation) |

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \
        python -m pytest molmetal/tests/test_wf335p2_sm120_gpu_wrapper.py -v

Result: **15 passed in 0.36s** (2026-09-29).

Wf-331-X3 regression check::

    PYTHONPATH=/home/hugo/codes/mol-metal \
        python -m pytest molmetal/tests/test_wf331x3_sm120_retune.py -v

Result: **4 passed in 0.32s** (2026-09-29) — the Wf-331-X3 ship
surface is unaffected by the Wf-335-P2 wrapper (the wrapper imports
from `wf331x3_sm120_retune` rather than redefining the shared
surface).

Wf-333-T1 regression check::

    PYTHONPATH=/home/hugo/codes/mol-metal \
        python -m pytest molmetal/tests/test_wf333t1_sm120_autotune.py -v

Result: **4 passed in 0.24s** (2026-09-29) — the Wf-333-T1 ship
surface is unaffected by the Wf-335-P2 wrapper.

---

## Honest framing per AGENTS.md §5.2 / party-twenty-spirit §3.5

| Claim | Tag | Source |
|---|---|---|
| 24 Triton kernels inventoried | **MEASURED** | `molmetal/reports/wf326e_triton_audit/audit.json` `triton_kernels_inventory.total_kernels == 24` |
| 117 per-kernel configs (5*3 + 13*6 + 3*6 + 3*2) | **MEASURED** | `sum(len(PER_KERNEL_CONFIGS[k]))` at module load (Wf-333-T1) |
| 288 worst-case configs (24 × 12 generic grid) | **MEASURED** | `24 * len(SM120_AUTOTUNE_CONFIGS)` (Wf-331-X3) |
| 2.46× per-kernel/generic speedup ratio | **MEASURED** | `288 / 117` ratio computed at smoke time |
| Speedup ratio in [2.0, 3.5] band | **PROJECTED** | matches Wf-326E-B follow-up recommendation "~3x" |
| Launch plan JSON emitted | **MEASURED** | 2 JSON files written (per_kernel + generic) under `molmetal/reports/wf335p2_sm120_triton_retune_gpu/` |
| 15 wrapper tests PASS | **MEASURED** | `pytest test_wf335p2_sm120_gpu_wrapper.py -v` exits 0 |
| 3 launch prereqs documented | **MEASURED** | `LAUNCH_PREREQS` tuple in `wf335p2_sm120_triton_retune_gpu.py` |
| Launch gate refuses without `--launch-prereqs-confirmed` | **MEASURED** | test_13 verifies exit 2 + gate JSON |
| `--launch --launch-prereqs-confirmed` accepts but defers actual retune to follow-up | **MEASURED** | test_14 verifies exit 0 + note explicitly defers execution |
| Full GPU retune DEFERRED | **HONEST-NEGATIVE / DEFERRED** | AGENTS.md §5.1 + Wf-333-T1 §3 + this doc |
| 1.5-3× aggregate speedup once GPU retune + per-kernel configs land | **PROJECTED** | `audit.audit_summary.projected_speedup_with_retune_and_new_kernels` |

### Why this is honest-negative / deferred, not a ship

The wrapper provides the **CPU-side CLI surface + launch-plan JSON**
but does **NOT** ship the GPU retune itself.  The full retune
requires:

1. **VRAM budget check** — confirm free VRAM on RTX 5090 (32 GB) per
   AGENTS.md §5.1, via `molmetal/scripts/check_blackwell_env.py`.
2. **Pre-flight PASS** — `python molmetal/scripts/pre_flight_check.py`
   (TODO-207) per AGENTS.md §4.
3. **Live per-kernel grid sweep** — autotune over the 117 per-kernel
   configs (vs 288 worst case) on real Blackwell workloads
   (Wasserstein loss / D-MPNN MLP / fused norm).
4. **Wired default flip** — change `TRITON_USE_FUSED` default to `1`
   after the retune confirms 1.5-3× aggregate speedup.

Until those four steps complete, the 24-kernel library remains
default-OFF (per the audit's "PyTorch is now the default" state) —
this wrapper does **NOT** flip the default.  The wrapper's
`launch_accepted=True` exit is an honest acknowledgment that the
prereqs were confirmed by the operator, NOT a claim that the retune
actually ran.

---

## Cross-references

| Doc | Purpose |
|---|---|
| `molmetal/reports/wf326e_triton_audit/audit.json` | Wf-326E source-of-truth (24 kernels, dispatch status, projected_speedup) |
| `molmetal/tests/test_wf326e_triton_audit.py` | Wf-326E regression tests (8 tests) |
| `molmetal/scripts/wf331x3_sm120_retune.py` | Wf-331-X3 CPU scaffold (this wrapper imports `SM120_AUTOTUNE_CONFIGS` + `is_sm120_config_safe` from there) |
| `molmetal/tests/test_wf331x3_sm120_retune.py` | Wf-331-X3 tests (4 tests, all pass after Wf-335-P2) |
| `TODO/completed/331x3_sm120_retune_cpu_scaffold.md` | Wf-331-X3 ship doc |
| `molmetal/scripts/wf333t1_sm120_autotune_extended.py` | Wf-333-T1 per-kernel extension (this wrapper imports `PER_KERNEL_CONFIGS` + `get_kernel_configs` from there) |
| `molmetal/tests/test_wf333t1_sm120_autotune.py` | Wf-333-T1 tests (4 tests, all pass after Wf-335-P2) |
| `TODO/completed/333t1_sm120_autotune_extended.md` | Wf-333-T1 ship doc |
| `molmetal/scripts/wf335p2_sm120_triton_retune_gpu.py` | Wf-335-P2 ship surface (this wrapper) |
| `molmetal/tests/test_wf335p2_sm120_gpu_wrapper.py` | Wf-335-P2 tests (15 tests, all pass) |
| `molmetal/reports/wf335p2_sm120_triton_retune_gpu/launch_plan_per_kernel.json` | per-kernel launch plan (24 × 117) |
| `molmetal/reports/wf335p2_sm120_triton_retune_gpu/launch_plan_generic.json` | generic-grid launch plan (24 × 288) |
| `molmetal/scripts/check_blackwell_env.py` | VRAM probe (REQUIRED before `--launch`) |
| `molmetal/scripts/pre_flight_check.py` | Pre-flight v2 25/25 PASS gate (REQUIRED before `--launch`) |
| `AGENTS.md` §4 + §5.1 | Critical workflows + standing GPU authorization |
| `AGENTS.md` §5.2 | honest framing (PROJECTED vs MEASURED) |
| `party-twenty-spirit` §3.5 | honest-negative framing rules |
| `.claude/skills/mol-metal-gpu-execution` §3 / §4 / §6 | venv setup + standard train contract + 5-seed protocol |

---

**END of Wf-335-P2 ship surface.** Owner: Hugo Lee. Last update
2026-09-29.
