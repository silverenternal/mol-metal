# Wf-333-T1 — sm_120 Triton autotune config presets extension

**Wf-id:** Wf-333-T1 (2026-09-29)
**Status:** SHIPPED (CPU scaffold)
**Operator:** Claude (Sonnet 4.6)
**Pre-flight:** N/A (CPU-only scaffold per AGENTS.md §5.1; no retrain / panel / ablation launch)

---

## Summary

Extends the Wf-331-X3 CPU scaffold (`molmetal/scripts/wf331x3_sm120_retune.py`,
12-config generic grid) with **per-kernel config presets** so the
autotuner sweeps a much narrower config band per kernel — per
Wf-326E follow-up recommendation B ("per-kernel config subsets cut
autotune cost by ~3x on Blackwell").

**Full GPU retune deferred** per AGENTS.md §5.1 (standing GPU
authorization + VRAM budget + pre-flight PASS required before any
GPU launch).  This extension provides the CPU-side per-kernel
infrastructure so the eventual GPU retune run can pick up a
kernel-aware config grid + the Wf-331-X3 generic fallback path.

---

## What ships

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf333t1_sm120_autotune_extended.py` | per-kernel autotune config preset extension (script + library) |
| `molmetal/tests/test_wf333t1_sm120_autotune.py` | 4 NEW tests (CPU-only) |
| `molmetal/reports/wf333t1_sm120_autotune/per_kernel_configs.json` | 24-kernel per-category registry dump |

---

## Extension API

The extension reuses the Wf-331-X3 scaffold surface via
`import wf331x3_sm120_retune as _x3` — the Wf-331-X3 constants
(`SM120_AUTOTUNE_CONFIGS`, `SM120_AUTOTUNE_CONFIGS_FALLBACK`,
`GFX1101_AUTOTUNE_CONFIGS`, `is_sm120_config_safe`,
`safe_kernel_call`, `load_audit_kernels`) are imported rather than
re-defined.  The new per-kernel surface:

| Name | Type | Purpose |
|---|---|---|
| `MEMORY_BOUND_KERNELS` | `frozenset[str]` (5 entries) | bandwidth-limited kernels (elementwise / norm / residual / dropout) |
| `COMPUTE_BOUND_KERNELS` | `frozenset[str]` (14 entries) | ILP-heavy kernels (MLP variants / matmul / batched MLP / losses) |
| `MID_BOUND_KERNELS` | `frozenset[str]` (2 entries) | balanced kernels (softmax / scatter-gather) |
| `ODE_KERNELS` | `frozenset[str]` (3 entries) | low-arithmetic-intensity (rotation / Euler / RK4) |
| `PER_KERNEL_CONFIGS` | `dict[str, list[triton.Config]]` (24 entries) | per-kernel config registry |
| `get_kernel_category(kernel_name)` | function | returns category string or `None` |
| `get_kernel_configs(kernel_name)` | function | returns per-kernel configs (falls back to Wf-331-X3 generic grid for unknown kernels) |
| `sm120_autotune_per_kernel(kernel_name, *, key)` | decorator | wraps `triton.autotune` with per-kernel configs |
| `dump_per_kernel_configs_json(output_path)` | function | emits per-kernel registry as JSON |
| `cpu_smoke_extended(output_dir)` | function | CPU-only correctness test for the extension |

### Per-category config counts

| Category | Kernels | num_warps | num_stages | Configs each | Total |
|---|---:|---|---|---:|---:|
| memory_bound | 5 | `(2,)` | `(2, 3, 4)` | 3 | 15 |
| compute_bound | 14 | `(8, 16)` | `(2, 3, 4)` | 6 | 84 |
| mid_bound | 2 | `(4, 8)` | `(2, 3, 4)` | 6 | 12 |
| ode | 3 | `(4,)` | `(2, 3)` | 2 | 6 |
| **TOTAL** | **24** | — | — | — | **117** |

Worst case if every kernel swept the Wf-331-X3 generic grid:
`24 × 12 = 288`.  The per-kernel presets cut the autotune cost by
`(288 - 117) / 288 ≈ 59%` — agrees with the Wf-326E follow-up B
~3× projection (`288 / 117 ≈ 2.5×`).

---

## CLI usage

```bash
# Default: dump per-kernel registry JSON (no GPU launch).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf333t1_sm120_autotune_extended.py \
    --output-dir molmetal/reports/wf333t1_sm120_autotune

# CPU smoke (validates the per-kernel extension).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf333t1_sm120_autotune_extended.py --smoke

# Emit registry JSON only (skip smoke).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf333t1_sm120_autotune_extended.py --dump-configs-only
```

Sample `--smoke` output (2026-09-29):

```json
{
  "smoke_passed": true,
  "n_kernels": 24,
  "n_categories": 4,
  "n_memory_bound": 5,
  "n_compute_bound": 14,
  "n_mid_bound": 2,
  "n_ode": 3,
  "n_total_per_kernel_configs": 117,
  "fallback_to_generic_grid_ok": true,
  "safe_kernel_call_fallback_ok": true,
  "per_kernel_decorator_callable": true,
  "configs_json_path": "/home/hugo/codes/mol-metal/molmetal/reports/wf333t1_sm120_autotune/per_kernel_configs.json",
  "elapsed_seconds": 0.0012
}
```

---

## Tests (4 NEW)

| # | Group | Test | Verifies |
|---|---|---|---|
| 1 | A | `TestPerKernelAutotunePresets::test_01_per_kernel_presets_match_category_assignments` | audit has 24 kernels; every audit kernel has a `PER_KERNEL_CONFIGS` entry; category sizes are 5 + 14 + 2 + 3 = 24; every config is `is_sm120_config_safe`; per-category config counts match the expected per-category grids |
| 2 | B | `TestSm120FallbackToGeneric::test_02_unknown_kernel_falls_back_to_generic_grid` | unknown kernel name → 12-config Wf-331-X3 grid (set equality); `get_kernel_category` returns `None`; every fallback config passes `is_sm120_config_safe` |
| 3 | C | `TestPerKernelConfigsSubset::test_03_per_kernel_configs_are_subset_of_generic_grid` | every per-kernel `(num_warps, num_stages)` pair is in the Wf-331-X3 grid; per-kernel total (117) is strictly less than worst case (24×12=288); union of per-category grids is non-empty and is a subset of the Wf-331-X3 grid |
| 4 | D | `TestCpuCorrectnessExtended::test_04_cpu_smoke_validates_full_extended_surface` | `cpu_smoke_extended` returns `{"smoke_passed": True, ...}` with all expected fields; JSON dump succeeds; `gpu_retune_deferred == True`; registry has 24 entries |

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \
        python -m pytest molmetal/tests/test_wf333t1_sm120_autotune.py -v

Result: **4 passed in 0.24s** (2026-09-29).

Wf-331-X3 regression check::

    PYTHONPATH=/home/hugo/codes/mol-metal \
        python -m pytest molmetal/tests/test_wf331x3_sm120_retune.py -v

Result: **4 passed in 0.32s** (2026-09-29) — the Wf-331-X3 ship
surface is unaffected by the Wf-333-T1 extension (the extension
imports from `wf331x3_sm120_retune` rather than redefining the
shared surface).

---

## Honest framing per AGENTS.md §5.2 / party-twenty-spirit §3.5

| Claim | Tag | Source |
|---|---|---|
| 24 Triton kernels inventoried | **MEASURED** | `molmetal/reports/wf326e_triton_audit/audit.json` `triton_kernels_inventory.total_kernels == 24` |
| Per-kernel category assignment (5 / 14 / 2 / 3) | **PROJECTED** | derived from `audit.triton_kernels_inventory.kernels[].category` (elementwise/norm/MLP/etc.) + Wf-326E follow-up recommendation B |
| 117 total per-kernel configs (vs 288 worst case) | **MEASURED** | `len(sum(PER_KERNEL_CONFIGS.values()))` computed at module load |
| ~59% autotune cost reduction | **PROJECTED** | `(288 - 117) / 288` ratio; matches Wf-326E follow-up B ~3× projection (`288 / 117 ≈ 2.5×`) |
| `safe_kernel_call` fallback path | **MEASURED** | 4/4 Wf-333-T1 tests pass on CPU (the test mocks a `RuntimeError`); Wf-331-X3 still passes 4/4 |
| `sm120_autotune_per_kernel` decorator callable | **MEASURED** | Wf-333-T1 test 01c + test 04c pass on CPU |
| Per-kernel JSON dump | **MEASURED** | `molmetal/reports/wf333t1_sm120_autotune/per_kernel_configs.json` written; metadata `n_kernels=24, n_categories=4` |
| 1.5-3× aggregate speedup once GPU retune + per-kernel configs land | **PROJECTED** | `audit.audit_summary.projected_speedup_with_retune_and_new_kernels` |
| Full GPU retune deferred | **HONEST-NEGATIVE / DEFERRED** | AGENTS.md §5.1 (standing GPU authorization + VRAM + pre-flight gate required) |

### Why this is honest-negative / deferred, not a ship

The extension provides the **CPU-side per-kernel infrastructure**
(category assignments + per-kernel config registry + per-kernel
decorator + JSON dump + 4 tests) but does **NOT** ship the GPU
retune itself.  The full retune requires:

1. **VRAM budget check** — confirm free VRAM on RTX 5090 (32 GB) per
   AGENTS.md §5.1.
2. **Pre-flight PASS** — `python molmetal/scripts/pre_flight_check.py`
   (TODO-207) per AGENTS.md §4.
3. **Live per-kernel grid sweep** — autotune over the 117 per-kernel
   configs (vs 288 worst case) on real Blackwell workloads
   (Wasserstein loss / D-MPNN MLP / fused norm).
4. **Wired default flip** — change `TRITON_USE_FUSED` default to `1`
   after the retune confirms 1.5-3× aggregate speedup.

Until those four steps complete, the 24-kernel library remains
default-OFF (per the audit's "PyTorch is now the default" state) —
this extension does **NOT** flip the default.

---

## Cross-references

| Doc | Purpose |
|---|---|
| `molmetal/reports/wf326e_triton_audit/audit.json` | Wf-326E source-of-truth (24 kernels, dispatch status, per-kernel category) |
| `molmetal/tests/test_wf326e_triton_audit.py` | Wf-326E regression tests (8 tests) |
| `molmetal/scripts/wf331x3_sm120_retune.py` | Wf-331-X3 ship surface — extended (NOT modified) by Wf-333-T1 |
| `molmetal/tests/test_wf331x3_sm120_retune.py` | Wf-331-X3 tests (4 tests, all pass after Wf-333-T1) |
| `TODO/completed/331x3_sm120_retune_cpu_scaffold.md` | Wf-331-X3 ship doc |
| `molmetal/scripts/wf333t1_sm120_autotune_extended.py` | Wf-333-T1 ship surface (this extension) |
| `molmetal/tests/test_wf333t1_sm120_autotune.py` | Wf-333-T1 tests (4 tests, all pass) |
| `molmetal/reports/wf333t1_sm120_autotune/per_kernel_configs.json` | Wf-333-T1 24-kernel registry dump |
| `AGENTS.md` §5.1 | standing GPU authorization + VRAM + pre-flight |
| `AGENTS.md` §5.2 | honest framing (PROJECTED vs MEASURED) |

---

**END of Wf-333-T1 ship surface.** Owner: Hugo Lee. Last update
2026-09-29.
