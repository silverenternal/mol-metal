# Wf-331-X3 — sm_120 Triton retune CPU scaffold

**Wf-id:** Wf-331-X3 (2026-09-29)
**Status:** SHIPPED (CPU scaffold)
**Operator:** Claude (Sonnet 4.6)
**Pre-flight:** N/A (CPU-only scaffold per AGENTS.md §5.1; no retrain / panel / ablation launch)

---

## Summary

Per the Wf-326E audit (24 Triton kernels inventoried at
`molmetal/reports/wf326e_triton_audit/audit.json`):
the 24-kernel Triton library is **default-OFF on Blackwell sm_120**
because the `AUTOTUNE_CONFIGS` in `triton_kernels/autotune.py` L49-59
was tuned for `gfx1101` (RDNA3 iGPU), where
`num_warps in {2, 4, 8} × num_stages in {2, 3, 4}` is the only
legitimate grid.  On Blackwell the same configs run 3-11× slower
than PyTorch (per `config.py` L107), so PyTorch is the default
since 2026-09-27.

Wf-331-X3 ships a **CPU-only scaffold** for the Wf-326E follow-up
recommendation A: "sweep AUTOTUNE_CONFIGS over
`num_warps {4, 8, 16}, num_stages {2, 3, 4}` on a single RTX 5090;
expected 1-2 day implementation + 1 day validation".

**Full GPU retune deferred** per AGENTS.md §5.1 (standing GPU
authorization + VRAM budget + pre-flight PASS required before any
GPU launch).  This scaffold provides the CPU-side infrastructure
so the eventual GPU retune run can pick up a known-good config grid
+ a safe gfx1101 fallback path.

---

## What ships

| Path | Purpose |
|---|---|
| `molmetal/scripts/wf331x3_sm120_retune.py` | sm_120 retune scaffold (script + library) |
| `molmetal/tests/test_wf331x3_sm120_retune.py` | 4 NEW tests (CPU-only) |
| `molmetal/reports/wf331x3_sm120_retune/sm120_autotune_configs.json` | 12-config grid dump |

---

## Scaffold API

The script exports the following CPU-friendly surface:

| Name | Type | Purpose |
|---|---|---|
| `SM120_AUTOTUNE_CONFIGS` | `list[triton.Config]` (12 entries) | sm_120 config grid (`num_warps ∈ {4, 8, 16}` + memory-bound `num_warps=2`, `num_stages ∈ {2, 3, 4}`) |
| `SM120_AUTOTUNE_CONFIGS_FALLBACK` | `list[triton.Config]` (4 entries) | smaller eval-mode grid (`num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3}`) |
| `GFX1101_AUTOTUNE_CONFIGS` | `list[triton.Config]` (9 entries) | read-through to `triton_kernels.AUTOTUNE_CONFIGS` for the fallback dispatch |
| `sm120_autotune(key=...)` | decorator | wraps `triton.autotune` with sm_120 configs as default |
| `safe_kernel_call(sm120_fn, gfx1101_fn, *args, **kwargs)` | function | try sm_120 → fallback to gfx1101 on `Exception` |
| `is_sm120_config_safe(cfg)` | function | `num_warps ∈ {2, 4, 8, 16}` ∧ `num_stages ∈ {2, 3, 4}` |
| `dump_sm120_configs_json(path)` | function | emits configs as JSON for the eventual GPU retune |
| `cpu_smoke(output_dir)` | function | CPU-only correctness test (12 configs, 4 fallback, fallback path) |

---

## CLI usage

```bash
# Default: dump configs JSON (no GPU launch).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf331x3_sm120_retune.py \
    --output-dir molmetal/reports/wf331x3_sm120_retune

# CPU smoke (validates the wrapper + fallback + is_safe).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf331x3_sm120_retune.py --smoke

# Emit configs JSON only (skip smoke).
PYTHONPATH=/home/hugo/codes/mol-metal \
    python molmetal/scripts/wf331x3_sm120_retune.py --dump-configs-only
```

Sample default-CLI output (2026-09-29):

```json
{
  "n_kernels_inventoried": 24,
  "n_sm120_configs": 12,
  "n_sm120_fallback_configs": 4,
  "configs_json_path": "/home/hugo/codes/mol-metal/molmetal/reports/wf331x3_sm120_retune/sm120_autotune_configs.json",
  "gpu_retune_deferred": true
}
```

Sample `--smoke` output (2026-09-29):

```json
{
  "smoke_passed": true,
  "n_sm120_configs": 12,
  "n_sm120_fallback_configs": 4,
  "n_kernels_inventoried": 24,
  "gpu_retune_deferred": true,
  "safe_kernel_call_fallback_ok": true,
  "autotune_decorator_callable": true,
  "configs_json_path": "...",
  "elapsed_seconds": 0.0005
}
```

---

## Tests (4 NEW)

| # | Group | Test | Verifies |
|---|---|---|---|
| 1 | A | `TestAutotuneWrapper::test_01_sm120_autotune_decorator_is_callable` | `@sm120_autotune(key=[...])` produces an `Autotuner` with `len(.configs) == len(SM120_AUTOTUNE_CONFIGS)`; sm_120 grid (12) > gfx1101 grid (9) |
| 2 | B | `TestSm120ConfigPresets::test_02_sm120_configs_match_wf326e_a_grid` | every config has `num_warps ∈ {2, 4, 8, 16}` ∧ `num_stages ∈ {2, 3, 4}`; full Wf-326E-A grid covered (3×3 + 3); `is_sm120_config_safe` rejects `num_warps=64` |
| 3 | B | `TestSm120ConfigPresets::test_03_sm120_fallback_configs_smaller_for_eval` | fallback grid = 4 entries (`num_warps ∈ {4, 8}`); fallback ⊆ full grid |
| 4 | C | `TestGfx1101Fallback::test_04_safe_kernel_call_falls_back_to_gfx1101` | sm_120 `RuntimeError` → gfx1101 fallback; happy path does NOT invoke fallback; positional + kwarg passthrough |

Run::

    PYTHONPATH=/home/hugo/codes/mol-metal \
        python -m pytest molmetal/tests/test_wf331x3_sm120_retune.py -v

Result: **4 passed in 0.18s** (2026-09-29).

---

## Honest framing per AGENTS.md §5.2 / party-twenty-spirit §3.5

| Claim | Tag | Source |
|---|---|---|
| 24 Triton kernels inventoried | **MEASURED** | `molmetal/reports/wf326e_triton_audit/audit.json` `triton_kernels_inventory.total_kernels == 24` |
| Default-OFF on sm_120 (3-11× slower than PyTorch) | **MEASURED** | `triton_kernels/config.py` L107 + `audit.json` `dispatch_status.rationale` |
| 12 sm_120 configs (3×3 + 1×3 memory-bound) | **PROJECTED** | Wf-326E follow-up recommendation A: `num_warps {4, 8, 16}` × `num_stages {2, 3, 4}` + 3 memory-bound (`num_warps=2`) |
| 4 fallback configs | **PROJECTED** | smaller eval-mode grid (4 vs 12) |
| `safe_kernel_call` fallback path | **MEASURED** | 4/4 tests pass on CPU (the test mocks a `RuntimeError`) |
| 1.5-3× aggregate speedup once GPU retune + new kernels land | **PROJECTED** | `audit.json` `audit_summary.projected_speedup_with_retune_and_new_kernels` |
| Full GPU retune deferred | **HONEST-NEGATIVE / DEFERRED** | AGENTS.md §5.1 (standing GPU authorization + VRAM + pre-flight gate required) |

### Why this is honest-negative / deferred, not a ship

The scaffold provides the **CPU-side infrastructure** (config grid +
fallback dispatch + JSON dump + 4 tests) but does **NOT** ship the
GPU retune itself.  The full retune requires:

1. **VRAM budget check** — confirm free VRAM on RTX 5090 (32 GB) per
   AGENTS.md §5.1.
2. **Pre-flight PASS** — `python molmetal/scripts/pre_flight_check.py`
   (TODO-207) per AGENTS.md §4.
3. **Live grid sweep** — autotune over the 12 configs on real
   Blackwell workloads (Wasserstein loss / D-MPNN MLP / fused norm).
4. **Wired default flip** — change `TRITON_USE_FUSED` default to `1`
   after the retune confirms 1.5-3× aggregate speedup.

Until those four steps complete, the 24-kernel library remains
default-OFF (per the audit's "PyTorch is now the default" state) —
this scaffold does **NOT** flip the default.

---

## GPU retune procedure (deferred per AGENTS.md §5.1)

When the user authorizes the GPU retune launch (and VRAM + pre-flight
PASS), the procedure is:

1. Confirm `nvidia-smi` shows ≥ 30 GB free VRAM on GPU 0 (RTX 5090).
2. Run `python molmetal/scripts/pre_flight_check.py` — expect 25/25
   PASS.
3. Launch the grid sweep (projected ~1-2 days wall time per
   Wf-326E follow-up A):

   ```bash
   PYTHONPATH=/home/hugo/codes/mol-metal \
       python molmetal/scripts/wf331x3_sm120_retune.py \
       --gpu-retune --output-dir molmetal/reports/wf331x3_sm120_retune
   ```

   (The `--gpu-retune` flag is **not** implemented yet — it would be
   added when the GPU launch is authorized.)

4. Compare retuned MEASURED speedups against the audit's PROJECTED
   1.5-3× band.  If MEASURED < PROJECTED, label honest-negative and
   re-scope (smaller grid, or skip Blackwell for this workload).
5. Wire `TRITON_USE_FUSED=1` as default after MEASURED > 1.5×
   confirms the audit's PROJECTED claim.

---

## Cross-references

| Doc | Purpose |
|---|---|
| `molmetal/reports/wf326e_triton_audit/audit.json` | Wf-326E source-of-truth (24 kernels, dispatch status, root cause) |
| `molmetal/tests/test_wf326e_triton_audit.py` | Wf-326E regression tests (8 tests) |
| `triton_kernels/autotune.py` L49-59 | existing gfx1101 `AUTOTUNE_CONFIGS` (9 configs) |
| `triton_kernels/config.py` L107 | "Default changed 2026-09-27" rationale |
| `molmetal/tests/test_wf331x3_sm120_retune.py` | Wf-331-X3 tests (4 tests) |
| `AGENTS.md` §5.1 | standing GPU authorization + VRAM + pre-flight |
| `AGENTS.md` §5.2 | honest framing (PROJECTED vs MEASURED) |

---

**END of Wf-331-X3 ship surface.** Owner: Hugo Lee. Last update
2026-09-29.
