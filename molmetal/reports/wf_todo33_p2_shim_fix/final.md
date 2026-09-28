# TODO-33 P2 — ODE Solver Shim Fix (ModelWrapper unwrap)

**Date**: 2026-09-25
**Workflow**: ad-hoc (5 shim file edits + 3-seed verify)
**Inputs**: `molmetal/data/metallo_drugs_500_train.csv` (449 SMILES, 358 3D-embedded)
**Config**: steps=1000, batch=4, h=128, n_layers=3, bond_head=learned, joint_train=True, lr_warmup_steps=200, MOLMETAL_CFM_AMP_DTYPE=bfloat16

## Per-seed results

| seed | decode@250 | decode@500 | decode@750 | decode@1000 | max_decode | disconnected | wall_s |
|---|---|---|---|---|---|---|---|
| 0 | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 1.000 | 0 | 82.1 |
| 1 | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 1.000 | 0 | 88.5 |
| 2 | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 1.000 | 0 | 98.5 |

**Seed-mean ± std**: decode_ratio = **1.000 ± 0.000** at all 4 probes × 3 seeds = **12/12 successes**.

## Comparison

| Metric | BF16 R16 baseline | workflow E (F1+F2+F4) | **THIS (shim fix)** |
|---|---|---|---|
| atom_loss step 1 | Inf | Inf (FP32) / Inf (BF16) | **2.996** |
| atom_loss last | Inf | 1.227 (FP32+F4) | **3.81** |
| cfm_loss last | 252330 | 31.2 | **19.3** |
| decode_ratio | 0/192 (probe crash) | 0/192 (TypeError) | **48/48 (1.000)** |
| disconnected | 16/16 | n/a (probe never ran) | **0/48** |

## The shim fix (1 line defensive try/except + companion edits)

### Primary fix — solver shim defensive fallback

`molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:152-165`

```python
if self._sampler is not None and not (
    step_size is not None or time_grid is not None
):
    try:
        return self._sampler.sample(
            x_init, n_steps=n_steps, method=method, **kwargs,
        )
    except TypeError:
        # Upstream rejected the wrapper at sample-time (isinstance check
        # on VelocityNet); fall through to inline Euler/midpoint.
        self._sampler = None
```

**Why this works**: the upstream `FlowMatchingSampler.__init__` does NOT enforce `isinstance(model, VelocityNet)` — that check is deferred to `.sample()`. The shim's `_UPSTREAM_AVAILABLE` probe (lines 55-64) succeeds because `_ProbeModel` constructs OK, but the real `ModelWrapper` is rejected at sample-time. Wrapping the upstream `.sample()` call in try/except TypeError and falling back to the inline integration loop (already implemented at lines 167-187) makes the shim robust to this race.

### Companion fixes (required for inline path to actually run)

| File | Change | Reason |
|---|---|---|
| `_reference_loader.py:79-94` | Auto-detect flat clone root: `if (ref / "__init__.py").is_file() and ref.name == "flow_matching": root = ref` | The adapter's default passes `/home/hugo/codes/mol-metal/flow_matching` (the clone root), not its parent. The legacy code appended another `flow_matching/` segment, producing the non-existent `/home/hugo/codes/mol-metal/flow_matching/flow_matching/__init__.py`. |
| `_reference/__init__.py:34-56` | Hybrid `__path__ = [str(_ref_clone), _local_shim_dir]` + sys.path-injection proxy for `upstream_flow_matching` | The shim files (`solver/`, `path/`, `utils/`) live in the local dir; upstream modules (`sampler`, `interpolation`, `sampler_config`) live in the clone. Both must resolve under the same private namespace. |
| `_reference/path/__init__.py:95-96` | `from .upstream_flow_matching.interpolation import interpolate, target_velocity` | The bare `.upstream_flow_matching` import had no `interpolate` symbol (lives in `.interpolation` submodule). |
| `_reference/path/scheduler/__init__.py:9-10` | `from .upstream_flow_matching.sampler_config import SamplerConfig` | Same — `SamplerConfig` lives in `.sampler_config` submodule. |
| `_reference/solver/__init__.py:51-52` | `from .upstream_flow_matching.sampler import FlowMatchingSampler` | Same — `FlowMatchingSampler` lives in `.sampler` submodule. |
| `_reference/utils/__init__.py:35-54` | ModelWrapper tries `model(x,t)`, `model(x,t=t)`, `model(x,t=t,**extras)`, `model(x,t,**extras)` in order | The adapter wraps velocity as `partial(forward_velocity, atom_types=..., edge_index=..., pocket_embed=...)` — this makes `t` keyword-only (it's after `*` in `forward_velocity`'s signature), so the legacy `model(x, t)` positional call failed with `TypeError: got multiple values for argument 'atom_types'`. |

## Verification — 12/12 probes decode perfectly

Every probe in every seed:
- `decode_ratio = 1.000` (16/16 candidates valid)
- `disconnected = 0/16` (perfect graph connectivity)
- `mean_atoms = 8.0` (consistent with the 8-atom training set's average)
- Per-probe wall: 0.3-1.7s (inline integration is fast)

## Comparison to gate

| Gate | Threshold | Result |
|---|---|---|
| Any decode lift | ≥1/3 seeds with decode_ratio > 0 | **3/3 PASS** |
| ≥0.5 mean decode | mean across seeds > 0.5 | **1.000 PASS** |
| disconnected | disconnected < 2/16 | **0/16 PASS** |

**TODO-33 P2 — PASS.** The 0/192 decode floor is gone.

## Files changed

- `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py` (defensive try/except)
- `molmetal/adapters/flow_matching_lipman/_reference/utils/__init__.py` (ModelWrapper kw-time fallback)
- `molmetal/adapters/flow_matching_lipman/_reference_loader.py` (flat-clone auto-detect)
- `molmetal/adapters/flow_matching_lipman/_reference/__init__.py` (hybrid __path__ + upstream proxy)
- `molmetal/adapters/flow_matching_lipman/_reference/path/__init__.py` (submodule-qualified import)
- `molmetal/adapters/flow_matching_lipman/_reference/path/scheduler/__init__.py` (submodule-qualified import)

## Artifacts

- `logs/yuelbond_p1_shim_fix/seed_{0,1,2}/{final.json,final.md,ckpt.pt}` — 3 seed runs
- `logs/yuelbond_p1_shim_fix/seed_{0,1,2}.log` — full per-seed stdout
- `TODO/completed/33_r17_fixes_blackwell_loader_paths.md` — completion summary

## Next steps

- TODO-33 is now COMPLETE: P1 (F4 clamp + 20-atom vocab) + P2 (shim fix) lift both the Inf wall AND the decode=0 wall.
- 10K-step retrain can now ship on Blackwell with the locked config.
- Workflow G's `decoder-conn` phase is OBSOLETE (was attempting to fix decoder connectivity, but the real issue was upstream rejecting ModelWrapper). It will exit on its own.
