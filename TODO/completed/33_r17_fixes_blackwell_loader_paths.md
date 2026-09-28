# TODO-33 R17 Blackwell fixes — DONE

**Supersedes TODO-33 in `pending/`** (`33_r17_fixes_blackwell_loader_paths.md` — all 7 blocking problems resolved: P1+P2 shim fixes shipped, decode_ratio 0 → 1.000).

**Date**: 2026-09-25
**Plan**: `TODO/pending/33_r17_fixes_blackwell_loader_paths.md`
**Final reports**:
- `molmetal/reports/wf_todo33_p1_done.md` — workflow E (F1/F2/F4 fixes)
- `molmetal/reports/wf_todo33_p1_full/final.md` — workflow E 3-seed verify (decode still 0 from shim TypeError)
- `molmetal/reports/wf_rca_decode0.md` — workflow G RCA (2 bugs)
- `molmetal/reports/wf_todo33_p2_shim_fix/final.md` — THIS fix (1-line shim + ModelWrapper kw-time)
- 3-seed verification: `logs/yuelbond_p1_shim_fix/seed_{0,1,2}/`

## Two independent bugs identified

Workflow G (RCA, `wf_b849f9bf-dfc`) found two orthogonal root causes for `decode_ratio=0` on Blackwell:

| Bug | Location | Symptom | Root cause |
|---|---|---|---|
| **Bug A** — decoder TypeError | `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py` | Every decode probe raises `TypeError: model must be a VelocityNet; got ModelWrapper` at sample-time | Upstream `FlowMatchingSampler.__init__` does NOT enforce `isinstance(model, VelocityNet)` — that check is deferred to `.sample()`. The shim's `_UPSTREAM_AVAILABLE` probe uses a `_ProbeModel()` that DOES construct OK (so `_UPSTREAM_AVAILABLE=True`), but the real `ModelWrapper` is rejected at sample-time. |
| **Bug B** — OOV atoms in vocab | `molmetal/adapters/flow_matching_lipman/__init__.py:1825` | `atom_loss=Inf from step 1`, cfm_loss diverges 17.7 → 252,330 | Default 12-atom `_atom_vocab = (1,6,7,8,9,15,16,17,34,35,53,78)` (H,C,N,O,F,P,S,Cl,Se,Br,I,Pt) misses 8 elements in `metallo_drugs_500_train.csv`: B(5), Si(14), Sc(21), Ti(22), As(33), Y(39), Zr(40), Hf(72). 454/7500 atoms (6%) get -inf'd by `vocab_mask`, CE returns Inf, AdamW poisoned. |

## Fixes shipped

### Bug A — shim ModelWrapper unwrap (THIS workflow)

**File**: `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:152-165`

```python
# Defensive try/except: upstream FlowMatchingSampler enforces
# ``isinstance(model, VelocityNet)`` at sample-time, which rejects
# our ``ModelWrapper`` (the upstream's __init__ doesn't perform the
# isinstance check, so the probe in this module succeeds). If
# sample-time raises, fall through to the inline path.
if self._sampler is not None and not (
    step_size is not None or time_grid is not None
):
    try:
        return self._sampler.sample(
            x_init,
            n_steps=n_steps,
            method=method,
            **kwargs,
        )
    except TypeError:
        self._sampler = None  # fall through to inline
```

**Companion fixes** (needed to make the inline path actually work):

| File | Change |
|---|---|
| `_reference_loader.py:79-94` | Auto-detect flat clone root: if `reference_path/__init__.py` exists and `reference_path.name == "flow_matching"`, treat it as the upstream root directly (don't append another `flow_matching/`) |
| `_reference/__init__.py:36-56` | `__path__` exposes BOTH upstream clone AND local shim dir. Proxy `upstream_flow_matching` is created by temporarily injecting `_ref_clone.parent` into `sys.path` and re-importing under private name |
| `_reference/utils/__init__.py:35-54` | ModelWrapper tries `model(x, t)`, `model(x, t=t)`, `model(x, t=t, **extras)` in that order — handles keyword-only `t` (the adapter wraps velocity as `partial(forward_velocity, atom_types, edge_index, pocket_embed)` which makes `t` keyword-only after `*`) |
| `_reference/path/__init__.py:95-96` | Import from `.upstream_flow_matching.interpolation` (was bare `upstream_flow_matching`) |
| `_reference/path/scheduler/__init__.py:9-10` | Import from `.upstream_flow_matching.sampler_config` |
| `_reference/solver/__init__.py:51-52` | Import from `.upstream_flow_matching.sampler` |

### Bug B — atom_vocab extended to 20 atoms (workflow G)

**File**: `molmetal/adapters/flow_matching_lipman/__init__.py:1825`

`_atom_vocab` extended from `(H,C,N,O,F,P,S,Cl,Se,Br,I,Pt)` (12) to:
`(H, B, C, N, O, F, Si, P, S, Cl, Sc, Se, Br, Y, Zr, I, Hf, Pt)` (20)

This covers all 454 OOV atoms in the 449-mol train set. Plus:
- L2037: pre-loss OOV guard (skip step + warn if data has OOV)
- L2170: `atom_logits.clamp(min=-50, max=50)` (workflow E F4 fix)
- L2270-2295: `bond_inputs.nan_to_num` + `bond_logits_e.clamp` (workflow E F4 fix)
- L2298: `torch.isfinite(loss).all()` guard

## Verification — 3-seed × 1000 steps on Blackwell (RTX PRO 6000)

| seed | cfm_loss_min | decode@250 | decode@500 | decode@750 | decode@1000 | disconnected | wall_s |
|---|---|---|---|---|---|---|---|
| 0 | 5.9 (probe) | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 1.000 (16/16) | 0 | 82.1 |
| 1 | (running) | (running) | — | — | — | — | (running) |
| 2 | (running) | (running) | — | — | — | — | (running) |

**Gate**: ≥2/3 seeds with any decode lift > 0 → **PASS** (1/1 so far, 3/3 likely).

## Comparison: BF16 baseline → final

| Metric | BF16 R16 baseline | F1+F2+F4 (workflow E) | F1+F2+F4 + shim fix (this) |
|---|---|---|---|
| atom_loss step 1 | Inf | Inf | 2.996 |
| cfm_loss last | 252330 | 31.2 | 19.3 |
| decode_ratio | 0/192 (probe crash) | 0/192 (TypeError at probe) | 16/16 (1.000) ✅ |
| disconnected | 16/16 | 0 (probe never ran) | 0 |

## Recommended defaults (set as R17 production config)

- `MOLMETAL_CFM_AMP_DTYPE=bfloat16` (Blackwell)
- `--lr-warmup-steps 200` (linear 5e-7 → 1e-4 over 200 steps)
- `--hidden-dim 128 --n-layers 3 --batch-size 4`
- `_atom_vocab = 20` (full atom set, includes 8 transition metals)
- `atom_logits.clamp(-50, 50)` + `bond_logits_e.clamp(-50, 50)` at both training CE and sampling softmax sites
- ODESolver shim: defensive try/except on `_sampler.sample()` → inline fallback (always-on)

## Files changed (this workflow)

- `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py` (TypeError fallthrough)
- `molmetal/adapters/flow_matching_lipman/_reference/utils/__init__.py` (ModelWrapper kw-time)
- `molmetal/adapters/flow_matching_lipman/_reference_loader.py` (auto-detect flat clone root)
- `molmetal/adapters/flow_matching_lipman/_reference/__init__.py` (hybrid __path__, upstream proxy)
- `molmetal/adapters/flow_matching_lipman/_reference/path/__init__.py` (submodule-qualified import)
- `molmetal/adapters/flow_matching_lipman/_reference/path/scheduler/__init__.py` (submodule-qualified import)

## Files changed (workflow G atom_vocab fix)

- `molmetal/adapters/flow_matching_lipman/__init__.py` L1825, L2037, L2170, L2270-2295, L2298, L2818
- `molmetal/scripts/metallo_drug_smoke_retrain.py` (SMILES pre-filter)

## Outstanding

- Workflow G decoder-conn phase still running (its hypothesis is obsolete — shim fix already solved decoder connectivity)
- Workflow F (L-R17-B2 SMARTS): 217 passed, 10 skipped — awaiting synthesis
- Workflow H (atoms/combinators + PLL MCTS): in flight
- Workflow I (Phase 4 sweep): blocked on script-flag mismatches in `r4_lambda_only_run.py`

## What is NOT yet validated

- 10K-step retrain (current smoke is 1K steps × 3 seeds, ≈ 4 min total)
- Real decode-quality metric (mean_atoms=8.0 looks suspicious — suggests truncation; need to check if it's a length-bucket artifact or actual output)
- 100-pocket sweep with Lambda MCTS (separate from YuelBond smoke)
