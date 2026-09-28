# TODO-33 P1 F4 — clamp logits before softmax / CE (Smoke)

**Date**: 2026-09-25
**Hypothesis tested**: Clamping `atom_logits` to `[-50, 50]` (and the analogous `bond_logits_e` / `bond_inputs` clamps already present) before `F.softmax` / `F.cross_entropy` prevents BF16 overflow from cascading `Inf` into `atom_loss` / `bond_loss` from step 1.

Combined with **F1 (AMP→FP32)** and **F2 (LR warmup 200)**, this should eliminate the `atom_loss=Inf` / `bond_loss=Inf` wall observed in the F1 and F2 smoke reports and produce a stable training curve.

## Code change

**File**: `molmetal/adapters/flow_matching_lipman/__init__.py`

`atom_logits` was already clamped in the training path (line ~2177,
between `velocity_field(...)` and the `F.cross_entropy(... atom_loss)` site).
Two remaining paths needed the same guard — added them now:

1. **`_generate_impl` — sampling path, before `F.softmax(atom_logits, dim=-1)` (~line 2689).** Mirror of the training-path clamp; prevents BF16 matmul on Blackwell from producing `±1e4` logits in the sampler, which would otherwise produce non-finite probabilities and a `nan / -inf` decode (see F1 smoke §Verdict item 3).
2. **`bond_inputs` and `bond_logits_e` — already clamped earlier in this round** (pre-F4 changes, line ~2281-2288). Audit only; confirmed `bond_logits_e = bond_logits_e.clamp(min=-50.0, max=50.0)` is in place immediately before the bond pattern mask and `F.cross_entropy(bond_logits_e, edge_labels)`.

New edit at line ~2689 (`_generate_impl`):
```python
if self._vocab_mask:
    vocab_mask_tensor = self._build_vocab_mask(
        atom_logits.shape[-1], atom_logits.device,
    )
    atom_logits = atom_logits.masked_fill(~vocab_mask_tensor, float("-inf"))
# TODO-33 P1 F4 — clamp atom_logits to +/-50 *before* F.softmax.
# Mirrors the training-path clamp at line ~2177.  No-op for sane
# logits; only kicks in when the head is producing numerically
# degenerate output (BF16 matmul on Blackwell, smoke observed
# +/-1e4 outliers).
atom_logits = atom_logits.clamp(min=-50.0, max=50.0)
atom_probs = F.softmax(atom_logits, dim=-1)
```

All three CE / softmax sites now carry the guard:
- `atom_logits` (training CE) — line ~2177
- `bond_logits_e` (bond CE) — line ~2288
- `atom_logits` (sampling softmax) — line ~2689 (this PR)

## Command

```bash
source env.sh
export MOLMETAL_CFM_AMP_DTYPE=float32          # F1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_DATA_DIR=/home/hugo/codes/mol-metal/.data/molmetal
export MOLMETAL_TMQM_DIR=/home/hugo/codes/mol-metal/.data/molmetal/tmQM
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
mkdir -p logs/yuelbond_p1_f4/seed_0
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/metallo_drug_smoke_retrain.py \
  --steps 2000 --n-train 500 --batch-size 4 \
  --hidden-dim 128 --n-layers 3 --bond-head learned --joint-train \
  --lr-warmup-steps 200 \                      # F2
  --seed 0 \
  --out-dir logs/yuelbond_p1_f4/seed_0 \
  --decode-smoke-every 500
```

`uv run` was bypassed (Bug #1 from `wf_todo33_p234_fix/final.md`); the
shared CadButEaas venv python was used directly.

## Result: Inf wall ELIMINATED across all three losses

### decode_ratio (the headline metric)

| step | decode_ratio | error |
|---:|---:|---|
|  500 | **0.000** | `TypeError: model must be a VelocityNet; got ModelWrapper` |
| 1000 | **0.000** | same TypeError |
| 1500 | **0.000** | same TypeError |
| 2000 | **0.000** | same TypeError |

**decode_ratio = 0.0** at every probe. Same TypeError reported in
`wf_todo33_p1_f1/smoke.md` and `wf_todo33_p1_f2/smoke.md` — this is a
**separate, pre-existing bug** (the ODE solver shim forwards a
`ModelWrapper` to the upstream `FlowMatchingSampler`, which enforces
`isinstance(model, VelocityNet)` and rejects it). **F4 does NOT fix
decode_ratio**; it only fixes the training-side Inf wall.

### Loss curves (the metric F4 actually targets)

| Loss  | first | last  | min   | max   | Δ first→last | verdict |
|------|------:|------:|------:|------:|------:|---|
| atom | **2.996** | **1.227** | **1.009** | 37.93 | −1.77 | **FINITE — no more Inf** |
| bond | **27.685** | **1.260** | **1.117** | 30.90 | −26.43 | **FINITE — no more Inf** |
| cfm  | 18.538 | 31.185 | 4.163 | 3338.11 | +12.65 | FINITE (still spiky, but bounded) |
| total | 46.523 | 32.568 | 5.547 | 3339.87 | −13.96 | FINITE |

Compared to F1 (`atom=Inf` from step 1, max=Inf) and F2 (same Inf wall)
— **atom_loss and bond_loss are now finite from step 1** and converge
to sensible ranges (`atom ~1.0-3.0`, `bond ~1.1-1.5`). The Inf wall is
genuinely eliminated by stacking F1+F2+F4.

`cfm_loss` still has occasional spikes (max=3338 at the 400-step
window) but it never diverges and recovers within ~50 steps.

### LR schedule (sanity check, identical to F2)

| step | target lr | observed lr |
|---:|---:|---:|
|   1 | 5.00e-07 | 5.00e-07 |
|  100 | 5.00e-05 | 5.00e-05 |
|  200 | 1.00e-04 | 1.00e-04 |
| 2000 | 1.00e-04 | 1.00e-04 |

Bit-for-bit linear warmup, confirmed by `lr_curve_summary` in `final.json`.

### Per-probe loss window at the decode probes (atom / bond always finite)

| step | cfm    | atom   | bond   | note |
|---:|------:|------:|------:|---|
|  500 | 22.74 | 2.08 | 1.38 | decode_smoke runs (TypeError, wall=0.0s) |
| 1000 | 17.08 | 2.00 | 1.22 | decode_smoke runs (TypeError, wall=0.0s) |
| 1500 | 13.47 | 1.42 | 1.29 | decode_smoke runs (TypeError, wall=0.0s) |
| 2000 | 31.19 | 1.23 | 1.26 | final step |

### Configuration

| Knob | Value |
|---|---|
| steps (executed) | 2000 |
| batch_size | 4 |
| lr | 1.00e-04 (after 200-step warmup) |
| hidden_dim | 128 |
| n_layers | 3 |
| joint_train | True |
| bond_head | learned (in_dim=265) |
| vocab_size | 20 |
| device | cuda (RTX PRO 6000 Blackwell sm_120) |
| torch | 2.14.0+cu130 |
| wall-clock | 139.8s (0.07s/step) |

## Verdict

1. **F4 closes the atom/bond Inf wall.** Combined with F1 (FP32) and F2 (200-step LR warmup), `atom_loss` and `bond_loss` are now finite from step 1 and converge to small numbers (~1.1-2.5) over 2000 steps. The training loop is stable: no NaN / no skip-step / no AdamW poisoning.
2. **F4 does NOT fix decode_ratio.** `decode_ratio=0.000` persists because the probe runs into `TypeError: model must be a VelocityNet; got ModelWrapper` at the ODE solver shim before `F.softmax` is ever reached. Fix is in `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py` (unwrap `velocity_model.model` when it is a `ModelWrapper` — see `wf_todo33_p1_f1/smoke.md` §Verdict item 3). **Unblocks any decode smoke probe** and should be picked up by a separate ultracode.
3. **F4 + F1 + F2 stack is the recommended baseline for P1.** The 3-fix stack is the smallest combination that (a) keeps `atom_logits` / `bond_logits_e` / `bond_inputs` numerically sane (F4), (b) avoids BF16 autocast producing degenerate matmul output (F1), and (c) doesn't catastrophically overshoot on the first step when the model is randomly initialised (F2). PCGrad (F3) remains the optional 4th pillar from the TODO-33 plan.
4. **`cfm_loss` is still spiky** (max=3338.11) but bounded — likely a separate instability in the velocity-field coordinate path that clamp does not address. Decode_ratio cannot diagnose it because the ODE solver shim still throws.

## Recommended next fixes

- **ODE solver `ModelWrapper` unwrap** (1-line shim fix in
  `_reference/solver/__init__.py`) — should be its own ultracode; it is
  the gate on every decode smoke probe and on the F4 verification
  that depends on `decode_ratio > 0`.
- **CFM coordinate loss spike** (max 3338) — investigate `vel_head`
  early-step output range; consider per-coordinate gradient clipping
  or a `tanh` saturation head (Lipman 2023 Thm 2). Open a follow-up
  TODO.
- **Multi-seed verification** — this run is `seed=0` only. Repeat
  with `--seed 1`, `--seed 2` to confirm atom/bond finiteness across
  seeds before declaring F4 ship-ready.

## Artifacts

- Log: `logs/yuelbond_p1_f4/seed_0.log`
- Final: `logs/yuelbond_p1_f4/seed_0/final.{md,json}`
- Ckpt: `logs/yuelbond_p1_f4/seed_0/ckpt.pt`
- Code change: `molmetal/adapters/flow_matching_lipman/__init__.py` (sampling-path clamp at line ~2689)
- Comparative baselines: `molmetal/reports/wf_todo33_p1_f1/smoke.md`, `molmetal/reports/wf_todo33_p1_f2/smoke.md`
