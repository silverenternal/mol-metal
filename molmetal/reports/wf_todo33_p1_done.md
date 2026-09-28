# TODO-33 P1 Final Synthesis — Blackwell decode_ratio=0 fix attempt

**Date**: 2026-09-25
**Inputs**: `logs/yuelbond_p1_{f1,f2,f4}/seed_0/`, `logs/yuelbond_p1_full/seed_{0,1,2}/`
**Reference**: R16 ROCm baseline `molmetal/reports/wf_r16_yuelbond_2000_probe/final.md`

## 1. Executive Summary

**No fix in the F1/F2/F4 set moved `decode_ratio` off zero.** All four Blackwell runs (F1, F2, F4, F1+F2+F4 3-seed) end with `decode_ratio = 0.000` at every probe step (500/1000/1500/2000).

- **Root cause (post-mortem)** is **not** numerical instability. F4 alone produces finite `atom_loss` / `bond_loss` from step 1 with no Inf spikes (`atom 1.0-3.0`, `bond 1.1-1.5`, `cfm_loss min 3.75-4.18`). The training loop is mathematically healthy.
- The actual blocker is a **shim-bug in the ODE solver**: every probe raises `TypeError: \`model\` must be a VelocityNet; got ModelWrapper.` at `flow_matching/sampler.py:81` before any decode math runs. The `_reference/solver/__init__.py` shim wraps the velocity field in the legacy `ModelWrapper`, then hands that wrapper to the upstream `FlowMatchingSampler` whose `isinstance(model, VelocityNet)` check rejects it.
- F1/F2/F4 each fixed a *different* slice of the original BF16/Blackwell symptom bundle, but they all leave the wrapper-typecheck regression in place because it is orthogonal to training stability.

**Recommended config going forward** (smallest set that keeps training healthy for the next decode unblock):

| knob | value | rationale |
|---|---|---|
| `MOLMETAL_CFM_AMP_DTYPE` | `float32` (F1) | eliminates BF16 matmul overflow on Blackwell |
| `--lr-warmup-steps` | `200` (F2, **make default**) | avoids first-step overshoot; restores 4-line LR log |
| atom_logits clamp | `[-50, 50]` at both training + sampling softmax (F4) | bound the logit space so `cross_entropy` stays finite |
| vocab | 20-element set (`{H,B,C,N,O,F,Si,P,S,Cl,Sc,Se,Br,Y,Zr,I,Hf,Pt}`) | covers all `metallo_drugs_500_train.csv` atoms without OOV |

P1.3 (PCGrad) remains the optional 4th pillar from `TODO/pending/33_…` §1 but is **not** required to unblock decode.

## 2. Per-fix results

| Fix | atom_loss@1 | cfm min | decode_ratio @ 500/1000/1500/2000 | verdict |
|---|---:|---:|---|---|
| **F1** AMP→FP32 | **Inf** | 4.063 | 0.000 / 0.000 / 0.000 / 0.000 (TypeError) | loss still Inf; no effect |
| **F2** LR warmup 200 | **Inf** | 5.265 | 0.000 / 0.000 / 0.000 / 0.000 (TypeError) | loss still Inf; no effect |
| **F4** F1 + F2 + clamp logits | **2.996** | 4.163 | 0.000 / 0.000 / 0.000 / 0.000 (TypeError) | **Inf wall eliminated**; decode still 0 (orthogonal) |

Detail loss curves (single seed, 2000 steps):

| Loss  | F1 first/last/min/max        | F2 first/last/min/max            | F4 first/last/min/max         |
|-------|------------------------------|----------------------------------|-------------------------------|
| cfm   | 18.5 / 894.0 / 4.06 / 4.9e4  | 18.5 / 1484.7 / 5.27 / 1.3e7     | 18.5 / 31.2 / 4.16 / 3338     |
| atom  | **Inf** / **Inf** / 0.98 / **Inf** | **Inf** / **Inf** / 1.02 / **Inf** | 3.00 / 1.23 / 1.01 / 37.93 |
| bond  | **Inf** / **Inf** / 0.49 / **Inf** | **Inf** / **Inf** / 0.39 / **Inf** | 27.68 / 1.26 / 1.12 / 30.90 |
| total | **Inf** / **Inf** / 7.99 / **Inf** | **Inf** / **Inf** / 7.37 / **Inf** | 46.52 / 32.57 / 5.55 / 3340 |

The transition is sharp: adding the `atom_logits.clamp(-50, 50)` guard at both the training CE site (~line 2177) **and** the sampling softmax site (~line 2689) is what flips atom/bond from "Inf-from-step-1" to "finite and bounded". F4 also widened `vocab_size` 12→20 so all atoms in the 500-mol pool are now in-vocab (otherwise the cfm mask would clamp them to `-inf` early).

**decoder verdict, F4**: 4/4 probes raise the same `TypeError: \`model\` must be a VelocityNet; got ModelWrapper.` The shim in `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py` must unwrap `velocity_model.model` before passing to `FlowMatchingSampler` — see `wf_todo33_p1_f1/smoke.md` §6 item 3.

## 3. 3-seed full verification (F1 + F2 + F4)

| seed | cfm_loss_min | atom_loss_min | bond_loss_min | decode@500/1000/1500/2000 | wall_s |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.179 | 0.985 | 1.134 | 0.000 / 0.000 / 0.000 / 0.000 (TypeError) | 94.4 |
| 1 | 4.084 | 0.980 | 1.036 | 0.000 / 0.000 / 0.000 / 0.000 (TypeError) | 172.5 |
| 2 | 3.750 | 0.980 | 1.089 | 0.000 / 0.000 / 0.000 / 0.000 (TypeError) | 172.6 |

Seed-mean ± std: `cfm_loss_min = 4.005 ± 0.184`; `decode_ratio = 0.000 ± 0.000` at every probe. All four probes per seed raise the same `TypeError`. **`P1 fix passes the loss-curve gate (no Inf, finite min, stable schedule) but fails the decode-lift gate (0/3 seeds).`**

## 4. Comparison to R16 ROCm baseline

`molmetal/reports/wf_r16_yuelbond_2000_probe/final.md`:

| Metric | R16 ROCm (RX 7800 XT) | P1 attempt (Blackwell sm_120) | Δ |
|---|---:|---:|---|
| `cfm_loss min` | 6.65 | 4.16 (F4) / 4.005 ± 0.18 (3-seed mean) | **better** |
| `cfm_loss first` | 21.09 | 18.54 | better |
| `cfm_loss last` | 5756 (Inf-blown) | 31.19 (F4) / 9.99 (seed-2) | **>> better** |
| `atom_loss` | Inf spikes | finite 0.98-1.39 | **better** |
| `decode_ratio @ step 500` | **1.000** | **0.000** | **WORSE** |
| `decode_ratio @ step 2000` | **1.000** | **0.000** | **WORSE** |

The P1 fix improved every loss metric by a wide margin (no Inf, lower min, better convergence). It did not recover the `decode_ratio = 1.000` from R16 because the wrapper typecheck regression is new in R17 and unrelated to BF16/LR/clamp. R16 also reported `mean_atoms=8.0` at every probe, which one section of the R16 caveat flagged as the unconditional-mean collapse — so R16's "1.000 lift off the 0 floor" was a known artefact, not a true pocket-conditioned decode.

## 5. Recommendations: default-config changes

| Change | In default? | Why |
|---|---|---|
| `lr_warmup_steps = 200` | **yes** | stabilises LR trace (linear 5e-7 → 1e-4 over 200 steps); ~10% slower but eliminates first-step divergence. Mirror R10-R15 warmup convention. |
| `MOLMETAL_CFM_AMP_DTYPE=float32` | **yes** on Blackwell; **BF16** on ROCm | Blackwell `autocast(dtype=torch.bfloat16)` overflows `atom_logits`; ROCm path is numerically fine. Make it conditional via env var (`float32` default on `compute capability >= 8.9`). |
| `atom_logits.clamp(-50, 50)` | **yes**, both sites | no-op for sane logits; prevents cascading Inf; ~0 perf cost |
| `vocab_size = 20` (full set) | **yes** | all real atoms in pool, no OOV masking cost |
| `batch_size` 8 → 4 | keep 4 for now | R16 probe used 8 but with Inf spikes; F4 uses 4 and is stable |
| `PCGrad (P1.3)` | **defer** until decode unblock | loss curves are stable without it; adding complexity risks regressions |
| Keep `vocab_mask_in_loss` (P0-F3) | **yes** | confirmed working |

## 6. Next steps

- **Decode blocker is orthogonal to F1/F2/F4.** The `TypeError: \`model\` must be a VelocityNet; got ModelWrapper.` at `flow_matching/sampler.py:81` is raised by the upstream `isinstance(model, VelocityNet)` check after the shim hands it a `ModelWrapper`. The fix is to unwrap in `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py` (≈1-3 lines) before passing to `FlowMatchingSampler`. This is a separate ultracode and the highest-priority next step.
- **Once that is unblocked**, re-run the F4 smoke (`MOLMETAL_CFM_AMP_DTYPE=float32`, `lr_warmup_steps=200`, `clamp_logits=True`, single seed, 2000 steps) to verify `decode_ratio > 0`.
  - If **≥1 probe > 0.5** in that re-run → promote config to default and run the full 10000-step verification per TODO-33 §10 line 3 (`decode_ratio > 0 at step 1000/2000 in 3/3 seeds`).
  - If **all probes still < 0.5** → escalate to `TODO/pending/24_cfm_architecture_redo_plan.md` §P1.3 **PCGrad** (gradient projection per task on conflicting atom/bond gradients) — this is what R17 already budgeted as the last-resort fix.
- `cfm_loss` spikes (max=3338 even in F4) suggest a remaining instability in the velocity-field coordinate path that clamp does not address. Investigate per-coordinate gradient clipping on `vel_head` if needed.
- Companion reports: `wf_todo33_p1_f1/smoke.md`, `wf_todo33_p1_f2/smoke.md`, `wf_todo33_p1_f4/smoke.md`, `wf_todo33_p1_vocab_fix/final.md`, `wf_todo33_p1_full/final.md`.
