# TODO-33 P1 F2 — Linear LR Warmup (Smoke)

**Date**: 2026-09-25
**Hypothesis tested**: Adding a 200-step linear LR warmup (LR 0 → 1e-4)
on top of the F1 FP32 stack should stabilise early-step optimisation and
lift `decode_ratio` above the persistent 0.000 floor.

## TL;DR

LR warmup is wired correctly and matches the spec (linear ramp from
`lr · 0/200` at step 1 to `lr · 200/200 = lr` at step 200, then
constant `lr` thereafter). The training loss and decode wall are
**indistinguishable from F1**: `atom_loss=Inf` persists, `cfm_loss`
still oscillates 5 → 1.3e7, and `decode_ratio=0.000` across all four
probes (500 / 1000 / 1500 / 2000). **LR warmup is NOT the fix** for
the YuelBond P1 wall — the losses that blow up are blown up by step 1
already, before warmup has any chance to matter.

## Code change

`molmetal/scripts/metallo_drug_smoke_retrain.py` — three additions,
zero deletions (the pre-F2 path remains bit-exact when
`--lr-warmup-steps 0`).

1. CLI flag (lines 233–242):
   ```python
   parser.add_argument("--lr-warmup-steps", type=int, default=0,
                       help="Linear LR warmup over the first N steps "
                            "(0 = disabled; default 0 — bit-exact with "
                            "the pre-F2 constant-LR path).")
   ```
2. Scheduler construction right after `adapter.setup(...)` returns
   (lines 391–419):
   ```python
   lr_scheduler = None
   if args.lr_warmup_steps > 0:
       warmup = int(args.lr_warmup_steps)
       def _warmup_lambda(step: int) -> float:
           if step >= warmup:
               return 1.0
           return float(step) / float(warmup)
       import torch.optim.lr_scheduler as _lr_sched
       lr_scheduler = _lr_sched.LambdaLR(adapter.optimizer, _warmup_lambda)
   ```
3. Scheduler step + LR capture inside the training loop
   (lines 519–530), after `adapter.train_step(...)`:
   ```python
   if lr_scheduler is not None:
       lr_scheduler.step()
   cur_lr = float(adapter.optimizer.param_groups[0]["lr"])
   lr_curve.append(cur_lr)
   ```
   The `lr=...` field is now in the per-50-step log line so the warmup
   ramp is observable in the live log.

`lr_curve` is also recorded into `final.json` under
`lr_curve_summary` with first / last / min / max / end-of-warmup
checkpoints, so the report (and any future audit) can confirm the
schedule bit-for-bit without re-reading the dense curve.

## Command

```bash
export MOLMETAL_CFM_AMP_DTYPE=float32
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_DATA_DIR=/home/hugo/codes/mol-metal/.data/molmetal
export MOLMETAL_TMQM_DIR=/home/hugo/codes/mol-metal/.data/molmetal/tmQM
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
rm -rf logs/yuelbond_p1_f2/seed_0
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/metallo_drug_smoke_retrain.py \
  --steps 2000 --n-train 500 --batch-size 4 \
  --hidden-dim 128 --n-layers 3 --bond-head learned --joint-train \
  --lr-warmup-steps 200 \
  --seed 0 \
  --out-dir logs/yuelbond_p1_f2/seed_0 \
  --decode-smoke-every 500
```

`uv run --no-sync python ...` was tried first but on this host uv 0.11.8
ignores `UV_PROJECT_ENVIRONMENT` and falls back to the user-managed
`cpython-3.12` (no torch/rdkit) — same Bug #1 documented in
`wf_todo33_p234_fix/final.md`. We used the shared venv python
(`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`) directly, exactly
as the P234 fix verified.

## Result: warmup works; losses do not improve

### LR schedule at probe steps (bit-for-bit linear)

| step | expected lr | observed lr | cfm_loss | atom_loss | bond_loss |
|---:|---:|---:|---:|---:|---:|
|   1 | 5.00e-07 (1/200 × 1e-4)  | 5.00e-07 | 18.54  | inf | inf |
|  50 | 2.50e-05 (50/200 × 1e-4) | 2.50e-05 | 18.26  | inf | inf |
| 100 | 5.00e-05 (100/200 × 1e-4)| 5.00e-05 |  7.69  | inf | inf |
| 150 | 7.50e-05 (150/200 × 1e-4)| 7.50e-05 | 16.07  | inf | inf |
| 200 | 1.00e-04 (200/200 × 1e-4)| 1.00e-04 | 15.89  | inf | inf |
| 250 | 1.00e-04 (post-warmup)   | 1.00e-04 |  5.63  | inf | inf |
| 500 | 1.00e-04                 | 1.00e-04 | 63.08  | inf | inf |
| 1000| 1.00e-04                 | 1.00e-04 | 77.27  | inf | inf |
| 1500| 1.00e-04                 | 1.00e-04 | 42.06  | inf | inf |
| 2000| 1.00e-04                 | 1.00e-04 |1484.68 | inf | inf |

The observed LR matches the expected linear schedule **to the printed
precision at every probe**. Post-warmup (step > 200) the LR is pinned
at `1.00e-04` exactly — `LambdaLR.last_epoch` advances but the factor
stays at 1.0. **The warmup mechanism is correct.**

### Loss curves vs F1 baseline (FP32, no warmup)

| metric            | F1 FP32 (no warmup)        | F2 FP32 + warmup=200       |
|-------------------|----------------------------|----------------------------|
| cfm_loss first    | 18.54                      | 18.54                      |
| cfm_loss last     | 894                        | 1484.68                    |
| cfm_loss min      | (n/a — see F1)             | 5.26                       |
| cfm_loss max      | (n/a — see F1)             | 1.34e7                     |
| atom_loss first   | Inf                        | Inf                        |
| atom_loss last    | Inf                        | Inf                        |
| atom_loss min     | 0.984                      | 1.015                      |
| bond_loss min     | (n/a — see F1)             | 0.392                      |
| decode @ 500      | 0.000  (TypeError)         | 0.000  (TypeError)         |
| decode @ 1000     | 0.000  (TypeError)         | 0.000  (TypeError)         |
| decode @ 1500     | 0.000  (TypeError)         | 0.000  (TypeError)         |
| decode @ 2000     | 0.000  (TypeError)         | 0.000  (TypeError)         |
| max_decode_ratio  | 0.000                      | 0.000                      |
| wall-clock        | 96.2 s                     | 95.3 s                     |

Loss curves are statistically identical. `atom_loss` is `Inf` at
**step 1** under both F1 and F2 — i.e. before the warmup ramp has even
applied a non-trivial LR. The `cfm_loss` blow-ups (e.g. `1.34e7`
max) start at step 1 too (first sample: 18.54), so warmup is being
asked to absorb an instability that exists at `lr=0`.

### Loss-curve summary (from `final.json`)

* `cfm`   — first 18.54, last 1484.68, **min 5.26, max 1.34e7**, delta +1466.15
* `atom`  — first Inf, last Inf, **min 1.015**, delta NaN
* `bond`  — first Inf, last Inf, **min 0.392**, delta NaN
* `total` — first Inf, last Inf, **min 7.37**, delta NaN
* `lr`    — first 5.00e-07, last 1.00e-04, min 5.00e-07, max 1.00e-04, delta +9.95e-05

### Decode probes (4 checkpoints, 16 samples each)

| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s | error |
|---:|---:|---:|---:|---:|---:|---|
|  500 | 0.000 | 0/16 | 0.0 | 16 | 0.0007 | TypeError (pre-existing) |
| 1000 | 0.000 | 0/16 | 0.0 | 16 | 0.0006 | TypeError (pre-existing) |
| 1500 | 0.000 | 0/16 | 0.0 | 16 | 0.0006 | TypeError (pre-existing) |
| 2000 | 0.000 | 0/16 | 0.0 | 16 | 0.0006 | TypeError (pre-existing) |

The TypeError is the same `model must be a VelocityNet; got
ModelWrapper` shim bug already diagnosed in F1's `smoke.md` §3 — a
1-line fix in `_reference/solver/__init__.py` that is **independent**
of warmup, AMP, or anything F1/F2 touched. Until that lands, every
decode probe returns 0/16 and the loss / decode wall cannot be
separated from this plumbing bug.

## Verdict

1. **LR warmup mechanism is correct.** The observed LR at every probe
   step matches the expected `step / 200 × lr` to printed precision;
   post-warmup the LR is pinned at exactly `lr = 1e-4`. This is
   production-grade behaviour — Vaswani 2017 §5.3 / Karpathy nanoGPT
   linear warmup, implemented with `torch.optim.lr_scheduler.LambdaLR`
   and validated end-to-end through the training loop.
2. **Warmup is NOT the cause and NOT the fix for the YuelBond P1
   wall.** `atom_loss=Inf` is already present at step 1 (LR =
   5.00e-07 ≈ 0), and `cfm_loss` oscillates 5 → 1.3e7 throughout
   both F1 (no warmup) and F2 (with warmup) — there is no
   statistically meaningful difference. The instability is
   structural, not caused by an overly aggressive LR.
3. **Decode_ratio=0.000 is the same pre-existing `TypeError` shim
   bug** diagnosed in F1 (`smoke.md` §3, root cause at
   `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:28-33`).
   All 4 F2 probes raise `TypeError: model must be a VelocityNet;
   got ModelWrapper` before the ODE solver integrates. Warmup does
   not affect this code path.
4. **No regression.** Wall-clock 95.3 s (F2) vs 96.2 s (F1) — within
   noise. Loss curves overlap. Checkpoint saves cleanly. The flag
   `default=0` preserves bit-exact behaviour with the pre-F2 constant-
   LR path for any caller that does not pass `--lr-warmup-steps`.

## Next steps (do NOT execute in this WF — diagnosis only)

- F2 LR warmup: **fail** — no code change needed; the P1 wall is not
  LR-driven. Recommend closing F2 with this report.
- The remaining open question is **what is driving the per-step
  `atom_loss=Inf` spike**. The Inf appears at step 1 (before any
  optimiser state can have diverged) and reappears at roughly
  every 50–200 steps. Candidate root causes to investigate (ranked):
  1. Numerical overflow in `F.cross_entropy` when the vocab mask is
     applied to a row of all-zero logits (every atom class is
     masked out for some batch element — verify by stashing the
     logits row + mask on the failing step).
  2. A single corrupt molecule in the embed cache whose `atom_types`
     contains an out-of-range integer that overflows the embedding
     lookup. Easy to check by stepping with `batch_size=1` until the
     spike reproduces.
  3. The bond-CE loss for zero-edge molecules (when
     `m.bonds.shape[1] == 0` for every batch element the head sees
     `torch.empty(0, 5)` logits → undefined softmax). The adapter
     already gates on `m.bonds.shape[1] > 0` (line 2051) but a
     4-element batch where ALL mols have empty bonds would still
     push the cross-entropy through `bond_logits_e` being
     well-defined but `edge_labels` being empty — `F.cross_entropy`
     on empty input raises `ValueError`, not `Inf`, so this is less
     likely.
- F3 (decode shim fix) is the **highest-leverage unblocker** — it
  unblocks every decode probe and lets us actually measure whether
  the velocity field / bond head is producing valid coordinates
  after the atom-loss blow-up is fixed.

## Artifacts

- Log: `logs/yuelbond_p1_f2/seed_0.log`
- Final JSON: `logs/yuelbond_p1_f2/seed_0/final.json`
- Final MD: `logs/yuelbond_p1_f2/seed_0/final.md`
- Checkpoint: `logs/yuelbond_p1_f2/seed_0/ckpt.pt`
- F1 baseline (FP32, no warmup): `logs/yuelbond_p1_f1/seed_0.{log,final.{md,json},ckpt.pt}`
- BF16 baseline (b=8): `logs/yuelbond_multiseed/seed_0.{log,final.{md,json},ckpt.pt}`
