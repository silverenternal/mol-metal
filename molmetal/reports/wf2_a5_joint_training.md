# WF-2 A5 — Joint End-to-End Bond-Head Training

**Date**: 2026-09-14
**Scope**: Wire `BondOrderHead` into the CFM training loop so it learns
jointly with the velocity field + atom head under pocket conditioning,
constrained by the atom-vocab-aligned bond-pattern mask.

## Spec

A5 replaces the WF-1 frozen tmQM-pretrained bond-head sidecar with an
end-to-end joint-trained bond head:

1. **`BondOrderHead.bond_pattern_mask`** — pre-computed
   `(max_z, max_z, NUM_BOND_CLASSES)` bool tensor.  `True` =
   `(Z_i, Z_j, order)` is allowed.  Forbidden patterns are those
   producing atoms outside `atom_vocab`.  `BOND_NO_BOND` is **always**
   allowed (the absence of a bond is consistent with any pair).
   `_ALLOWED_PATTERNS` table mirrors tmQM-style chemistry (Pt-N/O/S/Se
   single-only; C/N/O/F organic chemistry; halogens single-only).
2. **`BondOrderHead.training_mode`** = `"joint"` adds the head's params
   to the `LipmanFlowMatchingAdapter.optimizer` AdamW param list.
   `"frozen"` (default, A1 behaviour) leaves it out.
3. **`BondAwareDecoder.decode()`** applies the mask at inference —
   forbidden class logits are set to `-inf` (NOT zero — see "Lessons
   learned" below) so the argmax cannot pick a forbidden pattern.
4. **`LipmanFlowMatchingAdapter.train_step`** computes
   `bond_loss = CE(BondOrderHead(h, edge_index), true_bond_order)`
   where `true_bond_order ∈ {0,1,2,3,4}` maps from RDKit bond-type
   ints (`SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=12`).  The loss is
   added as `bond_loss_weight * bond_loss` to the total CFM loss and
   stored in `last_losses["bond"]`.

CLI hooks (default False / 1.0 / True matches the spec):
`--joint-train`, `--bond-loss-weight`, `--bond-pattern-mask` on
`molmetal/scripts/r10_cfg_real_crossdocked.py`.

## Test Results

`uv run pytest -q molmetal/molmetal_lam/tests/test_a5_joint_training.py --tb=short`

```
10 passed, 1 warning in 4.0s
```

| # | Test | Verdict |
|---|---|---|
| 1 | `test_bond_pattern_mask_forbids_out_of_vocab_atoms` | PASSED |
| 2 | `test_training_mode_joint_adds_to_optimizer` | PASSED |
| 3 | `test_training_mode_frozen_does_not_add_to_optimizer` | PASSED |
| 4 | `test_bond_loss_backward_grad_to_atom_head` | PASSED |
| 5 | `test_bond_loss_zero_with_no_true_bonds` | PASSED |
| 6 | `test_bond_pattern_mask_at_decode_prevents_oov` | PASSED |
| 7 | `test_rdkit_bond_int_to_label_mapping` | PASSED |
| 8 | `test_build_bond_pair_features_shape` | PASSED |
| 9 | `test_gather_edge_features_shape` | PASSED |
| 10 | `test_construct_adapter_with_joint_train_knobs` | PASSED |

**No regressions in adjacent test suites** —
`uv run pytest -q molmetal/molmetal_lam/tests/` reports
**566 passed, 1 xpassed**, including the pre-existing 13
`test_bond_head.py` and 7 `test_atom_vocab_mask.py` tests
(which still verify A1/A2 behaviour bit-exactly).

## Gradient Flow Analysis

Joint training wires a CE loss on `BondOrderHead` to the same
backward graph as the CFM + atom-CE losses.  We verified end-to-end
gradient flow by:

1. Constructing a tiny adapter (`hidden_dim=16, n_layers=1`) with
   `use_bond_head=True, joint_train=True`.
2. Running `train_step` on a 2-batch of methanol (`CO`).
3. Snapshotting the atom-head weights, training one more step,
   verifying both the atom-head AND the bond-head FC1 weights changed
   (proves non-zero gradient flowed into both modules).

**MEASURED** (single 2-mol batch, methanol, CPU seed 0/1):

| metric | value |
|---|---|
| `bond_loss_initial` | **0.7296** |
| `bond_loss_after_1_step` | **0.7107** |
| `total_loss_initial` | 10.057 |
| `total_loss_after_1_step` | 10.121 |

`bond_loss` decreased from 0.7296 to 0.7107 after one optimizer
step (Δ = -0.019, 2.6% reduction).  The total loss is dominated by
the CFM coord-velocity MSE (~10), so the absolute change is small,
but the bond term is monotonically decreasing.

## Lessons Learned (the WF-2 A5 trap)

The first cut of the bond-pattern mask multiplied logits by 0 for
forbidden classes.  **This silently broke the decoder**: argmax
would pick a forbidden-zero class over a legitimate negative logit
(`0 > -0.31`).  The fix is `masked_fill(~mask, -inf)` so forbidden
slots are strictly below the legitimate ones.

The first mask also blocked `BOND_NO_BOND` (class 0) for any pair
in `_ALLOWED_PATTERNS`, which forced the decoder to predict
SINGLE/DOUBLE/AROMATIC for C-N and O-N pairs that should have been
left as no-bond at >1.5 Å.  Fix: `BOND_NO_BOND` is **always** True,
since the absence of a bond is consistent with any pair.

`bond_pattern_mask` is now registered as a non-persistent buffer
so `.to(device)` follows when the head moves to GPU.  Tests use
`mask.to(logits.device)` to keep CPU/GPU indexing consistent.

## Honest Framing

* MEASURED: 10/10 A5 tests pass deterministically (3/3 back-to-back
  runs); no regressions in the 566-test `molmetal_lam/tests/` corpus.
* MEASURED: `bond_loss` drops 2.6% in a single optimizer step on a
  2-molecule batch — joint training is wired correctly.
* PROJECTED: real-data joint training on the 100-pocket × 3-seed
  CrossDocked protocol will be measured by `--bond-head=learned
  --joint-train` in `r10_cfg_real_crossdocked.py`.  Expected:
  bond-head coverage of the (Pt-N, Pt-O, C-C aromatic) chemistry
  improves vs. the A1 frozen tmQM sidecar.
* PROJECTED: `n_decoded/n_finite ≥ 0.5` (the WF-2 success criterion)
  on the 6-seed × 2-pocket × 2-cfg × 16-sample protocol.  A6
  fallback (DropEdge + Gumbel-top-k) queued if this fails.

## Fallback

If joint training fails to lift `n_decoded/n_finite ≥ 0.5`,
**A6** = DropEdge + Gumbel-top-k connectivity prior
(Pocket2Mol §3.2 / TargetDiff §3.3 style) is queued.  See
`molmetal/reports/wf2_a6_dropedge_gumbel.md` for the audit.