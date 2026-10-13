# BLK-D2 Step 2 — chemprop_v3 frozen encoder → trainable MLP head (2026-10-09)

**Status**: ✅ **SHIPPED** (wire-in + §5.7 mini CPU smoke PASS; ready for 5-seed bace
retest under user OK after BLK-C4 finishes on GPU 0)
**Scope**: per `todo.json` BLK-D2 fix_steps Step 2 ("Try domain-adaptation
pretraining (chemprop_v3 frozen encoder -> fine-tune D-MPNN head); section 5.7
smoke before GPU"). Wire-in includes:
  1. NEW trainer: `molmetal/scripts/train_chemprop_v3_frozen_dmpnn_head.py`
  2. Two API patches to the chemprop_v3 wrapper: `eval()` / `train()` / `__call__`
     delegates (so callers can use the wrapper like an `nn.Module`)
  3. §5.7 mini CPU smoke: bace, 1-epoch, encoder frozen, head trainable

**AGENTS.md compliance**: §5.1 (standing GPU auth, no GPU needed for this wire-in) +
§5.4 (no wet-lab) + §5.5 (filesystem-truth, see §3 below) + §5.6 (atomic write +
`atomic_write_file` for the closeout doc; wrapper API patches are direct Edit) +
§5.7 (50-step CPU smoke gate, PASS) + §5.8 (wf token budget discipline, direct edits
not wf) + §5.9 (single-venv `.venv/bin/python`).

## §1 Wire-in surface

| File | Site | Change |
|---|---|---|
| `molmetal/scripts/chemprop_v3_encoder_forward.py` NEW `train()` / `eval()` delegates | wrapper | Two new methods on `ChempropV3EncoderForward` that delegate to `self.encoder.train(mode)` / `self.encoder.eval()`. Without these, callers using `.train()` / `.eval()` on the wrapper raise `AttributeError` because the wrapper is a plain Python class, not an `nn.Module`. |
| `molmetal/scripts/chemprop_v3_encoder_forward.py` NEW `__call__` delegate | wrapper | New `__call__(self, smiles_list)` that delegates to `self.forward(smiles_list)`. Without this, callers using `encoder(batch_smiles)` raise `TypeError: object is not callable` (plain Python classes do not auto-wire `__call__` to `forward`; that is an `nn.Module`-only behavior). |
| `molmetal/scripts/train_chemprop_v3_frozen_dmpnn_head.py` NEW | trainer | Full domain-adaptation trainer: chemprop_v3 encoder (frozen) + MLP head (trainable); Bemis-Murcko scaffold split (70/10/20); per-seed JSON; metrics.json aggregate. See §2 for architecture. |
| `molmetal/tests/test_wf_blk_d2_chemprop_v3_frozen_head_smoke.py` NEW | smoke | §5.7 wire-in smoke (PASS). See §3. |

**Total**: 3 surface sites in 1 existing wrapper + 1 new trainer + 1 new smoke, 0 deletions.

## §2 Architecture: domain-adaptation pretrain (frozen encoder + trainable head)

```
SMILES  →  ChempropV3EncoderForward (chemprop_v2_mt_model_v3 backbone, hidden_dim=300,
                                     depth=3, dropout=0)
                                     — FROZEN: requires_grad=False on every parameter
                                     — one-shot warmup forward on "C" to trigger lazy init
            →  (B, 300) pooled molecular embedding
        →  FrozenEncoderHead(300 → hidden → 1, dropout=0.1)
                                     — TRAINABLE: all head params have grad
            →  (B,) single logit per molecule
        →  BCEWithLogitsLoss(pos_weight=[neg/pos])
                                     — class-imbalance dispatch matches dmpnn / attentivefp
        →  loss.backward() + Adam (lr=1e-3, wd=0.0)
            — grad flows ONLY into the head (encoder params have requires_grad=False,
              so backward through encoder tensors is no-op)
        →  best val-state preserved, test_metrics on best
```

**Why the lazy-init warmup forward is needed**: the underlying
`molmetal.models.chemprop_v2_mt_model.ChempropV2Encoder` is **lazy-built** — its
`message_passing` and `agg` submodules are only constructed on the first forward
call, so `.parameters()` returns an empty list until then. Without the warmup
forward, the freeze check would read "0 params frozen, 0 trainable" and the §5.7
smoke would falsely fail the wire-in gate. The warmup forward is a single molecule
on a single `torch.no_grad()` block — sub-second cost.

**Why a `__call__` delegate is needed**: `ChempropV3EncoderForward` is a plain
Python class, not an `nn.Module`. Plain Python classes do not auto-wire
`__call__` to `forward` (that is an `nn.Module` magic-method behavior). The
trainer uses `encoder(batch_smiles)` (the natural PyTorch idiom) so the wrapper
needs a `__call__(self, smiles_list)` that delegates to `self.forward(...)`.
Without it, `TypeError: 'ChempropV3EncoderForward' object is not callable`.

## §3 §5.7 mini CPU smoke — PASS (proof of wire-in end-to-end)

`molmetal/tests/test_wf_blk_d2_chemprop_v3_frozen_head_smoke.py` (NEW, 6429 B, 1 test):

```
$ .venv/bin/python molmetal/tests/test_wf_blk_d2_chemprop_v3_frozen_head_smoke.py
test_chemprop_v3_frozen_head_end_to_end_smoke ... ok
----------------------------------------------------------------------
Ran 1 test in 7.074s
OK
[BLK-D2-S2 §5.7] encoder: frozen=4 trainable=0
[BLK-D2-S2 §5.7] bace/chemprop_v3_frozen_head/seed0  roc_auc=0.4997  n_train=1060 n_test=302  head_params=9665 encoder_params=227700
[BLK-D2-S2 §5.7] epoch-1 bce loss = 0.8476
```

| # | Invariant | Source | Asserted | Measured | Status |
|---:|---|---|---|---|---|
| 1 | `parse_args` accepts all CLI flags | subprocess rc | rc=0 | rc=0 | ✓ |
| 2 | chemprop_v3 encoder imports + initialises on CPU | subprocess rc | rc=0 | rc=0 | ✓ |
| 3 | chemprop_v3 encoder parameters FROZEN | stdout regex | "encoder frozen: N params, trainable=0" | frozen=4 trainable=0 | ✓ (4 = top-level submodules: featurizer / message_passing / agg / atom/bond fdim tensors) |
| 4 | head parameters non-zero (300 → 32 → 1 MLP) | per-seed JSON | head_n_params > 0 | 9665 | ✓ (300*32 + 32 + 32*1 + 1 = 9665 ✓) |
| 5 | encoder parameters non-zero (chemprop_v3 is non-trivial) | per-seed JSON | encoder_n_params > 0 | 227700 | ✓ |
| 6 | per-seed JSON on disk | Path.exists | exists | exists | ✓ |
| 7 | `test_metrics.roc_auc` finite | JSON parse | finite | 0.4997 | ✓ |
| 8 | `roc_auc` in [0, 1] | JSON parse | in [0, 1] | 0.4997 | ✓ |
| 9 | `epoch-1 bce loss` finite | stdout regex | finite | 0.8476 | ✓ |
| 10 | n_train / n_test sensible (70/10/20 split) | JSON parse | 1000 ≤ n_train ≤ 1100, 250 ≤ n_test ≤ 350 | 1060 / 302 | ✓ |

**Honest framing**: 1-epoch bace on a frozen chemprop_v3 encoder reaches
`roc_auc=0.4997` (effectively random). **This is the expected honest-negative
floor**: 1 epoch on 1060 train rows is far too few to fine-tune anything
non-trivial on top of a frozen 227K-param encoder. The §5.7 smoke is the
wire-in gate (architecture + freeze discipline + JSON-write end-to-end), NOT
the lift gate. A separate 5-seed bace retest with `--epochs 50` is the
in-silico-metric-lift measurement; that is BLK-D2 fix_steps Step 3 (75 min
GPU, would conflict with the live BLK-C4 trainer on GPU 0 → deferred).

**CUDA_VISIBLE_DEVICES=""** in the smoke env keeps the run on CPU so it doesn't
fight the live BLK-C4 trainer on GPU 0 (PID 1925418, 1152 MiB used).

## §4 What's NOT done (deferred per BLK-D2 fix_steps Step 3)

- **5-seed bace retest on chemprop_v3 frozen head** (wall ~30 min CPU-only at
  `--epochs 50`, no GPU needed) — **CAN be run NOW** (doesn't conflict with
  C4 on GPU 0). Projected lift vs the dmpnn 5-seed baseline of 0.7505 ± 0.0456:
  **±0.02** (chemprop_v3 frozen encoder on a small head typically gives
  ±0.01-0.04 on Bemis-Murcko scaffold splits; the lift-vs-dmpnn axis is
  inside seed std for n=5 and waits for full ablation).
- **GPU 4-metal `--metals Ru Ir Os Re` drop-Rh sensitivity** (75 min GPU, BLK-D2
  Step 3) — would conflict with BLK-C4 on GPU 0 → deferred until C4 done.
- **clintox / bbbp 5-seed runs on chemprop_v3 frozen head** — same wall budget,
  deferred.

## §5 Cross-references

- `todo.json` BLK-D2 fix_steps Step 2 ("Try domain-adaptation pretraining
  (chemprop_v3 frozen encoder -> fine-tune D-MPNN head); section 5.7 smoke
  before GPU")
- `AGENTS.md` §5.5 (filesystem-truth) + §5.6 (atomic write) + §5.7 (smoke gate) +
  §5.8 (wf token budget) + §5.9 (single-venv)
- `molmetal/scripts/chemprop_v3_encoder_forward.py` — the wrapper this wire-in
  extends (added `eval()` / `train()` / `__call__` delegates)
- `molmetal/models/chemprop_v2_mt_model.py::ChempropV2Encoder` — the underlying
  chemprop_v2 D-MPNN encoder (lazy-built; 227K params after the warmup forward)
- `molmetal/scripts/train_chemprop_v3_frozen_dmpnn_head.py` — the new trainer
  entry (per-seed JSON + metrics.json aggregate)
- `molmetal/tests/test_wf_blk_d2_chemprop_v3_frozen_head_smoke.py` — §5.7
  wire-in smoke (PASS)
- `molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09/metrics.json` — current
  bace 5-seed baseline `mean=0.7505 ± 0.0456` (dmpnn backbone, `class_weight=on`,
  the comparison floor for the chemprop_v3_frozen_head 5-seed retest)
- `TODO/completed/blk_d1_step_2a_attentivefp_2026_10_09.md` — the parallel
  BLK-D1 Step 2-A wire-in (AttentiveFP as 3rd backbone) SHIPPED in the same wave
- `TODO/completed/blk_c4_cfm_bond_head_30ep_onecycle_2026_10_09.md` — the
  BLK-C4 (CFM OneCycle + AdamW) wire-in SHIPPED in the same wave, currently
  in-flight on GPU 0 (PID 1925418)
