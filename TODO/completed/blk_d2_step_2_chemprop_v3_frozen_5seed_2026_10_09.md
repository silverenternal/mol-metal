# BLK-D2 Step 2: bace 5-seed chemprop_v3 frozen head retest — CLOSEOUT

**Date:** 2026-10-09
**Operator:** Claude (background detached, CPU-only)
**Status:** SHIPPED (5-seed lift measurement vs dmpnn baseline)
**Wire-in provenance:** `molmetal/scripts/train_chemprop_v3_frozen_dmpnn_head.py`
(per `TODO/completed/blk_d2_step_2_chemprop_v3_frozen_head_2026_10_09.md` SHIPPED
wire-in; the 1-epoch §5.7 smoke at bace/seed0 had `roc_auc=0.4997`, the random
floor expected for 1 epoch on 1060 train rows with a frozen encoder).

## §5.5 filesystem-truth (ls -la evidence)

```
$ ls -la molmetal/reports/blk_d2_chemprop_v3_frozen_5seed_2026_10_09/
-rw-r--r-- 1 hugo hugo   461 Oct  9 14:34 chemprop_v3_frozen_head_seed0.json
-rw-r--r-- 1 hugo hugo   462 Oct  9 14:35 chemprop_v3_frozen_head_seed1.json
-rw-r--r-- 1 hugo hugo   463 Oct  9 14:36 chemprop_v3_frozen_head_seed2.json
-rw-r--r-- 1 hugo hugo   462 Oct  9 14:38 chemprop_v3_frozen_head_seed3.json
-rw-r--r-- 1 hugo hugo   463 Oct  9 14:39 chemprop_v3_frozen_head_seed4.json
-rw-r--r-- 1 hugo hugo   579 Oct  9 14:39 metrics.json
```

All 5 seed JSONs + aggregate `metrics.json` are on disk.

## §5.5 MEASURED lift (raw numbers, no paraphrase)

| Backbone | bace 5-seed mean ± std | per-seed |
|---|---|---|
| **dmpnn baseline** (Wf-FIX-3LINES 2026-10-09) | **0.7505 ± 0.0456** | 0.665 / 0.800 / 0.757 / 0.771 / 0.760 |
| **chemprop_v3 frozen + MLP(32) head** (BLK-D2, this run) | **0.7299 ± 0.0435** | 0.787 / 0.706 / 0.755 / 0.742 / 0.660 |

- **Δ vs dmpnn = -0.0206** (chemprop_v3 frozen - dmpnn)
- **Honest framing per `verify-premises-before-proposing` + `no-self-made-tables`**:
  - Δ = -0.0206 is **inside the 1σ envelope of BOTH arms** (dmpnn std=0.0456,
    chemprop_v3 std=0.0435 → combined noise floor ≈ ±0.06).
  - However the **direction is NEGATIVE**: the frozen chemprop_v3 encoder +
    32-unit MLP head is **NOT a win** over a fully-trained dmpnn baseline.
  - This is the canonical "frozen + tiny head" honest negative: 9,665 trainable
    head params (vs dmpnn's 300-d message-passing encoder training end-to-end)
    is **structurally under-parameterised** for bace.
  - **vs literature Chemprop D-MPNN 0.876**: Δ = -0.146. Below literature by 14.6pp.
- **Encoder / head param budget per `chemprop_v3_frozen_head_seed0.json`**:
  - encoder_n_params = 227,700 (frozen)
  - head_n_params = 9,665 (trainable MLP 300→32→1)
  - **trainable / total = 9,665 / (227,700 + 9,665) = 4.1%** — a "frozen" recipe
    by construction.

## §5.6 wall + protocol

- **Wall (per `metrics.json::elapsed_seconds`)**: 318.7 s = **5.3 min** (5 seeds ×
  50 ep on CPU; `CUDA_VISIBLE_DEVICES=""`, keeps GPU 0 free for the parallel
  AttentiveFP run).
- **Protocol**: Bemis-Murcko scaffold 70/10/20, 50 epochs, batch 32, lr 1e-3,
  hidden 32 (MLP head), dropout 0.1, seeds 0 1 2 3 4 — same split as dmpnn
  baseline + same seed set, so the two are directly comparable.

## Verdict

- **BLK-D2 Step 2 SHIPPED** — the wire-in works end-to-end on 5 seeds.
- **Honest negative on the lift claim**: chemprop_v3 frozen + 32-unit MLP head
  is **BELOW** the dmpnn baseline (-0.021 within noise but negative direction).
  The 9,665-param head is **structurally under-parameterised** for bace.
- **Honest framing catalog entry** (per `docs/HONEST_NEGATIVE_CATALOG.md`):
  "chemprop_v3 frozen encoder + 32-unit MLP head on bace = 0.7299 ± 0.0435;
  -0.021 vs dmpnn 0.7505 ± 0.0456; inside noise but direction negative; tiny
  head budget is structural ceiling; future-work = larger head (256-unit) or
  full encoder fine-tune, NOT another frozen+small retry."

## What this closes

- BLK-D2 Step 2 5-seed retest is SHIPPED.
- BLK-D2 as a whole has now completed: Step 2 wire-in (SHIPPED 2026-10-09) + Step 2
  5-seed lift (this SHIPPED, honest-negative result).
- Future-work (NOT auto-pursued, awaiting user OK): if user wants to chase
  chemprop_v3 fine-tune lift, the path is a 256-unit head + small unfreeze
  of the top 1-2 message-passing layers (NOT another frozen+small retry).
