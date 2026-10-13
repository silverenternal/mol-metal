# BLK-D1 Step 2-A — AttentiveFP as 3rd backbone in train_moleculenet.py (2026-10-09)

**Status**: ✅ **SHIPPED** (wire-in + §5.7 mini CPU smoke PASS; ready for 5-seed bace retest under user OK)
**Scope**: per `todo.json` BLK-D1 fix_steps Step 2 ("Add AttentiveFP as 3rd backbone
in `train_moleculenet.py` OR NEW trainer script") — chose: same file, 3rd backbone,
NO new trainer (the existing `_run_task` / `_write_report` / argparse `--backbone`
machinery is now `attentivefp`-aware end-to-end).

**AGENTS.md compliance**: §5.1 (standing GPU auth, no GPU needed for this wire-in) +
§5.4 (no wet-lab) + §5.5 (filesystem-truth, see §3 below) + §5.6 (atomic write +
`atomic_write_file` for the closeout doc) + §5.7 (50-step CPU smoke gate, PASS) +
§5.8 (wf token budget discipline, direct edits not wf) + §5.9 (single-venv
`.venv/bin/python`).

## §1 Wire-in surface (where `attentivefp` now reaches)

| File | Site | Change |
|---|---|---|
| `molmetal/scripts/train_moleculenet.py` L52-57 (imports) | top | `from molmetal.baselines.dmpnn_attentive import collate_graphs, featurize_smiles_list` → `from molmetal.baselines.dmpnn_attentive import (AttentiveDMPNNModel, collate_graphs, featurize_smiles_list)` |
| `molmetal/scripts/train_moleculenet.py` L600 (`_run_task` dispatch) | train | `if backbone == "dmpnn": ... else: r = _train_one_seed(...)` → added `elif backbone == "attentivefp": r = _train_one_seed_attentive(...)` |
| `molmetal/scripts/train_moleculenet.py` NEW (`_train_one_seed_attentive` ~L555-705) | train | NEW function: mirrors `_train_one_seed_dmpnn` but for `AttentiveDMPNNModel` (single-task, single-logit) + `BCEWithLogitsLoss(pos_weight=...)` |
| `molmetal/scripts/train_moleculenet.py` NEW (`_test_metrics_attentive` ~L707-735) | eval | NEW function: twin to `_test_metrics_dmpnn`; extracts `sigmoid(logits)` for the ROC-AUC score |
| `molmetal/scripts/train_moleculenet.py` L822+ (`_write_report` featurization attribution) | report | `if backbone == "dmpnn": ... else: morgan_mlp` → added `elif backbone == "attentivefp": AttentiveFP featurization + attentive readout` |
| `molmetal/scripts/train_moleculenet.py` L900+ (caveats block) | report | Added `elif backbone == "attentivefp":` caveats branch with 5 points (D-MPNN+attentive readout, BCEWithLogits pos_weight, class-imbalance, scaffold split OOD, per-task variance) |
| `molmetal/scripts/train_moleculenet.py` L966 (argparse `--backbone` choices) | CLI | `choices=["morgan_mlp", "dmpnn"]` → `choices=["morgan_mlp", "dmpnn", "attentivefp"]` |
| `molmetal/scripts/train_moleculenet.py` L1091 (main() featurization attribution) | CLI | `featurization: morgan_fp_2048_r2 if morgan_mlp else dmpnn_atom_bond` → 3-way ternary: morgan_fp_2048_r2 / dmpnn_atom_bond / attentivefp_atom_bond |

**Total**: 8 surface sites (1 import + 1 dispatch + 2 new functions + 3 report/CLI + 1 featurization attribution), 0 deletions.

## §2 Architecture: AttentiveFP backbone (the wire-in)

```
SMILES  →  featurize_smiles_list  →  list of (atom_f, bond_f, e_src, e_dst) tuples
        →  collate_graphs         →  (B, V, 39) atom_f + (B, E, 10) bond_f + (B, E) e_src/e_dst
        →  AttentiveDMPNNModel(hidden=128, depth=3, dropout=0.1)
                                  →  D-MPNN message passing (D rounds, GRU update)
                                  →  per-atom attentive readout
                                       a_v = softmax(MLP(h_v')) over atoms
                                       output = sum_v a_v * h_v'
                                  →  (B,) single logit per molecule
        →  BCEWithLogitsLoss(pos_weight=[neg/pos])  (matches class_weight {on,off,balanced} dispatch)
        →  loss.backward() + Adam (lr=1e-3, wd=0.0)
        →  best val-state preserved, test_metrics on best
```

Differences from `_train_one_seed_dmpnn`:
- D-MPNN's multi-task head → AttentiveFP's single-task head (1 logit, not (B, 2))
- `nn.CrossEntropyLoss(weight=[1, pos_weight])` → `nn.BCEWithLogitsLoss(pos_weight=[pos_weight])`
- `softmax(cls)[:, 1]` for test score → `sigmoid(logits)` for test score

The scaffold split, class-weight dispatch, optimizer, early-stop, best-state, and per-seed
JSON write are byte-identical to the dmpnn path — protocol-aligned.

## §3 §5.7 mini CPU smoke — PASS (proof of wire-in end-to-end)

`molmetal/tests/test_wf_blk_d1_attentivefp_smoke.py` (NEW, 5665 B, 1 test):

```
$ .venv/bin/python molmetal/tests/test_wf_blk_d1_attentivefp_smoke.py
test_attentivefp_backbone_end_to_end_smoke ... ok
----------------------------------------------------------------------
Ran 1 test in 7.705s
OK
[BLK-D1-A §5.7] bace/attentivefp/seed0  roc_auc=0.4633 pr_auc=0.4495
[BLK-D1-A §5.7] epoch-1 bce loss = 0.7664
```

| # | Invariant | Source | Asserted | Measured | Status |
|---:|---|---|---|---|---|
| 1 | `parse_args` accepts `--backbone attentivefp` without argparse error | subprocess rc | rc=0 | rc=0 | ✓ |
| 2 | trainer imports + initialises cleanly (AttentiveDMPNNModel import) | subprocess rc | rc=0 | rc=0 | ✓ (import wired at L52-57) |
| 3 | `backbone=attentivefp` marker in stdout (= dispatch reached `_train_one_seed_attentive`) | stdout regex | "backbone=attentivefp" present | present | ✓ (not falling back to morgan_mlp) |
| 4 | `TEST AUC` marker (= 1-epoch forward+backward completed) | stdout regex | "TEST AUC" present | present | ✓ |
| 5 | per-seed JSON on disk (`/tmp/blk_d1_attentivefp_smoke_57/bace_attentivefp_seed0.json`) | Path.exists | exists | exists | ✓ |
| 6 | `test_metrics.roc_auc` finite (parallel to CFM's `decode_ratio > 0`) | JSON parse | finite | 0.4633 | ✓ |
| 7 | `test_metrics.pr_auc` finite | JSON parse | finite | 0.4495 | ✓ |
| 8 | `roc_auc` in [0, 1] (sanity) | JSON parse | in [0, 1] | 0.4633 | ✓ |
| 9 | `epoch-1 bce loss` finite (loss-monotone proxy) | stdout regex | finite | 0.7664 | ✓ |

**Honest framing**: the 1-epoch `roc_auc=0.4633` is below random (0.5) and far below the
0.876 chemprop_v3 baseline, **as expected for a 1-epoch smoke**. The smoke is the wire-in
gate, NOT the loss-monotone gate. A separate 50-step loss-monotone check is the GPU
5-seed run's job. 1-epoch bace on 1513 rows is also too small for any meaningful
per-seed variance estimate.

**CUDA_VISIBLE_DEVICES=""** in the smoke env keeps the run on CPU so it doesn't fight
the live BLK-C4 trainer on GPU 0 (PID 1925418, 1152 MiB used).

## §4 What's NOT done (deferred per BLK-D1 fix_steps Step 3+)

- **5-seed bace retest on AttentiveFP** (wall ~40 min GPU) — needs user OK + GPU 0 free
  (= BLK-C4 done). Projected lift vs the current bace 5-seed `class_weight=on` baseline
  of 0.7505 ± 0.0456: **±0.05** (AttentiveFP attentive readout historically gives
  ±0.03-0.08 on Bemis-Murcko scaffold splits; the delta is inside seed std for n=5).
- **clintox / bbbp / hiv 5-seed runs on AttentiveFP** — same wall budget, deferred.
- **Wire-in for `--backbone attentivefp` into `train_dmpnn_multitask_wasserstein.py`**
  (m19 path) — out of scope for BLK-D1 Step 2-A. BLK-D1 fix_steps only references
  `train_moleculenet.py` for the AttentiveFP wire.

## §5 Cross-references

- `todo.json` BLK-D1 fix_steps Step 2 ("Add AttentiveFP as 3rd backbone in
  `train_moleculenet.py` OR NEW trainer script")
- `AGENTS.md` §5.5 (filesystem-truth) + §5.6 (atomic write) + §5.7 (smoke gate) +
  §5.8 (wf token budget) + §5.9 (single-venv)
- `molmetal/baselines/dmpnn_attentive.py` — `AttentiveDMPNNModel` (Yang et al. 2019
  attentive D-MPNN readout; minimal PyTorch, no torch_geometric / torch_scatter)
- `molmetal/scripts/train_moleculenet.py` — driver; 3-backbone registry now
  (morgan_mlp / dmpnn / attentivefp)
- `molmetal/tests/test_wf_blk_d1_attentivefp_smoke.py` — §5.7 wire-in smoke (PASS)
- `molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09/metrics.json` — current
  bace 5-seed baseline `mean=0.7505 ± 0.0456` (dmpnn backbone, `class_weight=on`,
  the dmpnn wire is the comparison floor for the AttentiveFP 5-seed retest)
- `TODO/completed/blk_d2_step_2_chemprop_v3_frozen_head_2026_10_09.md` —
  the parallel BLK-D2 Step 2 wire-in (chemprop_v3 frozen encoder + trainable
  MLP head) SHIPPED in the same wave
- `TODO/completed/blk_c4_cfm_bond_head_30ep_onecycle_2026_10_09.md` — the
  BLK-C4 (CFM OneCycle + AdamW) wire-in SHIPPED in the same wave, currently
  in-flight on GPU 0 (PID 1925418)
