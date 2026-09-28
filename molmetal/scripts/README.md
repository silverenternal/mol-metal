# `scripts/` — Training + retraining entry points

All CLI entry points for training, fine-tuning, ablation, and analysis. Each
script is a **single-file argparse driver** (no class hierarchy) so it can be
launched by `wf_*` workflow wrappers without import-time side effects.

Naming conventions:

| Prefix | Meaning |
|--------|---------|
| `train_*` | Training entry points (D-MPNN, EGNN, hybrid, MPNN multi-task). |
| `finetune_*`, `retrain_*` | Fine-tuning / retrain on top of pretrained weights. |
| `pretrain_*` | Self-supervised pretraining (tmQM, coordination). |
| `r{4,10,11,17}_*` | Round-specific experiments (R4 ablation, R10 OT, R17 GPU retrain). |
| `calibrate_*`, `recalibrate_*` | pIC50 / uncertainty calibration. |
| `eval_*`, `evaluate_*` | Evaluator harness runs (PoseBusters, antic-cancer, generated poses). |
| `make_figures.py`, `generate_section*.py` | Paper-figure / LaTeX regenerators. |
| `wf_*` | Top-level workflow wrappers (bash + py). |
| `ablate_*` | Ablation sweeps. |

## Key files

| File | Role |
|------|------|
| `train_metal_hybrid.py` | T3 hybrid V3 trainer (Ru subset). |
| `train_metal_hybrid_v4.py` | T3 hybrid V4 trainer (temporal holdout). |
| `train_pocket_flow.py` | Pocket-conditioned Lipman CFM trainer. |
| `train_metal_cytotox.py` | Cytotoxicity regression baseline. |
| `train_dmpnn_multitask.py` | D-MPNN multi-task (Rusanov anchor). |
| `metallo_drug_smoke_retrain.py` | 6-shim smoke retrain (R17, M13-M17). |
| `r17_gpu_retrain.py` | Full R17 GPU retrain (ROCm). |
| `pretrain_tmqm.py`, `pretrain_coordination.py` | Self-supervised pretraining. |
| `lambda_100pocket_sweep.py` | Lambda pipeline on 100 pockets (R11/R13). |
| `calibrate_pic50_uncertainty.py` | pIC50 uncertainty calibration (R10). |
| `evaluate_anticancer_metrics.py` | Anticancer pIC50 evaluator. |
| `eval_pocket_macro_skeleton.py` | Pocket macro-skeleton sanity check. |
| `make_figures.py` | Paper figure regenerator. |
| `wf_*` | Workflow orchestration shells. |

## Usage

```bash
# Round-17 GPU retrain with 6 shim fixes applied
python -m molmetal.scripts.metallo_drug_smoke_retrain \
    --metal Ru --epochs 5 --seeds 1 2 3 --splits scaffold temporal

# Pocket-conditioned CFM pretraining
python -m molmetal.scripts.train_pocket_flow \
    --pocket_dir data/crossdocked100 \
    --batch_size 32 --epochs 20
```

```bash
# R17 full GPU retrain (long)
bash molmetal/scripts/wf_reproduce_R17.sh
```

## Common pitfalls

* `train_metal_hybrid_v4.py` **must** be used for temporal splits — older
  versions leak.
* `pretrain_tmqm.py` requires the **tmQM block-copied** dataset (use
  `load_tmqm.py` first).
* `metallo_drug_smoke_retrain.py` is the **only** trainer that exercises
  M13-M17 shims — do not skip it before claiming R17 fixes.
* All scripts assume `cd /home/hugo/codes/mol-metal` first.
* Tests: `pytest molmetal/tests/test_smoke_flag.py test_dry_run_flag.py -q`.