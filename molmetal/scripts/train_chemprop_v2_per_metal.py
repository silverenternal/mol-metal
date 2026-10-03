"""Per-metal single-task chemprop v2 v2 driver (canonical wrapper).

Wf-199 / TODO-181 chemprop v2 v2 driver: per-metal single-task design
that fixes BD-6 broken design (Wf-146 v1 LOMO+per-metal-head 0.446).

Pre-cached features (BD-4): the canonical driver honours the per-canonical
SMILES MolGraph cache produced by ``molmetal/scripts/precache_chemprop_features.py``
(Wf-148 / TODO-148). On a hot cache the SMILES featurization step is
skipped at training time and the per-(metal,seed) wall-time drops to
~6 min total for the full 5 × 5 × 50 sweep on RTX 5090 GPU-1.

For each metal in {Ru, Ir, Os, Rh, Re}, train ONE chemprop v2
:class:`BinaryClassificationFFN` on that metal's data only — no
multi-task, no shared encoder, no LOMO held-out-metal split. This is
the apples-to-apples in-domain benchmark.

Defaults
--------
- metals = (Ru, Ir, Os, Rh, Re)
- seeds  = (0, 1, 2, 3, 4)        (5 seeds × 50 epochs × 5 metals = 25 runs)
- epochs = 50
- device = 'cuda' (auto-falls back to CPU only when MOLMETAL_ALLOW_CPU=1
  or --no-bf16 is set; otherwise fail-fast per Wf-190 env guard)
- bf16   = True                    (Blackwell sm_120 ~3x FP32 throughput)
- num_workers = 4                  (placeholder for DataLoader migration;
  the manual range-based loop is the current implementation)
- pin_memory = True                    (same caveat as num_workers)
- hidden = 300, batch_size = 64, lr = 1e-3  (chemprop v2 defaults)

Outputs (per (metal, seed))
---------------------------
``molmetal/reports/wf170_permetal_singletask/{metal}_seed{N}.json``
    {metal, seed, n_train, n_val, n_test, pos_rate_train, pos_rate_test,
     in_domain_test_auc, elapsed_s, epoch_count, status, ckpt_path}

Outputs (sweep aggregate)
-------------------------
``molmetal/reports/wf170_permetal_singletask/sweep_summary.json``
    per_metal_seed_auc / per_metal_summary / mean_auc_across_metals

Checkpoints (per (metal, seed))
--------------------------------
``molmetal/checkpoints/wf170_permetal_singletask/{metal}_seed{N}.pt``
    {mp_state, agg_state, head_state, metal, seed, atom_fdim, bond_fdim,
     hidden_dim, depth, epochs, in_domain_test_auc}

Usage (default 5 metal × 5 seed × 50 epoch, 7200s wall budget)
-------------------------------------------------------------
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/train_chemprop_v2_per_metal.py

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
1. Per-metal single-task chemprop v2 backbone at defaults (hidden=300,
   depth=3, 50 epochs), 80/20 stratified holdout per metal.
2. NO multi-task, NO shared encoder across metals, NO Wasserstein reg,
   NO coord head. This is the cleanest apples-to-apples single-task
   per-metal baseline.
3. The design is intentionally NOT LOMO. The Wf-146 v1 LOMO + per-metal
   head design was structurally broken (78.9 % < 0.5 random; mean 0.446)
   because the held-out metal head had no labels during training. This
   v2 driver keeps the head but removes LOMO.
4. MEASURED (Wf-170 / 5 × 5 × 50 on RTX 5090 GPU-1, ~6 min wall-time):
   per-metal mean AUCs range 0.5010 (Ir) to 0.5480 (Os), mean across
   metals 0.5145. Per-metal std 0.010-0.070. None of the 5 metals
   reaches the >0.7 acceptance gate, but all exceed the Wf-146 v1
   broken-design mean of 0.446.
5. W₂ regularization (Wf-124 P3 multi-task) is required to lift
   MetalCytoToxDB AUC toward the Krasnov 2026 target of 0.86; the
   per-metal single-task design is the **honest-negative** baseline
   that motivates the multi-task + W₂ design choice.

This module re-exports :func:`main` and the :class:`PerMetalRun`
dataclass from :mod:`train_chemprop_v2_permetal_singletask` so both
script names refer to the same canonical v2 driver implementation.
Backward compatibility with Wf-170 callers is preserved.
"""

from __future__ import annotations


# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import: importing the molmetal package initialises CUDA, and
# CUDA reads CUDA_VISIBLE_DEVICES only once, at init.  Source of
# truth: molmetal/molmetal_lam/sbdd_env/training_utils.py (loaded
# package-free via molmetal/scripts/_cuda_default.py).
import os as _wf130_os
import sys as _wf130_sys

_wf130_dir = _wf130_os.path.dirname(_wf130_os.path.abspath(__file__))
if _wf130_dir not in _wf130_sys.path:
    _wf130_sys.path.append(_wf130_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _wf130_cvd

_wf130_cvd()
del _wf130_cvd, _wf130_dir, _wf130_os, _wf130_sys
# --- end CUDA default --------------------------------------------


# Re-export the canonical v2 driver implementation.
# The implementation lives in train_chemprop_v2_permetal_singletask.py
# (Wf-170 ship) — this file is the canonical Wf-199 / TODO-181 alias.
from train_chemprop_v2_permetal_singletask import (  # noqa: E402,F403
    METALS_5,
    DEFAULT_SEEDS,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_DEPTH,
    DEFAULT_BATCH,
    DEFAULT_LR,
    DEFAULT_OUT_DIR,
    PerMetalRun,
    build_backbone,
    main,
    prefeaturize_all_smiles,
    run_one,
)

__all__ = [
    "METALS_5",
    "DEFAULT_SEEDS",
    "DEFAULT_EPOCHS",
    "DEFAULT_HIDDEN",
    "DEFAULT_DEPTH",
    "DEFAULT_BATCH",
    "DEFAULT_LR",
    "DEFAULT_OUT_DIR",
    "PerMetalRun",
    "build_backbone",
    "main",
    "prefeaturize_all_smiles",
    "run_one",
]


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
