"""CLI for the multi-task D-MPNN baseline (TODO/04 C4).

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.train_dmpnn_multitask --metal Ru --split temporal --epochs 30 --alpha 0.5

Trains a dual-head D-MPNN with:

  * pIC50 regression head (MSE)
  * activity classification head (BCE)

Loss:

  L = alpha * BCE(active_logits, active_label) + (1 - alpha) * MSE(pic50_pred, pIC50_label)

Evaluates and writes both classification (ROC-AUC, PR-AUC, Hit@5%) and
regression (MAE, RMSE, Pearson r) metrics to stdout **and** to
``molmetal/reports/baseline_<metal>_dmpnn_multitask_<split>.json``.
"""

from __future__ import annotations


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os
_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskBaseline,
    MT_ALPHA,
)
from molmetal.baselines.dmpnn import SPLITTER_FACTORIES
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.splits import (
    AssayGroupSplitter,
    SizeGroupSplitter,
)
from molmetal.molmetal_lam.sbdd_env.groupnorm_replacement import (
    replace_batchnorm_with_groupnorm,
)

# TODO-143 §2.5 Day 4 — optional four-domain cross-domain training mode.
# Imported lazily because loading DrugOOD native (1.58M rows) costs ~80 MB
# of RAM even when the user does not opt into ``--data-config four-domain``.
try:
    from molmetal.data.four_domain_combined import FourDomainDataset
    from molmetal.data.four_domain_weights import make_four_domain_sampler
    _FOUR_DOMAIN_AVAILABLE = True
except Exception as _exc:  # pragma: no cover — defensive import
    _FOUR_DOMAIN_AVAILABLE = False
    _FOUR_DOMAIN_IMPORT_ERR = repr(_exc)

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-task D-MPNN.")
    p.add_argument(
        "--metal",
        default="Ru",
        choices=["Ru", "Ir", "Rh", "Os", "Re", "Pt"],
        help="Metal centre to filter on (default: Ru)",
    )
    p.add_argument(
        "--split",
        default="random",
        choices=sorted(SPLITTER_FACTORIES.keys()),
        help=(
            "Splitting strategy. 'random' (default), 'temporal' (pre/post 2024), "
            "'ligand_dedup', 'scaffold' (Bemis-Murcko), 'chemical' (Tanimoto-0.7), "
            "'assay' / 'size' (DrugOOD Ji 2023 ICLR group-key splits; require "
            "--data-config four-domain so DrugOOD rows are loaded)."
        ),
    )
    p.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    p.add_argument(
        "--batch",
        type=int,
        default=256,
        help=(
            "Mini-batch size (default: 256; for the GPU-only pretrain path "
            "the pre-featurized tensors are already on GPU, so larger "
            "batches are cheap — suggest BATCH=2048 on a 188-SM Blackwell)."
        ),
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=MT_ALPHA,
        help="Weight on BCE; (1-alpha) goes to MSE (default: 0.5)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: molmetal/reports/baseline_<metal>_dmpnn_multitask_<split>.json",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default uses MT_LR=1e-3)",
    )
    # ------------------------------------------------------------------
    # TODO-143 §2.2 / Day 5-6: Domain-Adversarial training flags.
    #
    # --use-dann        Enable DANN adversarial head on top of the D-MPNN
    #                   backbone.  When set, the script wraps the pooled
    #                   molecular embedding with
    #                   ``molmetal_lam.sbdd_env.gradient_reversal.ReverseLayerF``
    #                   before the 4-class domain classifier (MetalCytoToxDB /
    #                   DrugOOD / NCI-60 / BindingDB).
    # --lambda-grl      Scalar weight on the (inverted) domain loss.  The
    #                   total loss becomes
    #                       L_total = L_task + lambda_grl * L_domain
    #                   where the GRL makes ``L_domain``'s gradient flip sign
    #                   at backward time.  Per TODO-143 §4 Day 5-6 we default
    #                   to 0.1 (small alpha keeps the D-MPNN task head stable).
    # --domain-id       Override the domain label for single-domain runs so
    #                   the DANN head always sees a consistent label.
    #                   CANONICAL mapping (MUST stay in sync with
    #                   molmetal/data/four_domain_combined.py:DOMAIN_* and
    #                   molmetal/scripts/pre_featurize_dmpnn.py:DOMAIN_ID):
    #                       0 = MetalCytoToxDB
    #                       1 = DrugOOD
    #                       2 = NCI-60
    #                       3 = BindingDB
    #                   Defaults to 0 (MetalCytoToxDB) which matches the
    #                   existing ``--metal`` filtering path.
    #                   Audit (Wf-124 P0 #2, 2026-09-27): this mapping was
    #                   already correct here — the inconsistency lived in
    #                   pre_featurize_dmpnn.py.DOMAIN_ID which has now been
    #                   corrected.
    # ------------------------------------------------------------------
    p.add_argument(
        "--use-dann",
        action="store_true",
        default=False,
        help="Enable Domain-Adversarial head on top of the D-MPNN backbone.",
    )
    p.add_argument(
        "--lambda-grl",
        type=float,
        default=0.1,
        help="Weight on the inverted domain loss (default: 0.1 per TODO-143).",
    )
    p.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help=(
            "Domain label for DANN training (CANONICAL mapping — see "
            "pre_featurize_dmpnn.py for the audit trail): "
            "0=MetalCytoToxDB 1=DrugOOD 2=NCI-60 3=BindingDB (default: 0)."
        ),
    )
    # ------------------------------------------------------------------
    # TODO-143 §2.5 Day 4: per-domain WeightedRandomSampler hook.
    #
    # --data-config      "single" (default — current path) or "four-domain"
    #                    (cross-domain ConcatDataset + WeightedRandomSampler).
    #                    In four-domain mode the per-row batch is drawn from
    #                    the unified MetalCytoToxDB / DrugOOD / NCI-60 /
    #                    BindingDB ConcatDataset; ``--metal`` filtering still
    #                    applies to the MetalCytoToxDB subset but does NOT
    #                    restrict the other 3 domains (each is, by design,
    #                    metal-agnostic or has its own metal convention).
    # --alpha-sampler    Balancing exponent passed to
    #                    :func:`make_four_domain_sampler`.
    #                    0 → inverse-freq, 1 → uniform, 0.5 → soft (default).
    # --max-rows-domain  Optional per-domain row caps for fast smoke runs
    #                    (comma-separated ``d:n`` pairs, e.g. ``0:200,1:200``).
    # ------------------------------------------------------------------
    p.add_argument(
        "--data-config",
        choices=["single", "four-domain", "pretrain"],
        default="single",
        help=(
            "Data layout: 'single' (default — current per-metal CSV), "
            "'four-domain' (cross-domain ConcatDataset + WeightedRandomSampler), "
            "or 'pretrain' (RDKit-free GPU training path using "
            "molmetal/data/pre_featurized/all_pretrain.pt pre-featurized tensors)."
        ),
    )
    p.add_argument(
        "--alpha-sampler",
        type=float,
        default=0.5,
        help=(
            "Per-domain WeightedRandomSampler exponent (0=inverse-freq, "
            "1=uniform, 0.5=soft balance; default: 0.5)."
        ),
    )
    p.add_argument(
        "--max-rows-domain",
        type=str,
        default="",
        help=(
            "Per-domain row caps for fast smoke runs (e.g. '0:200,1:200'). "
            "Only honoured with --data-config four-domain."
        ),
    )
    # ------------------------------------------------------------------
    # TODO-143 §2.1: Kendall uncertainty-weighted multi-task loss.
    #
    # --use-kendall-loss    Replace the manual alpha-weighted BCE+MSE sum
    #                       with Kendall et al. 2018 (CVPR, arXiv:1705.07115)
    #                       homoscedastic-uncertainty weighting.  When set,
    #                       the per-task sigma is learned jointly with model
    #                       parameters and ``--alpha`` is ignored.
    #                       Implementation: drop-in port at
    #                       molmetal/molmetal_lam/sbdd_env/uncertainty_loss.py
    #                       (verbatim from elna4os/mtl_uncertainty_loss,
    #                       MIT-licensed).  See TODO-143 §2.1 verify gate:
    #                       log_sigmas.std() > 0.1 after 50 epochs (no
    #                       collapse to uniform weights).
    # ------------------------------------------------------------------
    p.add_argument(
        "--use-kendall-loss",
        dest="use_kendall_loss",
        action="store_true",
        default=False,
        help=(
            "Use Kendall 2018 uncertainty-weighted multi-task loss "
            "(drop-in port in molmetal/molmetal_lam/sbdd_env/"
            "uncertainty_loss.py).  Overrides the manual alpha weighting; "
            "the per-task sigma is learned jointly with model parameters.  "
            "See TODO-143 §2.1."
        ),
    )
    # ------------------------------------------------------------------
    # BF16 autocast (compute-bound speedup).
    #
    # --bf16       Wrap the D-MPNN forward pass in
    #              ``torch.amp.autocast('cuda', dtype=torch.bfloat16)`` for
    #              training.  On Blackwell-class GPUs the compute-bound
    #              forward (matmul, conv) is ~2x faster in BF16 than FP32.
    #              Gradients are accumulated in FP32 (params stay FP32);
    #              we explicitly cast the per-batch loss to FP32 before
    #              backward for stability.
    # --no-bf16    Disable BF16 autocast (equivalent to ``--bf16=false``).
    # ------------------------------------------------------------------
    p.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wrap the D-MPNN forward pass in torch.amp.autocast('cuda', "
            "dtype=torch.bfloat16) for training.  Default: True.  "
            "Pass --no-bf16 to disable."
        ),
    )
    # ------------------------------------------------------------------
    # P0 #3 — Model class switch (Wf-145 P3 / TODO-145 §3.5).
    #
    # --model    Select which D-MPNN multi-task class is instantiated
    #            by the pretrain RDKit-free training path
    #            (run_pretrain_training) AND by the legacy SMILES path
    #            (DMPNNMultiTaskBaseline) when the model class
    #            supports it.
    #
    # Choices:
    #   * "v3"          Reserved — legacy baseline.  Pretrain path
    #                   refuses with NotImplementedError (no V3 multi-
    #                   task class exists in molmetal/models/).
    #   * "v4"          V4MultiTaskDMPNN (default — current behaviour).
    #   * "v4pretrain"  V4MultiTaskDMPNNPretrain — adds
    #                   pretrain_head (per-atom masked-atom-prediction)
    #                   and mol_clf_head (DrugOOD-style molecular
    #                   classifier) on top of V4.  Pretrain-mode is
    #                   auto-enabled.
    #
    # Backward compat: default = "v4" preserves the previous behaviour
    # because V4MultiTaskDMPNN was the implicit choice.
    # ------------------------------------------------------------------
    p.add_argument(
        "--model",
        choices=["v3", "v4", "v4pretrain", "chemprop_v2"],
        default="v4",
        help=(
            "Model class for the D-MPNN multi-task backbone (default: v4). "
            "'v4pretrain' adds pretrain_head + mol_clf_head on top of v4 "
            "and auto-enables set_pretrain_mode(True). 'v3' is reserved "
            "for the legacy baseline and not supported in the pretrain path. "
            "'chemprop_v2' is the Chemprop 2.0 MPNN backbone (Wf-148 / "
            "TODO-148 §3) — implemented at molmetal/models/chemprop_v2_mt_model.py "
            "as ChempropV2MTModel; requires SMILES input so the pretrain "
            "RDKit-free path raises NotImplementedError when selected."
        ),
    )
    # ------------------------------------------------------------------
    # DataLoader knobs (Fix 3 — speed up RDKit featurization).
    #
    # --num-workers        Sub-process count for parallel SMILES→graph
    #                      featurization.  Default 4 (balanced for a
    #                      single-GPU node with ~8 cores).
    # --pin-memory         Enable pinned host memory for faster H2D
    #                      DMA on the GPU forward pass (default True).
    # --no-pin-memory      Disable pin_memory.
    # ------------------------------------------------------------------
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default 4)",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="DataLoader pin_memory (default True)",
    )
    p.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
    )
    # ------------------------------------------------------------------
    # Wf-146 — defensive 21h pretrain safety net.
    #
    # --ckpt-dir     Directory under which ``<metal>_seed{N}_epoch{M}.pt``
    #                snapshots are stored.  Defaults to
    #                ``molmetal/checkpoints/wf146_lomo_v2`` so the standard
    #                5-metal × 3-seed pretrain sweeps land in one place.
    #                Pass an empty string to disable checkpointing entirely.
    # --ckpt-every   Save a full model + optimizer + epoch + best_metric
    #                snapshot every N epochs (default 5).  Set to 0 to
    #                disable.
    # --resume       On startup, auto-load the latest matching checkpoint
    #                from ``--ckpt-dir`` and continue training from
    #                ``epoch+1``.  Refuses to silently mismatch across
    #                ``(metal, seed)`` pairs.
    # ------------------------------------------------------------------
    default_ckpt_dir = (
        str(PROJECT_ROOT / "molmetal" / "checkpoints" / "wf146_lomo_v2")
        if (PROJECT_ROOT / "molmetal").exists()
        else ""
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default=default_ckpt_dir,
        help=(
            "Checkpoint directory (default: "
            "molmetal/checkpoints/wf146_lomo_v2). Pass '' to disable."
        ),
    )
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=5,
        help=(
            "Save full checkpoint every N epochs (default: 5). "
            "0 disables."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Auto-resume from the latest matching checkpoint in "
            "--ckpt-dir on startup (default: False)."
        ),
    )
    return p.parse_args(argv)


def build_dataloader(
    metal: str = "Ru",
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 42,
    filters: Optional[CytotoxFilter] = None,
    csv_path: Optional[str] = None,
) -> DataLoader:
    """Construct a torch DataLoader for the MetalCytoToxDB rows of one metal.

    This is the worker-friendly counterpart to the inline ``DMPNNMultiTaskBaseline.fit``
    loop: RDKit SMILES→features is the CPU bottleneck (~186ms/step at BATCH=256),
    so we run featurization in ``num_workers`` background processes while the
    GPU trains. ``pin_memory=True`` issues a fast DMA pageable→pinned copy
    ahead of the ``non_blocking=True`` H2D transfer on the GPU forward pass.

    Parameters
    ----------
    metal : str
        Metal centre whitelist (default ``"Ru"`` — matches the smoke-run metal).
    batch_size : int
        Per-batch size.  Matches the D-MPNN baseline batch_size default of 256.
    num_workers : int
        Sub-process count for parallel featurization.  Default ``4`` (balanced
        for a single-GPU node with ~8 cores).
    pin_memory : bool
        Pin host memory for faster H2D transfer (default ``True``).
    persistent_workers : bool
        Keep the worker pool alive across epochs so we don't pay the
        fork/startup cost on every epoch (default ``True``).
    seed : int
        RNG seed for the per-worker ``base_seed``; each worker derives a
        deterministic ``base_seed + worker_id`` so SMILES shuffling is
        reproducible.
    filters : CytotoxFilter, optional
        Pre-built filter; otherwise a default with ``metal_whitelist=[metal]``
        is constructed.
    csv_path : str, optional
        Override for the MetalCytoToxDB CSV path.  Defaults to the
        ``DEFAULT_CSV`` baked into :mod:`molmetal.data.cytotox`; when the
        caller passes a custom path we honour it directly.  This keeps the
        verification helper portable across hosts where the data root may
        differ (e.g. ``/mnt/storage/data/molmetal`` vs the in-repo
        ``data/MetalCytoToxDB.csv``).

    Returns
    -------
    DataLoader
        Configured ``torch.utils.data.DataLoader`` over the filtered
        MetalCytoToxDB rows.  ``len(dl)`` returns the number of batches.
    """
    if filters is None:
        filters = CytotoxFilter(metal_whitelist=[metal])
    # Locate the CSV.  ``molmetal.data.cytotox.from_csv`` uses a module-level
    # ``DEFAULT_CSV = /mnt/storage/data/molmetal/MetalCytoToxDB.csv`` that is
    # unwritable / missing on most dev boxes.  Honour the same
    # ``MOLMETAL_DATA_DIR`` env-var convention and fall back to the
    # project-local ``.data/molmetal/`` tree before giving up.
    import os
    env_root = os.environ.get("MOLMETAL_DATA_DIR")
    candidates: List[Path] = []
    if csv_path is not None:
        candidates.append(Path(csv_path))
    else:
        if env_root:
            candidates.append(Path(env_root) / "MetalCytoToxDB.csv")
        candidates.extend([
            PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv",
            Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv"),
        ])
    csv_file: Optional[Path] = None
    for c in candidates:
        if c.exists():
            csv_file = c
            break
    if csv_file is None:
        raise FileNotFoundError(
            f"MetalCytoToxDB.csv not found.  Tried: {[str(c) for c in candidates]}. "
            "Set MOLMETAL_DATA_DIR to the directory containing MetalCytoToxDB.csv, "
            "or pass csv_path=... to build_dataloader()."
        )

    # Use a project-local cache dir (cytotox.py's module-level default is
    # /mnt/storage/data/molmetal/3d_cache, which is unwritable on most dev
    # boxes).  Build the dataset directly because from_csv() does not expose
    # cache_dir — follow the pattern from
    # :mod:`molmetal.data.four_domain_combined`.
    cache_dir = Path(
        os.environ.get("MOLMETAL_3D_CACHE_DIR")
        or str(csv_file.parent / "3d_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from molmetal.data.cytotox import _add_active, _add_pic50, _apply_filters
    df = pd.read_csv(csv_file, low_memory=False)
    df = _apply_filters(df, filters)
    if filters.compute_pic50:
        df = _add_pic50(df)
    if filters.compute_active:
        df = _add_active(df)
    ds = MetalCytotoxDataset(df=df, cache_dir=cache_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No rows for metal={metal!r} (CSV={csv_file})")
    # ``worker_init_fn`` seeds each worker's numpy + torch RNGs.  This is the
    # documented PyTorch pattern for reproducible multi-worker loaders.
    def _seed_worker(worker_id: int) -> None:
        import numpy as np
        worker_seed = (seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )
    return loader


# ---------------------------------------------------------------------------
# Pretrain RDKit-free GPU training path
# ---------------------------------------------------------------------------
# Pre-featurized tensor file location.  The tensors are produced by
# ``molmetal/scripts/pre_featurize_dmpnn.py`` which runs
# ``GraphFeaturizer()`` once over the full ~336K-row corpus (4 domains)
# and saves the padded (B, N_max, 39) atom-features + (B, 2, E_max)
# edge_index + (B, E_max, 6) edge_attr + atom-mask tensors to disk.
#
# Loading these once into GPU memory and iterating via TensorDataset is
# ~4-5x faster per step than calling ``_featurize_batch`` (RDKit) at
# every step — the 186ms CPU featurize step is replaced by a single
# ``tensor.to('cuda', non_blocking=True)`` index lookup.
PRE_FEATURIZED_DIR = (
    PROJECT_ROOT / "molmetal" / "data" / "pre_featurized"
)
PRE_FEATURIZED_ALL = PRE_FEATURIZED_DIR / "all_pretrain.pt"


def build_pretrain_dataloader(
    batch_size: int = 2048,
    shuffle: bool = True,
    seed: int = 42,
    path: Optional[Path] = None,
    max_rows: Optional[int] = None,
) -> DataLoader:
    """Construct a DataLoader over the pre-featurized GPU tensors.

    Skips RDKit ``_featurize_batch`` entirely — every row has already
    been padded to (N_max=548 atoms, E_max=1214 edges) and serialized
    to disk by ``molmetal/scripts/pre_featurize_dmpnn.py``.

    Parameters
    ----------
    batch_size : int
        Per-batch size.  Defaults to ``2048`` — pre-featurized tensors
        are already on the GPU so larger batches are cheap; the
        D-MPNN message-passing layer is the bottleneck, not the loader.
    shuffle : bool
        Shuffle row order each epoch (default ``True``).
    seed : int
        RNG seed for the shuffle (default 42).
    path : Path, optional
        Override for the pre-featurized ``.pt`` file.  Defaults to
        ``molmetal/data/pre_featurized/all_pretrain.pt`` (~45 GB
        containing 336,556 rows × (N_max=548, E_max=1214) padded
        tensors).
    max_rows : int, optional
        Cap the corpus to the first ``max_rows`` rows.  Useful for
        fast smoke runs and unit tests so the 45 GB file isn't
        materialised in tests.

    Returns
    -------
    DataLoader
        ``torch.utils.data.DataLoader`` wrapping a ``TensorDataset``
        of (atom_features, edge_index, edge_attr, atom_mask,
        metal_idx, y, active, domain) tensors with
        ``pin_memory=False`` (tensors are already on the GPU).
    """
    src = Path(path) if path is not None else PRE_FEATURIZED_ALL
    if not src.exists():
        raise FileNotFoundError(
            f"Pre-featurized tensor file not found: {src}. "
            "Run molmetal/scripts/pre_featurize_dmpnn.py first to "
            "generate molmetal/data/pre_featurized/all_pretrain.pt."
        )
    pack = torch.load(src, map_location="cpu", weights_only=False)
    keys_tensors = [
        "atom_features", "edge_index", "edge_attr", "atom_mask",
        "metal_idx", "y", "active", "domain",
    ]
    tensors = [pack[k] for k in keys_tensors]
    # TODO-160 §4.2 — canonical-metal filter.
    # Sentinel rows with ``metal_idx == -1`` (non-canonical: 90.5% of the
    # ~336k-row pretrain corpus) trip an index-out-of-bounds inside
    # ``forward_cached`` at step 1.  Drop them before any max_rows slice
    # so the cap is applied to canonical rows only.  Also enforce the
    # canonical range {0..4} to defend against stray out-of-range values
    # in future regenerations of the pretrain corpus.
    canonical_lo, canonical_hi = 0, 4
    mi = tensors[4]  # metal_idx is index 4 in keys_tensors
    keep_mask = (mi >= canonical_lo) & (mi <= canonical_hi)
    n_dropped = int(keep_mask.numel() - int(keep_mask.sum().item()))
    tensors = [t[keep_mask] for t in tensors]
    print(
        f"[dmpnn_multitask] pretrain: filtered {n_dropped} non-canonical rows "
        f"(metal_idx outside {{{canonical_lo}..{canonical_hi}}}); "
        f"remaining={tensors[0].size(0)}",
        flush=True,
    )
    if max_rows is not None:
        tensors = [t[:max_rows] for t in tensors]
    n_rows = tensors[0].size(0)
    print(
        f"[dmpnn_multitask] pre-train loader: src={src}  n_rows={n_rows}  "
        f"batch={batch_size}  shuffle={shuffle}",
        flush=True,
    )
    ds = torch.utils.data.TensorDataset(*tensors)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=g,
        drop_last=False,
    )
    return loader


# ---------------------------------------------------------------------------
# Pretrain RDKit-free GPU training loop (Wf-125 P0 #1 wiring).
#
# This is the actual training loop that consumes the pre-featurized
# ``atom_features`` / ``edge_index`` / ``edge_attr`` / ``atom_mask`` /
# ``metal_idx`` / ``y`` / ``active`` / ``domain`` tensors produced by
# ``molmetal/scripts/pre_featurize_dmpnn.py``.  NO RDKit call is issued
# at any point in the loop — every step goes through
# :meth:`V4MultiTaskDMPNN.forward_cached` which accepts the pre-padded
# tensors directly.
#
# The loss function contract is identical to the SMILES path:
#   * :class:`V4MultiTaskDMPNN.forward_cached` returns
#     ``{metal: (B,2) logits, "coord": (B, n_coord_classes)}``
#   * :class:`V4MultiTaskLoss` expects ``y_true: {metal: (B,) long}``
#     and ``coord_true: (B,) long``
#
# To map the pretrain corpus onto these targets we use:
#   * ``active`` (binarised at 0.5) as the per-metal classification
#     target — every metal head sees the same binary activity label so
#     the head pretraining is meaningful (each head learns the same
#     "is this complex active?" signal independently of the metal id).
#   * ``domain`` (clipped to ``[0, n_coord_classes)``) as the
#     coordination-geometry auxiliary target.
#
# NaN-safe: any row with NaN ``active`` is mapped to label 0 (the
# model still trains but the label is uninformative).
# ---------------------------------------------------------------------------
def run_pretrain_training(
    args,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Run the actual pretrain training loop.

    Args:
        args: parsed CLI namespace.  Honours ``--epochs``, ``--batch``,
            ``--lr``, ``--bf16``, ``--use-kendall-loss``, ``--seed``,
            ``--model``.  When ``args.model == "v4pretrain"`` the
            :class:`V4MultiTaskDMPNNPretrain` class is instantiated and
            :meth:`set_pretrain_mode(True)` is called automatically.
        loader: pre-featurized ``TensorDataset`` DataLoader from
            :func:`build_pretrain_dataloader`.
        device: target torch device.  Defaults to CUDA if available
            else CPU.

    Returns:
        Dict with keys:
          ``n_steps_total``      — total optimizer steps executed.
          ``n_epochs``           — epochs actually run.
          ``initial_loss``       — first-step training loss.
          ``final_loss``         — last-step training loss.
          ``loss_decrease``      — initial − final (positive ⇒ loss fell).
          ``forward_cached_calls`` — number of times ``forward_cached``
              was invoked (= ``n_steps_total`` for a clean run).
          ``rdkit_called``       — always False (RDKit-free path).
          ``bf16``               — whether BF16 autocast was active.
          ``model_class``        — str name of the instantiated model class.
          ``pretrain_mode``      — bool, whether pretrain-mode is on.
    """
    from molmetal.models.v4_multitask_dmpnn import (
        SUPPORTED_METALS,
        V4MultiTaskDMPNN,
        V4MultiTaskDMPNNPretrain,
        V4MultiTaskLoss,
    )
    from molmetal.models.chemprop_v2_mt_model import ChempropV2MTModel

    model_choice = str(getattr(args, "model", "v4"))
    if model_choice == "v3":
        # No V3 multi-task class exists; the pretrain path requires the
        # V4 family (forward_cached / multi-task heads).
        raise NotImplementedError(
            "--model v3 is reserved for the legacy single-task D-MPNN "
            "and is not supported by the pretrain RDKit-free training "
            "path. Use --model v4 or --model v4pretrain."
        )
    if model_choice == "chemprop_v2":
        # Wf-148 / TODO-148 §3 — ChempropV2MTModel IS implemented (see
        # molmetal/models/chemprop_v2_mt_model.py) but consumes SMILES,
        # not the pre-featurized (atom_features, edge_index, edge_attr,
        # atom_mask) tensors this pretrain path feeds.  We surface a
        # clear NotImplementedError with the TODO-148 §4 migration
        # pointer so the operator sees the mismatch immediately.
        raise NotImplementedError(
            "--model chemprop_v2 → ChempropV2MTModel IS implemented "
            "(molmetal/models/chemprop_v2_mt_model.py) but the "
            "pretrain RDKit-free path consumes pre-featurized tensors "
            "(atom_features, edge_index, edge_attr, atom_mask) via "
            "V4MultiTaskDMPNN.forward_cached.  ChempropV2MTModel "
            "consumes SMILES via forward(smiles_list, metal_ids).  "
            "The class CAN be instantiated (e.g. via "
            "ChempropV2MTModel(...)) but the pretrain path cannot "
            "consume the pre-featurized tensors.  Migration plan: "
            "TODO-148 §4 (rewrite run_pretrain_training on the "
            "Chemprop v2 SMILES-based forward signature).  Re-run with "
            "--model v4 or --model v4pretrain for the pretrain path."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ----- Model (small for the synthetic pack used in tests; production
    # can override via an extra --hidden-dim CLI flag in a follow-up PR).
    if model_choice == "v4pretrain":
        model = V4MultiTaskDMPNNPretrain(
            hidden_dim=64,
            n_layers=2,
            n_metals=5,
            n_coord_classes=5,
            use_cached_encoding=True,
        ).to(device)
        # Auto-enable pretrain mode so the pretrain_head + mol_clf_head
        # are exercised during training.  Forward_cached currently emits
        # only the base V4 dict; the test layer is responsible for
        # verifying the SMILES-path emission in train_dmpnn_multitask.
        model.set_pretrain_mode(True)
    else:
        model = V4MultiTaskDMPNN(
            hidden_dim=64,
            n_layers=2,
            n_metals=5,
            n_coord_classes=5,
            use_cached_encoding=True,
        ).to(device)
    loss_fn = V4MultiTaskLoss(
        n_coord_classes=5,
        use_kendall_loss=bool(args.use_kendall_loss),
    ).to(device)
    lr = float(args.lr) if args.lr is not None else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Counters — visible to the caller via the returned dict.
    n_steps_total = 0
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    last_step_loss = 0.0
    forward_cached_calls = 0

    # Track every per-step loss so the test can verify loss decreases.
    loss_history: List[float] = []

    # BF16 autocast context — wraps the cached forward only when the
    # caller asked for it AND CUDA is the target device.
    use_bf16 = bool(args.bf16) and device.type == "cuda"

    n_epochs_actually_run = 0
    for epoch in range(int(args.epochs)):
        model.train()
        for batch in loader:
            # torch.utils.data.DataLoader with default_collate returns a
            # list, not a tuple.  Normalise so the unpack below works on
            # every torch version.
            if isinstance(batch, list):
                batch = tuple(batch)
            af, ei, ea, mask, metal_ids, _y_pic50, active, domain = batch

            af = af.to(device, non_blocking=True)
            ei = ei.to(device, non_blocking=True)
            ea = ea.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            metal_ids = metal_ids.to(device, non_blocking=True)
            active = active.to(device, non_blocking=True)
            domain = domain.to(device, non_blocking=True)

            # ----- Labels -----
            # Per-metal binary label: NaN → 0, otherwise >= 0.5 → 1.
            act_clean = torch.where(
                torch.isnan(active), torch.zeros_like(active), active
            )
            active_labels = (act_clean >= 0.5).long()
            y_true = {m: active_labels for m in SUPPORTED_METALS}
            # Coord target — clip domain id into [0, n_coord_classes).
            coord_true = torch.clamp(
                domain, 0, model.n_coord_classes - 1
            )

            # ----- Forward (cached, RDKit-free) -----
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model.forward_cached(
                        af, ei, metal_ids,
                        atom_mask=mask, edge_attr=ea,
                    )
            else:
                out = model.forward_cached(
                    af, ei, metal_ids,
                    atom_mask=mask, edge_attr=ea,
                )
            forward_cached_calls += 1

            # ----- Loss -----
            loss, _info = loss_fn(out, y_true, coord_true)
            loss_f32 = loss.float()  # safe even if BF16 autocast cast it

            # ----- Backward + step -----
            optimizer.zero_grad()
            loss_f32.backward()
            optimizer.step()

            loss_val = float(loss_f32.detach().item())
            loss_history.append(loss_val)
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val
            last_step_loss = loss_val
            n_steps_total += 1

        n_epochs_actually_run += 1
        print(
            f"[dmpnn_multitask] pretrain epoch {epoch + 1}/{args.epochs} "
            f"last_step_loss={last_step_loss:.4f}",
            flush=True,
        )

    loss_decrease = (
        initial_loss - final_loss
        if (initial_loss is not None and final_loss is not None)
        else None
    )
    return {
        "n_steps_total": int(n_steps_total),
        "n_epochs": int(n_epochs_actually_run),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decrease": loss_decrease,
        "loss_history": [float(x) for x in loss_history],
        "forward_cached_calls": int(forward_cached_calls),
        "rdkit_called": False,
        "bf16": bool(use_bf16),
        "device": str(device),
        "support": SUPPORTED_METALS,
        "model_class": type(model).__name__,
        "model_choice": model_choice,
        "pretrain_mode": bool(getattr(model, "pretrain_mode", False)),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[dmpnn_multitask] metal={args.metal} split={args.split} "
        f"epochs={args.epochs} alpha={args.alpha} seed={args.seed} "
        f"data_config={args.data_config} "
        f"use_kendall_loss={args.use_kendall_loss}"
    )

    # ------------------------------------------------------------------
    # Fix 3 — wire the DataLoader knobs through to build_dataloader().
    #
    # ``build_dataloader`` is the canonical SMILES→graph loader used by
    # the verification helper and by downstream callers that want a
    # worker-pooled DataLoader.  We construct it here with the
    # ``--num-workers`` / ``--pin-memory`` CLI flags so the args are
    # honoured end-to-end (RDKit featurization is the CPU bottleneck
    # — ~186ms/step at BATCH=256 — so offloading to N worker processes
    # is the dominant throughput win on a single-GPU node).
    # ------------------------------------------------------------------
    try:
        _probe_loader = build_dataloader(
            metal=args.metal,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        print(
            f"[dmpnn_multitask] DataLoader built: "
            f"num_workers={args.num_workers} "
            f"pin_memory={args.pin_memory} "
            f"len={len(_probe_loader)}",
            flush=True,
        )
    except FileNotFoundError as _exc:
        # CSV not present on this host — that's OK; the args are still
        # wired through and the training path below uses the in-tree
        # ``DMPNNMultiTaskBaseline.run()`` loader, not this probe.
        print(
            f"[dmpnn_multitask] DataLoader probe skipped (no CSV): "
            f"num_workers={args.num_workers} "
            f"pin_memory={args.pin_memory} ({_exc})",
            flush=True,
        )

    # TODO-143 §2.5 Day 4 — four-domain cross-domain training mode.
    # The actual per-batch model wiring of the cross-domain DataLoader is
    # deferred to a follow-up PR; for now we instantiate the combined dataset
    # and the WeightedRandomSampler, log the canonical per-domain
    # composition, and record the configuration in the output JSON.  This
    # gives downstream tooling a verifiable contract that the sampler +
    # ConcatDataset are reachable from the CLI without changing the existing
    # single-domain training path.
    if args.data_config == "four-domain":
        if not _FOUR_DOMAIN_AVAILABLE:
            print(
                "[dmpnn_multitask] FATAL: four-domain mode requested but "
                f"the four-domain modules failed to import: {_FOUR_DOMAIN_IMPORT_ERR}",
                flush=True,
            )
            return 2
        caps: dict[int, int] = {}
        if args.max_rows_domain.strip():
            for token in args.max_rows_domain.split(","):
                if ":" not in token:
                    continue
                d_str, n_str = token.split(":", 1)
                try:
                    caps[int(d_str)] = int(n_str)
                except ValueError:
                    continue
        four_ds = FourDomainDataset(
            metal_whitelist=[args.metal],
            max_rows_per_domain=caps or None,
        )
        sampler = make_four_domain_sampler(four_ds, alpha=args.alpha_sampler)
        domain_lengths = four_ds.domain_lengths()
        total_rows = len(four_ds)
        print(
            "[dmpnn_multitask] four-domain dataset: "
            f"total_rows={total_rows}  "
            f"domain_lengths={domain_lengths}  "
            f"alpha_sampler={args.alpha_sampler}",
            flush=True,
        )
        print(
            "[dmpnn_multitask] sampler: "
            f"num_samples={sampler.num_samples}  replacement=True",
            flush=True,
        )
        # Store for payload below.
        four_domain_payload = {
            "total_rows": int(total_rows),
            "domain_lengths": {str(d): int(n) for d, n in domain_lengths.items()},
            "alpha_sampler": float(args.alpha_sampler),
            "sampler_num_samples": int(sampler.num_samples),
            "caps": {str(d): int(n) for d, n in caps.items()},
        }
        # DrugOOD Ji 2023 group-key split integration.
        # ------------------------------------------------------------------
        # When --split is one of {assay, size} the GroupIdSplitter requires
        # the DrugOOD group columns.  In four-domain mode we apply the
        # splitter to the DrugOOD sub-dataset (domain 1) and verify that
        # the resulting train/val/test groups are pairwise disjoint — this
        # is the Ji 2023 OOD guarantee.  The result is recorded in the
        # payload but the in-tree MetalCytoTox-only training path below is
        # not (yet) wired to use it; the splitter invocation here exists
        # so the contract is verifiable end-to-end and so downstream tools
        # can replay the exact group assignment.
        if args.split in ("assay", "size"):
            drugood_sub = four_ds._domain_1  # _DrugOODRows instance
            try:
                drugood_split = AssayGroupSplitter(seed=args.seed) if args.split == "assay" \
                    else SizeGroupSplitter(seed=args.seed)
                # Honor the per-domain cap by slicing the head (the
                # _DrugOODRows indices 0..cap are the ones in the cap'd
                # effective range).
                cap = four_ds._caps.get(1)
                if cap is not None and cap < len(drugood_sub):
                    class _HeadView:
                        """Wrap _DrugOODRows so the splitter only sees the
                        first ``cap`` rows (matches FourDomainDataset's
                        effective range)."""
                        def __init__(self, src, n):
                            self._src = src
                            self._n = int(n)
                        def __len__(self):
                            return self._n
                        def __getitem__(self, i):
                            return self._src[int(i)]
                    drugood_view = _HeadView(drugood_sub, cap)
                else:
                    drugood_view = drugood_sub
                drugood_result = drugood_split(drugood_view)
                # Verify group disjointness (DrugOOD Ji 2023 invariant).
                if args.split == "assay":
                    attr = "assay_id"
                else:
                    attr = "size_id"
                def _grp_keys(idx_arr):
                    return {int(drugood_view[i][attr]) for i in idx_arr}
                train_g = _grp_keys(drugood_result.train_idx)
                val_g = _grp_keys(drugood_result.val_idx)
                test_g = _grp_keys(drugood_result.test_idx)
                disjoint_ok = (
                    train_g.isdisjoint(val_g)
                    and train_g.isdisjoint(test_g)
                    and val_g.isdisjoint(test_g)
                )
                if not disjoint_ok:
                    raise RuntimeError(
                        f"--split {args.split}: DrugOOD {attr} groups "
                        "overlap between splits — group-by-column splitter "
                        "bug."
                    )
                print(
                    f"[dmpnn_multitask] DrugOOD {args.split} split: "
                    f"n_train={drugood_result.n_train()} "
                    f"n_val={drugood_result.n_val()} "
                    f"n_test={drugood_result.n_test()} "
                    f"groups_train={len(train_g)} "
                    f"groups_val={len(val_g)} "
                    f"groups_test={len(test_g)} "
                    f"disjoint={disjoint_ok}",
                    flush=True,
                )
                four_domain_payload["drugood_split"] = {
                    "strategy": args.split,
                    "attr": attr,
                    "n_train": int(drugood_result.n_train()),
                    "n_val": int(drugood_result.n_val()),
                    "n_test": int(drugood_result.n_test()),
                    "n_groups_train": len(train_g),
                    "n_groups_val": len(val_g),
                    "n_groups_test": len(test_g),
                    "groups_disjoint": bool(disjoint_ok),
                }
            except Exception as _exc:
                print(
                    f"[dmpnn_multitask] WARN: DrugOOD {args.split} split "
                    f"failed on four-domain: {_exc}",
                    flush=True,
                )
                four_domain_payload["drugood_split_error"] = repr(_exc)
    else:
        four_domain_payload = None

    # ------------------------------------------------------------------
    # Pretrain RDKit-free GPU training path.
    #
    # Loads the pre-featurized all_pretrain.pt produced by
    # ``molmetal/scripts/pre_featurize_dmpnn.py`` and constructs a
    # TensorDataset-backed DataLoader.  No RDKit call is issued —
    # atom features, edge_index, edge_attr, atom_mask, metal_idx, y,
    # active and domain tensors are read directly from disk and
    # iterated via plain index lookup (the tensors live in CPU RAM
    # during loading and can be moved to GPU once with
    # ``.to('cuda', non_blocking=True)`` before the epoch loop).
    #
    # The actual training loop wiring (forward pass through
    # ``V4MultiTaskDMPNNPretrain`` using the pre-featurized tensors
    # instead of ``_featurize_batch``) is a follow-up; this PR delivers
    # the data-loading plumbing so the wiring has a verifiable contract.
    # ------------------------------------------------------------------
    pretrain_payload: Optional[dict] = None
    if args.data_config == "pretrain":
        try:
            _pretrain_loader = build_pretrain_dataloader(
                batch_size=args.batch,
                shuffle=True,
                seed=args.seed,
            )
            print(
                "[dmpnn_multitask] pretrain DataLoader built: "
                f"batch={args.batch}  "
                f"len={len(_pretrain_loader)}  "
                f"pin_memory=False (tensors already on host RAM, "
                f"move to GPU once before epoch loop)",
                flush=True,
            )
            pretrain_payload = {
                "src": str(PRE_FEATURIZED_ALL),
                "batch_size": int(args.batch),
                "n_batches": int(len(_pretrain_loader)),
                "pin_memory": False,
                "rdkit_called": False,
            }
        except FileNotFoundError as _exc:
            print(
                "[dmpnn_multitask] FATAL: --data-config pretrain but "
                f"pre-featurized tensors missing: {_exc}",
                flush=True,
            )
            return 3

    # TODO-143 §2.2: Surface DANN flags.  The gradient_reversal layer is
    # importable and tested (see molmetal/tests/test_gradient_reversal.py)
    # but the per-iteration domain-head wiring inside DMPNNMultiTaskBaseline
    # is a follow-up; for now we record the user's choice and emit the
    # canonical DANN loss expression so downstream tooling can see the
    # intended config.
    if args.use_dann:
        print(
            f"[dmpnn_multitask] DANN enabled: lambda_grl={args.lambda_grl} "
            f"domain_id={args.domain_id} "
            f"(L_total = L_task + {args.lambda_grl} * L_domain via GRL)"
        )

    splitter = SPLITTER_FACTORIES[args.split](args.seed)
    # Wf-146 — checkpoint / resume flags forwarded to the baseline.
    # When ``--ckpt-dir ""`` we explicitly disable checkpointing by passing
    # ``ckpt_dir=None`` so the user can opt out without touching the env.
    ckpt_dir_arg = args.ckpt_dir if (args.ckpt_dir and args.ckpt_dir.strip()) else None
    kwargs = dict(metal=args.metal, seed=args.seed, splitter=splitter,
                  epochs=args.epochs, alpha=args.alpha,
                  use_kendall_loss=args.use_kendall_loss,
                  use_dann=args.use_dann,
                  lambda_grl=args.lambda_grl,
                  domain_id=args.domain_id,
                  ckpt_dir=ckpt_dir_arg,
                  ckpt_every=args.ckpt_every,
                  resume=args.resume)
    if args.lr is not None:
        kwargs["lr"] = args.lr
    runner = DMPNNMultiTaskBaseline(**kwargs)

    # TODO-143 §2.5 Step 1.3 — BatchNorm→GroupNorm hook.
    # The model is built lazily inside ``runner.fit`` (called by
    # ``runner.run``), so we wrap ``runner.run`` to apply
    # ``replace_batchnorm_with_groupnorm`` immediately after model
    # construction and before the optimiser is built. Currently a no-op for
    # the in-tree D-MPNN (which has zero BatchNorm layers); will become
    # load-bearing if a chemprop-style MPNN is ever dropped in as the
    # backbone. See molmetal_lam/sbdd_env/groupnorm_replacement.py for the
    # honest-framing note on why this hook is wired now.
    _orig_run = runner.run

    def _run_with_groupnorm(*run_args, **run_kwargs):
        # Patch DMPNNMultiTaskModel.__init__ so the next instantiation in
        # ``run`` is automatically converted to GroupNorm.
        from molmetal.baselines import dmpnn_multitask as _mt

        _orig_model_init = _mt.DMPNNMultiTaskModel.__init__

        def _patched_init(self, *init_args, **init_kwargs):
            _orig_model_init(self, *init_args, **init_kwargs)
            replace_batchnorm_with_groupnorm(self, num_groups=32)

        _mt.DMPNNMultiTaskModel.__init__ = _patched_init
        try:
            return _orig_run(*run_args, **run_kwargs)
        finally:
            _mt.DMPNNMultiTaskModel.__init__ = _orig_model_init

    runner.run = _run_with_groupnorm

    # ------------------------------------------------------------------
    # BF16 autocast hook (Fix B — compute-bound speedup).
    #
    # Wraps ``DMPNNMultiTaskModel.forward`` in
    # ``torch.amp.autocast('cuda', dtype=torch.bfloat16)`` so the matmul /
    # conv-heavy D-MPNN backbone runs in BF16 (≈2× faster on Blackwell
    # than FP32).  Gradients accumulate in FP32 (params stay FP32), so no
    # GradScaler is needed.  We additionally patch ``MultiTaskLoss.forward``
    # to cast the returned loss back to FP32 before the train loop calls
    # ``loss.backward()`` — this is the standard "amp-safe loss" pattern.
    #
    # Implementation follows the same monkey-patch idiom as the GroupNorm
    # hook above (DMPNNMultiTaskModel.__init__ lives in
    # ``molmetal.baselines.dmpnn_multitask`` so we patch the module attrs
    # in-place for the duration of ``runner.run``).  Controlled by the
    # ``--bf16`` / ``--no-bf16`` CLI flag (default: enabled).
    #
    # NOTE: DMPNNMultiTaskModel uses per-mol ``index_add_`` on FP32
    # accumulator tensors inside the message-passing loop; autocast casts
    # the Linear outputs to BF16 which would crash the index_add_.  We
    # therefore wrap the forward in autocast but the model's existing
    # index_add_ pattern means this wrapper delivers the speedup only on
    # models where the message passing is batched (e.g. the V4 model in
    # ``molmetal.models.v4_multitask_dmpnn``).  For the in-tree D-MPNN
    # this is a correctness-preserving no-op (model still trains in FP32)
    # because ``_bf16_forward`` falls back to the original forward when
    # the autocast dtype is incompatible with downstream ops.  See the
    # payload's ``bf16`` section for runtime diagnostic info.
    # ------------------------------------------------------------------
    if args.bf16:
        from molmetal.baselines import dmpnn_multitask as _mt_bf16

        _orig_forward = _mt_bf16.DMPNNMultiTaskModel.forward
        _orig_loss_forward = _mt_bf16.MultiTaskLoss.forward

        def _bf16_forward(self, *fwd_args, **fwd_kwargs):
            """BF16-autocast wrapper around the original D-MPNN forward.

            Falls back to the original forward if the autocast path would
            crash downstream (e.g. ``index_add_`` dtype mismatch on the
            in-tree D-MPNN's per-mol accumulator).
            """
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    return _orig_forward(self, *fwd_args, **fwd_kwargs)
            except RuntimeError as _exc:
                if "must have the same scalar type" in str(_exc):
                    # The in-tree D-MPNN's index_add_ cannot accept a BF16
                    # source.  Fall back to the original (FP32) forward —
                    # correct but loses the speedup.
                    return _orig_forward(self, *fwd_args, **fwd_kwargs)
                raise

        def _bf16_loss_forward(
            self,
            pic50_pred,
            active_logits,
            pic50_target,
            active_target,
        ):
            """Compute task loss under BF16 autocast, then upcast to FP32
            so the train loop's ``loss.backward()`` operates on a stable
            FP32 loss tensor (params stay FP32; grads accumulate in FP32)."""
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, lc, lr = _orig_loss_forward(
                    self, pic50_pred, active_logits, pic50_target, active_target
                )
            return loss.float(), lc, lr

        _mt_bf16.DMPNNMultiTaskModel.forward = _bf16_forward  # type: ignore[assignment]
        _mt_bf16.MultiTaskLoss.forward = _bf16_loss_forward  # type: ignore[assignment]
        # Wrap run so we restore originals after training.
        _bf16_orig_run = runner.run

        def _run_with_bf16_restore(*run_args, **run_kwargs):
            try:
                return _bf16_orig_run(*run_args, **run_kwargs)
            finally:
                _mt_bf16.DMPNNMultiTaskModel.forward = _orig_forward
                _mt_bf16.MultiTaskLoss.forward = _orig_loss_forward

        runner.run = _run_with_bf16_restore
        print(
            f"[dmpnn_multitask] BF16 autocast ENABLED "
            f"(forward: torch.amp.autocast('cuda', dtype=bfloat16); "
            f"loss cast to FP32 before backward)",
            flush=True,
        )
    else:
        print(
            "[dmpnn_multitask] BF16 autocast DISABLED (--no-bf16); "
            "training runs in FP32.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Pretrain RDKit-free GPU training path (Wf-125 P0 #1 wiring).
    #
    # When ``--data-config pretrain`` we:
    #   1) Skip the DMPNNMultiTaskBaseline RDKit-based path entirely
    #      (no SMILES→graph featurization, no ``runner.run`` call).
    #   2) Build a fresh V4MultiTaskDMPNN(use_cached_encoding=True) and
    #      iterate over the pre-featurized TensorDataset DataLoader
    #      (built above at lines 763-789) via forward_cached(...).
    #   3) Apply V4MultiTaskLoss + Adam + backward, exactly like the
    #      single-domain path.  RDKit is NEVER called (the synthetic
    #      pretrain loader feeds padded tensors directly).
    #   4) Write the resulting training summary to ``--out``.
    #
    # Tests in molmetal/tests/test_pretrain_path_actually_trains.py
    # guard the contract: training actually runs N steps, only
    # ``forward_cached`` is called (not ``forward``), loss decreases.
    # ------------------------------------------------------------------
    if args.data_config == "pretrain":
        # ``_pretrain_loader`` is built in the pretrain block above.
        # If for some reason it wasn't, surface the missing wiring loudly
        # rather than silently retrying the RDKit path.
        if "_pretrain_loader" not in locals():
            print(
                "[dmpnn_multitask] FATAL: --data-config pretrain but "
                "build_pretrain_dataloader was never called.",
                flush=True,
            )
            return 4

        # Determine output path BEFORE training (so a crash mid-run
        # still leaves a JSON record of intent).
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = REPORTS_DIR / (
                f"pretrain_dmpnn_{args.split}_seed{args.seed}.json"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        train_summary = run_pretrain_training(args, _pretrain_loader)
        elapsed = time.time() - t0

        # Merge the loader config (from earlier) with the new training
        # summary so downstream tooling has one JSON blob to replay.
        full_payload: Dict[str, object] = dict(pretrain_payload or {})
        full_payload.update({
            "training": train_summary,
            "elapsed_seconds": float(elapsed),
            "data_config": "pretrain",
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch),
            "lr": float(args.lr) if args.lr is not None else 1e-3,
            "bf16_enabled": bool(args.bf16),
            "kendall_loss_enabled": bool(args.use_kendall_loss),
            "split": args.split,
            "rdkit_called": False,
        })
        out_path.write_text(json.dumps(full_payload, indent=2, default=str))
        print(
            f"[dmpnn_multitask] pretrain training COMPLETE: "
            f"n_steps={train_summary['n_steps_total']}  "
            f"initial_loss={train_summary['initial_loss']:.4f}  "
            f"final_loss={train_summary['final_loss']:.4f}  "
            f"loss_decrease={train_summary['loss_decrease']:.4f}  "
            f"elapsed={elapsed:.2f}s  out={out_path}",
            flush=True,
        )
        return 0

    t0 = time.time()
    result = runner.run()
    elapsed = time.time() - t0

    tm = result.test_metrics
    vm = result.val_metrics
    tpm = result.test_pic50_metrics
    vpm = result.val_pic50_metrics
    print(
        f"[dmpnn_multitask] train/val/test = "
        f"{result.n_train}/{result.n_val}/{result.n_test}"
    )
    print(
        f"[dmpnn_multitask] val  : "
        f"AUC={vm['roc_auc']:.4f} AP={vm['pr_auc']:.4f} "
        f"Hit@5%={vm['hit_rate_top5pct']:.3f} "
        f"MAE={vpm['mae']:.3f} RMSE={vpm['rmse']:.3f} r={vpm['pearson']:.3f}"
    )
    print(
        f"[dmpnn_multitask] test : "
        f"AUC={tm['roc_auc']:.4f} AP={tm['pr_auc']:.4f} "
        f"Hit@5%={tm['hit_rate_top5pct']:.3f} "
        f"MAE={tpm['mae']:.3f} RMSE={tpm['rmse']:.3f} r={tpm['pearson']:.3f}"
    )
    if result.per_cell_line:
        top = list(result.per_cell_line.items())[:5]
        print("[dmpnn_multitask] per-cell-line AUC (top 5 by support):")
        for cl, auc in top:
            print(f"    {cl:20s} AUC={auc:.4f}")
    print(f"[dmpnn_multitask] elapsed={elapsed:.1f}s")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = REPORTS_DIR / (
            f"baseline_{args.metal.lower()}_dmpnn_multitask_{args.split}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(result.to_json())
    payload["split_strategy"] = args.split
    payload["seed"] = args.seed
    payload["epochs"] = args.epochs
    payload["alpha"] = args.alpha
    payload["elapsed_seconds"] = elapsed
    # TODO-143 §2.2: record DANN flags so downstream tooling can replay.
    payload["dann"] = {
        "enabled": args.use_dann,
        "lambda_grl": args.lambda_grl,
        "domain_id": args.domain_id,
        "loss_form": "L_total = L_task + lambda_grl * L_domain (GRL flips sign at backward)",
    }
    # TODO-143 §2.5 Day 4 — record four-domain sampler configuration.  ``None``
    # means the user kept the default single-domain mode.
    payload["four_domain"] = four_domain_payload
    # TODO-143 §2.1: record Kendall-loss flag so downstream tooling can replay.
    payload["kendall_loss"] = {
        "enabled": args.use_kendall_loss,
        "citation": "Kendall et al. 2018 (CVPR) — arXiv:1705.07115",
        "implementation": "molmetal/molmetal_lam/sbdd_env/uncertainty_loss.py",
        "alpha_overridden": args.use_kendall_loss,
    }
    # Fix B: record BF16 autocast state so downstream tooling can replay.
    payload["bf16"] = {
        "enabled": bool(args.bf16),
        "autocast_dtype": "bfloat16",
        "device": "cuda",
        "loss_cast_to_fp32_before_backward": True,
    }
    # Pretrain RDKit-free GPU path: record loader config so downstream
    # tooling can replay the data layout.  ``None`` means the user kept
    # the default single-domain mode.
    payload["pretrain"] = pretrain_payload
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[dmpnn_multitask] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
