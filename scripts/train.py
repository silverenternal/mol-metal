"""Training entry-point for MolFlow-Triton.

This script wires together the YAML config loader, a molecular dataset,
the EGNN-style :class:`VelocityNet`, the conditional flow-matching loss
and an Adam optimizer into a small training loop.  It is intentionally
framework-light:

* The device is auto-detected (``cuda`` when available, else ``cpu``).
  On CPU the ODE-solver / scatter-sum Triton kernels cannot run, so
  the script raises a clear error instead of silently falling back.
* Logging is done with ``print`` (no ``logging`` config yet).
* Checkpoints are written to ``checkpoints/model.pt`` at the end of
  every epoch.

Usage
-----

    python scripts/train.py \\
        --config configs/default.yaml \\
        --epochs 100 \\
        --batch-size 32 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Make the project root importable so ``models``, ``flow_matching``,
# ``triton_kernels`` and ``utils`` resolve when this file is run as
# ``python scripts/train.py`` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from flow_matching import ConditionalFlowMatchingLoss
from models import VelocityNet
from utils.config import load_config


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pt"

# How many optimization steps between eval passes.
EVAL_EVERY_N_STEPS = 25

# How many optimization steps between summary log lines.
SUMMARY_EVERY_N_STEPS = 100

# Tabular datasets (e.g. MetalCytoToxDB) don't carry 3-D coordinates; we
# project the descriptor vector into a small synthetic ``(N, 3)`` lattice
# so the EGNN-style velocity net has something to chew on.  This is a
# pipeline-only placeholder: a real 3D-aware dataset (QM9 / GEOM-Drugs)
# will replace this once the loading code lands.
_TABULAR_N_ATOMS = 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the MolFlow-Triton flow-matching velocity net.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Path to YAML config (default: {DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override ``train.epochs`` from the config.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override ``train.batch_size`` from the config.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override ``device`` from the config "
        "('cuda' or 'cpu').  When omitted, auto-detected via "
        "torch.cuda.is_available().",
    )
    p.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Override ``data.target_index`` from the config.  When set, "
        "the dataset is loaded with this QM9 target index and a "
        ":class:`models.conditioner.Conditioner` is built so the "
        "velocity net is conditioned on the per-molecule scalar label.",
    )
    return p


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(requested: str | None) -> torch.device:
    """Pick ``cuda`` (or its ROCm alias) when available, else ``cpu``.

    Raises a clear error when CUDA is requested but unavailable, instead
    of silently downgrading — silent fallbacks hide real hardware
    configuration issues on the ROCm box.
    """
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = requested.lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested device 'cuda' but torch.cuda.is_available() is "
                "False.  Check that the ROCm / CUDA toolchain is "
                "installed and that torch+triton-rocm were built against "
                "the matching runtime."
            )
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(
        f"Unsupported --device value {requested!r}; expected 'cuda' or 'cpu'."
    )


def _require_cuda_for_triton(device: torch.device) -> None:
    """The flow-matching ODE solver / scatter-sum live in Triton kernels.

    These kernels target AMD ROCm GPUs via ``triton-rocm`` and have no
    CPU fallback.  Raise early with an actionable error so users don't
    see a cryptic JIT failure later in the loop.
    """
    if device.type != "cuda":
        raise RuntimeError(
            f"Training requested device {device!r}, but the MolFlow-Triton "
            "Triton kernels (scatter-sum, fused LayerNorm, ODE step) "
            "have no CPU implementation.  Run on a CUDA / ROCm GPU or "
            "wait for the CPU fallback to land.  See README.md → "
            "'Running on CPU'."
        )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_dataset(cfg: dict, target_index_override: int | None = None):
    """Construct the dataset specified by ``cfg['data']['dataset']``.

    Supported values:

    * ``"metalcytotox"`` / ``"metalcytotoxdb"`` — synthetic QSAR
      toxicity CSV.  Each sample contributes ``(features, label)``
      and the collate projects the descriptor vector into a small
      ``(N_atoms, 3)`` lattice.
    * ``"qm9"`` — real 3D molecules parsed from QM9 with RDKit.  Each
      sample contributes ``(coords, atom_types, label)`` and the
      collate pads variable-size molecules to a fixed ``MAX_ATOMS``.

    ``target_index_override`` (when not ``None``) wins over
    ``cfg['data']['target_index']`` so the ``--target-index`` CLI flag
    can take precedence without rewriting the config file.

    Returns ``(dataset, effective_target_index)`` so the caller can
    inspect which QM9 target was actually used (if any) and decide
    whether to build a :class:`models.conditioner.Conditioner`.

    Any other value raises :class:`NotImplementedError` so the user
    sees a clear message instead of a deep stack trace.
    """
    data_cfg = cfg.get("data", {}) or {}
    name = str(data_cfg.get("dataset", "qm9")).lower()

    if name in ("metalcytotox", "metalcytotoxdb"):
        from data.mol_dataset import MetalCytoToxDataset

        csv_path = PROJECT_ROOT / "data" / "MetalCytoToxDB.csv"
        return MetalCytoToxDataset(root=csv_path.parent, csv_name=csv_path.name), None

    if name == "qm9":
        from data.mol_dataset import QM9Dataset

        split = str(data_cfg.get("split", "train")).lower()
        cfg_target_index = data_cfg.get("target_index", None)
        target_index = (
            target_index_override
            if target_index_override is not None
            else cfg_target_index
        )
        if target_index is not None:
            target_index = int(target_index)
        return QM9Dataset(split=split, target_index=target_index), target_index

    raise NotImplementedError(
        f"Unknown dataset {name!r}.  Available: 'qm9' (default, real "
        f"3D molecules) and 'metalcytotox' (synthetic QSAR)."
    )


# ---------------------------------------------------------------------------
# Batch shaping: 3D-aware samples (coords, atom_types) -> batch dict
# ---------------------------------------------------------------------------
# QM9 (with explicit hydrogens) tops out at 29 atoms per molecule, so a
# fixed cap keeps the collate path branch-free and matches the
# encoder's expected (B, N, H) layout.
_QM9_MAX_ATOMS = 29


# Distance cutoff (Å) for the per-molecule edge graph.  5 Å is the
# common choice in EGNN/SchNet literature; small enough to keep the
# message passing local, large enough that adjacent atoms always
# share an edge in QM9.
_QM9_EDGE_CUTOFF = 5.0


def compute_qm9_position_scale(dataset, n_samples: int = 1000) -> float:
    """Estimate the per-atom coordinate std on a sample of the dataset.

    Used to normalise coordinates to unit variance for the flow
    matching task.  Computed once at training startup and stored in
    the checkpoint so :mod:`scripts.generate` can denormalize the
    integrated state back to Ångström before SMILES decoding.
    """
    from data.mol_dataset import QM9Dataset as _QM9Dataset

    if not isinstance(dataset, _QM9Dataset):
        return 1.0
    n = min(n_samples, len(dataset))
    stds: list[float] = []
    for i in range(n):
        c = dataset[i].coords
        if c.shape[0] < 2:
            continue
        c_centered = c - c.mean(dim=0, keepdim=True)
        stds.append(float(c_centered.std().item()))
    if not stds:
        return 1.0
    import statistics
    return float(statistics.median(stds))


def collate_qm9(
    items,
    device: torch.device,
    position_scale: float = 1.0,
    edge_cutoff: float = _QM9_EDGE_CUTOFF,
) -> dict:
    """Pad QM9 :class:`MoleculeSample` and normalize for flow-matching.

    Three transforms happen here:

    1. **Per-molecule centering**.  Each molecule is shifted so its
       centroid over real atoms is at the origin.  This makes the
       flow-matching task translation-invariant — without it the model
       has to learn that the absolute position of a molecule carries
       no information.
    2. **Global scaling**.  All coordinates are divided by a fixed
       ``position_scale`` (computed once at training startup from a
       sample of the training set), so the resulting positions have
       unit variance.  This keeps the velocity net's
       ``phi_ij * (x_j - x_i)`` products on a sane scale and prevents
       the runaway that a raw-Ångström coordinate system caused
       before normalization.  Using a fixed scale (rather than
       per-batch) keeps train and generation on the same coordinate
       system so :mod:`scripts.generate` can denormalize the
       integrated state back to Ångström before SMILES decoding.
    3. **Distance-cutoff edges**.  For each molecule we only keep
       edges with ``dist < edge_cutoff`` (no self-loops, no
       padding-to-padding edges, no padding-to-real edges).  This
       replaces the previous fully-connected graph; padding atoms
       naturally end up isolated.  ``edge_cutoff`` is read from
       ``cfg.train.edge_cutoff`` (default ``_QM9_EDGE_CUTOFF`` = 5.0).

    Returns a dict with keys ``positions``, ``atomic_numbers``,
    ``edge_index``, ``edge_mask``, ``node_mask`` and ``position_scale``
    (the global std, so :mod:`scripts.generate` can denormalize the
    final coordinates before decoding).
    """
    b = len(items)
    max_atoms = _QM9_MAX_ATOMS

    positions = torch.zeros(b, max_atoms, 3, dtype=torch.float32)
    atomic_numbers = torch.zeros(b, max_atoms, dtype=torch.long)
    node_mask = torch.zeros(b, max_atoms, dtype=torch.bool)

    # Conditional path: only populated when the dataset was constructed
    # with a ``target_index`` so each sample carries a (1,) scalar
    # label.  ``None`` for the unconditional dataset (12-D label
    # concatenated or no label at all).
    cond_list: list[torch.Tensor] = []
    has_cond = bool(items) and getattr(items[0], "label", None) is not None and tuple(
        items[0].label.shape
    ) == (1,)

    for i, s in enumerate(items):
        n = min(s.coords.shape[0], max_atoms)
        positions[i, :n] = s.coords[:n].to(torch.float32)
        atomic_numbers[i, :n] = s.atom_types[:n].to(torch.long)
        node_mask[i, :n] = True
        if has_cond:
            cond_list.append(s.label.to(torch.float32).reshape(1))

    positions = positions.to(device)
    atomic_numbers = atomic_numbers.to(device)
    node_mask = node_mask.to(device)

    # (1) Per-molecule centering over real atoms only.
    n_real = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    masked_sum = (positions * node_mask.unsqueeze(-1).float()).sum(dim=1)  # (B, 3)
    centroid = masked_sum / n_real                                       # (B, 3)
    positions = (positions - centroid.unsqueeze(1)) * node_mask.unsqueeze(-1)

    # (2) Global scaling by a fixed ``position_scale`` (computed once
    # at training startup).  Using a constant rather than a per-batch
    # std means train and generation are on the same coordinate
    # system, so :mod:`scripts.generate` can denormalize.
    positions = positions / float(position_scale)

    # (3) Distance-cutoff edges.
    dists = torch.cdist(positions, positions)                            # (B, N, N)
    real_pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)         # (B, N, N)
    no_self = ~torch.eye(max_atoms, dtype=torch.bool, device=device).unsqueeze(0)
    cutoff_mask = dists < float(edge_cutoff)
    edge_valid = real_pair & no_self & cutoff_mask                      # (B, N, N)

    # Pack to (B, 2, E_max).  E_max is the worst case (fully connected
    # minus self-loops); with the cutoff most molecules use far fewer.
    e_max = max_atoms * (max_atoms - 1)
    src_all = torch.arange(max_atoms, device=device).view(1, max_atoms, 1).expand(
        b, max_atoms, max_atoms
    )
    dst_all = src_all.transpose(1, 2)
    edge_src = src_all[edge_valid]                                       # (total_real,)
    edge_dst = dst_all[edge_valid]
    # Bucket edges back per-batch by counting how many each batch
    # element contributed.  ``scatter_add`` into a (b, e_max) buffer
    # is awkward; we use a simple cumulative-index approach.
    counts = edge_valid.view(b, -1).sum(dim=1)                          # (B,)
    # Build (B, e_max) tensors filled with the first valid edge per row.
    edge_src_b = torch.zeros(b, e_max, dtype=torch.long, device=device)
    edge_dst_b = torch.zeros(b, e_max, dtype=torch.long, device=device)
    edge_mask_b = torch.zeros(b, e_max, dtype=torch.bool, device=device)
    cursor = 0
    for i in range(b):
        n_e = int(counts[i].item())
        if n_e > 0:
            edge_src_b[i, :n_e] = edge_src[cursor : cursor + n_e]
            edge_dst_b[i, :n_e] = edge_dst[cursor : cursor + n_e]
            edge_mask_b[i, :n_e] = True
            cursor += n_e
    edge_index = torch.stack([edge_src_b, edge_dst_b], dim=1)           # (B, 2, e_max)

    batch = {
        "positions": positions,
        "edge_index": edge_index,
        "edge_mask": edge_mask_b,
        "atomic_numbers": atomic_numbers,
        "node_mask": node_mask,
        "position_scale": float(position_scale),
    }
    if has_cond:
        batch["cond"] = torch.stack(cond_list, dim=0).to(device)
    return batch


# ---------------------------------------------------------------------------
# Batch shaping: tabular (F,) -> 3D-aware (B, N, 3)
# ---------------------------------------------------------------------------
def tabular_sample_to_batch(
    features: torch.Tensor,
    n_atoms: int,
    device: torch.device,
) -> dict:
    """Reshape a flat descriptor vector into the inputs the velocity net wants.

    The ``VelocityNet`` / EGNN layers expect ``(B, N, 3)`` positions and
    a graph topology.  For tabular inputs we project the descriptor
    into a small fully-connected lattice of ``n_atoms`` nodes (a
    pipeline-only placeholder — see ``_TABULAR_N_ATOMS``).

    Returns
    -------
    dict with keys ``positions`` ``(B, N, 3)``, ``edge_index``
    ``(B, 2, N*N)`` and ``atomic_numbers`` ``(B, N)``.
    """
    if features.dim() == 1:
        features = features.unsqueeze(0)
    b = features.shape[0]

    # Pad / truncate so the reshape to (b, n_atoms, 3) is exact.
    needed = n_atoms * 3
    f = features
    if f.shape[-1] < needed:
        pad = torch.zeros(b, needed - f.shape[-1], dtype=f.dtype)
        f = torch.cat([f, pad], dim=-1)
    elif f.shape[-1] > needed:
        f = f[..., :needed]

    positions = f.view(b, n_atoms, 3).to(device)

    # Fully-connected edge index (no self-loops) per batch element.
    idx = torch.arange(n_atoms, device=device)
    src = idx.view(1, n_atoms, 1).expand(b, n_atoms, n_atoms)
    dst = idx.view(1, 1, n_atoms).expand(b, n_atoms, n_atoms)
    mask = src != dst
    src = src[mask].view(b, -1)
    dst = dst[mask].view(b, -1)
    edge_index = torch.stack([src, dst], dim=1)  # (B, 2, N*(N-1))

    atomic_numbers = torch.ones(b, n_atoms, dtype=torch.long, device=device)

    return {
        "positions": positions,
        "edge_index": edge_index,
        "atomic_numbers": atomic_numbers,
    }


def collate_tabular(items, device: torch.device):
    """Collate a list of :class:`MoleculeSample` into a 3D-shaped batch."""
    feats = torch.stack([s.features for s in items if s.features is not None], dim=0)
    return collate_tabular_items(feats, device)


def collate_tabular_items(feats: torch.Tensor, device: torch.device):
    """Reshape a stacked ``(B, F)`` feature tensor into a 3D batch dict."""
    batch = tabular_sample_to_batch(feats, _TABULAR_N_ATOMS, device)
    return batch


# ---------------------------------------------------------------------------
# Velocity net construction
# ---------------------------------------------------------------------------
def build_velocity_net(
    cfg: dict, device: torch.device, cond_dim: int = 0
) -> VelocityNet:
    """Build the :class:`VelocityNet`.

    ``cond_dim`` is forwarded to the network so the conditional path
    (``VelocityNet(... cond_dim=...)``) is enabled when the caller has
    built a :class:`Conditioner`.  When ``cond_dim == 0`` the
    unconditional path is taken (the network simply ignores ``cond=``
    arguments and substitutes the time embedding in their place).
    """
    model_cfg = cfg.get("model", {}) or {}
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    n_layers = int(model_cfg.get("n_layers", 4))
    net = VelocityNet(hidden_dim=hidden_dim, n_layers=n_layers, cond_dim=cond_dim)
    return net.to(device)


def build_conditioner(
    cfg: dict, device: torch.device, n_scalar_channels: int = 1
) -> "Conditioner":
    """Build the :class:`Conditioner` whose ``hidden_dim`` matches the velocity net.

    The conditioner encodes raw scalar physical / molecular
    properties (here a single QM9 target per sample) into a dense
    vector that is injected into every EGNN layer via
    ``VelocityNet.cond_proj``.  ``n_scalar_channels`` defaults to 1
    to match :class:`QM9Dataset(target_index=k)``'s ``label`` shape of
    ``(1,)``.
    """
    from models import Conditioner

    model_cfg = cfg.get("model", {}) or {}
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    n_layers = int(model_cfg.get("conditioner_n_layers", 2))
    cond = Conditioner(
        n_scalar_channels=n_scalar_channels,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    return cond.to(device)


def build_encoder(cfg: dict, device: torch.device) -> "MolEncoder":
    """Build the :class:`MolEncoder` whose hidden_dim matches the velocity net.

    The encoder is checkpointed alongside the velocity net so that
    :mod:`scripts.generate` can re-use the trained hidden states
    instead of feeding the sampler with a randomly-initialised
    encoder.  ``max_atomic_number`` defaults to 20 (covering H..Ca)
    and ``encoder_n_layers`` defaults to 2 message-passing rounds.
    """
    from models import MolEncoder

    model_cfg = cfg.get("model", {}) or {}
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    n_layers = int(model_cfg.get("encoder_n_layers", 2))
    max_atomic_number = int(model_cfg.get("max_atomic_number", 20))
    encoder = MolEncoder(
        max_atomic_number=max_atomic_number,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    return encoder.to(device)


def encoder_config_from(cfg: dict) -> dict:
    """Subset of the config that fully describes the encoder architecture."""
    model_cfg = cfg.get("model", {}) or {}
    return {
        "max_atomic_number": int(model_cfg.get("max_atomic_number", 20)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "encoder_n_layers": int(model_cfg.get("encoder_n_layers", 2)),
    }


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------
def evaluate(
    loss_fn: ConditionalFlowMatchingLoss,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 4,
    ema_state_dict: dict | None = None,
) -> float:
    """Quick validation pass — averaged loss over ``max_batches`` batches.

    When ``ema_state_dict`` is provided the velocity net is swapped to
    the EMA shadow weights for the duration of the eval pass (and
    restored afterwards).  This gives a stable eval signal that is not
    polluted by the most recent (noisy) optimizer step.

    The loss function owns the encoder now; we let it compute
    ``h_node`` internally rather than passing a pre-computed one.
    """
    loss_fn.eval()
    velocity_net = loss_fn.model
    # Swap velocity net -> EMA shadow weights (if any) for the eval pass.
    backup: dict | None = None
    if ema_state_dict is not None:
        backup = {k: v.detach().clone() for k, v in velocity_net.state_dict().items()}
        velocity_net.load_state_dict(ema_state_dict)
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x1 = batch["positions"]
            x0 = x1 + 0.1 * torch.randn_like(x1) * x1.std()
            out = loss_fn(
                x0=x0,
                x1=x1,
                cond=batch.get("cond"),
                atomic_numbers=batch["atomic_numbers"],
                edge_index=batch["edge_index"],
                edge_mask=batch["edge_mask"],
                node_mask=batch["node_mask"],
            )
            total += float(out.loss.item())
            n += 1
    if backup is not None:
        velocity_net.load_state_dict(backup)
    loss_fn.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: dict, args: argparse.Namespace) -> None:
    device = resolve_device(args.device or cfg.get("device"))
    _require_cuda_for_triton(device)
    print(f"[train] device = {device}")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    epochs = int(args.epochs if args.epochs is not None else cfg.get("train", {}).get("epochs", 100))
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else cfg.get("train", {}).get("batch_size", 32)
    )
    lr = float(cfg.get("train", {}).get("lr", 1.0e-4))
    warmup_steps = int(cfg.get("train", {}).get("warmup_steps", 200))
    grad_clip_norm = float(cfg.get("train", {}).get("grad_clip_norm", 1.0))
    ema_decay = float(cfg.get("train", {}).get("ema_decay", 0.999))
    edge_cutoff = float(cfg.get("train", {}).get("edge_cutoff", _QM9_EDGE_CUTOFF))
    print(f"[train] epochs={epochs} batch_size={batch_size} lr={lr} warmup_steps={warmup_steps} grad_clip_norm={grad_clip_norm} ema_decay={ema_decay} edge_cutoff={edge_cutoff}")

    # 1. dataset + loader ----------------------------------------------------
    dataset, target_index_eff = build_dataset(cfg, target_index_override=args.target_index)
    print(f"[train] dataset = {type(dataset).__name__} (size={len(dataset)})")
    if target_index_eff is not None:
        from data.qm9 import QM9Dataset as _CanonicalQM9
        target_name = _CanonicalQM9.TARGETS[target_index_eff] if hasattr(
            _CanonicalQM9, "TARGETS"
        ) else str(target_index_eff)
        print(
            f"[train] conditional path enabled on QM9 target index "
            f"{target_index_eff} ({target_name}); a Conditioner will be "
            f"built and the per-sample (B, 1) label will be threaded "
            f"through the loss."
        )

    from data.mol_dataset import QM9Dataset as _QM9Dataset

    # Compute (or default) the global position scale for QM9.
    position_scale = compute_qm9_position_scale(dataset) if isinstance(
        dataset, _QM9Dataset
    ) else 1.0
    print(f"[train] position_scale = {position_scale:.4f}")

    def _collate(items):
        # Pick the collate by dataset type so the training loop stays
        # dataset-agnostic below this line.
        if isinstance(dataset, _QM9Dataset):
            return collate_qm9(
                items,
                device=device,
                position_scale=position_scale,
                edge_cutoff=edge_cutoff,
            )
        # Tabular (MetalCytoToxDB).  Heavy normalization: raw
        # descriptors are ~10^2, so divide them down to keep projected
        # positions on a tiny lattice scale.
        feats = torch.stack([s.features for s in items if s.features is not None], dim=0)
        feats = feats / 1000.0
        return collate_tabular_items(feats, device=device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate,
    )

    # 2. model + encoder + loss + optimizer ---------------------------------
    # When conditioning on a QM9 target, the velocity net needs
    # ``cond_dim == hidden_dim`` (the conditioner's output width) so it
    # can build ``cond_proj``.  Otherwise we keep the unconditional path.
    hidden_dim = int((cfg.get("model", {}) or {}).get("hidden_dim", 128))
    cond_dim = hidden_dim if target_index_eff is not None else 0
    velocity_net = build_velocity_net(cfg, device, cond_dim=cond_dim)
    encoder = build_encoder(cfg, device)
    conditioner = (
        build_conditioner(cfg, device, n_scalar_channels=1)
        if target_index_eff is not None
        else None
    )
    n_params = (
        sum(p.numel() for p in velocity_net.parameters())
        + sum(p.numel() for p in encoder.parameters())
        + (sum(p.numel() for p in conditioner.parameters()) if conditioner is not None else 0)
    )
    print(f"[train] params (velocity+encoder{'+conditioner' if conditioner is not None else ''}) = {n_params:,}")

    # Wire the encoder (and conditioner) into the loss so it computes
    # ``h_node`` internally from (atomic_numbers, positions, edge_index)
    # at every step, and runs the raw (B, 1) scalar ``cond`` through the
    # conditioner before forwarding it to the velocity net.  This also
    # trains the conditioner end-to-end via backprop through the loss.
    loss_fn = ConditionalFlowMatchingLoss(
        model=velocity_net, encoder=encoder, conditioner=conditioner
    ).to(device)
    optimizer_params = list(velocity_net.parameters()) + list(encoder.parameters())
    if conditioner is not None:
        optimizer_params += list(conditioner.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)

    # Exponential moving average of velocity_net weights.  Initialised to
    # the live parameters so the very first eval / checkpoint is a
    # faithful copy.  Updated in-place every optimizer.step; used for
    # the eval pass and persisted in the checkpoint under
    # ``velocity_net_ema_state_dict`` so :mod:`scripts.generate` can
    # prefer the smoother shadow over the noisy live weights.
    velocity_net_ema: dict[str, torch.Tensor] = {
        name: param.detach().clone()
        for name, param in velocity_net.named_parameters()
    }
    print(f"[train] EMA shadow initialised for {len(velocity_net_ema)} velocity_net parameters "
          f"(decay={ema_decay})")

    # 3. training loop -------------------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    n_batches = len(loader)
    total_steps = epochs * n_batches

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    running_loss_mean = 0.0
    running_loss_count = 0
    initial_loss = None
    final_loss = None
    # Sliding window of the last 100 loss values for the 100-step summary line.
    recent_window: list[float] = []
    for epoch in range(1, epochs + 1):
        velocity_net.train()
        encoder.train()
        epoch_loss = 0.0
        n_batches_epoch = 0
        t0 = time.time()
        for batch in loader:
            x1 = batch["positions"]
            # Match x0 scale to x1 so the model never sees OOD scales.
            x0 = x1 + 0.1 * torch.randn_like(x1) * x1.std()
            # Let the loss function compute h_node via the encoder.
            # ``cond`` is the per-sample (B, 1) QM9 label when the
            # conditional path is enabled, otherwise ``None``.
            out = loss_fn(
                x0=x0,
                x1=x1,
                cond=batch.get("cond"),
                atomic_numbers=batch["atomic_numbers"],
                edge_index=batch["edge_index"],
                edge_mask=batch["edge_mask"],
                node_mask=batch["node_mask"],
            )
            loss = out.loss
            loss_val = float(loss.item())

            if initial_loss is None:
                initial_loss = loss_val

            optimizer.zero_grad()
            loss.backward()

            params_with_grads = list(velocity_net.parameters()) + list(encoder.parameters())
            if conditioner is not None:
                params_with_grads += list(conditioner.parameters())

            has_nan_inf = False
            for p in params_with_grads:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    has_nan_inf = True
                    break

            if has_nan_inf:
                print(f"[train] step={global_step + 1} SKIPPED (nan gradient)")
                optimizer.zero_grad()
                epoch_loss += loss_val
                n_batches_epoch += 1
                global_step += 1
                scheduler.step()
                continue

            torch.nn.utils.clip_grad_norm_(params_with_grads, grad_clip_norm)

            # Loss spike detection: maintain running mean and halve lr on spikes.
            spike_halved = False
            if running_loss_count > 0:
                if loss_val > 5.0 * running_loss_mean and loss_val > 100.0:
                    for pg in optimizer.param_groups:
                        pg["lr"] *= 0.5
                    spike_halved = True

            optimizer.step()
            scheduler.step()

            if spike_halved:
                # Reset to scheduled lr so the halving is one-shot for this step.
                scheduled = lr_lambda(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * scheduled

            # EMA update for velocity_net parameters: shadow = decay*shadow + (1-decay)*param.
            # Done in-place on the EMA tensors (no autograd) so the live
            # parameters remain the source of gradients for backprop.
            with torch.no_grad():
                for name, param in velocity_net.named_parameters():
                    ema_t = velocity_net_ema[name]
                    ema_t.mul_(ema_decay).add_(param.detach(), alpha=1.0 - ema_decay)

            # Update running mean (use the actual loss seen this step).
            running_loss_count += 1
            running_loss_mean += (loss_val - running_loss_mean) / running_loss_count

            # Track a sliding-window loss mean over the last 100 optimizer
            # steps for the 100-step summary line.
            recent_window.append(loss_val)
            if len(recent_window) > 100:
                recent_window.pop(0)

            epoch_loss += loss_val
            n_batches_epoch += 1
            global_step += 1
            final_loss = loss_val

            current_lr = optimizer.param_groups[0]["lr"]
            if global_step % EVAL_EVERY_N_STEPS == 0:
                eval_loss = evaluate(
                    loss_fn, loader, device, ema_state_dict=velocity_net_ema
                )
                print(
                    f"[train] step={global_step} epoch={epoch} "
                    f"train_loss={epoch_loss / max(n_batches_epoch, 1):.6f} "
                    f"eval_loss={eval_loss:.6f} "
                    f"lr={current_lr:.6e}"
                )

            # 100-step summary: EMA train loss, eval loss (under EMA
            # weights), current lr, and the EMA-decayed velocity-net
            # weight-norm so we can see whether the EMA shadow is
            # still drifting.
            if global_step % SUMMARY_EVERY_N_STEPS == 0:
                train_loss_ema = (
                    sum(recent_window) / len(recent_window)
                    if recent_window
                    else float("nan")
                )
                eval_loss = evaluate(
                    loss_fn, loader, device, ema_state_dict=velocity_net_ema
                )
                ema_weight_norm = float(
                    torch.sqrt(
                        sum(t.detach().pow(2).sum() for t in velocity_net_ema.values())
                    ).item()
                )
                print(
                    f"[train] summary step={global_step} epoch={epoch} "
                    f"train_loss_ema100={train_loss_ema:.6f} "
                    f"eval_loss={eval_loss:.6f} "
                    f"lr={current_lr:.6e} "
                    f"ema_w_norm={ema_weight_norm:.4f}"
                )

        avg = epoch_loss / max(n_batches_epoch, 1)
        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[train] epoch {epoch}/{epochs}  avg_loss={avg:.6f}  ({dt:.1f}s) "
            f"lr={current_lr:.6e}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": velocity_net.state_dict(),
            "velocity_net_ema_state_dict": {
                name: t.detach().clone() for name, t in velocity_net_ema.items()
            },
            "encoder_state_dict": encoder.state_dict(),
            "encoder_config": encoder_config_from(cfg),
            "position_scale": float(position_scale),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        }
        if conditioner is not None:
            ckpt["conditioner_state_dict"] = conditioner.state_dict()
            ckpt["conditioner_config"] = {
                "n_scalar_channels": 1,
                "hidden_dim": hidden_dim,
                "n_layers": int(
                    (cfg.get("model", {}) or {}).get("conditioner_n_layers", 2)
                ),
            }
        torch.save(ckpt, CHECKPOINT_PATH)
        print(f"[train] checkpoint saved -> {CHECKPOINT_PATH}")

    return initial_loss, final_loss if final_loss is not None else (epoch_loss / max(n_batches_epoch, 1)), optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = args.config or os.environ.get("MOLFLOW_CONFIG")
    cfg = load_config(config_path)
    result = train(cfg, args)
    if result is not None:
        initial_loss, final_loss, lr_at_end = result
        print(f"[train] summary initial_loss={initial_loss:.6f} final_loss={final_loss:.6f} lr_at_end={lr_at_end:.6e}")


if __name__ == "__main__":
    main()
