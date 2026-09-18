"""Sampling / generation entry-point for MolFlow-Triton.

Loads a trained ``VelocityNet`` from a checkpoint, builds a
:class:`FlowMatchingSampler`, draws ``--n-samples`` configurations by
integrating the learned ODE from noise to data, and reports Validity
and Uniqueness via :mod:`utils.metrics`.

The generated positions are decoded into canonical SMILES by
``utils.chem_utils.positions_to_smiles``: atom identities come from the
``atomic_numbers`` tensor, bonds are perceived from interatomic
distances via bond-length thresholds, and the resulting graph is
canonicalised with RDKit.  ``positions_to_smiles`` below is a thin
wrapper kept for backwards compatibility with existing call sites.

Usage
-----

    python scripts/generate.py \\
        --checkpoint checkpoints/model.pt \\
        --n-samples 64 \\
        --n-steps 100 \\
        --method rk4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable so ``models``, ``flow_matching``,
# ``triton_kernels`` and ``utils`` resolve when this file is run as
# ``python scripts/generate.py`` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from flow_matching import FlowMatchingSampler
from models import VelocityNet
from utils.chem_utils import positions_to_smiles as chem_positions_to_smiles
from utils.config import load_config
from utils.metrics import uniqueness, validity


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "model.pt"
DEFAULT_N_SAMPLES = 64
# Default number of atoms per generated molecule.  Matches the fixed
# cap used by scripts/train.py::collate_qm9 (QM9-with-hydrogens tops
# out at 29 atoms, so the training loop pads every batch to 29).
# Setting this to a different value here would break the velocity
# net's hidden_dim contract.
DEFAULT_N_ATOMS = 29


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = requested.lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested device 'cuda' but torch.cuda.is_available() is "
                "False.  Check the ROCm / CUDA toolchain."
            )
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(
        f"Unsupported --device value {requested!r}; expected 'cuda' or 'cpu'."
    )


def _require_cuda_for_triton(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            f"Generation requested device {device!r}, but the MolFlow-Triton "
            "ODE-solver / scatter-sum Triton kernels have no CPU "
            "fallback.  Run on a CUDA / ROCm GPU or wait for the CPU "
            "fallback.  See README.md -> 'Running on CPU'."
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_velocity_net(
    checkpoint_path: Path, device: torch.device
):
    """Rebuild a ``VelocityNet`` from a checkpoint dict.

    The checkpoint layout (see ``scripts/train.py``) is::

        {
            "epoch": int,
            "model_state_dict": OrderedDict,
            "velocity_net_ema_state_dict": OrderedDict | absent (older ckpts),
            "encoder_state_dict": ... | absent (older checkpoints),
            "encoder_config": {...} | absent,
            "optimizer_state_dict": ...,
            "config": dict,
        }

    When the checkpoint carries ``velocity_net_ema_state_dict`` (the
    smoother shadow updated every optimizer.step during training)
    we prefer it over ``model_state_dict``; otherwise we fall back to
    the live weights so older checkpoints keep working.

    Returns ``(velocity_net, blob)`` so the caller can also load the
    encoder in a second pass without re-reading the file.  Older
    checkpoints that lack ``encoder_state_dict`` still load the
    velocity net correctly; the caller is then responsible for
    falling back to a randomly-initialised encoder (or raising).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}.  Run scripts/train.py "
            "first or pass --checkpoint."
        )

    blob = torch.load(checkpoint_path, map_location=device)
    cfg = blob.get("config") if isinstance(blob, dict) else None
    model_cfg = (cfg or {}).get("model", {}) if cfg else {}
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    n_layers = int(model_cfg.get("n_layers", 4))

    if isinstance(blob, dict) and "velocity_net_ema_state_dict" in blob:
        state = blob["velocity_net_ema_state_dict"]
        print("[generate] using EMA velocity_net weights from checkpoint")
    else:
        state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
        print("[generate] using live velocity_net weights (no EMA in checkpoint)")
    # Detect ``cond_dim`` from the saved weights: ``cond_proj.0.weight``
    # has shape ``(hidden_dim, cond_dim)`` when conditioning is enabled,
    # or is absent entirely in the unconditional case.
    cond_dim = 0
    if isinstance(state, dict) and "cond_proj.0.weight" in state:
        cond_dim = int(state["cond_proj.0.weight"].shape[1])
        print(
            f"[generate] conditional velocity net detected "
            f"(cond_dim={cond_dim}); will load conditioner"
        )

    net = VelocityNet(hidden_dim=hidden_dim, n_layers=n_layers, cond_dim=cond_dim)
    net.load_state_dict(state)
    net = net.to(device)
    net.eval()
    return net, blob


def load_encoder(checkpoint_blob, device: torch.device, cfg: dict):
    """Rebuild the trained :class:`MolEncoder` from a checkpoint.

    Falls back to a freshly-initialised encoder (matching the
    velocity net's hidden_dim) when the checkpoint was produced by
    an older train.py that did not save encoder weights.  A warning
    is emitted so the user knows generation is using random hidden
    states.
    """
    from models import MolEncoder

    encoder_cfg: dict | None = None
    state: dict | None = None
    if checkpoint_blob is not None and isinstance(checkpoint_blob, dict):
        encoder_cfg = checkpoint_blob.get("encoder_config")
        state = checkpoint_blob.get("encoder_state_dict")

    if state is not None and encoder_cfg is not None:
        encoder = MolEncoder(
            max_atomic_number=int(encoder_cfg.get("max_atomic_number", 20)),
            hidden_dim=int(encoder_cfg.get("hidden_dim", 128)),
            n_layers=int(encoder_cfg.get("encoder_n_layers", 2)),
        )
        encoder.load_state_dict(state)
        print(
            "[generate] loaded trained encoder "
            f"(max_atomic_number={encoder_cfg.get('max_atomic_number')}, "
            f"hidden_dim={encoder_cfg.get('hidden_dim')}, "
            f"n_layers={encoder_cfg.get('encoder_n_layers')})"
        )
    else:
        model_cfg = cfg.get("model", {}) or {}
        encoder = MolEncoder(
            max_atomic_number=int(model_cfg.get("max_atomic_number", 20)),
            hidden_dim=velocity_net_hidden_dim(cfg),
            n_layers=int(model_cfg.get("encoder_n_layers", 2)),
        )
        print(
            "[generate] WARNING: checkpoint has no encoder weights; using "
            "freshly-initialised encoder. Re-run scripts/train.py to "
            "produce a checkpoint that includes the encoder."
        )
    return encoder.to(device)


def load_conditioner(checkpoint_blob, device: torch.device, cfg: dict):
    """Rebuild the trained :class:`Conditioner` from a checkpoint.

    Returns ``None`` when no conditioner was checkpointed (the
    unconditional path).  Falls back to a freshly-initialised
    conditioner when the velocity net was trained with conditioning but
    the checkpoint predates the conditioner save path (a warning is
    printed so the user knows the cond embedding is random).
    """
    from models import Conditioner

    cond_state = None
    cond_cfg = None
    if checkpoint_blob is not None and isinstance(checkpoint_blob, dict):
        cond_state = checkpoint_blob.get("conditioner_state_dict")
        cond_cfg = checkpoint_blob.get("conditioner_config")

    if cond_state is None or cond_cfg is None:
        # No conditioner checkpointed.  Either the velocity net is
        # unconditional (the user didn't ask for conditioning) or the
        # training run predates the conditioner save path.  Return None
        # and let the caller decide.
        return None

    conditioner = Conditioner(
        n_scalar_channels=int(cond_cfg.get("n_scalar_channels", 1)),
        hidden_dim=int(cond_cfg.get("hidden_dim", velocity_net_hidden_dim(cfg))),
        n_layers=int(cond_cfg.get("n_layers", 2)),
    )
    conditioner.load_state_dict(cond_state)
    print(
        "[generate] loaded trained conditioner "
        f"(n_scalar_channels={cond_cfg.get('n_scalar_channels')}, "
        f"hidden_dim={cond_cfg.get('hidden_dim')}, "
        f"n_layers={cond_cfg.get('n_layers')})"
    )
    return conditioner.to(device)


def velocity_net_hidden_dim(cfg: dict) -> int:
    return int(cfg.get("model", {}).get("hidden_dim", 128))


# ---------------------------------------------------------------------------
# QM9-shaped synthetic inputs (atomic_numbers from real test molecules)
# ---------------------------------------------------------------------------
# Cap that matches scripts/train.py::_QM9_MAX_ATOMS.
GENERATE_MAX_ATOMS = 29
# Edge cutoff that matches scripts/train.py::_QM9_EDGE_CUTOFF.
GENERATE_EDGE_CUTOFF = 5.0


def build_qm9_inputs(
    n_samples: int,
    device: torch.device,
    seed_dataset=None,
) -> tuple:
    """Build 29-atom (B, 29, 3) QM9-shaped noise + real atom-type graph.

    ``atomic_numbers``, ``edge_index``, ``edge_mask`` and ``node_mask``
    are taken from a sample of QM9 molecules (the test split by
    default).  ``x0`` is pure noise in *normalised* space (mean 0,
    std 1) — the caller's flow-matching sampler integrates from this
    noise to data, then :func:`denormalize_positions` rescales back to
    Ångström before SMILES decoding.

    Parameters
    ----------
    n_samples : int
        Number of generation samples.
    device : torch.device
        Target device.
    seed_dataset : optional ``Dataset``
        Source of real atomic numbers / topology.  When ``None`` the
        function falls back to carbon-only defaults (everything
        becomes C, which the decoder will reject — useful only for
        pipeline smoke tests).

    Returns
    -------
    ``(x0, atomic_numbers, edge_index, edge_mask, node_mask)``
    """
    max_atoms = GENERATE_MAX_ATOMS

    if seed_dataset is not None and len(seed_dataset) >= n_samples:
        real_items = [seed_dataset[i] for i in range(n_samples)]
    else:
        real_items = None

    atomic_numbers = torch.zeros(n_samples, max_atoms, dtype=torch.long, device=device)
    node_mask = torch.zeros(n_samples, max_atoms, dtype=torch.bool, device=device)
    for i in range(n_samples):
        if real_items is not None:
            s = real_items[i]
            n = min(s.atom_types.shape[0], max_atoms)
            atomic_numbers[i, :n] = s.atom_types[:n].to(device).to(torch.long)
            node_mask[i, :n] = True
        else:
            atomic_numbers[i, 0] = 6  # carbon
            node_mask[i, 0] = True

    # Distance-cutoff edges: build x0 first so we can compute pairwise
    # distances — for a QM9-shaped noise distribution the cutoff
    # threshold should be in the *normalised* space too.  We use
    # GENERATE_EDGE_CUTOFF / position_scale (the script stores the
    # training-time scale).
    x0 = torch.randn(n_samples, max_atoms, 3, device=device, dtype=torch.float32)

    # Edges: keep cutoff edges between real atoms only.  We compute
    # distances on the fly; pad to (B, 2, E_max) with edge_mask.
    e_max = max_atoms * (max_atoms - 1)
    edge_src_b = torch.zeros(n_samples, e_max, dtype=torch.long, device=device)
    edge_dst_b = torch.zeros(n_samples, e_max, dtype=torch.long, device=device)
    edge_mask_b = torch.zeros(n_samples, e_max, dtype=torch.bool, device=device)
    cutoff_in_unit = GENERATE_EDGE_CUTOFF  # already in normalised space
    for i in range(n_samples):
        real_idx = node_mask[i].nonzero(as_tuple=True)[0]
        if real_idx.numel() < 2:
            continue
        rp = x0[i, real_idx]
        d = torch.cdist(rp.unsqueeze(0), rp.unsqueeze(0)).squeeze(0)
        src_off, dst_off = (d < cutoff_in_unit).nonzero(as_tuple=True)
        # Map back to global atom indices.
        src = real_idx[src_off]
        dst = real_idx[dst_off]
        n_e = src.numel()
        if n_e > e_max:
            src = src[:e_max]
            dst = dst[:e_max]
            n_e = e_max
        edge_src_b[i, :n_e] = src
        edge_dst_b[i, :n_e] = dst
        edge_mask_b[i, :n_e] = True
    edge_index = torch.stack([edge_src_b, edge_dst_b], dim=1)

    return x0, atomic_numbers, edge_index, edge_mask_b, node_mask


# ---------------------------------------------------------------------------
# Position (de)normalization
# ---------------------------------------------------------------------------
def denormalize_positions(
    positions: torch.Tensor, position_scale: float
) -> torch.Tensor:
    """Rescale flow-matching output back to Ångström."""
    return positions * float(position_scale)


# ---------------------------------------------------------------------------
# 3D -> SMILES bridge
# ---------------------------------------------------------------------------
def positions_to_smiles(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor | None = None,
) -> list[str]:
    """Decode generated ``(B, N, 3)`` configurations into canonical SMILES.

    Thin wrapper around :func:`utils.chem_utils.positions_to_smiles`, kept
    here so existing call sites (and ``from scripts.generate import
    positions_to_smiles``) keep working.  Atom identities come from
    ``atomic_numbers``; bonds are perceived from interatomic distances.

    Returns ``"[]"`` for a sample with no inferable bonds and
    ``"INVALID_<hash>"`` when RDKit cannot sanitize the perceived graph.
    Raises ``ImportError`` if rdkit is not installed.
    """
    return chem_positions_to_smiles(positions, atomic_numbers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate molecules with the trained flow-matching model.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to a trained model checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples to draw (default: {DEFAULT_N_SAMPLES}).",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of ODE integration steps (overrides config.flow.n_steps).",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=("euler", "rk4"),
        default=None,
        help="Integration scheme (overrides config.flow.method).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (otherwise taken from the checkpoint).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'.  Auto-detected by default.",
    )
    p.add_argument(
        "--cond-value",
        type=float,
        default=None,
        help=(
            "Fixed scalar property value (e.g. QM9 gap in atomic units) "
            "broadcast to every generated molecule.  When omitted, "
            "samples are drawn from N(0, 1) per sample instead.  Only "
            "used when the loaded checkpoint was trained with a "
            "Conditioner."
        ),
    )
    p.add_argument(
        "--target-index",
        type=int,
        default=None,
        help=(
            "QM9 target index to generate for (0..11).  Mirrors the "
            "``--target-index`` flag of ``scripts/train.py``: when set, "
            "the conditional path is enabled and ``--cond-value`` (or a "
            "per-sample Gaussian fallback) is interpreted as the value "
            "for that target.  When omitted, generation follows whatever "
            "the checkpoint encoded (unconditional or whatever "
            "target the trainer was using)."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------
def generate(args: argparse.Namespace) -> None:
    # 1. Config: explicit --config wins, else fall back to checkpoint.
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = None

    checkpoint_path = Path(args.checkpoint)
    blob = torch.load(checkpoint_path, map_location="cpu")
    if cfg is None and isinstance(blob, dict):
        cfg = blob.get("config") or {}

    device = resolve_device(args.device or (cfg or {}).get("device"))
    _require_cuda_for_triton(device)
    print(f"[generate] device = {device}")

    # 2. Resolve sampler hyper-parameters ------------------------------------
    flow_cfg = (cfg or {}).get("flow", {}) or {}
    n_steps = int(args.n_steps if args.n_steps is not None else flow_cfg.get("n_steps", 100))
    method = args.method or flow_cfg.get("method", "euler")
    if method not in ("euler", "rk4"):
        raise ValueError(f"Unsupported method {method!r}; expected 'euler' or 'rk4'.")
    print(f"[generate] n_samples={args.n_samples} n_steps={n_steps} method={method}")

    # 3. Load model + encoder + sampler -------------------------------------
    velocity_net, blob = load_velocity_net(checkpoint_path, device)
    encoder = load_encoder(blob, device, cfg or {})
    conditioner = load_conditioner(blob, device, cfg or {})
    position_scale = float((blob or {}).get("position_scale", 1.0))
    print(f"[generate] position_scale = {position_scale:.4f}")
    sampler = FlowMatchingSampler(
        model=velocity_net,
        encoder=encoder,
        default_method=method,
        default_n_steps=n_steps,
    )

    # 4. Sample --------------------------------------------------------------
    # Try to source real QM9 test molecules for atom types + topology so
    # we generate molecules whose atom identities match the training
    # distribution.  When ``--target-index`` is set we also slice the
    # QM9 labels down to that single target so the per-sample ``cond``
    # we build below has a meaningful interpretation.  Falls back to
    # carbon-only defaults if QM9 is not available (e.g. the user wiped
    # ``data/qm9_processed.pt``).
    target_index = getattr(args, "target_index", None)
    if target_index is not None:
        from data.qm9 import QM9Dataset as _QM9Dataset
        target_name = (
            _QM9Dataset.TARGETS[target_index]
            if 0 <= target_index < len(_QM9Dataset.TARGETS)
            else f"index_{target_index}"
        )
        print(
            f"[generate] --target-index {target_index} ({target_name}); "
            "loading QM9 test split sliced to that scalar target."
        )

    seed_dataset = None
    try:
        from data.mol_dataset import QM9Dataset
        seed_dataset = QM9Dataset(split="test", target_index=target_index)
    except Exception as exc:
        print(f"[generate] WARNING: could not load QM9 test split ({exc}); "
              "falling back to carbon-only defaults.")

    x0, atomic_numbers, edge_index, edge_mask, node_mask = build_qm9_inputs(
        args.n_samples, device, seed_dataset=seed_dataset
    )

    # Build the per-sample condition tensor when the velocity net was
    # trained with a Conditioner.  When the user passes --cond-value the
    # same scalar is broadcast to every sample; otherwise we sample from
    # N(0, 1) per sample so each generated molecule sees a different
    # property value.  When the velocity net is unconditional we leave
    # ``cond=None``.
    cond: torch.Tensor | None = None
    if getattr(velocity_net, "cond_dim", 0) > 0:
        if args.cond_value is not None:
            cond = torch.full(
                (args.n_samples, 1),
                float(args.cond_value),
                dtype=torch.float32,
                device=device,
            )
            target_label = (
                f" ({target_name})" if target_index is not None else ""
            )
            print(
                f"[generate] conditioning every sample on "
                f"cond_value={float(args.cond_value):.6f}{target_label}"
            )
        else:
            cond = torch.randn(
                args.n_samples, 1, dtype=torch.float32, device=device
            )
            target_label = (
                f" ({target_name})" if target_index is not None else ""
            )
            print(
                f"[generate] conditioning on N(0, 1) noise{target_label} "
                f"(cond min={cond.min().item():.3f}, "
                f"max={cond.max().item():.3f})"
            )
        # When a conditioner is checkpointed we lift the raw scalar
        # through it once here so the velocity net sees the same
        # embedding format it was trained on.  This is purely a
        # convenience for callers that don't want to instantiate the
        # conditioner themselves; if no conditioner is present we
        # forward the raw tensor directly.
        if conditioner is not None:
            cond = conditioner(cond)
            print("[generate] applied conditioner to raw cond tensor")

    x_final = sampler.sample(
        x0=x0,
        cond=cond,
        n_steps=n_steps,
        method=method,
        atomic_numbers=atomic_numbers,
        edge_index=edge_index,
        edge_mask=edge_mask,
        node_mask=node_mask,
    )
    print(f"[generate] integrated state shape = {tuple(x_final.shape)}")

    # Denormalize back to Ångström before SMILES decoding — the
    # bond-length thresholds in utils/chem_utils.py are calibrated in
    # Angstrom, not normalised units.
    x_real = denormalize_positions(x_final, position_scale)

    # 5. Decode + metrics ----------------------------------------------------
    smiles = positions_to_smiles(x_real, atomic_numbers)
    val = validity(smiles)
    uniq = uniqueness(smiles)
    print(f"[generate] validity    = {val:.4f}  ({int(round(val * len(smiles)))}/{len(smiles)})")
    print(f"[generate] uniqueness  = {uniq:.4f}  ({int(round(uniq * len(smiles)))}/{len(smiles)})")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
