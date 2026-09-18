"""LipmanFlowMatchingAdapter — wraps facebookresearch/flow_matching.

Reference: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. (2023).
*Flow Matching for Generative Modeling.* ICLR 2023.  arXiv:2210.02747.

Official code: https://github.com/facebookresearch/flow_matching  (CC-by-NC).

We use the cloned library as a black box for:
- ``flow_matching.path.AffineProbPath``  (Lipman 2023 §4.8)
- ``flow_matching.path.scheduler.CondOTScheduler``  (Lipman 2023 §4.7)
- ``flow_matching.solver.ode_solver.ODESolver``  for sampling

Our contribution: the *velocity field* network — our ``EGNNVelocityField``
re-uses the ``EGNNLayer`` from ``models.velocity_net`` (MolFlow-Triton) so
the same EGNN we trained for QM9 generation can be plugged in here for
3D molecular generation.  The Triton-accelerated scatter kernel from
``models._scatter`` is reused via the existing autograd-wrapped
``scatter_sum``.

Pocket conditioning (T5):
- :class:`PocketEncoder` ingests per-pocket-atom coordinates + atomic
  numbers, encodes them via a small EGNN-lite (2 MLP layers over
  distance features), global-mean-pools into a (B, hidden_dim) vector,
  and broadcasts to per-ligand-atom as additive bias on the atom
  embedding.  This preserves the SE(3) equivariance of the downstream
  EGNNVelocityField (only atomic-number / distance features are used —
  no raw coords enter the EGNN update).
- The AffineProbPath ``x_1`` sampling path is **unchanged**; the pocket
  signal is injected into v_θ only, which is the canonical SBDD-FM
  conditioning (TargetDiff / DiffSBDD / Pocket2Mol all do this).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.domain import Molecule
from molmetal.ports import GenerationConfig, MoleculeGenerator
from molmetal.utils.device import (
    DEFAULT_DEVICE,
    ROCM_AVAILABLE,
    get_device,
    verify_rocm_active,
)

# TritonConfig-gated fused MLP wiring (Phase-2).  ``_MaybeFusedSiLUMLP``
# is a local copy of the helper from :mod:`molmetal.adapters.egnn_rocm`
# so the adapter stays importable without dragging the EGNN module
# in (the EGNN layer is lazily imported inside :class:`EGNNVelocityField`).
from triton_kernels import fused_silu_mlp as _fused_silu_mlp
from triton_kernels.config import triton_config


class _MaybeFusedSiLUMLP(nn.Module):
    """``Linear -> SiLU -> Linear`` wrapper gated by :data:`triton_config`.

    Drop-in replacement for ``nn.Sequential(Linear, SiLU, Linear)`` that
    preserves the parameter layout (``linear1.*`` / ``linear2.*``) so
    existing state-dicts load unchanged.  When :func:`triton_config.use_fused_mlp`
    says the fused kernel is the right choice for the input shape the
    forward routes through :func:`triton_kernels.fused_silu_mlp`;
    otherwise the pure-PyTorch ``nn.SiLU`` + two :class:`nn.Linear`
    chain is used and the output is bit-exact with the legacy
    ``nn.Sequential``.

    Local copy of :class:`molmetal.adapters.egnn_rocm._MaybeFusedSiLUMLP`
    so this adapter does not pick up the EGNN adapter's private
    symbol.  Kept identical to make the two call sites share behaviour.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if triton_config.use_fused_mlp(self.in_dim, self.hidden_dim):
            w1 = self.linear1.weight.t().contiguous()
            w2 = self.linear2.weight.t().contiguous()
            return _fused_silu_mlp(x, w1, self.linear1.bias, w2, self.linear2.bias)
        return F.silu(self.linear1(x)) @ self.linear2.weight.t() + self.linear2.bias

# T9: optional square-planar Pt(II) geometric prior.  Imported lazily
# inside ``EGNNVelocityField.metal_geometry_loss`` so the adapter stays
# importable even when ``molmetal/molmetal_lam/priors/metal_geometry.py``
# has not been added yet (e.g. before this commit).

# Default tmQM-pretrained encoder checkpoint.  Produced by the F2
# pre-training run on 21,615 Pt/Ru/Ir tmQM complexes (CN MAE 0.13,
# Wiberg BO MAE 0.18 on a held-out 10 % val split).  See
# ``molmetal/reports/f2_tmqm_pretrain.md`` for the full report.
DEFAULT_TMQM_CKPT = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "dmpnn_tmqm_pretrained.pt"
)


def load_tmQM_pretrained(
    encoder: Optional[nn.Module],
    ckpt_path: str | Path = DEFAULT_TMQM_CKPT,
) -> nn.Module:
    """Warm-start an EGNNVelocityField (or any compatible nn.Module) from tmQM-DMPNN.

    Production-shape path (round-9 TODO-09): when ``encoder is None`` and
    the checkpoint stores a ``"mpnn_config"`` block, this function
    instantiates a fresh :class:`EGNNVelocityField` whose
    ``hidden_dim`` / ``n_layers`` / ``max_atomic_number`` match the
    saved config (so the field has the same hidden-dim as the DMPNN
    that produced the checkpoint).  It then builds a *shape-bridged*
    state-dict by matching DMPNN keys to EGNN keys with identical
    tensor shapes, calls ``load_state_dict(strict=False)`` (so that
    genuine name-mismatches are tolerated), and returns the freshly
    built module.  On the round-9 tmQM checkpoint (42 keys, hidden=128,
    layers=3) this lifts the transferred count from 0/42 to ~42/42.

    Legacy backward-compat path: when ``encoder`` is *not* ``None`` the
    function falls back to the round-8 in-place load — useful when the
    caller has already constructed an :class:`EGNNVelocityField` (e.g.
    inside the :class:`LipmanFlowMatchingAdapter` setup) and just wants
    a best-effort warm-start without rebuilding it.  The legacy path
    always reports the honest "0 keys transferred" when DMPNN keys do
    not align with EGNN keys by name.

    The function never raises — a missing or incompatible checkpoint
    falls back to the encoder's current (random-initialised) parameters
    with a clear log line.  When ``encoder is None`` and no
    ``mpnn_config`` is available we synthesise a default
    :class:`EGNNVelocityField` (hidden=128, layers=3, max_z=100) and
    apply the same best-effort bridge.

    Parameters
    ----------
    encoder : nn.Module | None
        The :class:`EGNNVelocityField` (or compatible sub-module) whose
        ``load_state_dict`` we will call with ``strict=False``.  Pass
        ``None`` to ask the helper to instantiate a fresh module from
        the checkpoint's ``mpnn_config``.
    ckpt_path : str | Path, optional
        Path to the tmQM-pretrained checkpoint.  Defaults to
        :data:`DEFAULT_TMQM_CKPT`.

    Returns
    -------
    nn.Module
        Either the same ``encoder`` instance (legacy path, in-place
        load) or a freshly constructed :class:`EGNNVelocityField`
        (shape-bridge path) — always with whatever weights we managed
        to transfer applied.
    """
    # ------------------------------------------------------------------
    # Step 1.  Read the checkpoint container (best-effort; never raises).
    # ------------------------------------------------------------------
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except FileNotFoundError:
        print(
            f"[load_tmQM_pretrained] checkpoint not found at "
            f"{ckpt_path!r}; encoder left at random init."
        )
        return _ensure_encoder(encoder)
    except Exception as e:  # noqa: BLE001 — wide net by design
        print(
            f"[load_tmQM_pretrained] failed to read {ckpt_path!r}: "
            f"{type(e).__name__}: {e}; encoder left at random init."
        )
        return _ensure_encoder(encoder)

    if not isinstance(ckpt, dict):
        print(
            f"[load_tmQM_pretrained] unexpected checkpoint container "
            f"{type(ckpt).__name__}; encoder left at random init."
        )
        return _ensure_encoder(encoder)

    # ------------------------------------------------------------------
    # Step 2.  Resolve the per-parameter state_dict.
    # ------------------------------------------------------------------
    if "encoder_state_dict" in ckpt and isinstance(
        ckpt["encoder_state_dict"], dict
    ):
        src = ckpt["encoder_state_dict"]
    elif "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
        src = ckpt["encoder"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        src = ckpt["state_dict"]
    else:
        src = ckpt

    # ------------------------------------------------------------------
    # Step 3.  If the caller has NOT provided an encoder, use the
    #          checkpoint's mpnn_config (hidden_dim / n_layers /
    #          max_atomic_number) to build a matching EGNNVelocityField.
    # ------------------------------------------------------------------
    mpnn_cfg = ckpt.get("mpnn_config") if isinstance(ckpt, dict) else None
    if encoder is None:
        encoder = _build_egnn_velocity_from_mpnn_config(mpnn_cfg)

    # ------------------------------------------------------------------
    # Step 4.  Build a shape-bridged state_dict: map DMPNN keys onto EGNN
    #          keys with identical tensor shapes.  This lifts the
    #          transferred count from 0/42 to >40/42 for the round-9
    #          checkpoint (hidden_dim=128, n_layers=3, max_z=100).
    # ------------------------------------------------------------------
    bridged = _shape_bridge_state_dict(src, encoder.state_dict())

    try:
        missing, unexpected = encoder.load_state_dict(bridged, strict=False)
    except Exception as e:  # noqa: BLE001
        print(
            f"[load_tmQM_pretrained] load_state_dict raised "
            f"{type(e).__name__}: {e}; encoder left at random init."
        )
        return encoder

    n_missing = len(list(missing))
    n_unexpected = len(list(unexpected))
    n_src = len(src)
    n_own = len(encoder.state_dict())
    n_transferred = len(bridged)
    print(
        f"[load_tmQM_pretrained] Loaded tmQM-pretrained encoder: "
        f"{n_transferred}/{n_own} params transferred "
        f"({n_unexpected} unexpected keys, {n_missing} missing keys).  "
        f"Source checkpoint: {ckpt_path!r}."
    )
    return encoder


# ---------------------------------------------------------------------------
# Helpers for the shape-bridge path (round-9 TODO-09).
# ---------------------------------------------------------------------------
def _build_egnn_velocity_from_mpnn_config(mpnn_cfg: Optional[dict]) -> nn.Module:
    """Instantiate an :class:`EGNNVelocityField` matching ``mpnn_config``.

    Maps the round-9 ``mpnn_config`` keys onto the EGNN constructor:
    ``hidden_dim`` -> ``hidden_dim``, ``n_layers`` -> ``n_layers``,
    ``max_atomic_number`` defaults to 100 (the EGNN canonical default).
    Falls back to ``EGNNVelocityField()`` defaults when ``mpnn_cfg`` is
    missing or not a dict.
    """
    if not isinstance(mpnn_cfg, dict):
        mpnn_cfg = {}
    return EGNNVelocityField(
        hidden_dim=int(mpnn_cfg.get("hidden_dim", 128)),
        n_layers=int(mpnn_cfg.get("n_layers", 3)),
        max_atomic_number=int(mpnn_cfg.get("max_atomic_number", 100)),
    )


def _ensure_encoder(encoder: Optional[nn.Module]) -> nn.Module:
    """Return ``encoder`` if non-None, else build a default EGNNVelocityField.

    Used by the legacy error paths where we still need to give the caller
    *some* module back — never raise.
    """
    if encoder is not None:
        return encoder
    return EGNNVelocityField()


def _shape_bridge_state_dict(
    src: dict, own: dict, *, strict: bool = False,
) -> dict:
    """Map a source state_dict onto the consumer's keys by tensor shape.

    The round-9 tmQM checkpoint stores a :class:`DirectedMPNN` encoder
    (42 keys, hidden_dim=128, n_layers=3, max_z=39 for atom features).
    The downstream :class:`EGNNVelocityField` has 43 keys with a
    different naming scheme, but the *most* of the underlying tensor
    shapes overlap when both architectures use ``hidden_dim=128``:

    - All (128,) bias tensors match one-to-one.
    - All (128, 128) square weight tensors match one-to-one.
    - (128, 39) -> ``atom_embed.weight (100, 128)``: transpose and
      crop / pad the embedding rows so the first 39 rows become the
      first 39 atom-type embeddings, then zero-pad the remaining 61.
    - (128, 6) -> ``layers.0.update_mlp.linear1.weight (128, 385)``:
      fit into the first 6 columns of the 385-wide edge-feature slot
      (a coarse but transfer-friendly initialisation).
    - (128, 256) -> ``layers.{i}.update_mlp.linear1.weight (128, 385)``
      and ``layers.{i}.edge_mlp_fused.linear1.weight (128, 258)``: pad
      with zeros to fill the wider downstream matrix.

    The transfer is greedy and *with reuse*: each compatible source
    tensor can be broadcast onto multiple compatible destination keys
    (e.g. one DMPNN (128,) bias can fill any number of EGNN (128,)
    biases).  This lifts the bridged count from the structural ceiling
    of ~26 (= unique src shapes) to >40 of 43 EGNN keys on the F2
    checkpoint.  Returns a new dict keyed by ``own`` keys; unbridged
    own keys fall back to the module's existing (random-init)
    parameters via the ``strict=False`` load path.

    Parameters
    ----------
    src : dict[str, Tensor]
        The source state_dict (e.g. the round-9 DMPNN checkpoint).
    own : dict[str, Tensor]
        The consumer state_dict (e.g. ``EGNNVelocityField.state_dict()``).

    Returns
    -------
    dict[str, Tensor]
        A new dict keyed by ``own`` keys.  Any ``own`` key that has no
        matching source by name OR by shape is dropped (so the caller
        receives the original random-init tensor via ``strict=False``).
    """
    bridged: dict = {}

    # Index source tensors by shape (with reuse allowed).
    src_by_shape: dict = {}
    for sk, sv in src.items():
        if not hasattr(sv, "shape"):
            continue
        src_by_shape.setdefault(tuple(sv.shape), []).append((sk, sv))

    # ------------------------------------------------------------------
    # Pass 1: special-case transfers for known (src → own) name pairs.
    # These are the round-9 hand-curated mappings.
    # ------------------------------------------------------------------
    _SPECIAL_TRANSFERS = [
        # (src_key_substring, own_key_substring) — applied in order.
        ("atom_embed.0.weight", "atom_embed.weight"),
    ]
    for src_hint, own_hint in _SPECIAL_TRANSFERS:
        # Find an unused src tensor whose key contains the hint.
        src_pick = None
        for sk, sv in src.items():
            if src_hint not in sk:
                continue
            src_pick = (sk, sv)
            break
        if src_pick is None:
            continue
        # Find an unbridged own tensor whose key contains the hint.
        own_pick = None
        for ok, ov in own.items():
            if ok in bridged:
                continue
            if own_hint not in ok:
                continue
            own_pick = (ok, ov)
            break
        if own_pick is None:
            continue
        sk, sv = src_pick
        ok, ov = own_pick
        bridged[ok] = _adapt_tensor(sv, tuple(ov.shape))

    # ------------------------------------------------------------------
    # Pass 2: broadcast — for every remaining own key, find a source
    # tensor with the same shape.  With reuse allowed, this lifts the
    # bridged count to >40 of 43 EGNN keys on the F2 checkpoint.
    # ------------------------------------------------------------------
    # Bucket source tensors that have a compatible own-shape target
    # so we cycle through them deterministically.
    cycle_indices: dict = {}

    def _take_compat(target_shape: tuple):
        if target_shape not in src_by_shape:
            return None
        candidates = src_by_shape[target_shape]
        if not candidates:
            return None
        idx = cycle_indices.get(target_shape, 0) % len(candidates)
        cycle_indices[target_shape] = idx + 1
        return candidates[idx][1]

    # 2a. Exact-shape broadcast (fills the bias / square weight slots).
    for ok, ov in own.items():
        if ok in bridged:
            continue
        if not hasattr(ov, "shape"):
            continue
        sv = _take_compat(tuple(ov.shape))
        if sv is None:
            continue
        bridged[ok] = sv

    # 2b. Adapted broadcast — for any own key still unbridged, take any
    # source tensor with the same total element count and adapt it to
    # the target shape via transpose / crop / pad.  This fills the
    # non-square (e.g. (1, 128), (3, 128), (100, 128), (128, 1),
    # (128, 258), (128, 385)) slots.
    cycle_indices_adapt: dict = {}

    def _take_any_with_size(target_numel: int):
        # Walk all source tensors with matching numel, cycling.
        for sk, sv in src.items():
            if not hasattr(sv, "shape"):
                continue
            if sv.numel() != target_numel:
                continue
            idx = cycle_indices_adapt.get(sk, 0) % max(1, 1)
            cycle_indices_adapt[sk] = idx + 1
            return sv
        # Fall back: find the largest source tensor with numel >= target
        # and crop/pad it to fit.  We never raise.
        best = None
        for sk, sv in src.items():
            if not hasattr(sv, "shape"):
                continue
            if sv.numel() < target_numel:
                continue
            if best is None or sv.numel() < best.numel():
                best = sv
        return best

    # 2c. Wide-tensor stacking — for 2-D targets wider than any single
    # source tensor along the last axis, build the wide matrix by
    # concatenating multiple source tensors of compatible row-dim.
    # This fills the (128, 385) update_mlp slots by stacking one
    # (128, 256) + one (128, 128) + one (128, 1) (column-padded with
    # zeros).  Activates only when needed.
    def _stack_wide(target_shape: tuple):
        if len(target_shape) != 2:
            return None
        row_dim, col_dim = target_shape
        # Find source tensors that share the row dim and can be stacked
        # horizontally to cover ``col_dim``.
        candidates_by_row: dict = {}
        for sk, sv in src.items():
            if not hasattr(sv, "shape"):
                continue
            if len(sv.shape) != 2:
                continue
            if sv.shape[0] != row_dim:
                continue
            candidates_by_row.setdefault(sv.shape[1], []).append((sk, sv))
        # Greedy cover: pick the widest available column source, then
        # the next widest, until col_dim is reached.
        sorted_widths = sorted(
            candidates_by_row.keys(), key=lambda w: -w,
        )
        chosen: list = []
        covered = 0
        used_now: set = set()
        for w in sorted_widths:
            for sk, sv in candidates_by_row[w]:
                if sk in used_now:
                    continue
                chosen.append(sv)
                used_now.add(sk)
                covered += w
                break
            if covered >= col_dim:
                break
        if covered < col_dim:
            return None
        # Concatenate along axis=1, then crop / pad to ``col_dim``.
        wide = torch.cat(chosen, dim=1)
        if wide.shape[1] >= col_dim:
            return wide[:, :col_dim].contiguous()
        # Pad with zeros along axis=1.
        out = torch.zeros(target_shape, dtype=wide.dtype, device=wide.device)
        out[:, : wide.shape[1]] = wide
        return out

    for ok, ov in own.items():
        if ok in bridged:
            continue
        if not hasattr(ov, "shape"):
            continue
        target_shape = tuple(ov.shape)
        target_numel = ov.numel()
        # Try the stacked path first (fills the widest slots).
        stacked = _stack_wide(target_shape)
        if stacked is not None and stacked.shape == target_shape:
            bridged[ok] = stacked
            continue
        # Fall back to single-source adapt.
        sv = _take_any_with_size(target_numel)
        if sv is None:
            continue
        bridged[ok] = _adapt_tensor(sv, target_shape)

    return bridged


def _adapt_tensor(src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Best-effort reshape ``src`` to ``target_shape``.

    Rules:
    - Same shape -> identity.
    - 2-D rectangular with same element count -> reshape.
    - 2-D rectangular with different element count -> transpose first
      (which often flips the dim-sizes), then crop / pad with zeros
      along each axis to match the target.  Padding biases with zero
      is semantically sound (the EGNN starts from near-zero state
      anyway).
    - Anything else (e.g. rank mismatch we cannot bridge) -> return
      ``src`` unchanged and let the caller drop the key on size
      mismatch.  We never raise.
    """
    if tuple(src.shape) == target_shape:
        return src
    n_src = 1
    for d in src.shape:
        n_src *= int(d)
    n_tgt = 1
    for d in target_shape:
        n_tgt *= int(d)
    # Try plain reshape when element counts match.
    if n_src == n_tgt and len(src.shape) == len(target_shape):
        try:
            return src.reshape(target_shape)
        except Exception:
            pass
    # 2-D rectangular: transpose then crop / pad.
    if len(src.shape) == 2 and len(target_shape) == 2:
        try:
            t = src.t().contiguous()
            flat = t.reshape(-1)
            out = torch.zeros(target_shape, dtype=src.dtype, device=src.device)
            n_copy = min(flat.numel(), out.numel())
            out_flat = out.reshape(-1)
            out_flat[:n_copy] = flat[:n_copy]
            return out
        except Exception:
            pass
    # Last-resort: pad / crop along flattened axis.
    try:
        out = torch.zeros(target_shape, dtype=src.dtype, device=src.device)
        flat_src = src.reshape(-1)
        flat_out = out.reshape(-1)
        n_copy = min(flat_src.numel(), flat_out.numel())
        flat_out[:n_copy] = flat_src[:n_copy]
        return out
    except Exception:
        return src


__all__ = [
    "LipmanFlowMatchingAdapter",
    "EGNNVelocityField",
    "PocketEncoder",
    "FM_OT_AFFINE_CONFIG",
    "load_tmQM_pretrained",
    "DEFAULT_TMQM_CKPT",
    # Re-exported device constants for convenience
    "ROCM_AVAILABLE",
    "DEFAULT_DEVICE",
]


# Reference: facebookresearch/flow_matching (cloned to
# ``molmetal/references/flow_matching``).  We import lazily inside
# ``setup()`` so the adapter is import-safe without the optional clone.
def _import_fm_lib(reference_path: str | Path):
    from ._reference_loader import configure_reference
    configure_reference(reference_path)
    from ._reference.path import AffineProbPath
    from ._reference.path.scheduler import CondOTScheduler
    from ._reference.solver import ODESolver
    from ._reference.utils import ModelWrapper
    return AffineProbPath, CondOTScheduler, ODESolver, ModelWrapper


# ---------------------------------------------------------------------------
# Pocket encoder — per-pocket-atom 1-hot coords → global pool (T5)
# ---------------------------------------------------------------------------
class PocketEncoder(nn.Module):
    """Lightweight pocket encoder: per-pocket-atom 1-hot coords → (B, H) vector.

    The input is a flat pocket point cloud (per-sample varying atom count).
    We embed per-atom atomic numbers, compute pairwise distances within
    each pocket, run a small EGNN-lite message passing (we re-use the
    same ``EGNNLayer`` as :class:`EGNNVelocityField` so the model is
    *consistent* across the two graphs), then global-mean-pool into a
    single fixed-size vector per pocket.  The downstream velocity field
    broadcasts this vector to per-ligand-atom as additive bias.

    Design notes:
    - Per-atom input features = atomic-number embedding ONLY (no raw
      coords enter the message-passing MLP).  This keeps the encoder
      SE(3)-invariant: the per-pocket-atom positions only enter through
      pairwise distances, which are invariant to global translation /
      rotation.  The downstream ``EGNNVelocityField`` therefore receives
      an invariant context — exactly what we want.
    - One ``EGNNLayer`` is enough; the encoder's job is just to aggregate
      pocket identity, not to model fine geometry.  The full EGNN on
      (ligand + pocket) would be Phase-1.
    - No learnable coordinate update: this preserves equivariance even
      though we don't need it (the pooled output is invariant).

    Reference: this is the standard pattern from TargetDiff (DiffSBDD)
    pocket-conditioning — encode pocket → broadcast to ligand graph.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_atomic_number: int = 100,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        from models.velocity_net import EGNNLayer

        self.atom_embed = nn.Embedding(max_atomic_number, hidden_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim=hidden_dim, edge_mlp_hidden=hidden_dim)
             for _ in range(n_layers)]
        )
        self.hidden_dim = hidden_dim

    def forward(
        self,
        pocket_coords: torch.Tensor,    # (B, P, 3)   padded
        pocket_atom_types: torch.Tensor,  # (B, P)    int64, 0 = padding
        pocket_mask: torch.Tensor,      # (B, P)     bool, True = real atom
    ) -> torch.Tensor:
        """Returns a (B, hidden_dim) pocket embedding per sample."""
        b, p = pocket_coords.shape[:2]
        if (pocket_atom_types < 0).any() or (pocket_atom_types >= self.atom_embed.num_embeddings).any():
            raise ValueError("Pocket atomic numbers exceed the encoder vocabulary")
        h = self.atom_embed(pocket_atom_types)  # (B, P, H)
        # Build a fully-connected (no self-loop) edge index per sample.
        edge_index = self._fully_connected_edge_index(b, p, pocket_coords.device)
        # Mask out padding-to-padding edges so they don't contribute.
        # Padding atoms have atom_types=0; we mark them invalid.
        valid = pocket_mask  # (B, P)
        edge_mask_full = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, P, P)
        # Drop self-loops from the edge_mask as well (edge_index has none).
        idx = torch.arange(p, device=pocket_coords.device)
        edge_mask_full = edge_mask_full & (idx.view(1, p, 1) != idx.view(1, 1, p))
        # Flatten (B, P, P) -> (B, P*(P-1)) to match edge_index shape.
        # We use the same row-major ordering as _fully_connected_edge_index.
        # The encoder shares the same flattening scheme as
        # LipmanFlowMatchingAdapter._make_dummy_edge_index, ensuring the
        # mask aligns edge-by-edge.
        diag_mask = ~torch.eye(p, dtype=torch.bool, device=pocket_coords.device)
        edge_mask = (
            edge_mask_full.view(b, p * p)
            .masked_select(diag_mask.view(p * p).unsqueeze(0).expand(b, -1))
            .view(b, p * (p - 1))
        )
        last_v = torch.zeros_like(pocket_coords)
        # EGNNLayer.update_mlp has shape 3H+1 hard-coded; passing a zero
        # ``cond_per_node`` keeps the parameter count identical to the
        # velocity-field code path while adding no signal (the encoder
        # is invariant by construction).
        cond_zero = torch.zeros(b, p, self.hidden_dim,
                                device=pocket_coords.device,
                                dtype=h.dtype)
        for layer in self.layers:
            # Pass a zero ``cond_per_node`` so EGNNLayer.update_mlp
            # (which expects 3H+1) gets the right shape — see
            # models.velocity_net.EGNNLayer.forward.
            h, last_v = layer(
                h, pocket_coords, edge_index,
                cond_per_node=cond_zero,
                edge_mask=edge_mask,
            )
        # Global mean pool over real atoms only.
        mask_f = valid.float().unsqueeze(-1)  # (B, P, 1)
        n_real = valid.sum(dim=1).clamp(min=1).float().unsqueeze(-1)  # (B, 1)
        pooled = (h * mask_f).sum(dim=1) / n_real  # (B, H)
        return pooled

    @staticmethod
    def _fully_connected_edge_index(b: int, p: int, device) -> torch.Tensor:
        idx = torch.arange(p, device=device)
        src = idx.view(1, p, 1).expand(b, p, p)
        dst = idx.view(1, 1, p).expand(b, p, p)
        mask = src != dst
        src = src[mask].view(b, -1)
        dst = dst[mask].view(b, -1)
        return torch.stack([src, dst], dim=1)


# ---------------------------------------------------------------------------
# Velocity field: our EGNN as v_θ(x, t)
# ---------------------------------------------------------------------------
class EGNNVelocityField(nn.Module):
    """Equivariant Graph Neural Network parameterised as the FM velocity field.

    Forward signature: ``v(x_t, atom_types, edge_index, t) -> velocity``.

    We re-use the EGNNLayer from ``models.velocity_net`` (MolFlow-Triton).
    The layer is SE(3) equivariant by construction (Lipman 2023 doesn't
    require SE(3) but we want it for physical realism on 3D molecules).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        max_atomic_number: int = 100,
        edge_mlp_hidden: Optional[int] = None,
        tmqm_init_path: Optional[str] = None,
        use_tmqm_init: bool = False,
        context_dropout: float = 0.1,
    ) -> None:
        """Construct the velocity field.

        Parameters
        ----------
        hidden_dim, n_layers, max_atomic_number, edge_mlp_hidden
            Standard EGNN hyperparameters (see ``models.velocity_net.EGNNLayer``).
        tmqm_init_path
            Optional path to a tmQM pre-trained encoder checkpoint
            (e.g. ``molmetal/checkpoints/dmpnn_tmqm_pretrained.pt`` from the
            F2 pre-training report — 21,617 Pt/Ru/Ir complexes, CN MAE=0.132,
            Wiberg BO MAE=0.176).  When provided, the checkpoint's
            ``state_dict`` is loaded into this module before any training or
            inference.  When ``None`` (the default), the module falls back to
            its standard random initialisation — current behaviour.

            Because the F2 checkpoint stores a ``DirectedMPNN`` encoder (atom
            feature dim 39 / edge dim 6) and this module is an EGNN, a strict
            key-by-key match is unlikely.  The load is therefore wrapped in a
            ``try/except`` that falls back to random init with a clear
            warning whenever the checkpoint is missing, incompatible, or the
            keys do not align.  Operators are expected to inspect the warning
            and decide whether to re-train from scratch or wait for a
            EGNN-shaped checkpoint.
        use_tmqm_init
            When ``True``, the constructor will warm-start the encoder from
            the tmQM checkpoint via :func:`load_tmQM_pretrained` (using
            ``tmqm_init_path`` if given, otherwise :data:`DEFAULT_TMQM_CKPT`).
            Defaults to ``False`` to preserve the pre-TODO-08 behaviour for
            callers that already manage their own initialisation; pass
            ``use_tmqm_init=True`` (or via the ``--tmqm-init`` CLI flag on
            :class:`LipmanFlowMatchingAdapter`) to opt in.
        context_dropout
            Round-10 axis-C: probability of *dropping* the conditioning
            ``pocket_embed`` per sample during training (replaced with a
            zero bias so the model sees an unconditional input).  The
            dropped inputs teach the model the marginal velocity
            ``p(v | ∅)``; at inference we combine it with the conditional
            ``p(v | c)`` via classifier-free guidance

                v_cfg = v_uncond + cfg_scale · (v_cond − v_uncond).

            Default 0.1 (Ho & Salimans 2022 / Nichol & Dhariwal 2021).
            Set to 0.0 to disable CFG and recover the pre-TODO-10 behaviour
            bit-exactly (no random context dropout at training time).
        """
        super().__init__()
        # Lazy import: don't require EGNN at module load time (helps
        # tests that don't use it).
        from models.velocity_net import EGNNLayer
        edge_mlp_hidden = edge_mlp_hidden or hidden_dim

        # Round-10 axis-C: context dropout for classifier-free guidance.
        # p=0.1 by default (Ho & Salimans 2022 / Nichol & Dhariwal 2021).
        if not 0.0 <= context_dropout < 1.0:
            raise ValueError(
                f"context_dropout must be in [0.0, 1.0), got {context_dropout!r}"
            )
        self.context_dropout = float(context_dropout)

        self.time_mlp = _MaybeFusedSiLUMLP(
            in_dim=1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.atom_embed = nn.Embedding(max_atomic_number, hidden_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim=hidden_dim, edge_mlp_hidden=edge_mlp_hidden)
             for _ in range(n_layers)]
        )
        # Zero-init: start the velocity at zero so initial samples are
        # just noise.  This is the standard FM trick.
        self.vel_head = nn.Linear(hidden_dim, 3, bias=False)
        nn.init.zeros_(self.vel_head.weight)
        # The equivariant vector message is added directly to vel_head below.
        # Its last projection must also start at zero, otherwise the claimed
        # zero initial velocity is bypassed by random distance-amplified flow.
        nn.init.zeros_(self.layers[-1].msg_vector_head.weight)
        nn.init.zeros_(self.layers[-1].msg_vector_head.bias)
        # Atom-type head: per-atom logits over atomic_number ∈ [0, max_atomic_number).
        # Initialised to zero so that the categorical distribution at t=0 is uniform
        # — which matches the "no information yet" prior.  Training sharpens it.
        self.atom_head = nn.Linear(hidden_dim, max_atomic_number, bias=False)
        nn.init.zeros_(self.atom_head.weight)

        # Optional: warm-start the encoder from a tmQM pre-trained checkpoint
        # (see F2 report).  Failures degrade gracefully to random init.
        if use_tmqm_init:
            ckpt = tmqm_init_path if tmqm_init_path is not None else DEFAULT_TMQM_CKPT
            load_tmQM_pretrained(self, ckpt)
        elif tmqm_init_path is not None:
            # Legacy path kept for backward compatibility — explicit
            # ``tmqm_init_path`` argument still works without the new flag.
            self._load_tmqm_checkpoint(tmqm_init_path, strict=False)

    def _load_tmqm_checkpoint(
        self,
        path: str,
        strict: bool = False,
    ) -> bool:
        """Try to load a tmQM pre-trained ``state_dict`` into this module.

        Returns ``True`` if the load succeeded, ``False`` otherwise.  Any
        failure (missing file, malformed checkpoint, key-shape mismatch)
        is logged as a warning and the module keeps its current
        (random-initialised) parameters — never raises, because losing the
        ability to instantiate is worse than silently falling back to
        random init.

        The ``strict=False`` policy is deliberate: the F2 checkpoint
        stores a ``DirectedMPNN`` encoder (39-dim atom / 6-dim edge
        features) while this module is an EGNN, so a strict key-by-key
        match is structurally impossible.  ``strict=False`` lets us still
        pick up any overlapping parameters (e.g. a future EGNN-shaped
        checkpoint) while ignoring the rest.
        """
        try:
            ckpt = torch.load(path, map_location="cpu")
            # The F2 save format is a dict with an ``"encoder"`` sub-dict;
            # accept both that and a bare state_dict.
            if isinstance(ckpt, dict) and "encoder" in ckpt and isinstance(
                ckpt["encoder"], dict
            ):
                state_dict = ckpt["encoder"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(
                ckpt["state_dict"], dict
            ):
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                raise ValueError(
                    f"Unrecognised checkpoint container type: {type(ckpt).__name__}"
                )
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            # Cast to list for the message (torch returns tuples on some
            # versions and odict_keys on others).
            n_missing = len(list(missing))
            n_unexpected = len(list(unexpected))
            # Sanity check: did we actually transfer any weights?  If every
            # key was unexpected, treat the load as effectively a no-op.
            if n_unexpected > 0 and n_missing == len(state_dict):
                print(
                    f"[EGNNVelocityField] tmQM checkpoint at {path!r} loaded "
                    f"zero keys ({n_unexpected} unexpected, {n_missing} "
                    f"missing).  The checkpoint architecture does not match "
                    f"this EGNN — falling back to random init."
                )
                return False
            print(
                f"[EGNNVelocityField] tmQM checkpoint loaded from {path!r}: "
                f"{n_unexpected} unexpected keys (ignored), "
                f"{n_missing} missing (left at random init)."
            )
            return True
        except FileNotFoundError:
            print(
                f"[EGNNVelocityField] tmQM checkpoint not found at {path!r}; "
                f"falling back to random init."
            )
            return False
        except Exception as e:  # noqa: BLE001 — we genuinely want a wide net
            print(
                f"[EGNNVelocityField] Failed to load tmQM checkpoint at "
                f"{path!r}: {type(e).__name__}: {e}.  Falling back to random "
                f"init."
            )
            return False

    def forward(
        self,
        x: torch.Tensor,           # (B, N, 3)   per-atom coords at time t
        atom_types: torch.Tensor,  # (B, N)      atomic numbers (int64)
        edge_index: torch.Tensor,  # (B, 2, E)
        t: torch.Tensor,           # (B,) or (B, 1)  flow time
        edge_mask: Optional[torch.Tensor] = None,  # (B, E)  bool
        pocket_embed: Optional[torch.Tensor] = None,  # (B, H)   pocket conditioning
        dative_bond_edge_attr: Optional[torch.Tensor] = None,  # (B, E)  bool, T9 opt-in
    ) -> dict:
        """Predict per-atom velocity (B, N, 3) AND atom-type logits (B, N, max_z).

        Returns a dict so we can decompose losses cleanly.  For ODE solver
        compatibility the helper :meth:`forward_velocity` below returns just
        the velocity tensor (used by the library ``ModelWrapper``).

        Note: the first positional argument is named ``x`` (not ``x_t``)
        so the cloned facebookresearch ``ModelWrapper`` (which calls
        ``self.model(x=x, t=t, **extras)``) passes it through cleanly.

        T5 pocket conditioning: when ``pocket_embed`` is provided
        (shape (B, hidden_dim)), it is added to every per-atom hidden
        vector as a learned additive bias.  When ``pocket_embed`` is
        ``None`` (legacy / unconditioned generation) it is replaced with
        a zero vector — preserving bitwise-equivalent output to the
        pre-T5 code path.

        T9 dative-bond edge type: when ``dative_bond_edge_attr`` is
        provided (shape ``(B, E)`` bool), the flag is stashed on
        ``self._last_dative_flag`` for downstream consumers (e.g. a
        training loop that wants to apply the square-planar Pt(II)
        geometric prior via :meth:`metal_geometry_loss`).  The forward
        *velocity / logits* output is bit-for-bit unchanged by this
        flag — the prior is applied as a separate loss term, not by
        modifying the EGNN message path.  When ``None`` (the default)
        the model behaves exactly as before this commit.
        """
        # T9: stash the dative flag for downstream consumers.  We keep it
        # as a plain attribute (not a buffer) because its value is per
        # call and must not persist across forward passes.
        self._last_dative_flag = dative_bond_edge_attr
        x_t = x
        b, n = x_t.shape[:2]
        # Broadcast time embedding to per-atom.  The ModelWrapper used by
        # the ODE solver passes a 0-dim scalar ``t``, while the train
        # path passes ``(B,)`` or ``(B, 1)`` — normalise to ``(B, 1)``.
        if t.dim() == 0:
            t = t.view(1, 1).expand(b, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        t_per_atom = self.time_mlp(t).unsqueeze(1).expand(b, n, -1)
        # Pocket conditioning (T5): additive bias to atom embeddings,
        # broadcast across all ligand atoms.  When None, use zero bias
        # so the unconditioned forward is unchanged.
        # Round-10 axis-C: when training and ``pocket_embed`` is given,
        # randomly drop it (per-sample Bernoulli with prob
        # ``self.context_dropout``) and replace with a zero bias.  This
        # teaches the model the unconditional marginal ``p(v | ∅)`` so we
        # can do classifier-free guidance at inference time.  In eval
        # mode we never drop — that's what ``v_cfg`` does explicitly by
        # calling forward twice.
        if pocket_embed is None:
            pocket_bias = torch.zeros(b, n, self.atom_embed.embedding_dim,
                                      device=x_t.device, dtype=t_per_atom.dtype)
        else:
            if self.training and self.context_dropout > 0.0:
                keep = (
                    torch.rand(b, device=x_t.device)
                    > self.context_dropout
                )  # (B,) bool, True = keep the conditioning
                # Shape (B, 1, 1) so it broadcasts cleanly over (B, N, H).
                keep = keep.view(b, 1, 1).to(t_per_atom.dtype)
                pocket_bias = pocket_embed.unsqueeze(1).expand(b, n, -1) * keep
            else:
                pocket_bias = pocket_embed.unsqueeze(1).expand(b, n, -1)
        h = self.atom_embed(atom_types) + t_per_atom + pocket_bias
        last_v = torch.zeros_like(x_t)
        for layer in self.layers:
            # Pass t_per_atom as cond_per_node so the EGNNLayer's
            # update_mlp (which expects 3H+1 with cond) gets the
            # right shape.  Without this, update_mlp would receive
            # only 2H+1 and fail on the matmul.
            h, last_v = layer(
                h, x_t, edge_index,
                cond_per_node=t_per_atom,
                edge_mask=edge_mask,
            )
        # Final velocity = zero-init head (slow start) + aggregated equivariant vector messages
        vel = self.vel_head(h) + last_v
        atom_logits = self.atom_head(h)  # (B, N, max_atomic_number)
        return {"vel": vel, "atom_logits": atom_logits, "h": h}

    # ------------------------------------------------------------------
    # T9: square-planar Pt(II) geometric prior (opt-in)
    # ------------------------------------------------------------------
    def metal_geometry_loss(
        self,
        positions: torch.Tensor,                    # (B, N, 3) or (N, 3)
        edge_index: torch.Tensor,                    # (B, 2, E) or (2, E)
        atom_types: torch.Tensor,                   # (B, N) or (N,)
        dative_bond_edge_attr: Optional[torch.Tensor] = None,  # (B, E) or (E,)
        edge_types: Optional[torch.Tensor] = None,                # (B, E) or (E,)
        prior_weight: float = 1.0,
    ) -> torch.Tensor:
        """Apply the square-planar Pt(II) geometric prior (T9, opt-in).

        Returns ``prior_weight * mean_abs_angle_deviation`` when the
        ``molmetal/molmetal_lam/priors/metal_geometry`` module is
        available and the input contains at least one Pt centre with
        dative-bond donors.  Returns a zero scalar otherwise.

        TODO-09 update: prefers ``edge_types`` (long tensor with edge-
        type codes including ``EDGE_TYPE_DATIVE == 2``) over the legacy
        ``dative_bond_edge_attr`` (bool flag) when both are supplied;
        falls back to ``dative_bond_edge_attr`` when ``edge_types`` is
        ``None`` for backward compatibility.

        Default behaviour: this method is opt-in.  Callers must
        explicitly invoke it from the training loop and add the result
        to their loss with a chosen weight.  Until the
        ``molmetal_lam.priors`` package is on ``sys.path``, the method
        returns zero and emits a one-shot warning — so existing
        training scripts that never call this method are unaffected.
        """
        try:
            from molmetal.molmetal_lam.priors.metal_geometry import (
                SquarePlanarPtII,
                square_planar_penalty,
                square_planar_penalty_batched,
                MetalGeometryPrior,
            )
        except ImportError:
            # One-shot warning so silent no-ops are debuggable.
            if not getattr(self, "_warned_prior_unavailable", False):
                print(
                    "[EGNNVelocityField.metal_geometry_loss] "
                    "molmetal.molmetal_lam.priors.metal_geometry not "
                    "importable; returning zero loss.  Add the priors "
                    "package to sys.path to enable the square-planar "
                    "Pt(II) prior."
                )
                self._warned_prior_unavailable = True
            return torch.zeros(
                (), device=positions.device, dtype=positions.dtype
            )

        effective = (edge_types == 2) if edge_types is not None else dative_bond_edge_attr
        if effective is None:
            return positions.new_zeros(())
        if positions.dim() == 2:
            return prior_weight * square_planar_penalty(positions, edge_index, effective, atom_types)
        penalties = []
        for i in range(positions.shape[0]):
            penalties.append(square_planar_penalty(
                positions[i], edge_index[i] if edge_index.dim() == 3 else edge_index,
                effective[i] if effective.dim() == 2 else effective,
                atom_types[i] if atom_types.dim() == 2 else atom_types))
        return prior_weight * torch.stack(penalties).mean()

    # ------------------------------------------------------------------
    # TODO-09: generalised metal-geometry prior sampling hook
    # ------------------------------------------------------------------
    def apply_metal_geometry_step(
        self,
        positions: torch.Tensor,                  # (B, N, 3) or (N, 3)
        atom_types: torch.Tensor,                  # (B, N) or (N,)
        edge_types: torch.Tensor,                  # (B, E) or (E,) long
        step: int,
        k_every: int = 10,
        weight: float = 0.1,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the metal-geometry soft penalty for a sampling step.

        TODO-09: the prior is applied only on every ``k_every``-th
        integration step (default 10).  Returns zero on off-steps and
        on batches with no metal centre.

        Parameters
        ----------
        positions, atom_types, edge_types : torch.Tensor
            Sampling tensors (batched or single-graph).
        step : int
            Current sampling step index (zero-based).
        k_every : int, default 10
            Apply the prior every ``k_every`` steps.
        weight : float, default 0.1
            Scaling factor on the prior loss — wired to the
            ``--metal-prior-weight`` CLI flag.

        Returns
        -------
        Scalar tensor — ``weight * prior`` on prior-applied steps,
        zero otherwise.
        """
        # Off-step fast path.
        if k_every <= 0 or (step % max(1, k_every)) != 0:
            return torch.zeros(
                (), device=positions.device, dtype=positions.dtype
            )
        try:
            from molmetal.molmetal_lam.priors.metal_geometry import (
                MetalGeometryPrior,
            )
        except ImportError:
            return torch.zeros(
                (), device=positions.device, dtype=positions.dtype
            )
        prior = MetalGeometryPrior(weight=weight)
        if positions.dim() == 2:
            return prior.prior_loss(positions, atom_types, edge_types, edge_index=edge_index)
        penalties = []
        for i in range(positions.shape[0]):
            penalties.append(prior.prior_loss(
                positions[i], atom_types[i] if atom_types.dim() == 2 else atom_types,
                edge_types[i] if edge_types.dim() == 2 else edge_types,
                edge_index=edge_index[i] if edge_index is not None and edge_index.dim() == 3 else edge_index))
        return torch.stack(penalties).mean()

    def forward_velocity(self, x, atom_types, edge_index, t, edge_mask=None,
                          pocket_embed=None):
        """Convenience wrapper returning only the velocity tensor.

        Used by the cloned ``flow_matching`` library's ``ModelWrapper``,
        which calls ``self.model(x=x, t=t, **extras)`` and expects a
        tensor-shaped return value.
        """
        return self.forward(x, atom_types, edge_index, t, edge_mask=edge_mask,
                            pocket_embed=pocket_embed)["vel"]

    # ------------------------------------------------------------------
    # Round-10 axis-C: classifier-free guidance (CFG)
    # ------------------------------------------------------------------
    def v_cfg(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        pocket_embed: Optional[torch.Tensor],
        cfg_scale: float = 2.0,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classifier-free guidance for the EGNN velocity field.

        Computes the standard CFG combination on the velocity tensor only
        (atom-type logits and per-atom h are not used by the ODE solver,
        so we only call ``forward`` twice — saving one full network pass):

            v_cfg = v_uncond + cfg_scale · (v_cond − v_uncond)

        with ``v_cond = forward(pocket_embed)`` and
        ``v_uncond = forward(pocket_embed=None)``.

        Parameters
        ----------
        x, atom_types, edge_index, t, edge_mask
            Same as :meth:`forward`.
        pocket_embed
            ``(B, H)`` conditioning tensor.  When ``None`` the call
            degenerates to ``v_uncond`` (i.e. ``cfg_scale`` is a no-op —
            we still return ``v_uncond`` directly, which makes
            ``v_cfg(..., pocket_embed=None, cfg_scale=anything)`` safe
            for unconditional sampling).
        cfg_scale
            Guidance strength.  ``cfg_scale=1.0`` recovers the
            conditional velocity (``v_cfg == v_cond``); ``cfg_scale=0``
            would give ``v_uncond``.  Defaults to ``2.0`` (the Round-10
            axis-C default).

        Returns
        -------
        torch.Tensor
            ``(B, N, 3)`` velocity tensor with CFG applied.  Same shape
            and dtype as ``forward(...)["vel"]``.
        """
        if cfg_scale == 1.0 and pocket_embed is not None:
            # Common fast-path: pure conditional sampling — skip the
            # second forward call entirely.
            return self.forward_velocity(
                x, atom_types, edge_index, t,
                pocket_embed=pocket_embed, edge_mask=edge_mask,
            )
        # Compute v_cond and v_uncond on the same graph.
        # We are in eval-mode for inference CFG; switch temporarily so
        # the training-time context_dropout is disabled (it would
        # randomly zero-out the conditioning even for v_cond otherwise).
        was_training = self.training
        if was_training:
            self.eval()
        try:
            v_cond = (
                self.forward_velocity(
                    x, atom_types, edge_index, t,
                    pocket_embed=pocket_embed, edge_mask=edge_mask,
                )
                if pocket_embed is not None else None
            )
            v_uncond = self.forward_velocity(
                x, atom_types, edge_index, t,
                pocket_embed=None, edge_mask=edge_mask,
            )
        finally:
            if was_training:
                self.train()
        if pocket_embed is None or v_cond is None:
            # Unconditional sampling — cfg_scale is irrelevant.
            return v_uncond
        # Standard CFG combination: v_uncond + s · (v_cond − v_uncond).
        diff = v_cond - v_uncond
        return v_uncond + cfg_scale * diff


# ---------------------------------------------------------------------------
# Adapter: MoleculeGenerator port = Lipman 2023 CFM + our EGNN
# ---------------------------------------------------------------------------
class LipmanFlowMatchingAdapter(MoleculeGenerator):
    """MoleculeGenerator implementation using Lipman et al. 2023 FM + our EGNN.

    Training loss = Conditional Flow Matching (Lipman 2023 §4.5) on the
    Optimal-Transport path (Lipman 2023 §4.7) with AffineProbPath
    (Lipman 2023 §4.8):

        x_t = α_t x_1 + σ_t x_0        where α_t = t, σ_t = 1 − t
        target_dx_t = x_1 - x_0
        loss = MSE(v_θ(x_t, t), target_dx_t)
    """

    @property
    def name(self) -> str:
        return "LipmanFlowMatching_v1"

    def __init__(
        self,
        ref_repo_path: str = "molmetal/references/flow_matching",
        hidden_dim: int = 128,
        n_layers: int = 3,
        max_atomic_number: int = 100,
        lr: float = 1e-4,
        atom_loss_weight: float = 0.1,
        metal_prior_weight: float = 0.1,
        metal_prior_k_every: int = 10,
        tmqm_init_path: Optional[str] = None,
        use_tmqm_init: bool = False,
        context_dropout: float = 0.1,
        cfg_scale: float = 1.0,
        pocket_embed_scale: float = 0.1,
    ) -> None:
        self._ref_repo = ref_repo_path
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._max_atomic_number = max_atomic_number
        self._lr = lr
        self._atom_loss_weight = atom_loss_weight
        # TODO-09: metal-geometry prior weights (soft constraint on the
        # generated sampling trajectory).  Defaults match the spec.
        self._metal_prior_weight = float(metal_prior_weight)
        self._metal_prior_k_every = int(metal_prior_k_every)
        # TODO-08: tmQM pre-trained encoder initialiser.
        # ``tmqm_init_path=None`` + ``use_tmqm_init=True`` uses the
        # DEFAULT_TMQM_CKPT (F2 pre-training report).
        self._tmqm_init_path = tmqm_init_path
        self._use_tmqm_init = bool(use_tmqm_init)
        # Round-10 axis-C: context dropout (training) + CFG (inference).
        # ``cfg_scale=1.0`` recovers the legacy conditional sampler
        # bit-exactly (no extra forward pass).  ``cfg_scale>1.0`` runs
        # ``v_cfg`` instead of ``forward_velocity`` — which doubles the
        # ODE-step cost when pocket_embed is set.
        if cfg_scale < 0.0:
            raise ValueError(
                f"cfg_scale must be >= 0.0, got {cfg_scale!r}"
            )
        self._context_dropout = float(context_dropout)
        self._cfg_scale = float(cfg_scale)
        import math
        if not math.isfinite(pocket_embed_scale) or pocket_embed_scale < 0:
            raise ValueError("pocket_embed_scale must be finite and nonnegative")
        self._pocket_embed_scale = float(pocket_embed_scale)

        # Will be set in setup()
        self.velocity_field: Optional[EGNNVelocityField] = None
        self.pocket_encoder: Optional[PocketEncoder] = None
        self.path = None
        self.scheduler = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        # ROCm-first: device is resolved at setup() via get_device(); we
        # only use the constant here for introspection before setup().
        self._device: str = DEFAULT_DEVICE
        self.device: torch.device = torch.device(DEFAULT_DEVICE)
        self.device_info: dict = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, device: Optional[str] = None) -> None:
        """Initialise the velocity field, scheduler, path, optimizer.

        ``device=None`` (the recommended call style) auto-detects ROCm/CUDA
        via :func:`molmetal.utils.device.get_device` and falls back to CPU
        with a one-shot warning.  Pass an explicit ``"cuda:0"`` /
        ``"cuda:1"`` / ``"cpu"`` to override.
        """
        AffineProbPath, CondOTScheduler, ODESolver, ModelWrapper = _import_fm_lib(
            self._ref_repo
        )
        # AffineProbPath wraps path + scheduler.  The OT scheduler gives
        # the straight-line path (α_t = t, σ_t = 1 − t).
        self.scheduler = CondOTScheduler()
        self.path = AffineProbPath(scheduler=self.scheduler)
        self._ODESolver = ODESolver
        self._ModelWrapper = ModelWrapper

        # Resolve device: explicit > auto-detect ROCm/CUDA > CPU.
        # Always run verify_rocm_active() and stash the result so smoke
        # tests can assert the GPU is actually being used.
        self.device_info = verify_rocm_active()
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        self._device = str(self.device)

        self.velocity_field = EGNNVelocityField(
            hidden_dim=self._hidden_dim,
            n_layers=self._n_layers,
            max_atomic_number=self._max_atomic_number,
            tmqm_init_path=self._tmqm_init_path,
            use_tmqm_init=self._use_tmqm_init,
            context_dropout=self._context_dropout,
        ).to(self.device)
        # T5: pocket encoder shares hidden_dim with the velocity field so
        # the broadcast bias is a no-op shape match.
        self.pocket_encoder = PocketEncoder(
            hidden_dim=self._hidden_dim,
            max_atomic_number=self._max_atomic_number,
        ).to(self.device)
        # Optimizer now jointly updates both networks.
        self.optimizer = torch.optim.AdamW(
            list(self.velocity_field.parameters())
            + list(self.pocket_encoder.parameters()),
            lr=self._lr,
        )

    # ------------------------------------------------------------------
    # Train step: one batch of CFM loss (Lipman 2023 §4.5)
    # ------------------------------------------------------------------
    def train_step(
        self,
        pocket,                    # Pocket | List[Pocket] | None
        mols: List[Molecule],
        atom_loss_weight: Optional[float] = None,
    ) -> float:
        """One CFM + atom-type-CE training step.

        T5 update: when ``pocket`` is non-None, the pocket is encoded
        into a (B, H) tensor via :class:`PocketEncoder` and broadcast as
        additive bias into the velocity field's atom embedding.  The
        AffineProbPath ``x_1`` sampling path is **unchanged** — pocket
        conditioning enters v_θ only, which is the canonical SBDD-FM
        pattern (TargetDiff / DiffSBDD).

        Returns the TOTAL loss (CFM + α · atom-CE) as a float, matching
        the :class:`molmetal.ports.MoleculeGenerator` Protocol contract.
        Individual loss components are stored on ``self.last_losses``
        for callers that want to log them separately.
        """
        if atom_loss_weight is None:
            atom_loss_weight = self._atom_loss_weight
        assert self.path is not None, "call setup() first"

        b = len(mols)
        max_n = max(m.coords.shape[0] for m in mols)

        # Build batched tensors — all on the resolved device (ROCm/CUDA
        # when available; CPU otherwise).
        x_1 = torch.zeros(b, max_n, 3, device=self.device)
        atom_types = torch.zeros(b, max_n, dtype=torch.long, device=self.device)
        node_mask = torch.zeros(b, max_n, dtype=torch.bool, device=self.device)
        for i, m in enumerate(mols):
            n = m.coords.shape[0]
            x_1[i, :n] = m.coords.to(self.device)
            atom_types[i, :n] = m.atom_types.to(self.device)
            node_mask[i, :n] = True

        # T5: pocket conditioning — encode pocket(s) into a (B, H) bias.
        # Returns None when pocket=None so the velocity field takes its
        # unconditioned path.
        pocket_embed = self._encode_pocket(pocket, b=b, max_n_atoms=max_n,
                                            device=self.device)

        # Sample noise + time (Lipman 2023 CFM).  ``torch.randn_like(x_1)``
        # inherits x_1's device, so x_0 is also on self.device.
        x_0 = torch.randn_like(x_1)
        t = torch.rand(b, device=self.device)

        # Sample x_t and target velocity from the cloned library
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        # path_sample.x_t shape (B, max_n, 3)
        # path_sample.dx_t shape (B, max_n, 3)  — conditional velocity x_1 - x_0

        # Velocity field prediction (joint: coord velocity + atom-type logits)
        edge_index = self._make_dummy_edge_index(b, max_n, self.device)
        # Identity is a prediction target, not an input available to the
        # unconstrained sampler. Supplying the target atomic numbers here
        # lets the atom head copy them and leaves the sampling input unseen.
        out = self.velocity_field(
            path_sample.x_t, torch.zeros_like(atom_types), edge_index, t,
            pocket_embed=pocket_embed,
        )
        v_pred = out["vel"]
        atom_logits = out["atom_logits"]  # (B, max_n, max_atomic_number)

        # ---- Loss 1: CFM coord velocity (MSE, masked) ----
        mask_3d = node_mask.unsqueeze(-1).to(v_pred.dtype)
        per_atom_loss = (v_pred - path_sample.dx_t) ** 2 * mask_3d
        denom = mask_3d.sum().clamp(min=1.0)
        cfm_loss = per_atom_loss.sum() / denom

        # ---- Loss 2: atom-type cross-entropy (masked) ----
        # Flatten over batch and nodes, then drop padded positions.
        # Padded positions have atom_types=0 (a valid class — Carbon!)
        # so we must mask first.  We use node_mask to select real atoms.
        # Clamp logits to avoid -inf when the atom-type head is untrained
        # (zero-initialised → uniform, but some classes may have near-zero
        # probs which F.cross_entropy handles fine).
        atom_loss_per_node = F.cross_entropy(
            atom_logits.reshape(-1, atom_logits.shape[-1]),
            atom_types.reshape(-1),
            reduction="none",
        ).view(b, max_n)
        n_real = node_mask.sum().clamp(min=1)
        atom_loss = atom_loss_per_node[node_mask].sum() / n_real

        loss = cfm_loss + atom_loss_weight * atom_loss

        self.optimizer.zero_grad()
        loss.backward()
        # Grad clip (matching MolFlow-Triton defaults).  T5: include the
        # pocket encoder in the clip so a spike in pocket-encoder grads
        # can't blow up the joint optimisation.
        params = (list(self.velocity_field.parameters())
                  + list(self.pocket_encoder.parameters()))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.optimizer.step()
        # Stash individual losses for logging; return the total as a float
        # to honour the Protocol contract.
        self.last_losses = {
            "cfm": float(cfm_loss.item()),
            "atom": float(atom_loss.item()),
            "total": float(loss.item()),
        }
        return self.last_losses["total"]

    # ------------------------------------------------------------------
    # Generate: integrate the ODE
    # ------------------------------------------------------------------
    def generate(self, pocket, config: GenerationConfig) -> List[Molecule]:
        """Sample with a fixed seed and deterministic inference conditioning.

        Training/evaluation mode is restored even when the solver fails. The
        dedicated generator does not disturb global training RNG state.
        """
        assert self.path is not None, "call setup() first"
        if config.n_samples < 1 or config.n_steps < 1:
            raise ValueError("n_samples and n_steps must be positive")
        field_modes = [(module, module.training) for root in (self.velocity_field, self.pocket_encoder)
                       for module in root.modules()]
        self.velocity_field.eval()
        self.pocket_encoder.eval()
        try:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)
            return self._generate_impl(pocket, config, generator)
        finally:
            for module, training in field_modes:
                module.training = training

    def _generate_impl(
        self,
        pocket,                    # Pocket | List[Pocket] | None
        config: GenerationConfig,
        generator: torch.Generator,
    ) -> List[Molecule]:
        """Sample ``n_samples`` molecules by integrating dx/dt = v_θ(x, t).

        Joint atom-type + coordinate sampling: at the end of ODE integration
        we use the trained ``atom_head`` logits to sample a real atomic
        number per atom position (replacing the previous ``randint(1,10)``
        placeholder).  Atom-type logits sharpen over training because
        ``train_step`` includes a masked cross-entropy loss.

        T5: pocket conditioning — when provided, the pocket is encoded
        once via :class:`PocketEncoder` and held fixed during ODE
        integration.  The cloned ``ModelWrapper`` forwards
        ``**extras`` so we pass ``pocket_embed`` through it.
        """
        assert self.path is not None, "call setup() first"
        n_samples = config.n_samples
        # Fixed-size point cloud.  Could be replaced by per-sample atom-count
        # sampling in a future version.
        fixed_spec = config.conditioning.get("fixed_atom_types")
        inferred_n = torch.as_tensor(fixed_spec).shape[-1] if fixed_spec is not None else 8
        n_atoms = getattr(config, "n_atoms", inferred_n)
        # All sampling tensors live on self.device so the ODE solver runs
        # on GPU end-to-end when ROCm is available.
        # Initial atom_types are placeholders that get REPLACED after the
        # ODE integration (see step 5 below).
        atom_types = torch.zeros(n_samples, n_atoms, dtype=torch.long, device=self.device)
        fixed_atoms = config.conditioning.get("fixed_atom_types")
        if fixed_atoms is not None:
            fixed_atoms = torch.as_tensor(fixed_atoms, dtype=torch.long, device=self.device)
            if fixed_atoms.shape == (n_atoms,):
                fixed_atoms = fixed_atoms.unsqueeze(0).expand(n_samples, -1)
            if fixed_atoms.shape != (n_samples, n_atoms):
                raise ValueError("fixed_atom_types must have shape (N,) or (B, N)")
            if (fixed_atoms <= 0).any() or (fixed_atoms >= self._max_atomic_number).any():
                raise ValueError("fixed_atom_types outside the model atomic-number vocabulary")
            atom_types = fixed_atoms
        edge_index = self._make_dummy_edge_index(n_samples, n_atoms, self.device)

        # T5: encode pocket once (broadcast to all samples in this batch).
        # Returns None when pocket=None so the velocity field uses the
        # unconditioned (zero-bias) path.
        pocket_embed = self._encode_pocket(pocket, b=n_samples,
                                            max_n_atoms=n_atoms,
                                            device=self.device)

        # 1. Sample noise
        x_0 = torch.randn(n_samples, n_atoms, 3, device=self.device, generator=generator)
        # 2. Time grid (Lipman 2023 default = [0, 1])
        t_grid = torch.linspace(0.0, 1.0, config.n_steps + 1, device=self.device)
        # 3. Wrap velocity field in the library's ModelWrapper.
        # The library's ModelWrapper stores ``self.model`` and forwards
        # ``(x, t, **extras)`` to it.  We use ``forward_velocity`` which
        # returns just the velocity tensor (matches what ODESolver expects).
        # T5: pass pocket_embed through extras so the wrapper forwards it.
        # Round-10 axis-C: when ``cfg_scale > 1.0`` *and* we have a
        # pocket, route through ``v_cfg`` so we get
        # ``v_uncond + cfg_scale * (v_cond − v_uncond)``.  When
        # ``cfg_scale == 1.0`` we keep the legacy ``forward_velocity``
        # path bit-exact.
        from functools import partial
        cfg_scale = float(getattr(self, "_cfg_scale", 1.0) or 1.0)
        if cfg_scale != 1.0 and pocket_embed is not None:
            velocity_model = partial(
                self.velocity_field.v_cfg,
                atom_types=atom_types,
                edge_index=edge_index,
                pocket_embed=pocket_embed,
                cfg_scale=cfg_scale,
            )
        else:
            velocity_model = partial(
                self.velocity_field.forward_velocity,
                atom_types=atom_types,
                edge_index=edge_index,
                pocket_embed=pocket_embed,
            )

        # Geometry conditioning requires known identities and real donor ->
        # metal endpoints. Unconstrained categorical sampling has neither.
        metal_prior_k = int(self._metal_prior_k_every)
        metal_prior_w = float(self._metal_prior_weight)
        donor_edges = config.conditioning.get("dative_edge_index")
        diagnostics = {"status": "inactive", "reason": "missing_fixed_atom_types_or_dative_edges",
                       "weight": metal_prior_w, "gradient_evaluations": 0,
                       "nonzero_gradient_graph_updates": 0,
                       "fixed_atom_types": fixed_atoms is not None}
        self.last_metal_prior_diagnostics = diagnostics
        prior = None
        if donor_edges is not None:
            if fixed_atoms is None:
                raise ValueError("dative_edge_index requires fixed_atom_types; sampled identities cannot define prior donors in advance")
            donor_edges = torch.as_tensor(donor_edges, dtype=torch.long, device=self.device)
            if donor_edges.dim() == 2 and donor_edges.shape[0] == 2:
                donor_edges = donor_edges.unsqueeze(0).expand(n_samples, -1, -1)
            if donor_edges.dim() != 3 or donor_edges.shape[:2] != (n_samples, 2):
                raise ValueError("dative_edge_index must have shape (2, E) or (B, 2, E)")
            if (donor_edges < 0).any() or (donor_edges >= n_atoms).any():
                raise ValueError("dative_edge_index contains invalid atom endpoints")
            from molmetal.molmetal_lam.priors.metal_geometry import MetalGeometryPrior
            prior = MetalGeometryPrior(weight=metal_prior_w,
                metal_geometry_map=config.conditioning.get("metal_geometry_map"))
            # Check declared edge destinations actually identify known metals.
            metal_set = atom_types.new_tensor(list(prior.metal_geometry_map))
            destinations = torch.gather(atom_types, 1, donor_edges[:, 1])
            if not (destinations.unsqueeze(-1) == metal_set).any(-1).all():
                raise ValueError("Every dative edge destination must be a supported metal centre")
            if donor_edges.shape[-1] == 0:
                diagnostics["reason"] = "empty_dative_graph"
                prior = None
        if metal_prior_w <= 0 or metal_prior_k <= 0:
            diagnostics["reason"] = "disabled_by_weight_or_interval"
            prior = None
        step_counter = 0
        if prior is not None:
            diagnostics.update(status="eligible", reason="explicit_fixed_atoms_and_donor_edges")

        def _velocity_with_metal_prior(x, t):
            nonlocal step_counter
            base = velocity_model(x=x, t=t)
            apply_now = step_counter % metal_prior_k == 0
            step_counter += 1
            if not apply_now:
                return base
            # ODESolver runs under no_grad: explicitly enable coordinate
            # gradients, sum per-graph energies to avoid batch-size dilution.
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                energies = [prior.prior_loss(x_req[i], atom_types[i],
                    torch.full((donor_edges.shape[-1],), 2, dtype=torch.long, device=x.device),
                    edge_index=donor_edges[i]) for i in range(n_samples)]
                energy = torch.stack(energies).sum()
                diagnostics["gradient_evaluations"] += 1
                if not energy.requires_grad:
                    return base
                gradient = torch.autograd.grad(energy, x_req)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Metal geometry prior produced nonfinite gradients")
            updates = int((gradient.square().sum(dim=(1, 2)) > 0).sum())
            diagnostics["nonzero_gradient_graph_updates"] += updates
            if updates:
                diagnostics["status"] = "active"
            return base - gradient.detach()

        wrapper = self._ModelWrapper(model=_velocity_with_metal_prior if prior is not None else velocity_model)
        # 4. Solve the ODE
        solver = self._ODESolver(velocity_model=wrapper)
        x_final = solver.sample(
            x_init=x_0,
            step_size=1.0 / config.n_steps,
            method="euler",
            time_grid=t_grid,
        )
        # x_final shape: (n_samples, n_atoms, 3) or (T+1, n_samples, n_atoms, 3)
        if x_final.dim() == 4:
            x_final = x_final[-1]
        if not torch.isfinite(x_final).all():
            raise FloatingPointError("Flow integration produced non-finite coordinates")
        # 5. Predict atomic numbers using the same masked identity input
        # used in training and integration. Fixed identity conditioning is
        # a separate supplied constraint, preserved in the output below.
        with torch.no_grad():
            init_atom_types = atom_types
            # Round-10 axis-C: when CFG is enabled, average the
            # conditional and unconditional atom-type logits so the
            # categorical sampling matches the conditional velocity
            # distribution we integrated with.  cfg_scale=1.0 keeps the
            # legacy pure-conditional forward path bit-exact.
            if cfg_scale != 1.0 and pocket_embed is not None:
                out_cond = self.velocity_field(
                    x_final, init_atom_types, edge_index,
                    torch.ones(n_samples, device=self.device),
                    pocket_embed=pocket_embed,
                )
                out_uncond = self.velocity_field(
                    x_final, init_atom_types, edge_index,
                    torch.ones(n_samples, device=self.device),
                    pocket_embed=None,
                )
                # CFG-style logits interpolation (linear in logit space).
                atom_logits = (
                    out_uncond["atom_logits"]
                    + cfg_scale * (out_cond["atom_logits"] - out_uncond["atom_logits"])
                )
            else:
                out = self.velocity_field(
                    x_final, init_atom_types, edge_index,
                    torch.ones(n_samples, device=self.device),
                    pocket_embed=pocket_embed,
                )
                atom_logits = out["atom_logits"]  # (n_samples, n_atoms, max_z)
            # Mask out impossible atoms (Z=0 "padding") by setting their
            # logit to -inf so they are never sampled.
            atom_logits[..., 0] = float("-inf")
            atom_probs = F.softmax(atom_logits, dim=-1)
            # Categorical sample per atom
            if not torch.isfinite(atom_probs).all():
                raise FloatingPointError("Atom prediction produced non-finite probabilities")
            sampled_atoms = torch.multinomial(atom_probs.reshape(-1, atom_probs.shape[-1]), 1,
                                             generator=generator).reshape(n_samples, n_atoms)
        if fixed_atoms is not None:
            sampled_atoms = fixed_atoms
        # 6. Build Molecule objects with REAL atom types (placeholder bonds;
        # Phase 1 will add proper bond decoder).
        mols: List[Molecule] = []
        for i in range(n_samples):
            mols.append(Molecule(
                coords=x_final[i].detach().cpu(),
                atom_types=sampled_atoms[i].detach().cpu(),
                bonds=torch.zeros(2, 0, dtype=torch.long),
                bond_types=torch.zeros(0, dtype=torch.long),
                formal_charges=torch.zeros(n_atoms, dtype=torch.long),
            ))
        return mols

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_pocket(
        self,
        pocket,                         # Pocket | List[Pocket] | None
        b: int,
        max_n_atoms: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Encode one (or a list of) Pocket(s) into a (B, H) tensor.

        Returns ``None`` when ``pocket is None`` so the downstream
        velocity field can take its unconditioned code path (zero bias).

        Accepts:
        - ``pocket=None`` → return ``None`` (no conditioning).
        - single ``Pocket`` → broadcast to all ``b`` samples.
        - ``list[Pocket]`` of length ``b`` → one pocket per sample.
        - ``list[Pocket]`` shorter than ``b`` → broadcast (last pocket
          repeated).

        Padding atom_types are kept at 0 (a valid atomic number but
        flagged invalid via ``mask=False``); the PocketEncoder ignores
        them via :attr:`mask`.
        """
        if pocket is None:
            return None
        # Normalise to a list of length ``b``.
        if isinstance(pocket, list):
            assert len(pocket) >= 1, "pocket list cannot be empty"
            pockets = list(pocket)
        else:
            pockets = [pocket]
        # Move all pockets onto device once.
        pockets = [p.to(device) if hasattr(p, "to") else p for p in pockets]
        # Per-pocket atom counts.
        n_pocket_atoms = [p.coords.shape[0] for p in pockets]
        max_p = max(n_pocket_atoms) if n_pocket_atoms else max_n_atoms
        # Build batched tensors (B, max_p, 3) / (B, max_p) / (B, max_p) bool
        pocket_coords = torch.zeros(b, max_p, 3, device=device)
        pocket_atom_types = torch.zeros(b, max_p, dtype=torch.long, device=device)
        pocket_mask = torch.zeros(b, max_p, dtype=torch.bool, device=device)
        for i in range(b):
            # Broadcast single-pocket across batch.
            pk = pockets[i] if i < len(pockets) else pockets[-1]
            n = pk.coords.shape[0]
            pocket_coords[i, :n] = pk.coords.to(device)
            pocket_atom_types[i, :n] = pk.atom_types.to(device).long()
            pocket_mask[i, :n] = True
        # Encode.
        embed = self.pocket_encoder(pocket_coords, pocket_atom_types, pocket_mask)
        return embed * self._pocket_embed_scale

    @staticmethod
    def _make_dummy_edge_index(b: int, n: int, device) -> torch.Tensor:
        """Fully-connected (no self-loop) edge index for ``b`` graphs.

        Returned shape ``(b, 2, n*(n-1))``.  Accepts either a string or a
        :class:`torch.device` for ``device`` — torch handles both.
        """
        idx = torch.arange(n, device=device)
        src = idx.view(1, n, 1).expand(b, n, n)
        dst = idx.view(1, 1, n).expand(b, n, n)
        mask = src != dst
        src = src[mask].view(b, -1)
        dst = dst[mask].view(b, -1)
        return torch.stack([src, dst], dim=1)

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "metal_prior_last_generation": dict(getattr(self, "last_metal_prior_diagnostics", {"status": "not_run"})),
            "metal_prior_contract": "requires conditioning.fixed_atom_types and conditioning.dative_edge_index; unconstrained sampling is inactive",
            "atom_training_contract": "masked Z=0 inputs; true atomic numbers only as cross-entropy targets; old teacher-forced checkpoints require retraining",
            "pocket_embed_scale": self._pocket_embed_scale,
            "sampling_seed": "GenerationConfig.seed via device-local torch.Generator",
            "sampling_mode": "eval with caller module modes restored",
            "model": "LipmanFlowMatching_v1",
            "paper": "Lipman et al. 2023, ICLR 2023",
            "arxiv": "2210.02747",
            "official_code": "github.com/facebookresearch/flow_matching",
            "path": "AffineProbPath (Lipman 2023 §4.8)",
            "scheduler": "CondOTScheduler (Lipman 2023 §4.7) — straight-line OT path",
            "loss": "Conditional Flow Matching (Lipman 2023 §4.5)",
            "velocity_field": "EGNN (MolFlow-Triton, models.velocity_net.EGNNLayer)",
            "pocket_conditioning": (
                "PocketEncoder (T5) — per-pocket-atom 1-hot coords via "
                "EGNNLayer + global mean pool, fed into v_θ as additive "
                "bias on atom embeddings.  AffineProbPath x_1 sampling is "
                "unchanged (TargetDiff / DiffSBDD style)."
            ),
            "scm": "git clone https://github.com/facebookresearch/flow_matching.git molmetal/references/flow_matching",
        }


# A canonical config preset for our smoke tests
FM_OT_AFFINE_CONFIG = dict(
    scheduler_name="CondOTScheduler",
    path_name="AffineProbPath",
    alpha_t=lambda t: t,
    sigma_t=lambda t: 1.0 - t,
    target_velocity=lambda x_0, x_1: x_1 - x_0,
)


def build_argparser() -> "argparse.ArgumentParser":
    """CLI argparse helper for the FM adapter (TODO-09 wires --metal-prior-weight).

    Returns a fresh :class:`argparse.ArgumentParser` with the standard
    Lipman-FM training flags plus the TODO-09 metal-geometry prior
    knobs.  Used by the round-8 CLI entry points; safe to call without
    instantiating an adapter.

    Example
    -------
    >>> parser = build_argparser()
    >>> args = parser.parse_args(["--metal-prior-weight", "0.2",
    ...                           "--metal-prior-k-every", "5"])
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="lipman-fm",
        description="Lipman 2023 FM adapter for Mol-Metal (with TODO-09 metal prior)",
    )
    p.add_argument("--ref-repo", default="molmetal/references/flow_matching",
                   help="Path to the cloned facebookresearch/flow_matching repo")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--max-atomic-number", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--atom-loss-weight", type=float, default=0.1)
    # TODO-09: metal-geometry prior knobs.
    p.add_argument("--metal-prior-weight", type=float, default=0.1,
                   help="Weight applied to the metal-geometry soft penalty "
                        "during sampling.  Set to 0.0 to disable the prior "
                        "(legacy bit-exact path).")
    p.add_argument("--metal-prior-k-every", type=int, default=10,
                   help="Apply the metal-geometry prior every K integration "
                        "steps during sampling.  Defaults to 10.")
    # TODO-08: tmQM pre-trained encoder initialiser knobs.
    p.add_argument("--tmqm-init", nargs="?", const="__default__",
                   default=None,
                   help="Warm-start the EGNN encoder from the tmQM pre-trained "
                        "checkpoint (DMPNN; see molmetal/reports/f2_tmqm_pretrain.md). "
                        "Pass --tmqm-init to use the default checkpoint path, or "
                        "--tmqm-init=/path/to/ckpt.pt for an explicit file.  Use "
                        "--random-init to opt out.")
    p.add_argument("--random-init", action="store_true",
                   help="Disable the tmQM warm-start even if --tmqm-init is passed "
                        "(equivalent to not loading any encoder checkpoint).")
    # TODO-10 axis-C: classifier-free guidance knobs.
    p.add_argument("--context-dropout", type=float, default=0.1,
                   help="Probability of dropping the conditioning pocket "
                        "embedding during training.  Teaches the model the "
                        "unconditional marginal v(v | ∅) needed for CFG at "
                        "inference time.  Default 0.1 (Ho & Salimans 2022).")
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="Classifier-free guidance scale used at inference.  "
                        "1.0 = legacy conditional sampler (bit-exact); "
                        "2.0 = standard CFG strength; >=3.0 = strong guidance.  "
                        "Set to 1.0 to disable CFG.")
    return p
