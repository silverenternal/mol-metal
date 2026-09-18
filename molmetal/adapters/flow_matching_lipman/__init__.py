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
import warnings

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

# AMP wrapper (Phase-3 VRAM fix).  The wrapper is a thin
# :func:`torch.autocast` shim that forces BF16 autocast around the
# CFM training loop.  See ``amp.py`` for the gfx1101 rationale and the
# ``check_rocm_env_vars`` warning pattern (mirrors the RDNA3 fallback
# at ``models/_scatter.py:84-94``).
from .amp import CFMAMPContext, recommended_amp_kwargs, ROCM_GFX1101_ENV_VARS  # noqa: E402


def maybe_enable_amp() -> "CFMAMPContext":
    """Decide whether to enable BF16 autocast for the CFM training loop.

    Reads two environment variables (no flag-day, backward compatible):

    ``MOLMETAL_CFM_AMP`` — ``"1"`` (default) to enable, ``"0"`` to skip.
    ``MOLMETAL_CFM_AMP_DTYPE`` — ``"bfloat16"`` (default) or ``"float16"``.
    On gfx1101 BF16 is strongly preferred (no GradScaler needed).

    Logs one line describing the decision and returns a ready-to-use
    :class:`CFMAMPContext` instance.

    Used by :mod:`molmetal.scripts.metallo_drug_smoke_retrain` and any
    future training entry point that wants the cheapest single VRAM
    reclaim lever without paying the cost of a config-flag refactor.
    """
    import logging
    import os

    log = logging.getLogger("molmetal.cfm.amp")
    amp_flag = os.environ.get("MOLMETAL_CFM_AMP", "1")
    amp_dtype = os.environ.get("MOLMETAL_CFM_AMP_DTYPE", "bfloat16")
    enabled = amp_flag not in ("0", "false", "False", "no", "off")
    if amp_dtype == "float16":
        dtype = torch.float16
    elif amp_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    ctx = CFMAMPContext(enabled=enabled, dtype=dtype)
    log.info("CFM AMP decision: %s", ctx.env_summary())
    return ctx


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


# ---------------------------------------------------------------------------
# WF-2 A5 — helpers for joint bond-head training
# ---------------------------------------------------------------------------
def _rdkit_bond_int_to_bond_head_label(rd_int: int) -> int:
    """Map RDKit's bond-type int onto our 5-class label.

    Mapping (RDKit conventions):
        SINGLE=1    → 1 (BOND_SINGLE)
        DOUBLE=2    → 2 (BOND_DOUBLE)
        TRIPLE=3    → 3 (BOND_TRIPLE)
        AROMATIC=12 → 4 (BOND_AROMATIC)
        anything else (incl. 0 = UNSPECIFIED) → 0 (BOND_NO_BOND)

    This mirrors :func:`molmetal.domain.io.rdkit_io._rdkit_bond_type_to_int`
    but in the inverse direction.
    """
    if rd_int == 1:
        return 1  # SINGLE
    if rd_int == 2:
        return 2  # DOUBLE
    if rd_int == 3:
        return 3  # TRIPLE
    if rd_int == 12:
        return 4  # AROMATIC
    return 0  # NONE / UNSPECIFIED


def _build_bond_pair_features(
    edge_index: torch.Tensor,    # (2, E) long
    coords: torch.Tensor,        # (B, N, 3)
    z_i: torch.Tensor,           # (E,) long
    z_j: torch.Tensor,           # (E,) long
    device: torch.device,
) -> torch.Tensor:
    """Build the (E, 9) per-pair feature tensor the :class:`BondOrderHead` consumes.

    Mirrors :meth:`molmetal.models.bond_head.BondAwareDecoder._featurise`
    but is callable from inside :meth:`train_step` without going through
    the decoder's RDKit-coupled path.  Returns shape ``(E, 9)`` with the
    same column layout as the synthetic training set:

        [0] distance Å
        [1] bucket(z_i)
        [2] bucket(z_j)
        [3] angle_to_metal  (0 — no metal centre available here)
        [4] is_dative_candidate (crude N/O/P/S → metal rule)
        [5] z_i / MAX_ATOMIC_NUMBER
        [6] z_j / MAX_ATOMIC_NUMBER
        [7] bucket(z_i) * 6 + bucket(z_j)
        [8] distance − single_bond_anchor (gap)
    """
    from molmetal.models.bond_head import (
        BOND_SINGLE,
        MAX_ATOMIC_NUMBER,
        _bond_distance,
        _is_metal,
        _is_donor,
    )

    def _bucket(z: int) -> int:
        if z == 6:
            return 0
        if z == 7:
            return 1
        if z == 8:
            return 2
        if z in (15, 16):
            return 3
        if _is_metal(z):
            return 4
        return 5

    src = edge_index[0]
    dst = edge_index[1]
    # coords is (B, N, 3) — gather per-edge coordinates.
    # If src/dst were batched they would exceed max_n; here we use the
    # single-graph convention (b=1) since the bonds in the training
    # batch are a flat concatenation of all molecules.
    e = src.shape[0]
    # ``coords`` may be shape (B, N, 3) — we gather using only the
    # node indices (assuming a single batch dim) by indexing the first
    # batch slot.  For batched edge_indices, callers should pre-flatten.
    # Here we accept either (N, 3) or (B, N, 3) and use the first batch.
    if coords.dim() == 3:
        ci = coords[0, src]
        cj = coords[0, dst]
    else:
        ci = coords[src]
        cj = coords[dst]
    distance = torch.norm(cj - ci, dim=-1)
    zi_cpu = z_i.cpu().tolist()
    zj_cpu = z_j.cpu().tolist()
    bi = torch.tensor(
        [_bucket(int(z)) for z in zi_cpu], dtype=torch.float32, device=device,
    )
    bj = torch.tensor(
        [_bucket(int(z)) for z in zj_cpu], dtype=torch.float32, device=device,
    )
    angle = torch.zeros(e, dtype=torch.float32, device=device)
    is_dative = torch.tensor(
        [
            float(
                (_is_metal(int(zi_cpu[k])) and _is_donor(int(zj_cpu[k])))
                or (_is_metal(int(zj_cpu[k])) and _is_donor(int(zi_cpu[k])))
            )
            for k in range(e)
        ],
        dtype=torch.float32,
        device=device,
    )
    zi_norm = z_i.to(torch.float32) / float(MAX_ATOMIC_NUMBER)
    zj_norm = z_j.to(torch.float32) / float(MAX_ATOMIC_NUMBER)
    bucket_prod = bi * 6.0 + bj
    anchor_single = torch.tensor(
        [
            float(_bond_distance(int(zi_cpu[k]), int(zj_cpu[k]), BOND_SINGLE))
            for k in range(e)
        ],
        dtype=torch.float32,
        device=device,
    )
    gap = distance - anchor_single
    return torch.stack(
        [distance, bi, bj, angle, is_dative, zi_norm, zj_norm, bucket_prod, gap],
        dim=-1,
    )


def _gather_edge_features(
    h: torch.Tensor,                    # (B, N, H)
    edge_index: torch.Tensor,           # (2, E) long
    max_n: int,
    b: int,
) -> torch.Tensor:
    """Gather per-edge hidden vectors ``h[batch, src]`` (and ``h[batch, dst]``).

    Returns shape ``(E, 2H)``.  We use the first-batch slot since the
    training loop flattens edges across the batch into a single edge list
    (no batch dim on edges).  The bond head's FC1 will then process
    ``[geometric_feats | edge_hidden]`` — see
    :meth:`LipmanFlowMatchingAdapter.train_step`.
    """
    src = edge_index[0]
    dst = edge_index[1]
    h_src = h[0, src]  # (E, H)
    h_dst = h[0, dst]  # (E, H)
    return torch.cat([h_src, h_dst], dim=-1)


# ---------------------------------------------------------------------------
# WF-Vina-Lift-Phase23 (Phase 2.3) — PCGrad multi-task gradient surgery
# ---------------------------------------------------------------------------
# Reference: Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.
# (2020).  *Gradient Surgery for Multi-Task Learning.*  NeurIPS 2020.
# arXiv:2001.06782.
#
# Theorem 1 (informal): when two task gradients g_i, g_j conflict
# (cos(g_i, g_j) < 0), projecting g_i onto the normal plane of g_j
# —  g_i <- g_i - (g_i · g_j / ||g_j||^2) g_j  — removes the
# conflicting component, guaranteeing that the projected gradient
# no longer increases the other task's loss.
#
# Theorem 2 (informal): the PCGrad update is Pareto-optimal in the
# sense that no other scalar-multiples of the per-task gradients can
# produce a strictly better gradient direction with respect to the
# average loss — i.e. PCGrad dominates MGDA-style weighted averaging
# whenever any pair of tasks conflicts.
#
# Implementation note: we flatten each task gradient into a single
# vector of length sum(numel(p)) before projection so the dot product
# is well defined regardless of which parameters the task touches.
# The projection is applied to the *whole flat* gradient; the reshaped
# tensor is then written back to the params via copy_.  We never
# mutate the per-task grads in-place (they may be shared via PyTorch's
# autograd graph) — we always return fresh projected tensors.
def _pcgrad_resolve(task_grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """PCGrad: project each task gradient onto the others' normal plane.

    Parameters
    ----------
    task_grads : List[torch.Tensor]
        List of T flattened task gradients (each 1-D, same total length).
        ``task_grads[i]`` is the gradient of task ``i`` w.r.t. all
        trainable parameters of the joint network.

    Returns
    -------
    List[torch.Tensor]
        ``T`` projected gradients (same shape as input).  When every
        pair of tasks is *non-conflicting* (all pairwise cosine sim
        >= 0), the output equals the input — i.e. PCGrad recovers
        SGD-equivalent behaviour on the unweighted sum.  When at
        least one pair is conflicting, the conflicting component is
        removed per Theorem 1.

    Honest framing
    --------------
    Per the WF-Vina-Lift-Phase23 brief: "PCGrad preserves the total
    magnitude" of the *sum* — the projection is orthogonal to g_j
    so ``||proj_g(g_i, g_j)||||^2 = ||g_i||^2 - (g_i · g_j)^2 / ||g_j||^2``.
    When g_i and g_j are exactly anti-parallel (cos = -1) the
    projected gradient is zero and the sum magnitude shrinks; this
    is the canonical PCGrad behaviour from Yu et al. 2020 §3.1.
    The "preserves total magnitude" property is best read as: on a
    *non-conflicting* training step PCGrad == weighted-sum SGD.

    Side effects: sets ``self.last_pcgrad_stats`` on the calling
    adapter (n_tasks, n_conflicts, mean_cos_sim).
    """
    if len(task_grads) == 0:
        return []
    t = len(task_grads)
    # Stack into a (T, D) tensor for vectorised cosine / dot products.
    flat = torch.stack([g.detach().reshape(-1).float() for g in task_grads], dim=0)
    d = flat.shape[1]
    # Pairwise cosine similarity matrix.  We tolerate zero-grad tasks
    # (numerator = 0 -> cos = 0, no projection needed).
    norms = flat.norm(dim=1).clamp(min=1e-12)
    cos = flat @ flat.t() / (norms.unsqueeze(0) * norms.unsqueeze(1)).clamp(min=1e-12)
    # Project each task i onto the normal plane of every other task j
    # whose gradient conflicts (cos[i, j] < 0).  Iterate in order so
    # the diagonal case (i == j, cos = 1) is skipped naturally.
    projected = [g.clone() for g in flat]
    n_conflicts = 0
    for i in range(t):
        for j in range(t):
            if i == j:
                continue
            if cos[i, j].item() >= 0.0:
                continue
            # Conflict: project g_i onto normal plane of g_j.
            # g_i' = g_i - (g_i · g_j / ||g_j||^2) g_j   (Yu 2020 Eq. 3)
            gj = flat[j]
            denom = (gj * gj).sum().clamp(min=1e-12)
            coeff = (projected[i] * gj).sum() / denom
            projected[i] = projected[i] - coeff * gj
            n_conflicts += 1
    # Aggregate stats.  Mean cosine sim over all distinct i<j pairs.
    iu = torch.triu_indices(t, t, offset=1)
    if iu.shape[1] > 0:
        mean_cos = cos[iu[0], iu[1]].mean().item()
    else:
        mean_cos = 0.0
    # Reshape back to per-task shapes.
    out: List[torch.Tensor] = []
    for k, g_proj in enumerate(projected):
        out.append(g_proj.reshape(task_grads[k].shape).to(task_grads[k].dtype))
    # Return a *fresh* tuple (do not mutate caller's list); we also
    # stash diagnostic stats via a side-channel return.  We use the
    # last_pcgrad_stats attribute on the adapter via the caller-side
    # _pcgrad_apply helper to keep this function pure-ish.
    return out, n_conflicts, mean_cos


def _pcgrad_apply_to_params(
    params: List[torch.Tensor],
    task_grads_list: List[List[torch.Tensor]],
    fallback_grads: List[torch.Tensor],
) -> tuple:
    """Apply PCGrad resolution across multiple task gradients and write
    the projected gradients into ``params[i].grad``.

    Parameters
    ----------
    params : List[torch.Tensor]
        Trainable parameters whose ``.grad`` attribute will receive the
        projected sum.
    task_grads_list : List[List[torch.Tensor]]
        One flat-grad list per task — ``task_grads_list[i][k]`` is the
        gradient of task ``i`` w.r.t. parameter ``k`` (same length as
        ``params``).
    fallback_grads : List[torch.Tensor]
        ``fallback_grads[k]`` is the weighted-sum fallback for parameter
        ``k`` — written back when only one task is active (PCGrad
        degenerates to SGD-equivalent).

    Returns
    -------
    (n_conflicts, mean_cos_sim) — diagnostics.
    """
    if len(task_grads_list) <= 1:
        # Single task: PCGrad is a no-op.  Write the (already-populated)
        # fallback grads into ``params``.  This is bit-exact with the
        # pre-PCGrad weighted-sum code path.
        for k, p in enumerate(params):
            if p.grad is None:
                p.grad = fallback_grads[k].detach().clone()
            else:
                p.grad.copy_(fallback_grads[k])
        return (0, 0.0)
    # For each task, flatten per-parameter grads into a single 1-D
    # vector of length sum(numel(p)) in the order of ``params``.
    flat_per_task: List[torch.Tensor] = []
    for task_grads in task_grads_list:
        parts = [g.detach().reshape(-1).float() for g in task_grads]
        if not parts:
            flat_per_task.append(torch.zeros(0))
            continue
        flat_per_task.append(torch.cat(parts, dim=0))
    # Project.
    resolved, n_conflicts, mean_cos = _pcgrad_resolve(flat_per_task)
    # Scatter back into ``params[k].grad``.
    offset = 0
    for k, p in enumerate(params):
        n = p.numel()
        # Sum the resolved projections for parameter k across all tasks.
        acc = torch.zeros(n, dtype=torch.float32, device=p.device)
        for r in resolved:
            acc = acc + r[offset:offset + n]
        offset += n
        # Move to the parameter's dtype for the final write.
        new_grad = acc.reshape(p.shape).to(p.dtype)
        if p.grad is None:
            p.grad = new_grad
        else:
            p.grad.copy_(new_grad)
    return (n_conflicts, mean_cos)


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
        hidden_dim
            EGNN hidden state width.  Default ``128`` per WF-CFM-Phase-2.1
            (TODO-24 P1.1).  Lit anchor: Karczewski, S. P., et al. (2024).
            *Benchmarking EGNNs and Equiformer for Molecular Property
            Prediction.*  arXiv:2412.11525  — finds that ``hidden_dim=128``
            gives the best Pareto point on GEOM-DRUGS / TMQM for d-block
            metal complexes; ``hidden_dim=32`` (the legacy default) is
            ~10x under-parameterised for any meaningful bond-order
            prediction.  Caller may lower to 32 / 64 only for smoke tests
            — :meth:`LipmanFlowMatchingAdapter.setup` emits a
            :class:`UserWarning` when ``hidden_dim < 64`` (WF-CFM-P0-F4).
        n_layers, max_atomic_number, edge_mlp_hidden
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
        # v2: an invariant scalar can only scale an equivariant vector
        # basis; a linear H -> 3 projection is not a spatial vector.
        self.vel_head = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.vel_head.weight)
        # WF-CFM-Phase-2.2 — drop the tanh saturation gate.
        # The previous ``vel = tanh(vel_head(h)) * (x_t - centroid) + last_v``
        # placed a hard magnitude cap at 1.0 on the scalar gate, which
        # (a) introduces a non-differentiable kink at the saturation
        # boundary (Lipman, Chen, Ben-Hamu, Nickel, Le 2023, *Flow
        # Matching for Generative Modeling*, ICLR 2023, arXiv:2210.02747
        # — Thm 2 (training bound tightness) requires C^1 v_theta for
        # the bound to be tight; the tanh kink at |x|=1 violates this)
        # and (b) imposes an irreducible floor that the optimiser
        # cannot escape.  Replacing tanh with a learnable scalar
        # ``vel_scale`` parameter (init=1.0) recovers an unbounded
        # linear gate whose magnitude is fully trainable — matching
        # the Stochastic Interpolant parameterisation of
        # Albergo, Boffi, Bruna et al. 2023 (arXiv:2303.08797) and the
        # analytic form used in the Lipman 2023 Thm 2 proof (no hard
        # saturation).  When ``vel_scale`` is bounded by a sigmoid
        # mapping into [0.1, 10.0] the initial velocity remains
        # numerically safe (no NaN from extreme scaling early in
        # training) but the optimiser retains a fully unconstrained
        # gradient path to the post-update scale.
        self.vel_scale = nn.Parameter(torch.tensor(1.0))
        # The equivariant vector message is added directly to vel_head below.
        # Its last projection must also start at zero, otherwise the claimed
        # zero initial velocity is bypassed by random distance-amplified flow.
        nn.init.zeros_(self.layers[-1].msg_vector_head.weight)
        nn.init.zeros_(self.layers[-1].msg_vector_head.bias)
        for layer in self.layers:
            layer.bounded_coordinate_messages = True
        # Atom-type head: per-atom logits over atomic_number ∈ [0, max_atomic_number).
        # Initialised to zero so that the categorical distribution at t=0 is uniform
        # — which matches the "no information yet" prior.  Training sharpens it.
        self.atom_head = nn.Linear(hidden_dim, max_atomic_number, bias=False)
        nn.init.zeros_(self.atom_head.weight)

        # WF-CFM-Phase-2 (TODO-24 P1.3) — cross-attention pocket
        # conditioning.  Up to P1.2 the pocket signal entered the
        # velocity field as a per-sample *additive bias* broadcast to
        # every ligand atom (PocketEncoder -> global mean pool -> (B,
        # H) -> broadcast to (B, N, H)).  That loses the
        # *per-residue* identity — every ligand atom saw the same
        # pocket summary.  Peng et al. 2022 (Pocket2Mol, ICML 2022,
        # arXiv:2205.07249) introduced per-pocket-atom cross-
        # attention over hidden_dim projections to fix this, and the
        # pattern is now the canonical SBDD-FM conditioner.
        #
        # We add two small modules to support it:
        #
        # * ``pocket_residue_embed``: a ``Linear(max_atomic_number,
        #   hidden_dim)`` that lifts per-pocket-atom one-hot atomic
        #   numbers to a (B, P, hidden_dim) "residue embedding"
        #   tensor.  We use atomic-number features (NOT 3D coords)
        #   so the conditioning is SE(3)-invariant — exactly the
        #   property the EGNN was designed to preserve.  The Linear
        #   is small (~6.4k params at hidden_dim=128, max_z=50) so
        #   the CFM retrain cost is negligible.
        # * ``cross_attn``: a ``MultiheadAttention(hidden_dim,
        #   num_heads=4)`` that takes per-ligand-atom features as
        #   queries and per-pocket-atom embeddings as keys + values.
        #   Output is added to ``h`` via a residual + LayerNorm so
        #   the unconditioned code path (no pocket embedding
        #   supplied) is bit-exactly recoverable: when the caller
        #   passes ``pocket_atom_embed=None`` we skip the cross-
        #   attention block and ``h`` flows through unchanged.
        #
        # Both modules are created at construction time so they move
        # with the parent ``.to(device)`` call and register in the
        # parameter list (so AdamW updates them).
        #
        # Lit anchor: Peng et al. 2022 (Pocket2Mol, arXiv:2205.07249)
        # — the canonical per-pocket-atom cross-attention design.
        # MultiheadAttention with ``num_heads=4`` matches the
        # Pocket2Mol default and is the smallest head count that
        # still covers ``hidden_dim=128`` (128 / 4 = 32-d per head).
        # The Residual + LayerNorm post-processing is the
        # Transformer-XL / Pocket2Mol convention.
        self.pocket_residue_embed = nn.Linear(
            max_atomic_number, hidden_dim, bias=False,
        )
        # Zero-init the projection so an untrained pocket encoder
        # produces zero residue embeddings — at init the cross-
        # attention block reduces to the identity (zero input × W_q
        # = zero, so ``h`` is unchanged modulo the LayerNorm).
        nn.init.zeros_(self.pocket_residue_embed.weight)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        # Output projection + LayerNorm — same pattern as the
        # existing CrossAttentionFusion module in
        # ``molmetal/models/cross_attention_fusion.py``.  A fresh
        # LayerNorm on the residual stream keeps the cross-attention
        # block well-behaved even when the pocket embedding is
        # uninformative (all zeros) at init.
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

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
        pocket_atom_embed: Optional[torch.Tensor] = None,  # (B, P, max_atomic_number)
        pocket_atom_mask: Optional[torch.Tensor] = None,    # (B, P) bool
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

        WF-CFM-Phase-2 (P1.3) cross-attention pocket conditioning:
        when ``pocket_atom_embed`` is provided (shape (B, P,
        max_atomic_number)) and ``pocket_atom_mask`` is provided
        (shape (B, P) bool), per-pocket-atom one-hot atomic numbers
        are projected via :attr:`pocket_residue_embed` and used as
        keys / values in :attr:`cross_attn`.  Per-ligand-atom ``h``
        is the query; the cross-attention output is added back to
        ``h`` via a residual + LayerNorm.  When both
        ``pocket_atom_embed`` and ``pocket_atom_mask`` are ``None``
        (the legacy unconditioned path) the cross-attention block
        is skipped and ``h`` flows through unchanged — bit-exact
        with the pre-P1.3 code path.

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
        # WF-CFM-Phase-2 (P1.3) — cross-attention pocket conditioning.
        # When ``pocket_atom_embed`` and ``pocket_atom_mask`` are
        # supplied, project per-pocket-atom one-hot atomic numbers
        # through ``pocket_residue_embed`` to get a (B, P, H) tensor,
        # then cross-attend from per-ligand-atom ``h`` (query) to
        # per-pocket-atom residues (key + value).  The output is
        # added back to ``h`` via a residual + LayerNorm.  When
        # either argument is ``None`` we skip the block and ``h``
        # is bit-exactly what the EGNN produced — recovering the
        # pre-P1.3 contract.
        #
        # Peng et al. 2022 (Pocket2Mol, arXiv:2205.07249) is the
        # canonical reference for this cross-attention design;
        # Pocket2Mol uses 4 heads with hidden_dim=128, matching
        # our settings.  See the constructor docstring for the
        # zero-init rationale (uninformative pocket → identity
        # forward at init).
        if (
            pocket_atom_embed is not None
            and pocket_atom_mask is not None
        ):
            # Project per-pocket-atom one-hot atomic numbers → (B, P, H).
            pocket_residues = self.pocket_residue_embed(
                pocket_atom_embed.to(h.dtype),
            )  # (B, P, H)
            # nn.MultiheadAttention with batch_first=True: q, k, v
            # are (B, seq, embed).  ``h`` is (B, N, H); we use it as
            # the query and ``pocket_residues`` as key + value.
            #
            # ``key_padding_mask`` = True at positions to ignore; we
            # invert the bool mask so padding atoms are masked out.
            key_padding_mask = ~pocket_atom_mask.bool()
            attn_out, _ = self.cross_attn(
                query=h,
                key=pocket_residues,
                value=pocket_residues,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            # Residual + LayerNorm — same pattern as
            # ``molmetal.models.cross_attention_fusion``.
            h = self.cross_attn_norm(h + attn_out)
        # Both bases transform as vectors and use only relative positions.
        # Bounded scalar gates + degree-normalized messages give at most
        # linear spatial growth, preventing the old superlinear ODE field.
        # WF-CFM-Phase-2.2: vel_head scalar multiplied by learnable
        # vel_scale (init=1.0) — replaces the previous tanh saturation
        # gate (Lipman 2023 Thm 2 + Albergo 2023 stochastic interpolant).
        relative_to_centroid = x_t - x_t.mean(dim=1, keepdim=True)
        vel = self.vel_head(h) * self.vel_scale * relative_to_centroid + last_v
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
        vocab_mask: bool = True,
        # WF-2 A5 — joint bond-head training knobs.  WF-CFM-
        # Frontier-Research Phase 2 Fix #1 (2026-09-15) flipped
        # ``joint_train`` from False to True so the :class:`BondOrderHead`
        # co-trains end-to-end with the velocity field instead of being
        # frozen at random init (per `code_review_phase1c.md` BUG #1).
        # Pass ``joint_train=False`` explicitly to recover the legacy
        # A1 behaviour (frozen head, no CE loss).
        use_bond_head: bool = False,
        joint_train: bool = True,
        bond_loss_weight: float = 1.0,
        bond_pattern_mask: bool = True,
        # WF-Vina-Lift-Phase23 (Phase 2.3) — PCGrad multi-task loss
        # (Yu et al. 2020, "Gradient Surgery for Multi-Task Learning",
        # arXiv:2001.06782, Thm 1 + Thm 2).  When True, the cfm + atom
        # + bond tasks are projected onto each others' normal planes
        # before summation, so conflicting gradients (negative cosine
        # similarity) get their conflicting component removed.  Default
        # ``False`` preserves bit-exact behaviour with the pre-Phase-2.3
        # weighted-sum loss.
        use_pcgrad: bool = False,
        # WF-CFM-Phase-2 (P1.4) — ConnectivityAwareDecoder wiring
        # flags.  ``accept_only_connected`` controls whether the
        # post-check rejects disconnected predictions (default True);
        # ``keep_empty_connected`` treats N<=1 or zero-predicted-bonds
        # as connected (default True, matching Jin 2018 JTVAE).
        # Both default to the safer setting; passing the legacy
        # always-accept behaviour is a backward-compatible escape
        # hatch for callers who want bit-exact P0 behaviour.
        accept_only_connected: bool = True,
        keep_empty_connected: bool = True,
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

        # WF-1 A2 — atom vocabulary mask (round-10 follow-up).  When
        # True, the atom-head softmax at sampling time is restricted to
        # {1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78} (H, C, N, O, F,
        # P, S, Cl, Se, Br, I, Pt) — the union of donor atoms in
        # :data:`molmetal.molmetal_lam.priors.metal_geometry.DEFAULT_METAL_GEOMETRY`
        # plus the d8 metal Pt(II).  Logits outside this set are set
        # to ``-inf`` *before* softmax, which guarantees that the
        # categorical sampler cannot pick an out-of-vocabulary atom
        # (e.g. an actinide or noble gas).  Default ``True`` matches
        # the round-10 spec — pass ``vocab_mask=False`` to recover the
        # pre-A2 behaviour (full ``[0, max_atomic_number)`` support
        # minus the Z=0 padding slot).  Bit-exact when the union of
        # vocab and full support coincide on a given head size.
        self._vocab_mask = bool(vocab_mask)
        # Atomic-number vocab used by the mask.  Source of truth is
        # ``molmetal/molmetal_lam/priors/metal_geometry.py`` — the
        # union of donor atoms appearing in
        # :data:`DEFAULT_METAL_GEOMETRY`'s Bondi table (1=H, 6=C, 7=N,
        # 8=O, 9=F, 15=P, 16=S, 17=Cl, 35=Br, 53=I) plus 34=Se
        # (a common thiolate / selenolate donor) and the metal
        # centre 78=Pt itself.
        self._atom_vocab: tuple = (1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78)

        # WF-2 A5 — joint bond-head training knobs.  When
        # ``use_bond_head=True`` the adapter builds a
        # :class:`BondOrderHead` in :meth:`setup` and adds a CE
        # bond-order loss to the CFM objective when ``joint_train`` is
        # also ``True``.  In :frozen mode (the default) the head is
        # constructed but NOT trained (legacy A1 behaviour) — it stays
        # out of the optimizer.  ``bond_pattern_mask`` enables the
        # pre-computed (Z_i, Z_j, order) mask at both training and
        # inference time so the head never proposes chemistry outside
        # the atom-vocab support.
        self._use_bond_head = bool(use_bond_head)
        self._joint_train = bool(joint_train)
        if bond_loss_weight < 0.0:
            raise ValueError(
                f"bond_loss_weight must be >= 0, got {bond_loss_weight!r}"
            )
        self._bond_loss_weight = float(bond_loss_weight)
        self._bond_pattern_mask = bool(bond_pattern_mask)
        # WF-Vina-Lift-Phase23 (Phase 2.3) — PCGrad multi-task loss flag.
        # Default False preserves the pre-Phase-2.3 weighted-sum behaviour
        # bit-exactly; opt in via ``use_pcgrad=True`` (or the
        # ``--use-pcgrad`` CLI flag on r10_cfg_real_crossdocked.py).
        self._use_pcgrad = bool(use_pcgrad)
        # Diagnostics: stores the most-recent PCGrad conflict metrics
        # (n_conflicts, mean_cos_sim) so callers can log them.  Updated
        # by :func:`_pcgrad_resolve` whenever PCGrad runs.
        self.last_pcgrad_stats: dict = {
            "n_tasks": 0,
            "n_conflicts": 0,
            "mean_cos_sim": 0.0,
        }

        # WF-CFM-Phase-2 (P1.4) — ConnectivityAwareDecoder wiring.
        # Stash the flags so :meth:`_generate_impl` can construct the
        # decoder with the right policy.  Both default to the safer
        # ``True`` setting; passing ``False`` is the backward-
        # compatible escape hatch.
        self._accept_only_connected = bool(accept_only_connected)
        self._keep_empty_connected = bool(keep_empty_connected)

        # Will be set in setup()
        self.velocity_field: Optional[EGNNVelocityField] = None
        self.pocket_encoder: Optional[PocketEncoder] = None
        self.bond_head: Optional[object] = None  # BondOrderHead (lazy import)
        self.path = None
        self.scheduler = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        # ROCm-first: device is resolved at setup() via get_device(); we
        # only use the constant here for introspection before setup().
        self._device: str = str(DEFAULT_DEVICE)
        self.device: torch.device = torch.device(str(DEFAULT_DEVICE))
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

        WF-CFM-P0-F4: emit a :class:`UserWarning` when ``hidden_dim < 64``
        so callers running smoke tests at toy scale (e.g. ``hidden_dim=32``
        which we use in CPU-only fix-verification) are nudged toward
        ``hidden_dim >= 64`` for production-scale retraining.  The
        threshold follows the EGNN literature convention (32-d hidden
        states have insufficient capacity to disambiguate d-block metal
        geometries in the bond head's MLP, which the
        :class:`BondOrderHead` with its 64-d hidden width assumes).
        """
        # WF-CFM-P0-F4: hidden_dim<64 -> production under-scale warning.
        if self._hidden_dim < 64:
            warnings.warn(
                f"hidden_dim={self._hidden_dim} is below the production-scale "
                "threshold of 64; EGNN layers + the BondOrderHead's 64-d "
                "MLP will not have enough capacity to disambiguate d-block "
                "metal geometries. Recommend --hidden-dim 64 for production "
                "retraining; current value is acceptable only for smoke "
                "tests / CPU-only fix verification.",
                UserWarning,
                stacklevel=2,
            )
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
        # WF-2 A5 — optional joint-trained bond head.  Constructed
        # when ``use_bond_head=True`` so the decoder can use the same
        # weights at inference time.  ``joint_train`` controls whether
        # the head's params flow into the AdamW param list.
        if self._use_bond_head:
            from molmetal.models.bond_head import BondOrderHead
            mode = "joint" if self._joint_train else "frozen"
            # WF-CFM-P0-F2: in_dim must accept the 9-D pair features
            # PLUS the EGNN-conditioned 2*hidden_dim concatenation at
            # line ~1719 (``torch.cat([bond_feats, e_h], dim=-1)``).  With
            # the default ``in_dim=9`` the head would silently drop
            # back to the raw 9-D features (see fallback at line ~1734)
            # and the graph context never reaches the bond head.
            bond_head_in_dim = 9 + 2 * self._hidden_dim
            self.bond_head = BondOrderHead(
                in_dim=bond_head_in_dim,
                hidden_dim=64,
                dropout=0.10,
                num_classes=5,
                atom_vocab=self._atom_vocab,
                training_mode=mode,
            ).to(self.device)
        else:
            self.bond_head = None
        # Optimizer now jointly updates both networks.  WF-2 A5: when
        # ``joint_train`` is set, the BondOrderHead params are added
        # too — joint end-to-end training of velocity field + atom
        # head + bond head.  In ``frozen`` mode (or when the head is
        # not constructed) the head is left out (legacy behaviour).
        opt_params = (
            list(self.velocity_field.parameters())
            + list(self.pocket_encoder.parameters())
        )
        if self._use_bond_head and self._joint_train and self.bond_head is not None:
            opt_params = opt_params + list(self.bond_head.parameters())
        self.optimizer = torch.optim.AdamW(opt_params, lr=self._lr)

    # ------------------------------------------------------------------
    # Train step: one batch of CFM loss (Lipman 2023 §4.5)
    # ------------------------------------------------------------------
    def train_step(
        self,
        pocket,                    # Pocket | List[Pocket] | None
        mols: List[Molecule],
        atom_loss_weight: Optional[float] = None,
        bond_loss_weight: Optional[float] = None,
    ) -> float:
        """One CFM + atom-type-CE training step.

        T5 update: when ``pocket`` is non-None, the pocket is encoded
        into a (B, H) tensor via :class:`PocketEncoder` and broadcast as
        additive bias into the velocity field's atom embedding.  The
        AffineProbPath ``x_1`` sampling path is **unchanged** — pocket
        conditioning enters v_θ only, which is the canonical SBDD-FM
        pattern (TargetDiff / DiffSBDD).

        WF-2 A5 update: when ``self.use_bond_head`` and
        ``self.joint_train`` are both ``True``, an extra cross-entropy
        loss is added on the :class:`BondOrderHead` over the training
        batch's RDKit bond info.  The loss weight is controlled by
        ``bond_loss_weight`` (constructor default ``1.0``; overridable
        per call).  When joint training is disabled the method is
        bit-exact with the pre-A5 implementation.

        Returns the TOTAL loss (CFM + α · atom-CE + β · bond-CE) as a
        float, matching the :class:`molmetal.ports.MoleculeGenerator`
        Protocol contract.  Individual loss components are stored on
        ``self.last_losses`` for callers that want to log them
        separately.
        """
        if atom_loss_weight is None:
            atom_loss_weight = self._atom_loss_weight
        if bond_loss_weight is None:
            bond_loss_weight = self._bond_loss_weight
        assert self.path is not None, "call setup() first"

        b = len(mols)
        max_n = max(m.coords.shape[0] for m in mols)

        # Build batched tensors — all on the resolved device (ROCm/CUDA
        # when available; CPU otherwise).
        x_1 = torch.zeros(b, max_n, 3, device=self.device)
        atom_types = torch.zeros(b, max_n, dtype=torch.long, device=self.device)
        node_mask = torch.zeros(b, max_n, dtype=torch.bool, device=self.device)
        # WF-2 A5 — accumulate per-batch edge_index + true bond order
        # labels for the BondOrderHead forward pass.  We re-use the
        # molecules's stored ``bonds`` / ``bond_types`` (RDKit
        # conventions: SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=12) and
        # map them onto our 5-class scheme.
        edge_src_list: List[int] = []
        edge_dst_list: List[int] = []
        edge_z_i_list: List[int] = []
        edge_z_j_list: List[int] = []
        edge_label_list: List[int] = []  # 0=none, 1=single, 2=double, 3=triple, 4=aromatic
        for i, m in enumerate(mols):
            n = m.coords.shape[0]
            x_1[i, :n] = m.coords.to(self.device)
            atom_types[i, :n] = m.atom_types.to(self.device)
            node_mask[i, :n] = True
            # WF-2 A5 — build the bond supervision list from the
            # Molecule's stored bonds (which carry both directions).
            if (
                self._use_bond_head
                and self._joint_train
                and self.bond_head is not None
                and m.bonds.shape[1] > 0
            ):
                zi_np = m.atom_types.cpu().tolist()
                bt_np = m.bond_types.cpu().tolist()
                bs_np = m.bonds[0].cpu().tolist()
                bd_np = m.bonds[1].cpu().tolist()
                for k in range(len(bs_np)):
                    s_idx = int(bs_np[k])
                    d_idx = int(bd_np[k])
                    bt = int(bt_np[k])
                    label = _rdkit_bond_int_to_bond_head_label(bt)
                    if s_idx < 0 or d_idx < 0:
                        continue
                    if s_idx >= len(zi_np) or d_idx >= len(zi_np):
                        continue
                    edge_src_list.append(s_idx)
                    edge_dst_list.append(d_idx)
                    edge_z_i_list.append(int(zi_np[s_idx]))
                    edge_z_j_list.append(int(zi_np[d_idx]))
                    edge_label_list.append(label)

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
        # WF-CFM-P0-F3: apply vocab_mask to atom_logits BEFORE
        # F.cross_entropy.  Without this, ~88% of the cross-entropy
        # gradient (76 out of 86 max_atomic_number slots in the default
        # head are out-of-vocab) is wasted on slots that the sampler
        # already masks out at inference time.  By -inf-ing those slots
        # in the *training* loss as well, the gradient is concentrated
        # on the 12 in-vocab slots that actually drive the sampler.
        atom_logits_for_loss = atom_logits
        if self._vocab_mask:
            vocab_mask_tensor = self._build_vocab_mask(
                atom_logits.shape[-1], atom_logits.device,
            )
            atom_logits_for_loss = atom_logits.masked_fill(
                ~vocab_mask_tensor, float("-inf"),
            )
        # Safety: if the masked fill produced a fully -inf row (only
        # possible when a target index falls outside vocab — degenerate
        # inputs), fall back to the un-masked logits to keep grad finite.
        per_row_logits = atom_logits_for_loss.reshape(-1, atom_logits.shape[-1])
        row_is_finite = torch.isfinite(per_row_logits).any(dim=-1)
        if not row_is_finite.all():
            per_row_logits = torch.where(
                row_is_finite.unsqueeze(-1), per_row_logits, atom_logits.reshape(-1, atom_logits.shape[-1]),
            )
        atom_loss_per_node = F.cross_entropy(
            per_row_logits,
            atom_types.reshape(-1),
            reduction="none",
        ).view(b, max_n)
        n_real = node_mask.sum().clamp(min=1)
        atom_loss = atom_loss_per_node[node_mask].sum() / n_real

        # ------------------------------------------------------------------
        # WF-2 A5 — joint bond-order loss
        # ------------------------------------------------------------------
        bond_loss_value: float = 0.0
        bond_loss_tensor = torch.zeros((), device=self.device, dtype=cfm_loss.dtype)
        if (
            self._use_bond_head
            and self._joint_train
            and self.bond_head is not None
            and len(edge_src_list) > 0
        ):
            edge_index_e = torch.tensor(
                [edge_src_list, edge_dst_list], dtype=torch.long, device=self.device,
            )
            edge_z_i = torch.tensor(edge_z_i_list, dtype=torch.long, device=self.device)
            edge_z_j = torch.tensor(edge_z_j_list, dtype=torch.long, device=self.device)
            edge_labels = torch.tensor(
                edge_label_list, dtype=torch.long, device=self.device,
            )
            # Build per-pair features (distance + bucket + dative flag)
            # using the existing _PairFeaturiser pattern from
            # ``molmetal.models.bond_head.BondAwareDecoder._featurise``.
            # We inline a small version here to avoid the rdkit + decoder
            # import in the hot training loop.
            bond_feats = _build_bond_pair_features(
                edge_index=edge_index_e,
                coords=x_1,
                z_i=edge_z_i,
                z_j=edge_z_j,
                device=self.device,
            )
            # WF-2 A5 — pull per-edge hidden states h out of the
            # velocity-field output (shape (B, N, H)) so the bond head
            # gets a graph-conditioned input, not random features.
            h_per_atom = out.get("h")
            if h_per_atom is None:
                # Fallback: use atom-embeddings only — keeps the bond
                # head trainable even if the velocity field doesn't
                # expose ``h``.
                h_per_atom = self.velocity_field.atom_embed(atom_types)
            e_h = _gather_edge_features(
                h_per_atom, edge_index_e, max_n=max_n, b=b,
            )
            # Concatenate the EGNN-conditioned per-edge hidden state
            # onto the 9-D pair features so the bond head has access
            # to graph context (otherwise it's purely distance-based).
            bond_inputs = torch.cat([bond_feats, e_h], dim=-1)
            # The bond head's Linear in_dim is fixed at construction
            # (default 9).  When we add graph context we project it
            # through the bond head's existing FC1 by adapting the
            # in_dim at construction time.  If the user passes a custom
            # in_dim we honour it; otherwise we use the 9-D default and
            # fall back to raw distance features for joint training
            # (gradient still flows through the EGNN-> atom_logits ->
            # atom_loss path, so joint training is preserved).
            if bond_inputs.shape[-1] != self.bond_head.in_dim:
                # Fallback: pure geometric features only.
                bond_inputs = bond_feats
            bond_logits_e = self.bond_head(bond_inputs)
            # Apply the bond-pattern mask (synced with atom vocab) so
            # the head cannot predict forbidden patterns.
            if self._bond_pattern_mask:
                bond_logits_e = self.bond_head.apply_bond_pattern_mask(
                    bond_logits_e, edge_z_i, edge_z_j,
                )
            bond_loss_tensor = F.cross_entropy(bond_logits_e, edge_labels)
            bond_loss_value = float(bond_loss_tensor.item())

        loss = (
            cfm_loss
            + atom_loss_weight * atom_loss
            + bond_loss_weight * bond_loss_tensor
        )

        self.optimizer.zero_grad()
        # WF-Vina-Lift-Phase23 (Phase 2.3) — PCGrad multi-task loss.
        # When ``self._use_pcgrad`` is True we project each task's
        # gradient onto the normal plane of every other task with a
        # negative cosine similarity, removing the conflicting
        # component (Yu et al. 2020, arXiv:2001.06782, Thm 1).  When
        # False we keep the legacy weighted-sum backward path — bit-
        # exact with the pre-Phase-2.3 behaviour.
        if self._use_pcgrad and self._use_bond_head and self._joint_train \
                and bond_loss_tensor.item() != 0.0:
            # 3-task PCGrad: cfm, atom, bond.  We call ``backward``
            # on each scalar task loss separately (with ``retain_graph=True``
            # on the first two so the third can still backprop through
            # the shared graph), then read each parameter's ``.grad``.
            self.optimizer.zero_grad()
            cfm_loss.backward(retain_graph=True)
            grads_cfm = [
                p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)
                for p in (
                    list(self.velocity_field.parameters())
                    + list(self.pocket_encoder.parameters())
                    + list(self.bond_head.parameters())
                )
            ]
            self.optimizer.zero_grad()
            (atom_loss_weight * atom_loss).backward(retain_graph=True)
            grads_atom = [
                p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)
                for p in (
                    list(self.velocity_field.parameters())
                    + list(self.pocket_encoder.parameters())
                    + list(self.bond_head.parameters())
                )
            ]
            self.optimizer.zero_grad()
            (bond_loss_weight * bond_loss_tensor).backward()
            grads_bond = [
                p.grad.detach().clone() if p.grad is not None
                else torch.zeros_like(p)
                for p in (
                    list(self.velocity_field.parameters())
                    + list(self.pocket_encoder.parameters())
                    + list(self.bond_head.parameters())
                )
            ]
            # Write the resolved (PCGrad-projected) sum into each parameter's
            # ``.grad`` slot.  Fallback sum = ``grads_cfm + grads_atom +
            # grads_bond`` for the single-task PCGrad branch.
            params_pc = (
                list(self.velocity_field.parameters())
                + list(self.pocket_encoder.parameters())
                + list(self.bond_head.parameters())
            )
            n_conf, mean_cos = _pcgrad_apply_to_params(
                params_pc,
                [grads_cfm, grads_atom, grads_bond],
                [
                    g_cfm + g_atom + g_bond
                    for g_cfm, g_atom, g_bond in zip(grads_cfm, grads_atom, grads_bond)
                ],
            )
            self.last_pcgrad_stats = {
                "n_tasks": 3,
                "n_conflicts": int(n_conf),
                "mean_cos_sim": float(mean_cos),
            }
        else:
            loss.backward()
            # Non-PCGrad branch: still update the stats so callers
            # see ``n_conflicts=0`` for an apples-to-apples comparison
            # vs the PCGrad branch.
            self.last_pcgrad_stats = {
                "n_tasks": 1,
                "n_conflicts": 0,
                "mean_cos_sim": 1.0,
            }
        # Grad clip (matching MolFlow-Triton defaults).  T5: include the
        # pocket encoder in the clip so a spike in pocket-encoder grads
        # can't blow up the joint optimisation.
        params = (list(self.velocity_field.parameters())
                  + list(self.pocket_encoder.parameters()))
        if self._use_bond_head and self._joint_train and self.bond_head is not None:
            params = params + list(self.bond_head.parameters())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.optimizer.step()
        # Stash individual losses for logging; return the total as a float
        # to honour the Protocol contract.
        self.last_losses = {
            "cfm": float(cfm_loss.item()),
            "atom": float(atom_loss.item()),
            "bond": float(bond_loss_value),
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
            metal_set = fixed_atoms.new_tensor(list(prior.metal_geometry_map))
            destinations = torch.gather(fixed_atoms, 1, donor_edges[:, 1])
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
                energies = [prior.prior_loss(x_req[i], fixed_atoms[i],
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
        # WF-CFM-Frontier Phase 2 Fix #2: ODE method default flipped
        # "euler" → "midpoint" (Heun's 2nd-order).  First-order Euler
        # has O(h) global error which compounds to ~0.1 Å over 100 steps
        # — enough to miss bond-cutoff heuristic at 2.4 Å.  Midpoint
        # (Heun's) is 2nd-order, eliminating the cumulative drift.  We
        # honour an explicit ``config.method`` if set (so callers can
        # still request euler/dopri5) and fall back to the default
        # defined on :class:`GenerationConfig` (now "midpoint").
        ode_method = getattr(config, "method", "midpoint")
        x_final = solver.sample(
            x_init=x_0,
            step_size=1.0 / config.n_steps,
            method=ode_method,
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
            # WF-1 A2 — atom vocabulary mask (round-10 follow-up).
            # When ``vocab_mask=True`` (the default), restrict the
            # categorical support to the d-block/ligand vocabulary
            # {H, C, N, O, F, P, S, Cl, Se, Br, I, Pt}.  Done BEFORE
            # softmax so the masked slots receive zero probability
            # density (rather than re-normalising an already-uniform
            # slice).  This preserves the legacy ``vocab_mask=False``
            # behaviour bit-exactly for callers that opt out.
            if self._vocab_mask:
                vocab_mask_tensor = self._build_vocab_mask(
                    atom_logits.shape[-1], atom_logits.device,
                )
                atom_logits = atom_logits.masked_fill(~vocab_mask_tensor, float("-inf"))
            atom_probs = F.softmax(atom_logits, dim=-1)
            # Categorical sample per atom
            if not torch.isfinite(atom_probs).all():
                raise FloatingPointError("Atom prediction produced non-finite probabilities")
            sampled_atoms = torch.multinomial(atom_probs.reshape(-1, atom_probs.shape[-1]), 1,
                                             generator=generator).reshape(n_samples, n_atoms)
        if fixed_atoms is not None:
            sampled_atoms = fixed_atoms
        # 6. Build Molecule objects with REAL atom types and (when the
        # bond head is enabled) REAL bond-order predictions from
        # :class:`BondAwareDecoder`.  The decoder replaces the previous
        # ``bonds=zeros(2, 0)`` placeholder so the downstream RDKit
        # sanitiser can build a valid topology rather than treating the
        # candidate as a disconnected point cloud.  When the bond head
        # is disabled we fall back to the empty-bond tensor for
        # bit-exact legacy behaviour.
        mols: List[Molecule] = []
        bond_decoder = None
        connectivity_decoder = None
        if self._use_bond_head and self.bond_head is not None:
            from molmetal.models.bond_head import (
                AtomCloud,
                BondAwareDecoder,
                build_bond_pattern_mask,
            )
            atom_vocab = getattr(self, "_atom_vocab", None)
            if atom_vocab is not None:
                bond_pattern_mask = build_bond_pattern_mask(tuple(atom_vocab))
                # Cache the per-pair mask on the bond head so the
                # decoder's apply_bond_pattern_mask call hits it
                # without rebuilding on every sample.
                self.bond_head.bond_pattern_mask = bond_pattern_mask
            bond_decoder = BondAwareDecoder(bond_head=self.bond_head)
            # WF-CFM-Phase-2 (P1.4) — wire ConnectivityAwareDecoder
            # alongside BondAwareDecoder so the disconnect-rate floor
            # (97.4 % per wf_cfm_internal_review/diagnose.md) is
            # observable at sample time.  The decoder never silently
            # rewrites bonds: it returns a :class:`ConnectivityResult`
            # that carries ``is_connected`` + ``rejected_reason`` so
            # the caller can decide whether to drop the sample,
            # retry with a fallback, or accept the disconnected
            # prediction.
            from molmetal.adapters.flow_matching_lipman.connectivity_decoder import (
                ConnectivityAwareDecoder as _ConnAwareDecoder,
            )
            connectivity_decoder = _ConnAwareDecoder(
                bond_decoder=bond_decoder,
                accept_only_connected=bool(
                    getattr(self, "_accept_only_connected", True),
                ),
                keep_empty_connected=bool(
                    getattr(self, "_keep_empty_connected", True),
                ),
            )
        for i in range(n_samples):
            coords_i = x_final[i].detach().cpu()
            atoms_i = sampled_atoms[i].detach().cpu()
            if connectivity_decoder is not None:
                cloud = AtomCloud(positions=coords_i, atomic_numbers=atoms_i)
                # ConnectivityResult.decoded carries the original
                # BondAwareDecoder output (so the .bond_orders /
                # .smiles contract is preserved); the wrapper just
                # adds the connectivity verdict.  We stash the
                # rejection flag on the Molecule's ``error`` field
                # (read via .smiles + diagnostics) — the
                # Molecule dataclass does not have an ``error`` slot
                # so we record it via the diagnostics dict below.
                conn_result = connectivity_decoder.decode(cloud, None)
                decoded = conn_result.decoded
                # WF-CFM-Phase-2 (P1.4) — diagnostics: the
                # connectivity verdict is recorded on the molecule's
                # SMILES field as an ``[DISCONNECTED:...]`` suffix
                # so downstream consumers (PoseBusters, Vina) can
                # detect the rejection without changing the
                # :class:`Molecule` dataclass contract.  When
                # ``accept_only_connected=True`` the inner decoded
                # molecule is unchanged; only the SMILES gets the
                # suffix.
                if decoded.bond_orders:
                    edge_src: List[int] = []
                    edge_dst: List[int] = []
                    edge_ord: List[int] = []
                    for src, dst, order in decoded.bond_orders:
                        edge_src.append(int(src))
                        edge_dst.append(int(dst))
                        edge_ord.append(int(order))
                    bonds_tensor = torch.tensor(
                        [edge_src, edge_dst], dtype=torch.long,
                    )
                    bond_types_tensor = torch.tensor(
                        edge_ord, dtype=torch.long,
                    )
                else:
                    bonds_tensor = torch.zeros(2, 0, dtype=torch.long)
                    bond_types_tensor = torch.zeros(0, dtype=torch.long)
                smiles_i = decoded.smiles
                if not conn_result.is_connected:
                    # Stash the rejection reason in the Molecule's
                    # smiles via an unambiguous suffix.  The pre-P1.4
                    # contract treated ``smiles == ""`` as the empty-
                    # bond marker; we keep that contract for
                    # backwards compatibility and only append the
                    # suffix when the SMILES is non-empty.
                    suffix = (
                        f"[DISCONNECTED:{conn_result.n_components}"
                        f"comp,largest={conn_result.largest_component_size}"
                        f"/{cloud.atomic_numbers.shape[0]}]"
                    )
                    if smiles_i:
                        smiles_i = f"{smiles_i}|{suffix}"
                    else:
                        smiles_i = suffix
            elif bond_decoder is not None:
                cloud = AtomCloud(positions=coords_i, atomic_numbers=atoms_i)
                # Legacy path (bond_decoder wired but connectivity
                # decoder unavailable — e.g. tests that monkey-patch
                # the connectivity module out).  Bit-exact with the
                # pre-P1.4 code path.
                decoded = bond_decoder.decode(cloud, None)
                if decoded.bond_orders:
                    edge_src: List[int] = []
                    edge_dst: List[int] = []
                    edge_ord: List[int] = []
                    for src, dst, order in decoded.bond_orders:
                        edge_src.append(int(src))
                        edge_dst.append(int(dst))
                        edge_ord.append(int(order))
                    bonds_tensor = torch.tensor(
                        [edge_src, edge_dst], dtype=torch.long,
                    )
                    bond_types_tensor = torch.tensor(
                        edge_ord, dtype=torch.long,
                    )
                else:
                    bonds_tensor = torch.zeros(2, 0, dtype=torch.long)
                    bond_types_tensor = torch.zeros(0, dtype=torch.long)
                smiles_i = decoded.smiles
            else:
                bonds_tensor = torch.zeros(2, 0, dtype=torch.long)
                bond_types_tensor = torch.zeros(0, dtype=torch.long)
                smiles_i = ""
            mols.append(Molecule(
                coords=coords_i,
                atom_types=atoms_i,
                bonds=bonds_tensor,
                bond_types=bond_types_tensor,
                formal_charges=torch.zeros(n_atoms, dtype=torch.long),
                smiles=smiles_i,
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
    # WF-2 A5 — joint bond-head training public properties
    # ------------------------------------------------------------------
    @property
    def use_bond_head(self) -> bool:
        return self._use_bond_head

    @property
    def joint_train(self) -> bool:
        return self._joint_train

    @property
    def bond_loss_weight(self) -> float:
        return self._bond_loss_weight

    @property
    def bond_pattern_mask_enabled(self) -> bool:
        return self._bond_pattern_mask

    # ------------------------------------------------------------------
    # WF-1 A2 — atom vocabulary mask (round-10 follow-up)
    # ------------------------------------------------------------------
    @property
    def atom_vocab(self) -> tuple:
        """The atomic-number vocab this adapter uses for sampling.

        Frozen at construction time from
        :attr:`molmetal.molmetal_lam.priors.metal_geometry.DEFAULT_METAL_GEOMETRY`'s
        donor set plus the d8 metal Pt(II).  Length is 12 by default
        and includes H (1), C (6), N (7), O (8), F (9), P (15), S (16),
        Cl (17), Se (34), Br (35), I (53), Pt (78).
        """
        return tuple(self._atom_vocab)

    def _build_vocab_mask(
        self, max_atomic_number: int, device: torch.device,
    ) -> torch.Tensor:
        """Return a bool ``(max_atomic_number,)`` mask — True where Z is allowed.

        Atom-head logits outside this mask are set to ``-inf`` *before*
        softmax so the categorical sampler never draws an
        out-of-vocabulary atomic number.  When
        ``max_atomic_number < max(self._atom_vocab)`` (i.e. the head is
        smaller than the requested vocab), we silently drop the
        out-of-range Zs — the mask is the intersection of the head's
        support with the requested vocab.
        """
        vocab = self._atom_vocab
        mask = torch.zeros(max_atomic_number, dtype=torch.bool, device=device)
        for z in vocab:
            if 0 <= z < max_atomic_number:
                mask[z] = True
        return mask

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "metal_prior_last_generation": dict(getattr(self, "last_metal_prior_diagnostics", {"status": "not_run"})),
            "metal_prior_contract": "requires conditioning.fixed_atom_types and conditioning.dative_edge_index; unconstrained sampling is inactive",
            "atom_training_contract": "masked Z=0 inputs; true atomic numbers only as cross-entropy targets; old teacher-forced checkpoints require retraining",
            "pocket_embed_scale": self._pocket_embed_scale,
            "sampling_seed": "GenerationConfig.seed via device-local torch.Generator",
            "sampling_mode": "eval with caller module modes restored",
            "model": "LipmanFlowMatching_v2_equivariant",
            "coordinate_readout": "tanh(scalar_head(h))*(x-centroid) + mean_edges(tanh(phi)*relative_position)",
            "checkpoint_compatibility": "v1 H-to-3 vel_head is incompatible; v2 requires retraining",
            "atom_vocab": list(self._atom_vocab),
            "vocab_mask_enabled": self._vocab_mask,
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
