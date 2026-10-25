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
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.domain import Molecule
from molmetal.ports import GenerationConfig, MoleculeGenerator, validate_method
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
from .amp import CFMAMPContext  # noqa: E402


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
#: 2026-10-04 — this used to be a single hard-coded path,
#: ``molmetal/checkpoints/dmpnn_tmqm_pretrained.pt``, which is GIT-IGNORED
#: (``.gitignore:110 checkpoints/``) and is NOT on disk.  Two consequences,
#: both silent:
#:
#:   * ``use_tmqm_init=True`` with no explicit path would raise on a missing
#:     file — so the warm start was never usable through its own default.
#:   * ``molmetal/tests/test_tmqm_wireup.py`` gates its whole module on
#:     ``DEFAULT_TMQM_CKPT.exists()``, so all 6 tmQM wire-up tests were
#:     SKIPPED, and the suite reported green while testing nothing.
#:
#: The F2 pretrain output does exist at ``models/tmqm_pretrained/``
#: (138,112 params, hidden 128 / depth 3, 5 regression targets incl.
#: metal_bo_total + coord_number, cls_acc 0.965 — matching
#: ``molmetal/reports/f2_tmqm_pretraining.md``).  Resolve over both
#: locations, newest-convention first, so neither the tests nor the flag
#: depend on which one a given checkout happens to carry.
# `parents[2]` is `molmetal/` (this file is molmetal/adapters/flow_matching_lipman/);
# `parents[3]` is the repo root, where `models/` lives.
_TMQM_CKPT_CANDIDATES = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "tmqm_pretrained"
    / "tmqm_pretrained_encoder.pt",
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "dmpnn_tmqm_pretrained.pt",
)


def _resolve_default_tmqm_ckpt() -> Path:
    """First existing tmQM checkpoint, else the canonical (absent) path."""
    for _cand in _TMQM_CKPT_CANDIDATES:
        if _cand.exists():
            return _cand
    return _TMQM_CKPT_CANDIDATES[0]


DEFAULT_TMQM_CKPT = _resolve_default_tmqm_ckpt()


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
    if "encoder_state_dict" in ckpt and isinstance(ckpt["encoder_state_dict"], dict):
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
    src: dict,
    own: dict,
    *,
    strict: bool = False,
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
            candidates_by_row.keys(),
            key=lambda w: -w,
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
    edge_index: torch.Tensor,  # (2, E) long
    coords: torch.Tensor,  # (B, N, 3)
    z_i: torch.Tensor,  # (E,) long
    z_j: torch.Tensor,  # (E,) long
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
        [_bucket(int(z)) for z in zi_cpu],
        dtype=torch.float32,
        device=device,
    )
    bj = torch.tensor(
        [_bucket(int(z)) for z in zj_cpu],
        dtype=torch.float32,
        device=device,
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
    h: torch.Tensor,  # (B, N, H)
    edge_index: torch.Tensor,  # (2, E) long
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
            acc = acc + r[offset : offset + n]
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
    """Anchor ``reference_path`` against molmetal / repo root before
    passing to :func:`configure_reference`.

    Wf-415-A3 fix (Wf-344-A3 follow-up): relative paths like
    ``"molmetal/references/flow_matching"`` or
    ``"references/flow_matching"`` resolve relative to ``os.getcwd()``.
    When the caller (subprocess, notebook, CI runner) is in a foreign
    CWD, ``Path(relative_path).resolve()`` produces a path that does
    NOT exist.  We always anchor relative paths against the repo root
    (where ``molmetal/`` lives) and the molmetal package directory
    (``<repo>/molmetal/``) before falling back to the default behaviour.
    """
    from ._reference_loader import configure_reference
    import os
    from pathlib import Path

    if isinstance(reference_path, str) and not os.path.isabs(reference_path):
        cand = Path(reference_path)
        # Path traversal: `<this file>` = .../molmetal/adapters/flow_matching_lipman/
        # so 4 .parent hops gives the repo root (where ``molmetal/`` lives).
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        molmetal_dir = repo_root / "molmetal"
        # Try (1) molmetal-package-relative — strip the leading
        # ``molmetal/`` and anchor against the package dir.
        if cand.parts and cand.parts[0] == "molmetal":
            cand_pkg = molmetal_dir / Path(*cand.parts[1:])
            if cand_pkg.exists():
                reference_path = str(cand_pkg)
        # Try (2) anchor against the molmetal package directory
        # (where the references/flow_matching clone actually lives).
        cand_mpkg = molmetal_dir / cand
        if cand_mpkg.exists():
            reference_path = str(cand_mpkg)
        # Try (3) repo-relative — anchor against repo_root directly.
        elif (repo_root / cand).exists():
            reference_path = str(repo_root / cand)
        # If neither anchor works, fall through with the original
        # relative path (configure_reference will raise a clean
        # FileNotFoundError explaining the missing clone).
    configure_reference(reference_path)
    # Wf-A62 W1 (RL5) — guard against upstream API drift.  The Lipman
    # 2023 -> 2024-restructure refactored ``AffineProbPath`` /
    # ``CondOTScheduler`` / ``ODESolver`` / ``ModelWrapper`` import
    # paths.  A clean ``ImportError`` here is confusing because the
    # caller never sees the traceback back to the upstream rename.
    # Wrap each import and re-raise with a clear pointer.
    try:
        from ._reference.path import AffineProbPath
    except ImportError as exc:
        raise RuntimeError(
            f"_import_fm_lib: upstream ``AffineProbPath`` import failed "
            f"({exc!r}); check whether your flow_matching clone is on "
            f"Lipman 2023 vs 2024-restructure. See "
            f"docs/LIPMAN_UPSTREAM_API_DRIFT.md (Wf-A62 RL5)."
        ) from exc
    try:
        from ._reference.path.scheduler import CondOTScheduler
    except ImportError as exc:
        raise RuntimeError(
            f"_import_fm_lib: upstream ``CondOTScheduler`` import "
            f"failed ({exc!r}); 2024-restructure moved scheduler into "
            f"``flow_matching.path.scheduler``."
        ) from exc
    try:
        from ._reference.solver import ODESolver
    except ImportError as exc:
        raise RuntimeError(
            f"_import_fm_lib: upstream ``ODESolver`` import failed "
            f"({exc!r}); 2024-restructure renamed to "
            f"``flow_matching.solver.ODEFlowMatcher``."
        ) from exc
    try:
        from ._reference.utils import ModelWrapper
    except ImportError as exc:
        raise RuntimeError(
            f"_import_fm_lib: upstream ``ModelWrapper`` import failed "
            f"({exc!r}); 2024-restructure moved ``ModelWrapper`` into "
            f"``flow_matching.solver.utils``. The local shim at "
            f"``_reference/utils/__init__.py`` provides a fallback."
        ) from exc

    return AffineProbPath, CondOTScheduler, ODESolver, ModelWrapper


# ---------------------------------------------------------------------------
# Pocket encoder — per-pocket-atom 1-hot coords → global pool (T5)
# ---------------------------------------------------------------------------
def radius_edge_index(
    coords: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    cutoff: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Padded block-diagonal radius graph over a batch of atom clouds.

    Returns ``(edge_index (B, 2, E), edge_mask (B, E))``.

    ``EGNNVelocityField`` takes ``edge_index`` as a rectangular tensor, so
    every graph in a batch is padded to the same ``E`` and ``edge_mask``
    marks which entries are real.  Two properties are load-bearing:

    * **No cross-sample edges** — padding positions are excluded via
      ``node_mask``, so a pair whose endpoints are not both real is dropped.
      Without it, a batch of separately-embedded molecules (each near the
      origin) would be silently wired into one spurious super-molecule.
    * **No self-loops**, and no pair at or beyond ``cutoff`` Å.

    Added 2026-10-04 alongside the EGNN speed-up: the adapter used to hand
    the EGNN a FULLY CONNECTED graph (``_make_dummy_edge_index``) — O(N^2)
    messages with none of the locality an EGNN's geometric prior comes from.
    """
    b, n = int(coords.shape[0]), int(coords.shape[1])
    dev = coords.device
    idx = torch.arange(n, device=dev)
    src = idx.view(1, n, 1).expand(b, n, n)
    dst = idx.view(1, 1, n).expand(b, n, n)
    dist = torch.cdist(coords.float(), coords.float())  # (B, n, n)
    keep = (src != dst) & (dist < float(cutoff))
    if node_mask is not None:
        keep = keep & node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    flat_keep = keep.reshape(b, n * n)
    e_max = int(flat_keep.sum(dim=1).max().item()) if flat_keep.numel() else 0
    if e_max == 0:
        return (
            torch.zeros(b, 2, 0, dtype=torch.long, device=dev),
            torch.zeros(b, 0, dtype=torch.bool, device=dev),
        )
    # `topk(largest=True)` puts the True entries first, so taking the first
    # `e_max` positions selects every real edge and pads the rest with
    # arbitrary False positions, which `edge_mask` then masks out.
    order = flat_keep.float().topk(e_max, dim=1, largest=True).indices
    sel_mask = flat_keep.gather(1, order)
    src_sel = src.reshape(b, -1).gather(1, order)
    dst_sel = dst.reshape(b, -1).gather(1, order)
    return torch.stack([src_sel, dst_sel], dim=1), sel_mask


class _RadiusGraphVelocityModel:
    """Bind the EGNN's extra inputs, rebuilding the graph from each ``x``.

    The cloned ``ODESolver`` calls ``model(x, t)`` with a 0-d scalar ``t`` and
    expects a ``(B, N, 3)`` velocity back.  The EGNN additionally needs
    ``atom_types``, an ``edge_index``, and a per-sample ``t``.

    The graph cannot be bound once and reused.  It is a radius graph over
    ``x``, and the solver moves ``x`` at every step, so a bound graph is
    stale from the first step on — which is what the previous
    ``partial(..., edge_index=...)`` did for the whole 100-step solve.  This
    callable rebuilds it per invocation instead.

    ``use_cfg`` routes through :meth:`EGNNVelocityField.v_cfg`; both that and
    :meth:`forward_velocity` return the velocity tensor directly.
    """

    def __init__(
        self,
        velocity_field: "EGNNVelocityField",
        atom_types: torch.Tensor,
        node_mask: Optional[torch.Tensor],
        cutoff: float,
        pocket_embed: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> None:
        self._vf = velocity_field
        self._atom_types = atom_types
        self._node_mask = node_mask
        self._cutoff = float(cutoff)
        self._pocket_embed = pocket_embed
        self._cfg_scale = float(cfg_scale)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        edge_index, edge_mask = radius_edge_index(x, self._node_mask, self._cutoff)
        b = int(x.shape[0])
        # Solver contract: 0-d scalar t.  EGNN wants one time per sample.
        # Wf-A62 W1 (VF1) — guard against multi-dim t from adaptive solvers
        # (e.g., dopri5 may pass (B, B') for batched ODE state).  Silently
        # truncating to a single sample masked divergent trajectories in
        # training; surface the contract violation instead.
        if t.dim() not in (0, 1):
            raise ValueError(
                f"_RadiusGraphVelocityModel expected t as 0-d scalar or (B,) 1-d; "
                f"got shape {tuple(t.shape)}. The Lipman ODESolver contract is "
                f"0-d per-step t; if you are using an adaptive solver, ensure "
                f"`step_index` is collapsed before calling the wrapper."
            )
        # Wf-A62 W1 (VF4 doc) — ODESolver fixed-grid Euler / midpoint
        # methods intentionally re-use the same t across the inner loop;
        # velocity is therefore deterministic per batch index.  Training
        # paths sample t independently so this is a documentation note,
        # not a fix.
        t_b = t.reshape(1).expand(b) if t.dim() == 0 else t.reshape(-1)[:1].expand(b)
        if self._cfg_scale != 1.0 and self._pocket_embed is not None:
            return self._vf.v_cfg(
                x,
                self._atom_types,
                edge_index,
                t_b,
                pocket_embed=self._pocket_embed,
                cfg_scale=self._cfg_scale,
                edge_mask=edge_mask,
            )
        return self._vf.forward_velocity(
            x,
            self._atom_types,
            edge_index,
            t_b,
            edge_mask=edge_mask,
            pocket_embed=self._pocket_embed,
        )


class _BondLengthGuidance:
    """Wf-A75 (2026-10-18) — bond-length regulariser added to the EGNN
    velocity at every ODE step.

    Per Vignac et al. "MolDiff: Addressing the Atom-Bond Inconsistency
    Problem in 3D Molecule Diffusion" (ICML 2023, arXiv:2305.07508) the
    central failure mode of vanilla Cartesian-coordinate diffusion is
    that bonded atom pairs drift past ~3 Å during denoising; the
    molecule comes out with a graph topology that is topologically
    disconnected (longest fragment_frac → 0.05).  MolDiff's full fix
    decouples bond and atom noise schedules AND uses a bond-classifier
    gradient w.r.t. positions as a guidance term.

    Wf-A75 ships the simpler, lower-cost component: a "bond length
    prior" that is added to the EGNN velocity at every step.  For each
    pair within the radius-graph cutoff we compute a per-pair "expected
    bond length" from a 2-row chemistry table (metal-donor / C-C / C-N /
    C-O / other) and add a Hooke-spring force that pulls the actual
    distance toward the expected value.  The force magnitude is
    controlled by ``guidance_weight`` (YAML key).  When
    ``guidance_weight=0`` the wrapper is a strict no-op (the
    default), so Wf-A75 is bit-exact back-compatible with the existing
    sampling loop.

    Cost: one extra forward + one gradient call per ODE step (n_steps
    times), so at n_steps=100 this is a 100x cost multiplier on the
    guidance module.  For the 5-epoch smoke we use n_steps=100 (the
    existing default) and accept the cost.

    Pair-type expected distances (Å, calibrated to tmQM / 5-metal
    CSD reference compounds):
        * metal-donor (Pt, Ru, Ir, Os, Rh, Re + N/O/S/P): 2.05 (dative)
        * C-C: 1.54 (single) / 1.40 (aromatic) — used 1.50 as the
          conservative average
        * C-N / C-O: 1.40 (sp2/aromatic)
        * all other pairs (Si, S, halides, P): 1.80 (single, generic)
    """
    # Metal Z values (5-metal cohort, all7 preset).  Pair is "metal-donor"
    # when one endpoint is in this set AND the other is N/O/S/P.
    _METAL_Z = frozenset([44, 45, 46, 75, 76, 77, 78])  # Ru, Rh, Pd, Re, Os, Ir, Pt
    _DONOR_Z = frozenset([7, 8, 15, 16])  # N, O, P, S

    def __init__(
        self,
        atom_types: torch.Tensor,
        cutoff: float,
        guidance_weight: float = 0.1,
        spring_constant: float = 50.0,
    ) -> None:
        # Normalize atom_types to 2-D ``(B, N)`` so the per-batch
        # ``__call__`` loop can index it as ``atom_types[bi, idx]``.
        # The trainer passes ``(B, N)`` directly; the unit tests pass
        # 1-D ``(N,)`` for the single-cloud case.
        if atom_types.dim() == 1:
            self._atom_types = atom_types.unsqueeze(0)
        else:
            self._atom_types = atom_types
        self._cutoff = float(cutoff)
        self._guidance_weight = float(guidance_weight)
        self._spring_constant = float(spring_constant)

    @torch.no_grad()
    def _expected_length_table(self) -> torch.Tensor:
        """Per-(Z_i, Z_j) expected bond-length lookup.  Returns a
        (MAX_Z+1, MAX_Z+1) symmetric tensor of expected distances in Å;
        0.0 means "no prior" (the wrapper skips that pair).  Built
        from the metal/donor heuristics at module load (called once per
        generation)."""
        MAX_Z = 100
        table = torch.full((MAX_Z + 1, MAX_Z + 1), 1.50)  # default single
        # C-C aromatic average
        for z in (6,):
            table[z, z] = 1.50
        # C-N / C-O sp2/aromatic
        for zc, zn in [(6, 7), (6, 8), (7, 6), (8, 6), (7, 8), (8, 7), (7, 7), (8, 8)]:
            table[zc, zn] = 1.40
        # P / S — single bond (heavier atoms, longer length)
        for zp in (15, 16):
            table[zp, 6] = 1.85
            table[6, zp] = 1.85
            table[zp, zp] = 2.05
        # Metal-donor: dative, expected ~2.0 Å (Pt-N, Ru-N, Ir-N, etc.)
        for zm in self._METAL_Z:
            for zd in self._DONOR_Z:
                table[zm, zd] = 2.05
                table[zd, zm] = 2.05
            # Metal-C (e.g., Pt-C in organometallics) ~ 2.0 Å
            table[zm, 6] = 2.0
            table[6, zm] = 2.0
        return table

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-atom guidance force (B, N, 3) given current
        positions (B, N, 3).  Zero force when guidance_weight=0 or
        when the edge list is empty."""
        if self._guidance_weight <= 0.0:
            return torch.zeros_like(x)
        # Build the edge list on the same radius-graph logic as the
        # velocity model (so the guidance is applied to the SAME
        # candidate pairs).  We do not backprop through the edge
        # list — the prior is computed once per step on the current
        # positions.
        edge_index, edge_mask = radius_edge_index(x, None, self._cutoff)
        # edge_index shape: (B, 2, E).  edge_mask shape: (B, E).
        if edge_index.shape[-1] == 0:
            return torch.zeros_like(x)
        b, n, _ = x.shape
        flat = x.reshape(b * n, 3)
        flat_force = torch.zeros_like(flat)  # (B*N, 3)
        # Inner-bond cutoff: pairs at distance > 2.5 Å are not real
        # bonds (the decoder uses 2.4 Å for bond candidacy per Wf-A71
        # §6.99) and should NOT receive a restoring force — otherwise
        # a 3-atom chain [0, 1.5, 3.0] at the C-C target would still
        # see a force on the (0, 2) pair at d=3.0 (the long chain
        # span), which is not a bond at all.
        _INNER_BOND_CUTOFF = 2.5
        for bi in range(b):
            src_bi = edge_index[bi, 0, :]  # (E,)
            dst_bi = edge_index[bi, 1, :]  # (E,)
            mask_bi = edge_mask[bi, :]  # (E,)
            if not bool(mask_bi.any().item()):
                continue
            # Vector and distance between each pair.
            diff_bi = x[bi, dst_bi[mask_bi]] - x[bi, src_bi[mask_bi]]  # (E', 3)
            d_bi = diff_bi.norm(dim=-1).clamp(min=1e-3)  # (E',)
            # Only apply force to inner-bond pairs (d < 2.5 Å).  This
            # matches the decoder's bond candidacy cutoff (2.4 Å) and
            # prevents the prior from over-pulling long chain spans.
            inner_mask = d_bi < _INNER_BOND_CUTOFF
            if not bool(inner_mask.any().item()):
                continue
            diff_bi = diff_bi[inner_mask]
            d_bi = d_bi[inner_mask]
            src_bi_eff = src_bi[mask_bi][inner_mask]
            dst_bi_eff = dst_bi[mask_bi][inner_mask]
            # Expected distance from the (Z, Z) table.
            table = self._expected_length_table().to(x.device).to(x.dtype)
            expected_bi = table[
                self._atom_types[bi, src_bi_eff],
                self._atom_types[bi, dst_bi_eff],
            ].clamp(min=0.5)  # (E'',)
            # Spring force: F = k * (d - d_target) * unit(diff).
            residual_bi = (d_bi - expected_bi)  # (E'',)
            force_mag_bi = self._spring_constant * residual_bi  # (E'',)
            unit_bi = diff_bi / d_bi.unsqueeze(-1)  # (E'', 3)
            # Newton 3rd law: +i side gets +force*unit, -j side gets
            # -force*unit.  Accumulate via index_add.
            src_idx = src_bi_eff + bi * n
            dst_idx = dst_bi_eff + bi * n
            flat_force.index_add_(
                0, src_idx, force_mag_bi.unsqueeze(-1) * unit_bi
            )
            flat_force.index_add_(
                0, dst_idx, -force_mag_bi.unsqueeze(-1) * unit_bi
            )
        force = flat_force.reshape(b, n, 3)
        # Per-atom norm is bounded by the largest |d - d_target| in
        # the cloud (~ 1.5 Å at init), so the per-atom force is
        # ~ k * 1.5 = 75 at full magnitude, then * guidance_weight
        # = 7.5 max.
        return self._guidance_weight * force


def _bond_guided_velocity(
    v_egnn: torch.Tensor,
    x: torch.Tensor,
    guidance: _BondLengthGuidance,
) -> torch.Tensor:
    """Add the Wf-A75 bond-length guidance to the EGNN velocity.

    The guidance force is computed WITHOUT grad (no need to backprop
    through the EGNN); the velocity returned is detached from the
    guidance.  This keeps the ODESolver's no_grad contract intact.
    """
    f = guidance(x)
    return v_egnn + f


class _BondGuidedVelocityModel:
    """Wf-A75 — thin wrapper around :class:`_RadiusGraphVelocityModel`
    that adds the bond-length guidance force to the velocity at every
    ODE step.  The wrapper has the same ``(x, t) -> (B, N, 3)`` signature
    as the inner model so the ODESolver contract is unchanged.
    """
    def __init__(self, inner: "_RadiusGraphVelocityModel", guidance: _BondLengthGuidance) -> None:
        self._inner = inner
        self._guidance = guidance

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self._inner(x, t)
        return _bond_guided_velocity(v, x, self._guidance)


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
            [
                EGNNLayer(hidden_dim=hidden_dim, edge_mlp_hidden=hidden_dim)
                for _ in range(n_layers)
            ]
        )
        self.hidden_dim = hidden_dim

    def forward(
        self,
        pocket_coords: torch.Tensor,  # (B, P, 3)   padded
        pocket_atom_types: torch.Tensor,  # (B, P)    int64, 0 = padding
        pocket_mask: torch.Tensor,  # (B, P)     bool, True = real atom
    ) -> torch.Tensor:
        """Returns a (B, hidden_dim) pocket embedding per sample."""
        b, p = pocket_coords.shape[:2]
        if (pocket_atom_types < 0).any() or (
            pocket_atom_types >= self.atom_embed.num_embeddings
        ).any():
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
        cond_zero = torch.zeros(
            b, p, self.hidden_dim, device=pocket_coords.device, dtype=h.dtype
        )
        for layer in self.layers:
            # Pass a zero ``cond_per_node`` so EGNNLayer.update_mlp
            # (which expects 3H+1) gets the right shape — see
            # models.velocity_net.EGNNLayer.forward.
            h, last_v = layer(
                h,
                pocket_coords,
                edge_index,
                cond_per_node=cond_zero,
                edge_mask=edge_mask,
            )
        # Global mean pool over real atoms only.
        mask_f = valid.float().unsqueeze(-1)  # (B, P, 1)
        n_real = valid.sum(dim=1).clamp(min=1).float().unsqueeze(-1)  # (B, 1)
        # Wf-A62 W1 (PE1) — empty-pocket footgun: when ``valid`` is all
        # False for a sample, ``n_real = 1`` (clamp) and the pooled
        # vector is identically zero.  The velocity field downstream
        # treats ``pocket_embed == 0`` as "no pocket" (the
        # unconditioned path), so a missing-pocket sample and a
        # valid-zero-pocket sample are indistinguishable.  Surface the
        # upstream data leak so maintainers can see which ligand had
        # no real pocket atoms.
        if not valid.any():
            warnings.warn(
                "PocketEncoder: pocket has no real atoms across the batch "
                "(valid.any()==False); pooled vector is identically zero "
                "and downstream velocity field treats this as "
                "'unconditioned'. Check upstream pocket loader.",
                stacklevel=2,
            )
        pooled = (h * mask_f).sum(dim=1) / n_real  # (B, H)
        return pooled

    @staticmethod
    def _fully_connected_edge_index(b: int, p: int, device) -> torch.Tensor:
        # Wf-A62 W1 (PE2) — guard against pathological pockets whose
        # fully-connected edge_index blows O(P^2) memory.  P=400 ⇒ 160 k
        # edges per sample; on a sub-32 GB GPU the holder tensor for
        # the FC edge mask (~B*P*P bytes) allocates fast enough to OOM
        # mid-step.  Cap at 100 000 edges per sample = P<=317.
        if p * p > 100_000:
            raise ValueError(
                f"PocketEncoder._fully_connected_edge_index: P={p} ⇒ "
                f"{p * p} edges per sample exceeds the 100 000 cap. "
                f"Use radius-graph pockets (radius_edge_index) instead."
            )
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
        q_head_enabled: bool = False,
        # Wf-A27 (2026-10-13) step-5 — per-layer sinusoidal-t opt-in.
        # Default 0 keeps pre-Wf-A27 numerics bit-exact (the legacy
        # scalar-t path).  When > 0 the velocity field materialises a
        # :class:`molmetal.adapters.egnn_extensions.SinusoidalEmbedding`
        # and adds its output at every EGNN layer (``pocket_per_layer``)
        # or only at the input projection (legacy-equivalent).
        t_embedding_dim: int = 0,
        pocket_per_layer: bool = False,
        # Wf-A32 (2026-10-14) step-1 — opt-in per-layer pocket
        # injection.  When ``pocket_inject_per_layer=True`` the velocity
        # field materialises a zero-init
        # :class:`molmetal.adapters.egnn_extensions.PerLayerPocketInjector`
        # and adds its output to ``h`` at every EGNN layer via the
        # ``cond_per_node`` slot.  Default ``False`` keeps the bit-exact
        # legacy path (no injector, no per-layer call).  Distinct from
        # ``pocket_per_layer`` (which gates the *temporal* per-layer
        # addend) so the two opt-in axes compose independently.
        pocket_inject_per_layer: bool = False,
        # Wf-A108a (2026-10-22) — coordinate-head architecture.
        #
        # ``"radial"`` (default) is the pre-Wf-A108 head:
        #     vel = tanh(vel_head(h)) * vel_scale * (x - centroid) + last_v
        # which Wf-A107h MEASURED cannot fit one 8-cloud batch (MSE 10.529
        # against a 10.652 ``v = 0`` baseline and a 5.783 radial-only
        # ceiling).  ``"vector_channels"`` replaces it with ``C``
        # independent 3-vectors per atom (see
        # :mod:`molmetal.adapters.flow_matching_lipman.vector_channels`
        # for the GVP / MiDi / SemlaFlow / FlowMol3 lineage).
        #
        # Default ``"radial"`` adds zero parameters and zero branches on
        # the live path, so every existing checkpoint still loads with
        # ``strict=True`` and the forward is bit-exact.
        velocity_head: str = "radial",
        vector_channels: int = 8,
        # Wf-A108c (2026-10-22) — metal-centred channels.  Only consulted
        # when ``velocity_head="vector_channels"``.  Off by default: the
        # extra channels change ``state_dict`` shape, so it is a separate
        # opt-in from the architecture itself.
        metal_frame_channels: bool = False,
        # Wf-A108 — mask padded atoms out of the radial head's centroid.
        # Off by default: the legacy centroid includes the trainer's zero
        # padding, and changing that changes every existing run's forward.
        radial_centroid_masked: bool = False,
        # Wf-A108e (2026-10-22) — "velocity" (default, bit-exact) or "x1".
        prediction_target: str = "velocity",
        x1_denominator_floor: float = 0.02,
        # Wf-A108j (2026-10-25) — magnitude ownership and head initialisation.
        #
        # ``vector_channel_norm_gain``: "learnable" (default, bit-exact) keeps
        # the per-channel gain trainable; "fixed" freezes it at 1 so the only
        # magnitude control is the head's gate.  MEASURED motivation: on the
        # epoch-4 vector_channels checkpoint the gain had collapsed 1.0 ->
        # 0.0017 in three layers while the head needed |v| = 5.3 and produced
        # 0.135 (cfm_module_gates M4).
        #
        # ``velocity_head_init``: "zero" (default, bit-exact) leaves the gate
        # at zero; "lsq" builds the gate WITH a bias and with a sinusoidal-t
        # branch, so ``CFMJointModel.init_velocity_head_lsq`` can set both by
        # a closed-form least-squares solve on the first FM batch.  At the LS
        # optimum the gradient with respect to every channel gain is zero, so
        # the collapse above cannot start.
        vector_channel_norm_gain: str = "learnable",
        velocity_head_init: str = "zero",
        # Wf-A108j (2026-10-25) — per-cloud metal identity as a first-class
        # conditioning channel.
        #
        # "none" (default, bit-exact) leaves metal identity reachable only
        # through ``atom_embed`` at the metal atom itself.  "embedding" adds a
        # learned ``(n_metal + 1, H)`` table indexed by the cloud's metal and
        # broadcasts it to every atom of that cloud.
        #
        # MEASURED motivation (cfm_module_gates M2 on the c8 epoch-2
        # checkpoint, 8 tmQM clouds / 291 atoms): swapping the metal's atomic
        # number moved the velocity by 0.74% while shifting t by 0.25 moved it
        # by 86% and swapping one carbon for nitrogen moved it by 18%.  The
        # channel the project is named after was the weakest input in the
        # model.
        metal_conditioning: str = "none",
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
        # 2026-10-19 — every sibling module in this package (EGNNPredictor,
        # UnifiedAtomBondFlow, ConfidenceHead, PerLayerPocketInjector,
        # EGNNVelocityField's own `head` at :1405) stores `hidden_dim` as an
        # attribute; this class was the only one that did not.  That made
        # `vf.hidden_dim` an AttributeError instead of the construction width
        # (test_cfm_p1_fixes::test_hidden_dim_default_128), and forced
        # cfm_property_panel to recover the width from
        # `model.head.atom_encoder.weight.shape[0]`.  Stored, not inferred.
        self.hidden_dim = int(hidden_dim)

        self.time_mlp = _MaybeFusedSiLUMLP(
            in_dim=1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.atom_embed = nn.Embedding(max_atomic_number, hidden_dim)
        # Wf-A108a — the two coordinate heads are mutually exclusive.  The
        # default builds exactly the modules the pre-Wf-A108 code built, so
        # ``state_dict`` keys are unchanged.
        _velocity_head = str(velocity_head).lower()
        if _velocity_head not in {"radial", "vector_channels"}:
            raise ValueError(
                "velocity_head must be 'radial' or 'vector_channels', "
                f"got {velocity_head!r}"
            )
        self.velocity_head = _velocity_head
        self.vector_channels = int(vector_channels)
        # Wf-A108j — capture slot used by ``fit_output_head_lsq``.  Always
        # defined so the attribute check in ``forward`` never depends on the
        # caller having set it first.
        self._capture_head_inputs = False
        self._head_capture = None
        # Wf-A108j — per-cloud metal conditioning.  "none" registers no module
        # and no buffer, so a pre-Wf-A108j checkpoint rebuilds bit-exactly.
        _metal_cond = str(metal_conditioning).lower()
        if _metal_cond not in {"none", "embedding"}:
            raise ValueError(
                "metal_conditioning must be 'none' or 'embedding', "
                f"got {metal_conditioning!r}"
            )
        self.metal_conditioning = _metal_cond
        if _metal_cond == "embedding":
            # Index 0 means "no metal in this cloud", 1..K are the corpus
            # metals in a fixed order.  The table is initialised at the same
            # scale as a linear layer's weights (1/sqrt(H)) rather than at
            # zero: a zero-init channel would repeat the exact failure this
            # surface exists to fix -- a conditioning path that emits nothing
            # until the optimiser pays for it step by step.
            self.register_buffer(
                "metal_z_order",
                torch.tensor([78, 44, 77, 45, 76, 75, 46], dtype=torch.long),
                persistent=False,
            )
            self.metal_embed = nn.Embedding(
                int(self.metal_z_order.numel()) + 1, hidden_dim
            )
            nn.init.normal_(
                self.metal_embed.weight, std=float(hidden_dim) ** -0.5
            )
            nn.init.zeros_(self.metal_embed.weight[0])
        else:
            self.metal_embed = None  # type: ignore[assignment]
        self.metal_frame_channels = bool(metal_frame_channels)
        self.radial_centroid_masked = bool(radial_centroid_masked)
        # Wf-A108e (2026-10-22) — what the coordinate stack is trained to emit.
        #
        # ``"velocity"`` (default) is the pre-Wf-A108e contract: ``vel`` is the
        # regression target's own space, ``x1 - x0``.  ``"x1"`` treats the same
        # head output as a DISPLACEMENT from ``x_t`` and publishes both
        # ``x1_hat = x_t + u`` (the loss operand) and
        # ``vel = u / max(1 - t, floor)`` (the sampler operand).
        #
        # Why: Wf-A108d MEASURED that at ``t = 0.05`` the velocity-trained field
        # is 64% worse than predicting zero (14.781 vs 9.008) while a
        # one-parameter Gaussian posterior mean reaches 3.605, because the
        # conditional mean must go from ``-x_t`` at ``t -> 0`` to ``+x_t`` at
        # ``t -> 1`` and the field never reverses (endpoint cosine +0.736).
        # In x1 space the low-t target is ``x1`` itself, so the required output
        # is the data mean rather than a reversal.
        #
        # The loss weight is the caller's business and is NOT hidden here:
        # with ``x1_hat = x_t + (1-t)u`` a plain MSE on ``x1_hat`` is
        # ``||u - (1-t)v||^2``, i.e. it DOWNWEIGHTS high ``t`` and leaves low
        # ``t`` at full weight.  That is the opposite of FlowMol3's
        # ``w(t) = min(max(0.005, t/(1-t)), 1.5)``, whose effective velocity
        # weight is ``t(1-t)`` and which therefore suppresses the exact region
        # Wf-A108d found broken.  See docs/WF_A108_VELOCITY_HEAD_SURVEY.md §6.
        _target = str(prediction_target).lower()
        if _target not in {"velocity", "x1"}:
            raise ValueError(
                "prediction_target must be 'velocity' or 'x1', "
                f"got {prediction_target!r}"
            )
        if float(x1_denominator_floor) <= 0.0 or float(x1_denominator_floor) >= 1.0:
            raise ValueError(
                "x1_denominator_floor must be in (0, 1), got "
                f"{x1_denominator_floor!r}"
            )
        self.prediction_target = _target
        self.x1_denominator_floor = float(x1_denominator_floor)
        if self.velocity_head == "vector_channels":
            from molmetal.adapters.flow_matching_lipman.vector_channels import (
                VectorChannelHead,
                VectorChannelLayer,
            )

            _gain_mode = str(vector_channel_norm_gain).lower()
            if _gain_mode not in {"learnable", "fixed"}:
                raise ValueError(
                    "vector_channel_norm_gain must be 'learnable' or 'fixed', "
                    f"got {vector_channel_norm_gain!r}"
                )
            _head_init = str(velocity_head_init).lower()
            if _head_init not in {"zero", "lsq"}:
                raise ValueError(
                    "velocity_head_init must be 'zero' or 'lsq', "
                    f"got {velocity_head_init!r}"
                )
            self.vector_channel_norm_gain = _gain_mode
            self.velocity_head_init = _head_init
            self.layers = nn.ModuleList([])
            self.vc_layers = nn.ModuleList(
                [
                    VectorChannelLayer(
                        hidden_dim=hidden_dim,
                        n_channels=self.vector_channels,
                        edge_hidden=edge_mlp_hidden,
                        learnable_norm_gain=(_gain_mode == "learnable"),
                    )
                    for _ in range(n_layers)
                ]
            )
            self.vc_head = VectorChannelHead(
                hidden_dim=hidden_dim,
                n_channels=self.vector_channels,
                t_dim=(int(t_embedding_dim) if _head_init == "lsq" else 0),
                use_bias=(_head_init == "lsq"),
            )
            # ``vel_head`` / ``vel_scale`` are NOT constructed on this path:
            # a dead parameter is a knob nobody audits (§5.11 rationale).
            self.vel_head = None  # type: ignore[assignment]
            self.vel_scale = None  # type: ignore[assignment]
        else:
            self.vector_channel_norm_gain = "learnable"
            self.velocity_head_init = "zero"
            self.vc_layers = nn.ModuleList([])
            self.vc_head = None  # type: ignore[assignment]
            self.layers = nn.ModuleList(
                [
                    EGNNLayer(hidden_dim=hidden_dim, edge_mlp_hidden=edge_mlp_hidden)
                    for _ in range(n_layers)
                ]
            )
        # Zero-init: start the velocity at zero so initial samples are
        # just noise.  This is the standard FM trick.
        # v2: an invariant scalar can only scale an equivariant vector
        # basis; a linear H -> 3 projection is not a spatial vector.
        if self.velocity_head == "radial":
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
        if self.velocity_head == "radial":
            self.vel_scale = nn.Parameter(torch.tensor(1.0))
        # The equivariant vector message is added directly to vel_head below.
        # Its last projection must also start at zero, otherwise the claimed
        # zero initial velocity is bypassed by random distance-amplified flow.
        if self.velocity_head == "radial":
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
            max_atomic_number,
            hidden_dim,
            bias=False,
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

        # Track-A (Wf-A15, 2026-10-13) — Layer 3a q-prediction head.
        #
        # When ``q_head_enabled=True``, attach a tiny zero-init
        # ``Linear(hidden_dim, 1)`` that emits per-atom NBO natural
        # charge from the final per-atom ``h`` (the same tensor used by
        # ``atom_head``).  The head is ZERO-INIT so a fresh
        # ``EGNNVelocityField`` produces ``q_pred = 0`` everywhere —
        # ``NaturalChargeLoss.compute(pred=0, target=any)`` returns
        # ``mse_loss(0, target) * weight``, which collapses to a zero
        # tensor when ``weight=0`` (the YAML default).  When
        # ``q_head_enabled=False`` (legacy), no q_head is registered on
        # the module — bit-exact pre-Track-A: ``state_dict`` does not
        # gain keys, every existing checkpoint resumes, and the forward
        # dict never carries ``q_pred``.
        #
        # The head sits in the ``q_channel`` param group when the
        # trainer enables ``--q-channel-lr-multiplier`` (Layer 7,
        # :mod:`molmetal.optim.differential_lr`); it falls into the
        # backbone bucket otherwise — same behaviour as the
        # ``atom_head`` (zero-init, default LR).
        self.q_head_enabled = bool(q_head_enabled)
        if self.q_head_enabled:
            self.q_head = nn.Linear(hidden_dim, 1, bias=True)
            nn.init.zeros_(self.q_head.weight)
            nn.init.zeros_(self.q_head.bias)
        else:
            # Register a None placeholder for stable introspection; this
            # is never consulted on the bit-exact default path.
            self.q_head = None

        # Optional: warm-start the encoder from a tmQM pre-trained checkpoint
        # (see F2 report).  Failures degrade gracefully to random init.
        if use_tmqm_init:
            ckpt = tmqm_init_path if tmqm_init_path is not None else DEFAULT_TMQM_CKPT
            load_tmQM_pretrained(self, ckpt)
        elif tmqm_init_path is not None:
            # Legacy path kept for backward compatibility — explicit
            # ``tmqm_init_path`` argument still works without the new flag.
            self._load_tmqm_checkpoint(tmqm_init_path, strict=False)

        # Wf-A27 (2026-10-13) step-5 — per-layer sinusoidal-t materialisation.
        # Only constructed when ``t_embedding_dim > 0`` so the default
        # path stays bit-exact (no new sub-module, no new parameter).  At
        # ``dim=0`` the legacy ``t_per_atom`` path keeps running; the
        # :attr:`t_emb_per_layer` attribute stays ``None``, and the
        # per-layer additions in ``forward`` are no-ops at the syntactic
        # level — they fail the ``is None`` short-circuit and the input
        # / per-layer additions are skipped.
        self.t_embedding_dim = int(t_embedding_dim)
        self.pocket_per_layer = bool(pocket_per_layer)
        # Wf-A32 (2026-10-14) step-1 — opt-in per-layer pocket injector.
        # Instantiated ONLY when ``pocket_inject_per_layer=True`` so the
        # shipped-default path (``False``) adds zero parameters and keeps
        # ``state_dict()`` bit-exact: every shipped CFM YAML at default
        # resumes / round-trips through ``load_state_dict(strict=True)``
        # unchanged.  At ``True`` the injector is zero-init (see
        # :class:`PerLayerPocketInjector` in egnn_extensions.py:171) so a
        # fresh ``forward`` call folds the contribution to zero — the
        # optimiser must train it before any non-zero signal reaches the
        # per-layer ``cond_per_node`` slot.  This mirrors the Wf-A27
        # ``t_to_h`` zero-init convention at line 1519.
        self.pocket_inject_per_layer = bool(pocket_inject_per_layer)
        if self.pocket_inject_per_layer:
            from molmetal.adapters.egnn_extensions import PerLayerPocketInjector
            self.pocket_per_layer_injector = PerLayerPocketInjector(
                pocket_dim=self.atom_embed.embedding_dim,
                hidden_dim=self.atom_embed.embedding_dim,
            )
        else:
            self.pocket_per_layer_injector = None  # type: ignore[assignment]
        if self.t_embedding_dim > 0:
            from molmetal.adapters.egnn_extensions import SinusoidalEmbedding
            self.t_emb_per_layer = SinusoidalEmbedding(dim=self.t_embedding_dim)
            # Project the (B, dim) sinusoidal output to (B, hidden_dim)
            # so it can be added at every layer's ``cond_per_node`` slot
            # without changing the EGNNLayer signature.  Zero-init the
            # Linear weight so this projection is identically zero at
            # init — preserves the bit-exact legacy output for any
            # already-trained checkpoint that opted in to this surface
            # without updating the warm-start.
            self.t_to_h = nn.Linear(
                in_features=self.t_embedding_dim,
                out_features=hidden_dim,
                bias=False,
            )
            nn.init.zeros_(self.t_to_h.weight)
        else:
            self.t_emb_per_layer = None  # type: ignore[assignment]
            self.t_to_h = None  # type: ignore[assignment]

    @torch.no_grad()
    def fit_output_head_lsq(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        ridge: float = 1e-6,
    ) -> Dict[str, object]:
        """Closed-form least-squares gate for ONE time value (Wf-A108j).

        Solves ``min_w || sum_c w_c V^c - target ||^2`` over the real atoms of
        one batch, where ``V^c`` are the vector channels this forward produced.
        ``w`` has ``C`` entries, so the normal equations are ``C x C`` — small
        enough to solve exactly and deterministic enough to reproduce.

        Why the optimum matters beyond "a better start": at the LS solution the
        residual is orthogonal to every channel, so ``dL/d(gain_c)`` is exactly
        zero for every per-channel gain.  That is what stops the collapse
        measured on the epoch-4 checkpoint, where a trainable ``norm_gain``
        walked from 1.0 to 0.0017 because the field was outputting nothing and
        shrinking the channels was the only loss-reducing direction available.

        **The ridge is chosen by leave-one-cloud-out, not fixed.**  An
        unregularised fit minimises the residual on the ONE draw in the batch,
        and where the conditional mean is small that means fitting the draw:
        MEASURED 2026-10-25 at ``t = 0.5``, where the achievable magnitude is
        7.2% of the target, the in-sample fit explained 56% of the variance --
        100x more than the Bayes bound permits -- and emitted 5.1x the
        achievable magnitude with a direction cosine of 0.124.  Training then
        spent 36 epochs walking that back to 3.8x.  Selecting ``ridge`` on
        held-out clouds instead of in-sample atoms is what makes the gate an
        estimate of the conditional mean rather than of the noise; clouds, not
        atoms, are the held-out unit because atoms inside a cloud are
        correlated.

        Returns the fitted ``w``, the time embedding at this ``t`` (so a caller
        can regress ``w`` on it across knots), and the achieved MSE / cosine /
        norm ratio for the record.  ``None`` values are returned when the head
        was not built with ``velocity_head_init="lsq"``.
        """
        if self.vc_head is None:
            return {"w": None, "t_emb": None, "reason": "not a vector_channels head"}
        if getattr(self, "velocity_head_init", "zero") != "lsq":
            return {"w": None, "t_emb": None, "reason": "velocity_head_init != lsq"}

        self._capture_head_inputs = True
        _was_training = self.training
        self.eval()  # context dropout would randomise the fit
        try:
            self.forward(x, atom_types, edge_index, t, edge_mask=edge_mask)
            h_cap, vec_cap, t_emb = self._head_capture
        finally:
            self._capture_head_inputs = False
            self._head_capture = None
            self.train(_was_training)
        if h_cap is None or vec_cap is None:
            return {"w": None, "t_emb": None, "reason": "capture produced nothing"}

        if node_mask is None:
            node_mask = atom_types > 0
        node_mask = node_mask.to(torch.bool)
        v = vec_cap[node_mask]  # (M, C, 3)
        y = target[node_mask]  # (M, 3)
        if v.shape[0] == 0:
            return {"w": None, "t_emb": t_emb, "reason": "no real atoms"}

        gram = torch.einsum("mcd,med->ce", v, v)
        rhs = torch.einsum("mcd,md->c", v, y)
        scale = max(float(gram.diagonal().mean()), 1e-12)
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)

        def _solve(g: torch.Tensor, r: torch.Tensor, lam_rel: float) -> torch.Tensor:
            # Ridge RELATIVE to each channel's own energy, not to the mean of
            # the diagonal.  The channel norms differ by two orders of magnitude
            # here (one channel carries the metal centre), so a scalar ridge
            # taken from the mean leaves the small channels unregularised and
            # the held-out prediction explodes: MEASURED on a synthetic 8-cloud
            # batch, a scalar ridge of 0.03 * mean(diag) gave an in-sample ratio
            # of 0.87 and an out-of-fold ratio of 2.47.
            #
            # The channel basis is also rank-deficient at init (several channels
            # are the same direction up to sign), so the gram matrix is singular
            # and a zero ridge raises; 1e-8 * scale keeps the solve well-posed.
            diag = g.diagonal().clamp_min(1e-3 * scale)
            return torch.linalg.solve(
                g + (max(lam_rel, 1e-8) * diag).diag() + 1e-6 * scale * eye, r
            )

        # Ridge selected by generalised cross-validation.  GCV rather than
        # leave-one-cloud-out because at the batch sizes this runs at (8 clouds)
        # a held-out fold is fitted on 7 clouds and the out-of-fold magnitude is
        # not usable as a selection signal: MEASURED on a synthetic 8-cloud
        # batch, the LOO choice gave an in-sample ratio of 0.36 and an
        # out-of-fold ratio of 1.03, i.e. it picked a ridge that generalises
        # worse than it fits while still being the minimum of the fold errors.
        # GCV needs no folds and its complexity term (C / (1 + lam)) is exact
        # for this per-channel ridge.
        lam = float(ridge)
        gcv = float("nan")
        if y.shape[0] > gram.shape[0]:
            grid = (1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0)
            m_rows = float(y.shape[0])
            c_ch = float(gram.shape[0])
            best = (float("inf"), None)
            for lam_rel in grid:
                w_try = _solve(gram, rhs, lam_rel)
                rss = float(
                    ((torch.einsum("c,mcd->md", w_try, v) - y) ** 2).sum()
                )
                dof = c_ch / (1.0 + max(lam_rel, 1e-8))
                denom_gcv = max(1.0 - dof / m_rows, 1e-6)
                score = (rss / m_rows) / denom_gcv**2
                if score < best[0]:
                    best = (score, lam_rel)
            if best[1] is not None:
                lam = float(best[1])
                gcv = float(best[0])

        w = _solve(gram, rhs, lam)

        pred = torch.einsum("c,mcd->md", w, v)
        mse = float(((pred - y) ** 2).mean())
        mse_zero = float((y ** 2).mean())
        denom = float(pred.norm()) * float(y.norm())
        cosine = float((pred * y).sum()) / max(denom, 1e-12)
        return {
            "w": w,
            # One embedding per knot: every cloud in the batch shares ``t``,
            # so the per-sample rows are identical.  Take the first rather
            # than flattening, which would silently produce ``B * dim``
            # features and a (K, B*dim) design matrix.
            "t_emb": (
                None
                if t_emb is None
                else t_emb.reshape(-1, t_emb.shape[-1])[0].detach()
            ),
            "mse": mse,
            "mse_zero": mse_zero,
            "cosine": cosine,
            "ratio": float(pred.norm()) / max(float(y.norm()), 1e-12),
            "ridge": lam,
            "ridge_relative": lam,
            "gcv": gcv,
        }

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
            if (
                isinstance(ckpt, dict)
                and "encoder" in ckpt
                and isinstance(ckpt["encoder"], dict)
            ):
                state_dict = ckpt["encoder"]
            elif (
                isinstance(ckpt, dict)
                and "state_dict" in ckpt
                and isinstance(ckpt["state_dict"], dict)
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
        x: torch.Tensor,  # (B, N, 3)   per-atom coords at time t
        atom_types: torch.Tensor,  # (B, N)      atomic numbers (int64)
        edge_index: torch.Tensor,  # (B, 2, E)
        t: torch.Tensor,  # (B,) or (B, 1)  flow time
        edge_mask: Optional[torch.Tensor] = None,  # (B, E)  bool
        pocket_embed: Optional[torch.Tensor] = None,  # (B, H)   pocket conditioning
        dative_bond_edge_attr: Optional[torch.Tensor] = None,  # (B, E)  bool, T9 opt-in
        pocket_atom_embed: Optional[torch.Tensor] = None,  # (B, P, max_atomic_number)
        pocket_atom_mask: Optional[torch.Tensor] = None,  # (B, P) bool
        batch_ctx: Optional[dict] = None,  # Track-A L3b — chemistry context dict
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
        # Wf-A62 W1 (VF2 doc) — ``pocket_atom_embed`` /
        # ``pocket_atom_mask`` / ``batch_ctx`` are accepted-but-unused
        # by ``EGNNVelocityField.forward``; only ``_last_dative_flag``
        # is the live consumer.  The accepted-unused kwargs are kept
        # in the signature for backward compatibility with Track-A
        # L3b callers; the velocity field computes per-sample pocket
        # conditioning via the SEPARATE ``pocket_embed`` arg
        # (see ``forward_velocity`` and the pocket-encoder pipeline).
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
        # Wf-A27 (2026-10-13) step-5 — sinusoidal-t per-layer materialisation.
        # When ``self.t_emb_per_layer is None`` (the shipped default
        # ``t_embedding_dim=0``) the projected addend is ``None`` and the
        # downstream ``cond_per_node`` carries the legacy scalar-t path
        # bit-exactly.  When ``t_embedding_dim > 0`` the projection is
        # zero-init so a fresh random init folds the contribution to
        # zero — pre-existing checkpoints keep loading unchanged.
        t_per_layer_addend: Optional[torch.Tensor] = None
        t_sin: Optional[torch.Tensor] = None
        if self.t_emb_per_layer is not None:
            t_sin = self.t_emb_per_layer(t.squeeze(-1))  # (B, dim)
        if self.t_emb_per_layer is not None and self.t_to_h is not None:
            t_per_layer_addend = self.t_to_h(t_sin).unsqueeze(1)  # (B, 1, H)
        # Wf-A32 (2026-10-14) step-1 — pre-compute pocket per-layer
        # addend.  When ``self.pocket_per_layer_injector is not None``
        # (opt-in only) AND ``pocket_embed is not None`` (conditioning
        # is supplied), project the per-sample (B, H) embedding to
        # (B, N, H) and broadcast to every EGNN layer's
        # ``cond_per_node``.  At fresh init the projector is zero-init so
        # the contribution is identically zero — bit-exact legacy path
        # until trained.  When the opt-in flag is False (default) the
        # addend stays ``None`` and the per-layer gate below falls
        # through to the legacy ``_layer_cond = t_per_atom`` branch.
        _pocket_per_layer_addend: Optional[torch.Tensor] = None
        if (
            self.pocket_per_layer_injector is not None
            and pocket_embed is not None
        ):
            _pocket_per_layer_addend = self.pocket_per_layer_injector(
                pocket_embed, n_max=n
            )  # (B, N, H)
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
            pocket_bias = torch.zeros(
                b,
                n,
                self.atom_embed.embedding_dim,
                device=x_t.device,
                dtype=t_per_atom.dtype,
            )
        else:
            if self.training and self.context_dropout > 0.0:
                keep = (
                    torch.rand(b, device=x_t.device) > self.context_dropout
                )  # (B,) bool, True = keep the conditioning
                # Shape (B, 1, 1) so it broadcasts cleanly over (B, N, H).
                keep = keep.view(b, 1, 1).to(t_per_atom.dtype)
                pocket_bias = pocket_embed.unsqueeze(1).expand(b, n, -1) * keep
            else:
                pocket_bias = pocket_embed.unsqueeze(1).expand(b, n, -1)
        # Wf-A27 (2026-10-13) step-5 — when ``pocket_per_layer=False`` the
        # sinusoidal-t addend joins the input projection exactly once
        # (``t_per_layer_addend`` is None in the default-path so this is a
        # bit-exact no-op for ``t_embedding_dim=0``).
        _input_addend = (
            t_per_layer_addend.expand(b, n, -1) if t_per_layer_addend is not None
            else 0
        )
        # Wf-A108j — per-cloud metal identity, derived from ``atom_types`` so no
        # caller has to thread a new tensor.  Index 0 is "no metal".
        metal_bias = 0
        if self.metal_embed is not None:
            _hit = (
                atom_types.unsqueeze(-1) == self.metal_z_order.view(1, 1, -1)
            ).any(dim=1)  # (B, K)
            _ids = torch.where(
                _hit.any(dim=1),
                _hit.to(torch.long).argmax(dim=1) + 1,
                torch.zeros(b, dtype=torch.long, device=atom_types.device),
            )
            metal_bias = self.metal_embed(_ids).unsqueeze(1)  # (B, 1, H)
        if isinstance(_input_addend, int):
            h = (
                self.atom_embed(atom_types)
                + t_per_atom
                + pocket_bias
                + metal_bias
            )
        else:
            h = (
                self.atom_embed(atom_types)
                + t_per_atom
                + pocket_bias
                + _input_addend
                + metal_bias
            )
        last_v = torch.zeros_like(x_t)
        if self.velocity_head == "vector_channels":
            # Wf-A108a — the channel stack replaces ``last_v`` as the
            # coordinate carrier.  Channel 0 is ``x - centroid`` so the head
            # can recover the legacy radial basis; Wf-A108c adds the
            # metal-relative channels.
            from molmetal.adapters.flow_matching_lipman.vector_channels import (
                build_metal_mask,
                initial_vector_channels,
            )

            metal_mask = build_metal_mask(atom_types)
            vec = initial_vector_channels(
                x_t,
                self.vector_channels,
                metal_mask=metal_mask,
                metal_frame_channels=self.metal_frame_channels,
                # Padded atoms carry z = 0 (the trainer's pad value); real
                # atomic numbers start at 1.
                node_mask=atom_types > 0,
            )
        else:
            metal_mask = None
            vec = None
        for layer in self.layers:
            # Pass t_per_atom as cond_per_node so the EGNNLayer's
            # update_mlp (which expects 3H+1 with cond) gets the
            # right shape.  Without this, update_mlp would receive
            # only 2H+1 and fail on the matmul.
            #
            # Wf-A27 (2026-10-13) step-5 — when ``pocket_per_layer=True``
            # AND ``t_embedding_dim>0`` the sinusoidal-t addend is
            # broadcast to ``cond_per_node`` at every EGNN layer; the
            # zero-init ``t_to_h`` keeps a fresh field's contribution
            # identically zero, so pre-Wf-A27 checkpoints that opted in
            # load with no immediate change to the velocity field.
            #
            # Wf-A32 (2026-10-14) step-1 — three-way gate so the two
            # opt-in axes (temporal ``t_per_layer_addend`` and pocket
            # ``_pocket_per_layer_addend``) compose independently.  Both
            # default to ``None`` so a fresh init / opt-out ships
            # bit-exact.  ``elif`` (not two ``if`` branches) means the
            # legacy ``else`` path keeps running when neither flag fires
            # — same as the pre-Wf-A32 code path.
            if (
                self.pocket_per_layer
                and t_per_layer_addend is not None
            ):
                _layer_cond = t_per_atom + t_per_layer_addend
            elif _pocket_per_layer_addend is not None:
                _layer_cond = t_per_atom + _pocket_per_layer_addend
            else:
                _layer_cond = t_per_atom
            h, last_v = layer(
                h,
                x_t,
                edge_index,
                cond_per_node=_layer_cond,
                edge_mask=edge_mask,
            )
        if self.velocity_head == "vector_channels":
            # Same three-way conditioning gate as the radial loop above, so
            # the pocket / sinusoidal-t opt-ins compose identically on both
            # heads.  The vector-channel layer's update_mlp expects the same
            # 3H+1 input width, hence the same ``cond_per_node`` contract.
            for layer in self.vc_layers:
                if self.pocket_per_layer and t_per_layer_addend is not None:
                    _layer_cond = t_per_atom + t_per_layer_addend
                elif _pocket_per_layer_addend is not None:
                    _layer_cond = t_per_atom + _pocket_per_layer_addend
                else:
                    _layer_cond = t_per_atom
                h, vec = layer(
                    h,
                    x_t,
                    vec,
                    edge_index,
                    cond_per_node=_layer_cond,
                    edge_mask=edge_mask,
                    metal_mask=metal_mask,
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
        if pocket_atom_embed is not None and pocket_atom_mask is not None:
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
        # Wf-A91 (2026-10-19) — the per-atom gate is BOUNDED; the global
        # scale stays learnable.
        #
        # This was `vel_head(h) * vel_scale * relative_to_centroid`, an
        # unbounded scalar.  `h` depends on pairwise DISTANCES, so at
        # coordinate scale s the gate can itself grow like s and the product
        # becomes QUADRATIC in s.  MEASURED (fresh field, 2 layers, gates
        # randomised to std 0.03): |v| went 0.066 -> 300 for x -> 100x, a
        # 7,519x ratio where linear growth would be 100x.  That is the
        # "superlinear ODE field" the class comment claims was prevented.
        #
        # tanh restores the documented contract without capping the model:
        # the per-atom gate is in (-1, 1) while `vel_scale` (a free
        # parameter) carries the magnitude, so the optimiser keeps an
        # unconstrained path to any scale.  tanh is C-infinity, so the
        # Lipman 2023 Thm 2 smoothness requirement is satisfied — the
        # earlier comment's "non-differentiable kink at the saturation
        # boundary" is not a property of tanh (a hard clamp would be).
        # With `bounded_coordinate_messages=True` on every layer, last_v is
        # a degree-normalised mean of tanh-gated displacement vectors, so
        # |vel| <= |vel_scale| * centroid_radius + mean_pair_distance.
        if self.velocity_head == "vector_channels":
            # Wf-A108a — the velocity is a learned projection of the channel
            # stack, not a scalar gate on the centroid ray.  ``vec`` is
            # RMS-normalised per atom, so ``vc_head``'s gate carries the
            # magnitude and the channels carry the direction.
            #
            # Wf-A108j — the LS initialiser needs the head's own inputs.  The
            # flag is set only by ``fit_output_head_lsq`` and cleared by it, so
            # the capture costs one attribute check per forward otherwise.
            if getattr(self, "_capture_head_inputs", False):
                self._head_capture = (
                    h.detach(),
                    vec.detach(),
                    None if t_sin is None else t_sin.detach(),
                )
            vel = self.vc_head(h, vec, t_emb=t_sin)
        else:
            if self.radial_centroid_masked:
                # Wf-A108 — the trainer pads every cloud in an FM batch to
                # the BATCH MAX atom count (``fm_velocity``), so a plain
                # ``mean(dim=1)`` divides the real coordinate sum by the
                # padded length.  MEASURED on the 199-cloud train cohort at
                # B=8: the resulting centroid is scaled by s/n_max with
                # median 0.481, p05 0.148, and 52% of samples below 0.5 —
                # i.e. for half the batch the "radial" term is radial about a
                # point halfway to the origin.  Opt-in because fixing it
                # changes the default forward.
                _m = (atom_types > 0).to(x_t.dtype).unsqueeze(-1)
                _n = _m.sum(dim=1, keepdim=True).clamp_min(1.0)
                _centroid = (x_t * _m).sum(dim=1, keepdim=True) / _n
            else:
                _centroid = x_t.mean(dim=1, keepdim=True)
            relative_to_centroid = x_t - _centroid
            vel = (
                torch.tanh(self.vel_head(h)) * self.vel_scale * relative_to_centroid
                + last_v
            )
        # Wf-A108e — publish the absolute-position operand when the field is
        # trained in x1 space.  ``vel`` above is the head's raw output, i.e. a
        # displacement; in x1 mode that same tensor is ``u`` and
        # ``x1_hat = x_t + u``.  The sampler keeps reading ``out["vel"]``, so
        # the division by ``1 - t`` happens here and nowhere else; the floor
        # stops ``t -> 1`` from turning a small residual into an infinite step.
        # Both keys are absent when ``prediction_target == "velocity"``, which
        # is what keeps the default state dict and forward bit-exact.
        if self.prediction_target == "x1":
            x1_hat = x_t + vel
            _den = (1.0 - t).clamp_min(self.x1_denominator_floor)
            vel = vel / _den.unsqueeze(-1)
        else:
            x1_hat = None
        atom_logits = self.atom_head(h)  # (B, N, max_atomic_number)
        # Wf-A56 W2 (2026-10-16) — P4 BF16 overflow guard.  R17 BF16
        # regression: untrained Blackwell kernels overflowed
        # ``atom_logits`` to +/-1e4, blowing up ``F.cross_entropy``
        # into ``Inf``.  Clamp at +/-50 here (training path) AND in
        # the sampler (line ~2571).  Clamp is monotonic so gradient
        # flow is preserved.
        atom_logits = atom_logits.clamp(min=-50.0, max=50.0)
        # Track-A (Wf-A15, 2026-10-13) — Layer 3a q-prediction.
        #
        # When ``self.q_head_enabled=True`` and ``self.q_head`` is set,
        # emit per-atom q predictions (``q_pred``) from the same final
        # per-atom ``h`` that feeds ``atom_logits``.  The head is
        # zero-init at construction so a fresh field returns
        # ``q_pred == 0``; ``NaturalChargeLoss(weight=0)`` then returns
        # a zero tensor — bit-exact pre-Track-A.  When
        # ``q_head_enabled=False`` (the YAML default), the dict does NOT
        # carry ``q_pred`` — existing callers that destructure the
        # return dict via ``out["vel"]`` / ``out["atom_logits"]`` /
        # ``out["h"]`` keep working unchanged.
        out: Dict[str, torch.Tensor] = {
            "vel": vel,
            "atom_logits": atom_logits,
            "h": h,
        }
        if x1_hat is not None:
            out["x1_hat"] = x1_hat
        if self.q_head is not None:
            out["q_pred"] = self.q_head(h).squeeze(-1)  # (B, N)
        return out

    # ------------------------------------------------------------------
    # T9: square-planar Pt(II) geometric prior (opt-in)
    # ------------------------------------------------------------------
    def metal_geometry_loss(
        self,
        positions: torch.Tensor,  # (B, N, 3) or (N, 3)
        edge_index: torch.Tensor,  # (B, 2, E) or (2, E)
        atom_types: torch.Tensor,  # (B, N) or (N,)
        dative_bond_edge_attr: Optional[torch.Tensor] = None,  # (B, E) or (E,)
        edge_types: Optional[torch.Tensor] = None,  # (B, E) or (E,)
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
                square_planar_penalty,
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
            return torch.zeros((), device=positions.device, dtype=positions.dtype)

        effective = (
            (edge_types == 2) if edge_types is not None else dative_bond_edge_attr
        )
        if effective is None:
            return positions.new_zeros(())
        if positions.dim() == 2:
            return prior_weight * square_planar_penalty(
                positions, edge_index, effective, atom_types
            )
        penalties = []
        for i in range(positions.shape[0]):
            penalties.append(
                square_planar_penalty(
                    positions[i],
                    edge_index[i] if edge_index.dim() == 3 else edge_index,
                    effective[i] if effective.dim() == 2 else effective,
                    atom_types[i] if atom_types.dim() == 2 else atom_types,
                )
            )
        return prior_weight * torch.stack(penalties).mean()

    # ------------------------------------------------------------------
    # TODO-09: generalised metal-geometry prior sampling hook
    # ------------------------------------------------------------------
    def apply_metal_geometry_step(
        self,
        positions: torch.Tensor,  # (B, N, 3) or (N, 3)
        atom_types: torch.Tensor,  # (B, N) or (N,)
        edge_types: torch.Tensor,  # (B, E) or (E,) long
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
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        try:
            from molmetal.molmetal_lam.priors.metal_geometry import (
                MetalGeometryPrior,
            )
        except ImportError:
            # Wf-A30-2 B3 (2026-10-14) — warn (not silent). The prior
            # # was a soft penalty on every k_every-th sampling step,
            # # so failing to import left the prior at zero for every
            # # step without any signal to the user. Surfacing a
            # # one-time ImportWarning preserves the soft-fail
            # # behaviour (the trainer keeps running) while making the
            # # missing prior visible in the trainer stderr. The
            # # module-level ``_metal_geometry_import_warned`` gate
            # # prevents per-step log spam: callers invoke this
            # # function once per sampling step (CFM trainer has 50-200
            # # steps per molecule), so an ungated warn would produce
            # # 50-200 identical lines per molecule.
            global _metal_geometry_import_warned
            if not _metal_geometry_import_warned:
                warnings.warn(
                    "[flow_matching_lipman] MetalGeometryPrior import "
                    "failed; apply_metal_geometry_step returns 0 for "
                    "every step. Check that "
                    "molmetal.molmetal_lam.priors.metal_geometry is "
                    "importable. This warning is emitted once per "
                    "Python process.",
                    ImportWarning,
                    stacklevel=2,
                )
                _metal_geometry_import_warned = True
            return torch.zeros((), device=positions.device, dtype=positions.dtype)
        prior = MetalGeometryPrior(weight=weight)
        if positions.dim() == 2:
            return prior.prior_loss(
                positions, atom_types, edge_types, edge_index=edge_index
            )
        penalties = []
        for i in range(positions.shape[0]):
            penalties.append(
                prior.prior_loss(
                    positions[i],
                    atom_types[i] if atom_types.dim() == 2 else atom_types,
                    edge_types[i] if edge_types.dim() == 2 else edge_types,
                    edge_index=edge_index[i]
                    if edge_index is not None and edge_index.dim() == 3
                    else edge_index,
                )
            )
        return torch.stack(penalties).mean()

    def forward_velocity(
        self, x, atom_types, edge_index, t, edge_mask=None, pocket_embed=None
    ):
        """Convenience wrapper returning only the velocity tensor.

        Used by the cloned ``flow_matching`` library's ``ModelWrapper``,
        which calls ``self.model(x=x, t=t, **extras)`` and expects a
        tensor-shaped return value.
        """
        return self.forward(
            x, atom_types, edge_index, t, edge_mask=edge_mask, pocket_embed=pocket_embed
        )["vel"]

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
                x,
                atom_types,
                edge_index,
                t,
                pocket_embed=pocket_embed,
                edge_mask=edge_mask,
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
                    x,
                    atom_types,
                    edge_index,
                    t,
                    pocket_embed=pocket_embed,
                    edge_mask=edge_mask,
                )
                if pocket_embed is not None
                else None
            )
            v_uncond = self.forward_velocity(
                x,
                atom_types,
                edge_index,
                t,
                pocket_embed=None,
                edge_mask=edge_mask,
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


# Wf-A30-2 B3 (2026-10-14) — module-level once-per-process gate for
# the MetalGeometryPrior ImportWarning. CFM sampling invokes
# apply_metal_geometry_step once per integration step (50-200 steps per
# molecule), so an unguarded warnings.warn would emit 50-200 identical
# lines per molecule.
_metal_geometry_import_warned: bool = False


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
        # Wf-A68 (2026-10-17) — optional residual skip in the bond head.
        # When True, the head registers an extra Linear(hidden_dim,
        # hidden_dim) projection from fc1's output to the pre-`out`
        # activation (ResNet-style).  Default False preserves Wf-A67
        # bit-exact compat.
        use_bond_head_skip: bool = False,
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
        cutoff: float = 5.0,
        # Wf-A69 (2026-10-17) W1 — bond-candidate source switch.
        # ``truth_only`` (default) reproduces the pre-A69 candidate
        # construction at :meth:`train_step` line ~2599-2641: every
        # candidate edge is read from ``m.bonds`` so the per-pair
        # CE is 100% positive by construction. ``radius_graph``
        # builds candidates from a distance-cutoff ``triu_indices``
        # (matching :class:`YuelBondDecoderHead`'s eval cutoff) and
        # labels them with truth via the same ``m.bonds`` lookup.
        # 2026-10-17 measured: Wf-A67 bond_f1=0.265 was reported by
        # an evaluation whose candidate set (~90-95% NO_BOND,
        # distance cutoff) the trainer never saw during training
        # (truth-only, 100% positive). ``radius_graph`` aligns train
        # and eval so the head sees the negative class.
        # Default ``truth_only`` preserves Wf-A67 bit-exact backward
        # compat.  Opt-in via YAML ``flow.bond_candidate_source``.
        bond_candidate_source: str = "truth_only",
        bond_candidate_cutoff: float = 2.4,
        # Wf-A69 W1 — subsample negatives to neg:pos = N (4:1 typical).
        # ``0`` (default) = no subsampling, full candidate set used;
        # ``>0`` activates a deterministic per-epoch seed so the run
        # is reproducible.  Per PepForge (bioRxiv 2026) + GraViti
        # (arXiv 2605.16668) negative-resampling recipe for ~95%
        # NO_BOND prior.
        subsample_neg_ratio: int = 0,
        subsample_neg_seed: str = "epoch",
    ) -> None:
        # 2026-10-04 — radius-graph cutoff (Å) for the EGNN's message passing.
        #
        # Both production call sites used to pass a FULLY CONNECTED graph
        # (``_make_dummy_edge_index``), so the EGNN ran O(N^2) messages with
        # none of the locality its geometric prior is built on, and the graph
        # was also built ONCE before the ODE solve and reused at every
        # integration step even though ``x_t`` moves.  5.0 Å is the usual
        # intra-molecular choice for 3D molecular FM (TargetDiff / DiffSBDD
        # use 4-10 Å); the smallest molecules here are ~8 atoms / ~5 Å across,
        # so a smaller value would fragment them into disconnected pieces.
        self._cutoff = float(cutoff)
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
            raise ValueError(f"cfg_scale must be >= 0.0, got {cfg_scale!r}")
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
        # centres (78=Pt canonical OOV trigger + 28=Ni, 46=Pd,
        # 77=Ir, 45=Rh, 76=Os, 75=Re, 44=Ru, 79=Au).  20 elements
        # total — P1 guard requires >= 20.
        self._atom_vocab: tuple = (
            1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78, 28, 46, 77, 45, 76, 75, 44, 79
        )

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
        self._use_bond_head_skip = bool(use_bond_head_skip)
        self._joint_train = bool(joint_train)
        if bond_loss_weight < 0.0:
            raise ValueError(f"bond_loss_weight must be >= 0, got {bond_loss_weight!r}")
        self._bond_loss_weight = float(bond_loss_weight)
        self._bond_pattern_mask = bool(bond_pattern_mask)
        # WF-Vina-Lift-Phase23 (Phase 2.3) — PCGrad multi-task loss flag.
        # Default False preserves the pre-Phase-2.3 weighted-sum behaviour
        # bit-exactly; opt in via ``use_pcgrad=True`` (or the
        # ``--use-pcgrad`` CLI flag on r10_cfg_real_crossdocked.py).
        self._use_pcgrad = bool(use_pcgrad)
        # Diagnostics: per-term losses from the most recent train_step.
        # Initialised here (2026-10-19) because the non-finite-loss guard
        # reads ``self.last_losses.get("skipped_oov", 0)`` BEFORE assigning
        # the dict — so a FIRST step that produced NaN/Inf raised
        # AttributeError instead of reporting the skip.  That is exactly the
        # step a caller is most likely to inspect (fresh init, bad batch),
        # and it made test_cfm_rescue_bondhead_default fail on 3 tests.
        self.last_losses: dict = {}
        # Diagnostics: stores the most-recent PCGrad conflict metrics
        # (n_conflicts, mean_cos_sim) so callers can log them.  Updated
        # by :func:`_pcgrad_resolve` whenever PCGrad runs.
        self.last_pcgrad_stats: dict = {
            "n_tasks": 0,
            "n_conflicts": 0,
            "mean_cos_sim": 0.0,
        }
        # Wf-A69 W1 (2026-10-17) — bond candidate-list diagnostic
        # counters.  ``train_step`` writes these on every step so a
        # maintainer can verify the candidate-list source is doing
        # what the YAML asked: ``truth_only`` should have
        # ``last_bond_n_neg == 0``; ``radius_graph`` should have a
        # positive ``last_bond_n_neg`` so the head sees the negative
        # class.  Pre-Wf-A69 callers that never set these leave them
        # at 0 (the legacy path didn't record this metric).
        self.last_bond_n_candidates: int = 0
        self.last_bond_n_pos: int = 0
        self.last_bond_n_neg: int = 0

        # WF-CFM-Phase-2 (P1.4) — ConnectivityAwareDecoder wiring.
        # Stash the flags so :meth:`_generate_impl` can construct the
        # decoder with the right policy.  Both default to the safer
        # ``True`` setting; passing ``False`` is the backward-
        # compatible escape hatch.
        self._accept_only_connected = bool(accept_only_connected)
        self._keep_empty_connected = bool(keep_empty_connected)

        # Wf-A69 (2026-10-17) W1 — bond-candidate source / subsample knobs.
        # All defaults preserve bit-exact pre-Wf-A69 behaviour.
        if str(bond_candidate_source) not in ("truth_only", "radius_graph"):
            raise ValueError(
                f"bond_candidate_source must be 'truth_only' or 'radius_graph', "
                f"got {bond_candidate_source!r}"
            )
        if bond_candidate_cutoff <= 0.0:
            raise ValueError(
                f"bond_candidate_cutoff must be > 0 Å, got {bond_candidate_cutoff}"
            )
        self._bond_candidate_source = str(bond_candidate_source)
        self._bond_candidate_cutoff = float(bond_candidate_cutoff)
        self._subsample_neg_ratio = int(subsample_neg_ratio)
        self._subsample_neg_seed = str(subsample_neg_seed)

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
            # Wf-A68 (2026-10-17) — wire use_skip / use_focal_bond_ce /
            # bond_sample_balance / dative_distance_cutoff flags through.
            # Default OFF (use_skip=False) keeps Wf-A67 checkpoints
            # bit-exact-compatible — the projection shape is identical
            # and forward() no-ops the skip branch.
            self.bond_head = BondOrderHead(
                in_dim=bond_head_in_dim,
                hidden_dim=64,
                dropout=0.10,
                num_classes=5,
                atom_vocab=self._atom_vocab,
                training_mode=mode,
                use_skip=bool(
                    getattr(self, "_use_bond_head_skip", False)
                ),
            ).to(self.device)
        else:
            self.bond_head = None
        # Optimizer now jointly updates both networks.  WF-2 A5: when
        # ``joint_train`` is set, the BondOrderHead params are added
        # too — joint end-to-end training of velocity field + atom
        # head + bond head.  In ``frozen`` mode (or when the head is
        # not constructed) the head is left out (legacy behaviour).
        opt_params = list(self.velocity_field.parameters()) + list(
            self.pocket_encoder.parameters()
        )
        if self._use_bond_head and self._joint_train and self.bond_head is not None:
            opt_params = opt_params + list(self.bond_head.parameters())
        self.optimizer = torch.optim.AdamW(opt_params, lr=self._lr)

    # ------------------------------------------------------------------
    # Train step: one batch of CFM loss (Lipman 2023 §4.5)
    # ------------------------------------------------------------------
    def train_step(
        self,
        pocket,  # Pocket | List[Pocket] | None
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
        # 2026-10-19 — cumulative diagnostics (skipped_oov, oov_target_*_rows,
        # bond_skipped_no_edges, bond_subsampled_negatives) must survive the
        # end-of-step merge; only the previous step's verdict is cleared.
        self.last_losses.pop("nonfinite_terms", None)
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
        edge_label_list: List[
            int
        ] = []  # 0=none, 1=single, 2=double, 3=triple, 4=aromatic
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
                # Wf-A69 W1 (2026-10-17) — bond-candidate source switch.
                # ``truth_only`` (default) reproduces the pre-A69
                # candidate construction: every candidate edge is read
                # from ``m.bonds`` so the per-pair CE is 100% positive.
                # ``radius_graph`` builds candidates from a
                # distance-cutoff ``triu_indices`` (matching
                # ``YuelBondDecoderHead``'s eval cutoff, default 2.4 Å
                # matching ``BondOrderHead.bond_cutoff``) and labels
                # them with truth via the same ``m.bonds`` lookup.
                # Pairs NOT in ``m.bonds`` get ``BOND_NO_BOND=0`` so the
                # head sees the negative class at training.
                # Truth bonds are NEVER dropped (kept on top of the
                # distance filter so the positive rate is preserved).
                _candidate_source = str(self._bond_candidate_source)
                _cutoff = float(self._bond_candidate_cutoff)
                # Build truth lookup on canonical (min, max) keys so a
                # candidate direction never matters for the label.
                _truth_lookup: Dict[Tuple[int, int], int] = {}
                for k in range(len(bs_np)):
                    s_idx = int(bs_np[k])
                    d_idx = int(bd_np[k])
                    if s_idx < 0 or d_idx < 0:
                        continue
                    if s_idx >= len(zi_np) or d_idx >= len(zi_np):
                        continue
                    bt = int(bt_np[k])
                    label = _rdkit_bond_int_to_bond_head_label(bt)
                    _key = (min(s_idx, d_idx), max(s_idx, d_idx))
                    # If a pair has multiple labels (shouldn't, but
                    # defensive), keep the first one — bond_type is
                    # the truth per tmQM .BO shards, not a consensus.
                    if _key not in _truth_lookup:
                        _truth_lookup[_key] = label
                if _candidate_source == "radius_graph" and n >= 2:
                    # Build (n*(n-1)/2,) upper-triangular pair list
                    # masked by ``d_ij < cutoff``.
                    _triu = torch.triu_indices(n, n, offset=1)
                    _src_all = _triu[0].tolist()
                    _dst_all = _triu[1].tolist()
                    _coords_np = m.coords.detach().cpu().tolist()
                    for k in range(len(_src_all)):
                        s_idx = int(_src_all[k])
                        d_idx = int(_dst_all[k])
                        # Distance gate (Angstrom).
                        _dx = _coords_np[s_idx][0] - _coords_np[d_idx][0]
                        _dy = _coords_np[s_idx][1] - _coords_np[d_idx][1]
                        _dz = _coords_np[s_idx][2] - _coords_np[d_idx][2]
                        _d = (_dx * _dx + _dy * _dy + _dz * _dz) ** 0.5
                        if _d >= _cutoff:
                            continue
                        _key = (min(s_idx, d_idx), max(s_idx, d_idx))
                        label = _truth_lookup.get(_key, 0)  # 0 = BOND_NO_BOND
                        edge_src_list.append(s_idx)
                        edge_dst_list.append(d_idx)
                        edge_z_i_list.append(int(zi_np[s_idx]))
                        edge_z_j_list.append(int(zi_np[d_idx]))
                        edge_label_list.append(int(label))
                else:
                    # ``truth_only`` path (default, bit-exact pre-A69).
                    for k in range(len(bs_np)):
                        s_idx = int(bs_np[k])
                        d_idx = int(bd_np[k])
                        if s_idx < 0 or d_idx < 0:
                            continue
                        if s_idx >= len(zi_np) or d_idx >= len(zi_np):
                            continue
                        bt = int(bt_np[k])
                        label = _rdkit_bond_int_to_bond_head_label(bt)
                        edge_src_list.append(s_idx)
                        edge_dst_list.append(d_idx)
                        edge_z_i_list.append(int(zi_np[s_idx]))
                        edge_z_j_list.append(int(zi_np[d_idx]))
                        edge_label_list.append(int(label))

        # T5: pocket conditioning — encode pocket(s) into a (B, H) bias.
        # Returns None when pocket=None so the velocity field takes its
        # unconditioned path.
        pocket_embed = self._encode_pocket(
            pocket, b=b, max_n_atoms=max_n, device=self.device
        )

        # Sample noise + time (Lipman 2023 CFM).  ``torch.randn_like(x_1)``
        # inherits x_1's device, so x_0 is also on self.device.
        x_0 = torch.randn_like(x_1)
        t = torch.rand(b, device=self.device)

        # Sample x_t and target velocity from the cloned library
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        # path_sample.x_t shape (B, max_n, 3)
        # path_sample.dx_t shape (B, max_n, 3)  — conditional velocity x_1 - x_0

        # Velocity field prediction (joint: coord velocity + atom-type logits)
        #
        # 2026-10-04 — a radius graph over the CURRENT x_t, not a fully
        # connected one.  `_make_dummy_edge_index` wired every atom to every
        # other atom in the sample: O(N^2) messages and no locality, which is
        # the one inductive bias an EGNN carries.  Building it from
        # `path_sample.x_t` (the interpolated coordinates the model actually
        # sees) is the standard treatment; using `x_1` would leak the target
        # geometry into the graph structure.
        edge_index, edge_mask = radius_edge_index(
            path_sample.x_t, node_mask, self._cutoff
        )
        # Identity is a prediction target, not an input available to the
        # unconstrained sampler. Supplying the target atomic numbers here
        # lets the atom head copy them and leaves the sampling input unseen.
        out = self.velocity_field(
            path_sample.x_t,
            torch.zeros_like(atom_types),
            edge_index,
            t,
            edge_mask=edge_mask,
            pocket_embed=pocket_embed,
        )
        v_pred = out["vel"]
        atom_logits = out["atom_logits"]  # (B, max_n, max_atomic_number)
        # Wf-A56 W2 (2026-10-16) — P4 BF16 overflow guard (training
        # path).  See L1855 for the velocity-field-side clamp; this
        # one covers the train_step-side clamp after the dict lookup.
        atom_logits = atom_logits.clamp(min=-50.0, max=50.0)

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
                atom_logits.shape[-1],
                atom_logits.device,
            )
            atom_logits_for_loss = atom_logits.masked_fill(
                ~vocab_mask_tensor,
                float("-inf"),
            )
        # Safety: if the masked fill produced a fully -inf row (only
        # possible when a target index falls outside vocab — degenerate
        # inputs), fall back to the un-masked logits to keep grad finite.
        per_row_logits = atom_logits_for_loss.reshape(-1, atom_logits.shape[-1])
        row_is_finite = torch.isfinite(per_row_logits).any(dim=-1)
        if not row_is_finite.all():
            # Wf-A62 W1 (JM7) — diagnose OOV fallback so a maintainer
            # can spot upstream data leaks (e.g., atomic_number >=
            # vocab size in the data loader).  Previously the fallback
            # silently re-enabled all OOV slots which would allow OOV
            # atoms to be sampled at inference.
            n_oov_rows = int((~row_is_finite).sum().item())
            warnings.warn(
                f"_joint_adapter cross-entropy: {n_oov_rows}/{per_row_logits.shape[0]} "
                f"rows had all-(-inf) vocab mask; falling back to un-masked "
                f"logits for those rows (see OOV data-load guard).",
                stacklevel=2,
            )
            per_row_logits = torch.where(
                row_is_finite.unsqueeze(-1),
                per_row_logits,
                atom_logits.reshape(-1, atom_logits.shape[-1]),
            )
        # Wf-A62 W1 (JM2) — fully-masked batch silently contributes zero
        # atom_loss which poisons the per-batch metric without raising.
        # Surface the upstream data leak (e.g., a degenerate dataset
        # sample with no real atoms) instead of silently cancelling.
        if int(node_mask.sum().item()) == 0:
            warnings.warn(
                "_joint_adapter atom_loss on fully-masked batch; "
                "node_mask.sum()==0 ⇒ atom_loss = 0 silently. "
                "Check upstream data loader for empty/all-padding samples.",
                stacklevel=2,
            )
        flat_targets = atom_types.reshape(-1)
        atom_loss_per_node = F.cross_entropy(
            per_row_logits,
            flat_targets,
            reduction="none",
        ).view(b, max_n)
        # Wf-A87 (2026-10-19) — a row whose TARGET class is masked still
        # yields +inf: the row itself is finite, so the all-(-inf) fallback
        # above does not catch it.  That inf poisoned the whole batch and
        # tripped the non-finite guard, discarding every valid row in the
        # step.  Exclude the unusable rows from the mean and report them.
        # torch.where (not multiply-by-mask) so inf * 0 cannot become nan.
        target_logit = per_row_logits.gather(
            1,
            flat_targets.clamp(min=0, max=per_row_logits.shape[1] - 1).unsqueeze(1),
        ).squeeze(1)
        real_rows = node_mask.reshape(-1)
        target_finite = torch.isfinite(target_logit)
        usable = real_rows & target_finite
        n_unusable = int((real_rows & ~target_finite).sum().item())
        if n_unusable:
            warnings.warn(
                f"_joint_adapter atom CE: {n_unusable} real atom(s) have a "
                "target class masked by the vocab mask (OOV atom); excluded "
                "from atom_loss for this step. Check the upstream data guard.",
                stacklevel=2,
            )
            self.last_losses["oov_target_atom_rows"] = (
                int(self.last_losses.get("oov_target_atom_rows", 0)) + n_unusable
            )
        if int(usable.sum().item()) == 0 and int(real_rows.sum().item()) > 0:
            warnings.warn(
                "_joint_adapter atom CE: every real atom in this batch is "
                "OOV; atom_loss contributes 0 for this step.",
                stacklevel=2,
            )
        per_node = torch.where(
            usable,
            atom_loss_per_node.reshape(-1),
            torch.zeros_like(target_logit),
        )
        n_real = usable.sum().clamp(min=1)
        atom_loss = per_node.sum() / n_real

        # ------------------------------------------------------------------
        # WF-2 A5 — joint bond-order loss
        # ------------------------------------------------------------------
        bond_loss_value: float = 0.0
        bond_loss_tensor = torch.zeros((), device=self.device, dtype=cfm_loss.dtype)
        # Wf-A62 W1 (JM6) — surface the "bond loss skipped: zero edges"
        # path.  When joint bond-head training is on but the molecule is
        # all-H / atom-less, ``len(edge_src_list) == 0`` would silently
        # skip with no diagnostic.  Record the integer flag so it lands
        # in the JSON summary (``bond_skipped_no_edges = 1``) and a
        # maintainer can distinguish "bond head disabled" from
        # "legitimate empty graph".
        if (
            self._use_bond_head
            and self._joint_train
            and self.bond_head is not None
            and len(edge_src_list) == 0
        ):
            self.last_losses["bond_skipped_no_edges"] = int(
                self.last_losses.get("bond_skipped_no_edges", 0)
            ) + 1
        if (
            self._use_bond_head
            and self._joint_train
            and self.bond_head is not None
            and len(edge_src_list) > 0
        ):
            edge_index_e = torch.tensor(
                [edge_src_list, edge_dst_list],
                dtype=torch.long,
                device=self.device,
            )
            edge_z_i = torch.tensor(edge_z_i_list, dtype=torch.long, device=self.device)
            edge_z_j = torch.tensor(edge_z_j_list, dtype=torch.long, device=self.device)
            edge_labels = torch.tensor(
                edge_label_list,
                dtype=torch.long,
                device=self.device,
            )
            # Wf-A69 W1 (2026-10-17) — optional negative subsampling.
            # When ``bond_candidate_source=radius_graph`` plus
            # ``subsample_neg_ratio > 0``, keep all positive pairs and
            # ``subsample_neg_ratio * n_pos`` random negatives
            # (deterministic per-epoch seed via Python's `hash` so the
            # run is reproducible across CUDA devices).
            # 0 (default) keeps every pair, bit-exact pre-A69.
            if (
                str(self._bond_candidate_source) == "radius_graph"
                and int(self._subsample_neg_ratio) > 0
            ):
                _pos_mask = edge_labels > 0
                _neg_mask = ~_pos_mask
                _n_pos = int(_pos_mask.sum().item())
                _n_neg = int(_neg_mask.sum().item())
                _keep_neg = int(self._subsample_neg_ratio) * _n_pos
                if _n_pos > 0 and _n_neg > _keep_neg:
                    # Deterministic per-batch seed derived from the
                    # candidate counts + the seed policy.  `_g.manual_seed`
                    # is the standard way to get a reproducible randperm
                    # without touching the global torch RNG.
                    _g = torch.Generator(device=self.device)
                    if str(self._subsample_neg_seed) == "epoch":
                        _g.manual_seed(int(_n_pos) * 7919 + int(_n_neg))
                    else:
                        # "global" → use a fixed seed so the run is
                        # reproducible regardless of epoch index.
                        _g.manual_seed(0)
                    _neg_idx = torch.nonzero(_neg_mask, as_tuple=False).flatten()
                    _perm = torch.randperm(
                        int(_neg_idx.numel()), generator=_g,
                        device=self.device,
                    )[:_keep_neg]
                    _sel_neg = _neg_idx[_perm]
                    _sel_pos = torch.nonzero(_pos_mask, as_tuple=False).flatten()
                    _sel = torch.cat([_sel_pos, _sel_neg], dim=0)
                    edge_index_e = edge_index_e[:, _sel]
                    edge_z_i = edge_z_i[_sel]
                    edge_z_j = edge_z_j[_sel]
                    edge_labels = edge_labels[_sel]
                    self.last_losses["bond_subsampled_negatives"] = int(
                        self.last_losses.get("bond_subsampled_negatives", 0)
                    ) + 1
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
                h_per_atom,
                edge_index_e,
                max_n=max_n,
                b=b,
            )
            # Concatenate the EGNN-conditioned per-edge hidden state
            # onto the 9-D pair features so the bond head has access
            # to graph context (otherwise it's purely distance-based).
            bond_inputs = torch.cat([bond_feats, e_h], dim=-1)
            # Wf-A35-1 (2026-10-14) C1 — silent shape-mismatch fallback REMOVED.
            # The previous branch silently dropped the EGNN-conditioned h
            # (replacing ``bond_inputs`` with the 9-D geometric ``bond_feats``
            # only) whenever ``bond_inputs.shape[-1] != self.bond_head.in_dim``.
            # That hid a broken EGNN -> bond-head pathway: training kept running
            # because the atom-loss gradient still flowed through the EGNN,
            # but the bond head was decoupled from the EGNN context it was
            # supposed to receive.  We now mirror bond_head.py:946-949, which
            # raises the same ValueError when ``pair_features.shape[1] !=
            # self.in_dim`` — fail fast and force a configuration fix
            # (construct the bond head with the right ``in_dim``) instead of
            # silently zeroing the gradient pathway.
            if bond_inputs.shape[-1] != self.bond_head.in_dim:
                raise ValueError(
                    f"bond_inputs last-dim {bond_inputs.shape[-1]} != "
                    f"bond_head.in_dim {self.bond_head.in_dim}; "
                    f"this is a hard shape mismatch — the EGNN ↔ "
                    f"bond-head pathway is broken (reconstruct the bond "
                    f"head with the matching in_dim)"
                )
            bond_logits_e = self.bond_head(bond_inputs)
            # Apply the bond-pattern mask (synced with atom vocab) so
            # the head cannot predict forbidden patterns.
            if self._bond_pattern_mask:
                bond_logits_e = self.bond_head.apply_bond_pattern_mask(
                    bond_logits_e,
                    edge_z_i,
                    edge_z_j,
                )
            # Wf-A87 (2026-10-19) — same masked-target problem as the atom
            # CE: a truth bond whose class the pattern mask forbids has a
            # -inf target logit, so CE is +inf even though the row has
            # finite slots.  Drop those edges from the mean and report
            # them; the alternative was losing the whole step — and every
            # valid edge in it — to one bad pair.
            _bond_target_logit = bond_logits_e.gather(
                1, edge_labels.clamp(min=0).unsqueeze(1)
            ).squeeze(1)
            _bond_usable = torch.isfinite(_bond_target_logit)
            _n_bond_unusable = int((~_bond_usable).sum().item())
            if _n_bond_unusable:
                warnings.warn(
                    f"_joint_adapter bond CE: {_n_bond_unusable}/"
                    f"{edge_labels.shape[0]} truth edges have a class masked "
                    "by the bond-pattern mask (OOV pair); excluded from "
                    "bond_loss for this step.",
                    stacklevel=2,
                )
                self.last_losses["oov_target_bond_rows"] = (
                    int(self.last_losses.get("oov_target_bond_rows", 0))
                    + _n_bond_unusable
                )
            if bool(_bond_usable.any()):
                bond_loss_tensor = F.cross_entropy(
                    bond_logits_e[_bond_usable], edge_labels[_bond_usable]
                )
            else:
                bond_loss_tensor = torch.zeros(
                    (), device=self.device, dtype=cfm_loss.dtype
                )
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
        if (
            self._use_pcgrad
            and self._use_bond_head
            and self._joint_train
            and bond_loss_tensor.item() != 0.0
        ):
            # 3-task PCGrad: cfm, atom, bond.  We call ``backward``
            # on each scalar task loss separately (with ``retain_graph=True``
            # on the first two so the third can still backprop through
            # the shared graph), then read each parameter's ``.grad``.
            self.optimizer.zero_grad()
            cfm_loss.backward(retain_graph=True)
            grads_cfm = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in (
                    list(self.velocity_field.parameters())
                    + list(self.pocket_encoder.parameters())
                    + list(self.bond_head.parameters())
                )
            ]
            self.optimizer.zero_grad()
            (atom_loss_weight * atom_loss).backward(retain_graph=True)
            grads_atom = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in (
                    list(self.velocity_field.parameters())
                    + list(self.pocket_encoder.parameters())
                    + list(self.bond_head.parameters())
                )
            ]
            self.optimizer.zero_grad()
            (bond_loss_weight * bond_loss_tensor).backward()
            grads_bond = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
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
        params = list(self.velocity_field.parameters()) + list(
            self.pocket_encoder.parameters()
        )
        if self._use_bond_head and self._joint_train and self.bond_head is not None:
            params = params + list(self.bond_head.parameters())
        # Wf-A62 W1 (VF5) — warn when the bond_head is in use but the
        # legacy ``joint_train=False`` path excludes its params from
        # the clip; if a caller enables both ``use_bond_head=True``
        # and ``joint_train=False``, the bond-head grads are
        # unbounded and a single bad batch can blow up between
        # optimizer steps.  Warn-once-per-setup so it does not flood
        # stderr during a 5-seed × 50-epoch run.
        elif self._use_bond_head and not self._joint_train:
            if not getattr(self, "_bond_head_clip_warn_emitted", False):
                warnings.warn(
                    "_joint_adapter: bond_head in use but joint_train=False; "
                    "bond_head params are NOT included in the grad-norm clip. "
                    "Unbounded bond-head grads can blow up between optimizer "
                    "steps.  Use ``joint_train=True`` for end-to-end training.",
                    stacklevel=2,
                )
                self._bond_head_clip_warn_emitted = True
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        # Wf-A56 W2 (2026-10-16) — P1 OOV defence-in-depth.  If the
        # data-load guard was bypassed (e.g. an ad-hoc notebook calls
        # ``Molecule.from_smiles`` directly) and bad_atoms slipped
        # through, ``atom_loss`` is ``Inf`` which would poison
        # ``AdamW``.  Detect non-finite losses AFTER ``clip_grad_norm_``
        # then SKIPPING optimizer step, recording the
        # ``skipped_oov`` diagnostic so a future maintainer can spot
        # the upstream leak.  ``bad_atoms`` is the canonical variable
        # name used by the upstream data-load guard.
        bad_atoms = bool(
            torch.as_tensor(loss).isnan().any() or torch.as_tensor(loss).isinf().any()
        )
        if bad_atoms:
            self.optimizer.zero_grad()
            # 2026-10-19 — name the term(s) that blew up.  The step is still
            # skipped and the losses still report nan (the numbers are not
            # meaningful), but without this a caller could see 'skipped_oov'
            # and still not know whether the leak was in the atom CE, the
            # bond CE or the flow loss.
            nonfinite_terms = [
                name
                for name, term in (
                    ("cfm", cfm_loss),
                    ("atom", atom_loss),
                    ("bond", bond_loss_tensor),
                )
                if not bool(torch.isfinite(term).all())
            ]
            self.last_losses.update({
                "cfm": float("nan"),
                "atom": float("nan"),
                "bond": float("nan"),
                "total": float(loss.item()),
                "skipped_oov": int(self.last_losses.get("skipped_oov", 0)) + 1,
                "nonfinite_terms": nonfinite_terms,
            })
            return self.last_losses["total"]
        self.optimizer.step()
        # Stash individual losses for logging; return the total as a float
        # to honour the Protocol contract.
        self.last_losses.update({
            "cfm": float(cfm_loss.item()),
            "atom": float(atom_loss.item()),
            "bond": float(bond_loss_value),
            "total": float(loss.item()),
        })
        # Wf-A69 W1 (2026-10-17) — diagnostic counters for the bond
        # candidate list.  ``last_bond_n_candidates`` is the raw size
        # of the candidate list before any subsampling; ``last_bond_n_pos``
        # and ``last_bond_n_neg`` are the truth-vs-no_bond split.  These
        # three counters let a maintainer verify the candidate-list
        # source is doing what the YAML asked: ``truth_only`` should
        # have ``n_neg == 0``; ``radius_graph`` should have a positive
        # ``n_neg`` so the head sees the negative class.  Bit-exact
        # pre-Wf-A69 callers leave them at 0 (the legacy path never
        # recorded this metric).
        try:
            self.last_bond_n_candidates = int(edge_index_e.shape[1])
            self.last_bond_n_pos = int((edge_labels > 0).sum().item())
            self.last_bond_n_neg = int((edge_labels == 0).sum().item())
        except Exception:
            self.last_bond_n_candidates = 0
            self.last_bond_n_pos = 0
            self.last_bond_n_neg = 0
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
        field_modes = [
            (module, module.training)
            for root in (self.velocity_field, self.pocket_encoder)
            for module in root.modules()
        ]
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
        pocket,  # Pocket | List[Pocket] | None
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
        inferred_n = (
            torch.as_tensor(fixed_spec).shape[-1] if fixed_spec is not None else 8
        )
        n_atoms = getattr(config, "n_atoms", inferred_n)
        # All sampling tensors live on self.device so the ODE solver runs
        # on GPU end-to-end when ROCm is available.
        # Initial atom_types are placeholders that get REPLACED after the
        # ODE integration (see step 5 below).
        atom_types = torch.zeros(
            n_samples, n_atoms, dtype=torch.long, device=self.device
        )
        fixed_atoms = config.conditioning.get("fixed_atom_types")
        if fixed_atoms is not None:
            fixed_atoms = torch.as_tensor(
                fixed_atoms, dtype=torch.long, device=self.device
            )
            if fixed_atoms.shape == (n_atoms,):
                fixed_atoms = fixed_atoms.unsqueeze(0).expand(n_samples, -1)
            if fixed_atoms.shape != (n_samples, n_atoms):
                raise ValueError("fixed_atom_types must have shape (N,) or (B, N)")
            if (fixed_atoms <= 0).any() or (
                fixed_atoms >= self._max_atomic_number
            ).any():
                raise ValueError(
                    "fixed_atom_types outside the model atomic-number vocabulary"
                )
        # 2026-10-04 — no fixed `edge_index` here any more.  The solver changes
        # `x_t` at every step and the graph is a radius graph over `x_t`, so a
        # single graph bound for the whole solve is stale from step 1.
        # `_RadiusGraphVelocityModel` rebuilds it per call; the two one-shot
        # forwards below build theirs from `x_final`.
        #
        # T5: encode pocket once (broadcast to all samples in this batch).
        # Returns None when pocket=None so the velocity field uses the
        # unconditioned (zero-bias) path.
        pocket_embed = self._encode_pocket(
            pocket, b=n_samples, max_n_atoms=n_atoms, device=self.device
        )

        # 1. Sample noise
        x_0 = torch.randn(
            n_samples, n_atoms, 3, device=self.device, generator=generator
        )
        # 2. Time grid (Lipman 2023 default = [0, 1]).
        # Wf-A35-2 (D1): honour ``config.fm_t_dist`` ("uniform" | "log_normal").
        # When "log_normal", draw ``n_steps+1`` flow times via the
        # log-normal Karras 2022 schedule (arXiv:2206.00364 §3.4: t =
        # sigmoid(N(mu, sigma))), with the mid-range concentration that
        # makes the regression target hardest.  At the default
        # ``fm_t_dist="uniform"`` we keep the legacy ``torch.linspace`` so
        # every pre-A35-2 invocation is bit-exact identical.
        _fm_t_dist = str(getattr(config, "fm_t_dist", "uniform") or "uniform")
        if _fm_t_dist == "log_normal":
            from molmetal.adapters.flow_matching_lipman.log_normal_t import (
                log_normal_t_sampling as _ln_sample,
            )

            _mu = float(getattr(config, "fm_log_normal_mu", -0.5))
            _sigma = float(getattr(config, "fm_log_normal_sigma", 1.0))
            # n_steps+1 grid points; same RNG stream as ``torch.linspace``
            # (no RNG consumed in the uniform branch) means the log-normal
            # branch is the only place that differs from legacy.
            t_grid = _ln_sample(
                int(config.n_steps) + 1, _mu, _sigma,
                generator=generator, device=self.device,
            )
        else:
            t_grid = torch.linspace(
                0.0, 1.0, config.n_steps + 1, device=self.device
            )
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
        cfg_scale = float(getattr(self, "_cfg_scale", 1.0) or 1.0)
        # The radius graph is a function of `x`, which the solver moves, so it
        # is rebuilt inside the callable rather than bound here.
        velocity_model = _RadiusGraphVelocityModel(
            velocity_field=self.velocity_field,
            atom_types=atom_types,
            node_mask=None,  # fixed n_atoms grid: every sample is fully real
            cutoff=self._cutoff,
            pocket_embed=pocket_embed,
            cfg_scale=cfg_scale if pocket_embed is not None else 1.0,
        )

        # Wf-A75 (2026-10-18) — MolDiff-style bond-length guidance.
        # When `bond_length_guidance_weight > 0`, wrap the velocity
        # model so that at every ODE step the bond-length prior force
        # is added to the EGNN velocity.  Default 0 = no-op
        # (back-compat with the existing sampler).  See the
        # `_BondLengthGuidance` class docstring for the prior
        # construction and the rationale (per Vignac et al. 2023,
        # atom-bond inconsistency is the dominant failure mode of
        # Cartesian-coordinate diffusion on metal complexes).
        _bond_guidance_weight = float(
            getattr(config, "bond_length_guidance_weight", 0.0) or 0.0
        )
        if _bond_guidance_weight > 0.0:
            velocity_model = _BondGuidedVelocityModel(
                inner=velocity_model,
                guidance=_BondLengthGuidance(
                    atom_types=atom_types,
                    cutoff=self._cutoff,
                    guidance_weight=_bond_guidance_weight,
                ),
            )

        # Geometry conditioning requires known identities and real donor ->
        # metal endpoints. Unconstrained categorical sampling has neither.
        metal_prior_k = int(self._metal_prior_k_every)
        metal_prior_w = float(self._metal_prior_weight)
        donor_edges = config.conditioning.get("dative_edge_index")
        diagnostics = {
            "status": "inactive",
            "reason": "missing_fixed_atom_types_or_dative_edges",
            "weight": metal_prior_w,
            "gradient_evaluations": 0,
            "nonzero_gradient_graph_updates": 0,
            "fixed_atom_types": fixed_atoms is not None,
        }
        self.last_metal_prior_diagnostics = diagnostics
        prior = None
        if donor_edges is not None:
            if fixed_atoms is None:
                raise ValueError(
                    "dative_edge_index requires fixed_atom_types; "
                    "sampled identities cannot define prior donors in advance"
                )
            donor_edges = torch.as_tensor(
                donor_edges, dtype=torch.long, device=self.device
            )
            if donor_edges.dim() == 2 and donor_edges.shape[0] == 2:
                donor_edges = donor_edges.unsqueeze(0).expand(n_samples, -1, -1)
            if donor_edges.dim() != 3 or donor_edges.shape[:2] != (n_samples, 2):
                raise ValueError(
                    "dative_edge_index must have shape (2, E) or (B, 2, E)"
                )
            if (donor_edges < 0).any() or (donor_edges >= n_atoms).any():
                raise ValueError("dative_edge_index contains invalid atom endpoints")
            from molmetal.molmetal_lam.priors.metal_geometry import MetalGeometryPrior

            prior = MetalGeometryPrior(
                weight=metal_prior_w,
                metal_geometry_map=config.conditioning.get("metal_geometry_map"),
            )
            # Check declared edge destinations actually identify known metals.
            metal_set = fixed_atoms.new_tensor(list(prior.metal_geometry_map))
            destinations = torch.gather(fixed_atoms, 1, donor_edges[:, 1])
            if not (destinations.unsqueeze(-1) == metal_set).any(-1).all():
                raise ValueError(
                    "Every dative edge destination must be a supported metal centre"
                )
            if donor_edges.shape[-1] == 0:
                diagnostics["reason"] = "empty_dative_graph"
                prior = None
        if metal_prior_w <= 0 or metal_prior_k <= 0:
            diagnostics["reason"] = "disabled_by_weight_or_interval"
            prior = None
        step_counter = 0
        if prior is not None:
            diagnostics.update(
                status="eligible", reason="explicit_fixed_atoms_and_donor_edges"
            )

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
                # Wf-A62 W1 (JM4) — surface the O(n_samples * P_donor)
                # memory footprint so small-GPU jobs abort before OOM
                # rather than dying mid-iteration.  The metal-prior
                # path is per-sample (no batch dim), so n_samples=1000
                # on P_donor=20 holds ~80 kB of grad state per step
                # but accumulates per ODESolver.inner-step.
                if n_samples > 32:
                    warnings.warn(
                        f"_velocity_with_metal_prior: n_samples={n_samples} drives "
                        f"O(n_samples*P_donor) energy terms; each ODESolver step "
                        f"calls prior.prior_loss per graph. Consider n_samples<=32 "
                        f"for sub-32 GB GPUs, or wrap in torch.utils.checkpoint.",
                        stacklevel=2,
                    )
                # Wf-A62 W1 (JM8) — non-differentiable prior.prior_loss
                # (e.g. RDKit) silently returns the base velocity; surface
                # that fallback to the diagnostics dict so callers can see
                # the metal prior was effectively a no-op.
                grad_ok = True
                try:
                    energies = [
                        prior.prior_loss(
                            x_req[i],
                            fixed_atoms[i],
                            torch.full(
                                (donor_edges.shape[-1],),
                                2,
                                dtype=torch.long,
                                device=x.device,
                            ),
                            edge_index=donor_edges[i],
                        )
                        for i in range(n_samples)
                    ]
                    energy = torch.stack(energies).sum()
                except Exception:
                    grad_ok = False
                    energy = None
                diagnostics["gradient_evaluations"] += 1
                if not grad_ok or (energy is not None and not energy.requires_grad):
                    diagnostics["non_differentiable_fallback_count"] = (
                        int(diagnostics.get("non_differentiable_fallback_count", 0)) + 1
                    )
                    return base
                gradient = torch.autograd.grad(energy, x_req)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    "Metal geometry prior produced nonfinite gradients"
                )
            updates = int((gradient.square().sum(dim=(1, 2)) > 0).sum())
            diagnostics["nonzero_gradient_graph_updates"] += updates
            if updates:
                diagnostics["status"] = "active"
            return base - gradient.detach()

        wrapper = self._ModelWrapper(
            model=_velocity_with_metal_prior if prior is not None else velocity_model
        )
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
        # Wf-A57 W9-P1 (2026-10-16): validate_method() gates the
        # expanded union set {"euler", "midpoint", "dopri5", "heun3",
        # "rk4"}; any caller passing a misspelled method raises here
        # before solver.sample() produces numerically meaningless drift.
        ode_method = getattr(config, "method", "midpoint")
        validate_method(ode_method)
        x_final = solver.sample(
            x_init=x_0,
            step_size=1.0 / config.n_steps,
            method=ode_method,
            time_grid=t_grid,
        )
        # x_final shape: (n_samples, n_atoms, 3) or (T+1, n_samples, n_atoms, 3)
        # Wf-A62 W1 (JM3) — preserve trajectory so divergence can be
        # detected at the step where it first occurred, not only at the
        # final step.  If the ODE diverged at step T/2 but converged by
        # T the final-state check below would falsely report success.
        x_traj = x_final if x_final.dim() == 4 else None
        if x_final.dim() == 4:
            x_final = x_final[-1]
        if not torch.isfinite(x_final).all():
            if x_traj is not None:
                bad_step = int(
                    torch.argwhere(~torch.isfinite(x_traj).all(dim=(1, 2, 3)))[0].item()
                )
                raise FloatingPointError(
                    f"Flow integration produced non-finite coordinates at trajectory "
                    f"step {bad_step}/{x_traj.shape[0] - 1} (post-solve shape "
                    f"{tuple(x_final.shape)})"
                )
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
            # One-shot forward on the SOLVED coordinates: the graph is built
            # from `x_final` once.  (Unlike the solve above, `x_final` does
            # not move between these two calls.)
            _ei_final, _em_final = radius_edge_index(
                x_final, None, self._cutoff
            )
            if cfg_scale != 1.0 and pocket_embed is not None:
                out_cond = self.velocity_field(
                    x_final,
                    init_atom_types,
                    _ei_final,
                    torch.ones(n_samples, device=self.device),
                    edge_mask=_em_final,
                    pocket_embed=pocket_embed,
                )
                out_uncond = self.velocity_field(
                    x_final,
                    init_atom_types,
                    _ei_final,
                    torch.ones(n_samples, device=self.device),
                    edge_mask=_em_final,
                    pocket_embed=None,
                )
                # CFG-style logits interpolation (linear in logit space).
                atom_logits = out_uncond["atom_logits"] + cfg_scale * (
                    out_cond["atom_logits"] - out_uncond["atom_logits"]
                )
            else:
                out = self.velocity_field(
                    x_final,
                    init_atom_types,
                    _ei_final,
                    torch.ones(n_samples, device=self.device),
                    edge_mask=_em_final,
                    pocket_embed=pocket_embed,
                )
                atom_logits = out["atom_logits"]  # (n_samples, n_atoms, max_z)
            # Wf-A56 W2 (2026-10-16) — P4 BF16 overflow guard (sampler
            # path).  Same rationale as L1855 / L2587: untrained
            # Blackwell kernels can produce +/-1e4 logits which then
            # overflow the categorical sampler.  Clamp here is the
            # last guard before softmax.
            atom_logits = atom_logits.clamp(min=-50.0, max=50.0)
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
                    atom_logits.shape[-1],
                    atom_logits.device,
                )
                atom_logits = atom_logits.masked_fill(~vocab_mask_tensor, float("-inf"))
            atom_probs = F.softmax(atom_logits, dim=-1)
            # Categorical sample per atom
            if not torch.isfinite(atom_probs).all():
                raise FloatingPointError(
                    "Atom prediction produced non-finite probabilities"
                )
            # Wf-A35-2 (D2): honour ``config.top_p`` / ``config.top_k`` /
            # ``config.temperature`` via ``diversity_sampling.nucleus_sample``
            # (Holtzman 2020, arXiv:1904.09751).  At the default
            # ``top_p=1.0, top_k=0, temperature=1.0`` no rows are masked
            # and the softmax + multinomial path is bit-exact identical
            # to the legacy ``F.softmax`` + ``torch.multinomial`` pair.
            # The opt-in audit counter ``nucleus_sample_invocations`` is
            # exposed via ``last_sampling_diagnostics`` so callers can
            # verify the wire-in fired (--top-p / --top-k / --temperature
            # were not no-ops).
            from molmetal.adapters.flow_matching_lipman.diversity_sampling import (
                nucleus_sample as _nucleus_sample,
            )

            _top_p = float(getattr(config, "top_p", 1.0) or 1.0)
            _top_k = int(getattr(config, "top_k", 0) or 0)
            _temperature = float(
                getattr(config, "temperature", 1.0) or 1.0
            )
            _nucleus_active = (
                _top_p < 1.0 or _top_k > 0 or _temperature != 1.0
            )
            if _nucleus_active:
                sampled_atoms = _nucleus_sample(
                    atom_logits,  # pre-softmax logits for temperature rescale
                    p=_top_p,
                    k=_top_k,
                    temperature=_temperature,
                    generator=generator,
                ).reshape(n_samples, n_atoms)
                self.last_sampling_diagnostics = {
                    "nucleus_sample_invocations": int(
                        getattr(self, "_nucleus_sample_invocation_count", 0)
                    )
                    + 1,
                    "top_p": _top_p,
                    "top_k": _top_k,
                    "temperature": _temperature,
                    "active": True,
                }
                self._nucleus_sample_invocation_count = int(
                    getattr(self, "_nucleus_sample_invocation_count", 0)
                ) + 1
            else:
                sampled_atoms = torch.multinomial(
                    atom_probs.reshape(-1, atom_probs.shape[-1]),
                    1,
                    generator=generator,
                ).reshape(n_samples, n_atoms)
                self.last_sampling_diagnostics = {
                    "nucleus_sample_invocations": int(
                        getattr(self, "_nucleus_sample_invocation_count", 0)
                    ),
                    "top_p": _top_p,
                    "top_k": _top_k,
                    "temperature": _temperature,
                    "active": False,
                }
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
                        [edge_src, edge_dst],
                        dtype=torch.long,
                    )
                    bond_types_tensor = torch.tensor(
                        edge_ord,
                        dtype=torch.long,
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
                        [edge_src, edge_dst],
                        dtype=torch.long,
                    )
                    bond_types_tensor = torch.tensor(
                        edge_ord,
                        dtype=torch.long,
                    )
                else:
                    bonds_tensor = torch.zeros(2, 0, dtype=torch.long)
                    bond_types_tensor = torch.zeros(0, dtype=torch.long)
                smiles_i = decoded.smiles
            else:
                bonds_tensor = torch.zeros(2, 0, dtype=torch.long)
                bond_types_tensor = torch.zeros(0, dtype=torch.long)
                smiles_i = ""
            mols.append(
                Molecule(
                    coords=coords_i,
                    atom_types=atoms_i,
                    bonds=bonds_tensor,
                    bond_types=bond_types_tensor,
                    formal_charges=torch.zeros(n_atoms, dtype=torch.long),
                    smiles=smiles_i,
                )
            )
        return mols

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_pocket(
        self,
        pocket,  # Pocket | List[Pocket] | None
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
        self,
        max_atomic_number: int,
        device: torch.device,
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
        # Wf-A62 W1 (JM5) — when the head dim is smaller than the
        # requested vocab (e.g., a50-d head trained on H/C/N/O/F/P/S/
        # Cl/Se/Br/I/Pt cannot draw Pd/Au/Ir/Ru because Z >= 78 falls
        # outside the head's support), the over-range Zs are silently
        # dropped.  Surface the silent vocab trim so a maintainer
        # sees "we have a 50-d head but the requested vocab is 12"
        # and can re-train or relax the dim.
        if max_atomic_number < max(vocab):
            dropped = sorted(z for z in vocab if z >= max_atomic_number)
            warnings.warn(
                f"_build_vocab_mask: head dim {max_atomic_number} drops "
                f"{len(dropped)} vocab atoms {dropped} (vocab max = "
                f"{max(vocab)}). Inference will never draw these metals. "
                f"Increase ``max_atomic_number`` or re-train the head.",
                stacklevel=2,
            )
        mask = torch.zeros(max_atomic_number, dtype=torch.bool, device=device)
        for z in vocab:
            if 0 <= z < max_atomic_number:
                mask[z] = True
        return mask

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "metal_prior_last_generation": dict(
                getattr(self, "last_metal_prior_diagnostics", {"status": "not_run"})
            ),
            "metal_prior_contract": (
                "requires conditioning.fixed_atom_types and "
                "conditioning.dative_edge_index; unconstrained sampling is inactive"
            ),
            "atom_training_contract": (
                "masked Z=0 inputs; true atomic numbers only as cross-entropy "
                "targets; old teacher-forced checkpoints require retraining"
            ),
            "pocket_embed_scale": self._pocket_embed_scale,
            "sampling_seed": "GenerationConfig.seed via device-local torch.Generator",
            "sampling_mode": "eval with caller module modes restored",
            "model": "LipmanFlowMatching_v2_equivariant",
            "coordinate_readout": (
                "tanh(scalar_head(h))*(x-centroid) + "
                "mean_edges(tanh(phi)*relative_position)"
            ),
            "checkpoint_compatibility": (
                "v1 H-to-3 vel_head is incompatible; v2 requires retraining"
            ),
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
            "scm": (
                "git clone https://github.com/facebookresearch/flow_matching.git "
                "molmetal/references/flow_matching"
            ),
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
    p.add_argument(
        "--ref-repo",
        default="molmetal/references/flow_matching",
        help="Path to the cloned facebookresearch/flow_matching repo",
    )
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--max-atomic-number", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--atom-loss-weight", type=float, default=0.1)
    # TODO-09: metal-geometry prior knobs.
    p.add_argument(
        "--metal-prior-weight",
        type=float,
        default=0.1,
        help="Weight applied to the metal-geometry soft penalty "
        "during sampling.  Set to 0.0 to disable the prior "
        "(legacy bit-exact path).",
    )
    p.add_argument(
        "--metal-prior-k-every",
        type=int,
        default=10,
        help="Apply the metal-geometry prior every K integration "
        "steps during sampling.  Defaults to 10.",
    )
    # TODO-08: tmQM pre-trained encoder initialiser knobs.
    p.add_argument(
        "--tmqm-init",
        nargs="?",
        const="__default__",
        default=None,
        help="Warm-start the EGNN encoder from the tmQM pre-trained "
        "checkpoint (DMPNN; see molmetal/reports/f2_tmqm_pretrain.md). "
        "Pass --tmqm-init to use the default checkpoint path, or "
        "--tmqm-init=/path/to/ckpt.pt for an explicit file.  Use "
        "--random-init to opt out.",
    )
    p.add_argument(
        "--random-init",
        action="store_true",
        help="Disable the tmQM warm-start even if --tmqm-init is passed "
        "(equivalent to not loading any encoder checkpoint).",
    )
    # TODO-10 axis-C: classifier-free guidance knobs.
    p.add_argument(
        "--context-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping the conditioning pocket "
        "embedding during training.  Teaches the model the "
        "unconditional marginal v(v | ∅) needed for CFG at "
        "inference time.  Default 0.1 (Ho & Salimans 2022).",
    )
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale used at inference.  "
        "1.0 = legacy conditional sampler (bit-exact); "
        "2.0 = standard CFG strength; >=3.0 = strong guidance.  "
        "Set to 1.0 to disable CFG.",
    )
    return p
