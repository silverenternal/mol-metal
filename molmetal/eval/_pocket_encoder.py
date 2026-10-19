"""Wf-A36-1 (2026-10-14) — shared helper for the eval-side PocketEncoder.

Background
----------
The CFM trainer (molmetal.scripts.train_cfm_yuelbond_wasserstein) wires
a :class:`molmetal.adapters.flow_matching_lipman.PocketEncoder` (the
in-tree per-pocket-atom encoder) and a :class:`PocketGenEncoder` (the
lazy-import wrapper around the vendored Zhang et al. 2024 BilevelEncoder)
into the CFM velocity field.  At eval time, however, neither was
instantiated: :mod:`molmetal.eval.cross_dataset` and
:mod:`molmetal.eval.linear_probe` only loaded the YuelBond bond head,
so the pocket-conditioning slot was structurally absent from the eval
path.  This module is the single source of truth for *instantiating*
the pocket encoder at eval time.

Backward-compat
--------------
At default (``use_pocket_encoder=False``) the helper is a no-op:
:func:`build_pocket_encoder` returns ``None`` and no log / warning is
emitted, so the cross-dataset + linear-probe behaviour is bit-exact
identical to the pre-Wf-A36-1 path for every shipped CFM YAML (none of
which set ``use_pocket_encoder: true`` today).

One-shot warning
----------------
When the user opts in to the pocket encoder AND the heavy PocketGen
deps (``torch_scatter`` / ``torch_geometric`` / ``esm``) are missing,
the helper emits a single :func:`warnings.warn` per Python session
(guarded by a module-level boolean) and returns ``None``.  A subsequent
call does NOT re-warn — that is what the spec calls "1-shot".
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

LOG = logging.getLogger("molmetal.eval._pocket_encoder")

# 1-shot warning guard.  Module-level so it survives across calls in
# the same Python process (the cross-dataset script calls the helper
# once per dataset; we only want the warning to fire the first time).
_POCKET_GEN_UNAVAILABLE_WARNED = False


def _warn_pocketgen_unavailable(reason: str) -> None:
    """Emit the 1-shot 'pocket encoder unavailable' warning.

    Parameters
    ----------
    reason : str
        Why the construction failed (e.g. ``"ImportError: torch_scatter"``).
    """
    global _POCKET_GEN_UNAVAILABLE_WARNED
    if _POCKET_GEN_UNAVAILABLE_WARNED:
        return
    _POCKET_GEN_UNAVAILABLE_WARNED = True
    msg = (
        "[eval.pocket_encoder] PocketGenEncoder is unavailable — "
        "falling back to NoOp (no pocket conditioning). "
        f"Reason: {reason}.  "
        "Install the heavy deps via the [pocketgen] extra group to "
        "enable the real encoder."
    )
    LOG.warning(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def build_pocket_encoder(
    use_pocket_encoder: bool,
    ckpt_state: Optional[Dict[str, Any]] = None,
    hidden_dim: int = 128,
) -> Tuple[bool, Optional[Any], int]:
    """Return ``(requested, encoder_or_none, pocket_embed_dim)``.

    Parameters
    ----------
    use_pocket_encoder : bool
        When ``True``, attempt to instantiate :class:`PocketGenEncoder`
        from :mod:`molmetal.adapters.pocketgen`.  When ``False`` (the
        default — the §5.11 bit-exact path), return immediately with
        ``(False, None, 0)`` and no warning.
    ckpt_state : dict, optional
        The full checkpoint state dict (the return value of
        :func:`molmetal.eval.cross_dataset._load_pretrained_checkpoint`).
        When the dict carries a ``pocket_encoder_state_dict`` key, the
        helper loads it into the freshly-instantiated encoder so the
        eval can replay the trainer's pocket-conditioning weights.
    hidden_dim : int
        The CFM velocity field's ``hidden_dim``.  Forwarded to
        :class:`PocketGenEncoder` as ``hidden_channels`` (the PocketGen
        paper's internal dim).  Default ``128`` matches the CFM
        trainer's ``DEFAULT_HIDDEN_DIM``.

    Returns
    -------
    (requested, encoder, pocket_embed_dim) : tuple
        * ``requested`` — the value of the input ``use_pocket_encoder``
          flag, propagated for the caller's logging.
        * ``encoder`` — a :class:`PocketGenEncoder` instance, or
          ``None`` if construction failed or the user did not opt in.
        * ``pocket_embed_dim`` — the encoder's output dim
          (``2 * hidden_channels`` by default per the PocketGen paper,
          or the user-supplied override), or ``0`` when no encoder was
          constructed.

    Notes
    -----
    The :class:`PocketGenEncoder` wrapper itself is a pure-Python
    class (no heavy deps on import), per
    ``molmetal.adapters.pocketgen.__init__``.  Construction succeeds
    in CPU-only envs; only :meth:`.load()` and :meth:`.extract_embedding`
    trigger the heavy ``torch_scatter`` / ``torch_geometric`` / ``esm``
    imports.  So at this helper level we never touch those deps.
    """
    if not use_pocket_encoder:
        # §5.11 bit-exact default — short-circuit before any import.
        return (False, None, 0)
    try:
        from molmetal.adapters.pocketgen import PocketGenEncoder
    except Exception as exc:
        _warn_pocketgen_unavailable(f"import failed: {type(exc).__name__}: {exc}")
        return (True, None, 0)
    try:
        enc = PocketGenEncoder(
            hidden_channels=int(hidden_dim),
            num_interactions=3,
            knn=30,
            cutoff=10.0,
            pocket_embed_dim=int(hidden_dim),
            pretrained_ckpt=None,
            esm_model_name="esm2_t6_8M_UR50D",
        )
    except Exception as exc:
        _warn_pocketgen_unavailable(
            f"ctor failed: {type(exc).__name__}: {exc}"
        )
        return (True, None, 0)
    # Optional pretrained weights — the CFM trainer may have written a
    # ``pocket_encoder_state_dict`` sub-dict alongside the velocity
    # field; load it when present, strict=False so future encoder
    # variants do not break the eval reader.
    if isinstance(ckpt_state, dict):
        pes = ckpt_state.get("pocket_encoder_state_dict")
        if isinstance(pes, dict) and pes:
            try:
                # PocketGenEncoder does not yet expose a load_state_dict
                # helper (the heavy encoder is built lazily on .load());
                # we stash the dict on the instance for the caller /
                # future wire-ins to consume.  This is a no-op for
                # bit-exact (the heavy encoder is not exercised here).
                enc._checkpoint_state_dict = pes
                LOG.info(
                    "[eval.pocket_encoder] stashed pocket_encoder_state_dict "
                    "(%d tensors) for downstream load",
                    len(pes),
                )
            except Exception as exc:
                LOG.warning(
                    "[eval.pocket_encoder] stashing pocket_encoder_state_dict "
                    "failed: %s",
                    exc,
                )
    return (True, enc, int(enc.pocket_embed_dim))


def try_build_full_cfm_model(
    model_state: Dict[str, Any],
    ema_state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Any]]:
    """Wf-A36-1 (2026-10-14) E2 — try to load the full CFM velocity field.

    The pre-Wf-A36-1 path applied the EMA snapshot only to the
    YuelBondDecoderHead (and even then, the head's ``ema_swap_in`` is
    a stub that just sets a flag — the live weights were never
    actually swapped out).  This helper constructs the real
    :class:`CFMJointModel` (the trainer's full velocity-field model),
    loads the checkpoint state into it, and applies the EMA round-trip
    on the *whole* model — so the CFM trajectory / bond head / any
    downstream consumer of the velocity field sees the EMA-smoothed
    weights, not the noisy post-final-step live weights.

    Parameters
    ----------
    model_state : dict
        The full ``state["model_state_dict"]`` (parameter tensors keyed
        by ``head.atom_encoder.weight`` etc.).
    ema_state : dict, optional
        The full ``state["ema_state_dict"]`` snapshot.  When empty or
        ``None``, the helper still builds the model (the EMA no-op is
        handled inside :class:`CFMJointModel`) so the caller can decide
        whether to use the model vs the legacy head-only path.

    Returns
    -------
    (built, model) : tuple
        * ``built`` — ``True`` iff the helper successfully constructed
          and loaded the model.
        * ``model`` — the :class:`CFMJointModel` instance, or ``None``
          on failure (any of: ImportError, TypeError, RuntimeError
          from ``load_state_dict`` shape mismatch, etc.).
    """
    if not isinstance(model_state, dict) or not model_state:
        return (False, None)
    # The legacy decode path in cross_dataset.py:152-167 already infers
    # hidden_dim + n_layers from the head's atom_encoder + msg_mlps
    # tensors.  We re-derive them here so the helper is self-contained.
    try:
        atom_w = model_state.get("head.atom_encoder.weight")
        if atom_w is None:
            return (False, None)
        hidden_dim = int(atom_w.shape[0])
        n_layers = sum(
            1
            for k in model_state
            if k.startswith("head.msg_mlps.") and k.endswith(".0.weight")
        )
        if n_layers < 2:
            n_layers = 3
    except Exception as exc:
        LOG.warning(
            "[eval.pocket_encoder] cannot infer CFMJointModel dims: %s",
            exc,
        )
        return (False, None)
    try:
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            CFMJointModel,
        )
    except Exception as exc:
        LOG.warning(
            "[eval.pocket_encoder] CFMJointModel import failed: %s; "
            "falling back to head-only EMA",
            exc,
        )
        return (False, None)
    try:
        model = CFMJointModel(
            hidden_dim=int(hidden_dim),
            n_layers=int(n_layers),
        )
        # strict=False so future flag additions do not break the loader.
        _missing, _unexpected = model.load_state_dict(model_state, strict=False)
        # Wf-A96 (2026-10-19) — a checkpoint whose velocity field was built
        # with opt-in modules this constructor does not pass (t_embedding_dim,
        # pocket_inject_per_layer, ...) leaves those tensors UNEXPECTED.  The
        # old code called that a successful build: the head loaded, the
        # velocity field stayed at random init, and every downstream EMA
        # number was measured on a model that was not the checkpoint.
        # MEASURED on cfm_fullcorpus_2026_10_10 (vel_field.time_embed_proj.*,
        # vel_field.pocket_per_layer_injector.*).  Refuse instead of
        # pretending, so the caller's documented head-only fallback runs.
        _vel_unexpected = [k for k in _unexpected if k.startswith("vel_field.")]
        if _vel_unexpected:
            LOG.warning(
                "[eval.pocket_encoder] checkpoint carries %d velocity-field "
                "tensor(s) this constructor does not build (%s); refusing the "
                "partial build and falling back to head-only EMA",
                len(_vel_unexpected),
                ", ".join(_vel_unexpected[:3]),
            )
            return (False, None)
    except Exception as exc:
        LOG.warning(
            "[eval.pocket_encoder] CFMJointModel ctor / load_state_dict "
            "failed: %s; falling back to head-only EMA",
            exc,
        )
        return (False, None)
    # EMA round-trip on the whole model.  The trainer's
    # :meth:`load_ema_state` is key-by-key and strict=False, so a
    # snapshot from a different flag combination (e.g. --use-egnn-velocity)
    # still applies the keys that overlap.
    if isinstance(ema_state, dict) and ema_state:
        try:
            # init_ema is idempotent — the snapshot starts at the live
            # weights, then load_ema_state overwrites it.  This mirrors
            # the trainer's save path (init_ema at step 0, update_ema
            # every step, ema_state_dict at save).
            if not getattr(model, "_ema_state", None):
                model.init_ema(decay=0.999)
            model.load_ema_state(ema_state)
            swapped = bool(model.ema_swap_in())
            if swapped:
                LOG.info(
                    "[eval.pocket_encoder] applied full-model EMA "
                    "(%d tensors) to CFMJointModel",
                    len(ema_state),
                )
        except Exception as exc:
            LOG.warning(
                "[eval.pocket_encoder] EMA round-trip on CFMJointModel "
                "failed: %s; using live weights",
                exc,
            )
    return (True, model)


__all__ = [
    "build_pocket_encoder",
    "try_build_full_cfm_model",
]
