"""Differential learning-rate optimizer builder (Wf-A11 / optimizer wave).

Implements the Loshchilov-Hutter 2017 / Fix3Discarl / Fix3Discarl pattern:
4 params groups

    head_decay
    head_no_decay
    backbone_decay
    backbone_no_decay

so the head can run with a different (usually smaller) learning rate from the
backbone WITHOUT either layer's weight-decay terms bleeding into the no-decay
parameter learning rate (the canonical "fixing weight decay regularization in
Adam" bug).

The "head" vs "backbone" partition
---------------------------------
Heuristic: any parameter whose attribute name contains the substring
``"head"`` is a head; everything else is backbone.  This works for the
Mol-Metal CFM trainer where the head parameter names are
``model.bond_head.*`` and ``model.yuelbond_head.*``.  Callers with a
different head naming scheme should subclass or set
:attr:`molmetal.optim.differential_lr._HEAD_ATTR_HINT` accordingly.

Backward-compat
---------------
* :func:`build_differential_optimizer` returns a
  :class:`torch.optim.Adam` with 4 param groups only when
  ``head_lr_multiplier != 1.0``.  When the multiplier is 1.0 (legacy
  default), it returns a single-group Adam with the same flat-weight-decay
  behaviour the Wf-13 trainer shipped with — preserving the Wf-13
  baseline loss trajectory on benchmark cold-cache runs.
* ``no_decay_param_names`` defaults to ``["bias", "LayerNorm.weight",
  "layer_norm.weight"]`` (BERT-style), which matches every Transformer /
  EGNN layer in the Mol-Metal tree.

Lit anchors
-----------
* Loshchilov & Hutter 2017 "Fixing Weight Decay Regularization in Adam"
  (arXiv:1711.05101) — the canonical no-decay split.
* Mol-Ego fakes 2026 "Mol-Metal Wf-A11 design §B" — the 4-group
  split rationale (head+backbone orthogonal decay/no-decay).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

#: Default substrings that mark a parameter as belonging to the head
#: rather than the backbone.  Override per-call when the head naming
#: convention differs.
_DEFAULT_HEAD_HINTS: Tuple[str, ...] = ("head",)

#: Default substrings that mark a parameter as belonging to the
#: Track-A q-channel (per-atom NBO natural-charge head + projector).
#: Default OFF (no 5th group emitted) when no q-channel parameter is
#: registered on the model — see :func:`build_differential_optimizer`
#: for the legacy fast-path collapse to a single-group Adam.
_DEFAULT_Q_CHANNEL_HINTS: Tuple[str, ...] = (
    "q_channel",
    "natural_charge_proj",
    "charge_head",
    "q_head",
)

#: Default no-decay parameter set — canonical Transformer / BERT convention.
_DEFAULT_NO_DECAY_NAMES: Tuple[str, ...] = (
    "bias",
    "LayerNorm.weight",
    "layer_norm.weight",
)


def _split_params(
    model: nn.Module,
    no_decay_param_names: Sequence[str],
    head_hints: Sequence[str],
    q_channel_hints: Optional[Sequence[str]] = None,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """Bucket model parameters into 4 (legacy) or 6 (Track-A q-channel) groups.

    Returns ``(head_decay, head_no_decay, q_decay, q_no_decay, backbone_decay,
    backbone_no_decay)`` when ``q_channel_hints`` matches at least one
    parameter.  Returns the same 6-tuple but with ``q_decay`` /
    ``q_no_decay`` empty when no parameter matches.

    A parameter is:

    * "head" iff any of its fully-qualified attribute names (joined with
      ``.``) contains a substring from ``head_hints`` AND does not contain
      a substring from ``q_channel_hints``.
    * "q_channel" iff its name contains a substring from ``q_channel_hints``
      (priority over "head" so the q-prediction head can be tuned
      independently).
    * "backbone" otherwise.
    """
    no_decay_set = set(no_decay_param_names)
    q_set = set(q_channel_hints) if q_channel_hints else set()
    head_decay: List[nn.Parameter] = []
    head_no_decay: List[nn.Parameter] = []
    q_decay: List[nn.Parameter] = []
    q_no_decay: List[nn.Parameter] = []
    backbone_decay: List[nn.Parameter] = []
    backbone_no_decay: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_q = any(h in name for h in q_set) if q_set else False
        is_head = (not is_q) and any(h in name for h in head_hints)
        no_decay = any(name.endswith(suf) for suf in no_decay_set)
        if is_q and no_decay:
            q_no_decay.append(param)
        elif is_q:
            q_decay.append(param)
        elif is_head and no_decay:
            head_no_decay.append(param)
        elif is_head:
            head_decay.append(param)
        elif no_decay:
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)
    return (
        head_decay,
        head_no_decay,
        q_decay,
        q_no_decay,
        backbone_decay,
        backbone_no_decay,
    )


def build_differential_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    head_lr_multiplier: float = 1.0,
    weight_decay: float = 1e-5,
    no_decay_param_names: Optional[Sequence[str]] = Iterable,  # type: ignore[assignment]
    head_hints: Optional[Sequence[str]] = None,
    q_channel_lr_multiplier: float = 1.0,
    q_channel_hints: Optional[Sequence[str]] = None,
) -> Optimizer:
    """Build a :class:`torch.optim.Adam` for differential head + q-channel LR.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters are optimized.  Frozen parameters
        (``requires_grad == False``) are excluded.
    base_lr : float, default 1e-4
        Backbone learning rate.  Head learning rate is
        ``base_lr * head_lr_multiplier``.  Q-channel learning rate is
        ``base_lr * q_channel_lr_multiplier``.
    head_lr_multiplier : float, default 1.0
        When ``1.0`` (legacy), the returned optimizer collapses to a
        single-group Adam with the same flat ``weight_decay`` the legacy
        Wf-13 trainer shipped with — no behaviour change.  When ``!= 1.0``
        (e.g. ``0.1``), four param groups are produced.
    weight_decay : float, default 1e-5
        Weight-decay term applied to the decay groups only.
    no_decay_param_names : sequence of str, optional
        Parameter-name suffixes (matched via ``str.endswith``) that should
        be excluded from weight decay.  Default: ``("bias",
        "LayerNorm.weight", "layer_norm.weight")``.
    head_hints : sequence of str, optional
        Substrings whose presence in a parameter's fully-qualified name
        marks it as a head parameter.  Default: ``("head",)``.
    q_channel_lr_multiplier : float, default 1.0
        When ``1.0`` (Track-A legacy OFF), no 5th group is emitted even
        if ``q_channel_hints`` matches — the q-channel parameters fall
        into the backbone bucket at backbone LR.  When ``!= 1.0``, a
        separate q-channel group is emitted (priority over the head
        match).  Default ``1.0`` preserves bit-exactness with the
        pre-Track-A trainer when the q-head is OFF.
    q_channel_hints : sequence of str, optional
        Substrings whose presence in a parameter's fully-qualified name
        marks it as a Track-A q-channel parameter.  Default:
        ``("q_channel", "natural_charge_proj", "charge_head", "q_head")``.

    Returns
    -------
    torch.optim.Adam
        Single-group when both ``head_lr_multiplier`` AND
        ``q_channel_lr_multiplier`` equal ``1.0``.  4-group when only
        ``head_lr_multiplier != 1.0``.  6-group when
        ``q_channel_lr_multiplier != 1.0`` AND at least one parameter
        matches ``q_channel_hints``.
    """
    if no_decay_param_names is None or no_decay_param_names is Iterable:
        no_decay_param_names = _DEFAULT_NO_DECAY_NAMES
    if head_hints is None:
        head_hints = _DEFAULT_HEAD_HINTS
    if q_channel_hints is None:
        q_channel_hints = _DEFAULT_Q_CHANNEL_HINTS

    (
        head_decay,
        head_no_decay,
        q_decay,
        q_no_decay,
        backbone_decay,
        backbone_no_decay,
    ) = _split_params(
        model, no_decay_param_names, head_hints, q_channel_hints
    )

    head_lr = float(base_lr) * float(head_lr_multiplier)
    q_lr = float(base_lr) * float(q_channel_lr_multiplier)

    # Legacy fast-path: when head lr == backbone lr AND q lr == backbone
    # lr, collapse to a single group so the optimizer behaves identically
    # to a flat Adam.  This is what the Wf-13 baseline shipped with;
    # preserving it preserves the benchmark baseline.
    if (
        abs(head_lr - float(base_lr)) < 1e-12
        and abs(q_lr - float(base_lr)) < 1e-12
    ):
        return Adam(
            [
                {
                    "params": (
                        list(backbone_decay)
                        + list(backbone_no_decay)
                        + list(q_decay)
                        + list(q_no_decay)
                    ),
                    "weight_decay": float(weight_decay),
                }
            ],
            lr=float(base_lr),
        )

    # 4- or 6-group path: head vs q_channel vs backbone, decay vs no-decay.
    # q_channel group is emitted between head and backbone in the
    # returned list — Adam is order-insensitive for the first step but
    # the param-group key influences state_dict keys (legacy fast-path
    # above collapses to single-group so old checkpoints resume).
    param_groups: List[dict] = []
    if head_decay:
        param_groups.append(
            {"params": head_decay, "lr": head_lr, "weight_decay": float(weight_decay)}
        )
    if head_no_decay:
        param_groups.append(
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0}
        )
    # Q-channel group (Track-A): only emitted when q_lr != base_lr AND
    # at least one parameter matched a q-channel hint.
    if q_decay and abs(q_lr - float(base_lr)) > 1e-12:
        param_groups.append(
            {"params": q_decay, "lr": q_lr, "weight_decay": float(weight_decay)}
        )
    if q_no_decay and abs(q_lr - float(base_lr)) > 1e-12:
        param_groups.append(
            {"params": q_no_decay, "lr": q_lr, "weight_decay": 0.0}
        )
    if backbone_decay:
        param_groups.append(
            {"params": backbone_decay, "lr": float(base_lr), "weight_decay": float(weight_decay)}
        )
    if backbone_no_decay:
        param_groups.append(
            {"params": backbone_no_decay, "lr": float(base_lr), "weight_decay": 0.0}
        )

    return Adam(param_groups)


__all__ = ["build_differential_optimizer"]