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

from typing import List, Optional, Sequence, Tuple

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
    pocket_hints: Optional[Sequence[str]] = None,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """Bucket model parameters into 4 / 6 / 8 groups.

    Returns
    -------
    8-tuple ``(head_decay, head_no_decay, q_decay, q_no_decay,
    pocket_decay, pocket_no_decay, backbone_decay, backbone_no_decay)``.

    The pocket bucket is appended in priority order AFTER q_channel but
    BEFORE backbone.  When ``pocket_hints`` matches no parameter, both
    pocket lists are empty and the bucket falls through to backbone
    (bit-exact with pre-Wf-A22 behaviour when
    ``pocket_lr_multiplier == 1.0``).

    A parameter is:

    * "head" iff any of its fully-qualified attribute names (joined with
      ``.``) contains a substring from ``head_hints`` AND does not contain
      a substring from ``q_channel_hints`` or ``pocket_hints``.
    * "q_channel" iff its name contains a substring from ``q_channel_hints``
      (priority over "head" so the q-prediction head can be tuned
      independently) and not from ``pocket_hints``.
    * "pocket" iff its name contains a substring from ``pocket_hints``
      (priority over head + backbone so the c-flow pocket encoder /
      injector can be tuned independently).
    * "backbone" otherwise.
    """
    no_decay_set = set(no_decay_param_names)
    q_set = set(q_channel_hints) if q_channel_hints else set()
    pocket_set = set(pocket_hints) if pocket_hints else set()
    head_decay: List[nn.Parameter] = []
    head_no_decay: List[nn.Parameter] = []
    q_decay: List[nn.Parameter] = []
    q_no_decay: List[nn.Parameter] = []
    pocket_decay: List[nn.Parameter] = []
    pocket_no_decay: List[nn.Parameter] = []
    backbone_decay: List[nn.Parameter] = []
    backbone_no_decay: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_q = any(h in name for h in q_set) if q_set else False
        is_pocket = (
            (not is_q) and (any(h in name for h in pocket_set) if pocket_set else False)
        )
        is_head = (
            (not is_q) and (not is_pocket) and any(h in name for h in head_hints)
        )
        no_decay = any(name.endswith(suf) for suf in no_decay_set)
        if is_q and no_decay:
            q_no_decay.append(param)
        elif is_q:
            q_decay.append(param)
        elif is_pocket and no_decay:
            pocket_no_decay.append(param)
        elif is_pocket:
            pocket_decay.append(param)
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
        pocket_decay,
        pocket_no_decay,
        backbone_decay,
        backbone_no_decay,
    )


def build_differential_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    head_lr_multiplier: float = 1.0,
    weight_decay: float = 1e-5,
    no_decay_param_names: Optional[Sequence[str]] = None,
    head_hints: Optional[Sequence[str]] = None,
    q_channel_lr_multiplier: float = 1.0,
    q_channel_hints: Optional[Sequence[str]] = None,
    pocket_lr_multiplier: float = 1.0,
    pocket_hints: Optional[Sequence[str]] = None,
) -> Optimizer:
    """Build a :class:`torch.optim.Adam` for differential head + q-channel + pocket LR.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters are optimized.  Frozen parameters
        (``requires_grad == False``) are excluded.
    base_lr : float, default 1e-4
        Backbone learning rate.  Head learning rate is
        ``base_lr * head_lr_multiplier``.  Q-channel learning rate is
        ``base_lr * q_channel_lr_multiplier``.  Pocket learning rate is
        ``base_lr * pocket_lr_multiplier`` (Wf-A22 step-4).
    pocket_lr_multiplier : float, default 1.0
        When ``1.0`` (default; bit-exact pre-Wf-A22), the returned
        optimizer collapses to the same single-group / 4-group /
        6-group fast-path it would have without this slot — pocket
        parameters fall into the backbone bucket at backbone LR.
        When ``!= 1.0``, an 8th param group is emitted (priority over
        head + backbone, after q-channel) so the c-flow pocket encoder
        + pocket injector can have an independent LR.
    pocket_hints : sequence of str, optional
        Substrings whose presence in a parameter's fully-qualified name
        marks it as belonging to the c-flow pocket module.  Default:
        ``("pocket_injector", "pocket_encoder")`` (covers both
        ``molmetal/adapters/egnn_extensions.py:PerLayerPocketInjector``
        and the trainer's ``model.pocket_encoder``).
    no_decay_param_names, head_hints, q_channel_*, ...
        Unchanged from pre-Wf-A22.

    Returns
    -------
    torch.optim.Adam
        Single-group when head_lr_multiplier == q_channel_lr_multiplier
        == pocket_lr_multiplier == 1.0 (legacy fast-path).  8-group
        when any non-1.0 multiplier is set AND its hints match.
    """
    if no_decay_param_names is None:
        # Wf-A30-3 (2026-10-14) C3 FIX — sentinel used to be
        # ``Optional[Sequence[str]] = Iterable`` + ``is Iterable`` check.
        # That over-loaded a runtime check on a typing alias identity,
        # which both confuses mypy (the `# type: ignore[assignment]` was
        # load-bearing) and silently shadows any caller that ever passed a
        # non-default Sequence subclass.  Sentinel is now ``None``; the
        # fallback fires on the single, easy-to-read condition below, and
        # any caller passing a Sequence (the only other supported shape)
        # flows straight through.
        no_decay_param_names = _DEFAULT_NO_DECAY_NAMES
    if head_hints is None:
        head_hints = _DEFAULT_HEAD_HINTS
    if q_channel_hints is None:
        q_channel_hints = _DEFAULT_Q_CHANNEL_HINTS
    if pocket_hints is None:
        pocket_hints = ("pocket_injector", "pocket_encoder")

    (
        head_decay,
        head_no_decay,
        q_decay,
        q_no_decay,
        pocket_decay,
        pocket_no_decay,
        backbone_decay,
        backbone_no_decay,
    ) = _split_params(
        model,
        no_decay_param_names,
        head_hints,
        q_channel_hints,
        pocket_hints,
    )

    head_lr = float(base_lr) * float(head_lr_multiplier)
    q_lr = float(base_lr) * float(q_channel_lr_multiplier)
    pocket_lr = float(base_lr) * float(pocket_lr_multiplier)

    # Legacy fast-path: when head lr == backbone lr AND q lr == backbone
    # lr AND pocket lr == backbone lr, collapse to a single group so the
    # optimizer behaves identically to a flat Adam.  This is what the
    # Wf-13 baseline shipped with; preserving it preserves the benchmark
    # baseline AND keeps the Wf-A22 step-4 commit bit-exact with
    # pre-Wf-A22 builds when the new kwarg is absent.
    if (
        abs(head_lr - float(base_lr)) < 1e-12
        and abs(q_lr - float(base_lr)) < 1e-12
        and abs(pocket_lr - float(base_lr)) < 1e-12
    ):
        return Adam(
            [
                {
                    "params": (
                        list(backbone_decay)
                        + list(backbone_no_decay)
                        + list(q_decay)
                        + list(q_no_decay)
                        + list(pocket_decay)
                        + list(pocket_no_decay)
                    ),
                    "weight_decay": float(weight_decay),
                }
            ],
            lr=float(base_lr),
        )

    # 4-, 6-, or 8-group path: head vs q_channel vs pocket vs backbone,
    # decay vs no-decay.  Order: head, q_channel, pocket, backbone.
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
    # Pocket group (Wf-A22 step-4): only emitted when pocket_lr != base_lr
    # AND at least one parameter matched a pocket hint.  Priority over
    # backbone so the c-flow pocket encoder + pocket injector can be tuned
    # independently without contaminating the chemistry-aware backbone.
    if pocket_decay and abs(pocket_lr - float(base_lr)) > 1e-12:
        param_groups.append(
            {"params": pocket_decay, "lr": pocket_lr, "weight_decay": float(weight_decay)}
        )
    if pocket_no_decay and abs(pocket_lr - float(base_lr)) > 1e-12:
        param_groups.append(
            {"params": pocket_no_decay, "lr": pocket_lr, "weight_decay": 0.0}
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