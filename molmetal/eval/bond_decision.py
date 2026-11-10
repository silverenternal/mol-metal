"""Single source of truth for the (decision, threshold, class_logit_bias) triplet.

Wf-A35-1 (2026-10-14) C3 — until now the
:class:`molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond.YuelBondBondDecoder`
constructor and :func:`molmetal.scripts.train_cfm_yuelbond_wasserstein.evaluate_bond_accuracy`
both accepted the triplet as separate kwargs, and the two callers in
:mod:`molmetal.eval.cross_dataset` and the trainer each inlined their own
``getattr(args, ...)`` lookups. That layout had two failure modes:

1. ``cross_dataset._predict_smiles`` (eval/cross_dataset.py:198) constructed
   the decoder without passing ``decision`` / ``decision_threshold`` /
   ``class_logit_bias`` at all, so every shipped YAML that set those flags
   silently fell back to the constructor's ``("argmax", 0.5, None)``
   defaults at decode time. The metrics side read the same flags from
   ``args`` directly, so the reported bond F1 and the molecules that were
   actually assembled could come from two different rules.
2. ``evaluate_bond_accuracy`` and the trainer's decoder construction
   (``train_cfm_yuelbond_wasserstein.py:9844``) each spelled the lookup
   differently (``getattr(args, "bond_decision", "argmax")`` vs.
   ``str(_bond_state.get(...))`` vs. ``_aromatic_bias_vector(args)``),
   so a future rename of any flag would have to be applied in N places
   to stay consistent.

The helper centralises the lookup so both code paths read the same triplet
from the same kind of state object, and a future flag rename is one diff.

API
---

>>> bond_decision_settings({"bond_decision": "threshold",
...                         "bond_decision_threshold": 0.3,
...                         "class_logit_bias": [0, 0, 0, 0, 0.5]})
("threshold", 0.3, [0.0, 0.0, 0.0, 0.0, 0.5])

>>> bond_decision_settings({})   # missing keys → canonical defaults
("argmax", 0.5, None)

>>> bond_decision_settings({"args": {"bond_decision": "threshold", ...}})  # ckpt layout
("threshold", 0.5, None)

Bit-exact default contract (~~§5.11~~ → AGENTS.md §3.3 audit-ledger; corrected 2026-10-31)
-------------------------------------------------------------------------------------------
The canonical defaults are ``("argmax", 0.5, None)``. At those defaults the
helper is the identity for the historical call sites:

* :func:`molmetal.scripts.train_cfm_yuelbond_wasserstein.evaluate_bond_accuracy`
  accepted ``decision="argmax"``, ``threshold=0.5`` as its default kwargs; with
  no ``class_logit_bias`` key on the live state the helper returns ``None``
  exactly like ``_aromatic_bias_vector(args)`` does when the user did not set
  ``--bond-aromatic-logit-bias``.
* :class:`YuelBondBondDecoder` accepts ``decision="argmax"``,
  ``decision_threshold=0.5``, ``class_logit_bias=None`` as its default kwargs;
  every shipped CFM YAML that does not opt in to
  ``flow.bond_decision="threshold"`` therefore produces bit-identical output.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

# Trainer-level canonical defaults — fixed; do not change without a
# Wf-XXX retrain and a ~~§5.11~~ → AGENTS.md §3.3 audit-ledger
# bit-exact-default check across every shipped YAML. (corrected 2026-10-31)
DEFAULT_DECISION: str = "argmax"
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_CLASS_LOGIT_BIAS: Optional[Sequence[float]] = None

# State layouts the helper accepts.
# 1. ``argparse.Namespace`` — the trainer's ``args``.
# 2. ``dict`` — a flat dict of argparse values (the Wf-356 probe passes
#    one of these to the decoder construction site).
# 3. ``dict`` with an ``"args"`` sub-dict — the full ckpt layout as
#    returned by ``molmetal.eval.cross_dataset._load_pretrained_checkpoint``;
#    the helper resolves the triplet from the inner ``args``.


def _flat_lookup(state: Any, key: str, default: Any) -> Any:
    """Look ``key`` up on an argparse Namespace, dict-like, or any object.

    Tolerates both ``state["bond_decision"]`` (dict) and
    ``state.bond_decision`` (Namespace).  Returns ``default`` when the
    key is missing or the value is ``None``.
    """
    if state is None:
        return default
    try:
        v = state[key]  # type: ignore[index]
    except (KeyError, TypeError):
        try:
            v = getattr(state, key)
        except AttributeError:
            return default
    if v is None:
        return default
    return v


def bond_decision_settings(
    state: Any,
) -> Tuple[str, float, Optional[Sequence[float]]]:
    """Read the ``(decision, threshold, class_logit_bias)`` triplet.

    Parameters
    ----------
    state
        Any of: an ``argparse.Namespace`` (``args``), a flat ``dict``
        mirroring one, or a full ckpt ``dict`` that carries an
        ``"args"`` sub-dict. Unknown layouts degrade to the canonical
        defaults, which matches the historical behaviour of every call
        site.

    Returns
    -------
    (decision, threshold, class_logit_bias) : tuple
        ``decision`` is always ``"argmax"`` or ``"threshold"``; any other
        value the caller supplied passes through verbatim and is rejected
        later by :class:`YuelBondBondDecoder`. ``threshold`` is cast to
        ``float``; ``class_logit_bias`` is cast to a list of ``float``
        when present, or left as ``None`` for the no-op path.
    """
    # Trainr ckpt layout: {"args": {...}, "model_state_dict": ..., ...}.
    # Fall through to a flat lookup if "args" is missing or not a dict.
    if isinstance(state, dict) and isinstance(state.get("args"), dict):
        inner = state["args"]
    else:
        inner = state
    decision = str(_flat_lookup(inner, "bond_decision", DEFAULT_DECISION))
    threshold = float(_flat_lookup(inner, "bond_decision_threshold", DEFAULT_THRESHOLD))
    bias_raw = _flat_lookup(inner, "class_logit_bias", DEFAULT_CLASS_LOGIT_BIAS)
    if bias_raw is None:
        bias_out: Optional[Sequence[float]] = None
    else:
        bias_out = [float(x) for x in bias_raw]
    return decision, threshold, bias_out


__all__ = ["bond_decision_settings", "DEFAULT_DECISION", "DEFAULT_THRESHOLD", "DEFAULT_CLASS_LOGIT_BIAS"]
