#!/usr/bin/env python
"""Wf-A80 (2026-10-18) — load the trainer's per-class bond thresholds.

WHY THIS EXISTS
---------------
The trainer has calibrated one decision threshold per bond class since
Wf-A70 (bond_class_k_best_threshold, on a held-out fraction of the
validation pairs) and reports the result as
bond_per_class_test_macro_f1.  On the Wf-A75a runs that number is
0.457 against a pooled 0.219 -- the per-class calibration is worth more
than a factor of two in macro F1.

None of it reached inference.  The decoder took one global threshold on
P(bonded), so the classes with the measured PR-AUC deficit (SINGLE 0.38
vs a 0.7 target, DOUBLE 0.055 vs 0.10) were all decided by the same bar
as AROMATIC, which is the one class that is not rare.

This module is the missing link: it reads the thresholds the trainer
already writes into per_epoch_metrics.json and hands them to
bond_decision(..., decision="per_class").  Nothing here changes a
default: a decoder built without thresholds behaves exactly as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

N_CLASSES = 5


def thresholds_from_metrics(
    metrics: Dict[str, Any],
    epoch: Optional[int] = None,
) -> List[float]:
    """Extract per-class thresholds from a per_epoch_metrics payload.

    metrics: either the full JSON (with a per_epoch_metrics list) or a
        single epoch record.
    epoch: which epoch to read.  None -> the last epoch (the most
        recently calibrated).  An integer selects by epoch value.

    Missing classes fall back to 0.5 so the returned vector always has
    one entry per BOND_LABELS class; a payload with NO calibration at
    all raises, because silently returning five 0.5s would look like a
    calibrated decoder and decode with the global rule.
    """
    if "per_epoch_metrics" in metrics:
        rows = metrics["per_epoch_metrics"]
        if not rows:
            raise ValueError("per_epoch_metrics is empty; nothing to calibrate from")
        row = rows[-1] if epoch is None else next(
            (r for r in rows if int(r.get("epoch", -1)) == int(epoch)), None
        )
        if row is None:
            raise ValueError(f"epoch {epoch!r} not present in per_epoch_metrics")
    else:
        row = metrics

    found = {
        k: float(row[f"bond_class_{k}_best_threshold"])
        for k in range(N_CLASSES)
        if f"bond_class_{k}_best_threshold" in row
    }
    if not found:
        raise ValueError(
            "no bond_class_*_best_threshold fields in this metrics payload; "
            "per-class calibration was not run for this config"
        )
    return [found.get(k, 0.5) for k in range(N_CLASSES)]


def load_per_class_thresholds(
    path: str | Path,
    epoch: Optional[int] = None,
) -> List[float]:
    """Read per-class thresholds from a per_epoch_metrics.json on disk."""
    payload = json.loads(Path(path).read_text())
    return thresholds_from_metrics(payload, epoch=epoch)


def calibration_from_decoder(decoder: Any, epoch: Optional[int] = None) -> Dict[str, Any]:
    """Serialise the decision rule a decoder is ACTUALLY using.

    Wf-A81 (2026-10-18) — this is the training half of the train->inference
    hand-off.  The trainer already updates ``decoder.per_class_thresholds``
    every epoch before gen_eval, but the checkpoint carried only
    ``args`` (which says ``bond_decision: per_class`` and no thresholds), so
    a reader that loaded the .pt and rebuilt a decoder got the argparse
    default -- argmax -- while the run's own metrics were produced under a
    different rule.  Stamping the resolved rule into the checkpoint means
    the artifact that carries the weights carries the way to read them.
    """
    t = getattr(decoder, "per_class_thresholds", None)
    cal: Dict[str, Any] = {
        "decision": str(getattr(decoder, "decision", "argmax")),
        "decision_threshold": float(
            getattr(decoder, "decision_threshold", 0.5)
        ),
        "per_class_thresholds": (
            [float(x) for x in t] if t is not None else None
        ),
    }
    bias = getattr(decoder, "class_logit_bias", None)
    if bias is not None:
        try:
            cal["class_logit_bias"] = [float(x) for x in bias]
        except TypeError:
            cal["class_logit_bias"] = None
    if epoch is not None:
        cal["source_epoch"] = int(epoch)
    return cal


def decoder_kwargs_from_checkpoint(ckpt: Any) -> Dict[str, Any]:
    """Turn a checkpoint's ``decode_calibration`` into decoder kwargs.

    Returns ``{}`` for any checkpoint that predates the stamp, so every
    existing artifact keeps loading with the decoder's own defaults.  For a
    ``per_class`` stamp the thresholds are REQUIRED: handing a per_class
    decoder no thresholds raises at decode time, so a malformed stamp falls
    back to ``{}`` instead of producing a decoder that cannot run.
    """
    if not isinstance(ckpt, dict):
        return {}
    cal = ckpt.get("decode_calibration")
    if not isinstance(cal, dict) or not cal:
        return {}
    decision = str(cal.get("decision") or "argmax")
    if decision not in ("argmax", "threshold", "per_class"):
        return {}
    kw: Dict[str, Any] = {"decision": decision}
    if cal.get("decision_threshold") is not None:
        kw["decision_threshold"] = float(cal["decision_threshold"])
    if decision == "per_class":
        t = cal.get("per_class_thresholds")
        if not isinstance(t, (list, tuple)) or len(t) != N_CLASSES:
            return {}
        kw["per_class_thresholds"] = [float(x) for x in t]
    bias = cal.get("class_logit_bias")
    if isinstance(bias, (list, tuple)) and len(bias) == N_CLASSES:
        kw["class_logit_bias"] = [float(x) for x in bias]
    return kw

