"""Drift guard — retire injected samples whose on-policy hardness diverges.

================================================================
Drift guard for RSI curriculum learning (TODO-150 §P1.7)
================================================================
After the predictor is re-evaluated every 5 epochs, each sample still
in the injection set has two hardness numbers:

* **predicted** — what the offline MLP said at injection time.
* **on_policy** — what the *current* trainer reports when the
  sample is fed through the live pipeline.

If ``|on_policy - predicted| > DRIFT_DIVERGENCE_THRESHOLD`` (0.5),
the sample is **retired** from the injection set.  This is the
mechanism that prevents the predictor from silently poisoning the
cohort with samples it has not learned well.

The guard is intentionally cheap (one float compare per sample); the
``DriftReport`` returned by :meth:`DriftGuard.evaluate` is JSON-safe
so it can be persisted to ``metrics/by_metric/rsi_curriculum/<metal>/
drift_report_epoch_<k>.json`` for the audit ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .hardness import DRIFT_DIVERGENCE_THRESHOLD


@dataclass(frozen=True)
class DriftReport:
    """Summary of one drift evaluation."""

    metal: str
    epoch: int
    n_total: int
    n_retired: int
    n_kept: int
    threshold: float = DRIFT_DIVERGENCE_THRESHOLD
    retired_row_ids: List[int] = field(default_factory=list)
    kept_row_ids: List[int] = field(default_factory=list)

    @property
    def retire_fraction(self) -> float:
        """Convenience: retired / total.  Zero when n_total == 0."""
        return float(self.n_retired) / float(self.n_total) if self.n_total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metal": str(self.metal),
            "epoch": int(self.epoch),
            "n_total": int(self.n_total),
            "n_retired": int(self.n_retired),
            "n_kept": int(self.n_kept),
            "threshold": float(self.threshold),
            "retired_row_ids": [int(r) for r in self.retired_row_ids],
            "kept_row_ids": [int(r) for r in self.kept_row_ids],
            "retire_fraction": self.retire_fraction,
        }


class DriftGuard:
    """Per-sample retirement decision based on the divergence threshold."""

    def __init__(
        self,
        threshold: float = DRIFT_DIVERGENCE_THRESHOLD,
    ) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0; got {threshold}")
        self.threshold = float(threshold)

    def evaluate(
        self,
        metal: str,
        epoch: int,
        samples: Iterable[Dict[str, Any]],
    ) -> DriftReport:
        """Evaluate drift for a batch of samples.

        ``samples`` is an iterable of dicts each carrying::

            {"row_id": int, "predicted": float, "on_policy": float}

        Returns a :class:`DriftReport` listing retired and kept ids.
        """
        metal = str(metal)
        epoch = int(epoch)
        retired: List[int] = []
        kept: List[int] = []
        n_total = 0
        for s in samples:
            n_total += 1
            row_id = int(s["row_id"])
            predicted = float(s.get("predicted", 0.0))
            on_policy = float(s.get("on_policy", 0.0))
            if abs(on_policy - predicted) > self.threshold:
                retired.append(row_id)
            else:
                kept.append(row_id)
        return DriftReport(
            metal=metal,
            epoch=epoch,
            n_total=n_total,
            n_retired=len(retired),
            n_kept=len(kept),
            threshold=self.threshold,
            retired_row_ids=retired,
            kept_row_ids=kept,
        )

    def is_retired(self, predicted: float, on_policy: float) -> bool:
        """Single-sample decision — exposed for unit tests / callers."""
        return abs(float(on_policy) - float(predicted)) > self.threshold


__all__ = [
    "DriftGuard",
    "DriftReport",
]