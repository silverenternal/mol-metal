"""wetlab_extensions — TODO-30 P5.2 Tier 1 EXTENSIONS to wet-lab plumbing.

============================================================
TODO-30 P5.2 Rank-15 extensions (2026-09-26)
============================================================
Adds 5 lightweight, CPU-only extensions on top of the R17 Tier-1 wet-lab
plumbing (:mod:`wetlab_protocol` + :mod:`wetlab_reward_channel`):

1. :class:`WetLabSchedulingHelper`      — capacity-aware assay scheduling.
2. :class:`CostEstimator`               — volume-discount cost projection.
3. :class:`CollaboratorTracker`         — in-memory collaborator registry.
4. :class:`AssayDataParser`             — CSV + JSON assay-feed parsers.
5. :class:`WetLabFeedbackIntegrator`    — feedback → reward recalibration.

Design contract (inherited from R17)
------------------------------------
* **CPU-only** — no GPU / no network / no external chemistry facility.
* **Opt-in** — every helper degrades gracefully on missing data.
* **No mutation of R17 modules** — :mod:`wetlab_protocol` and
  :mod:`wetlab_reward_channel` are untouched.  This module only
  *reads* from them via the published public surface.
* **Honest framing** — every numeric estimator (cost, scheduling,
  calibration delta) is a *projection*; the wet-lab channel's
  reward bit-for-bit contract is unchanged.

These helpers are deliberately small + import-safe — they live
inside the :mod:`lam_chem` package so the lambda-calculus chemistry
stack can pull them in without touching :mod:`molmetal.scripts`.

Public surface
--------------
* :class:`WetLabSchedulingHelper`      + :meth:`schedule_assays`
* :class:`CostEstimator`               + :meth:`estimate_total_cost`
* :class:`CollaboratorTracker`         + :meth:`add_collaborator`,
  :meth:`list_collaborators`
* :class:`AssayDataParser`             + :meth:`parse_csv_assay_data`,
  :meth:`parse_json_assay_data`
* :class:`WetLabFeedbackIntegrator`    + :meth:`integrate_assay_results`
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


__all__ = [
    "WetLabSchedulingHelper",
    "CostEstimator",
    "CollaboratorTracker",
    "AssayDataParser",
    "WetLabFeedbackIntegrator",
]
__version__ = "0.0.1-wetlab-extensions"


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class WetLabSchedulingHelper:
    """Capacity-aware assay scheduler.

    Parameters
    ----------
    assay_capacity_per_week : int, default 10
        How many assays the wet-lab cohort can run per ISO week.
        Used by :meth:`schedule_assays` to bucket the priority list
        into weeks.

    Notes
    -----
    * The scheduler is *greedy* — assays are scheduled in the order
      they appear in the priority list, higher-priority items first.
      Each item is placed in the earliest week that still has
      capacity.  No re-balancing / no LPT / no relaxation — this is
      a Tier-1 helper, not a full ILP solver.
    * Each item in the priority list must be a ``dict`` with at
      least a ``"smiles"`` (or ``"assay_id"``) key and a
      ``"priority"`` numeric key (``"high"``, ``"medium"``, ``"low"``
      string is also accepted and mapped to 1.0, 0.5, 0.1).
    * Items with unknown priority map to ``priority=0.0`` and are
      scheduled last.
    """

    _PRIORITY_MAP: Dict[str, float] = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.1,
    }

    def __init__(self, assay_capacity_per_week: int = 10) -> None:
        if assay_capacity_per_week is None:
            assay_capacity_per_week = 10
        try:
            n = int(assay_capacity_per_week)
        except (TypeError, ValueError):
            n = 10
        if n < 1:
            n = 1
        self.assay_capacity_per_week = n

    @staticmethod
    def _priority_key(item: Mapping[str, Any]) -> float:
        """Extract the numeric priority from an item; high > medium > low."""
        if not isinstance(item, Mapping):
            return 0.0
        p = item.get("priority", 0.0)
        if isinstance(p, (int, float)):
            try:
                pf = float(p)
                if pf != pf:  # NaN
                    return 0.0
                return pf
            except (TypeError, ValueError):
                return 0.0
        if isinstance(p, str):
            return WetLabSchedulingHelper._PRIORITY_MAP.get(p.strip().lower(), 0.0)
        return 0.0

    @staticmethod
    def _item_label(item: Mapping[str, Any]) -> str:
        """Return a stable label for the scheduled item."""
        if not isinstance(item, Mapping):
            return ""
        for key in ("smiles", "assay_id", "id"):
            v = item.get(key)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    def schedule_assays(
        self,
        priority_list: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bucket ``priority_list`` into weeks respecting capacity.

        Returns
        -------
        list[dict]
            One dict per scheduled assay with fields::

                {
                  "label": str,           # smiles / assay_id / id
                  "week": int,            # 1-indexed ISO-week bucket
                  "priority": float,      # numeric priority after mapping
                  "raw": dict,            # original input row
                }

            The list is sorted by ``(week, -priority, label)`` so the
            output is deterministic regardless of input ordering
            quirks.  When the priority list has more items than the
            cumulative capacity, the surplus items are still returned
            with their bucket number (degraded but never silently
            dropped — the caller can inspect ``week`` to detect
            overflow).
        """
        if not priority_list:
            return []

        cap = max(1, int(self.assay_capacity_per_week))
        enriched: List[Dict[str, Any]] = []
        for idx, item in enumerate(priority_list):
            if not isinstance(item, Mapping):
                continue
            enriched.append(
                {
                    "_idx": idx,
                    "label": self._item_label(item),
                    "priority": self._priority_key(item),
                    "raw": dict(item),
                }
            )

        # Sort by (-priority, original_index) — stable.
        enriched.sort(key=lambda x: (-x["priority"], x["_idx"]))

        out: List[Dict[str, Any]] = []
        week_counts: Dict[int, int] = defaultdict(int)
        for row in enriched:
            # Find the earliest week with capacity.
            week = 1
            while week_counts[week] >= cap:
                week += 1
            week_counts[week] += 1
            out.append(
                {
                    "label": row["label"],
                    "week": week,
                    "priority": float(row["priority"]),
                    "raw": row["raw"],
                }
            )

        out.sort(key=lambda x: (x["week"], -x["priority"], x["label"]))
        return out


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


class CostEstimator:
    """Project wet-lab cost with a simple volume-discount curve.

    Parameters
    ----------
    cost_per_assay_usd : float, default 200
        Baseline per-assay cost in USD.  Volume discounts kick in
        above 50 / 200 / 1000 assays (5 % / 15 % / 30 %).

    Notes
    -----
    The discount curve is hard-coded and **intentionally simple** —
    it is a Tier-1 projection, not a procurement-grade quote.  Real
    Tier-3 outreach would override the discount brackets from the
    collaborator agreement file.
    """

    _DISCOUNT_BANDS: List[tuple] = [
        (0, 0.0),
        (50, 0.05),
        (200, 0.15),
        (1000, 0.30),
    ]

    def __init__(self, cost_per_assay_usd: float = 200.0) -> None:
        try:
            c = float(cost_per_assay_usd)
        except (TypeError, ValueError):
            c = 200.0
        if c < 0.0 or c != c:  # NaN guard
            c = 200.0
        self.cost_per_assay_usd = c

    @classmethod
    def _discount_for(cls, n_assays: int) -> float:
        """Return the active discount fraction for the given volume."""
        if n_assays < 0:
            return 0.0
        disc = 0.0
        for threshold, frac in cls._DISCOUNT_BANDS:
            if n_assays >= threshold:
                disc = frac
        return disc

    def estimate_total_cost(self, n_assays: int) -> float:
        """Return ``n_assays * cost_per_assay * (1 - discount)``.

        Parameters
        ----------
        n_assays : int
            Number of assays in the cohort.  Negative values are
            clamped to 0 (cost is 0.0).

        Returns
        -------
        float
            Total cost in USD, rounded to 2 decimals.  ``0.0`` when
            ``n_assays <= 0``.
        """
        try:
            n = int(n_assays)
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            return 0.0
        disc = self._discount_for(n)
        return round(float(n) * self.cost_per_assay_usd * (1.0 - disc), 2)


# ---------------------------------------------------------------------------
# Collaborator tracker
# ---------------------------------------------------------------------------


class CollaboratorTracker:
    """In-memory registry of wet-lab collaborators.

    Notes
    -----
    * Pure data structure — no DB / no API calls.  Tier-3 outreach
      would persist this to disk via a CSV export.
    * Each collaborator is stored as a dict with at minimum
      ``name``, ``institution``, ``expertise``.
    """

    def __init__(self) -> None:
        self._collaborators: List[Dict[str, str]] = []

    def add_collaborator(
        self,
        name: str,
        institution: str,
        expertise: str,
        *,
        email: str = "",
        notes: str = "",
    ) -> Dict[str, str]:
        """Register a collaborator.  Returns the stored record.

        Silently no-ops on empty ``name`` (returns ``{}``) so the
        caller can iterate a CSV without crashing on blank rows.
        """
        if not name or not isinstance(name, str):
            return {}
        record = {
            "name": name.strip(),
            "institution": str(institution or "").strip(),
            "expertise": str(expertise or "").strip(),
            "email": str(email or "").strip(),
            "notes": str(notes or "").strip(),
        }
        # Dedup on (name, institution) — same person from a new lab
        # would be added again with their new affiliation.
        for existing in self._collaborators:
            if (
                existing["name"] == record["name"]
                and existing["institution"] == record["institution"]
            ):
                # Merge in any non-empty new fields.
                for k, v in record.items():
                    if v and not existing.get(k):
                        existing[k] = v
                return dict(existing)
        self._collaborators.append(record)
        return dict(record)

    def list_collaborators(self) -> List[Dict[str, str]]:
        """Return a *copy* of the registry.  Mutation by callers is safe."""
        return [dict(r) for r in self._collaborators]

    def __len__(self) -> int:
        return len(self._collaborators)


# ---------------------------------------------------------------------------
# Assay data parser
# ---------------------------------------------------------------------------


class AssayDataParser:
    """CSV + JSON parsers for wet-lab assay feeds.

    Notes
    -----
    * Both parsers return ``List[Dict[str, str]]`` — the same shape
      :func:`wetlab_protocol.load_assays` accepts via wide schema.
    * CSV: reads ``smiles`` + ``pIC50`` columns (case-insensitive).
      Extra columns are preserved verbatim.
    * JSON: accepts either a list-of-dicts shape (``[ {smiles, pIC50, ...} ]``)
      or a dict-of-dicts keyed by SMILES (``{smiles: {pIC50, ...}}``).
    """

    @staticmethod
    def parse_csv_assay_data(path: str) -> List[Dict[str, str]]:
        """Read a CSV file of wet-lab measurements.

        Returns an empty list on missing / unreadable files so the
        caller can safely chain the result through downstream
        pipelines.
        """
        out: List[Dict[str, str]] = []
        try:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if not row:
                        continue
                    cleaned = {
                        str(k).strip(): str(v).strip()
                        for k, v in row.items()
                        if k is not None
                    }
                    if any(cleaned.values()):
                        out.append(cleaned)
        except (OSError, UnicodeDecodeError, csv.Error):
            return []
        return out

    @staticmethod
    def parse_json_assay_data(path: str) -> List[Dict[str, str]]:
        """Read a JSON file of wet-lab measurements.

        Supports two on-disk shapes:

        1. ``List[Dict]`` — each row is a self-contained record.
        2. ``Dict[str, Dict]`` — keyed by SMILES; the SMILES is
           injected into the row as ``smiles`` for shape parity with
           the list-of-dicts form.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []

        out: List[Dict[str, str]] = []
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                out.append({str(k): str(v) for k, v in row.items()})
            return out
        if isinstance(payload, dict):
            for smi, row in payload.items():
                if not isinstance(row, dict):
                    continue
                rec = {str(k): str(v) for k, v in row.items()}
                rec.setdefault("smiles", str(smi))
                out.append(rec)
            return out
        # Scalar / unrecognised shape — graceful empty.
        return []


# ---------------------------------------------------------------------------
# Feedback integrator
# ---------------------------------------------------------------------------


class WetLabFeedbackIntegrator:
    """Bridge wet-lab results into a reward-channel recalibration delta.

    Parameters
    ----------
    recalibrate_fn : Callable
        The recalibration function (typically
        :func:`molmetal.scripts.recalibrate_from_assay.main` or a
        test double).  The integrator calls
        ``recalibrate_fn(results)`` and derives a calibration delta
        from the returned report.

    Notes
    -----
    The "calibration delta" is defined as the absolute change in the
    reward-channel weight (``w_wetlab``) implied by the report's
    ``recommendation`` field:

    * ``"OK"``                              → ``+0.0`` (no change)
    * ``"RE-TRAIN with weight..."``         → ``+0.5``
    * ``"RE-TRAIN with bias-correction..."``→ ``+1.0``
    * ``"MARGINAL"``                        → ``+0.25``
    * any other recommendation              → ``+0.0`` (no automatic promotion)

    The delta is a *diagnostic* — the integrator NEVER mutates the
    aggregator's weight on its own.  The user must opt in via the
    next :func:`register_wetlab_channel` call.  This mirrors the
    "no-silent-promotion" contract used everywhere else.
    """

    _DELTA_BY_RECOMMENDATION: Dict[str, float] = {
        "ok": 0.0,
        "marginal": 0.25,
        "re-train with weight": 0.5,
        "re-train with bias-correction": 1.0,
        "re-train with re-weight": 0.5,
    }

    def __init__(
        self,
        recalibrate_fn: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    ) -> None:
        if not callable(recalibrate_fn):
            raise TypeError("recalibrate_fn must be callable")
        self.recalibrate_fn = recalibrate_fn

    @classmethod
    def _delta_from_report(cls, report: Mapping[str, Any]) -> float:
        """Map a recalibration report to a calibration delta."""
        if not isinstance(report, Mapping):
            return 0.0
        rec = report.get("recommendation", "")
        if not isinstance(rec, str):
            return 0.0
        rec_l = rec.strip().lower()
        for prefix, delta in cls._DELTA_BY_RECOMMENDATION.items():
            if rec_l.startswith(prefix):
                return float(delta)
        return 0.0

    def integrate_assay_results(
        self,
        results: Sequence[Mapping[str, Any]],
    ) -> float:
        """Run ``recalibrate_fn(results)`` and return the calibration delta.

        Parameters
        ----------
        results : Sequence[Mapping]
            List of wet-lab rows (e.g. output of
            :meth:`AssayDataParser.parse_csv_assay_data`).

        Returns
        -------
        float
            The calibration delta (>= 0.0).  Always 0.0 when the
            recalibration report does not yield a recommendation we
            recognise.
        """
        try:
            report = self.recalibrate_fn(results)
        except Exception:
            return 0.0
        delta = self._delta_from_report(report)
        # NaN / negative guard.
        try:
            if delta != delta or delta < 0.0:  # NaN check
                return 0.0
        except Exception:
            return 0.0
        return float(delta)


# ---------------------------------------------------------------------------
# Backwards-compat aliases (in case downstream code expected them).
# ---------------------------------------------------------------------------

try:
    SchedulingHelper = WetLabSchedulingHelper  # type: ignore[misc,assignment]
except NameError:  # pragma: no cover
    SchedulingHelper = None  # type: ignore[assignment]
