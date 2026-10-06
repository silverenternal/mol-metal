"""Headline metric loader — gate for ``metrics/by_metric/*.json`` headline use.

================================================================
Wf-A11 (2026-10-05) — closes the "headline metric is mis-readable" defect
surfaced by the prior ultracode audit of
``metrics/by_metric/metal_lomo_auc_wasserstein.json``. The audit found 5
issues with that JSON as a Tier-2 headline:

  1. ``rh_oversample=500`` mixes 500 synthetic Rh rows into the TRAIN split
     of the held-out Rh fold (synthetic Rh was designed to balance that
     fold — the OOD-LOMO measurement is contaminated).
  2. ``n_folds=2`` but the JSON was historically referenced as a "5-metal
     LOMO sweep" — actually only Rh + Re were covered.
  3. The cited overall mean (0.7577) cannot be honestly quoted without
     disclosing all three of the above.
  4. ``paper_dashboard.py`` and downstream consumers read the value
     directly without an "as_headline" opt-in.
  5. There was no programmatic gate that prevented ``USABLE_AS_HEADLINE``
     from being True by default.

This module provides a single canonical gate
(:func:`load_headline_metric`) that a caller MUST go through. The gate
RAISES if a JSON file has ``USABLE_AS_HEADLINE: false`` AND the caller
passed ``as_headline=True`` (and did NOT pass ``allow_disqualified=True``).
The exception class (:class:`HeadlineNotUsableError`) carries the
``disqualifying_reasons`` so the caller can surface them.

CPU-only, stdlib-only. Mirror of OmegaFold's small-pure-helper pattern
(no torch, no sklearn). Safe to call from anywhere in the repo.

Usage::

    from molmetal.scripts.headline_metric_loader import (
        load_headline_metric, HeadlineNotUsableError,
    )

    # SAFE: returns dict regardless of USABLE_AS_HEADLINE.
    payload = load_headline_metric(
        "metrics/by_metric/metal_lomo_auc_wasserstein.json",
    )
    print(payload["value"])

    # RAISES: the headline is disqualified; caller did not opt-in.
    try:
        payload = load_headline_metric(
            "metrics/by_metric/metal_lomo_auc_wasserstein.json",
            as_headline=True,
        )
    except HeadlineNotUsableError as e:
        # e.reasons is a list[str]; surface them in any paper / dashboard.
        print("Refusing to use as headline:", e.reasons)

    # OK: explicit allow_disqualified=True bypasses the gate. The caller
    # is then on the hook for honest framing per AGENTS.md §5.2.
    payload = load_headline_metric(
        "metrics/by_metric/metal_lomo_auc_wasserstein.json",
        as_headline=True,
        allow_disqualified=True,
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HeadlineNotUsableError(RuntimeError):
    """Raised when a JSON declares ``USABLE_AS_HEADLINE: false`` and the
    caller asked for it as a headline.

    Attributes
    ----------
    path : str
        Filesystem path to the headline JSON that was refused.
    reasons : list[str]
        The ``disqualifying_reasons`` list from the JSON, surfaced verbatim.
    value : float | None
        The ``value`` field of the JSON (if present), exposed so callers
        can render it in honest-framing outputs that explicitly say
        "this number is disqualified; see reasons".
    """

    def __init__(self, path: str, reasons: list[str], value: Any | None = None) -> None:
        self.path = path
        self.reasons = list(reasons)
        self.value = value
        msg = (
            f"Refusing to use {path} as a headline metric: "
            + "; ".join(reasons)
            + (
                f" (value={value!r} is disqualified; use allow_disqualified=True "
                "with explicit honest-framing, OR re-measure per the JSON's "
                "clean_re_measurement_command field)"
            )
        )
        super().__init__(msg)


def load_headline_metric(
    json_path: str | Path,
    *,
    as_headline: bool = False,
    allow_disqualified: bool = False,
) -> dict[str, Any]:
    """Load a ``metrics/by_metric/*.json`` headline metric with gate.

    Parameters
    ----------
    json_path : str | Path
        Filesystem path to the headline JSON.
    as_headline : bool, default False
        If True, the caller intends to use this number as a Tier-2 headline
        (paper §4 LEAD, dashboard KPI, etc.). The gate enforces that the
        JSON's ``USABLE_AS_HEADLINE`` is True OR ``allow_disqualified=True``.
    allow_disqualified : bool, default False
        Explicit opt-in to use a disqualified number. The caller is then
        on the hook for honest framing per AGENTS.md §5.2.

    Returns
    -------
    dict[str, Any]
        The full JSON payload.

    Raises
    ------
    FileNotFoundError
        If ``json_path`` does not exist.
    HeadlineNotUsableError
        If ``as_headline=True``, ``USABLE_AS_HEADLINE`` is not True, and
        ``allow_disqualified`` is False.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Headline metric JSON not found: {path}")
    with path.open() as f:
        payload = json.load(f)

    if not as_headline:
        return payload

    usable = bool(payload.get("USABLE_AS_HEADLINE", False))
    if usable:
        return payload

    reasons = payload.get("disqualifying_reasons") or [
        "USABLE_AS_HEADLINE is not True and no disqualifying_reasons were declared"
    ]
    if allow_disqualified:
        return payload

    raise HeadlineNotUsableError(
        path=str(path),
        reasons=reasons,
        value=payload.get("value"),
    )


__all__ = [
    "HeadlineNotUsableError",
    "load_headline_metric",
]
