#!/usr/bin/env python3
"""recalibrate_from_assay.py — Pearson r + residual analysis for dry-lab vs wet-lab.

============================================================
TODO-30 P5.2 Tier 1 (2026-09-17)
============================================================
This script reads a wet-lab assay TSV (the same format accepted by
``--wetlab-input`` on ``r4_lambda_only_run.py``) and a parallel dry-lab
prediction TSV, then emits a JSON calibration report:

    {
      "n_assays": 42,
      "n_overlap": 31,
      "pearson_r": 0.612,
      "spearman_rho": 0.587,
      "rmse": 0.487,
      "within_05_log_units": 18,  # |pred - meas| <= 0.5
      "within_10_log_units": 26,
      "max_abs_err": 1.732,
      "recommendation": "OK" | "RE-TRAIN with margin >= 1.0" | "RE-TRAIN with weight += 0.5",
      "residual_regression": { "intercept": 0.31, "slope": -0.18 },
      "report_path": "calibration_report.json"
    }

CLI
---
    python -m molmetal.scripts.recalibrate_from_assay \
        --wetlab-input path/to/assays.tsv \
        --drylab-input path/to/predictions.tsv \
        --output path/to/calibration_report.json

File formats
------------
* ``--wetlab-input``: same TSV as the reward channel (``smiles<TAB>pIC50``
  one per row; wide header auto-detected).
* ``--drylab-input``: same TSV format.  The first column is the SMILES
  and the second is the dry-lab pIC50 prediction.  Wide header is also
  auto-detected (looks for ``smiles`` + ``pIC50_pred`` or ``prediction``
  column names).

Honest framing
--------------
The ``recommendation`` field is a *diagnostic* signal derived from the
residual regression + Pearson r — it is NOT a silently-applied
weight re-tuning.  The script never mutates the aggregator or the
training code; it only emits a JSON file the user can read.  Any
weight adjustment requires the user to opt in via the next
``r4_lambda_only_run.py`` invocation.  This is the same
"no-silent-promotion" contract used by every other workflow
(`wf_p0_metrics_smoke`, `wf_sa_penalty`, etc.).

CPU-only, no GPU dependency, no external API calls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Lightweight stats helpers (no scipy / sklearn dependency)
# ---------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Return Pearson r; ``None`` if the sample is degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 <= 0.0 or dy2 <= 0.0:
        return None
    return float(num / math.sqrt(dx2 * dy2))


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Return Spearman rho via rank-Pearson (no scipy)."""
    if len(xs) < 2:
        return None
    rx = _rank(xs)
    ry = _rank(ys)
    return _pearson(rx, ry)


def _rank(xs: Sequence[float]) -> List[float]:
    """Convert a list of floats to ranks (average rank for ties)."""
    sorted_pairs = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _linear_regression(
    xs: Sequence[float], ys: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (intercept, slope) of an OLS fit; ``None`` if degenerate."""
    n = len(xs)
    if n < 2:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    return float(intercept), float(slope)


def _rmse(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Root-mean-square error between two equally-sized sequences."""
    n = len(xs)
    if n == 0:
        return None
    sse = sum((x - y) ** 2 for x, y in zip(xs, ys))
    return float(math.sqrt(sse / n))


# ---------------------------------------------------------------------------
# TSV loaders
# ---------------------------------------------------------------------------


def _load_pair_tsv(
    path: str,
    *,
    smiles_col: str = "smiles",
    value_col_candidates: Sequence[str] = ("pIC50", "pIC50_pred", "prediction", "pIC50_predicted"),
    sep: str = "\t",
    comment: str = "#",
) -> Dict[str, float]:
    """Load a 2-column TSV into ``{smiles: value}``.

    Mirrors :func:`molmetal_lam.lam_chem.wetlab_protocol.load_assays`
    but is more lenient about the value column name — it accepts the
    canonical ``pIC50`` (wet-lab) OR a ``pIC50_pred`` /
    ``prediction`` (dry-lab prediction) header.
    """
    out: Dict[str, float] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.readlines()
                     if ln.strip() and not ln.lstrip().startswith(comment)]
    except OSError:
        return out
    if not lines:
        return out
    first = lines[0].rstrip("\n").rstrip("\r").split(sep)
    has_header = any(tok.strip().lower() in {smiles_col.lower(), *value_col_candidates}
                     for tok in first)
    if has_header:
        header = [tok.strip().lower() for tok in first]
        try:
            i_smiles = header.index(smiles_col.lower())
        except ValueError:
            return out
        i_value = -1
        for cand in value_col_candidates:
            if cand.lower() in header:
                i_value = header.index(cand.lower())
                break
        if i_value < 0:
            return out
        data_lines = lines[1:]
    else:
        i_smiles = 0
        i_value = 1
        data_lines = lines
    for raw in data_lines:
        toks = raw.rstrip("\n").rstrip("\r").split(sep)
        if len(toks) < 2:
            continue
        smi = toks[i_smiles].strip()
        if not smi:
            continue
        try:
            val = float(toks[i_value].strip())
        except (TypeError, ValueError):
            continue
        if val != val:
            continue
        out[smi] = float(val)
    return out


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def _recommend(
    n_overlap: int,
    pearson: Optional[float],
    rmse: Optional[float],
    intercept: Optional[float],
    slope: Optional[float],
) -> str:
    """Return a human-readable recommendation string.

    Honest framing — this is a *diagnostic* hint, not a silent
    weight re-tune.  The user must opt in via the next
    ``r4_lambda_only_run.py`` invocation.
    """
    if n_overlap < 5:
        return ("INSUFFICIENT_DATA — n_overlap<5; collect more wet-lab "
                "measurements before drawing conclusions")
    if pearson is None:
        return "DEGENERATE_PREDICTIONS — dry-lab predictions are constant; cannot calibrate"
    if rmse is None:
        return "DEGENERATE_MEASUREMENTS — wet-lab measurements are constant; cannot calibrate"
    # Tier 1: OK
    if pearson >= 0.7 and rmse <= 0.5:
        return ("OK — Pearson r >= 0.7 and RMSE <= 0.5 log-units; "
                "the dry-lab predictor is well-calibrated")
    # Tier 2: RE-TRAIN with margin
    if 0.4 <= pearson < 0.7:
        return ("RE-TRAIN with censor-aware margin >= 1.0 — Pearson r in [0.4, 0.7); "
                "the predictor discriminates but the residuals are large. "
                "Use ``--reward-wetlab-weight 0.5`` to let wet-lab evidence "
                "shape the MCTS reward without dominating Vina / SA / QED")
    # Tier 3: RE-TRAIN with weight
    if intercept is not None and abs(intercept) > 1.0:
        return (f"RE-TRAIN with bias-correction — intercept={intercept:+.3f}; "
                "dry-lab predictions have a systematic offset. "
                "Add ``--reward-wetlab-weight 1.0`` to compensate")
    if slope is not None and (slope <= 0.0 or slope >= 1.5):
        return (f"RE-TRAIN with re-weight — slope={slope:+.3f}; "
                "dry-lab predictions are not monotonically proportional to "
                "wet-lab outcomes. ``--reward-wetlab-weight 1.0`` recommended")
    # Default: marginal
    return (f"MARGINAL — Pearson r={pearson:+.3f}, RMSE={rmse:.3f}; "
            "the predictor is informative but not production-grade. "
            "Increase n_assays and re-run before promoting the wet-lab "
            "channel to unit weight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recalibrate_from_assay",
        description=(
            "TODO-30 P5.2 Tier 1 — compute Pearson r + residual "
            "diagnostics between dry-lab predictions and wet-lab "
            "measurements.  Emits a JSON calibration report.  NEVER "
            "mutates the aggregator — all weight changes require the "
            "user to opt in via ``r4_lambda_only_run.py``."
        ),
    )
    parser.add_argument(
        "--wetlab-input",
        type=str,
        required=True,
        help="Path to the wet-lab assay TSV (smiles<TAB>pIC50).",
    )
    parser.add_argument(
        "--drylab-input",
        type=str,
        required=True,
        help="Path to the dry-lab predictions TSV (same format).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_report.json",
        help="Output JSON report path (default: calibration_report.json).",
    )
    parser.add_argument(
        "--within-margin",
        type=float,
        default=0.5,
        help="|err| threshold in log-units for the within_X metric "
             "(default 0.5 per Settles 2012).",
    )
    args = parser.parse_args(argv)

    wetlab = _load_pair_tsv(args.wetlab_input)
    drylab = _load_pair_tsv(args.drylab_input)

    n_assays = len(wetlab)
    n_predictions = len(drylab)

    # Compute the overlap on the intersection of the two dicts.
    overlap = sorted(set(wetlab.keys()) & set(drylab.keys()))
    n_overlap = len(overlap)
    if n_overlap == 0:
        report = {
            "n_assays": n_assays,
            "n_predictions": n_predictions,
            "n_overlap": 0,
            "pearson_r": None,
            "spearman_rho": None,
            "rmse": None,
            "within_margin_count": 0,
            "within_margin_frac": None,
            "max_abs_err": None,
            "recommendation": (
                "NO_OVERLAP — the wet-lab and dry-lab SMILES sets do "
                "not intersect. Check canonicalisation (RDKit) on both sides."
            ),
            "residual_regression": {"intercept": None, "slope": None},
            "report_path": str(args.output),
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    meas = [wetlab[s] for s in overlap]
    pred = [drylab[s] for s in overlap]

    pearson = _pearson(pred, meas)
    spearman = _spearman(pred, meas)
    rmse = _rmse(pred, meas)
    abs_errs = [abs(p - m) for p, m in zip(pred, meas)]
    within_margin_count = sum(1 for e in abs_errs if e <= args.within_margin)
    within_margin_frac = within_margin_count / n_overlap if n_overlap else None
    max_abs_err = max(abs_errs) if abs_errs else None

    # Residual regression: residuals = measured - predicted.
    residuals = [m - p for m, p in zip(meas, pred)]
    intercept, slope = _linear_regression(pred, residuals)

    recommendation = _recommend(n_overlap, pearson, rmse, intercept, slope)

    report = {
        "n_assays": n_assays,
        "n_predictions": n_predictions,
        "n_overlap": n_overlap,
        "pearson_r": (round(pearson, 4) if pearson is not None else None),
        "spearman_rho": (round(spearman, 4) if spearman is not None else None),
        "rmse": (round(rmse, 4) if rmse is not None else None),
        "within_margin_threshold_log_units": args.within_margin,
        "within_margin_count": within_margin_count,
        "within_margin_frac": (
            round(within_margin_frac, 4) if within_margin_frac is not None else None
        ),
        "max_abs_err": (round(max_abs_err, 4) if max_abs_err is not None else None),
        "recommendation": recommendation,
        "residual_regression": {
            "intercept": (round(intercept, 4) if intercept is not None else None),
            "slope": (round(slope, 4) if slope is not None else None),
            "interpretation": (
                "Residual regression is (measured - predicted) ~ a + b*predicted. "
                "Large intercept → systematic offset; slope near 0 → no "
                "proportional bias; slope > 1 or < 0 → under/over-confident "
                "predictor."
            ),
        },
        "report_path": str(args.output),
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
