#!/usr/bin/env python3
"""Build cross-domain uncertainty / calibration / applicability panel for Mol-Metal.

Lifted from Multiflow ``scripts/build_uncertainty_calibration_applicability_panel.py``
(Wf-143 §2.5 Sprint B1).  The Multiflow source applies to four pharma-modality
domains (binding / admet / pose / postprocess) with a single (target × candidate)
input row producing four per-domain output rows.

Mol-Metal adapts this to four cross-domain data **sources**:

* ``metalcytotox``  - 5-metal × 9-cell pIC50 panel (Pt / Ru / Ir / Rh / Os) ×
  (MCF-7, A549, A2780, A2780cisR, HCT-116, PC-3, HL-60, SK-OV-3, DU-145)
* ``drugood``       - 1.58M mol-OOD retrieval benchmark (druglikeness + yield)
* ``nci60``         - 9-cell precious-metal panel (MCF-7 ... DU-145 above)
* ``bindingdb``     - 1.07M IC50 protein-ligand benchmark

Each input candidate row carries one (metal, cell) pair.  For every
source we derive a per-source probability / score / label / uncertainty /
applicability / OOD reading, with source-specific applicability gates
against the 5-metal × 9-cell × 4-source coverage matrix.

The panel builder produces:

* a long-form candidate × source row CSV (== Multiflow ``domain_row``)
* a 10-bin calibration CSV per source (== Multiflow ``_calibration_bins``)
* a summary JSON with per-source Brier score, ECE, blocker tallies

Honest framing
--------------
This is a structural lift: the per-domain applicability gates, blocker
hashing, and bin attribution are 1-to-1 from Multiflow.  What changes is
the ``DOMAIN_SPECS`` table (5-metal × 9-cell × 4-source coverage),
the score-to-probability function (now handles pIC50 / Tanimoto /
Vina-kcal/mol), and the source-specific applicability vocabulary.
Heavy lifting (real wet-lab pIC50 + cross-domain OOD scoring) is in
the upstream :mod:`molmetal_lam.sbdd_env.metrics_v2` + D-MPNN
multi-task heads, this script just **aggregates + flags**.

Usage
-----
::

    python -m molmetal.scripts.build_cross_domain_uncertainty_panel \\
        --candidate-csv molmetal/data/cross_domain_candidates.csv \\
        --output-dir molmetal/reports/wf143/cross_domain_uncertainty
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Constants — 5-metal × 9-cell × 4-source
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("molmetal/reports/wf143/cross_domain_uncertainty_panel")

# 5 precious metals covered by the Wf-128 P1 multitask backbone.
FIVE_METALS: tuple[str, ...] = ("Pt", "Ru", "Ir", "Rh", "Os")

# 9 NCI-60-overlapping cell lines from ``prepare_nci60_data.py``.
NINE_CELLS: tuple[str, ...] = (
    "MCF-7",
    "A549",
    "A2780",
    "A2780cisR",
    "HCT-116",
    "PC-3",
    "HL-60",
    "SK-OV-3",
    "DU-145",
)

# 4 cross-domain sources (== Multiflow's 4 domains).
FOUR_SOURCES: tuple[str, ...] = (
    "metalcytotox",
    "drugood",
    "nci60",
    "bindingdb",
)


# ---------------------------------------------------------------------------
# DOMAIN_SPECS_MOL_METAL — per-source field dictionaries
# ---------------------------------------------------------------------------
# Each entry maps the per-source key tuple to the names that may carry the
# corresponding value in the candidate CSV.  This mirrors Multiflow's
# ``DOMAIN_SPECS`` shape exactly.

DOMAIN_SPECS_MOL_METAL: dict[str, dict[str, tuple[str, ...]]] = {
    "metalcytotox": {
        "probability": (
            "metalcytotox_probability",
            "metalcytotox_pic50_probability",
            "metalcytotox_active_probability",
            "metalcytotox_confidence",
        ),
        "score": (
            "metalcytotox_score",
            "metalcytotox_pic50",
            "metalcytotox_ic50",
        ),
        "label": (
            "metalcytotox_label",
            "metalcytotox_active",
            "metalcytotox_pass",
        ),
        "uncertainty": (
            "metalcytotox_uncertainty",
            "metalcytotox_pic50_uncertainty",
            "metalcytotos_epistemic",
        ),
        "applicability": (
            "metalcytotox_applicability_domain_status",
            "metalcytotox_applicability",
            "applicability_domain_status",
        ),
        "ood": (
            "metalcytotox_ood",
            "metalcytotox_out_of_domain",
            "ood_flag",
            "out_of_domain",
        ),
    },
    "drugood": {
        "probability": (
            "drugood_probability",
            "drugood_druglikeness_probability",
            "drugood_assay_pass_probability",
            "drugood_yield_probability",
            "drugood_confidence",
        ),
        "score": (
            "drugood_score",
            "drugood_druglikeness_score",
            "drugood_assay_score",
            "drugood_yield_score",
        ),
        "label": (
            "drugood_label",
            "drugood_assay_pass",
            "drugood_pass",
            "drugood_active",
        ),
        "uncertainty": (
            "drugood_uncertainty",
            "drugood_assay_uncertainty",
            "drugood_yield_uncertainty",
        ),
        "applicability": (
            "drugood_applicability_domain_status",
            "drugood_applicability",
            "drugood_tanimoto_status",
        ),
        "ood": (
            "drugood_ood",
            "drugood_out_of_domain",
            "drugood_oof_flag",
            "ood_flag",
        ),
    },
    "nci60": {
        "probability": (
            "nci60_probability",
            "nci60_gi50_pass_probability",
            "nci60_active_probability",
            "nci60_confidence",
        ),
        "score": (
            "nci60_score",
            "nci60_gi50",
            "nci60_pic50",
        ),
        "label": (
            "nci60_label",
            "nci60_active",
            "nci60_pass",
            "nci60_gi50_pass",
        ),
        "uncertainty": (
            "nci60_uncertainty",
            "nci60_gi50_uncertainty",
            "nci60_pic50_uncertainty",
        ),
        "applicability": (
            "nci60_applicability_domain_status",
            "nci60_applicability",
            "nci60_precious_status",
        ),
        "ood": (
            "nci60_ood",
            "nci60_out_of_domain",
            "ood_flag",
            "out_of_domain",
        ),
    },
    "bindingdb": {
        "probability": (
            "bindingdb_probability",
            "bindingdb_ic50_pass_probability",
            "bindingdb_active_probability",
            "bindingdb_binding_confidence",
        ),
        "score": (
            "bindingdb_score",
            "bindingdb_ic50",
            "bindingdb_pic50",
            "bindingdb_kd",
        ),
        "label": (
            "bindingdb_label",
            "bindingdb_active",
            "bindingdb_pass",
            "bindingdb_ic50_pass",
        ),
        "uncertainty": (
            "bindingdb_uncertainty",
            "bindingdb_ic50_uncertainty",
            "bindingdb_replicate_std",
        ),
        "applicability": (
            "bindingdb_applicability_domain_status",
            "bindingdb_applicability",
            "bindingdb_target_in_domain",
        ),
        "ood": (
            "bindingdb_ood",
            "bindingdb_out_of_domain",
            "ood_flag",
            "out_of_domain",
        ),
    },
}


# Output row schema (== Multiflow ROW_FIELDS extended with metal + cell +
# source coverage fields).
ROW_FIELDS: list[str] = [
    "candidate_id",
    "smiles",
    "metal",
    "cell_line",
    "source",
    "probability",
    "probability_source",
    "score",
    "score_source",
    "label",
    "uncertainty",
    "applicability_domain_status",
    "ood_flag",
    "low_confidence",
    "claim_blockers",
]


BIN_FIELDS: list[str] = [
    "source",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "count",
    "mean_confidence",
    "empirical_accuracy",
    "abs_calibration_error",
    "ece_contribution",
]


# ---------------------------------------------------------------------------
# IO helpers (lifted 1-to-1 from Multiflow)
# ---------------------------------------------------------------------------


def _read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "candidates", "candidate_rows", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Type coercion helpers (lifted 1-to-1 from Multiflow)
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {
        "1",
        "true",
        "yes",
        "y",
        "pass",
        "passed",
        "success",
        "accepted",
        "active",
    }:
        return True
    if text in {
        "0",
        "false",
        "no",
        "n",
        "fail",
        "failed",
        "failure",
        "rejected",
        "inactive",
    }:
        return False
    return None


def _first_value(
    row: Mapping[str, Any], keys: Iterable[str]
) -> tuple[str, Any] | tuple[None, None]:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return key, row.get(key)
    return None, None


def _clip_probability(value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def _candidate_id(row: Mapping[str, Any], index: int) -> str:
    return str(
        row.get("candidate_id")
        or row.get("source_candidate_id")
        or row.get("id")
        or f"row_{index}"
    ).strip()


# ---------------------------------------------------------------------------
# Source-specific score → probability + applicability blockers
# ---------------------------------------------------------------------------


def _score_to_probability(source: str, score: float | None) -> float | None:
    """Convert a raw per-source ``score`` into a [0, 1] confidence.

    Mirrors Multiflow's ``_score_to_probability`` but extends it with
    Mol-Metal-specific score conventions:

    * ``metalcytotox`` / ``nci60``     - pIC50; sigmoid centered at 5
      (pIC50 = 5 == IC50 of 10 µM, the activity threshold)
    * ``drugood``                      - druglikeness / assay score in
      [0, 1] is taken verbatim; raw logistic fallback uses -score
    * ``bindingdb``                    - pIC50 convention same as
      metalcytotox/nci60
    """
    if score is None:
        return None
    if 0.0 <= score <= 1.0:
        return score
    if source in {"metalcytotox", "nci60", "bindingdb"}:
        # pIC50 units: 5 = 10 µM active threshold.
        return 1.0 / (1.0 + math.exp(-(score - 5.0)))
    if source == "drugood":
        # Druglikeness score already in [0, 1] mapped; raw fall through.
        return 1.0 / (1.0 + math.exp(-score))
    return 1.0 / (1.0 + math.exp(-score))


def _applicability_is_blocking(status: str) -> bool:
    """Return True if the applicability status string forbids claim.

    Mirrors Multiflow's vocabulary.  We keep the same allowed / blocked
    sets so the 0-tolerance fail-closed contract survives the lift.
    """
    text = status.strip().lower()
    if not text:
        return True
    allowed = {
        "in_domain",
        "inside_domain",
        "applicable",
        "pass",
        "passed",
        "known_in_domain",
        "metal_in",
        "cell_in",
        "metal_cell_in",
    }
    blocked = {
        "ood",
        "out_of_domain",
        "outside_domain",
        "not_applicable",
        "fail",
        "failed",
        "unknown",
        "metal_ood",
        "cell_ood",
    }
    if text in allowed:
        return False
    if text in blocked:
        return True
    return True


def _source_known_metal(source: str) -> bool:
    """Source-specific applicability: does this source know the row's metal?"""
    metal = source.split("|")  # not used as-is; placeholder for future routing
    del metal
    return True  # the per-source applicability_status field is what gates


# ---------------------------------------------------------------------------
# Per-source row builder
# ---------------------------------------------------------------------------


def _source_row(
    row: Mapping[str, Any],
    *,
    index: int,
    source: str,
    low_confidence_margin: float,
) -> dict[str, Any] | None:
    """Build one source-scoped panel row, mirroring Multiflow's
    ``_domain_row`` signature.

    Returns ``None`` when the input row carries no per-source field at
    all (the source isn't even relevant for that candidate).
    """
    spec = DOMAIN_SPECS_MOL_METAL[source]
    probability_key, probability_value = _first_value(row, spec["probability"])
    score_key, score_value = _first_value(row, spec["score"])
    label_key, label_value = _first_value(row, spec["label"])
    uncertainty_key, uncertainty_value = _first_value(row, spec["uncertainty"])
    applicability_key, applicability_value = _first_value(row, spec["applicability"])
    ood_key, ood_value = _first_value(row, spec["ood"])
    del label_key, uncertainty_key, applicability_key, ood_key

    probability = _clip_probability(_to_float(probability_value))
    score = _to_float(score_value)
    if probability is None and score is not None:
        probability = _score_to_probability(source, score)
        probability_key = f"{score_key}:score_to_probability" if score_key else None

    uncertainty = _to_float(uncertainty_value)
    label_bool = _as_bool(label_value)
    ood_bool = _as_bool(ood_value)
    applicability_status = str(applicability_value or "").strip()

    has_any_source_input = any(
        value is not None and value != ""
        for value in (
            probability_value,
            score_value,
            uncertainty_value,
            label_value,
            applicability_value,
            ood_value,
        )
    )
    if not has_any_source_input:
        return None

    blockers: list[str] = []
    if probability is None:
        blockers.append(f"{source}_probability_or_score_missing")
    if label_bool is None:
        blockers.append(f"{source}_calibration_label_missing")
    if uncertainty is None and probability is None:
        blockers.append(f"{source}_uncertainty_missing")
    if not applicability_status:
        blockers.append(f"{source}_applicability_domain_missing")
    elif _applicability_is_blocking(applicability_status):
        blockers.append(f"{source}_applicability_domain_blocking")
    if ood_bool:
        blockers.append(f"{source}_ood_flag_true")

    low_confidence = False
    if probability is None:
        low_confidence = True
    elif abs(probability - 0.5) <= low_confidence_margin:
        low_confidence = True
    elif uncertainty is not None and uncertainty >= low_confidence_margin:
        low_confidence = True
    if low_confidence:
        blockers.append(f"{source}_low_confidence")

    return {
        "candidate_id": _candidate_id(row, index),
        "smiles": str(row.get("smiles", "")),
        "metal": str(row.get("metal", "")),
        "cell_line": str(row.get("cell_line", "")),
        "source": source,
        "probability": "" if probability is None else probability,
        "probability_source": probability_key or "",
        "score": "" if score is None else score,
        "score_source": score_key or "",
        "label": "" if label_bool is None else label_bool,
        "uncertainty": "" if uncertainty is None else uncertainty,
        "applicability_domain_status": applicability_status,
        "ood_flag": bool(ood_bool),
        "low_confidence": low_confidence,
        "claim_blockers": "|".join(dict.fromkeys(blockers)),
    }


# ---------------------------------------------------------------------------
# Calibration bin attribution (lifted 1-to-1 from Multiflow, per source)
# ---------------------------------------------------------------------------


def _brier_score(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """Brier score = mean (p - y)^2 over calibrated (probability, label) pairs.

    Added to support Mol-Metal's 5-metal × 9-cell × 4-source heads
    needing a headline accuracy metric alongside ECE.
    """
    if not pairs:
        return None
    total = sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs)
    return total / len(pairs)


def _calibration_bins(
    rows: Sequence[Mapping[str, Any]], *, bin_count: int
) -> tuple[list[dict[str, Any]], dict[str, float | None], dict[str, float | None]]:
    """Compute 10-bin reliability histogram + per-source ECE + Brier.

    Returns:
        bin_rows: long-form ``BIN_FIELDS`` rows
        ece_by_source: dict source -> ECE (None when no calibrated pairs)
        brier_by_source: dict source -> Brier (None when no calibrated pairs)
    """
    by_source: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        probability = _to_float(row.get("probability"))
        label = _as_bool(row.get("label"))
        if probability is not None and label is not None:
            by_source[str(row.get("source"))].append((probability, label))

    bin_rows: list[dict[str, Any]] = []
    ece_by_source: dict[str, float | None] = {}
    brier_by_source: dict[str, float | None] = {}
    for source in DOMAIN_SPECS_MOL_METAL:
        pairs = by_source.get(source, [])
        if not pairs:
            ece_by_source[source] = None
            brier_by_source[source] = None
            continue
        ece = 0.0
        total = len(pairs)
        for bin_index in range(bin_count):
            lower = bin_index / bin_count
            upper = (bin_index + 1) / bin_count
            in_bin = [
                (probability, label)
                for probability, label in pairs
                if probability >= lower
                and (
                    probability < upper
                    or (bin_index == bin_count - 1 and probability <= upper)
                )
            ]
            if not in_bin:
                mean_confidence = None
                accuracy = None
                error = None
                contribution = 0.0
            else:
                mean_confidence = sum(p for p, _ in in_bin) / len(in_bin)
                accuracy = sum(1.0 for _, label in in_bin if label) / len(in_bin)
                error = abs(mean_confidence - accuracy)
                contribution = (len(in_bin) / total) * error
                ece += contribution
            bin_rows.append(
                {
                    "source": source,
                    "bin_index": bin_index,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": len(in_bin),
                    "mean_confidence": ""
                    if mean_confidence is None
                    else mean_confidence,
                    "empirical_accuracy": "" if accuracy is None else accuracy,
                    "abs_calibration_error": "" if error is None else error,
                    "ece_contribution": contribution,
                }
            )
        ece_by_source[source] = ece
        brier_by_source[source] = _brier_score(pairs)
    return bin_rows, ece_by_source, brier_by_source


# ---------------------------------------------------------------------------
# Panel builder (entry point)
# ---------------------------------------------------------------------------


def build_cross_domain_uncertainty_panel(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    panel_id: str = "cross_domain_uncertainty",
    candidate_csv: Path | None = None,
    candidate_json: Path | None = None,
    bin_count: int = 10,
    low_confidence_margin: float = 0.10,
) -> dict[str, Any]:
    """Build the 4-source × N-row panel and write CSV + JSON artefacts.

    Mirrors the Multiflow function signature + output schema.
    """
    candidate_rows = [*_read_csv(candidate_csv), *_read_json_rows(candidate_json)]
    source_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows):
        for source in DOMAIN_SPECS_MOL_METAL:
            source_row = _source_row(
                row,
                index=index,
                source=source,
                low_confidence_margin=low_confidence_margin,
            )
            if source_row is not None:
                source_rows.append(source_row)

    bin_rows, ece_by_source, brier_by_source = _calibration_bins(
        source_rows, bin_count=bin_count
    )

    # Coverage tallies on the 5-metal × 9-cell × 4-source matrix.
    metal_coverage: Counter[str] = Counter()
    cell_coverage: Counter[str] = Counter()
    for row in source_rows:
        if row.get("metal"):
            metal_coverage[str(row["metal"])] += 1
        if row.get("cell_line"):
            cell_coverage[str(row["cell_line"])] += 1

    blocker_counts: Counter[str] = Counter()
    ood_count = 0
    low_confidence_count = 0
    calibrated_source_count = 0
    for row in source_rows:
        if _as_bool(row.get("ood_flag")):
            ood_count += 1
        if _as_bool(row.get("low_confidence")):
            low_confidence_count += 1
        if (
            _to_float(row.get("probability")) is not None
            and _as_bool(row.get("label")) is not None
        ):
            calibrated_source_count += 1
        for blocker in str(row.get("claim_blockers") or "").split("|"):
            if blocker:
                blocker_counts[blocker] += 1

    claim_blockers: list[str] = list(blocker_counts)
    if not source_rows:
        claim_blockers.append("uncertainty_or_calibration_inputs_missing")
    if calibrated_source_count == 0:
        claim_blockers.append("calibration_probability_label_pairs_missing")
    missing_sources = [
        src for src in DOMAIN_SPECS_MOL_METAL if ece_by_source.get(src) is None
    ]
    if missing_sources:
        claim_blockers.append("source_calibration_missing:" + "|".join(missing_sources))
    if ood_count:
        claim_blockers.append("ood_or_applicability_blocker_present")
    if low_confidence_count:
        claim_blockers.append("low_confidence_candidates_present")

    claim_blockers = list(dict.fromkeys(claim_blockers))
    broad_claim_allowed = not claim_blockers

    # Brier + ECE aggregated by source.
    per_source_metric: dict[str, dict[str, float | None]] = {}
    for source in DOMAIN_SPECS_MOL_METAL:
        per_source_metric[source] = {
            "ece": ece_by_source.get(source),
            "brier": brier_by_source.get(source),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / f"{panel_id}_source_uncertainty_rows.csv"
    bins_csv = output_dir / f"{panel_id}_calibration_bins.csv"
    summary_json = output_dir / f"{panel_id}_summary.json"
    coverage_json = output_dir / f"{panel_id}_coverage.json"
    _write_csv(rows_csv, source_rows, ROW_FIELDS)
    _write_csv(bins_csv, bin_rows, BIN_FIELDS)
    summary = {
        "schema_version": "cross_domain_uncertainty_panel_v1",
        "panel_id": panel_id,
        "lifted_from": "multiflow/scripts/build_uncertainty_calibration_applicability_panel.py",
        "candidate_csv": "" if candidate_csv is None else str(candidate_csv),
        "candidate_json": "" if candidate_json is None else str(candidate_json),
        "candidate_count": len(candidate_rows),
        "source_row_count": len(source_rows),
        "calibrated_source_row_count": calibrated_source_count,
        "source_ece": ece_by_source,
        "source_brier": brier_by_source,
        "per_source_metric": per_source_metric,
        "calibration_bin_count": bin_count,
        "ood_applicability_flag_count": ood_count,
        "low_confidence_count": low_confidence_count,
        "metal_coverage": dict(metal_coverage.most_common()),
        "cell_coverage": dict(cell_coverage.most_common()),
        "expected_coverage": {
            "metals": list(FIVE_METALS),
            "cells": list(NINE_CELLS),
            "sources": list(FOUR_SOURCES),
            "max_panel_cells": len(FIVE_METALS) * len(NINE_CELLS) * len(FOUR_SOURCES),
        },
        "claim_blocker_counts": dict(blocker_counts.most_common()),
        "claim_blockers": claim_blockers,
        "cross_domain_uncertainty_evidence_present": bool(
            source_rows and calibrated_source_count
        ),
        "broad_claim_allowed": broad_claim_allowed,
        "promotion_note": (
            "Fail closed: broad cross-domain claims require calibrated "
            "probability/label pairs in every source and nonblocking "
            "applicability for every 5-metal × 9-cell × 4-source cell."
        ),
        "outputs": {
            "source_uncertainty_rows_csv": str(rows_csv),
            "calibration_bins_csv": str(bins_csv),
            "summary_json": str(summary_json),
            "coverage_json": str(coverage_json),
        },
    }
    coverage = {
        "metals_seen": sorted(metal_coverage.keys()),
        "cells_seen": sorted(cell_coverage.keys()),
        "metals_missing": [m for m in FIVE_METALS if m not in metal_coverage],
        "cells_missing": [c for c in NINE_CELLS if c not in cell_coverage],
        "metal_row_counts": dict(metal_coverage),
        "cell_row_counts": dict(cell_coverage),
        "expected_full_panel_cells": (
            len(FIVE_METALS) * len(NINE_CELLS) * len(FOUR_SOURCES)
        ),
    }
    _write_json(summary_json, summary)
    _write_json(coverage_json, coverage)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-id", default="cross_domain_uncertainty")
    parser.add_argument("--bin-count", type=int, default=10)
    parser.add_argument("--low-confidence-margin", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_cross_domain_uncertainty_panel(
        candidate_csv=args.candidate_csv,
        candidate_json=args.candidate_json,
        output_dir=args.output_dir,
        panel_id=args.panel_id,
        bin_count=max(1, int(args.bin_count)),
        low_confidence_margin=float(args.low_confidence_margin),
    )
    print(json.dumps(summary["outputs"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
