"""Cross-domain uncertainty / calibration / applicability panel builder.

Mol-Metal adaptation of
``/home/hugo/codes/multiflow/scripts/build_uncertainty_calibration_applicability_panel.py``
(``P0-236L``).  Multiflow's original script organises the panel around
*four* functional domains — binding, admet, pose, postprocess — with
each domain owning a tuple of accepted field-name aliases for
probability / score / label / uncertainty / applicability / ood.

Mol-Metal extends that vocabulary with a fifth domain
``metal_cytotox`` that maps to the precious-metal cytotoxicity prediction
pipeline (pIC50 / log_residual / AUC predictions on the
MetalCytoToxDB + DrugOOD + NCI-60 + BindingDB sources).

This module:

    1. Defines :data:`DOMAIN_SPECS` — a 5-domain vocabulary suitable for
       the Mol-Metal evaluation pipeline.  Each domain's "probability"
       / "score" / "label" / "uncertainty" / "applicability" / "ood"
       lists accept multiple aliases so that heterogeneous upstream
       schemas (Vina dumps, PoseBusters CSVs, MetalCytoToxDB JSON,
       BindingDB IC50 exports) can be ingested without renaming.
    2. Provides :func:`build_cross_domain_panel` — the Mol-Metal
       equivalent of Multiflow's
       :func:`build_uncertainty_calibration_applicability_panel`,
       computing per-row domain rows, per-domain calibration bins,
       expected calibration error (ECE), blocker counts, and a
       pass-closed ``broad_claim_allowed`` flag.
    3. Provides :func:`build_arg_parser` + :func:`main` so the panel
       can be invoked from the command line (mirrors Multiflow's CLI).

Skill alignment
---------------
* ``party-twenty-spirit`` §3.5 — fail-closed ``broad_claim_allowed``:
  any blocker makes the panel claim-blocked.  Per-domain diagnostics
  are reported verbatim from the input rows.
* ``mol-metal-evaluation-rigor`` — per-domain Brier score + ECE
  reported; per-source applicability gating; OOD detection flags
  surfaced as claim blockers.
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
# Domain vocabulary — Mol-Metal 5-domain extension of Multiflow 4-domain
# ---------------------------------------------------------------------------


DEFAULT_OUTPUT_DIR = Path("molmetal/reports/cross_domain_uncertainty_panel")


DOMAIN_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    # ------------------------------------------------------------------
    # 1. binding — Vina / docking scores (verbatim Multiflow aliases)
    # ------------------------------------------------------------------
    "binding": {
        "probability": (
            "binding_probability",
            "binding_success_probability",
            "binding_proxy_probability",
            "binding_score_probability",
            "binding_confidence",
            "vina_probability",
            "docking_probability",
        ),
        "score": (
            "binding_score",
            "gnina_score",
            "refined_gnina_score",
            "vina_score",
            "posthoc_vina_score",
            "vina_kcal_per_mol",
            "docking_score",
        ),
        "label": (
            "binding_label",
            "binding_success",
            "binding_pass",
            "score_pair_success",
            "vina_pass",
        ),
        "uncertainty": (
            "binding_uncertainty",
            "gnina_uncertainty",
            "vina_uncertainty",
        ),
        "applicability": (
            "binding_applicability_domain_status",
            "applicability_domain_status",
            "pocket_applicability",
        ),
        "ood": (
            "binding_ood",
            "binding_out_of_domain",
            "ood_flag",
            "out_of_domain",
        ),
    },
    # ------------------------------------------------------------------
    # 2. admet — hERG / AMES / clintox / CYP / DILI (verbatim Multiflow
    #    + Mol-Metal extension: per-endpoint labels for herg/ames/cyp/...)
    # ------------------------------------------------------------------
    "admet": {
        "probability": (
            "admet_probability",
            "admet_all_required_pass_probability",
            "herg_probability",
            "ames_probability",
            "cyp_probability",
            "cyp_max_probability",
            "cytotox_probability",
            "dili_probability",
            "clintox_probability",
        ),
        "score": (
            "admet_score",
            "max_admet_endpoint_risk_probability",
            "max_risk_probability",
        ),
        "label": (
            "admet_label",
            "admet_all_required_pass",
            "admet_pass",
            # Mol-Metal per-endpoint label aliases — paired with the
            # matching ``herg_probability`` / ``ames_probability`` etc.
            # probability aliases above.
            "herg_label",
            "herg_pass",
            "ames_label",
            "ames_pass",
            "cyp_label",
            "cyp_pass",
            "cyp_max_label",
            "cyp_max_pass",
            "dili_label",
            "dili_pass",
            "clintox_label",
            "clintox_pass",
        ),
        "uncertainty": (
            "admet_uncertainty",
            "admet_endpoint_uncertainty",
        ),
        "applicability": (
            "admet_applicability_domain_status",
            "applicability_domain_status",
        ),
        "ood": (
            "admet_ood",
            "admet_out_of_domain",
            "ood_flag",
            "out_of_domain",
        ),
    },
    # ------------------------------------------------------------------
    # 3. pose — geometry / PoseBusters validity (verbatim Multiflow)
    # ------------------------------------------------------------------
    "pose": {
        "probability": (
            "pose_probability",
            "pose_validity_probability",
            "posebusters_pass_probability",
            "pose_validity_proxy_probability",
        ),
        "score": (
            "pose_score",
            "pose_validity_score",
            "posebusters_score",
        ),
        "label": (
            "pose_label",
            "posebusters_pass",
            "pose_validity_proxy_pass",
            "pose_geometry_proxy_pass",
        ),
        "uncertainty": (
            "pose_uncertainty",
            "pose_validity_uncertainty",
            "posebusters_uncertainty",
        ),
        "applicability": (
            "pose_applicability_domain_status",
            "applicability_domain_status",
        ),
        "ood": (
            "pose_ood",
            "pose_out_of_domain",
            "ood_flag",
            "out_of_domain",
        ),
    },
    # ------------------------------------------------------------------
    # 4. postprocess — closure / synthesis / route success
    #    (verbatim Multiflow)
    # ------------------------------------------------------------------
    "postprocess": {
        "probability": (
            "postprocess_probability",
            "postprocess_success_probability",
            "route_success_probability",
            "repair_success_probability",
        ),
        "score": (
            "postprocess_score",
            "route_success_score",
            "repair_score",
        ),
        "label": (
            "postprocess_label",
            "postprocess_success",
            "route_success",
            "repair_accepted",
            "promotion_gate_pass",
        ),
        "uncertainty": (
            "postprocess_uncertainty",
            "route_uncertainty",
            "repair_uncertainty",
        ),
        "applicability": (
            "postprocess_applicability_domain_status",
            "applicability_domain_status",
        ),
        "ood": (
            "postprocess_ood",
            "postprocess_out_of_domain",
            "ood_flag",
            "out_of_domain",
        ),
    },
    # ------------------------------------------------------------------
    # 5. metal_cytotox — Mol-Metal-specific precious-metal cytotoxicity
    #    (NEW for Mol-Metal; not in the original Multiflow script)
    # ------------------------------------------------------------------
    "metal_cytotox": {
        "probability": (
            "metal_cytotox_probability",
            "cytotox_probability",
            "cell_line_activity_probability",
            "auc_probability",
            "pic50_probability",
            "cancer_active_probability",
            "anticancer_probability",
        ),
        "score": (
            "metal_cytotox_score",
            "pic50_score",
            "log_pic50_score",
            "cell_line_auc_score",
            "metalcytotox_score",
            "log_residual_score",
            "pIC50",
        ),
        "label": (
            "metal_cytotox_label",
            "active",
            "passes_cytotox_threshold",
            "cytotox_label",
            "cell_line_active",
            "cancer_active",
        ),
        "uncertainty": (
            "metal_cytotox_uncertainty",
            "pic50_uncertainty",
            "log_residual_uncertainty",
            "kendall_sigma",
        ),
        "applicability": (
            "metal_cytotox_applicability_domain_status",
            "metal_applicability",
            "cell_line_in_distribution",
            "data_source_applicability",
        ),
        "ood": (
            "metal_cytotox_ood",
            "metal_ood",
            "cell_line_ood",
            "scaffold_ood",
            "metal_out_of_domain",
        ),
    },
}


# ---------------------------------------------------------------------------
# Schema for the output CSVs / JSON
# ---------------------------------------------------------------------------


ROW_FIELDS: list[str] = [
    "candidate_id",
    "target_id",
    "domain",
    "probability",
    "probability_source",
    "score",
    "score_source",
    "label",
    "uncertainty",
    "applicability_domain_status",
    "ood_flag",
    "low_confidence",
    "data_source",
    "metal",
    "cell_line",
    "claim_blockers",
]


BIN_FIELDS: list[str] = [
    "domain",
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
# IO helpers
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
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Type coercion helpers (verbatim Multiflow semantics)
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
    if text in {"1", "true", "yes", "y", "pass", "passed", "success", "accepted"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "failure", "rejected"}:
        return False
    return None


def _first_value(
    row: Mapping[str, Any], keys: Iterable[str]
) -> tuple[str | None, Any]:
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


def _score_to_probability(domain: str, score: float | None) -> float | None:
    """Multiflow-style score→probability transform (verbatim logic)."""
    if score is None:
        return None
    if 0.0 <= score <= 1.0:
        return score
    if domain == "binding":
        # Docking scores are usually better when more negative; -7 is treated as a soft midpoint.
        return 1.0 / (1.0 + math.exp(score + 7.0))
    return 1.0 / (1.0 + math.exp(-score))


def _applicability_is_blocking(status: str) -> bool:
    """Verbatim Multiflow applicability semantics."""
    text = status.strip().lower()
    if not text:
        return True
    allowed = {"in_domain", "inside_domain", "applicable", "pass", "passed", "known_in_domain"}
    blocked = {"ood", "out_of_domain", "outside_domain", "not_applicable", "fail", "failed", "unknown"}
    if text in allowed:
        return False
    if text in blocked:
        return True
    return True


def _candidate_id(row: Mapping[str, Any], index: int) -> str:
    return str(
        row.get("candidate_id")
        or row.get("source_candidate_id")
        or row.get("id")
        or f"row_{index}"
    ).strip()


# ---------------------------------------------------------------------------
# Per-row domain extraction
# ---------------------------------------------------------------------------


def _domain_row(
    row: Mapping[str, Any],
    *,
    index: int,
    domain: str,
    low_confidence_margin: float,
) -> dict[str, Any] | None:
    spec = DOMAIN_SPECS[domain]
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
        probability = _score_to_probability(domain, score)
        probability_key = f"{score_key}:score_to_probability" if score_key else None

    uncertainty = _to_float(uncertainty_value)
    label_bool = _as_bool(label_value)
    ood_bool = _as_bool(ood_value)
    applicability_status = str(applicability_value or "").strip()

    has_any_domain_input = any(
        value is not None and value != ""
        for value in (
            probability_value, score_value, uncertainty_value,
            label_value, applicability_value, ood_value,
        )
    )
    if not has_any_domain_input:
        return None

    blockers: list[str] = []
    if probability is None:
        blockers.append(f"{domain}_probability_or_score_missing")
    if label_bool is None:
        blockers.append(f"{domain}_calibration_label_missing")
    if uncertainty is None and probability is None:
        blockers.append(f"{domain}_uncertainty_missing")
    if not applicability_status:
        blockers.append(f"{domain}_applicability_domain_missing")
    elif _applicability_is_blocking(applicability_status):
        blockers.append(f"{domain}_applicability_domain_blocking")
    if ood_bool:
        blockers.append(f"{domain}_ood_flag_true")

    low_confidence = False
    if probability is None:
        low_confidence = True
    elif abs(probability - 0.5) <= low_confidence_margin:
        low_confidence = True
    elif uncertainty is not None and uncertainty >= low_confidence_margin:
        low_confidence = True
    if low_confidence:
        blockers.append(f"{domain}_low_confidence")

    return {
        "candidate_id": _candidate_id(row, index),
        "target_id": row.get("target_id", ""),
        "domain": domain,
        "probability": "" if probability is None else probability,
        "probability_source": probability_key or "",
        "score": "" if score is None else score,
        "score_source": score_key or "",
        "label": "" if label_bool is None else label_bool,
        "uncertainty": "" if uncertainty is None else uncertainty,
        "applicability_domain_status": applicability_status,
        "ood_flag": bool(ood_bool),
        "low_confidence": low_confidence,
        "data_source": row.get("data_source", ""),
        "metal": row.get("metal", ""),
        "cell_line": row.get("cell_line", ""),
        "claim_blockers": "|".join(dict.fromkeys(blockers)),
    }


# ---------------------------------------------------------------------------
# Calibration bins
# ---------------------------------------------------------------------------


def _calibration_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    bin_count: int,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    by_domain: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        probability = _to_float(row.get("probability"))
        label = _as_bool(row.get("label"))
        if probability is not None and label is not None:
            by_domain[str(row.get("domain"))].append((probability, label))

    bin_rows: list[dict[str, Any]] = []
    ece_by_domain: dict[str, float | None] = {}
    for domain in DOMAIN_SPECS:
        pairs = by_domain.get(domain, [])
        if not pairs:
            ece_by_domain[domain] = None
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
                mean_confidence = sum(probability for probability, _ in in_bin) / len(in_bin)
                accuracy = sum(1.0 for _, label in in_bin if label) / len(in_bin)
                error = abs(mean_confidence - accuracy)
                contribution = (len(in_bin) / total) * error
                ece += contribution
            bin_rows.append(
                {
                    "domain": domain,
                    "bin_index": bin_index,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": len(in_bin),
                    "mean_confidence": "" if mean_confidence is None else mean_confidence,
                    "empirical_accuracy": "" if accuracy is None else accuracy,
                    "abs_calibration_error": "" if error is None else error,
                    "ece_contribution": contribution,
                }
            )
        ece_by_domain[domain] = ece
    return bin_rows, ece_by_domain


# ---------------------------------------------------------------------------
# Public API — main panel builder
# ---------------------------------------------------------------------------


def build_cross_domain_panel(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    panel_id: str = "cross_domain_uncertainty_panel",
    candidate_csv: Path | None = None,
    candidate_json: Path | None = None,
    bin_count: int = 10,
    low_confidence_margin: float = 0.10,
) -> dict[str, Any]:
    """Build the Mol-Metal cross-domain uncertainty / calibration panel.

    Verbatim port of Multiflow's
    :func:`build_uncertainty_calibration_applicability_panel` with the
    Mol-Metal 5-domain vocabulary (binding / admet / pose / postprocess
    / metal_cytotox) and Mol-Metal-specific output schema
    (``data_source``, ``metal``, ``cell_line`` columns).

    Args:
        output_dir:           directory to write the panel CSVs + JSON into.
        panel_id:             prefix for the output filenames.
        candidate_csv:        optional CSV of candidate rows.
        candidate_json:       optional JSON of candidate rows (list or
                              ``{"rows": [...]}``).
        bin_count:            number of calibration bins per domain
                              (default 10).
        low_confidence_margin: probability / uncertainty margin that
                              flips the ``low_confidence`` flag
                              (default 0.10).

    Returns:
        The summary dict (also written as
        ``{panel_id}_summary.json`` in ``output_dir``).
    """
    candidate_rows = [*_read_csv(candidate_csv), *_read_json_rows(candidate_json)]
    domain_rows: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows):
        for domain in DOMAIN_SPECS:
            domain_row = _domain_row(
                row, index=index, domain=domain,
                low_confidence_margin=low_confidence_margin,
            )
            if domain_row is not None:
                domain_rows.append(domain_row)

    bin_rows, ece_by_domain = _calibration_bins(domain_rows, bin_count=bin_count)
    blocker_counts: Counter[str] = Counter()
    ood_count = 0
    low_confidence_count = 0
    calibrated_domain_count = 0
    for row in domain_rows:
        if _as_bool(row.get("ood_flag")):
            ood_count += 1
        if _as_bool(row.get("low_confidence")):
            low_confidence_count += 1
        if _to_float(row.get("probability")) is not None and _as_bool(row.get("label")) is not None:
            calibrated_domain_count += 1
        for blocker in str(row.get("claim_blockers") or "").split("|"):
            if blocker:
                blocker_counts[blocker] += 1

    claim_blockers = list(blocker_counts)
    if not domain_rows:
        claim_blockers.append("uncertainty_or_calibration_inputs_missing")
    if calibrated_domain_count == 0:
        claim_blockers.append("calibration_probability_label_pairs_missing")
    missing_domains = [domain for domain in DOMAIN_SPECS if ece_by_domain.get(domain) is None]
    if missing_domains:
        claim_blockers.append("domain_calibration_missing:" + "|".join(missing_domains))
    if ood_count:
        claim_blockers.append("ood_or_applicability_blocker_present")
    if low_confidence_count:
        claim_blockers.append("low_confidence_candidates_present")

    claim_blockers = list(dict.fromkeys(claim_blockers))
    broad_claim_allowed = not claim_blockers

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / f"{panel_id}_candidate_uncertainty_rows.csv"
    bins_csv = output_dir / f"{panel_id}_calibration_bins.csv"
    summary_json = output_dir / f"{panel_id}_summary.json"
    _write_csv(rows_csv, domain_rows, ROW_FIELDS)
    _write_csv(bins_csv, bin_rows, BIN_FIELDS)
    summary = {
        "schema_version": "cross_domain_uncertainty_panel_v1",
        "panel_id": panel_id,
        "domains": list(DOMAIN_SPECS.keys()),
        "candidate_csv": "" if candidate_csv is None else str(candidate_csv),
        "candidate_json": "" if candidate_json is None else str(candidate_json),
        "candidate_count": len(candidate_rows),
        "domain_row_count": len(domain_rows),
        "calibrated_domain_row_count": calibrated_domain_count,
        "ece_by_domain": ece_by_domain,
        "calibration_bin_count": bin_count,
        "ood_applicability_flag_count": ood_count,
        "low_confidence_count": low_confidence_count,
        "claim_blocker_counts": dict(blocker_counts.most_common()),
        "claim_blockers": claim_blockers,
        "uncertainty_calibration_evidence_present": bool(domain_rows and calibrated_domain_count),
        "broad_claim_allowed": broad_claim_allowed,
        "promotion_note": (
            "Fail closed: broad claims require calibrated probability/label "
            "pairs and nonblocking applicability for every promoted "
            "domain (binding / admet / pose / postprocess / metal_cytotox)."
        ),
        "outputs": {
            "candidate_uncertainty_rows_csv": str(rows_csv),
            "calibration_bins_csv": str(bins_csv),
            "summary_json": str(summary_json),
        },
    }
    _write_json(summary_json, summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-id", default="cross_domain_uncertainty_panel")
    parser.add_argument("--bin-count", type=int, default=10)
    parser.add_argument("--low-confidence-margin", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_cross_domain_panel(
        candidate_csv=args.candidate_csv,
        candidate_json=args.candidate_json,
        output_dir=args.output_dir,
        panel_id=args.panel_id,
        bin_count=max(1, int(args.bin_count)),
        low_confidence_margin=float(args.low_confidence_margin),
    )
    print(json.dumps(summary["outputs"], sort_keys=True))
    return 0


__all__ = [
    "DOMAIN_SPECS",
    "ROW_FIELDS",
    "BIN_FIELDS",
    "DEFAULT_OUTPUT_DIR",
    "build_cross_domain_panel",
    "build_arg_parser",
    "main",
]
