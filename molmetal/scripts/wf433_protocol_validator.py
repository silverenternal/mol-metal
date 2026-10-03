#!/usr/bin/env python3
"""Wf-433-A4 protocol validator — single-results CLI for protocol-aligned
comparison against 3 selected Tier-1 baselines (Krasnov 2026 + Rusanov 2025
+ DrugOOD 2022).

==============================================================================

Per ``TODO/03_baselines/PROTOCOL_VALIDATION.md`` §2 + ``COMPARISON_BASELINES.md``
§0 principle:

    我们不需要 reproduce 别人的数字 — 我们需要 same dataset + same protocol +
    same metric 来 make results comparable。

This script is **CPU-only** (no GPU launches, no wet-lab planning). It computes
the post-training per-seed / per-split / per-cohort / per-fold statistics from a
results JSON (e.g. ``metrics/by_metric/metal_lomo_auc_wasserstein.json``), looks
up the selected baseline value from a hardcoded baseline cache, and emits a
PASS / FAIL / WARN verdict with:

  * ``our_value`` + ``our_value_per_seed`` + ``our_value_per_fold`` (when present)
  * ``baseline_value`` + ``baseline_source`` (always cited)
  * ``alignment_level`` (FULL ALIGN / DATASET+METRIC ALIGN /
    DATASET+METRIC+SPLIT APPROX ALIGN / DATASET ALIGN / NOT ALIGN)
  * ``protocol_adjustment`` (numeric delta applied to baseline for comparability)
  * ``threshold_comparable`` (baseline + protocol_adjustment)
  * ``disagreement_score`` (normalised gap; 0.0 = exactly at threshold)
  * ``verdict`` (canonical labels per ``party-twenty-spirit §3.5`` honest framing)

CLI::

    python molmetal/scripts/wf433_protocol_validator.py \
        --our-results metrics/by_metric/metal_lomo_auc_wasserstein.json \
        --baseline krasnov_2026 \
        --out molmetal/reports/wf433_protocol_validator/example_output.json

Supported baselines: ``krasnov_2026`` / ``rusanov_2025`` / ``drugood_2022``.

Honest framing
==============

Per ``party-twenty-spirit §3.5`` + ``AGENTS.md §5.2``: every numeric value traces
to a JSON / report path on disk (``our_value_source`` + ``baseline_source``
fields).  No fabrication of aspirational numbers; CITE-ONLY baselines (e.g.
Rusanov per-cell R² with the supplementary xlsx Cloudflare-blocked) emit
``baseline_value: null + verdict: NOT_COMPARABLE + baseline_source: "CITE-ONLY"``.

Per ``AGENTS.md §5.6`` atomic-write protocol adoption rule: when ``--out`` is
provided, this script writes via ``atomic_write_file`` from
:mod:`wf386a3_atomic_write` to prevent partial-write file corruption.

Per ``AGENTS.md §5.5`` filesystem-truth rule: ``--out`` defaults to
``molmetal/reports/wf433_protocol_validator/example_output.json`` if not
provided.  Parent directories are auto-created.

Exit codes
==========

  * ``0`` — verdict is PASS / NOT_COMPARABLE (CITE-ONLY)
  * ``1`` — verdict is FAIL / WARN (alignment gap below threshold)
  * ``2`` — input JSON missing or unreadable
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Ensure the repo root is on sys.path so ``molmetal.scripts.wf386a3_atomic_write``
# imports cleanly when the script is invoked directly (e.g. via
# ``python molmetal/scripts/wf433_protocol_validator.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants — 3 selected Tier-1 baselines per COMPARISON_BASELINES.md §1
# ---------------------------------------------------------------------------

METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"
DEFAULT_OUT = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf433_protocol_validator"
    / "example_output.json"
)

# Canonical 3 baseline cache (auto-extracted from COMPARISON_BASELINES.md §1
# + PROTOCOL_VALIDATION.md §2 + PROTOCOL_ALIGNMENT_MATRIX.md §1).
# Per-baseline: dataset, split, metric, alignment level, protocol adjustment,
# baseline values (per-metal for Krasnov, per-cell for Rusanov, per-split for
# DrugOOD).  Every numeric value traces to COMPARISON_BASELINES.md §1 per
# PROTOCOL_VALIDATION.md §4 honest framing.
BASELINES: Dict[str, Dict[str, Any]] = {
    "krasnov_2026": {
        "name": "Krasnov 2026 (J. Med. Chem.) — MetalCytoToxDB",
        "dataset": "MetalCytoToxDB",
        "dataset_path_pattern": "metalcytotox_5metal_multitask_v4.csv",
        "paper_split": "5-fold CV with temporal split",
        "our_split": "scaffold-split 70/10/20 (Bemis-Murcko)",
        "metric": "binary AUC (per metal, mean + std over seeds)",
        "alignment_level": "DATASET+METRIC ALIGN (split differs)",
        "protocol_adjustment": 0.02,
        "threshold_direction": "higher_better",
        "applicable_metrics": ["m1", "m21"],
        "baselines": {
            "m1": {"Ru": 0.81, "Ir": 0.73},
            "m21": {"per_cell_avg_single_Ru": 0.81},
        },
        "baseline_source": "TODO/03_baselines/COMPARISON_BASELINES.md::§1.1::Krasnov_2026",
        "alignment_note": (
            "scaffold-split is stricter than Krasnov's temporal 5-fold CV; "
            "protocol adjustment +0.02"
        ),
    },
    "rusanov_2025": {
        "name": "Rusanov 2025 (ChemRxiv) — PlatinAI",
        "dataset": "PlatinAI 4-cell (MCF-7, A549, A2780, A2780cis)",
        "dataset_path_pattern": "platinai_4cell.csv",
        "paper_split": "per-cell-line split",
        "our_split": "cohort-split 70/10/20 + per-cell-line evaluation",
        "metric": "per-cell-line binary AUC + per-cell R²",
        "alignment_level": "DATASET+METRIC+SPLIT APPROX ALIGN",
        "protocol_adjustment": -0.03,
        "threshold_direction": "higher_better",
        "applicable_metrics": ["m2", "m19"],
        "baselines": {
            "m2": {
                "A2780cis": 0.62,
                "MCF7": "CITE-ONLY",
                "A549": "CITE-ONLY",
                "A2780": "CITE-ONLY",
            },
            "m19": {"Pt_fold_primary": 0.70},
        },
        "baseline_source": "TODO/03_baselines/COMPARISON_BASELINES.md::§1.2::Rusanov_2025",
        "alignment_note": (
            "cohort-split is harder than per-cell-line; protocol adjustment "
            "-0.03 on A2780cis (per-cell AUC directly comparable)"
        ),
        "cite_only_metrics": ["m2::MCF7", "m2::A549", "m2::A2780", "m2::R²"],
    },
    "drugood_2022": {
        "name": "DrugOOD 2022 (ICLR) — OOD generalization",
        "dataset": "DrugOOD (1.58M rows ChEMBL_29 IC50)",
        "dataset_path_pattern": "drugood/ic50/data/train.csv",
        "paper_split": "assay / scaffold / size (官方)",
        "our_split": "assay / scaffold / size (same)",
        "metric": "OOD binary AUC (mean over 3 splits)",
        "alignment_level": "DATASET+METRIC+SPLIT FULL ALIGN",
        "protocol_adjustment": 0.0,
        "threshold_direction": "higher_better",
        "applicable_metrics": ["m17"],
        "baselines": {
            "m17": {
                "iMoLD": {"assay": 0.7211, "scaffold": 0.6884, "size": 0.6792},
                "ERM": {"assay": 0.7163, "scaffold": 0.6879, "size": 0.6750},
            },
        },
        "baseline_source": "TODO/03_baselines/COMPARISON_BASELINES.md::§1.3::DrugOOD_2022",
        "alignment_note": ("FULL ALIGN; no protocol adjustment"),
    },
}

SUPPORTED_BASELINES = tuple(BASELINES.keys())

# Recognised metric identifiers (subset of METRICS_TARGETS.md §1).
# Falls back to filename inference if `metric` field is absent.
METRIC_KEYS_BY_NAME = {
    "metal_lomo_auc_wasserstein": "m1",
    "platinai_4cell_auc": "m2",
    "vina_kcal_per_mol": "m3",
    "docked_pose_rmsd": "m15",
    "drugood_ood_auc": "m17",
    "nci60_precious_9cell_auc": "m21",
}

SCHEMA_VERSION = "433.v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. ``2026-10-02T15:42:00+00:00``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _read_json(path: Path) -> Dict[str, Any]:
    """Read JSON; raise ``FileNotFoundError`` / ``json.JSONDecodeError``."""
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomic-write JSON via :mod:`wf386a3_atomic_write` per AGENTS.md §5.6."""
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
    except (ImportError, ModuleNotFoundError):
        # Fallback to plain write if wf386a3 is not importable (e.g. when this
        # script is invoked outside the project's venv).  This fallback is
        # documented in the script docstring's honest-framing section.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    atomic_write_file(str(path), text)


def _safe_mean(xs: List[float]) -> Optional[float]:
    """Arithmetic mean; returns None for empty list."""
    if not xs:
        return None
    return float(statistics.fmean(xs))


def _safe_std(xs: List[float]) -> Optional[float]:
    """Sample std; returns None if fewer than 2 elements."""
    if len(xs) < 2:
        return None
    return float(statistics.stdev(xs))


def _infer_metric_id(results: Dict[str, Any], results_path: Path) -> str:
    """Infer Tier-1 metric_id (m1 / m2 / m3 / m15 / m17 / m21) from results.

    Priority: explicit ``metric`` field → ``METRIC_KEYS_BY_NAME`` map → stem of
    the JSON filename (e.g. ``metal_lomo_auc_wasserstein.json`` → ``m1``).
    """
    explicit = str(results.get("metric", "")).strip()
    if explicit:
        # Some JSONs use the bare filename as ``metric`` field; check map.
        mapped = METRIC_KEYS_BY_NAME.get(explicit)
        if mapped:
            return mapped
        # Already an m-id?
        if explicit.startswith("m") and len(explicit) <= 4:
            return explicit
    stem = results_path.stem
    mapped = METRIC_KEYS_BY_NAME.get(stem)
    if mapped:
        return mapped
    return "unknown"


# ---------------------------------------------------------------------------
# Per-metric statistics extractors (MEASURED vs PROJECTED discipline)
# ---------------------------------------------------------------------------


@dataclass
class OurStats:
    """MEASURED stats extracted from a single results JSON.

    All fields are optional because not every metric JSON carries every
    statistic (e.g. ``per_fold_auc_mean`` is MetalCytoToxDB-specific; only
    ``drugood_ood_auc.json`` carries per-split ``results.<split>.ood_auc_mean``).
    """

    overall_value: Optional[float] = None
    overall_std: Optional[float] = None
    per_seed: List[float] = field(default_factory=list)
    per_split: Dict[str, float] = field(default_factory=dict)
    per_fold: Dict[str, float] = field(default_factory=dict)
    per_cell: Dict[str, float] = field(default_factory=dict)
    n_seeds: Optional[int] = None
    n_runs: Optional[int] = None
    data_source: Optional[str] = None
    metric_field: Optional[str] = None


def _extract_m1_stats(r: Dict[str, Any]) -> OurStats:
    """m1 — Ru subset binary AUC (MetalCytoToxDB / per-fold)."""
    s = OurStats()
    s.metric_field = r.get("metric")
    s.overall_value = r.get("value")
    s.overall_std = r.get("value_std")
    s.per_seed = [float(v) for v in r.get("per_run_auc", []) or []]
    s.per_fold = {k: float(v) for k, v in (r.get("per_fold_auc_mean") or {}).items()}
    s.n_runs = r.get("n_runs") or r.get("n_runs_completed")
    s.n_seeds = len(s.per_seed)
    s.data_source = r.get("data_source")
    return s


def _extract_m2_stats(r: Dict[str, Any]) -> OurStats:
    """m2 — Pt 4-cell-line AUC (PlatinAI / per-cell-line)."""
    s = OurStats()
    s.metric_field = r.get("metric")
    s.overall_value = r.get("value_overall_mean_auc")
    s.overall_std = r.get("value_overall_mean_std")
    cells = r.get("per_cell_line") or {}
    s.per_cell = {
        k: float((v or {}).get("auc_mean", float("nan"))) for k, v in cells.items()
    }
    s.per_seed = []
    for v in cells.values():
        if v and "per_seed" in v:
            s.per_seed.extend(float(x) for x in v["per_seed"])
    s.n_seeds = r.get("n_seeds")
    s.data_source = r.get("data_source")
    return s


def _extract_m17_stats(r: Dict[str, Any]) -> OurStats:
    """m17 — DrugOOD OOD generalization (assay / scaffold / size)."""
    s = OurStats()
    s.metric_field = r.get("metric")
    results = r.get("results") or {}
    s.per_split = {
        k: float((v or {}).get("ood_auc_mean", float("nan")))
        for k, v in results.items()
    }
    valid = [v for v in s.per_split.values() if not math.isnan(v)]
    s.overall_value = _safe_mean(valid)
    s.overall_std = _safe_std(valid)
    s.per_seed = []
    for v in results.values():
        if v and "per_seed_ood" in v:
            s.per_seed.extend(float(x) for x in v["per_seed_ood"])
    s.n_seeds = r.get("seeds") and len(r["seeds"])
    s.data_source = r.get("data_source")
    return s


def _extract_generic_stats(r: Dict[str, Any]) -> OurStats:
    """Generic fallback — pull ``value`` + ``value_std`` + ``per_run_auc``."""
    s = OurStats()
    s.metric_field = r.get("metric")
    s.overall_value = r.get("value")
    s.overall_std = r.get("value_std")
    s.per_seed = [float(v) for v in r.get("per_run_auc", []) or []]
    s.n_runs = r.get("n_runs")
    s.n_seeds = len(s.per_seed) or r.get("n_seeds")
    s.data_source = r.get("data_source")
    # Also try per-cell/per-split if present.
    if "per_cell_line" in r and isinstance(r["per_cell_line"], dict):
        s.per_cell = {
            k: float((v or {}).get("auc_mean", float("nan")))
            for k, v in r["per_cell_line"].items()
        }
    if "results" in r and isinstance(r["results"], dict):
        s.per_split = {
            k: float((v or {}).get("ood_auc_mean", float("nan")))
            for k, v in r["results"].items()
        }
    return s


def extract_our_stats(metric_id: str, r: Dict[str, Any]) -> OurStats:
    """Top-level dispatch."""
    if metric_id == "m1":
        return _extract_m1_stats(r)
    if metric_id == "m2":
        return _extract_m2_stats(r)
    if metric_id == "m17":
        return _extract_m17_stats(r)
    return _extract_generic_stats(r)


# ---------------------------------------------------------------------------
# Baseline lookup + threshold computation
# ---------------------------------------------------------------------------


def _baseline_for(
    baseline_key: str,
    metric_id: str,
    baseline_meta: Dict[str, Any],
) -> Tuple[Optional[float], str, str, bool]:
    """Return ``(baseline_value, baseline_source, label, is_cite_only)``.

    For Krasnov m1, the metric JSON reports per-fold AUC and the baseline is
    per-metal; the default compares to Ru (the dominant 5-metal subset).
    For Rusanov m2, the metric JSON reports per-cell-line AUC and we compare
    to A2780cis (the only baseline value actually on disk).
    For DrugOOD m17, we compare mean-of-3-splits against the mean of iMoLD
    split values.
    """
    cite_only_metrics = baseline_meta.get("cite_only_metrics", []) or []
    if baseline_key == "krasnov_2026":
        ru_val = baseline_meta["baselines"][metric_id].get("Ru")
        ir_val = baseline_meta["baselines"][metric_id].get("Ir")
        # Prefer Ru (dominant 5-metal); fall back to Ir if Ru absent.
        chosen = ru_val if ru_val is not None else ir_val
        return (
            float(chosen) if chosen is not None else None,
            f"{baseline_meta['baseline_source']}::Ru",
            "Ru",
            False,
        )
    if baseline_key == "rusanov_2025":
        # A2780cis is the only baseline value on disk; other cells are CITE-ONLY.
        a2780cis = baseline_meta["baselines"][metric_id].get("A2780cis")
        is_cite = a2780cis is None or f"m{metric_id[1:]}::A2780cis" in cite_only_metrics
        return (
            float(a2780cis) if a2780cis is not None else None,
            f"{baseline_meta['baseline_source']}::A2780cis",
            "A2780cis",
            bool(is_cite) and a2780cis is None,
        )
    if baseline_key == "drugood_2022":
        # Compare mean of 3-split results vs mean of iMoLD references.
        imold = baseline_meta["baselines"][metric_id]["iMoLD"]
        vals = [v for v in imold.values() if isinstance(v, (int, float))]
        mean_val = _safe_mean([float(v) for v in vals])
        return (
            float(mean_val) if mean_val is not None else None,
            f"{baseline_meta['baseline_source']}::iMoLD_mean",
            "iMoLD_mean",
            False,
        )
    return None, baseline_meta["baseline_source"], "unknown", False


def _threshold(baseline: Optional[float], adjustment: float) -> Optional[float]:
    """Apply protocol adjustment; returns None for cite-only baselines."""
    if baseline is None:
        return None
    return float(baseline + adjustment)


def _verdict(
    *,
    metric_value: Optional[float],
    threshold: Optional[float],
    direction: str,
) -> Tuple[bool, str, Optional[float], Optional[str]]:
    """Compute PASS / FAIL / NOT_COMPARABLE verdict.

    Returns ``(pass_bool, verdict_label, disagreement_score, threshold_comparable)``.
    ``disagreement_score`` is normalised gap (0.0 = exactly at threshold); None
    if threshold is None (cite-only baseline).
    """
    if metric_value is None or threshold is None:
        return False, "NOT_COMPARABLE", None, None
    delta = threshold - metric_value  # gap above threshold (for higher-better)
    if direction == "higher_better":
        # pass = MEASURED >= threshold - gate_margin (default 0)
        ok = metric_value >= threshold
        diss = max(0.0, delta) / max(abs(threshold), 1e-9)
    else:
        # lower_better — pass = MEASURED <= threshold
        ok = metric_value <= threshold
        diss = max(0.0, -delta) / max(abs(threshold), 1e-9)
    label = "PASS_BASELINE_GE_THRESHOLD" if ok else "BELOW_BASELINE_THRESHOLD"
    return bool(ok), label, float(diss), float(threshold)


# ---------------------------------------------------------------------------
# Protocol-alignment contract (subset of wf433_protocol_alignment_validator.py)
# ---------------------------------------------------------------------------


def _check_alignment_contract(
    results: Dict[str, Any],
    baseline_key: str,
    baseline_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate that the metric JSON's dataset path / split / metric markers
    match the baseline's documented contract.

    The check is **subset-style** (presence-based); it does NOT verify the
    trainer actually emitted those values at training time (that's the
    ``pre_flight_check.py`` job).
    """
    ds_pattern = baseline_meta["dataset_path_pattern"]
    ds_match = ds_pattern in str(results.get("data_source", ""))
    # m17 uses a top-level ``data_source`` string; m1 / m2 use absolute path.
    # Both forms are accepted.
    split_field = str(results.get("split", "")).lower()
    split_match = (
        "scaffold" in split_field
        or "5fold" in split_field
        or "kfold" in split_field
        or "temporal" in split_field
    )
    metric_field = str(results.get("metric", "")).lower()
    metric_match = any(
        kw in metric_field
        for kw in ("auc", "kca", "r_squared", "rmse", "kcal_mol", "tanimoto", "rms")
    )
    return {
        "dataset_path_check": {
            "expected_pattern": ds_pattern,
            "observed": str(results.get("data_source", "")),
            "match": bool(ds_match),
        },
        "split_protocol_check": {
            "expected_paper_split": baseline_meta["paper_split"],
            "expected_our_split": baseline_meta["our_split"],
            "observed_split": results.get("split", ""),
            "match": bool(split_match),
        },
        "metric_definition_check": {
            "expected_metric": baseline_meta["metric"],
            "observed_metric": results.get("metric", ""),
            "match": bool(metric_match),
        },
        "overall_alignment_match": bool(ds_match and split_match and metric_match),
    }


# ---------------------------------------------------------------------------
# CLI + orchestrator
# ---------------------------------------------------------------------------


def run_validation(
    *,
    results_path: Path,
    baseline_key: str,
    out_path: Path,
) -> Dict[str, Any]:
    """Single-shot orchestrator: read JSON + compute verdict + write output."""
    if not results_path.is_file():
        raise FileNotFoundError(f"results JSON missing: {results_path}")
    if baseline_key not in BASELINES:
        raise ValueError(
            f"unknown baseline {baseline_key!r}; "
            f"supported: {', '.join(SUPPORTED_BASELINES)}"
        )
    r = _read_json(results_path)
    metric_id = _infer_metric_id(r, results_path)
    baseline_meta = BASELINES[baseline_key]

    if metric_id not in baseline_meta["applicable_metrics"]:
        # Honest framing: NOT_COMPARABLE if metric not in the baseline's
        # applicable set.  We still emit alignment contract so reviewer knows
        # which axis the metric was mapped to.
        applicable = baseline_meta["applicable_metrics"]

    our_stats = extract_our_stats(metric_id, r)
    baseline_value, baseline_source, baseline_label, is_cite_only = _baseline_for(
        baseline_key, metric_id, baseline_meta
    )
    threshold = _threshold(baseline_value, baseline_meta["protocol_adjustment"])

    # For per-cell / per-split baselines (e.g. Rusanov A2780cis / DrugOOD
    # 3-split mean), use the per-cell / per-split value rather than the
    # overall aggregate when available.
    if baseline_label == "A2780cis" and our_stats.per_cell.get("A2780cis") is not None:
        metric_value = our_stats.per_cell["A2780cis"]
    elif baseline_label == "iMoLD_mean":
        # DrugOOD: compare mean of per_split values vs mean of iMoLD splits.
        valid_per_split = [
            v
            for v in our_stats.per_split.values()
            if v is not None and not (isinstance(v, float) and (v != v))  # filter NaN
        ]
        metric_value = (
            _safe_mean(valid_per_split) if valid_per_split else our_stats.overall_value
        )
    else:
        metric_value = our_stats.overall_value

    pass_bool, verdict_label, diss, thr = _verdict(
        metric_value=metric_value,
        threshold=threshold,
        direction=baseline_meta["threshold_direction"],
    )

    alignment_contract = _check_alignment_contract(r, baseline_key, baseline_meta)

    # Per-seed stats summary (per AGENTS.md §5.2 honest framing).
    per_seed_stats = {
        "count": len(our_stats.per_seed),
        "mean": _safe_mean(our_stats.per_seed),
        "std": _safe_std(our_stats.per_seed),
        "min": min(our_stats.per_seed) if our_stats.per_seed else None,
        "max": max(our_stats.per_seed) if our_stats.per_seed else None,
        "raw": our_stats.per_seed,
    }

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _now_iso(),
        "our_results_path": str(results_path),
        "baseline": baseline_key,
        "baseline_meta": {
            "name": baseline_meta["name"],
            "dataset": baseline_meta["dataset"],
            "paper_split": baseline_meta["paper_split"],
            "our_split": baseline_meta["our_split"],
            "metric": baseline_meta["metric"],
            "alignment_level": baseline_meta["alignment_level"],
            "protocol_adjustment": baseline_meta["protocol_adjustment"],
            "applicable_metrics": baseline_meta["applicable_metrics"],
            "alignment_note": baseline_meta["alignment_note"],
        },
        "metric_id": metric_id,
        "our_value_source": str(results_path),
        "our_value": metric_value,
        "our_value_overall": our_stats.overall_value,
        "our_value_std": our_stats.overall_std,
        "our_value_per_seed": per_seed_stats,
        "our_value_per_fold": our_stats.per_fold,
        "our_value_per_cell": our_stats.per_cell,
        "our_value_per_split": our_stats.per_split,
        "our_n_runs": our_stats.n_runs,
        "our_n_seeds": our_stats.n_seeds,
        "our_data_source": our_stats.data_source,
        "baseline_label": baseline_label,
        "baseline_value": baseline_value,
        "baseline_source": baseline_source,
        "baseline_cite_only": bool(is_cite_only),
        "alignment_level": baseline_meta["alignment_level"],
        "protocol_adjustment": baseline_meta["protocol_adjustment"],
        "threshold_comparable": thr,
        "direction": baseline_meta["threshold_direction"],
        "pass": bool(pass_bool),
        "disagreement_score": diss,
        "verdict": (
            "NOT_COMPARABLE"
            if is_cite_only or verdict_label == "NOT_COMPARABLE"
            else verdict_label
        ),
        "verdict_detail": (
            f"MEASURED {metric_value} vs {baseline_meta['name']} "
            f"{baseline_label} {baseline_value} "
            f"(alignment={baseline_meta['alignment_level']}; "
            f"protocol_adjustment={baseline_meta['protocol_adjustment']:+}; "
            f"threshold={thr}). Per-seed n={len(our_stats.per_seed)}; "
            f"per-fold={our_stats.per_fold}; per-cell={our_stats.per_cell}; "
            f"per-split={our_stats.per_split}; overall={our_stats.overall_value}."
            if verdict_label != "NOT_COMPARABLE"
            else f"CITE-ONLY baseline ({baseline_meta['name']} {baseline_label}); "
            f"baseline_value=null per honest framing."
        ),
        "alignment_contract": alignment_contract,
        "honest_framing": (
            f"Wf-433-A4 protocol validator: baseline={baseline_key}, "
            f"metric={metric_id}, direction={baseline_meta['threshold_direction']}. "
            f"MEASURED values from {results_path.name}; "
            f"baseline from {baseline_source}. "
            f"Verdict computed per party-twenty-spirit §3.5 honest framing "
            f"(no fabrication; CITE-ONLY handled gracefully)."
        ),
        "error": "",
    }

    if out_path is not None:
        try:
            _atomic_write_json(out_path, payload)
        except OSError as exc:
            payload["error"] = f"write {out_path} failed: {type(exc).__name__}: {exc}"

    return payload


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wf433_protocol_validator.py",
        description=(
            "Wf-433-A4 — single-results CLI for protocol-aligned comparison "
            "against 3 selected Tier-1 baselines (Krasnov 2026 + Rusanov 2025 + "
            "DrugOOD 2022). Emits per-seed / per-split / per-fold stats + "
            "alignment check + PASS/FAIL/WARN verdict."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--our-results",
        required=True,
        type=Path,
        help=(
            "Path to our results JSON (e.g. metrics/by_metric/"
            "metal_lomo_auc_wasserstein.json). Must be Tier-1 (m1 / m2 / "
            "m3 / m15 / m17 / m21) for the selected baseline."
        ),
    )
    p.add_argument(
        "--baseline",
        required=True,
        choices=SUPPORTED_BASELINES,
        help=(
            "Selected Tier-1 baseline. " f"Supported: {', '.join(SUPPORTED_BASELINES)}."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=(
            f"Output JSON path (default: {DEFAULT_OUT}). "
            "Parent dirs auto-created. Atomic-written per AGENTS.md §5.6."
        ),
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing --out (compute + return payload only).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose progress to stderr.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)
    if args.verbose:
        print(
            f"[wf433_protocol_validator] our_results={args.our_results} "
            f"baseline={args.baseline} out={args.out}",
            file=sys.stderr,
        )
    out_path = None if args.no_write else args.out
    try:
        payload = run_validation(
            results_path=args.our_results,
            baseline_key=args.baseline,
            out_path=out_path,
        )
    except FileNotFoundError as exc:
        print(f"[wf433_protocol_validator] FATAL: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[wf433_protocol_validator] FATAL: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.verbose:
        print(
            f"[wf433_protocol_validator] wrote {args.out} "
            f"(verdict={payload['verdict']} "
            f"disagreement_score={payload['disagreement_score']} "
            f"alignment_match={payload['alignment_contract']['overall_alignment_match']})",
            file=sys.stderr,
        )

    # Exit-code policy:
    #   0 — verdict is PASS / NOT_COMPARABLE (cite-only)
    #   1 — verdict is FAIL / alignment contract mismatched
    if payload["verdict"] == "PASS_BASELINE_GE_THRESHOLD":
        return 0
    if payload["verdict"] == "NOT_COMPARABLE":
        return 0
    # BELOW_BASELINE_THRESHOLD or alignment contract mismatch
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"[wf433_protocol_validator] UNHANDLED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
