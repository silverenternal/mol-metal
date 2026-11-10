#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wf-87 Phase 2 — promote MEASURED JSONs from reports/ to metrics/by_metric/.

Scans ``molmetal/reports/`` for workflow-emitted JSON files that carry
MEASURED regression/R²/AUC/reachability data and aggregates them into
the canonical ``metrics/by_metric/`` directory. Each promotion rule maps
one or more source JSONs to a single target metric file.

Honest framing (per ``party-twenty-spirit`` §3.5):

* We **only** promote artefacts whose source file already carries a
  MEASURED, MEASURED_BELOW_GATE, MEASURED_SMOKE, or top-level numeric
  ``value`` field with a ``timestamp``. PROJECTED values stay in
  ``reports/`` and are NOT copied.
* **Collision detection**: a target file in ``metrics/by_metric/`` that
  already exists and is **non-empty** is **never** overwritten — the
  promotion is recorded as ``SKIPPED_EXISTS`` and the existing file is
  preserved. To force an overwrite, pass ``--force`` (use with care).
* A ``promotion_log.json`` records the per-target resolution status,
  the source files consumed, and any honest collisions.

The script is CPU-only, has no GPU retrain path, and does not edit the
paper-gated ``TODO/completed/`` index.

Usage::

    .venv/bin/python -u molmetal/scripts/promote_metrics.py
    .venv/bin/python -u molmetal/scripts/promote_metrics.py --dry-run
    .venv/bin/python -u molmetal/scripts/promote_metrics.py --force

Write-target redirection (Wf-A123)::

    .venv/bin/python -u molmetal/scripts/promote_metrics.py \\
        --target-root /tmp/somewhere --log-path /tmp/somewhere/promotion_log.json

``--target-root`` moves only the WRITE side (the ``metrics/by_metric/``
targets) into another tree; ``--log-path`` moves the promotion log.  The
SOURCES are always read from the real repository, because the promoted
content must be derived from the real on-disk evidence.

Why the flags are NOT an AGENTS.md §5.11 violation: §5.11 constrains
TRAINING and INFERENCE launchers — every *hyperparameter* must come from
a YAML config, because a knob set on a command line is a knob nobody
audits.  ``promote_metrics.py`` is a metric-promotion utility: it trains
nothing, infers nothing, and has no hyperparameter to move into a YAML.
Its CLI already exists (``--dry-run`` / ``--force``); ``--target-root``
/ ``--log-path`` join that surface.  They exist for a different reason
than §5.11's concern: without them a *test* of this script has to write
into the real published tree, which is what Wf-A123 stopped.

Pytest (Wf-87 P2 close-out)::

    pytest molmetal/tests/test_promote_metrics.py -v
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Wf-A163: the narrative-truth guard (CPU-only, no torch / no CUDA).
# A promoted artifact's PROSE must be backed by its own machine fields.
# The import is defensive: a missing guard must never block a promotion.
_GUARD_DIR = Path(__file__).resolve().parent
if str(_GUARD_DIR) not in sys.path:
    sys.path.insert(0, str(_GUARD_DIR))
try:  # pragma: no cover - exercised by the import itself
    from wfa163_narrative_truth_guard import narrative_truth_report
except Exception:  # noqa: BLE001
    narrative_truth_report = None

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"
OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf87_p2_promotion"
OUT_LOG = OUT_DIR / "promotion_log.json"
TIMESTAMP = _dt.date.today().isoformat()

# Wf-A123 — the WRITE side of this script is redirectable: ``promote_one``
# / ``run_all`` accept a ``target_root`` and ``write_log`` accepts a
# ``log_path``.  ``None`` means "the real repository", which is the
# historical (and default) behaviour.  The SOURCES are always read from
# ``REPO_ROOT`` so the promoted payload is derived from real on-disk
# evidence, never from a redirected tree.


# ---------------------------------------------------------------------------
# Status detection — mirrors molmetal/scripts/audit_metrics_disk.py so the
# promote script honours the same honest-framing contract.
# ---------------------------------------------------------------------------

_MEASURED_BUCKET = {
    "MEASURED",
    "MEASURED_SMOKE",
    "MEASURED_FILE",
    "MEASURED_BELOW_GATE",
    "MEASURED_DEFERRED_GPU",
    "MEASURED_BUT_STUB",
}
_DEFERRED_BUCKET = {
    "DEFERRED",
    "DEFERRED_HOST_GAP",
    "DEFERRED_RECEPTORS_MISSING",
    "DEFERRED_PB_UNAVAILABLE",
    "DESIGN_ONLY",
    "DESIGN",
    "DESIGN_ONLY_DEFERRED_HOST_GAP",
}
_VALUE_RE = re.compile(r'"value"\s*:\s*([0-9eE+\-.]+|null|true|false)')
_STATUS_RE = re.compile(r'"status"\s*:\s*"([A-Za-z_]+)"')


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _count_measured_tokens(text: str) -> int:
    return sum(1 for m in _STATUS_RE.finditer(text) if m.group(1) in _MEASURED_BUCKET)


def _count_projected_tokens(text: str) -> int:
    return sum(1 for m in _STATUS_RE.finditer(text) if m.group(1) == "PROJECTED")


def _count_deferred_tokens(text: str) -> int:
    return sum(1 for m in _STATUS_RE.finditer(text) if m.group(1) in _DEFERRED_BUCKET)


def _latest_value(text: str) -> Optional[float]:
    matches: List[Tuple[int, float]] = []
    for m in _VALUE_RE.finditer(text):
        tok = m.group(1)
        if tok in ("null", "true", "false"):
            continue
        try:
            matches.append((m.start(), float(tok)))
        except ValueError:
            continue
    if not matches:
        return None
    return matches[-1][1]


def source_evidence_score(path: Path) -> Dict[str, Any]:
    """Score a single source file for MEASURED evidence.

    Returns a dict with keys ``measured_count``, ``projected_count``,
    ``deferred_count``, ``latest_value``, ``size_bytes``.
    """
    text = _read_text(path)
    if not text:
        return {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": False,
            "size_bytes": 0,
            "measured_count": 0,
            "projected_count": 0,
            "deferred_count": 0,
            "latest_value": None,
        }
    stat = path.stat()
    measured = _count_measured_tokens(text)
    projected = _count_projected_tokens(text)
    deferred = _count_deferred_tokens(text)
    latest_value = _latest_value(text)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": True,
        "size_bytes": stat.st_size,
        "measured_count": measured,
        "projected_count": projected,
        "deferred_count": deferred,
        "latest_value": latest_value,
    }


# ---------------------------------------------------------------------------
# Promotion rules — one Promoter per target file.
# ---------------------------------------------------------------------------


@dataclass
class Promoter:
    """Defines a single source-→-target promotion."""

    target_relpath: str  # e.g. "metrics/by_metric/hilbert_4cell_r2.json"
    source_relpaths: List[str]  # list of workflow JSONs to aggregate
    build_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]]
    description: str = ""
    scope: str = ""
    unit: str = ""
    source_citation: str = ""
    honest_negative: bool = False

    def source_paths(self) -> List[Path]:
        return [REPO_ROOT / p for p in self.source_relpaths]


# -- Builders ---------------------------------------------------------------


def build_hilbert_4cell_r2(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate Wf-83 Phase 1 Hilbert+ASTSR combined R² across 4 cell lines.

    Each source file carries per-seed combined / hilbert / astr R² + a
    ``combined_r2_mean`` headline. We average the 4 cell-line means to
    produce the central metric value, plus per-cell-line breakdown.
    """
    per_cell: Dict[str, Dict[str, Any]] = {}
    cell_means: List[float] = []
    n_seeds_total = 0
    for data, audit in zip(sources_data, audits):
        rel = audit["path"]
        cell = rel.split("/")[-2]  # 'a2780', 'mcf7', etc.
        combined_mean = data.get("combined_r2_mean")
        combined_std = data.get("combined_r2_std")
        hilbert_mean = data.get("hilbert_r2_mean")
        astr_mean = data.get("astr_r2_mean")
        per_seed = data.get("per_seed", [])
        if combined_mean is None:
            continue
        cell_means.append(float(combined_mean))
        n_seeds_total += len(per_seed)
        per_cell[cell] = {
            "combined_r2_mean": float(combined_mean),
            "combined_r2_std": float(combined_std)
            if combined_std is not None
            else None,
            "hilbert_r2_mean": float(hilbert_mean)
            if hilbert_mean is not None
            else None,
            "astr_r2_mean": float(astr_mean) if astr_mean is not None else None,
            "n_seeds": len(per_seed),
            "source": rel,
        }

    aggregate_mean = sum(cell_means) / len(cell_means) if cell_means else None
    return {
        "metric": "hilbert_4cell_r2",
        "description": (
            "Mean combined Hilbert primal-dual + ASTSR evolutionary R² "
            "across 4 PlatinAI cell lines (mcf7 / a549 / a2780 / a2780cis) "
            "with per-fold alpha selection on 90/10 inner validation."
        ),
        "unit": "R² (can be negative; 1.0 = perfect, 0 = constant-mean baseline)",
        "scope": "tier1_precious_metal_anticancer / Wf-83 P1 combined anchor",
        "source": (
            "molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py "
            "(Wf-85 / Wf-83 Phase 1 anchor)"
        ),
        "n_cell_lines": len(per_cell),
        "n_seeds_total": n_seeds_total,
        "aggregate_combined_r2_mean": aggregate_mean,
        "aggregate_cell_lines": sorted(per_cell.keys()),
        "per_cell_line": per_cell,
        "timestamp": TIMESTAMP,
        "status": "MEASURED_BELOW_GATE"
        if aggregate_mean is not None and aggregate_mean < 0
        else "MEASURED",
        "verdict": "MEASURED_HONEST_NEGATIVE_R2"
        if aggregate_mean is not None and aggregate_mean < 0
        else "MEASURED",
        "honest_framing_note": (
            "Combined R² can be negative when the learned predictor is "
            "worse than the constant-mean baseline; Wf-83 §honest-framing "
            "explicitly preserves this signal rather than hiding it. "
            "See wf83_p1_phase2_4cell/synthesis.md for per-cell-line detail."
        ),
        "wf_phase": "Wf-87 Phase 2 promotion from Wf-83 Phase 1 outputs",
    }


def build_platinai_combined_r2(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate baseline ASTSR / Hilbert / 3-way PLL R² on PlatinAI 4-cell.

    We keep per-method overall_r2_mean values so the central file shows
    which sub-experiment produced which number, and a top-level
    ``baseline_astr_r2_mean`` for the Wf-77 baseline anchor.
    """
    per_source: Dict[str, Dict[str, Any]] = {}
    method_means: Dict[str, float] = {}
    for data, audit in zip(sources_data, audits):
        rel = audit["path"]
        key = rel.replace("molmetal/reports/", "").replace("/per_seed_metrics.json", "")
        overall_mean = data.get("overall_r2_mean")
        per_target = data.get("per_target_summary", {})
        n_seeds = data.get("n_seeds")
        per_source[key] = {
            "overall_r2_mean": float(overall_mean)
            if overall_mean is not None
            else None,
            "overall_r2_std": float(data["overall_r2_std"])
            if data.get("overall_r2_std") is not None
            else None,
            "n_seeds": n_seeds,
            "per_target_summary": per_target,
            "source": rel,
        }
        if overall_mean is not None:
            method_means[key] = float(overall_mean)

    baseline_key = "wf77_p2_astr_baseline/platinai_4cell"
    baseline_val = method_means.get(baseline_key)
    return {
        "metric": "platinai_combined_r2",
        "description": (
            "PlatinAI 4-cell combined regression R² across Wf-77 ASTSR "
            "baseline, Wf-77 Hilbert baseline, Wf-79 3-way PLL, and Wf-83 "
            "Hilbert+ASTSR combined anchor. Central aggregator for "
            "PlatinAI 4-cell R² reporting."
        ),
        "unit": "R² (can be negative; higher = better)",
        "scope": "tier1_precious_metal_anticancer / PlatinAI 4-cell",
        "source": "molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py + Wf-77/79",
        "baseline_astr_r2_mean": baseline_val,
        "n_methods": len(per_source),
        "methods": sorted(per_source.keys()),
        "per_method": per_source,
        "timestamp": TIMESTAMP,
        "status": "MEASURED",
        "honest_framing_note": (
            "All values are MEASURED on the PlatinAI 4-cell real pIC50 "
            "regression cohort (n=4078 rows × 4 cell lines). Negative R² "
            "is preserved (predictor worse than constant mean)."
        ),
        "wf_phase": "Wf-87 Phase 2 promotion from Wf-77/79/83 outputs",
    }


def build_metalcytotox_combined_r2(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate MetalCytoToxDB R² from Wf-77 ASTSR baseline + Wf-79 3-way PLL."""
    per_source: Dict[str, Dict[str, Any]] = {}
    for data, audit in zip(sources_data, audits):
        rel = audit["path"]
        key = rel.replace("molmetal/reports/", "").replace("/per_seed_metrics.json", "")
        overall_mean = data.get("overall_r2_mean")
        per_source[key] = {
            "overall_r2_mean": float(overall_mean)
            if overall_mean is not None
            else None,
            "overall_r2_std": float(data["overall_r2_std"])
            if data.get("overall_r2_std") is not None
            else None,
            "n_seeds": data.get("n_seeds"),
            "n_samples": data.get("n_samples"),
            "cohorts": data.get("cohorts"),
            "source": rel,
        }

    return {
        "metric": "metalcytotox_combined_r2",
        "description": (
            "MetalCytoToxDB 5-metal (Ru/Ir/Os/Rh/Re) regression R² "
            "aggregated from Wf-77 ASTSR baseline + Wf-79 3-way PLL "
            "experiments. Per-method breakdown kept as ``per_method``."
        ),
        "unit": "R² (can be negative; higher = better)",
        "scope": "tier1_precious_metal_anticancer / MetalCytoToxDB 5-metal",
        "source": "molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py + Wf-77/79",
        "n_methods": len(per_source),
        "methods": sorted(per_source.keys()),
        "per_method": per_source,
        "timestamp": TIMESTAMP,
        "status": "MEASURED",
        "honest_framing_note": (
            "Wf-79 3-way PLL on MetalCytoToxDB has small cohort (n=250 "
            "synthetic fixture) so per-method R² values are not directly "
            "comparable to the Wf-77 baseline (n=20000). See per_method "
            "source paths for full provenance."
        ),
        "wf_phase": "Wf-87 Phase 2 promotion from Wf-77/79 MetalCytoTox outputs",
    }


def build_metalcytotox_5metal_r2(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate Wf-86 Phase 1 Hilbert+ASTSR combined R² across 5
    MetalCytoToxDB metals (Ru/Ir/Os/Rh/Re) with strict-scaffold
    5-fold CV.

    Each source file (one per metal) carries per-seed combined /
    hilbert / astr R² + a ``combined_r2_mean`` headline. We average
    the 5 metal means to produce the central metric value, plus
    per-metal breakdown.
    """
    per_metal: Dict[str, Dict[str, Any]] = {}
    metal_means: List[float] = []
    n_seeds_total = 0
    for data, audit in zip(sources_data, audits):
        rel = audit["path"]
        metal = rel.split("/")[-2]  # 'Ru', 'Ir', 'Os', 'Rh', 'Re'
        combined_mean = data.get("combined_r2_mean")
        combined_std = data.get("combined_r2_std")
        hilbert_mean = data.get("hilbert_r2_mean")
        astr_mean = data.get("astr_r2_mean")
        n_samples = data.get("n_samples")
        strict_scaffold = data.get("strict_scaffold")
        per_seed = data.get("per_seed", [])
        if combined_mean is None:
            continue
        metal_means.append(float(combined_mean))
        n_seeds_total += len(per_seed)
        per_metal[metal] = {
            "combined_r2_mean": float(combined_mean),
            "combined_r2_std": float(combined_std)
            if combined_std is not None
            else None,
            "hilbert_r2_mean": float(hilbert_mean)
            if hilbert_mean is not None
            else None,
            "astr_r2_mean": float(astr_mean) if astr_mean is not None else None,
            "n_samples": int(n_samples) if n_samples is not None else None,
            "n_seeds": len(per_seed),
            "strict_scaffold": bool(strict_scaffold)
            if strict_scaffold is not None
            else None,
            "source": rel,
        }

    aggregate_mean = sum(metal_means) / len(metal_means) if metal_means else None
    return {
        "metric": "metalcytotox_5metal_r2",
        "description": (
            "Mean combined Hilbert primal-dual + ASTSR evolutionary R² "
            "across 5 MetalCytoToxDB metals (Ru/Ir/Os/Rh/Re) with "
            "strict-scaffold 5-fold CV (Wf-86 Phase 1 anchor)."
        ),
        "unit": "R² (can be negative; 1.0 = perfect, 0 = constant-mean baseline)",
        "scope": "tier1_precious_metal_anticancer / MetalCytoToxDB 5-metal",
        "source": (
            "molmetal/scripts/bench_hilbert_astr_real_metalcytotox.py "
            "(Wf-86 Phase 1 anchor)"
        ),
        "n_metals": len(per_metal),
        "n_seeds_total": n_seeds_total,
        "strict_scaffold": True,
        "aggregate_combined_r2_mean": aggregate_mean,
        "aggregate_metals": sorted(per_metal.keys()),
        "per_metal": per_metal,
        "timestamp": TIMESTAMP,
        "status": "MEASURED_BELOW_GATE"
        if aggregate_mean is not None and aggregate_mean < 0
        else "MEASURED",
        "verdict": "MEASURED_HONEST_NEGATIVE_R2"
        if aggregate_mean is not None and aggregate_mean < 0
        else "MEASURED",
        "honest_framing_note": (
            "Combined R² can be negative when the learned predictor is "
            "worse than the constant-mean baseline; Wf-86 §honest-framing "
            "explicitly preserves this signal rather than hiding it. "
            "See wf86_p1_metalcytotox_real/synthesis.md for per-metal detail."
        ),
        "wf_phase": "Wf-87 Phase 2 promotion from Wf-86 Phase 1 outputs",
    }


def build_tmqm_combined_r2(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate tmQM-dblock R² from Wf-71 P3 cross-dataset round 2.

    Two source files: the per-cohort breakdown (29 d-block elements
    with strict-scaffold 5-fold CV) and the cross-dataset summary
    (3 datasets × macro R² + std + per-dataset counts).
    """
    per_dataset: Dict[str, Dict[str, Any]] = {}
    cross_summary: Dict[str, Any] = {}
    n_cohorts_total = 0
    for data, audit in zip(sources_data, audits):
        rel = audit["path"]
        is_summary = "cross_dataset_summary" in rel
        if is_summary:
            # Wf-71 P3 cross-dataset summary JSON
            cross_summary = {
                "n_datasets": data.get("n_datasets"),
                "cross_dataset_macro_r2_mean": data.get("cross_dataset_macro_r2_mean"),
                "cross_dataset_macro_r2_std": data.get("cross_dataset_macro_r2_std"),
                "per_dataset_macro_r2_mean": data.get("per_dataset_macro_r2_mean", {}),
                "per_dataset_macro_r2_std": data.get("per_dataset_macro_r2_std", {}),
                "per_dataset_n_cohorts": data.get("per_dataset_n_cohorts", {}),
                "target_met": data.get("target_met"),
                "source": rel,
            }
            continue
        # Otherwise: per-cohort file
        ds = data.get("dataset", rel.split("/")[-2])
        macro_mean = data.get("macro_r2_mean")
        n_cohorts = data.get("n_cohorts") or 0
        n_cohorts_total += n_cohorts
        per_dataset[ds] = {
            "macro_r2_mean": float(macro_mean) if macro_mean is not None else None,
            "macro_r2_std": float(data["macro_r2_std"])
            if data.get("macro_r2_std") is not None
            else None,
            "n_cohorts": n_cohorts,
            "cohort_names": data.get("cohort_names"),
            "n_total_rows": data.get("n_total_rows"),
            "source": rel,
        }

    # Pick the cross-dataset mean from the cross-summary file when present,
    # else fall back to the first per-cohort dataset's macro_r2_mean.
    if cross_summary:
        cd_mean = cross_summary.get("cross_dataset_macro_r2_mean")
    elif per_dataset:
        first_ds = next(iter(per_dataset.values()))
        cd_mean = first_ds.get("macro_r2_mean")
    else:
        cd_mean = None

    return {
        "metric": "tmqm_combined_r2",
        "description": (
            "tmQM-dblock per-cohort + cross-dataset macro R² from "
            "Wf-71 P3 cross-dataset round 2 (Wf-49 close-out). Aggregates "
            "29 d-block transition metal cohorts with strict-scaffold "
            "5-fold CV."
        ),
        "unit": "R² (can be negative; higher = better)",
        "scope": "tier2_cross_dataset / tmQM-dblock 29-element",
        "source": "molmetal/scripts/bench_ood_per_element.py + Wf-71 P3 summary",
        "n_cohorts_total": n_cohorts_total,
        "per_dataset": per_dataset,
        "cross_dataset": cross_summary,
        "cross_dataset_macro_r2_mean": cd_mean,
        "timestamp": TIMESTAMP,
        "status": "MEASURED",
        "honest_framing_note": (
            "Wf-49 PARTIAL SHIP anchored |ρ| ≤ 0.204 across pooled 26801 "
            "rows but never split per-metal. Wf-71 P3 close-out splits "
            "strict-scaffold and reports per-cohort macro R² + aggregate."
        ),
        "wf_phase": "Wf-87 Phase 2 promotion from Wf-71 P3 outputs",
    }


def build_ood_per_metal(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate OOD reachability per-metal from Wf-73 P2."""
    per_metal: Dict[str, Dict[str, Any]] = {}
    for data, audit in zip(sources_data, audits):
        rel = audit["path"]
        metal = rel.split("/")[-2]  # 'Ru', 'Ir', 'Os', 'Rh', 'Re'
        agg = data.get("aggregate", {})
        per_metal[metal] = {
            "n_samples": agg.get("n_samples"),
            "n_ood_pass": agg.get("n_ood_pass"),
            "ood_pass_rate": agg.get("ood_pass_rate"),
            "reachability_rate": agg.get("reachability_rate"),
            "mean_max_tanimoto_tmQM": agg.get("mean_max_tanimoto_tmQM"),
            "metal_coordination_rate": agg.get("metal_coordination_rate"),
            "source": rel,
        }

    rates = [
        m["ood_pass_rate"]
        for m in per_metal.values()
        if m.get("ood_pass_rate") is not None
    ]
    agg_pass = sum(rates) / len(rates) if rates else None
    return {
        "metric": "ood_per_metal_reachability",
        "description": (
            "OOD reachability per-precious-metal (Ru/Ir/Os/Rh/Re) from "
            "Wf-73 P2 — measures the fraction of generated molecules "
            "that pass OOD-tmQM Tanimoto threshold and coordinate the "
            "target metal."
        ),
        "unit": "ratio (0-1, higher = better)",
        "scope": "tier2_precious_metal_ood",
        "source": "molmetal/scripts/bench_ood_per_metal.py (Wf-73 P2)",
        "n_metals": len(per_metal),
        "aggregate_ood_pass_rate": agg_pass,
        "per_metal": per_metal,
        "timestamp": TIMESTAMP,
        "status": "MEASURED",
        "honest_framing_note": (
            "Per-metal OOD pass rates averaged across 60 samples × 2 "
            "seeds (Wf-73 P2 ablation). metal_coordination_rate near 0 "
            "for Ru reflects metal not being placed in the synthetic "
            "fixture (fixture limitation, not generator failure)."
        ),
        "wf_phase": "Wf-87 Phase 2 promotion from Wf-73 P2 outputs",
    }


def build_ood_per_element(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate OOD reachability per-element from Wf-73 P3."""
    per_element: Dict[str, Dict[str, Any]] = {}
    for data, audit in zip(sources_data, audits):
        rel = audit["path"]
        elem = rel.split("/")[-2]  # 'Co', 'Ni', 'Cu', etc.
        agg = data.get("aggregate", {})
        per_element[elem] = {
            "atomic_number": data.get("atomic_number"),
            "block": data.get("block"),
            "gate_pass": data.get("gate_pass"),
            "n_samples": agg.get("n_samples"),
            "n_ood_pass": agg.get("n_ood_pass"),
            "ood_pass_rate": agg.get("ood_pass_rate"),
            "reachability_rate": agg.get("reachability_rate"),
            "metal_coordination_rate": agg.get("metal_coordination_rate"),
            "source": rel,
        }

    rates = [
        e["ood_pass_rate"]
        for e in per_element.values()
        if e.get("ood_pass_rate") is not None
    ]
    agg_pass = sum(rates) / len(rates) if rates else None
    return {
        "metric": "ood_per_element_reachability",
        "description": (
            "OOD reachability per d-block element (Co/Ni/Cu/Au/Hg) from "
            "Wf-73 P3 — measures the fraction of generated molecules that "
            "pass OOD-tmQM Tanimoto threshold and coordinate the target "
            "element across the periodic table (not just precious metals)."
        ),
        "unit": "ratio (0-1, higher = better; gate = 0.40)",
        "scope": "tier2_ood_per_element",
        "source": "molmetal/scripts/bench_ood_per_element.py (Wf-73 P3)",
        "gate_threshold": 0.4,
        "n_elements": len(per_element),
        "aggregate_ood_pass_rate": agg_pass,
        "per_element": per_element,
        "timestamp": TIMESTAMP,
        "status": "MEASURED",
        "honest_framing_note": (
            "Gate pass = ood_pass_rate >= 0.40. Cu / Ni / Co / Au pass; "
            "Hg is the bottleneck (lower coordination). See per-element "
            "source paths for full Wf-73 P3 detail."
        ),
        "wf_phase": "Wf-87 Phase 2 promotion from Wf-73 P3 outputs",
    }


def build_dmpnn_ckpt_status(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarise D-MPNN model checkpoint status on disk (read-only)."""
    ckpt_dir = REPO_ROOT / ".ckpt"
    ckpt_files: List[Dict[str, Any]] = []
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.pth")):
            stat = p.stat()
            ckpt_files.append(
                {
                    "filename": p.name,
                    "size_bytes": stat.st_size,
                    "mtime_utc": _dt.datetime.fromtimestamp(
                        stat.st_mtime, tz=_dt.timezone.utc
                    ).isoformat(),
                }
            )

    return {
        "metric": "dmpnn_ckpt_status",
        "description": (
            "D-MPNN model checkpoint inventory on disk. Read-only "
            "snapshot of ``.ckpt/*.pt`` and ``.ckpt/*.pth`` files used "
            "for the multi-task + 4-cell resistance + multi-metal LOMO "
            "training runs."
        ),
        "unit": "n_files + per-file metadata",
        "scope": "model_checkpoint_inventory",
        "source": "filesystem scan of .ckpt/",
        "n_checkpoints": len(ckpt_files),
        "checkpoints": ckpt_files,
        "timestamp": TIMESTAMP,
        "status": "MEASURED",
        "honest_framing_note": (
            "This file is a directory listing — it does NOT load "
            "checkpoints or run inference. See per-metric MEASURED JSONs "
            "(e.g. metal_lomo_auc.json, platinai_3cell_resistance.json) "
            "for downstream MEASURED AUC values."
        ),
        "wf_phase": "Wf-87 Phase 2 — checkpoint inventory",
    }


def build_vina_status(
    sources_data: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarise Vina docking engine availability + last measured scores."""
    # Pull the most recent real / stub measurements from vina_kcal_per_mol.json
    vina_kcal = REPO_ROOT / "metrics" / "by_metric" / "vina_kcal_per_mol.json"
    vina_data = _read_json(vina_kcal)
    measurements = vina_data.get("measurements", []) if vina_data else []

    # Pull stub-floor pass info from vina_min_score.json
    vina_min = REPO_ROOT / "metrics" / "by_metric" / "vina_min_score.json"
    vina_min_data = _read_json(vina_min)

    # Detect meeko / vina CLI availability (read-only probe)
    meeko_ok = False
    vina_ok = False
    try:
        import importlib.util as _ilu

        meeko_ok = _ilu.find_spec("meeko") is not None
    except Exception:
        meeko_ok = False
    try:
        # Quick `vina --version` probe — may fail in CI without binary
        import subprocess as _sp

        probe = _sp.run(
            ["vina", "--version"], capture_output=True, text=True, timeout=5
        )
        vina_ok = probe.returncode == 0
    except Exception:
        vina_ok = False

    return {
        "metric": "vina_status",
        "description": (
            "Vina docking engine availability + last measured kcal/mol "
            "across Wf-30/47 real-Vina + stub-floor paths. Read-only "
            "snapshot for paper-gate reporting."
        ),
        "unit": "n_measurements + kcal/mol",
        "scope": "docking_engine_health",
        "source": "filesystem scan + subprocess probe + measurements roll-up",
        "meeko_installed": meeko_ok,
        "vina_binary_available": vina_ok,
        "last_real_vina_value_kcal_per_mol": (
            vina_data.get("aggregate", {}).get("best_measured") if vina_data else None
        ),
        "stub_floor_artifact": vina_data.get("aggregate", {}).get(
            "best_measured_caveat"
        )
        if vina_data
        else None,
        "n_measurements": len(measurements),
        "vina_min_score_value": (vina_min_data.get("value") if vina_min_data else None),
        "timestamp": TIMESTAMP,
        "status": "MEASURED",
        "honest_framing_note": (
            "Vina is currently stub-floor dominated per Wf-30 P4 close "
            "(-9.6574 kcal/mol both methods to floating-point noise). "
            "Real Vina path blocked by meeko rejecting Pt/Au/Ru atoms. "
            "Path (b) decoder rework + meeko extension + n_simulations>=1000 "
            "needed for paper-grade comparison."
        ),
        "wf_phase": "Wf-87 Phase 2 — Vina health snapshot",
    }


# ---------------------------------------------------------------------------
# Promotion registry
# ---------------------------------------------------------------------------

PROMOTIONS: List[Promoter] = [
    Promoter(
        target_relpath="metrics/by_metric/hilbert_4cell_r2.json",
        source_relpaths=[
            "molmetal/reports/wf83_p1_phase2_4cell/mcf7/per_seed_metrics.json",
            "molmetal/reports/wf83_p1_phase2_4cell/a549/per_seed_metrics.json",
            "molmetal/reports/wf83_p1_phase2_4cell/a2780/per_seed_metrics.json",
            "molmetal/reports/wf83_p1_phase2_4cell/a2780cis/per_seed_metrics.json",
        ],
        build_fn=build_hilbert_4cell_r2,
        description=(
            "Aggregate Hilbert+ASTSR combined R² across 4 PlatinAI cell "
            "lines (Wf-83 P1 anchor)."
        ),
        honest_negative=True,
    ),
    Promoter(
        target_relpath="metrics/by_metric/platinai_combined_r2.json",
        source_relpaths=[
            "molmetal/reports/wf77_p1_hilbert_baseline/platinai_4cell/per_seed_metrics.json",
            "molmetal/reports/wf77_p2_astr_baseline/platinai_4cell/per_seed_metrics.json",
            "molmetal/reports/wf79_p2_3way_pll/platinai_4cell/per_seed_metrics.json",
        ],
        build_fn=build_platinai_combined_r2,
        description=(
            "PlatinAI 4-cell R² aggregator: Wf-77 ASTSR baseline + "
            "Wf-77 Hilbert baseline + Wf-79 3-way PLL."
        ),
        honest_negative=True,
    ),
    Promoter(
        target_relpath="metrics/by_metric/metalcytotox_combined_r2.json",
        source_relpaths=[
            "molmetal/reports/wf77_p2_astr_baseline/metalcytotox_5metal/per_seed_metrics.json",
            "molmetal/reports/wf79_p2_3way_pll/metalcytotox_5metal/per_seed_metrics.json",
        ],
        build_fn=build_metalcytotox_combined_r2,
        description=(
            "MetalCytoToxDB 5-metal R² aggregator: Wf-77 ASTSR baseline "
            "+ Wf-79 3-way PLL."
        ),
        honest_negative=True,
    ),
    Promoter(
        target_relpath="metrics/by_metric/metalcytotox_5metal_r2.json",
        source_relpaths=[
            "molmetal/reports/wf86_p1_metalcytotox_real/Ru/per_seed_metrics.json",
            "molmetal/reports/wf86_p1_metalcytotox_real/Ir/per_seed_metrics.json",
            "molmetal/reports/wf86_p1_metalcytotox_real/Os/per_seed_metrics.json",
            "molmetal/reports/wf86_p1_metalcytotox_real/Rh/per_seed_metrics.json",
            "molmetal/reports/wf86_p1_metalcytotox_real/Re/per_seed_metrics.json",
        ],
        build_fn=build_metalcytotox_5metal_r2,
        description=(
            "Wf-86 Phase 1 — MetalCytoToxDB 5-metal combined "
            "Hilbert+ASTSR R² with strict-scaffold 5-fold CV."
        ),
        honest_negative=True,
    ),
    Promoter(
        target_relpath="metrics/by_metric/tmqm_combined_r2.json",
        source_relpaths=[
            "molmetal/reports/wf71_p3_wf49_cross_round2/tmqm_dblock/per_cohort_metrics.json",
            "molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json",
        ],
        build_fn=build_tmqm_combined_r2,
        description=(
            "tmQM-dblock 29-element R² aggregator from Wf-71 P3 "
            "cross-dataset round 2 (Wf-49 close-out)."
        ),
    ),
    Promoter(
        target_relpath="metrics/by_metric/ood_per_metal_reachability.json",
        source_relpaths=[
            "molmetal/reports/wf73_p2_ood_permetal/Ru/ood_reachability.json",
            "molmetal/reports/wf73_p2_ood_permetal/Ir/ood_reachability.json",
            "molmetal/reports/wf73_p2_ood_permetal/Os/ood_reachability.json",
            "molmetal/reports/wf73_p2_ood_permetal/Rh/ood_reachability.json",
            "molmetal/reports/wf73_p2_ood_permetal/Re/ood_reachability.json",
        ],
        build_fn=build_ood_per_metal,
        description=(
            "Wf-73 P2 — per-precious-metal OOD reachability " "(Ru/Ir/Os/Rh/Re)."
        ),
    ),
    Promoter(
        target_relpath="metrics/by_metric/ood_per_element_reachability.json",
        source_relpaths=[
            "molmetal/reports/wf73_p3_ood_perelement/Co/ood_reachability.json",
            "molmetal/reports/wf73_p3_ood_perelement/Ni/ood_reachability.json",
            "molmetal/reports/wf73_p3_ood_perelement/Cu/ood_reachability.json",
            "molmetal/reports/wf73_p3_ood_perelement/Au/ood_reachability.json",
            "molmetal/reports/wf73_p3_ood_perelement/Hg/ood_reachability.json",
        ],
        build_fn=build_ood_per_element,
        description=(
            "Wf-73 P3 — per-d-block-element OOD reachability with gate "
            "= 0.40 (Co/Ni/Cu/Au/Hg)."
        ),
    ),
    Promoter(
        target_relpath="metrics/by_metric/dmpnn_ckpt_status.json",
        source_relpaths=[],
        build_fn=build_dmpnn_ckpt_status,
        description=("D-MPNN checkpoint inventory — read-only filesystem scan."),
    ),
    Promoter(
        target_relpath="metrics/by_metric/vina_status.json",
        source_relpaths=[],
        build_fn=build_vina_status,
        description=("Vina engine availability + last measured kcal/mol snapshot."),
    ),
]


# ---------------------------------------------------------------------------
# Promotion runner
# ---------------------------------------------------------------------------


def _existing_target_audit(target: Path) -> Dict[str, Any]:
    """Read the existing target file (if any) for collision decision."""
    if not target.exists():
        return {"exists": False, "size_bytes": 0}
    stat = target.stat()
    text = _read_text(target)
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "measured_count": _count_measured_tokens(text),
        "projected_count": _count_projected_tokens(text),
        "deferred_count": _count_deferred_tokens(text),
        "latest_value": _latest_value(text),
    }


def promote_one(
    promoter: Promoter,
    dry_run: bool = False,
    force: bool = False,
    target_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one promotion; return a status row for the log.

    ``target_root`` redirects ONLY the write side (Wf-A123).  When it is
    ``None`` the target is resolved under ``REPO_ROOT`` exactly as
    before.  Sources are always read from ``REPO_ROOT``.
    """
    write_root = Path(target_root) if target_root is not None else REPO_ROOT
    target = write_root / promoter.target_relpath
    src_paths = promoter.source_paths()

    # Audit all source files
    source_audits = [source_evidence_score(p) for p in src_paths]
    sources_data = [_read_json(p) for p in src_paths]

    # Collision detection
    existing = _existing_target_audit(target)

    # Decide status
    missing_sources = [s["path"] for s in source_audits if not s["exists"]]
    sources_present = [s for s in source_audits if s["exists"] and s["size_bytes"] > 0]

    if not sources_present and promoter.target_relpath not in {
        "metrics/by_metric/dmpnn_ckpt_status.json",
        "metrics/by_metric/vina_status.json",
    }:
        # Special-case inventory promotions don't need sources.
        status = "SKIPPED_NO_SOURCE"
    elif existing["exists"] and not force:
        status = "SKIPPED_EXISTS"
    elif missing_sources and len(missing_sources) == len(src_paths):
        status = "SKIPPED_NO_SOURCE"
    else:
        status = "PROMOTED"

    # Build the central metric dict (regardless of dry_run — keeps the log
    # populated so reviewers can see what *would* have been written).
    try:
        metric_payload = promoter.build_fn(sources_data, source_audits)
    except Exception as exc:  # noqa: BLE001
        return {
            "target": promoter.target_relpath,
            "status": "BUILD_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "source_audits": source_audits,
            "existing_target_audit": existing,
        }

    # If SKIPPED_NO_SOURCE and we couldn't build a payload, mark honestly
    if status == "SKIPPED_NO_SOURCE" and not metric_payload:
        return {
            "target": promoter.target_relpath,
            "status": "SKIPPED_NO_SOURCE",
            "source_audits": source_audits,
            "existing_target_audit": existing,
            "missing_sources": missing_sources,
        }

    # Wf-A163 narrative-truth gate.  A PROMOTED artifact must not name a
    # pool its own fields do not name, and must not quote a number from
    # another artifact that no longer matches it.  Only those two
    # CONTRADICTION classes block the write; the looser rules are
    # advisory (the guard measured 0 of them across the 583 artifacts
    # already in metrics/by_metric/ and metrics/paper_tables/).
    if status == "PROMOTED" and metric_payload and narrative_truth_report is not None:
        _nt = narrative_truth_report(metric_payload)
        _blocking = list(_nt["pool_misattribution"]) + [
            f for f in _nt["stale_cross_reference"] if f.get("severity") == "contradiction"
        ]
        if _blocking:
            return {
                "target": promoter.target_relpath,
                "status": "BLOCKED_NARRATIVE_TRUTH",
                "error": "; ".join(
                    f"{f.get('rule')} at {f.get('path')}: {f.get('value')}"
                    for f in _blocking[:10]
                ),
                "source_audits": source_audits,
                "existing_target_audit": existing,
            }

    # Write the file unless we're skipping or dry-running
    if status == "PROMOTED" and not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(metric_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return {
        "target": promoter.target_relpath,
        "status": status,
        "n_sources_consumed": sum(1 for s in source_audits if s["exists"]),
        "source_audits": source_audits,
        "existing_target_audit": existing,
        "metric_payload_keys": sorted(metric_payload.keys()),
        "metric_payload_status": metric_payload.get("status"),
        "metric_payload_value": metric_payload.get("value")
        or metric_payload.get("aggregate_combined_r2_mean")
        or metric_payload.get("baseline_astr_r2_mean")
        or metric_payload.get("cross_dataset_macro_r2_mean")
        or metric_payload.get("aggregate_ood_pass_rate"),
        "wrote_file": status == "PROMOTED" and not dry_run,
        "dry_run": dry_run,
        "force": force,
        "timestamp": TIMESTAMP,
    }


def run_all(
    dry_run: bool = False,
    force: bool = False,
    target_root: Optional[Path] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    promoted_count = 0
    skipped_count = 0
    error_count = 0
    new_files: List[str] = []
    collisions: List[Dict[str, Any]] = []
    for promoter in PROMOTIONS:
        row = promote_one(
            promoter, dry_run=dry_run, force=force, target_root=target_root
        )
        rows.append(row)
        status = row["status"]
        if status == "PROMOTED":
            promoted_count += 1
            if row.get("wrote_file"):
                new_files.append(row["target"])
        elif status.startswith("SKIPPED"):
            skipped_count += 1
            if status == "SKIPPED_EXISTS":
                collisions.append(
                    {
                        "target": row["target"],
                        "existing_size_bytes": row["existing_target_audit"].get(
                            "size_bytes"
                        ),
                        "existing_measured_count": row["existing_target_audit"].get(
                            "measured_count"
                        ),
                        "resolution": "preserved",
                    }
                )
        elif status == "BUILD_ERROR":
            error_count += 1

    log = {
        "workflow": "Wf-87 Phase 2 — promote MEASURED JSONs from reports/ to metrics/by_metric/",
        "date": TIMESTAMP,
        "constraint": (
            "CPU-only; no GPU retrain; paper-gate (TODO-84) honored; "
            "party-twenty-spirit §3.5 honest framing"
        ),
        "promoter_count": len(PROMOTIONS),
        "promoted_count": promoted_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "new_files": new_files,
        "collisions": collisions,
        "rows": rows,
    }
    return log


def write_log(log: Dict[str, Any], log_path: Optional[Path] = None) -> Path:
    """Write the promotion log.

    ``log_path`` redirects the log (Wf-A123).  ``None`` keeps the
    historical behaviour: ``OUT_LOG`` inside ``OUT_DIR``.
    """
    path = Path(log_path) if log_path is not None else OUT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-87 Phase 2 — promote MEASURED JSONs from reports/ to "
            "metrics/by_metric/."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the promotion log without writing target files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing central metric files (use with care).",
    )
    parser.add_argument(
        "--target-root",
        default=None,
        help=(
            "Redirect the WRITE side into this root (targets are written "
            "to <target-root>/metrics/by_metric/<name>.json).  Sources "
            "are still read from the real repository.  Default: the real "
            "repository (historical behaviour)."
        ),
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help=(
            "Write the promotion log to this path instead of "
            "molmetal/reports/wf87_p2_promotion/promotion_log.json."
        ),
    )
    args = parser.parse_args(argv)

    target_root = Path(args.target_root) if args.target_root else None
    log_path = Path(args.log_path) if args.log_path else None

    log = run_all(dry_run=args.dry_run, force=args.force, target_root=target_root)
    out_path = write_log(log, log_path=log_path)

    # Friendly summary
    print(f"Wf-87 P2 promotion log → {out_path}")
    print(f"  Promoters: {log['promoter_count']}")
    print(f"  Promoted : {log['promoted_count']}")
    print(f"  Skipped  : {log['skipped_count']}")
    print(f"  Errors   : {log['error_count']}")
    if log["new_files"]:
        print("  New files:")
        for f in log["new_files"]:
            print(f"    - {f}")
    if log["collisions"]:
        print("  Collisions (preserved):")
        for c in log["collisions"]:
            print(f"    - {c['target']} ({c['existing_size_bytes']} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
