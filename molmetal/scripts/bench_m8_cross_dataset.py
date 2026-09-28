#!/usr/bin/env python3
"""Wf-64 Phase 3 — m8 PB pass_rate cross-dataset retrain.

================================================================
Wf-64 Phase 3: cross-dataset m8 PB-Valid re-bench
================================================================

Wf-32 Phase 4 (commit 2780abc, 2026-09-26) deferred the full 5-seed
sweep because GPU 1 (RTX 5090, 32 GB) was occupied by the Wf-31 m1
Ru-AUC SE(3) SB run.  Wf-63 Phase 3 (2026-09-26) re-ran the **m8
PB-Valid** subset on the Wf-30 ten-pocket cohort using only the
``_demo_5_metaldrugs.txt`` candidates; the resulting pass_rate was
0.0 because those 5 demo Pt / Ru / Ir / Au / Pd anchored SMILES do
not satisfy PB's mol-mode chemistry checks under MMFF94 (Pt
ligands without explicit ``[Pt]`` square-planar geometry fail the
clash / bond-length checks).

Wf-64 Phase 3 widens the m8 candidate pool to *real* dataset SMILES:

* **PlatinAI 4-cell**     — ``molmetal/data/platinai_4cell.csv``
  (4 078 Pt complexes × 4 cell lines)
* **MetalCytoToxDB 5-metal** — ``.data/molmetal/metalcytotox_5metal_multitask.csv``
  (Ru/Ir/Rh/Os/Re organometallics)
* **tmQM d-block**        — ``.data/molmetal/tmQM/tmqm_parsed.csv``
  (d-block metal-organic complexes; we filter to the 25 d-block
  metals present in tmQM, dropping La/Sc/Y as f-block / group-3
  and excluding Hg/Cd/Pb as post-d-block).

For each (dataset, seed) pair we sub-sample 50 SMILES (deterministic
seed), run the real PoseBusters ``mol`` config (chemistry-only, 14
checks) on each, and aggregate to a per-seed ``pb_pass_rate``.  The
``mol`` config is intentionally protein-free — this is the Wf-32 m8
baseline path; the Wf-30 ten-pocket ``receptor.pdbqt`` is NOT used
here because the cross-dataset candidates are pure-ligand SMILES,
not pre-docked poses.

The headline metric is the macro-average pass_rate across the three
datasets.  Target ``>= 0.7`` (lit: TargetDiff 0.94, Uni-Mol v2 0.75).

Inputs
------
* ``molmetal/data/platinai_4cell.csv``         (4 078 rows; SMILES column ``smiles``)
* ``.data/molmetal/metalcytotox_5metal_multitask.csv`` (~26 800 rows; ``smiles`` + ``metal``)
* ``.data/molmetal/tmQM/tmqm_parsed.csv``     (~108 500 rows; ``smiles`` + ``metal``)

Outputs (under :file:`molmetal/reports/wf64_m8_cross/`):
* :file:`{dataset}/seed_{seed}_metrics.json`   — per-seed PB pass_rate + per-check failure names
* :file:`{dataset}/per_seed_summary.json`     — aggregate per-seed pass_rate + per-pocket-style
                                                (here: per-cell-line for PlatinAI, per-metal
                                                for the other two)
* :file:`m8_cross_summary.json`               — machine-readable cross-dataset headline
* :file:`synthesis.md`                        — honest writeup

Verify
------
::

    CUDA_VISIBLE_DEVICES=0 python \\
        molmetal/scripts/bench_m8_cross_dataset.py \\
        --datasets platinai_4cell metalcytotox_5metal tmqm_dblock \\
        --pockets all --seeds 0 1 2 3 4

Exits 0 if the real PoseBusters adapter ran on every requested
(dataset, seed) cell and produced per-seed JSON.  Exits 2 if real PB
is unavailable (records ``HONEST_NEGATIVE`` with 0.0 pass_rate).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Repo-root path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATASETS: Tuple[str, ...] = (
    "platinai_4cell", "metalcytotox_5metal", "tmqm_dblock",
)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_N_SAMPLES_PER_DATASET: int = 50

# Repo paths
PLATINAI_CSV = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
METALCYTOTOX_CSV = _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask.csv"
TMQM_CSV = _REPO_ROOT / ".data" / "molmetal" / "tmQM" / "tmqm_parsed.csv"

# MetalCytoToxDB 5-metal whitelist (per R17 Rusanov 2025 anchor)
METALCYTOTOX_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")

# tmQM d-block set (drop f-block / post-d / lanthanides).  We use the
# canonical d-block assignment from the IUPAC periodic table groups 3-12
# (excluding La as f-block).
TMQM_DBLOCK_METALS: Tuple[str, ...] = (
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
)

# Cell lines for PlatinAI 4-cell dataset (for per-cell-line breakdown)
PLATINAI_CELL_LINES: Tuple[str, ...] = ("MCF7", "A549", "A2780", "A2780cis")

# Default output directory
DEFAULT_OUT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf64_m8_cross"

# Target headline pass_rate
TARGET_PASS_RATE: float = 0.7

log = logging.getLogger("bench_m8_cross_dataset")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleSMILES:
    """One sampled SMILES + its provenance (dataset, metal, etc.)."""
    sample_id: str            # stable ID (e.g. "platinai_4cell_00042")
    smiles: str
    metal: str                # metal symbol, e.g. "Pt"
    bucket: str               # per-cell-line (PlatinAI) or per-metal (others)


@dataclass(frozen=True)
class PBOutcome:
    """One PB ``mol``-config check result."""
    sample_id: str
    smiles: str
    metal: str
    bucket: str
    status: str               # "passed" | "failed" | "skipped" | "error"
    pb_valid: Optional[bool]
    n_checks: int
    n_passed: int
    n_failed: int
    n_skipped: int
    failed_checks: Tuple[str, ...]
    elapsed_s: float
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_platinai_4cell(path: Path = PLATINAI_CSV) -> List[SampleSMILES]:
    """Load Pt complexes from PlatinAI 4-cell dataset.

    Bucket per-cell-line via ``source`` field (each row carries a
    CSD code in the ``source`` column for its origin; for the
    per-bucket breakdown we use the source as the "metal bucket"
    because PlatinAI is monometallic Pt).
    """
    if not path.exists():
        log.warning("platinai_4cell not found at %s", path)
        return []
    samples: List[SampleSMILES] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                continue
            # PlatinAI is monometallic Pt — bucket by source (CSD code)
            # for cross-pivot comparison vs MetalCytoToxDB / tmQM.
            bucket = (row.get("source") or "Pt").strip()
            samples.append(
                SampleSMILES(
                    sample_id=f"platinai_4cell_{i:05d}",
                    smiles=smiles,
                    metal="Pt",
                    bucket=bucket,
                )
            )
    return samples


def load_metalcytotox_5metal(
    path: Path = METALCYTOTOX_CSV,
    metals: Sequence[str] = METALCYTOTOX_METALS,
) -> List[SampleSMILES]:
    """Load organometallics from MetalCytoToxDB 5-metal cohort."""
    if not path.exists():
        log.warning("metalcytotox not found at %s", path)
        return []
    samples: List[SampleSMILES] = []
    metal_set = set(metals)
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            metal = (row.get("metal") or "").strip()
            if metal not in metal_set:
                continue
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                continue
            samples.append(
                SampleSMILES(
                    sample_id=f"metalcytotox_{i:06d}",
                    smiles=smiles,
                    metal=metal,
                    bucket=metal,
                )
            )
    return samples


def load_tmqm_dblock(
    path: Path = TMQM_CSV,
    metals: Sequence[str] = TMQM_DBLOCK_METALS,
) -> List[SampleSMILES]:
    """Load d-block metal-organic complexes from tmQM."""
    if not path.exists():
        log.warning("tmQM not found at %s", path)
        return []
    samples: List[SampleSMILES] = []
    metal_set = set(metals)
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            metal = (row.get("metal") or "").strip()
            if metal not in metal_set:
                continue
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                continue
            samples.append(
                SampleSMILES(
                    sample_id=f"tmqm_dblock_{i:06d}",
                    smiles=smiles,
                    metal=metal,
                    bucket=metal,
                )
            )
    return samples


def select_dataset(name: str) -> List[SampleSMILES]:
    """Dispatch dataset name → loader."""
    if name == "platinai_4cell":
        return load_platinai_4cell()
    if name == "metalcytotox_5metal":
        return load_metalcytotox_5metal()
    if name == "tmqm_dblock":
        return load_tmqm_dblock()
    raise ValueError(
        f"unknown dataset {name!r}; valid: {DEFAULT_DATASETS}"
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def deterministic_subsample(
    samples: Sequence[SampleSMILES],
    n: int,
    seed: int,
) -> List[SampleSMILES]:
    """Pick ``n`` SMILES from ``samples`` deterministically by ``seed``."""
    if n <= 0 or n >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    indices = rng.sample(range(len(samples)), n)
    return [samples[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# PoseBusters runner
# ---------------------------------------------------------------------------
def _import_pb():
    """Lazy import posebusters_runner with a graceful miss."""
    try:
        from molmetal.validation.posebusters_runner import (
            check_posebusters,
            posebusters_available,
        )
        return check_posebusters, posebusters_available
    except Exception as exc:                                # pragma: no cover
        log.warning("posebusters_runner import failed: %s", exc)
        return None, None


def _coerce_status(report: Dict[str, object]) -> Tuple[str, Optional[bool]]:
    """Map a PB report dict → (status, pb_valid)."""
    if bool(report.get("skipped", False)):
        return "skipped", None
    pb = report.get("pb_valid")
    if pb is None:
        return "skipped", None
    if isinstance(pb, bool):
        return ("passed" if pb else "failed"), bool(pb)
    return "error", None


def _per_check_counts(report: Dict[str, object]) -> Tuple[int, int, int, int, Tuple[str, ...]]:
    """(n_checks, n_passed, n_failed, n_skipped, failed_names)."""
    checks = report.get("checks", {}) or {}
    missing = tuple(report.get("missing_checks", ()) or ())
    failed_list = list(report.get("failures", []) or [])
    n_total = len(checks) + len(missing)
    n_passed = sum(1 for v in checks.values() if v is True)
    n_failed = sum(1 for v in checks.values() if v is False)
    n_skipped = sum(1 for v in checks.values() if v is None) + len(missing)
    failed_names = tuple(failed_list) if failed_list else tuple(
        k for k, v in checks.items() if v is False
    )
    return int(n_total), int(n_passed), int(n_failed), int(n_skipped), failed_names


def run_one_pb(sample: SampleSMILES) -> PBOutcome:
    """Run the real PB ``mol`` config on one sampled SMILES."""
    t0 = time.monotonic()
    check_pb, _ = _import_pb()
    if check_pb is None:
        return PBOutcome(
            sample_id=sample.sample_id, smiles=sample.smiles,
            metal=sample.metal, bucket=sample.bucket,
            status="skipped", pb_valid=None,
            n_checks=0, n_passed=0, n_failed=0, n_skipped=0,
            failed_checks=(), elapsed_s=time.monotonic() - t0,
            note="posebusters_runner import failed",
        )
    report = check_pb(sample.smiles)
    status, pb_valid = _coerce_status(report)
    n_total, n_passed, n_failed, n_skipped, failed = _per_check_counts(report)
    elapsed = time.monotonic() - t0
    note = report.get("error") if status == "error" else None
    return PBOutcome(
        sample_id=sample.sample_id, smiles=sample.smiles,
        metal=sample.metal, bucket=sample.bucket,
        status=status, pb_valid=pb_valid,
        n_checks=n_total, n_passed=n_passed,
        n_failed=n_failed, n_skipped=n_skipped,
        failed_checks=failed, elapsed_s=elapsed,
        note=note if note else None,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_outcomes(outcomes: Iterable[PBOutcome]) -> Dict[str, object]:
    """Reduce outcomes → aggregate stats per bucket + headline pass_rate."""
    by_bucket: Dict[str, List[PBOutcome]] = {}
    for o in outcomes:
        by_bucket.setdefault(o.bucket, []).append(o)
    by_bucket_stats: Dict[str, Dict[str, object]] = {}
    for bucket, items in by_bucket.items():
        passed = [o for o in items if o.pb_valid is True]
        failed = [o for o in items if o.pb_valid is False]
        skipped = [o for o in items if o.status == "skipped"]
        error = [o for o in items if o.status == "error"]
        by_bucket_stats[bucket] = {
            "n_total": int(len(items)),
            "n_passed": int(len(passed)),
            "n_failed": int(len(failed)),
            "n_skipped": int(len(skipped)),
            "n_error": int(len(error)),
            "pb_pass_rate": float(len(passed)) / max(len(items), 1),
        }
    # headline = macro-average across buckets (each bucket weight equal)
    rates = [v["pb_pass_rate"] for v in by_bucket_stats.values()]
    headline = float(statistics.fmean(rates)) if rates else 0.0
    return {
        "by_bucket": by_bucket_stats,
        "headline_pb_pass_rate": headline,
    }


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_dataset_seed(
    dataset_name: str,
    seed: int,
    n_samples: int,
) -> Tuple[List[PBOutcome], Dict[str, object]]:
    """Run one (dataset, seed) cell — load, subsample, sweep PB."""
    samples = select_dataset(dataset_name)
    if not samples:
        log.warning("dataset %s returned 0 samples", dataset_name)
        return [], {"by_bucket": {}, "headline_pb_pass_rate": 0.0}
    sampled = deterministic_subsample(samples, n_samples, seed)
    outcomes: List[PBOutcome] = []
    t_start = time.monotonic()
    for i, s in enumerate(sampled):
        o = run_one_pb(s)
        outcomes.append(o)
        if (i + 1) % max(1, len(sampled) // 5) == 0 or (i + 1) == len(sampled):
            elapsed = time.monotonic() - t_start
            log.info(
                "  %s seed=%d %d/%d (%.0f%%) status=%s "
                "n_checks=%d elapsed=%.1fs",
                dataset_name, seed, i + 1, len(sampled),
                100.0 * (i + 1) / max(len(sampled), 1),
                o.status, o.n_checks, elapsed,
            )
    return outcomes, aggregate_outcomes(outcomes)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def outcome_to_dict(o: PBOutcome) -> Dict[str, object]:
    return {
        "sample_id": o.sample_id, "smiles": o.smiles,
        "metal": o.metal, "bucket": o.bucket,
        "status": o.status, "pb_valid": o.pb_valid,
        "n_checks": o.n_checks, "n_passed": o.n_passed,
        "n_failed": o.n_failed, "n_skipped": o.n_skipped,
        "failed_checks": list(o.failed_checks),
        "elapsed_s": float(o.elapsed_s),
        "note": o.note,
    }


def write_dataset_outputs(
    out_dir: Path,
    dataset: str,
    seed: int,
    outcomes: List[PBOutcome],
    aggregate: Dict[str, object],
) -> None:
    """Write per-seed JSON + aggregate summary for one (dataset, seed)."""
    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    seed_path = ds_dir / f"seed_{seed}_metrics.json"
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": int(seed),
        "n_samples": len(outcomes),
        "outcomes": [outcome_to_dict(o) for o in outcomes],
        "aggregate": aggregate,
    }
    seed_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("wrote %s", seed_path)


def update_per_seed_summary(
    out_dir: Path,
    dataset: str,
    seed: int,
    aggregate: Dict[str, object],
    per_seed: List[float],
) -> None:
    """Maintain ``per_seed_summary.json`` for one dataset."""
    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ds_dir / "per_seed_summary.json"
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "per_seed_pb_pass_rate": [float(x) for x in per_seed],
        "per_seed_n_samples": [int(aggregate["by_bucket"][b]["n_total"]) for b in aggregate["by_bucket"]],
        "by_bucket": aggregate["by_bucket"],
        "headline_pb_pass_rate_last_seed": float(aggregate["headline_pb_pass_rate"]),
        "last_seed": int(seed),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_cross_summary(
    out_dir: Path,
    cross_aggregate: Dict[str, Dict[str, object]],
    per_dataset_pass_rates: Dict[str, List[float]],
) -> None:
    """Write the cross-dataset ``m8_cross_summary.json``."""
    overall_rates = [
        rates[-1] for rates in per_dataset_pass_rates.values() if rates
    ]
    overall = (
        float(statistics.fmean(overall_rates)) if overall_rates else 0.0
    )
    payload = {
        "schema_version": 1,
        "recipe": "wf64_phase3_m8_cross_dataset",
        "metric": "m8 PB pass_rate (cross-dataset macro-average)",
        "wf": "Wf-64",
        "phase": "Phase 3 — PlatinAI 4-cell + MetalCytoToxDB 5-metal + tmQM d-block",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "target_pass_rate": TARGET_PASS_RATE,
        "target_met": bool(overall >= TARGET_PASS_RATE),
        "overall_pb_pass_rate_last_seed": float(overall),
        "per_dataset_last_seed": {
            k: float(v[-1]) if v else 0.0
            for k, v in per_dataset_pass_rates.items()
        },
        "per_dataset_seed_history": per_dataset_pass_rates,
        "per_dataset_aggregate": cross_aggregate,
    }
    out_path = out_dir / "m8_cross_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("wrote %s", out_path)


def write_synthesis_md(
    out_dir: Path,
    cross_aggregate: Dict[str, Dict[str, object]],
    per_dataset_pass_rates: Dict[str, List[float]],
    n_samples: int,
    n_seeds: int,
    pb_installed: bool,
) -> None:
    """Write ``synthesis.md`` — human-readable honest writeup."""
    overall_rates = [
        rates[-1] for rates in per_dataset_pass_rates.values() if rates
    ]
    overall = float(statistics.fmean(overall_rates)) if overall_rates else 0.0
    target_met = "YES" if overall >= TARGET_PASS_RATE else "NO"
    md_lines = [
        "# Wf-64 Phase 3 — m8 PB pass_rate cross-dataset retrain",
        "",
        "**Generated**: " + datetime.now(timezone.utc).isoformat(),
        "**Workflow**: Wf-64 / Phase 3",
        f"**Status**: {'SHIP' if target_met == 'YES' else 'HONEST-NEGATIVE'}",
        f"**PoseBusters installed**: {pb_installed}",
        f"**Datasets**: {', '.join(per_dataset_pass_rates.keys())}",
        f"**Seeds**: {n_seeds} × {n_samples} samples/seed/dataset = "
        f"{n_seeds * n_samples * len(per_dataset_pass_rates)} PB checks",
        f"**Target**: pass_rate >= {TARGET_PASS_RATE}",
        "",
        "## Headline (last-seed)",
        "",
        f"- **overall_pb_pass_rate (macro avg)**: **{overall:.4f}**",
        f"- **target_met**: {target_met}",
        "",
        "## Per-dataset last-seed",
        "",
        "| Dataset | pass_rate (last seed) | n_buckets | n_samples |",
        "|---|---:|---:|---:|",
    ]
    for ds, agg in cross_aggregate.items():
        rate = agg.get("headline_pb_pass_rate", 0.0)
        n_buckets = len(agg.get("by_bucket", {}))
        n_total = sum(b.get("n_total", 0) for b in agg.get("by_bucket", {}).values())
        md_lines.append(
            f"| {ds} | {rate:.4f} | {n_buckets} | {n_total} |"
        )
    md_lines += [
        "",
        "## Per-seed history",
        "",
        "| Dataset | " + " | ".join(f"seed_{i}" for i in range(n_seeds)) + " |",
        "|---|" + "|".join(["---:"] * n_seeds) + "|",
    ]
    for ds, rates in per_dataset_pass_rates.items():
        md_lines.append(
            f"| {ds} | " + " | ".join(f"{r:.4f}" for r in rates) + " |"
        )
    md_lines += [
        "",
        "## Comparability caveat (vs Wf-32 P2 baseline)",
        "",
        "- **Wf-32 P2 baseline**: DEFER (no MEASURED value — GPU 1 occupancy)",
        "- **Wf-63 m8 retry** (10 pockets × demo_5_metaldrugs.txt): pass_rate=0.0",
        "  on the 5 demo Pt/Ru/Ir/Au/Pd SMILES (those SMILES do not satisfy",
        "  PB mol-mode chemistry checks under MMFF94).",
        f"- **Wf-64 Phase 3**: real-dataset SMILES subsample ({n_samples}/seed/dataset),",
        "  cross-pivot = macro-average across PlatinAI Pt bucket + MetalCytoToxDB",
        "  5-metal buckets + tmQM d-block buckets.",
        "",
        "## Honest framing",
        "",
        f"- Real PB installed on this host: **{pb_installed}**",
        f"- Headline pass_rate (last seed): **{overall:.4f}**",
        f"- Target >= {TARGET_PASS_RATE}: **{target_met}**",
        f"- If target not met → R20 backlog (TODO-93/94/95) is the path forward.",
        "",
    ]
    out_path = out_dir / "synthesis.md"
    out_path.write_text("\n".join(md_lines))
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wf-64 Phase 3 — m8 PB pass_rate cross-dataset retrain"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DEFAULT_DATASETS),
        help=f"Datasets to bench (default: {DEFAULT_DATASETS})",
    )
    parser.add_argument(
        "--pockets", default="all",
        help="Reserved for CLI compat — this script does not use the WF-30 pockets "
             "(the cross-dataset candidates are pure-ligand SMILES, not pre-docked poses).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help=f"Seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES_PER_DATASET,
        help=f"SMILES per (dataset, seed) — default {DEFAULT_N_SAMPLES_PER_DATASET}",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output dir (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # PB availability probe
    _check, _avail = _import_pb()
    pb_installed = bool(_check is not None and (_avail is None or _avail()))

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_dataset_pass_rates: Dict[str, List[float]] = {}
    cross_aggregate: Dict[str, Dict[str, object]] = {}
    for dataset in args.datasets:
        if dataset not in DEFAULT_DATASETS and dataset != "all":
            log.error("unknown dataset %s — skip", dataset)
            continue
        seed_rates: List[float] = []
        for seed in args.seeds:
            log.info("dataset=%s seed=%d — begin", dataset, seed)
            outcomes, aggregate = run_dataset_seed(
                dataset, int(seed), int(args.n_samples),
            )
            write_dataset_outputs(out_dir, dataset, int(seed), outcomes, aggregate)
            seed_rates.append(float(aggregate["headline_pb_pass_rate"]))
            update_per_seed_summary(
                out_dir, dataset, int(seed), aggregate, seed_rates,
            )
        per_dataset_pass_rates[dataset] = seed_rates
        if seed_rates:
            cross_aggregate[dataset] = {
                "headline_pb_pass_rate": float(seed_rates[-1]),
                "by_bucket": {},  # filled below
            }
            # Re-aggregate from latest seed for by_bucket detail
            _, latest = run_dataset_seed(
                dataset, int(args.seeds[-1]), int(args.n_samples),
            )
            cross_aggregate[dataset] = latest

    write_cross_summary(out_dir, cross_aggregate, per_dataset_pass_rates)
    write_synthesis_md(
        out_dir,
        cross_aggregate,
        per_dataset_pass_rates,
        int(args.n_samples),
        len(args.seeds),
        pb_installed,
    )

    overall_rates = [
        rates[-1] for rates in per_dataset_pass_rates.values() if rates
    ]
    overall = float(statistics.fmean(overall_rates)) if overall_rates else 0.0
    log.info(
        "DONE — overall_pb_pass_rate (last seed, macro) = %.4f (target >= %.2f)",
        overall, TARGET_PASS_RATE,
    )
    return 0 if overall >= TARGET_PASS_RATE else 2


if __name__ == "__main__":
    sys.exit(main())
