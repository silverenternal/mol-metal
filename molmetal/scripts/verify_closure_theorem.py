"""Wf-36 Phase 1 — Closure theorem end-to-end verifier (L1).

================================================================
Wf-36 Phase 1: per-metal closure reachability benchmark.

This is the **end-to-end** closure-theorem verifier, building on:

* :mod:`molmetal_lam.lam_chem.closure` (R17 ship) — the closure-theorem
  BFS + ``closure_theorem()`` assertion hook.
* :mod:`molmetal_lam.atoms.combinators` (R17 ship, 42-entry
  ``PRIMITIVE_ATOMS``) — the 30-transition-metal atom library.
* :mod:`molmetal.scripts.closure_real_benchmark` (Wf-17 P4.2 ship) —
  PlatinAI 5K benchmark framework that this script extends.
* :mod:`molmetal.scripts.wf_lambda4_verify` (WF-Lambda-4 ship) —
  cisplatin-class closure scenarios.
* ``molmetal/data/tmqm_dblock_subset.csv`` (Wf-28 ship, 5K samples
  spanning 30 d-block metals).
* ``molmetal/data/platinai_5k.csv`` (R17 ship, 5K real Pt-group
  anticancer samples).

What this script measures
-------------------------
For each metal symbol (default: the 9 task-spec metals
``Pt/Ru/Ir/Rh/Re/Os/Au/Pd/Ag``), the script:

  1. Loads up to ``--n-samples`` SMILES from the chosen dataset,
     filtered to ``metal_center == metal`` (tmQM) or ``metal == metal``
     (PlatinAI).
  2. Parses each SMILES into a :class:`MoleculeClosedTerm`.
  3. Runs :func:`molmetal_lam.lam_chem.closure.closure_theorem` with
     the canonical 5 click rules at ``--max-depth`` against the
     default 7-partner-tile seed set.
  4. Records per-sample closure-ok, n_visited, n_products, β-depth,
     wall-time, peak memory, and Pt-coordination flag.

The per-metal aggregate is the **closure reachability rate** — the
fraction of metal-tagged samples that pass the closure-theorem
assertion.  Together with per-metal wall-time + memory, this fills the
**L1 gap** in ``TODO/pending/83_lambda_algorithm_enhancement.md``.

Outputs
-------
* ``molmetal/reports/wf36_closure_theorem/closure_bench.json`` — full
  per-sample + per-metal aggregate benchmark.
* ``molmetal/reports/wf36_closure_theorem/per_metal.csv`` — one row
  per metal (spreadsheet audit).
* ``molmetal/reports/wf36_closure_theorem/closure_bench.md`` — human-
  readable summary.

Honest framing
--------------
* **CPU-only.** No CUDA / Triton / CFM retraining is exercised.
* **Deterministic.** No RNG; rerunning twice on the same input
  yields the same closure verdict (closure_theorem is pure-functional
  modulo dict-iteration order).
* **Real data.** Both CSV inputs are real Wf-28 / R17 ship datasets.
  Missing CSV ⇒ graceful empty-aggregate (no synthetic fallback).
* **Hardness caveat.**  tmQM d-block complexes with dative
  ``[M+]<-`` bonds sometimes trigger valence warnings from RDKit
  during the click-rule reduce path.  These are caught by
  ``closure_theorem``'s valence-BNF predicate and count as
  ``closure_ok=False`` (not as errors).  We report both rates
  transparently.

Usage
-----
::

    # Default: 9 task-spec metals × 100 samples each = ~900 total.
    uv run --no-sync python molmetal/scripts/verify_closure_theorem.py

    # Smoke (10 samples per metal, ~90 total).
    uv run --no-sync python molmetal/scripts/verify_closure_theorem.py --n-samples 10

    # PlatinAI dataset only, all 7 PlatinAI metals.
    uv run --no-sync python molmetal/scripts/verify_closure_theorem.py --dataset platinai
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
import tracemalloc
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Repo-root path injection
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


log = logging.getLogger("wf36_closure_theorem")


# ---------------------------------------------------------------------------
# Defaults — task-spec metal set + dataset paths
# ---------------------------------------------------------------------------
#: The 9 task-spec metals (TODO-83 L1): the platinum-group + iridium +
#: rhenium + osmium + gold + palladium + silver set.  All are present
#: in BOTH the tmQM d-block subset AND the PlatinAI 5K CSV.
TASK_SPEC_METALS: Tuple[str, ...] = (
    "Pt", "Ru", "Ir", "Rh", "Re", "Os", "Au", "Pd", "Ag",
)

#: The 30 d-block metals shipped in ``PRIMITIVE_ATOMS`` (R17 UC-B1).
TMQM_DBLOCK_METALS: Tuple[str, ...] = (
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",   # 3d
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",   # 4d
    "La", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",   # 5d + La
)

#: PlatinAI 5K metal labels (subset of d-block).
PLATINAI_METALS: Tuple[str, ...] = ("Pt", "Ru", "Ir", "Rh", "Os", "Au", "Pd")

DEFAULT_DATASET: str = "tmqm"
DEFAULT_N_SAMPLES_PER_METAL: int = 100
DEFAULT_MAX_DEPTH: int = 1
DEFAULT_MAX_PRODUCTS: int = 64
DEFAULT_OUTPUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf36_closure_theorem"
DEFAULT_TMQM_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"

#: Partner terms (mirrors wf_lambda4_verify fixture).
PARTNER_SMILE_LIST: List[str] = [
    "C#CCN",                  # propargylamine  — alkyne + amine
    "CCN=[N+]=[N-]",          # ethyl azide     — CuAAC
    "C#CC",                   # propyne         — CuAAC
    "CS",                     # methanethiol    — ThiolEne
    "C=CC",                   # propene         — ThiolEne
    "OC(=O)c1ccccc1",         # benzoic acid    — AmideCoupling
    "NCC",                    # ethylamine      — AmideCoupling
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _utcnow_iso() -> str:
    """Return the current UTC time in ISO 8601 format with ``Z`` suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _memory_mb(snapshot_diff: int) -> float:
    """Convert a ``tracemalloc`` byte diff to MB."""
    return float(snapshot_diff) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Lazy heavy imports — loaded on demand inside benchmark functions
# ---------------------------------------------------------------------------
def _load_click_rules() -> List[Any]:
    """Return the 5 canonical click rules in the public-facing order."""
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
    )
    return [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]


def _load_partner_terms() -> List[Any]:
    """Build partner terms from :data:`PARTNER_SMILE_LIST`."""
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    out: List[Any] = []
    for smi in PARTNER_SMILE_LIST:
        try:
            out.append(MoleculeClosedTerm.from_smiles(smi, embed_3d=False))
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Failed to parse partner %r: %s", smi, exc)
    return out


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def _load_metal_samples(
    csv_path: Path,
    metal: str,
    n: int,
    *,
    dataset: str,
) -> List[str]:
    """Read up to ``n`` SMILES tagged with the given metal symbol.

    The CSV format differs slightly between tmQM and PlatinAI:

    * tmQM uses the ``metal_center`` column (e.g. ``"Pt"``, ``"Ru"``).
    * PlatinAI uses the ``metal`` column.

    Both are real ship datasets; if the CSV is missing we return ``[]``
    so the benchmark degrades gracefully.  Empty / whitespace SMILES
    are skipped.
    """
    out: List[str] = []
    if not csv_path.is_file():
        log.error("CSV not found at %s", csv_path)
        return out
    column = "metal_center" if dataset == "tmqm" else "metal"
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if column not in fields:
            log.error(
                "Column %r missing from %s; fields=%r",
                column, csv_path, fields,
            )
            return out
        if "smiles" not in fields:
            log.error(
                "Column 'smiles' missing from %s; fields=%r",
                csv_path, fields,
            )
            return out
        for row in reader:
            tag = (row.get(column, "") or "").strip()
            if tag != metal:
                continue
            smi = (row.get("smiles", "") or "").strip()
            if not smi:
                continue
            out.append(smi)
            if len(out) >= int(n):
                break
    return out


# ---------------------------------------------------------------------------
# Per-sample benchmark
# ---------------------------------------------------------------------------
def _is_pt_coordinated(term: Any) -> bool:
    """Return True iff ``term`` contains at least one Pt atom.

    Detection uses RDKit atom inspection (Z=78 for Pt).  Falls back
    to False on any failure so the benchmark never crashes here.
    """
    try:
        mol = term.to_rdkit()
    except Exception:
        return False
    if mol is None:
        return False
    try:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 78:
                return True
    except Exception:
        return False
    return False


def _benchmark_one(
    smiles: str,
    idx: int,
    metal: str,
    *,
    click_rules: Sequence[Any],
    partner_terms: Sequence[Any],
    max_depth: int,
    max_products: int,
) -> Dict[str, Any]:
    """Run closure_theorem on one SMILES; record per-sample metrics.

    Returns a JSON-safe dict.  Status is one of ``ok``, ``skipped``,
    ``error``.  ``closure_ok`` is True iff the closure-theorem
    assertion hook returned True.
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    record: Dict[str, Any] = {
        "idx": int(idx),
        "metal": str(metal),
        "smiles": str(smiles),
        "status": "skipped",
        "closure_ok": False,
        "n_visited": 0,
        "n_products": 0,
        "max_depth_seen": 0,
        "witness_rate": 0.0,
        "well_typed_rate": 0.0,
        "n_pt_coordinated": 0,
        "n_atoms_start": 0,
        "parse_error": "",
        "wall_time_s": 0.0,
        "memory_mb": 0.0,
        "depth": int(max_depth),
    }

    # Parse SMILES into a closed term.
    t_parse_0 = time.perf_counter()
    try:
        start_term = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)
        record["n_atoms_start"] = int(len(start_term.atoms))
    except Exception as exc:
        record["status"] = "error"
        record["parse_error"] = f"{type(exc).__name__}: {exc}"[:240]
        record["wall_time_s"] = float(time.perf_counter() - t_parse_0)
        return record

    # Run the BFS + closure_theorem assertion.
    tracemalloc.start()
    base_mem = tracemalloc.get_traced_memory()[0]
    t_bfs_0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ok = closure_theorem(
                start_term=start_term,
                click_rules=list(click_rules),
                max_depth=int(max_depth),
                max_products=int(max_products),
                partner_terms=list(partner_terms) if partner_terms else None,
            )
        record["closure_ok"] = bool(ok)
        record["status"] = "ok"
        last_result = getattr(closure_theorem, "last_result", {}) or {}
        record["n_visited"] = int(last_result.get("n_visited", 0))
        record["n_products"] = int(last_result.get("n_products", 0))
        record["max_depth_seen"] = int(last_result.get("max_depth_seen", 0))
        record["witness_rate"] = float(last_result.get("witness_rate", 1.0))
        record["well_typed_rate"] = float(last_result.get("well_typed_rate", 1.0))
    except Exception as exc:
        record["status"] = "error"
        record["parse_error"] = f"{type(exc).__name__}: {exc}"[:240]
    finally:
        wall_s = time.perf_counter() - t_bfs_0
        cur_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record["wall_time_s"] = float(wall_s)
        record["memory_mb"] = float(_memory_mb(peak_mem - base_mem))

    # Pt-coordinated detection — only on successful parses; cheap.
    if record["status"] == "ok":
        try:
            if _is_pt_coordinated(start_term):
                record["n_pt_coordinated"] = 1
        except Exception:
            pass

    return record


# ---------------------------------------------------------------------------
# Per-metal aggregate
# ---------------------------------------------------------------------------
def _aggregate_per_metal(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-sample records into a per-metal benchmark dict.

    Keys returned (all floats; counts are int-cast for JSON safety):

    * ``n_samples``             — number of records
    * ``n_closure_ok``          — number of records with closure_ok=True
    * ``closure_reachability_rate``  — n_closure_ok / n_samples
    * ``mean_beta_depth``       — mean of max_depth_seen over ok records
    * ``mean_n_products``       — mean of n_products over ok records
    * ``mean_wall_time``        — mean of wall_time_s over ok records
    * ``std_wall_time``         — sample stdev of wall_time_s
    * ``median_wall_time``      — median of wall_time_s
    * ``peak_memory_mb``        — max of memory_mb over ok records
    * ``total_wall_time``       — sum of wall_time_s over ok records
    * ``n_error``               — number of records with status="error"
    """
    n = len(records)
    if n == 0:
        return {
            "n_samples": 0,
            "n_closure_ok": 0,
            "closure_reachability_rate": 0.0,
            "mean_beta_depth": 0.0,
            "mean_n_products": 0.0,
            "mean_wall_time": 0.0,
            "std_wall_time": 0.0,
            "median_wall_time": 0.0,
            "peak_memory_mb": 0.0,
            "total_wall_time": 0.0,
            "n_error": 0,
        }

    n_ok = sum(1 for r in records if r["closure_ok"])
    n_err = sum(1 for r in records if r["status"] == "error")
    rate = (n_ok / n) if n else 0.0

    ok_records = [r for r in records if r["status"] == "ok"]
    if ok_records:
        depths = [float(r["max_depth_seen"]) for r in ok_records]
        prods = [float(r["n_products"]) for r in ok_records]
        walls = [float(r["wall_time_s"]) for r in ok_records]
        mems = [float(r["memory_mb"]) for r in ok_records]
        mean_beta_depth = float(statistics.fmean(depths))
        mean_n_products = float(statistics.fmean(prods))
        mean_wall = float(statistics.fmean(walls))
        std_wall = (
            float(statistics.stdev(walls)) if len(walls) >= 2 else 0.0
        )
        median_wall = float(statistics.median(walls))
        peak_mem = float(max(mems)) if mems else 0.0
        total_wall = float(sum(walls))
    else:
        mean_beta_depth = 0.0
        mean_n_products = 0.0
        mean_wall = 0.0
        std_wall = 0.0
        median_wall = 0.0
        peak_mem = 0.0
        total_wall = 0.0

    return {
        "n_samples": int(n),
        "n_closure_ok": int(n_ok),
        "closure_reachability_rate": float(rate),
        "mean_beta_depth": mean_beta_depth,
        "mean_n_products": mean_n_products,
        "mean_wall_time": mean_wall,
        "std_wall_time": std_wall,
        "median_wall_time": median_wall,
        "peak_memory_mb": peak_mem,
        "total_wall_time": total_wall,
        "n_error": int(n_err),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_closure_theorem_verifier(
    dataset: str = DEFAULT_DATASET,
    n_samples_per_metal: int = DEFAULT_N_SAMPLES_PER_METAL,
    *,
    metals: Sequence[str] = TASK_SPEC_METALS,
    tmqm_csv: Path = DEFAULT_TMQM_CSV,
    platinai_csv: Path = DEFAULT_PLATINAI_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_products: int = DEFAULT_MAX_PRODUCTS,
    seeds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the closure-theorem verifier across ``metals`` and return the JSON payload.

    The returned dict is also persisted to ``{output_dir}/closure_bench.json``
    by :func:`save_verifier_benchmark`.
    """
    log.info(
        "[wf36_closure_theorem] start: dataset=%s, metals=%s, n_per_metal=%d, "
        "max_depth=%d",
        dataset, list(metals), int(n_samples_per_metal), int(max_depth),
    )

    if dataset == "tmqm":
        csv_path = Path(tmqm_csv)
    else:
        csv_path = Path(platinai_csv)

    if not csv_path.is_file():
        log.error("CSV missing at %s; aborting benchmark.", csv_path)
        return {
            "schema_version": 1,
            "tool": "verify_closure_theorem.py",
            "pilot_id": "wf36_closure_theorem",
            "date_utc": _utcnow_iso(),
            "dataset": dataset,
            "csv_path": str(csv_path),
            "metals": list(metals),
            "n_samples_per_metal": int(n_samples_per_metal),
            "max_depth": int(max_depth),
            "error": f"CSV not found at {csv_path}",
            "per_metal": {},
            "per_sample": [],
            "aggregate": {},
        }

    click_rules = _load_click_rules()
    partner_terms = _load_partner_terms()
    click_rule_names = [getattr(r, "name", type(r).__name__) for r in click_rules]

    t_total_0 = time.perf_counter()
    per_metal_records: Dict[str, List[Dict[str, Any]]] = {}
    per_metal_aggregate: Dict[str, Dict[str, float]] = {}
    all_records: List[Dict[str, Any]] = []

    for metal in metals:
        smis = _load_metal_samples(
            csv_path, str(metal), int(n_samples_per_metal), dataset=dataset,
        )
        records: List[Dict[str, Any]] = []
        for i, smi in enumerate(smis):
            rec = _benchmark_one(
                smi, i, str(metal),
                click_rules=click_rules,
                partner_terms=partner_terms,
                max_depth=int(max_depth),
                max_products=int(max_products),
            )
            records.append(rec)
            all_records.append(rec)
        per_metal_records[metal] = records
        per_metal_aggregate[metal] = _aggregate_per_metal(records)
        agg = per_metal_aggregate[metal]
        log.info(
            "[wf36_closure_theorem] %s: n=%d closure_ok=%d rate=%.4f wall=%.3fs",
            metal, agg["n_samples"], agg["n_closure_ok"],
            agg["closure_reachability_rate"], agg["total_wall_time"],
        )

    # Overall aggregate (across all metals).
    overall = _aggregate_per_metal(all_records)
    total_wall_s = time.perf_counter() - t_total_0

    # Per-metal status counts.
    status_counts: Dict[str, Dict[str, int]] = {}
    for metal, recs in per_metal_records.items():
        sc = {"ok": 0, "skipped": 0, "error": 0}
        for r in recs:
            sc[r["status"]] = sc.get(r["status"], 0) + 1
        status_counts[metal] = sc

    try:
        import torch  # local; cheap
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "n/a"

    py_ver = ".".join(str(v) for v in sys.version_info[:3])

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "verify_closure_theorem.py",
        "pilot_id": "wf36_closure_theorem",
        "date_utc": _utcnow_iso(),
        "dataset": str(dataset),
        "csv_path": str(csv_path),
        "metals": list(metals),
        "n_samples_per_metal": int(n_samples_per_metal),
        "n_samples_total": int(sum(
            per_metal_aggregate[m]["n_samples"] for m in metals
        )),
        "max_depth": int(max_depth),
        "max_products": int(max_products),
        "seeds": int(seeds) if seeds is not None else 1,
        "click_rule_names": list(click_rule_names),
        "n_partners": int(len(partner_terms)),
        "partner_smiles": list(PARTNER_SMILE_LIST),
        "per_metal": {
            metal: {
                "aggregate": dict(per_metal_aggregate[metal]),
                "status_counts": dict(status_counts.get(metal, {})),
                "records": list(per_metal_records[metal]),
            }
            for metal in metals
        },
        "per_sample": list(all_records),
        "aggregate": dict(overall),
        "total_wall_time_s": float(total_wall_s),
        "timestamp_utc": _utcnow_iso(),
        "python": py_ver,
        "torch": torch_ver,
        "honest_framing": (
            "CPU-only closure-theorem end-to-end verifier. "
            "Real data from tmQM d-block subset or PlatinAI 5K CSV. "
            "Deterministic; runs the actual ProductiveSpace BFS + "
            "closure_theorem assertion hook on each metal-tagged sample."
        ),
    }
    return payload


def save_verifier_benchmark(
    payload: Dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path]:
    """Persist the verifier benchmark payload + mirrors to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "closure_bench.json"

    def _coerce(o: Any) -> Any:
        try:
            import numpy as _np
            if isinstance(o, (_np.floating, _np.integer)):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        return str(o)

    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=_coerce)

    csv_path = output_dir / "per_metal.csv"
    with csv_path.open("w", newline="") as fh:
        fields = [
            "metal", "n_samples", "n_closure_ok", "closure_reachability_rate",
            "mean_beta_depth", "mean_n_products", "mean_wall_time",
            "std_wall_time", "median_wall_time", "peak_memory_mb",
            "total_wall_time", "n_error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for metal in payload.get("metals", []):
            agg = payload["per_metal"][metal]["aggregate"]
            row = {f: agg.get(f, "") for f in fields}
            row["metal"] = metal
            writer.writerow(row)

    md_path = output_dir / "closure_bench.md"
    _write_markdown(payload, output_dir)

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
    }


def _write_markdown(payload: Dict[str, Any], output_dir: Path) -> None:
    """Write the human-readable ``closure_bench.md`` summary."""
    md_path = output_dir / "closure_bench.md"
    metals = payload.get("metals", [])
    agg = payload.get("aggregate", {}) or {}
    overall_rate = float(agg.get("closure_reachability_rate", 0.0))
    overall_n = int(agg.get("n_samples", 0))
    overall_ok = int(agg.get("n_closure_ok", 0))
    total_wall = float(payload.get("total_wall_time_s", 0.0))

    lines: List[str] = []
    lines.append("# Wf-36 Phase 1 — Closure theorem end-to-end verifier (L1)")
    lines.append("")
    lines.append(
        "Per-metal closure reachability benchmark on the tmQM d-block\n"
        "subset + PlatinAI 5K datasets.  Fills the **L1 gap** in\n"
        ":mod:`TODO/pending/83_lambda_algorithm_enhancement.md`.\n"
    )
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"**Date (UTC)**: {payload['timestamp_utc']}  ")
    lines.append(f"**Tool**: `{payload['tool']}`  ")
    lines.append(f"**Dataset**: `{payload['dataset']}` ({payload['csv_path']})  ")
    lines.append(f"**N metals**: {len(metals)}  ")
    lines.append(f"**N samples per metal**: {payload['n_samples_per_metal']}  ")
    lines.append(f"**N samples total**: {overall_n}  ")
    lines.append(f"**Max depth**: {payload['max_depth']}  ")
    lines.append(f"**Click rules**: {', '.join(payload['click_rule_names'])}  ")
    lines.append(f"**Partner terms**: {payload['n_partners']}  ")
    lines.append(f"**Total wall**: {total_wall:.2f} s  ")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| N samples | {overall_n} |")
    lines.append(f"| N closure_ok | {overall_ok} |")
    lines.append(f"| closure_reachability_rate | {overall_rate:.4f} |")
    lines.append(f"| mean_beta_depth | {float(agg.get('mean_beta_depth', 0.0)):.4f} |")
    lines.append(f"| mean_n_products | {float(agg.get('mean_n_products', 0.0)):.4f} |")
    lines.append(f"| mean_wall_time (s) | {float(agg.get('mean_wall_time', 0.0)):.6f} |")
    lines.append(f"| std_wall_time (s) | {float(agg.get('std_wall_time', 0.0)):.6f} |")
    lines.append(f"| peak_memory_mb | {float(agg.get('peak_memory_mb', 0.0)):.4f} |")
    lines.append(f"| N error | {int(agg.get('n_error', 0))} |")
    lines.append("")
    lines.append("## Per-metal reachability rate")
    lines.append("")
    lines.append("| Metal | N | closure_ok | reachability_rate | mean_wall_s |")
    lines.append("|-------|---|------------|-------------------|-------------|")
    for metal in metals:
        m = payload["per_metal"][metal]["aggregate"]
        lines.append(
            f"| {metal} | {int(m['n_samples'])} | {int(m['n_closure_ok'])} "
            f"| {float(m['closure_reachability_rate']):.4f} "
            f"| {float(m['mean_wall_time']):.6f} |"
        )
    lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **CPU-only.** No CUDA / Triton / CFM retraining is exercised.\n"
        "- **Deterministic.** closure_theorem is pure-functional.\n"
        "- **Real data.** Both CSV inputs are real ship datasets.\n"
        "- **Valence-BNF rejection.** Many tmQM samples fail the\n"
        "  valence-BNF check on the start term (dative-bond organometallics\n"
        "  with ``[M+]<-`` notation that the MLC ledger treats as\n"
        "  under-saturated).  These count as ``closure_ok=False`` but\n"
        "  ``status=ok`` — the BFS runs, the assertion rejects the seed.\n"
    )
    lines.append("")

    md_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_closure_theorem.py",
        description=(
            "Wf-36 Phase 1 closure-theorem end-to-end verifier. "
            "Runs the real ProductiveSpace BFS + closure_theorem "
            "assertion hook on per-metal-tagged SMILES from the tmQM "
            "d-block subset or the PlatinAI 5K CSV."
        ),
    )
    p.add_argument(
        "--dataset",
        choices=("tmqm", "platinai"),
        default=DEFAULT_DATASET,
        help="Dataset to load (default tmqm).",
    )
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES_PER_METAL,
        help="Samples per metal (default 100).",
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(TASK_SPEC_METALS),
        help=(
            "Metal symbols to benchmark. Default: the 9 task-spec "
            f"metals {list(TASK_SPEC_METALS)}. Pass --metals all_dblock "
            "to use all 30 metals from PRIMITIVE_ATOMS, or supply "
            "an explicit list like --metals Pt Ru Pd."
        ),
    )
    p.add_argument(
        "--tmqm-csv", type=Path, default=DEFAULT_TMQM_CSV,
        help="Path to tmQM d-block subset CSV.",
    )
    p.add_argument(
        "--platinai-csv", type=Path, default=DEFAULT_PLATINAI_CSV,
        help="Path to PlatinAI 5K CSV.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for closure_bench.json + per_metal.csv + .md.",
    )
    p.add_argument(
        "--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
        help="Closure BFS max depth (default 1).",
    )
    p.add_argument(
        "--max-products", type=int, default=DEFAULT_MAX_PRODUCTS,
        help="ProductiveSpace BFS safety cap (default 64).",
    )
    p.add_argument(
        "--seeds", type=int, default=1,
        help="Number of independent seeds (default 1; placeholder for future).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode (caller controls --n-samples, e.g. 10).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    metals_arg = list(args.metals)
    # Allow "all_dblock" / "all" alias for the 30-metal set.
    if len(metals_arg) == 1 and metals_arg[0].lower() in ("all_dblock", "all", "30"):
        metals_arg = list(TMQM_DBLOCK_METALS)
    # Filter to known set.
    if args.dataset == "tmqm":
        known = set(TMQM_DBLOCK_METALS)
    else:
        known = set(PLATINAI_METALS)
    metals = [m for m in metals_arg if m in known]
    if not metals:
        log.error(
            "No known metals in --metals %s; valid for dataset=%s: %s",
            metals_arg, args.dataset, sorted(known),
        )
        return 2

    payload = run_closure_theorem_verifier(
        dataset=str(args.dataset),
        n_samples_per_metal=int(args.n_samples),
        metals=metals,
        tmqm_csv=Path(args.tmqm_csv),
        platinai_csv=Path(args.platinai_csv),
        output_dir=Path(args.output_dir),
        max_depth=int(args.max_depth),
        max_products=int(args.max_products),
        seeds=int(args.seeds),
    )

    if "error" in payload:
        log.error("Benchmark aborted: %s", payload["error"])
        return 1

    paths = save_verifier_benchmark(payload, output_dir=Path(args.output_dir))
    agg = payload["aggregate"]
    log.info(
        "[wf36_closure_theorem] DONE: dataset=%s metals=%d n=%d "
        "closure_ok=%d rate=%.4f mean_wall=%.6fs peak_mem=%.2fMB "
        "total_wall=%.2fs",
        payload["dataset"], len(metals), agg["n_samples"],
        agg["n_closure_ok"], agg["closure_reachability_rate"],
        agg["mean_wall_time"], agg["peak_memory_mb"],
        payload["total_wall_time_s"],
    )
    print(
        f"[wf36_closure_theorem] dataset={payload['dataset']} "
        f"metals={len(metals)} n_samples={agg['n_samples']} "
        f"closure_ok={agg['n_closure_ok']} "
        f"reachability_rate={agg['closure_reachability_rate']:.4f} "
        f"mean_wall={agg['mean_wall_time']:.6f}s "
        f"peak_mem={agg['peak_memory_mb']:.2f}MB"
    )
    for name, p in paths.items():
        print(f"[wf36_closure_theorem] wrote {name}: {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())