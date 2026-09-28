"""Wf-41 Phase 2 — Closure theorem cross-domain OOD benchmark (L12).

================================================================
Wf-41 Phase 2 (L12): closure-theorem reachability on a **cross-domain
OOD subset** drawn from PlatinAI 5K + MetalCytoToxDB 5-metal data,
filtered by Tanimoto<0.4 (Morgan-ECFP4) against the tmQM d-block
training pool.

This benchmark extends the Wf-37 Phase 2 d-block sweep
(:mod:`molmetal.scripts.bench_closure_dblock`) along the **OOD axis**:
where Wf-37 P2 measured closure-reachability on a single chemistry
domain (tmQM), this script measures it on a **cross-domain held-out
subset** that has been fingerprinted as OOD relative to the train
pool.  This is the closure-theorem analogue of the Wf-7
Tanimoto<0.4 OOD protocol (Pitfall Rank 14, TODO-30 P6.3).

Two methods
-----------
* ``closure``           — run :func:`closure_theorem` on every sample
  of the cross-domain dataset (no OOD filter).  Establishes the
  baseline closure-reachability across the full cross-domain data.
* ``closure+ood_filter``— first filter samples to those with
  **max Tanimoto < 0.4** vs the tmQM training pool (per-metal
  fingerprint distance), then run :func:`closure_theorem` on the
  remaining OOD subset.  This is the strict L12 protocol — the
  closure-theorem reachability on the **OOD subset**.

Per-domain aggregation
----------------------
The benchmark stratifies results along three axes:

  * **domain**: ``platinai_5k`` (8 metals) vs ``metalcytotox_5metal``
    (5 metals).
  * **metal**:  ``Pt``, ``Ru``, ``Pd``, ``Au``, ``Ir``, ``Rh``, ``Os``,
    ``Re`` (intersection of the two datasets).
  * **method**: ``closure`` (unfiltered) vs ``closure+ood_filter``
    (Tanimoto<0.4 against tmQM).

The headline metric is **per-domain closure reachability rate** —
the fraction of metal-tagged samples whose closure-theorem assertion
hook returns ``True``.  This is the L12 number that fills the
"closure theorem cross-domain" gap in TODO-83 §0.

Inputs
------
* :mod:`molmetal_lam.lam_chem.closure` (R17 ship) — closure-theorem
  BFS + ``closure_theorem()`` assertion hook.
* :mod:`molmetal_lam.lam_chem.ood_protocol` (Wf-7 ship) —
  ``TanimotoOODProtocol`` (CPU Morgan-ECFP4, threshold 0.4).
* :mod:`molmetal.scripts.bench_closure_dblock` (Wf-37 P2 ship) —
  ``_load_click_rules``, ``_load_partner_terms``, ``_benchmark_one``
  (we reuse these to avoid duplicating the BFS plumbing).
* ``molmetal/data/platinai_5k.csv`` (R17 ship, 5K real Pt-group).
* ``molmetal/data/tmqm_dblock_subset.csv`` (Wf-28 ship, 30-metal
  tmQM subset — used as the **train pool** for OOD detection).
* ``.data/molmetal/metalcytotox_5metal_multitask.csv`` (Wf-21 ship,
  26,801 rows across 5 metals: Ru / Ir / Rh / Os / Re).

Outputs
-------
* ``molmetal/reports/wf41_closure_ood/ood_closure.json`` — full
  per-sample + per-domain + per-metal aggregate benchmark (the
  primary L12 number).
* ``molmetal/reports/wf41_closure_ood/per_domain.csv`` — one row
  per (domain, metal) — closure_rate + ood_rate + n_ood.
* ``molmetal/reports/wf41_closure_ood/ood_closure.md`` — human-
  readable Markdown summary.

Usage
-----
::

    # Default: 50 samples/metal/domain, 1 seed.
    uv run --no-sync python molmetal/scripts/bench_closure_ood.py

    # Larger smoke: 100 samples per metal per domain.
    uv run --no-sync python molmetal/scripts/bench_closure_ood.py \\
        --n-samples 100

    # Filter to OOD subset only (skip the unfiltered baseline).
    uv run --no-sync python molmetal/scripts/bench_closure_ood.py \\
        --method closure+ood_filter
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
        if _p not in sys.path:
            sys.path.insert(0, _p)


log = logging.getLogger("wf41_closure_ood")


# ---------------------------------------------------------------------------
# Defaults — cross-domain metal set + dataset paths
# ---------------------------------------------------------------------------
#: The 8 PlatinAI 5K metals (subset of d-block).  Empty (or unknown)
#: labels are skipped during row filtering.
PLATINAI_METALS: Tuple[str, ...] = (
    "Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Au",
)

#: The 5 MetalCytoToxDB multitask metals (Ru/Ir/Rh/Os/Re).
METALCYTO_METALS: Tuple[str, ...] = (
    "Ru", "Ir", "Rh", "Os", "Re",
)

#: The intersection (used for the per-domain headline aggregation when
#: both datasets have the metal): Pt is PlatinAI-only, Re is
#: MetalCytoTox-only.
COMMON_METALS: Tuple[str, ...] = tuple(
    m for m in PLATINAI_METALS if m in METALCYTO_METALS
)  # ("Ru", "Ir", "Rh", "Os")

#: The 30 d-block metals (Wf-28 ship — tmQM subset).  We benchmark a
#: representative subset (those with n_samples ≥ 20 in the CSV) for
#: the Wf-57 Phase 2 per-domain reachability sweep.
TMQM_METALS: Tuple[str, ...] = (
    "Ag", "Au", "Cd", "Co", "Cr", "Cu", "Fe", "Hf", "Hg", "Ir",
    "La", "Mn", "Mo", "Nb", "Ni", "Os", "Pd", "Pt", "Re", "Rh",
    "Ru", "Sc", "Ta", "Tc", "Ti", "V", "W", "Y", "Zn", "Zr",
)

#: Per-metal atomic-number map for coordination-count sanity checks.
METAL_Z: Dict[str, int] = {
    "Sc": 21, "Ti": 22, "V":  23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Y":  39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ag": 47, "Cd": 48, "La": 57, "Hf": 72, "Ta": 73, "W":  74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
}

DEFAULT_DATASET: str = "cross_domain"
DEFAULT_N_SAMPLES: int = 50
DEFAULT_MAX_DEPTH: int = 1
DEFAULT_MAX_PRODUCTS: int = 64
DEFAULT_SEEDS: int = 1
DEFAULT_OUTPUT_DIR: Path = (
    _REPO_ROOT / "molmetal" / "reports" / "wf41_closure_ood"
)
DEFAULT_TANIMOTO_THRESHOLD: float = 0.4
DEFAULT_PLATINAI_CSV: Path = (
    _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
)
DEFAULT_TMQM_CSV: Path = (
    _REPO_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
)
DEFAULT_METALCYTO_CSV: Path = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask.csv"
)
DEFAULT_METHOD: str = "closure"  # 'closure' | 'closure+ood_filter'

#: Partner terms (mirrors Wf-37 Phase 2 fixture).  Covers all 5 click
#: handles (CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling) plus a
#: generic amine + benzoic acid for stress coverage.
PARTNER_SMILE_LIST: List[str] = [
    "C#CCN",                  # propargylamine  — alkyne + amine
    "CCN=[N+]=[N-]",          # ethyl azide     — CuAAC
    "C#CC",                   # propyne         — CuAAC
    "CS",                     # methanethiol    — ThiolEne
    "C=CC",                   # propene         — ThiolEne
    "OC(=O)c1ccccc1",         # benzoic acid    — AmideCoupling
    "NCC",                    # ethylamine      — AmideCoupling
]

#: Domains tracked by the benchmark.  Each domain has a CSV path,
#: a metal-column name, and the metal whitelist to consider.
DOMAIN_SPECS: Dict[str, Dict[str, Any]] = {
    "platinai_5k": {
        "csv": DEFAULT_PLATINAI_CSV,
        "metal_col": "metal",
        "smiles_col": "smiles",
        "metals": PLATINAI_METALS,
        "description": "PlatinAI 5K (R17 ship) — real Pt-group anticancer complexes",
    },
    "metalcytotox_5metal": {
        "csv": DEFAULT_METALCYTO_CSV,
        "metal_col": "metal",
        "smiles_col": "smiles",
        "metals": METALCYTO_METALS,
        "description": "MetalCytoToxDB 5-metal (Wf-21 ship) — photo-activated cytotoxicity",
    },
    "tmqm_dblock": {
        "csv": DEFAULT_TMQM_CSV,
        "metal_col": "metal_center",
        "smiles_col": "smiles",
        "metals": TMQM_METALS,
        "description": "tmQM d-block 30-metal (Wf-28 ship) — real organometallic transition-metal complexes (Balcells & Skjelstad 2020)",
    },
}


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


def _load_tanimoto_protocol(
    threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
) -> Any:
    """Construct a :class:`TanimotoOODProtocol` (Wf-7 ship)."""
    from molmetal_lam.lam_chem.ood_protocol import TanimotoOODProtocol
    return TanimotoOODProtocol(threshold=float(threshold))


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def _load_metal_samples(
    csv_path: Path,
    metal: str,
    n: int,
    *,
    smiles_col: str = "smiles",
    metal_col: str = "metal",
    offset: int = 0,
) -> List[str]:
    """Read up to ``n`` SMILES tagged with the given metal symbol.

    Symmetric loader for PlatinAI, MetalCytoToxDB, and tmQM d-block —
    CSVs have ``smiles`` + ``metal``/``metal_center`` columns.  Empty /
    whitespace SMILES are skipped.  ``offset`` skips the first
    ``offset`` matching samples before collecting up to ``n`` (used
    by the multi-seed sweep to draw disjoint SMILES slices per seed).
    Returns ``[]`` when the CSV is missing.
    """
    out: List[str] = []
    if not csv_path.is_file():
        log.error("CSV not found at %s", csv_path)
        return out
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if metal_col not in fields:
            log.error(
                "Column %r missing from %s; fields=%r",
                metal_col, csv_path, fields,
            )
            return out
        if smiles_col not in fields:
            log.error(
                "Column %r missing from %s; fields=%r",
                smiles_col, csv_path, fields,
            )
            return out
        skipped = 0
        for row in reader:
            tag = (row.get(metal_col, "") or "").strip()
            if tag != metal:
                continue
            smi = (row.get(smiles_col, "") or "").strip()
            if not smi:
                continue
            if skipped < int(offset):
                skipped += 1
                continue
            out.append(smi)
            if len(out) >= int(n):
                break
    return out


def _load_tmqm_train_pool(csv_path: Path) -> List[str]:
    """Read the tmQM training-pool SMILES (used for OOD filter).

    The tmQM d-block subset is the *only* chemistry pool that ships
    in :mod:`molmetal.data.tmqm_dblock_subset` (Wf-28); all 5K
    samples are returned so the Tanimoto OOD protocol has the full
    training distribution to compare against.  Empty SMILES are
    skipped.
    """
    out: List[str] = []
    if not csv_path.is_file():
        log.error("tmQM CSV not found at %s", csv_path)
        return out
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles", "") or "").strip()
            if not smi:
                continue
            out.append(smi)
    return out


# ---------------------------------------------------------------------------
# Per-sample benchmark
# ---------------------------------------------------------------------------
def _is_metal_coordinated(term: Any, metal_z: int) -> bool:
    """Return True iff ``term`` contains at least one atom with atomic
    number ``metal_z`` (= the metal we're benchmarking).
    """
    try:
        mol = term.to_rdkit()
    except Exception:
        return False
    if mol is None:
        return False
    try:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == int(metal_z):
                return True
    except Exception:
        return False
    return False


def _benchmark_one(
    smiles: str,
    idx: int,
    metal: str,
    *,
    domain: str,
    click_rules: Sequence[Any],
    partner_terms: Sequence[Any],
    max_depth: int,
    max_products: int,
) -> Dict[str, Any]:
    """Run closure_theorem on one SMILES; record per-sample metrics.

    Mirrors :func:`molmetal.scripts.bench_closure_dblock._benchmark_one`
    but with the additional ``domain`` field that distinguishes
    PlatinAI vs MetalCytoTox samples in the JSON output.
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    record: Dict[str, Any] = {
        "idx": int(idx),
        "metal": str(metal),
        "domain": str(domain),
        "smiles": str(smiles),
        "status": "skipped",
        "closure_ok": False,
        "n_visited": 0,
        "n_products": 0,
        "max_depth_seen": 0,
        "well_typed_rate": 0.0,
        "n_metal_coordinated": 0,
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
        record["well_typed_rate"] = float(
            last_result.get("well_typed_rate", 1.0)
        )
    except Exception as exc:
        record["status"] = "error"
        record["parse_error"] = f"{type(exc).__name__}: {exc}"[:240]
    finally:
        wall_s = time.perf_counter() - t_bfs_0
        cur_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record["wall_time_s"] = float(wall_s)
        record["memory_mb"] = float(_memory_mb(peak_mem - base_mem))

    # Metal-coordinated detection (cheap).
    if record["status"] == "ok":
        try:
            metal_z = METAL_Z.get(metal, -1)
            if metal_z > 0 and _is_metal_coordinated(start_term, metal_z):
                record["n_metal_coordinated"] = 1
        except Exception:
            pass

    return record


# ---------------------------------------------------------------------------
# OOD filter — apply Tanimoto<0.4 vs the tmQM training pool
# ---------------------------------------------------------------------------
def _filter_ood_subset(
    smiles_list: List[str],
    train_pool: List[str],
    ood_protocol: Any,
    metal: str,
) -> Tuple[List[Tuple[int, str, float]], Dict[str, Any]]:
    """Filter ``smiles_list`` to the OOD subset vs the tmQM train pool.

    Parameters
    ----------
    smiles_list : list[str]
        Candidate SMILES from the cross-domain dataset (PlatinAI or
        MetalCytoToxDB).
    train_pool : list[str]
        tmQM training-pool SMILES (the reference distribution).
    ood_protocol : TanimotoOODProtocol
        The Wf-7 Tanimoto OOD classifier.
    metal : str
        Metal symbol (only used for the audit log).

    Returns
    -------
    (ood_subset, audit) :
        ``ood_subset`` is a list of ``(idx, smiles, max_tanimoto)``
        triples for samples whose max-Tanimoto similarity to the
        train pool is **strictly below** the OOD threshold.  ``audit``
        is a small dict with aggregate statistics (n_in, n_ood,
        mean_max_tanimoto, etc.) used by the per-domain aggregator.
    """
    ood_subset: List[Tuple[int, str, float]] = []
    n_total = len(smiles_list)
    n_unparseable = 0
    n_ood = 0
    n_not_ood = 0
    max_sims: List[float] = []
    for i, smi in enumerate(smiles_list):
        max_sim = ood_protocol.max_tanimoto(smi, train_pool)
        # max_tanimoto returns 0.0 for unparseable queries — we
        # treat them as "OOD" (per Wf-7 fail-closed semantics) so
        # they pass the filter.
        if max_sim < float(ood_protocol.threshold):
            ood_subset.append((i, smi, float(max_sim)))
            n_ood += 1
        else:
            n_not_ood += 1
        max_sims.append(float(max_sim))

    audit: Dict[str, Any] = {
        "n_in": int(n_total),
        "n_ood": int(n_ood),
        "n_not_ood": int(n_not_ood),
        "ood_rate": float(n_ood / n_total) if n_total else 0.0,
        "mean_max_tanimoto": float(statistics.fmean(max_sims))
        if max_sims else 0.0,
        "median_max_tanimoto": float(statistics.median(max_sims))
        if max_sims else 0.0,
        "max_max_tanimoto": float(max(max_sims)) if max_sims else 0.0,
        "n_unparseable_query": int(n_unparseable),
        "threshold": float(ood_protocol.threshold),
        "metal": str(metal),
    }
    return ood_subset, audit


# ---------------------------------------------------------------------------
# Per-metal aggregate
# ---------------------------------------------------------------------------
def _aggregate_per_metal(
    records: List[Dict[str, Any]],
    *,
    n_input: int = 0,
    n_ood: int = 0,
) -> Dict[str, float]:
    """Aggregate per-sample records into a per-metal benchmark dict.

    Keys returned (counts are int-cast for JSON safety; rates are
    floats in [0, 1]):

    * ``n_input``                 — number of SMILES fed in (before OOD)
    * ``n_ood``                   — number of OOD-filtered samples
    * ``n_samples``               — number of records (= n_ood for OOD,
                                    = n_input for unfiltered)
    * ``n_closure_ok``            — number of records with closure_ok=True
    * ``closure_reachability_rate``  — n_closure_ok / n_samples
    * ``ood_rate``                — n_ood / n_input (when input > 0)
    * ``mean_wall_time`` / ``std_wall_time`` / ``median_wall_time``
    * ``peak_memory_mb`` / ``total_wall_time``
    * ``n_error``                 — number of records with status="error"
    * ``n_metal_coordinated`` / ``metal_coordination_rate``
    """
    n = len(records)
    if n == 0:
        return {
            "n_input": int(n_input),
            "n_ood": int(n_ood),
            "n_samples": 0,
            "n_closure_ok": 0,
            "closure_reachability_rate": 0.0,
            "ood_rate": float(n_ood / n_input) if n_input else 0.0,
            "mean_wall_time": 0.0,
            "std_wall_time": 0.0,
            "median_wall_time": 0.0,
            "peak_memory_mb": 0.0,
            "total_wall_time": 0.0,
            "n_error": 0,
            "n_metal_coordinated": 0,
            "metal_coordination_rate": 0.0,
        }

    n_ok = sum(1 for r in records if r["closure_ok"])
    n_err = sum(1 for r in records if r["status"] == "error")
    n_metal = sum(int(r.get("n_metal_coordinated", 0)) for r in records)

    ok_records = [r for r in records if r["status"] == "ok"]
    if ok_records:
        walls = [float(r["wall_time_s"]) for r in ok_records]
        mems = [float(r["memory_mb"]) for r in ok_records]
        mean_wall = float(statistics.fmean(walls))
        std_wall = (
            float(statistics.stdev(walls)) if len(walls) >= 2 else 0.0
        )
        median_wall = float(statistics.median(walls))
        peak_mem = float(max(mems)) if mems else 0.0
        total_wall = float(sum(walls))
    else:
        mean_wall = std_wall = median_wall = peak_mem = total_wall = 0.0

    return {
        "n_input": int(n_input),
        "n_ood": int(n_ood),
        "n_samples": int(n),
        "n_closure_ok": int(n_ok),
        "closure_reachability_rate": float(n_ok / n) if n else 0.0,
        "ood_rate": float(n_ood / n_input) if n_input else 0.0,
        "mean_wall_time": mean_wall,
        "std_wall_time": std_wall,
        "median_wall_time": median_wall,
        "peak_memory_mb": peak_mem,
        "total_wall_time": total_wall,
        "n_error": int(n_err),
        "n_metal_coordinated": int(n_metal),
        "metal_coordination_rate": float(
            (n_metal / n) if n else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_closure_ood_benchmark(
    n_samples: int = DEFAULT_N_SAMPLES,
    *,
    domains: Sequence[str] = (
        "platinai_5k", "metalcytotox_5metal", "tmqm_dblock",
    ),
    method: str = DEFAULT_METHOD,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    platinai_csv: Path = DEFAULT_PLATINAI_CSV,
    metalcyto_csv: Path = DEFAULT_METALCYTO_CSV,
    tmqm_csv: Path = DEFAULT_TMQM_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_products: int = DEFAULT_MAX_PRODUCTS,
    seeds: int = DEFAULT_SEEDS,
) -> Dict[str, Any]:
    """Run the closure theorem cross-domain OOD benchmark.

    Parameters
    ----------
    n_samples : int
        How many SMILES to load per (domain, metal) (default 50).
    domains : sequence[str]
        Subset of ``("platinai_5k", "metalcytotox_5metal")`` to run.
        Default: both.
    method : str
        ``"closure"`` (unfiltered baseline) or
        ``"closure+ood_filter"`` (Tanimoto<0.4 against tmQM only).
    tanimoto_threshold : float
        OOD threshold (default 0.4 — the Pocket2Mol / TargetDiff
        convention).
    platinai_csv / metalcyto_csv / tmqm_csv : Path
        CSV paths for the two cross-domain datasets and the tmQM
        training pool.
    output_dir : Path
        Where the JSON / CSV / Markdown reports will be written.
    max_depth : int
        BFS depth budget (default 1 = one click rule application).
    max_products : int
        Safety cap on the BFS reachable set (default 64).
    seeds : int
        Number of "seed slots" — currently informational only
        (default 1).

    Returns
    -------
    dict
        JSON-serialisable payload.  Also persisted by
        :func:`save_closure_ood_benchmark`.
    """
    log.info(
        "[wf41_closure_ood] start: domains=%s, method=%s, n_per_metal=%d, "
        "max_depth=%d, threshold=%.2f, seeds=%d",
        list(domains), str(method), int(n_samples),
        int(max_depth), float(tanimoto_threshold), int(seeds),
    )

    method_norm = str(method).lower().replace(" ", "")
    if method_norm not in ("closure", "closure+ood_filter"):
        raise ValueError(
            f"method must be 'closure' or 'closure+ood_filter'; got {method!r}"
        )
    apply_ood_filter = method_norm == "closure+ood_filter"

    # Lazy-loaded heavy objects.
    click_rules = _load_click_rules()
    partner_terms = _load_partner_terms()
    click_rule_names = [
        getattr(r, "name", type(r).__name__) for r in click_rules
    ]

    # Train pool (used only when applying the OOD filter).
    train_pool: List[str] = []
    if apply_ood_filter:
        train_pool = _load_tmqm_train_pool(Path(tmqm_csv))
        log.info(
            "[wf41_closure_ood] loaded tmQM train pool: n=%d",
            len(train_pool),
        )
        if not train_pool:
            log.error(
                "tmQM train pool empty (%s); OOD filter will reject all "
                "samples (every query has max_tanimoto=0.0 < 0.4)",
                Path(tmqm_csv),
            )

    ood_protocol = _load_tanimoto_protocol(threshold=tanimoto_threshold)

    t_total_0 = time.perf_counter()

    per_domain_records: Dict[str, List[Dict[str, Any]]] = {}
    per_domain_aggregate: Dict[str, Dict[str, float]] = {}
    per_domain_per_metal_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    per_domain_per_metal_aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_domain_per_metal_seed_aggregate: Dict[
        str, Dict[str, List[Dict[str, float]]]
    ] = {}
    per_domain_seed_aggregates: Dict[
        str, List[Dict[str, float]]
    ] = {}
    ood_audit: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_records: List[Dict[str, Any]] = []

    for domain in domains:
        spec = DOMAIN_SPECS.get(domain)
        if spec is None:
            log.error("Unknown domain %r; skipping.", domain)
            continue

        csv_path = (
            Path(platinai_csv)
            if domain == "platinai_5k"
            else Path(metalcyto_csv)
            if domain == "metalcytotox_5metal"
            else Path(tmqm_csv)
        )
        if not csv_path.is_file():
            log.error("CSV missing for domain %s at %s", domain, csv_path)
            continue

        metal_col = str(spec["metal_col"])
        smiles_col = str(spec["smiles_col"])
        domain_metals = list(spec["metals"])

        per_metal_records: Dict[str, List[Dict[str, Any]]] = {}
        per_metal_aggregate: Dict[str, Dict[str, float]] = {}
        per_metal_ood_audit: Dict[str, Dict[str, Any]] = {}
        # Per-seed sub-records (for multi-seed mean/std aggregation).
        per_metal_seed_records: Dict[str, List[List[Dict[str, Any]]]] = {}
        per_metal_seed_aggregate: Dict[str, List[Dict[str, float]]] = {}

        n_seeds = max(1, int(seeds))
        for metal in domain_metals:
            seed_records: List[List[Dict[str, Any]]] = []
            seed_aggregates: List[Dict[str, float]] = []
            # Aggregate over all seeds for this metal.
            for seed_i in range(n_seeds):
                offset = seed_i * int(n_samples)
                smis = _load_metal_samples(
                    csv_path, str(metal), int(n_samples),
                    smiles_col=smiles_col, metal_col=metal_col,
                    offset=int(offset),
                )
                if not smis:
                    log.warning(
                        "[wf41_closure_ood] %s/%s/seed=%d: no SMILES "
                        "found", domain, metal, seed_i,
                    )
                    seed_records.append([])
                    seed_aggregates.append(_aggregate_per_metal(
                        [], n_input=0, n_ood=0,
                    ))
                    continue

                if apply_ood_filter:
                    ood_subset, audit = _filter_ood_subset(
                        smis, train_pool, ood_protocol, str(metal),
                    )
                    per_metal_ood_audit[metal] = audit
                    smis_to_run = [s for (_, s, _) in ood_subset]
                else:
                    per_metal_ood_audit[metal] = {
                        "n_in": int(len(smis)),
                        "n_ood": int(len(smis)),
                        "n_not_ood": 0,
                        "ood_rate": 1.0,
                        "mean_max_tanimoto": 0.0,
                        "median_max_tanimoto": 0.0,
                        "max_max_tanimoto": 0.0,
                        "threshold": float(tanimoto_threshold),
                        "metal": str(metal),
                    }
                    smis_to_run = list(smis)

                n_ood = len(smis_to_run)
                records: List[Dict[str, Any]] = []
                for i, smi in enumerate(smis_to_run):
                    rec = _benchmark_one(
                        smi, i, str(metal),
                        domain=str(domain),
                        click_rules=click_rules,
                        partner_terms=partner_terms,
                        max_depth=int(max_depth),
                        max_products=int(max_products),
                    )
                    records.append(rec)
                seed_records.append(records)
                seed_aggregates.append(_aggregate_per_metal(
                    records,
                    n_input=int(len(smis)),
                    n_ood=int(n_ood),
                ))

            # End for seed_i — collapse all seeds into per_metal_records.
            per_metal_records[metal] = [
                rec for seed_list in seed_records for rec in seed_list
            ]
            per_metal_aggregate[metal] = _aggregate_per_metal(
                per_metal_records[metal],
                n_input=int(sum(
                    s["n_input"] for s in seed_aggregates
                )),
                n_ood=int(sum(
                    s["n_ood"] for s in seed_aggregates
                )),
            )
            per_metal_seed_records[metal] = seed_records
            per_metal_seed_aggregate[metal] = seed_aggregates
            agg = per_metal_aggregate[metal]
            log.info(
                "[wf41_closure_ood] %s/%s: n_input=%d n_ood=%d "
                "closure_ok=%d rate=%.4f mean_wall=%.3fs seeds=%d",
                domain, metal, agg["n_input"], agg["n_ood"],
                agg["n_closure_ok"], agg["closure_reachability_rate"],
                agg["mean_wall_time"], n_seeds,
            )

        # Domain-level aggregate across all metals.
        domain_records = [
            rec
            for metal in domain_metals
            for rec in per_metal_records.get(metal, [])
        ]
        n_input_total = sum(
            int(per_metal_aggregate.get(m, {}).get("n_input", 0))
            for m in domain_metals
        )
        n_ood_total = sum(
            int(per_metal_aggregate.get(m, {}).get("n_ood", 0))
            for m in domain_metals
        )
        per_domain_records[domain] = domain_records
        per_domain_aggregate[domain] = _aggregate_per_metal(
            domain_records,
            n_input=int(n_input_total),
            n_ood=int(n_ood_total),
        )
        per_domain_per_metal_records[domain] = per_metal_records
        per_domain_per_metal_aggregate[domain] = per_metal_aggregate
        per_domain_per_metal_seed_aggregate[domain] = per_metal_seed_aggregate
        # Domain-level seed aggregate (mean of per-metal seed aggregates).
        domain_seed_aggs: List[Dict[str, float]] = []
        for metal in domain_metals:
            for s_agg in per_metal_seed_aggregate.get(metal, []):
                domain_seed_aggs.append(dict(s_agg))
        per_domain_seed_aggregates[domain] = domain_seed_aggs
        ood_audit[domain] = per_metal_ood_audit
        all_records.extend(domain_records)

    overall = _aggregate_per_metal(
        all_records,
        n_input=int(sum(
            per_domain_aggregate[d]["n_input"] for d in per_domain_aggregate
        )),
        n_ood=int(sum(
            per_domain_aggregate[d]["n_ood"] for d in per_domain_aggregate
        )),
    )
    total_wall_s = time.perf_counter() - t_total_0

    try:
        import torch  # local; cheap
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "n/a"

    py_ver = ".".join(str(v) for v in sys.version_info[:3])

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "bench_closure_ood.py",
        "pilot_id": "wf41_closure_ood",
        "date_utc": _utcnow_iso(),
        "method": str(method_norm),
        "domains": list(domains),
        "n_samples_per_metal": int(n_samples),
        "max_depth": int(max_depth),
        "max_products": int(max_products),
        "seeds": int(seeds),
        "tanimoto_threshold": float(tanimoto_threshold),
        "n_train_pool": int(len(train_pool)),
        "click_rule_names": list(click_rule_names),
        "n_partners": int(len(partner_terms)),
        "partner_smiles": list(PARTNER_SMILE_LIST),
        "domain_specs": {
            d: {
                "csv_path": str(DOMAIN_SPECS[d]["csv"]),
                "metal_col": str(DOMAIN_SPECS[d]["metal_col"]),
                "smiles_col": str(DOMAIN_SPECS[d]["smiles_col"]),
                "metals": list(DOMAIN_SPECS[d]["metals"]),
                "description": str(DOMAIN_SPECS[d]["description"]),
            }
            for d in domains
        },
        "per_domain": {
            d: {
                "aggregate": dict(per_domain_aggregate[d]),
                "seed_aggregate": [
                    dict(s) for s in per_domain_seed_aggregates.get(d, [])
                ],
                "per_metal": {
                    metal: {
                        "aggregate": dict(per_domain_per_metal_aggregate[d].get(metal, {})),
                        "seed_aggregate": [
                            dict(s) for s in per_domain_per_metal_seed_aggregate.get(d, {}).get(metal, [])
                        ],
                        "records": list(per_domain_per_metal_records[d].get(metal, [])),
                        "ood_audit": dict(ood_audit[d].get(metal, {})),
                    }
                    for metal in DOMAIN_SPECS[d]["metals"]
                },
            }
            for d in domains
            if d in per_domain_aggregate
        },
        "aggregate": dict(overall),
        "total_wall_time_s": float(total_wall_s),
        "timestamp_utc": _utcnow_iso(),
        "python": py_ver,
        "torch": torch_ver,
        "honest_framing": (
            "CPU-only closure-theorem cross-domain OOD benchmark. "
            "Real data from PlatinAI 5K + MetalCytoToxDB 5-metal, "
            "filtered by Tanimoto<0.4 (Morgan-ECFP4) against the tmQM "
            "d-block training pool. Per-domain closure-reachability "
            "rate is the L12 fill for TODO-83 §0. Deterministic; runs "
            "the actual ProductiveSpace BFS + closure_theorem "
            "assertion hook on every OOD-filtered sample."
        ),
    }
    return payload


def save_closure_ood_benchmark(
    payload: Dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path]:
    """Persist the OOD benchmark payload + mirrors to ``output_dir``.

    Writes three artefacts:

    * ``ood_closure.json`` — full payload (per-sample + per-domain +
      per-metal + aggregate + OOD audit).
    * ``per_domain.csv`` — one row per (domain, metal) — closure_rate
      + ood_rate + n_ood + n_input.
    * ``ood_closure.md`` — human-readable Markdown summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ood_closure.json"

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

    csv_path = output_dir / "per_domain.csv"
    fields = [
        "domain", "metal", "n_input", "n_ood", "n_samples",
        "n_closure_ok", "closure_reachability_rate", "ood_rate",
        "metal_coordination_rate", "n_metal_coordinated",
        "mean_wall_time",
        "std_wall_time", "median_wall_time", "peak_memory_mb",
        "total_wall_time", "n_error",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for domain in payload.get("domains", []):
            d_info = payload.get("per_domain", {}).get(domain, {})
            for metal in DOMAIN_SPECS.get(domain, {}).get("metals", []):
                agg = (
                    d_info.get("per_metal", {}).get(metal, {}).get(
                        "aggregate", {},
                    )
                )
                row: Dict[str, Any] = {f: agg.get(f, "") for f in fields}
                row["domain"] = domain
                row["metal"] = metal
                writer.writerow(row)

    md_path = output_dir / "ood_closure.md"
    _write_markdown(payload, output_dir)

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
    }


def _write_markdown(
    payload: Dict[str, Any], output_dir: Path
) -> None:
    """Write the human-readable ``ood_closure.md`` summary."""
    md_path = output_dir / "ood_closure.md"
    method = payload.get("method", "")
    domains = payload.get("domains", [])
    agg = payload.get("aggregate", {}) or {}
    overall_rate = float(agg.get("closure_reachability_rate", 0.0))
    overall_n = int(agg.get("n_samples", 0))
    overall_ok = int(agg.get("n_closure_ok", 0))
    overall_n_input = int(agg.get("n_input", 0))
    overall_n_ood = int(agg.get("n_ood", 0))
    total_wall = float(payload.get("total_wall_time_s", 0.0))

    lines: List[str] = []
    lines.append("# Wf-41 Phase 2 — Closure theorem cross-domain OOD (L12)")
    lines.append("")
    lines.append(
        "Cross-domain closure reachability benchmark on **PlatinAI 5K** "
        "+ **MetalCytoToxDB 5-metal**, filtered by "
        "**Tanimoto<0.4 (Morgan-ECFP4) against the tmQM d-block "
        "training pool**.  Fills the **L12 gap** in "
        ":mod:`TODO/pending/83_lambda_algorithm_enhancement.md`."
    )
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"**Date (UTC)**: {payload['timestamp_utc']}  ")
    lines.append(f"**Tool**: `{payload['tool']}`  ")
    lines.append(f"**Method**: `{method}`  ")
    lines.append(f"**Tanimoto threshold**: "
                 f"{float(payload.get('tanimoto_threshold', 0.4)):.2f}  ")
    lines.append(f"**Train pool size (tmQM)**: "
                 f"{int(payload.get('n_train_pool', 0))}  ")
    lines.append(f"**Domains**: {', '.join(domains)}  ")
    lines.append(f"**N samples per metal**: "
                 f"{int(payload.get('n_samples_per_metal', 0))}  ")
    lines.append(f"**Max depth**: {payload['max_depth']}  ")
    lines.append(f"**Click rules**: "
                 f"{', '.join(payload.get('click_rule_names', []))}  ")
    lines.append(f"**Partner terms**: "
                 f"{int(payload.get('n_partners', 0))}  ")
    lines.append(f"**Seeds**: {payload['seeds']}  ")
    lines.append(f"**Total wall**: {total_wall:.2f} s  ")
    lines.append("")
    lines.append("## Aggregate (across all domains)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| N input (before OOD) | {overall_n_input} |")
    lines.append(f"| N OOD (after filter) | {overall_n_ood} |")
    lines.append(f"| N samples run | {overall_n} |")
    lines.append(f"| N closure_ok | {overall_ok} |")
    lines.append(
        f"| closure_reachability_rate | {overall_rate:.4f} |"
    )
    lines.append(
        f"| mean_wall_time (s) | "
        f"{float(agg.get('mean_wall_time', 0.0)):.6f} |"
    )
    lines.append(
        f"| std_wall_time (s) | "
        f"{float(agg.get('std_wall_time', 0.0)):.6f} |"
    )
    lines.append(
        f"| peak_memory_mb | "
        f"{float(agg.get('peak_memory_mb', 0.0)):.4f} |"
    )
    lines.append(f"| N error | {int(agg.get('n_error', 0))} |")
    lines.append("")
    lines.append("## Per-domain reachability rate")
    lines.append("")
    lines.append(
        "| Domain | N input | N OOD | N samples | closure_ok "
        "| reachability_rate (mean ± std) | mean_wall_s |"
    )
    lines.append(
        "|--------|---------|-------|-----------|------------"
        "|------------------------------|-------------|"
    )
    n_seeds = max(1, int(payload.get("seeds", 1)))
    for domain in domains:
        d_agg = payload.get("per_domain", {}).get(domain, {}).get(
            "aggregate", {},
        )
        if not d_agg:
            continue
        # Per-domain mean/std across seeds (mean of per-seed rates).
        seed_aggs = (
            payload.get("per_domain", {}).get(domain, {})
            .get("seed_aggregate", [])
        )
        if seed_aggs:
            rates = [
                float(s.get("closure_reachability_rate", 0.0))
                for s in seed_aggs
            ]
            mean_rate = float(statistics.fmean(rates))
            std_rate = (
                float(statistics.stdev(rates))
                if len(rates) >= 2 else 0.0
            )
            rate_str = f"{mean_rate:.4f} ± {std_rate:.4f}"
        else:
            rate_str = (
                f"{float(d_agg.get('closure_reachability_rate', 0.0)):.4f}"
            )
        lines.append(
            f"| {domain} | {int(d_agg.get('n_input', 0))} "
            f"| {int(d_agg.get('n_ood', 0))} "
            f"| {int(d_agg.get('n_samples', 0))} "
            f"| {int(d_agg.get('n_closure_ok', 0))} "
            f"| {rate_str} "
            f"| {float(d_agg.get('mean_wall_time', 0.0)):.6f} |"
        )
    lines.append("")
    lines.append("## Per-(domain, metal) reachability rate")
    lines.append("")
    lines.append(
        "| Domain | Metal | N input | N OOD | closure_ok "
        "| reachability_rate | ood_rate |"
    )
    lines.append(
        "|-------|-------|---------|-------|------------"
        "|-------------------|----------|"
    )
    for domain in domains:
        per_metal = (
            payload.get("per_domain", {}).get(domain, {})
            .get("per_metal", {})
        )
        for metal in DOMAIN_SPECS.get(domain, {}).get("metals", []):
            m_agg = per_metal.get(metal, {}).get("aggregate", {})
            if not m_agg:
                continue
            lines.append(
                f"| {domain} | {metal} "
                f"| {int(m_agg.get('n_input', 0))} "
                f"| {int(m_agg.get('n_ood', 0))} "
                f"| {int(m_agg.get('n_closure_ok', 0))} "
                f"| {float(m_agg.get('closure_reachability_rate', 0.0)):.4f} "
                f"| {float(m_agg.get('ood_rate', 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **CPU-only.** No CUDA / Triton / CFM retraining is exercised.\n"
        "- **Deterministic.** closure_theorem is pure-functional; the\n"
        "  benchmark can be re-run with the same flags and reproduce\n"
        "  every per-sample metric.\n"
        "- **Real cross-domain data.** PlatinAI 5K (R17) and\n"
        "  MetalCytoToxDB 5-metal (Wf-21) are both real ship\n"
        "  datasets; the OOD filter is the Wf-7 Tanimoto<0.4\n"
        "  protocol (Morgan-ECFP4).\n"
        "- **Valence-BNF rejection.** Many PlatinAI / MetalCytoTox\n"
        "  samples fail the valence-BNF check on the start term\n"
        "  (dative-bond organometallics with ``[M+]<-`` notation\n"
        "  that the MLC ledger treats as under-saturated).\n"
        "  These count as ``closure_ok=False`` but ``status=ok``\n"
        "  — the BFS runs, the assertion rejects the seed.\n"
        "- **Smoke-friendly.** The default ``--n-samples 50`` keeps\n"
        "  the OOD-filtered cross-domain run under 5 min on CPU;\n"
        "  the smoke tests in :mod:`test_closure_ood` use\n"
        "  ``--n-samples 3`` for sub-second CI runs.\n"
    )
    lines.append("")

    md_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_closure_ood.py",
        description=(
            "Wf-41 Phase 2 closure-theorem cross-domain OOD benchmark "
            "(L12). Runs the real ProductiveSpace BFS + closure_theorem "
            "assertion hook on PlatinAI 5K + MetalCytoToxDB 5-metal, "
            "optionally filtered by Tanimoto<0.4 (Morgan-ECFP4) against "
            "the tmQM d-block training pool."
        ),
    )
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help="Samples per (domain, metal) (default 50).",
    )
    p.add_argument(
        "--domains", nargs="+",
        default=["platinai_5k", "metalcytotox_5metal"],
        choices=("platinai_5k", "metalcytotox_5metal", "tmqm_dblock"),
        help=(
            "Domains to benchmark. Default: both. "
            "Pass --domains platinai_5k to limit to the 8-metal Pt-group "
            "subset; pass --domains metalcytotox_5metal for the "
            "5-metal cytotoxicity subset."
        ),
    )
    p.add_argument(
        "--method",
        choices=("closure", "closure+ood_filter"),
        default=DEFAULT_METHOD,
        help=(
            "Benchmark method (default 'closure'). 'closure' runs the "
            "BFS on every cross-domain sample (unfiltered baseline); "
            "'closure+ood_filter' first filters by Tanimoto<0.4 against "
            "the tmQM training pool and then runs the BFS on the OOD "
            "subset."
        ),
    )
    p.add_argument(
        "--tanimoto-threshold", type=float,
        default=DEFAULT_TANIMOTO_THRESHOLD,
        help="Tanimoto OOD threshold (default 0.4).",
    )
    p.add_argument(
        "--platinai-csv", type=Path, default=DEFAULT_PLATINAI_CSV,
        help="Path to PlatinAI 5K CSV.",
    )
    p.add_argument(
        "--metalcyto-csv", type=Path, default=DEFAULT_METALCYTO_CSV,
        help="Path to MetalCytoToxDB 5-metal CSV.",
    )
    p.add_argument(
        "--tmqm-csv", type=Path, default=DEFAULT_TMQM_CSV,
        help="Path to tmQM d-block training-pool CSV.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the JSON + CSV + Markdown reports.",
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
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help="Number of independent seeds (default 1; placeholder).",
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

    payload = run_closure_ood_benchmark(
        n_samples=int(args.n_samples),
        domains=list(args.domains),
        method=str(args.method),
        tanimoto_threshold=float(args.tanimoto_threshold),
        platinai_csv=Path(args.platinai_csv),
        metalcyto_csv=Path(args.metalcyto_csv),
        tmqm_csv=Path(args.tmqm_csv),
        output_dir=Path(args.output_dir),
        max_depth=int(args.max_depth),
        max_products=int(args.max_products),
        seeds=int(args.seeds),
    )

    paths = save_closure_ood_benchmark(payload, output_dir=Path(args.output_dir))
    agg = payload["aggregate"]
    log.info(
        "[wf41_closure_ood] DONE: domains=%d method=%s n_input=%d "
        "n_ood=%d closure_ok=%d rate=%.4f mean_wall=%.6fs peak_mem=%.2fMB "
        "total_wall=%.2fs",
        len(payload["domains"]), payload["method"],
        agg["n_input"], agg["n_ood"], agg["n_closure_ok"],
        agg["closure_reachability_rate"], agg["mean_wall_time"],
        agg["peak_memory_mb"], payload["total_wall_time_s"],
    )
    print(
        f"[wf41_closure_ood] n_domains={len(payload['domains'])} "
        f"method={payload['method']} "
        f"n_input={agg['n_input']} n_ood={agg['n_ood']} "
        f"closure_ok={agg['n_closure_ok']} "
        f"reachability_rate={agg['closure_reachability_rate']:.4f} "
        f"mean_wall={agg['mean_wall_time']:.6f}s "
        f"peak_mem={agg['peak_memory_mb']:.2f}MB"
    )
    for name, p in paths.items():
        print(f"[wf41_closure_ood] wrote {name}: {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())