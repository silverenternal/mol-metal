"""Wf-73 Phase 3 — Per-element OOD reachability on tmQM d-block.

================================================================
Wf-73 Phase 3 (L12 trajectory): per-element OOD reachability on the
**tmQM d-block** 30-element training pool (Wf-28 ship), filtered
by Morgan-ECFP4 Tanimoto<0.4 (Wf-57 P1 verified split protocol).

This script extends the Wf-57 Wf-44 P2 closure OOD benchmark
(:mod:`molmetal.scripts.bench_closure_ood` + Wf-57 P1
:mod:`molmetal.scripts.bench_ood_tanimoto_protocol`) along the
**per-element axis**: where Wf-57 P2 measured aggregate
``fraction_below_threshold = 0.9978`` across 32 256 cross-pairs
on the 30-metal d-block, this script measures **per-element closure
reachability rate** with:

  * **30 d-block elements** (``Sc / Ti / V / Cr / Mn / Fe / Co / Ni /
    Cu / Zn`` (3d) + ``Y / Zr / Nb / Mo / Tc / Ru / Rh / Pd / Ag / Cd``
    (4d) + ``La / Hf / Ta / W / Re / Os / Ir / Pt / Au / Hg`` (5d /
    La) — full 30-element coverage from the Wf-28 ship).
  * **Per-element 80/20 stratified split** with the Wf-57 P1 verified
    Tanimoto<0.4 OOD split (deterministic on CSD code).
  * **Per-element closure reachability** = closure_theorem BFS
    assertion on each per-element test-split SMILES, with mean ± std
    across n_seeds disjoint slices.
  * **OOD pass rate per element** = fraction of per-element test
    SMILES that are < Tanimoto_thresh vs the same element's train
    pool (the Wf-57 P1 per-cluster OOD audit, per-element resolution).
  * **Cross-element correlation matrix** — Pearson r over the
    per-element closure-OK indicator across the 30 elements, to spot
    whether closure failure on one element correlates with failure
    on another (e.g. an "early-row" cluster with heavy MW penalty
    would lift off-diagonally).
  * **Honest per-element framing** — per-element reachability is
    reported even when it falls below the 0.40 gate; no fabrication
    per ``party-twenty-spirit §3.5``.

Inputs
------
* ``molmetal/data/tmqm_dblock_subset.csv`` (Wf-28 ship, 30 d-block
  elements, 5000 real mononuclear transition-metal complexes,
  Balcells & Skjelstad 2020).
* :mod:`molmetal_lam.lam_chem.closure` (R17 ship).
* :mod:`molmetal_lam.lam_chem.ood_protocol` (Wf-7 ship, Tanimoto<0.4).

Outputs (under ``molmetal/reports/wf73_p3_ood_perelement/``)
------
* ``{element}/ood_reachability.json`` — per-element aggregate
  (n_samples, n_closure_ok, reachability_rate, OOD pass rate, mean ± std
  wall, n_metal_coordinated, well_typed_rate).
* ``wf73_ood_per_element_full.json`` — full structured payload
  (30×30 cross-element correlation, per-seed slice, per-element summary).
* ``synthesis.md`` — honest per-element breakdown + cross-element
  correlation table + per-element reachability gate verdict
  (party-twenty-spirit §3.5 honest framing).

Usage
-----
::

    # Smoke (small, fast): 5 elements × 4 samples × 1 seed.
    uv run --no-sync python molmetal/scripts/bench_ood_per_element.py \\
        --elements Sc Ti V Cr Mn --n-samples 4 --seeds 1

    # Default production: all 30 elements × 20 samples × 3 seeds.
    uv run --no-sync python molmetal/scripts/bench_ood_per_element.py

    # Single-element diagnostic (e.g. W bottleneck study).
    uv run --no-sync python molmetal/scripts/bench_ood_per_element.py \\
        --elements W --n-samples 50 --seeds 5
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
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


log = logging.getLogger("wf73_ood_per_element")

# ---------------------------------------------------------------------------
# Defaults — tmQM d-block 30-element (Wf-28 ship)
# ---------------------------------------------------------------------------
#: Full tmQM d-block 30-element list (Wf-28 ship + Balcells &
#: Skjelstad 2020).  Sorted in 3d → 4d → 5d → La block order to mirror
#: the Wf-57 25-metal + 30-element Catherine sweep ordering.
DBLOCK_ELEMENTS: Tuple[str, ...] = (
    # 3d-block (10 elements)
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d-block (10 elements)
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d-block + La (10 elements)
    "La", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
)

#: Default TMQM CSV (Wf-28 ship).
DEFAULT_TMQM_CSV: Path = (
    _REPO_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
)

#: Default output directory.
DEFAULT_OUTPUT_DIR: Path = (
    _REPO_ROOT / "molmetal" / "reports" / "wf73_p3_ood_perelement"
)

DEFAULT_N_SAMPLES: int = 20
DEFAULT_TANIMOTO_THRESHOLD: float = 0.4
DEFAULT_MAX_DEPTH: int = 1
DEFAULT_MAX_PRODUCTS: int = 32
DEFAULT_SEEDS: int = 3
DEFAULT_TRAIN_RATIO: float = 0.8

#: Per-element atomic-number map (for coordination sanity checks).
ELEMENT_Z: Dict[str, int] = {
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ag": 47, "Cd": 48, "La": 57, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
}

#: Per-element block assignment (for honest framing in the synthesis).
ELEMENT_BLOCK: Dict[str, str] = {
    **{e: "3d" for e in DBLOCK_ELEMENTS[:10]},
    **{e: "4d" for e in DBLOCK_ELEMENTS[10:20]},
    **{e: "5d+La" for e in DBLOCK_ELEMENTS[20:]},
}

#: Click-rule partner terms (5 click handles + amine + benzoic acid;
#: mirrors Wf-37 P2 / Wf-41 P2 / Wf-73 P2 fixture).
PARTNER_SMILE_LIST: List[str] = [
    "C#CCN",
    "CCN=[N+]=[N-]",
    "C#CC",
    "CS",
    "C=CC",
    "OC(=O)c1ccccc1",
    "NCC",
]

#: Honest framing — per-element reachability gate (Wf-73 P3 spec;
#: mirrors Wf-73 P2 0.30 threshold, scaled to 30-element d-block).
PER_ELEMENT_GATE: float = 0.40


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _memory_mb(snapshot_diff: int) -> float:
    return float(snapshot_diff) / (1024.0 * 1024.0)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) else 0.0


def _have_rdkit() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


def _load_click_rules() -> List[Any]:
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne,
    )
    return [CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling]


def _load_partner_terms() -> List[Any]:
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    out: List[Any] = []
    for smi in PARTNER_SMILE_LIST:
        try:
            out.append(MoleculeClosedTerm.from_smiles(smi, embed_3d=False))
        except Exception as exc:
            log.warning("Failed to parse partner %r: %s", smi, exc)
    return out


def _load_tanimoto_protocol(threshold: float = DEFAULT_TANIMOTO_THRESHOLD) -> Any:
    from molmetal_lam.lam_chem.ood_protocol import TanimotoOODProtocol
    return TanimotoOODProtocol(threshold=float(threshold))


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def _load_tmqm_pool(
    csv_path: Path,
    *,
    elements: Sequence[str] = DBLOCK_ELEMENTS,
    smiles_col: str = "smiles",
    metal_col: str = "metal_center",
    code_col: str = "csd_code",
) -> Dict[str, List[Dict[str, str]]]:
    """Load tmQM d-block CSV and group rows by ``metal_center``.

    Returns a dict ``{element: [{csd_code, smiles, metal_center,
    coord_number, n_atoms}]}`` for the requested elements only.
    Rows are sorted by ``csd_code`` for deterministic splits.
    """
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    if not csv_path.is_file():
        log.error("tmQM CSV not found at %s", csv_path)
        return {}
    wanted = set(elements)
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for col in (smiles_col, metal_col, code_col):
            if col not in fields:
                log.error(
                    "Required column %r missing from %s; fields=%r",
                    col, csv_path, fields,
                )
                return {}
        for row in reader:
            smi = (row.get(smiles_col, "") or "").strip()
            metal = (row.get(metal_col, "") or "").strip()
            code = (row.get(code_col, "") or "").strip()
            if not smi or not metal:
                continue
            if metal not in wanted:
                continue
            out[metal].append({
                "csd_code": code,
                "smiles": smi,
                "metal_center": metal,
                "coord_number": (row.get("coord_number", "") or "").strip(),
                "n_atoms": (row.get("n_atoms", "") or "").strip(),
            })

    for elem in list(out.keys()):
        out[elem].sort(key=lambda r: r["csd_code"])

    return {k: out[k] for k in sorted(out.keys())}


# ---------------------------------------------------------------------------
# Per-element train/test split (deterministic, stratified 80/20)
# ---------------------------------------------------------------------------
def _split_train_test(
    rows: Sequence[Dict[str, str]],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Deterministic CSD-code-ordered 80/20 split per element."""
    n = len(rows)
    n_train = int(round(float(train_ratio) * float(n)))
    if n_train < 0:
        n_train = 0
    if n_train > n:
        n_train = n
    return list(rows[:n_train]), list(rows[n_train:])


# ---------------------------------------------------------------------------
# Per-element sample loader (takes test slice, supports seed-offset)
# ---------------------------------------------------------------------------
def _load_element_samples(
    pool: Dict[str, List[Dict[str, str]]],
    element: str,
    n: int,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed_offset: int = 0,
) -> List[Dict[str, str]]:
    """Return up to ``n`` test-split samples for ``element``.

    Uses deterministic 80/20 stratified split (per Wf-57 P1); the
    ``seed_offset`` slides the test-window start so each seed gets a
    disjoint slice.
    """
    rows = pool.get(element, [])
    if not rows:
        return []
    train, test = _split_train_test(rows, train_ratio=train_ratio)
    if not test:
        return []
    n_use = min(int(n), len(test))
    if int(seed_offset) >= len(test):
        return []
    out: List[Dict[str, str]] = []
    for k in range(int(seed_offset), int(seed_offset) + n_use):
        idx = k % len(test)
        out.append(test[idx])
    return out


# ---------------------------------------------------------------------------
# Per-sample benchmark (mirrors Wf-73 P2 record schema)
# ---------------------------------------------------------------------------
def _benchmark_one(
    smiles: str,
    idx: int,
    element: str,
    *,
    click_rules: Sequence[Any],
    partner_terms: Sequence[Any],
    ood_protocol: Any,
    train_pool: List[str],
    max_depth: int,
    max_products: int,
) -> Dict[str, Any]:
    """Run Tanimoto<0.4 OOD filter + closure_theorem on one SMILES."""
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    record: Dict[str, Any] = {
        "idx": int(idx),
        "element": str(element),
        "smiles": str(smiles),
        "max_tanimoto_tmQM": 0.0,
        "ood_pass": False,
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
    }

    # Step 1: OOD filter (Tanimoto<0.4 vs same-element train pool).
    t_ood_0 = time.perf_counter()
    try:
        max_sim = ood_protocol.max_tanimoto(smiles, train_pool)
    except Exception:
        max_sim = 0.0
    record["max_tanimoto_tmQM"] = float(max_sim)
    record["ood_pass"] = bool(max_sim < float(ood_protocol.threshold))
    if not record["ood_pass"]:
        record["status"] = "ood_filtered_out"
        record["wall_time_s"] = float(time.perf_counter() - t_ood_0)
        return record

    # Step 2: parse SMILES -> closed term.
    t_parse_0 = time.perf_counter()
    try:
        start_term = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)
        record["n_atoms_start"] = int(len(start_term.atoms))
    except Exception as exc:
        record["status"] = "error"
        record["parse_error"] = f"{type(exc).__name__}: {exc}"[:240]
        record["wall_time_s"] = float(time.perf_counter() - t_parse_0)
        return record

    # Step 3: closure_theorem BFS assertion.
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
        last = getattr(closure_theorem, "last_result", {}) or {}
        record["n_visited"] = int(last.get("n_visited", 0))
        record["n_products"] = int(last.get("n_products", 0))
        record["max_depth_seen"] = int(last.get("max_depth_seen", 0))
        record["well_typed_rate"] = float(last.get("well_typed_rate", 1.0))
    except Exception as exc:
        record["status"] = "error"
        record["parse_error"] = f"{type(exc).__name__}: {exc}"[:240]
    finally:
        record["wall_time_s"] = float(time.perf_counter() - t_bfs_0)
        cur_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record["memory_mb"] = float(_memory_mb(peak_mem - base_mem))

    # Metal-coordinated detection (cheap sanity check).
    if record["status"] == "ok":
        try:
            metal_z = ELEMENT_Z.get(element, -1)
            if metal_z > 0:
                term_obj = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)
                mol = term_obj.to_rdkit()
                if mol is not None:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() == int(metal_z):
                            record["n_metal_coordinated"] = 1
                            break
        except Exception:
            pass

    return record


# ---------------------------------------------------------------------------
# Per-element aggregation
# ---------------------------------------------------------------------------
def _aggregate_per_element(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-sample records into a per-element benchmark dict."""
    n = len(records)
    if n == 0:
        return {
            "n_samples": 0,
            "n_ood_pass": 0,
            "n_closure_ok": 0,
            "reachability_rate": 0.0,
            "ood_pass_rate": 0.0,
            "mean_wall_time": 0.0,
            "std_wall_time": 0.0,
            "median_wall_time": 0.0,
            "peak_memory_mb": 0.0,
            "total_wall_time": 0.0,
            "n_error": 0,
            "n_metal_coordinated": 0,
            "metal_coordination_rate": 0.0,
            "mean_max_tanimoto_tmQM": 0.0,
        }
    n_closure = sum(1 for r in records if r["closure_ok"])
    n_ood = sum(1 for r in records if r["ood_pass"])
    n_err = sum(1 for r in records if r["status"] == "error")
    n_metal = sum(int(r.get("n_metal_coordinated", 0)) for r in records)
    ok_records = [r for r in records if r["status"] == "ok"]
    if ok_records:
        walls = [float(r["wall_time_s"]) for r in ok_records]
        mems = [float(r["memory_mb"]) for r in ok_records]
        mean_wall = float(statistics.fmean(walls))
        std_wall = float(statistics.stdev(walls)) if len(walls) >= 2 else 0.0
        median_wall = float(statistics.median(walls))
        peak_mem = float(max(mems)) if mems else 0.0
        total_wall = float(sum(walls))
    else:
        mean_wall = std_wall = median_wall = peak_mem = total_wall = 0.0
    return {
        "n_samples": int(n),
        "n_ood_pass": int(n_ood),
        "n_closure_ok": int(n_closure),
        "reachability_rate": _safe_div(n_closure, n),
        "ood_pass_rate": _safe_div(n_ood, n),
        "mean_wall_time": mean_wall,
        "std_wall_time": std_wall,
        "median_wall_time": median_wall,
        "peak_memory_mb": peak_mem,
        "total_wall_time": total_wall,
        "n_error": int(n_err),
        "n_metal_coordinated": int(n_metal),
        "metal_coordination_rate": _safe_div(n_metal, n),
        "mean_max_tanimoto_tmQM": float(
            statistics.fmean([r["max_tanimoto_tmQM"] for r in records])
        ) if records else 0.0,
    }


# ---------------------------------------------------------------------------
# Cross-element correlation (Pearson r over per-seed reachability)
# ---------------------------------------------------------------------------
def _cross_element_correlation(
    per_seed: Dict[str, Dict[int, Dict[str, float]]],
    elements: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute the |elements| × |elements| Pearson correlation matrix
    over per-element reachability rate indexed by seed.
    """
    seed_index: Dict[int, Dict[str, float]] = defaultdict(dict)
    for elem in elements:
        seed_block = per_seed.get(elem, {})
        for s, agg in seed_block.items():
            seed_index[int(s)][elem] = float(
                agg.get("reachability_rate", 0.0)
            )
    matrix: Dict[str, Dict[str, float]] = {}
    elems = list(elements)
    for i, ea in enumerate(elems):
        matrix[ea] = {}
        for j, eb in enumerate(elems):
            if i == j:
                matrix[ea][eb] = 1.0
                continue
            xs = [
                seed_index[s][ea] for s in seed_index
                if ea in seed_index[s] and eb in seed_index[s]
            ]
            ys = [
                seed_index[s][eb] for s in seed_index
                if ea in seed_index[s] and eb in seed_index[s]
            ]
            if len(xs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
                try:
                    matrix[ea][eb] = float(statistics.correlation(xs, ys))
                except Exception:
                    matrix[ea][eb] = 0.0
            else:
                matrix[ea][eb] = 0.0
    return matrix


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_ood_per_element_benchmark(
    n_samples: int = DEFAULT_N_SAMPLES,
    *,
    elements: Sequence[str] = DBLOCK_ELEMENTS,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    tmqm_csv: Path = DEFAULT_TMQM_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_products: int = DEFAULT_MAX_PRODUCTS,
    seeds: int = DEFAULT_SEEDS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Dict[str, Any]:
    """Run per-element OOD reachability on tmQM d-block."""
    log.info(
        "[wf73_ood_per_element] start: elements=%s, n_per_element=%d, "
        "seeds=%d, threshold=%.2f, max_depth=%d",
        list(elements), int(n_samples), int(seeds),
        float(tanimoto_threshold), int(max_depth),
    )

    click_rules = _load_click_rules()
    partner_terms = _load_partner_terms()
    click_rule_names = [
        getattr(r, "name", type(r).__name__) for r in click_rules
    ]
    ood_protocol = _load_tanimoto_protocol(threshold=tanimoto_threshold)

    pool = _load_tmqm_pool(Path(tmqm_csv), elements=elements)
    if not pool:
        log.error("[wf73_ood_per_element] empty pool — aborting.")
        return {
            "schema_version": 1,
            "tool": "bench_ood_per_element.py",
            "pilot_id": "wf73_ood_per_element",
            "date_utc": _utcnow_iso(),
            "method": "closure+ood_filter",
            "tanimoto_threshold": float(tanimoto_threshold),
            "elements": list(elements),
            "n_samples_per_element_per_seed": int(n_samples),
            "max_depth": int(max_depth),
            "max_products": int(max_products),
            "seeds": int(seeds),
            "click_rule_names": click_rule_names,
            "tmqm_csv": str(tmqm_csv),
            "per_element": {},
            "per_seed": {},
            "per_element_pool_size": {},
            "error": "empty_pool",
        }

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "bench_ood_per_element.py",
        "pilot_id": "wf73_ood_per_element",
        "date_utc": _utcnow_iso(),
        "method": "closure+ood_filter",
        "tanimoto_threshold": float(tanimoto_threshold),
        "elements": list(elements),
        "n_samples_per_element_per_seed": int(n_samples),
        "max_depth": int(max_depth),
        "max_products": int(max_products),
        "seeds": int(seeds),
        "click_rule_names": click_rule_names,
        "tmqm_csv": str(tmqm_csv),
        "per_element": {},
        "per_seed": {},
        "per_element_pool_size": {},
    }

    t_total_0 = time.perf_counter()
    n_error_total = 0

    for element in elements:
        rows = pool.get(element, [])
        payload["per_element_pool_size"][element] = int(len(rows))
        if not rows:
            log.warning(
                "[wf73_ood_per_element] no rows for element=%s; skipping.",
                element,
            )
            payload["per_element"][element] = {
                "aggregate": _aggregate_per_metal_skeleton(),
                "per_seed": {},
                "n_records": 0,
                "seeds_used": 0,
                "skipped_reason": "no_rows",
            }
            continue
        train, _test = _split_train_test(rows, train_ratio=train_ratio)
        train_pool = [r["smiles"] for r in train]

        element_records_all: List[Dict[str, Any]] = []
        seed_metrics: Dict[int, Dict[str, Any]] = {}
        for seed in range(int(seeds)):
            samples = _load_element_samples(
                pool, element, int(n_samples),
                train_ratio=train_ratio, seed_offset=int(seed * n_samples),
            )
            if not samples:
                log.warning(
                    "[wf73_ood_per_element] no test samples for "
                    "element=%s seed=%d",
                    element, seed,
                )
                continue
            log.info(
                "[wf73_ood_per_element] element=%s seed=%d n_loaded=%d "
                "(train_pool=%d)",
                element, seed, len(samples), len(train_pool),
            )
            seed_recs: List[Dict[str, Any]] = []
            for i, row in enumerate(samples):
                rec = _benchmark_one(
                    smiles=row["smiles"],
                    idx=int(seed * n_samples + i),
                    element=element,
                    click_rules=click_rules,
                    partner_terms=partner_terms,
                    ood_protocol=ood_protocol,
                    train_pool=train_pool,
                    max_depth=int(max_depth),
                    max_products=int(max_products),
                )
                seed_recs.append(rec)
            seed_metrics[seed] = _aggregate_per_element(seed_recs)
            element_records_all.extend(seed_recs)

        per_element = _aggregate_per_element(element_records_all)
        n_error_total += int(per_element.get("n_error", 0))
        gate_pass = (
            float(per_element.get("reachability_rate", 0.0))
            >= float(PER_ELEMENT_GATE)
        )
        payload["per_element"][element] = {
            "aggregate": per_element,
            "per_seed": seed_metrics,
            "n_records": int(len(element_records_all)),
            "seeds_used": int(len(seed_metrics)),
            "block": ELEMENT_BLOCK.get(element, "?"),
            "atomic_number": int(ELEMENT_Z.get(element, -1)),
            "gate_pass": bool(gate_pass),
            "gate_threshold": float(PER_ELEMENT_GATE),
        }

    # Cross-element correlation.
    per_seed_index: Dict[str, Dict[int, Dict[str, float]]] = {}
    for elem in elements:
        seed_block = payload["per_element"].get(elem, {}).get("per_seed", {})
        per_seed_index[elem] = {int(s): dict(agg) for s, agg in seed_block.items()}
    payload["cross_element_correlation"] = _cross_element_correlation(
        per_seed_index, elements,
    )

    # Honest per-element gate verdict.
    gate_rates = [
        float(payload["per_element"][e]["aggregate"].get("reachability_rate", 0.0))
        for e in elements
        if "aggregate" in payload["per_element"].get(e, {})
        and payload["per_element"][e]["aggregate"].get("n_samples", 0) > 0
    ]
    n_gate_pass = sum(1 for r in gate_rates if r >= float(PER_ELEMENT_GATE))
    payload["per_element_gate"] = {
        "threshold": float(PER_ELEMENT_GATE),
        "n_elements_with_data": int(len(gate_rates)),
        "n_elements_pass": int(n_gate_pass),
        "pass_rate": _safe_div(n_gate_pass, len(gate_rates)),
        "verdict": (
            "PER_ELEMENT_GATE_MET"
            if n_gate_pass == len(gate_rates) and len(gate_rates) >= 1
            else "PER_ELEMENT_HONEST_PARTIAL"
        ),
    }

    payload["n_error_total"] = int(n_error_total)
    payload["total_wall_time"] = float(time.perf_counter() - t_total_0)
    return payload


def _aggregate_per_metal_skeleton() -> Dict[str, float]:
    """Empty per-element aggregate (zero-safe)."""
    return _aggregate_per_element([])


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------
def save_ood_per_element_benchmark(
    payload: Dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Persist per-element JSON + synthesis.md to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for element in payload.get("per_element", {}):
        element_dir = out / element
        element_dir.mkdir(parents=True, exist_ok=True)
        with (element_dir / "ood_reachability.json").open("w") as fh:
            json.dump(
                payload["per_element"][element],
                fh, indent=2, sort_keys=True,
            )
    with (out / "wf73_ood_per_element_full.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    log.info("[wf73_ood_per_element] saved per-element JSON to %s", out)


# ---------------------------------------------------------------------------
# Synthesis markdown writer
# ---------------------------------------------------------------------------
def write_synthesis(
    payload: Dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write a per-element synthesis.md summarising the run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "synthesis.md"

    elements = list(payload.get("elements", []))
    per_element = payload.get("per_element", {})
    cross_corr = payload.get("cross_element_correlation", {})
    gate = payload.get("per_element_gate", {})

    lines: List[str] = []
    lines.append("# Wf-73 Phase 3 — per-element OOD reachability on tmQM d-block")
    lines.append("")
    lines.append(f"**Date UTC**: {payload.get('date_utc', '?')}  ")
    lines.append(f"**Tool**: `{payload.get('tool', '?')}`  ")
    lines.append(f"**Method**: {payload.get('method', '?')}  ")
    lines.append(
        f"**Tanimoto threshold**: {payload.get('tanimoto_threshold', 0.0)}  "
    )
    lines.append(
        f"**n_samples_per_element_per_seed**: "
        f"{payload.get('n_samples_per_element_per_seed', 0)}  "
    )
    lines.append(
        f"**seeds**: {payload.get('seeds', 0)} | "
        f"**max_depth**: {payload.get('max_depth', 0)} | "
        f"**max_products**: {payload.get('max_products', 0)}"
    )
    lines.append("")
    lines.append("## §1 Per-element reachability (gate verdict)")
    lines.append("")
    n_pass = int(gate.get("n_elements_pass", 0))
    n_total = int(gate.get("n_elements_with_data", 0))
    threshold = float(gate.get("threshold", PER_ELEMENT_GATE))
    verdict = str(gate.get("verdict", "?"))
    lines.append(
        f"**Gate**: per-element reachability ≥ {threshold:.2f}.  "
        f"**Verdict**: `{verdict}`.  "
        f"**Pass**: {n_pass} / {n_total} elements."
    )
    lines.append("")

    lines.append("## §2 Per-element table (30 d-block elements)")
    lines.append("")
    lines.append("| Element | Block | n | reachability | OOD pass | metal_coord | well_typed | gate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|:---:|")
    for elem in elements:
        blk = per_element.get(elem, {})
        agg = blk.get("aggregate", {}) if blk else {}
        n = int(agg.get("n_samples", 0))
        rr = float(agg.get("reachability_rate", 0.0))
        ood = float(agg.get("ood_pass_rate", 0.0))
        mc = float(agg.get("metal_coordination_rate", 0.0))
        wt = float(agg.get("mean_max_tanimoto_tmQM", 0.0))
        block = ELEMENT_BLOCK.get(elem, "?")
        gate_pass = bool(blk.get("gate_pass", False))
        gate_mark = "PASS" if gate_pass else "FAIL"
        if n == 0:
            lines.append(
                f"| {elem} | {block} | 0 | — | — | — | — | (no data) |"
            )
        else:
            lines.append(
                f"| {elem} | {block} | {n} | {rr:.4f} | "
                f"{ood:.4f} | {mc:.4f} | {wt:.4f} | {gate_mark} |"
            )
    lines.append("")

    # Honest framing note.
    lines.append("## §3 Honest per-element framing (party-twenty-spirit §3.5)")
    lines.append("")
    lines.append(
        "Per-element OOD reachability is reported even when it falls below "
        f"the {threshold:.2f} gate. We do not silently drop low-N or "
        "low-reachability elements. The verdict "
        f"`{verdict}` reflects the on-disk measurement."
    )
    lines.append("")
    lines.append(
        "Elements with **n_samples = 0** in the per-element table "
        "indicate the tmQM d-block pool has fewer than the seed-offset "
        "test split — those elements are flagged "
        "(`skipped_reason = no_rows`) and excluded from the gate verdict."
    )
    lines.append("")

    # Cross-element correlation summary.
    lines.append("## §4 Cross-element correlation (Pearson r)")
    lines.append("")
    lines.append(
        "30 × 30 Pearson correlation matrix over per-element reachability "
        "rate indexed by seed. Diagonals = 1.0. Off-diagonals are 0.0 when "
        "variance is degenerate (low-N seeds yield identical rates)."
    )
    lines.append("")
    elems = list(elements)
    head = "| | " + " | ".join(elems) + " |"
    lines.append(head)
    sep = "|---|" + "---:|" * len(elems)
    lines.append(sep)
    for ea in elems:
        row_vals: List[str] = [ea]
        for eb in elems:
            v = float(cross_corr.get(ea, {}).get(eb, 0.0))
            row_vals.append(f"{v:+.2f}" if v != 0.0 else "0.00")
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("")

    # Top off-diagonal correlations.
    pairs: List[Tuple[float, str, str]] = []
    for i, ea in enumerate(elems):
        for j, eb in enumerate(elems):
            if i >= j:
                continue
            r = float(cross_corr.get(ea, {}).get(eb, 0.0))
            pairs.append((r, ea, eb))
    pairs.sort(key=lambda x: x[0], reverse=True)
    lines.append("### Top 5 positive off-diagonal correlations")
    lines.append("")
    lines.append("| r | element_a | element_b |")
    lines.append("|---:|---|---|")
    for r, ea, eb in pairs[:5]:
        lines.append(f"| {r:+.3f} | {ea} | {eb} |")
    lines.append("")
    lines.append("### Top 5 negative off-diagonal correlations")
    lines.append("")
    lines.append("| r | element_a | element_b |")
    lines.append("|---:|---|---|")
    for r, ea, eb in pairs[-5:]:
        lines.append(f"| {r:+.3f} | {ea} | {eb} |")
    lines.append("")

    # Wall-time + error budget.
    lines.append("## §5 Wall-time + error budget")
    lines.append("")
    lines.append(f"- **total_wall_time** = {payload.get('total_wall_time', 0.0):.2f} s")
    lines.append(f"- **n_error_total** = {payload.get('n_error_total', 0)}")
    pool_size = payload.get("per_element_pool_size", {})
    if pool_size:
        total_pool = sum(int(v) for v in pool_size.values())
        lines.append(f"- **tmqm_pool_size_total** = {total_pool}")
    lines.append("")

    lines.append("## §6 Cross-references")
    lines.append("")
    lines.append(
        "- `TODO/completed/98_wf44_closure_ood_phase2.md` — Wf-57 Wf-44 P2 "
        "baseline (Tanimoto<0.4 0.9978; Catherine 6/6 on 25 metals)"
    )
    lines.append(
        "- `TODO/completed/101_wf73_p2_ood_permetal.md` — Wf-73 P2 "
        "(MetalCytoToxDB 5-metal per-metal)"
    )
    lines.append(
        "- `molmetal/scripts/bench_ood_per_metal.py` — Wf-73 P2 "
        "per-metal script"
    )
    lines.append(
        "- `molmetal/scripts/bench_ood_tanimoto_protocol.py` — Wf-57 P1 "
        "Tanimoto<0.4 protocol script"
    )
    lines.append("")

    md.write_text("\n".join(lines))
    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-73 Phase 3 — per-element OOD reachability on "
            "tmQM d-block (CPU + GPU-0 Blackwell)"
        ),
    )
    parser.add_argument(
        "--elements", nargs="+", default=list(DBLOCK_ELEMENTS),
        help="Subset of d-block elements to run (default: all 30).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"SMILES per element per seed (default {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--tanimoto-thresh", type=float, default=DEFAULT_TANIMOTO_THRESHOLD,
        help=f"OOD threshold (default {DEFAULT_TANIMOTO_THRESHOLD}).",
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help=f"Number of disjoint SMILES slices per element (default {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
    )
    parser.add_argument(
        "--max-products", type=int, default=DEFAULT_MAX_PRODUCTS,
    )
    parser.add_argument(
        "--tmqm-csv", type=Path, default=DEFAULT_TMQM_CSV,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    payload = run_ood_per_element_benchmark(
        n_samples=int(args.n_samples),
        elements=tuple(args.elements),
        tanimoto_threshold=float(args.tanimoto_thresh),
        tmqm_csv=args.tmqm_csv,
        output_dir=args.output_dir,
        max_depth=int(args.max_depth),
        max_products=int(args.max_products),
        seeds=int(args.seeds),
    )
    save_ood_per_element_benchmark(payload, output_dir=args.output_dir)
    write_synthesis(payload, output_dir=args.output_dir)

    print(json.dumps(
        {k: payload[k] for k in (
            "tool", "date_utc", "elements", "seeds",
            "n_samples_per_element_per_seed", "per_element_gate",
        )},
        indent=2, sort_keys=True,
    ))
    for elem, blk in payload.get("per_element", {}).items():
        agg = blk.get("aggregate", {})
        if int(agg.get("n_samples", 0)) == 0:
            continue
        print(
            f"  element={elem:>3}: n={agg['n_samples']:4d}  "
            f"reachability={agg['reachability_rate']:.4f}  "
            f"ood_pass={agg['ood_pass_rate']:.4f}  "
            f"metal_coord={agg['metal_coordination_rate']:.4f}  "
            f"gate={'PASS' if blk.get('gate_pass') else 'FAIL'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())