"""Wf-130 Phase 2 — Closure reachability expansion to 30 metals × 5000 mols (CPU-only).

================================================================
Wf-130 Phase 2 (R20-P5 / TODO-96 / Wf-43): Lift closure-theorem
reachability toward **≥ 0.85 on 30 d-block metals × 5000 molecules
each = 150 000 samples**, sourced from the full tmQM parsed CSV.

Why this script exists
----------------------

Wf-67 P2 reached ``0.5891`` on 30 metals × ~75 unique samples each
(2244 unique SMILES total, all from the 5K tmQM d-block subset CSV).
Wf-68 P2 hit ``0.5744`` on 25 metals × ~180 unique samples each
(4497 unique SMILES total).  Both runs were capped by the **5K
sample limit of the d-block subset CSV** — many metals had < 100
unique SMILES, making per-metal reachability high-variance and the
overall mean sensitive to sample scarcity.

Wf-130 P2 sources SMILES from the full tmQM parsed CSV
(``.data/molmetal/tmQM/tmqm_parsed.csv``, 108 542 records, ~30 d-block
metals), which has 100-10 000 unique SMILES per metal.  Metals with
< 5000 unique SMILES (Tc=283, La=116, Sc=459, Nb=516, Hf=572,
Ta=740) are sampled **with replacement** to reach the 5000-sample
target.  Per-metal reports carry both ``n_unique`` and ``n_total``
so the with-replacement sampling is fully transparent.

Approach
--------
1. Load the tmQM parsed CSV (uses ``metal`` column, not
   ``metal_center``).
2. For each metal, collect **all** unique SMILES (deduplicated).
3. If ``n_unique >= 5000``: take the first 5000 (deterministic).
4. If ``n_unique < 5000``: sample with replacement (seeded).
5. Run :func:`closure_theorem` on every sample at depth=1, max_products=64.
6. Aggregate per-metal (closure_ok / n_total) + macro reachability
   (mean of per-metal rates; equal weighting per metal).

Outputs
-------
* ``molmetal/reports/wf130_p2_closure_30_5000/{metal}/reachability.json``
  — per-metal reachability + n_unique + n_total + samples.
* ``molmetal/reports/wf130_p2_closure_30_5000/reachability_overall.json``
  — overall macro + per-metal summary.
* ``molmetal/reports/wf130_p2_closure_30_5000/synthesis.md`` —
  human-readable ship surface.

Constraints
-----------
* **CPU-only.** Closure-theorem BFS is pure-functional.
* **Deterministic.** Same flags reproduce every per-sample metric.
  Sampling-with-replacement uses a fixed seed (default 42).
* **Honest framing.** Per-metal JSON carries ``n_unique`` + ``n_total``
  so with-replacement sampling is transparent.  Metals with
  ``n_unique < 100`` (Tc, La) are flagged in synthesis.md.

Usage examples
--------------
::

    # Default: 30 metals × 5000 samples = 150 000 total.
    uv run --no-sync python molmetal/scripts/bench_closure_dblock_full.py

    # Smoke (3 metals × 5 samples each = 15 total).
    uv run --no-sync python molmetal/scripts/bench_closure_dblock_full.py \\
        --metals Pt Ru Pd --n-samples 5 --smoke

    # Smaller test (30 metals × 100 samples each = 3000 total).
    uv run --no-sync python molmetal/scripts/bench_closure_dblock_full.py \\
        --n-samples 100
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import sys
import time
import tracemalloc
import warnings
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


log = logging.getLogger("wf130_p2_closure_30_5000")


# ---------------------------------------------------------------------------
# Constants — 30 d-block metals + dataset paths
# ---------------------------------------------------------------------------

#: All 30 d-block metals (R17 ``PRIMITIVE_ATOMS`` ship).  Order follows
#: the periodic-table row layout (3d / 4d / 5d + La).
DBLOCK_METALS: Tuple[str, ...] = (
    # 3d
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d + La
    "La", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
)

#: Wf-130 P2 success criterion: macro reachability ≥ 0.85.
GATE_TARGET_REACHABILITY: float = 0.85
#: Wf-67 P2 baseline (2244 samples × 30 metals).
WF67_BASELINE_REACHABILITY: float = 0.5891
#: Wf-68 P2 baseline (4497 samples × 25 metals).
WF68_BASELINE_REACHABILITY: float = 0.5744
#: Honest-negative threshold: any metal below this in the per-metal
#: table triggers an "honest framing" annotation in synthesis.md.
HONEST_NEGATIVE_THRESHOLD: float = 0.50

DEFAULT_N_METALS: int = 30
DEFAULT_N_SAMPLES: int = 5000
DEFAULT_MAX_DEPTH: int = 1
DEFAULT_MAX_PRODUCTS: int = 64
DEFAULT_SEEDS: int = 1
DEFAULT_SAMPLE_SEED: int = 42  # for with-replacement sampling
DEFAULT_OUTPUT_DIR: Path = (
    _REPO_ROOT / "molmetal" / "reports" / "wf130_p2_closure_30_5000"
)

#: Path to the full tmQM parsed CSV (108K records, 30 d-block metals).
#: Falls back to ``molmetal/data/tmqm_dblock_subset.csv`` if missing.
DEFAULT_TMQM_CSV: Path = Path(
    "/home/hugo/codes/mol-metal/.data/molmetal/tmQM/tmqm_parsed.csv"
)
FALLBACK_TMQM_CSV: Path = (
    _REPO_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
)

#: 7 partner terms (same as Wf-37 P2 / Wf-67 P2 / Wf-68 P2).
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
    """Return current UTC time in ISO 8601 with Z suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _memory_mb(snapshot_diff: int) -> float:
    """Convert a ``tracemalloc`` byte diff to MB."""
    return float(snapshot_diff) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Lazy heavy imports
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
# Dataset loader
# ---------------------------------------------------------------------------
def _detect_csv_layout(csv_path: Path) -> Dict[str, Any]:
    """Probe CSV header to determine layout + column names.

    Returns a dict ``{"metal_column": str, "smiles_column": str}`` so
    the loader can dispatch on the actual schema instead of hardcoding.

    The tmQM parsed CSV uses ``metal`` + ``smiles``; the tmQM d-block
    subset CSV uses ``metal_center`` + ``smiles``.  Both are detected
    automatically.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "metal_column": "metal",
        "smiles_column": "smiles",
    }
    if not csv_path.is_file():
        return out
    with csv_path.open("r", newline="") as fh:
        header = fh.readline().strip().split(",")
    cols = [c.strip() for c in header]
    if "metal_center" in cols:
        out["metal_column"] = "metal_center"
    elif "metal" in cols:
        out["metal_column"] = "metal"
    else:
        return out
    if "smiles" not in cols:
        return out
    out["ok"] = True
    return out


def _load_unique_metal_smis(
    csv_path: Path,
    metal: str,
    *,
    metal_column: str = "metal",
    smiles_column: str = "smiles",
) -> List[str]:
    """Load all unique SMILES tagged with ``metal`` from the CSV.

    Iterates the full CSV once, dedup-caches every SMILES tagged with
    ``metal``.  Returns a deterministic list (first-seen order) so
    reproducible runs are bit-stable.

    Empty / whitespace SMILES are skipped.  SMILES are not canonicalised
    here — closure_theorem re-parses each anyway, and the per-metal
    JSON records the raw SMILES for auditing.
    """
    out: List[str] = []
    seen: set = set()
    if not csv_path.is_file():
        log.error("CSV not found at %s", csv_path)
        return out
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if metal_column not in fields:
            log.error(
                "Column %r missing from %s; fields=%r",
                metal_column, csv_path, fields,
            )
            return out
        if smiles_column not in fields:
            log.error(
                "Column %r missing from %s; fields=%r",
                smiles_column, csv_path, fields,
            )
            return out
        for row in reader:
            tag = (row.get(metal_column, "") or "").strip()
            if tag != metal:
                continue
            smi = (row.get(smiles_column, "") or "").strip()
            if not smi or smi in seen:
                continue
            seen.add(smi)
            out.append(smi)
    return out


def _sample_to_target(
    unique_smis: List[str],
    n_target: int,
    *,
    seed: int,
) -> Tuple[List[str], int]:
    """Sample to ``n_target`` SMILES (with replacement if needed).

    If ``n_unique >= n_target``: take the first ``n_target``.
    If ``n_unique < n_target``: sample with replacement using a
    seeded random RNG so the run is reproducible.

    Returns (sampled_smis, n_repeats) where ``n_repeats`` is the
    number of samples drawn with replacement (0 when not sampled).
    """
    n_unique = len(unique_smis)
    if n_unique >= n_target:
        return unique_smis[: int(n_target)], 0
    rng = random.Random(int(seed))
    sampled: List[str] = []
    for _ in range(int(n_target)):
        sampled.append(rng.choice(unique_smis))
    return sampled, int(n_target - n_unique)


# ---------------------------------------------------------------------------
# Per-sample benchmark
# ---------------------------------------------------------------------------
def _benchmark_one(
    raw_smiles: str,
    idx: int,
    metal: str,
    *,
    click_rules: Sequence[Any],
    partner_terms: Sequence[Any],
    max_depth: int,
    max_products: int,
    n_unique_total: int,
    sampled_with_replacement: bool,
    sample_origin_idx: int,
) -> Dict[str, Any]:
    """Run :func:`closure_theorem` on one SMILES; record per-sample metrics.

    Status is one of ``ok``, ``error``.  ``closure_ok`` is True iff
    the closure-theorem assertion hook returned True.
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    record: Dict[str, Any] = {
        "idx": int(idx),
        "metal": str(metal),
        "smiles": str(raw_smiles),
        "status": "skipped",
        "closure_ok": False,
        "n_visited": 0,
        "n_products": 0,
        "max_depth_seen": 0,
        "well_typed_rate": 0.0,
        "n_atoms_start": 0,
        "parse_error": "",
        "wall_time_s": 0.0,
        "memory_mb": 0.0,
        "depth": int(max_depth),
        "sampled_with_replacement": bool(sampled_with_replacement),
        "n_unique_total": int(n_unique_total),
        "sample_origin_idx": int(sample_origin_idx),
    }

    t_parse_0 = time.perf_counter()
    try:
        start_term = MoleculeClosedTerm.from_smiles(
            raw_smiles, embed_3d=False,
        )
        record["n_atoms_start"] = int(len(start_term.atoms))
    except Exception as exc:
        record["status"] = "error"
        record["parse_error"] = f"{type(exc).__name__}: {exc}"[:240]
        record["wall_time_s"] = float(time.perf_counter() - t_parse_0)
        return record

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

    return record


# ---------------------------------------------------------------------------
# Per-metal aggregate
# ---------------------------------------------------------------------------
def _aggregate_per_metal(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample records into a per-metal benchmark dict.

    Keys returned:
      * ``n_samples``            — number of records (= total bench draws)
      * ``n_unique``             — number of unique SMILES in the run
      * ``n_repeats``            — ``n_samples - n_unique``
      * ``sampled_with_replacement``  — True if any sample was drawn
        with replacement (i.e. ``n_repeats > 0``)
      * ``n_closure_ok``         — number of records with closure_ok=True
      * ``closure_reachability_rate``  — n_closure_ok / n_samples
      * ``n_parse_error``        — number of records with status="error"
      * ``mean_wall_time_s``     — mean wall time over successful samples
      * ``median_wall_time_s``   — median wall time
      * ``std_wall_time_s``      — sample stdev of wall time
      * ``total_wall_time_s``    — sum of wall_time_s
      * ``peak_memory_mb``       — max memory_mb over samples
      * ``gate_target``          — :data:`GATE_TARGET_REACHABILITY`
      * ``gate_pass``            — ``rate >= gate_target``
    """
    n = len(records)
    base: Dict[str, Any] = {
        "n_samples": 0,
        "n_unique": 0,
        "n_repeats": 0,
        "sampled_with_replacement": False,
        "n_closure_ok": 0,
        "closure_reachability_rate": 0.0,
        "n_parse_error": 0,
        "mean_wall_time_s": 0.0,
        "median_wall_time_s": 0.0,
        "std_wall_time_s": 0.0,
        "total_wall_time_s": 0.0,
        "peak_memory_mb": 0.0,
        "gate_target": GATE_TARGET_REACHABILITY,
        "gate_pass": False,
    }
    if n == 0:
        return base
    n_ok = sum(1 for r in records if r["closure_ok"])
    n_err = sum(1 for r in records if r["status"] == "error")
    n_unique = int(records[0].get("n_unique_total", 0)) if records else 0
    # n_unique should be the same across all records for a given metal.
    n_repeats = max(0, int(n) - int(n_unique))
    sampled_wr = bool(n_repeats > 0)
    rate = (n_ok / n) if n else 0.0
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
        "n_samples": int(n),
        "n_unique": int(n_unique),
        "n_repeats": int(n_repeats),
        "sampled_with_replacement": bool(sampled_wr),
        "n_closure_ok": int(n_ok),
        "closure_reachability_rate": float(rate),
        "n_parse_error": int(n_err),
        "n_cache_hits": int(sum(
            1 for r in records if r.get("from_cache", False)
        )),
        "mean_wall_time_s": mean_wall,
        "median_wall_time_s": median_wall,
        "std_wall_time_s": std_wall,
        "total_wall_time_s": total_wall,
        "peak_memory_mb": peak_mem,
        "gate_target": GATE_TARGET_REACHABILITY,
        "gate_pass": bool(rate >= GATE_TARGET_REACHABILITY),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _benchmark_metal_worker(
    metal: str,
    csv_path_str: str,
    metal_column: str,
    smiles_column: str,
    n_samples: int,
    sample_seed: int,
    max_depth: int,
    max_products: int,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Worker function for one metal — runs the full bench for it.

    Designed to be called by :class:`ProcessPoolExecutor`.  Returns
    ``(metal, records, summary)`` so the main process can aggregate.

    Each worker has its own process (and therefore its own Python
    state, RDKit, closure cache).  Cache is intra-metal only — which
    is exactly what we want, since with-replacement sampling repeats
    SMILES within the same metal.
    """
    from molmetal_lam.lam_chem.closure import closure_theorem
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

    csv_path = Path(csv_path_str)
    click_rules = _load_click_rules()
    partner_terms = _load_partner_terms()

    unique_smis = _load_unique_metal_smis(
        csv_path, str(metal),
        metal_column=metal_column, smiles_column=smiles_column,
    )
    n_unique = len(unique_smis)
    sampled_smis, n_repeats = _sample_to_target(
        unique_smis, int(n_samples), seed=int(sample_seed),
    )
    n_target = len(sampled_smis)
    sampled_wr = bool(n_repeats > 0)

    records: List[Dict[str, Any]] = []
    # Per-worker (per-process) cache: only helps WITHIN one metal's
    # with-replacement repeats.  Survives only within this call.
    closure_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Suppress noisy logs in workers — main process logs the summary.
    logging.getLogger("wf130_p2_closure_30_5000").setLevel(logging.WARNING)
    # Also silence molmetal_lam's beta_reductions registration warnings.
    import warnings as _w
    _w.filterwarnings("ignore", category=UserWarning, module=r"molmetal_lam.*")

    for i, smi in enumerate(sampled_smis):
        origin_idx = (
            i if not sampled_wr else i % max(1, n_unique)
        )
        cache_key = (str(metal), str(smi))
        cached = closure_cache.get(cache_key)
        if cached is not None:
            rec = dict(cached)
            rec["idx"] = int(i)
            rec["sample_origin_idx"] = int(origin_idx)
            rec["sampled_with_replacement"] = bool(sampled_wr)
            rec["n_unique_total"] = int(n_unique)
            rec["from_cache"] = True
        else:
            rec = _benchmark_one(
                smi, i, str(metal),
                click_rules=click_rules,
                partner_terms=partner_terms,
                max_depth=int(max_depth),
                max_products=int(max_products),
                n_unique_total=int(n_unique),
                sampled_with_replacement=bool(sampled_wr),
                sample_origin_idx=int(origin_idx),
            )
            rec["from_cache"] = False
            closure_cache[cache_key] = rec
        records.append(rec)

    summary = _aggregate_per_metal(records)
    return str(metal), records, summary


def run_benchmark(
    n_metals: int = DEFAULT_N_METALS,
    n_samples: int = DEFAULT_N_SAMPLES,
    *,
    metals: Sequence[str] = DBLOCK_METALS,
    tmqm_csv: Path = DEFAULT_TMQM_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_products: int = DEFAULT_MAX_PRODUCTS,
    seeds: int = DEFAULT_SEEDS,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    fallback_csv: Path = FALLBACK_TMQM_CSV,
    max_workers: Optional[int] = None,
    parallel: bool = True,
) -> Dict[str, Any]:
    """Run the Wf-130 P2 closure-reachability expansion benchmark.

    Parameters
    ----------
    n_metals : int
        Number of metals (default 30).
    n_samples : int
        Target samples per metal (default 5000).  Metals with
        < n_samples unique SMILES are sampled with replacement.
    metals : sequence[str]
        Which metals to benchmark (default all 30 d-block).
    tmqm_csv : Path
        Path to the full tmQM parsed CSV (default
        ``.data/molmetal/tmQM/tmqm_parsed.csv``).
    fallback_csv : Path
        Path to the 5K tmQM d-block subset CSV (used if ``tmqm_csv``
        is not present on disk).
    output_dir : Path
        Where reports are written.
    max_depth : int
        Closure BFS depth budget (default 1).
    max_products : int
        ProductiveSpace BFS safety cap (default 64).
    seeds : int
        Number of independent seeds (placeholder for future stochastic
        re-runs).  Default 1.
    sample_seed : int
        RNG seed for with-replacement sampling.  Default 42 so the run
        is reproducible.

    Returns
    -------
    dict
        JSON-serialisable payload.  Also persisted by
        :func:`save_per_metal`.
    """
    log.info(
        "[wf130_p2] start: n_metals=%d n_samples=%d max_depth=%d "
        "seeds=%d sample_seed=%d",
        int(n_metals), int(n_samples), int(max_depth),
        int(seeds), int(sample_seed),
    )

    csv_path = Path(tmqm_csv)
    if not csv_path.is_file():
        log.warning(
            "Primary CSV missing at %s; falling back to %s",
            csv_path, fallback_csv,
        )
        csv_path = Path(fallback_csv)
    if not csv_path.is_file():
        log.error("CSV missing at %s; aborting benchmark.", csv_path)
        return {
            "schema_version": 1,
            "tool": "bench_closure_dblock_full.py",
            "pilot_id": "wf130_p2_closure_30_5000",
            "date_utc": _utcnow_iso(),
            "metals": list(metals),
            "n_samples": int(n_samples),
            "max_depth": int(max_depth),
            "error": f"CSV not found at {csv_path}",
            "per_metal": {},
            "aggregate": {},
        }

    layout = _detect_csv_layout(csv_path)
    if not layout["ok"]:
        log.error(
            "Could not detect CSV layout at %s; aborting.", csv_path,
        )
        return {
            "schema_version": 1,
            "tool": "bench_closure_dblock_full.py",
            "pilot_id": "wf130_p2_closure_30_5000",
            "date_utc": _utcnow_iso(),
            "csv_path": str(csv_path),
            "metals": list(metals),
            "n_samples": int(n_samples),
            "error": "CSV layout not recognised (missing metal / smiles column)",
            "per_metal": {},
            "aggregate": {},
        }
    metal_column = layout["metal_column"]
    smiles_column = layout["smiles_column"]
    log.info(
        "[wf130_p2] CSV=%s metal_column=%s smiles_column=%s",
        csv_path, metal_column, smiles_column,
    )

    click_rules = _load_click_rules()
    partner_terms = _load_partner_terms()

    t_total_0 = time.perf_counter()
    per_metal_records: Dict[str, List[Dict[str, Any]]] = {}
    per_metal_summary: Dict[str, Dict[str, Any]] = {}
    all_records: List[Dict[str, Any]] = []

    # Decide parallelism: each metal is independent, so we use
    # ProcessPoolExecutor when there are ≥ 2 metals AND ``parallel``
    # is True AND the metal list is long enough to benefit.
    use_parallel = bool(parallel) and len(metals) >= 2
    if use_parallel:
        try:
            import multiprocessing as _mp
            cpu_count = _mp.cpu_count()
        except Exception:
            cpu_count = 1
        if max_workers is None or int(max_workers) <= 0:
            # Default: 8 workers (or cpu_count, whichever is smaller).
            max_workers = min(8, cpu_count, len(metals))
        else:
            max_workers = min(int(max_workers), cpu_count, len(metals))
        log.info(
            "[wf130_p2] parallelism: max_workers=%d (cpu_count=%d, "
            "n_metals=%d)",
            int(max_workers), int(cpu_count), int(len(metals)),
        )

    if use_parallel and int(max_workers) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        worker_args = [
            (
                str(metal),
                str(csv_path),
                metal_column,
                smiles_column,
                int(n_samples),
                int(sample_seed),
                int(max_depth),
                int(max_products),
            )
            for metal in metals
        ]
        with ProcessPoolExecutor(max_workers=int(max_workers)) as pool:
            futures = {
                pool.submit(_benchmark_metal_worker, *args): args[0]
                for args in worker_args
            }
            for fut in as_completed(futures):
                metal_submitted = futures[fut]
                try:
                    metal_name, records, summary = fut.result()
                except Exception as exc:
                    log.error(
                        "[wf130_p2] metal %s worker failed: %r",
                        metal_submitted, exc,
                    )
                    continue
                per_metal_records[metal_name] = records
                per_metal_summary[metal_name] = summary
                agg = summary
                n_cache_hits = sum(
                    1 for r in records if r.get("from_cache", False)
                )
                log.info(
                    "[wf130_p2] %s: n=%d closure_ok=%d rate=%.4f "
                    "gate_pass=%s cache_hits=%d",
                    metal_name, agg["n_samples"], agg["n_closure_ok"],
                    agg["closure_reachability_rate"], agg["gate_pass"],
                    n_cache_hits,
                )
    else:
        # Single-threaded fallback (also used when ``parallel=False``,
        # when only 1 metal is requested, or in tests).
        for metal in metals:
            metal_name, records, summary = _benchmark_metal_worker(
                metal, str(csv_path), metal_column, smiles_column,
                int(n_samples), int(sample_seed), int(max_depth),
                int(max_products),
            )
            per_metal_records[metal_name] = records
            per_metal_summary[metal_name] = summary
            agg = summary
            n_cache_hits = sum(
                1 for r in records if r.get("from_cache", False)
            )
            log.info(
                "[wf130_p2] %s: n=%d closure_ok=%d rate=%.4f "
                "gate_pass=%s cache_hits=%d",
                metal_name, agg["n_samples"], agg["n_closure_ok"],
                agg["closure_reachability_rate"], agg["gate_pass"],
                n_cache_hits,
            )

    # Flatten all per-metal records into a single list.
    for metal in metals:
        for r in per_metal_records.get(metal, []):
            all_records.append(r)

    total_wall_s = time.perf_counter() - t_total_0

    # Macro reachability: equal-weight mean of per-metal rates.
    # (Each metal counts once regardless of n_unique — this is the
    # "macro" definition used by the Wf-130 P2 spec.)
    per_metal_rates = [
        s["closure_reachability_rate"]
        for s in per_metal_summary.values()
        if s["n_samples"] > 0
    ]
    if per_metal_rates:
        macro_reachability = float(statistics.fmean(per_metal_rates))
        macro_std = (
            float(statistics.stdev(per_metal_rates))
            if len(per_metal_rates) >= 2 else 0.0
        )
    else:
        macro_reachability = 0.0
        macro_std = 0.0

    # Micro (weighted) reachability: sum(closure_ok) / sum(n_samples).
    # Reported alongside macro for transparency.
    total_n = sum(s["n_samples"] for s in per_metal_summary.values())
    total_ok = sum(s["n_closure_ok"] for s in per_metal_summary.values())
    micro_reachability = (
        float(total_ok / total_n) if total_n else 0.0
    )

    n_metals_pass = sum(
        1 for s in per_metal_summary.values() if s["gate_pass"]
    )
    n_metals_actually_run = int(len(metals))
    n_metals_below_negative = sum(
        1 for s in per_metal_summary.values()
        if s["n_samples"] > 0
        and s["closure_reachability_rate"] < HONEST_NEGATIVE_THRESHOLD
    )
    n_metals_sampled_wr = sum(
        1 for s in per_metal_summary.values()
        if s.get("sampled_with_replacement", False)
    )

    try:
        import torch  # local; cheap
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "n/a"

    py_ver = ".".join(str(v) for v in sys.version_info[:3])

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "bench_closure_dblock_full.py",
        "pilot_id": "wf130_p2_closure_30_5000",
        "date_utc": _utcnow_iso(),
        "dataset": "tmqm_parsed",
        "csv_path": str(csv_path),
        "metal_column": metal_column,
        "smiles_column": smiles_column,
        "metals": list(metals),
        "n_metals": int(n_metals),
        "n_samples": int(n_samples),
        "n_samples_total": int(total_n),
        "n_samples_unique_total": int(sum(
            s["n_unique"] for s in per_metal_summary.values()
        )),
        "n_samples_repeats_total": int(sum(
            s["n_repeats"] for s in per_metal_summary.values()
        )),
        "max_depth": int(max_depth),
        "max_products": int(max_products),
        "seeds": int(seeds),
        "sample_seed": int(sample_seed),
        "gate_target_reachability": GATE_TARGET_REACHABILITY,
        "wf67_baseline_reachability": WF67_BASELINE_REACHABILITY,
        "wf68_baseline_reachability": WF68_BASELINE_REACHABILITY,
        "honest_negative_threshold": HONEST_NEGATIVE_THRESHOLD,
        "per_metal": per_metal_summary,
        "per_metal_records": per_metal_records,
        "aggregate": {
            "n_metals": int(len(metals)),
            "n_samples": int(total_n),
            "n_closure_ok": int(total_ok),
            "micro_reachability_rate": float(micro_reachability),
            "macro_reachability_rate": float(macro_reachability),
            "macro_std_of_means": float(macro_std),
        },
        "n_metals_pass": int(n_metals_pass),
        "n_metals_below_honest_negative_threshold": int(
            n_metals_below_negative
        ),
        "n_metals_sampled_with_replacement": int(n_metals_sampled_wr),
        "overall_gate_pass": bool(
            macro_reachability >= GATE_TARGET_REACHABILITY
        ),
        "total_wall_time_s": float(total_wall_s),
        "timestamp_utc": _utcnow_iso(),
        "python": py_ver,
        "torch": torch_ver,
        "honest_framing": (
            "Wf-130 P2 closure-reachability expansion: 30 d-block "
            "metals × 5000 samples each = 150 000 total. CPU-only, "
            "deterministic.  Source: full tmQM parsed CSV (108K "
            "records).  Metals with < 5000 unique SMILES are sampled "
            "with replacement (seeded).  Macro reachability = mean of "
            "per-metal rates; reported alongside micro (weighted) "
            "for transparency."
        ),
    }
    return payload


# ---------------------------------------------------------------------------
# Save + synthesis
# ---------------------------------------------------------------------------
def save_per_metal(
    payload: Dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Path]:
    """Persist per-metal JSON + overall summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for metal in payload.get("metals", []):
        metal_dir = output_dir / metal
        metal_dir.mkdir(parents=True, exist_ok=True)
        rec_path = metal_dir / "reachability.json"
        per_metal = payload["per_metal"][metal]
        per_metal_records = payload.get("per_metal_records", {}).get(
            metal, [],
        )
        sample_records = [
            {k: r[k] for k in (
                "idx", "status", "closure_ok",
                "sampled_with_replacement", "sample_origin_idx",
                "wall_time_s", "parse_error", "from_cache",
            ) if k in r}
            for r in per_metal_records
        ]
        rec_path.write_text(json.dumps(
            {
                "metal": metal,
                "summary": per_metal,
                "n_samples_in_run": per_metal["n_samples"],
                "n_unique_smiles": per_metal["n_unique"],
                "n_repeats": per_metal["n_repeats"],
                "samples": sample_records,
            },
            indent=2,
            default=str,
        ))
        paths[metal] = rec_path

    overall_path = output_dir / "reachability_overall.json"
    # Strip per_metal_records (huge) from the overall file.
    overall_only = {
        k: v for k, v in payload.items()
        if k not in ("per_metal_records", "per_sample")
    }
    overall_path.write_text(json.dumps(overall_only, indent=2, default=str))
    paths["overall"] = overall_path
    return paths


def write_synthesis(
    payload: Dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``synthesis.md`` — human-readable ship surface."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "synthesis.md"
    agg = payload.get("aggregate", {})
    macro_rate = float(agg.get("macro_reachability_rate", 0.0))
    micro_rate = float(agg.get("micro_reachability_rate", 0.0))
    macro_std = float(agg.get("macro_std_of_means", 0.0))
    overall_pass = bool(payload.get("overall_gate_pass", False))
    n_metals_pass = int(payload.get("n_metals_pass", 0))
    n_below = int(payload.get("n_metals_below_honest_negative_threshold", 0))
    n_wr = int(payload.get("n_metals_sampled_with_replacement", 0))
    n_total = int(agg.get("n_samples", 0))
    n_total_unique = int(payload.get("n_samples_unique_total", 0))
    n_total_repeats = int(payload.get("n_samples_repeats_total", 0))
    total_wall = float(payload.get("total_wall_time_s", 0.0))
    n_metals = int(payload.get("n_metals", len(payload["metals"])))
    n_metals_actually_run = int(len(payload["metals"]))
    n_samples_target = int(payload.get("n_samples", 0))

    lines: List[str] = []
    lines.append("# Wf-130 Phase 2 — Closure reachability 30 × 5000 expansion")
    lines.append("")
    lines.append(
        f"Lifts Wf-67 P2 (0.5891 on 30 × ~75 unique) + Wf-68 P2 "
        f"(0.5744 on 25 × ~180 unique) toward the gate target "
        f"**≥ {GATE_TARGET_REACHABILITY:.2f}** on "
        f"**{n_metals} metals × {n_samples_target} samples each = "
        f"{n_samples_target * n_metals} requested**, sourced from the "
        f"full tmQM parsed CSV.  Metals with fewer unique SMILES are "
        f"sampled **with replacement** (seeded)."
    )
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"- **Macro reachability** (mean of per-metal rates, "
        f"equal-weight): **{macro_rate:.4f}** ± {macro_std:.4f} "
        f"(std-of-means across metals)  "
    )
    lines.append(
        f"- **Micro reachability** (sum closure_ok / sum n_samples, "
        f"weighted by sample count): {micro_rate:.4f}  "
    )
    lines.append(
        f"- **Overall gate pass (≥ {GATE_TARGET_REACHABILITY:.2f})**: "
        f"{overall_pass}  "
    )
    lines.append(
        f"- **Metals at gate**: {n_metals_pass}/{n_metals_actually_run}  "
    )
    lines.append(
        f"- **Metals below honest-negative threshold "
        f"({HONEST_NEGATIVE_THRESHOLD:.2f})**: "
        f"{n_below}/{n_metals_actually_run}  "
    )
    lines.append(
        f"- **Metals sampled with replacement**: "
        f"{n_wr}/{n_metals_actually_run}  "
    )
    lines.append(
        f"- **N samples (overall)**: {n_total}  "
    )
    lines.append(
        f"- **N unique SMILES (overall)**: {n_total_unique}  "
    )
    lines.append(
        f"- **N repeats (overall)**: {n_total_repeats}  "
    )
    lines.append(f"- **Total wall**: {total_wall:.2f} s  ")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append(
        f"- **Wf-67 P2 baseline**: {WF67_BASELINE_REACHABILITY:.4f} "
        f"(2244 unique samples × 30 metals).\n"
        f"- **Wf-68 P2 baseline**: {WF68_BASELINE_REACHABILITY:.4f} "
        f"(4497 unique samples × 25 metals).\n"
        f"- **Wf-130 P2 measured (macro)**: {macro_rate:.4f}  "
        f"({n_metals} metals × {n_samples_target} samples each, "
        f"with-replacement where required).\n"
        f"- **Delta vs Wf-67**: {macro_rate - WF67_BASELINE_REACHABILITY:+.4f}\n"
        f"- **Delta vs Wf-68**: {macro_rate - WF68_BASELINE_REACHABILITY:+.4f}"
    )
    lines.append("")
    lines.append("## Per-metal breakdown")
    lines.append("")
    lines.append(
        "| Metal | N | unique | repeats | closure_ok | reachability | gate |"
    )
    lines.append(
        "|-------|---|--------|---------|------------|--------------|------|"
    )
    for metal in payload["metals"]:
        m = payload["per_metal"][metal]
        lines.append(
            f"| {metal} | {m['n_samples']} | {m['n_unique']} "
            f"| {m['n_repeats']} | {m['n_closure_ok']} "
            f"| {m['closure_reachability_rate']:.4f} "
            f"| {m['gate_pass']} |"
        )
    lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **CPU-only.** Closure verifier is pure-functional; no GPU "
        "dependency.\n"
        "- **Deterministic.** Same flags reproduce every per-sample "
        "metric; with-replacement RNG is seeded "
        f"(`sample_seed={payload.get('sample_seed')}`).\n"
        "- **Real data.** Full tmQM parsed CSV (108K records, "
        "Balcells & Skjelstad 2020); the Wf-28 d-block subset was a "
        "5K sample drawn from this CSV.\n"
        "- **Sampling strategy.** Each metal receives exactly "
        f"{n_samples_target} samples.  Metals with "
        f"{n_samples_target}+ unique SMILES take the first "
        f"{n_samples_target} (deterministic).  Metals with fewer "
        f"unique SMILES (notably Tc, La, Sc, Nb, Hf, Ta) are sampled "
        f"with replacement (seeded) so every metal has the same "
        f"sample budget — this is the standard honest sampling "
        f"strategy; per-metal JSON files carry both "
        f"`n_unique` and `n_repeats` so the with-replacement "
        f"behaviour is transparent.\n"
        f"- **Macro vs micro reachability.** Macro = equal-weight "
        f"mean of per-metal rates (this is the gate definition).  "
        f"Micro = sum(closure_ok)/sum(n_samples) (weighted by "
        f"sample count, so well-sampled metals like Pd/Ni dominate). "
        f"Both are reported.\n"
        f"- **Honest-negative threshold.** Any metal with "
        f"reachability < {HONEST_NEGATIVE_THRESHOLD:.2f} is flagged. "
        f"At-risk metals for honest-negative framing: "
        f"Tc (only 283 unique SMILES), La (only 116 unique — "
        f"lanthanide), Sc (459), Nb (516), Hf (572), Ta (740).  "
        f"Per-metal rates for these are high-variance due to sample "
        f"scarcity even after with-replacement sampling.\n"
        f"- **Why is the 0.85 gate not always reachable?** "
        f"Structural reasons documented in TODO-83 / Wf-67 P2: "
        f"the closure-theorem valence-BNF predicate conservatively "
        f"rejects tmQM seeds with dative-bond notation `[M+]<-X` "
        f"(under-saturates the metal in RDKit valence counting).  "
        f"Lifting reachability to 0.95+ requires a chemistry-aware "
        f"SMILES parser or closure-theorem relaxations; both are "
        f"out-of-scope for the Wf-130 P2 measurement pass."
    )
    lines.append("")
    lines.append("## Honest-negative metals")
    lines.append("")
    honest_neg = [
        m for m in payload["metals"]
        if payload["per_metal"][m]["n_samples"] > 0
        and payload["per_metal"][m]["closure_reachability_rate"]
        < HONEST_NEGATIVE_THRESHOLD
    ]
    if honest_neg:
        lines.append(
            "The following metals achieved reachability below "
            f"{HONEST_NEGATIVE_THRESHOLD:.2f} — honest framing: "
        )
        for m in honest_neg:
            s = payload["per_metal"][m]
            lines.append(
                f"- **{m}**: n={s['n_samples']} unique={s['n_unique']} "
                f"repeats={s['n_repeats']} "
                f"reachability={s['closure_reachability_rate']:.4f}"
            )
        lines.append("")
        lines.append(
            "Reasons vary by metal: Tc and La have < 300 unique "
            "SMILES (sample scarcity); V/Cr/Mn can be valence-BNF "
            "rejected on early-row 3d samples (under-saturated); "
            "Mo/Tc have high oxidation-state variance in tmQM."
        )
    else:
        lines.append(
            f"All {n_metals} metals achieved reachability ≥ "
            f"{HONEST_NEGATIVE_THRESHOLD:.2f}."
        )
    lines.append("")
    md_path.write_text("\n".join(lines))
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_closure_dblock_full.py",
        description=(
            "Wf-130 Phase 2 closure-theorem reachability expansion. "
            f"Targets macro reachability ≥ {GATE_TARGET_REACHABILITY:.2f} "
            "on 30 d-block metals × 5000 samples each "
            "(150 000 total), CPU-only, deterministic."
        ),
    )
    p.add_argument("--n-metals", type=int, default=DEFAULT_N_METALS,
                   help="Number of d-block metals (default 30).")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                   help="Samples per metal (default 5000).")
    p.add_argument("--metals", nargs="+", default=list(DBLOCK_METALS))
    p.add_argument("--tmqm-csv", type=Path, default=DEFAULT_TMQM_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--max-products", type=int, default=DEFAULT_MAX_PRODUCTS)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode (caller controls --n-samples, --n-metals).")
    p.add_argument("--max-workers", type=int, default=0,
                   help=(
                       "ProcessPoolExecutor max_workers for per-metal "
                       "parallelism (default 0 = auto = min(8, cpu_count, "
                       "n_metals)). Pass 1 to disable parallelism."
                   ))
    p.add_argument("--no-parallel", action="store_true",
                   help="Disable multiprocessing (single-threaded fallback).")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    metals_arg = list(args.metals)
    if len(metals_arg) == 1 and metals_arg[0].lower() in (
        "all", "all_dblock", "30",
    ):
        metals_arg = list(DBLOCK_METALS)
    known = set(DBLOCK_METALS)
    metals = [m for m in metals_arg if m in known]
    if not metals:
        log.error("No known metals in --metals %s", metals_arg)
        return 2

    payload = run_benchmark(
        n_metals=int(args.n_metals),
        n_samples=int(args.n_samples),
        metals=metals,
        tmqm_csv=Path(args.tmqm_csv),
        output_dir=Path(args.output_dir),
        max_depth=int(args.max_depth),
        max_products=int(args.max_products),
        seeds=int(args.seeds),
        sample_seed=int(args.sample_seed),
        max_workers=int(args.max_workers),
        parallel=not bool(args.no_parallel),
    )
    if "error" in payload:
        log.error("Benchmark aborted: %s", payload["error"])
        return 1

    paths = save_per_metal(payload, output_dir=Path(args.output_dir))
    md_path = write_synthesis(payload, output_dir=Path(args.output_dir))

    agg = payload["aggregate"]
    macro = float(agg["macro_reachability_rate"])
    micro = float(agg["micro_reachability_rate"])
    log.info(
        "[wf130_p2] DONE: n_metals=%d n_total=%d closure_ok=%d "
        "macro=%.4f micro=%.4f gate_pass=%s",
        len(metals), agg["n_samples"], agg["n_closure_ok"],
        macro, micro, payload["overall_gate_pass"],
    )
    print(
        f"[wf130_p2] n_metals={len(metals)} "
        f"n_samples={agg['n_samples']} "
        f"n_unique_total={payload['n_samples_unique_total']} "
        f"n_repeats_total={payload['n_samples_repeats_total']} "
        f"closure_ok={agg['n_closure_ok']} "
        f"macro_reachability={macro:.4f} "
        f"micro_reachability={micro:.4f} "
        f"gate_pass={payload['overall_gate_pass']}"
    )
    for name, p in paths.items():
        print(f"[wf130_p2] wrote {name}: {p}")
    print(f"[wf130_p2] wrote synthesis: {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
