"""Wf-73 Phase 2 — Per-metal OOD reachability (MetalCytoToxDB 5-metal).

================================================================
Wf-73 Phase 2 (L12 trajectory): per-metal OOD reachability on the
**MetalCytoToxDB 5-metal** dataset (Ru / Ir / Os / Rh / Re), filtered
by Morgan-ECFP4 Tanimoto<0.4 against the tmQM d-block training pool.

This script extends the Wf-41 Phase 2 cross-domain OOD benchmark
(:mod:`molmetal.scripts.bench_closure_ood`) along the **per-metal
axis**: where Wf-41 P2 measured closure reachability per (domain,
metal) at coarse resolution (n_samples_per_metal=20–50, 1 seed), this
script measures it at **fine per-metal resolution** with:

  * **5-metal, n_samples_per_metal=80** (default) — quadruples the
    Wf-41 P2 per-metal sample budget to get statistically stable
    reachability estimates per Ru / Ir / Os / Rh / Re.
  * **3 seeds** — disjoint SMILES slices per seed (deterministic via
    start-row offset on the CSV), giving mean ± std per metal.
  * **Tanimoto<0.4 OOD filter against tmQM** — the Wf-7 protocol
    lifted to per-metal OOD reachability rather than per-domain
    aggregate.
  * **Rh bottleneck lift diagnostic** — explicitly compares the
    Rh (n=1134) reachability against the other 4 metals and gives
    an honest bottleneck readout per
    ``party-twenty-spirit §3.5`` (no fabricated lift claims).
  * **Cross-metal correlation matrix** — Pearson r over the
    per-sample closure-OK indicator across the 5 metals, to spot
    whether closure failure on one metal is correlated with failure
    on another.

Inputs
------
* ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` (Wf-21 ship,
  26801 rows × 5 metals).
* ``molmetal/data/tmqm_dblock_subset.csv`` (Wf-28 ship, 30 d-block
  metals — the OOD train pool).
* :mod:`molmetal_lam.lam_chem.closure` (R17 ship).
* :mod:`molmetal_lam.lam_chem.ood_protocol` (Wf-7 ship).

Outputs (under ``molmetal/reports/wf73_p2_ood_permetal/``)
------
* ``{Ru,Ir,Os,Rh,Re}/ood_reachability.json`` — per-metal aggregate
  (n_samples, n_closure_ok, reachability_rate, mean ± std wall).
* ``synthesis.md`` — honest per-metal breakdown + cross-metal
  correlation table + Rh bottleneck diagnosis (party-twenty-spirit
  §3.5 honest framing).

Usage
-----
::

    # Smoke (small, fast): 5 samples per metal × 1 seed.
    uv run --no-sync python molmetal/scripts/bench_ood_per_metal.py \\
        --n-samples 5 --seeds 1

    # Default production: 80 samples × 3 seeds × 5 metals.
    uv run --no-sync python molmetal/scripts/bench_ood_per_metal.py

    # Single-metal diagnostic (e.g. Rh bottleneck study).
    uv run --no-sync python molmetal/scripts/bench_ood_per_metal.py \\
        --metals Rh --n-samples 200 --seeds 5
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


log = logging.getLogger("wf73_ood_per_metal")

# ---------------------------------------------------------------------------
# Defaults — MetalCytoToxDB 5-metal + tmQM d-block
# ---------------------------------------------------------------------------
METALCYTOTOX_METALS: Tuple[str, ...] = (
    "Ru",
    "Ir",
    "Os",
    "Rh",
    "Re",
)

DEFAULT_METALCYTOTOX_CSV: Path = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
DEFAULT_TMQM_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
DEFAULT_OUTPUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf73_p2_ood_permetal"

DEFAULT_N_SAMPLES: int = 80
DEFAULT_TANIMOTO_THRESHOLD: float = 0.4
DEFAULT_MAX_DEPTH: int = 1
DEFAULT_MAX_PRODUCTS: int = 32
DEFAULT_SEEDS: int = 3

# Per-metal atomic-number map (for coordination sanity checks).
METAL_Z: Dict[str, int] = {
    "Ru": 44,
    "Ir": 77,
    "Os": 76,
    "Rh": 45,
    "Re": 75,
}

# Click-rule partner terms (5 click handles + amine + benzoic acid;
# mirrors Wf-37 P2 / Wf-41 P2 fixture).
PARTNER_SMILE_LIST: List[str] = [
    "C#CCN",
    "CCN=[N+]=[N-]",
    "C#CC",
    "CS",
    "C=CC",
    "OC(=O)c1ccccc1",
    "NCC",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _memory_mb(snapshot_diff: int) -> float:
    return float(snapshot_diff) / (1024.0 * 1024.0)


def _load_click_rules() -> List[Any]:
    from molmetal_lam.lam_chem.rules import (
        AmideCoupling,
        CuAAC,
        SPAAC,
        Suzuki,
        ThiolEne,
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
    """Load up to ``n`` SMILES tagged with ``metal`` starting at ``offset``."""
    out: List[str] = []
    if not csv_path.is_file():
        log.error("CSV not found at %s", csv_path)
        return out
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if metal_col not in fields or smiles_col not in fields:
            log.error("Columns missing from %s; fields=%r", csv_path, fields)
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
    """Load the full tmQM d-block SMILES (OOD train pool)."""
    out: List[str] = []
    if not csv_path.is_file():
        log.error("tmQM CSV not found at %s", csv_path)
        return out
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles", "") or "").strip()
            if smi:
                out.append(smi)
    return out


# ---------------------------------------------------------------------------
# Per-sample benchmark
# ---------------------------------------------------------------------------
def _benchmark_one(
    smiles: str,
    idx: int,
    metal: str,
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
        "metal": str(metal),
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

    # Step 1: OOD filter (Tanimoto<0.4 vs tmQM).
    t_ood_0 = time.perf_counter()
    try:
        max_sim = ood_protocol.max_tanimoto(smiles, train_pool)
    except Exception:
        max_sim = 0.0  # fail-closed: unparseable counts as OOD
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
            metal_z = METAL_Z.get(metal, -1)
            if metal_z > 0:
                from molmetal_lam.molecules.closed_term import (
                    MoleculeClosedTerm as _MCT,
                )

                term_obj = _MCT.from_smiles(smiles, embed_3d=False)
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
# Per-metal aggregation
# ---------------------------------------------------------------------------
def _aggregate_per_metal(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-sample records into a per-metal benchmark dict."""
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
        "reachability_rate": float(n_closure / n) if n else 0.0,
        "ood_pass_rate": float(n_ood / n) if n else 0.0,
        "mean_wall_time": mean_wall,
        "std_wall_time": std_wall,
        "median_wall_time": median_wall,
        "peak_memory_mb": peak_mem,
        "total_wall_time": total_wall,
        "n_error": int(n_err),
        "n_metal_coordinated": int(n_metal),
        "metal_coordination_rate": float((n_metal / n) if n else 0.0),
        "mean_max_tanimoto_tmQM": float(
            statistics.fmean([r["max_tanimoto_tmQM"] for r in records])
        )
        if records
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_ood_per_metal_benchmark(
    n_samples: int = DEFAULT_N_SAMPLES,
    *,
    metals: Sequence[str] = METALCYTOTOX_METALS,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    metalcyto_csv: Path = DEFAULT_METALCYTOTOX_CSV,
    tmqm_csv: Path = DEFAULT_TMQM_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_products: int = DEFAULT_MAX_PRODUCTS,
    seeds: int = DEFAULT_SEEDS,
) -> Dict[str, Any]:
    """Run per-metal OOD reachability on MetalCytoToxDB 5-metal."""
    log.info(
        "[wf73_ood_per_metal] start: metals=%s, n_per_metal=%d, "
        "seeds=%d, threshold=%.2f, max_depth=%d",
        list(metals),
        int(n_samples),
        int(seeds),
        float(tanimoto_threshold),
        int(max_depth),
    )

    click_rules = _load_click_rules()
    partner_terms = _load_partner_terms()
    click_rule_names = [getattr(r, "name", type(r).__name__) for r in click_rules]
    train_pool = _load_tmqm_train_pool(Path(tmqm_csv))
    ood_protocol = _load_tanimoto_protocol(threshold=tanimoto_threshold)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "bench_ood_per_metal.py",
        "pilot_id": "wf73_ood_per_metal",
        "date_utc": _utcnow_iso(),
        "method": "closure+ood_filter",
        "tanimoto_threshold": float(tanimoto_threshold),
        "metals": list(metals),
        "n_samples_per_metal_per_seed": int(n_samples),
        "max_depth": int(max_depth),
        "max_products": int(max_products),
        "seeds": int(seeds),
        "click_rule_names": click_rule_names,
        "tmqm_train_pool_size": int(len(train_pool)),
        "metalcyto_csv": str(metalcyto_csv),
        "tmqm_csv": str(tmqm_csv),
        "per_metal": {},
        "per_seed": {},
    }

    t_total_0 = time.perf_counter()
    n_error_total = 0

    for metal in metals:
        metal_records_all: List[Dict[str, Any]] = []
        seed_metrics: Dict[int, Dict[str, Any]] = {}
        for seed in range(int(seeds)):
            offset = int(seed * n_samples)
            smiles_list = _load_metal_samples(
                csv_path=Path(metalcyto_csv),
                metal=metal,
                n=int(n_samples),
                offset=int(offset),
            )
            log.info(
                "[wf73_ood_per_metal] metal=%s seed=%d offset=%d n_loaded=%d",
                metal,
                seed,
                offset,
                len(smiles_list),
            )
            if not smiles_list:
                log.warning(
                    "[wf73_ood_per_metal] no SMILES for metal=%s seed=%d",
                    metal,
                    seed,
                )
                continue

            seed_recs: List[Dict[str, Any]] = []
            for i, smi in enumerate(smiles_list):
                rec = _benchmark_one(
                    smiles=smi,
                    idx=int(offset + i),
                    metal=metal,
                    click_rules=click_rules,
                    partner_terms=partner_terms,
                    ood_protocol=ood_protocol,
                    train_pool=train_pool,
                    max_depth=int(max_depth),
                    max_products=int(max_products),
                )
                seed_recs.append(rec)
            seed_metrics[seed] = _aggregate_per_metal(seed_recs)
            metal_records_all.extend(seed_recs)

        per_metal = _aggregate_per_metal(metal_records_all)
        n_error_total += int(per_metal.get("n_error", 0))
        payload["per_metal"][metal] = {
            "aggregate": per_metal,
            "per_seed": seed_metrics,
            "n_records": int(len(metal_records_all)),
            "seeds_used": int(len(seed_metrics)),
        }

    # Cross-metal correlation (Pearson r over per-seed reachability
    # rate across the 5 metals — diagnostic, low-N friendly).
    seed_index: Dict[int, Dict[str, float]] = defaultdict(dict)
    for metal in metals:
        seed_block = payload["per_metal"].get(metal, {}).get("per_seed", {})
        for s, agg in seed_block.items():
            seed_index[int(s)][metal] = float(agg.get("reachability_rate", 0.0))
    cross_metal_corr: Dict[str, Dict[str, float]] = {}
    metals_list = list(metals)
    for i, ma in enumerate(metals_list):
        cross_metal_corr[ma] = {}
        for j, mb in enumerate(metals_list):
            if i == j:
                cross_metal_corr[ma][mb] = 1.0
                continue
            xs = [
                seed_index[s][ma]
                for s in seed_index
                if ma in seed_index[s] and mb in seed_index[s]
            ]
            ys = [
                seed_index[s][mb]
                for s in seed_index
                if ma in seed_index[s] and mb in seed_index[s]
            ]
            if len(xs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
                try:
                    cross_metal_corr[ma][mb] = float(statistics.correlation(xs, ys))
                except Exception:
                    cross_metal_corr[ma][mb] = 0.0
            else:
                cross_metal_corr[ma][mb] = 0.0
    payload["cross_metal_correlation"] = cross_metal_corr

    # Rh bottleneck diagnosis.
    rh = payload["per_metal"].get("Rh", {}).get("aggregate", {})
    others = [
        payload["per_metal"][m]["aggregate"]
        for m in metals
        if m != "Rh" and m in payload["per_metal"]
    ]
    rh_rate = float(rh.get("reachability_rate", 0.0))
    others_mean = (
        float(statistics.fmean([o["reachability_rate"] for o in others]))
        if others
        else 0.0
    )
    payload["rh_bottleneck"] = {
        "rh_reachability_rate": rh_rate,
        "other_metals_reachability_mean": others_mean,
        "lift_gap": float(others_mean - rh_rate),
        "verdict": (
            "RH_BOTTLENECK_LIFTED"
            if rh_rate >= max(0.30, others_mean - 0.10)
            else "RH_HONEST_PARTIAL"
        ),
    }

    payload["n_error_total"] = int(n_error_total)
    payload["total_wall_time"] = float(time.perf_counter() - t_total_0)
    return payload


def save_ood_per_metal_benchmark(
    payload: Dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Persist per-metal JSON + synthesis.md to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for metal in payload["per_metal"]:
        metal_dir = out / metal
        metal_dir.mkdir(parents=True, exist_ok=True)
        with (metal_dir / "ood_reachability.json").open("w") as fh:
            json.dump(
                payload["per_metal"][metal],
                fh,
                indent=2,
                sort_keys=True,
            )
    with (out / "wf73_ood_per_metal_full.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    log.info("[wf73_ood_per_metal] saved per-metal JSON to %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-73 Phase 2 — per-metal OOD reachability on "
            "MetalCytoToxDB 5-metal (CPU)"
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"SMILES per metal per seed (default {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--metals",
        nargs="+",
        default=list(METALCYTOTOX_METALS),
        help="Subset of metals to run (default: all 5).",
    )
    parser.add_argument(
        "--tanimoto-thresh",
        type=float,
        default=DEFAULT_TANIMOTO_THRESHOLD,
        help=f"OOD threshold (default {DEFAULT_TANIMOTO_THRESHOLD}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Number of disjoint SMILES slices per metal (default {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=DEFAULT_MAX_PRODUCTS,
    )
    parser.add_argument(
        "--metalcyto-csv",
        type=Path,
        default=DEFAULT_METALCYTOTOX_CSV,
    )
    parser.add_argument(
        "--tmqm-csv",
        type=Path,
        default=DEFAULT_TMQM_CSV,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    payload = run_ood_per_metal_benchmark(
        n_samples=int(args.n_samples),
        metals=tuple(args.metals),
        tanimoto_threshold=float(args.tanimoto_thresh),
        metalcyto_csv=args.metalcyto_csv,
        tmqm_csv=args.tmqm_csv,
        output_dir=args.output_dir,
        max_depth=int(args.max_depth),
        max_products=int(args.max_products),
        seeds=int(args.seeds),
    )
    save_ood_per_metal_benchmark(payload, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "tool",
                    "date_utc",
                    "metals",
                    "seeds",
                    "n_samples_per_metal_per_seed",
                    "rh_bottleneck",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    for metal, blk in payload["per_metal"].items():
        agg = blk["aggregate"]
        print(
            f"  metal={metal:>4}: n={agg['n_samples']:4d}  "
            f"reachability={agg['reachability_rate']:.4f}  "
            f"ood_pass={agg['ood_pass_rate']:.4f}  "
            f"metal_coord={agg['metal_coordination_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
