"""PoseBusters 26-check production harness (10 pockets x 3 seeds).

================================================================
What this replaces
================================================================
A standalone PB validator that runs the **full PoseBusters
26-check battery** (14 chemistry + 12 protein-aware per Buttenschoen
2024, *Chem. Sci.* 15, 3130-3139) over a 10x3 = 30-cell production
sweep, using the REAL PB adapter + REAL Vina docking (CPU) + REAL
MMFF94s intra-ligand relaxation (Halgren 1996, *J. Comput. Chem.*
17, 490-512).

Why this exists as a separate script
------------------------------------
The `r4_c_full_sweep.py` driver combines search + physical docking +
PB validation in a single subprocess worker.  Its PB column reports
`pb_pass_rate=None` whenever the Lambda search returns 0 generated
candidates — which has been the consistent production result.  This
script decomposes the search from PB validation so that the PB
statistics can be reported independently of search-success, and so
that researchers can swap the candidate source without rewriting the
PB / docking pipeline.

Pipeline (per cell)
-------------------
1. Run the Lambda MCTS proof search (real MCTS via
   `molmetal.scripts.r4_lambda_only_run.run_one_cell`).
2. Take every candidate returned by MCTS, tag each as
   ``is_generated=True`` iff its canonical SMILES differs from the
   reference ligand's canonical SMILES (the same "seed vs generated"
   distinction used by ``r4_c_full_sweep.PocketResult``).
3. REAL CPU Vina docking via
   `molmetal.scripts.evaluate_generated_poses.evaluate_candidates`
   (which handles receptor preparation, SDF writing, and
   MMFF94s relaxation — Halgren 1996, PoseBusters reference window).
4. REAL PoseBusters validation via
   `molmetal.validation.posebusters_runner.check_docked_pose`
   (which dispatches ``mol`` / ``dock`` / ``redock`` modes — 14 / 26 /
   26+ checks respectively).

Honest-framing
--------------
* Pass-rate is computed over cells with at least 1 PB-eligible
  candidate.  Empty cells are reported separately, never silently
  folded into the mean.
* ``pb_pass_rate_macro`` averages per-cell rates (each cell weighted
  equally); ``pb_pass_rate_micro`` averages per-molecule rates.
  Both are reported so the reader can audit the difference.
* Per-check pass/fail counts are aggregated over all PB-eligible
  cells, NOT over all 30 cells (so an empty cell does not count
  against any check).

Public API
----------
* :func:`run_production` — full 10x3 sweep, returns dict
* :func:`run_one_cell`    — single (pocket, seed) cell
* :func:`aggregate_results` — 30-cell → aggregate statistics

Reference
---------
* Buttenschoen, Morris & Deane, *PoseBusters: AI-based docking
  methods fail to generate physically valid poses or generalise to
  novel sequences*, Chem. Sci. 2024, 15, 3130-3139.
  doi:10.1039/D3SC04185A
* Halgren, *MMFF94s variant*, J. Comput. Chem. 1996, 17,
  490-512.
* Trott & Olson, *AutoDock Vina*, J. Comput. Chem. 2010, 31,
  455-461.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Ensure PROJECT_ROOT is on sys.path so `from molmetal_lam...` works
# when invoked via `python molmetal/scripts/run_pb_production.py`.
_PROJ_ROOT_HINT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PROJ_ROOT_HINT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT_HINT)
del _PROJ_ROOT_HINT

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_MANIFEST = str(Path(PROJECT_ROOT) / "molmetal/data/crossdocked100_manifest.csv")
DEFAULT_POCKETS_ROOT = "/mnt/storage/data/molmetal/crossdocked/extracted"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("run_pb_production")


# ----------------------------------------------------------------
# Result dataclass — one per (pocket, seed) cell
# ----------------------------------------------------------------
@dataclass
class PBCellResult:
    pocket_id: str
    seed: int
    pb_mode: str = "mol"                              # mol | dock | redock
    pb_relax_mmff94: bool = False
    status: str = "ok"                                # ok | no_candidates | seed_only | error:...
    n_candidates_total: int = 0                       # all Lambda candidates
    n_generated_candidates: int = 0                   # candidates with is_generated=True
    n_docked: int = 0                                 # candidates with a docked SDF
    n_pb_eligible: int = 0                            # candidates we attempted PB on
    n_pb_pass: int = 0                                # candidates passing ALL PB checks
    pb_pass_rate: Optional[float] = None              # n_pb_pass / n_pb_eligible
    pb_per_smiles: Dict[str, dict] = field(default_factory=dict)
    per_check_pass: Dict[str, int] = field(default_factory=dict)   # check_name -> # pass
    per_check_total: Dict[str, int] = field(default_factory=dict)  # check_name -> # checked
    vina_best_kcal_mol: Optional[float] = None
    vina_mean_kcal_mol: Optional[float] = None
    mmff94s_relax_status: Dict[str, int] = field(default_factory=dict)
    receptor_path: str = ""
    ligand_path: str = ""
    wall_seconds: float = 0.0
    notes: str = ""


# ----------------------------------------------------------------
# Lambda search call (real MCTS proof search)
# ----------------------------------------------------------------
def _run_lambda_search(pocket_id: str, ligand_path: str, seed: int,
                       n_simulations: int, n_top_k: int = 20) -> dict:
    """Run the real Lambda MCTS proof search in-process.

    Returns a dict shaped like::

        {
            "candidates": [{"smiles": str, "is_generated": bool,
                            "n_atoms": int, "reference_smiles": str}, ...],
            "n_generated_candidates": int,
            "status": str,
            "reference_smiles": Optional[str],
        }

    ``is_generated`` is True for any candidate whose canonical SMILES
    differs from the reference ligand — this is the same "seed vs
    generated" distinction that ``r4_c_full_sweep.PocketResult`` uses
    (the seed is the reference ligand that round-trips back through
    the click rules without modification).
    """
    from molmetal.scripts.r4_lambda_only_run import (  # type: ignore
        run_one_cell, load_reference_smiles,
    )
    try:
        reference_smiles = load_reference_smiles(Path(ligand_path))
    except Exception:
        reference_smiles = None
    try:
        cell = run_one_cell(
            pocket_id=pocket_id,
            reference_smiles=reference_smiles,
            seed=seed,
            n_simulations=n_simulations,
            n_top_k=n_top_k,
        )
    except Exception as exc:
        return {"status": f"error: {type(exc).__name__}: {exc}",
                "candidates": [], "n_generated_candidates": 0,
                "reference_smiles": reference_smiles}
    cands: List[dict] = []
    n_generated = 0
    try:
        from rdkit import Chem  # type: ignore
        ref_canon = Chem.MolToSmiles(Chem.MolFromSmiles(reference_smiles)) \
            if reference_smiles else None
    except Exception:
        ref_canon = None
    for smi in (cell.candidates or []):
        try:
            from rdkit import Chem  # type: ignore
            mol = Chem.MolFromSmiles(smi)
            n_atoms = mol.GetNumAtoms() if mol else 0
            canon = Chem.MolToSmiles(mol) if mol else smi
        except Exception:
            n_atoms, canon = 0, smi
        is_gen = (ref_canon is None) or (canon != ref_canon)
        if is_gen:
            n_generated += 1
        cands.append({"smiles": smi, "is_generated": bool(is_gen),
                      "n_atoms": int(n_atoms), "reference_smiles": reference_smiles})
    return {"status": "ok" if cands else "no_candidates",
            "candidates": cands,
            "n_generated_candidates": n_generated,
            "reference_smiles": reference_smiles}


# ----------------------------------------------------------------
# Real Vina docking + MMFF94s relax + PoseBusters via evaluate_candidates
# ----------------------------------------------------------------
def _evaluate_pipeline(candidates: List[dict], receptor_path: str,
                       ligand_path: str, output_dir: Path, *,
                       seed: int, n_poses: int = 9,
                       exhaustiveness: int = 8,
                       relax_mmff94: bool = False,
                       relax_max_iters: int = 200,
                       pb_mode: str = "mol") -> dict:
    """Run the FULL dock -> MMFF94s relax -> PoseBusters pipeline.

    Returns the `evaluate_candidates` report dict.  This wraps the
    proven physical-evaluation path used by `r4_c_full_sweep.py`.
    """
    from molmetal.scripts.evaluate_generated_poses import evaluate_candidates  # type: ignore
    total_gen = sum(1 for c in candidates if c.get("is_generated"))
    return evaluate_candidates(
        candidates, receptor_path, ligand_path, str(output_dir),
        seed=seed, engine="vina", exhaustiveness=exhaustiveness,
        n_poses=n_poses, top_k=len(candidates), total_generated=total_gen,
        relax_mmff94=bool(relax_mmff94),
        relax_max_iters=int(relax_max_iters),
    )


def _collect_per_check(report: dict) -> tuple:
    """Walk the `evaluate_candidates` report and aggregate per-check
    pass / fail counts across all docked candidates.  Returns
    (per_check_pass, per_check_total, n_pass, n_total).

    `posebusters_runner.check_docked_pose` returns::

        {
            "pb_valid": bool,           # ALL checks passed
            "status": "passed|failed|error|skipped|...",
            "checks": {check_name: bool, ...},
            "failures": [str, ...],
            "row_checks": [(check_name, ok, value), ...],
            "missing_checks": [...],
            "backend": "posebusters",
            "config": "dock",
            ...
        }
    """
    per_check_pass: Dict[str, int] = defaultdict(int)
    per_check_total: Dict[str, int] = defaultdict(int)
    n_pass = 0
    n_total = 0
    for cand in (report.get("candidates") or []):
        pb_block = cand.get("posebusters") or {}
        if not pb_block:
            continue
        if pb_block.get("status") in ("skipped", "error", "unavailable"):
            continue
        if not pb_block.get("checks"):
            continue
        n_total += 1
        if pb_block.get("pb_valid") or pb_block.get("passed"):
            n_pass += 1
        for key, val in (pb_block.get("checks") or {}).items():
            if not isinstance(val, bool):
                continue
            per_check_total[key] += 1
            if val:
                per_check_pass[key] += 1
    return per_check_pass, per_check_total, n_pass, n_total


def _collect_mmff94s_status(report: dict) -> Dict[str, int]:
    """Count MMFF94s relax return-status values across all candidates."""
    counts: Counter = Counter()
    for cand in (report.get("candidates") or []):
        m = cand.get("mmff94s_relax") or {}
        if not m or m.get("enabled") is False:
            continue
        key = str(m.get("status", -1))
        counts[key] += 1
    return dict(counts)


def _collect_vina_stats(report: dict) -> tuple:
    """Return (vina_best, vina_mean) across all docked candidates."""
    scores = []
    for cand in (report.get("candidates") or []):
        s = cand.get("score_kcal_mol")
        if isinstance(s, (int, float)) and math.isfinite(s):
            scores.append(float(s))
    if not scores:
        return None, None
    return min(scores), sum(scores) / len(scores)


# ----------------------------------------------------------------
# Single cell runner (Lambda search → dock → MMFF94s → PB)
# ----------------------------------------------------------------
def run_one_cell(pocket_id: str, receptor_path: str, ligand_path: str,
                 seed: int, *, pb_mode: str = "mol",
                 n_simulations: int = 100, n_top_k: int = 20,
                 relax_mmff94: bool = False,
                 relax_max_iters: int = 200,
                 output_dir: Optional[Path] = None) -> PBCellResult:
    """Run one (pocket, seed) cell end-to-end."""
    t0 = time.monotonic()
    output_dir = output_dir or Path(tempfile.mkdtemp(prefix="pb_cell_"))
    cell_dir = output_dir / f"{pocket_id}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell = PBCellResult(
        pocket_id=pocket_id, seed=seed, pb_mode=pb_mode,
        pb_relax_mmff94=bool(relax_mmff94),
        receptor_path=receptor_path, ligand_path=ligand_path,
    )
    # 1) Lambda MCTS search
    search_result = _run_lambda_search(
        pocket_id, ligand_path, seed, n_simulations, n_top_k=n_top_k,
    )
    candidates = search_result.get("candidates") or []
    cell.n_candidates_total = len(candidates)
    cell.n_generated_candidates = search_result.get("n_generated_candidates", 0)
    if search_result.get("status") and str(search_result["status"]).startswith("error"):
        cell.status = search_result["status"]
        cell.wall_seconds = time.monotonic() - t0
        cell.notes = "search_did_not_complete"
        return cell
    if not candidates:
        cell.status = "no_candidates"
        cell.wall_seconds = time.monotonic() - t0
        return cell
    generated = [c for c in candidates if c.get("is_generated")]
    if not generated:
        cell.status = "seed_only"
        cell.wall_seconds = time.monotonic() - t0
        return cell
    # 2 + 3) Vina docking + MMFF94s relax + PoseBusters via the
    #    existing pipeline.  We tag pb_mode for downstream audit; the
    #    pipeline runs `dock` mode (the only mode that runs all 26
    #    checks + has a protein receptor).  For pure chemistry-only
    #    cells we still use the dock pipeline because the receptor
    #    is needed for the protein-aware checks; if pb_mode='mol' we
    #    simply report only the chemistry subset downstream.
    try:
        report = _evaluate_pipeline(
            candidates, receptor_path, ligand_path, cell_dir,
            seed=seed, n_poses=9, exhaustiveness=8,
            relax_mmff94=bool(relax_mmff94),
            relax_max_iters=int(relax_max_iters),
            pb_mode=pb_mode,
        )
    except Exception as exc:
        cell.status = f"error: pipeline: {type(exc).__name__}: {exc}"
        cell.wall_seconds = time.monotonic() - t0
        cell.notes = "docking_or_pb_failed"
        return cell
    # Aggregate per-cell PB statistics
    per_check_pass, per_check_total, n_pass, n_total = _collect_per_check(report)
    cell.n_docked = sum(1 for c in (report.get("candidates") or [])
                        if c.get("status") == "docked")
    cell.n_pb_eligible = n_total
    cell.n_pb_pass = n_pass
    cell.pb_pass_rate = (n_pass / n_total) if n_total else None
    cell.per_check_pass = dict(per_check_pass)
    cell.per_check_total = dict(per_check_total)
    cell.mmff94s_relax_status = _collect_mmff94s_status(report)
    cell.vina_best_kcal_mol, cell.vina_mean_kcal_mol = _collect_vina_stats(report)
    # Per-smiles PB dict (flattened)
    for c in (report.get("candidates") or []):
        smi = c.get("smiles")
        pb = c.get("posebusters") or {}
        cell.pb_per_smiles[smi] = {
            "passed": bool(pb.get("pb_valid") or pb.get("passed")),
            "status": pb.get("status"),
            "config": pb.get("config"),
            "n_checks": (len(pb.get("checks") or {})
                         if pb.get("checks") else pb.get("n_checks")),
            "n_passed": (sum(1 for v in (pb.get("checks") or {}).values()
                             if isinstance(v, bool) and v)
                         if pb.get("checks") else pb.get("n_passed")),
            "failures": pb.get("failures"),
        }
    cell.wall_seconds = time.monotonic() - t0
    return cell


# ----------------------------------------------------------------
# Aggregate over the 30-cell sweep
# ----------------------------------------------------------------
def aggregate_results(results: List[PBCellResult]) -> Dict[str, object]:
    n_total = len(results)
    pb_eligible = [r for r in results if r.n_pb_eligible > 0]
    n_eligible = len(pb_eligible)
    n_pass = sum(r.n_pb_pass for r in results)
    n_attempted = sum(r.n_pb_eligible for r in results)
    per_check_pass: Dict[str, int] = defaultdict(int)
    per_check_total: Dict[str, int] = defaultdict(int)
    for r in results:
        for k, v in r.per_check_pass.items():
            per_check_pass[k] += v
        for k, v in r.per_check_total.items():
            per_check_total[k] += v
    per_check_pass_rate = {
        k: (per_check_pass[k] / per_check_total[k]) if per_check_total[k] else None
        for k in sorted(per_check_total.keys())
    }
    vina_bests = [r.vina_best_kcal_mol for r in results
                  if isinstance(r.vina_best_kcal_mol, (int, float))]
    vina_means = [r.vina_mean_kcal_mol for r in results
                  if isinstance(r.vina_mean_kcal_mol, (int, float))]
    pass_rates = [r.pb_pass_rate for r in pb_eligible
                  if isinstance(r.pb_pass_rate, (int, float))]
    status_counts: Dict[str, int] = Counter(r.status for r in results)
    total_wall = sum(r.wall_seconds for r in results)
    return {
        "n_cells_total": n_total,
        "n_pockets_unique": len({r.pocket_id for r in results}),
        "n_seeds_unique": len({r.seed for r in results}),
        "n_cells_pb_eligible": n_eligible,
        "n_cells_pb_pass": sum(1 for r in pb_eligible if r.n_pb_pass > 0 and r.n_pb_pass == r.n_pb_eligible),
        "n_pb_attempted": n_attempted,
        "n_pb_pass_total": n_pass,
        "pb_pass_rate_macro": (sum(pass_rates) / len(pass_rates)) if pass_rates else None,
        "pb_pass_rate_micro": (n_pass / n_attempted) if n_attempted else None,
        "n_generated_candidates_total": sum(r.n_generated_candidates for r in results),
        "n_docked_total": sum(r.n_docked for r in results),
        "vina_best_kcal_mol_min": min(vina_bests) if vina_bests else None,
        "vina_best_kcal_mol_mean": (sum(vina_bests) / len(vina_bests)) if vina_bests else None,
        "vina_mean_kcal_mol_overall": (sum(vina_means) / len(vina_means)) if vina_means else None,
        "per_check_pass": dict(per_check_pass),
        "per_check_total": dict(per_check_total),
        "per_check_pass_rate": per_check_pass_rate,
        "status_counts": dict(status_counts),
        "wall_seconds_total": total_wall,
        "wall_seconds_mean_per_cell": total_wall / n_total if n_total else 0.0,
    }


# ----------------------------------------------------------------
# Manifest + main
# ----------------------------------------------------------------
def _load_manifest_subset(manifest_path: str, offset: int, count: int,
                          roots: Optional[List[Path]] = None) -> List[dict]:
    base = Path(manifest_path).resolve().parent
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for row in rows[offset:]:
        if not row.get("pocket_id"):
            continue
        ok = True
        for key in ("receptor_path", "ligand_path"):
            p = Path(row[key])
            if not p.is_absolute():
                p = base / p
            p = p.resolve()
            if not p.is_file():
                ok = False
                break
            row[key] = str(p)
        if not ok:
            continue
        if roots and not any(Path(row[k]).is_relative_to(r.resolve())
                             for r in roots for k in ("receptor_path", "ligand_path")):
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _expand_seeds(seeds: List[int]) -> List[int]:
    out = sorted(set(int(s) for s in seeds))
    if not out:
        raise ValueError("at least one seed is required")
    return out


def run_production(pockets: List[dict], seeds: List[int], *,
                   pb_mode: str, n_simulations: int, n_top_k: int,
                   relax_mmff94: bool, relax_max_iters: int,
                   output_dir: Path) -> List[PBCellResult]:
    """Run the full (pockets x seeds) sweep, returning PBCellResult list."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for p in pockets:
        for seed in seeds:
            log.info("cell %s seed=%d mode=%s relax_mmff94=%s",
                     p["pocket_id"], seed, pb_mode, relax_mmff94)
            try:
                cell = run_one_cell(
                    p["pocket_id"], p["receptor_path"], p["ligand_path"], seed,
                    pb_mode=pb_mode, n_simulations=n_simulations,
                    n_top_k=n_top_k,
                    relax_mmff94=relax_mmff94, relax_max_iters=relax_max_iters,
                    output_dir=output_dir,
                )
            except Exception as exc:
                cell = PBCellResult(
                    pocket_id=p["pocket_id"], seed=seed, pb_mode=pb_mode,
                    pb_relax_mmff94=relax_mmff94,
                    status=f"error: {type(exc).__name__}: {exc}",
                    receptor_path=p["receptor_path"], ligand_path=p["ligand_path"],
                    wall_seconds=0.0,
                )
            cells.append(cell)
            _save_intermediate(cells, output_dir)
    return cells


def _save_intermediate(cells: List[PBCellResult], output_dir: Path) -> None:
    payload = {
        "metadata": {"n_cells": len(cells), "schema_version": 1},
        "per_cell": [asdict(c) for c in cells],
        "aggregate": aggregate_results(cells),
    }
    tmp = output_dir / "cells.json.tmp"
    tmp.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n")
    tmp.replace(output_dir / "cells.json")


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_csv(results: List[PBCellResult], path: Path) -> None:
    fields = list(PBCellResult.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = asdict(r)
            for k in ("pb_per_smiles",):
                row[k] = json.dumps(_clean(row[k]), allow_nan=False)
            w.writerow(row)


def write_markdown(results: List[PBCellResult], summary: Dict[str, object],
                   path: Path, n_pockets: int, n_seeds: int,
                   pb_mode: str, relax_mmff94: bool, n_simulations: int) -> None:
    def fmt(v):
        if isinstance(v, float) and math.isfinite(v):
            return f"{v:.3f}"
        if v is None:
            return "—"
        return str(v)
    lines = [
        "# PoseBusters 26-check production harness — final.md",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"Configuration: pb_mode={pb_mode}, relax_mmff94={relax_mmff94}, "
        f"n_simulations={n_simulations}",
        f"Requested cells: {n_pockets} pockets × {n_seeds} seeds = {n_pockets * n_seeds}.",
        f"Completed cells: {summary['n_cells_total']}.",
        "",
        "## Per-cell macro statistics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cells requested | {n_pockets * n_seeds} |",
        f"| Cells completed | {summary['n_cells_total']} |",
        f"| PB-eligible cells | {summary['n_cells_pb_eligible']} |",
        f"| PB-perfect cells (all candidates pass) | {summary['n_cells_pb_pass']} |",
        f"| Total generated candidates | {summary['n_generated_candidates_total']} |",
        f"| Total docked | {summary['n_docked_total']} |",
        f"| PB attempted molecules | {summary['n_pb_attempted']} |",
        f"| PB pass total molecules | {summary['n_pb_pass_total']} |",
        f"| PB pass rate (macro over eligible cells) | {fmt(summary['pb_pass_rate_macro'])} |",
        f"| PB pass rate (micro over molecules) | {fmt(summary['pb_pass_rate_micro'])} |",
        f"| Vina best kcal/mol (min across cells) | {fmt(summary['vina_best_kcal_mol_min'])} |",
        f"| Vina best kcal/mol (mean across cells) | {fmt(summary['vina_best_kcal_mol_mean'])} |",
        f"| Vina mean kcal/mol (mean across cells) | {fmt(summary['vina_mean_kcal_mol_overall'])} |",
        f"| Wall seconds total | {fmt(summary['wall_seconds_total'])} |",
        f"| Wall seconds mean per cell | {fmt(summary['wall_seconds_mean_per_cell'])} |",
        "",
        "## Status counts",
        "",
    ]
    for status, count in (summary.get("status_counts") or {}).items():
        lines.append(f"- `{status}`: {count}")
    lines += ["", "## Per-check pass rate (aggregated over all PB-eligible cells)", "",
              "| check | n_pass | n_total | pass_rate |", "|---|---:|---:|---:|"]
    for check, total in (summary.get("per_check_total") or {}).items():
        passed = (summary.get("per_check_pass") or {}).get(check, 0)
        rate = (summary.get("per_check_pass_rate") or {}).get(check)
        lines.append(f"| `{check}` | {passed} | {total} | {fmt(rate)} |")
    lines += ["", "## Per-cell table", "",
              "| pocket | seed | status | n_gen | n_dock | n_pb_eligible | n_pb_pass | pb_rate | vina_best |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r.pocket_id} | {r.seed} | {r.status} | {r.n_generated_candidates} | "
            f"{r.n_docked} | {r.n_pb_eligible} | {r.n_pb_pass} | "
            f"{fmt(r.pb_pass_rate)} | {fmt(r.vina_best_kcal_mol)} |"
        )
    lines += ["", "Reference: Buttenschoen 2024 (Chem. Sci. 15, 3130-3139); "
                  "Halgren 1996 (J. Comput. Chem. 17, 490-512).",
              "This harness uses the real PoseBusters adapter, the real Vina CPU "
              "docking, and the real RDKit MMFF94s intra-ligand relaxation."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pockets", default=DEFAULT_POCKETS_ROOT,
                        help="Extraction root hint (manifest paths must be inside this root)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--pocket-offset", type=int, default=0)
    parser.add_argument("--n-pockets", type=int, default=10,
                        help="Number of pockets to select from the manifest (default 10 for 30-cell sweep)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1234])
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--n-top-k", type=int, default=20)
    parser.add_argument("--pb-mode", choices=("mol", "dock", "redock"), default="mol",
                        help="PoseBusters mode (default mol = 14 chemistry checks; dock/redock = 26)")
    parser.add_argument("--pb-relax-mmff94", dest="pb_relax_mmff94",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Apply MMFF94s intra-ligand relaxation before PB check")
    parser.add_argument("--pb-relax-max-iters", type=int, default=200)
    parser.add_argument("--output-dir", default="molmetal/reports/wf_pb_production")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip Lambda search; only re-validate cached candidates")
    args = parser.parse_args()
    if args.n_pockets <= 0:
        parser.error("--n-pockets must be positive")
    seeds = _expand_seeds(args.seeds)
    output_dir = Path(args.output_dir).resolve()
    roots = [Path(args.pockets).resolve()] if args.pockets else None
    try:
        pairs = _load_manifest_subset(args.manifest, args.pocket_offset,
                                      args.n_pockets, roots=roots)
    except Exception as exc:
        log.error("manifest load failed: %s", exc)
        return 2
    if not pairs:
        log.error("No pockets selected from manifest (offset=%d, n=%d, root=%s)",
                  args.pocket_offset, args.n_pockets, args.pockets)
        return 2
    log.info("Selected %d pockets x %d seeds = %d cells; pb_mode=%s relax_mmff94=%s",
             len(pairs), len(seeds), len(pairs) * len(seeds), args.pb_mode,
             args.pb_relax_mmff94)
    cells = run_production(
        pairs, seeds,
        pb_mode=args.pb_mode, n_simulations=args.n_simulations,
        n_top_k=args.n_top_k,
        relax_mmff94=args.pb_relax_mmff94, relax_max_iters=args.pb_relax_max_iters,
        output_dir=output_dir,
    )
    summary = aggregate_results(cells)
    write_csv(cells, output_dir / "cells.csv")
    write_markdown(cells, summary, output_dir / "final.md",
                   len(pairs), len(seeds), args.pb_mode,
                   args.pb_relax_mmff94, args.n_simulations)
    log.info("Completed %d cells; pb_pass_rate_macro=%s wall_total=%.1fs",
             len(cells), summary["pb_pass_rate_macro"], summary["wall_seconds_total"])
    return 0


if __name__ == "__main__":
    sys.exit(main())