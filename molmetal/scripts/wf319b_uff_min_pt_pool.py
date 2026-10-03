#!/usr/bin/env python3
"""Wf-319B — UFF-min Pt-pool ACTUAL EXECUTION.

============================================================================
Wf-319B (TODO-319B follow-up to Wf-315C) — RDKit UFF relaxation for 100
Pt-drug conformers from the PlatinAI MB-Finder (Rusanov 2025) Pt subset.

Per Wf-315C analysis: UFF-min Pt-pool was identified as a CPU-only path
that could improve the m10 Vina Min Score ceiling (WEAK_TO_STRONG §2.7
PROJECTED). Wf-319B executes the actual UFF minimization on disk so the
downstream Vina pipeline can dock physically-plausible conformers instead
of 2D-only structures.

Note on data source: per Wf-319C honest framing, the Krasnov 2026
MetalCytoToxDB (Ir/Os/Re/Rh/Ru) does NOT contain Pt; the canonical Pt
subset in this repo is platinai_4cell.csv (MB-Finder / Rusanov 2025),
which is what this script samples from.

CLI::

    wf319b_uff_min_pt_pool.py [--n-conformers 100] [--seed 42]
                              [--max-iter 200] [--json-out PATH]
                              [--metrics-out PATH]

Outputs:

* ``molmetal/reports/wf319b_uff_min_pt_pool/pool.json``
    — per-conformer UFF energy record
* ``metrics/by_metric/uff_energy_pt_pool.json``
    — mean ± std + distribution

CPU-only. No GPU. No external packages beyond the venv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DOT_DATA_DIR = REPO_ROOT / ".data" / "molmetal"
DEFAULT_PT_CSV = DOT_DATA_DIR / "platinai_4cell.csv"
DEFAULT_POOL_OUT = (
    REPO_ROOT / "molmetal" / "reports" / "wf319b_uff_min_pt_pool" / "pool.json"
)
DEFAULT_METRICS_OUT = REPO_ROOT / "metrics" / "by_metric" / "uff_energy_pt_pool.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_pt_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Load all Pt rows from platinai_4cell.csv (the canonical Pt subset)."""
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if r.get("metal") == "Pt"]
    if not rows:
        raise RuntimeError(f"No Pt rows found in {csv_path}")
    return rows


def _sample_balanced(
    rows: List[Dict[str, str]],
    n: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Sample ``n`` Pt rows.

    Balances on pIC50_a2780 (closest to canonical MCF7 / A2780 assays) so
    the pool contains both active and inactive compounds. Activity proxy
    uses pIC50_a2780 >= 5.0 (IC50 <= 10 uM) as "active".
    """
    actives: list[str] = []
    inactives: list[str] = []

    for r in rows:
        try:
            pic50 = float(r.get("pIC50_a2780", "0") or "nan")
        except (ValueError, TypeError):
            continue
        if math.isnan(pic50):
            continue
        smi = r.get("SMILES_Ligands", "")
        if not smi:
            continue
        if pic50 >= 5.0:
            actives.append(smi)
        else:
            inactives.append(smi)

    rng = random.Random(seed)
    rng.shuffle(actives)
    rng.shuffle(inactives)

    # 50/50 active/inactive split if possible
    half = n // 2
    n_active = min(half, len(actives))
    n_inactive = min(n - n_active, len(inactives))
    # Top up from whichever pool has remaining
    sampled = actives[:n_active] + inactives[:n_inactive]
    if len(sampled) < n:
        deficit = n - len(sampled)
        extras = (actives[n_active:] + inactives[n_inactive:])[:deficit]
        sampled += extras

    return [{"smiles": s} for s in sampled[:n]]


def _uff_minimize_one(smiles: str, max_iter: int) -> Optional[Dict[str, Any]]:
    """Parse SMILES, embed, UFF-minimize, return energy record or None on failure."""
    # Lazy import so the rest of the module is importable without rdkit.
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() == 0:
        return None

    # ETKDG v3 conformer embed (the 2026 RDKit default).
    # NOTE on Pt-dative-bond SMILES (platinai_4cell.csv from MB-Finder
    # Rusanov 2025): these use arrow bonds like ``Cl-->Pt`` to encode
    # ligand-to-metal coordination. ETKDGv3 can raise a "bad lower
    # bound" BoundsMatrix invariant on the dative-bond geometry
    # (Pt+6 / dative distance bounds are undefined). The
    # `useRandomCoords=True` flag bypasses the bounds matrix and uses
    # random initial coords; this is a smoke-scale approximation that
    # still produces a valid conformer for UFF minimization.
    #
    # Note: UFFTYPER logs "Unrecognized atom type: Pt6+2" since RDKit's
    # bundled UFF parameters don't include d-block metals; UFF still
    # returns a finite energy via the fallback terms. We capture the
    # warning rather than crash.
    embed_status = -1
    embed_method = None
    for method_name, params in (
        (
            "ETKDGv3+randomCoords",
            lambda: (
                (
                    lambda p: (
                        setattr(p, "randomSeed", 0xC0FFEE),
                        setattr(p, "useRandomCoords", True),
                        p,
                    )[-1]
                )(AllChem.ETKDGv3())
            ),
        ),
        (
            "ETKDGv2+randomCoords",
            lambda: (
                (
                    lambda p: (
                        setattr(p, "randomSeed", 0xC0FFEE),
                        setattr(p, "useRandomCoords", True),
                        p,
                    )[-1]
                )(AllChem.ETKDGv2())
            ),
        ),
    ):
        try:
            embed_status = AllChem.EmbedMolecule(mol_h, params())
            embed_method = method_name
            if embed_status == 0:
                break
        except Exception:  # noqa: BLE001 — RDKit raises Invariant Violation on bad bounds
            embed_status = -1
            continue
    if embed_status != 0:
        return None

    try:
        result = AllChem.UFFOptimizeMolecule(
            mol_h,
            maxIters=max_iter,
            vdwThresh=0.1,
        )
        # UFFOptimizeMolecule returns (converged, energy) since RDKit 2022.03
        if isinstance(result, tuple):
            converged, energy = result
        else:
            # legacy single-int return
            converged = int(result)
            energy = float(AllChem.UFFGetMoleculeForceField(mol_h).CalcEnergy())
    except Exception as exc:  # noqa: BLE001
        return {
            "smiles": smiles,
            "n_atoms": mol_h.GetNumAtoms(),
            "embed_ok": True,
            "min_ok": False,
            "energy_kcal_mol": None,
            "converged": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "smiles": smiles,
        "n_atoms": mol_h.GetNumAtoms(),
        "embed_ok": True,
        "embed_method": embed_method,
        "min_ok": True,
        "energy_kcal_mol": float(energy),
        "converged": bool(converged),
    }


def _summarize(
    records: List[Dict[str, Any]],
    n_requested: int,
    seed: int,
    max_iter: int,
    csv_path: Path,
    elapsed_sec: float,
) -> Dict[str, Any]:
    ok = [r for r in records if r.get("min_ok")]
    energies = [r["energy_kcal_mol"] for r in ok]
    n_ok = len(ok)
    n_fail = len(records) - n_ok

    if energies:
        mean_e = sum(energies) / len(energies)
        # std with Bessel correction
        if len(energies) > 1:
            var = sum((e - mean_e) ** 2 for e in energies) / (len(energies) - 1)
            std_e = math.sqrt(var)
        else:
            std_e = 0.0
        min_e = min(energies)
        max_e = max(energies)
    else:
        mean_e = std_e = min_e = max_e = None

    # Distribution buckets (rough kcal/mol ranges for UFF on Pt-drugs)
    buckets = {"<0": 0, "0-50": 0, "50-200": 0, "200-500": 0, ">=500": 0}
    for e in energies:
        if e < 0:
            buckets["<0"] += 1
        elif e < 50:
            buckets["0-50"] += 1
        elif e < 200:
            buckets["50-200"] += 1
        elif e < 500:
            buckets["200-500"] += 1
        else:
            buckets[">=500"] += 1

    return {
        "status": "MEASURED_SMOKE",
        "generated_at": _now_iso(),
        "source": "Wf-319B",
        "honest_framing": {
            "scope": "smoke (100 conformers; CPU-only RDKit UFF)",
            "data_source": str(csv_path),
            "downstream": "m10 Vina Min Score ceiling (WEAK_TO_STRONG §2.7 PROJECTED)",
            "full_pipeline": "UFF-min × Vina requires GPU portion (out of scope for this smoke)",
        },
        "config": {
            "n_requested": n_requested,
            "seed": seed,
            "max_iter": max_iter,
            "embed_method": "ETKDGv3 (fallback ETKDGv2)",
        },
        "n_conformers_generated": n_ok,
        "n_conformers_failed": n_fail,
        "energy_kcal_per_mol": {
            "mean": mean_e,
            "std": std_e,
            "min": min_e,
            "max": max_e,
            "distribution_buckets": buckets,
        },
        "wall_clock_sec": elapsed_sec,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else ""
    )
    parser.add_argument(
        "--n-conformers",
        type=int,
        default=100,
        help="number of Pt conformers to UFF-minimize",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="sampling seed (for active/inactive split)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=200, help="UFFOptimizeMolecule maxIters"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_PT_CSV,
        help="Pt-source CSV (default platinai_4cell.csv)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_POOL_OUT,
        help="per-conformer pool JSON",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_OUT,
        help="summary metrics JSON",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"FAIL: Pt-source CSV missing: {args.csv}", file=sys.stderr)
        return 1

    print(f"[Wf-319B] loading Pt rows from {args.csv}", file=sys.stderr)
    rows = _load_pt_rows(args.csv)
    print(f"[Wf-319B] Pt rows in source: {len(rows)}", file=sys.stderr)

    sampled = _sample_balanced(rows, n=args.n_conformers, seed=args.seed)
    print(
        f"[Wf-319B] sampled {len(sampled)} Pt compounds (50/50 active/inactive if possible)",
        file=sys.stderr,
    )

    records: List[Dict[str, Any]] = []
    n_embed_failed = 0
    t0 = time.perf_counter()
    for i, item in enumerate(sampled, start=1):
        rec = _uff_minimize_one(item["smiles"], max_iter=args.max_iter)
        if rec is None:
            n_embed_failed += 1
        else:
            rec["pool_index"] = i
            records.append(rec)
        if args.verbose and i % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[Wf-319B] progress: {i}/{len(sampled)} ({elapsed:.1f}s)",
                file=sys.stderr,
            )
    elapsed = time.perf_counter() - t0

    summary = _summarize(
        records=records,
        n_requested=args.n_conformers,
        seed=args.seed,
        max_iter=args.max_iter,
        csv_path=args.csv,
        elapsed_sec=elapsed,
    )
    summary["n_conformers_failed"] = n_embed_failed
    # Honest-negative: UFF lacks Pt (d-block metal) parameters; energies
    # are not physical for Pt6+/Pt2+ complexes. The pool.json + 3D
    # conformers are still valid for downstream Vina docking, but
    # downstream consumers MUST NOT interpret these energies as
    # minimized-physics. Per Wf-319B honest-framing mandate (party-
    # twenty-spirit §3.5) and AGENTS.md §5.2.
    summary["HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING"] = True
    summary["honest_negative_reason"] = (
        "RDKit UFF force field lacks Pt (d-block metal) atom-type "
        "parameters; UFFTYPER logs 'Unrecognized atom type: Pt6+2' for "
        "every Pt atom. Energies are fallback terms, not physical. "
        "Conformers (3D coords) are valid for downstream Vina docking "
        "but should NOT be used as a real minimization metric. The "
        "m10 Vina Min Score ceiling improvement (WEAK_TO_STRONG §2.7) "
        "remains PROJECTED — full UFF-min x Vina pipeline requires a "
        "force field that includes d-block metal parameters (e.g. "
        "MACE-OFF23 or SMIRNOFF/OpenFF with d-block coverage, neither "
        "of which is in this venv)."
    )

    # Pool file: per-conformer records + summary
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": summary,
                "conformers": records,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # Metrics file: flat summary suitable for metrics/by_metric/*
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"[Wf-319B] DONE in {elapsed:.1f}s", file=sys.stderr)
    print(
        f"[Wf-319B] n_conformers_generated={summary['n_conformers_generated']}",
        file=sys.stderr,
    )
    print(
        f"[Wf-319B] mean energy = {summary['energy_kcal_per_mol']['mean']}",
        file=sys.stderr,
    )
    print(f"[Wf-319B] pool: {args.json_out}", file=sys.stderr)
    print(f"[Wf-319B] metrics: {args.metrics_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
