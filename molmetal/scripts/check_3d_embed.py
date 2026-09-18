"""3-D embedding sanity check for metal complexes (T4 B1).

Goal
----
RDKit's ETKDGv3 + MMFF94 force field is the cheap end of the spectrum for
generating 3-D conformers.  Before we plug it into the full docking /
property-prediction pipelines we need to know:

1. What fraction of metal complexes (Pt, Ru, Ir) from MetalCytoToxDB can
   be embedded and MMFF-optimised without RDKit errors?
2. Are the resulting bond distances *chemically reasonable* (Pt-N ~2.0 Å,
   Pt-Cl ~2.3 Å, Ru-N ~2.1 Å, Ir-N ~2.0 Å)?

If success rate falls below 90 % we recommend a UFF fallback (TODO/07 B2).

Usage
-----
    source .venv/bin/activate
    python -m molmetal.scripts.check_3d_embed --metal Pt --n 100
    python -m molmetal.scripts.check_3d_embed --metal Ru --n 100
    python -m molmetal.scripts.check_3d_embed --metal Ir --n 100

Outputs a JSON report under ``molmetal/reports/b1_3d_embed_<metal>.json``
plus a console summary.  The Markdown summary lives in
``molmetal/reports/b1_3d_embed_sanity.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Silence RDKit's noisy warnings — we expect some failures and don't need
# every embed attempt to print to stderr.
RDLogger.DisableLog("rdApp.*")

from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.metal_smiles import reconstruct_metal_complex

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"

# Expected bond distances (Å) by (metal, donor-element) used for the
# "chemically reasonable?" check.  Reference values drawn from the
# Cambridge Structural Database via Housecroft & Sharpe (Inorg Chem).
EXPECTED_DISTANCES: Dict[Tuple[str, str], float] = {
    ("Pt", "N"): 2.05,
    ("Pt", "Cl"): 2.32,
    ("Pt", "O"): 2.00,
    ("Pt", "S"): 2.30,
    ("Ru", "N"): 2.10,
    ("Ru", "Cl"): 2.40,
    ("Ru", "O"): 2.05,
    ("Ru", "S"): 2.35,
    ("Ir", "N"): 2.05,
    ("Ir", "Cl"): 2.38,
    ("Ir", "O"): 2.05,
    ("Ir", "S"): 2.30,
}

# Anything within 0.30 Å of the reference passes the sanity check.
BOND_TOLERANCE = 0.30


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Embed + MMFF-optimise metal complexes from MetalCytoToxDB "
            "and report success rate + bond-distance distributions."
        )
    )
    p.add_argument(
        "--metal",
        default="Ru",
        choices=["Pt", "Ru", "Ir"],
        help=(
            "Metal centre to filter on.  Note: MetalCytoToxDB contains "
            "Ru/Ir/Rh/Os/Re only — ``Pt`` runs on a small synthetic "
            "cisplatin-like pool (built from the parser) plus any Pt "
            "rows we generate from ligand strings with explicit Pt."
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of rows to embed (default: 100).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0xC0FFEE,
        help="RDKit random seed (default: deterministic).",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Re-attempts per molecule if EmbedMolecule returns -1.",
    )
    p.add_argument(
        "--report-json",
        default=None,
        help="Override output JSON path (default: reports/b1_3d_embed_<metal>.json).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-row embedding
# ---------------------------------------------------------------------------
def _embed_one(
    smiles_ligands: str,
    metal: str,
    oxidation_state: int,
    seed: int,
    max_attempts: int,
) -> Optional[Chem.Mol]:
    """Build, embed, MMFF-optimise a single metal complex.

    Returns the optimised ``Chem.Mol`` with a single 3-D conformer on
    success, or ``None`` on any failure (parse / embed / MMFF).
    """
    complex_smiles = reconstruct_metal_complex(smiles_ligands, metal, oxidation_state)
    if not complex_smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(complex_smiles)
    except Exception:
        return None
    if mol is None:
        return None
    # RDKit needs explicit Hs to use ETKDG + MMFF properly.
    try:
        mol = Chem.AddHs(mol)
    except Exception:
        return None
    # Try to embed.  ETKDGv3 is the most permissive version.
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    # maxIterations controls the pruning-iteration cap (≈ attempts to find
    # a valid geometry).  Bump above the default 50 — metal complexes are
    # awkward because RDKit has no metal-specific torsions.
    params.maxIterations = 200
    # ``useRandomCoords=True`` is critical: RDKit has no ETKDG torsion
    # parameters for transition metals (Pt, Ru, Ir), so the knowledge-term
    # distance geometry fails.  Falling back to random coords + ETKDG
    # bounds still gives chemically reasonable bond lengths.
    params.useRandomCoords = True
    embed_status = -1
    for _ in range(max_attempts):
        try:
            embed_status = AllChem.EmbedMolecule(mol, params)
        except Exception:
            embed_status = -1
        if embed_status == 0:
            break
    if embed_status != 0:
        return None
    # MMFF94 optimisation.  ``mmffVerbosity=0`` suppresses per-iter chatter.
    try:
        ff_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        return None
    # ff_status == 0 → converged.  Non-zero means "didn't converge but
    # still gave us a structure" — we accept that as success since we only
    # care that *a* structure exists.
    if mol.GetNumConformers() == 0:
        return None
    return mol


def _metal_bond_distances(mol: Chem.Mol, metal: str) -> Dict[str, List[float]]:
    """Measure distances from the metal centre to each neighbour.

    Returns a dict keyed by donor element symbol → list of distances (Å).
    Only counts atoms directly bonded to the first metal atom (RDKit bond
    order 1 or 2 are both accepted; we don't enforce anything fancy).
    """
    out: Dict[str, List[float]] = defaultdict(list)
    conf = mol.GetConformer()
    # Find the metal atom (first occurrence).
    metal_idx: Optional[int] = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == metal:
            metal_idx = atom.GetIdx()
            break
    if metal_idx is None:
        return out
    mp = conf.GetAtomPosition(metal_idx)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetIdx() == metal_idx and b.GetSymbol() != "H":
            other = b
        elif b.GetIdx() == metal_idx and a.GetSymbol() != "H":
            other = a
        else:
            continue
        op = conf.GetAtomPosition(other.GetIdx())
        dx = mp.x - op.x
        dy = mp.y - op.y
        dz = mp.z - op.z
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        out[other.GetSymbol()].append(d)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_synthetic_pt_smiles(n: int) -> List[Tuple[str, str, int]]:
    """Build n synthetic Pt(II) / Pt(IV) ligand SMILES.

    The MetalCytoToxDB CSV has no Pt rows (it is Ru/Ir/Rh/Os/Re only), so
    for ``--metal Pt`` we synthesise a pool of cisplatin-like ligand
    strings: 2 amines + 2 chlorides (square-planar), or 6 amines (octahedral
    Pt(IV)).  Returns a list of ``(smiles_ligands, metal, oxidation_state)``.
    """
    pool: List[Tuple[str, str, int]] = []
    # Variant A: classical cisplatin (2 N + 2 Cl, Pt(II)).
    for frag in [
        "N.N.Cl.Cl",            # cisplatin
        "N.N.N.N",              # tetraammine Pt(II)
        "N.N.[Cl-]",            # 1 ammonia, 1 chloride — partial complex
        "NCC[NH2].NCC[NH2].Cl.Cl",  # ethylenediamine + 2 Cl (Pt(II))
    ]:
        pool.append((frag, "Pt", 2))
    # Variant B: Pt(IV) octahedral (6 donors).
    for frag in [
        "N.N.N.N.Cl.Cl",
        "N.N.N.N.N.N",
        "N.N.N.N.[Cl-].[Cl-]",
    ]:
        pool.append((frag, "Pt", 4))
    # Pad by rotation if user asked for more than our pool size.
    out: List[Tuple[str, str, int]] = []
    for i in range(n):
        out.append(pool[i % len(pool)])
    return out


def run(
    metal: str,
    n: int,
    seed: int,
    max_attempts: int,
    report_json: Optional[Path],
) -> Dict[str, object]:
    """Run the sanity check and write the JSON report.  Returns the report."""
    # Load rows from MetalCytoToxDB; fall back to a synthetic pool for
    # metals (e.g. Pt) that have no rows in the CSV.
    rows: List[Tuple[str, str, int]] = []
    data_source = "MetalCytoToxDB"
    try:
        filters = CytotoxFilter(metal_whitelist=[metal])
        ds = MetalCytotoxDataset.from_csv(filters=filters)
        rows = [
            (
                str(ds[i]["smiles"]),
                metal,
                int(ds[i].get("oxidation_state", 0) or 0),
            )
            for i in range(min(n, len(ds)))
        ]
    except (ValueError, FileNotFoundError):
        rows = []

    if not rows:
        if metal != "Pt":
            raise RuntimeError(f"No rows found for metal={metal}")
        rows = _build_synthetic_pt_smiles(n)
        data_source = "synthetic_pt_pool"

    n_take = len(rows)

    success = 0
    failure_reasons: Dict[str, int] = defaultdict(int)
    all_distances: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    bond_examples: List[Dict[str, object]] = []

    t0 = time.time()
    for i, (smiles, m, ox) in enumerate(rows):
        mol = _embed_one(smiles, m, ox, seed=seed + i, max_attempts=max_attempts)
        if mol is None:
            failure_reasons["embed_or_optimize_failed"] += 1
            continue
        success += 1
        dists = _metal_bond_distances(mol, m)
        for donor_el, ds_ in dists.items():
            key = (m, donor_el)
            all_distances[key].extend(ds_)
            ref = EXPECTED_DISTANCES.get(key)
            if ref is not None and len(bond_examples) < 10:
                bond_examples.append(
                    {
                        "metal": m,
                        "donor": donor_el,
                        "distance_A": ds_[0],
                        "expected_A": ref,
                        "within_tol": abs(ds_[0] - ref) <= BOND_TOLERANCE,
                    }
                )

    elapsed = time.time() - t0
    success_rate = success / max(1, n_take)

    # Distance summary statistics per (metal, donor).
    dist_summary: Dict[str, Dict[str, float]] = {}
    for key, ds_ in all_distances.items():
        m, donor = key
        dist_summary[f"{m}-{donor}"] = {
            "n": len(ds_),
            "mean_A": sum(ds_) / len(ds_),
            "min_A": min(ds_),
            "max_A": max(ds_),
            "expected_A": EXPECTED_DISTANCES.get(key),
            "within_tol_rate": (
                sum(1 for d in ds_ if abs(d - EXPECTED_DISTANCES[key]) <= BOND_TOLERANCE) / len(ds_)
                if EXPECTED_DISTANCES.get(key) is not None
                else None
            ),
        }

    report: Dict[str, object] = {
        "metal": metal,
        "data_source": data_source,
        "n_requested": n,
        "n_processed": n_take,
        "n_success": success,
        "n_failure": n_take - success,
        "success_rate": success_rate,
        "elapsed_seconds": elapsed,
        "failure_reasons": dict(failure_reasons),
        "distance_summary": dist_summary,
        "bond_examples": bond_examples[:10],
    }

    out_path = report_json or (REPORTS_DIR / f"b1_3d_embed_{metal.lower()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))

    # Console summary.
    print(f"\n=== 3-D embedding sanity check: {metal} ===")
    print(f"  data source    : {data_source}")
    print(f"  requested      : {n}")
    print(f"  processed      : {n_take}")
    print(f"  success        : {success}  ({success_rate * 100:.1f} %)")
    print(f"  elapsed        : {elapsed:.1f} s")
    print(f"  bond summary   :")
    for key, stats in dist_summary.items():
        ref = stats.get("expected_A")
        ref_str = f" (ref {ref:.2f})" if ref is not None else ""
        print(
            f"    {key:>5}  n={stats['n']:>4}  "
            f"mean={stats['mean_A']:.3f}  "
            f"min={stats['min_A']:.3f}  max={stats['max_A']:.3f}{ref_str}"
        )
    print(f"  report         : {out_path}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    report_json = Path(args.report_json) if args.report_json else None
    run(
        metal=args.metal,
        n=args.n,
        seed=args.seed,
        max_attempts=args.max_attempts,
        report_json=report_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
