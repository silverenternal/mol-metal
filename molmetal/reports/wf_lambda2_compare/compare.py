"""WF-Lambda-2 verify — paper-grade comparison between Homotype and Tanimoto.

Builds a 10-molecule test set:
  * 5 constitutional isomers of C6H12  — high Tanimoto similarity but
    different homotype signatures (where the typed-variable multiset
    actually differs; honest fall-back noted in the final report).
  * 5 unrelated drug-like mols (cisplatin, benzene, naphthalene, aspirin,
    caffeine) — low similarity in both metrics.

Computes:
  * Tanimoto pairwise distance matrix (Morgan fingerprint via RDKit).
  * Homotype pairwise distance matrix (via the WF-Lambda-2 module).

Prints and persists:
  * A per-pair result table for both subsets.
  * The summary metrics required by the task schema.

Honest framing — what's MEASURED vs PROJECTED:
  * MEASURED on this 10-mol, 45-pairs-each test set.
  * PROJECTED to production-scale (100k+ mols × same metric).  The
    test set is too small to draw distributional conclusions, only
    *direction* (does homotype move with / against Tanimoto on
    constitutional diversity?).

Run with: ``uv run python compare.py`` (cwd-agnostic).
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# The homotype-diversity module lives in the molmetal/molmetal_lam package.
# Insert the molmetal/ directory to sys.path so we can import
# ``molmetal_lam.metrics.homotype_diversity`` regardless of where this
# script is invoked from.
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_MOLMETAL_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _MOLMETAL_ROOT not in sys.path:
    sys.path.insert(0, _MOLMETAL_ROOT)

from molmetal_lam.metrics.homotype_diversity import (
    HomotypeSignature,
    homotype_distance,
    homotype_diversity,
)


# ---------------------------------------------------------------------------
# 10-molecule test set
# ---------------------------------------------------------------------------
# 5 constitutional isomers of C6H12 (heavy atom count = 6, all-carbon).
# These have HIGH Morgan-Tanimoto similarity because the Morgan
# fingerprint is built from substructures of the same atomic alphabet
# at the same atomic count, but the homotype metric is supposed to
# give them *higher* distance because their typed-variable multisets
# differ in connectivity-derived counts (number of H attached, ring
# closure points, etc.) when the typed-var vocabulary is enriched.
#
# Note: under the *current* typed-variable vocabulary (atomic symbols
# only) all five collapse to {"C": 6} — i.e. cos distance = 0 on the
# typed-var channel.  This is honest-fail and the report documents
# the projected enriched-vocabulary version (H-count channel,
# ring-channel).
ISOMER_SMILES = [
    ("cyclohexane", "C1CCCCC1"),
    ("hex-1-ene", "CCCCC=C"),
    ("methylcyclopentane", "CC1CCCC1"),
    ("3-methylpent-1-ene", "CCC(C)C=C"),
    ("2,3-dimethylbut-2-ene", "CC(C)=C(C)C"),
]

# 5 unrelated drug-like mols — low similarity in BOTH metrics.
UNRELATED_SMILES = [
    ("cisplatin", "[H][N]([H])([H])[Pt](Cl)(Cl)([N]([H])([H])[H])[N]([H])([H])[H]"),
    ("benzene", "c1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------
def tanimoto_distance(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """Tanimoto *distance* = 1 - Tanimoto similarity (Morgan r=2, 2048 bits)."""
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    return float(1.0 - sim)


def homotype_distance_for_mols(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """Homotype distance on RDKit mols via the WF-Lambda-2 module."""
    sig_a = HomotypeSignature.from_mol(mol_a)
    sig_b = HomotypeSignature.from_mol(mol_b)
    return float(homotype_distance(sig_a, sig_b))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_test_set() -> List[Tuple[str, str, Chem.Mol]]:
    rows: List[Tuple[str, str, Chem.Mol]] = []
    for name, smi in ISOMER_SMILES + UNRELATED_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse {name}: {smi}")
        rows.append((name, smi, mol))
    return rows


def pairwise(rows: List[Tuple[str, str, Chem.Mol]]) -> List[Dict]:
    out: List[Dict] = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            ni, si, mi = rows[i]
            nj, sj, mj = rows[j]
            tan_d = tanimoto_distance(mi, mj)
            hom_d = homotype_distance_for_mols(mi, mj)
            out.append({
                "i": ni,
                "j": nj,
                "smi_i": si,
                "smi_j": sj,
                "tanimoto_distance": tan_d,
                "homotype_distance": hom_d,
            })
    return out


def subset(pairs: List[Dict], names: List[str]) -> List[Dict]:
    nset = set(names)
    return [p for p in pairs if p["i"] in nset and p["j"] in nset]


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def main() -> Dict:
    rows = build_test_set()
    pairs = pairwise(rows)
    iso_names = [n for n, _ in ISOMER_SMILES]
    unrel_names = [n for n, _ in UNRELATED_SMILES]
    iso_pairs = subset(pairs, iso_names)
    unrel_pairs = subset(pairs, unrel_names)

    iso_tan = [p["tanimoto_distance"] for p in iso_pairs]
    iso_hom = [p["homotype_distance"] for p in iso_pairs]
    unrel_tan = [p["tanimoto_distance"] for p in unrel_pairs]
    unrel_hom = [p["homotype_distance"] for p in unrel_pairs]

    metrics = {
        "n_pairs": len(pairs),
        "n_iso_pairs": len(iso_pairs),
        "n_unrelated_pairs": len(unrel_pairs),
        "mean_tanimoto_isomers": mean(iso_tan),
        "mean_homotype_isomers": mean(iso_hom),
        "mean_tanimoto_unrelated": mean(unrel_tan),
        "mean_homotype_unrelated": mean(unrel_hom),
        "homotype_exceeds_tanimoto_on_isomers": mean(iso_hom) > mean(iso_tan),
        "homotype_matches_tanimoto_on_unrelated": abs(mean(unrel_hom) - mean(unrel_tan)) < 0.15,
    }

    # Persist raw tables for the report.
    here = os.path.dirname(os.path.abspath(__file__))
    table_path = os.path.join(here, "pairs.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset", "i", "j", "smi_i", "smi_j",
            "tanimoto_distance", "homotype_distance",
            "homotype_minus_tanimoto",
        ])
        for p in iso_pairs:
            writer.writerow([
                "isomers", p["i"], p["j"], p["smi_i"], p["smi_j"],
                f"{p['tanimoto_distance']:.4f}",
                f"{p['homotype_distance']:.4f}",
                f"{p['homotype_distance'] - p['tanimoto_distance']:+.4f}",
            ])
        for p in unrel_pairs:
            writer.writerow([
                "unrelated", p["i"], p["j"], p["smi_i"], p["smi_j"],
                f"{p['tanimoto_distance']:.4f}",
                f"{p['homotype_distance']:.4f}",
                f"{p['homotype_distance'] - p['tanimoto_distance']:+.4f}",
            ])

    metrics_path = os.path.join(here, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary.
    print("== WF-Lambda-2 verify: 10-mol dual-distance comparison ==")
    print(f"Total pairs: {len(pairs)}")
    print(f"  - Isomer subset: {len(iso_pairs)} pairs")
    print(f"  - Unrelated subset: {len(unrel_pairs)} pairs")
    print()
    print("Per-pair tables:")
    print()
    print("-- ISOMERS (C6H12) --")
    print(f"{'pair':<60}  {'tanimoto':>9}  {'homotype':>9}  {'Δ':>7}")
    for p in iso_pairs:
        label = f"{p['i']} vs {p['j']}"
        d = p["homotype_distance"] - p["tanimoto_distance"]
        print(f"{label:<60}  {p['tanimoto_distance']:>9.4f}  {p['homotype_distance']:>9.4f}  {d:>+7.4f}")
    print()
    print("-- UNRELATED drug-like mols --")
    print(f"{'pair':<60}  {'tanimoto':>9}  {'homotype':>9}  {'Δ':>7}")
    for p in unrel_pairs:
        label = f"{p['i']} vs {p['j']}"
        d = p["homotype_distance"] - p["tanimoto_distance"]
        print(f"{label:<60}  {p['tanimoto_distance']:>9.4f}  {p['homotype_distance']:>9.4f}  {d:>+7.4f}")
    print()
    print("-- Aggregate metrics --")
    for k, v in metrics.items():
        if isinstance(v, bool):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")
    print()
    print(f"Per-pair CSV:    {table_path}")
    print(f"Metrics JSON:    {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()