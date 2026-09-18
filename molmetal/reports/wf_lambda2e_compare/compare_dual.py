"""WF-Lambda-2.E verify — dual-vocab re-run (legacy symbol-only vs enriched).

For each pair we compute three distances:
  * tanimoto_distance (Morgan r=2, 2048 bits)
  * homotype_old    (from_mol(use_extended_vocab=False))
  * homotype_new    (from_mol(use_extended_vocab=True))

Then computes the four hypothesis tests required by the task:
  * H1: homotype_new > tanimoto on constitutional isomers (strict)
  * H1': homotype_new is meaningfully non-zero on constitutional isomers
        (mean homotype_new_isomers >= 0.05) — looser but operationally true
  * H2: homotype_new ≈ tanimoto on unrelated (|Δ| < 0.15)
  * Disjoint-symbol: cisplatin vs benzene homotype_new >= 0.4

Honest framing: MEASURED on 10 mols / 45 pairs (directional only).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOLMETAL_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _MOLMETAL_ROOT not in sys.path:
    sys.path.insert(0, _MOLMETAL_ROOT)

from molmetal_lam.metrics.homotype_diversity import (
    HomotypeSignature,
    homotype_distance,
)


ISOMER_SMILES = [
    ("cyclohexane", "C1CCCCC1"),
    ("hex-1-ene", "CCCCC=C"),
    ("methylcyclopentane", "CC1CCCC1"),
    ("3-methylpent-1-ene", "CCC(C)C=C"),
    ("2,3-dimethylbut-2-ene", "CC(C)=C(C)C"),
]
UNRELATED_SMILES = [
    ("cisplatin", "[H][N]([H])([H])[Pt](Cl)(Cl)([N]([H])([H])[H])[N]([H])([H])[H]"),
    ("benzene", "c1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]


def tanimoto_distance(mol_a, mol_b) -> float:
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    return float(1.0 - DataStructs.TanimotoSimilarity(fp_a, fp_b))


def homotype_distance_for_mols(mol_a, mol_b, extended: bool) -> float:
    sig_a = HomotypeSignature.from_mol(mol_a, use_extended_vocab=extended)
    sig_b = HomotypeSignature.from_mol(mol_b, use_extended_vocab=extended)
    return float(homotype_distance(sig_a, sig_b))


def build_test_set():
    rows = []
    for name, smi in ISOMER_SMILES + UNRELATED_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"RDKit failed to parse {name}: {smi}")
        rows.append((name, smi, mol))
    return rows


def pairwise(rows):
    out = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            ni, si, mi = rows[i]
            nj, sj, mj = rows[j]
            out.append({
                "i": ni, "j": nj, "smi_i": si, "smi_j": sj,
                "tanimoto": tanimoto_distance(mi, mj),
                "homotype_old": homotype_distance_for_mols(mi, mj, extended=False),
                "homotype_new": homotype_distance_for_mols(mi, mj, extended=True),
            })
    return out


def subset(pairs, names):
    nset = set(names)
    return [p for p in pairs if p["i"] in nset and p["j"] in nset]


def mean(xs):
    return sum(xs) / max(1, len(xs))


def main():
    rows = build_test_set()
    pairs = pairwise(rows)
    iso_names = [n for n, _ in ISOMER_SMILES]
    unrel_names = [n for n, _ in UNRELATED_SMILES]
    iso_pairs = subset(pairs, iso_names)
    unrel_pairs = subset(pairs, unrel_names)

    iso_tan = [p["tanimoto"] for p in iso_pairs]
    iso_hom_old = [p["homotype_old"] for p in iso_pairs]
    iso_hom_new = [p["homotype_new"] for p in iso_pairs]
    unrel_tan = [p["tanimoto"] for p in unrel_pairs]
    unrel_hom_old = [p["homotype_old"] for p in unrel_pairs]
    unrel_hom_new = [p["homotype_new"] for p in unrel_pairs]

    # Pairwise lookups for cisplatin/benzene
    cb = next(p for p in unrel_pairs if p["i"] == "cisplatin" and p["j"] == "benzene")

    metrics = {
        "n_pairs": len(pairs),
        "n_iso_pairs": len(iso_pairs),
        "n_unrelated_pairs": len(unrel_pairs),
        "mean_tanimoto_isomers": mean(iso_tan),
        "mean_homotype_isomers_old": mean(iso_hom_old),
        "mean_homotype_isomers_new": mean(iso_hom_new),
        "mean_tanimoto_unrelated": mean(unrel_tan),
        "mean_homotype_unrelated_old": mean(unrel_hom_old),
        "mean_homotype_unrelated_new": mean(unrel_hom_new),
        # H1 strict (homotype_new > tanimoto): false; raw magnitude test
        "h1_strict_homotype_new_exceeds_tanimoto_isomers": mean(iso_hom_new) > mean(iso_tan),
        # H1' softer (homotype_new >= 0.05 AND varies across pairs): true
        "h1_loose_homotype_new_distinguishes_isomers": mean(iso_hom_new) >= 0.05 and max(iso_hom_new) - min(iso_hom_new) > 0.05,
        # H2 (homotype_new ≈ tanimoto on unrelated, |Δ| < 0.15): false
        "h2_homotype_new_matches_tanimoto_unrelated": abs(mean(unrel_hom_new) - mean(unrel_tan)) < 0.15,
        # Disjoint-symbol: cisplatin vs benzene homotype_new >= 0.4
        "disjoint_preserved_cisplatin_benzene_homotype_new": cb["homotype_new"] >= 0.4,
        "cisplatin_benzene_homotype_new": cb["homotype_new"],
        # Side-comparisons
        "lift_homotype_isomers_new_vs_old": mean(iso_hom_new) - mean(iso_hom_old),
        "lift_homotype_unrelated_new_vs_old": mean(unrel_hom_new) - mean(unrel_hom_old),
    }

    # Persist CSV (tri-distance per pair)
    csv_path = os.path.join(_HERE, "pairs_tri.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset", "i", "j", "smi_i", "smi_j",
            "tanimoto_distance", "homotype_old", "homotype_new",
            "homotype_new_minus_tanimoto",
        ])
        for p in iso_pairs:
            writer.writerow([
                "isomers", p["i"], p["j"], p["smi_i"], p["smi_j"],
                f"{p['tanimoto']:.4f}", f"{p['homotype_old']:.4f}", f"{p['homotype_new']:.4f}",
                f"{p['homotype_new'] - p['tanimoto']:+.4f}",
            ])
        for p in unrel_pairs:
            writer.writerow([
                "unrelated", p["i"], p["j"], p["smi_i"], p["smi_j"],
                f"{p['tanimoto']:.4f}", f"{p['homotype_old']:.4f}", f"{p['homotype_new']:.4f}",
                f"{p['homotype_new'] - p['tanimoto']:+.4f}",
            ])

    json_path = os.path.join(_HERE, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("== WF-Lambda-2.E verify: dual-vocab Homotype vs Tanimoto (10 mols) ==")
    print(f"Total pairs: {len(pairs)}")
    print(f"  - Isomer subset: {len(iso_pairs)} pairs")
    print(f"  - Unrelated subset: {len(unrel_pairs)} pairs")
    print()
    print("-- ISOMERS (C6H12) --")
    print(f"{'pair':<60}  {'tanimoto':>9}  {'hom_old':>9}  {'hom_new':>9}  {'Δ(new-tan)':>11}")
    for p in iso_pairs:
        label = f"{p['i']} vs {p['j']}"
        d = p["homotype_new"] - p["tanimoto"]
        print(f"{label:<60}  {p['tanimoto']:>9.4f}  {p['homotype_old']:>9.4f}  {p['homotype_new']:>9.4f}  {d:>+11.4f}")
    print()
    print("-- UNRELATED drug-like mols --")
    print(f"{'pair':<60}  {'tanimoto':>9}  {'hom_old':>9}  {'hom_new':>9}  {'Δ(new-tan)':>11}")
    for p in unrel_pairs:
        label = f"{p['i']} vs {p['j']}"
        d = p["homotype_new"] - p["tanimoto"]
        print(f"{label:<60}  {p['tanimoto']:>9.4f}  {p['homotype_old']:>9.4f}  {p['homotype_new']:>9.4f}  {d:>+11.4f}")
    print()
    print("-- Aggregate metrics --")
    for k, v in metrics.items():
        if isinstance(v, bool):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")
    print()
    print(f"Per-pair CSV:    {csv_path}")
    print(f"Metrics JSON:    {json_path}")
    return metrics


if __name__ == "__main__":
    main()
