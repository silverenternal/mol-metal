#!/usr/bin/env python3
"""Measure (a) — What fraction of truth-aromatic bonds are METAL-BOUND?

Following the test scaffold in molmetal/tests/test_wf_blkc_qed_lever_aromatic_head.py:
- iterate the tmQM BO shards
- per complex, build the (i, j, bo) truth pair list
- restrict to non-H heavy-atom pairs (same scope as bench_tmqm_bond_perception.py)
- count truth-aromatic bonds (BO in [1.3, 1.7))
- count how many involve a metal atom (Pt/Ru/Ir/Rh/Os/Re)
- count how many of the AROMATIC->SINGLE predicted collapses involve a metal-bound pair
  (the (A)-defect "70.7% collapse" reported in docs/CFM_STATUS_2026_10_12.md BLK-C-OBJ)

Output: JSON dump with measured counts to stdout.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TMQM_DIR = REPO / ".data" / "molmetal" / "tmQM"
BO_FILES = (
    TMQM_DIR / "tmQM_X1.BO.gz",
    TMQM_DIR / "tmQM_X2.BO.gz",
    TMQM_DIR / "tmQM_X3.BO.gz",
)

AROMATIC_LO = 1.3
AROMATIC_HI = 1.7
TRAIN_METALS = frozenset({"Pt", "Ru", "Ir", "Rh", "Os", "Re"})
METAL_Z = {
    "Pt": 78, "Ru": 44, "Ir": 77, "Rh": 45, "Os": 76, "Re": 75,
}


def iter_bo_blocks(bo_paths):
    """Yield (csd_code, [(idx, sym, [(nbr_idx, bo), ...]), ...]) per complex."""
    for path in bo_paths:
        if not Path(path).is_file():
            continue
        code, atoms = None, []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith("CSD_code"):
                    if code is not None and atoms:
                        yield code, atoms
                    code = line.split("=", 1)[1].split("|", 1)[0].strip()
                    atoms = []
                    continue
                parts = line.split()
                if len(parts) < 3:
                    if code is not None and atoms:
                        yield code, atoms
                        code, atoms = None, []
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                rest = parts[3:]
                nbrs = []
                for k in range(0, len(rest) - 2, 3):
                    try:
                        nbrs.append((int(rest[k + 1]), float(rest[k + 2])))
                    except (ValueError, IndexError):
                        continue
                atoms.append((idx, parts[1], nbrs))
        if code is not None and atoms:
            yield code, atoms


def ground_truth_bonds(atoms, threshold=0.3):
    """Return {(min_i1b, max_1b): bo} for all pairs with bo >= threshold."""
    bonds = {}
    for idx, _sym, nbrs in atoms:
        for nbr_idx, bo in nbrs:
            if bo < threshold or nbr_idx == idx:
                continue
            key = (min(idx, nbr_idx), max(idx, nbr_idx))
            if key not in bonds or bo > bonds[key]:
                bonds[key] = bo
    return bonds


def main():
    if not all(p.is_file() for p in BO_FILES):
        print("ERROR: tmQM BO shards not on disk:", BO_FILES, file=sys.stderr)
        sys.exit(2)

    n_complexes_scanned = 0
    n_complexes_with_metal = 0
    n_aromatic_truth_total = 0        # BO in [1.3, 1.7), in scope (non-H heavy-atom)
    n_aromatic_truth_metal_bound = 0  # of those, at least one endpoint is a metal
    n_aromatic_truth_organic = 0      # of those, both endpoints organic
    n_aromatic_metal_per_metal = {m: 0 for m in TRAIN_METALS}
    n_aromatic_metal_bound_pairs_total = 0  # one entry per pair (i,j) with one metal end
    n_pairs_total_in_scope = 0

    # aromatic->single collapse counts (replay meta-report):
    # truth-aromatic bonds that the bench test recorded as predicted single
    # = 11,209 of 15,851 = 70.7% per the docstring.
    # We cannot run the head here (CPU 50-step smoke budget), so report the
    # counts of truth-aromatic pairs BY metal involvement — this is the
    # upper bound on how many of the 11,209 could have a metal endpoint.
    # If the answer is small, the 70.7% is an organic-organic defect (the
    # head fails on aromatic C-C within the ligand), NOT a metal-label defect.
    # If the answer is large, the 70.7% is contaminated by a population that
    # is in the test scope but labelled with the wrong vocabulary (dative).

    by_complex_aromatic = []  # per-complex truth aromatic counts

    for code, atoms in iter_bo_blocks(BO_FILES):
        n_complexes_scanned += 1
        syms = [a[1] for a in atoms]
        if not any(s in TRAIN_METALS for s in syms):
            continue
        n_complexes_with_metal += 1

        bonds = ground_truth_bonds(atoms, threshold=0.3)

        # scope: drop hydrogens, drop metal-metal (covered elsewhere)
        per_complex_arom = 0
        per_complex_arom_metal = 0
        per_complex_arom_organic = 0
        per_complex_arom_metal_bound_pair = 0
        for (i, j), bo in bonds.items():
            si, sj = syms[i - 1], syms[j - 1]
            if si == "H" or sj == "H":
                continue
            n_pairs_total_in_scope += 1
            is_metal_i = si in TRAIN_METALS
            is_metal_j = sj in TRAIN_METALS
            if is_metal_i and is_metal_j:
                # metal-metal contact (rare, e.g. Pt-Pt); skip
                continue
            if not (AROMATIC_LO <= bo < AROMATIC_HI):
                continue
            # truth-aromatic, in scope
            n_aromatic_truth_total += 1
            per_complex_arom += 1
            if is_metal_i or is_metal_j:
                n_aromatic_truth_metal_bound += 1
                per_complex_arom_metal += 1
                per_complex_arom_metal_bound_pair += 1
                # which metal?
                metal_sym = si if is_metal_i else sj
                if metal_sym in n_aromatic_metal_per_metal:
                    n_aromatic_metal_per_metal[metal_sym] += 1
            else:
                n_aromatic_truth_organic += 1
                per_complex_arom_organic += 1
        by_complex_aromatic.append({
            "code": code,
            "n_aromatic": per_complex_arom,
            "n_aromatic_metal_bound": per_complex_arom_metal,
            "n_aromatic_organic": per_complex_arom_organic,
        })

    # Summary
    total = n_aromatic_truth_total
    metal_share = (
        n_aromatic_truth_metal_bound / total if total else 0.0
    )
    organic_share = (
        n_aromatic_truth_organic / total if total else 0.0
    )

    out = {
        "scope": "non-H heavy-atom pairs from 798-complex cohort (per docs/CFM_STATUS_2026_10_12.md BLK-C-OBJ); re-scanned over the entire tmQM corpus restricted to {Pt,Ru,Ir,Rh,Os,Re}",
        "aromatic_band": [AROMATIC_LO, AROMATIC_HI],
        "bo_threshold_bonded": 0.3,
        "n_complexes_scanned": n_complexes_scanned,
        "n_complexes_with_target_metal": n_complexes_with_metal,
        "n_pairs_in_scope_total": n_pairs_total_in_scope,
        "n_aromatic_truth_total": n_aromatic_truth_total,
        "n_aromatic_truth_metal_bound": n_aromatic_truth_metal_bound,
        "n_aromatic_truth_organic": n_aromatic_truth_organic,
        "metal_share_of_aromatic_truth": metal_share,
        "organic_share_of_aromatic_truth": organic_share,
        "n_aromatic_metal_bound_per_metal": n_aromatic_metal_per_metal,
        "docstring_70p7_collapse_aromatic_to_single": 11209,
        "docstring_aromatic_truth_total": 15851,
        "max_metal_involved_in_70p7_collapse": n_aromatic_truth_metal_bound,
        "verdict_a": (
            f"Metal-bound pairs account for {metal_share:.4f} of the truth-aromatic "
            f"bonds ({n_aromatic_truth_metal_bound}/{n_aromatic_truth_total}); "
            f"{(1.0 - metal_share) * 100:.1f}% of truth-aromatic bonds are "
            f"organic-organic. The 70.7% aromatic->single collapse is therefore "
            f"DOMINATED by organic-organic pairs and CANNOT be primarily caused "
            f"by a metal-label defect."
            if metal_share < 0.10
            else (
                f"Metal-bound pairs account for {metal_share:.4f} of truth-aromatic "
                f"bonds ({n_aromatic_truth_metal_bound}/{n_aromatic_truth_total}); "
                f"this is a SIGNIFICANT share and a metal-label defect could be "
                f"responsible for a meaningful slice of the 70.7% collapse."
            )
        ),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()