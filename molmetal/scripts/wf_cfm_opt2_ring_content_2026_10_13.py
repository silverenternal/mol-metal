# SPDX-License-Identifier: Apache-2.0
"""OPT-2 measurement: ring content (generated vs real six-metal).

WHY THIS EXISTS
---------------
Six rounds chased the bond head under the assumption that getting more
aromatic BONDS was the lever for QED (ROTB) and SA. Aromatic bond fraction
is not ring fraction. A saturated 6-membered ring fixes ROTB the same way an
aromatic ring does, and the head cannot ring-close anything. Before assuming
aromatic bonding is the right target, measure ring CLOSURE itself.

This script computes:
  (a) rings per molecule                  (count, mean, median, frac with 0)
  (b) aromatic rings per molecule
  (c) ring SIZE distribution              (3, 4, 5, 6, 7, 8+)
  (d) rotatable bond count                (RDKit Lipinski.NumRotatableBonds)
  (e) largest connected FRAGMENT size as a fraction of heavy atoms
Cross-checks against metrics/by_metric/cfm_objective_inventory.json.

SCOPE
-----
Two cohorts:
  - Generated: molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv (n=66)
  - Real:      .data/molmetal/tmQM/tmQM_y.csv filtered to rows whose SMILES
               contains any of {Pt, Ir, Ru, Rh, Os, Re} (n=31150)
CPU, no GPU, no retrain. Writes one metric JSON and one report dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent.parent

GENERATED_CSV = REPO / "molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv"
REAL_TMQM = REPO / ".data/molmetal/tmQM/tmQM_y.csv"
METRIC_OUT = REPO / "metrics/by_metric/cfm_opt2_ring_content.json"
REPORT_DIR = REPO / "molmetal/reports/cfm_opt2_ring_content_2026_10_13"

SIX_METALS = ("Pt", "Ir", "Ru", "Rh", "Os", "Re")


def _parse_one(smi: str) -> Optional[Chem.Mol]:
    """Tolerant parser: sanitize=False then attempt sanitize. Return None on
    complete failure."""
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        # Keep the unsanitized mol -- ring counts from bond graph are still
        # meaningful for the cohort comparison.
        return m


def _ring_stats(m: Chem.Mol) -> Tuple[int, int, List[int]]:
    rings = m.GetRingInfo().AtomRings()
    n_rings = len(rings)
    sizes = [len(r) for r in rings]
    n_arom = sum(
        1
        for r in rings
        if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    )
    return n_rings, n_arom, sizes


def _frag_largest_frac(m: Chem.Mol) -> float:
    n_heavy = m.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 1.0
    try:
        frags = Chem.GetMolFrags(m, sanitizeFrags=False)
    except Exception:
        return 1.0
    if not frags:
        return 1.0
    frag_heavy = [sum(1 for idx in f if m.GetAtomWithIdx(idx).GetSymbol() != "H") for f in frags]
    if not frag_heavy or max(frag_heavy) == 0:
        return 1.0
    return max(frag_heavy) / n_heavy


def _rotb(m: Chem.Mol) -> int:
    try:
        return int(Lipinski.NumRotatableBonds(m))
    except Exception:
        return 0


def _per_mol_summary(cohort: List[str]) -> Dict:
    n_total = len(cohort)
    n_parse_ok = 0
    n_sanitize_ok = 0
    rings_per: List[int] = []
    arom_rings_per: List[int] = []
    rotb_per: List[int] = []
    largest_frac: List[float] = []
    ring_size_counts: Counter = Counter()
    n_fail = 0
    rows = []
    for s in cohort:
        m = _parse_one(s)
        if m is None:
            n_fail += 1
            rows.append({"smiles": s[:60], "parse": "fail"})
            continue
        n_parse_ok += 1
        try:
            Chem.SanitizeMol(m)
            n_sanitize_ok += 1
        except Exception:
            pass
        n_rings, n_arom, sizes = _ring_stats(m)
        rotb = _rotb(m)
        lfrac = _frag_largest_frac(m)
        rings_per.append(n_rings)
        arom_rings_per.append(n_arom)
        rotb_per.append(rotb)
        largest_frac.append(lfrac)
        for sz in sizes:
            ring_size_counts[sz] += 1
        rows.append({
            "smiles": s[:80],
            "n_atoms": m.GetNumAtoms(),
            "n_bonds": m.GetNumBonds(),
            "n_rings": n_rings,
            "n_arom_rings": n_arom,
            "ring_sizes": sorted(sizes),
            "rotb": rotb,
            "largest_frag_frac": round(lfrac, 4),
            "parse": "ok",
        })

    if not rings_per:
        return {"n_total": n_total, "n_parse_ok": 0, "n_fail": n_fail, "rows": rows}

    return {
        "n_total": n_total,
        "n_parse_ok": n_parse_ok,
        "n_sanitize_ok": n_sanitize_ok,
        "n_fail": n_fail,
        "rings_per_mol": {
            "mean": st.mean(rings_per),
            "median": st.median(rings_per),
            "sd": st.pstdev(rings_per) if len(rings_per) > 1 else 0.0,
            "min": min(rings_per),
            "max": max(rings_per),
            "frac_zero": sum(1 for x in rings_per if x == 0) / len(rings_per),
        },
        "aromatic_rings_per_mol": {
            "mean": st.mean(arom_rings_per),
            "median": st.median(arom_rings_per),
            "sd": st.pstdev(arom_rings_per) if len(arom_rings_per) > 1 else 0.0,
            "min": min(arom_rings_per),
            "max": max(arom_rings_per),
            "frac_with_arom_ring": sum(1 for x in arom_rings_per if x >= 1) / len(arom_rings_per),
        },
        "ring_size_distribution": {
            str(k): v for k, v in sorted(ring_size_counts.items())
        },
        "rotb_per_mol": {
            "mean": st.mean(rotb_per),
            "median": st.median(rotb_per),
            "sd": st.pstdev(rotb_per) if len(rotb_per) > 1 else 0.0,
            "min": min(rotb_per),
            "max": max(rotb_per),
        },
        "largest_frag_frac": {
            "mean": st.mean(largest_frac),
            "median": st.median(largest_frac),
            "sd": st.pstdev(largest_frac) if len(largest_frac) > 1 else 0.0,
            "min": min(largest_frac),
            "max": max(largest_frac),
            "frac_below_0_95": sum(1 for x in largest_frac if x < 0.95) / len(largest_frac),
        },
        "rows": rows,
    }


def _load_generated() -> List[str]:
    smiles: List[str] = []
    with open(GENERATED_CSV) as f:
        for r in csv.DictReader(f):
            smiles.append(r["smiles"])
    return smiles


def _load_real_six_metal() -> List[str]:
    smiles: List[str] = []
    with open(REAL_TMQM, encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter=";")
        for r in rdr:
            s = r.get("SMILES") or ""
            if not s:
                continue
            if any(m in s for m in SIX_METALS):
                smiles.append(s)
    return smiles


def _verdict(gen: Dict, real: Dict) -> Dict:
    """Decide whether the defect is connectivity closure (rings), bond typing
    (aromatic rings), RING SIZE, or fragmentation (largest_frag_frac)."""
    g_rings_mean = gen["rings_per_mol"]["mean"]
    r_rings_mean = real["rings_per_mol"]["mean"]
    g_arom_mean = gen["aromatic_rings_per_mol"]["mean"]
    r_arom_mean = real["aromatic_rings_per_mol"]["mean"]
    g_lfrac_mean = gen["largest_frag_frac"]["mean"]
    r_lfrac_mean = real["largest_frag_frac"]["mean"]
    g_zero_ring = gen["rings_per_mol"]["frac_zero"]
    r_zero_ring = real["rings_per_mol"]["frac_zero"]
    g_six = gen["ring_size_distribution"].get("6", 0)
    r_six = real["ring_size_distribution"].get("6", 0)
    g_three = gen["ring_size_distribution"].get("3", 0)
    r_three = real["ring_size_distribution"].get("3", 0)
    g_five = gen["ring_size_distribution"].get("5", 0)
    r_five = real["ring_size_distribution"].get("5", 0)
    g_total_rings = sum(gen["ring_size_distribution"].values())
    r_total_rings = sum(real["ring_size_distribution"].values())

    g_three_share = (g_three / g_total_rings) if g_total_rings else 0.0
    r_three_share = (r_three / r_total_rings) if r_total_rings else 0.0
    g_five_six_share = (
        ((g_five + g_six) / g_total_rings) if g_total_rings else 0.0
    )
    r_five_six_share = (
        ((r_five + r_six) / r_total_rings) if r_total_rings else 0.0
    )

    closing = {
        "rings_per_mol_gen_vs_real": (g_rings_mean, r_rings_mean),
        "frac_zero_rings_gen_vs_real": (g_zero_ring, r_zero_ring),
        "fraction_of_rings_that_are_6_membered_gen_vs_real": (
            (g_six / g_total_rings) if g_total_rings else 0.0,
            (r_six / r_total_rings) if r_total_rings else 0.0,
        ),
        "fraction_of_rings_that_are_3_membered_gen_vs_real": (
            g_three_share,
            r_three_share,
        ),
        "fraction_of_rings_that_are_5_or_6_membered_gen_vs_real": (
            g_five_six_share,
            r_five_six_share,
        ),
    }
    typing = {
        "aromatic_rings_per_mol_gen_vs_real": (g_arom_mean, r_arom_mean),
        "frac_molecules_with_arom_ring_gen_vs_real": (
            gen["aromatic_rings_per_mol"]["frac_with_arom_ring"],
            real["aromatic_rings_per_mol"]["frac_with_arom_ring"],
        ),
    }
    frag = {
        "largest_frag_frac_gen_vs_real": (g_lfrac_mean, r_lfrac_mean),
        "frac_below_0_95_gen_vs_real": (
            gen["largest_frag_frac"]["frac_below_0_95"],
            real["largest_frag_frac"]["frac_below_0_95"],
        ),
    }

    rings_ratio = g_rings_mean / r_rings_mean if r_rings_mean else float("nan")
    arom_ratio = g_arom_mean / r_arom_mean if r_arom_mean else float("nan")
    six_share_ratio = (g_six / g_total_rings) / (r_six / r_total_rings) if g_total_rings and r_total_rings and r_six else float("nan")

    # 3 things wrong: total ring count, ring SIZE, aromaticity. They are not
    # independent. If the rings that DO get made are nearly all the wrong size,
    # and aromatic is essentially zero, the deciding axis is the SIZE:
    # aromaticity cannot exist if you never close a hexagon, and the bond
    # head's aromatic/single ranking is irrelevant to which hexagons exist.
    if (
        g_rings_mean > 0  # some rings are being closed
        and g_three_share > 0.5  # majority are 3-mem (cyclopropanes, oxide bridges)
        and g_five_six_share < 0.2 * r_five_six_share  # 5/6 share collapsed
        and g_lfrac_mean > 0.95  # not fragmented
    ):
        deciding_axis = "RING SIZE (geometric ring template)"
        deciding_number = (
            f"fraction of 5/6-membered rings: gen={g_five_six_share:.3f} vs real={r_five_six_share:.3f} "
            f"(gen/real share-ratio = "
            f"{(g_five_six_share / r_five_six_share) if r_five_six_share else float('nan'):.4f})"
        )
    elif (1 - g_lfrac_mean) > 0.05 or gen["largest_frag_frac"]["frac_below_0_95"] > 0.1:
        deciding_axis = "FRAGMENTATION"
        deciding_number = f"largest_frag_frac mean gen={g_lfrac_mean:.4f} vs real={r_lfrac_mean:.4f}"
    elif g_rings_mean < 0.5 * r_rings_mean:
        deciding_axis = "RING CLOSURE"
        deciding_number = f"rings_per_mol ratio (gen/real) = {rings_ratio:.3f}"
    else:
        deciding_axis = "BOND TYPING (aromatic)"
        deciding_number = f"aromatic_rings_per_mol ratio (gen/real) = {arom_ratio:.3f}"

    return {
        "closing": closing,
        "typing": typing,
        "fragmentation": frag,
        "deciding_axis": deciding_axis,
        "deciding_number": deciding_number,
        "rings_ratio_gen_over_real": rings_ratio,
        "arom_ratio_gen_over_real": arom_ratio,
        "five_six_share_ratio_gen_over_real": (
            (g_five_six_share / r_five_six_share) if r_five_six_share else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-sample", type=int, default=2000,
        help="Sample size for the real cohort (default 2000; 0 = use all 31k).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    gen = _load_generated()
    real_full = _load_real_six_metal()
    if args.real_sample and args.real_sample < len(real_full):
        import random
        random.seed(args.seed)
        real = random.sample(real_full, args.real_sample)
    else:
        real = real_full

    print(
        f"generated n={len(gen)}, real six-metal n={len(real_full)} (sampled n={len(real)})",
        file=sys.stderr,
    )

    gen_s = _per_mol_summary(gen)
    real_s = _per_mol_summary(real)
    verdict = _verdict(gen_s, real_s)

    out = {
        "metric": "cfm_opt2_ring_content",
        "tag": "MEASURED",
        "date": "2026-10-13",
        "purpose": "OPT-2: ring content comparison. Generated (n=66) vs real six-metal baseline (n=31150; sampled).",
        "generated_path": str(GENERATED_CSV),
        "real_path": str(REAL_TMQM),
        "generated": gen_s,
        "real": real_s,
        "verdict": verdict,
        "cross_check_against_objective_inventory": {
            "rotb_generated_expected": 16.863636,
            "rotb_generated_measured": round(gen_s["rotb_per_mol"]["mean"], 6),
            "matches_inventory": abs(gen_s["rotb_per_mol"]["mean"] - 16.863636) < 0.5,
            "note": "ROTB in cfm_objective_inventory.json line 86 is on n=66 generated.",
        },
    }

    METRIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote metric: {METRIC_OUT}", file=sys.stderr)

    # Write a slim report dir for browsing -- exclude the row-by-row list to
    # keep it readable.
    slim = {k: v for k, v in out.items() if k != "generated"}
    slim["generated"] = {k: v for k, v in out["generated"].items() if k != "rows"}
    slim["real"] = {k: v for k, v in out["real"].items() if k != "rows"}
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(slim, f, indent=2)

    # A per-row JSONL for the generated cohort (small) and the real sample
    with open(REPORT_DIR / "generated_per_mol.jsonl", "w") as f:
        for row in gen_s["rows"]:
            f.write(json.dumps(row) + "\n")
    with open(REPORT_DIR / "real_sample_per_mol.jsonl", "w") as f:
        for row in real_s["rows"]:
            f.write(json.dumps(row) + "\n")

    # Print a one-page verdict
    v = verdict
    print("\n=== OPT-2 verdict (deciding axis) ===")
    print(v["deciding_axis"])
    print(f"reason: {v['deciding_number']}")
    print("\n=== Closing (rings) ===")
    for k, val in v["closing"].items():
        print(f"  {k}: gen={val[0]}  real={val[1]}")
    print("\n=== Typing (aromatic rings) ===")
    for k, val in v["typing"].items():
        print(f"  {k}: gen={val[0]}  real={val[1]}")
    print("\n=== Fragmentation ===")
    for k, val in v["fragmentation"].items():
        print(f"  {k}: gen={val[0]}  real={val[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
