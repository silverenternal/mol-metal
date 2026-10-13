#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the OPT-1 stereocentre numbers from disk, with a size-matched control.

WHY THIS SCRIPT EXISTS
----------------------
Round 8's OPT-1 feasibility probe concluded "SA moves the WRONG WAY under
every intervention that lowers n_chiral" and closed OPT-1 as
INFEASIBLE_AT_DECODE_TIME. That conclusion came from
``molmetal/eval/scoring/sascorer.py``, a vendored fork whose fragment term
is a hardcoded constant (see ``molmetal/tests/test_sascorer_fork_divergence.py``:
the fork folds its Morgan fingerprint to 2048 bits while the Ertl table is
keyed on sparse 32-bit ids, so 0 of 66 molecules match a single fragment).

Under the canonical scorer the sign reverses. This script produces the
numbers so the reversal is reproducible rather than asserted.

THE SIZE-MATCHED CONTROL
------------------------
The obvious attack on "remove_atom lifts SA" is that SA carries a size
penalty (``nAtoms**1.005 - nAtoms``) and the intervention deletes 6.26
atoms per molecule, so any atom deletion would look good. The control drops
the SAME COUNT of atoms chosen from the non-stereocentre atoms instead. Any
SA difference between the two arms is attributable to which atoms went,
not how many.

Usage
-----
    .venv/bin/python molmetal/scripts/wf_cfm_opt1_stereo_suppression_2026_10_13.py
    .venv/bin/python molmetal/scripts/wf_cfm_opt1_stereo_suppression_2026_10_13.py --out-json PATH
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

from rdkit import Chem, RDConfig
from rdkit.Chem import QED

REPO = Path(__file__).resolve().parent.parent.parent
COHORT = REPO / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10" / "generated_smiles.csv"
DEFAULT_OUT = REPO / "metrics" / "by_metric" / "cfm_opt1_stereo_suppression.json"
FORK = REPO / "molmetal" / "eval" / "scoring" / "sascorer.py"


def load_canonical_sascorer():
    """RDKit's bundled Ertl scorer, by path.

    Loaded by path rather than by sys.path so that a stray ``sascorer`` on
    sys.path -- including this repo's own broken fork -- cannot shadow it.
    That shadowing is the exact failure this script exists to correct.
    """
    path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    spec = importlib.util.spec_from_file_location("_canonical_sa", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical sascorer from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.readFragmentScores(str(path.parent / "fpscores.pkl.gz"))
    return mod


def read_cohort(path: Path = COHORT) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"cohort not on disk: {path}")
    out = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("smiles") or "").strip()
            if v:
                out.append(v)
    return out


def drop_atoms(mol: Chem.Mol, idxs: Sequence[int]) -> Optional[Chem.Mol]:
    """Remove atoms by index, highest first, then sanitise."""
    if not idxs:
        return mol
    rw = Chem.RWMol(mol)
    for i in sorted(set(idxs), reverse=True):
        try:
            rw.RemoveAtom(i)
        except Exception:
            return None
    out = rw.GetMol()
    if out.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def stereocentre_atoms(mol: Chem.Mol) -> list[int]:
    return [i for i, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=True)]


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def arm_stats(sascorer, mols: Sequence[Chem.Mol]) -> dict:
    return {
        "n_survived": len(mols),
        "sa_mean": _mean([sascorer.calculateScore(m) for m in mols]),
        "qed_mean": _mean([QED.qed(m) for m in mols]),
        "atoms_mean": _mean([m.GetNumAtoms() for m in mols]),
        "n_chiral_mean": _mean(
            [len(Chem.FindMolChiralCenters(m, includeUnassigned=True)) for m in mols]
        ),
    }


def measure(cohort_path: Path = COHORT) -> dict:
    sa = load_canonical_sascorer()
    smis = read_cohort(cohort_path)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]
    if not mols:
        raise RuntimeError("cohort yielded no parseable molecules")

    base = arm_stats(sa, mols)

    # Arm A: drop the stereocentre atoms (the intervention under test).
    arm_a = []
    for m in mols:
        out = drop_atoms(m, stereocentre_atoms(m))
        if out is not None:
            arm_a.append(out)

    # Arm B: size-matched control. Drop the SAME COUNT of atoms, chosen from
    # the non-stereocentre ones, so any SA difference is about WHICH atoms
    # were removed rather than HOW MANY.
    arm_b = []
    for m in mols:
        centres = set(stereocentre_atoms(m))
        others = [a.GetIdx() for a in m.GetAtoms() if a.GetIdx() not in centres]
        k = min(len(centres), len(others))
        if k == 0:
            continue
        out = drop_atoms(m, others[:k])
        if out is not None:
            arm_b.append(out)

    a = arm_stats(sa, arm_a)
    b = arm_stats(sa, arm_b)

    d_sa_a = a["sa_mean"] - base["sa_mean"]
    d_sa_b = b["sa_mean"] - base["sa_mean"]
    d_q_a = a["qed_mean"] - base["qed_mean"]
    d_q_b = b["qed_mean"] - base["qed_mean"]

    return {
        "metric": "cfm_opt1_stereo_suppression",
        "tag": "MEASURED",
        "date": "2026-10-13",
        "scorer": (
            "RDKit bundled rdkit/Contrib/SA_Score/sascorer.py (canonical Ertl "
            "2009). NOT molmetal/eval/scoring/sascorer.py, whose fragment term "
            "is a hardcoded constant -- see "
            "molmetal/tests/test_sascorer_fork_divergence.py."
        ),
        "cohort": {
            "path": str(cohort_path.relative_to(REPO)),
            "n_smiles": len(smis),
            "n_parsed": len(mols),
        },
        "baseline": base,
        "arm_A_remove_stereocentre_atoms": a,
        "arm_B_size_matched_control": b,
        "deltas": {
            "arm_A_d_sa": d_sa_a,
            "arm_B_d_sa": d_sa_b,
            "stereocentre_attributable_d_sa": d_sa_a - d_sa_b,
            "arm_A_d_qed": d_q_a,
            "arm_B_d_qed": d_q_b,
            "stereocentre_attributable_d_qed": d_q_a - d_q_b,
            "arm_A_d_atoms": a["atoms_mean"] - base["atoms_mean"],
            "arm_B_d_atoms": b["atoms_mean"] - base["atoms_mean"],
        },
        "why_the_control_is_required": (
            "SA contains a size penalty (nAtoms**1.005 - nAtoms), and the "
            "intervention deletes atoms, so an unguarded SA improvement would "
            "be indistinguishable from simply having fewer atoms. Arm B "
            "deletes the same count of non-stereocentre atoms; the difference "
            "between the arms is what is attributable to the stereocentres."
        ),
        "supersedes": (
            "cfm_todo.json::OPT_1_MEASURED_2026_10_13, which reported "
            "SA 3.3039 -> 4.4189 (+1.115, worse) and closed OPT-1 as "
            "INFEASIBLE_AT_DECODE_TIME. Those numbers came from the broken "
            "in-tree fork; the sign reverses under the canonical scorer."
        ),
        "what_does_not_change": (
            "set_tag and remove_annotation remain VACUOUS -- scorer-"
            "independent, because FindMolChiralCenters(includeUnassigned="
            "True) counts assigned centres too. That half of round 8 stands."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    res = measure(args.cohort)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(args.out_json, json.dumps(res, indent=2) + "\n")

    d = res["deltas"]
    print(f"baseline            SA {res['baseline']['sa_mean']:.4f}  "
          f"QED {res['baseline']['qed_mean']:.4f}  "
          f"atoms {res['baseline']['atoms_mean']:.2f}")
    print(f"A remove_stereo     SA {res['arm_A_remove_stereocentre_atoms']['sa_mean']:.4f} "
          f"({d['arm_A_d_sa']:+.4f})  atoms "
          f"{res['arm_A_remove_stereocentre_atoms']['atoms_mean']:.2f}")
    print(f"B size-matched ctrl SA {res['arm_B_size_matched_control']['sa_mean']:.4f} "
          f"({d['arm_B_d_sa']:+.4f})  atoms "
          f"{res['arm_B_size_matched_control']['atoms_mean']:.2f}")
    print(f"\nstereocentre-attributable d_sa  = "
          f"{d['stereocentre_attributable_d_sa']:+.4f}")
    print(f"stereocentre-attributable d_qed = "
          f"{d['stereocentre_attributable_d_qed']:+.4f}")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
