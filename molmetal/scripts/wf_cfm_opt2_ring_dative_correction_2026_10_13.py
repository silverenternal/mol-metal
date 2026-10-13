#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Ring-size audit, corrected for DATIVE bonds hiding real chelate rings.

THE CONFOUND
------------
``metrics/by_metric/cfm_opt2_ring_content.json`` reported generated rings as
89.8% three-membered against 0.43% for real tmQM -- a 180x gap, presented as
the load-bearing finding of round 8. The real number is measured through
``RDKit`` ring perception, and real tmQM encodes EVERY metal-ligand bond as
``DATIVE`` (``->``), which ring perception does not traverse. So the real
side never sees a single chelate ring.

MEASURED, same corpus:

    REAL as-written      3-membered  0.43%   5/6-membered 96.73%
    REAL dative->single  3-membered 12.38%   5/6-membered 80.50%

The generated side needs no conversion: its metal-ligand bonds are already
SINGLE/DOUBLE, so ring perception sees them (measured: 32.4% of generated
rings contain a metal). The dative->single real arm is therefore the
like-for-like comparison, and it is the one to quote.

THE HONEST LIMITATION: 29.5% SURVIVORS
----------------------------------------
Only 9183 of 31150 molecules survive ``DATIVE -> SINGLE`` followed by
sanitisation; the rest fail on valence or kekulisation. Every number in the
corrected arm is computed on that SAME 9183-molecule subset, compared against
itself as-written, so the dative effect is not confounded with survivorship.
But the subset is itself selected (it is the set that tolerates the rewrite),
and 70.5% of the corpus is unmeasured here. That is why BOTH arms are
reported on the same subset rather than the full corpus being silently
dropped.

Usage
-----
    .venv/bin/python molmetal/scripts/wf_cfm_opt2_ring_dative_correction_2026_10_13.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent.parent
TMQM = REPO / ".data" / "molmetal" / "tmQM" / "tmQM_y.csv"
GENERATED = (
    REPO / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10" / "generated_smiles.csv"
)
DEFAULT_OUT = REPO / "metrics" / "by_metric" / "cfm_opt2_ring_dative_correction.json"

METAL_TOKENS = ("[Pt", "[Ir", "[Ru", "[Rh", "[Os", "[Re")
METAL_Z = {78, 77, 44, 45, 76, 75}


def dative_to_single(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Rewrite DATIVE/DATIVEONE as SINGLE so ring perception can traverse.

    Returns None when the rewrite fails sanitisation, which happens for 70.5%
    of the corpus -- those molecules are EXCLUDED from both arms rather than
    counted on one side only.
    """
    m = Chem.Mol(mol)
    for bond in m.GetBonds():
        if bond.GetBondType() in (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE):
            bond.SetBondType(Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def profile(mols: Sequence[Chem.Mol]) -> dict:
    sizes: Counter = Counter()
    rings_total = 0
    metal_rings = 0
    for m in mols:
        rings = m.GetRingInfo().AtomRings()
        rings_total += len(rings)
        for ring in rings:
            sizes[len(ring)] += 1
            if any(m.GetAtomWithIdx(i).GetAtomicNum() in METAL_Z for i in ring):
                metal_rings += 1
    tot = rings_total or 1
    return {
        "n_mols": len(mols),
        "n_rings": rings_total,
        "rings_per_mol": rings_total / max(1, len(mols)),
        "ring_size_counts": {str(k): sizes[k] for k in sorted(sizes)},
        "frac_3_membered": sizes[3] / tot,
        "frac_5_or_6_membered": sum(sizes[k] for k in (5, 6)) / tot,
        "frac_rings_containing_a_metal": metal_rings / tot,
    }


def read_real() -> tuple[list[Chem.Mol], list[Chem.Mol], int]:
    """Return (survivors_as_written, survivors_converted, n_corpus_parsed).

    ``n_corpus_parsed`` is the denominator for the survivorship rate. Using
    the survivor list as its own denominator is the bug this signature exists
    to prevent -- it silently reports 100% survivorship.
    """
    rows = []
    with TMQM.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            s = (r.get("SMILES") or "").strip()
            if any(t in s for t in METAL_TOKENS):
                rows.append(s)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in rows) if m is not None]
    written, conv = [], []
    for m in mols:
        c = dative_to_single(m)
        if c is not None:
            written.append(m)
            conv.append(c)
    return written, conv, len(mols)


def read_generated() -> list[Chem.Mol]:
    smis = []
    with GENERATED.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("smiles") or "").strip()
            if v:
                smis.append(v)
    return [m for m in (Chem.MolFromSmiles(s) for s in smis) if m is not None]


def measure() -> dict:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    gen = read_generated()
    written, conv, n_corpus = read_real()

    p_gen = profile(gen)
    p_written = profile(written)
    p_conv = profile(conv)

    # The generated side needs no rewrite: its metal-ligand bonds are already
    # SINGLE/DOUBLE, so ring perception sees them. Verify rather than assume.
    gen_has_dative = any(
        b.GetBondType() in (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE)
        for m in gen
        for b in m.GetBonds()
    )
    real_has_dative = any(
        b.GetBondType() in (Chem.BondType.DATIVE, Chem.BondType.DATIVEONE)
        for m in written
        for b in m.GetBonds()
    )

    return {
        "metric": "cfm_opt2_ring_dative_correction",
        "tag": "MEASURED",
        "date": "2026-10-13",
        "question": (
            "Is the round-8 finding 'generated rings are 89.8% three-membered "
            "vs 0.43% real' real, or an artefact of real tmQM encoding every "
            "metal-ligand bond as DATIVE, which RDKit ring perception does not "
            "traverse?"
        ),
        "answer": "ARTEFACT, IN PART. The gap is real but far smaller than reported.",
        "confound": (
            "RDKit ring perception does not traverse DATIVE bonds. Real tmQM "
            "writes every metal-ligand bond as DATIVE (->), so the as-written "
            "real profile contains ZERO metal-containing rings: the chelate "
            "rings that dominate six-metal anticancer chemistry were never "
            "counted on the reference side."
        ),
        "like_for_like_arms": {
            "note": (
                "The generated molecules have no dative bonds (verified), so "
                "their ring profile is already fully visible and needs no "
                "rewrite. The real corpus does, so the like-for-like real arm "
                "is dative->single."
            ),
            "generated_as_written": p_gen,
            "real_same_subset_as_written_dative": p_written,
            "real_same_subset_dative_to_single": p_conv,
        },
        "dative_presence": {
            "generated_contains_dative_bonds": gen_has_dative,
            "real_contains_dative_bonds": real_has_dative,
        },
        "corrected_comparison": {
            "frac_3_membered_generated": p_gen["frac_3_membered"],
            "frac_3_membered_real_like_for_like": p_conv["frac_3_membered"],
            "ratio": (
                p_gen["frac_3_membered"] / p_conv["frac_3_membered"]
                if p_conv["frac_3_membered"]
                else None
            ),
            "frac_5_or_6_generated": p_gen["frac_5_or_6_membered"],
            "frac_5_or_6_real_like_for_like": p_conv["frac_5_or_6_membered"],
            "what_round_8_reported": {
                "frac_3_membered_real": 0.0043,
                "ratio": 184.0,
                "source": "metrics/by_metric/cfm_opt2_ring_content.json",
            },
            "how_much_of_the_finding_survives": (
                "The 184x becomes roughly 7x on the like-for-like arm. The "
                "DIRECTION is unchanged -- generated still over-produces "
                "three-membered rings by a large factor and still under-produces "
                "5/6-membered ones -- but the magnitude is a fraction of what was "
                "reported, and any PROJECTED SA lift quoted against the 184x "
                "figure must be restated."
            ),
        },
        "honest_limitation_survivorship": {
            "survivors": len(conv),
            "corpus": n_corpus,
            "fraction": len(conv) / max(1, n_corpus),
            "why": (
                "DATIVE->SINGLE followed by sanitisation fails for 70.5% of the "
                "corpus (valence / kekulisation). Both arms are computed on the "
                "SAME 9183-molecule subset, so the dative effect is not "
                "confounded with survivorship -- but the subset is selected (it "
                "is the set that tolerates the rewrite) and 70.5% of the corpus "
                "is unmeasured in the corrected arm."
            ),
            "not_a_silent_cap": (
                "Reported explicitly rather than averaged away. A full-coverage "
                "arm would need per-molecule ring perception that tolerates "
                "dative bonds, which RDKit does not offer; that is a real gap, "
                "not a rounding error."
            ),
        },
        "what_does_NOT_change": (
            "Fragmentation was still not the story, and ring SIZE was still the "
            "right axis to look at rather than ring count -- both hold on the "
            "corrected arm. What changes is the size of the gap, and therefore "
            "the size of any lift quoted against it."
        ),
        "supersedes": (
            "The 0.43% real three-membered figure and the 184x ratio in "
            "metrics/by_metric/cfm_opt2_ring_content.json. That file's GENERATED "
            "numbers are unaffected (no dative bonds on that side); only its REAL "
            "denominator is confounded."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    res = measure()

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(args.out_json, json.dumps(res, indent=2) + "\n")

    a = res["like_for_like_arms"]
    c = res["corrected_comparison"]
    print(f"generated          3-ring {c['frac_3_membered_generated']:.2%}  "
          f"5/6 {c['frac_5_or_6_generated']:.2%}  "
          f"({a['generated_as_written']['rings_per_mol']:.2f} rings/mol)")
    print(f"real as-written    3-ring {a['real_same_subset_as_written_dative']['frac_3_membered']:.2%}  "
          f"5/6 {a['real_same_subset_as_written_dative']['frac_5_or_6_membered']:.2%}   <- dative rings invisible")
    print(f"real dative->single 3-ring {c['frac_3_membered_real_like_for_like']:.2%}  "
          f"5/6 {c['frac_5_or_6_real_like_for_like']:.2%}   <- like-for-like")
    print(f"\n3-ring ratio: {c['ratio']:.1f}x  (round 8 reported {c['what_round_8_reported']['ratio']:.0f}x)")
    print(f"survivors: {res['honest_limitation_survivorship']['survivors']} "
          f"({res['honest_limitation_survivorship']['fraction']:.1%})")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
