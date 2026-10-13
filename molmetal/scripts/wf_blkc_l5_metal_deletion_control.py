#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""BLK-C-L5 control: does deleting the metal make REAL tmQM complexes embed?

WHY THIS EXISTS
---------------
``metrics/by_metric/cfm_embed_failure_localisation.json`` localises the 18.8%
embed failure at the metal coordination centre, on this measurement:

    11 of the 12 failing GENERATED molecules become embeddable once the
    metal atom is deleted.

That measurement has **no control**. The same trick has never been applied to
molecules that are known to be chemically real. Two readings are consistent
with the number as recorded:

  (a) the generated molecules' metals are the defect -- deleting them repairs
      the graph; or
  (b) RDKit simply cannot embed a large fraction of metal complexes, and
      deleting the metal repairs ANY of them.

Under (b) the localisation says nothing about our bond head, and
``next_actions[0]`` -- "BOND HEAD, METAL COORDINATION SPHERE -- the first
localised, evidence-backed defect this line has produced" -- is built on a base
rate rather than on a signal.

The control cohort is ``.data/molmetal/tmQM/tmqm_parsed.csv``, which carries
100,849 real tmQM SMILES with their metal identity. This is NOT
``real_tmqm_reference.csv``: per the BLK-C-L4 audit that file is 636/636
decoder output with bracket-atom notation, i.e. our own bond head's output
wearing a "real" label. Using it here would measure the generator against
itself.

WHAT TO REPORT
--------------
For both cohorts, under the identical protocol already recorded as
reproducible in the localisation JSON (ETKDGv3, useRandomCoords=True,
maxAttempts=1, randomSeed=0xF00D):

  * embed rate WITH the metal
  * embed rate with EVERY metal deleted
  * the paired transition counts, so "fixed by deletion" is not confused with
    "was already embeddable"

The comparison that decides it is the **conditional**: of the molecules that
FAIL with the metal present, what fraction become embeddable without it --
measured separately for generated and for real. If the real cohort's fraction
is close to the generated cohort's 11/12, the generated result carries no
information about our model.

It also grows n. The caveat on ``next_actions[0]`` is that 11 molecules is too
few to design against, and this is the cheap way to find out whether the
effect survives on real molecules before spending anything on it.

MEASURES. Trains nothing, launches no GPU work, writes nothing outside
``--out-json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The metals this project trains on and decodes. Deliberately the SAME set the
# localisation used, so the two measurements differ only in the cohort.
TRAIN_METALS = frozenset({"Pt", "Ru", "Ir", "Rh", "Os", "Re"})

# The protocol recorded as reproducing the audit's 54/66 = 0.8182 exactly.
SEED = 0xF00D


def _rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    return Chem, AllChem


def embeddable(mol) -> Optional[bool]:
    """True/False for one molecule under the recorded protocol.

    MUST match ``wf_blkc4_metric_validity_audit.py::embeddable`` exactly,
    because that is the definition under which the recorded 54/66 = 0.8182 was
    produced, and this control is only meaningful if both cohorts are measured
    the same way the original number was.

    Two details are load-bearing and both were wrong in the first version of
    this function, which reported an embed rate of 0.0 for BOTH cohorts and
    therefore "proved" the localisation wrong on the strength of its own bug:

    * ``Chem.AddHs``. Without explicit hydrogens ETKDG has no torsion
      definitions for most bonds and fails on essentially every organometallic.
    * ``probe.GetNumConformers() > 0`` as the success test rather than the raw
      return code alone.

    Returns None when the molecule cannot be parsed, so a parse failure is
    never silently counted as an embedding failure -- the two have different
    causes and averaging them together hides both.
    """
    if mol is None:
        return None
    Chem, AllChem = _rdkit()
    try:
        probe = Chem.AddHs(Chem.Mol(mol))
    except Exception:
        return False
    try:
        if AllChem.EmbedMolecule(
            probe, useRandomCoords=True, maxAttempts=1, randomSeed=SEED
        ) != 0:
            return False
        return probe.GetNumConformers() > 0
    except Exception:
        return False


def _has_any_metal(smiles: str) -> bool:
    """True if the molecule contains at least one of the six train metals."""
    Chem, _ = _rdkit()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    z = {Chem.GetPeriodicTable().GetAtomicNumber(sym) for sym in TRAIN_METALS}
    return any(a.GetAtomicNum() in z for a in m.GetAtoms())


def strip_metals(smiles: str) -> Optional[str]:
    """Delete every metal atom and return the remaining SMILES, or None.

    2026-10-12. This used ``Chem.DeleteSubstructs(mol, smarts,
    onlyFrags=True)``, which was VERIFIED NOT TO DELETE ANYTHING: on
    ``[Pt](Cl)(Cl)(N)N`` the SMARTS matches once and
    ``DeleteSubstructs`` returns the molecule unchanged, Pt still present. A
    run would then have compared "with metal" against "still with metal",
    measured a repair rate of 0 on both cohorts, and concluded the two are
    identical -- a conclusion that is an artefact of the deletion never
    happening.

    Uses ``RWMol.RemoveAtom`` on the atomic numbers directly, then sanitizes.
    Removing the atom without relabelling would leave dangling valences, which
    is why the sanitize step is mandatory and its failure is a None rather than
    a silent pass-through.
    """
    Chem, _ = _rdkit()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    # Atomic numbers, not symbols: TRAIN_METALS is the single source of truth
    # for both this deletion and the cohort filter below.
    z = {Chem.GetPeriodicTable().GetAtomicNumber(sym) for sym in TRAIN_METALS}
    rw = Chem.RWMol(m)
    metal_idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() in z]
    if not metal_idx:
        # Nothing to delete. Returning the input would make a metal-free
        # molecule count as "repaired", which is the opposite of the truth.
        return None
    for i in sorted(metal_idx, reverse=True):
        rw.RemoveAtom(i)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
    except Exception:
        return None
    if out.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(out)


def evaluate(smiles_list: List[str], label: str) -> Dict[str, object]:
    Chem, _ = _rdkit()

    n_parsed = n_unparsed = 0
    with_metal_ok = 0
    fails: List[str] = []

    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            n_unparsed += 1
            continue
        n_parsed += 1
        ok = embeddable(m)
        if ok:
            with_metal_ok += 1
        else:
            fails.append(s)

    # Of the molecules that FAILED, how many does deleting the metal repair?
    #
    # Three buckets, kept apart deliberately. Collapsing them is how a control
    # ends up flattering the thing it is supposed to test:
    #   repaired              -- failed with the metal, embeds without it
    #   still_failing         -- there was no metal to delete, or deleting it
    #                            left something that still will not embed
    #   unparsable_after_delete-- the stripped graph does not even sanitise.
    #                            This is a harness fault, not a model property,
    #                            so it is counted separately and never folded
    #                            into the repair rate.
    repaired = 0
    still_failing = 0
    no_metal_to_delete = 0
    unparsable_after_delete = 0
    for s in fails:
        stripped = strip_metals(s)
        if stripped is None:
            # Either there was no metal (nothing to repair) or the stripped
            # graph will not sanitise. Distinguish the two: the first is a
            # legitimate "still failing", the second is our own bug.
            if _has_any_metal(s):
                unparsable_after_delete += 1
            else:
                no_metal_to_delete += 1
                still_failing += 1
            continue
        if embeddable(Chem.MolFromSmiles(stripped)):
            repaired += 1
        else:
            still_failing += 1

    considered = repaired + still_failing
    return {
        "cohort": label,
        "n_input": len(smiles_list),
        "n_parsed": n_parsed,
        "n_unparsed_smiles": n_unparsed,
        "n_embeddable_with_metal": with_metal_ok,
        "embed_rate_with_metal": round(with_metal_ok / n_parsed, 4) if n_parsed else None,
        "n_failed_with_metal": len(fails),
        "of_which_repaired_by_deleting_metals": repaired,
        "of_which_still_failing": still_failing,
        "of_which_had_no_metal_to_delete": no_metal_to_delete,
        "unparsable_after_deletion": unparsable_after_delete,
        "REPAIR_RATE_given_failure": (
            round(repaired / considered, 4) if considered else None
        ),
        "repair_rate_n": considered,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--real-tmqm-csv",
        type=Path,
        default=REPO / ".data" / "molmetal" / "tmQM" / "tmqm_parsed.csv",
    )
    ap.add_argument(
        "--generated-csv",
        type=Path,
        default=REPO / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10"
        / "generated_smiles.csv",
    )
    ap.add_argument("--n-real", type=int, default=600,
                    help="real tmQM molecules to sample (seeded)")
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()

    if not a.real_tmqm_csv.is_file():
        print(f"MISSING: {a.real_tmqm_csv}", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(
        a.real_tmqm_csv.open(encoding="utf-8", errors="replace")))
    real = [(r["smiles"].strip(), r.get("metal", "").strip())
            for r in rows if (r.get("smiles") or "").strip()]
    # Restrict to the six metals this line actually trains on, so the control
    # cohort matches the generated one in metal identity and not just in size.
    real_matched = [(s, m) for s, m in real if m in TRAIN_METALS]
    print(f"real tmQM with SMILES: {len(real)}; "
          f"on the 6 train metals: {len(real_matched)}")

    rnd = random.Random(SEED)
    if len(real_matched) > a.n_real:
        real_sampled = rnd.sample(real_matched, a.n_real)
    else:
        real_sampled = real_matched
    print(f"control cohort: {len(real_sampled)} real tmQM molecules, "
          f"metal counts {dict(Counter(m for _, m in real_sampled).most_common())}")

    cohorts = []
    if a.generated_csv.is_file():
        gen_rows = [
            r for r in csv.DictReader(a.generated_csv.open(encoding="utf-8"))
            if (r.get("smiles") or "").strip()
        ]
        gen_smi = [r["smiles"].strip() for r in gen_rows]
        print(f"generated cohort: {len(gen_smi)}")
        cohorts.append(evaluate(gen_smi, "generated (our bond head, ODE)"))
    else:
        print(f"MISSING: {a.generated_csv}", file=sys.stderr)

    cohorts.append(evaluate([s for s, _ in real_sampled], "real tmQM (CONTROL)"))

    print()
    hdr = (f"{'cohort':26} {'n':>5} {'embed':>7} {'fail':>5} "
           f"{'repaired':>9} {'repair|given fail':>18}")
    print(hdr)
    for c in cohorts:
        rr = c["REPAIR_RATE_given_failure"]
        print(f"{c['cohort']:26} {c['n_parsed']:>5} "
              f"{str(c['embed_rate_with_metal']):>7} {c['n_failed_with_metal']:>5} "
              f"{c['of_which_repaired_by_deleting_metals']:>9} "
              f"{('%.4f (n=%d)' % (rr, c['repair_rate_n'])) if rr is not None else 'NA':>18}")

    g = next((c for c in cohorts if c["cohort"].startswith("generated")), None)
    r = next((c for c in cohorts if c["cohort"].startswith("real")), None)
    verdict = None
    if g and r and g["REPAIR_RATE_given_failure"] is not None \
            and r["REPAIR_RATE_given_failure"] is not None:
        gr, rr = g["REPAIR_RATE_given_failure"], r["REPAIR_RATE_given_failure"]
        verdict = {
            "generated_repair_rate": gr,
            "real_repair_rate": rr,
            "difference": round(gr - rr, 4),
            "verdict": (
                "LOCALISATION SURVIVES the control: deleting the metal repairs a "
                "materially larger fraction of the GENERATED failures than of the "
                "real ones, so the generated metal sphere is doing something the "
                "real ones do not."
                if gr - rr > 0.15 else
                "LOCALISATION DOES NOT SURVIVE the control: RDKit repairs real "
                "tmQM metal complexes at nearly the same rate as it repairs the "
                "generated ones, so '11 of 12 embed once the metal is deleted' is "
                "a base rate of the embedder, not evidence about our bond head. "
                "next_actions[0] and the L5 localisation must be re-scoped before "
                "any fix is designed."
            ),
            "threshold_used": 0.15,
        }

    out = {
        "metric": "cfm_l5_metal_deletion_control",
        "tag": "MEASURED",
        "date": "2026-10-12",
        "question": (
            "Is 'deleting the metal repairs the embedding' a property of our "
            "generated complexes, or of RDKit's embedder applied to metal "
            "complexes in general?"
        ),
        "protocol": {
            "embedder": "RDKit ETKDGv3, useRandomCoords=True, maxAttempts=1, "
                        "randomSeed=0xF00D",
            "identical_to": "metrics/by_metric/cfm_embed_failure_localisation.json",
            "deletion": "Chem.DeleteSubstructs(metal SMARTS, onlyFrags=True)",
            "metals": sorted(TRAIN_METALS),
        },
        "cohort_provenance": {
            "real": (
                f"{a.real_tmqm_csv} -- REAL tmQM SMILES with their metal identity. "
                "NOT real_tmqm_reference.csv, which the BLK-C-L4 audit showed is "
                "636/636 decoder output with bracket-atom notation and would "
                "measure our bond head against itself."
            ),
            "generated": str(a.generated_csv),
        },
        "cohorts": cohorts,
        "VERDICT": verdict,
        "what_this_does_not_settle": [
            "n is still small on the generated side (12 failures). This "
            "establishes whether the effect is worth pursuing, not its size.",
            "Embeddability is a proxy. It says nothing about whether the "
            "complex is a plausible anticancer agent -- the ligand set and "
            "geometry still have to be judged separately.",
        ],
    }
    if verdict:
        print()
        print("VERDICT:", verdict["verdict"])
    if a.out_json:
        a.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())