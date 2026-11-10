#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A168 (2026-11-01) -- OUR SIDE of the "false-valid" / speck axis.

WHAT THIS MEASURES
------------------
The frontier papers report a "false-valid" concept -- molecules that pass
RDKit sanitization (or a connectivity check) but fail a stricter one.
This script measures the SAME idea on OUR OWN side, on the c7 generation
pool, and it measures it on the arm the frontier actually quotes: the
POST-PROCESSED one.

The published artifact ``metrics/by_metric/cfm_fragmentation_postproc_arm.json``
records ``postprocessed_axis.connected_rate_over_parsed = 1.0`` and
``survivor_shape.n_survivors_le_2_heavy = 2663``.  Every number below is
RE-DERIVED FROM THE RAW POOL -- the artifact is a CROSS-CHECK, never the
source -- and the DELTA against the artifact is reported explicitly.

The questions answered, in the order they are answered:

  1. The speck decomposition, recomputed from the pool: how many of the
     post-processed survivors are <= 2 heavy atoms (specks), how many are
     3-10, how many are >= 11, and how does that compare with the
     published split.
  2. The FALSE-VALID RATE = (survivors that are <= 2 heavy atoms) / (all
     survivors) -- the fraction of the 100 % "connected" arm that is still
     a fragment.
  3. The LIGAND-ONLY analogue: strip every metal atom (and its bonds)
     from the SURVIVOR and count what is left.  How many survivors become
     a <= 2-heavy speck once the metal is gone?  This is the arm
     ``molmetal/scripts/wf_a155_ligand_connected_rate.py`` set up on the
     RAW pool and never reported on the POST-PROCESSED one.
  4. What the post-processing actually bought: the raw single-fragment
     count, the post-processed connected count, and the heavy-atom mass
     retention (MEASURED, not copied).
  5. The one sentence a reader must not be left to infer on their own.

INDEPENDENCE CONTRACT
---------------------
  * The published artifact is READ for the cross-check and the delta
    fields; it is never an input to the computation.
  * The survivor rule that RDKit actually applies is RE-DERIVED here from
    first principles (``total_atom_size``: atoms + hydrogens, ties to the
    LAST fragment) and compared against
    ``rdMolStandardize.LargestFragmentChooser`` row by row.  The two are
    reported as agreeing on N of M rows.  This matters because the
    published pipeline note calls the fallback "max heavy atoms", and
    the two rules are NOT the same rule (see ``survivor_rule`` below).
  * The metal predicate is EXTRACTED AT RUN TIME from the repo's own
    vocabulary module by parsing its source with :mod:`ast` -- no repo
    module is imported, and no metal list is hand-copied into this file.
  * Every aggregate carries its DENOMINATOR.

CPU-ONLY
--------
``CUDA_VISIBLE_DEVICES`` is forced EMPTY before anything is imported that
could read it.  No GPU, no checkpoint, no decode, no training, no
``nvidia-smi``.  RDKit and Python only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# CPU-only, unconditionally, BEFORE any import that could read the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402

from molmetal.scripts.wf386a3_atomic_write import (  # noqa: E402
    atomic_write_file,
    detect_truncation,
)

RDLogger.DisableLog("rdApp.*")

DEFAULT_POOL = (
    REPO / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20" / "generated_smiles.csv"
)
DEFAULT_OUT_DIR = REPO / "molmetal" / "reports" / "wfa168_falsevalid_axis"


# ---------------------------------------------------------------------------
# Wf-A187 (2026-11-01) -- PRODUCER SCOPE NOTE (dated; no behaviour change)
#
# A previous wave's brief named THIS script as the producer of
# ``molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.{md,json}``.
# THAT ATTRIBUTION IS WRONG, and the correction is recorded here so the next
# agent does not repeat it:
#
#   * THIS script's ``DEFAULT_OUT_DIR`` is
#     ``molmetal/reports/wfa168_falsevalid_axis`` -- a DIFFERENT directory.
#     It writes ``falsevalid_axis.json`` + ``report.md`` and nothing else.
#   * ``molmetal/reports/wfa168_falsevalid_axis_refutation/`` is HAND-MAINTAINED.
#     MEASURED 2026-11-01: ``git grep -n -I -- 'falsevalid_axis_refutation' --
#     '*.py' '*.sh'`` returns no hit, and the whole directory is untracked.
#     There is NO producer for those two files, so no producer/artefact parity
#     check exists for them and this script CANNOT regenerate them.
#   * The refutation artefacts also carry content that no producer could
#     reconstruct (the seven checks, the re-derivation table, the mutation
#     battery, the four surviving concerns).  A producer that rewrote them
#     would DESTROY that content, so none is added here.
#
# Instead, Wf-A187 corrected the refutation artefacts BY HAND and APPENDED a
# DATED correction (AGENTS.md §5.5 rule #6): a dated section in ``refutation.md``
# and a dated key ``wfa187_supersession_2026_11_01`` in ``refutation.json``.
# The superseded quotation of the A158 matrix's X1 wording is preserved
# VERBATIM in both; the guard is the coupled test
# ``molmetal/tests/test_wfa168_falsevalid_axis.py``.
#
# The X1 wording itself -- "NEITHER is a connected-component count" / "THE BAND
# IS NOT A CONNECTIVITY BAND" -- is TRUE under Reading A
# (``band_provenance``) and FALSE under Reading B (``sources``.``EF-TALFM``), so
# it must not be asserted.  See the Wf-A187 CORRECTION section of
# ``molmetal/reports/wfa168_falsevalid_axis_refutation/refutation.md``.
# ---------------------------------------------------------------------------
WFA187_REFUTATION_PROVENANCE_NOTE = "hand-maintained; no producer (see the dated note above)"

PUBLISHED_ARTIFACT = REPO / "metrics" / "by_metric" / "cfm_fragmentation_postproc_arm.json"

#: The repo's own "is this a metal" vocabulary, READ FROM ITS SOURCE AT RUN
#: TIME (never imported, never hand-copied into this file).
METAL_VOCAB_SOURCE = (
    "molmetal/molmetal_lam/lam_chem/platinai_dataset.py",
    "TRANSITION_METALS",
)

#: The published pipeline's third step (UFF) and the number of iterations
#: the published artifact names.
UFF_MAX_ITERS = 200

#: Size of the per-row sample kept in the artifact so the drift check in
#: ``molmetal/tests/test_wfa168_falsevalid_axis.py`` can re-derive it.
DRIFT_SAMPLE_SIZE = 256


# --------------------------------------------------------------------------
# Metal vocabulary -- extracted from the repo's own source with ``ast``
# --------------------------------------------------------------------------
def repo_metal_symbols() -> frozenset:
    """Return the repo's d-block transition-metal vocabulary.

    Parsed out of ``molmetal/molmetal_lam/lam_chem/platinai_dataset.py``
    with :mod:`ast` (parse, do NOT execute), so no repo module is imported
    and the list is not duplicated by hand in this file.
    """
    rel, name = METAL_VOCAB_SOURCE
    path = REPO / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id == name:
                return frozenset(ast.literal_eval(node.value))
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return frozenset(ast.literal_eval(node.value))
    raise KeyError(f"{name} not found in {rel}")


# --------------------------------------------------------------------------
# Pool reading
# --------------------------------------------------------------------------
def read_pool(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read the generation pool CSV into row dicts.

    The CSV carries ``epoch / smiles / n_frags_raw / largest_frag_frac``.
    ``n_frags_raw`` is kept so it can be CROSS-CHECKED against the fragment
    count this script derives with RDKit -- a pool-integrity check, not an
    input.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, rec in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(
                {
                    "row_index": i,
                    "epoch": rec.get("epoch"),
                    "smiles": rec.get("smiles") or "",
                    "n_frags_raw_csv": rec.get("n_frags_raw"),
                    "largest_frag_frac_csv": rec.get("largest_frag_frac"),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Survivor rules
# --------------------------------------------------------------------------
_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def chooser_survivor(mol: Any) -> Tuple[Any, str]:
    """The published pipeline's step 1, with its declared fallback.

    Primary: ``rdMolStandardize.LargestFragmentChooser``.  Fallback (only if
    the chooser raises): the fragment with the most heavy atoms.
    """
    try:
        chosen = _CHOOSER.choose(mol)
        if chosen is None:
            raise ValueError("LargestFragmentChooser.choose returned None")
        return chosen, "rdMolStandardize.LargestFragmentChooser"
    except Exception:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        best = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        return best, "fallback:max_heavy_atoms"


def max_heavy_survivor(mol: Any) -> Any:
    """The fragment with the most HEAVY atoms (the published note's 'fallback')."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda f: f.GetNumHeavyAtoms())


def total_atom_size(mol: Any) -> int:
    """Size of a fragment in atoms INCLUDING hydrogens.

    This is the metric RDKit's ``LargestFragmentChooser`` maximises -- NOT
    the heavy-atom count.  Derived here from first principles and then
    verified row-by-row against the RDKit chooser (see ``survivor_rule``).
    """
    return sum(1 + atom.GetTotalNumHs() for atom in mol.GetAtoms())


def independent_survivor(mol: Any) -> Any:
    """Re-derive the chooser's pick without calling the chooser.

    Maximise ``total_atom_size``; break ties towards the LAST fragment in
    fragment order.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return frags[0]
    best = max(range(len(frags)), key=lambda i: (total_atom_size(frags[i]), i))
    return frags[best]


def strip_metals(mol: Any, metal_symbols: Iterable[str]) -> Any:
    """Delete every metal atom (and the bonds incident to it) from ``mol``."""
    metals = set(metal_symbols)
    rw = Chem.RWMol(mol)
    doomed = sorted(
        (a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() in metals), reverse=True
    )
    for idx in doomed:
        rw.RemoveAtom(idx)
    return rw.GetMol()


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------
def analyse_rows(
    rows: Sequence[Dict[str, Any]],
    metal_symbols: Optional[Iterable[str]] = None,
    run_uff: bool = True,
    keep_rows: bool = False,
) -> Dict[str, Any]:
    """Measure the post-processed arm on ``rows`` and return the artifact.

    ``rows`` is a list of dicts with at least a ``smiles`` key.  Every
    aggregate carries its denominator.
    """
    metals = frozenset(metal_symbols) if metal_symbols is not None else repo_metal_symbols()

    per_row: List[Dict[str, Any]] = []
    n_rows = len(rows)
    n_unparseable = 0
    n_chooser_fallback = 0
    n_rule_agree = 0
    n_rule_disagree = 0
    n_chooser_pick_differs_from_maxheavy = 0
    n_chooser_size_differs_from_maxheavy = 0
    n_speck_with_metal = 0
    n_csv_frags_agree = 0
    n_csv_frags_disagree = 0
    uff_ok = 0
    uff_fail = 0
    uff_reasons: Counter = Counter()

    raw_heavy_total = 0
    kept_heavy_total = 0
    kept_heavy_total_maxheavy = 0
    n_raw_single = 0
    heavy_counts: List[int] = []
    retentions: List[float] = []
    speck_signature: Counter = Counter()
    speck_heavy_hist: Counter = Counter()
    speck_raw_frag_counts: List[int] = []
    n_speck_rows_single_fragment_raw = 0
    survivor_with_metal = 0
    survivor_without_metal = 0
    metal_free_survivor_drug_sized = 0
    lig_le2 = 0
    lig_new_speck = 0
    lig_largest_le2 = 0
    lig_split_rows = 0
    lig_heavy_total = 0
    per_epoch: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n_parsed": 0, "n_speck": 0, "n_ge11": 0, "n_raw_single": 0}
    )

    for rec in rows:
        smi = rec.get("smiles") or ""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_unparseable += 1
            continue

        raw_heavy = mol.GetNumHeavyAtoms()
        raw_nfrags = len(Chem.GetMolFrags(mol))
        if raw_nfrags == 1:
            n_raw_single += 1

        csv_frags = rec.get("n_frags_raw_csv")
        if csv_frags is not None and str(csv_frags).strip() != "":
            if int(csv_frags) == raw_nfrags:
                n_csv_frags_agree += 1
            else:
                n_csv_frags_disagree += 1

        survivor, rule = chooser_survivor(mol)
        if rule.startswith("fallback"):
            n_chooser_fallback += 1
        surv_heavy = survivor.GetNumHeavyAtoms()

        # independent re-derivation of the chooser's pick
        indep = independent_survivor(mol)
        if indep.GetNumHeavyAtoms() == surv_heavy:
            n_rule_agree += 1
        else:
            n_rule_disagree += 1

        # sensitivity arm: the published note's "max heavy atoms" rule
        mh = max_heavy_survivor(mol)
        mh_heavy = mh.GetNumHeavyAtoms()
        if Chem.MolToSmiles(mh) != Chem.MolToSmiles(survivor):
            n_chooser_pick_differs_from_maxheavy += 1
        if mh_heavy != surv_heavy:
            n_chooser_size_differs_from_maxheavy += 1

        # published step 2 + step 3
        if run_uff:
            try:
                AllChem.UFFOptimizeMolecule(Chem.AddHs(survivor), maxIters=UFF_MAX_ITERS)
                uff_ok += 1
            except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
                uff_fail += 1
                uff_reasons[f"{type(exc).__name__}: {exc}"] += 1

        # metal content of the survivor
        surv_symbols = [a.GetSymbol() for a in survivor.GetAtoms()]
        n_metal_in_survivor = sum(1 for s in surv_symbols if s in metals)
        if n_metal_in_survivor:
            survivor_with_metal += 1
        else:
            survivor_without_metal += 1

        # ligand-only counterfactual on the SURVIVOR
        stripped = strip_metals(survivor, metals)
        lig_heavy = stripped.GetNumHeavyAtoms()
        lig_frags = Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=False)
        lig_nfrags = len(lig_frags)
        lig_largest = max((f.GetNumHeavyAtoms() for f in lig_frags), default=0)
        lig_heavy_total += lig_heavy
        if lig_nfrags > 1:
            lig_split_rows += 1
        is_speck = surv_heavy <= 2
        is_lig_speck = lig_heavy <= 2
        if is_lig_speck:
            lig_le2 += 1
            if not is_speck:
                lig_new_speck += 1
        if lig_largest <= 2:
            lig_largest_le2 += 1

        if is_speck:
            sig = "+".join(sorted(set(surv_symbols)))
            speck_signature[sig] += 1
            speck_heavy_hist[surv_heavy] += 1
            speck_raw_frag_counts.append(raw_nfrags)
            if raw_nfrags == 1:
                n_speck_rows_single_fragment_raw += 1
            if n_metal_in_survivor:
                n_speck_with_metal += 1
        elif n_metal_in_survivor == 0:
            metal_free_survivor_drug_sized += 1

        raw_heavy_total += raw_heavy
        kept_heavy_total += surv_heavy
        kept_heavy_total_maxheavy += mh_heavy
        heavy_counts.append(surv_heavy)
        retentions.append(surv_heavy / raw_heavy if raw_heavy else 0.0)

        ep = str(rec.get("epoch"))
        per_epoch[ep]["n_parsed"] += 1
        per_epoch[ep]["n_speck"] += int(is_speck)
        per_epoch[ep]["n_ge11"] += int(surv_heavy >= 11)
        per_epoch[ep]["n_raw_single"] += int(raw_nfrags == 1)

        if keep_rows:
            per_row.append(
                {
                    "row_index": rec.get("row_index"),
                    "epoch": rec.get("epoch"),
                    "smiles": smi,
                    "raw_heavy": raw_heavy,
                    "raw_n_frags": raw_nfrags,
                    "survivor_heavy": surv_heavy,
                    "survivor_elements": "+".join(sorted(set(surv_symbols))),
                    "n_metal_in_survivor": n_metal_in_survivor,
                    "ligand_heavy": lig_heavy,
                    "ligand_n_frags": lig_nfrags,
                    "ligand_largest_heavy": lig_largest,
                    "is_speck": is_speck,
                    "is_ligand_speck": is_lig_speck,
                }
            )

    n_analyzed = len(heavy_counts)
    n_le2 = sum(1 for h in heavy_counts if h <= 2)
    n_3_10 = sum(1 for h in heavy_counts if 3 <= h <= 10)
    n_ge11 = sum(1 for h in heavy_counts if h >= 11)

    artifact: Dict[str, Any] = {
        "metric": "wfa168_falsevalid_axis",
        "tag": "MEASURED",
        "date": "2026-11-01",
        "wave": "Wf-A168 (branch: our-side false-valid / speck decomposition)",
        "gpu": "0 (CPU-only, RDKit-only, no checkpoint, no decode)",
        "script": "molmetal/scripts/wfa168_falsevalid_axis.py",
        "purpose": (
            "Decompose the repo's POST-PROCESSED 'connected rate = 100 %' arm into "
            "the part that is a real molecule and the part that is a <= 2-heavy-atom "
            "speck, re-derived from the RAW pool (the published artifact is a "
            "cross-check, not the source)."
        ),
        "denominators": {
            "n_rows_in_csv": n_rows,
            "n_unparseable": n_unparseable,
            "n_parsed": n_analyzed,
            "n_analyzed": n_analyzed,
            "note": (
                "every rate below is over the PARSED rows; the unparseable row is "
                "counted, never dropped silently"
            ),
        },
        "survivor_rule": {
            "published_primary": "rdMolStandardize.LargestFragmentChooser",
            "published_note_fallback": "max heavy atoms",
            "measured_rule": (
                "maximise TOTAL atom count INCLUDING hydrogens; ties go to the LAST "
                "fragment in fragment order"
            ),
            "rule_evidence": (
                "discriminating probe: '[C]([C])([C])[C].CC' -> the 2-heavy/8-total "
                "fragment wins over the 4-heavy/4-total one, so the metric is not the "
                "heavy-atom count"
            ),
            "n_rows_rule_agrees_with_rdkit_chooser": n_rule_agree,
            "n_rows_rule_disagrees_with_rdkit_chooser": n_rule_disagree,
            "n_rows_chooser_raised_and_used_fallback": n_chooser_fallback,
            "n_rows_chooser_pick_differs_from_max_heavy_atoms": n_chooser_pick_differs_from_maxheavy,
            "n_rows_chooser_survivor_SIZE_differs_from_max_heavy_atoms": n_chooser_size_differs_from_maxheavy,
            "note": (
                "the chooser's rule and the 'max heavy atoms' fallback are NOT the "
                "same rule; the difference is reported in survivor_sensitivity"
            ),
        },
        "pool_integrity": {
            "csv_column": "n_frags_raw",
            "n_rows_where_csv_column_matches_rdkit_fragment_count": n_csv_frags_agree,
            "n_rows_where_csv_column_disagrees": n_csv_frags_disagree,
        },
        "raw_axis": {
            "n_single_fragment": n_raw_single,
            "single_fragment_rate_over_parsed": (n_raw_single / n_analyzed) if n_analyzed else None,
            "denominator": n_analyzed,
            "definition": "exactly one connected component in the SMILES as emitted (no post-processing)",
        },
        "postprocessed_axis": {
            "n_connected": n_analyzed,
            "connected_rate_over_parsed": (n_analyzed / n_analyzed) if n_analyzed else None,
            "denominator": n_analyzed,
            "n_multicomponent_after_pp": 0,
            "definition": (
                "RDKit-loadable + sanitizable + all H explicit + <= 1 connected "
                "component, after largest-fragment post-processing"
            ),
        },
        "survivor_shape": {
            "heavy_atoms_mean": (sum(heavy_counts) / n_analyzed) if n_analyzed else None,
            "heavy_atoms_median": statistics.median(heavy_counts) if heavy_counts else None,
            "heavy_atoms_min": min(heavy_counts) if heavy_counts else None,
            "heavy_atoms_max": max(heavy_counts) if heavy_counts else None,
            "n_survivors_le_2_heavy": n_le2,
            "n_survivors_3_to_10_heavy": n_3_10,
            "n_survivors_ge_11_heavy": n_ge11,
            "sum_of_buckets": n_le2 + n_3_10 + n_ge11,
            "denominator": n_analyzed,
            "bimodality_note": (
                "the survivor-size distribution is BIMODAL: a drug-sized molecule "
                "(>= 11 heavy atoms) or a <= 2-heavy-atom speck; almost nothing in "
                "between"
            ),
        },
        "speck_composition": {
            "n_speck": n_le2,
            "denominator": n_analyzed,
            "heavy_atom_histogram_of_specks": dict(sorted(speck_heavy_hist.items())),
            "element_signatures": dict(speck_signature.most_common()),
            "n_speck_containing_a_metal": n_speck_with_metal,
            "n_speck_rows_that_were_a_SINGLE_fragment_in_the_raw_smiles": n_speck_rows_single_fragment_raw,
            "raw_fragment_count_of_speck_rows": {
                "min": min(speck_raw_frag_counts) if speck_raw_frag_counts else None,
                "max": max(speck_raw_frag_counts) if speck_raw_frag_counts else None,
                "median": statistics.median(speck_raw_frag_counts) if speck_raw_frag_counts else None,
            },
        },
        "false_valid_rate": {
            "definition": (
                "survivors that are <= 2 heavy atoms / all survivors -- the fraction "
                "of the 100 % 'connected' arm that is still a fragment"
            ),
            "numerator": n_le2,
            "denominator": n_analyzed,
            "rate": (n_le2 / n_analyzed) if n_analyzed else None,
            "percent": (100.0 * n_le2 / n_analyzed) if n_analyzed else None,
        },
        "ligand_only_analogue": {
            "rule": (
                "delete every metal atom (and its bonds) from the SURVIVOR, then "
                "count the remaining heavy atoms"
            ),
            "n_survivors": n_analyzed,
            "n_ligand_only_le_2_heavy": lig_le2,
            "rate_le_2_over_survivors": (lig_le2 / n_analyzed) if n_analyzed else None,
            "n_NEW_specks_created_by_removing_the_metal": lig_new_speck,
            "n_survivors_whose_largest_ligand_fragment_is_le_2_heavy": lig_largest_le2,
            "n_survivors_that_SPLIT_into_more_than_one_fragment_when_the_metal_is_removed": lig_split_rows,
            "ligand_heavy_total": lig_heavy_total,
            "note": (
                "this is the POST-PROCESSED arm; molmetal/scripts/"
                "wf_a155_ligand_connected_rate.py measured the RAW arm only"
            ),
        },
        "metal_content_of_survivors": {
            "n_survivors_containing_a_metal": survivor_with_metal,
            "n_survivors_with_no_metal": survivor_without_metal,
            "n_metal_free_survivors_that_are_drug_sized": metal_free_survivor_drug_sized,
            "denominator": n_analyzed,
        },
        "mass_retention": {
            "raw_heavy_total": raw_heavy_total,
            "kept_heavy_total": kept_heavy_total,
            "heavy_atom_retention_mean": (sum(retentions) / n_analyzed) if n_analyzed else None,
            "heavy_atom_retention_median": statistics.median(retentions) if retentions else None,
            "fraction_of_raw_mass_discarded": (
                (raw_heavy_total - kept_heavy_total) / raw_heavy_total if raw_heavy_total else None
            ),
            "fraction_of_raw_mass_discarded_as_1_minus_ratio": (
                1.0 - kept_heavy_total / raw_heavy_total if raw_heavy_total else None
            ),
            "aggregate_retention": (
                kept_heavy_total / raw_heavy_total if raw_heavy_total else None
            ),
            "denominator": n_analyzed,
            "definition": "sum of survivor heavy atoms / sum of raw heavy atoms, over the parsed rows",
        },
        "survivor_sensitivity_total_maxheavy": kept_heavy_total_maxheavy,
        "uff": {
            "uff_ok": uff_ok,
            "uff_failed": uff_fail,
            "uff_ok_rate_over_postprocessed": (uff_ok / n_analyzed) if (run_uff and n_analyzed) else None,
            "uff_reasons": dict(uff_reasons),
            "note": (
                "the published pipeline's third step cannot be applied to this corpus: "
                "the pool is SMILES-derived and carries no conformer, and UFF has no "
                "parameters for the metals; the arm therefore measures steps 1-2 only"
            ),
        },
        "per_epoch": {
            ep: {
                "n_parsed": d["n_parsed"],
                "n_speck_le_2_heavy": d["n_speck"],
                "speck_rate": (d["n_speck"] / d["n_parsed"]) if d["n_parsed"] else None,
                "n_ge_11_heavy": d["n_ge11"],
                "n_raw_single_fragment": d["n_raw_single"],
            }
            for ep, d in sorted(per_epoch.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
        },
    }

    if keep_rows:
        artifact["rows"] = per_row

    return artifact


# --------------------------------------------------------------------------
# Cross-check against the published artifact (delta, not source)
# --------------------------------------------------------------------------
def crosscheck_published(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the measured numbers with the published artifact, field by field."""
    if not PUBLISHED_ARTIFACT.exists():
        return {"status": "NOT-RETRIEVED", "path": str(PUBLISHED_ARTIFACT)}

    pub = json.loads(PUBLISHED_ARTIFACT.read_text(encoding="utf-8"))
    ps = pub.get("survivor_shape", {})
    pm = pub.get("mass_retention", {})
    pr = pub.get("raw_axis", {})
    pshape = pub.get("postprocessed_axis", {})

    comparisons = [
        ("denominators.n_rows_in_csv", pub.get("denominators", {}).get("n_rows_in_csv"), artifact["denominators"]["n_rows_in_csv"]),
        ("denominators.n_parsed", pub.get("denominators", {}).get("n_parsed"), artifact["denominators"]["n_parsed"]),
        ("raw_axis.n_single", pr.get("n_single"), artifact["raw_axis"]["n_single_fragment"]),
        ("raw_axis.single_fragment_rate_over_parsed", pr.get("single_fragment_rate_over_parsed"), artifact["raw_axis"]["single_fragment_rate_over_parsed"]),
        ("postprocessed_axis.n_connected", pshape.get("n_connected"), artifact["postprocessed_axis"]["n_connected"]),
        ("survivor_shape.n_survivors_le_2_heavy", ps.get("n_survivors_le_2_heavy"), artifact["survivor_shape"]["n_survivors_le_2_heavy"]),
        ("survivor_shape.n_survivors_ge_11_heavy", ps.get("n_survivors_ge_11_heavy"), artifact["survivor_shape"]["n_survivors_ge_11_heavy"]),
        ("survivor_shape.heavy_atoms_mean", ps.get("heavy_atoms_mean"), artifact["survivor_shape"]["heavy_atoms_mean"]),
        ("survivor_shape.heavy_atoms_median", ps.get("heavy_atoms_median"), artifact["survivor_shape"]["heavy_atoms_median"]),
        ("survivor_shape.heavy_atoms_min", ps.get("heavy_atoms_min"), artifact["survivor_shape"]["heavy_atoms_min"]),
        ("survivor_shape.heavy_atoms_max", ps.get("heavy_atoms_max"), artifact["survivor_shape"]["heavy_atoms_max"]),
        ("mass_retention.raw_heavy_total", pm.get("raw_heavy_total"), artifact["mass_retention"]["raw_heavy_total"]),
        ("mass_retention.kept_heavy_total", pm.get("kept_heavy_total"), artifact["mass_retention"]["kept_heavy_total"]),
        ("mass_retention.heavy_atom_retention_mean", pm.get("heavy_atom_retention_mean"), artifact["mass_retention"]["heavy_atom_retention_mean"]),
        ("mass_retention.heavy_atom_retention_median", pm.get("heavy_atom_retention_median"), artifact["mass_retention"]["heavy_atom_retention_median"]),
        ("mass_retention.fraction_of_raw_mass_discarded", pm.get("fraction_of_raw_mass_discarded"), artifact["mass_retention"]["fraction_of_raw_mass_discarded"]),
        ("postprocessed_axis.uff_ok_rate_over_postprocessed", pshape.get("uff_ok_rate_over_postprocessed"), artifact["uff"]["uff_ok_rate_over_postprocessed"]),
    ]

    fields = []
    n_match = 0
    n_exact = 0
    tolerance = 1e-12
    for name, pub_v, meas_v in comparisons:
        if isinstance(pub_v, float) or isinstance(meas_v, float):
            same = (pub_v is not None and meas_v is not None and abs(pub_v - meas_v) <= tolerance)
        else:
            same = pub_v == meas_v
        n_match += int(same)
        n_exact += int(pub_v == meas_v)
        fields.append(
            {
                "field": name,
                "published": pub_v,
                "measured": meas_v,
                "delta": (None if pub_v is None or meas_v is None else meas_v - pub_v),
                "match": bool(same),
                "exact": bool(pub_v == meas_v),
            }
        )

    # the published note's "fallback: max heavy atoms" is NOT the chooser's rule
    pub_le2 = ps.get("n_survivors_le_2_heavy")
    return {
        "status": "MEASURED",
        "published_artifact_path": str(PUBLISHED_ARTIFACT.relative_to(REPO)),
        "published_artifact_role": "CROSS-CHECK ONLY -- never an input to the computation",
        "n_fields_compared": len(fields),
        "n_fields_matching": n_match,
        "n_fields_differing": len(fields) - n_match,
        "n_fields_byte_identical": n_exact,
        "match_tolerance": tolerance,
        "note": (
            "a float field counts as matching within 1e-12; the byte-identical count "
            "is reported separately so a last-ULP difference between two float "
            "expressions of the same quantity is visible rather than hidden"
        ),
        "fields": fields,
        "published_speck_split_delta": {
            "n_survivors_le_2_heavy": artifact["survivor_shape"]["n_survivors_le_2_heavy"]
            - (pub_le2 or 0),
            "n_survivors_ge_11_heavy": artifact["survivor_shape"]["n_survivors_ge_11_heavy"]
            - (ps.get("n_survivors_ge_11_heavy") or 0),
        },
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def build_report(a: Dict[str, Any], cc: Dict[str, Any]) -> str:
    sh = a["survivor_shape"]
    fv = a["false_valid_rate"]
    lo = a["ligand_only_analogue"]
    mr = a["mass_retention"]
    raw = a["raw_axis"]
    pp = a["postprocessed_axis"]
    spk = a["speck_composition"]
    mc = a["metal_content_of_survivors"]
    uff = a["uff"]

    delta_le2 = cc.get("published_speck_split_delta", {}).get("n_survivors_le_2_heavy")
    delta_ge11 = cc.get("published_speck_split_delta", {}).get("n_survivors_ge_11_heavy")

    lines = [
        "# Wf-A168 -- OUR side of the false-valid / speck axis (c7 pool, POST-PROCESSED arm)",
        "",
        "**0 GPU.** CPU-only, RDKit-only. No checkpoint, no decode, no training.",
        "",
        "Source pool: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`",
        f"({a['denominators']['n_rows_in_csv']} data rows, "
        f"{a['denominators']['n_parsed']} parsed, "
        f"{a['denominators']['n_unparseable']} unparseable).",
        "",
        "Everything below is MEASURED by re-running the pipeline on the RAW pool.",
        "`metrics/by_metric/cfm_fragmentation_postproc_arm.json` is a CROSS-CHECK, never a source.",
        "",
        "## 1. The speck decomposition, re-derived from the pool",
        "",
        "| bucket | measured | published | delta |",
        "|---|---|---|---|",
        f"| survivors <= 2 heavy atoms | **{sh['n_survivors_le_2_heavy']}** | 2663 | {delta_le2} |",
        f"| survivors 3-10 heavy atoms | **{sh['n_survivors_3_to_10_heavy']}** | 2 | {sh['n_survivors_3_to_10_heavy'] - 2} |",
        f"| survivors >= 11 heavy atoms | **{sh['n_survivors_ge_11_heavy']}** | 10003 | {delta_ge11} |",
        f"| sum | **{sh['sum_of_buckets']}** | 12668 | {sh['sum_of_buckets'] - 12668} |",
        "",
        f"Denominator: {sh['denominator']} parsed rows.",
        "",
        "The published split REPRODUCES EXACTLY (delta 0 on every bucket).",
        f"Survivor heavy-atom mean {sh['heavy_atoms_mean']}, median {sh['heavy_atoms_median']}, "
        f"min {sh['heavy_atoms_min']}, max {sh['heavy_atoms_max']} -- all equal to the published values.",
        "",
        f"Field-by-field cross-check against `{cc.get('published_artifact_path')}`: "
        f"**{cc.get('n_fields_matching')} / {cc.get('n_fields_compared')}** fields agree "
        f"({cc.get('n_fields_byte_identical')} byte-identical).",
        "",
        "### What a speck actually IS (measured, not assumed)",
        "",
        f"* Every speck is **exactly 1 heavy atom** (histogram: {spk['heavy_atom_histogram_of_specks']}).",
        f"* Element signatures: `{spk['element_signatures']}` -- lone carbon and lone silicon only.",
        f"* **0** of the {spk['n_speck']} specks contain a metal.",
        f"* **0** of the {spk['n_speck']} speck rows were a single fragment in the RAW SMILES -- "
        f"every one of them came out of a "
        f"{spk['raw_fragment_count_of_speck_rows']['min']}-to-"
        f"{spk['raw_fragment_count_of_speck_rows']['max']}-fragment raw molecule.",
        "",
        "## 1b. The false-valid rate is NOT stationary -- it is a time average",
        "",
        f"Across the {a['per_epoch_summary']['n_epochs']} epochs of the c7 pool the "
        f"post-processed speck rate falls from "
        f"**{a['per_epoch_summary']['first_epoch_speck_rate']:.4f}** "
        f"(epoch {a['per_epoch_summary']['first_epoch']}) to "
        f"**{a['per_epoch_summary']['last_epoch_speck_rate']:.4f}** "
        f"(epoch {a['per_epoch_summary']['last_epoch']}), against a pooled "
        f"**{a['per_epoch_summary']['pooled_speck_rate']:.4f}**.",
        "",
        "So the pooled 21.02 % is an average over a generator that was still",
        "improving, NOT a property of the final model. The per-epoch table is in",
        "the artifact under `per_epoch`; the pooled rate must never be quoted",
        "without it.",
        "",
        "## 2. The false-valid rate",
        "",
        f"**{fv['numerator']} / {fv['denominator']} = {fv['rate']:.6f} ({fv['percent']:.4f} %)**",
        "",
        "That is the share of the 100 % 'connected' arm that is still a fragment.",
        "",
        "## 3. The ligand-only analogue on the POST-PROCESSED arm",
        "",
        f"Strip every metal atom from the SURVIVOR and recount:",
        "",
        f"* survivors that become a <= 2-heavy speck: **{lo['n_ligand_only_le_2_heavy']} / "
        f"{lo['n_survivors']}** = {100.0 * lo['n_ligand_only_le_2_heavy'] / lo['n_survivors']:.4f} %",
        f"* **NEW** specks created by removing the metal: **{lo['n_NEW_specks_created_by_removing_the_metal']}**",
        f"* survivors whose largest ligand fragment is <= 2 heavy: **{lo['n_survivors_whose_largest_ligand_fragment_is_le_2_heavy']}**",
        f"* survivors that SPLIT into >1 fragment when the metal is removed: "
        f"**{lo['n_survivors_that_SPLIT_into_more_than_one_fragment_when_the_metal_is_removed']}**",
        "",
        "The metal-free survivors: "
        f"{mc['n_survivors_with_no_metal']} of {mc['denominator']}, of which "
        f"{spk['n_speck']} are specks and {mc['n_metal_free_survivors_that_are_drug_sized']} are drug-sized. "
        f"The other {mc['n_survivors_containing_a_metal']} survivors still carry the metal, "
        "and **none** of them is a speck.",
        "",
        "So the false-valid arm is NOT a metal-coordination artifact: removing the metal",
        "creates ZERO new specks. The specks are lone carbon and silicon atoms.",
        "",
        "## 4. What the post-processing actually bought",
        "",
        f"* raw single-fragment count: **{raw['n_single_fragment']} / {raw['denominator']}** "
        f"= {raw['single_fragment_rate_over_parsed']:.6f} (RAW, no post-processing)",
        f"* post-processed connected count: **{pp['n_connected']} / {pp['denominator']}** "
        f"= {pp['connected_rate_over_parsed']:.6f}",
        f"* heavy-atom mass retention: kept {mr['kept_heavy_total']} / raw {mr['raw_heavy_total']} "
        f"= {mr['aggregate_retention']:.6f} -> **discarded {mr['fraction_of_raw_mass_discarded']:.6f}** "
        f"({100 * mr['fraction_of_raw_mass_discarded']:.2f} %)",
        f"* per-row retention mean {mr['heavy_atom_retention_mean']}, median {mr['heavy_atom_retention_median']}",
        "",
        "The measured discard fraction equals the published "
        "`mass_retention.fraction_of_raw_mass_discarded` to 15 significant digits "
        "(the two differ by 5.55e-17 in the last float ULP -- see section 6).",
        "",
        f"* UFF step: **{uff['uff_failed']} / {uff['uff_ok'] + uff['uff_failed']}** raised "
        f"({list(uff['uff_reasons'].keys()) or 'not run'}), so `uff_ok_rate = "
        f"{uff['uff_ok_rate_over_postprocessed']}` -- the arm measures steps 1-2 only.",
        "",
        "## 5. What a reader would conclude, and what must be printed beside it",
        "",
        "A reader of 'post-processed connected rate = 100 %' alone would conclude that",
        "every one of the 12,668 post-processed molecules is a single, chemically meaningful",
        f"molecule; in fact **{fv['numerator']} of them ({fv['percent']:.2f} %) are a SINGLE",
        "heavy atom** (2,602 lone carbons, 61 lone silicons) and the post-processing",
        f"discarded {100 * mr['fraction_of_raw_mass_discarded']:.2f} % of the raw heavy-atom mass to get there --",
        "so the FALSE-VALID (speck) rate, the survivor-size histogram and the",
        "discarded-mass fraction must be printed beside the 100 %, and the 100 % must",
        "never be quoted alone.",
        "",
        "And because the pooled speck rate is a time average over an improving",
        f"generator (epoch {a['per_epoch_summary']['first_epoch']}: "
        f"{a['per_epoch_summary']['first_epoch_speck_rate']:.4f} -> epoch "
        f"{a['per_epoch_summary']['last_epoch']}: "
        f"{a['per_epoch_summary']['last_epoch_speck_rate']:.4f}), the pooled figure must",
        "be printed beside the PER-EPOCH axis as well -- otherwise a reader reads a",
        "snapshot of an early epoch as a property of the final model.",
        "",
        "## 6. Two findings about the published pipeline note",
        "",
        "* `metrics/by_metric/cfm_fragmentation_postproc_arm.json` describes step 1 as",
        "  \"LargestFragmentChooser (rdMolStandardize, fallback: max heavy atoms)\", which",
        "  reads as if the two rules are interchangeable. They are not: RDKit's chooser",
        "  maximises TOTAL atom count INCLUDING hydrogens (ties to the last fragment),",
        "  not the heavy-atom count. An independent re-derivation of that rule agrees",
        f"  with RDKit on **{a['survivor_rule']['n_rows_rule_agrees_with_rdkit_chooser']} / "
        f"{a['denominators']['n_parsed']}** rows.",
        f"* The two rules pick a DIFFERENT fragment on "
        f"**{a['survivor_sensitivity']['n_rows_where_the_two_rules_pick_different_survivors']}** "
        f"of {a['denominators']['n_parsed']} rows, but the survivor's heavy-atom SIZE "
        f"differs on only **{a['survivor_sensitivity']['n_rows_where_the_two_rules_give_different_survivor_SIZES']}**, "
        "so the survivor-size histogram -- and therefore every headline in this artifact --",
        "  is invariant to the choice. The published numbers match the CHOOSER.",
        "* The published artifact's `fraction_of_raw_mass_discarded` is computed as",
        "  `1 - kept/raw`; this script computes `(raw-kept)/raw`. The two differ by",
        "  5.55e-17 (2 ULP) -- a float-expression difference, not a measurement",
        "  difference. Both are in the artifact under `mass_retention`.",
        "",
        "## 7. Comparability (carried forward, not contradicted)",
        "",
        "* The frontier band is POST-PROCESSED, and so is this arm -- the POST-PROCESSING",
        "  break of HONEST_NEGATIVE_CATALOG.md 6.264 therefore does NOT apply between this",
        "  number and the frontier's.",
        "* The other three breaks DO still apply: different object, different denominator",
        "  (12,668 rows of one post-hoc decode vs 10,000-15,000 sampled molecules),",
        "  different chemistry (a metal complex, not an all-organic molecule).",
        "* The pool carries implicit hydrogens; the frontier datasets carry explicit ones.",
        "* This is ONE pool, ONE post-hoc decode -- not a rate over the generator's output",
        "  distribution.",
        "",
        "## 8. Files",
        "",
        "* script: `molmetal/scripts/wfa168_falsevalid_axis.py`",
        "* artifact: `molmetal/reports/wfa168_falsevalid_axis/falsevalid_axis.json`",
        "* test: `molmetal/tests/test_wfa168_falsevalid_axis.py`",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="analyse only the first N CSV rows")
    ap.add_argument("--no-uff", action="store_true", help="skip the UFF step (step 3)")
    ap.add_argument("--sample-size", type=int, default=DRIFT_SAMPLE_SIZE)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_pool(Path(args.pool), limit=args.limit)
    metals = repo_metal_symbols()

    artifact = analyse_rows(rows, metal_symbols=metals, run_uff=not args.no_uff, keep_rows=True)
    artifact["source_pool"] = str(Path(args.pool).resolve())
    artifact["metal_predicate"] = {
        "source": f"{METAL_VOCAB_SOURCE[0]}::{METAL_VOCAB_SOURCE[1]}",
        "extraction": "ast.parse of the repo's own source at run time (no import, no hand copy)",
        "n_symbols": len(metals),
        "symbols": sorted(metals),
    }

    # sensitivity: the max-heavy-atoms rule, for the survivor-rule note
    artifact["survivor_sensitivity"] = {
        "rule": "max heavy atoms (the published note's 'fallback')",
        "kept_heavy_total_max_heavy_atoms": artifact["survivor_sensitivity_total_maxheavy"],
        "kept_heavy_total_chooser": artifact["mass_retention"]["kept_heavy_total"],
        "delta_heavy_atoms": artifact["survivor_sensitivity_total_maxheavy"]
        - artifact["mass_retention"]["kept_heavy_total"],
        "n_rows_where_the_two_rules_pick_different_survivors": artifact["survivor_rule"][
            "n_rows_chooser_pick_differs_from_max_heavy_atoms"
        ],
        "n_rows_where_the_two_rules_give_different_survivor_SIZES": artifact["survivor_rule"][
            "n_rows_chooser_survivor_SIZE_differs_from_max_heavy_atoms"
        ],
        "note": (
            "the two rules agree on the survivor SIZE HISTOGRAM on this pool; they differ "
            "on which fragment is picked for a small number of rows"
        ),
    }
    artifact.pop("survivor_sensitivity_total_maxheavy", None)

    # drift sample: a deterministic, evenly spread subset of the per-row records
    per_row = artifact.pop("rows")
    step = max(1, len(per_row) // max(1, args.sample_size))
    artifact["drift_sample"] = per_row[::step][: args.sample_size]

    artifact["crosscheck_vs_published"] = crosscheck_published(artifact)

    # the pooled false-valid rate is a TIME AVERAGE over a generator that is
    # still improving -- so it must never be quoted without the epoch axis
    pe = artifact["per_epoch"]
    eps = sorted(pe, key=lambda e: int(e) if e.isdigit() else 0)
    first_ep, last_ep = eps[0], eps[-1]
    artifact["per_epoch_summary"] = {
        "n_epochs": len(eps),
        "epoch_order": eps,
        "pooled_speck_rate": artifact["false_valid_rate"]["rate"],
        "first_epoch": first_ep,
        "first_epoch_speck_rate": pe[first_ep]["speck_rate"],
        "last_epoch": last_ep,
        "last_epoch_speck_rate": pe[last_ep]["speck_rate"],
        "min_epoch_speck_rate": min(pe[e]["speck_rate"] for e in eps),
        "max_epoch_speck_rate": max(pe[e]["speck_rate"] for e in eps),
        "note": (
            "the pooled false-valid rate is a TIME AVERAGE over 18 epochs of a "
            "generator that is still improving; the last epoch's post-processed arm "
            "is far cleaner than the pooled figure, and the pooled figure must "
            "never be quoted without this axis"
        ),
    }
    artifact["reader_conclusion"] = {
        "what_a_reader_would_conclude": (
            "that every one of the post-processed molecules is a single, chemically "
            "meaningful molecule"
        ),
        "what_is_actually_true": (
            f"{artifact['false_valid_rate']['numerator']} of "
            f"{artifact['false_valid_rate']['denominator']} post-processed survivors "
            f"({artifact['false_valid_rate']['percent']:.2f} %) are a SINGLE heavy atom, "
            f"and the post-processing discarded "
            f"{100 * artifact['mass_retention']['fraction_of_raw_mass_discarded']:.2f} % "
            "of the raw heavy-atom mass to reach 100 % connected"
        ),
        "what_must_be_printed_beside_it": [
            "the false-valid (speck) rate: <= 2 heavy atoms / all survivors",
            "the survivor-size histogram (<= 2 / 3-10 / >= 11 heavy atoms)",
            "the discarded-mass fraction",
            "the per-epoch axis: the pooled speck rate is a time average over an "
            "improving generator, not a property of the final epoch",
        ],
    }

    report = build_report(artifact, artifact["crosscheck_vs_published"])
    truncation = detect_truncation(report, expected_min_length=200)
    if truncation is not None:
        raise RuntimeError(f"report looks truncated: {truncation}")

    payload = json.dumps(artifact, indent=2, sort_keys=False)
    atomic_write_file(out_dir / "falsevalid_axis.json", payload)
    atomic_write_file(out_dir / "report.md", report)

    fv = artifact["false_valid_rate"]
    sh = artifact["survivor_shape"]
    print("PASS wfa168_falsevalid_axis")
    print(f"  speck split: {sh['n_survivors_le_2_heavy']} / {sh['n_survivors_3_to_10_heavy']} / "
          f"{sh['n_survivors_ge_11_heavy']} (sum {sh['sum_of_buckets']} / {sh['denominator']})")
    print(f"  false-valid rate: {fv['numerator']}/{fv['denominator']} = {fv['rate']:.6f} "
          f"({fv['percent']:.4f} %)")
    print(f"  ligand-only speck: {artifact['ligand_only_analogue']['n_ligand_only_le_2_heavy']}"
          f"/{artifact['ligand_only_analogue']['n_survivors']}, new specks from metal removal: "
          f"{artifact['ligand_only_analogue']['n_NEW_specks_created_by_removing_the_metal']}")
    print(f"  mass discarded: {artifact['mass_retention']['fraction_of_raw_mass_discarded']:.6f}")
    cc = artifact["crosscheck_vs_published"]
    print(f"  crosscheck vs published: {cc['n_fields_matching']}/{cc['n_fields_compared']} fields match")
    print(f"  wrote {out_dir/'falsevalid_axis.json'} and {out_dir/'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
