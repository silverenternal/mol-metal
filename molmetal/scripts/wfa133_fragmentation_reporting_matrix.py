#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A133 fragmentation REPORTING MATRIX -- CPU-only, RDKit-only, no GPU.

WHY THIS EXISTS
---------------
The project publishes ONE fragmentation scalar (``fragmented_ratio``) whose
denominator is implicit and whose population is implicit.  A 2024-2026
reviewer does not read one scalar: the field's own protocol (Buttenschoen et
al., arXiv:2505.00518; DiffSBDD, arXiv:2210.13695 App. G.2 Table 6; MOSAIC,
arXiv:2610.04161) is a MATRIX of SEPARATELY-DENOMINATED rates -- chemical
validity, connectivity, connected-valid, the largest-component fraction, the
post-processed arm, the field band -- each over a stated denominator.

This tool BUILDS that matrix from files already on disk, so the matrix can be
produced (and later registered) without a GPU, without a decode and without a
checkpoint.  It is a REPORTING tool, not a training tool.

WHAT IT GUARANTEES
------------------
1. **The fragment definition cannot drift.**  Every RAW connectivity cell is
   derived from :func:`molmetal.scripts.cfm_fragmentation_gate.score_smiles_pool`,
   which calls the trainer's OWN ``decode_chemistry``
   (``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``).  This tool does
   NOT re-implement "is this one connected molecule?".
2. **Every cell names its denominator.**  A rate without its denominator is
   unreadable (``cfm_fragmentation_gate.py:60-70``): a pool file holds the
   molecules that SURVIVED, so a rate over its rows is a rate over survivors.
   The denominator ladder -- drawn / pool rows / parsed / valid / connected --
   is published for every slice.
3. **RAW vs POST-PROCESSED is stated per cell**, because the field's headline
   connectivity numbers are POST-PROCESSED and ours are RAW.  Comparing the
   two without the label is a category error (END, arXiv:2506.10532 App.
   A.6.1: "% valid" does not penalize fragmented samples).
4. **An uncomputable cell is NAMED, not omitted.**  Each UNAVAILABLE cell
   carries the blocker, so a reader can tell "we did not measure this" from
   "this cell does not exist".
5. **The population is named.**  The c7 pool is an 18-epoch trajectory
   spanning a KILLED pre-resume attempt (epochs 0-3) and the run of record
   (epochs 4-17); the pooled headline and the final-epoch headline describe
   DIFFERENT things.  Every slice says which rows it holds.

RAW vs POST-PROCESSED, EXACTLY
-------------------------------
* **RAW** -- the SMILES as the sampler emitted it.  Connectivity is the number
  of connected components in that string.  This is the arm that describes the
  generator.
* **POST-PROCESSED** -- the field's repair pipeline: RDKit
  ``LargestFragmentChooser(preferOrganic=True)`` -> ``SanitizeMol`` ->
  ``AddHs`` -> the repo's own ``min_atoms >= 2`` rule
  (``molmetal/eval/scoring/decode_gate.py:18-19``).
  **The UFF step is OMITTED and the cell is marked UNAVAILABLE**: the pool
  holds SMILES only, so there is no model geometry to minimise.
* The post-processed CONNECTIVITY cell is **1.000000 by construction**
  (``LargestFragmentChooser`` returns exactly one fragment).  It is published
  with an explicit ``tautology: true`` flag rather than dropped, because
  reporting it as a result would be reporting a tautology, and hiding it would
  hide the reason the informative post-processed cells are validity-after-
  selection and mass-discarded.

WHAT IS UNAVAILABLE ON THIS HOST, AND WHY
------------------------------------------
* ``posebusters_valid`` / ``strain_or_internal_energy`` / the UFF arm -- the
  pool holds SMILES only; there are no coordinates on disk.
* ``bonds_dropped_histogram`` -- the CFM checkpoints cannot be loaded on a
  CPU-only host (``torch.load(..., map_location="cpu")`` raises
  ``RuntimeError: Attempting to deserialize object on a CUDA device but
  torch.cuda.is_available() is False``).  This tool never loads a checkpoint.
* ``novelty`` / ``intdiv1`` -- the published diversity artifact lives on a
  DIFFERENT pool (``cfm_fullcorpus_2026_10_10``, 66 rows); computing them here
  would mix populations.  Uniqueness IS computed, on THIS pool.
* ``downstream_properties_on_connected_subset`` -- the property artifacts are
  on a different pool; this tool does not recompute them.
* ``docking_redock`` -- needs generated poses and a docking engine; the Vina
  panel is BLOCKED on Pt+2.

USAGE
-----
    # print the matrix as JSON on stdout (nothing is written to disk)
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa133_fragmentation_reporting_matrix.py

    # same, but human-readable
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa133_fragmentation_reporting_matrix.py --format text

    # write the JSON to a path the CALLER names (never under metrics/ or
    # registry/ -- this tool refuses those roots)
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa133_fragmentation_reporting_matrix.py \
        --out /tmp/wfa133/fragmentation_matrix.json

CPU-only and RDKit-only.  No GPU, no model, no sampler, no checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.cfm_fragmentation_gate import score_smiles_pool  # noqa: E402
from molmetal.scripts.cfm_pool_epoch_rule import (  # noqa: E402
    census_row as _census_row,
    pool_population_composition,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: The run this tool was written for (the pool the project's fragmentation
#: metrics are measured on).
DEFAULT_RUN_DIR = REPO / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20"

#: The real corpus the generation pool is compared against.  It is a CONTROL,
#: not a matrix cell: it answers "what does real chemistry look like under the
#: same scorer".
DEFAULT_REFERENCE = REPO / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"

#: The run census -- the SSOT of which run wrote what.
CENSUS_PATH = REPO / "metrics" / "by_metric" / "cfm_run_census.json"

#: The metals the corpus writes as their own component.  Same set as the
#: trainer's ``_TARGET_METALS`` (Ru Rh Re Os Ir Pt), so the "metal in its own
#: component" question is asked with the corpus's own vocabulary.
TARGET_METALS = frozenset({44, 45, 75, 76, 77, 78})

#: CBGBench's published rule (arXiv:2406.10840 App. E.4): "the largest fragment
#: holds more than 85% of the atoms".
CBGBENCH_LARGEST_FRAG_MIN = 0.85

#: The repo's own declared gate floor
#: (``cfm_fragmentation_gate.py::DEFAULT_MIN_LARGEST_FRAGMENT_FRAC``).  It is
#: declared POLICY, not a published optimum, and is labelled as such.
REPO_GATE_LARGEST_FRAG_MIN = 0.90

#: The published field band this project's honest negative is measured
#: against.  Every number here is PUBLISHED (quoted from the Wf-A130 frontier
#: scans); this tool does NOT re-measure them and does NOT re-fetch them.
FIELD_BAND: Dict[str, Any] = {
    "tag": "PUBLISHED",
    "convention": (
        "one connected molecule, NO post-processing, GEOM-Drugs / CrossDocked "
        "as stated per source"
    ),
    "published": {
        "GeoLDM": 0.374,
        "FlowMol": 0.686,
        "EQGAT_diff": 0.844,
        "SemlaFlow": 0.923,
        "EDM_connected": 0.409,
        "EDM_connected_valid": 0.349,
        "MOSAIC_connected": 0.800,
        "MegalodonFlow_connected": 0.960,
    },
    "worst_published_connected": 0.374,
    "citations": [
        "arXiv:2505.00518 (Buttenschoen et al., GEM @ ICLR 2025)",
        "arXiv:2610.04161 (MOSAIC)",
        "arXiv:2603.12734 (VecMol)",
        "arXiv:2210.13695 (DiffSBDD)",
    ],
    "provenance": (
        "quoted from the Wf-A130 frontier scans "
        "(/tmp/wfa130/l1-fragment-validity.md, l2-postproc-assembly.md, "
        "l3-eval-protocol.md); NOT re-fetched and NOT re-measured by this "
        "tool -- re-fetch before publication"
    ),
}


# ---------------------------------------------------------------------------
# READING THE POOL AND THE RUN'S OWN EVENTS
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    """The pool file's SHA-256, so a reader can tell WHICH pool was scored."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pool(
    path: Path,
) -> Tuple[List[str], List[Optional[int]], List[Dict[str, str]]]:
    """The pool's rows: the SMILES, their epoch, and the raw row dicts.

    The epoch column is what makes the pool a TRAJECTORY rather than a
    checkpoint, and the raw dicts carry the trainer's own ``n_frags_raw`` /
    ``largest_frag_frac`` columns, which this tool CROSS-CHECKS against the
    re-derivation (the audit's own ``csv_column_cross_check`` idiom).
    """
    smiles: List[str] = []
    epochs: List[Optional[int]] = []
    raw_rows: List[Dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            raw_rows.append(row)
            smiles.append((row.get("smiles") or "").strip())
            try:
                epochs.append(int(float(row.get("epoch", ""))))
            except (TypeError, ValueError):
                epochs.append(None)
    return smiles, epochs, raw_rows


def _events_sample_counts(run_dir: Path) -> Dict[int, float]:
    """``epoch -> samples drawn``, from the run's own ``events.jsonl``.

    The pool file holds the RDKit-strict SURVIVORS only, so its row count is
    NOT the number of molecules drawn.  Same reader as
    ``cfm_fragmentation_audit._sample_counts`` (mirrored here so this tool
    stands alone); the source is named next to every rate that uses it.
    """
    out: Dict[int, float] = {}
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return out
    try:
        for line in path.open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "gen_validity_n" not in row:
                continue
            try:
                ep = int(float(row.get("epoch", 0)))
                n = float(row["gen_validity_n"])
            except (TypeError, ValueError):
                continue
            out[ep] = n
    except OSError:
        return out
    return out


def _census_for(run_name: str) -> Optional[Dict[str, Any]]:
    """The run census's own row for ``run_name``, or ``None``."""
    if not CENSUS_PATH.is_file():
        return None
    try:
        census = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _census_row(census, run_name)


# ---------------------------------------------------------------------------
# THE SECOND PASS -- the cells the repo's scorer does not compute
# ---------------------------------------------------------------------------
def _composition_facts(mol) -> Dict[str, Any]:
    """How the molecule's components are COMPOSED.

    ``n_organic_components`` counts the components that hold at least one
    heavy atom that is not one of the corpus's own metals: the corpus writes
    the metal as its own component, so a CORRECT generation has several
    SUBSTANTIAL components and "one component" is not "one molecule"
    (``c7_fullgen_pool_quality.py:14-24``).
    """
    from rdkit import Chem

    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    sizes = [len(f) for f in frags]
    zs = [int(a.GetAtomicNum()) for a in mol.GetAtoms()]
    n_organic = 0
    metal_only = False
    metal_single_atom = False
    for f in frags:
        frag_z = [zs[i] for i in f]
        if all(z in TARGET_METALS for z in frag_z):
            metal_only = True
            if len(f) == 1:
                metal_single_atom = True
        else:
            n_organic += 1
    ordered = sorted(sizes, reverse=True)
    return {
        "n_components": len(sizes),
        "n_organic_components": n_organic,
        "metal_only_component": metal_only,
        "metal_single_atom_component": metal_single_atom,
        "largest_frag_single_heavy_atom": (max(sizes) == 1) if sizes else False,
        "second_largest_size": ordered[1] if len(ordered) >= 2 else 0,
    }


def _postprocess_facts(mol, chooser) -> Dict[str, Any]:
    """The POST-PROCESSED arm's facts for ONE molecule.

    The pipeline is the field's, minus the step this host cannot run:

        LargestFragmentChooser(preferOrganic=True) -> SanitizeMol -> AddHs
        -> at least 2 heavy atoms (the repo's own min_atoms rule)

    The UFF minimisation of the model's geometry is NOT here: the pool holds
    SMILES only, so there is no model geometry to minimise.  The cell that
    would need it is reported as UNAVAILABLE, not silently dropped.
    """
    from rdkit import Chem

    facts: Dict[str, Any] = {
        "postproc_parsed": False,
        "postproc_sanitized": False,
        "postproc_kept_heavy": 0,
        "postproc_min2heavy": False,
        "mass_discarded_frac": None,
    }
    try:
        kept = chooser.choose(mol)
    except Exception:
        return facts
    facts["postproc_parsed"] = True
    try:
        Chem.SanitizeMol(kept)
    except Exception:
        return facts
    facts["postproc_sanitized"] = True
    try:
        kept = Chem.AddHs(kept)
    except Exception:
        pass
    n_kept = int(kept.GetNumHeavyAtoms())
    facts["postproc_kept_heavy"] = n_kept
    facts["postproc_min2heavy"] = n_kept >= 2
    n_heavy = int(mol.GetNumHeavyAtoms())
    if n_heavy:
        facts["mass_discarded_frac"] = (n_heavy - n_kept) / n_heavy
    return facts


def _second_pass(smiles: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    """Per-row facts the repo's scorer does not produce.

    Returns a list parallel to ``smiles`` with ``None`` for the rows RDKit
    refused (the same rows the repo's scorer drops), so the two passes can be
    lined up by INDEX and cross-checked.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    out: List[Optional[Dict[str, Any]]] = []
    for text in smiles:
        if not text:
            out.append(None)
            continue
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            out.append(None)
            continue
        facts = _composition_facts(mol)
        facts.update(_postprocess_facts(mol, chooser))
        facts["canonical_smiles"] = Chem.MolToSmiles(mol)
        out.append(facts)
    return out


# ---------------------------------------------------------------------------
# CELL CONSTRUCTION
# ---------------------------------------------------------------------------
def _cell(
    value: Optional[float],
    *,
    num: Optional[int] = None,
    den: Optional[int] = None,
    denominator: str = "",
    arm: str = "RAW",
    source: str = "",
    note: Optional[str] = None,
    tautology: bool = False,
) -> Dict[str, Any]:
    """One matrix cell: the value, its numerator/denominator, and its arm."""
    record: Dict[str, Any] = {
        "value": value,
        "arm": arm,
        "num": num,
        "den": den,
        "denominator": denominator,
        "source": source,
    }
    if tautology:
        record["tautology"] = True
    if note:
        record["note"] = note
    return record


def _slice_cells(
    *,
    slice_name: str,
    slice_note: str,
    n_pool_rows: int,
    n_parsed: int,
    n_drawn: Optional[int],
    n_single: int,
    n_unparseable: int,
    n_empty: int,
    lcf_mean: Optional[float],
    n_fragments_mean: Optional[float],
    n_largest_gt_085: int,
    n_largest_ge_090: int,
    n_one_organic: int,
    n_metal_only: int,
    n_metal_single_atom: int,
    n_largest_single_heavy_atom: int,
    n_second_frag_single_atom: int,
    n_rows_with_2plus_components: int,
    n_frags_2_to_6: int,
    n_frags_ge21: int,
    ring_counts: Dict[int, int],
    n_unique: int,
    n_postproc_valid: int,
    n_postproc_min2heavy: int,
    n_postproc_single_heavy_atom: int,
    mass_discarded: List[float],
    cell_source: str,
) -> Dict[str, Any]:
    """Every cell of ONE population slice, each with its denominator."""
    den_drawn = n_drawn if n_drawn else None
    raw: Dict[str, Any] = {}
    raw["validity"] = _cell(
        (n_parsed / n_pool_rows) if n_pool_rows else None,
        num=n_parsed,
        den=n_pool_rows,
        denominator="n_pool_rows",
        source=cell_source,
        note="RDKit parse + strict sanitize; a string RDKit refused is NOT valid",
    )
    raw["connectivity_conditional"] = _cell(
        (n_single / n_parsed) if n_parsed else None,
        num=n_single,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note="exactly one connected component, conditioned on the row having parsed",
    )
    raw["success_rate_over_pool_rows"] = _cell(
        (n_single / n_pool_rows) if n_pool_rows else None,
        num=n_single,
        den=n_pool_rows,
        denominator="n_pool_rows",
        source=cell_source,
        note="a rate over SURVIVORS: the pool file holds the molecules that survived",
    )
    raw["success_rate_end_to_end"] = _cell(
        (n_single / den_drawn) if den_drawn else None,
        num=n_single,
        den=den_drawn,
        denominator="n_drawn_samples",
        source=cell_source,
        note=(
            "the field's end-to-end convention: unparseable and empty draws "
            "stay in the denominator"
        ),
    )
    raw["fragmented_ratio_over_pool_rows"] = _cell(
        (1.0 - n_single / n_pool_rows) if n_pool_rows else None,
        num=n_single,
        den=n_pool_rows,
        denominator="n_pool_rows",
        source=cell_source,
        note="complement of success_rate_over_pool_rows",
    )
    raw["fragmented_ratio_end_to_end"] = _cell(
        (1.0 - n_single / den_drawn) if den_drawn else None,
        num=n_single,
        den=den_drawn,
        denominator="n_drawn_samples",
        source=cell_source,
        note="complement of success_rate_end_to_end",
    )
    raw["largest_component_frac_mean"] = _cell(
        lcf_mean,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note="mean over the parsed rows of (largest fragment heavy atoms / heavy atoms)",
    )
    raw["largest_frac_gt_0p85"] = _cell(
        (n_largest_gt_085 / n_parsed) if n_parsed else None,
        num=n_largest_gt_085,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note=(
            "CBGBench rule (arXiv:2406.10840 App. E.4): the largest fragment "
            "holds MORE THAN 85% of the heavy atoms"
        ),
    )
    raw["largest_frac_ge_0p90"] = _cell(
        (n_largest_ge_090 / n_parsed) if n_parsed else None,
        num=n_largest_ge_090,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note=(
            "the repo's OWN declared gate floor "
            "(cfm_fragmentation_gate.py) -- declared POLICY, not a published "
            "optimum"
        ),
    )
    raw["metal_aware_connectivity_one_ligand"] = _cell(
        (n_one_organic / n_parsed) if n_parsed else None,
        num=n_one_organic,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note=(
            "the metal-aware reading: exactly ONE component holds a non-metal "
            "heavy atom (the corpus writes the metal as its own component, so "
            "one component is not one molecule)"
        ),
    )
    raw["n_fragments_mean"] = _cell(
        n_fragments_mean,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
    )
    raw["share_n_frags_2_to_6"] = _cell(
        (n_frags_2_to_6 / n_parsed) if n_parsed else None,
        num=n_frags_2_to_6,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
    )
    raw["share_n_frags_ge21"] = _cell(
        (n_frags_ge21 / n_parsed) if n_parsed else None,
        num=n_frags_ge21,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
    )
    raw["second_frag_is_single_atom"] = _cell(
        (n_second_frag_single_atom / n_rows_with_2plus_components)
        if n_rows_with_2plus_components
        else None,
        num=n_second_frag_single_atom,
        den=n_rows_with_2plus_components,
        denominator="n_parsed_rows_with_>=2_components",
        source=cell_source,
        note=(
            "one fused blob plus leftover atoms has the same component COUNT "
            "as several ligands; only this separates them"
        ),
    )
    total_rings = sum(ring_counts.values())
    raw["three_ring_share"] = _cell(
        (ring_counts.get(3, 0) / total_rings) if total_rings else None,
        num=ring_counts.get(3, 0),
        den=total_rings,
        denominator="n_rings_total",
        source=cell_source,
    )
    raw["six_ring_share"] = _cell(
        (ring_counts.get(6, 0) / total_rings) if total_rings else None,
        num=ring_counts.get(6, 0),
        den=total_rings,
        denominator="n_rings_total",
        source=cell_source,
    )
    raw["uniqueness"] = _cell(
        (n_unique / n_parsed) if n_parsed else None,
        num=n_unique,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note="distinct canonical SMILES over the parsed rows; NOVELTY is NOT computed here",
    )
    raw["metal_in_own_component"] = _cell(
        (n_metal_only / n_parsed) if n_parsed else None,
        num=n_metal_only,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note="a component whose heavy atoms are all corpus metals (Ru Rh Re Os Ir Pt)",
    )
    raw["metal_single_atom_component"] = _cell(
        (n_metal_single_atom / n_parsed) if n_parsed else None,
        num=n_metal_single_atom,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note="the metal stands alone as a one-atom component",
    )
    raw["largest_fragment_is_single_heavy_atom"] = _cell(
        (n_largest_single_heavy_atom / n_parsed) if n_parsed else None,
        num=n_largest_single_heavy_atom,
        den=n_parsed,
        denominator="n_parsed",
        source=cell_source,
        note=(
            "the largest fragment is a BARE heavy atom -- the field's "
            "post-processing keeps such a fragment and MOSES's "
            "'at least one heavy atom' rule accepts it"
        ),
    )

    post: Dict[str, Any] = {}
    post["validity_after_selection_min2heavy"] = _cell(
        (n_postproc_min2heavy / n_parsed) if n_parsed else None,
        num=n_postproc_min2heavy,
        den=n_parsed,
        denominator="n_parsed",
        arm="POST-PROCESSED",
        source=cell_source,
        note=(
            "LargestFragmentChooser(preferOrganic=True) -> SanitizeMol -> "
            "AddHs -> at least 2 heavy atoms (the repo's min_atoms rule)"
        ),
    )
    post["validity_after_selection_min2heavy_over_pool_rows"] = _cell(
        (n_postproc_min2heavy / n_pool_rows) if n_pool_rows else None,
        num=n_postproc_min2heavy,
        den=n_pool_rows,
        denominator="n_pool_rows",
        arm="POST-PROCESSED",
        source=cell_source,
        note=(
            "the SAME numerator over the SURVIVOR denominator -- published "
            "alongside the n_parsed row so the two are never confused"
        ),
    )
    post["single_component_after_selection"] = _cell(
        (n_postproc_valid / n_parsed) if n_parsed else None,
        num=n_postproc_valid,
        den=n_parsed,
        denominator="n_parsed",
        arm="POST-PROCESSED",
        source=cell_source,
        tautology=True,
        note=(
            "1.000000 BY CONSTRUCTION -- LargestFragmentChooser returns "
            "exactly one fragment.  NOT a result: it is the reason the "
            "informative post-processed cells are validity-after-selection "
            "and mass-discarded."
        ),
    )
    post["largest_fragment_is_single_heavy_atom"] = _cell(
        (n_postproc_single_heavy_atom / n_parsed) if n_parsed else None,
        num=n_postproc_single_heavy_atom,
        den=n_parsed,
        denominator="n_parsed",
        arm="POST-PROCESSED",
        source=cell_source,
    )
    post["heavy_atom_mass_discarded_mean"] = _cell(
        (sum(mass_discarded) / len(mass_discarded)) if mass_discarded else None,
        den=len(mass_discarded),
        denominator="n_parsed_rows_with_a_discarded_mass",
        arm="POST-PROCESSED",
        source=cell_source,
        note="the honest cost of the repair: heavy atoms dropped by the selection",
    )
    post["heavy_atom_mass_discarded_median"] = _cell(
        (statistics.median(mass_discarded) if mass_discarded else None),
        den=len(mass_discarded),
        denominator="n_parsed_rows_with_a_discarded_mass",
        arm="POST-PROCESSED",
        source=cell_source,
    )

    return {
        "slice": slice_name,
        "population_note": slice_note,
        "counts": {
            "n_drawn_samples": den_drawn,
            "n_pool_rows": n_pool_rows,
            "n_parsed": n_parsed,
            "n_unparseable": n_unparseable,
            "n_empty": n_empty,
            "n_connected": n_single,
            "n_valid": n_parsed,
        },
        "raw": raw,
        "post_processed": post,
    }


# ---------------------------------------------------------------------------
# THE MATRIX
# ---------------------------------------------------------------------------
def build_matrix(
    run_dir: Path,
    *,
    pool_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the whole matrix for ONE run directory.

    Every number it returns is MEASURED on this host, except the
    ``field_band`` block, which is PUBLISHED and labelled as such.
    """
    run_dir = Path(run_dir)
    if pool_path is None:
        pool_path = run_dir / "generated_smiles.csv"
    pool_path = Path(pool_path)

    smiles, epochs, raw_rows = _read_pool(pool_path)
    sample_counts = _events_sample_counts(run_dir)
    census = _census_for(run_dir.name)
    composition = pool_population_composition(run_dir, pool_path, census_row=census)
    run_of_record_epochs = list(composition.get("run_of_record_epochs") or [])
    pool_epochs = list(composition.get("pool_epochs") or [])
    killed_epochs = [e for e in pool_epochs if e not in set(run_of_record_epochs)]
    pool_last_epoch = composition.get("pool_epoch")

    # --- the repo's OWN scorer: the RAW fragment definition -----------------
    scored = score_smiles_pool(smiles)
    gate_rows = scored.get("rows") or []
    gate_indexes = list(scored.get("row_indexes") or [])
    gate_by_input: Dict[int, Dict[str, Any]] = {
        int(idx): row for idx, row in zip(gate_indexes, gate_rows)
    }

    # --- the second pass: the cells the repo's scorer does not compute -----
    second = _second_pass(smiles)

    # --- the cross-checks: the pool's OWN columns vs the re-derivation ------
    n_frags_checked = 0
    n_frags_mismatch = 0
    n_frac_checked = 0
    n_frac_mismatch = 0
    n_comp_mismatch = 0
    for k, idx in enumerate(gate_indexes):
        raw = raw_rows[idx]
        try:
            recorded = int(float(raw.get("n_frags_raw", "")))
        except (TypeError, ValueError):
            recorded = None
        if recorded is not None:
            n_frags_checked += 1
            if recorded != int(gate_rows[k].get("n_fragments", 0) or 0):
                n_frags_mismatch += 1
        try:
            recorded_frac = float(raw.get("largest_frag_frac", ""))
        except (TypeError, ValueError):
            recorded_frac = None
        if recorded_frac is not None:
            n_frac_checked += 1
            # The pool's own column is ROUNDED to 4 decimals, so the
            # agreement is asserted to the column's own precision (half of
            # the last printed digit), not to floating-point equality.
            if abs(recorded_frac - float(gate_rows[k]["largest_fragment_frac"])) > 1e-4:
                n_frac_mismatch += 1
        facts = second[idx]
        if facts is not None and facts["n_components"] != int(
            gate_rows[k].get("n_fragments", 0) or 0
        ):
            n_comp_mismatch += 1

    # --- the slices --------------------------------------------------------
    def _slice(
        name: str,
        note: str,
        positions: List[int],
        drawn_epochs: Optional[Sequence[int]],
    ) -> Dict[str, Any]:
        """One population slice, derived from the repo scorer's per-row facts.

        The slice re-uses the SAME ``score_smiles_pool`` per-row facts the
        full pool was scored with (indexed by the pool's own row index, via
        the scorer's ``row_indexes``), so a slice's fragment definition is
        literally the full pool's.
        """
        sub_rows = [gate_by_input[p] for p in positions if p in gate_by_input]
        n_parsed = len(sub_rows)
        n_single = sum(1 for r in sub_rows if int(r.get("n_fragments", 0) or 0) == 1)
        lcf_values = [
            float(r.get("largest_fragment_frac", 0.0) or 0.0) for r in sub_rows
        ]
        nfrag_values = [int(r.get("n_fragments", 0) or 0) for r in sub_rows]
        ring_counts: Dict[int, int] = {}
        for r in sub_rows:
            for size in r.get("ring_sizes") or []:
                ring_counts[int(size)] = ring_counts.get(int(size), 0) + 1

        n_largest_gt_085 = sum(1 for v in lcf_values if v > CBGBENCH_LARGEST_FRAG_MIN)
        n_largest_ge_090 = sum(1 for v in lcf_values if v >= REPO_GATE_LARGEST_FRAG_MIN)
        n_one_organic = 0
        n_metal_only = 0
        n_metal_single = 0
        n_largest_single_heavy = 0
        n_second_single = 0
        n_rows_2plus = 0
        n_frags_2to6 = 0
        n_frags_ge21 = 0
        unique = set()
        n_postproc_valid = 0
        n_postproc_min2 = 0
        n_postproc_single_heavy = 0
        mass_discarded: List[float] = []
        n_empty = 0
        n_unparseable = 0
        for pos in positions:
            facts = second[pos]
            if facts is None:
                if smiles[pos]:
                    n_unparseable += 1
                else:
                    n_empty += 1
                continue
            if facts["n_organic_components"] == 1:
                n_one_organic += 1
            if facts["metal_only_component"]:
                n_metal_only += 1
            if facts["metal_single_atom_component"]:
                n_metal_single += 1
            if facts["largest_frag_single_heavy_atom"]:
                n_largest_single_heavy += 1
            n_comp = int(facts["n_components"])
            if n_comp >= 2:
                n_rows_2plus += 1
                if int(facts["second_largest_size"]) == 1:
                    n_second_single += 1
            if 2 <= n_comp <= 6:
                n_frags_2to6 += 1
            if n_comp >= 21:
                n_frags_ge21 += 1
            unique.add(facts["canonical_smiles"])
            if facts["postproc_parsed"] and facts["postproc_sanitized"]:
                n_postproc_valid += 1
                if int(facts["postproc_kept_heavy"]) == 1:
                    # The retained fragment is a BARE heavy atom.  Counted
                    # OUTSIDE the min-2-heavy-atom branch: a one-heavy-atom
                    # fragment is exactly what the min-2 rule REJECTS, so
                    # counting it inside would make the counter always zero.
                    n_postproc_single_heavy += 1
                if facts["postproc_min2heavy"]:
                    n_postproc_min2 += 1
                if facts["mass_discarded_frac"] is not None:
                    mass_discarded.append(float(facts["mass_discarded_frac"]))

        if drawn_epochs is None:
            drawn = (
                sum(
                    float(v)
                    for k, v in sample_counts.items()
                    if int(k) in set(pool_epochs)
                )
                or None
            )
        else:
            wanted = set(int(e) for e in drawn_epochs)
            drawn = (
                sum(float(v) for k, v in sample_counts.items() if int(k) in wanted)
                or None
            )
        return _slice_cells(
            slice_name=name,
            slice_note=note,
            n_pool_rows=len(positions),
            n_parsed=n_parsed,
            n_drawn=int(drawn) if drawn else None,
            n_single=n_single,
            n_unparseable=n_unparseable,
            n_empty=n_empty,
            lcf_mean=(sum(lcf_values) / n_parsed) if n_parsed else None,
            n_fragments_mean=(sum(nfrag_values) / n_parsed) if n_parsed else None,
            n_largest_gt_085=n_largest_gt_085,
            n_largest_ge_090=n_largest_ge_090,
            n_one_organic=n_one_organic,
            n_metal_only=n_metal_only,
            n_metal_single_atom=n_metal_single,
            n_largest_single_heavy_atom=n_largest_single_heavy,
            n_second_frag_single_atom=n_second_single,
            n_rows_with_2plus_components=n_rows_2plus,
            n_frags_2_to_6=n_frags_2to6,
            n_frags_ge21=n_frags_ge21,
            ring_counts=ring_counts,
            n_unique=len(unique),
            n_postproc_valid=n_postproc_valid,
            n_postproc_min2heavy=n_postproc_min2,
            n_postproc_single_heavy_atom=n_postproc_single_heavy,
            mass_discarded=mass_discarded,
            cell_source=(
                "molmetal/scripts/cfm_fragmentation_gate.py::score_smiles_pool "
                "(the trainer's own decode_chemistry)"
            ),
        )

    all_positions = list(range(len(smiles)))
    ro_positions = [
        i
        for i, ep in enumerate(epochs)
        if ep is None or ep in set(run_of_record_epochs)
    ]
    final_positions = [
        i for i, ep in enumerate(epochs) if ep is None or ep == pool_last_epoch
    ]

    full = _slice(
        "full_pool",
        (
            "EVERY row of the pool file (all epochs).  The pool is an "
            "18-epoch TRAJECTORY spanning a killed pre-resume attempt and the "
            "run of record, so this slice is the POOL's composition, not a "
            "checkpoint's behaviour."
        ),
        all_positions,
        drawn_epochs=None,
    )
    ro = _slice(
        "run_of_record",
        "the epochs the run of record ran (the census's own run_of_record_epoch_indices).",
        ro_positions,
        drawn_epochs=run_of_record_epochs,
    )
    final = _slice(
        "final_epoch",
        (
            "the pool's own last epoch -- the only slice that describes the "
            "final checkpoint."
        ),
        final_positions,
        drawn_epochs=[pool_last_epoch] if pool_last_epoch is not None else None,
    )

    # The slice derivation is INDEPENDENTLY re-checked against the repo
    # scorer, on the smallest slice, so "a slice re-uses the full pool's
    # per-row facts" is a tested claim and not an assertion.
    slice_consistency: Dict[str, Any] = {"checked": False}
    if final_positions:
        sub = score_smiles_pool([smiles[i] for i in final_positions])
        slice_consistency = {
            "checked": True,
            "slice": "final_epoch",
            "n_parsed_rederived": int(sub.get("n_scored", 0) or 0),
            "n_parsed_from_slice": final["counts"]["n_parsed"],
            "n_single_rederived": int(sub.get("n_single", 0) or 0),
            "n_single_from_slice": final["counts"]["n_connected"],
            "largest_fragment_frac_mean_rederived": sub.get(
                "largest_fragment_frac_mean"
            ),
            "largest_fragment_frac_mean_from_slice": final["raw"][
                "largest_component_frac_mean"
            ]["value"],
            "source": (
                "an independent score_smiles_pool() call on the slice's own "
                "SMILES, compared with the slice derived from the full pool's "
                "per-row facts"
            ),
        }

    controls: Dict[str, Any] = {}
    if reference_path is not None and Path(reference_path).is_file():
        controls["reference_corpus"] = _reference_control(Path(reference_path))

    payload: Dict[str, Any] = {
        "metric": "wfa133_fragmentation_reporting_matrix",
        "tag": "MEASURED",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": {
            "script": "molmetal/scripts/wfa133_fragmentation_reporting_matrix.py",
            "cpu_only": True,
            "gpu_used": False,
            "checkpoint_loaded": False,
            "fragment_definition_owner": (
                "molmetal/scripts/train_cfm_yuelbond_wasserstein.py::"
                "decode_chemistry, via "
                "molmetal/scripts/cfm_fragmentation_gate.py::score_smiles_pool"
            ),
            "postprocessing_pipeline": (
                "rdkit.Chem.MolStandardize.LargestFragmentChooser("
                "preferOrganic=True) -> SanitizeMol -> AddHs -> "
                "heavy atoms >= 2 (the repo's min_atoms rule)"
            ),
            "postprocessing_uff_step": (
                "OMITTED -- the pool holds SMILES only, so the model's "
                "geometry does not exist to minimise; the cell that needs it "
                "is reported as UNAVAILABLE"
            ),
            "writes_under_metrics_or_registry": False,
        },
        "population": {
            "run_dir": _repo_relative(run_dir),
            "pool_path": _repo_relative(pool_path),
            "pool_sha256": _sha256(pool_path),
            "n_pool_rows": len(smiles),
            "pool_epochs": pool_epochs,
            "run_of_record_epochs": run_of_record_epochs,
            "killed_pre_resume_epochs": killed_epochs,
            "pool_last_epoch": pool_last_epoch,
            "population_note": composition.get("note"),
            "population_source": composition.get("population_source"),
            "census_row_found": census is not None,
        },
        "denominators": {
            "n_drawn_samples": full["counts"]["n_drawn_samples"],
            "n_drawn_samples_source": (
                "events.jsonl:train_epoch[*].gen_validity_n summed over the "
                "pool's epochs (the trainer's own epoch-level draws)"
            ),
            "n_pool_rows": full["counts"]["n_pool_rows"],
            "n_parsed": full["counts"]["n_parsed"],
            "n_valid": full["counts"]["n_valid"],
            "n_connected": full["counts"]["n_connected"],
            "ladder_note": (
                "drawn >= pool rows >= parsed; a rate over pool rows is a rate "
                "over SURVIVORS and must never be called a per-sample rate"
            ),
        },
        "cells": {
            "full_pool": full,
            "run_of_record": ro,
            "final_epoch": final,
        },
        "cross_checks": {
            "pool_n_frags_raw_column": {
                "n_checked": n_frags_checked,
                "n_mismatch": n_frags_mismatch,
                "source": "the pool's own 'n_frags_raw' column vs the re-derivation",
            },
            "pool_largest_frag_frac_column": {
                "n_checked": n_frac_checked,
                "n_mismatch": n_frac_mismatch,
                "source": (
                    "the pool's own 'largest_frag_frac' column (ROUNDED to 4 "
                    "decimals) vs the re-derivation; agreement is asserted to "
                    "the column's own precision"
                ),
            },
            "two_pass_component_agreement": {
                "n_checked": len(gate_rows),
                "n_mismatch": n_comp_mismatch,
                "source": "the repo scorer's component count vs this tool's second pass",
            },
            "slice_derivation_consistency": slice_consistency,
        },
        "field_band": FIELD_BAND,
        "unavailable_cells": [
            {
                "cell": "posebusters_valid",
                "reason": (
                    "the pool holds SMILES only -- there is no coordinate dump "
                    "on disk, and every PoseBusters check is a geometry check"
                ),
                "blocker": "no coordinates",
            },
            {
                "cell": "strain_or_internal_energy",
                "reason": (
                    "same: an energy/strain measurement needs the model's "
                    "geometry, and the pool stores only SMILES"
                ),
                "blocker": "no coordinates",
            },
            {
                "cell": "postproc_validity_with_uff_on_model_geometry",
                "reason": (
                    "the field's pipeline (arXiv:2505.00518) ends in a UFF "
                    "minimisation of the MODEL's geometry; the pool stores no "
                    "geometry, so the 2-D arm is published WITHOUT the UFF "
                    "step and this cell is NOT the field's number"
                ),
                "blocker": "no coordinates",
            },
            {
                "cell": "bonds_dropped_histogram",
                "reason": (
                    "needs a decode; the CFM checkpoints cannot be loaded on a "
                    "CPU-only host (torch.load with map_location='cpu' raises "
                    "RuntimeError: Attempting to deserialize object on a CUDA "
                    "device but torch.cuda.is_available() is False)"
                ),
                "blocker": "CUDA-serialised checkpoint",
            },
            {
                "cell": "novelty_and_intdiv1",
                "reason": (
                    "the published diversity artifact lives on a DIFFERENT pool "
                    "(cfm_fullcorpus_2026_10_10, 66 rows); computing them here "
                    "would mix populations.  Uniqueness IS computed, on THIS "
                    "pool."
                ),
                "blocker": "population mismatch",
            },
            {
                "cell": "downstream_properties_on_connected_subset",
                "reason": (
                    "the property artifacts are on a different pool; this tool "
                    "does not recompute them.  The field's convention is to "
                    "report them on the valid AND connected subset (END, "
                    "arXiv:2506.10532 App. A.6.1)."
                ),
                "blocker": "population mismatch",
            },
            {
                "cell": "docking_and_redock_rms",
                "reason": (
                    "needs generated poses and a docking engine; the Vina panel "
                    "is BLOCKED on Pt+2"
                ),
                "blocker": "no poses / no docking engine",
            },
        ],
        "controls": controls,
        "notes": [
            "RAW means the SMILES as the sampler emitted it; POST-PROCESSED "
            "means the field's repair pipeline (largest-fragment selection -> "
            "explicit H -> heavy-atom floor), which here OMITS the UFF step "
            "for lack of coordinates.",
            "The post-processed CONNECTIVITY cell is 1.000000 by construction "
            "and is flagged with tautology=true; it must not be quoted as a "
            "result.",
            "The published 'fragmented_ratio' and the published end-to-end "
            "success rate are complements of DIFFERENT denominators (pool rows "
            "vs drawn samples) and differ in the 5th decimal; each cell above "
            "names its own.",
        ],
    }
    return payload


def _reference_control(path: Path) -> Dict[str, Any]:
    """The real corpus under the SAME scorer -- a control, not a cell.

    Without it, a generator number has nothing to be wrong against.
    """
    smiles: List[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if (row.get("split") or "").strip() != "train":
                continue
            text = (row.get("smiles") or "").strip()
            if text:
                smiles.append(text)
    scored = score_smiles_pool(smiles)
    rows = scored.get("rows") or []
    n_parsed = len(rows)
    n_single = sum(1 for r in rows if int(r.get("n_fragments", 0) or 0) == 1)
    lcf = [float(r.get("largest_fragment_frac", 0.0) or 0.0) for r in rows]
    nfrag = [int(r.get("n_fragments", 0) or 0) for r in rows]
    second_sizes: List[int] = []
    n_metal_only = 0
    for text in smiles:
        facts = _composition_facts_for_smiles(text)
        if facts is None:
            continue
        if facts["metal_only_component"]:
            n_metal_only += 1
        if facts["n_components"] >= 2:
            second_sizes.append(int(facts["second_largest_size"]))
    return {
        "source": _repo_relative(path),
        "population": "the train split of the real corpus",
        "n_rows": len(smiles),
        "n_parsed": n_parsed,
        "one_component": n_single,
        "one_component_rate_over_parsed": (n_single / n_parsed) if n_parsed else None,
        "largest_component_frac_mean": (sum(lcf) / n_parsed) if n_parsed else None,
        "n_fragments_mean": (sum(nfrag) / n_parsed) if n_parsed else None,
        "second_frag_is_single_atom": (
            (sum(1 for s in second_sizes if s == 1) / len(second_sizes))
            if second_sizes
            else None
        ),
        "metal_in_own_component": (n_metal_only / n_parsed) if n_parsed else None,
        "note": (
            "the corpus writes the metal as its own component, so a correct "
            "metal complex has SEVERAL components: one component is NOT a "
            "molecule here, and this control exists so the generator's "
            "component COUNT is read against real chemistry"
        ),
    }


def _composition_facts_for_smiles(smiles: str) -> Optional[Dict[str, Any]]:
    """The composition facts for one SMILES (used by the reference control)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _composition_facts(mol)


def _repo_relative(path: Path) -> str:
    """``path`` relative to the repo root when it is inside it, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except (ValueError, OSError):
        return str(path)


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------
def render_text(payload: Dict[str, Any]) -> str:
    """A human-readable rendering of the same payload (nothing is lost)."""
    lines: List[str] = []
    pop = payload.get("population") or {}
    dens = payload.get("denominators") or {}
    lines.append("=" * 78)
    lines.append("Wf-A133 FRAGMENTATION REPORTING MATRIX")
    lines.append("=" * 78)
    lines.append(f"pool      : {pop.get('pool_path')}")
    lines.append(f"sha256    : {pop.get('pool_sha256')}")
    lines.append(
        "epochs    : pool={} run_of_record={} killed={}".format(
            pop.get("pool_epochs"),
            pop.get("run_of_record_epochs"),
            pop.get("killed_pre_resume_epochs"),
        )
    )
    lines.append(
        "denominators: drawn={} pool_rows={} parsed={} valid={} connected={}".format(
            dens.get("n_drawn_samples"),
            dens.get("n_pool_rows"),
            dens.get("n_parsed"),
            dens.get("n_valid"),
            dens.get("n_connected"),
        )
    )
    lines.append(f"drawn src : {dens.get('n_drawn_samples_source')}")
    for name, block in (payload.get("cells") or {}).items():
        counts = block.get("counts") or {}
        lines.append("")
        lines.append("-" * 78)
        lines.append(
            f"SLICE {name}: drawn={counts.get('n_drawn_samples')} "
            f"pool_rows={counts.get('n_pool_rows')} parsed={counts.get('n_parsed')} "
            f"connected={counts.get('n_connected')}"
        )
        lines.append(f"  note: {block.get('population_note')}")
        for arm in ("raw", "post_processed"):
            cells = block.get(arm) or {}
            if not cells:
                continue
            label = "RAW" if arm == "raw" else "POST-PROCESSED"
            lines.append(f"  [{label}]")
            for key, cell in cells.items():
                value = cell.get("value")
                num = cell.get("num")
                den = cell.get("den")
                taut = " [TAUTOLOGY]" if cell.get("tautology") else ""
                den_label = cell.get("denominator")
                numden = f"{num}/{den}" if num is not None and den is not None else "-"
                lines.append(
                    f"    {key:42s} {value!r:>24}  {numden:>14}  ({den_label}){taut}"
                )
    checks = payload.get("cross_checks") or {}
    lines.append("")
    lines.append("-" * 78)
    lines.append("CROSS-CHECKS")
    for name, block in checks.items():
        if not isinstance(block, dict):
            continue
        if "n_checked" in block:
            lines.append(
                f"  {name:42s} checked={block.get('n_checked')} "
                f"mismatch={block.get('n_mismatch')}"
            )
            continue
        # The slice-derivation check has its own key names.
        for key, value in sorted(block.items()):
            if key == "source":
                continue
            lines.append(f"    {key:40s} {value}")
    controls = payload.get("controls") or {}
    if controls.get("reference_corpus"):
        ref = controls["reference_corpus"]
        lines.append("")
        lines.append("-" * 78)
        lines.append("CONTROL -- the real corpus under the same scorer")
        lines.append(
            "  n_parsed={} one_component={} rate={} LCF_mean={} n_frags_mean={}".format(
                ref.get("n_parsed"),
                ref.get("one_component"),
                ref.get("one_component_rate_over_parsed"),
                ref.get("largest_component_frac_mean"),
                ref.get("n_fragments_mean"),
            )
        )
    lines.append("")
    lines.append("-" * 78)
    lines.append("UNAVAILABLE CELLS (named, not omitted)")
    for item in payload.get("unavailable_cells") or []:
        lines.append(f"  {item.get('cell')}: {item.get('reason')}")
    band = payload.get("field_band") or {}
    lines.append("")
    lines.append("-" * 78)
    lines.append(
        "FIELD BAND [{}]: worst published connected = {} ({})".format(
            band.get("tag"),
            band.get("worst_published_connected"),
            ", ".join(band.get("citations") or []),
        )
    )
    lines.append(f"provenance: {band.get('provenance')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _forbid_registry_write(path: Path) -> Optional[str]:
    """Refuse a write under metrics/ or registry/.

    This tool publishes NOTHING into the project's SSOT; the main thread
    decides separately whether to register a metric.
    """
    try:
        resolved = Path(path).resolve()
        for forbidden in ("metrics", "registry"):
            root = (REPO / forbidden).resolve()
            if resolved == root or root in resolved.parents:
                return (
                    f"refusing to write under {forbidden}/ -- this tool prints "
                    "to stdout and writes only where the caller names a path; "
                    "metric registration is a separate decision"
                )
    except (OSError, ValueError):
        return None
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="the run directory holding generated_smiles.csv and events.jsonl",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="the pool CSV (default: <run-dir>/generated_smiles.csv)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=(
            "the real corpus used as a control (default: the metalcytotox "
            "corpus).  Pass --reference none to skip it."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "write the JSON here IN ADDITION to stdout.  Writes under "
            "metrics/ or registry/ are REFUSED."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="stdout format (default: json)",
    )
    args = parser.parse_args(argv)

    reference = args.reference
    if reference is not None and str(reference).strip().lower() in ("none", ""):
        reference = None

    payload = build_matrix(args.run_dir, pool_path=args.pool, reference_path=reference)

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        refusal = _forbid_registry_write(args.out)
        if refusal:
            print(f"[wfa133] {refusal}", file=sys.stderr)
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_file(args.out, text)
        print(f"[wfa133] wrote {args.out}", file=sys.stderr)

    if args.format == "text":
        print(render_text(payload))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
