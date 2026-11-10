#!/usr/bin/env python3
"""Wf-A142 R1 — FragFlex-style C-1 spanning-tree completion probe (CPU only).

Question
--------
Does a FragFlex-style valence-aware spanning-tree completion — add EXACTLY
``C-1`` edges to a ``C``-component molecule, chosen by shortest inter-component
atom distance among valence-feasible pairs, with an RDKit-sanitize-gated
rollback that never deletes an atom or an edge — raise our connectivity
WITHOUT the sanitize gate dropping molecules (i.e. without mass deletion)?

Frontier source
---------------
FragFlex — Sihyun Park & David Ryan Koes, "FragFlex: Fragment-Based Flexible
Molecular Generation for Exploring Vast Chemical Space", arXiv:2610.15284
(NeurIPS 2026 Workshop on AI for Drug Discovery).  Verified mechanics used
here: *"If the final graph contains C connected components before this repair,
exactly C-1 edges are added"*; RDKit strict-valence filter as the arbiter.

What this probe is NOT
----------------------
It is a SMILES-level post-hoc graph edit on the EXISTING generation pool
(``molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv``).  No
model is loaded, no decode is run, no GPU is touched.  The pool carries NO
coordinates, so the "shortest inter-component distance" that ranks the
candidate pairs comes from an ETKDG embedding of the multi-fragment SMILES —
a measurement of this probe's own geometry, not of the decoder's.  The
``gated_index`` variant (no geometry at all) and the ``gated_fragsep``
variant (closest approach between independently embedded fragment shapes)
bound how much that un-evaluable ordering decides; see the report's
"Honest negative 1" section (``/tmp/wfa142/r1-fragflex-cminus1.md``).

Repo conventions reused (NOT re-invented here)
----------------------------------------------
* ``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py``:
  ``_COVALENT_VALENCE`` (via :func:`covalent_valence_table`),
  ``_endpoint_valence_cost`` and ``_replay_remaining_budget``.
* ``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py``:
  ``_METAL_SET_PRESETS["all7"]`` (the head's metal set) and ``_DONOR_ZS``.
* ``molmetal/models/bond_head.py``: ``BOND_SINGLE`` / ``BOND_DOUBLE`` /
  ``BOND_TRIPLE`` / ``BOND_AROMATIC`` class ids.

Cohort and pool facts are MEASURED, never asserted
--------------------------------------------------
Every row count and epoch count this probe reports is COMPUTED from the
pool it was pointed at.  ``--cohort`` selects a COHORT and the cohort's
size depends on that pool, so a literal row count is true for one pool
and false for every other (Wf-A164).  The narrative is checked against
the artifact's own machine fields before it is written: an unbacked
number fails the run (see ``narrative_self_consistency``).

CLI
---
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \\
        molmetal/scripts/wfa142_fragflex_cminus1_probe.py \\
        --cohort epoch17 --out molmetal/reports/wfa142_fragflex_cminus1/probe.json

Exit codes: 0 = ran, 1 = usage/IO error OR a narrative self-consistency
FAIL (``status: FAIL`` is written to the artifact either way).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# CPU-only guard: this module must never touch a GPU.  The wave directive
# ("禁止使用gpu") OVERRIDES the project's standing GPU grant, so the guard is
# set before numpy/torch/rdkit are imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from molmetal.models.bond_head import (  # noqa: E402
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    _endpoint_valence_cost,
    _replay_remaining_budget,
    covalent_valence_table,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (  # noqa: E402
    _DONOR_ZS,
    _METAL_SET_PRESETS,
)

DEFAULT_POOL = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c7_fullgen_2026_10_20"
    / "generated_smiles.csv"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wfa142_fragflex_cminus1"
    / "probe.json"
)
DEFAULT_CORPUS = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)

#: The cohort ``epoch17`` selects: the pool's rows whose ``epoch`` column is
#: 17.  The NAME fixes the epoch and nothing else — the row count, the epoch
#: count and the per-epoch counts are MEASURED at run time (Wf-A164).
COHORT_EPOCH = 17

#: The head's metal set — the SINGLE source of truth for "is this atom a
#: metal" in the decode path (``yuelbond_decoder.py`` ``_METAL_SET_PRESETS``).
METAL_SET = frozenset(_METAL_SET_PRESETS["all7"])
DONOR_SET = frozenset(_DONOR_ZS)

#: ``_cap_for_z`` with the decoder's default ``unknown_z_default=None``
#: (an unknown Z is capped at 4, the carbon value).
_VALENCE_TABLE: Dict[int, int] = covalent_valence_table()


def cap_for_z(z: int) -> float:
    """The repo's own covalent-valence cap for ``z``.

    Mirrors ``CFMYuelBondDecoder._cap_for_z`` with the decoder's default
    ``unknown_z_default=None`` (unknown Z -> 4.0, the legacy carbon
    fallback).
    """
    return float(_VALENCE_TABLE.get(int(z), 4))


def is_metal(z: int) -> bool:
    """Is ``z`` a metal under the head's ``metal_set``?"""
    return int(z) in METAL_SET


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
_RDKIT_ORDER_TO_CLASS = {
    Chem.BondType.SINGLE: BOND_SINGLE,
    Chem.BondType.DOUBLE: BOND_DOUBLE,
    Chem.BondType.TRIPLE: BOND_TRIPLE,
    Chem.BondType.AROMATIC: BOND_AROMATIC,
    # A dative bond is stored as a BOND_SINGLE with a separate dative list in
    # the repo's decode path (``_select_bonds``); same accounting here.
    Chem.BondType.DATIVE: BOND_SINGLE,
}


def components(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    """Connected components as tuples of atom indices.

    ``Chem.GetMolFrags(mol, asMols=True)`` is the component READ used by the
    repo's own pool metric (``c7_fullgen_pool_quality.py``); the INDEX form
    is used here because the edit needs atom indices, and the two agree
    component-for-component (``GetMolFrags`` returns them in the same order).
    The index form is used because it is a pure graph traversal: it works on
    a molecule that does NOT sanitize, which the ungated counterfactual
    needs.
    """
    return [tuple(int(i) for i in frag) for frag in Chem.GetMolFrags(mol)]


def bond_classes(mol: Chem.Mol) -> List[Tuple[int, int, int]]:
    """``(i, j, BOND_* class)`` for every bond in ``mol``."""
    out: List[Tuple[int, int, int]] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        cls = _RDKIT_ORDER_TO_CLASS.get(bond.GetBondType())
        if cls is None:
            cls = BOND_SINGLE
        out.append((i, j, cls))
    return out


def remaining_budget(mol: Chem.Mol) -> Tuple[List[float], List[int], List[int]]:
    """Per-atom remaining valence budget, from the repo's own replay.

    Returns ``(remaining, arom_kept, z_list)``.
    """
    z_list = [int(a.GetAtomicNum()) for a in mol.GetAtoms()]
    caps = [cap_for_z(z) for z in z_list]
    remaining, arom_kept = _replay_remaining_budget(
        bond_classes(mol), z_list, caps, is_metal
    )
    return remaining, arom_kept, z_list


def metal_state(mol: Chem.Mol) -> Dict[str, float]:
    """How the METAL sits in the molecule — the corpus-convention probe.

    The corpus writes the metal as its OWN component by convention, so the
    two numbers that matter are (i) how many bonds the metal carries and
    (ii) whether the metal is a component on its own.
    """
    bonds = 0
    alone = False
    found = False
    frags = components(mol)
    for frag in frags:
        zs = [int(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in frag]
        if any(is_metal(z) for z in zs):
            found = True
            alone = len(frag) == 1
            for i in frag:
                if is_metal(int(mol.GetAtomWithIdx(i).GetAtomicNum())):
                    bonds += int(mol.GetAtomWithIdx(i).GetDegree())
    return {
        "n_metal_bonds": bonds if found else -1,
        "metal_alone": bool(alone) if found else False,
    }


def min_intercomponent_distance(
    mol: Chem.Mol, positions: Optional[np.ndarray]
) -> float:
    """Shortest distance between two DIFFERENT components (a geometry probe).

    On a multi-fragment SMILES this distance is a property of the EMBEDDING,
    not of the chemistry: RDKit places disconnected fragments wherever the
    embedding algorithm puts them.  The number is reported so a reader can
    see how arbitrary the "shortest inter-component distance" ordering is
    on a coordinate-free pool.
    """
    if positions is None:
        return float("nan")
    comps = components(mol)
    if len(comps) <= 1:
        return float("nan")
    comp_of = {}
    for ci, frag in enumerate(comps):
        for atom in frag:
            comp_of[atom] = ci
    best = float("inf")
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(i + 1, n):
            if comp_of[i] == comp_of[j]:
                continue
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < best:
                best = d
    return best


def _fragment_submol(mol: Chem.Mol, frag: Sequence[int]) -> Optional[Chem.Mol]:
    """A sanitized sub-molecule carrying one component's atoms AND bonds.

    The atom order is exactly ``frag``'s order, so the embedded coordinates
    map straight back onto the parent.
    """
    rw = Chem.RWMol()
    index_map: Dict[int, int] = {}
    for k, i in enumerate(frag):
        atom = mol.GetAtomWithIdx(int(i))
        rw.AddAtom(Chem.Atom(int(atom.GetAtomicNum())))
        index_map[int(i)] = k
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        if i in index_map and j in index_map:
            rw.AddBond(index_map[i], index_map[j], bond.GetBondType())
    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        return None
    return sub


def embed_positions(
    mol: Chem.Mol, seed: int = 0, geometry: str = "whole"
) -> Tuple[Optional[np.ndarray], str]:
    """Coordinates for the molecule.

    ``geometry="whole"``
        ETKDG on the WHOLE (multi-fragment) SMILES — the method the brief
        specifies.  RDKit places disconnected fragments wherever the
        embedding algorithm puts them, so the "shortest inter-component
        distance" this yields is a property of the EMBEDDING, not of the
        chemistry.

    ``geometry="fragsep"``
        ETKDG on each component SEPARATELY, then each component's centroid is
        moved to the origin.  The inter-component distance is then the
        CLOSEST APPROACH between two fragment shapes — the quantity FragFlex's
        assembler actually ranks on — instead of an arbitrary relative
        placement.

    Returns ``(positions, status)`` where ``status`` is one of ``"3d"``,
    ``"2d"`` (a 3-D embed failed and a 2-D fallback was used) or ``"none"``.

    MEASURED determinism note: RDKit's ``randomSeed=0`` IS deterministic
    (three consecutive embeddings of the same SMILES gave bit-identical
    coordinates); the seed is nonetheless exposed so a caller can move off 0.
    """
    if geometry == "whole":
        work = Chem.Mol(mol)
        rc = AllChem.EmbedMolecule(work, useRandomCoords=True, randomSeed=seed)
        if rc != 0:
            try:
                AllChem.Compute2DCoords(work)
            except Exception:
                return None, "none"
            if work.GetNumConformers() == 0:
                return None, "none"
            conf = work.GetConformer()
            pos = np.array(
                [
                    [
                        conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                    ]
                    for i in range(work.GetNumAtoms())
                ],
                dtype=float,
            )
            return pos, "2d"
        conf = work.GetConformer()
        pos = np.array(conf.GetPositions(), dtype=float)
        return pos, "3d"

    if geometry == "index":
        # NULL CONTROL: no geometric information at all.  Every pair is at
        # distance 0, so the candidate order degenerates to the atom-index
        # order.  Comparing this variant against the two geometries above
        # measures how much of the outcome the (unmeasurable) distance
        # ordering actually decides.
        return None, "index"

    if geometry != "fragsep":
        raise ValueError(f"unknown geometry {geometry!r}")

    frags = components(mol)
    n = mol.GetNumAtoms()
    pos = np.zeros((n, 3), dtype=float)
    worst = "3d"
    for frag in frags:
        sub = _fragment_submol(mol, frag)
        if sub is None:
            return None, "none"
        work = Chem.Mol(sub)
        rc = AllChem.EmbedMolecule(work, useRandomCoords=True, randomSeed=seed)
        if rc != 0:
            try:
                AllChem.Compute2DCoords(work)
            except Exception:
                return None, "none"
            if work.GetNumConformers() == 0:
                return None, "none"
            conf = work.GetConformer()
            sub_pos = np.array(conf.GetPositions(), dtype=float)
            if worst == "3d":
                worst = "2d"
        else:
            conf = work.GetConformer()
            sub_pos = np.array(conf.GetPositions(), dtype=float)
        # Centre the fragment on the origin: two fragments are then compared
        # by their closest approach, not by an arbitrary relative placement.
        sub_pos = sub_pos - sub_pos.mean(axis=0)
        for k, i in enumerate(frag):
            pos[int(i)] = sub_pos[k]
    return pos, worst


# ---------------------------------------------------------------------------
# The FragFlex-style repair
# ---------------------------------------------------------------------------
def _try_add_single(mol: Chem.Mol, i: int, j: int) -> Tuple[Optional[Chem.Mol], str]:
    """Add ONE single bond and re-sanitize.  Never deletes anything.

    Returns ``(sanitized_mol_or_None, failure_reason)``.  On failure the
    caller keeps the previous molecule untouched — no atom and no edge is
    ever removed (the FragFlex / Yao et al. "no edge is deleted" rule).
    """
    rw = Chem.RWMol(mol)
    try:
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    except Exception as exc:  # pragma: no cover - RDKit raises RuntimeError
        return None, f"AddBond:{type(exc).__name__}"
    probe = rw.GetMol()
    try:
        Chem.SanitizeMol(probe)
    except Exception as exc:
        return None, type(exc).__name__
    return probe, ""


def _add_no_sanitize(mol: Chem.Mol, i: int, j: int) -> Tuple[Optional[Chem.Mol], str]:
    """Add ONE single bond WITHOUT sanitizing — the blind counterfactual.

    Used only by the ``ungated`` variant, whose purpose is to measure how
    many molecules a NO-GATE FragFlex-style assembler would reject wholesale.
    """
    rw = Chem.RWMol(mol)
    try:
        rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    except Exception as exc:
        return None, f"AddBond:{type(exc).__name__}"
    return rw.GetMol(), ""


def fragflex_complete(
    mol: Chem.Mol,
    positions: Optional[np.ndarray],
    *,
    gate: bool = True,
    allow_metal_endpoints: bool = True,
    max_edges: Optional[int] = None,
) -> Dict[str, object]:
    """Add EXACTLY ``C-1`` edges to a ``C``-component molecule.

    Parameters
    ----------
    gate
        ``True`` (the FragFlex rule): a candidate pair must be
        valence-feasible under the repo's own accounting AND survive
        ``Chem.SanitizeMol`` after the add.  ``False`` (counterfactual): the
        closest pair is added blind and the molecule is sanitized once at the
        end, so the caller can measure how much wholesale rejection the gate
        is preventing.
    allow_metal_endpoints
        ``False`` keeps metal atoms out of the bridge endpoints entirely, so
        the metal's component stays a separate component — the corpus's own
        convention (see the report's framing section).

    The returned dict is the per-molecule trace.  Keys:

    ``n_components_before`` / ``n_components_after``
        component counts before and after the repair.
    ``edges_added``
        how many edges were actually added (``<= C-1``).
    ``stranded``
        ``True`` when the repair could not reach ``C-1`` edges (no candidate
        pair survived), i.e. the molecule is left with more than one
        component.
    ``edges_metal_nonmetal`` / ``edges_nonmetal_nonmetal``
        the split of added edges by whether a METAL atom is an endpoint.
    ``sanitize_ok``
        whether the FINAL molecule survives ``Chem.SanitizeMol``.
    """
    comps = components(mol)
    n_before = len(comps)
    shape0 = molecule_shape(mol)
    metal0 = metal_state(mol)
    result: Dict[str, object] = {
        "n_components_before": n_before,
        "n_components_after": n_before,
        "n_singletons_before": shape0["n_singletons"],
        "n_singletons_after": shape0["n_singletons"],
        "n_heavy_before": shape0["n_heavy"],
        "n_heavy_after": shape0["n_heavy"],
        "largest_frag_heavy_before": shape0["largest_frag_heavy"],
        "largest_frag_heavy_after": shape0["largest_frag_heavy"],
        "n_metal_bonds_before": metal0["n_metal_bonds"],
        "n_metal_bonds_after": metal0["n_metal_bonds"],
        "metal_alone_before": metal0["metal_alone"],
        "metal_alone_after": metal0["metal_alone"],
        "edges_added": 0,
        "edges_metal_nonmetal": 0,
        "edges_nonmetal_nonmetal": 0,
        "stranded": False,
        "stranded_components": 0,
        "sanitize_ok": True,
        "sanitize_fail_reason": "",
        "rejected_by_valence": 0,
        "rejected_by_sanitize": 0,
        "bonds_rewritten_by_sanitizer": 0,
        "n_atoms_before": int(mol.GetNumAtoms()),
        "n_atoms_after": int(mol.GetNumAtoms()),
        "n_bonds_before": int(mol.GetNumBonds()),
        "n_bonds_after": int(mol.GetNumBonds()),
        # The final molecule's bond list, indexed by ATOM INDEX (the edit
        # never renumbers atoms, so these indices are the parent's).  It is
        # popped by the runner so it never reaches the JSON artifact; the
        # unit test uses it to prove that no original edge was deleted.
        "bonds_after": [
            [int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()), str(b.GetBondType())]
            for b in mol.GetBonds()
        ],
        "smiles_after": "",
    }
    if n_before <= 1:
        return result

    target = n_before - 1
    if max_edges is not None:
        target = min(target, int(max_edges))

    z_list = [int(a.GetAtomicNum()) for a in mol.GetAtoms()]
    n_atoms = mol.GetNumAtoms()

    # The candidate ORDER is a property of the geometry and the atom indices
    # only, so it is computed ONCE per molecule and then filtered at each
    # step to the pairs that are still inter-component.  (The relative order
    # of the inter-component subset is the same as re-sorting it per step.)
    ordered_pairs: List[Tuple[float, int, int]] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if not allow_metal_endpoints and (
                is_metal(z_list[i]) or is_metal(z_list[j])
            ):
                continue
            if positions is None:
                dist = 0.0
            else:
                dist = float(np.linalg.norm(positions[i] - positions[j]))
            ordered_pairs.append((dist, i, j))
    ordered_pairs.sort()

    current = mol
    added = 0
    n_metal_edges = 0
    n_ligand_edges = 0
    rejected_valence = 0
    rejected_sanitize = 0
    final_reason = ""

    while added < target:
        comps = components(current)
        if len(comps) <= 1:
            break
        comp_of = {}
        for ci, frag in enumerate(comps):
            for atom in frag:
                comp_of[atom] = ci

        # The valence budget only changes when a bond is ACCEPTED, so it is
        # computed once per step, not once per candidate.
        remaining = None
        if gate:
            remaining, _arom, z_now = remaining_budget(current)

        placed = False
        for _dist, i, j in ordered_pairs:
            if comp_of[i] == comp_of[j]:
                continue
            if gate:
                need_i = _endpoint_valence_cost(
                    BOND_SINGLE, z_now[i], z_now[j], is_metal
                )
                need_j = _endpoint_valence_cost(
                    BOND_SINGLE, z_now[j], z_now[i], is_metal
                )
                if remaining[i] < need_i or remaining[j] < need_j:
                    rejected_valence += 1
                    continue
                trial, reason = _try_add_single(current, i, j)
                if trial is None:
                    rejected_sanitize += 1
                    final_reason = reason
                    continue
            else:
                # Blind add: no valence check, no per-step sanitize gate.
                trial, reason = _add_no_sanitize(current, i, j)
                if trial is None:
                    rejected_sanitize += 1
                    final_reason = reason
                    continue
            current = trial
            added += 1
            if is_metal(z_list[i]) or is_metal(z_list[j]):
                n_metal_edges += 1
            else:
                n_ligand_edges += 1
            placed = True
            break

        if not placed:
            break

    # Final sanitize of the whole molecule (the ungated counterfactual needs
    # this to measure wholesale rejection).
    final_ok = True
    final_reason = ""
    try:
        Chem.SanitizeMol(current)
    except Exception as exc:
        final_ok = False
        final_reason = type(exc).__name__

    comps_after = components(current)
    shape_after = molecule_shape(current)
    metal1 = metal_state(current)
    # How many ORIGINAL edges kept their bond TYPE?  RDKit's
    # ``cleanupOrganometallics`` (which runs inside ``SanitizeMol``) rewrites
    # a single bond between a metal and a hypervalent atom into a DATIVE
    # bond — the repo's documented behaviour
    # (``cfm_decoder_yuelbond.py`` ``_endpoint_valence_cost`` rule 1).  That
    # is a REWRITE, not a deletion: the edge survives.
    orig_types: Dict[Tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        orig_types[(i, j)] = str(bond.GetBondType())
        orig_types[(j, i)] = str(bond.GetBondType())
    n_rewritten = 0
    for bond in current.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        if (i, j) in orig_types and orig_types[(i, j)] != str(bond.GetBondType()):
            n_rewritten += 1
    try:
        smiles_after = Chem.MolToSmiles(current)
    except Exception as exc:
        # RDKit can refuse to write a SMILES for a highly symmetric fused
        # system ("Maximum BFS search size exceeded").  The molecule itself is
        # fine; only the string is unavailable.
        smiles_after = f"<MolToSmiles failed: {type(exc).__name__}>"
    result.update(
        {
            "n_components_after": len(comps_after),
            "edges_added": added,
            "edges_metal_nonmetal": n_metal_edges,
            "edges_nonmetal_nonmetal": n_ligand_edges,
            "stranded": len(comps_after) > 1,
            "stranded_components": max(0, len(comps_after) - 1),
            "sanitize_ok": final_ok,
            "sanitize_fail_reason": final_reason,
            "rejected_by_valence": rejected_valence,
            "rejected_by_sanitize": rejected_sanitize,
            "n_atoms_after": int(current.GetNumAtoms()),
            "n_bonds_after": int(current.GetNumBonds()),
            "n_singletons_after": shape_after["n_singletons"],
            "n_heavy_after": shape_after["n_heavy"],
            "largest_frag_heavy_after": shape_after["largest_frag_heavy"],
            "n_metal_bonds_after": metal1["n_metal_bonds"],
            "metal_alone_after": metal1["metal_alone"],
            "bonds_rewritten_by_sanitizer": n_rewritten,
            "smiles_after": smiles_after,
            "bonds_after": [
                [
                    int(b.GetBeginAtomIdx()),
                    int(b.GetEndAtomIdx()),
                    str(b.GetBondType()),
                ]
                for b in current.GetBonds()
            ],
        }
    )
    return result


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def molecule_shape(mol: Chem.Mol) -> Dict[str, float]:
    """The per-molecule shape measures (components, singletons, heavy atoms).

    Computed from the index form of ``Chem.GetMolFrags`` so it also works on
    a molecule that failed to sanitize (the ungated counterfactual).  The
    partition is identical to the ``asMols=True`` form.
    """
    frags = components(mol)
    sizes = sorted((len(f) for f in frags), reverse=True)
    heavy_sizes = sorted(
        (
            sum(
                1
                for i in frag
                if int(mol.GetAtomWithIdx(int(i)).GetAtomicNum()) != 1
            )
            for frag in frags
        ),
        reverse=True,
    )
    n_singletons = sum(1 for f in frags if len(f) == 1)
    return {
        "n_components": len(frags),
        "n_singletons": n_singletons,
        "n_heavy": int(mol.GetNumHeavyAtoms()),
        "largest_frag_heavy": int(heavy_sizes[0]) if heavy_sizes else 0,
        "second_frag_size": int(sizes[1]) if len(sizes) > 1 else 0,
    }


def _aggregate(records: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate the per-molecule records into the reported numbers."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    out: Dict[str, object] = {"n": n}
    for key in (
        "n_components_before",
        "n_components_after",
        "n_singletons_before",
        "n_singletons_after",
        "n_heavy_before",
        "n_heavy_after",
        "edges_added",
        "edges_metal_nonmetal",
        "edges_nonmetal_nonmetal",
        "rejected_by_valence",
        "rejected_by_sanitize",
        "bonds_rewritten_by_sanitizer",
    ):
        vals = [float(r[key]) for r in records]
        out[f"mean_{key}"] = round(sum(vals) / n, 6)
        out[f"sum_{key}"] = round(sum(vals), 6)
    out["n_connected_before"] = sum(
        1 for r in records if r["n_components_before"] == 1
    )
    out["n_connected_after"] = sum(
        1 for r in records if r["n_components_after"] == 1
    )
    out["n_stranded"] = sum(1 for r in records if r["stranded"])
    out["n_sanitize_fail"] = sum(1 for r in records if not r["sanitize_ok"])
    out["n_embed_3d"] = sum(1 for r in records if r["embed_status"] == "3d")
    out["n_embed_2d"] = sum(1 for r in records if r["embed_status"] == "2d")
    out["n_embed_none"] = sum(1 for r in records if r["embed_status"] == "none")
    out["n_parse_fail_before"] = sum(1 for r in records if r["parse_fail"])
    out["n_metal_atoms"] = sum(int(r["n_metal_atoms"]) for r in records)
    metal_bonds = [float(r["n_metal_bonds_before"]) for r in records]
    out["mean_metal_bonds_before"] = round(
        sum(metal_bonds) / n, 6
    )
    metal_bonds_after = [float(r["n_metal_bonds_after"]) for r in records]
    out["mean_metal_bonds_after"] = round(
        sum(metal_bonds_after) / n, 6
    )
    out["n_metal_alone_before"] = sum(
        1 for r in records if r.get("metal_alone_before")
    )
    out["n_metal_alone_after"] = sum(
        1 for r in records if r.get("metal_alone_after")
    )
    dists = [
        float(r["min_intercomponent_distance"])
        for r in records
        if r.get("min_intercomponent_distance") == r.get("min_intercomponent_distance")
    ]
    if dists:
        srt = sorted(dists)
        out["min_intercomponent_distance_mean"] = round(
            sum(srt) / len(srt), 4
        )
        out["min_intercomponent_distance_median"] = round(
            srt[len(srt) // 2], 4
        )
        out["min_intercomponent_distance_p10"] = round(
            srt[int(0.10 * (len(srt) - 1))], 4
        )
        out["min_intercomponent_distance_p90"] = round(
            srt[int(0.90 * (len(srt) - 1))], 4
        )
    # Largest-fragment selection (the pool's own post-processing) discards
    # mass: retention = sum(largest frag heavy) / sum(total heavy).
    tot_heavy_before = sum(float(r["n_heavy_before"]) for r in records)
    tot_heavy_after = sum(float(r["n_heavy_after"]) for r in records)
    kept_before = sum(float(r["largest_frag_heavy_before"]) for r in records)
    kept_after = sum(float(r["largest_frag_heavy_after"]) for r in records)
    out["heavy_total_before"] = tot_heavy_before
    out["heavy_total_after"] = tot_heavy_after
    out["largest_frag_kept_before"] = kept_before
    out["largest_frag_kept_after"] = kept_after
    out["largest_frag_retention_before"] = (
        round(kept_before / tot_heavy_before, 6) if tot_heavy_before else None
    )
    out["largest_frag_retention_after"] = (
        round(kept_after / tot_heavy_after, 6) if tot_heavy_after else None
    )
    out["largest_frag_mass_discarded_before_pct"] = (
        round(100.0 * (1.0 - kept_before / tot_heavy_before), 4)
        if tot_heavy_before
        else None
    )
    out["largest_frag_mass_discarded_after_pct"] = (
        round(100.0 * (1.0 - kept_after / tot_heavy_after), 4)
        if tot_heavy_after
        else None
    )
    out["heavy_retention_no_selection"] = (
        round(tot_heavy_after / tot_heavy_before, 6) if tot_heavy_before else None
    )
    return out


def read_pool(path: Path) -> List[Dict[str, str]]:
    """Read EVERY row of the generation pool (no cohort selection)."""
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def _epoch_sort_key(key: str) -> Tuple[int, float, str]:
    """Sort epoch labels numerically when they are numeric, else lexically.

    The default pool's epochs are integers; a caller-supplied pool may not
    be, and the old ``key=lambda e: int(e)`` raised on such a pool.
    """
    try:
        return (0, float(str(key)), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(key))


def epoch_histogram(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    """Row count per epoch label, in epoch order."""
    hist: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("epoch", ""))
        hist[key] = hist.get(key, 0) + 1
    return {k: hist[k] for k in sorted(hist, key=_epoch_sort_key)}


def select_cohort(
    rows: Sequence[Dict[str, str]],
    cohort: str,
    *,
    per_epoch: int = 0,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """Select a cohort from the pool and MEASURE the facts about it.

    Wf-A164 (2026-11-01): the cohort's row count and the pool's epoch count
    used to be string literals ("704 rows", "18 epochs") that were TRUE for
    the default pool and FALSE for every other pool — and every other
    ``--cohort``.  They are COMPUTED here, from the rows this function
    actually read.

    Cohorts
    -------
    ``epoch17``        the pool's rows whose ``epoch`` column is 17.
    ``all``            every row of every epoch in the pool.
    ``all_stratified`` a deterministic SYSTEMATIC sample of at most
                       ``per_epoch`` rows from EVERY epoch.  The pool file is
                       epoch-ordered, so the sample is a fixed stride inside
                       each epoch — no RNG, reproducible by construction, and
                       not a prefix (which would be biased towards whatever the
                       generator emitted first).
    """
    pool_rows = list(rows)
    hist = epoch_histogram(pool_rows)
    epoch_label = str(COHORT_EPOCH)
    n_epoch_rows = hist.get(epoch_label, 0)

    selected: List[Dict[str, str]] = []
    if cohort == "epoch17":
        selected = [
            r for r in pool_rows if str(r.get("epoch")) == epoch_label
        ]
    elif cohort == "all":
        selected = pool_rows
    elif cohort == "all_stratified":
        if per_epoch <= 0:
            raise ValueError("all_stratified needs --per-epoch-limit > 0")
        by_epoch: Dict[str, List[Dict[str, str]]] = {}
        for row in pool_rows:
            by_epoch.setdefault(str(row.get("epoch", "")), []).append(row)
        keep: List[Dict[str, str]] = []
        for epoch in sorted(by_epoch, key=_epoch_sort_key):
            bucket = by_epoch[epoch]
            stride = max(1, -(-len(bucket) // per_epoch))  # ceil
            keep.extend(bucket[::stride][:per_epoch])
        selected = keep
    else:
        raise ValueError(f"unknown cohort {cohort!r}")

    sel_hist = epoch_histogram(selected)
    facts: Dict[str, object] = {
        "cohort": cohort,
        "cohort_epoch": COHORT_EPOCH if cohort == "epoch17" else None,
        "n_rows_in_pool": len(pool_rows),
        "n_epochs_in_pool": len(hist),
        "rows_per_epoch_in_pool": hist,
        "n_rows_in_cohort_epoch": n_epoch_rows,
        "n_rows_in_cohort_selected": len(selected),
        "n_epochs_in_cohort": len(sel_hist),
        "rows_per_epoch_in_cohort": sel_hist,
        "per_epoch_limit": int(per_epoch),
        "note": (
            "every count here is MEASURED from the pool that was read; none "
            "of them is a constant of this script"
        ),
    }
    return selected, facts


def load_pool(
    path: Path, cohort: str, *, per_epoch: int = 0
) -> List[Dict[str, str]]:
    """Read the generation pool and select a cohort (rows only).

    Kept as the public entry point the pre-existing test suite uses.
    Wf-A164: the cohort's row count and the pool's epoch count are
    MEASURED by :func:`select_cohort`; this wrapper returns the selected
    rows and asserts no count of its own.
    """
    rows, _facts = select_cohort(
        read_pool(path), cohort, per_epoch=per_epoch
    )
    return rows


def derive_cohort_detail(facts: Dict[str, object]) -> str:
    """The cohort description — DERIVED from the measured cohort facts.

    Wf-A164 (2026-11-01): this used to be one literal per cohort ("epoch 17
    rows of the pool (704 rows)" / "every row of every epoch (12,669
    rows)" / "all 18 epochs"), each of them true only for the default pool
    and every default cohort.  It is now a pure function of the facts, so it
    cannot disagree with the rows the tool actually processed.
    """
    cohort = str(facts["cohort"])
    n = int(facts["n_rows_processed"])
    sel = int(facts["n_rows_in_cohort_selected"])
    # Thousands separators, so the rendering matches the one the published
    # default-pool artifact already carries ("12,669 rows").
    extra = "" if n == sel else f" of {sel:,} selected"
    if cohort == "epoch17":
        if int(facts["n_rows_in_cohort_epoch"]) == 0:
            return (
                f"epoch {facts['cohort_epoch']} rows of the pool ({n:,} rows; "
                "the pool carries no rows for that epoch)"
            )
        return (
            f"epoch {facts['cohort_epoch']} rows of the pool ({n:,} rows{extra})"
        )
    if cohort == "all":
        return f"every row of every epoch ({n:,} rows{extra})"
    if cohort == "all_stratified":
        return (
            "systematic sample: first-fit stride inside each epoch, at most "
            f"{facts['per_epoch_limit']} rows per epoch, "
            f"all {facts['n_epochs_in_cohort']} epochs ({n:,} rows{extra})"
        )
    raise ValueError(f"unknown cohort {cohort!r}")


# ---------------------------------------------------------------------------
# The narrative self-consistency guard (Wf-A164)
# ---------------------------------------------------------------------------
#: Every number rendered in a narrative string must be backed by a machine
#: field of the SAME artifact.  A token with no backing is a defect: it was
#: written by a human or by a tool constant, not derived from the run the
#: artifact describes — so it is true for one population and FALSE for
#: another.  This is the Wf-A155 defect class, closed at the producer.
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

#: The artifact's PROSE paths.  Every leaf under these is prose; every other
#: leaf is a MACHINE field.  Note the scope: these are the paths that can
#: carry a CLAIM ABOUT THIS POPULATION.  The method strings (``question``,
#: ``variants``) are population-invariant by construction — their only digit
#: is the FragFlex rule's "C-1", which is stated as a machine field too
#: (``method.edges_added_rule``).  Paths are NOT restated in prose: a path's
#: digits are identifiers, not claims, so a path belongs in a machine field.
NARRATIVE_PATHS: Tuple[str, ...] = (
    "cohort_detail",
    "population.statement",
    "caveats",
    "question",
    "variants",
)


def _leaf_map(artifact: Any) -> Dict[str, Any]:
    """Flatten an artifact to ``dotted.path -> leaf``."""
    out: Dict[str, Any] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                walk(val, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for idx, val in enumerate(node):
                walk(val, f"{path}[{idx}]")
        else:
            out[path] = node

    walk(artifact, "")
    return out


def _resolve(artifact: Any, path: str) -> List[str]:
    """The leaf paths under a dotted narrative path.

    A path may designate a LEAF (``cohort_detail``), a MAPPING
    (``variants`` -> ``variants.gated_whole``) or a LIST (``caveats`` ->
    ``caveats[0]``, ``caveats[1]``, ...).  The list form matters: with a
    naive ``path + "."`` prefix the list elements would never match, they
    would be silently classified as MACHINE fields, and every number inside
    them would go unchecked.
    """
    leaves = _leaf_map(artifact)
    if path in leaves:
        return [path]
    prefixes = (path + ".", path + "[")
    return sorted(p for p in leaves if p.startswith(prefixes))


def _standalone_in_any(token: str, strings: List[str]) -> bool:
    """True when ``token`` appears in one of ``strings`` as a STANDALONE
    number — delimited by characters that are neither digits nor a decimal
    point.

    Deliberately stricter than a substring test: the token ``1`` must NOT
    count as "backed" by the ``1`` inside ``12669``.
    """
    pattern = re.compile(r"(?<![0-9.])" + re.escape(token) + r"(?![0-9.])")
    return any(pattern.search(text) for text in strings)


def _narrative_self_consistency(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """THE GUARD for the defect class, not just this instance.

    Every number rendered in a NARRATIVE string must be backed by a MACHINE
    field of the same artifact, either as a standalone number inside a
    machine string, or equal (to the token's own decimal count) to a machine
    number.  A token with no backing is a DEFECT and fails the run.
    """
    leaves = _leaf_map(artifact)
    prose_paths: set = set()
    for path in NARRATIVE_PATHS:
        prose_paths.update(_resolve(artifact, path))
    machine = {p: v for p, v in leaves.items() if p not in prose_paths}

    machine_numbers: List[float] = []
    machine_strings: List[str] = []
    for value in machine.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            machine_numbers.append(float(value))
        elif isinstance(value, str):
            machine_strings.append(value)

    unbacked: Dict[str, List[str]] = {}
    n_prose_numbers = 0
    for path in sorted(prose_paths):
        text = leaves.get(path)
        if not isinstance(text, str):
            continue
        for token in _NUM_RE.findall(text):
            n_prose_numbers += 1
            norm = token.replace(",", "")
            try:
                value = float(norm)
            except ValueError:  # pragma: no cover - defensive
                continue
            decimals = len(norm.split(".")[1]) if "." in norm else 0
            tol = 0.5 * (10.0 ** (-decimals)) if decimals else 0.0
            backed = _standalone_in_any(token, machine_strings)
            if not backed:
                backed = any(
                    abs(mv - value) <= tol + 1e-12 for mv in machine_numbers
                )
            if not backed:
                unbacked.setdefault(token, []).append(path)
    return {
        "rule": (
            "every number rendered in a narrative string must be backed by a "
            "machine field of the same artifact -- as a standalone number "
            "inside a machine string, or equal (to the token's own decimal "
            "count) to a machine number"
        ),
        "scope": (
            "the population-describing prose: the cohort description, the "
            "population statement, the caveats, and the method strings (whose "
            "only digit is the FragFlex rule's C-1, itself stated in the "
            "machine field method.edges_added_rule). Paths are machine "
            "fields, not prose: a path's digits are identifiers, not claims."
        ),
        "narrative_paths": list(NARRATIVE_PATHS),
        "n_prose_leaves": len(prose_paths),
        "n_prose_numbers": n_prose_numbers,
        "n_machine_leaves": len(machine),
        "n_machine_numbers": len(machine_numbers),
        "n_machine_strings": len(machine_strings),
        "unbacked_numbers": {
            tok: sorted(set(paths)) for tok, paths in sorted(unbacked.items())
        },
        "pass": not unbacked,
    }


def narrative_text(artifact: Dict[str, Any]) -> Dict[str, str]:
    """The artifact's PROSE leaves, by path (used by the regression test)."""
    leaves = _leaf_map(artifact)
    prose_paths: set = set()
    for path in NARRATIVE_PATHS:
        prose_paths.update(_resolve(artifact, path))
    return {
        p: leaves[p]
        for p in sorted(prose_paths)
        if isinstance(leaves.get(p), str)
    }


def run_variant(
    rows: Sequence[Dict[str, str]],
    *,
    gate: bool,
    allow_metal_endpoints: bool,
    seed: int,
    label: str,
    geometry: str = "whole",
) -> List[Dict[str, object]]:
    """Run one variant over ``rows`` and return the per-molecule records."""
    records: List[Dict[str, object]] = []
    for row in rows:
        smiles = str(row.get("smiles", ""))
        mol = Chem.MolFromSmiles(smiles)
        rec: Dict[str, object] = {
            "epoch": str(row.get("epoch", "")),
            "smiles": smiles,
            "variant": label,
            "geometry": geometry,
        }
        if mol is None:
            rec.update(
                {
                    "parse_fail": True,
                    "embed_status": "none",
                    "n_components_before": 0,
                    "n_components_after": 0,
                    "n_singletons_before": 0,
                    "n_singletons_after": 0,
                    "n_heavy_before": 0,
                    "n_heavy_after": 0,
                    "largest_frag_heavy_before": 0,
                    "largest_frag_heavy_after": 0,
                    "edges_added": 0,
                    "edges_metal_nonmetal": 0,
                    "edges_nonmetal_nonmetal": 0,
                    "rejected_by_valence": 0,
                    "rejected_by_sanitize": 0,
                    "bonds_rewritten_by_sanitizer": 0,
                    "stranded": False,
                    "sanitize_ok": False,
                    "sanitize_fail_reason": "parse_fail",
                    "n_metal_atoms": 0,
                    "n_metal_bonds_before": -1,
                    "n_metal_bonds_after": -1,
                    "metal_alone_before": False,
                    "metal_alone_after": False,
                    "min_intercomponent_distance": float("nan"),
                }
            )
            records.append(rec)
            continue
        rec["parse_fail"] = False
        shape_before = molecule_shape(mol)
        rec["n_components_before"] = shape_before["n_components"]
        rec["n_singletons_before"] = shape_before["n_singletons"]
        rec["n_heavy_before"] = shape_before["n_heavy"]
        rec["largest_frag_heavy_before"] = shape_before["largest_frag_heavy"]
        rec["n_metal_atoms"] = sum(
            1 for a in mol.GetAtoms() if is_metal(a.GetAtomicNum())
        )
        positions, status = embed_positions(mol, seed=seed, geometry=geometry)
        rec["embed_status"] = status
        rec["min_intercomponent_distance"] = min_intercomponent_distance(
            mol, positions
        )
        outcome = fragflex_complete(
            mol,
            positions,
            gate=gate,
            allow_metal_endpoints=allow_metal_endpoints,
        )
        outcome.pop("bonds_after", None)
        rec.update(outcome)
        records.append(rec)
    return records


def corpus_reference(path: Path) -> Dict[str, object]:
    """The corpus baseline, measured with the SAME definitions as the pool.

    The published metric artifact
    (``metrics/by_metric/c7_fullgen_pool_quality.json``) reports ITS
    reference over the TRAIN split of ITS corpus.  The numbers below are
    recomputed from the corpus named by ``--corpus``; they are comparable to
    that artifact only when ``--corpus`` IS that corpus — for any other
    corpus they are that corpus's own numbers and the comparison is not
    claimed.

    The function also records how many of the corpus's components contain a
    METAL atom — which is what decides whether "mean components" is
    comparable between the pool and the corpus at all.  Note that the
    corpus's ``smiles`` column is LIGAND-ONLY (the metal is carried in a
    separate column), so the metal-component count is a property of that
    column; ``n_rows_with_metal_column`` is reported alongside it so the
    zero is not misread.
    """
    if not path.exists():
        return {"source": str(path), "present": False}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: Dict[str, object] = {
        "source": str(path),
        "present": True,
        "n_rows": len(rows),
        "n_rows_with_metal_column": sum(
            1
            for r in rows
            if str(r.get("metal") or r.get("Metal") or "").strip()
        ),
    }
    for split_name, subset in (
        ("all", rows),
        ("train", [r for r in rows if r.get("split") == "train"]),
    ):
        n = 0
        comps: List[int] = []
        singletons: List[int] = []
        metal_comps = 0
        tot_heavy = 0
        kept_heavy = 0
        for row in subset:
            mol = Chem.MolFromSmiles(str(row.get("smiles", "")))
            if mol is None:
                continue
            n += 1
            frags = Chem.GetMolFrags(mol, asMols=True)
            comps.append(len(frags))
            singletons.append(sum(1 for f in frags if f.GetNumAtoms() == 1))
            for frag in frags:
                if any(is_metal(a.GetAtomicNum()) for a in frag.GetAtoms()):
                    metal_comps += 1
            sizes = [f.GetNumHeavyAtoms() for f in frags]
            tot_heavy += mol.GetNumHeavyAtoms()
            kept_heavy += max(sizes) if sizes else 0
        out[split_name] = {
            "n_parse": n,
            "mean_n_frags": round(sum(comps) / n, 4) if n else None,
            "mean_singleton_components": round(sum(singletons) / n, 4) if n else None,
            "mean_components_containing_a_metal": round(metal_comps / n, 6) if n else None,
            "n_components_containing_a_metal": metal_comps,
            "largest_frag_retention": round(kept_heavy / tot_heavy, 6) if tot_heavy else None,
            "frac_n_frags_3": round(
                sum(1 for c in comps if c == 3) / n, 5
            ) if n else None,
            "frac_n_frags_1": round(
                sum(1 for c in comps if c == 1) / n, 5
            ) if n else None,
        }
    return out


def _sha256(path: Path) -> str:
    """SHA-256 of a file — pins the artifact to the exact pool it was read from."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (extracted so the HELP TEXT is testable)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-molecule-out", type=Path, default=None)
    parser.add_argument(
        "--cohort",
        default="epoch17",
        choices=("epoch17", "all", "all_stratified"),
        help=(
            "which rows to analyse: 'epoch17' = the pool's rows whose 'epoch' "
            "column is 17; 'all' = every row of every epoch in the pool; "
            "'all_stratified' = a deterministic systematic sample of at most "
            "--per-epoch-limit rows from every epoch in the pool. The row and "
            "epoch counts depend on the pool and are measured at run time -- "
            "see cohort_facts / cohort_detail in the artifact"
        ),
    )
    parser.add_argument(
        "--per-epoch-limit",
        type=int,
        default=120,
        help="rows per epoch for the 'all_stratified' cohort (0 = no limit)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.pool.exists():
        print(f"FAIL: pool not found: {args.pool}", file=sys.stderr)
        return 1

    rows, cohort_facts = select_cohort(
        read_pool(args.pool), args.cohort, per_epoch=args.per_epoch_limit
    )
    if args.limit:
        rows = rows[: args.limit]
    cohort_facts["n_rows_processed"] = len(rows)
    cohort_facts["limit"] = int(args.limit)
    started = time.time()

    variants = {
        # PRIMARY: the FragFlex rule exactly as the brief states it, on the
        # WHOLE-molecule ETKDG geometry the brief specifies.
        "gated_whole": dict(
            gate=True, allow_metal_endpoints=True, geometry="whole"
        ),
        # PRIMARY, faithful geometry: the same repair, but the inter-component
        # distance is the CLOSEST APPROACH between two independently embedded
        # fragment shapes rather than an arbitrary relative placement.
        "gated_fragsep": dict(
            gate=True, allow_metal_endpoints=True, geometry="fragsep"
        ),
        # NULL CONTROL: the same repair with NO geometric information at all
        # (the candidate order degenerates to the atom-index order).
        "gated_index": dict(
            gate=True, allow_metal_endpoints=True, geometry="index"
        ),
        # COUNTERFACTUAL: no valence check, no per-step sanitize gate.
        "ungated_whole": dict(
            gate=False, allow_metal_endpoints=True, geometry="whole"
        ),
        # CORPUS-CONVENTION: metals are not bridge endpoints.
        "metal_preserving_whole": dict(
            gate=True, allow_metal_endpoints=False, geometry="whole"
        ),
    }
    per_molecule: List[Dict[str, object]] = []
    summary: Dict[str, object] = {}
    for label, kwargs in variants.items():
        t0 = time.time()
        recs = run_variant(
            rows, seed=args.seed, label=label, **kwargs
        )
        per_molecule.extend(recs)
        agg = _aggregate(recs)
        agg["wall_seconds"] = round(time.time() - t0, 2)
        summary[label] = agg

    # per-epoch breakdown for the primary variant
    by_epoch: Dict[str, Dict[str, object]] = {}
    for rec in per_molecule:
        if rec["variant"] != "gated_whole":
            continue
        key = str(rec.get("epoch", ""))
        by_epoch.setdefault(key, []).append(rec)
    epoch_summary = {
        k: _aggregate(v)
        for k, v in sorted(by_epoch.items(), key=lambda kv: _epoch_sort_key(kv[0]))
    }

    payload = {
        "artifact": "wfa142_fragflex_cminus1_probe",
        "tag": "MEASURED",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_used": False,
        "question": (
            "Does a FragFlex-style valence-aware C-1 spanning-tree completion "
            "raise our connectivity WITHOUT the sanitize gate dropping "
            "molecules (mass deletion)?"
        ),
        "cohort": args.cohort,
        # DERIVED from the measured cohort facts (Wf-A164).  It used to be a
        # literal per cohort -- true for the default pool, false for every
        # other pool and every other cohort.
        "cohort_detail": derive_cohort_detail(cohort_facts),
        "cohort_facts": cohort_facts,
        "n_rows_in_cohort": len(rows),
        "pool": str(args.pool),
        "pool_sha256": _sha256(args.pool),
        "pool_n_rows_total": cohort_facts["n_rows_in_pool"],
        "seed": args.seed,
        "method": {
            "edges_added_rule": (
                "exactly C-1 edges are added to a C-component molecule"
            ),
            "gate": (
                "valence-feasible under the repo's own accounting AND "
                "SanitizeMol-surviving after the add"
            ),
            "geometry": "ETKDG on the whole multi-fragment SMILES",
            "gpu": "0 (CPU-only; no checkpoint, no decode, no trainer)",
        },
        "variants": {
            "gated_whole": (
                "PRIMARY — valence-feasible + SanitizeMol-gated, C-1 edges, "
                "whole-molecule ETKDG geometry (the method the brief specifies)"
            ),
            "gated_fragsep": (
                "PRIMARY, faithful geometry — same repair, inter-component "
                "distance = closest approach between independently embedded "
                "fragment shapes"
            ),
            "ungated_whole": (
                "COUNTERFACTUAL — closest pairs added blind, one final sanitize"
            ),
            "gated_index": (
                "NULL CONTROL — the same repair with NO geometric information "
                "at all (candidate order degenerates to the atom-index order)"
            ),
            "metal_preserving_whole": (
                "CORPUS-CONVENTION — metals excluded as bridge endpoints"
            ),
        },
        "repo_conventions": {
            "metal_set_all7": sorted(METAL_SET),
            "donor_zs": sorted(DONOR_SET),
            "valence_table_source": (
                "molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py::"
                "covalent_valence_table() (== _COVALENT_VALENCE)"
            ),
            "valence_cost_source": (
                "molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py::"
                "_endpoint_valence_cost / _replay_remaining_budget"
            ),
            "unknown_z_cap": 4.0,
        },
        "summary": summary,
        "per_epoch_gated": epoch_summary,
        "corpus_reference": corpus_reference(args.corpus),
        "population": {
            "pool_rows": cohort_facts["n_rows_in_pool"],
            "pool_epochs": cohort_facts["n_epochs_in_pool"],
            "cohort": args.cohort,
            "rows_processed": len(rows),
            "statement": (
                f"ONE pool, ONE cohort ('{args.cohort}'): {len(rows):,} rows "
                f"processed out of {cohort_facts['n_rows_in_pool']:,} pool rows "
                f"spanning {cohort_facts['n_epochs_in_pool']} epochs."
            ),
        },
        "caveats": [
            f"ONE pool and ONE cohort: {len(rows):,} rows processed out of "
            f"{cohort_facts['n_rows_in_pool']:,} pool rows spanning "
            f"{cohort_facts['n_epochs_in_pool']} epochs.",
            "The pool carries NO coordinates: the 'shortest inter-component "
            "distance' that ranks the candidate pairs comes from an ETKDG "
            "embedding of the multi-fragment SMILES, so it measures this "
            "probe's own geometry, not the decoder's.",
            "The 'gated_index' and 'gated_fragsep' variants bound how much "
            "the (un-evaluable) distance ordering decides.",
            "No model was loaded, no decode was run, no GPU was touched.",
        ],
        "wall_seconds_total": round(time.time() - started, 2),
    }

    # -- Wf-A164: the narrative must be backed by this artifact's own fields --
    # The guard runs BEFORE its own result is added to the payload, so the
    # result can never be used to back a number in the prose it checks.
    guard = _narrative_self_consistency(payload)
    payload["narrative_self_consistency"] = guard
    # The strongest form of the same invariant: the cohort description must
    # be re-derivable from the measured cohort facts alone.  A hand-written
    # literal would disagree with the rows the tool actually processed.
    recomputed_detail = derive_cohort_detail(dict(cohort_facts))
    recheck = {
        "rule": (
            "cohort_detail must be re-derivable from cohort_facts alone; a "
            "hand-written literal would disagree with the rows the tool "
            "actually processed"
        ),
        "recomputed": recomputed_detail,
        "matches": recomputed_detail == payload["cohort_detail"],
    }
    payload["cohort_detail_recheck"] = recheck
    if guard["pass"] and recheck["matches"]:
        payload["status"] = "PASS"
    else:
        payload["status"] = "FAIL"
        payload["status_reason"] = (
            "narrative self-consistency FAILED: unbacked numbers "
            f"{sorted(guard['unbacked_numbers'])}"
            if not guard["pass"]
            else "cohort_detail is not re-derivable from cohort_facts"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False)
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(out_path, text)
    except Exception as exc:  # pragma: no cover
        print(f"WARN: atomic write failed ({exc}); falling back", file=sys.stderr)
        out_path.write_text(text)

    if payload["status"] != "PASS":
        # The artifact is written either way -- a FAIL is a MEASURED result,
        # not a missing one -- but the run's exit code says so.
        print(f"FAIL: {payload['status_reason']}", file=sys.stderr)
        print(
            f"unbacked numbers: {guard['unbacked_numbers']}", file=sys.stderr
        )
        print(f"wrote {out_path}", file=sys.stderr)
        return 1

    if args.per_molecule_out:
        pm_path = Path(args.per_molecule_out)
        pm_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "variant",
            "geometry",
            "epoch",
            "n_components_before",
            "n_components_after",
            "n_singletons_before",
            "n_singletons_after",
            "n_heavy_before",
            "n_heavy_after",
            "largest_frag_heavy_before",
            "largest_frag_heavy_after",
            "edges_added",
            "edges_metal_nonmetal",
            "edges_nonmetal_nonmetal",
            "rejected_by_valence",
            "rejected_by_sanitize",
            "stranded",
            "sanitize_ok",
            "sanitize_fail_reason",
            "embed_status",
            "min_intercomponent_distance",
            "n_metal_atoms",
            "n_metal_bonds_before",
            "n_metal_bonds_after",
            "metal_alone_before",
            "metal_alone_after",
            "smiles",
        ]
        lines = [",".join(fields)]
        for rec in per_molecule:
            row = []
            for f in fields:
                val = rec.get(f, "")
                text_val = str(val)
                if any(ch in text_val for ch in ',"\n'):
                    text_val = '"' + text_val.replace('"', '""') + '"'
                row.append(text_val)
            lines.append(",".join(row))
        csv_text = "\n".join(lines) + "\n"
        try:
            from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

            atomic_write_file(pm_path, csv_text)
        except Exception as exc:  # pragma: no cover
            print(f"WARN: atomic write failed ({exc}); falling back", file=sys.stderr)
            pm_path.write_text(csv_text)

    gated = summary["gated_whole"]
    print(
        "cohort={cohort} n={n} | gated_whole: comps {b:.4f}->{a:.4f} sing "
        "{sb:.4f}->{sa:.4f} edges/mol {e:.4f} stranded {st} sanitize_fail {sf} | "
        "gated_fragsep: comps {a2:.4f} edges/mol {e2:.4f} metal-edges {me:.4f} | "
        "ungated sanitize_fail {uf} | metal_preserving comps {mp:.4f}".format(
            cohort=args.cohort,
            n=gated.get("n", 0),
            b=gated.get("mean_n_components_before", float("nan")),
            a=gated.get("mean_n_components_after", float("nan")),
            sb=gated.get("mean_n_singletons_before", float("nan")),
            sa=gated.get("mean_n_singletons_after", float("nan")),
            e=gated.get("mean_edges_added", float("nan")),
            st=gated.get("n_stranded", 0),
            sf=gated.get("n_sanitize_fail", 0),
            a2=summary["gated_fragsep"].get(
                "mean_n_components_after", float("nan")
            ),
            e2=summary["gated_fragsep"].get(
                "mean_edges_added", float("nan")
            ),
            me=summary["gated_fragsep"].get(
                "mean_edges_metal_nonmetal", float("nan")
            ),
            uf=summary["ungated_whole"].get("n_sanitize_fail", 0),
            mp=summary["metal_preserving_whole"].get(
                "mean_n_components_after", float("nan")
            ),
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
