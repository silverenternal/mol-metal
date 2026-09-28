"""Finite molecule sheaf/cover atlas helpers for cross-domain similarity.

================================================================
What this replaces / adds
================================================================
Adapted from multiflow/pocket_core/contact_sheaf.py (P0-236L) for
Mol-Metal's cross-molecule similarity axis (TODO-143 §2.5 Sprint B2).

Multiflow's :class:`ContactSection` describes a *pocket-ligand* region:
ligand atoms, pocket patches, contact edges.  Mol-Metal operates one
level up — on *molecules* (no pocket) — and needs the same finite
overlap/cover diagnostic to score 4-domain × cross-molecule similarity:

    atoms       -> frozenset[int]      (was ligand_atoms)
    fragments   -> frozenset[str]      (was pocket_patches; here,
                                         functional group / scaffold
                                         names like ``"amide"``,
                                         ``"aromatic_ring"``)
    bonds       -> frozenset[tuple]    (was contact_edges)

The residual formula is intentionally identical (max of bond Jaccard
disagreement and 0.5× atom / fragment Jaccard disagreement) so the
upstream ``failed_overlap`` threshold stays comparable across pocket
and molecule atlases.

Public API
----------
* :class:`MoleculeSection` — frozen dataclass (name, atoms, fragments,
  bonds, weight).
* :func:`section_from_payload` — inverse of :meth:`MoleculeSection.to_payload`.
* :func:`overlap_residual(left, right)` — disagreement dict between two
  sections; ``overlap_residual`` is the headline scalar in [0, 1].
* :func:`molecule_atlas_payload(sections, *, atlas_id,
  residual_threshold)` — cross-mol pair payload, used by the
  ``cross_mol_similarity`` metric.

Honest framing
--------------
This is a finite-cover diagnostic, not a theorem certificate.  It tells
us whether two molecule sections can be glued consistently on the
shared atom/fragment cover.  A residual > threshold flags a
"patch_gluing_obstruction" for the caller to refine.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MoleculeSection:
    """Finite cover of a single molecule (or sub-molecule region).

    Parameters
    ----------
    name      : str — section identifier (e.g. ``"mol_Pt_cisplatin"``).
    atoms     : frozenset[int] — heavy-atom indices in this section.
    fragments : frozenset[str] — functional-group / scaffold labels
                 covering the section's atoms (e.g. ``{"amide",
                 "aromatic_ring"}``).
    bonds     : frozenset[tuple[int, int]] — covalent bond index
                 pairs (undirected: ``(i, j)`` and ``(j, i)`` are
                 treated identically by the cover diagnostic).
    weight    : float — section importance (default 1.0).
    """

    name: str
    atoms: frozenset[int]
    fragments: frozenset[str]
    bonds: frozenset[tuple[int, int]]
    weight: float = 1.0

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "name": self.name,
            "atoms": sorted(int(x) for x in self.atoms),
            "fragments": sorted(self.fragments),
            "bonds": [[int(i), int(j)] for i, j in sorted(self.bonds)],
            "weight": float(self.weight),
        }


def section_from_payload(payload: Mapping[str, Any]) -> MoleculeSection:
    """Inverse of :meth:`MoleculeSection.to_payload`."""
    raw_bonds = payload.get("bonds", [])
    return MoleculeSection(
        name=str(payload.get("name", payload.get("section_id", "section"))),
        atoms=frozenset(int(x) for x in payload.get("atoms", [])),
        fragments=frozenset(str(x) for x in payload.get("fragments", [])),
        bonds=frozenset(
            (int(b[0]), int(b[1])) for b in raw_bonds if len(b) >= 2
        ),
        weight=float(payload.get("weight", 1.0)),
    )


def overlap_residual(left: MoleculeSection, right: MoleculeSection) -> dict[str, Any]:
    """Jaccard-disagreement scores between two molecule sections.

    Returns a dict with these keys:

    * ``shared_atom_count``, ``shared_fragment_count``,
      ``bond_overlap_count``, ``bond_union_count`` — overlap sizes.
    * ``bond_disagreement`` — ``1 - |L ∩ R| / |L ∪ R|`` (1.0 if no bonds).
    * ``atom_disagreement`` — ``1 - |L ∩ R| / |L ∪ R|`` (1.0 if no atoms).
    * ``fragment_disagreement`` — same formula on the fragment label set.
    * ``overlap_residual`` — ``max(bond_disagreement, 0.5·atom_disagreement,
      0.5·fragment_disagreement)``; the headline scalar in [0, 1].

    A residual of 0.0 means the two sections are *identical* on the
    shared cover; a residual of 1.0 means *disjoint*.  Cross-mol
    similarity uses ``1 - residual`` as the headline similarity in
    [0, 1].
    """
    shared_atoms = left.atoms & right.atoms
    shared_fragments = left.fragments & right.fragments
    bond_union = left.bonds | right.bonds
    bond_overlap = left.bonds & right.bonds
    atom_union = left.atoms | right.atoms
    fragment_union = left.fragments | right.fragments
    bond_disagreement = 1.0 - (
        (len(bond_overlap) / len(bond_union)) if bond_union else 1.0
    )
    atom_disagreement = 1.0 - (
        (len(shared_atoms) / len(atom_union)) if atom_union else 1.0
    )
    fragment_disagreement = 1.0 - (
        (len(shared_fragments) / len(fragment_union)) if fragment_union else 1.0
    )
    residual = max(
        bond_disagreement,
        0.5 * atom_disagreement,
        0.5 * fragment_disagreement,
    )
    return {
        "left_section": left.name,
        "right_section": right.name,
        "shared_atom_count": len(shared_atoms),
        "shared_fragment_count": len(shared_fragments),
        "bond_overlap_count": len(bond_overlap),
        "bond_union_count": len(bond_union),
        "bond_disagreement": bond_disagreement,
        "atom_disagreement": atom_disagreement,
        "fragment_disagreement": fragment_disagreement,
        "overlap_residual": float(residual),
    }


def molecule_atlas_payload(
    sections: Sequence[MoleculeSection | Mapping[str, Any]],
    *,
    atlas_id: str,
    residual_threshold: float = 0.35,
) -> dict[str, Any]:
    """Cross-molecule sheaf/cover diagnostic for 4-domain × cross-mol similarity.

    For each unordered pair of sections that shares at least one atom
    or fragment label, compute :func:`overlap_residual`.  The aggregated
    payload reports ``max`` / ``mean`` residuals, the count of failed
    pairs (``residual > threshold``), and the unique-atom /
    unique-fragment / unique-bond counts across the entire atlas.

    Parameters
    ----------
    sections : sequence of MoleculeSection or payload dicts.
    atlas_id : str — human-readable identifier for the cross-mol atlas.
    residual_threshold : float — cutoff above which a pair counts as a
                            gluing obstruction (default 0.35).

    Returns
    -------
    dict — atlas summary; ``global_section_exists`` is True iff there
    are no failed pairs (i.e. the finite cover glues consistently).
    """
    normalized = [
        section if isinstance(section, MoleculeSection) else section_from_payload(section)
        for section in sections
    ]
    overlaps = []
    for idx, left in enumerate(normalized):
        for right in normalized[idx + 1 :]:
            if not (left.atoms & right.atoms or left.fragments & right.fragments):
                continue
            overlaps.append(overlap_residual(left, right))
    max_residual = max((item["overlap_residual"] for item in overlaps), default=0.0)
    mean_residual = (
        sum(item["overlap_residual"] for item in overlaps) / len(overlaps)
        if overlaps
        else 0.0
    )
    failed = [item for item in overlaps if item["overlap_residual"] > residual_threshold]
    bonds: set[tuple[int, int]] = set()
    atoms: set[int] = set()
    fragments: set[str] = set()
    for section in normalized:
        bonds |= set(section.bonds)
        atoms |= set(section.atoms)
        fragments |= set(section.fragments)
    return {
        "atlas_id": atlas_id,
        "section_count": len(normalized),
        "section_payloads": [section.to_payload() for section in normalized],
        "unique_atom_count": len(atoms),
        "unique_fragment_count": len(fragments),
        "unique_bond_count": len(bonds),
        "overlap_pair_count": len(overlaps),
        "max_overlap_residual": float(max_residual),
        "mean_overlap_residual": float(mean_residual),
        "failed_overlap_count": len(failed),
        "global_section_exists": len(failed) == 0,
        "overlap_residual_threshold": float(residual_threshold),
        "overlap_payloads": overlaps,
    }
