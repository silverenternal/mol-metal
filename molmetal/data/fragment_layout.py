# SPDX-License-Identifier: Apache-2.0
"""Separate the disconnected fragments of an embedded multi-component SMILES.

THE DEFECT THIS FIXES (Wf-A113 / T57)
--------------------------------------
RDKit's ``EmbedMolecule`` embeds each disconnected fragment independently and
then translates them all to the same place.  Two waters of the same topology
end up with the SAME conformer at the SAME coordinates:

    >>> smi = "[OH2].[OH2].[OH2].[OH2].[OH2].[OH2].[Ru]"
    >>> AllChem.EmbedMolecule(Chem.AddHs(Chem.MolFromSmiles(smi)), randomSeed=0)
    0                                    # "success"
    >>> min pairwise distance
    0.0                                  # six oxygens on one point

MEASURED on 399 clouds built from ``metalcytotox_5metal_multitask_v3.csv``:
327 (82.0%) contained at least one pair of atoms closer than 1e-3 A, 8,782
such pairs in total.  The flow-matching objective was therefore trained to
bring solvent fragments onto the ligand, and the geometry term's clash value
on GROUND-TRUTH geometry was 0.807 — it was measuring the data, not a bug.

WHAT THIS DOES
--------------
Keeps the largest fragment and the fragment that owns ``keep_atom`` where they
are, then translates every other fragment along a deterministic direction
until its closest contact with the already-placed atoms is at least
``clearance``.  No RNG: the direction set is a fixed Fibonacci sphere and the
radius ladder is fixed, so the same input always gives the same output.

WHAT IT DOES NOT DO
-------------------
It does not claim the separated layout is physical.  A solvent molecule's
position in a crystal is a fact about the crystal; this loader has no crystal.
The claim is only that the layout no longer teaches the model that atoms
overlap.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch

#: Minimum allowed closest contact between a moved fragment and the atoms
#: already placed, in Angstrom.  2.0 A is below every covalent bond and above
#: every hard clash, so a moved fragment can never create a bond-like contact.
DEFAULT_CLEARANCE_A: float = 2.0

#: Radius ladder, Angstrom.  Tried in order; the first radius that clears wins,
#: so fragments sit as close to the complex as the clearance allows.
_RADIUS_LADDER: Sequence[float] = (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0)


def _fibonacci_directions(n: int, device, dtype) -> torch.Tensor:
    """``(n, 3)`` unit vectors, evenly spread, deterministic."""
    i = torch.arange(n, device=device, dtype=dtype) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / float(n))
    theta = torch.pi * (1.0 + 5.0 ** 0.5) * i
    return torch.stack(
        [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)],
        dim=-1,
    )


def separate_fragments(
    coords: torch.Tensor,
    fragments: Iterable[Sequence[int]],
    *,
    keep_atom: Optional[int] = None,
    clearance: float = DEFAULT_CLEARANCE_A,
    n_directions: int = 64,
) -> torch.Tensor:
    """Return ``coords`` with disconnected fragments moved apart.

    Parameters
    ----------
    coords
        ``(N, 3)`` positions.
    fragments
        Atom indices per connected component (``Chem.GetMolFrags`` output).
    keep_atom
        An atom whose fragment must NOT move (the metal, which the caller
        places afterwards).  Its fragment is treated as already placed.
    clearance
        Minimum closest contact between a moved fragment and the placed atoms.

    Returns
    -------
    A new ``(N, 3)`` tensor.  ``coords`` is not modified.
    """
    frags: List[List[int]] = [list(f) for f in fragments if len(f)]
    out = coords.clone()
    if len(frags) <= 1:
        return out

    # Largest fragment first, then the one holding keep_atom, are anchors.
    order = sorted(range(len(frags)), key=lambda k: (-len(frags[k]), min(frags[k])))
    anchors: List[int] = [order[0]]
    if keep_atom is not None:
        for k in order:
            if int(keep_atom) in frags[k] and k not in anchors:
                anchors.append(k)
    moving = [k for k in order if k not in anchors]

    placed_idx: List[int] = []
    for k in anchors:
        placed_idx.extend(frags[k])
    if not placed_idx:
        return out

    dirs = _fibonacci_directions(int(n_directions), coords.device, coords.dtype)
    placed = out[torch.tensor(placed_idx, dtype=torch.long, device=coords.device)]

    for k in moving:
        idx = torch.tensor(frags[k], dtype=torch.long, device=coords.device)
        block = out[idx]
        chosen = None
        best_any = None  # (min_dist, translation) — fallback when nothing clears
        for radius in _RADIUS_LADDER:
            for d in dirs:
                translation = d * float(radius)
                dist = float(torch.cdist(block + translation, placed).min())
                if best_any is None or dist > best_any[0]:
                    best_any = (dist, translation)
                if dist >= float(clearance):
                    chosen = translation
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = best_any[1]
        out[idx] = block + chosen
        placed = torch.cat([placed, out[idx]], dim=0)
    return out


def min_pair_distance(coords: torch.Tensor) -> float:
    """Closest contact in ``coords``, or ``inf`` for fewer than two atoms."""
    n = int(coords.shape[0])
    if n < 2:
        return float("inf")
    d = torch.cdist(coords, coords) + torch.eye(n, device=coords.device) * 1e9
    return float(d.min())
