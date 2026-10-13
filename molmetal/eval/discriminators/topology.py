"""Topology discriminator (Wf-A11 / eval wave).

Cheap per-decoded-mol topology audit:
  * ``single_fragment_rate``    — fraction of mols that are 1 fragment
  * ``ring_size_distribution``  — {3..7: count} of each ring size
  * ``five_six_ring_fraction``  — fraction of rings that are 5/6
  * ``largest_fragment_atom_count`` — heavy-atom count of the largest
    fragment (median across the input)

All metrics are computed from RDKit's :class:`GetMolFrags` /
:class:`GetRingInfo`.  The discriminator is tolerant of a ``None`` mol
(parse failure -> all keys say ``0.0``).

Backward-compat
---------------
* :meth:`forward` returns ``None`` (not ``{}``) so callers can tell
  "this mol could not be parsed at all" apart from "this mol parsed but
  had no topologically-meaningful signal".
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rdkit import Chem

#: RDKit ring-size histogram keys.
RING_SIZES: tuple = (3, 4, 5, 6, 7)


class TopologyDiscriminator:
    """Topology audit over one decoded RDKit Mol."""

    name = "topology"

    def forward(self, decoded_mol: Optional[Chem.Mol]) -> Optional[Dict[str, float]]:
        """Return the topology audit dict for *decoded_mol*."""
        if decoded_mol is None:
            return None
        try:
            frags = Chem.GetMolFrags(decoded_mol, asMols=False)
            n_atoms = decoded_mol.GetNumAtoms()
            if n_atoms <= 0:
                return None
            n_frags = len(frags)
            single_fragment = 1.0 if n_frags == 1 else 0.0
            largest_frag_atoms = max(len(f) for f in frags) if frags else 0

            ring_info = decoded_mol.GetRingInfo()
            ring_sizes = [len(r) for r in ring_info.AtomRings()]
            ring_hist: Dict[int, int] = {k: 0 for k in RING_SIZES}
            for rs in ring_sizes:
                if rs in ring_hist:
                    ring_hist[rs] += 1
            n_rings = sum(ring_hist.values())
            five_six = (
                float(ring_hist[5] + ring_hist[6]) / float(n_rings)
                if n_rings > 0
                else 0.0
            )
            return {
                "single_fragment_rate": single_fragment,
                "largest_fragment_atom_count": float(largest_frag_atoms),
                "five_six_ring_fraction": five_six,
                **{f"ring_size_{k}": float(v) for k, v in ring_hist.items()},
            }
        except Exception:
            return None

    def aggregate(self, rows: List[Dict[str, float]]) -> Dict[str, float]:
        """Mean-aggregate a list of topology dicts."""
        if not rows:
            return {
                "single_fragment_rate": 0.0,
                "largest_fragment_atom_count": 0.0,
                "five_six_ring_fraction": 0.0,
                **{f"ring_size_{k}": 0.0 for k in RING_SIZES},
            }
        n = float(len(rows))
        keys = rows[0].keys()
        return {k: sum(float(r.get(k, 0.0)) for r in rows) / n for k in keys}


__all__ = ["TopologyDiscriminator", "RING_SIZES"]