"""PDB file ⇄ Pocket dataclass converters.

A "Pocket" in our framework is the set of protein atoms within a sphere
around the binding-site centroid, plus minimal metadata (chain/residue ids).
We use BioPython's ``PDBParser`` because it handles standard PDB quirks
(alt-locs, insertion codes, multi-model) and exposes a clean per-atom view.

Heavy imports (BioPython) are lazy so the rest of molmetal/ stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch

from molmetal.domain import Pocket

if TYPE_CHECKING:
    from Bio.PDB.Structure import Structure  # noqa: F401


# ---------------------------------------------------------------------------
# Public converter
# ---------------------------------------------------------------------------
def _load_pocket(
    path: Union[str, Path],
    ligand_center: torch.Tensor,
    radius: float = 6.0,
    chain_whitelist: list[str] | None = None,
    include_heteroatoms: bool = False,
) -> Pocket:
    """Parse a PDB file and extract a pocket centred on ``ligand_center``.

    Parameters
    ----------
    path : str | Path
        PDB file path.
    ligand_center : (3,) tensor
        Binding-site centroid (Å).  Typically the centre of mass of the
        bound ligand.
    radius : float
        Sphere radius (Å) around ``ligand_center``.
    chain_whitelist : list[str] | None
        If provided, only atoms whose chain id is in the list are kept.
    include_heteroatoms : bool
        By default we drop HETATM (waters, ligands, ions).  Set to True
        if you want the full pocket including e.g. catalytic Zn²⁺.

    Returns
    -------
    Pocket
    """
    from Bio.PDB import PDBParser, PPBuilder, is_aa
    # Three-letter to one-letter (API change in BioPython >= 1.83:
    # ``three_to_one`` was removed in favour of ``protein_letters_3to1``).
    try:
        from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one  # type: ignore
    except ImportError:  # pragma: no cover — kept for very old BioPython
        from Bio.PDB.Polypeptide import three_to_one  # type: ignore[unreachable]

    parser = PDBParser(QUIET=True)
    pdb_id = Path(path).stem
    structure: "Structure" = parser.get_structure(pdb_id, str(path))

    # Use first model only (NMR structures often have many)
    model = next(structure.get_models())

    centre = ligand_center.float() if isinstance(ligand_center, torch.Tensor) \
        else torch.as_tensor(ligand_center, dtype=torch.float32)
    if centre.dim() > 1:
        centre = centre.flatten()
    assert centre.shape == (3,), f"ligand_center must be (3,), got {centre.shape}"

    coords, atom_types, residue_ids, chain_ids, mask = [], [], [], [], []
    kept_residues: set = set()

    for atom in model.get_atoms():
        # Standard residues only by default
        residue = atom.get_parent()
        hetflag = residue.get_id()[0].strip()
        if hetflag.startswith("H_") or hetflag == "W":
            # HETATM (waters, ligands, ions)
            if not include_heteroatoms:
                continue
        if chain_whitelist is not None and residue.get_parent().id not in chain_whitelist:
            continue

        pos = atom.get_vector().get_array()  # (3,) numpy
        d = float(((pos - centre.numpy()) ** 2).sum() ** 0.5)
        if d > radius:
            continue

        coords.append(pos)
        atom_types.append(_atomic_number_from_element(atom.element))
        residue_ids.append(residue.get_id()[1])
        chain_ids.append(_chain_id_to_int(residue.get_parent().id))
        mask.append(True)
        kept_residues.add(residue.get_full_id())

    if not coords:
        raise ValueError(
            f"No atoms within {radius} Å of {centre.tolist()} in {path}. "
            "Check that ligand_center is correct."
        )

    coords_t = torch.tensor(coords, dtype=torch.float32)
    return Pocket(
        pdb_id=pdb_id,
        coords=coords_t,
        atom_types=torch.tensor(atom_types, dtype=torch.long),
        residue_ids=torch.tensor(residue_ids, dtype=torch.long),
        chain_ids=torch.tensor(chain_ids, dtype=torch.long),
        mask=torch.tensor(mask, dtype=torch.bool),
        center=centre,
        radius=radius,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PERIODIC_NUMBER = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "P": 15, "S": 16, "CL": 17, "BR": 35, "I": 53,
    # Common d-block in metalloproteins:
    "ZN": 30, "FE": 26, "MG": 12, "MN": 25, "CU": 29, "CA": 20,
    "NI": 28, "CO": 27, "MO": 42, "PT": 78, "PD": 46, "AU": 79,
}


def _atomic_number_from_element(elem: str) -> int:
    if elem is None:
        return 0
    return _PERIODIC_NUMBER.get(elem.strip().upper(), 0)


def _chain_id_to_int(chain_id: str) -> int:
    """Stable mapping: 'A'->0, 'B'->1, … but accepts arbitrary 1-char ids."""
    if not chain_id:
        return 0
    return ord(chain_id[0].upper()) - ord("A")
