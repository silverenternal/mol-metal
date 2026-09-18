"""Tile — A frozen building-block of the click-chemistry tile library.

This module implements the **Tile layer** of the LamClick framework,
formalized in ``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §9.

Core thesis
-----------
A tile is a small, drug-like reactive fragment (typically 5-30 heavy atoms)
that carries one or more functional handles suitable for click chemistry
(CuAAC, SPAAC, SPC, Diels-Alder, Thiol-Ene, ...).

In the lambda-calculus reading, a tile is a *closed term* (a complete
molecule) of bounded complexity, tagged with the reactions it can enter
into. The *tile library* is therefore the *constant pool* used by the
proof search to construct larger drug candidates.

Tile invariants
---------------
* ``smiles``            canonical (RDKit) SMILES string of the tile
* ``coords``            optional 3D torch.Tensor of shape ``(N_atoms, 3)``
                        (None if no conformer has been embedded)
* ``functional_groups`` list of string tags — e.g. ``['azide', 'primary_amine']``
* ``tile_id``           deterministic hex hash of the canonical SMILES,
                        used as the primary key in the library dict
* ``sas_score``         synthetic accessibility (1=easy, 10=very hard)
* ``mw`` ``logp`` ``tpsa`` physico-chemical descriptors (RDKit Crippen)

Operations
----------
* ``.has_group(name) -> bool``  test whether the tile carries a given tag
* ``.to_smiles() -> str``       canonical SMILES (alias for ``self.smiles``)

Frozen dataclass
----------------
``Tile`` is ``frozen=True`` so that the library is effectively immutable
once built; this mirrors the read-only nature of the constant pool in the
MLC search algorithm.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

# Lazy torch import — only required when 3D coords are present.
# Lazy rdkit import — only required when computing descriptors.
__all__ = ["Tile"]


# ---------------------------------------------------------------------------
# The Tile dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """A click-chemistry building-block tile.

    Parameters
    ----------
    smiles : str
        Canonical SMILES of the tile (RDKit-canonicalized).
    coords : Optional[torch.Tensor]
        Optional 3D coordinates of shape ``(N_atoms, 3)``. May be ``None``
        if no conformer has been embedded.
    functional_groups : List[str]
        Tags describing the reactive handles (e.g. ``"azide"``,
        ``"terminal_alkyne"``, ``"cyclooctyne"``, ``"diene"``,
        ``"dienophile"``, ``"thiol"``, ``"alkene"``).
    tile_id : str
        Hex digest used as the primary key. If empty, a deterministic
        hash of the canonical SMILES is generated automatically.
    sas_score : float
        Synthetic accessibility score (1.0=easy, 10.0=hard). Default 1.0.
    mw : float
        Molecular weight (g/mol). Default 0.0.
    logp : float
        Wildman-Crippen logP. Default 0.0.
    tpsa : float
        Topological polar surface area (Angstrom^2). Default 0.0.
    """

    smiles: str
    coords: Optional["object"] = None  # torch.Tensor (lazy import)
    functional_groups: List[str] = field(default_factory=list)
    tile_id: str = ""
    sas_score: float = 1.0
    mw: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0

    def __post_init__(self) -> None:
        """Auto-generate ``tile_id`` if caller left it empty.

        A frozen dataclass cannot assign to ``self`` directly, so we use
        ``object.__setattr__`` to patch the id field.
        """
        if not self.tile_id:
            # Deterministic 12-char hex hash from the canonical SMILES.
            digest = hashlib.sha256(self.smiles.encode("utf-8")).hexdigest()
            object.__setattr__(self, "tile_id", digest[:12])

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def has_group(self, name: str) -> bool:
        """Return True if ``name`` appears in ``functional_groups``.

        Parameters
        ----------
        name : str
            The functional-group tag to look up (e.g. ``"azide"``).

        Returns
        -------
        bool
            ``True`` if the tag is present, ``False`` otherwise.
        """
        return name in self.functional_groups

    def to_smiles(self) -> str:
        """Return the canonical SMILES of the tile."""
        return self.smiles

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Compact representation showing id, SMILES and tags."""
        tags = ",".join(self.functional_groups) if self.functional_groups else "-"
        return f"Tile(id={self.tile_id}, smiles={self.smiles!r}, tags=[{tags}])"
