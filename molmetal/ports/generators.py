"""Generator ports (Protocol interfaces) for the molmetal framework.

This module hosts the generator-side :class:`typing.Protocol` definitions
that are separate from the heavy ``ports/__init__.py`` to keep the
generator-only surface lightweight and importable from environments
that do not need the docking / scoring / loop ports.

Hexagonal architecture
----------------------
Any concrete implementation of a port can be swapped without changing
the orchestration code.  All ports are ``typing.Protocol`` classes
(PEP 544) — they are duck-typed, so adapters don't need to inherit
explicitly.

Ports in this file
------------------
- :class:`MetalLigandGenerator` — emits ``List[Complex]`` from a metal
  token plus ligand SMILES, reconstructing the full metal-complex
  SMILES via :mod:`molmetal.data.metal_smiles` (multi-component form
  ``L1.L2....Ln.[M]``).

This port extends the abstract ``MoleculeGenerator`` contract from
:mod:`molmetal.ports` with metal-specific inputs/outputs, while
remaining *duck-typed* so existing :class:`MoleculeGenerator`
adapters can be substituted where appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

import torch

from molmetal.domain import Complex, Pocket


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetalLigandConfig:
    """Inputs to :meth:`MetalLigandGenerator.generate`.

    Attributes
    ----------
    n_samples : int
        How many Complexes to produce (per ``ligand_smiles`` entry).
    oxidation_state : int
        Oxidation state of the metal centre (drives coordination
        geometry; e.g. ``Pt(IV) = 4`` → octahedral, ``Pt(II) = 2`` →
        square-planar).
    seed : int
        RNG seed for any downstream sampling noise.
    embed_3d : bool
        If True, the adapter should run an ETKDGv3 3-D embedding of
        the reconstructed SMILES before returning the Complex.  When
        False (the default for the stub), the Complex carries a
        :class:`Molecule` with empty bonds and zero 3-D coordinates,
        leaving embedding to the downstream Vina adapter.
    extra_metadata : dict
        Adapter-specific overrides (charge, geometry override, …).
    """

    n_samples: int = 1
    oxidation_state: int = 2
    seed: int = 42
    embed_3d: bool = False
    extra_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Port: MetalLigandGenerator
# ---------------------------------------------------------------------------
@runtime_checkable
class MetalLigandGenerator(Protocol):
    """Generate metal-complex candidates (List[Complex]) from a metal token.

    Concrete implementations are *adapters* (e.g.
    :class:`molmetal_lam.sbdd_env.metal_generator_adapter.MetalLigandAdapter`).
    The adapter contract is:

    1. Take a :class:`Pocket` (for context — the pocket drives metal
       choice in real pipelines, but for the protocol stub the pocket
       is just passed through).
    2. Take a metal element symbol (``"Pt"``, ``"Ru"``, ``"Ir"`` …).
    3. Take one or more ligand SMILES (multi-component ``.``-separated
       strings are fine, as is a list of fragments).
    4. Return ``List[Complex]`` — one Complex per generated candidate,
       each with ``molecule.smiles`` set to the **reconstructed
       multi-component metal-complex SMILES** (``L1.L2....Ln.[M]``).

    The metal token injection happens *inside* :meth:`generate` so that
    downstream callers can treat this port as a drop-in replacement
    for :class:`molmetal.ports.MoleculeGenerator` on metalloprotein
    targets.
    """

    @property
    def name(self) -> str:
        """Identifier — e.g. ``"MetalLigandAdapter_v1"``."""
        ...

    def setup(self, device: str = "cpu") -> None:
        """Load any heavy resources (RDKit, MMFF, etc.).  No-op for stubs."""
        ...

    def generate(
        self,
        pocket: Pocket,
        metal: str,
        ligand_smiles: str | List[str],
        config: MetalLigandConfig,
    ) -> List[Complex]:
        """Produce ``config.n_samples`` metal-complex Complexes.

        Parameters
        ----------
        pocket : Pocket
            Conditioning pocket (used for downstream pose scoring).
        metal : str
            Element symbol of the metal centre (``"Pt"``, ``"Ru"`` …).
        ligand_smiles : str | list[str]
            One or more ligand SMILES.  When a single ``str`` is given,
            multi-component ``.``-separated form is honoured.
        config : MetalLigandConfig
            See :class:`MetalLigandConfig`.
        """
        ...

    def get_metadata(self) -> dict:
        """Reproducibility info: paper, code, dataset, config."""
        ...


__all__ = [
    "MetalLigandConfig",
    "MetalLigandGenerator",
]
