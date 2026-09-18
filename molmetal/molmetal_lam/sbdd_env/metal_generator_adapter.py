"""MetalLigandAdapter — implements the :class:`MetalLigandGenerator` port.

================================================================
What this replaces
================================================================
The original ``molmetal/molmetal_lam/scripts/baselines.py`` had no
metal-specific generator at all — generated SMILES were organic-only
ligands, and the metalloprotein extension was hand-rolled.  This
adapter wraps :mod:`molmetal.data.metal_smiles` (multi-component
form ``L1.L2....Ln.[M]``) so that any caller of the
:class:`molmetal.ports.generators.MetalLigandGenerator` Protocol can
behave identically on metalloprotein targets.

Public API
----------
* :class:`MetalLigandAdapter` — implements the Protocol; reconstructs
  the metal-complex SMILES, optionally embeds 3-D via RDKit
  (ETKDGv3 + MMFF94 fallback to UFF), and returns
  ``List[Complex]``.
* :func:`reconstruct_complex_smiles` — convenience helper that calls
  :func:`molmetal.data.metal_smiles.reconstruct_multi_component` with
  sensible defaults.

Notes
-----
* The reconstruction defaults to the *multi-component* form
  ``L1.L2....Ln.[M]`` because it round-trips through RDKit at 100%
  on MetalCytoToxDB, vs. ~96.7% for the legacy bracket form
  ``[M](L1)(L2)...``.  See the docstring of
  :mod:`molmetal.data.metal_smiles` for details.
* The ``[OH2]`` water placeholders emitted for under-coordinated
  complexes are *valid* SMILES but they are intentionally not
  desolvated — downstream PoseBusters / Vina passes can drop them.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import torch

from molmetal.data.metal_smiles import reconstruct_multi_component
from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports.generators import MetalLigandConfig, MetalLigandGenerator

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------
def reconstruct_complex_smiles(
    ligand_smiles: Union[str, List[str]],
    metal: str,
    oxidation_state: int = 2,
) -> str:
    """Reconstruct a full metal-complex SMILES in multi-component form.

    Wraps :func:`molmetal.data.metal_smiles.reconstruct_multi_component`
    so that callers may pass a single ``str`` *or* a list of fragments.

    Examples
    --------
    >>> reconstruct_complex_smiles("N.N.Cl.Cl", "Pt", 2)
    'N.N.Cl.Cl.[Pt]'
    >>> reconstruct_complex_smiles(["N", "N", "Cl", "Cl"], "Pt", 2)
    'N.N.Cl.Cl.[Pt]'
    """
    if isinstance(ligand_smiles, list):
        merged = ".".join(s.strip() for s in ligand_smiles if s and s.strip())
    else:
        merged = str(ligand_smiles or "").strip()
    return reconstruct_multi_component(merged, metal, oxidation_state)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
@dataclass
class MetalLigandAdapter:
    """Implements :class:`MetalLigandGenerator`.

    The adapter is **stateless apart from configuration**: each call to
    :meth:`generate` reconstructs the SMILES from scratch, so callers
    can swap metal centres and ligands freely.

    Parameters
    ----------
    embed_3d : bool
        When True, run RDKit ETKDGv3 + MMFF94/UFF on the reconstructed
        SMILES and populate the returned Complex's ``molecule.coords``
        with 3-D coordinates.  When False (default), coords are zeros
        and bonds are empty — the downstream Vina adapter is expected
        to re-embed.
    fallback_to_bracket : bool
        Reserved for future use.  The adapter currently always emits
        the multi-component form; the flag is accepted for Protocol
        completeness and round-trip with legacy callers.
    """

    embed_3d: bool = False
    fallback_to_bracket: bool = False
    _device: str = field(default="cpu", init=False)
    _initialised: bool = field(default=False, init=False)

    # ---------------------------------------------------------- Protocol
    @property
    def name(self) -> str:
        return "MetalLigandAdapter_v1"

    def setup(self, device: str = "cpu") -> None:
        self._device = str(device)
        # Lazy-import RDKit; we don't want to force the dependency at
        # module import time (the metal-reconstruction path itself is
        # pure-string and doesn't need RDKit).
        try:
            from rdkit import Chem  # noqa: F401
            self._initialised = True
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "MetalLigandAdapter.setup(): RDKit unavailable (%s); "
                "3-D embedding will be skipped even if embed_3d=True.",
                exc,
            )
            self._initialised = False

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "engine": "metal_smiles.reconstruct_multi_component",
            "rdkit_available": self._initialised,
            "embed_3d": self.embed_3d,
            "device": self._device,
        }

    # ---------------------------------------------------------- Public API
    def generate(
        self,
        pocket: Pocket,
        metal: str,
        ligand_smiles: Union[str, List[str]],
        config: Optional[MetalLigandConfig] = None,
    ) -> List[Complex]:
        """Return ``config.n_samples`` reconstructed metal-complex Complexes.

        Parameters
        ----------
        pocket : Pocket
            Conditioning pocket — carried through into each Complex
            so downstream scoring (Vina, PoseBusters) has context.
        metal : str
            Element symbol of the metal centre.
        ligand_smiles : str | list[str]
            Ligand SMILES (single ``str`` with ``.``-separated
            fragments, or a list of fragments).
        config : MetalLigandConfig | None
            Defaults to :class:`MetalLigandConfig()` when ``None``.
        """
        if config is None:
            config = MetalLigandConfig()
        if not self._initialised:
            self.setup(self._device)

        # 1. Reconstruct the full multi-component SMILES (string-only;
        #    no RDKit dependency).
        full_smi = reconstruct_complex_smiles(
            ligand_smiles, str(metal), int(config.oxidation_state),
        )
        if not full_smi:
            logger.warning(
                "MetalLigandAdapter.generate: empty reconstruction for "
                "metal=%r ligands=%r — returning no Complexes.",
                metal, ligand_smiles,
            )
            return []

        # 2. Optionally embed 3-D via RDKit (ETKDGv3 + MMFF94 → UFF).
        mols: List[Molecule] = []
        if self.embed_3d and config.embed_3d:
            mols = self._embed_to_molecules(
                full_smi, n_samples=int(config.n_samples),
            )
        else:
            # No 3-D embedding — single placeholder Molecule; downstream
            # Vina adapter is expected to re-embed from the SMILES.
            placeholder = self._placeholder_molecule(full_smi)
            mols = [placeholder] * int(config.n_samples)

        # 3. Wrap each Molecule in a Complex, threading the pocket
        #    and the original ligand SMILES through for traceability.
        complexes: List[Complex] = []
        for mol in mols[: int(config.n_samples)]:
            complexes.append(Complex(pocket=pocket, molecule=mol))
        return complexes

    # ---------------------------------------------------------- Internals
    def _placeholder_molecule(self, full_smi: str) -> Molecule:
        """Build a 0-atom/0-bond Molecule that carries only the SMILES."""
        coords = torch.zeros((0, 3), dtype=torch.float32)
        atom_types = torch.zeros(0, dtype=torch.long)
        bonds = torch.zeros((2, 0), dtype=torch.long)
        bond_types = torch.zeros(0, dtype=torch.long)
        formal_charges = torch.zeros(0, dtype=torch.long)
        return Molecule(
            coords=coords,
            atom_types=atom_types,
            bonds=bonds,
            bond_types=bond_types,
            formal_charges=formal_charges,
            smiles=full_smi,
        )

    def _embed_to_molecules(
        self, full_smi: str, n_samples: int,
    ) -> List[Molecule]:
        """Embed ``full_smi`` via RDKit; return up to ``n_samples`` Molecules.

        On failure, falls back to a placeholder Molecule carrying only
        the SMILES so callers always get ``n_samples`` results.
        """
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        out: List[Molecule] = []
        try:
            rdmol = Chem.MolFromSmiles(full_smi)
            if rdmol is None:
                raise ValueError(f"RDKit could not parse {full_smi!r}")
            rdmol = Chem.AddHs(rdmol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            embed_status = AllChem.EmbedMolecule(rdmol, params)
            if embed_status == -1:
                # 2-D fallback (still produces a Molecule)
                Chem.RemoveHs(rdmol)
                return [self._placeholder_molecule(full_smi)] * n_samples
            # MMFF94 first, UFF fallback (matches PoseBusters path).
            try:
                mmff_status = AllChem.MMFFOptimizeMolecule(rdmol, maxIters=200)
            except Exception:
                mmff_status = -1
            if mmff_status != 0:
                try:
                    AllChem.UFFOptimizeMolecule(rdmol, maxIters=200)
                except Exception:
                    pass
            Chem.RemoveHs(rdmol)
            coords = np.asarray(
                [list(rdmol.GetConformer().GetAtomPosition(a.GetIdx()))
                 for a in rdmol.GetAtoms()],
                dtype=np.float32,
            )
            atom_types = np.asarray(
                [a.GetAtomicNum() for a in rdmol.GetAtoms()], dtype=np.int64,
            )
            n_bonds = rdmol.GetNumBonds()
            bond_index = np.zeros((2, n_bonds), dtype=np.int64)
            bond_types = np.zeros(n_bonds, dtype=np.int64)
            for i, b in enumerate(rdmol.GetBonds()):
                bond_index[0, i] = b.GetBeginAtomIdx()
                bond_index[1, i] = b.GetEndAtomIdx()
                bond_types[i] = int(b.GetBondType())
            mol = Molecule(
                coords=torch.from_numpy(coords),
                atom_types=torch.from_numpy(atom_types),
                bonds=torch.from_numpy(bond_index),
                bond_types=torch.from_numpy(bond_types),
                formal_charges=torch.zeros(len(atom_types), dtype=torch.long),
                smiles=full_smi,
            )
            out.append(mol)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "MetalLigandAdapter._embed_to_molecules failed for %r (%s); "
                "returning placeholder.",
                full_smi, exc,
            )
            return [self._placeholder_molecule(full_smi)] * n_samples
        # Replicate to n_samples — sampling multiple distinct 3-D
        # conformers is left to a future "live" adapter (cf. Pocket2Mol).
        return out * max(1, n_samples)


# ---------------------------------------------------------------------------
# Protocol conformance hint
# ---------------------------------------------------------------------------
def _assert_protocol_conformance() -> None:
    """Static check that the adapter shape matches MetalLigandGenerator."""
    adapter: MetalLigandGenerator = MetalLigandAdapter()  # type: ignore[assignment]
    _ = adapter.name
    _ = adapter.setup
    _ = adapter.generate
    _ = adapter.get_metadata


__all__ = [
    "MetalLigandAdapter",
    "reconstruct_complex_smiles",
]
