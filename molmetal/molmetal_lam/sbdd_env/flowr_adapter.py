"""FLOWR adapter — protocol stub.

================================================================
What this replaces
================================================================
``FLOWR`` (Alvira et al., 2024) is a flow-matching generative model
targeted at *3D pocket-conditioned* molecule generation. Its key
contribution is an SE(3)-equivariant message-passing backbone that
produces both atom types *and* 3D coordinates in a single flow.

Like FlowDock, FLOWR depends on ``torch_cluster`` + ``torch_scatter``
CUDA extensions, so we keep a :class:`typing.Protocol`-shaped stub
in the abstract layer and never link the heavy code.

Reference
---------
Alvira, S., et al. (2024).
*FLOWR: Flow Matching for Structure-Based Drug Design.*
arXiv:2404.02819.
https://arxiv.org/abs/2404.02819

Public API
----------
* :class:`FLOWRReferenceAdapter` — implements the
  :class:`molmetal.ports.MoleculeGenerator` protocol.
* :func:`is_flowr_available` — runtime probe for the cloned repo.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

from molmetal.domain import Molecule, Pocket
from molmetal.ports import GenerationConfig, MoleculeGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------
def is_flowr_available(repo_path: str = "molmetal/references/FLOWR") -> bool:
    """Return True if the FLOWR clone is present on disk."""
    return os.path.isdir(repo_path)


# ---------------------------------------------------------------------------
# Reference adapter
# ---------------------------------------------------------------------------
@dataclass
class FLOWRReferenceAdapter:
    """Protocol stub adapter for the cloned FLOWR repo.

    Implements :class:`molmetal.ports.MoleculeGenerator` by shape; a real
    adapter would load the ``flowr`` package + its pretrained
    Pocket2Mol/FLOWR weights from the cloned directory.
    """

    repo_path: str = "molmetal/references/FLOWR"
    fallback_smiles: List[str] = field(
        default_factory=lambda: [
            "CCO",                         # ethanol
            "CCN(CC)CC",                   # triethylamine
            "c1ccccc1",                    # benzene
        ],
    )
    _loaded: bool = False

    @property
    def name(self) -> str:
        return "FLOWR_reference_stub_v1"

    def setup(self, device: str = "cuda") -> None:
        """Probe-only — never imports the real ``flowr`` package."""
        self._loaded = is_flowr_available(self.repo_path)
        if self._loaded:
            logger.info(
                "FLOWR repo at %s — STUB MODE (upstream weights NOT loaded).",
                self.repo_path,
            )
        else:
            logger.info(
                "FLOWR repo NOT found at %s — adapter stays a stub.",
                self.repo_path,
            )

    def is_stub(self) -> bool:
        return not self._loaded

    def generate(
        self,
        pocket: Pocket,
        config: GenerationConfig,
    ) -> List[Molecule]:
        """Return ``n_samples`` SMILES drawn from a deterministic fallback list.

        A real adapter would invoke ``flowr.sample(pocket, n=config.n_samples)``
        and return :class:`Molecule` instances with 3D coordinates.
        """
        if not self._loaded:
            logger.warning(
                "FLOWRReferenceAdapter.generate() in stub mode — "
                "returning %d placeholder Molecules.",
                config.n_samples,
            )
        out: List[Molecule] = []
        for i in range(config.n_samples):
            smi = self.fallback_smiles[i % len(self.fallback_smiles)]
            out.append(Molecule.from_smiles(smi))
        return out

    def train_step(self, pocket: Pocket, mols: List[Molecule]) -> float:
        """Stub optimisation step — returns 0.0."""
        return 0.0

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "engine": "FLOWR (reference stub)",
            "engine_version": "arXiv:2404.02819",
            "repo_path": self.repo_path,
            "stub": not self._loaded,
            "paper": "Alvira et al., 2024, arXiv:2404.02819",
        }


# ---------------------------------------------------------------------------
# Protocol conformance hint
# ---------------------------------------------------------------------------
def _assert_protocol_conformance() -> None:
    """Static check that the adapter shape matches MoleculeGenerator."""
    adapter: MoleculeGenerator = FLOWRReferenceAdapter()  # type: ignore[assignment]
    _ = adapter.name
    _ = adapter.setup
    _ = adapter.generate
    _ = adapter.train_step
    _ = adapter.get_metadata


__all__ = [
    "FLOWRReferenceAdapter",
    "is_flowr_available",
]
