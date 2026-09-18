"""REINVENT4 adapter — protocol stub.

================================================================
What this replaces
================================================================
REINVENT4 (He, J., et al., 2024) is a modern molecular-design tool
built on top of REINVENT (3.x). It exposes a TOML-driven scoring
pipeline and an RL loop that maximises a weighted sum of *scoring
components*. The reference repo ships heavy optional plugins
(``reinvent_plugins`` — QSAR, docking, shape, RAscore) that depend
on proprietary models and binaries.

This module wraps the cloned repo *conceptually*: it discovers the
TOML presets on disk, exposes a :class:`molmetal.ports.ScoringFunction`
adapter, but degrades to the lightweight RDKit-only
:class:`REINVENT4Scorer` (see :mod:`molmetal_lam.sbdd_env.reinvent_wrapper`)
when the real plugins are unavailable.

Reference
---------
He, J., et al. (2024).
*REINVENT4: Modern AI-Driven Molecular Design.*
ChemRxiv 2024 (or the journal version once it appears).
doi:10.26434/chemrxiv-2024-XXXXX (placeholder until the canonical
DOI lands). The MolDQN and REINVENT line originated in:
Olivecrona, M., et al. (2017). *Molecular de-novo design through
deep reinforcement learning.* J. Cheminform. 9, 48.
doi:10.1186/s13321-017-0235-x

Public API
----------
* :class:`REINVENT4Adapter` — implements :class:`molmetal.ports.ScoringFunction`.
* :func:`discover_reinvent4_tomls` — list available TOML scoring presets.
* :func:`is_reinvent4_available` — runtime probe for the cloned repo.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from molmetal.domain import Molecule
from molmetal.ports import (
    PropertyPrediction,
    ScoredCandidate,
    ScoringFunction,
)
from molmetal_lam.sbdd_env.reinvent_wrapper import (
    DEFAULT_SCORER_WEIGHTS,
    REINVENT4Scorer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def is_reinvent4_available(repo_path: str = "molmetal/references/REINVENT4") -> bool:
    """Return True if the REINVENT4 clone is on disk."""
    return os.path.isdir(repo_path)


def discover_reinvent4_tomls(
    repo_path: str = "molmetal/references/REINVENT4",
) -> List[str]:
    """Return a list of TOML preset paths shipped with the cloned repo.

    We do **not** parse them — just enumerate. A real adapter would
    ``tomllib.load`` each file and pick the one that matches the
    user's objective (qed, binding, sa, novelty).
    """
    if not is_reinvent4_available(repo_path):
        return []
    matches: List[str] = []
    for root, _dirs, files in os.walk(repo_path):
        for f in files:
            if f.endswith(".toml"):
                matches.append(os.path.join(root, f))
    return sorted(matches)


# ---------------------------------------------------------------------------
# Reference adapter
# ---------------------------------------------------------------------------
@dataclass
class REINVENT4Adapter:
    """Protocol stub implementing :class:`molmetal.ports.ScoringFunction`.

    Delegates to :class:`REINVENT4Scorer` (RDKit fallback) so the
    pipeline keeps working without the heavy REINVENT4 plugins.
    """

    repo_path: str = "molmetal/references/REINVENT4"
    weights: dict = field(default_factory=lambda: dict(DEFAULT_SCORER_WEIGHTS))
    _scorer: Optional[REINVENT4Scorer] = field(default=None, init=False)

    @property
    def name(self) -> str:
        return "REINVENT4_reference_adapter_v1"

    def setup(self) -> None:
        """Build the underlying RDKit fallback scorer."""
        self._scorer = REINVENT4Scorer(
            reinvent_repo_path=self.repo_path,
            weights=dict(self.weights),
        )
        tomls = discover_reinvent4_tomls(self.repo_path)
        logger.info(
            "REINVENT4Adapter.setup: repo=%s, %d TOML presets discovered, "
            "running in RDKit-fallback mode.",
            self.repo_path,
            len(tomls),
        )

    def is_stub(self) -> bool:
        """Always True: we never load the upstream RL loop or heavy plugins."""
        return True

    def score(
        self,
        candidates: List[Tuple[Molecule, object, PropertyPrediction]],
    ) -> List[ScoredCandidate]:
        """Combine each candidate's properties with the configured weights.

        The dummy ``object`` slot is :class:`molmetal.domain.Complex`
        passed through unchanged.
        """
        if self._scorer is None:
            self.setup()

        assert self._scorer is not None  # for type-checkers
        out: List[ScoredCandidate] = []
        for rank, (mol, cmplx, prop) in enumerate(candidates):
            raw = self._scorer.score(mol.smiles) if hasattr(mol, "smiles") else 0.0
            out.append(
                ScoredCandidate(
                    molecule=mol,
                    complex=cmplx,  # type: ignore[arg-type]
                    property_pred=prop,
                    combined_score=float(raw),
                    rank=rank,
                )
            )
        # Highest combined score first. ``ScoredCandidate`` is frozen, so
        # we replace each item via ``dataclasses.replace``.
        from dataclasses import replace

        out.sort(key=lambda c: c.combined_score, reverse=True)
        return [replace(c, rank=i) for i, c in enumerate(out)]

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "engine": "REINVENT4 (fallback adapter)",
            "engine_version": "RDKit fallback (no plugins loaded)",
            "repo_path": self.repo_path,
            "toml_presets": discover_reinvent4_tomls(self.repo_path),
            "stub": True,
            "paper": "He et al., REINVENT4, ChemRxiv 2024",
        }


# ---------------------------------------------------------------------------
# Protocol conformance hint
# ---------------------------------------------------------------------------
def _assert_protocol_conformance() -> None:
    """Static check that the adapter shape matches ScoringFunction."""
    adapter: ScoringFunction = REINVENT4Adapter()  # type: ignore[assignment]
    _ = adapter.name
    _ = adapter.setup
    _ = adapter.score
    _ = adapter.get_metadata


__all__ = [
    "REINVENT4Adapter",
    "discover_reinvent4_tomls",
    "is_reinvent4_available",
]
