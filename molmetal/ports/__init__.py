"""Abstract ports (Protocol interfaces) for the molmetal framework.

Hexagonal architecture: any concrete implementation of a port can be
swapped without changing the orchestration code.  All ports are
``typing.Protocol`` classes (PEP 544) — they are duck-typed, so adapters
don't need to inherit explicitly.

Five ports in v1:

- :class:`MoleculeGenerator` — de novo molecule generation
- :class:`DockingEngine` — pose prediction
- :class:`PropertyPredictor` — 2D / 3D / biological property prediction
- :class:`ScoringFunction` — multi-objective weighted combination
- :class:`DesignLoop` — orchestrator (generate → dock → score → refine)

Each port comes with a frozen dataclass holding the configuration
inputs (``GenerationConfig``, ``DockingConfig`` …) so that downstream
adapters can read them without ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import torch

from molmetal.domain import Complex, Molecule, Pocket


# ---------------------------------------------------------------------------
# Common config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """Inputs to :meth:`MoleculeGenerator.generate`."""

    n_samples: int = 100
    n_steps: int = 50                # number of ODE integration steps
    temperature: float = 1.0         # noise scale (>1 → more diverse)
    seed: int = 42
    conditioning: dict = field(default_factory=dict)  # pocket embeddings etc.
    # ODE solver method. One of {"euler", "midpoint", "dopri5", "heun3"}.
    # Default flipped "euler" → "midpoint" (Heun's 2nd-order) per
    # WF-CFM-Frontier Phase 2 Fix #2 to reduce O(h) global integration
    # error (~0.01 Å per coord → ~0.1 Å cloud drift over 100 steps for
    # first-order Euler).  Midpoint is 2nd-order, eliminating the
    # cumulative drift while doubling the per-step cost — the decoder
    # bond-cutoff heuristic (2.4 Å) lands more often in the valid range
    # when coords are not drifted away from intended positions.
    # Existing callers passing ``method="euler"`` explicitly remain
    # unchanged; new callers get the better method by default.
    method: str = "midpoint"


@dataclass(frozen=True)
class DockingConfig:
    """Inputs to :meth:`DockingEngine.dock`."""

    n_poses: int = 10                 # top-K poses per (mol, pocket)
    exhaustiveness: int = 8
    use_confidence: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Port 1: MoleculeGenerator
# ---------------------------------------------------------------------------
class MoleculeGenerator(Protocol):
    """De novo molecule generation conditioned on a binding pocket.

    Concrete implementations are *adapters* in ``molmetal.adapters.*``.
    The canonical v1 adapter wraps the Facebook Research
    `flow_matching` library (Lipman et al. 2023, ICLR).
    """

    @property
    def name(self) -> str:
        """Identifier — e.g. ``"LipmanFlowMatching_v1"``."""
        ...

    def setup(self, device: str = "cuda") -> None:
        """Load weights, allocate buffers, compile if needed."""
        ...

    def generate(self, pocket: Pocket, config: GenerationConfig) -> List[Molecule]:
        """Produce ``config.n_samples`` candidate molecules."""
        ...

    def train_step(self, pocket: Pocket, mols: List[Molecule]) -> float:
        """One optimisation step; returns the loss value (for logging)."""
        ...

    def get_metadata(self) -> dict:
        """Reproducibility info: paper, code, dataset, config."""
        ...


# ---------------------------------------------------------------------------
# Port 2: DockingEngine
# ---------------------------------------------------------------------------
class DockingEngine(Protocol):
    """Predict binding pose(s) of a molecule in a pocket."""

    @property
    def name(self) -> str: ...
    def setup(self, device: str = "cuda") -> None: ...
    def dock(
        self,
        molecule: Molecule,
        pocket: Pocket,
        config: DockingConfig,
    ) -> List[Complex]:
        """Return top-N poses (each is a Complex with transformed molecule)."""
        ...
    def get_metadata(self) -> dict: ...


# ---------------------------------------------------------------------------
# Port 3: PropertyPredictor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PropertyPrediction:
    """Outputs of :meth:`PropertyPredictor.predict`."""

    qed: float = 0.0
    sa_score: float = 0.0
    logp: float = 0.0
    mol_weight: float = 0.0
    tpsa: float = 0.0
    num_h_donors: int = 0
    num_h_acceptors: int = 0
    num_rotatable_bonds: int = 0
    binding_affinity_pic50: Optional[float] = None  # from EGNN/D-MPNN
    metal_binding_score: Optional[float] = None     # for our metal extension


@runtime_checkable
class PropertyPredictor(Protocol):
    """Predict 2D / 3D / biological properties of a Molecule (or Complex)."""

    @property
    def name(self) -> str: ...
    def setup(self, device: str = "cuda") -> None: ...
    def predict(
        self,
        molecule: Molecule,
        complex: Optional[Complex] = None,
    ) -> PropertyPrediction: ...
    def get_metadata(self) -> dict: ...

# ---------------------------------------------------------------------------
# Port 4: ScoringFunction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredCandidate:
    """A (molecule, complex, properties) tuple with a combined score + rank."""

    molecule: Molecule
    complex: Optional[Complex]
    property_pred: PropertyPrediction
    combined_score: float
    rank: int = 0


class ScoringFunction(Protocol):
    """Combine multiple signals into a single ranking score."""

    @property
    def name(self) -> str: ...
    def setup(self) -> None: ...
    def score(
        self,
        candidates: List[Tuple[Molecule, Optional[Complex], PropertyPrediction]],
    ) -> List[ScoredCandidate]: ...
    def get_metadata(self) -> dict: ...


# ---------------------------------------------------------------------------
# Port 5: DesignLoop
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DesignLoopConfig:
    """Inputs to :meth:`DesignLoop.run`."""

    n_iterations: int = 3
    n_samples_per_round: int = 1000
    n_top_k: int = 100
    convergence_threshold: float = 0.01
    seed: int = 42


class DesignLoop(Protocol):
    """Orchestrate generate → score → refine cycles."""

    def __init__(
        self,
        generator: MoleculeGenerator,
        docker: DockingEngine,
        predictor: PropertyPredictor,
        scorer: ScoringFunction,
    ): ...

    @property
    def name(self) -> str: ...
    def setup(self, device: str = "cuda") -> None: ...
    def run(
        self, pocket: Pocket, config: DesignLoopConfig
    ) -> List[ScoredCandidate]: ...
    def get_history(self) -> List[dict]: ...


__all__ = [
    "MoleculeGenerator",
    "DockingEngine",
    "PropertyPredictor",
    "ScoringFunction",
    "DesignLoop",
    "GenerationConfig",
    "DockingConfig",
    "PropertyPrediction",
    "ScoredCandidate",
    "DesignLoopConfig",
]
