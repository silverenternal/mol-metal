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

from molmetal.domain import Complex, Molecule, Pocket


# ---------------------------------------------------------------------------
# Common config dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """Inputs to :meth:`MoleculeGenerator.generate`."""

    #: Canonical ODE solver method vocabulary.  Wf-A52 W1 — exposed as
    #: a module-level constant so callers (and the dispatch entry in
    #: :mod:`molmetal.adapters.flow_matching_lipman`) can validate
    #: ``config.method`` against the canonical set without duplicating
    #: the literal list.  See :func:`validate_method`.
    #:
    #: Wf-A57 W9-P1 (2026-10-16) — union with the dispatch whitelist in
    #: :mod:`molmetal.adapters.flow_matching_lipman`.  ``rk4`` is the
    #: trainer-side recommended method per the Lipman reference paper;
    #: ``dopri5`` and ``heun3`` remain in the set because they are the
    #: canonical GenerationConfig surface referenced in
    #: ``se3_sb_integration.py`` docstring (line 469), even when the
    #: live reference solver / w_flow only dispatch euler / heun /
    #: midpoint / rk4.  This is the asymmetry the Wf-A56 honest-negative
    #: catalogue recommended resolving with Option A (union set).
    VALID_METHODS: frozenset = frozenset(
        {"euler", "midpoint", "dopri5", "heun3", "rk4"}
    )

    n_samples: int = 100
    n_steps: int = 50  # number of ODE integration steps
    temperature: float = 1.0  # noise scale (>1 → more diverse)
    seed: int = 42
    conditioning: dict = field(default_factory=dict)  # pocket embeddings etc.
    # Wf-A75 (2026-10-18) — bond-length guidance weight applied at every
    # ODE step.  0.0 (default) is a no-op (back-compat with the pre-Wf-A75
    # sampler).  Per MolDiff (Vignac et al. 2023, arXiv:2305.07508),
    # Cartesian-coordinate diffusion on metal complexes suffers from
    # the "atom-bond inconsistency" failure mode: bonded pairs drift
    # past ~3 Å during denoising, fragmenting the output.  The
    # guidance is a Hooke-spring force on (Z_i, Z_j) pairs that pulls
    # actual distance toward an expected per-pair value (metal-donor
    # ~2.05 Å, C-C ~1.50 Å, C-N/O ~1.40 Å, etc.).  At weight 0.1 the
    # guidance is ~10% of the EGNN velocity at full magnitude.
    bond_length_guidance_weight: float = 0.0
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
    # Wf-A35-2 (D2): nucleus-sampling hyperparameters for atom-type
    # decoding.  Defaults preserve the pre-A35-2 bit-exact path
    # (``top_p=1.0`` keeps all tokens, ``top_k=0`` disables top-k
    # truncation, ``temperature=1.0`` does not rescale logits).  At
    # any non-default value the adapter routes the atom-type
    # multinomial through ``diversity_sampling.nucleus_sample``
    # (Holtzman 2020); see ``last_sampling_diagnostics`` for the
    # opt-in audit counter.
    top_p: float = 1.0
    top_k: int = 0
    # Wf-A35-2 (D1): flow-time distribution flag for the ODE time
    # grid.  ``"uniform"`` (default) keeps the legacy ``torch.linspace``
    # grid; ``"log_normal"`` draws ``n_steps+1`` times via
    # ``log_normal_t_sampling`` (Karras 2022 §3.4: t = sigmoid(N(mu, sigma)))
    # concentrating t in the mid-range where FM regression is hardest.
    fm_t_dist: str = "uniform"
    fm_log_normal_mu: float = -0.5
    fm_log_normal_sigma: float = 1.0

    def __post_init__(self) -> None:
        # Wf-A52 W1 — fail loud on unknown ODE method at config
        # construction time.  Before this, a misspelled
        # ``method="rk4"`` silently fell through to the default path in
        # :class:`flow_matching_lipman.LipmanFlowMatchingAdapter` and
        # produced numerically meaningless drift without any error trace.
        if self.method not in self.VALID_METHODS:
            raise ValueError(
                f"GenerationConfig.method must be one of "
                f"{sorted(self.VALID_METHODS)}; got {self.method!r}"
            )


def validate_method(method: str) -> None:
    """Raise ``ValueError`` if ``method`` is not in VALID_METHODS.

    Wf-A52 W1 — convenience export so adapters can call
    ``validate_method(config.method)`` at dispatch entry without
    duplicating the literal set.  See :class:`GenerationConfig`.
    """
    if method not in GenerationConfig.VALID_METHODS:
        raise ValueError(
            f"ODE method must be one of "
            f"{sorted(GenerationConfig.VALID_METHODS)}; got {method!r}"
        )


@dataclass(frozen=True)
class DockingConfig:
    """Inputs to :meth:`DockingEngine.dock`."""

    n_poses: int = 10  # top-K poses per (mol, pocket)
    exhaustiveness: int = 8
    use_confidence: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        # Wf-A52 W2 — fail loud on zero/negative integer configs at
        # construction time. Before this, ``n_poses=0`` or
        # ``exhaustiveness=-1`` silently produced empty pose lists
        # without any error trace, masking the config typo until
        # downstream scoring reported zero candidates.
        if self.n_poses < 1:
            raise ValueError(
                f"DockingConfig.n_poses must be >= 1; got {self.n_poses}"
            )
        if self.exhaustiveness < 1:
            raise ValueError(
                f"DockingConfig.exhaustiveness must be >= 1; got {self.exhaustiveness}"
            )


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
    metal_binding_score: Optional[float] = None  # for our metal extension


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

    def __post_init__(self) -> None:
        # Wf-A52 W2 — fail loud on zero/negative iteration counts at
        # construction time. Before this, ``n_iterations=0`` short-
        # circuited the loop to zero candidates, masking the config
        # typo until downstream scoring reported zero rank rows.
        if self.n_iterations < 1:
            raise ValueError(
                f"DesignLoopConfig.n_iterations must be >= 1; got {self.n_iterations}"
            )
        if self.n_samples_per_round < 1:
            raise ValueError(
                "DesignLoopConfig.n_samples_per_round must be >= 1; "
                f"got {self.n_samples_per_round}"
            )
        if self.n_top_k < 1:
            raise ValueError(
                f"DesignLoopConfig.n_top_k must be >= 1; got {self.n_top_k}"
            )
        if self.convergence_threshold < 0.0:
            raise ValueError(
                f"DesignLoopConfig.convergence_threshold must be >= 0.0; "
                f"got {self.convergence_threshold}"
            )


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
