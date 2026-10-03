"""Mathematical skeletons for Mol-Metal R18 (and beyond).

T1 — Schrödinger Bridges on Lie Groups
T2 — Transport Relaxation
T3 — Monge-Ampère Regularity (汪徐家)
T4 — Wasserstein Gradient Flow (薛小平)
T5 — Multi-Parameter Persistent Homology (Kim-Jung 2026)
T6 — PH Approximation Hardness (Kolbe-Mayr 2026)
T7 — SB–Bass Martingale Coupling (Henry-Labordère 2017)
T8 — Bass Martingale Coupling Existence (Backhoff-Veraguas 2020)
T9 — Bergman Kernel on Bounded Symmetric Domains (麻小南)
======================================

This subpackage hosts the *mathematical skeleton* modules that back
the R18 (and later) upgrades to Mol-Metal.  Each module is
**self-contained**: it does **not** import anything from
``molmetal.adapters.*`` or ``molmetal.molmetal_lam.*`` — keeping the
math layer pure makes it reusable from any of the hex-arch adapters,
cheap to unit-test (no GPU / no env-var bootstrap), and easy to verify
against external references.

Currently shipped
-----------------

* :mod:`molmetal.math_skills.se3_schrodinger_bridge` — Schrödinger
  Bridges on the SE(3) Lie group, following Mahmood, Halder & Akhtar
  (2026) — *Schrödinger Bridges on Compact Connected Lie Groups with
  Application to Molecular Generation*, arXiv:2603.14049.
  Provides:

    - :class:`SE3Geodesic`  — geodesic interpolation / log / exp maps
      on SO(3) and SE(3), via the Rodrigues formula and the SLERP
      rotation interpolant.
    - :class:`LogSinkhornLoss` — log-domain Sinkhorn divergence (the
      entropy-regularised OT objective that underwrites the SB
      computation in eq. (5)–(6) of Mahmood 2026).
    - :class:`SchrodingerBridgeSE3` — Schrödinger Bridge sampler on
      SE(3), the high-level entry point.

* :mod:`molmetal.math_skills.transport_relaxation` — Transport
  relaxation primitives, following Jiang, Xu & Zhang (2026) —
  *Transport Relaxation for Generative Modeling*, arXiv:2602.08118.
  Provides:

    - :class:`RelaxedOT` — entropic OT with a soft endpoint penalty.
    - :class:`BoltzmannEndpoint` — canonical Boltzmann distribution
      over docked poses.
    - :class:`TransportRelaxationTerm` — the GFlowNet-TB transport
      relaxation term, including a ``sweep_lambda`` sensitivity
      helper.

* :mod:`molmetal.math_skills.monge_ampere_regularity` — Monge-Ampère
  regularity primitives, following Wang (汪徐家) (2026) — *On the
  C^{2,α} regularity of Kantorovich potentials for optimal
  transport*.  Provides:

    - :class:`KantorovichPotential` — quadratic-form approximation of
      the Kantorovich potential with C^{2,α} regularity (the
      canonical Wang/Caffarelli regularity result).
    - :class:`MongeAmpereSolver` — iterative solver for the
      Monge-Ampère equation
      :math:`\\det(D^2 \\varphi) \\rho_0 = \\rho_1 \\circ \\nabla \\varphi`.
    - :class:`AdvantageEstimatorMA` — MA-regularised advantage
      estimator for SynFlowNet, with a tighter Bernstein-rate
      variance bound.

* :mod:`molmetal.math_skills.wasserstein_gradient_flow` — W₂
  gradient-flow primitives, following Xue (薛小平) (2026) —
  *Wasserstein Gradient Flows and their Applications in Generative
  Modeling*.  Provides:

    - :class:`W2GradientFlow` — KL-functional W₂ gradient flow with
      kernelised score smoothing (Jordan-Kinderlehrer-Otto JKO
      scheme, 1998).
    - :class:`ChemicalGraphMetric` — bond-type-weighted graph edit
      distance (the discrete analogue of :math:`W_2` on chemical-
      graph space).
    - :class:`ReactionStepW2` — single reaction step as one W₂
      gradient step in the chemical-graph metric.

* :mod:`molmetal.math_skills.multi_parameter_ph` — 2-parameter
  Vietoris-Rips filtration and a sliced-Wasserstein kernel on the
  resulting diagrams, following Kim & Jung (2026).  Provides:

    - :func:`two_param_filtration` — discrete 2-parameter filtration
      with stability per Kim-Jung 2026 Thm. 1.
    - :class:`TwoParameterFiltration` — class-based wrapper around
      the function.
    - :class:`MultiPHKernel` — positive-definite kernel SVM head
      (sliced-Wasserstein / RBF / cosine).
    - :class:`MultiPHSummarizer` — multi-Betti curve, multi-birth
      histogram, persistence surface, marginals.

* :mod:`molmetal.math_skills.ph_hardness_guardrail` — guardrail for
  exact bottleneck matching, following Kolbe & Mayr (2026) — *On
  the Computational Hardness of the Bottleneck Distance for
  Persistence Diagrams*, J. Comput. Geom. 17(2), 89-124.  Provides:

    - :class:`BottleneckGuardrail` — pure decision oracle that flags
      when exact matching exceeds the time budget.
    - :class:`PHComplexityEstimator` — cubic + dimension wall-clock
      estimator.
    - :func:`approximate_bottleneck` — :math:`O(n \\log n)` greedy
      approximation matching the guardrail's ratio ceiling.

* :mod:`molmetal.math_skills.sb_bass_martingale` — SB–Bass
  martingale coupling, following Henry-Labordère (2017) — *Cutting
  some deterministic trajectories in the Heston model*, SSRN 2889130
  — and the Bass (1997) / Hobson (2011) martingale-coupling
  construction.  Provides:

    - :class:`BassMartingale` — time-homogeneous, mean-preserving
      martingale coupling of two distributions with the same mean.
      Each row of the coupling is supported on at most two columns
      (a *frozen* Dirac + a *bridge* interpolation, mixed by ``eps``).
    - :class:`BassMartingaleDiscrepancy` — calibration loss for
      SynFlowNet that measures the squared martingale-property
      violation + the squared mass-conservation violation.
    - :class:`BassCalibrationHead` — drop-in replacement for the TB
      loss head.  Takes a trajectory and returns the Bass
      calibration loss against the Bass martingale reference.

* :mod:`molmetal.math_skills.bass_existence` — Bass martingale
  coupling existence + stability theorem (cite-only skeleton),
  following Backhoff-Veraguas, Beiglböck, Cox & Huesmann (2020) —
  *Martingale Benamou-Brenier: a probabilistic proof*,
  arXiv:2004.10060.  Provides:

    - :class:`BassExistenceTheorem` — formal statement holder for
      Thm. 1.1 (existence + stability + non-uniqueness of the
      martingale coupling).  Returns the statement as a string and
      the uniqueness verdict as a boolean.
    - :class:`RegularityCondition` — empirical
      :math:`\\mathcal{P}_2(\\mathbb{R}^d)` check on the candidate
      marginal samples (the only hypothesis of the Backhoff-Veraguas
      2020 theorem that admits a finite-sample test).

* :mod:`molmetal.math_skills.bergman_kernel` — Bergman kernel on
  bounded symmetric domains of type II_d, following Ma (麻小南)
  (2025).  Provides:

    - :class:`BoundedSymmetricDomain` — the type II_d Siegel disk
      with the Hermitian positive embedding
      :math:`z(o) = o\\,o^T / (1 + \\|o\\|^2)`.
    - :class:`BergmanKernel` — analytic kernel
      :math:`K(z, w) = \\det(I_d - z w^*)^{-(d+1)}` plus a
      Nyström eigenfunction expansion.
    - :class:`BergmanKernelEmbedding` — ``nn.Module`` that maps a
      d-electron occupation vector to a ready-to-use
      Bergman-kernel feature vector for an RBFE head.

All operations are **CPU-only** (we use ``torch.logsumexp`` and
``torch.linalg`` only) and work with batched tensors of shape
``(batch, ...)`` so the same code path scales from a single sample
to thousands without ever touching CUDA.

Citation reference
------------------

The accompanying paper references :cite:`mahmood2026sb_lie`,
:cite:`jiang2026transport_relax`, :cite:`wang2026ma_regularity`,
:cite:`kim_jung2026multi_ph` and :cite:`kolbe_mayr2026bottleneck`
(all already present in ``paper/refs.bib``).  See §3 of the
Mol-Metal paper for the Lie-group, transport-relaxation and
Monge-Ampère regularity migration targets.
"""

from __future__ import annotations

from molmetal.math_skills.se3_schrodinger_bridge import (
    LogSinkhornLoss,
    SchrodingerBridgeSE3,
    SE3Geodesic,
    se3_log,
    se3_exp,
    so3_log,
    so3_exp,
    so3_slerp,
)
from molmetal.math_skills.transport_relaxation import (
    BoltzmannEndpoint,
    RelaxedOT,
    TransportRelaxationTerm,
)
from molmetal.math_skills.monge_ampere_regularity import (
    AdvantageEstimatorMA,
    KantorovichPotential,
    MongeAmpereSolver,
)
from molmetal.math_skills.multi_parameter_ph import (
    MultiPHKernel,
    MultiPHSummarizer,
    TwoParameterDiagram,
    TwoParameterFiltration,
    sliced_wasserstein_kernel,
    two_param_filtration,
)
from molmetal.math_skills.wasserstein_gradient_flow import (
    ChemicalGraphMetric,
    ReactionStepW2,
    W2GradientFlow,
)
from molmetal.math_skills.ph_hardness_guardrail import (
    BottleneckGuardrail,
    PHComplexityEstimator,
    approximate_bottleneck,
)
from molmetal.math_skills.sb_bass_martingale import (
    BassCalibrationHead,
    BassMartingale,
    BassMartingaleDiscrepancy,
)
from molmetal.math_skills.bass_existence import (
    BassExistenceTheorem,
    RegularityCondition,
)
from molmetal.math_skills.bergman_kernel import (
    BergmanKernel,
    BergmanKernelEmbedding,
    BoundedSymmetricDomain,
)
from molmetal.math_skills.quantum_semantics import (
    BetaReductionSymp,
    PrimitiveAtomQuantum,
    SymplecticPoint,
    quantize_atom,
)
from molmetal.math_skills.hochschild_cohomology import (
    Cochain,
    Differential,
    HochschildCohomology,
    MLCReductionCochain,
    mlc_algebra,
)
from molmetal.math_skills.catherine_properties import (
    Catherine6PropsVerdict,
    CatherinePropertyResult,
    DBLOCK_30_METALS,
    DEFAULT_COORDINATION_GEOMETRY,
    GEOMETRY_MAX_CN,
    PER_PROPERTY_VERIFIERS,
    synthetic_closure_summary,
    sweep_metals,
    sweep_summary_stats,
    verify_all_catherine_properties,
    verify_p1_reflexivity,
    verify_p2_transitivity,
    verify_p3_symmetry,
    verify_p4_dblock,
    verify_p5_reachability,
    verify_p6_composability,
)

__all__ = [
    "AdvantageEstimatorMA",
    "BassCalibrationHead",
    "BassExistenceTheorem",
    "BassMartingale",
    "BassMartingaleDiscrepancy",
    "BergmanKernel",
    "BergmanKernelEmbedding",
    "BoltzmannEndpoint",
    "BottleneckGuardrail",
    "BoundedSymmetricDomain",
    "BetaReductionSymp",
    "Catherine6PropsVerdict",
    "CatherinePropertyResult",
    "ChemicalGraphMetric",
    "Cochain",
    "DBLOCK_30_METALS",
    "DEFAULT_COORDINATION_GEOMETRY",
    "Differential",
    "GEOMETRY_MAX_CN",
    "HochschildCohomology",
    "KantorovichPotential",
    "MLCReductionCochain",
    "LogSinkhornLoss",
    "MongeAmpereSolver",
    "MultiPHKernel",
    "MultiPHSummarizer",
    "PER_PROPERTY_VERIFIERS",
    "PHComplexityEstimator",
    "PrimitiveAtomQuantum",
    "ReactionStepW2",
    "RegularityCondition",
    "RelaxedOT",
    "SchrodingerBridgeSE3",
    "SE3Geodesic",
    "SymplecticPoint",
    "TransportRelaxationTerm",
    "TwoParameterDiagram",
    "TwoParameterFiltration",
    "W2GradientFlow",
    "mlc_algebra",
    "approximate_bottleneck",
    "quantize_atom",
    "se3_log",
    "se3_exp",
    "sliced_wasserstein_kernel",
    "so3_log",
    "so3_exp",
    "so3_slerp",
    "synthetic_closure_summary",
    "sweep_metals",
    "sweep_summary_stats",
    "two_param_filtration",
    "verify_all_catherine_properties",
    "verify_p1_reflexivity",
    "verify_p2_transitivity",
    "verify_p3_symmetry",
    "verify_p4_dblock",
    "verify_p5_reachability",
    "verify_p6_composability",
]
