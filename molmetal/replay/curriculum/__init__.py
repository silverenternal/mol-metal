"""molmetal.replay.curriculum — RSI for curriculum learning (hard-example mining).

Implements the FOUNDATION primitive of TODO-150 P1 (curriculum learning
via hard-example mining replay).  Three reasons this lives next to
``molmetal.replay.pool`` and not under ``molmetal_lam.search_alg``:

* the **operational level is the training loop** (per-sample loss /
  gradient / fingerprint), not the search-tree level;
* the **data flow is per-epoch parquet shards** under
  ``metrics/by_metric/rsi_sample_ledger/<metal>/seed_<n>/epoch_<k>.parquet``
  rather than the AlphaGo ``(s, a, z, pi)`` tuples of MCTS replay;
* the **acquisition policy** (top-K by predicted hardness + Tanimoto
  diversity < 0.4 + 10 % random fallback) is closer to a curriculum
  scheduler than a proof-search expansion.

Honest framing (per party-twenty-spirit §3.5)
==============================================
All scoring functions in this module return predictions of *hardness*
(scalar + 3-way decomposition).  None of these are validated to lift
Rh AUC until the Wf-124 5-seed × 50-epoch protocol reports a
MEASURED Rh AUC delta ≥ -0.01 (must NOT regress).  Until that gate
fires, every number produced here is a **PROJECTION** — call sites
must not promote projections to paper claims.

Module map (one file per primitive — atomic scope)
===================================================
* :mod:`.sample_ledger`  — append-only per-sample parquet writer.
* :mod:`.hardness`       — ``HardnessSample`` + ``HardnessPredictor``
                            (lightweight 2-layer MLP, 64-d) + 3-way
                            hardness decomposition (m_geometry,
                            m_binding, m_reduction).
* :mod:`.validity_gate`  — SMILES round-trip + RDKit sanitization
                            gate (closes the Wf-77 / TODO-13 dream
                            validity contract).
* :mod:`.acquisition`    — per-metal top-K + Tanimoto<0.4 diversity +
                            10 % random fallback.
* :mod:`.drift`          — drift guard (retire injected samples whose
                            on-policy hardness diverges > 0.5 from
                            predicted).
* :mod:`.curriculum`     — ``RSICurriculum`` orchestrator wiring the
                            five primitives together.
"""

from __future__ import annotations

from .sample_ledger import (
    LEDGER_VERSION,
    LEDGER_ROOT_DEFAULT,
    SampleLedger,
    SampleLedgerShard,
    PerSampleRecord,
)
from .hardness import (
    HardnessSample,
    HardnessDecomposition,
    HardnessPredictor,
    PREDICTOR_INPUT_DIM,
    PREDICTOR_HIDDEN_DIM,
    DRIFT_DIVERGENCE_THRESHOLD,
)
from .validity_gate import (
    ValidityGate,
    ValidityResult,
)
from .acquisition import (
    AcquisitionPolicy,
    AcquisitionResult,
    DEFAULT_K_BY_METAL,
    DEFAULT_TANIMOTO_THRESHOLD,
    DEFAULT_RANDOM_FRACTION,
)
from .drift import (
    DriftGuard,
    DriftReport,
)
from .curriculum import (
    RSICurriculum,
    CurriculumConfig,
    CurriculumStepReport,
)

__all__ = [
    "LEDGER_VERSION",
    "LEDGER_ROOT_DEFAULT",
    "SampleLedger",
    "SampleLedgerShard",
    "PerSampleRecord",
    "HardnessSample",
    "HardnessDecomposition",
    "HardnessPredictor",
    "PREDICTOR_INPUT_DIM",
    "PREDICTOR_HIDDEN_DIM",
    "DRIFT_DIVERGENCE_THRESHOLD",
    "ValidityGate",
    "ValidityResult",
    "AcquisitionPolicy",
    "AcquisitionResult",
    "DEFAULT_K_BY_METAL",
    "DEFAULT_TANIMOTO_THRESHOLD",
    "DEFAULT_RANDOM_FRACTION",
    "DriftGuard",
    "DriftReport",
    "RSICurriculum",
    "CurriculumConfig",
    "CurriculumStepReport",
]
