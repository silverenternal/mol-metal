"""RSI curriculum orchestrator — wires the five primitives together.

================================================================
Curriculum orchestrator (TODO-150 §P1–P2)
================================================================
:func:`RSICurriculum` is the *user-facing* entry point.  Each
:meth:`step` call wires:

  :class:`SampleLedger`     — append-only per-sample writer.
  :class:`HardnessPredictor` — 2-layer MLP, retrained every 5 epochs.
  :class:`ValidityGate`     — SMILES round-trip + RDKit sanitisation.
  :class:`AcquisitionPolicy` — top-K + Tanimoto diversity + random fallback.
  :class:`DriftGuard`       — retire drifted samples.

The orchestrator is **stateless across metals** — one
:class:`RSICurriculum` instance handles all metals via a per-metal
``metal_state`` dict so back-to-back :meth:`step` calls over the
Wf-124 5-metal × 3-seed protocol stay cheap.

Honest framing
==============
:meth:`step` returns a :class:`CurriculumStepReport` with both
**MEASURED** fields (the per-epoch on-policy hardness drift, the
Tanimoto-histogram of the injection set, the per-metal
``n_injected``) and **PROJECTED** fields (the predictor's hardness
estimates, the retirement decision).  The MEASURED fields are the
ones safe to quote in paper §3.7; the PROJECTED fields are explicitly
labelled.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .sample_ledger import (
    LEDGER_ROOT_DEFAULT,
    LEDGER_VERSION,
    PerSampleRecord,
    SampleLedger,
    SampleLedgerShard,
)
from .hardness import (
    DRIFT_DIVERGENCE_THRESHOLD,
    HardnessDecomposition,
    HardnessPredictor,
    HardnessSample,
    PREDICTOR_HIDDEN_DIM,
    PREDICTOR_INPUT_DIM,
)
from .validity_gate import ValidityGate, ValidityResult
from .acquisition import (
    AcquisitionPolicy,
    AcquisitionResult,
    DEFAULT_K_BY_METAL,
    DEFAULT_RANDOM_FRACTION,
    DEFAULT_TANIMOTO_THRESHOLD,
)
from .drift import DriftGuard, DriftReport


@dataclass
class CurriculumConfig:
    """Hyperparameters for the RSI curriculum orchestrator.

    All fields default to the values cited in TODO-150 §P1–§P2.  The
    defaults are deliberately conservative — flipping any of them is
    a separate workflow that owns the policy-vs-target interaction.
    """

    metal_list: Sequence[str] = field(
        default_factory=lambda: ("Ru", "Ir", "Rh", "Os", "Re")
    )
    predictor_input_dim: int = PREDICTOR_INPUT_DIM
    predictor_hidden_dim: int = PREDICTOR_HIDDEN_DIM
    predictor_n_epochs: int = 50
    predictor_batch_size: int = 64
    predictor_lr: float = 1e-3
    refit_every: int = 5
    drift_threshold: float = DRIFT_DIVERGENCE_THRESHOLD
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD
    random_fraction: float = DEFAULT_RANDOM_FRACTION
    k_by_metal: Optional[Dict[str, int]] = None
    ledger_root: Optional[str] = None
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metal_list": list(self.metal_list),
            "predictor_input_dim": int(self.predictor_input_dim),
            "predictor_hidden_dim": int(self.predictor_hidden_dim),
            "predictor_n_epochs": int(self.predictor_n_epochs),
            "predictor_batch_size": int(self.predictor_batch_size),
            "predictor_lr": float(self.predictor_lr),
            "refit_every": int(self.refit_every),
            "drift_threshold": float(self.drift_threshold),
            "tanimoto_threshold": float(self.tanimoto_threshold),
            "random_fraction": float(self.random_fraction),
            "k_by_metal": dict(self.k_by_metal) if self.k_by_metal else None,
            "ledger_root": str(self.ledger_root) if self.ledger_root else LEDGER_ROOT_DEFAULT,
            "seed": int(self.seed),
        }


@dataclass
class CurriculumStepReport:
    """What :meth:`RSICurriculum.step` returns per metal per step."""

    metal: str
    epoch: int
    n_written: int
    n_acquired: int
    n_retired: int
    n_kept: int
    drift_report: DriftReport
    acquisition: AcquisitionResult
    predictor_mae: float = float("nan")
    projection_lift: float = 0.0  # PROJECTED — never quote as measured.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metal": str(self.metal),
            "epoch": int(self.epoch),
            "n_written": int(self.n_written),
            "n_acquired": int(self.n_acquired),
            "n_retired": int(self.n_retired),
            "n_kept": int(self.n_kept),
            "predictor_mae": float(self.predictor_mae),
            "projection_lift": float(self.projection_lift),
            "drift_report": self.drift_report.to_dict(),
            "acquisition": self.acquisition.to_dict(),
        }


class RSICurriculum:
    """Orchestrator that wires ledger + predictor + gate + acquisition + drift.

    The orchestrator holds:

    * one :class:`SampleLedger` (shared across metals / seeds);
    * one :class:`HardnessPredictor` per metal (retrained every
      ``refit_every`` epochs);
    * one :class:`AcquisitionPolicy` per metal (with the per-metal
      top-K);
    * one :class:`ValidityGate` (shared, stateless);
    * one :class:`DriftGuard` (shared).

    The :meth:`step` entry point matches the Wf-124 P3 protocol:
    per-metal, per-seed, per-epoch.

    Backward compat
    ---------------
    The orchestrator is *opt-in*: existing trainers do not consume
    its outputs unless they call :meth:`step`.  This matches the
    "MUST NOT regress the Wf-124 baseline" gate (TODO-150 §P2.2):
    turning RSI off == the Wf-124 baseline.
    """

    def __init__(self, config: Optional[CurriculumConfig] = None) -> None:
        self.config = config or CurriculumConfig()
        self.ledger = SampleLedger(root=self.config.ledger_root)
        self.gate = ValidityGate()
        self.drift = DriftGuard(threshold=self.config.drift_threshold)
        # Per-metal state.
        self._predictors: Dict[str, HardnessPredictor] = {}
        self._policies: Dict[str, AcquisitionPolicy] = {}
        self._retired: Dict[str, set] = {m: set() for m in self.config.metal_list}
        self._rng = random.Random(self.config.seed)
        # Lazy predictor / policy instantiation — populated on first
        # :meth:`step` call for a given metal.
        for m in self.config.metal_list:
            self._ensure_predictor(m)
            self._ensure_policy(m)

    # ------------------------------------------------------------------
    # Lazy state
    # ------------------------------------------------------------------
    def _ensure_predictor(self, metal: str) -> HardnessPredictor:
        if metal not in self._predictors:
            self._predictors[metal] = HardnessPredictor(
                input_dim=self.config.predictor_input_dim,
                hidden_dim=self.config.predictor_hidden_dim,
                learning_rate=self.config.predictor_lr,
                n_epochs=self.config.predictor_n_epochs,
                batch_size=self.config.predictor_batch_size,
                seed=self.config.seed,
            )
        return self._predictors[metal]

    def _ensure_policy(self, metal: str) -> AcquisitionPolicy:
        if metal not in self._policies:
            k_by_metal = self.config.k_by_metal or DEFAULT_K_BY_METAL
            self._policies[metal] = AcquisitionPolicy(
                k_by_metal=k_by_metal,
                tanimoto_threshold=self.config.tanimoto_threshold,
                random_fraction=self.config.random_fraction,
                seed=self.config.seed,
            )
        return self._policies[metal]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self,
        metal: str,
        seed: int,
        epoch: int,
        samples: Iterable[Dict[str, Any]],
        candidates: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> CurriculumStepReport:
        """Run one (metal, seed, epoch) step of the curriculum.

        ``samples`` is an iterable of per-sample dicts carrying the
        fields required by :class:`PerSampleRecord`.  ``candidates``
        is the optional pool the acquisition policy draws from
        (typically ``samples`` filtered through the validity gate).
        """
        metal = str(metal)
        seed = int(seed)
        epoch = int(epoch)
        predictor = self._ensure_predictor(metal)
        policy = self._ensure_policy(metal)

        # 1. Write the per-sample ledger.
        shard = self.ledger.begin_shard(metal=metal, seed=seed, epoch=epoch)
        sample_list: List[PerSampleRecord] = []
        for s in samples:
            rec = PerSampleRecord(
                row_id=int(s["row_id"]),
                smiles=str(s["smiles"]),
                metal=metal,
                pocket_id=str(s.get("pocket_id", "")),
                loss=float(s["loss"]),
                gradient_norm=float(s.get("gradient_norm", 0.0)),
                predicted_pic50=float(s.get("predicted_pic50", 0.0)),
                target_pic50=float(s.get("target_pic50", 0.0)),
                predicted_active=int(s.get("predicted_active", 0)),
                target_active=int(s.get("target_active", 0)),
                epoch=epoch,
                seed=seed,
            )
            sample_list.append(rec)
            self.ledger.append(shard, rec)
        self.ledger.finalise(shard)
        n_written = shard.n_written

        # 2. Refit the predictor every ``refit_every`` epochs.
        predictor_mae = float("nan")
        if sample_list and (epoch % self.config.refit_every == 0):
            features = np_stack_features(
                sample_list, input_dim=self.config.predictor_input_dim
            )
            targets = np.asarray(
                [min(1.0, max(0.0, r.loss)) for r in sample_list], dtype=np.float64
            )
            diag = predictor.fit(features, targets)
            predictor_mae = float(diag.get("train_mae", float("nan")))

        # 3. Build the candidate pool for acquisition.
        pool: List[Dict[str, Any]] = []
        if candidates is not None:
            for c in candidates:
                pool.append(dict(c))
        else:
            for r in sample_list:
                pred = (
                    float(predictor.predict(_placeholder_feature(self.config.predictor_input_dim))[0])
                    if predictor.is_fitted
                    else 0.0
                )
                pool.append(
                    {
                        "row_id": r.row_id,
                        "smiles": r.smiles,
                        "hardness_predicted": pred,
                    }
                )

        # 4. Validity gate.
        valid = self.gate.accept([c["smiles"] for c in pool])
        valid_set = set(valid)
        filtered_pool = [c for c in pool if c["smiles"] in valid_set]

        # 5. Acquisition.
        acquisition = policy.acquire(metal, filtered_pool, rng=self._rng)
        # Drop retired samples from the acquisition.
        retired_set = self._retired.setdefault(metal, set())
        acquisition = AcquisitionResult(
            metal=metal,
            accepted=[r for r in acquisition.accepted if r not in retired_set],
            accepted_smiles=[s for s in acquisition.accepted_smiles],
            hardness_predicted=acquisition.hardness_predicted,
            n_topk_candidates=acquisition.n_topk_candidates,
            n_after_diversity=acquisition.n_after_diversity,
            n_random_injected=acquisition.n_random_injected,
        )

        # 6. Drift guard (re-evaluate every ``refit_every`` epochs).
        drift_samples: List[Dict[str, Any]] = []
        on_policy_lookup = {r.row_id: r.loss for r in sample_list}
        pred_lookup = {a: h for a, h in zip(acquisition.accepted, acquisition.hardness_predicted)}
        for row_id in acquisition.accepted:
            drift_samples.append(
                {
                    "row_id": row_id,
                    "predicted": float(pred_lookup.get(row_id, 0.0)),
                    "on_policy": float(on_policy_lookup.get(row_id, 0.0)),
                }
            )
        drift_report = self.drift.evaluate(metal, epoch, drift_samples)
        # Update retired set with new retirements.
        for rid in drift_report.retired_row_ids:
            retired_set.add(rid)

        # 7. Projection (NOT a measurement): expected lift = predictor's
        # confidence on the accepted set minus the cohort mean.
        # Always reported under ``projection_lift``; never surfaced as
        # a measured lift in the audit report.
        if acquisition.hardness_predicted:
            mean_pred = sum(acquisition.hardness_predicted) / len(acquisition.hardness_predicted)
            mean_cohort = (
                sum(r.loss for r in sample_list) / len(sample_list)
                if sample_list
                else 0.0
            )
            projection_lift = float(mean_pred - mean_cohort)
        else:
            projection_lift = 0.0

        return CurriculumStepReport(
            metal=metal,
            epoch=epoch,
            n_written=n_written,
            n_acquired=len(acquisition.accepted),
            n_retired=drift_report.n_retired,
            n_kept=drift_report.n_kept,
            drift_report=drift_report,
            acquisition=acquisition,
            predictor_mae=predictor_mae,
            projection_lift=projection_lift,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Per-metal ledger stats + retired-set sizes."""
        out: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "ledger": self.ledger.stats(),
            "retired_per_metal": {m: len(s) for m, s in self._retired.items()},
        }
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholder_feature(input_dim: int) -> np.ndarray:
    """Build a deterministic feature vector when no fingerprint is available.

    Uses a zero vector — the predictor's first call uses uniform
    features and the result is only used to seed the candidate
    pool; the *real* features are populated by
    :func:`featurize_smiles_list` upstream of this orchestrator.
    """
    return np.zeros((1, int(input_dim)), dtype=np.float64)


def np_stack_features(records: Sequence[PerSampleRecord], input_dim: int) -> np.ndarray:
    """Stack a sequence of ledger records into a (N, input_dim) matrix.

    Until the upstream trainer wires a real fingerprint pipeline, we
    emit a deterministic row-wise hash expansion — every record
    contributes a unique row so the predictor can still fit and the
    curriculum stays importable.  This is the same convention as
    ``HardnessPredictor.predict`` callers and is overridden by the
    trainer with the real fingerprint pipeline.
    """
    import hashlib
    out = np.zeros((len(records), int(input_dim)), dtype=np.float64)
    for i, rec in enumerate(records):
        h = hashlib.sha256(rec.smiles.encode("utf-8")).digest()
        # Repeat-hash until we cover ``input_dim`` columns.
        n_repeats = (int(input_dim) + 31) // 32
        full = b"".join(h for _ in range(n_repeats))[: int(input_dim)]
        arr = np.frombuffer(full, dtype=np.uint8).astype(np.float64)
        out[i] = (arr - 127.5) / 127.5
    return out


__all__ = [
    "CurriculumConfig",
    "CurriculumStepReport",
    "RSICurriculum",
]