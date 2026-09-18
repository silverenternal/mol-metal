"""SOTA-aligned run config — TargetDiff protocol as a frozen dataclass.

This module encodes the canonical SOTA-aligned experimental config
(TargetDiff / CrossDocked100 protocol — Guan et al. 2023) as a frozen
dataclass that mirrors the YAML schema at
``molmetal/configs/sota_aligned_targetdiff.yaml``.

The dataclass is intentionally *frozen* so accidental mutation from
a sweep harness can never silently change the run protocol.  Use
:func:`dataclasses.replace` to derive per-pocket overrides.

Public API
----------
- :class:`SOTAAlignedConfig`     — frozen root dataclass
- :func:`load_sota_aligned_config` — YAML -> :class:`SOTAAlignedConfig`
- :func:`to_mcts_kwargs`           — config -> kwargs for
  :class:`molmetal_lam.search_alg.proof_search.MCTSProofSearch`

Cite-only path rationale (TODO/pending/decisions.md D1):
The numbers in the YAML are CITED from TargetDiff (Guan 2023); we do
not re-run DiffDock / TargetDiff ourselves on this hardware.  Strict
head-to-head is gated on DiffSBDD ckpt + torch_geometric ROCm 7.2
wheels (TODO/pending/risks.md R1).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Nested config dataclasses — mirror the YAML schema 1:1
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DockingConfig:
    """Vina docking knobs."""

    engine: str = "vina"
    version: str = "1.2.7"
    exhaustiveness: int = 8
    n_poses: int = 9


@dataclass(frozen=True)
class ScoringConfig:
    """Property scoring mix — SA/QED/Lipinski."""

    sa: str = "ertl_2000"
    qed: str = "rdkit"
    lipinski: bool = True
    logp_threshold: float = 5.0


@dataclass(frozen=True)
class SuccessRateConfig:
    """Thresholds for declaring a ligand "successful" (SOTA-paper def)."""

    vina_threshold_kcal: float = -8.0
    sa_threshold: float = 4.0
    qed_threshold: float = 0.5
    logp_max: float = 5.0


@dataclass(frozen=True)
class NFEConfig:
    """Number-of-Function-Evaluations budget per pocket."""

    target_per_pocket: int = 1000
    method: str = "mcts_rollouts"


@dataclass(frozen=True)
class PoseBustersConfig:
    """PoseBusters validity gate (SOTA 2024+ validity table)."""

    enabled: bool = True
    threshold: str = "pass_all"


@dataclass(frozen=True)
class MCTSConfig:
    """MCTS knobs — calibrated from round-0 sweep observations.

    Notes
    -----
    - ``top_k=100`` lifts per-pocket candidates from ~1.7 toward SOTA
      bands (50-100 cands/pocket per TransDiff/MolCRAFT/Pocket2Mol).
    - ``max_depth=3`` is the round-0 sweet spot (depth=2 throttles
      diversity; depth=4+ blows up compute).
    - ``patience=50`` + ``early_stop=True`` — round-0 showed UCB
      plateau by conv_iter 2-7 at branching=12 / 16-37 at
      branching=1020; the rest of the budget is compute burned.
    - ``branching_target=1020`` is the Phase-1 ceiling (204 tiles x
      5 click rules) — the dominant quality signal per round-0
      ablation (12/60 -> 1.7046 vs 1020 -> 1.9338, +13.5%).
    """

    top_k: int = 100
    max_depth: int = 3
    patience: int = 50
    early_stop: bool = True
    branching_target: int = 1020


@dataclass(frozen=True)
class LambdaSpecificConfig:
    """Lambda-only levers — explicit surface of what makes Lambda Lambda.

    These are NOT SOTA-comparable; they are Mol-Metal advantages over
    plain flow-matching baselines and the sweep markdown labels them
    as Lambda-only contributions.
    """

    tile_library: str = "extended_204"
    click_rules: str = "all_5"
    synthesis_oracle: bool = True
    symbolic_prior: bool = True
    symbolic_prior_refit_every: int = 5


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SOTAAlignedConfig:
    """Canonical SOTA-aligned run config (TargetDiff protocol).

    Fields mirror the YAML at
    ``molmetal/configs/sota_aligned_targetdiff.yaml`` 1:1.
    """

    test_set: str = "crossdocked_pocket10"
    n_test_pockets: int = 100
    docking: DockingConfig = field(default_factory=DockingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    success_rate: SuccessRateConfig = field(default_factory=SuccessRateConfig)
    nfe: NFEConfig = field(default_factory=NFEConfig)
    posebusters: PoseBustersConfig = field(default_factory=PoseBustersConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    lambda_specific: LambdaSpecificConfig = field(default_factory=LambdaSpecificConfig)

    def __getattr__(self, name: str) -> Any:
        # Allow ergonomic ``config.to_mcts_kwargs()`` calls in addition
        # to the module-level :func:`to_mcts_kwargs(config)` form.
        # ``__getattr__`` is only consulted when normal attribute lookup
        # fails, so real fields take precedence.
        if name == "to_mcts_kwargs":
            return lambda: to_mcts_kwargs(self)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def _build_docking(d: Dict[str, Any]) -> DockingConfig:
    return DockingConfig(
        engine=d.get("engine", "vina"),
        version=str(d.get("version", "1.2.7")),
        exhaustiveness=int(d.get("exhaustiveness", 8)),
        n_poses=int(d.get("n_poses", 9)),
    )


def _build_scoring(d: Dict[str, Any]) -> ScoringConfig:
    return ScoringConfig(
        sa=d.get("sa", "ertl_2000"),
        qed=d.get("qed", "rdkit"),
        lipinski=bool(d.get("lipinski", True)),
        logp_threshold=float(d.get("logp_threshold", 5.0)),
    )


def _build_success_rate(d: Dict[str, Any]) -> SuccessRateConfig:
    return SuccessRateConfig(
        vina_threshold_kcal=float(d.get("vina_threshold_kcal", -8.0)),
        sa_threshold=float(d.get("sa_threshold", 4.0)),
        qed_threshold=float(d.get("qed_threshold", 0.5)),
        logp_max=float(d.get("logp_max", 5.0)),
    )


def _build_nfe(d: Dict[str, Any]) -> NFEConfig:
    return NFEConfig(
        target_per_pocket=int(d.get("target_per_pocket", 1000)),
        method=d.get("method", "mcts_rollouts"),
    )


def _build_posebusters(d: Dict[str, Any]) -> PoseBustersConfig:
    return PoseBustersConfig(
        enabled=bool(d.get("enabled", True)),
        threshold=d.get("threshold", "pass_all"),
    )


def _build_mcts(d: Dict[str, Any]) -> MCTSConfig:
    return MCTSConfig(
        top_k=int(d.get("top_k", 100)),
        max_depth=int(d.get("max_depth", 3)),
        patience=int(d.get("patience", 50)),
        early_stop=bool(d.get("early_stop", True)),
        branching_target=int(d.get("branching_target", 1020)),
    )


def _build_lambda_specific(d: Dict[str, Any]) -> LambdaSpecificConfig:
    return LambdaSpecificConfig(
        tile_library=d.get("tile_library", "extended_204"),
        click_rules=d.get("click_rules", "all_5"),
        synthesis_oracle=bool(d.get("synthesis_oracle", True)),
        symbolic_prior=bool(d.get("symbolic_prior", True)),
        symbolic_prior_refit_every=int(d.get("symbolic_prior_refit_every", 5)),
    )


def load_sota_aligned_config(path: str) -> SOTAAlignedConfig:
    """Load :class:`SOTAAlignedConfig` from a YAML file.

    Parameters
    ----------
    path : str
        Path to a YAML file matching the
        ``molmetal/configs/sota_aligned_targetdiff.yaml`` schema.
        Missing sections fall back to dataclass defaults.

    Returns
    -------
    SOTAAlignedConfig
        Frozen dataclass with all sub-configs populated.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SOTA-aligned config not found: {path}")
    with p.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"SOTA-aligned config root must be a mapping; got {type(raw).__name__}")
    return SOTAAlignedConfig(
        test_set=raw.get("test_set", "crossdocked_pocket10"),
        n_test_pockets=int(raw.get("n_test_pockets", 100)),
        docking=_build_docking(raw.get("docking") or {}),
        scoring=_build_scoring(raw.get("scoring") or {}),
        success_rate=_build_success_rate(raw.get("success_rate") or {}),
        nfe=_build_nfe(raw.get("nfe") or {}),
        posebusters=_build_posebusters(raw.get("posebusters") or {}),
        mcts=_build_mcts(raw.get("mcts") or {}),
        lambda_specific=_build_lambda_specific(raw.get("lambda_specific") or {}),
    )


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------
def to_mcts_kwargs(config: SOTAAlignedConfig) -> Dict[str, Any]:
    """Return kwargs to construct :class:`MCTSProofSearch`.

    Maps the YAML's ``mcts:`` block onto the dataclass fields of
    :class:`molmetal_lam.search_alg.proof_search.MCTSProofSearch`:

    - ``top_k``       -> ``top_k``
    - ``early_stop``  -> ``early_stop``
    - ``patience``    -> ``patience``

    Other YAML fields (``max_depth``, ``branching_target``) are NOT
    fields on the :class:`MCTSProofSearch` dataclass — they are
    passed at *call time* (max_depth) or applied *before* construction
    (branching_target selects the tile library).  We surface them as
    ``__mcts_call_kwargs__`` so the orchestrator can pick them up.

    Also bound as an instance method via ``__get__`` below for ergonomic
    ``config.to_mcts_kwargs()`` calls.
    """
    return {
        # Direct MCTSProofSearch field kwargs:
        "top_k": config.mcts.top_k,
        "early_stop": config.mcts.early_stop,
        "patience": config.mcts.patience,
        # Call-time / pre-construction kwargs (consumed by the orchestrator):
        "__mcts_call_kwargs__": {
            "max_depth": config.mcts.max_depth,
            "branching_target": config.mcts.branching_target,
            "n_simulations": config.nfe.target_per_pocket,
        },
        # Lambda-specific levers (consumed by the orchestrator before
        # constructing MCTSProofSearch; not direct dataclass fields):
        "__lambda_kwargs__": {
            "tile_library": config.lambda_specific.tile_library,
            "click_rules": config.lambda_specific.click_rules,
            "synthesis_oracle": config.lambda_specific.synthesis_oracle,
            "symbolic_prior": config.lambda_specific.symbolic_prior,
            "symbolic_prior_refit_every": config.lambda_specific.symbolic_prior_refit_every,
        },
        # Echo the protocol fingerprint so logs / markdown can label
        # the sweep "SOTA-aligned TargetDiff" without re-reading the YAML.
        "__protocol_fingerprint__": {
            "test_set": config.test_set,
            "n_test_pockets": config.n_test_pockets,
            "docking_engine": config.docking.engine,
            "docking_version": config.docking.version,
            "docking_exhaustiveness": config.docking.exhaustiveness,
            "n_poses": config.docking.n_poses,
            "vina_threshold_kcal": config.success_rate.vina_threshold_kcal,
            "posebusters": config.posebusters.enabled,
        },
    }
