"""Configs sub-package — version-controlled experimental configurations.

This package exposes the canonical SOTA-aligned run config (TargetDiff
protocol) as a frozen dataclass with load + projection helpers, so future
sweeps can reproduce the SOTA-aligned setup from a single YAML.

Canonical entry points:
    - :func:`load_sota_aligned_config` — load YAML -> :class:`SOTAAlignedConfig`
    - :func:`to_mcts_kwargs`            — config -> kwargs for
      :class:`MCTSProofSearch`
"""
from molmetal_lam.configs.sota_aligned import (
    DockingConfig,
    LambdaSpecificConfig,
    MCTSConfig,
    NFEConfig,
    PoseBustersConfig,
    ScoringConfig,
    SOTAAlignedConfig,
    SuccessRateConfig,
    load_sota_aligned_config,
    to_mcts_kwargs,
)

__all__ = [
    "DockingConfig",
    "LambdaSpecificConfig",
    "MCTSConfig",
    "NFEConfig",
    "PoseBustersConfig",
    "ScoringConfig",
    "SOTAAlignedConfig",
    "SuccessRateConfig",
    "load_sota_aligned_config",
    "to_mcts_kwargs",
]
