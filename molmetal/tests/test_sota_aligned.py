"""Unit tests for the canonical SOTA-aligned config (``SOTAAlignedConfig``).

These tests guard the YAML <-> dataclass roundtrip and the
``to_mcts_kwargs()`` projection onto ``MCTSProofSearch`` kwargs.  The
configs live at ``molmetal/molmetal_lam/configs/sota_aligned.py`` and
the canonical YAML at
``molmetal/configs/sota_aligned_targetdiff.yaml``.

A regression in any of these surfaces as the R4-C sweep silently
running a non-canonical protocol — exactly the kind of breakage
this test is here to catch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_YAML = PROJECT_ROOT / "molmetal" / "configs" / "sota_aligned_targetdiff.yaml"


@pytest.fixture(scope="module")
def canonical_config() -> SOTAAlignedConfig:
    """Load the canonical TargetDiff YAML once per module."""
    assert CANONICAL_YAML.is_file(), f"YAML missing: {CANONICAL_YAML}"
    return load_sota_aligned_config(str(CANONICAL_YAML))


# ---------------------------------------------------------------------------
# YAML <-> dataclass roundtrip
# ---------------------------------------------------------------------------
def test_canonical_yaml_loads(canonical_config: SOTAAlignedConfig) -> None:
    """Canonical TargetDiff YAML parses into a frozen SOTAAlignedConfig."""
    assert isinstance(canonical_config, SOTAAlignedConfig)
    assert canonical_config.test_set == "crossdocked_pocket10"
    assert canonical_config.n_test_pockets == 100


def test_canonical_yaml_nested_blocks(canonical_config: SOTAAlignedConfig) -> None:
    """All nested config dataclasses populate from the YAML sections."""
    assert isinstance(canonical_config.docking, DockingConfig)
    assert canonical_config.docking.engine == "vina"
    assert canonical_config.docking.version == "1.2.7"
    assert canonical_config.docking.exhaustiveness == 8
    assert canonical_config.docking.n_poses == 9

    assert isinstance(canonical_config.scoring, ScoringConfig)
    assert canonical_config.scoring.sa == "ertl_2000"
    assert canonical_config.scoring.qed == "rdkit"
    assert canonical_config.scoring.lipinski is True

    assert isinstance(canonical_config.success_rate, SuccessRateConfig)
    assert canonical_config.success_rate.vina_threshold_kcal == pytest.approx(-8.0)

    assert isinstance(canonical_config.nfe, NFEConfig)
    assert canonical_config.nfe.target_per_pocket == 1000
    assert canonical_config.nfe.method == "mcts_rollouts"

    assert isinstance(canonical_config.posebusters, PoseBustersConfig)
    assert canonical_config.posebusters.enabled is True
    assert canonical_config.posebusters.threshold == "pass_all"

    assert isinstance(canonical_config.mcts, MCTSConfig)
    assert canonical_config.mcts.top_k == 100
    assert canonical_config.mcts.max_depth == 3
    assert canonical_config.mcts.patience == 50
    assert canonical_config.mcts.early_stop is True
    assert canonical_config.mcts.branching_target == 1020

    assert isinstance(canonical_config.lambda_specific, LambdaSpecificConfig)
    assert canonical_config.lambda_specific.tile_library == "extended_204"
    assert canonical_config.lambda_specific.click_rules == "all_5"
    assert canonical_config.lambda_specific.synthesis_oracle is True
    assert canonical_config.lambda_specific.symbolic_prior is True


def test_sota_aligned_config_is_frozen(canonical_config: SOTAAlignedConfig) -> None:
    """Root + nested configs must be frozen — accidental mutation forbidden."""
    with pytest.raises(Exception):
        canonical_config.n_test_pockets = 999  # type: ignore[misc]
    with pytest.raises(Exception):
        canonical_config.mcts.top_k = 999  # type: ignore[misc]


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loader raises FileNotFoundError on a missing YAML."""
    bogus = tmp_path / "no_such.yaml"
    with pytest.raises(FileNotFoundError):
        load_sota_aligned_config(str(bogus))


def test_load_minimal_yaml_uses_defaults(tmp_path: Path) -> None:
    """Loader fills dataclass defaults when YAML omits a section."""
    yaml_path = tmp_path / "minimal.yaml"
    yaml_path.write_text("test_set: crossdocked_pocket10\nn_test_pockets: 10\n")
    cfg = load_sota_aligned_config(str(yaml_path))
    assert cfg.n_test_pockets == 10
    # Defaults survive:
    assert cfg.docking.engine == "vina"
    assert cfg.mcts.top_k == 100
    assert cfg.posebusters.threshold == "pass_all"


# ---------------------------------------------------------------------------
# to_mcts_kwargs projection
# ---------------------------------------------------------------------------
def test_to_mcts_kwargs_direct_fields(canonical_config: SOTAAlignedConfig) -> None:
    """``to_mcts_kwargs`` emits the direct ``MCTSProofSearch`` field kwargs."""
    mcts = to_mcts_kwargs(canonical_config)
    assert mcts["top_k"] == 100
    assert mcts["early_stop"] is True
    assert mcts["patience"] == 50


def test_to_mcts_kwargs_call_time_section(canonical_config: SOTAAlignedConfig) -> None:
    """``__mcts_call_kwargs__`` carries max_depth + branching + n_sims."""
    mcts = to_mcts_kwargs(canonical_config)
    call = mcts["__mcts_call_kwargs__"]
    assert call["max_depth"] == 3
    assert call["branching_target"] == 1020
    assert call["n_simulations"] == 1000  # == nfe.target_per_pocket


def test_to_mcts_kwargs_lambda_section(canonical_config: SOTAAlignedConfig) -> None:
    """``__lambda_kwargs__`` carries Lambda-only levers."""
    mcts = to_mcts_kwargs(canonical_config)
    lam = mcts["__lambda_kwargs__"]
    assert lam["tile_library"] == "extended_204"
    assert lam["click_rules"] == "all_5"
    assert lam["synthesis_oracle"] is True
    assert lam["symbolic_prior"] is True
    assert lam["symbolic_prior_refit_every"] == 5


def test_to_mcts_kwargs_protocol_fingerprint(canonical_config: SOTAAlignedConfig) -> None:
    """``__protocol_fingerprint__`` is the cite-only citation line."""
    mcts = to_mcts_kwargs(canonical_config)
    fp = mcts["__protocol_fingerprint__"]
    assert fp["test_set"] == "crossdocked_pocket10"
    assert fp["n_test_pockets"] == 100
    assert fp["docking_engine"] == "vina"
    assert fp["docking_version"] == "1.2.7"
    assert fp["docking_exhaustiveness"] == 8
    assert fp["n_poses"] == 9
    assert fp["vina_threshold_kcal"] == pytest.approx(-8.0)
    assert fp["posebusters"] is True


def test_bound_to_mcts_kwargs_method_matches_module_fn(
    canonical_config: SOTAAlignedConfig,
) -> None:
    """``config.to_mcts_kwargs()`` (bound) equals module-level ``to_mcts_kwargs(config)``."""
    bound = canonical_config.to_mcts_kwargs()
    free = to_mcts_kwargs(canonical_config)
    assert bound == free
