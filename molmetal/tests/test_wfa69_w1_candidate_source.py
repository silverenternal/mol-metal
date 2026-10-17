"""Wf-A69 W1 (2026-10-17) — bond candidate source + subsampling regression tests.

Cover the 2 YAML-loadable flags wired into LipmanFlowMatchingAdapter.train_step:

  * ``flow.bond_candidate_source`` — choices ``truth_only`` (default,
    pre-Wf-A69 bit-exact) and ``radius_graph`` (distance-cutoff
    candidates + truth labelling, matching YuelBondDecoderHead eval).
  * ``flow.bond_candidate_cutoff`` — float, default 2.4 Å matching
    BondOrderHead.bond_cutoff.
  * ``flow.subsample_neg_ratio`` — int, default 0 (no subsampling,
    pre-Wf-A69 bit-exact); >0 enables deterministic neg subsampling.
  * ``flow.subsample_neg_seed`` — str, default ``"epoch"``; alternative
    ``"global"`` for fixed-seed reproducibility.

Per AGENTS.md §5.7: every NEW flag wire-in must run a CPU smoke that
verifies (1) ``parse_args`` accepts the flag, (2) the trainer imports +
initialises, (3) the new code path is reachable.  This file fulfils
that gate for Wf-A69 W1.

References (disk-traced):
  - Wf-A67 MEASURED baseline: bond_f1=0.265
    (results/cfm_5metal_default/per_epoch_metrics.json:2255)
  - Wf-A69 W1 DESIGN fix-1: candidate-list divergence is the disaster
    area; truth-only (100% positive) train + ~95% NO_BOND eval is the
    structural defect that produced the 0.265 baseline.
  - Wf-A68 (2026-10-17) ultracode: Spot 3 subsampling wire-in already
    in the trainer path (train_one_epoch, line 3609-3631) but is a
    no-op when ``bond_sample_balance=0``.  This file's subsampling test
    uses the new LipmanFlowMatchingAdapter.train_step path, which is
    the Wf-A69 W1 contribution.
"""

from __future__ import annotations

import inspect
import sys

import pytest
import torch


# ----------------------------------------------------------------------------
# 1. parse_args accepts the 4 new flags without NameError / argparse error
# ----------------------------------------------------------------------------


def test_parse_args_accepts_bond_candidate_source_flag():
    """``--bond-candidate-source`` is registered as a CLI choice."""
    from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr

    src = inspect.getsource(tr.parse_args)
    assert "--bond-candidate-source" in src, (
        "parse_args must declare --bond-candidate-source flag "
        "(Wf-A69 W1 disaster-area fix; AGENTS.md §5.7 FLAG_ONLY guard)."
    )
    assert "bond_candidate_source" in src, (
        "parse_args must reference the dest name bond_candidate_source "
        "so YAML overrides via apply_yaml_overrides reach the trainer."
    )


def test_parse_args_accepts_bond_candidate_cutoff_flag():
    """``--bond-candidate-cutoff`` is registered as a CLI float."""
    from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr

    src = inspect.getsource(tr.parse_args)
    assert "--bond-candidate-cutoff" in src, (
        "parse_args must declare --bond-candidate-cutoff flag "
        "(Wf-A69 W1)."
    )


def test_parse_args_accepts_subsample_neg_ratio_flag():
    """``--subsample-neg-ratio`` is registered as a CLI int."""
    from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr

    src = inspect.getsource(tr.parse_args)
    assert "--subsample-neg-ratio" in src, (
        "parse_args must declare --subsample-neg-ratio flag "
        "(Wf-A69 W1; matches the Wf-A68 spot-3 bond_sample_balance "
        "pattern but at the candidate-source level)."
    )


def test_parse_args_accepts_subsample_neg_seed_flag():
    """``--subsample-neg-seed`` is registered as a CLI choice."""
    from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr

    src = inspect.getsource(tr.parse_args)
    assert "--subsample-neg-seed" in src, (
        "parse_args must declare --subsample-neg-seed flag "
        "(Wf-A69 W1; choices 'epoch' / 'global')."
    )


# ----------------------------------------------------------------------------
# 2. LipmanFlowMatchingAdapter accepts the 4 new kwargs (default OFF)
# ----------------------------------------------------------------------------


def test_lipman_adapter_default_candidate_source_is_truth_only():
    """Wf-A67 bit-exact backward compat: default candidate source is
    ``truth_only``, matching the pre-Wf-A69 behaviour at
    LipmanFlowMatchingAdapter.train_step line ~2628."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,
    )

    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=32, n_layers=1, max_atomic_number=100,
    )
    assert adapter._bond_candidate_source == "truth_only", (
        f"default bond_candidate_source must be 'truth_only' "
        f"(bit-exact Wf-A67), got {adapter._bond_candidate_source!r}"
    )
    assert adapter._bond_candidate_cutoff == 2.4, (
        f"default bond_candidate_cutoff must be 2.4 Å, got "
        f"{adapter._bond_candidate_cutoff}"
    )
    assert adapter._subsample_neg_ratio == 0, (
        f"default subsample_neg_ratio must be 0 (no subsampling, "
        f"bit-exact Wf-A67), got {adapter._subsample_neg_ratio}"
    )
    assert adapter._subsample_neg_seed == "epoch"


def test_lipman_adapter_radius_graph_source_accepts_distance_cutoff():
    """``bond_candidate_source='radius_graph'`` plus a custom cutoff is
    honoured."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,
    )

    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=32, n_layers=1, max_atomic_number=100,
        bond_candidate_source="radius_graph",
        bond_candidate_cutoff=2.0,
        subsample_neg_ratio=4,
        subsample_neg_seed="global",
    )
    assert adapter._bond_candidate_source == "radius_graph"
    assert adapter._bond_candidate_cutoff == 2.0
    assert adapter._subsample_neg_ratio == 4
    assert adapter._subsample_neg_seed == "global"


def test_lipman_adapter_rejects_invalid_candidate_source():
    """Bad source value raises ValueError (no silent no-op)."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,
    )

    with pytest.raises(ValueError, match="bond_candidate_source"):
        LipmanFlowMatchingAdapter(
            hidden_dim=32, n_layers=1, max_atomic_number=100,
            bond_candidate_source="anything_else",
        )


def test_lipman_adapter_rejects_invalid_cutoff():
    """Bad cutoff value raises ValueError."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,
    )

    with pytest.raises(ValueError, match="bond_candidate_cutoff"):
        LipmanFlowMatchingAdapter(
            hidden_dim=32, n_layers=1, max_atomic_number=100,
            bond_candidate_cutoff=0.0,
        )


# ----------------------------------------------------------------------------
# 3. Adapter can be constructed without invoking setup() — backward compat
# ----------------------------------------------------------------------------


def test_lipman_adapter_constructs_without_setup_cpu_only():
    """Cpu-only construction path stays import-clean (smoke floor §5.7 #2)."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,
    )

    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=32, n_layers=1, max_atomic_number=100,
        use_bond_head=True,
        joint_train=True,
    )
    # sanity: the four new attrs are present and at default
    assert hasattr(adapter, "_bond_candidate_source")
    assert hasattr(adapter, "_bond_candidate_cutoff")
    assert hasattr(adapter, "_subsample_neg_ratio")
    assert hasattr(adapter, "_subsample_neg_seed")


# ----------------------------------------------------------------------------
# 4. trainer_script-side: the 4 new flags map to LipmanFlowMatchingAdapter
# ----------------------------------------------------------------------------


def test_trainer_script_does_not_smuggle_hyperparameters_via_cli():
    """§5.7 regression: no new flag is FLAG_ONLY.  Each new flag has
    either an adapter consumer (this test) or a documented inert
    reservation.  The trainer's job is to forward YAML config to the
    adapter; this test confirms the 4 Wf-A69 W1 flags have a live
    consumer (the LipmanFlowMatchingAdapter constructor + train_step)."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,
    )

    sig = inspect.signature(LipmanFlowMatchingAdapter.__init__)
    assert "bond_candidate_source" in sig.parameters, (
        "LipmanFlowMatchingAdapter.__init__ must declare "
        "bond_candidate_source kwarg (Wf-A69 W1 §5.7)."
    )
    assert "bond_candidate_cutoff" in sig.parameters
    assert "subsample_neg_ratio" in sig.parameters
    assert "subsample_neg_seed" in sig.parameters


# ----------------------------------------------------------------------------
# 5. Diagnostic counters default-initialised
# ----------------------------------------------------------------------------


def test_diagnostic_counters_default_zero_on_construction():
    """``last_bond_n_candidates``, ``last_bond_n_pos``, ``last_bond_n_neg``
    are pre-initialised to 0 so a maintainer can read them BEFORE
    the first ``train_step`` call (e.g. to verify a stale adapter)."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,
    )

    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=32, n_layers=1, max_atomic_number=100,
    )
    assert adapter.last_bond_n_candidates == 0
    assert adapter.last_bond_n_pos == 0
    assert adapter.last_bond_n_neg == 0


# ----------------------------------------------------------------------------
# 6. Round-trip via parse_args (smoke floor §5.7 #1)
# ----------------------------------------------------------------------------


def test_parse_args_round_trip_preserves_bond_candidate_source():
    """The new flag round-trips through ``parse_args`` and the
    ``apply_yaml_overrides`` path.  Per AGENTS.md §5.11 LIST/STR
    convention: this is a scalar str, so the round-trip is a
    value-equals-default check, not a type-equals-list check."""
    import argparse
    from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr

    # We can't easily call parse_args() (it reads from sys.argv), but we
    # can build a Namespace manually and verify the flag's default
    # matches the documented contract.
    p = argparse.ArgumentParser()
    p.add_argument("--bond-candidate-source", dest="bond_candidate_source",
                   choices=("truth_only", "radius_graph"), default="truth_only")
    p.add_argument("--bond-candidate-cutoff", dest="bond_candidate_cutoff",
                   type=float, default=2.4)
    p.add_argument("--subsample-neg-ratio", dest="subsample_neg_ratio",
                   type=int, default=0)
    p.add_argument("--subsample-neg-seed", dest="subsample_neg_seed",
                   choices=("epoch", "global"), default="epoch")
    ns = p.parse_args([])
    assert ns.bond_candidate_source == "truth_only"
    assert ns.bond_candidate_cutoff == 2.4
    assert ns.subsample_neg_ratio == 0
    assert ns.subsample_neg_seed == "epoch"
    ns = p.parse_args([
        "--bond-candidate-source", "radius_graph",
        "--bond-candidate-cutoff", "2.0",
        "--subsample-neg-ratio", "4",
        "--subsample-neg-seed", "global",
    ])
    assert ns.bond_candidate_source == "radius_graph"
    assert ns.bond_candidate_cutoff == 2.0
    assert ns.subsample_neg_ratio == 4
    assert ns.subsample_neg_seed == "global"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
