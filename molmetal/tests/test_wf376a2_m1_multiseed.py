"""Wf-376-A2 — m1_auc multi-seed evaluation scaffold tests.

=================================================================
Wf-376-A2 — CPU-only tests for the multi-seed evaluation scaffold
``molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py``. Extends
Wf-359-A2 with full 5-fold CV + multi-seed averaging + 95% CI +
ENV-gated dispatch + paper §6 framing fields.

Per the Wf-376-A2 task brief, 3 NEW tests verify the scaffold:

1. test_wf376a2_scaffold_structure
   — scaffold structure: CLI flags + main function signature +
     build_aggregate / aggregate_metal_multiseed / resolve_mode API.
2. test_wf376a2_env_dispatch_logic
   — ENV-gated dispatch logic: SCAFFOLD mode is the default
     (CPU-only, USER_OK_SCAFFOLD=1); LIVE mode requires USER_OK=1;
     --mode flag overrides ENV-gated detection.
3. test_wf376a2_per_metal_auc_aggregation
   — per-metal AUC aggregation logic: mean ± std + 95% CI math
     correct on synthetic 5-seed × 5-fold input; verdict strings
     MEASURED / HONEST_NEGATIVE_1SEED / PARTIAL_2SEED per
     Wf-357-B3 honest-negative contract.

All 3 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no
GPU launches). Per `party-twenty-spirit` §3.5: every claim
traces to a file/line or live-run output. No ``paper/*`` touches
(TODO-84 honored).
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the scaffold module via importlib to avoid the hyphen-name module
# file naming convention clashing with the dotted import path.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "wf376a2_scaffold_module",
    str(PROJECT_ROOT / "molmetal" / "scripts" / "wf376a2_m1_auc_multiseed_scaffold.py"),
)
assert _spec is not None and _spec.loader is not None
_wf376a2_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wf376a2_mod)

scaffold = _wf376a2_mod


# ---------------------------------------------------------------------------
# Test 1: scaffold structure (CLI flags + main function signature)
# ---------------------------------------------------------------------------


def test_wf376a2_scaffold_structure() -> None:
    """Wf-376-A2 task contract — scaffold has the expected public API.

    Verifies:
      * Public symbols exist: build_aggregate, aggregate_metal_multiseed,
        resolve_mode, synthetic_per_seed_auc, DEFAULT_METALS,
        DEFAULT_SEEDS, N_FOLDS, PROJECTED_LIFT_W335_M2, main.
      * ``main`` is callable and has argparse CLI flags: --metals,
        --seeds, --n-folds, --mode, --out, --print.
      * build_aggregate returns a dict with the required top-level keys.
    """
    # Required public symbols
    for sym in (
        "build_aggregate",
        "aggregate_metal_multiseed",
        "resolve_mode",
        "synthetic_per_seed_auc",
        "DEFAULT_METALS",
        "DEFAULT_SEEDS",
        "N_FOLDS",
        "PROJECTED_LIFT_W335_M2",
        "main",
    ):
        assert hasattr(scaffold, sym), f"scaffold missing public symbol: {sym}"

    # Defaults are 5-metal × 5-seed × 5-fold per Wf-376-A2 brief
    assert scaffold.DEFAULT_METALS == ("Pt", "Ru", "Ir", "Au", "Pd"), (
        f"DEFAULT_METALS must be 5-metal Wf-131 P3 subset; "
        f"got {scaffold.DEFAULT_METALS}"
    )
    assert scaffold.DEFAULT_SEEDS == (
        0,
        1,
        2,
        3,
        4,
    ), f"DEFAULT_SEEDS must be 5-seed; got {scaffold.DEFAULT_SEEDS}"
    assert scaffold.N_FOLDS == 5, f"N_FOLDS must be 5; got {scaffold.N_FOLDS}"

    # main() is callable
    assert callable(scaffold.main), "main must be callable"

    # CLI flag inspection: read the argparse parser from main()
    src = inspect.getsource(scaffold.main)
    for flag in ("--metals", "--seeds", "--n-folds", "--mode", "--out", "--print"):
        assert flag in src, f"CLI flag {flag} missing from main() source"

    # build_aggregate returns dict with required keys
    aggregate = scaffold.build_aggregate(mode="SCAFFOLD")
    for key in (
        "schema",
        "wf_origin",
        "parent",
        "mode",
        "env_user_ok",
        "env_user_ok_scaffold",
        "metals",
        "seeds",
        "n_folds",
        "per_metal",
        "overall_mean",
        "overall_std",
        "paper_section6_framing",
        "honest_framing",
        "verdict",
    ):
        assert key in aggregate, f"aggregate missing top-level key {key!r}"

    # schema string is the canonical wf376a2 marker
    assert aggregate["schema"] == "wf376a2_m1_auc_multiseed_scaffold.v1"


# ---------------------------------------------------------------------------
# Test 2: ENV-gated dispatch logic
# ---------------------------------------------------------------------------


def test_wf376a2_env_dispatch_logic() -> None:
    """Wf-376-A2 task contract — ENV-gated dispatch between SCAFFOLD and LIVE.

    Verifies:
      * Default (no env vars) → SCAFFOLD mode (CPU-only, safe per
        AGENTS.md §5.1 + kill_then_adapt §0).
      * USER_OK_SCAFFOLD=1 → SCAFFOLD mode (explicit opt-in to scaffold).
      * USER_OK=1 → LIVE mode (operator consent for real GPU retrain).
      * build_aggregate aggregates carry the ``mode`` field reflecting
        the dispatched mode.
      * LIVE mode without USER_OK=1 → main() returns 1 (FAIL); this
        path is not exercised here (we call resolve_mode + build_aggregate
        directly to avoid mutating process env at test time).
    """
    # Save and restore relevant env vars so we don't pollute the test process
    saved_user_ok = os.environ.pop("USER_OK", None)
    saved_user_ok_scaffold = os.environ.pop("USER_OK_SCAFFOLD", None)
    try:
        # Default: both unset → SCAFFOLD
        assert scaffold.resolve_mode() == "SCAFFOLD", (
            "default mode (no env vars) must be SCAFFOLD per "
            "kill_then_adapt §0 (CPU-only safe default)"
        )

        # Explicit USER_OK_SCAFFOLD=1 → SCAFFOLD
        os.environ["USER_OK_SCAFFOLD"] = "1"
        assert (
            scaffold.resolve_mode() == "SCAFFOLD"
        ), "USER_OK_SCAFFOLD=1 must yield SCAFFOLD mode"

        # USER_OK=1 wins over USER_OK_SCAFFOLD=1 (operator consent > scaffold)
        os.environ["USER_OK"] = "1"
        assert (
            scaffold.resolve_mode() == "LIVE"
        ), "USER_OK=1 must yield LIVE mode (operator consent for GPU)"

        # aggregate JSON carries the mode field + env snapshot
        aggregate = scaffold.build_aggregate(mode="SCAFFOLD")
        assert aggregate["mode"] == "SCAFFOLD"
        assert aggregate["env_user_ok_scaffold"] == "1"

        # Paper §6 framing field present + carries honest-negative posture
        s6 = aggregate["paper_section6_framing"]
        for k in (
            "subsection",
            "m1_metric_id",
            "wf_origin_short",
            "MEASURED_today",
            "PROJECTED_lift_envelope",
            "honest_negative_posture",
            "kill_then_adapt_directive",
        ):
            assert k in s6, f"paper_section6_framing missing key {k!r}"
        assert "§6" in s6["subsection"]
        assert "m1" in s6["m1_metric_id"]
        # MEASURED/PROJECTED honest-framing tags are present (per AGENTS.md §5.2)
        # either as key names or as field-content text — both must be present
        mea_field = str(s6.get("MEASURED_today", ""))
        proj_field = str(s6.get("PROJECTED_lift_envelope", ""))
        assert (
            mea_field and proj_field
        ), "MEASURED_today + PROJECTED_lift_envelope must both be non-empty"
        # Honest-negative posture must be present (per party-twenty-spirit §3.5)
        hn_field = str(s6.get("honest_negative_posture", ""))
        assert hn_field and (
            "PROJECTED" in hn_field or "honest" in hn_field.lower()
        ), f"honest_negative_posture must mention PROJECTED or honest; got {hn_field!r}"
    finally:
        # Restore env
        if saved_user_ok is not None:
            os.environ["USER_OK"] = saved_user_ok
        else:
            os.environ.pop("USER_OK", None)
        if saved_user_ok_scaffold is not None:
            os.environ["USER_OK_SCAFFOLD"] = saved_user_ok_scaffold
        else:
            os.environ.pop("USER_OK_SCAFFOLD", None)


# ---------------------------------------------------------------------------
# Test 3: per-metal AUC aggregation logic
# ---------------------------------------------------------------------------


def test_wf376a2_per_metal_auc_aggregation() -> None:
    """Wf-376-A2 task contract — per-metal AUC aggregation is correct.

    Verifies:
      * 5-seed × 5-fold synthetic input → mean AUC + std + 95% CI
        computed correctly per Wf-335-M2 aggregation math.
      * Honest-negative contract: n_seeds=1 → HONEST_NEGATIVE_1SEED;
        n_seeds=2 → PARTIAL_2SEED; n_seeds≥3 → MEASURED.
      * m1 Ru-specific gate target delta computed correctly.
      * 95% CI margin = z * std / sqrt(n_obs) with z=1.96.
    """
    # Synthetic 5-seed × 5-fold input — every fold AUC = 0.7
    # Expected: mean = 0.7, std = 0.0, 95% CI = (0.7, 0.7)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    per_seed_aucs: Dict[int, List[float]] = {
        0: [0.7, 0.7, 0.7, 0.7, 0.7],
        1: [0.7, 0.7, 0.7, 0.7, 0.7],
        2: [0.7, 0.7, 0.7, 0.7, 0.7],
        3: [0.7, 0.7, 0.7, 0.7, 0.7],
        4: [0.7, 0.7, 0.7, 0.7, 0.7],
    }

    block = scaffold.aggregate_metal_multiseed("Ru", seeds, per_seed_aucs)

    assert block["metal"] == "Ru"
    assert (
        block["n_seeds_used"] == 5
    ), f"expected n_seeds=5; got {block['n_seeds_used']}"
    assert block["n_obs_total"] == 25, f"expected n_obs=25; got {block['n_obs_total']}"
    assert (
        block["verdict"] == "MEASURED"
    ), f"5-seed × 5-fold → MEASURED; got {block['verdict']!r}"
    assert math.isclose(block["test_auc_mean"], 0.7, abs_tol=1e-9)
    assert math.isclose(block["test_auc_std"], 0.0, abs_tol=1e-9)
    # CI = mean ± z*std/sqrt(n); std=0 → CI = (mean, mean)
    assert math.isclose(block["auc_95ci_low"], 0.7, abs_tol=1e-9)
    assert math.isclose(block["auc_95ci_high"], 0.7, abs_tol=1e-9)

    # m1 Ru-specific: gate target delta
    assert block["baseline_krasnov_2026"] == scaffold.RU_KRASNOV_BASELINE
    assert block["gate_target"] == scaffold.GATE_TARGET
    # 0.7 < gate 0.86 → hit_gate=False
    assert block["hit_gate"] is False, "0.7 < 0.86 → hit_gate must be False"

    # ---- 1-seed HONEST_NEGATIVE contract (per Wf-357-B3 §4) ----
    block_1seed = scaffold.aggregate_metal_multiseed(
        "Ru", (0,), {0: [0.7613, 0.7613, 0.7613, 0.7613, 0.7613]}
    )
    assert block_1seed["n_seeds_used"] == 1
    assert (
        block_1seed["verdict"] == "HONEST_NEGATIVE_1SEED"
    ), f"1-seed → HONEST_NEGATIVE_1SEED; got {block_1seed['verdict']!r}"
    assert math.isclose(block_1seed["test_auc_mean"], 0.7613, abs_tol=1e-9)
    assert block_1seed["test_auc_std"] == 0.0

    # ---- 2-seed PARTIAL_2SEED contract ----
    block_2seed = scaffold.aggregate_metal_multiseed(
        "Ru", (0, 1), {0: [0.7] * 5, 1: [0.7] * 5}
    )
    assert block_2seed["n_seeds_used"] == 2
    assert (
        block_2seed["verdict"] == "PARTIAL_2SEED"
    ), f"2-seed → PARTIAL_2SEED; got {block_2seed['verdict']!r}"

    # ---- 0-seed NOT_MEASURED contract ----
    block_0seed = scaffold.aggregate_metal_multiseed("Ru", (0,), {})
    assert block_0seed["verdict"] == "NOT_MEASURED"
    assert block_0seed["test_auc_mean"] is None

    # ---- 95% CI math on non-degenerate input ----
    # 5-seed × 5-fold with each fold alternating 0.6 / 0.8 / 0.6 / 0.8 / 0.6
    # → 15 obs at 0.6 + 10 obs at 0.8 = 25 total → mean = (15*0.6 + 10*0.8) / 25 = 0.68
    alternating: Dict[int, List[float]] = {
        seed: [0.6, 0.8, 0.6, 0.8, 0.6] for seed in seeds
    }
    block_alt = scaffold.aggregate_metal_multiseed("Ru", seeds, alternating)
    assert math.isclose(block_alt["test_auc_mean"], 0.68, abs_tol=1e-9)
    # variance = (15 * 0.08² + 10 * 0.12²) / 25 = (15*0.0064 + 10*0.0144)/25
    #          = (0.096 + 0.144) / 25 = 0.24 / 25 = 0.0096
    # std = sqrt(0.0096) ≈ 0.09798
    expected_std = math.sqrt(0.0096)
    assert math.isclose(
        block_alt["test_auc_std"], expected_std, abs_tol=1e-4
    ), f"std {block_alt['test_auc_std']} != expected {expected_std:.5f}"
    # 95% CI margin = 1.96 * std / sqrt(n_obs) = 1.96 * std / sqrt(25) = 1.96 * std / 5
    expected_margin = 1.96 * expected_std / 5.0
    assert math.isclose(
        block_alt["auc_95ci_low"],
        0.68 - expected_margin,
        abs_tol=1e-6,
    )
    assert math.isclose(
        block_alt["auc_95ci_high"],
        0.68 + expected_margin,
        abs_tol=1e-6,
    )
