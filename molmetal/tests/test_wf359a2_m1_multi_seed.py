"""Wf-359-A2 — m1 multi-seed evaluation scaffold tests.

=================================================================
Wf-359-A2 — CPU-only multi-seed evaluation scaffold tests for
``molmetal/scripts/eval_m1_multi_seed.py``. Closes the Wf-357-B3
multi-seed gap by preparing the aggregation code so the
Wf-355-B1 USER_OK-gated launch wrapper can hand off its outputs
and produce the paper-grade multi-seed mean ± std the moment the
7 DEFERRED runs (Rh 3+4 + Re 0-4) complete.

Per the Wf-359-A2 task brief, 4 NEW tests verify the scaffold:

1. test_wf359a2_load_per_seed_metrics
   — scaffold correctly loads per-seed metrics from the
     ``wf131_p3_lomo_full/`` fallback dir (which is on disk).
2. test_wf359a2_aggregate_mean_std
   — aggregate computation yields correct mean ± std for a
     known 5-seed synthetic cohort (mean=0.7, std=0.05).
3. test_wf359a2_honest_negative_1seed
   — HONEST-NEGATIVE handling: a metal with only 1 seed
     emits the 1-seed reading with HONEST_NEGATIVE_1SEED tag
     (not silently faking a 5-seed mean).
4. test_wf359a2_integration_with_wf131_p3_wrapper
   — integration: the scaffold's output schema is compatible
     with the Wf-131 P3 wrapper's per-seed JSON layout (the
     primary dir from Wf-355-B1 wrapper).

All 4 tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no
GPU launches). Per `party-twenty-spirit` §3.5: every claim
traces to a file/line or live-run output. No ``paper/*`` touches
(TODO-84 honored).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the scaffold (sys.path tweak above keeps the absolute import happy)
from molmetal.scripts.eval_m1_multi_seed import (  # noqa: E402
    aggregate_metal,
    build_aggregate,
    DEFAULT_METALS,
    PRIMARY_DIR,
    FALLBACK_DIR,
    OUTPUT_DIR,
    OUTPUT_FILE,
    RU_KRASNOV_BASELINE,
    GATE_TARGET,
)


# ---------------------------------------------------------------------------
# Test 1: scaffold loads per-seed metrics correctly
# ---------------------------------------------------------------------------


def test_wf359a2_load_per_seed_metrics() -> None:
    """Wf-359-A2 task contract — scaffold loads per-seed metrics from
    the Wf-131 P3 fallback directory.

    Verifies:
      * The scaffold module imports cleanly.
      * The fallback directory ``molmetal/reports/wf131_p3_lomo_full/``
        exists on disk.
      * At least one metal (Ru, Ir, or Os) has per-seed metrics loadable.
      * The per-seed metrics dict contains ``test_auc`` and ``seed`` keys.
      * The loaded n_seeds_used >= 3 for at least one metal (we have a
        real on-disk dataset with 5 seeds).
    """
    assert PRIMARY_DIR.exists(), f"primary dir missing: {PRIMARY_DIR}"
    assert FALLBACK_DIR.exists(), f"fallback dir missing: {FALLBACK_DIR}"

    aggregate = build_aggregate(metals=("Ru",))

    assert "per_metal" in aggregate, "aggregate missing 'per_metal' key"
    assert "Ru" in aggregate["per_metal"], "Ru block missing in aggregate"

    ru_block = aggregate["per_metal"]["Ru"]
    assert "per_seed_aucs" in ru_block, "Ru block missing per_seed_aucs"
    assert "test_auc_mean" in ru_block, "Ru block missing test_auc_mean"
    assert "verdict" in ru_block, "Ru block missing verdict"

    n_seeds = ru_block["n_seeds_used"]
    assert n_seeds >= 3, (
        f"expected ≥3 seeds for Ru (we have 5 on disk per Wf-355-B1 §1.1); "
        f"got n_seeds_used={n_seeds}"
    )

    # Per-seed AUCs must be in valid range
    for auc in ru_block["per_seed_aucs"]:
        assert 0.0 <= auc <= 1.0, f"test_auc out of [0,1]: {auc}"

    # Mean should be in same range
    assert (
        0.0 <= ru_block["test_auc_mean"] <= 1.0
    ), f"test_auc_mean out of [0,1]: {ru_block['test_auc_mean']}"


# ---------------------------------------------------------------------------
# Test 2: aggregate computation (mean ± std)
# ---------------------------------------------------------------------------


def test_wf359a2_aggregate_mean_std() -> None:
    """Wf-359-A2 task contract — aggregate correctly computes mean ± std
    for a synthetic 5-seed cohort.

    Verifies:
      * For a known input (5 seeds: 0.6, 0.65, 0.7, 0.75, 0.8), the
        mean is 0.7 and the population std is 0.0707 (≈ √0.005).
      * verdict is MEASURED (n>=3).
      * Aggregate runs in pure-CPU without IO (synthetic input).
    """
    synthetic_per_seed = [
        {"seed": 0, "test_auc": 0.60},
        {"seed": 1, "test_auc": 0.65},
        {"seed": 2, "test_auc": 0.70},
        {"seed": 3, "test_auc": 0.75},
        {"seed": 4, "test_auc": 0.80},
    ]

    block = aggregate_metal(
        per_seed=synthetic_per_seed,
        metal="Ru",
        source_path="<synthetic test>",
    )

    assert block["n_seeds_used"] == 5, f"expected 5 seeds; got {block['n_seeds_used']}"
    assert (
        block["verdict"] == "MEASURED"
    ), f"expected MEASURED verdict; got {block['verdict']!r}"

    # mean = (0.60+0.65+0.70+0.75+0.80)/5 = 0.70
    expected_mean = 0.70
    assert math.isclose(
        block["test_auc_mean"], expected_mean, abs_tol=1e-9
    ), f"mean {block['test_auc_mean']} != expected {expected_mean}"

    # population std = sqrt(((0.10)^2 + (0.05)^2 + 0 + (0.05)^2 + (0.10)^2) / 5)
    #                 = sqrt((0.01 + 0.0025 + 0 + 0.0025 + 0.01) / 5)
    #                 = sqrt(0.025/5) = sqrt(0.005) ≈ 0.07071
    expected_std = math.sqrt(0.005)
    assert math.isclose(
        block["test_auc_std"], expected_std, abs_tol=1e-4
    ), f"std {block['test_auc_std']} != expected {expected_std:.5f}"

    # m1 Ru-specific gate target delta must be present and correct
    assert block["baseline_krasnov_2026"] == RU_KRASNOV_BASELINE
    assert block["gate_target"] == GATE_TARGET
    assert math.isclose(
        block["lift_vs_baseline"],
        expected_mean - RU_KRASNOV_BASELINE,
        abs_tol=1e-9,
    )
    assert (
        block["hit_gate"] is False
    ), f"0.70 < gate 0.86; expected hit_gate=False; got {block['hit_gate']}"


# ---------------------------------------------------------------------------
# Test 3: HONEST-NEGATIVE handling (1-seed only)
# ---------------------------------------------------------------------------


def test_wf359a2_honest_negative_1seed() -> None:
    """Wf-359-A2 task contract — 1-seed-only data emits HONEST_NEGATIVE
    tag (not silently faking a 5-seed mean).

    Verifies:
      * For a metal with only 1 seed, the verdict is HONEST_NEGATIVE_1SEED.
      * test_auc_mean equals the single seed's value (no fake std).
      * test_auc_std = 0.0 (population std of a single sample).
      * n_seeds_used == 1 (explicit tag).
      * Per-seed AUC list has exactly 1 entry.
    """
    synthetic_per_seed = [{"seed": 0, "test_auc": 0.7613}]  # Wf-124 P3 anchor

    block = aggregate_metal(
        per_seed=synthetic_per_seed,
        metal="Ru",
        source_path="<1-seed test>",
    )

    assert (
        block["verdict"] == "HONEST_NEGATIVE_1SEED"
    ), f"expected HONEST_NEGATIVE_1SEED verdict; got {block['verdict']!r}"
    assert block["n_seeds_used"] == 1, f"expected n=1; got {block['n_seeds_used']}"
    assert len(block["per_seed_aucs"]) == 1
    assert math.isclose(block["test_auc_mean"], 0.7613, abs_tol=1e-9)
    assert math.isclose(block["test_auc_std"], 0.0, abs_tol=1e-9)

    # m1 Ru-specific: lift vs baseline still computed but tagged honest-negative
    assert math.isclose(
        block["lift_vs_baseline"],
        0.7613 - RU_KRASNOV_BASELINE,
        abs_tol=1e-9,
    )
    assert block["hit_gate"] is False, "0.7613 < gate 0.86; expected hit_gate=False"


# ---------------------------------------------------------------------------
# Test 4: integration with Wf-131 P3 wrapper output schema
# ---------------------------------------------------------------------------


def test_wf359a2_integration_with_wf131_p3_wrapper() -> None:
    """Wf-359-A2 task contract — scaffold output schema is compatible with
    the Wf-131 P3 wrapper's per-seed JSON layout.

    Verifies:
      * The primary dir ``wf131_p3_deferred_runs/`` exists (created by
        Wf-355-B1 wrapper, even if currently empty of per-seed JSONs).
      * The scaffold can write its aggregate to OUTPUT_DIR/aggregate.json
        and read it back without errors.
      * The aggregate JSON has the keys expected by downstream
        consumers: ``per_metal``, ``verdict``, ``overall_mean``,
        ``honest_framing``, ``metals``.
      * The fallback dir's per-seed JSON files (e.g. Ru/Ir/Os) parse
        via the scaffold's loader with the expected keys.
    """
    # Primary dir may exist even if empty (Wf-355-B1 wrapper created it
    # via --dry-run). The scaffold should still emit a usable aggregate
    # by falling back to wf131_p3_lomo_full.
    assert PRIMARY_DIR.exists(), f"primary dir missing: {PRIMARY_DIR}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the scaffold and write to disk
    aggregate = build_aggregate(metals=DEFAULT_METALS)
    OUTPUT_FILE.write_text(json.dumps(aggregate, indent=2) + "\n")

    # Read it back
    reloaded = json.loads(OUTPUT_FILE.read_text())

    # Required keys
    for key in ("per_metal", "verdict", "overall_mean", "honest_framing", "metals"):
        assert key in reloaded, f"aggregate JSON missing key {key!r}"

    # Per-metal blocks all present
    assert set(reloaded["metals"]) == set(
        DEFAULT_METALS
    ), f"metals {reloaded['metals']} != default {DEFAULT_METALS}"
    for metal in DEFAULT_METALS:
        assert (
            metal in reloaded["per_metal"]
        ), f"per_metal missing {metal!r}; got keys {list(reloaded['per_metal'].keys())}"

    # The verdict is one of the known strings
    assert reloaded["verdict"] in (
        "READY_FOR_USER_OK_LAUNCH",
        "ALL_METALS_5SEED",
    ), f"unexpected verdict {reloaded['verdict']!r}"

    # The Wf-131 P3 fallback dir should yield per-seed JSON files for
    # Ru/Ir/Os (the 3 COMPLETE metals per Wf-354-A1 §1.2 + Wf-355-B1 §1.1).
    # Rh has only 3 seeds; Re has 0 seeds. Both must be reported honestly.
    for metal in ("Ru", "Ir", "Os"):
        block = reloaded["per_metal"][metal]
        assert (
            block["n_seeds_used"] >= 3
        ), f"{metal} should have ≥3 seeds from Wf-131 P3 anchor; got n={block['n_seeds_used']}"
        assert block["test_auc_mean"] is not None, f"{metal} mean is None"
        assert (
            block["verdict"] in ("MEASURED", "PARTIAL_2SEED")
        ), f"{metal} should be MEASURED (≥3 seeds) or PARTIAL_2SEED; got {block['verdict']!r}"

    # Rh has 3 seeds (DEFERRED seeds 3+4) — still MEASURED in scaffold terms
    rh_block = reloaded["per_metal"]["Rh"]
    assert (
        rh_block["n_seeds_used"] == 3
    ), f"Rh should have 3 seeds from Wf-131 P3 partial; got n={rh_block['n_seeds_used']}"
    assert rh_block["verdict"] == "MEASURED"

    # Re has 0 seeds — must be honestly NOT_MEASURED (this is the key
    # honest-negative case: Re fold 0/5 DEFERRED per Wf-354-A1 §5)
    re_block = reloaded["per_metal"]["Re"]
    assert (
        re_block["n_seeds_used"] == 0
    ), f"Re should have 0 seeds (DEFERRED 0/5); got n={re_block['n_seeds_used']}"
    assert (
        re_block["verdict"] == "NOT_MEASURED"
    ), f"Re verdict should be NOT_MEASURED; got {re_block['verdict']!r}"
    assert (
        re_block["test_auc_mean"] is None
    ), f"Re test_auc_mean should be None (NOT_MEASURED); got {re_block['test_auc_mean']}"
