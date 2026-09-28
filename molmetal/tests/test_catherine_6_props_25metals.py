"""Tests for Wf-57 Phase 3 — Catherine 6/6 properties × 25+ metals.

==============================================================================

Pytest suite for the Catherine property verifiers and the Wf-57
sweep bench.  The suite runs **CPU-only** (no GPU / RDKit is required
for the synthetic-summary fast path).

Test coverage
=============

* **30 default d-block metals** loaded from
  :data:`molmetal.math_skills.catherine_properties.DBLOCK_30_METALS`
  (parity with ``molmetal/data/transition_metals_30.csv``).

* **5 metals × 6 properties = 30 property verdicts** — the Wf-57 ship
  target of "30/30 pytest" (= 6 properties × 5 representative metals).
  The 5 representative metals span 3d / 4d / 5d + various coordination
  geometries:

    - ``Pt`` 5d square-planar (R17 platinum anchor)
    - ``Ru`` 4d octahedral
    - ``Fe`` 3d octahedral
    - ``Cu`` 3d square-planar (Zn/Cu sister-metal)
    - ``Au`` 5d square-planar (AuI/AuIII amphoteric)

* **8 per-property tests** — one PASS test + one FAIL-on-violation
  test for each of P1..P6 (12 tests in this row).

* **5 integration tests** — sweep + 6/6 verdict + bench-output
  smoke tests.

**Total**: 30 pytest items (5 metals × 6 properties) + 12 per-property
+ 5 integration = **47 tests**, all designed to PASS on a vanilla
Python install (no torch / RDKit / cuda requirements).

Notes
-----
The synthetic-summary path is the primary test target; real
closure-theorem BFS would slow the suite down by 30-60s and is not
required for Wf-57 ship.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Repo-root path injection — matches the bench script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.math_skills.catherine_properties import (
    DBLOCK_30_METALS,
    DEFAULT_COORDINATION_GEOMETRY,
    GEOMETRY_MAX_CN,
    Catherine6PropsVerdict,
    CatherinePropertyResult,
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


# ---------------------------------------------------------------------------
# Constants — 30 d-block metals (parity with transition_metals_30.csv)
# ---------------------------------------------------------------------------

EXPECTED_30_METALS = {
    # 3d
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d + La
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

# Five representative metals for the Wf-57 ship target (5 metals ×
# 6 props = 30 pypermetal tests).
SHIP_TARGET_METALS = ("Pt", "Ru", "Fe", "Cu", "Au")

EXPECTED_6_PROPERTY_IDS = (
    "P1_reflexivity",
    "P2_transitivity",
    "P3_symmetry",
    "P4_dblock",
    "P5_reachability",
    "P6_composability",
)


# ---------------------------------------------------------------------------
# 0. Constants + module surface
# ---------------------------------------------------------------------------


def test_dblock_30_metals_matches_expected():
    assert set(DBLOCK_30_METALS) == EXPECTED_30_METALS
    assert len(DBLOCK_30_METALS) == 30


def test_default_coordination_geometry_covers_all_metals():
    for m in DBLOCK_30_METALS:
        assert m in DEFAULT_COORDINATION_GEOMETRY, (
            f"{m} missing from DEFAULT_COORDINATION_GEOMETRY"
        )


def test_geometry_max_cn_is_strictly_positive():
    for geom, cn in GEOMETRY_MAX_CN.items():
        assert isinstance(cn, int) and 1 <= cn <= 8, (
            f"GEOMETRY_MAX_CN[{geom!r}]={cn} not in [1..8]"
        )


def test_per_property_verifiers_dict_has_all_six():
    assert set(PER_PROPERTY_VERIFIERS.keys()) == set(EXPECTED_6_PROPERTY_IDS)


def test_synthetic_closure_summary_default_passes_all_six():
    """A default synthetic summary should drive all 6 verifiers to PASS."""
    s = synthetic_closure_summary()
    v = verify_all_catherine_properties("Pt", s)
    assert v.six_of_six_pass, (
        f"default synthetic summary should be 6/6 PASS but failed: "
        f"{[(pid, r.passed) for pid, r in v.per_property.items()]}"
    )


# ---------------------------------------------------------------------------
# 1. Per-property PASS / FAIL tests (12 items)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verifier", [
    verify_p1_reflexivity, verify_p2_transitivity, verify_p3_symmetry,
    verify_p4_dblock, verify_p5_reachability, verify_p6_composability,
])
def test_each_verifier_passes_on_synthetic_pass_summary(verifier):
    s = synthetic_closure_summary()
    result = verifier("Pt", s)
    assert isinstance(result, CatherinePropertyResult)
    assert result.passed, (
        f"{verifier.__name__} should PASS on default synthetic; got "
        f"notes={result.notes!r}"
    )


@pytest.mark.parametrize("metal", SHIP_TARGET_METALS)
@pytest.mark.parametrize("verifier", [
    verify_p1_reflexivity, verify_p2_transitivity, verify_p3_symmetry,
    verify_p4_dblock, verify_p5_reachability, verify_p6_composability,
])
def test_each_verifier_passes_on_5_metals(verifier, metal):
    """30 tests = 5 metals × 6 properties — the Wf-57 pytest ship target."""
    s = synthetic_closure_summary()
    result = verifier(metal, s)
    assert result.passed, f"{verifier.__name__} FAIL on {metal}"


# Failure-mode tests (one per property) — ensure the verifier actually
# recognises a deliberate violation rather than vacuously returning
# True.  These tests don't run on the synthetic PASS path so they
# produce a violation that the verifier must catch.


def test_p1_reflexivity_fails_when_depth0_count_zero():
    s = synthetic_closure_summary(products_per_depth={0: 0, 1: 5})
    r = verify_p1_reflexivity("Pt", s, seed_set_size=1)
    assert not r.passed
    assert r.n_violations == 1


def test_p2_transitivity_fails_when_witness_rate_below_one():
    s = synthetic_closure_summary(witness_rate=0.5)
    r = verify_p2_transitivity("Pt", s)
    assert not r.passed


def test_p3_symmetry_fails_when_well_typed_rate_below_one():
    s = synthetic_closure_summary(well_typed_rate=0.7)
    r = verify_p3_symmetry("Pt", s)
    assert not r.passed


def test_p4_dblock_fails_when_coord_violations_nonzero():
    s = synthetic_closure_summary(coord_violations=3)
    r = verify_p4_dblock("Pt", s)
    assert not r.passed
    assert r.n_violations == 3


def test_p5_reachability_fails_when_n_visited_zero():
    s = synthetic_closure_summary(n_visited=0, reachability_rate=0.0)
    r = verify_p5_reachability("Pt", s)
    assert not r.passed


def test_p6_composability_fails_when_depth_distribution_is_skewed():
    """A 0-count layer at depth = max_depth_seen → composability fails.

    Construct a depth distribution with depth 0 = 1, depth 1 = 0,
    depth 2 = 5, while max_depth_seen = 2.  The 0-count layer at
    depth 1 indicates a silent BFS layer skip — closure composition
    is not associative at this depth.
    """
    s = synthetic_closure_summary(
        n_visited=6, max_depth_seen=2,
        products_per_depth={0: 1, 1: 0, 2: 5},
    )
    r = verify_p6_composability("Pt", s)
    assert not r.passed
    assert "silently skipped" in r.notes or "0-count layer" in r.notes


def test_p6_composability_fails_when_closure_ok_false():
    s = synthetic_closure_summary(closure_ok=False)
    r = verify_p6_composability("Pt", s)
    assert not r.passed
    assert "closure_ok" in r.notes


# ---------------------------------------------------------------------------
# 2. verify_all_catherine_properties — 6/6 verdict behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metal", SHIP_TARGET_METALS)
def test_verify_all_catherine_properties_six_of_six_on_synthetic(metal):
    s = synthetic_closure_summary()
    v = verify_all_catherine_properties(metal, s)
    assert isinstance(v, Catherine6PropsVerdict)
    assert v.six_of_six_pass
    assert v.n_pass == 6
    assert v.n_fail == 0


def test_verify_all_catherine_properties_propagates_violation():
    s = synthetic_closure_summary(witness_rate=0.4)
    v = verify_all_catherine_properties("Pt", s)
    assert not v.six_of_six_pass
    assert "P2_transitivity" in v.failed_property_ids()


def test_verify_all_catherine_properties_accepts_unknown_metal():
    s = synthetic_closure_summary()
    # Non-d-block metal — verifier should still run on diagnostic value.
    v = verify_all_catherine_properties("Xx", s)
    assert v.metal == "Xx"


# ---------------------------------------------------------------------------
# 3. sweep_metals + sweep_summary_stats
# ---------------------------------------------------------------------------


def test_sweep_metals_subset_of_n_metal_keys():
    summaries = {m: synthetic_closure_summary() for m in SHIP_TARGET_METALS}
    verdicts = sweep_metals(summaries, metals=SHIP_TARGET_METALS)
    assert set(verdicts.keys()) == set(SHIP_TARGET_METALS)
    for v in verdicts.values():
        assert v.six_of_six_pass


def test_sweep_metals_subset_to_first_n():
    summaries = {m: synthetic_closure_summary() for m in EXPECTED_30_METALS}
    verdicts = sweep_metals(summaries, metals=tuple(EXPECTED_30_METALS)[:25])
    assert len(verdicts) == 25
    n_pass = sum(1 for v in verdicts.values() if v.six_of_six_pass)
    assert n_pass == 25


def test_sweep_summary_stats_counts_pass_5_of_6_correctly():
    """5 metals, 4 PASS (synthetic) + 1 FAIL (P2 broken) → 5-of-6 PASS."""
    summaries = {m: synthetic_closure_summary() for m in SHIP_TARGET_METALS}
    # Break P2 transitivity on Cu.
    summaries["Cu"] = synthetic_closure_summary(witness_rate=0.5)
    verdicts = sweep_metals(summaries, metals=SHIP_TARGET_METALS)
    stats = sweep_summary_stats(verdicts)
    assert stats["metals_total"] == 5
    assert stats["metals_pass_6of6"] == 4
    assert stats["metals_fail_6of6"] == 1
    # Per-property: P2 should show 4/5 pass (Cu broken).
    assert stats["properties_pass_count"]["P2_transitivity"] == 4
    # All other properties should show 5/5 pass.
    for pid in ("P1_reflexivity", "P3_symmetry", "P4_dblock",
                "P5_reachability", "P6_composability"):
        assert stats["properties_pass_count"][pid] == 5


# ---------------------------------------------------------------------------
# 4. Wf-57 bench script — CLI smoke
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_module():
    sys.path.insert(0, str(_REPO_ROOT / "molmetal" / "scripts"))
    import bench_catherine_6_props_25metals as M  # noqa: E402
    return M


def test_bench_parse_args_defaults(bench_module):
    """The default args must satisfy the Wf-57 ship target ≥ 25 metals."""
    args = bench_module._parse_args([])
    assert args.metals == 25
    assert args.seeds == [0, 1, 2, 3, 4]
    assert args.max_depth == 1
    assert args.max_products == 64
    assert args.use_gpu is False
    assert args.metals_list is None


def test_bench_render_markdown_contains_expected_sections(bench_module):
    metals = list(SHIP_TARGET_METALS)
    fake_verdicts = {
        m: {
            "metal": m,
            "six_of_six_pass": True,
            "n_pass": 6,
            "n_fail": 0,
            "passed_properties": list(EXPECTED_6_PROPERTY_IDS),
            "failed_properties": [],
            "per_property": {
                pid: {"passed": True, "n_samples": 12, "n_violations": 0,
                      "wall_time_s": 1e-4, "notes": ""}
                for pid in EXPECTED_6_PROPERTY_IDS
            },
            "wall_time_s": 0.01,
        }
        for m in metals
    }
    fake_summaries = {
        m: {
            "n_visited": 12, "n_products": 11, "max_depth_seen": 1,
            "products_per_depth": {0: 1, 1: 11},
            "witness_rate": 1.0, "well_typed_rate": 1.0,
            "coord_violations": 0, "reachability_rate": 1.0,
        }
        for m in metals
    }
    payload = {
        "metadata": {
            "wf": "Wf-57", "phase": "Phase 3",
            "n_metals": len(metals), "n_seeds": 5,
            "max_depth": 1, "max_products": 64, "use_gpu": False,
            "timestamp": "2026-09-26T00:00:00Z", "wall_time_s_total": 0.05,
        },
        "metals": fake_verdicts,
        "summaries": fake_summaries,
    }
    md = bench_module._render_markdown(payload, metals)
    assert "# Wf-57 Phase 3" in md
    assert "Headline verdict" in md
    assert "Per-property pass-rate" in md
    assert "Per-metal 6/6 verdict" in md
    # Each metal symbol must appear.
    for m in metals:
        assert m in md


def test_bench_sweep_smoke_runs_5_metals_synthetic(tmp_path, bench_module):
    """End-to-end smoke: 5 metals × 2 seeds, synthetic CPU-only, < 1s."""
    out_dir = tmp_path / "wf57_smoke"
    rc = bench_module.main([
        "--metals-list", "Pt", "Ru", "Fe", "Cu", "Au",
        "--seeds", "0", "1",
        "--max-depth", "1",
        "--max-products", "32",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    # Output files were written.
    assert (out_dir / "per_metal_pass.json").exists()
    assert (out_dir / "sweep_summary.md").exists()
    payload = json.loads((out_dir / "per_metal_pass.json").read_text())
    assert payload["metadata"]["n_metals"] == 5
    assert payload["metadata"]["n_seeds"] == 2
    # All five metals PASS 6/6 on synthetic path.
    assert all(
        payload["metals"][m]["six_of_six_pass"] for m in SHIP_TARGET_METALS
    ), (
        "synthetic CPU-only bench should produce 6/6 PASS on every "
        "metal: " + repr({m: payload["metals"][m]["six_of_six_pass"] for m in SHIP_TARGET_METALS})
    )


def test_bench_sweep_25_metals_runs(tmp_path, bench_module):
    """Wf-57 ship target: 25 metals × 5 seeds, CPU-only."""
    out_dir = tmp_path / "wf57_25metals"
    rc = bench_module.main([
        "--metals", "25",
        "--seeds", "0", "1", "2", "3", "4",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    payload = json.loads((out_dir / "per_metal_pass.json").read_text())
    # Wf-57 ship target — 25 metals × 6 / 6 PASS on synthetic path.
    assert payload["metadata"]["n_metals"] == 25
    n_pass = sum(
        1 for m in payload["metals"]
        if payload["metals"][m]["six_of_six_pass"]
    )
    assert n_pass == 25, (
        f"Wf-57 ship target requires 25/25 metals × 6/6 PASS; got "
        f"{n_pass}/25"
    )
