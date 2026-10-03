"""Tests for Wf-339-C4 per-metal reachability matrix update (M25 + 30-metal + OOD).

================================================================
Wf-339-C4: per-metal reachability matrix contract.

This file ships the **4 pytest contracts** that pin the per-metal
reachability matrix used by the closure theorem d-block + cross-domain
OOD bench:

 1. test_per_metal_l6_baseline          — per-metal MEASURED closure_reachability_rate
                                          on the 30 d-block metals (Wf-37 P2 / L6)
 2. test_per_metal_ood_baseline         — per-metal MEASURED cross-domain OOD
                                          closure_reachability_rate (Wf-41 P2 / L12)
 3. test_m25_hilbert_attribution        — M25 Hilbert primal-dual attribution
                                          (PROJECTED; per party-twenty-spirit §3.5)
 4. test_per_periodic_block_split       — per-periodic-block aggregate
                                          (3d / 4d / 5d_La)

All tests are CPU-only, deterministic, and complete in < 30 s combined.

The MEASURED values cited below trace back to:
- molmetal/reports/wf37_closure_dblock/per_metal_bench.md
- molmetal/reports/wf37_closure_dblock/per_metal_bench.json
- molmetal/reports/wf37_closure_dblock/per_metal.csv
- molmetal/reports/wf41_closure_ood/ood_closure.md
- molmetal/reports/wf41_closure_ood/ood_closure.json

Per-party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED tag discipline):
- L6 + L12 baselines = MEASURED (Wf-37 P2 + Wf-41 P2)
- M25 Hilbert attribution = PROJECTED (L5 Hilbert-ASTSR R² ≥ 0.85 not yet
  MEASURED; Wf-94 R20-P2 / TODO-93 is the architectural fix path).

Run with::

    PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
        molmetal/tests/test_wf339c4_per_metal_reachability.py -v
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PER_METAL_BENCH_JSON = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf37_closure_dblock"
    / "per_metal_bench.json"
)
PER_METAL_CSV = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf37_closure_dblock" / "per_metal.csv"
)
PER_METAL_MD = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf37_closure_dblock" / "per_metal_bench.md"
)
OOD_CLOSURE_JSON = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf41_closure_ood" / "ood_closure.json"
)
OOD_CLOSURE_MD = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf41_closure_ood" / "ood_closure.md"
)


# ---------------------------------------------------------------------------
# Per-metal MEASURED baseline (Wf-37 P2 / L6 d-block bench)
# Source: per_metal_bench.md §Per-metal reachability rate
# ---------------------------------------------------------------------------
# Only the 9 task-spec metals are pinned here; full 30-metal matrix is
# in per_metal.csv.
L6_BASELINE: Dict[str, float] = {
    "Pt": 0.4700,
    "Pd": 0.4500,
    "Ru": 0.7200,
    "Rh": 0.5500,
    "Ir": 0.3800,
    "Os": 0.4000,
    "Au": 0.4400,
    "Ag": 0.6173,
    "Re": 0.5400,
}


# ---------------------------------------------------------------------------
# Per-(domain, metal) MEASURED baseline (Wf-41 P2 / L12 OOD bench)
# Source: ood_closure.md §Per-(domain, metal) reachability rate
# ---------------------------------------------------------------------------
L12_BASELINE: Dict[str, Dict[str, float]] = {
    "platinai_5k": {
        "Pt": 0.5833,
        "Pd": 0.3333,
        "Ru": 0.7000,
        "Rh": 0.6471,
        "Ir": 0.6000,
        "Os": 0.5294,
        "Au": 0.4545,
    },
    "metalcytotox_5metal": {
        "Ru": 0.1500,
        "Ir": 0.0000,
        "Rh": 0.0000,
        "Os": 0.0000,
        "Re": 0.0000,
    },
}


# ---------------------------------------------------------------------------
# Per-periodic-block MEASURED baseline (Wf-37 P2 / L6 d-block bench)
# Source: per_metal_bench.md §Per-periodic-block aggregate
# ---------------------------------------------------------------------------
PER_BLOCK_BASELINE: Dict[str, float] = {
    "3d": 0.6682,
    "4d": 0.5813,
    "5d_La": 0.4918,
}


# ---------------------------------------------------------------------------
# M25 Hilbert primal-dual attribution — PROJECTED (per honest framing)
#
# The M25 Hilbert-ASTSR R² ≥ 0.85 target is the architectural fix for the
# 12-d RDKit 2D signal floor (TODO-179 STATUS_HOLD + Wf-313G HONEST_NEGATIVE
# captured). It is NOT yet MEASURED; the Hilbert-augmented reachability
# values listed below are placeholders that will become MEASURED only
# after Wf-94 R20-P2 (TODO-93) ships.
#
# Per party-twenty-spirit §3.5, every PROJECTED value MUST be flagged.
# ---------------------------------------------------------------------------
M25_HILBERT_ATTRIBUTION_STATUS = "PROJECTED"  # NOT yet MEASURED
M25_HILBERT_ATTRIBUTION_TARGET_R2 = 0.85  # Wf-94 R20-P2 / TODO-93 target
M25_HILBERT_ATTRIBUTION_NOTE = (
    "M25 Hilbert-ASTSR attribution is PROJECTED per party-twenty-spirit §3.5. "
    "L5 R² ≥ 0.85 not yet MEASURED (12-d RDKit 2D signal floor per "
    "TODO/completed/177_wf313g_downstream_lift_dashboard.md §3.4); "
    "Wf-94 R20-P2 Multi-PH + PLL is the architectural fix path."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_per_metal_bench() -> Dict[str, Any]:
    """Load per_metal_bench.json (Wf-37 P2 L6 d-block bench)."""
    assert (
        PER_METAL_BENCH_JSON.is_file()
    ), f"per_metal_bench.json missing at {PER_METAL_BENCH_JSON}"
    return json.loads(PER_METAL_BENCH_JSON.read_text())


def _load_ood_closure() -> Dict[str, Any]:
    """Load ood_closure.json (Wf-41 P2 L12 cross-domain OOD bench)."""
    assert OOD_CLOSURE_JSON.is_file(), f"ood_closure.json missing at {OOD_CLOSURE_JSON}"
    return json.loads(OOD_CLOSURE_JSON.read_text())


def _load_per_metal_csv() -> Dict[str, Dict[str, Any]]:
    """Load per_metal.csv as a metal → row dict (Wf-37 P2 L6 d-block bench)."""
    assert PER_METAL_CSV.is_file(), f"per_metal.csv missing at {PER_METAL_CSV}"
    out: Dict[str, Dict[str, Any]] = {}
    with PER_METAL_CSV.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metal = row["metal"]
            out[metal] = dict(row)
    return out


# ---------------------------------------------------------------------------
# 1. Per-metal L6 baseline (30 d-block metals, MEASURED)
# ---------------------------------------------------------------------------
def test_per_metal_l6_baseline() -> None:
    """Per-metal MEASURED closure_reachability_rate on the 30 d-block
    metals (Wf-37 P2 / L6 bench).

    Asserts each of the 9 task-spec metals' value matches the Wf-37
    P2 ship artefact (within 1e-3 tolerance for JSON round-trip) and
    that every value lies in the closed unit interval [0, 1].

    Honest framing (party-twenty-spirit §3.5):
    - MEASURED today (Wf-37 P2, 2026-09-26).
    - Source: molmetal/reports/wf37_closure_dblock/per_metal_bench.md §
      Per-metal reachability rate; per_metal.csv; per_metal_bench.json.
    """
    bench = _load_per_metal_bench()
    csv_rows = _load_per_metal_csv()

    assert "per_metal" in bench, "per_metal_bench.json missing per_metal"
    bench_per_metal = bench["per_metal"]

    for metal, expected_rate in L6_BASELINE.items():
        # JSON aggregate (truth source)
        agg = bench_per_metal.get(metal, {}).get("aggregate", {})
        assert agg, f"per_metal_bench.json missing aggregate for {metal!r}"
        actual_rate_json = float(agg["closure_reachability_rate"])
        assert abs(actual_rate_json - expected_rate) < 1e-3, (
            f"L6 {metal}: expected {expected_rate}, got {actual_rate_json} "
            f"from per_metal_bench.json"
        )
        # CSV row (secondary source)
        csv_row = csv_rows.get(metal, {})
        assert csv_row, f"per_metal.csv missing row for {metal!r}"
        actual_rate_csv = float(csv_row["closure_reachability_rate"])
        assert abs(actual_rate_csv - expected_rate) < 1e-3, (
            f"L6 {metal}: expected {expected_rate}, got {actual_rate_csv} "
            f"from per_metal.csv"
        )
        # Bounded in [0, 1]
        assert (
            0.0 <= actual_rate_json <= 1.0
        ), f"L6 {metal}: rate {actual_rate_json} out of [0,1]"


# ---------------------------------------------------------------------------
# 2. Per-metal OOD L12 baseline (PlatinAI 5K + MetalCytoToxDB 5-metal, MEASURED)
# ---------------------------------------------------------------------------
def test_per_metal_ood_baseline() -> None:
    """Per-(domain, metal) MEASURED closure_reachability_rate on the
    OOD-filtered cross-domain cohort (Wf-41 P2 / L12 bench).

    Asserts each documented (domain, metal) pair's value matches the
    Wf-41 P2 ship artefact (within 1e-3 tolerance) and that every
    value lies in the closed unit interval [0, 1].

    Honest framing (party-twenty-spirit §3.5):
    - MEASURED today (Wf-41 P2, 2026-09-26).
    - Source: molmetal/reports/wf41_closure_ood/ood_closure.md §
      Per-(domain, metal) reachability rate; ood_closure.json.
    - MetalCytoTox 5-metal Ru 0.1500 + Ir/Rh/Os/Re 0.0000 are NOT
      failures — they reflect the MetalCytoTox data format (cyto
      activity labels, not d-block organometallics with dative bonds)
      that the closure BFS rejects as under-saturated. Per
      per_metal_bench.md §Honest framing, this is valence-BNF
      rejection, not a closure theorem bug.
    """
    bench = _load_ood_closure()

    assert "per_domain" in bench, "ood_closure.json missing per_domain"
    per_domain = bench["per_domain"]

    for domain, metal_rates in L12_BASELINE.items():
        assert domain in per_domain, f"ood_closure.json missing domain {domain!r}"
        per_metal = per_domain[domain].get("per_metal", {})
        for metal, expected_rate in metal_rates.items():
            agg = per_metal.get(metal, {}).get("aggregate", {})
            assert agg, (
                f"ood_closure.json missing aggregate for " f"{domain!r}/{metal!r}"
            )
            actual_rate = float(agg["closure_reachability_rate"])
            assert abs(actual_rate - expected_rate) < 1e-3, (
                f"L12 {domain}/{metal}: expected {expected_rate}, " f"got {actual_rate}"
            )
            assert (
                0.0 <= actual_rate <= 1.0
            ), f"L12 {domain}/{metal}: rate {actual_rate} out of [0,1]"


# ---------------------------------------------------------------------------
# 3. M25 Hilbert-ASTSR attribution — PROJECTED (per honest framing)
# ---------------------------------------------------------------------------
def test_m25_hilbert_attribution() -> None:
    """M25 Hilbert primal-dual attribution contract.

    Asserts that:
      (a) The attribution status is flagged PROJECTED (NOT MEASURED)
          per party-twenty-spirit §3.5.
      (b) The M25 attribution target R² ≥ 0.85 is consistent with
          Wf-94 R20-P2 (TODO-93) — the architectural fix path that
          replaces the 12-d RDKit 2D signal floor (TODO-179
          STATUS_HOLD + Wf-313G HONEST_NEGATIVE captured).
      (c) The note explicitly cites the source of the honest-negative
          framing (TODO/completed/177_wf313g_downstream_lift_dashboard.md
          §3.4) and the architectural fix (Wf-94 R20-P2).

    This test does NOT fabricate a Hilbert-augmented reachability
    value. Per party-twenty-spirit §3.5, do NOT cite L5 R² ≥ 0.85 as
    MEASURED until the architectural fix ships.

    Honest framing (party-twenty-spirit §3.5):
    - **STATUS**: PROJECTED (NOT YET MEASURED).
    - **Target**: R² ≥ 0.85 (Wf-94 R20-P2 / TODO-93).
    - **MEASURED today** (Wf-122 P3 + Wf-129 P3):
      - L5 Hilbert R² 4-cell: -5.66 ± 1.36
        (TODO-122 L5 / metrics/by_metric/hilbert_4cell_r2.json)
      - Wf-129 P3 Multi-PH + PLL guarded: -0.0780 ± 0.0148
    - **Block**: 12-d RDKit 2D feature ceiling |ρ| ≤ 0.019 →
      linear R² upper bound = 0.0043 (TODO/completed/137_*).
    - **Fix path**: Wf-94 R20-P2 Multi-PH + PLL real feature
      augmentation (32-bin Fourier + 128-d PLL signature).
    """
    # (a) Status flagged PROJECTED, not MEASURED
    assert M25_HILBERT_ATTRIBUTION_STATUS == "PROJECTED", (
        f"M25 Hilbert attribution must be PROJECTED; got "
        f"{M25_HILBERT_ATTRIBUTION_STATUS}"
    )
    # (b) Target R² ≥ 0.85
    assert M25_HILBERT_ATTRIBUTION_TARGET_R2 == 0.85, (
        f"M25 Hilbert target R² must be 0.85 (Wf-94 R20-P2); got "
        f"{M25_HILBERT_ATTRIBUTION_TARGET_R2}"
    )
    # (c) Note cites Wf-313G dashboard + Wf-94 R20-P2 + signal floor
    assert "PROJECTED" in M25_HILBERT_ATTRIBUTION_NOTE
    assert "wf313g" in M25_HILBERT_ATTRIBUTION_NOTE.lower()
    assert "wf-94" in M25_HILBERT_ATTRIBUTION_NOTE.lower()
    assert "12-d" in M25_HILBERT_ATTRIBUTION_NOTE  # signal floor cited
    assert "multi-ph" in M25_HILBERT_ATTRIBUTION_NOTE.lower()


# ---------------------------------------------------------------------------
# 4. Per-periodic-block split (3d / 4d / 5d_La, MEASURED)
# ---------------------------------------------------------------------------
def test_per_periodic_block_split() -> None:
    """Per-periodic-block MEASURED closure_reachability_rate (3d / 4d / 5d_La).

    Asserts each block's value matches the Wf-37 P2 ship artefact
    (within 1e-3 tolerance) and that the block split sums to the
    expected partition (3d: 892 + 4d: 683 + 5d_La: 669 = 2244
    samples total).

    Honest framing (party-twenty-spirit §3.5):
    - MEASURED today (Wf-37 P2, 2026-09-26).
    - Source: molmetal/reports/wf37_closure_dblock/per_metal_bench.md §
      Per-periodic-block aggregate; per_metal_bench.json §per_block.
    - Trend: 3d (0.6682) > 4d (0.5813) > 5d_La (0.4918) — i.e.,
      reachability decreases with period. Reported as MEASURED, NOT
      PROJECTED.
    """
    bench = _load_per_metal_bench()
    assert "per_block" in bench, "per_metal_bench.json missing per_block"
    per_block = bench["per_block"]

    for block, expected_rate in PER_BLOCK_BASELINE.items():
        assert block in per_block, f"per_metal_bench.json missing block {block!r}"
        agg = per_block[block]
        actual_rate = float(agg["closure_reachability_rate"])
        assert abs(actual_rate - expected_rate) < 1e-3, (
            f"per-block {block}: expected {expected_rate}, got " f"{actual_rate}"
        )
        assert (
            0.0 <= actual_rate <= 1.0
        ), f"per-block {block}: rate {actual_rate} out of [0,1]"

    # Partition sanity: 3d + 4d + 5d_La = 2244 total samples
    n_total = sum(int(per_block[b]["n_samples"]) for b in PER_BLOCK_BASELINE)
    assert n_total == 2244, (
        f"per-block n_samples must partition into 2244 total; got " f"{n_total}"
    )
    # Trend sanity: 3d > 4d > 5d_La (reachability decreases with period)
    assert (
        PER_BLOCK_BASELINE["3d"]
        > PER_BLOCK_BASELINE["4d"]
        > PER_BLOCK_BASELINE["5d_La"]
    ), f"per-block trend must be 3d > 4d > 5d_La; got " f"{PER_BLOCK_BASELINE}"


# ---------------------------------------------------------------------------
# Bonus invariants — sanity check the source artefacts are consistent
# ---------------------------------------------------------------------------
def test_per_metal_csv_consistent_with_json() -> None:
    """per_metal.csv closure_reachability_rate matches per_metal_bench.json."""
    bench = _load_per_metal_bench()
    csv_rows = _load_per_metal_csv()
    bench_per_metal = bench["per_metal"]

    assert set(csv_rows.keys()) == set(bench_per_metal.keys()), (
        f"per_metal.csv metals {set(csv_rows.keys())} != "
        f"per_metal_bench.json per_metal metals "
        f"{set(bench_per_metal.keys())}"
    )

    for metal in csv_rows:
        json_rate = float(
            bench_per_metal[metal]["aggregate"]["closure_reachability_rate"]
        )
        csv_rate = float(csv_rows[metal]["closure_reachability_rate"])
        assert (
            abs(json_rate - csv_rate) < 1e-3
        ), f"{metal}: json={json_rate} != csv={csv_rate}"


def test_md_artefacts_present() -> None:
    """The Markdown summaries for Wf-37 P2 + Wf-41 P2 ship on disk."""
    assert PER_METAL_MD.is_file(), f"per_metal_bench.md missing at {PER_METAL_MD}"
    assert OOD_CLOSURE_MD.is_file(), f"ood_closure.md missing at {OOD_CLOSURE_MD}"
    # Quick text sniff
    pmd = PER_METAL_MD.read_text()
    assert "Per-metal reachability rate" in pmd
    assert "Per-periodic-block aggregate" in pmd
    omd = OOD_CLOSURE_MD.read_text()
    assert "Per-(domain, metal) reachability rate" in omd


def test_l6_l12_metal_coverage_disjoint() -> None:
    """L6 (30 d-block) superset of L12 (7 PlatinAI + 5 MetalCytoTox = 12).
    All L12 metals must appear in L6 metal set.
    """
    csv_rows = _load_per_metal_csv()
    l6_metals = set(csv_rows.keys())
    l12_metals: set = set()
    for domain_rates in L12_BASELINE.values():
        l12_metals.update(domain_rates.keys())
    missing = l12_metals - l6_metals
    assert not missing, f"L12 metals missing from L6 30-metal set: {sorted(missing)}"


def test_ood_metalcyto_zero_rates_documented() -> None:
    """MetalCytoToxDB 5-metal Ir/Rh/Os/Re zero rates are documented
    honest-negative (per ood_closure.md §Honest framing, valence-BNF
    rejection — NOT closure theorem bug).
    """
    bench = _load_ood_closure()
    per_metal = bench["per_domain"]["metalcytotox_5metal"]["per_metal"]
    for metal in ("Ir", "Rh", "Os", "Re"):
        rate = float(per_metal[metal]["aggregate"]["closure_reachability_rate"])
        assert rate == 0.0, (
            f"metalcytotox_5metal/{metal} expected 0.0 (valence-BNF "
            f"rejection per ood_closure.md §Honest framing); got {rate}"
        )
