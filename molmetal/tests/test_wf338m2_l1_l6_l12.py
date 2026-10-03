"""Wf-338-M2 — L1 / L6 / L12 closure full bench verification suite.

=================================================================
**Wf-338-M2** — cross-cutting verification of the Wf-130 P1+P2+P3
L1+L6+L12 closure full bench ship surface (per skill
``mol-metal-closure-catherine`` §8.5.1 MEASURED anchors + §8.5.3
ledger reconciliation).

The 4 shipped Wf-130 sub-reports are:

  * L1  — ``molmetal/reports/wf130_p1_catherine_30/``        (Catherine 6/6 on 30 d-block)
  * L6  — ``molmetal/reports/wf130_p2_closure_30_5000/``     (d-block reachability 30 × 5000)
  * L12 — ``molmetal/reports/wf130_p3_ood_30/``             (PlatinAI OOD on 30 d-block)

This file ships **4 pytest contracts** that pin the cross-cutting
behaviour of the Wf-130 P1+P2+P3 ship surface:

 1. test_l1_closure_beta_nf_enumeration — L1 Catherine 6/6 audit
    on 30 d-block metals ships on disk; every per-metal JSON is
    ``six_of_six_pass=True``; 30 / 30 metals present.
 2. test_l6_closure_dblock_reachability — L6 closure reachability
    on 30 × 5000 samples ships on disk; macro reachability between
    Wf-67 P2 baseline 0.5891 and Wf-130 P2 measured 0.5589 (the
        honest-partial verdict per TODO-133 §B).
 3. test_l12_closure_platinai_ood — L12 closure OOD on 30 d-block
    ships on disk; at least 8 metals with cross-domain source and
    honest framing markers recorded.
 4. test_per_metal_reachability_consistency — per-metal reachability
    schema consistency across the 3 Wf-130 sub-reports; each per-metal
    JSON has the required keys; aggregate macros are present.

All tests are CPU-only, deterministic, no torch / RDKit / cuda
requirements beyond standard library + the existing audit / bench
script modules.

Run with::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf338m2_l1_l6_l12.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_ROOT = PROJECT_ROOT / "molmetal" / "reports"

L1_REPORT_DIR = REPORT_ROOT / "wf130_p1_catherine_30"
L6_REPORT_DIR = REPORT_ROOT / "wf130_p2_closure_30_5000"
L12_REPORT_DIR = REPORT_ROOT / "wf130_p3_ood_30"

L1_OVERALL_JSON = L1_REPORT_DIR / "catherine_audit_30metals.json"
L6_OVERALL_JSON = L6_REPORT_DIR / "reachability_overall.json"
L12_OVERALL_JSON = L12_REPORT_DIR / "ood_reachability_overall.json"

# Per skill §8.5.1 MEASURED anchors (Wf-67 P2 + Wf-41 P2 baseline; pre-Wf-130)
SKILL_L1_ANCHOR = 0.5108  # β-NF enumeration (TODO-83j §3.1, 2026-09-15)
SKILL_L6_ANCHOR = 0.5891  # Wf-37 P2 d-block reachability, 2026-09-26
SKILL_L12_ANCHOR = 0.5581  # Wf-41 P2 PlatinAI OOD, 2026-09-26

# Wf-130 measured (TODO-133 §0)
WF130_L6_MEASURED = 0.5589  # 30 × 5000 = 150 000 samples; macro 0.5589 ± 0.1526
WF130_L12_METALS_WITH_DATA = 8  # 8/30 cross-domain (Pt/Pd/Ru/Rh/Ir/Os/Au/Re)

# Honest-negative / gate thresholds (carried from existing bench scripts)
L6_GATE = 0.85
L6_HONEST_NEGATIVE = 0.50
L12_GATE = 0.30
L12_HONEST_NEGATIVE = 0.30

EXPECTED_30_METALS = [
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
]


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file with a clear error if it is missing."""
    if not path.is_file():
        pytest.fail(f"required Wf-130 ship surface missing: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. L1 closure (β-NF enumeration) — Catherine 6/6 on 30 d-block metals
# ---------------------------------------------------------------------------


def test_l1_closure_beta_nf_enumeration() -> None:
    """L1 Catherine 6/6 audit on 30 d-block metals — every metal 6/6 PASS.

    Per skill §8.5.1 + TODO-133 §A: Wf-130 P1 lifted L1 from 25 → 30
    d-block metals on the synthetic CPU-only path. The 30/30 × 6/6
    verdict must be on disk.
    """
    payload = _load_json(L1_OVERALL_JSON)

    # metadata sanity
    metadata = payload.get("metadata", {})
    assert metadata.get("wf") == "Wf-130", f"unexpected wf tag: {metadata.get('wf')}"
    assert metadata.get("n_metals") == 30
    assert metadata.get("synthetic_only") is True
    assert metadata.get("r20_plan") == "R20-P5"

    metals = payload.get("metals", {})
    assert len(metals) == 30, f"L1 expected 30 metals, got {len(metals)}"

    # All 30 canonical d-block metals must be present
    missing = sorted(set(EXPECTED_30_METALS) - set(metals.keys()))
    assert not missing, f"L1 audit missing metals: {missing}"

    # Every metal must be 6/6 PASS on the synthetic CPU-only path
    n_pass = 0
    for metal, row in metals.items():
        assert (
            row.get("six_of_six_pass") is True
        ), f"L1 audit FAIL on {metal}: expected six_of_six_pass=True"
        assert row.get("n_pass") == 6, f"L1 {metal} n_pass != 6"
        assert row.get("n_fail") == 0, f"L1 {metal} n_fail != 0"
        n_pass += 1
    assert n_pass == 30


# ---------------------------------------------------------------------------
# 2. L6 closure (d-block reachability) — 30 × 5000 = 150 000 samples
# ---------------------------------------------------------------------------


def test_l6_closure_dblock_reachability() -> None:
    """L6 closure reachability on 30 d-block × 5000 samples — on disk + in range.

    Per TODO-133 §B: Wf-130 P2 measured macro_reachability = 0.5589
    on 150 000 samples. The number is honest-partial (below L6_GATE
    0.85) but consistent with Wf-67 P2 baseline 0.5891 (skill §8.5.1)
    minus the structural 30-metal dilution (Δ = -0.0302).
    """
    payload = _load_json(L6_OVERALL_JSON)

    # metadata sanity
    assert payload.get("pilot_id") == "wf130_p2_closure_30_5000"
    assert payload.get("n_metals") == 30
    assert payload.get("n_samples") == 5000
    assert payload.get("n_samples_total") == 150000

    # WF-67 baseline cross-link
    assert payload.get("wf67_baseline_reachability") == pytest.approx(SKILL_L6_ANCHOR)
    assert payload.get("gate_target_reachability") == pytest.approx(L6_GATE)

    # aggregate carries macro_reachability_rate in expected range
    agg = payload.get("aggregate", {})
    macro = agg.get("macro_reachability_rate")
    assert macro is not None, "L6 aggregate missing macro_reachability_rate"
    # Wf-130 P2 measured 0.5589; allow ± 0.02 tolerance for any future re-runs
    assert L6_HONEST_NEGATIVE <= macro < L6_GATE, (
        f"L6 macro {macro} not in honest-negative band "
        f"[{L6_HONEST_NEGATIVE}, {L6_GATE}); gate PASS unexpected"
    )
    # tighter equality check vs recorded Wf-130 P2 measurement
    assert macro == pytest.approx(
        WF130_L6_MEASURED, abs=1e-4
    ), f"L6 macro {macro} != Wf-130 P2 measured {WF130_L6_MEASURED}"

    # aggregate honest framing
    assert payload.get("overall_gate_pass") is False
    assert payload.get("n_metals_pass") == 0
    n_below = payload.get("n_metals_below_honest_negative_threshold", 0)
    assert 0 < n_below <= 30

    # per-metal breakdown carries 30 metals with reachability rates
    per_metal = payload.get("per_metal", {})
    assert len(per_metal) == 30, f"L6 expected 30 per-metal rows, got {len(per_metal)}"
    # every per-metal row must have closure_reachability_rate in [0, 1]
    for metal, row in per_metal.items():
        rate = row.get("closure_reachability_rate")
        assert (
            rate is not None
        ), f"L6 per_metal[{metal}] missing closure_reachability_rate"
        assert 0.0 <= rate <= 1.0, f"L6 per_metal[{metal}] rate out of range: {rate}"

    # Per-metal highlights per TODO-133 §B: Zn 0.8180, La 0.8000, Cu 0.7678
    zn_rate = per_metal.get("Zn", {}).get("closure_reachability_rate")
    assert zn_rate == pytest.approx(
        0.8180, abs=1e-3
    ), f"L6 Zn reachability changed from 0.8180 to {zn_rate}"


# ---------------------------------------------------------------------------
# 3. L12 closure (PlatinAI OOD) — 30 d-block metals
# ---------------------------------------------------------------------------


def test_l12_closure_platinai_ood() -> None:
    """L12 closure OOD on 30 d-block metals — 8/30 with cross-domain data.

    Per TODO-133 §C: Wf-130 P3 measured 8/30 metals with PlatinAI +
    MetalCytoTox cross-domain source. The 22 metals without source
    are reported ``no_cross_domain_source`` — structural, not a
    measurement artifact. CPU-only + no-RDKit fallback returns
    0 reachability on the 8-metals subset (Wf-41 P2 RDKit rate is the
    MEASURED truth).
    """
    payload = _load_json(L12_OVERALL_JSON)

    # metadata sanity
    assert payload.get("pilot_id") == "wf130_p3_ood_30"
    assert payload.get("n_metals") == 30
    assert payload.get("tanimoto_threshold") == pytest.approx(0.4)

    # Wf-41 P2 baselines (skill §8.5.1 cross-link)
    assert payload.get("wf41_pt_reachability") == pytest.approx(0.5833)
    assert payload.get("wf41_platinai_macro_reachability") == pytest.approx(
        SKILL_L12_ANCHOR
    )
    assert payload.get("wf41_metalcyto_macro_reachability") == pytest.approx(0.0300)

    # Aggregate: 8 metals with data, 22 without, macro 0.0 (CPU-only no-RDKit)
    agg = payload.get("aggregate", {})
    assert agg.get("n_metals_with_data") == WF130_L12_METALS_WITH_DATA
    assert agg.get("n_metals_without_data") == 22

    # Honest framing gate markers
    assert payload.get("gate_target_per_metal_reachability") == pytest.approx(L12_GATE)
    assert payload.get("honest_negative_threshold") == pytest.approx(
        L12_HONEST_NEGATIVE
    )
    assert payload.get("overall_gate_pass") is False

    # Honest framing block must be present
    assert "honest_framing" in payload, "L12 missing honest_framing field"

    # Per-metal breakdown: 30 rows with status ok or no_cross_domain_source
    per_metal = payload.get("per_metal", {})
    assert len(per_metal) == 30, f"L12 expected 30 per-metal rows, got {len(per_metal)}"

    n_with_data = 0
    n_no_source = 0
    cross_domain_metals = {"Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Au", "Re"}
    for metal, row in per_metal.items():
        status = row.get("status")
        if status == "no_cross_domain_source":
            n_no_source += 1
        elif status == "ok":
            n_with_data += 1
            assert (
                metal in cross_domain_metals
            ), f"L12 cross-domain metal {metal} not in canonical 8"
        else:
            pytest.fail(f"L12 per_metal[{metal}] unexpected status: {status}")
    assert n_with_data == WF130_L12_METALS_WITH_DATA
    assert n_no_source == 22


# ---------------------------------------------------------------------------
# 4. Per-metal reachability consistency across L1 + L6 + L12 ship surfaces
# ---------------------------------------------------------------------------


def test_per_metal_reachability_consistency() -> None:
    """Cross-report per-metal consistency — all 30 metals present in L1+L6+L12.

    Verifies the Wf-130 P1 + P2 + P3 ship surfaces cover the same 30
    canonical d-block metals, and the aggregate macros + honest-framing
    fields are present in every report. This is the cross-cutting
    ledger reconciliation per skill §8.5.3.
    """
    # L1
    l1_payload = _load_json(L1_OVERALL_JSON)
    l1_metals = set(l1_payload.get("metals", {}).keys())

    # L6
    l6_payload = _load_json(L6_OVERALL_JSON)
    l6_per_metal = l6_payload.get("per_metal", {})
    l6_metals = set(l6_per_metal.keys())

    # L12
    l12_payload = _load_json(L12_OVERALL_JSON)
    l12_per_metal = l12_payload.get("per_metal", {})
    l12_metals = set(l12_per_metal.keys())

    canonical = set(EXPECTED_30_METALS)

    # every report covers the full 30 d-block set
    assert l1_metals == canonical, (
        f"L1 metal set differs from canonical: missing={sorted(canonical - l1_metals)}, "
        f"extra={sorted(l1_metals - canonical)}"
    )
    assert l6_metals == canonical, (
        f"L6 metal set differs from canonical: missing={sorted(canonical - l6_metals)}, "
        f"extra={sorted(l6_metals - canonical)}"
    )
    assert l12_metals == canonical, (
        f"L12 metal set differs from canonical: missing={sorted(canonical - l12_metals)}, "
        f"extra={sorted(l12_metals - canonical)}"
    )

    # Per-metal reachability schema consistency (L6 + L12)
    for metal in canonical:
        l6_row = l6_per_metal[metal]
        assert (
            "closure_reachability_rate" in l6_row
        ), f"L6 {metal} missing closure_reachability_rate"
        assert "n_samples" in l6_row, f"L6 {metal} missing n_samples"
        assert "n_closure_ok" in l6_row, f"L6 {metal} missing n_closure_ok"

        l12_row = l12_per_metal[metal]
        # L12 row may have closure_reachability_rate=None for no_cross_domain_source metals
        assert "status" in l12_row, f"L12 {metal} missing status"
        assert "n_input" in l12_row, f"L12 {metal} missing n_input"

    # Cross-skill anchor (skill §8.5.1): L6 Wf-67 P2 baseline 0.5891 → Wf-130 P2 0.5589
    # Delta = -0.0302 (honest-partial).
    l6_macro = l6_payload["aggregate"]["macro_reachability_rate"]
    delta_l6 = l6_macro - SKILL_L6_ANCHOR
    assert -0.05 <= delta_l6 <= 0.0, (
        f"L6 delta vs Wf-67 P2 baseline {SKILL_L6_ANCHOR} is {delta_l6:+.4f}; "
        f"expected honest-negative drift in [{-0.05}, 0]"
    )

    # Honest framing blocks must be present in L6 + L12 (L1 has metadata.synthetic_only)
    assert l6_payload.get("honest_framing"), "L6 missing honest_framing"
    assert l12_payload.get("honest_framing"), "L12 missing honest_framing"
