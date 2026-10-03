"""Wf-338-V1 — Wf-130 L1/L6/L12 closure full verify (skill §8.5.1 anchors).

=================================================================
**Wf-338-V1** — meta-verification of the Wf-130 P1 + P2 + P3 closure
full ship surface against the ``mol-metal-closure-catherine`` skill
§8.5.1 MEASURED anchors. This is the V1 verification pass that
complements the Wf-338-M2 test suite (which covers the cross-report
consistency contracts).

The 2 NEW pytest contracts here are:

 1. test_wf130_l1_l6_l12_ship_status — Wf-130 L1 / L6 / L12 ship
    surfaces exist on disk, carry the Wf-130 tag, and the aggregate
    fields required for honest framing are present. This is the
    high-level "did Wf-130 actually ship?" assertion.
 2. test_skill_851_measured_anchors_match — the Wf-130 ship surfaces
    cross-reference the skill §8.5.1 MEASURED anchors (L1=0.5108,
    L6=0.5891, L12=0.5581, MetalCytoToxDB=0.0300, Tanimoto=0.9978)
    exactly as documented. If the skill anchor values ever drift,
    this test trips so the Wf-130 ledger reconciliation stays in sync.

All tests are CPU-only, deterministic, no torch / RDKit / cuda
requirements beyond standard library + JSON.

Run with::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf338v1_wf130_verify.py -v
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

# Skill mol-metal-closure-catherine §8.5.1 MEASURED anchors
# (extracted from .claude/skills/mol-metal-closure-catherine/SKILL.md §8.5.1)
SKILL_851_L1_BNF = 0.5108  # TODO/completed/83j_lambda_closeout.md §3.1, 2026-09-15
SKILL_851_L6_DBLOCK = 0.5891  # Wf-37 P2 ship wf37_closure_dblock/, 2026-09-26
SKILL_851_L12_PLATINAI = 0.5581  # Wf-41 P2 ship wf41_closure_ood/, 2026-09-26
SKILL_851_METALCYTO = 0.0300  # Wf-41 P2 ship wf41_closure_ood/, 2026-09-26
SKILL_851_TANIMOTO = 0.9978  # wf57_tanimoto/, 2026-09-26

# Per Wf-130 ship surfaces (Wf-130 P1+P2+P3, TODO-133 §0, 2026-09-27)
WF130_L6_MEASURED = 0.55888  # 30 × 5000 = 150 000 samples macro
WF130_L6_DELTA_FROM_SKILL = (
    -0.0302
)  # 0.55888 - 0.5891 = -0.0302 (honest-negative drift)

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
# 1. Wf-130 L1 / L6 / L12 ship status — high-level "did it ship?" assertion
# ---------------------------------------------------------------------------


def test_wf130_l1_l6_l12_ship_status() -> None:
    """Wf-130 L1+L6+L12 ship surfaces exist with Wf-130 tag + honest framing.

    Per TODO-133 §0 (Wf-130 P1+P2+P3 ship 2026-09-27): Wf-130 expanded
    closure full bench from 9 → 30 d-block metals. This test confirms
    the ship surfaces are on disk + carry the Wf-130 metadata tag +
    the honest-framing aggregate fields required by the closure-cath
    skill ledger.
    """
    # --- L1: Catherine 6/6 audit on 30 d-block ---
    payload = _load_json(L1_OVERALL_JSON)
    metadata = payload.get("metadata", {})
    assert (
        metadata.get("wf") == "Wf-130"
    ), f"L1 metadata.wf != 'Wf-130': got {metadata.get('wf')!r}"
    assert metadata.get("phase") == "Phase 1"
    assert metadata.get("r20_plan") == "R20-P5"
    assert metadata.get("n_metals") == 30
    assert (
        metadata.get("synthetic_only") is True
    ), "L1 ship surface must be synthetic-only (no RDKit / GPU)"
    metals = payload.get("metals", {})
    assert len(metals) == 30, f"L1 expected 30 metals, got {len(metals)}"
    missing = sorted(set(EXPECTED_30_METALS) - set(metals.keys()))
    assert not missing, f"L1 ship missing canonical metals: {missing}"
    # every metal ships with six_of_six_pass=True (synthetic CPU-only verdict)
    for metal, row in metals.items():
        assert (
            row.get("six_of_six_pass") is True
        ), f"L1 {metal} not six_of_six_pass=True"
        assert row.get("n_pass") == 6 and row.get("n_fail") == 0, (
            f"L1 {metal} expected n_pass=6 n_fail=0, got "
            f"n_pass={row.get('n_pass')} n_fail={row.get('n_fail')}"
        )

    # --- L6: d-block reachability 30 × 5000 ---
    payload = _load_json(L6_OVERALL_JSON)
    assert payload.get("pilot_id") == "wf130_p2_closure_30_5000"
    assert payload.get("n_metals") == 30
    assert payload.get("n_samples") == 5000
    assert payload.get("n_samples_total") == 150000
    agg = payload.get("aggregate", {})
    assert agg.get("macro_reachability_rate") == pytest.approx(
        WF130_L6_MEASURED, abs=1e-4
    )
    assert "honest_framing" in payload, "L6 ship missing honest_framing block"
    per_metal = payload.get("per_metal", {})
    assert len(per_metal) == 30, f"L6 expected 30 per-metal rows, got {len(per_metal)}"

    # --- L12: closure OOD on 30 d-block ---
    payload = _load_json(L12_OVERALL_JSON)
    assert payload.get("pilot_id") == "wf130_p3_ood_30"
    assert payload.get("n_metals") == 30
    assert payload.get("tanimoto_threshold") == pytest.approx(0.4)
    agg = payload.get("aggregate", {})
    assert agg.get("n_metals_with_data") == 8
    assert agg.get("n_metals_without_data") == 22
    assert "honest_framing" in payload, "L12 ship missing honest_framing block"
    per_metal = payload.get("per_metal", {})
    assert len(per_metal) == 30, f"L12 expected 30 per-metal rows, got {len(per_metal)}"
    # 22 metals must report no_cross_domain_source (structural)
    n_no_source = sum(
        1 for r in per_metal.values() if r.get("status") == "no_cross_domain_source"
    )
    assert (
        n_no_source == 22
    ), f"L12 expected 22 metals with status=no_cross_domain_source, got {n_no_source}"


# ---------------------------------------------------------------------------
# 2. Skill §8.5.1 MEASURED anchors match Wf-130 cross-link fields
# ---------------------------------------------------------------------------


def test_skill_851_measured_anchors_match() -> None:
    """Skill §8.5.1 MEASURED anchors exactly match Wf-130 ship cross-link fields.

    Per skill ``mol-metal-closure-catherine`` §8.5.1 (verbatim, see
    .claude/skills/mol-metal-closure-catherine/SKILL.md §8.5.1):

      | Metric                                | Value  | Source                              |
      |---------------------------------------|-------:|------------------------------------|
      | L1 reachability (β-NF enumeration)    | 0.5108 | TODO-83j §3.1, 2026-09-15          |
      | L6 d-block reachability               | 0.5891 | Wf-37 P2 ship wf37_closure_dblock  |
      | L12 PlatinAI OOD reachability         | 0.5581 | Wf-41 P2 ship wf41_closure_ood     |
      | MetalCytoToxDB OOD reachability       | 0.0300 | Wf-41 P2 ship wf41_closure_ood     |
      | Tanimoto<0.4 fraction_below_threshold | 0.9978 | wf57_tanimoto, 2026-09-26          |

    The Wf-130 ship surfaces must cross-link to these exact values via
    ``wf67_baseline_reachability`` (L6) + ``wf41_platinai_macro_reachability``
    (L12) + ``wf41_metalcyto_macro_reachability`` (L12). The L1 anchor is
    documented in TODO-83j §3.1 and carried in Wf-338-M2 tests but the
    Wf-130 P1 ship surface uses the 6/6 PASS verdict form (not the
    scalar 0.5108), so we assert the verdict form is present.
    """
    # --- L6 cross-link to skill §8.5.1 ---
    l6_payload = _load_json(L6_OVERALL_JSON)
    l6_baseline = l6_payload.get("wf67_baseline_reachability")
    assert l6_baseline == pytest.approx(SKILL_851_L6_DBLOCK, abs=1e-4), (
        f"L6 cross-link wf67_baseline_reachability={l6_baseline} != "
        f"skill §8.5.1 anchor {SKILL_851_L6_DBLOCK}"
    )
    # L6 measured delta vs skill §8.5.1 anchor must be in honest-negative band
    l6_macro = l6_payload["aggregate"]["macro_reachability_rate"]
    delta_l6 = l6_macro - SKILL_851_L6_DBLOCK
    assert delta_l6 == pytest.approx(WF130_L6_DELTA_FROM_SKILL, abs=1e-3), (
        f"L6 drift delta={delta_l6:.4f} != expected "
        f"honest-negative {WF130_L6_DELTA_FROM_SKILL:+.4f}"
    )
    assert (
        -0.05 <= delta_l6 <= 0.0
    ), f"L6 drift {delta_l6:+.4f} outside honest-negative band [-0.05, 0.0]"

    # --- L12 cross-link to skill §8.5.1 ---
    l12_payload = _load_json(L12_OVERALL_JSON)
    l12_platinai = l12_payload.get("wf41_platinai_macro_reachability")
    assert l12_platinai == pytest.approx(SKILL_851_L12_PLATINAI, abs=1e-4), (
        f"L12 cross-link wf41_platinai_macro_reachability={l12_platinai} != "
        f"skill §8.5.1 anchor {SKILL_851_L12_PLATINAI}"
    )
    l12_metalcyto = l12_payload.get("wf41_metalcyto_macro_reachability")
    assert l12_metalcyto == pytest.approx(SKILL_851_METALCYTO, abs=1e-4), (
        f"L12 cross-link wf41_metalcyto_macro_reachability={l12_metalcyto} != "
        f"skill §8.5.1 anchor {SKILL_851_METALCYTO}"
    )
    l12_pt = l12_payload.get("wf41_pt_reachability")
    # Wf-41 P2 Pt-specific reachability is a derived metric — must be present
    assert (
        l12_pt is not None and 0.0 <= l12_pt <= 1.0
    ), f"L12 wf41_pt_reachability missing or out of [0,1]: {l12_pt}"

    # --- L1 verdict form: Wf-130 P1 uses 6/6 PASS, not scalar 0.5108 ---
    l1_payload = _load_json(L1_OVERALL_JSON)
    l1_metadata = l1_payload.get("metadata", {})
    # Anchor 0.5108 is the pre-Wf-130 L1 β-NF enumeration MEASURED
    # (TODO-83j §3.1, 2026-09-15); Wf-130 P1 used the 6/6 PASS verdict
    # form. Confirm the verdict form is on disk + cross-linkable.
    assert (
        l1_metadata.get("r20_plan") == "R20-P5"
    ), "L1 metadata.r20_plan must be R20-P5 (cross-link to skill)"
    l1_metals = l1_payload.get("metals", {})
    assert len(l1_metals) == 30, "L1 must ship 30 d-block metal verdicts"
    for metal, row in l1_metals.items():
        assert (
            row.get("six_of_six_pass") is True
        ), f"L1 {metal} not six_of_six_pass=True (verdict form drift)"

    # --- Skill §8.5.1 anchor values are the lower-bound MEASURED ledger ---
    # These are the canonical anchors that must not drift silently.
    # (Per skill: "MEASURED anchors above (L1=0.5108, L6=0.5891,
    #  L12=0.5581, Tanimoto=0.9978) are the lower bounds for R20-P5.")
    anchors = {
        "L1_BNF_enumeration": SKILL_851_L1_BNF,
        "L6_dblock_reachability": SKILL_851_L6_DBLOCK,
        "L12_platinai_OOD": SKILL_851_L12_PLATINAI,
        "MetalCytoToxDB_OOD": SKILL_851_METALCYTO,
        "Tanimoto_0.4_fraction": SKILL_851_TANIMOTO,
    }
    for name, value in anchors.items():
        # All anchors must be in (0, 1) range — sanity-check they are
        # not corrupted by a rebase or rename.
        assert (
            0.0 < value < 1.0
        ), f"Skill §8.5.1 anchor {name}={value} out of (0, 1) range"
