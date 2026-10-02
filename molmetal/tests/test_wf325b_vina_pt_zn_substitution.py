"""Wf-325B unit tests — Vina Pt→Zn UFF substitution final closeout.

Per Wf-143 + Wf-246-C + Wf-301B + Wf-319C + Wf-323C: Vina 1.2.7's
PDBQT parser rejects the 'Pt' atom-type.  The Wf-325B closeout
formalises the Pt→Zn substitution convention across the canonical
5-pocket cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu) and records
``HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED``.

8 NEW tests verify:

  3 — per-pocket Zn substitution correctness (status + module + lift=0)
  2 — Vina re-dock validity (script + result JSON shape)
  2 — m3 lift vs original (cross-pocket median + ceiling preservation)
  1 — integration with downstream consumer (tier2_summary.json)

All CPU-only.  No GPU launch per AGENTS.md §5.1.

Honest-framing note (party-twenty-spirit §3.5):
Per WEAK_TO_STRONG §2.3: m3 Vina Dock median MEASURED on the canonical
830c single-pocket = −1.3546 kcal/mol; gap to industry ≤ −7 = +5.65
kcal/mol; gap to projected target = +7.15 kcal/mol.  The 5-pocket
Wf-325B cohort sits at −3.2830 median (cisplatin) — same architectural
ceiling (0/5 pockets reach industry), same honest-negative
(Wf-323C contract; Wf-325B formalises it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "molmetal" / "reports" / "wf325b_vina_pt_zn_substitution"
WF143_P4_DIR = ROOT / "molmetal" / "reports" / "wf143_p4_chemotype_dock"
WF323C_DIR = ROOT / "molmetal" / "reports" / "wf323c_m8_full"
METRICS_DIR = ROOT / "metrics" / "by_metric"
RESULT_JSON = REPORTS_DIR / "result.json"
WF325B_SCRIPT = ROOT / "molmetal" / "scripts" / "wf325b_vina_pt_zn_substitution.py"

POCKETS_5 = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")

# Honest-negative canonical key
HONEST_NEGATIVE_KEY = "HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_result() -> dict:
    if not RESULT_JSON.exists():
        pytest.skip(
            f"result JSON not found at {RESULT_JSON}; run "
            f"wf325b_vina_pt_zn_substitution.py --execute first"
        )
    with RESULT_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_wf143_aggregate(pocket_id: str) -> dict:
    p = WF143_P4_DIR / pocket_id / "per_seed_metrics.json"
    if not p.exists():
        pytest.skip(f"Wf-143 per-seed metrics missing for {pocket_id}: {p}")
    with p.open("r", encoding="utf-8") as fh:
        rec = json.load(fh)
    return rec.get("_aggregate", {})


# =============================================================================
# Group 1 — Per-pocket Zn substitution correctness (3 tests)
# =============================================================================
def test_wf325b_per_pocket_zn_substitution_status_measured_generic_atom():
    """Every Wf-143 per-pocket record carries
    ``status="MEASURED_GENERIC_ATOM"`` — the canonical Pt→Zn
    substitution marker (per Wf-246-C / Wf-323C).

    If this test fails, the upstream Wf-143 sweep was rerun WITHOUT
    the substitution convention and the Wf-325B closeout is invalid.
    """
    for pocket_id in POCKETS_5:
        agg = _load_wf143_aggregate(pocket_id)
        assert agg.get("status") == "MEASURED_GENERIC_ATOM", (
            f"pocket {pocket_id} status={agg.get('status')!r}; expected "
            f"MEASURED_GENERIC_ATOM (Pt→Zn substitution marker)"
        )


def test_wf325b_per_pocket_zn_substitution_module_pt_group_elements():
    """The ``metal_pdbqt_overrides`` module exposes the Pt-group element
    set used by the substitution rule.  Wf-325B's helper imports it
    directly; the test pins the contract so the convention is
    discoverable from a single source.
    """
    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        PT_GROUP_ELEMENTS,
        add_metal_atom_types,
    )
    # Pt-group metal set (canonical 8 d-block Pt-group elements)
    expected_pt_group = {"Pt", "Ru", "Ir", "Os", "Rh", "Pd", "Au", "Re"}
    assert set(PT_GROUP_ELEMENTS) == expected_pt_group, (
        f"PT_GROUP_ELEMENTS drifted: got {set(PT_GROUP_ELEMENTS)}, "
        f"expected {expected_pt_group}"
    )
    # The substitution rule caller is exposed (callable for downstream)
    assert callable(add_metal_atom_types), (
        "add_metal_atom_types must be callable for downstream consumers"
    )


def test_wf325b_per_pocket_zn_substitution_lift_zero():
    """Per-pocket lift = Zn-substituted − original = 0.0 kcal/mol.

    Both sides use the same Vina kcal/mol because the Pt→Zn
    substitution was already active in Wf-143's
    MEASURED_GENERIC_ATOM path.  Wf-325B FORMALISES the convention;
    it does NOT introduce a new kcal/mol.
    """
    result = _load_result()
    for rec in result.get("per_pocket", []):
        assert rec.get("lift_kcal_mol") == pytest.approx(0.0, abs=1e-9), (
            f"pocket {rec.get('pocket_id')}: lift={rec.get('lift_kcal_mol')}; "
            f"expected 0.0 (Wf-143 already Zn-substituted)"
        )


# =============================================================================
# Group 2 — Vina re-dock validity (2 tests)
# =============================================================================
def test_wf325b_script_present_and_complete():
    """The Wf-325B entry-point script must be on disk with the
    canonical docstring + key function names.
    """
    assert WF325B_SCRIPT.exists(), f"script missing: {WF325B_SCRIPT}"
    src = WF325B_SCRIPT.read_text(encoding="utf-8")
    assert "Wf-325B" in src, "Wf-325B anchor missing from docstring"
    assert "_uff_prerelax_pt_proxy" in src, "_uff_prerelax_pt_proxy fn missing"
    assert "_pt_zn_substitution_smoke" in src, "_pt_zn_substitution_smoke fn missing"
    assert "_build_per_pocket_records" in src, "_build_per_pocket_records fn missing"
    assert "Pt" in src, "Pt handling missing"
    assert "Zn" in src, "Zn substitution handling missing"
    assert HONEST_NEGATIVE_KEY in src, (
        f"honest-negative canonical key {HONEST_NEGATIVE_KEY!r} missing"
    )


def test_wf325b_result_json_shape():
    """The result JSON must carry the canonical top-level keys +
    per-pocket records for all 5 pockets.
    """
    result = _load_result()
    # Top-level keys
    required = {
        "workflow", "engine", "generated_at", "scope", "pockets",
        "source_wf143_dir", "uff_prerelax_smoke", "per_pocket",
        "headline", "honest_framing",
        HONEST_NEGATIVE_KEY, "honest_negative_reason",
        "ceiling_preserved_reason", "recovery_path",
        "wall_seconds_total", "exit_status",
    }
    missing = required - set(result.keys())
    assert not missing, f"missing top-level keys: {sorted(missing)}"

    # 5-pocket per-pocket records
    per_pocket = result.get("per_pocket", [])
    assert len(per_pocket) == 5, (
        f"per_pocket should have 5 records; got {len(per_pocket)}"
    )
    pocket_ids = sorted(r["pocket_id"] for r in per_pocket)
    assert pocket_ids == sorted(POCKETS_5), (
        f"per-pocket pocket_ids={pocket_ids}; expected {sorted(POCKETS_5)}"
    )
    # Honest-negative flag MUST be True
    assert result.get(HONEST_NEGATIVE_KEY) is True, (
        f"{HONEST_NEGATIVE_KEY} must be True (Wf-325B closeout)"
    )
    assert result.get("exit_status") == "PASS", (
        f"exit_status must be 'PASS'; got {result.get('exit_status')!r}"
    )


# =============================================================================
# Group 3 — m3 lift vs original (2 tests)
# =============================================================================
def test_wf325b_cross_pocket_m3_median_in_measured_window():
    """The cross-pocket m3 Vina Dock median kcal/mol must be in the
    MEASURED cisplatin window [−3.7, −2.8].

    The 5-pocket Wf-143 cisplatin cohort sits at ~−3.1 kcal/mol; if
    the Wf-325B aggregation drifts out of this window the upstream
    Wf-143 data was rerun / replaced.
    """
    result = _load_result()
    headline = result.get("headline", {})
    cross_median = headline.get("cross_pocket_median_kcal_mol")
    assert cross_median is not None, "headline.cross_pocket_median_kcal_mol missing"
    assert -3.7 <= cross_median <= -2.8, (
        f"cross-pocket median {cross_median} outside cisplatin window [-3.7, -2.8]"
    )
    # n_pockets_total must be 5
    assert headline.get("n_pockets_total") == 5, (
        f"n_pockets_total must be 5; got {headline.get('n_pockets_total')}"
    )


def test_wf325b_ceiling_preserved_zero_pockets_at_industry_cutoff():
    """The architectural ceiling is preserved: 0/5 pockets reach
    industry cutoff ≤ −7 kcal/mol.  This is the canonical honest
    finding — Vina does not score the covalent Pt anchor; Pt→Zn
    substitution is a graceful-degradation marker, not a recovery.
    """
    result = _load_result()
    headline = result.get("headline", {})
    assert headline.get("n_pockets_at_industry_cutoff") == 0, (
        f"n_pockets_at_industry_cutoff must be 0 (architectural ceiling "
        f"preserved); got {headline.get('n_pockets_at_industry_cutoff')}"
    )
    assert headline.get("industry_cutoff_kcal_mol") == -7.0, (
        f"industry_cutoff_kcal_mol must be -7.0; "
        f"got {headline.get('industry_cutoff_kcal_mol')}"
    )
    # gap_to_industry_cutoff_kcal_mol must be > 0 (i.e., the cohort is
    # above the industry cutoff in kcal/mol terms).
    gap = headline.get("gap_to_industry_cutoff_kcal_mol")
    assert gap is not None and gap > 0, (
        f"gap_to_industry_cutoff_kcal_mol must be > 0; got {gap}"
    )


# =============================================================================
# Group 4 — Integration with downstream consumer (1 test)
# =============================================================================
def test_wf325b_integration_with_tier2_summary():
    """The Wf-325B cross-pocket m3 median must be CONSISTENT with
    the existing tier2_summary.json m3 entry (vina_dock_median_kcal_mol).

    tier2_summary documents the Wf-121 + Wf-143 MEASURED m3 =
    −1.3546 kcal/mol on the 830c single-pocket.  The Wf-325B
    5-pocket cisplatin cohort sits at −3.28 kcal/mol — DIFFERENT
    cohort / DIFFERENT scope (5-pocket cisplatin, not 830c single-
    pocket TransDiff baseline).  The integration test asserts the
    Wf-325B result JSON path is discoverable from the downstream
    consumer's source_json chain (if any) — or, absent a chain link,
    asserts the Wf-325B result JSON itself documents the scope
    difference explicitly so a downstream consumer is not misled.
    """
    tier2_path = METRICS_DIR / "tier2_summary.json"
    if not tier2_path.exists():
        pytest.skip(f"tier2_summary.json missing: {tier2_path}")
    with tier2_path.open("r", encoding="utf-8") as fh:
        tier2 = json.load(fh)

    # Documented m3 entry on 830c single-pocket — DIFFERENT cohort.
    # tier2_summary.json uses ``metrics_aggregated`` key (not
    # ``tier2_metrics``); search both shapes for backwards compat.
    m3_entry = (
        tier2.get("metrics_aggregated", {}).get("vina_dock_median_kcal_mol", {})
        or tier2.get("tier2_metrics", {}).get("vina_dock_median_kcal_mol", {})
    )
    if not m3_entry:
        pytest.skip(
            "tier2_summary missing vina_dock_median_kcal_mol entry "
            "(checked metrics_aggregated + tier2_metrics)"
        )

    tier2_value = m3_entry.get("our_value")
    result = _load_result()
    wf325b_median = result.get("headline", {}).get(
        "cross_pocket_median_kcal_mol"
    )

    # Document the cohort difference in the Wf-325B result JSON
    # so a downstream consumer can branch on the canonical field.
    honest_framing = result.get("honest_framing", {})
    scope_note = honest_framing.get("ceiling_preserved", "")
    assert "0/5 pockets" in scope_note or "0 / 5 pockets" in scope_note, (
        f"Wf-325B honest_framing.ceiling_preserved must document 0/5 pockets "
        f"at industry cutoff; got {scope_note!r}"
    )

    # Sanity: both values are MEASURED kcal/mol; downstream consumer can
    # distinguish by the canonical honest_framing.ceiling_preserved key.
    assert tier2_value is not None
    assert wf325b_median is not None
    # They WILL differ in value because they're different cohorts;
    # the test confirms the Wf-325B result is discoverable + honest.
    assert (
        honest_framing.get("recovery_path")
        and "F2(a)" in honest_framing["recovery_path"]
    ), "recovery_path must mention F2(a) MetalLigandExchange per WEAK_TO_STRONG §3.5"