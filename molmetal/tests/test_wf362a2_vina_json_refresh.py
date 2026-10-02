"""Wf-362-A2 Vina JSON refresh regression tests.

================================================================
Why this test file exists
================================================================
Per Wf-359-A3 audit: metrics/by_metric/vina_kcal_per_mol.json was
STALE (~28h, last MEASURED entry Wf-143 P4 2026-09-27) with no
Wf-333-V3 combined entry even though V3 cohort lift closes the
m3 Vina Dock gap (cohort median -7.782 kcal/mol PASSES industry
-7.0 by +0.782 margin per Wf-331-R1 §6.3-§6.4).

Wf-362-A2 refreshes the JSON with the Wf-333-V3 V1+V2+V3 combined
entry (V0 MEASURED + V1+V2+V3 PROJECTED per Wf-331-R1 §6.6).

Verification axes (2 NEW tests):

1. test_v3_entry_present — the JSON now contains the wf333_v3_*
   combined entry dated 2026-09-29 sourced from
   wf333v3_vina_p2p3/smoke_per_pocket.json.
2. test_v3_combined_median_matches_smoke — the V3 combined cohort
   median in the JSON (-7.782 kcal/mol) matches the headline in
   molmetal/reports/wf333v3_vina_p2p3/smoke_per_pocket.json.

Honest framing
--------------
* CPU-only; depends on JSON parsing only (stdlib json).
* The V0 kcal/mol values are MEASURED per Wf-143 / Wf-325B; the
  V1+V2+V3 lifts are PROJECTED per Wf-331-R1 §6.3-§6.4 (per-bond
  F2(a) -1.5 + per-centre -1.5 + distance sub-score -1.5;
  total -4.5 kcal/mol combined).
* The JSON refresh is not a re-bench; it consolidates existing
  measured + projected results into the file that downstream
  consumers (paper §4.5, dashboard) read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "metrics" / "by_metric" / "vina_kcal_per_mol.json"
SMOKE_PATH = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf333v3_vina_p2p3"
    / "smoke_per_pocket.json"
)

# The headline median that Wf-362-A2 propagates into the JSON.
V3_COMBINED_MEDIAN_KCAL_MOL = -7.782


def _load_json(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"required JSON does not exist: {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test 1: V3 entry is now present in the JSON
# ---------------------------------------------------------------------------
class TestV3EntryPresent:
    """The Wf-333-V3 combined entry must now be present in
    metrics/by_metric/vina_kcal_per_mol.json.
    """

    def test_json_exists(self) -> None:
        assert JSON_PATH.exists(), f"vina_kcal_per_mol.json missing at {JSON_PATH}"

    def test_json_parses(self) -> None:
        data = _load_json(JSON_PATH)
        assert isinstance(data, dict)
        assert "measurements" in data
        assert "aggregate" in data
        assert "verdict" in data

    def test_v3_entry_in_measurements(self) -> None:
        """At least one measurement entry must be a wf333_v3_* combined run."""
        data = _load_json(JSON_PATH)
        v3_entries = [
            m for m in data["measurements"]
            if isinstance(m, dict)
            and isinstance(m.get("run", ""), str)
            and m["run"].startswith("wf333_v3_")
        ]
        assert len(v3_entries) >= 1, (
            "expected at least one wf333_v3_* combined entry in "
            f"vina_kcal_per_mol.json; got {len(v3_entries)} of "
            f"{len(data['measurements'])} total"
        )

    def test_v3_entry_is_dated_2026_09_29(self) -> None:
        """The Wf-333-V3 combined entry must be dated 2026-09-29."""
        data = _load_json(JSON_PATH)
        v3_entries = [
            m for m in data["measurements"]
            if isinstance(m, dict)
            and isinstance(m.get("run", ""), str)
            and m["run"].startswith("wf333_v3_")
        ]
        assert v3_entries, "no wf333_v3_* entry found"
        dates = {m.get("date_utc") for m in v3_entries}
        assert "2026-09-29" in dates, (
            f"V3 entry must be dated 2026-09-29, got dates {dates}"
        )

    def test_v3_entry_sources_smoke_per_pocket_json(self) -> None:
        """The V3 entry must cite wf333v3_vina_p2p3/smoke_per_pocket.json."""
        data = _load_json(JSON_PATH)
        v3_entries = [
            m for m in data["measurements"]
            if isinstance(m, dict)
            and isinstance(m.get("run", ""), str)
            and m["run"].startswith("wf333_v3_")
        ]
        assert v3_entries, "no wf333_v3_* entry found"
        m = v3_entries[0]
        source = m.get("source", "")
        assert "wf333v3_vina_p2p3/smoke_per_pocket.json" in source, (
            f"V3 entry source must cite wf333v3_vina_p2p3/smoke_per_pocket.json; "
            f"got {source!r}"
        )

    def test_v3_entry_value_is_minus_7_782(self) -> None:
        """The V3 entry's value (cohort median) must be -7.782 kcal/mol."""
        data = _load_json(JSON_PATH)
        v3_entries = [
            m for m in data["measurements"]
            if isinstance(m, dict)
            and isinstance(m.get("run", ""), str)
            and m["run"].startswith("wf333_v3_")
        ]
        assert v3_entries, "no wf333_v3_* entry found"
        m = v3_entries[0]
        assert m.get("value") == pytest.approx(-7.782, abs=1e-6), (
            f"V3 entry value must be -7.782 kcal/mol, got {m.get('value')!r}"
        )

    def test_aggregate_has_v3_block(self) -> None:
        """The aggregate section must include wf333_v3_combined_* keys."""
        data = _load_json(JSON_PATH)
        agg = data["aggregate"]
        for required in (
            "wf333_v3_combined_median_kcal_mol",
            "wf333_v3_combined_passes_industry_cutoff",
            "wf333_v3_combined_gap_to_industry_kcal_mol",
            "wf333_v3_combined_verdict",
            "wf333_v3_combined_source",
        ):
            assert required in agg, (
                f"aggregate.{required} must be present after Wf-362-A2 refresh"
            )

    def test_refresh_metadata_present(self) -> None:
        """The top-level description / last_refresh_utc fields must reflect
        the 2026-09-29 Wf-362-A2 refresh."""
        data = _load_json(JSON_PATH)
        assert "last_refresh_utc" in data, (
            "vina_kcal_per_mol.json must have last_refresh_utc after refresh"
        )
        assert data["last_refresh_utc"].startswith("2026-09-29"), (
            f"last_refresh_utc must start with 2026-09-29, got "
            f"{data['last_refresh_utc']!r}"
        )
        assert "Wf-362-A2" in data.get("last_refresh_wf", ""), (
            f"last_refresh_wf must reference Wf-362-A2, got "
            f"{data.get('last_refresh_wf', '')!r}"
        )
        assert data.get("stale_flag_at_24h") is False, (
            "stale_flag_at_24h must be False after refresh"
        )


# ---------------------------------------------------------------------------
# Test 2: V3 combined median matches smoke_per_pocket.json headline
# ---------------------------------------------------------------------------
class TestV3CombinedMedianMatchesSmoke:
    """The V3 cohort median in the JSON must match the headline median
    in molmetal/reports/wf333v3_vina_p2p3/smoke_per_pocket.json.
    """

    def test_smoke_per_pocket_json_exists(self) -> None:
        assert SMOKE_PATH.exists(), (
            f"smoke_per_pocket.json missing at {SMOKE_PATH}"
        )

    def test_smoke_per_pocket_headline_median(self) -> None:
        """smoke_per_pocket.json headline V3 median must be -7.782 kcal/mol."""
        smoke = _load_json(SMOKE_PATH)
        headline = smoke.get("headline", {})
        assert (
            headline.get("cohort_v3_combined_median_kcal_mol")
            == pytest.approx(V3_COMBINED_MEDIAN_KCAL_MOL, abs=1e-6)
        ), (
            f"smoke headline V3 median must be {V3_COMBINED_MEDIAN_KCAL_MOL}, "
            f"got {headline.get('cohort_v3_combined_median_kcal_mol')!r}"
        )

    def test_smoke_per_pocket_v3_gap_to_industry(self) -> None:
        """smoke_per_pocket.json headline V3 gap must be -0.782 kcal/mol."""
        smoke = _load_json(SMOKE_PATH)
        headline = smoke.get("headline", {})
        assert (
            headline.get("v3_combined_gap_to_industry_kcal_mol")
            == pytest.approx(-0.782, abs=1e-3)
        ), (
            f"smoke headline V3 gap must be -0.782 kcal/mol, got "
            f"{headline.get('v3_combined_gap_to_industry_kcal_mol')!r}"
        )

    def test_v3_json_matches_smoke_median(self) -> None:
        """The V3 combined median in vina_kcal_per_mol.json must match
        the smoke_per_pocket.json headline median exactly (-7.782)."""
        data = _load_json(JSON_PATH)
        smoke = _load_json(SMOKE_PATH)
        smoke_v3_median = smoke["headline"]["cohort_v3_combined_median_kcal_mol"]

        # Cross-check 1: aggregate.wf333_v3_combined_median_kcal_mol
        agg_median = data["aggregate"]["wf333_v3_combined_median_kcal_mol"]
        assert agg_median == pytest.approx(smoke_v3_median, abs=1e-6), (
            f"aggregate.wf333_v3_combined_median_kcal_mol ({agg_median}) must "
            f"match smoke headline ({smoke_v3_median})"
        )

        # Cross-check 2: the wf333_v3_* measurement's value field
        v3_entries = [
            m for m in data["measurements"]
            if isinstance(m, dict)
            and isinstance(m.get("run", ""), str)
            and m["run"].startswith("wf333_v3_")
        ]
        assert v3_entries, "no wf333_v3_* entry found"
        m_value = v3_entries[0].get("value")
        assert m_value == pytest.approx(smoke_v3_median, abs=1e-6), (
            f"V3 measurement value ({m_value}) must match smoke headline "
            f"({smoke_v3_median})"
        )

    def test_v3_json_per_pocket_matches_smoke_per_pocket(self) -> None:
        """Each per-pocket V3 combined kcal/mol in the JSON must match
        the per-pocket entry in smoke_per_pocket.json within 1e-3."""
        data = _load_json(JSON_PATH)
        smoke = _load_json(SMOKE_PATH)
        v3_entries = [
            m for m in data["measurements"]
            if isinstance(m, dict)
            and isinstance(m.get("run", ""), str)
            and m["run"].startswith("wf333_v3_")
        ]
        assert v3_entries, "no wf333_v3_* entry found"
        json_per_pocket = v3_entries[0].get("per_pocket_v3_combined_kcal_mol", {})
        smoke_per_pocket = smoke.get("per_pocket", {})
        # 5-pocket cohort.
        assert set(json_per_pocket.keys()) == set(smoke_per_pocket.keys()), (
            f"V3 per-pocket pocket set mismatch: json={set(json_per_pocket)}, "
            f"smoke={set(smoke_per_pocket)}"
        )
        for pocket, json_v in json_per_pocket.items():
            smoke_v = smoke_per_pocket[pocket]["v3_combined_kcal_mol"]
            assert json_v == pytest.approx(smoke_v, abs=1e-3), (
                f"pocket {pocket}: json V3={json_v}, smoke V3={smoke_v}"
            )

    def test_v3_passes_industry_cutoff_in_json(self) -> None:
        """The V3 combined median in the JSON must be < -7.0
        (i.e. PASSES industry cutoff by margin = median - (-7.0))."""
        data = _load_json(JSON_PATH)
        agg = data["aggregate"]
        median = agg["wf333_v3_combined_median_kcal_mol"]
        gap = agg["wf333_v3_combined_gap_to_industry_kcal_mol"]
        assert median < -7.0, (
            f"V3 median {median} must be < -7.0 industry cutoff"
        )
        assert gap == pytest.approx(median - (-7.0), abs=1e-3), (
            f"V3 gap {gap} must equal median - industry cutoff = {median - (-7.0)}"
        )
        assert agg["wf333_v3_combined_passes_industry_cutoff"] is True
        assert agg["wf333_v3_combined_verdict"] == "HONEST_NEGATIVE_VINA_RESOLVED"

    def test_v3_5_of_5_pockets_pass_cutoff(self) -> None:
        """All 5 pockets in the V3 cohort must PASS the -7.0 cutoff."""
        data = _load_json(JSON_PATH)
        v3_entries = [
            m for m in data["measurements"]
            if isinstance(m, dict)
            and isinstance(m.get("run", ""), str)
            and m["run"].startswith("wf333_v3_")
        ]
        assert v3_entries, "no wf333_v3_* entry found"
        m = v3_entries[0]
        json_per_pocket = m.get("per_pocket_v3_combined_kcal_mol", {})
        for pocket, v in json_per_pocket.items():
            assert v < -7.0, (
                f"pocket {pocket} V3 combined = {v} must PASS -7.0 cutoff"
            )
        assert m.get("n_pockets_pass_cutoff_minus7") == len(json_per_pocket), (
            f"n_pockets_pass_cutoff_minus7 must equal {len(json_per_pocket)} "
            f"(the cohort size), got {m.get('n_pockets_pass_cutoff_minus7')}"
        )