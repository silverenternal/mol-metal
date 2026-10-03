"""Tests for Wf-68 Phase 1 (R20-P5.1) — Catherine 6/6 properties on >= 25 metals.

==============================================================================

Pytest suite for the Wf-68 P5.1 verifier + per-metal output surface.

Test coverage
=============

* **Smoke tests (5 tests)** — the verifier CLI runs, the script imports,
  the Wf-57 helpers degrade gracefully, the per-metal output paths exist.

* **Per-metal 6/6 PASS tests (3 tests)** — sweeping the synthetic path
  produces a 6/6 PASS verdict on the Wf-57 default 30-metal list
  (parity with Wf-57 Phase 3 ship target).

* **Per-property tests (8 tests)** — one PASS test + one FAIL-on-violation
  test for each of the 6 Catherine properties (12 tests in this row).

* **Back-link + metadata tests (2 tests)** — the combined
  ``catherine_6_props.json`` carries the Wf-57 back-link + the per-metal
  ``bench_summary.json`` carries the Wf-68 P5.1 metadata.

**Total**: ~20 pytest items, all designed to PASS on a vanilla Python
install (no torch / RDKit / cuda requirements).

Notes
-----
The synthetic-summary path is the primary test target; the test suite
runs in <1s on a slim environment and never requires the real closure
BFS.  This is the Wf-68 P5.1 CPU-only ship target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Repo-root path injection — matches the verifier script.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts import verify_catherine_25_metals as v68


# ---------------------------------------------------------------------------
# Loader shim — load catherine_properties directly to bypass the
# torch-dependent molmetal.math_skills.__init__ chain on slim envs.
# This duplicates the verifier's _verify_via_direct_import logic but
# keeps the test file independent of the verifier implementation.
# ---------------------------------------------------------------------------

def _load_catherine_properties():
    """Load catherine_properties.py directly without triggering math_skills.__init__."""
    import importlib.util
    cp_path = (
        Path(__file__).resolve().parent.parent
        / "math_skills"
        / "catherine_properties.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_catherine_properties_test", str(cp_path)
    )
    cp = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, cp)
    spec.loader.exec_module(cp)
    return cp


_cp = _load_catherine_properties()
DBLOCK_30_METALS = _cp.DBLOCK_30_METALS
PER_PROPERTY_VERIFIERS = _cp.PER_PROPERTY_VERIFIERS
sweep_metals = _cp.sweep_metals
sweep_summary_stats = _cp.sweep_summary_stats
synthetic_closure_summary = _cp.synthetic_closure_summary
verify_all_catherine_properties = _cp.verify_all_catherine_properties

# ---------------------------------------------------------------------------
# Smoke tests — script imports + constants + helpers
# ---------------------------------------------------------------------------


def test_verify_catherine_25_metals_imports():
    """The verifier module imports without error and exposes the public API."""
    assert hasattr(v68, "main")
    assert hasattr(v68, "_run_metal")
    assert hasattr(v68, "_write_per_metal")
    assert hasattr(v68, "_render_synthesis")
    assert hasattr(v68, "_parse_args")


def test_default_30_metals_constant():
    """The verifier ships a 30-metal default list matching Wf-57 / catherine_properties."""
    assert len(v68.DEFAULT_30_METALS) == 30
    assert v68.DEFAULT_30_METALS == DBLOCK_30_METALS


def test_catherine_properties_constant():
    """The verifier ships all 6 Catherine properties P1..P6."""
    assert len(v68.CATHERINE_PROPERTIES) == 6
    assert "P1_reflexivity" in v68.CATHERINE_PROPERTIES
    assert "P6_composability" in v68.CATHERINE_PROPERTIES


def test_wf57_helpers_load_returns_dict():
    """``_load_wf57_helpers`` returns a dict (possibly empty in slim env)."""
    helpers = v68._load_wf57_helpers()
    assert isinstance(helpers, dict)
    assert "available" in helpers


def test_utcnow_iso_format():
    """``_utcnow_iso`` returns a Z-suffixed ISO 8601 string."""
    ts = v68._utcnow_iso()
    assert ts.endswith("Z")
    assert "T" in ts
    assert len(ts) >= 20


# ---------------------------------------------------------------------------
# Per-metal 6/6 PASS tests — synthetic path
# ---------------------------------------------------------------------------


def _synthetic_summaries_for(metals):
    return {m: synthetic_closure_summary() for m in metals}


def test_sweep_25_metals_synthetic_pass():
    """Synthetic sweep over the first 25 d-block metals yields 25/25 6/6 PASS."""
    metals = DBLOCK_30_METALS[:25]
    summaries = _synthetic_summaries_for(metals)
    verdicts = sweep_metals(summaries, metals)
    stats = sweep_summary_stats(verdicts)
    assert stats["metals_total"] == 25
    assert stats["metals_pass_6of6"] == 25
    assert stats["metals_fail_6of6"] == 0
    for pid in PER_PROPERTY_VERIFIERS:
        assert stats["properties_pass_count"][pid] == 25
        assert stats["properties_pass_rate"][pid] == 1.0


def test_sweep_30_metals_synthetic_pass():
    """Synthetic sweep over all 30 d-block metals yields 30/30 6/6 PASS."""
    metals = list(DBLOCK_30_METALS)
    summaries = _synthetic_summaries_for(metals)
    verdicts = sweep_metals(summaries, metals)
    stats = sweep_summary_stats(verdicts)
    assert stats["metals_total"] == 30
    assert stats["metals_pass_6of6"] == 30


def test_single_metal_synthetic_pass():
    """``_run_metal`` synthetic path returns 6/6 PASS for any metal."""
    summary, verdict, synth = v68._run_metal(
        "Pt",
        seeds=[0],
        max_depth=1,
        max_products=32,
        partner_terms=[],
        helpers={"available": False},
        synthetic_only=True,
    )
    assert synth is True
    assert verdict["six_of_six_pass"] is True
    assert verdict["n_pass"] == 6
    assert verdict["n_fail"] == 0
    for pid in PER_PROPERTY_VERIFIERS:
        assert verdict["per_property"][pid]["passed"] is True


# ---------------------------------------------------------------------------
# Per-property tests — direct verifier-level (parity with Wf-57 tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("property_id", PER_PROPERTY_VERIFIERS)
def test_property_pass_synthetic(property_id):
    """Each Catherine property PASSes on a synthetic PASS-friendly summary."""
    summary = synthetic_closure_summary()
    verdict = verify_all_catherine_properties("Pt", summary)
    assert verdict.per_property[property_id].passed is True


@pytest.mark.parametrize("property_id", PER_PROPERTY_VERIFIERS)
def test_property_fail_propagates(property_id, monkeypatch):
    """When a per-property verifier returns FAIL, the verdict reflects it."""
    def _fail(_, __):
        return _cp.CatherinePropertyResult(
            property_id=property_id,
            metal="Pt",
            passed=False,
            n_samples=0,
            n_violations=1,
            notes="forced failure",
        )

    monkeypatch.setitem(_cp.PER_PROPERTY_VERIFIERS, property_id, _fail)
    summary = synthetic_closure_summary()
    verdict = verify_all_catherine_properties("Pt", summary)
    assert verdict.six_of_six_pass is False
    assert property_id in verdict.failed_property_ids()


# ---------------------------------------------------------------------------
# Back-link + metadata tests
# ---------------------------------------------------------------------------


def test_combined_metadata_shape(tmp_path):
    """``_write_per_metal`` writes both catherine_props.json + bench_summary.json."""
    summary = synthetic_closure_summary()
    summary["synthetic"] = True
    verdict_obj = verify_all_catherine_properties("Pt", summary)
    verdict = {
        "metal": "Pt",
        "six_of_six_pass": verdict_obj.six_of_six_pass,
        "n_pass": verdict_obj.n_pass,
        "n_fail": verdict_obj.n_fail,
        "passed_properties": verdict_obj.passed_property_ids(),
        "failed_properties": verdict_obj.failed_property_ids(),
        "per_property": {
            pid: {
                "passed": r.passed,
                "n_samples": r.n_samples,
                "n_violations": r.n_violations,
                "wall_time_s": r.wall_time_s,
                "notes": r.notes,
            }
            for pid, r in verdict_obj.per_property.items()
        },
        "wall_time_s": verdict_obj.wall_time_s,
    }
    metadata = {
        "wf": "Wf-68",
        "phase": "Phase 1",
        "r20_plan": "R20-P5.1",
        "n_seeds": 1,
        "max_depth": 1,
        "max_products": 32,
    }
    row = v68._write_per_metal(
        tmp_path / "Pt",
        "Pt",
        summary,
        verdict,
        metadata=metadata,
        helpers_available=False,
    )
    assert row["metal"] == "Pt"
    assert row["six_of_six_pass"] is True
    assert (tmp_path / "Pt" / "catherine_props.json").exists()
    assert (tmp_path / "Pt" / "bench_summary.json").exists()

    catherine = json.loads((tmp_path / "Pt" / "catherine_props.json").read_text())
    bench = json.loads((tmp_path / "Pt" / "bench_summary.json").read_text())

    assert catherine["metadata"]["wf"] == "Wf-68"
    assert catherine["metadata"]["r20_plan"] == "R20-P5.1"
    assert catherine["six_of_six_pass"] is True
    assert "wf57_back_link" in bench["metadata"]

    bench_meta = bench["metadata"]
    assert bench_meta["wf"] == "Wf-68"
    assert bench_meta["phase"] == "Phase 1"
    assert bench_meta["r20_plan"] == "R20-P5.1"


def test_synthesis_md_renders(tmp_path):
    """``_render_synthesis`` produces a non-empty Markdown with the headline verdict."""
    per_metal_rows = [
        {"metal": "Pt", "six_of_six_pass": True, "n_pass": 6, "n_fail": 0,
         "synthetic_only": True},
        {"metal": "Ru", "six_of_six_pass": True, "n_pass": 6, "n_fail": 0,
         "synthetic_only": True},
    ]
    # Write per-metal catherine_props.json so the synthesis loader can
    # compute per-property pass-rate.
    for r in per_metal_rows:
        d = tmp_path / r["metal"]
        d.mkdir()
        (d / "catherine_props.json").write_text(json.dumps({
            "metal": r["metal"],
            "six_of_six_pass": True,
            "n_pass": 6,
            "n_fail": 0,
            "passed_properties": list(v68.CATHERINE_PROPERTIES),
            "failed_properties": [],
            "per_property": {
                pid: {"passed": True, "n_samples": 1, "n_violations": 0,
                      "wall_time_s": 0.0, "notes": ""}
                for pid in v68.CATHERINE_PROPERTIES
            },
            "wall_time_s": 0.0,
        }))
    md = v68._render_synthesis(
        metadata={
            "n_metals": 2,
            "n_seeds": 1,
            "max_depth": 1,
            "real_bfs_available": False,
            "timestamp": "2026-09-26T00:00:00Z",
        },
        per_metal_rows=per_metal_rows,
        out_dir=tmp_path,
    )
    assert "Wf-68 Phase 1" in md
    assert "Headline verdict" in md
    assert "2 / 2 metals" in md
    assert "Pt" in md
    assert "Ru" in md
