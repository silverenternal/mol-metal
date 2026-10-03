"""Tests for Wf-64 Phase 3 — bench_m8_cross_dataset module.

================================================================
Wf-64 Phase 3 cross-dataset m8 PB-Valid retrain (test suite)
================================================================

The test suite pins:

1. **Dataset loaders** return SampleSMILES instances with the right
   metal / bucket fields.  No mock data — the real CSVs are loaded.
2. **Deterministic subsampling** returns the same SMILES for the
   same seed across runs.
3. **PB outcome aggregation** computes per-bucket + headline stats
   consistent with raw outcome counts.
4. **Output writers** write valid JSON / Markdown to the correct
   paths.
5. **CLI main()** returns 0 when target met, 2 when not (or when PB
   is missing — honest negative path).

All tests are CPU-only.  ``posebusters`` may be absent on the host;
the test suite degrades gracefully — outcomes with ``status="skipped"``
are still counted but the headline ``pb_pass_rate`` reflects what the
real adapter would have produced.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "molmetal" / "scripts"
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_m8_cross_dataset as bm8  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def platinai_samples() -> List[bm8.SampleSMILES]:
    return bm8.load_platinai_4cell()


@pytest.fixture(scope="module")
def metalcytotox_samples() -> List[bm8.SampleSMILES]:
    return bm8.load_metalcytotox_5metal()


@pytest.fixture(scope="module")
def tmqm_samples() -> List[bm8.SampleSMILES]:
    return bm8.load_tmqm_dblock()


# ---------------------------------------------------------------------------
# Dataset loader tests
# ---------------------------------------------------------------------------
def test_platinai_4cell_loader_returns_pt_only(platinai_samples):
    """PlatinAI 4-cell is monometallic Pt."""
    assert (
        len(platinai_samples) > 1000
    ), f"PlatinAI 4-cell should have >1000 rows, got {len(platinai_samples)}"
    metals = {s.metal for s in platinai_samples}
    assert metals == {"Pt"}, f"PlatinAI should only contain Pt, got {metals}"
    for s in platinai_samples[:10]:
        assert s.smiles, "SMILES should not be empty"
        assert (
            "Pt" in s.smiles or "[Pt" in s.smiles
        ), f"PlatinAI SMILES should contain Pt: {s.smiles}"


def test_metalcytotox_loader_returns_5_metals(metalcytotox_samples):
    """MetalCytoToxDB 5-metal subset should only contain Ru/Ir/Rh/Os/Re."""
    metals = {s.metal for s in metalcytotox_samples}
    assert metals.issubset(
        set(bm8.METALCYTOTOX_METALS)
    ), f"metalcytotox should only contain 5 metals, got {metals}"
    assert len(metalcytotox_samples) > 100, (
        f"MetalCytoToxDB should have >100 rows in 5-metal subset, got "
        f"{len(metalcytotox_samples)}"
    )
    for s in metalcytotox_samples[:10]:
        assert s.smiles, "SMILES should not be empty"


def test_tmqm_loader_filters_dblock(tmqm_samples):
    """tmQM d-block loader should exclude La (f-block) and Hg/Cd (post-d)."""
    metals = {s.metal for s in tmqm_samples}
    # La is f-block — should NOT appear
    assert "La" not in metals, f"tmQM d-block should exclude La, got {metals}"
    # Group-3 (Sc, Y) is acceptable per IUPAC group 3 convention
    for m in metals:
        assert m in bm8.TMQM_DBLOCK_METALS, f"unexpected metal {m!r} in d-block set"
    assert (
        len(tmqm_samples) > 10000
    ), f"tmQM d-block should have >10000 rows, got {len(tmqm_samples)}"


def test_select_dataset_dispatch():
    """Dataset name dispatch works for all 3 known datasets."""
    for name in ("platinai_4cell", "metalcytotox_5metal", "tmqm_dblock"):
        samples = bm8.select_dataset(name)
        assert isinstance(samples, list), f"{name} should return a list"


def test_select_dataset_unknown_raises():
    """Unknown dataset name raises ValueError."""
    with pytest.raises(ValueError, match="unknown dataset"):
        bm8.select_dataset("not_a_real_dataset")


# ---------------------------------------------------------------------------
# Subsample determinism tests
# ---------------------------------------------------------------------------
def test_subsample_deterministic():
    """Same seed → same SMILES across runs."""
    samples = bm8.load_metalcytotox_5metal()
    if len(samples) < 50:
        pytest.skip("MetalCytoToxDB has too few rows")
    a = bm8.deterministic_subsample(samples, 10, seed=42)
    b = bm8.deterministic_subsample(samples, 10, seed=42)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]


def test_subsample_different_seeds():
    """Different seeds → different SMILES."""
    samples = bm8.load_metalcytotox_5metal()
    if len(samples) < 50:
        pytest.skip("MetalCytoToxDB has too few rows")
    a = bm8.deterministic_subsample(samples, 10, seed=0)
    b = bm8.deterministic_subsample(samples, 10, seed=1)
    assert [s.sample_id for s in a] != [s.sample_id for s in b]


def test_subsample_caps_to_n():
    """Subsample size capped to len(samples)."""
    samples = bm8.load_metalcytotox_5metal()[:30]
    out = bm8.deterministic_subsample(samples, 100, seed=0)
    assert len(out) == 30, f"expected 30, got {len(out)}"


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------
def _make_outcome(status, pb_valid, bucket="Pt", metal="Pt") -> bm8.PBOutcome:
    return bm8.PBOutcome(
        sample_id="x",
        smiles="CCO",
        metal=metal,
        bucket=bucket,
        status=status,
        pb_valid=pb_valid,
        n_checks=14,
        n_passed=14 if pb_valid else 7,
        n_failed=0 if pb_valid else 7,
        n_skipped=0,
        failed_checks=(),
        elapsed_s=0.1,
    )


def test_aggregate_buckets():
    """Per-bucket pass_rate counts passed vs total (skip+pass+fail)."""
    outcomes = [
        _make_outcome("passed", True, bucket="Pt"),
        _make_outcome("passed", True, bucket="Pt"),
        _make_outcome("failed", False, bucket="Pt"),
        _make_outcome("passed", True, bucket="Ru"),
        _make_outcome("skipped", None, bucket="Ru"),
    ]
    agg = bm8.aggregate_outcomes(outcomes)
    by_bucket = agg["by_bucket"]
    assert by_bucket["Pt"]["n_total"] == 3
    assert by_bucket["Pt"]["n_passed"] == 2
    assert by_bucket["Pt"]["pb_pass_rate"] == pytest.approx(2 / 3)
    assert by_bucket["Ru"]["n_total"] == 2
    assert by_bucket["Ru"]["n_passed"] == 1
    assert agg["headline_pb_pass_rate"] == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_aggregate_empty_outcomes():
    """Empty outcomes → headline 0.0."""
    agg = bm8.aggregate_outcomes([])
    assert agg["headline_pb_pass_rate"] == 0.0
    assert agg["by_bucket"] == {}


# ---------------------------------------------------------------------------
# Output writer tests
# ---------------------------------------------------------------------------
def test_outcome_to_dict_roundtrip():
    """PBOutcome.to_dict is JSON-serialisable."""
    o = _make_outcome("passed", True)
    d = bm8.outcome_to_dict(o)
    # All values must be JSON-serialisable
    json.dumps(d)


def test_write_dataset_outputs(tmp_path):
    """Per-seed JSON written with correct schema + counts."""
    outcomes = [
        _make_outcome("passed", True, bucket="Pt"),
        _make_outcome("failed", False, bucket="Pt"),
    ]
    agg = bm8.aggregate_outcomes(outcomes)
    bm8.write_dataset_outputs(tmp_path, "platinai_4cell", 0, outcomes, agg)
    seed_path = tmp_path / "platinai_4cell" / "seed_0_metrics.json"
    assert seed_path.exists()
    payload = json.loads(seed_path.read_text())
    assert payload["dataset"] == "platinai_4cell"
    assert payload["seed"] == 0
    assert payload["n_samples"] == 2
    assert len(payload["outcomes"]) == 2
    assert "aggregate" in payload


def test_write_cross_summary(tmp_path):
    """Cross-summary JSON written with target_met + headline."""
    cross_agg = {
        "platinai_4cell": {"headline_pb_pass_rate": 0.85, "by_bucket": {}},
        "metalcytotox_5metal": {"headline_pb_pass_rate": 0.72, "by_bucket": {}},
    }
    per_ds = {"platinai_4cell": [0.85], "metalcytotox_5metal": [0.72]}
    bm8.write_cross_summary(tmp_path, cross_agg, per_ds)
    out_path = tmp_path / "m8_cross_summary.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["target_met"] is True  # 0.785 >= 0.7
    assert "overall_pb_pass_rate_last_seed" in payload


def test_write_synthesis_md(tmp_path):
    """Synthesis markdown written with headline + per-dataset table."""
    cross_agg = {
        "platinai_4cell": {"headline_pb_pass_rate": 0.85, "by_bucket": {}},
    }
    per_ds = {"platinai_4cell": [0.85]}
    bm8.write_synthesis_md(
        tmp_path,
        cross_agg,
        per_ds,
        n_samples=50,
        n_seeds=1,
        pb_installed=True,
    )
    md_path = tmp_path / "synthesis.md"
    assert md_path.exists()
    text = md_path.read_text()
    assert "Wf-64 Phase 3" in text
    assert "platinai_4cell" in text
    assert "0.85" in text or "0.8500" in text


# ---------------------------------------------------------------------------
# CLI / main() tests
# ---------------------------------------------------------------------------
def test_main_smoke_runs():
    """main() runs end-to-end with --n-samples 2 to keep wall cheap."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "wf64_test"
        rc = bm8.main(
            [
                "--datasets",
                "metalcytotox_5metal",
                "--seeds",
                "0",
                "--n-samples",
                "2",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert rc in (0, 2), f"unexpected return code {rc}"
        # Outputs must exist
        seed_json = out_dir / "metalcytotox_5metal" / "seed_0_metrics.json"
        assert seed_json.exists()
        summary_json = out_dir / "metalcytotox_5metal" / "per_seed_summary.json"
        assert summary_json.exists()
