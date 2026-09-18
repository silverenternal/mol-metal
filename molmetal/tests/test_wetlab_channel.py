"""test_wetlab_channel.py — unit tests for the TODO-30 P5.2 Tier 1
wet-lab reward channel + plumbing + recalibration.

These tests cover the contract:

1. ``r_wetlab`` returns ``-|predicted - measured|`` for assayed SMILES.
2. Empty ``wetlab_measurement_dict`` returns 0.0 for all candidates.
3. Missing measurement returns 0.0 (no error / no crash).
4. ``--wetlab-input`` parses the TSV correctly (wide header + two-column).
5. :mod:`molmetal.scripts.recalibrate_from_assay` produces a correct
   Pearson r on synthetic data.
6. The reward aggregator is **bit-for-bit identical** when no
   wet-lab channel is registered (regression-proof for all
   downstream adapters).

All tests are CPU-only and skip the GPU/CFM training stack.  The
``RewardAggregator`` is imported lazily so the test module is
collectible even when the heavier search_alg / torch stack is broken.

Run from the project root::

    uv run pytest molmetal/tests/test_wetlab_channel.py -q --tb=short
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest


# ---------------------------------------------------------------------------
# Subject under test — imported lazily so the test module can be
# collected even when the heavier search_alg / torch stack is broken.
# ---------------------------------------------------------------------------
try:
    from molmetal_lam.lam_chem.wetlab_reward_channel import (
        compute_wetlab_signed_error,
        make_wetlab_channel,
        register_wetlab_channel,
    )
    _HAS_CHANNEL = True
except Exception:  # pragma: no cover
    compute_wetlab_signed_error = None
    make_wetlab_channel = None
    register_wetlab_channel = None
    _HAS_CHANNEL = False

try:
    from molmetal_lam.lam_chem import wetlab_protocol as _WP
    _HAS_PROTOCOL = True
except Exception:  # pragma: no cover
    _WP = None
    _HAS_PROTOCOL = False

try:
    from molmetal_lam.search_alg.proof_search import RewardAggregator
    _HAS_AGGREGATOR = True
except Exception:  # pragma: no cover
    RewardAggregator = None
    _HAS_AGGREGATOR = False


# ---------------------------------------------------------------------------
# Mock state helper
# ---------------------------------------------------------------------------


@dataclass
class _MockState:
    """Minimal state object that exposes ``canonical_smiles()``."""

    smiles: str

    def canonical_smiles(self) -> str:
        return self.smiles


# ---------------------------------------------------------------------------
# Tests — compute_wetlab_signed_error
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_perfect_prediction_returns_zero():
    """Perfect prediction: |pred - meas| == 0 → signed error = 0."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error(
        "CCO", meas, predicted_pic50=5.0,
    )
    assert err == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_half_log_unit_returns_minus_half():
    """Half-log-unit error → signed error == -0.5."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error(
        "CCO", meas, predicted_pic50=5.5,
    )
    assert err == pytest.approx(-0.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_clamped_to_three_log_units():
    """Huge errors are clamped to ±3 log-units (defensive bound)."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error(
        "CCO", meas, predicted_pic50=20.0,  # 15 log-unit error → clamp to 3
    )
    assert err == pytest.approx(-3.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_missing_measurement_returns_zero():
    """Missing measurement returns 0.0 (graceful degradation)."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error(
        "CCN", meas, predicted_pic50=5.0,  # CCN not in meas dict
    )
    assert err == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_empty_dict_returns_zero():
    """Empty measurement dict returns 0.0 (graceful degradation)."""
    err = compute_wetlab_signed_error("CCO", None, predicted_pic50=5.0)
    assert err == pytest.approx(0.0, abs=1e-9)
    err = compute_wetlab_signed_error("CCO", {}, predicted_pic50=5.0)
    assert err == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_empty_smiles_returns_zero():
    """Empty SMILES returns 0.0."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error("", meas, predicted_pic50=5.0)
    assert err == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_nan_measured_returns_zero():
    """NaN measured value returns 0.0 (never NaN propagates)."""
    meas = {"CCO": float("nan")}
    err = compute_wetlab_signed_error("CCO", meas, predicted_pic50=5.0)
    assert err == pytest.approx(0.0, abs=1e-9)
    assert not math.isnan(err)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_nan_predicted_returns_zero():
    """NaN predicted value returns 0.0 (never NaN propagates)."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error(
        "CCO", meas, predicted_pic50=float("nan"),
    )
    assert err == pytest.approx(0.0, abs=1e-9)
    assert not math.isnan(err)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_lookup_callable_used_when_no_prediction():
    """When ``predicted_pic50`` is None, ``predicted_lookup`` is consulted."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error(
        "CCO", meas, predicted_pic50=None,
        predicted_lookup=lambda s: 5.3 if s == "CCO" else None,
    )
    assert err == pytest.approx(-0.3, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_signed_error_zero_when_missing_true_no_pred():
    """zero_when_missing=True with no prediction returns 0.0."""
    meas = {"CCO": 5.0}
    err = compute_wetlab_signed_error(
        "CCO", meas, predicted_pic50=None, zero_when_missing=True,
    )
    assert err == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — make_wetlab_channel + RewardAggregator integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_make_wetlab_channel_handles_canonical_smiles():
    """The closure accepts a state with ``canonical_smiles()``."""
    meas = {"CCO": 5.0}
    channel = make_wetlab_channel(
        meas, predicted_lookup=lambda s: {"CCO": 5.5, "CCN": 6.0}.get(s),
    )
    state = _MockState(smiles="CCO")
    val = channel(state)
    assert val == pytest.approx(-0.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_make_wetlab_channel_handles_plain_string():
    """The closure accepts a plain SMILES string."""
    meas = {"CCO": 5.0}
    channel = make_wetlab_channel(
        meas, predicted_lookup=lambda s: 5.5 if s == "CCO" else None,
    )
    val = channel("CCO")
    assert val == pytest.approx(-0.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_make_wetlab_channel_empty_dict_is_no_op():
    """Empty / None measurement dict → closure always returns 0.0."""
    channel = make_wetlab_channel(None)
    state = _MockState(smiles="CCO")
    assert channel(state) == 0.0


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_make_wetlab_channel_never_crashes_on_bad_state():
    """The closure never raises on bad state — MCTS safety contract."""
    meas = {"CCO": 5.0}
    channel = make_wetlab_channel(
        meas, predicted_lookup=lambda s: 5.5 if s == "CCO" else None,
    )

    class _Bad:
        @property
        def canonical_smiles(self):
            raise RuntimeError("boom")

    # Bad state — closure returns 0.0 (graceful degradation).
    assert channel(_Bad()) == 0.0


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_register_wetlab_channel_wires_aggregator():
    """register_wetlab_channel installs r_wetlab + w_wetlab on the aggregator."""
    agg = RewardAggregator(r_vina=lambda s: 2.5)
    meas = {"CCO": 5.0}
    register_wetlab_channel(
        agg, meas,
        weight=1.0,
        predicted_lookup=lambda s: {"CCO": 5.5}.get(s),
    )
    state = _MockState(smiles="CCO")
    assert agg.r_wetlab is not None
    assert float(agg.w_wetlab) == 1.0
    # Channel contribution = -0.5 (signed error)
    v = agg.r_wetlab(state)
    assert v == pytest.approx(-0.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_register_wetlab_channel_disabled_zero_weight_keeps_aggregator_bit_for_bit():
    """Default ``w_wetlab=0.0`` keeps the aggregator's reward bit-for-bit
    identical — backward-compat regression-proof."""
    agg = RewardAggregator(r_vina=lambda s: 2.5)
    register_wetlab_channel(
        agg, {"CCO": 5.0}, weight=0.0,
        predicted_lookup=lambda s: 5.5 if s == "CCO" else None,
    )
    state = _MockState(smiles="CCO")
    # vina_invert=True → -2.5 + 0.0 (channel weight zeroed)
    total = agg(state)
    assert total == pytest.approx(-2.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_aggregator_unregistered_wetlab_is_no_op():
    """A fresh RewardAggregator without register_wetlab_channel call has
    ``r_wetlab=None`` and ``w_wetlab=0.0`` — the channel is silently off."""
    agg = RewardAggregator(r_vina=lambda s: 2.5)
    assert agg.r_wetlab is None
    assert float(agg.w_wetlab) == 0.0
    state = _MockState(smiles="CCO")
    # vina_invert=True → -2.5; nothing else
    assert agg(state) == pytest.approx(-2.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_AGGREGATOR, reason="RewardAggregator unavailable")
def test_aggregator_aggregate_recognises_r_wetlab_key():
    """The aggregator's ``aggregate`` method honours ``{"r_wetlab": value}``
    scaled by ``w_wetlab`` (mirrors the per-channel aggregate contract)."""
    agg = RewardAggregator()
    register_wetlab_channel(agg, {"CCO": 5.0}, weight=0.5)
    # Direct aggregate call with per-channel value of -1.0 → 0.5 * -1.0 = -0.5
    val = agg.aggregate(
        smiles="CCO", channels={"r_wetlab": -1.0},
    )
    assert val == pytest.approx(-0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — wetlab_protocol (Assay dataclass + load_assays)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROTOCOL, reason="wetlab_protocol unavailable")
def test_load_assays_two_column_fallback(tmp_path: Path):
    """A TSV without a header is parsed as ``smiles<sep>value`` pairs."""
    p = tmp_path / "two_col.tsv"
    p.write_text(
        "# comment\n"
        "CCO\t5.0\n"
        "CCN\t6.0\n"
        "CCC\t7.0\n"
    )
    rows = _WP.load_assays(str(p))
    assert len(rows) == 3
    assert rows[0].smiles == "CCO"
    assert rows[0].outcome_value == pytest.approx(5.0, abs=1e-9)
    assert rows[1].smiles == "CCN"
    assert rows[2].smiles == "CCC"


@pytest.mark.skipif(not _HAS_PROTOCOL, reason="wetlab_protocol unavailable")
def test_load_assays_wide_header(tmp_path: Path):
    """A TSV with a wide header is parsed with optional columns."""
    p = tmp_path / "wide.tsv"
    p.write_text(
        "smiles\tpIC50\tcell_line\toutcome_metric\tassay_id\n"
        "CCO\t5.0\tHeLa\tpIC50\tA-001\n"
        "CCN\t6.0\tMCF7\tpIC50\tA-002\n"
    )
    rows = _WP.load_assays(str(p))
    assert len(rows) == 2
    r = rows[0]
    assert r.smiles == "CCO"
    assert r.outcome_value == pytest.approx(5.0, abs=1e-9)
    assert r.cell_line == "HeLa"
    assert r.outcome_metric == "pIC50"
    assert r.assay_id == "A-001"


@pytest.mark.skipif(not _HAS_PROTOCOL, reason="wetlab_protocol unavailable")
def test_load_assays_handles_missing_file(tmp_path: Path):
    """A missing file returns an empty list — graceful degradation."""
    rows = _WP.load_assays(str(tmp_path / "does_not_exist.tsv"))
    assert rows == []


@pytest.mark.skipif(not _HAS_PROTOCOL, reason="wetlab_protocol unavailable")
def test_load_assays_skips_malformed_values(tmp_path: Path):
    """Malformed numeric values are stored as NaN (not skipped)."""
    p = tmp_path / "bad.tsv"
    p.write_text("CCO\tnot_a_number\nCCN\t6.5\n")
    rows = _WP.load_assays(str(p))
    assert len(rows) == 2
    assert rows[0].smiles == "CCO"
    assert math.isnan(rows[0].outcome_value)
    assert rows[1].smiles == "CCN"
    assert rows[1].outcome_value == pytest.approx(6.5, abs=1e-9)


@pytest.mark.skipif(not _HAS_PROTOCOL, reason="wetlab_protocol unavailable")
def test_assays_to_dict_filters_by_metric(tmp_path: Path):
    """assays_to_dict keeps only rows matching the requested metric."""
    p = tmp_path / "mixed.tsv"
    p.write_text(
        "smiles\tpIC50\toutcome_metric\n"
        "CCO\t5.0\tpIC50\n"
        "CCN\t2.5\tIC50_uM\n"
        "CCC\t7.0\tpIC50\n"
    )
    rows = _WP.load_assays(str(p))
    pmap = _WP.assays_to_dict(rows, metric="pIC50")
    assert "CCO" in pmap
    assert "CCC" in pmap
    assert "CCN" not in pmap
    assert pmap["CCO"] == pytest.approx(5.0, abs=1e-9)
    assert pmap["CCC"] == pytest.approx(7.0, abs=1e-9)


@pytest.mark.skipif(not _HAS_PROTOCOL, reason="wetlab_protocol unavailable")
def test_append_assay_dedup():
    """append_assay silently drops duplicate SMILES (first-loaded wins)."""
    rows = [_WP.Assay(smiles="CCO", outcome_value=5.0)]
    new = _WP.Assay(smiles="CCO", outcome_value=99.0)
    merged = _WP.append_assay(rows, new)
    assert len(merged) == 1
    assert merged[0].outcome_value == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Tests — recalibrate_from_assay.py
# ---------------------------------------------------------------------------


def _write_tsv(path: Path, rows: list) -> None:
    """Helper — write a 2-column TSV with smiles<TAB>pIC50."""
    with path.open("w") as fh:
        fh.write("smiles\tpIC50\n")
        for smi, val in rows:
            fh.write(f"{smi}\t{val}\n")


def test_recalibrate_perfect_prediction_pearson_one(tmp_path: Path):
    """When predictions == measurements, Pearson r must be exactly 1.0."""
    wetlab = tmp_path / "wetlab.tsv"
    drylab = tmp_path / "drylab.tsv"
    rows = [("CCO", 5.0), ("CCN", 6.0), ("CCC", 7.0),
            ("CCl", 4.0), ("CBr", 3.0)]
    _write_tsv(wetlab, rows)
    _write_tsv(drylab, rows)

    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m", "molmetal.scripts.recalibrate_from_assay",
        "--wetlab-input", str(wetlab),
        "--drylab-input", str(drylab),
        "--output", str(report_path),
    ]
    # Run from the repo root so the package import works.
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    report = json.loads(report_path.read_text())
    assert report["n_overlap"] == 5
    assert report["pearson_r"] == pytest.approx(1.0, abs=1e-3)
    assert report["rmse"] == pytest.approx(0.0, abs=1e-3)
    assert report["within_margin_count"] == 5
    assert "OK" in report["recommendation"]


def test_recalibrate_noisy_prediction_pearson_bounded(tmp_path: Path):
    """A noisy predictor should give Pearson r < 1.0 but > 0.5."""
    wetlab_rows = [
        ("CCO", 5.0), ("CCN", 6.0), ("CCC", 7.0),
        ("CCl", 4.0), ("CBr", 3.0), ("CF", 8.0),
        ("CI", 2.0), ("CS", 9.0),
    ]
    # Add Gaussian-ish noise (deterministic, hand-crafted)
    drylab_rows = [
        ("CCO", 5.1), ("CCN", 5.8), ("CCC", 7.2),
        ("CCl", 3.8), ("CBr", 3.1), ("CF", 7.9),
        ("CI", 2.1), ("CS", 8.7),
    ]
    wetlab = tmp_path / "wetlab.tsv"
    drylab = tmp_path / "drylab.tsv"
    _write_tsv(wetlab, wetlab_rows)
    _write_tsv(drylab, drylab_rows)
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m", "molmetal.scripts.recalibrate_from_assay",
        "--wetlab-input", str(wetlab),
        "--drylab-input", str(drylab),
        "--output", str(report_path),
    ]
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    report = json.loads(report_path.read_text())
    assert report["n_overlap"] == 8
    r = report["pearson_r"]
    assert r is not None
    assert 0.5 < r < 1.0
    assert report["rmse"] > 0.0
    assert report["within_margin_count"] >= 4


def test_recalibrate_no_overlap_returns_zero(tmp_path: Path):
    """Disjoint SMILES sets → n_overlap=0 + NO_OVERLAP recommendation."""
    wetlab = tmp_path / "wetlab.tsv"
    drylab = tmp_path / "drylab.tsv"
    _write_tsv(wetlab, [("CCO", 5.0), ("CCN", 6.0)])
    _write_tsv(drylab, [("CCC", 7.0), ("CCl", 4.0)])
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m", "molmetal.scripts.recalibrate_from_assay",
        "--wetlab-input", str(wetlab),
        "--drylab-input", str(drylab),
        "--output", str(report_path),
    ]
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=60,
    )
    # NO_OVERLAP returns exit code 1 with NO_OVERLAP recommendation.
    assert result.returncode == 1
    report = json.loads(report_path.read_text())
    assert report["n_overlap"] == 0
    assert "NO_OVERLAP" in report["recommendation"]


# ---------------------------------------------------------------------------
# Honest-framing constraint guard — channel must never silently promote
# DESIGN -> MEASURED.  We assert this by checking that the channel
# returns a *signed* error (penalty) rather than a reward (bonus), so
# the noise floor in the MCTS leaf value remains well-bounded.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CHANNEL, reason="wetlab_reward_channel unavailable")
def test_honest_framing_channel_returns_signed_error_not_bonus():
    """The channel returns a *penalty* (always <= 0) — never a reward.

    Rationale: this is a calibration signal, not a primary oracle.
    A perfect prediction contributes 0.0 (not bonus); a worse
    prediction contributes a more negative value.  This is the same
    contract used by the residue-aware QSAR lit (Sheridan 2013).
    """
    meas = {"CCO": 5.0}
    # Probe a range of predicted values; all must be <= 0.0.
    for pred in [4.0, 5.0, 5.5, 6.0, 8.0]:
        err = compute_wetlab_signed_error("CCO", meas, predicted_pic50=pred)
        assert err <= 0.0, (
            f"wet-lab channel must never produce a bonus (err={err} > 0)"
        )


# ---------------------------------------------------------------------------
# Constraint guard — module must NOT pull RDKit at import time.
# ---------------------------------------------------------------------------


def test_module_import_does_not_require_rdkit():
    """Importing the module must NOT fail when RDKit is missing.

    Wet-lab plumbing is data-side; it should remain usable on bare
    CPU hosts that do not have RDKit installed.
    """
    # The test simply re-imports — if this doesn't raise, we're good.
    from molmetal_lam.lam_chem import wetlab_protocol  # noqa: F401
    from molmetal_lam.lam_chem import wetlab_reward_channel  # noqa: F401
