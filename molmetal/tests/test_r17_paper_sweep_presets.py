"""Tests for the R17 paper-sweep preset flags on r4_lambda_only_run.py.

Five tests cover the new --smoke and --r17-paper-grade preset wiring:

1. CLI --help lists the new --smoke / --r17-paper-grade flags.
2. ``_resolve_paper_sweep_preset`` resolves --smoke to its preset values.
3. ``_resolve_paper_sweep_preset`` resolves --r17-paper-grade to its preset
   values.
4. Explicit --pockets / --seeds / --n-simulations / --manifest overrides win
   over the preset values.
5. When both --smoke and --r17-paper-grade are set, --r17-paper-grade wins.

Tests use subprocess to invoke ``_build_argparser().parse_args()`` via a
small in-process driver; the real MCTS stack is NOT exercised (the smoke
end-to-end run is exercised separately in the manual verification at
``molmetal/reports/wf_paper_sweep_spec/final.md`` §7).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from molmetal.scripts import r4_lambda_only_run as r4


SCRIPT = Path(r4.__file__).resolve()
REPO_ROOT = Path("/home/hugo/codes/mol-metal")


# ---------------------------------------------------------------------------
# 1) CLI --help lists the new --smoke / --r17-paper-grade flags
# ---------------------------------------------------------------------------
def test_cli_help_lists_r17_presets():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    out = (proc.stdout or "") + (proc.stderr or "")
    for needle in ("--smoke", "--r17-paper-grade"):
        assert needle in out, f"--help missing flag {needle}"


# ---------------------------------------------------------------------------
# 2) _resolve_paper_sweep_preset resolves --smoke to its preset values
# ---------------------------------------------------------------------------
def test_resolve_smoke_preset():
    parser = r4._build_argparser()
    args = parser.parse_args(["--smoke", "--output-dir", "smoke_test"])
    assert args.smoke is True
    assert args.r17_paper_grade is False

    resolved = r4._resolve_paper_sweep_preset(args)
    assert resolved.r17_preset_active == "smoke"
    assert resolved.pockets == r4._R17_SMOKE_PRESET["pockets"]
    assert resolved.seeds == r4._R17_SMOKE_PRESET["seeds"]
    assert resolved.n_simulations == r4._R17_SMOKE_PRESET["n_simulations"]
    assert resolved.manifest == r4._R17_SMOKE_PRESET["manifest"]


# ---------------------------------------------------------------------------
# 3) _resolve_paper_sweep_preset resolves --r17-paper-grade to its preset
# ---------------------------------------------------------------------------
def test_resolve_paper_grade_preset():
    parser = r4._build_argparser()
    args = parser.parse_args(["--r17-paper-grade", "--output-dir", "paper_grade_test"])
    assert args.r17_paper_grade is True
    assert args.smoke is False

    resolved = r4._resolve_paper_sweep_preset(args)
    assert resolved.r17_preset_active == "paper_grade"
    assert resolved.pockets == r4._R17_PAPER_GRADE_PRESET["pockets"]
    assert resolved.seeds == r4._R17_PAPER_GRADE_PRESET["seeds"]
    assert resolved.n_simulations == r4._R17_PAPER_GRADE_PRESET["n_simulations"]
    assert resolved.manifest == r4._R17_PAPER_GRADE_PRESET["manifest"]


# ---------------------------------------------------------------------------
# 4) Explicit --pockets / --seeds / --n-simulations / --manifest overrides win
# ---------------------------------------------------------------------------
def test_explicit_overrides_win_over_preset():
    parser = r4._build_argparser()
    # User passes --smoke AND explicit overrides for every preset-affected
    # field; the explicit values MUST win (any non-None value wins).
    args = parser.parse_args(
        [
            "--smoke",
            "--pockets",
            "5",
            "--seeds",
            "7",
            "8",
            "9",
            "--n-simulations",
            "50",
            "--manifest",
            "/tmp/explicit_manifest.csv",
            "--output-dir",
            "explicit_test",
        ]
    )
    resolved = r4._resolve_paper_sweep_preset(args)
    assert resolved.r17_preset_active == "smoke"
    assert resolved.pockets == 5
    assert resolved.seeds == [7, 8, 9]
    assert resolved.n_simulations == 50
    assert resolved.manifest == "/tmp/explicit_manifest.csv"


# ---------------------------------------------------------------------------
# 5) When both --smoke and --r17-paper-grade are set, --r17-paper-grade wins
# ---------------------------------------------------------------------------
def test_paper_grade_overrides_smoke_when_both_set(caplog):
    parser = r4._build_argparser()
    args = parser.parse_args(
        [
            "--smoke",
            "--r17-paper-grade",
            "--output-dir",
            "both_set_test",
        ]
    )
    with caplog.at_level("WARNING"):
        resolved = r4._resolve_paper_sweep_preset(args)
    assert resolved.r17_preset_active == "paper_grade"
    assert resolved.pockets == r4._R17_PAPER_GRADE_PRESET["pockets"]
    assert resolved.seeds == r4._R17_PAPER_GRADE_PRESET["seeds"]
    assert resolved.n_simulations == r4._R17_PAPER_GRADE_PRESET["n_simulations"]
    assert resolved.manifest == r4._R17_PAPER_GRADE_PRESET["manifest"]
    # Confirm a warning was emitted (the resolution order must surface
    # the conflict to the operator).
    assert any(
        "Both --smoke and --r17-paper-grade" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# 6) When neither preset is set, legacy defaults apply
# ---------------------------------------------------------------------------
def test_legacy_defaults_when_no_preset():
    parser = r4._build_argparser()
    args = parser.parse_args(["--output-dir", "legacy_test"])
    assert args.smoke is False
    assert args.r17_paper_grade is False

    resolved = r4._resolve_paper_sweep_preset(args)
    assert resolved.r17_preset_active == "legacy"
    assert resolved.pockets == r4._R17_LEGACY_DEFAULTS["pockets"]
    assert resolved.seeds == r4._R17_LEGACY_DEFAULTS["seeds"]
    assert resolved.n_simulations == r4._R17_LEGACY_DEFAULTS["n_simulations"]
    assert resolved.manifest == r4._R17_LEGACY_DEFAULTS["manifest"]


# ---------------------------------------------------------------------------
# 7) Preset dictionaries are well-formed (defensive regression guard)
# ---------------------------------------------------------------------------
def test_preset_dicts_well_formed():
    """Each preset must contain the four (pockets, seeds, n_simulations,
    manifest) keys with non-None values; the legacy default and paper-grade
    preset share the crossdocked100_manifest path; the smoke preset uses
    test_010_019_subset_manifest."""
    for name, preset in (
        ("legacy", r4._R17_LEGACY_DEFAULTS),
        ("paper_grade", r4._R17_PAPER_GRADE_PRESET),
    ):
        assert "pockets" in preset, f"{name} preset missing pockets"
        assert "seeds" in preset, f"{name} preset missing seeds"
        assert "n_simulations" in preset, f"{name} preset missing n_simulations"
        assert "manifest" in preset, f"{name} preset missing manifest"
        assert preset["pockets"] >= 1
        assert isinstance(preset["seeds"], list) and len(preset["seeds"]) >= 1
        assert preset["n_simulations"] >= 1
        assert "crossdocked100_manifest" in preset["manifest"]
    # Smoke preset (different manifest path).
    assert "pockets" in r4._R17_SMOKE_PRESET
    assert "seeds" in r4._R17_SMOKE_PRESET
    assert "n_simulations" in r4._R17_SMOKE_PRESET
    assert "manifest" in r4._R17_SMOKE_PRESET
    assert "test_010_019_subset_manifest" in r4._R17_SMOKE_PRESET["manifest"]
    # Paper-grade is the larger config.
    assert (
        r4._R17_PAPER_GRADE_PRESET["n_simulations"]
        > r4._R17_SMOKE_PRESET["n_simulations"]
    )
    assert r4._R17_PAPER_GRADE_PRESET["pockets"] > r4._R17_SMOKE_PRESET["pockets"]


# ---------------------------------------------------------------------------
# 8) Legacy _build_argparser returns a Namespace with the r17-payload
# ---------------------------------------------------------------------------
def test_argparser_exposes_r17_attrs():
    """The new flags must appear on the parsed Namespace so the resolution
    helper can dispatch on them."""
    parser = r4._build_argparser()
    args = parser.parse_args(["--output-dir", "smoke"])
    assert hasattr(args, "smoke")
    assert hasattr(args, "r17_paper_grade")
    assert args.smoke is False
    assert args.r17_paper_grade is False
