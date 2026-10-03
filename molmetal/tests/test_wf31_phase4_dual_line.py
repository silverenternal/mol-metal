"""Pytest: Wf-31 Phase 4 — Dual-line parallel GPU benchmarks (smoke + structure).

This module validates the Wf-31 Phase 4 deliverables across two
parallel lines of work:

**Line 1 (GPU 1 / RTX 5090)** — actual 5-seed × 50-epoch × 3 benchmarks
on the unfreeze-free RTX 5090 card.  The full 6-12 hours-per-script
run cannot complete inside a single workflow session; this module
therefore mixes **MEASURED** assertions (against any smoke JSON that
the GPU run did manage to write) with **PROJECTED** structural
assertions (against the documented directory layout).

**Line 2 (foreground tests + docs)** — this test module + the
``wf31_m{1,2,7}_summary.json`` synthesis files that pin the dual-line
output.

Target test count: ~15 distinct checks.  All tests are CPU-only /
file-system-only (no GPU required to run them); they pass on a fresh
checkout.

Tests
-----
1. ``test_m1_se3sb_dir_exists`` — m1 se3sb out-dir exists.
2. ``test_m1_tmqm_dir_exists`` — m1 tmqm out-dir exists.
3. ``test_m2_se3sb_dir_exists`` — m2 se3sb out-dir exists.
4. ``test_m2_tmqm_dir_exists`` — m2 tmqm out-dir exists.
5. ``test_m7_dir_exists`` — m7 bench out-dir exists.
6. ``test_m7_metrics_json_well_formed`` — m7 bench JSON parses + has
   diversity + expressivity keys (skip if no GPU run finished).
7. ``test_m7_comparison_md_well_formed`` — m7 comparison_table.md is
   non-empty when present.
8. ``test_m1_se3sb_train_script_help_exits_zero`` — ``--help`` exits 0
   for ``train_ru_5metal_se3sb.py``.
9. ``test_m1_se3sb_train_script_exposes_metals_seeds_init`` —
   argparse exposes ``--metals``, ``--seeds``, ``--init`` (Wf-31 spec).
10. ``test_m2_cisplatin_train_script_help_exits_zero`` — ``--help`` exits 0
    for ``train_cisplatin_resistance_v2.py``.
11. ``test_m2_cisplatin_train_script_exposes_cell_line_seeds_init`` —
    argparse exposes ``--cell-line``, ``--seeds``, ``--init`` (Wf-31 spec).
12. ``test_m7_bench_script_help_exits_zero`` — ``--help`` exits 0 for
    ``bench_multi_ph_diversity.py``.
13. ``test_m7_bench_script_exposes_data_n_samples_output`` —
    argparse exposes ``--data``, ``--n-samples``, ``--output`` (Wf-31 spec).
14. ``test_wf31_m1_summary_json_path_well_formed`` — when the synthesis
    summary file exists, it is parseable JSON with documented keys.
15. ``test_wf31_m2_summary_json_path_well_formed`` — same for m2.
16. ``test_wf31_m7_summary_json_path_well_formed`` — same for m7.

The honest MEASURED-vs-PROJECTED framing (per ``party-twenty-spirit``
§3.5) is: tests 1-5, 8-13 are PROJECTED structural assertions (they
hold even before any GPU run touches the disk); tests 6-7, 14-16 are
MEASURED-conditioned (they ``pytest.skip`` when the GPU run hasn't
produced the corresponding JSON yet).

CI integration: ``.github/workflows/ci.yml`` job
``wf31-phase4-smoke-structure`` (planned, parallel to Wf-29 /
Wf-30 / Wf-35 CI jobs) gates this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
VENV_PY = PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"

# Three benchmark out-dirs (Wf-31 Phase 4 spec).
M1_SE3SB_DIR = REPORTS_DIR / "wf31_m1" / "se3sb"
M1_TMQM_DIR = REPORTS_DIR / "wf31_m1" / "tmqm"
M2_SE3SB_DIR = REPORTS_DIR / "wf31_m2" / "se3sb"
M2_TMQM_DIR = REPORTS_DIR / "wf31_m2" / "tmqm"
M7_DIR = REPORTS_DIR / "wf31_m7"

# Three per-benchmark synthesis summary JSONs (Line 2 deliverable).
M1_SUMMARY = REPORTS_DIR / "wf31_m1_summary.json"
M2_SUMMARY = REPORTS_DIR / "wf31_m2_summary.json"
M7_SUMMARY = REPORTS_DIR / "wf31_m7_summary.json"

# Three CLI scripts under test (Wf-31 spec).
TRAIN_RU_5METAL = SCRIPTS_DIR / "train_ru_5metal_se3sb.py"
TRAIN_CISPLATIN = SCRIPTS_DIR / "train_cisplatin_resistance_v2.py"
BENCH_MULTI_PH = SCRIPTS_DIR / "bench_multi_ph_diversity.py"


def _have_venv() -> bool:
    return VENV_PY.exists()


def _run_help(script: Path) -> int:
    """Run ``--help`` on ``script`` and return the exit code."""
    proc = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode


def _parse_args_help(script: Path) -> str:
    """Return the ``--help`` text for flag discovery."""
    proc = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 1-5. Output directory structure (PROJECTED — always pass on fresh checkout)
# ---------------------------------------------------------------------------
def test_m1_se3sb_dir_exists() -> None:
    assert M1_SE3SB_DIR.exists(), f"missing {M1_SE3SB_DIR}"


def test_m1_tmqm_dir_exists() -> None:
    assert M1_TMQM_DIR.exists(), f"missing {M1_TMQM_DIR}"


def test_m2_se3sb_dir_exists() -> None:
    assert M2_SE3SB_DIR.exists(), f"missing {M2_SE3SB_DIR}"


def test_m2_tmqm_dir_exists() -> None:
    assert M2_TMQM_DIR.exists(), f"missing {M2_TMQM_DIR}"


def test_m7_dir_exists() -> None:
    assert M7_DIR.exists(), f"missing {M7_DIR}"


# ---------------------------------------------------------------------------
# 6-7. m7 bench outputs (MEASURED-conditioned, skip if no GPU run)
# ---------------------------------------------------------------------------
def test_m7_metrics_json_well_formed() -> None:
    """m7 metrics.json exists, parses, has expressivity + diversity keys."""
    metrics_path = M7_DIR / "metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"m7 metrics.json not yet written by GPU run: {metrics_path}")
    with open(metrics_path) as f:
        payload = json.load(f)
    assert "expressivity" in payload, "metrics.json missing expressivity"
    assert "diversity" in payload, "metrics.json missing diversity"
    assert "multi_ph" in payload["expressivity"]
    assert "single_ph" in payload["expressivity"]
    assert "mean_tanimoto_distance" in payload["diversity"]


def test_m7_comparison_md_well_formed() -> None:
    """m7 comparison_table.md (or comparison.md) is non-empty when present."""
    for candidate in ("comparison_table.md", "comparison.md"):
        p = M7_DIR / candidate
        if p.exists():
            assert p.stat().st_size > 0, f"{p} exists but is empty"
            return
    pytest.skip("m7 comparison .md not yet written by GPU run")


# ---------------------------------------------------------------------------
# 8-9. train_ru_5metal_se3sb.py CLI (PROJECTED — runs --help, exits 0)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _have_venv(), reason="mol-metal-venv not present")
def test_m1_se3sb_train_script_help_exits_zero() -> None:
    assert _run_help(TRAIN_RU_5METAL) == 0


@pytest.mark.skipif(not _have_venv(), reason="mol-metal-venv not present")
def test_m1_se3sb_train_script_exposes_metals_seeds_init() -> None:
    """m1 CLI must expose --metals, --seeds, --init (Wf-31 spec)."""
    help_text = _parse_args_help(TRAIN_RU_5METAL)
    for flag in ("--metals", "--seeds", "--init"):
        assert flag in help_text, f"train_ru_5metal_se3sb.py --help missing {flag}"
    # Init choices must include both R17 baseline and R18 attach points.
    for choice in ("tmqm", "se3sb", "transport", "multiph"):
        assert (
            choice in help_text
        ), f"train_ru_5metal_se3sb.py --help missing init choice {choice}"


# ---------------------------------------------------------------------------
# 10-11. train_cisplatin_resistance_v2.py CLI (PROJECTED)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _have_venv(), reason="mol-metal-venv not present")
def test_m2_cisplatin_train_script_help_exits_zero() -> None:
    assert _run_help(TRAIN_CISPLATIN) == 0


@pytest.mark.skipif(not _have_venv(), reason="mol-metal-venv not present")
def test_m2_cisplatin_train_script_exposes_cell_line_seeds_init() -> None:
    """m2 CLI must expose --cell-line, --seeds, --init (Wf-31 spec)."""
    help_text = _parse_args_help(TRAIN_CISPLATIN)
    for flag in ("--cell-line", "--seeds", "--init"):
        assert (
            flag in help_text
        ), f"train_cisplatin_resistance_v2.py --help missing {flag}"
    # Cell-line choices include A2780cis (the Wf-31 spec target).
    assert (
        "A2780cis" in help_text
    ), "train_cisplatin_resistance_v2.py --help missing A2780cis cell-line"


# ---------------------------------------------------------------------------
# 12-13. bench_multi_ph_diversity.py CLI (PROJECTED)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _have_venv(), reason="mol-metal-venv not present")
def test_m7_bench_script_help_exits_zero() -> None:
    assert _run_help(BENCH_MULTI_PH) == 0


@pytest.mark.skipif(not _have_venv(), reason="mol-metal-venv not present")
def test_m7_bench_script_exposes_data_n_samples_output() -> None:
    """m7 CLI must expose --data, --n-samples, --output (Wf-31 spec)."""
    help_text = _parse_args_help(BENCH_MULTI_PH)
    for flag in ("--data", "--n-samples", "--output"):
        assert flag in help_text, f"bench_multi_ph_diversity.py --help missing {flag}"


# ---------------------------------------------------------------------------
# 14-16. Per-benchmark synthesis summary JSONs (MEASURED-conditioned)
# ---------------------------------------------------------------------------
def _check_summary_well_formed(path: Path, expected_keys: tuple) -> None:
    if not path.exists():
        pytest.skip(f"synthesis summary not yet written: {path}")
    with open(path) as f:
        payload = json.load(f)
    for k in expected_keys:
        assert k in payload, f"{path} missing key {k!r}"


def test_wf31_m1_summary_json_path_well_formed() -> None:
    """wf31_m1_summary.json — when present — has documented keys."""
    _check_summary_well_formed(
        M1_SUMMARY,
        expected_keys=("recipe", "init_modes", "per_init_summary"),
    )


def test_wf31_m2_summary_json_path_well_formed() -> None:
    """wf31_m2_summary.json — when present — has documented keys."""
    _check_summary_well_formed(
        M2_SUMMARY,
        expected_keys=("recipe", "init_modes", "per_init_summary"),
    )


def test_wf31_m7_summary_json_path_well_formed() -> None:
    """wf31_m7_summary.json — when present — has documented keys."""
    _check_summary_well_formed(
        M7_SUMMARY,
        expected_keys=("recipe", "n_samples", "expressivity", "diversity"),
    )
