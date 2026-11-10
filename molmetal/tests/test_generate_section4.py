"""Behavioral tests for molmetal/scripts/generate_section4.py.

Five tests verify the §4 auto-generator end-to-end:

1. ``--dry-run`` exits 0 and prints a LaTeX table containing every
   registered metric display label (catches JSON-file-key drift).
2. The generator writes ``paper/sections/04_results_table2.tex`` with
   a \\begin{tabular} block.
3. The generator writes the three figure data JSON stubs under
   ``paper/figures/data/`` with the expected keys.
4. The generator writes a summary.md under ``paper/reports/wf_section4_gen/``.
5. Missing metric JSON files render as ``\\textit{pending ---}`` cells
   (graceful-degrade contract; verifies the honest framing).

All tests are deterministic and exercise the public CLI; no MCTS / no
GPU is involved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "generate_section4.py"
METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"
DEFAULT_OUT = REPO_ROOT / "paper"
DEFAULT_REPORT = REPO_ROOT / "paper" / "reports" / "wf_section4_gen"

# A representative subset of registered metrics — at least one per tier.
EXPECTED_LABELS = (
    "Ru subset AUC",  # Tier 1
    "Vina Dock median",  # Tier 2
    "Synthesis rate",  # Tier 3
    "PB-Valid",  # Tier 2
    "NCI-60 precious 9-cell mean AUC",  # Tier 1
)


def _run_generator(
    *args: str, env_extra: dict | None = None
) -> subprocess.CompletedProcess:
    env = None
    if env_extra:
        import os

        env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


# ---------------------------------------------------------------------------
# 1) --dry-run exits 0 and contains expected metric labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", EXPECTED_LABELS)
def test_dry_run_contains_registered_metric(label):
    proc = _run_generator("--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "begin{tabular}" in proc.stdout
    assert label in proc.stdout, f"--dry-run output missing metric label {label!r}"


# ---------------------------------------------------------------------------
# 2) Generator writes Table 2 to paper/sections/04_results_table2.tex
# ---------------------------------------------------------------------------
def test_writes_table2_tex(tmp_path):
    paper_out = tmp_path / "paper"
    metrics_dir = METRICS_DIR  # real on-disk snapshot
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(metrics_dir),
        "--report-dir",
        str(paper_out / "reports" / "wf_section4_gen"),
    )
    assert proc.returncode == 0, proc.stderr
    tex = paper_out / "sections" / "04_results_table2.tex"
    assert tex.is_file(), f"missing {tex}"
    body = tex.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in body
    assert "\\toprule" in body and "\\bottomrule" in body
    # All 20 metrics should appear
    assert body.count("\\\\") >= 20


# ---------------------------------------------------------------------------
# 3) Generator writes the three figure data JSON stubs
# ---------------------------------------------------------------------------
def test_writes_figure_data_stubs(tmp_path):
    paper_out = tmp_path / "paper"
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(METRICS_DIR),
        "--report-dir",
        str(paper_out / "reports" / "wf_section4_gen"),
    )
    assert proc.returncode == 0, proc.stderr

    for name, expected_key in (
        ("fig7_cell_line.json", "platinai"),
        ("fig8_ood_radar.json", "imold_reference"),
        ("fig9_tier1_scorecard.json", "cards"),
    ):
        path = paper_out / "figures" / "data" / name
        assert path.is_file(), f"missing {path}"
        body = json.loads(path.read_text(encoding="utf-8"))
        assert expected_key in body, f"{name} missing key {expected_key!r}"


# ---------------------------------------------------------------------------
# 4) Generator writes paper/reports/wf_section4_gen/summary.md
# ---------------------------------------------------------------------------
def test_writes_summary_md(tmp_path):
    paper_out = tmp_path / "paper"
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(METRICS_DIR),
        "--report-dir",
        str(paper_out / "reports" / "wf_section4_gen"),
    )
    assert proc.returncode == 0, proc.stderr
    summary = paper_out / "reports" / "wf_section4_gen" / "summary.md"
    assert summary.is_file(), f"missing {summary}"
    body = summary.read_text(encoding="utf-8")
    assert "PASS:" in body and "PENDING:" in body
    # At least 20 row entries
    assert body.count("|") >= 20 * 6


# ---------------------------------------------------------------------------
# 5) Missing metric JSON renders as PENDING (graceful-degrade contract)
# ---------------------------------------------------------------------------
def test_missing_metric_renders_as_pending(tmp_path):
    # Build an empty metrics dir and confirm PENDING appears for every cell.
    empty = tmp_path / "empty_metrics"
    empty.mkdir()
    paper_out = tmp_path / "paper"
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(empty),
        "--report-dir",
        str(paper_out / "reports" / "wf_section4_gen"),
    )
    assert proc.returncode == 0, proc.stderr
    # The pending text lives in the Table 2 .tex file (and summary.md), not stdout.
    tex = paper_out / "sections" / "04_results_table2.tex"
    tex_body = tex.read_text(encoding="utf-8")
    assert "pending --- see TODO/METRICS" in tex_body
    # All 20 rows should render as PENDING.
    assert tex_body.count("pending --- see TODO/METRICS") == 20
    # The summary.md should also say PENDING for every row.
    summary = paper_out / "reports" / "wf_section4_gen" / "summary.md"
    summary_body = summary.read_text(encoding="utf-8")
    assert "PENDING: **20**" in summary_body
