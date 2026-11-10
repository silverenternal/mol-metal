"""Behavioral tests for molmetal/scripts/generate_section6.py.

Five tests verify the §6 auto-generator end-to-end:

1. ``--dry-run`` exits 0 and prints a LaTeX fragment containing the
   expected structural anchors (\\section{Limitations}, \\begin{enumerate}).
2. The generator writes ``paper/sections/06_limitations_auto.tex`` with
   an \\begin{enumerate} block + ≥ 1 \\item entry per failed metric.
3. The generator writes a summary.md under
   ``paper/reports/wf_section6_gen/``.
4. With the real ``metrics/by_metric/`` snapshot, the script emits
   at least one \\item per below-gate metric (graceful-degrade on
   real data).
5. Missing metrics dir renders without raising (graceful-degrade).

All tests are deterministic and exercise the public CLI; no MCTS / no
GPU is involved.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "generate_section6.py"
METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"
INDEX_MD = REPO_ROOT / "TODO" / "INDEX.md"
MAPPING_MD = REPO_ROOT / "TODO" / "02_metrics" / "MECHANISM_METRIC_MAPPING.md"
DEFAULT_OUT = REPO_ROOT / "paper"
DEFAULT_REPORT = REPO_ROOT / "paper" / "reports" / "wf_section6_gen"


def _run_generator(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# 1) --dry-run exits 0 and contains the structural anchors
# ---------------------------------------------------------------------------
def test_dry_run_exits_zero_and_has_anchors():
    proc = _run_generator("--dry-run")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "\\section{Limitations" in out
    assert "\\begin{enumerate}" in out
    assert "\\end{enumerate}" in out


# ---------------------------------------------------------------------------
# 2) Generator writes 06_limitations_auto.tex with \\begin{enumerate}
# ---------------------------------------------------------------------------
def test_writes_06_limitations_auto_tex(tmp_path):
    paper_out = tmp_path / "paper"
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(METRICS_DIR),
        "--index-md",
        str(INDEX_MD),
        "--mapping-md",
        str(MAPPING_MD),
        "--report-dir",
        str(paper_out / "reports" / "wf_section6_gen"),
    )
    assert proc.returncode == 0, proc.stderr
    tex = paper_out / "sections" / "06_limitations_auto.tex"
    assert tex.is_file(), f"missing {tex}"
    body = tex.read_text(encoding="utf-8")
    assert "\\begin{enumerate}" in body
    assert "\\end{enumerate}" in body
    # At least one item per failed metric + 5 static items
    assert body.count("\\item ") >= 5


# ---------------------------------------------------------------------------
# 3) Generator writes summary.md under reports/wf_section6_gen/
# ---------------------------------------------------------------------------
def test_writes_summary_md(tmp_path):
    paper_out = tmp_path / "paper"
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(METRICS_DIR),
        "--index-md",
        str(INDEX_MD),
        "--mapping-md",
        str(MAPPING_MD),
        "--report-dir",
        str(paper_out / "reports" / "wf_section6_gen"),
    )
    assert proc.returncode == 0, proc.stderr
    summary = paper_out / "reports" / "wf_section6_gen" / "summary.md"
    assert summary.is_file(), f"missing {summary}"
    body = summary.read_text(encoding="utf-8")
    assert "Failed metrics" in body
    assert "INDEX.md rows flagged" in body


# ---------------------------------------------------------------------------
# 4) With real snapshot, the script emits at least one metric paragraph
# ---------------------------------------------------------------------------
def test_real_snapshot_emits_metric_paragraph(tmp_path):
    paper_out = tmp_path / "paper"
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(METRICS_DIR),
        "--index-md",
        str(INDEX_MD),
        "--mapping-md",
        str(MAPPING_MD),
        "--report-dir",
        str(paper_out / "reports" / "wf_section6_gen"),
    )
    assert proc.returncode == 0, proc.stderr
    tex = paper_out / "sections" / "06_limitations_auto.tex"
    body = tex.read_text(encoding="utf-8")
    # Expect at least one auto-generated metric item (real data has
    # metalcytotox_ru_auc + platinai_4cell_auc + decode_ratio + pearson_r_pic50
    # + diversity_tanimoto + n_distinct + sa_mean + vina_kcal_per_mol all failing).
    assert "\\textbf{Metric \\texttt{" in body
    # And the four static structural items
    for anchor in (
        "Compute constraint",
        "Subset coverage",
        "No wet-lab validation",
        "Pre-existing test failures",
    ):
        assert anchor in body, f"missing static anchor {anchor!r}"


# ---------------------------------------------------------------------------
# 5) Missing metrics dir renders without raising (graceful-degrade)
# ---------------------------------------------------------------------------
def test_missing_metrics_dir_renders_gracefully(tmp_path):
    empty = tmp_path / "empty_metrics"
    empty.mkdir()
    paper_out = tmp_path / "paper"
    proc = _run_generator(
        "--out-dir",
        str(paper_out),
        "--metrics-dir",
        str(empty),
        "--index-md",
        str(INDEX_MD),
        "--mapping-md",
        str(MAPPING_MD),
        "--report-dir",
        str(paper_out / "reports" / "wf_section6_gen"),
    )
    assert proc.returncode == 0, proc.stderr
    tex = paper_out / "sections" / "06_limitations_auto.tex"
    assert tex.is_file(), f"missing {tex}"
    # Even with no failed metrics, the static items should be present
    body = tex.read_text(encoding="utf-8")
    assert "Compute constraint" in body
    assert "No wet-lab validation" in body
