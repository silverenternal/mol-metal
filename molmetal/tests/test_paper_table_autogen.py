"""Tests for molmetal.scripts.paper_table_autogen — R18 Table 2 auto-generator.

15 tests covering the public surface of ``paper_table_autogen.py``:

1.  test_autogen_module_imports                — module imports cleanly.
2.  test_autogen_catalog_nonempty              — catalog has >= 20 metrics.
3.  test_autogen_catalog_unique_metric_ids     — every metric id is unique.
4.  test_autogen_catalog_categories_present    — every row carries a category.
5.  test_autogen_build_with_empty_inputs       — empty inputs yield PENDING.
6.  test_autogen_build_with_real_platinai      — PlatinAI 500 fills R18 e2e.
7.  test_autogen_build_with_real_metalcytotox  — MetalCytoTox fills Ru rate.
8.  test_autogen_render_latex_basic            — basic LaTeX render.
9.  test_autogen_render_latex_has_table_env    — \\begin{table} + \\end{table}.
10. test_autogen_render_latex_columns          — 6 columns of header.
11. test_autogen_render_latex_section_headers  — midrule + section headers.
12. test_autogen_write_creates_file            — write_table_tex writes.
13. test_autogen_load_json_missing_returns_empty — _load_json safe on missing.
14. test_autogen_format_r18_e2e_handles_none   — _format_r18_e2e for None.
15. test_autogen_format_r18_e2e_handles_real   — _format_r18_e2e for real val.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so the module under test resolves
# regardless of pytest's working directory.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest

from molmetal.scripts import paper_table_autogen as autogen


# ---------------------------------------------------------------------------
# Helpers — tiny fixture inputs that mimic the real e2e JSON shape.
# ---------------------------------------------------------------------------
def _make_platinai(n: int = 500, ok_step3: int = 371) -> dict:
    """Build a PlatinAI-shape fixture dict with the minimum keys used."""
    return {
        "n_samples": n,
        "success_rate": 0.742,
        "total_wall_time_s": 154.41,
        "per_step_status_counts": {
            "step1_closure": {"ok": n, "skipped": 0, "error": 0},
            "step2_se3": {"ok": n, "skipped": 0, "error": 0},
            "step3_multiph": {"ok": ok_step3, "skipped": 0, "error": n - ok_step3},
            "step4_mcts": {"ok": n, "skipped": 0, "error": 0},
            "step5_hilbert": {"ok": n, "skipped": 0, "error": 0},
        },
    }


def _make_metalcytotox() -> dict:
    """Build a MetalCytoTox-shape fixture dict with the minimum keys used."""
    return {
        "n_samples_per_metal": 500,
        "per_metal_success_rate": {
            "Ru": 1.0,
            "Ir": 1.0,
            "Rh": 0.998,
            "Re": 1.0,
            "Os": 0.976,
        },
        "per_step_per_metal_status_counts": {
            "step3_multiph": {
                "Ru": {"ok": 500, "skipped": 0, "error": 0},
                "Ir": {"ok": 500, "skipped": 0, "error": 0},
                "Rh": {"ok": 499, "skipped": 0, "error": 1},
                "Re": {"ok": 500, "skipped": 0, "error": 0},
                "Os": {"ok": 488, "skipped": 0, "error": 12},
            },
        },
    }


# ---------------------------------------------------------------------------
# 1) Module imports cleanly
# ---------------------------------------------------------------------------
def test_autogen_module_imports():
    assert autogen is not None
    assert hasattr(autogen, "build_table_rows")
    assert hasattr(autogen, "render_latex")
    assert hasattr(autogen, "write_table_tex")
    assert hasattr(autogen, "PLATINAI_E2E_JSON")
    assert hasattr(autogen, "METALCYTOTOX_E2E_JSON")
    assert hasattr(autogen, "DEFAULT_OUT_TEX")


# ---------------------------------------------------------------------------
# 2) Catalog is non-empty and has at least 20 rows
# ---------------------------------------------------------------------------
def test_autogen_catalog_nonempty():
    rows = autogen.build_table_rows(platinai={}, metalcytotox={})
    assert len(rows) >= 20, f"expected >=20 rows, got {len(rows)}"
    # Every row is a Table2Row dataclass instance.
    assert all(isinstance(r, autogen.Table2Row) for r in rows)


# ---------------------------------------------------------------------------
# 3) Metric ids are unique across the catalog
# ---------------------------------------------------------------------------
def test_autogen_catalog_unique_metric_ids():
    rows = autogen.build_table_rows(platinai={}, metalcytotox={})
    ids = [r.metric_id for r in rows]
    dupes = [x for x in set(ids) if ids.count(x) > 1]
    assert not dupes, f"duplicate metric ids in catalog: {dupes}"


# ---------------------------------------------------------------------------
# 4) Every row carries a non-empty category + mechanism + direction
# ---------------------------------------------------------------------------
def test_autogen_catalog_categories_present():
    rows = autogen.build_table_rows(platinai={}, metalcytotox={})
    for r in rows:
        assert r.category, f"{r.metric_id}: empty category"
        assert r.mechanism, f"{r.metric_id}: empty mechanism"
        assert r.direction in (
            "higher",
            "lower",
        ), f"{r.metric_id}: bad direction {r.direction!r}"
        assert r.r17_value, f"{r.metric_id}: empty R17 value"
        assert r.target_value, f"{r.metric_id}: empty target value"


# ---------------------------------------------------------------------------
# 5) Empty inputs yield PENDING-tagged R18 e2e column
# ---------------------------------------------------------------------------
def test_autogen_build_with_empty_inputs():
    rows = autogen.build_table_rows(platinai={}, metalcytotox={})
    pending_count = sum(1 for r in rows if "PENDING" in r.r18_e2e_value)
    # Every row that depends on an e2e extractor must be PENDING.
    assert pending_count >= 1, "expected at least one PENDING row"
    # R17 columns must still be populated even with empty inputs.
    assert all(r.r17_value for r in rows)


# ---------------------------------------------------------------------------
# 6) Real PlatinAI input fills the R18 e2e column for a success-rate row
# ---------------------------------------------------------------------------
def test_autogen_build_with_real_platinai():
    platinai = _make_platinai(n=500, ok_step3=371)
    rows = autogen.build_table_rows(platinai=platinai, metalcytotox={})
    # The m2_pt_a2780cis row uses _extract_platinai_success_rate -> 0.742
    a2780 = next(r for r in rows if r.metric_id == "m2_pt_a2780cis")
    assert "0.742" in a2780.r18_e2e_value
    assert "500" in a2780.r18_e2e_value
    assert "PENDING" not in a2780.r18_e2e_value


# ---------------------------------------------------------------------------
# 7) Real MetalCytoTox input fills Ru rate + NCI-60 overall
# ---------------------------------------------------------------------------
def test_autogen_build_with_real_metalcytotox():
    metal = _make_metalcytotox()
    rows = autogen.build_table_rows(platinai={}, metalcytotox=metal)
    ru = next(r for r in rows if r.metric_id == "m1")
    assert "1.000" in ru.r18_e2e_value
    assert "500" in ru.r18_e2e_value

    nci = next(r for r in rows if r.metric_id == "m21_nci60")
    # mean over {1.0, 1.0, 0.998, 1.0, 0.976} = 0.9948
    assert "0.99" in nci.r18_e2e_value or "0.995" in nci.r18_e2e_value


# ---------------------------------------------------------------------------
# 8) render_latex returns a non-empty string
# ---------------------------------------------------------------------------
def test_autogen_render_latex_basic():
    rows = autogen.build_table_rows(
        platinai=_make_platinai(), metalcytotox=_make_metalcytotox()
    )
    body = autogen.render_latex(rows)
    assert isinstance(body, str)
    assert len(body) > 1000, "rendered LaTeX is suspiciously short"


# ---------------------------------------------------------------------------
# 9) render_latex wraps the table in a LaTeX table env
# ---------------------------------------------------------------------------
def test_autogen_render_latex_has_table_env():
    rows = autogen.build_table_rows(platinai=_make_platinai(), metalcytotox={})
    body = autogen.render_latex(rows)
    assert "\\begin{table}" in body
    assert "\\end{table}" in body
    assert "\\begin{tabular}" in body
    assert "\\end{tabular}" in body
    assert "\\toprule" in body
    assert "\\bottomrule" in body


# ---------------------------------------------------------------------------
# 10) Header has the 6 documented columns
# ---------------------------------------------------------------------------
def test_autogen_render_latex_columns():
    rows = autogen.build_table_rows(platinai=_make_platinai(), metalcytotox={})
    body = autogen.render_latex(rows)
    # Six columns by header field-name, in document order:
    for needle in (
        "\\textbf{Metric}",
        "\\textbf{R17 (MEASURED)}",
        "\\textbf{R18 e2e (MEASURED)}",
        "\\textbf{R18 GPU (PROJECTED)}",
        "\\textbf{Dir.}",
        "\\textbf{Responsible mechanism}",
    ):
        assert needle in body, f"missing column header: {needle}"


# ---------------------------------------------------------------------------
# 11) Midrule separators and category mid-rows are emitted
# ---------------------------------------------------------------------------
def test_autogen_render_latex_section_headers():
    rows = autogen.build_table_rows(platinai=_make_platinai(), metalcytotox={})
    body = autogen.render_latex(rows)
    # At least one \multicolumn{6}{l}{\textit{...}} midrow.
    assert "\\multicolumn{6}{l}{\\textit{" in body
    # The default catalog has at least 7 categories.
    assert body.count("\\multicolumn{6}{l}{\\textit{") >= 7


# ---------------------------------------------------------------------------
# 12) write_table_tex creates a file
# ---------------------------------------------------------------------------
def test_autogen_write_creates_file(tmp_path):
    out = tmp_path / "table_r18.tex"
    rows = autogen.build_table_rows(
        platinai=_make_platinai(), metalcytotox=_make_metalcytotox()
    )
    written = autogen.write_table_tex(rows, out, overwrite=True)
    assert written == out
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "\\begin{table}" in text
    assert len(text) > 500


# ---------------------------------------------------------------------------
# 13) _load_json returns {} for a missing path
# ---------------------------------------------------------------------------
def test_autogen_load_json_missing_returns_empty(tmp_path):
    missing = tmp_path / "nope.json"
    result = autogen._load_json(missing)
    assert result == {}
    # And it must not raise.
    rows = autogen.build_table_rows(platinai=result, metalcytotox=result)
    assert len(rows) >= 20


# ---------------------------------------------------------------------------
# 14) _format_r18_e2e handles None value
# ---------------------------------------------------------------------------
def test_autogen_format_r18_e2e_handles_none():
    out = autogen._format_r18_e2e("m1", None, 0)
    assert "PENDING" in out

    # With samples but None value (e.g. MetalCytoTox missing):
    out = autogen._format_r18_e2e("m1", None, 500)
    assert "PENDING" in out


# ---------------------------------------------------------------------------
# 15) _format_r18_e2e handles a real number
# ---------------------------------------------------------------------------
def test_autogen_format_r18_e2e_handles_real():
    out = autogen._format_r18_e2e("m1", 0.712, 500)
    assert "0.712" in out
    assert "MEASURED" in out
    assert "500" in out

    # Wall-time style metrics (m3_vina / m10_vina_min) emit seconds.
    out = autogen._format_r18_e2e("m3_vina", 154.41, 500)
    assert "\\,s" in out
    assert "500" in out

    # Out-of-range values fall back to percent.
    out = autogen._format_r18_e2e("unknown_metric", 1.0, 100)
    assert "MEASURED" in out
    assert "100" in out
