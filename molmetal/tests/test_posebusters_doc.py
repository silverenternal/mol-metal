"""Smoke test for the PoseBusters column in the paper-numbers doc.

The TODO-06 audit confirms:

- :func:`molmetal.molmetal_lam.sbdd_env.posebusters_adapter._optimize_3d`
  uses :func:`rdkit.Chem.AllChem.MMFF94OptimizeMolecule` (round-5
  refactor, line 158).
- :file:`molmetal/reports/mmff94_fix.md` is the canonical post-fix
  benchmark: 13/13 = 100% pass-rate on the hand-drawn CuAAC/SPAAC
  products and 12/12 = 100% on the 12 click tiles.
- :file:`molmetal/reports/lambda_vs_sbdd_paper_numbers.md` §3 and §3.1
  were updated by TODO-06 to reflect the post-fix pass-rate and to
  point to ``mmff94_fix.md`` as the canonical reference.

This test is deliberately light: a doc-update regression is caught by
a single grep on the canonical phrases.  Running the PoseBusters
adapter end-to-end is covered by
:mod:`molmetal.tests.test_posebusters_adapter`.
"""

from __future__ import annotations

from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
TARGET_DOC = REPORTS_DIR / "lambda_vs_sbdd_paper_numbers.md"
MMFF_FIX_DOC = REPORTS_DIR / "mmff94_fix.md"


def test_lambda_vs_sbdd_doc_exists():
    """The paper-numbers doc must exist."""
    assert TARGET_DOC.exists(), f"missing doc at {TARGET_DOC}"


def test_lambda_vs_sbdd_doc_references_mmff94_fix():
    """The doc must cross-reference the canonical ``mmff94_fix.md``."""
    text = TARGET_DOC.read_text()
    assert "mmff94_fix.md" in text, (
        "lambda_vs_sbdd_paper_numbers.md must reference mmff94_fix.md as the "
        "canonical source for the PoseBusters post-fix pass-rate."
    )


def test_lambda_vs_sbdd_doc_posebusters_column_updated():
    """The PoseBusters column must reflect the post-fix 12/12 + 13/13."""
    text = TARGET_DOC.read_text()
    # TL;DR table: click tiles column must say 12/12, not 0/12.
    assert "12/12 = 100%" in text, (
        "PoseBusters click-tiles column must be 12/12 = 100% after the "
        "MMFF94 fix; the old 0/12 entry has been retired."
    )
    # §3.1 CuAAC products: must show 13/13.
    assert "13 / 13 = 100%" in text or "13/13" in text, (
        "PoseBusters CuAAC/SPAAC products column must be 13/13 = 100% after "
        "the MMFF94 fix; the old 0/13 entry has been retired."
    )


def test_posebusters_adapter_uses_mmff94():
    """Adapter must call ``MMFF94OptimizeMolecule`` (round-5 audit point)."""
    adapter = (
        Path(__file__).resolve().parent.parent
        / "molmetal_lam"
        / "sbdd_env"
        / "posebusters_adapter.py"
    )
    text = adapter.read_text()
    assert "MMFF94OptimizeMolecule" in text, (
        f"posebusters_adapter.py must call MMFF94OptimizeMolecule; the file "
        f"at {adapter} does not."
    )


def test_mmff94_fix_doc_exists():
    """The canonical fix doc must exist."""
    assert MMFF_FIX_DOC.exists(), f"missing {MMFF_FIX_DOC}"