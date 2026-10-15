# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Regression tests for :mod:`molmetal.eval.reporting.cite_helper`.

Wf-A49 (2026-10-16) — A2 added ``cite_rusanov_ece`` which threads the
on-disk Rusanov 2025 SI Table S5 ECE JSON
(``.data/molmetal/rusanov_2025/parsed/s5_ece.json``) into the paper
Table 1 footnote.  This test asserts:

  * the new function is exported from :mod:`cite_helper` (the public
    surface change for downstream consumers),
  * the cite line names the on-disk JSON path so the source is
    verifiable per AGENTS.md §5.5 filesystem-truth,
  * the optional ``per_cell`` / ``per_split`` kwargs surface as text
    in the cite line,
  * the S7 Pt-wet-lab JSON cross-ref also appears.

The function is **read-only** on the JSON — it does not validate the
ECE numbers against the file.  Callers wanting a per-cell / per-split
ECE number must read the JSON themselves; this test only checks the
cite block, not the numbers (which is the cite_helper contract).
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


def _load_cite_helper():
    """Load cite_helper.py directly to avoid pulling the full eval tree."""
    # __file__ = molmetal/tests/test_cite_helper.py; parents[0] = molmetal/tests,
    # parents[1] = molmetal.  cite_helper.py lives under molmetal/eval/reporting.
    path = Path(__file__).resolve().parents[1] / "eval/reporting/cite_helper.py"
    spec = importlib.util.spec_from_file_location("cite_helper_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cite_rusanov_ece_in_all() -> None:
    """The public surface now exports cite_rusanov_ece."""
    ch = _load_cite_helper()
    assert "cite_rusanov_ece" in ch.__all__
    assert callable(ch.cite_rusanov_ece)


def test_cite_rusanov_ece_names_json_path() -> None:
    """The cite line names the on-disk s5_ece.json path (§5.5)."""
    ch = _load_cite_helper()
    line = ch.cite_rusanov_ece()
    assert ".data/molmetal/rusanov_2025/parsed/s5_ece.json" in line
    # S7 Pt-wet-lab cross-ref is also named.
    assert ".data/molmetal/rusanov_2025/parsed/s7_pt_wet_lab.json" in line


def test_cite_rusanov_ece_accepts_per_cell_per_split() -> None:
    """per_cell and per_split kwargs surface as text in the cite line."""
    ch = _load_cite_helper()
    line_a2780 = ch.cite_rusanov_ece(per_cell="A2780")
    assert "A2780" in line_a2780
    line_scaffold = ch.cite_rusanov_ece(per_split="scaffold-based")
    assert "scaffold-based" in line_scaffold
    line_both = ch.cite_rusanov_ece(
        per_cell="MCF-7", per_split="cluster-based",
    )
    assert "MCF-7" in line_both
    assert "cluster-based" in line_both


def test_cite_rusanov_ece_default_lists_all_cells_and_splits() -> None:
    """When no kwargs are passed, all 4 cell lines + 3 split types are listed."""
    ch = _load_cite_helper()
    line = ch.cite_rusanov_ece()
    for cell in ("A2780", "A2780cis", "MCF-7", "A549"):
        assert cell in line, f"missing cell line {cell!r} in cite line"
    for split in ("scaffold-based", "random", "cluster-based"):
        assert split in line, f"missing split type {split!r} in cite line"


def test_cite_rusanov_ece_is_str() -> None:
    """Return type is a plain str (BibTeX-ready)."""
    ch = _load_cite_helper()
    line = ch.cite_rusanov_ece()
    assert isinstance(line, str)
    assert line.endswith(".")  # sentence-terminated


@pytest.mark.parametrize(
    "cell",
    ["A2780", "A2780cis", "MCF-7", "A549"],
)
def test_cite_rusanov_ece_per_cell_parametrized(cell: str) -> None:
    """Each of the 4 documented cell lines surfaces in the cite line."""
    ch = _load_cite_helper()
    line = ch.cite_rusanov_ece(per_cell=cell)
    assert cell in line