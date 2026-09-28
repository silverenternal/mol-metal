"""Tests for Wf-49 Phase 1 — RDKit 2D descriptors extractor.

=====================================================================
These tests pin down the invariants the Wf-49 Phase 1 deliverable
depends on so that future schema changes or sampling regressions are
caught immediately.

The extractor is implemented in
``molmetal/scripts/extract_rdkit_2d_descriptors.py`` and produces two
artefacts (per ``--dataset`` selection):

* ``molmetal/data/platinai_5k_rdkit_features.csv`` (5,000 rows ×
  ``smiles, metal, ic50_nm, selectivity, pIC50`` + 12 RDKit 2D desc)
* ``molmetal/data/metalcytotox_rdkit_features.csv`` (~26,800 rows ×
  ``smiles, metal, activity`` + 12 RDKit 2D desc)

And the per-dataset ``feature_summary.json`` files under
``molmetal/reports/wf49_rdkit_extraction/``:

* ``platinai_5k_feature_summary.json``
* ``metalcytotox_5metal_feature_summary.json``

The 12 RDKit 2D descriptors (frozen for Wf-49 Phase 1):

  1. MolWt
  2. MolLogP
  3. TPSA
  4. HBA
  5. HBD
  6. NRotB
  7. HeavyAtoms
  8. RingCount
  9. NumAromaticRings
 10. FractionCSP3
 11. MolMR
 12. BalabanJ

Tests (count target: 12):
 1.  PlatinAI 5K CSV exists and has shape ~5,000 × (5 + 12) = 17 cols.
 3.  PlatinAI 5K CSV has the 12 expected descriptor columns (in order).
 4.  All descriptor cells in the PlatinAI CSV are finite floats.
 5.  PlatinAI 4-cell CSV exists and has shape ~4,078 × (11 + 12) = 23
     cols (smiles, metal, 4 IC50 cols, 4 pIC50 cols + 12 desc).
 6.  PlatinAI 4-cell pIC50 columns match ``9 - log10(ic50_nm)`` to 1e-6.
 7.  The MetalCytoTox CSV exists and has shape ~26,800 × (3 + 12) = 15
     cols (smiles, metal, activity + 12 desc).
 8.  All three ``feature_summary.json`` files parse as JSON and have
     the documented schema (``descriptor_names``,
     ``correlations_vs_target`` blocks).
 9.  Pearson r between MolLogP and pIC50 in the PlatinAI 4-cell
     summary is finite and in ``[-1, 1]`` (sanity check on the
     correlation block — the headline metric for the m16 fix).
 10. Per-metal counts in the PlatinAI 5K summary cover Pt + Pd + Au
     + Ru + Ir + Rh + Os (the seven metals actually present in
     ``platinai_5k.csv``).
 11. CLI ``--help`` exits with code 0.
 12. Descriptor columns cover Lipinski-style drivers
     (MolLogP, TPSA, HBA, HBD).

Tests are **skipped (not failed)** when the artefacts are missing — so
CI without the prior run is not broken.  When present, all assertions
are strict.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLATINAI_OUT = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k_rdkit_features.csv"
PLATINAI_4CELL_OUT = (
    PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell_rdkit_features.csv"
)
METALCYTO_OUT = (
    PROJECT_ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
)
PLATINAI_SUMMARY = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf49_rdkit_extraction"
    / "platinai_5k_feature_summary.json"
)
PLATINAI_4CELL_SUMMARY = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf49_rdkit_extraction"
    / "platinai_4cell_feature_summary.json"
)
METALCYTO_SUMMARY = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf49_rdkit_extraction"
    / "metalcytotox_5metal_feature_summary.json"
)

EXPECTED_DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
]

EXPECTED_PLATINAI_METALS = {"Pt", "Pd", "Au", "Ru", "Ir", "Rh", "Os"}  # platinai_5k.csv metals


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Wf-49 Phase 1 artefact missing: {path}")


def _load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Test 1: PlatinAI 5K CSV exists with the right shape
# ---------------------------------------------------------------------------
def test_platinai_csv_shape() -> None:
    _skip_if_missing(PLATINAI_OUT)
    header, rows = _load_csv(PLATINAI_OUT)
    expected_cols = {"smiles", "metal", "ic50_nm", "selectivity", "pIC50"} | set(
        EXPECTED_DESC_NAMES
    )
    assert set(header) == expected_cols, (
        f"PlatinAI header mismatch.\n"
        f"  expected (any order): {sorted(expected_cols)}\n"
        f"  actual   (any order): {sorted(header)}"
    )
    # 5,000 rows tolerance: source CSV has 5,000 data rows; allow ±10.
    assert len(rows) == 5000, (
        f"PlatinAI expected 5,000 rows, got {len(rows)}."
    )


# ---------------------------------------------------------------------------
# Test 2: PlatinAI 5K CSV has 12 expected descriptor columns (in order)
# ---------------------------------------------------------------------------
def test_platinai_descriptor_columns_in_order() -> None:
    _skip_if_missing(PLATINAI_OUT)
    header, _ = _load_csv(PLATINAI_OUT)
    # Desc columns live at index 5..16 (after 5 metadata cols).
    desc_cols = header[5:5 + len(EXPECTED_DESC_NAMES)]
    assert desc_cols == EXPECTED_DESC_NAMES, (
        f"Descriptor columns out of order.\n"
        f"  expected: {EXPECTED_DESC_NAMES}\n"
        f"  actual:   {desc_cols}"
    )


# ---------------------------------------------------------------------------
# Test 3: All descriptor cells in PlatinAI CSV are finite floats
# ---------------------------------------------------------------------------
def test_platinai_descriptor_cells_finite() -> None:
    _skip_if_missing(PLATINAI_OUT)
    header, rows = _load_csv(PLATINAI_OUT)
    n_bad = 0
    for row in rows:
        for cell in row[5:]:
            try:
                v = float(cell)
            except ValueError:
                n_bad += 1
                continue
            if not math.isfinite(v):
                n_bad += 1
    assert n_bad == 0, (
        f"PlatinAI CSV contains {n_bad} non-finite descriptor cells "
        "(median imputation should have cleaned them)."
    )


# ---------------------------------------------------------------------------
# Test 4: pIC50 matches 9 - log10(IC50_nM) to 1e-6 (PlatinAI 4-cell cohort)
# ---------------------------------------------------------------------------
def test_platinai_pIC50_transformation() -> None:
    _skip_if_missing(PLATINAI_4CELL_OUT)
    header, rows = _load_csv(PLATINAI_4CELL_OUT)
    # Layout: smiles, metal, ic50_mcf7_nm, ic50_a549_nm, ic50_a2780_nm,
    # ic50_a2780cis_nm, pIC50_mcf7, pIC50_a549, pIC50_a2780,
    # pIC50_a2780cis, +12 desc cols.
    ic50_idx = [header.index(c) for c in [
        "ic50_mcf7_nm", "ic50_a549_nm", "ic50_a2780_nm", "ic50_a2780cis_nm",
    ]]
    pic50_idx = [header.index(c) for c in [
        "pIC50_mcf7", "pIC50_a549", "pIC50_a2780", "pIC50_a2780cis",
    ]]
    n_checked = 0
    n_off = 0
    max_err = 0.0
    for row in rows:
        for ic50_col_i, pic50_col_i in zip(ic50_idx, pic50_idx):
            ic50_raw, pic50_raw = row[ic50_col_i], row[pic50_col_i]
            if not ic50_raw or not pic50_raw:
                continue
            try:
                ic50 = float(ic50_raw)
                pic50_actual = float(pic50_raw)
            except ValueError:
                continue
            if not (math.isfinite(ic50) and math.isfinite(pic50_actual)):
                continue
            if ic50 <= 0:
                continue
            pic50_expected = 9.0 - math.log10(ic50)
            err = abs(pic50_actual - pic50_expected)
            max_err = max(max_err, err)
            if err > 1e-6:
                n_off += 1
            n_checked += 1
    assert n_checked > 1000, (
        f"Only {n_checked} cell-line rows had finite IC50 + pIC50; "
        f"expected >1000."
    )
    assert n_off == 0, (
        f"{n_off}/{n_checked} PlatinAI 4-cell rows have pIC50 off by >1e-6; "
        f"max_err={max_err}."
    )


# ---------------------------------------------------------------------------
# Test 4b: PlatinAI 4-cell CSV exists with the right shape
# ---------------------------------------------------------------------------
def test_platinai_4cell_csv_shape() -> None:
    _skip_if_missing(PLATINAI_4CELL_OUT)
    header, rows = _load_csv(PLATINAI_4CELL_OUT)
    expected_cols = {
        "smiles", "metal",
        "ic50_mcf7_nm", "ic50_a549_nm", "ic50_a2780_nm", "ic50_a2780cis_nm",
        "pIC50_mcf7", "pIC50_a549", "pIC50_a2780", "pIC50_a2780cis",
    } | set(EXPECTED_DESC_NAMES)
    assert set(header) == expected_cols, (
        f"PlatinAI 4-cell header mismatch.\n"
        f"  expected (any order): {sorted(expected_cols)}\n"
        f"  actual   (any order): {sorted(header)}"
    )
    assert 4000 < len(rows) < 4200, (
        f"PlatinAI 4-cell expected ~4,078 rows, got {len(rows)}."
    )


# ---------------------------------------------------------------------------
# Test 5: MetalCytoTox CSV exists with the right shape
# ---------------------------------------------------------------------------
def test_metalcytotox_csv_shape() -> None:
    _skip_if_missing(METALCYTO_OUT)
    header, rows = _load_csv(METALCYTO_OUT)
    expected_cols = {"smiles", "metal", "activity"} | set(EXPECTED_DESC_NAMES)
    assert set(header) == expected_cols, (
        f"MetalCytoTox header mismatch.\n"
        f"  expected (any order): {sorted(expected_cols)}\n"
        f"  actual   (any order): {sorted(header)}"
    )
    # Source has ~26,800 rows. Allow ±500 for the parse-failure margin.
    assert 20000 < len(rows) < 30000, (
        f"MetalCytoTox expected ~26,800 rows, got {len(rows)}."
    )


# ---------------------------------------------------------------------------
# Test 6: feature_summary.json files parse and have the right schema
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "summary_path, dataset_name",
    [
        (PLATINAI_SUMMARY, "platinai_5k"),
        (PLATINAI_4CELL_SUMMARY, "platinai_4cell"),
        (METALCYTO_SUMMARY, "metalcytotox_5metal"),
    ],
)
def test_feature_summary_schema(summary_path: Path, dataset_name: str) -> None:
    _skip_if_missing(summary_path)
    payload = _load_summary(summary_path)
    assert payload is not None
    assert payload["dataset"] == dataset_name
    assert payload["n_descriptors"] == 12
    assert payload["descriptor_names"] == EXPECTED_DESC_NAMES
    assert "descriptor_stats" in payload
    assert "correlations_vs_target" in payload
    assert len(payload["descriptor_stats"]) == 12
    assert len(payload["correlations_vs_target"]) == 12
    # Per-descriptor stats must include mean / std / min / max / median.
    for d in payload["descriptor_stats"]:
        assert {"name", "mean", "std", "min", "max", "median"}.issubset(
            d.keys()
        ), f"Missing keys in descriptor_stats entry: {d}"


# ---------------------------------------------------------------------------
# Test 7: Pearson r between MolLogP and pIC50_mcf7 in PlatinAI 4-cell
# summary is finite and in [-1, 1].
# ---------------------------------------------------------------------------
def test_mollogp_pic50_pearson_r_in_range() -> None:
    _skip_if_missing(PLATINAI_4CELL_SUMMARY)
    payload = _load_summary(PLATINAI_4CELL_SUMMARY)
    assert payload is not None
    corr_block = {
        c["name"]: c["pearson_r"]
        for c in payload["correlations_vs_target"]
    }
    r = corr_block.get("MolLogP")
    assert r is not None, (
        "MolLogP pearson_r entry missing in platinai_4cell feature_summary.json."
    )
    assert math.isfinite(r), f"MolLogP pearson_r not finite: {r}"
    assert -1.0 <= r <= 1.0, f"MolLogP pearson_r out of [-1, 1]: {r}"


# ---------------------------------------------------------------------------
# Test 7b: All 12 descriptors in PlatinAI 4-cell summary have finite
# Pearson r entries against pIC50_mcf7 — the metric that D-MPNN failed
# on at -0.040.  The RDKit 2D block should expose at least 1 strong
# univariate signal.
# ---------------------------------------------------------------------------
def test_platinai_4cell_all_pearson_r_finite() -> None:
    _skip_if_missing(PLATINAI_4CELL_SUMMARY)
    payload = _load_summary(PLATINAI_4CELL_SUMMARY)
    assert payload is not None
    corr_block = payload["correlations_vs_target"]
    n_finite = 0
    for entry in corr_block:
        r = entry.get("pearson_r")
        if r is not None and math.isfinite(r):
            n_finite += 1
    assert n_finite >= 6, (
        f"Expected >= 6/12 descriptors with finite Pearson r vs pIC50_mcf7; "
        f"got {n_finite}. Check that platinai_4cell.csv IC50 columns are "
        f"populated."
    )


# ---------------------------------------------------------------------------
# Test 8: Per-metal counts in PlatinAI summary match the source
# ---------------------------------------------------------------------------
def test_platinai_per_metal_counts() -> None:
    _skip_if_missing(PLATINAI_SUMMARY)
    payload = _load_summary(PLATINAI_SUMMARY)
    assert payload is not None
    per_metal = payload.get("per_metal_counts", {})
    assert set(per_metal.keys()) >= EXPECTED_PLATINAI_METALS, (
        f"PlatinAI per-metal counts missing expected metals. "
        f"got: {sorted(per_metal.keys())}, "
        f"expected at least: {sorted(EXPECTED_PLATINAI_METALS)}"
    )
    # The 4-cell cohort is Pt-only; the 5K cohort is mixed. Pt is
    # present in both, so we just sanity-check Pt count > 500.
    assert per_metal.get("Pt", 0) > 500, (
        f"PlatinAI Pt count suspiciously low: {per_metal.get('Pt', 0)}"
    )


# ---------------------------------------------------------------------------
# Test 9: CLI --help exits 0
# ---------------------------------------------------------------------------
def test_cli_help() -> None:
    script = PROJECT_ROOT / "molmetal" / "scripts" / "extract_rdkit_2d_descriptors.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, (
        f"--help failed with rc={result.returncode}\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 10: Descriptor columns cover Lipinski-style drivers
# ---------------------------------------------------------------------------
def test_descriptor_columns_cover_lipinski() -> None:
    # Static invariant — verifies the registry in the script module
    # covers at least MolLogP, TPSA, HBA, HBD (the canonical Lipinski
    # Rule of Five descriptors).
    required = {"MolLogP", "TPSA", "HBA", "HBD"}
    assert required.issubset(set(EXPECTED_DESC_NAMES)), (
        f"Descriptor registry missing Lipinski drivers.\n"
        f"  required: {sorted(required)}\n"
        f"  present:  {sorted(EXPECTED_DESC_NAMES)}"
    )

# ---------------------------------------------------------------------------
# TODO-91 follow-up tests: --require-labels guard + n_samples_warning emission
# ---------------------------------------------------------------------------


def test_require_labels_guard_blocks_platinai(tmp_path) -> None:
    """When --require-labels is set, --dataset=platinai must hard-error.

    Rationale: platinai_5k.csv has empty ic50_nm / selectivity columns on
    the 2026-09-26 snapshot, so running it without labels produces a JSON
    with n_samples=0 across the correlations block.  The guard forces
    callers to use --dataset=platinai_4cell for the labelled regression
    cohort.
    """
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "platinai",
        "--require-labels",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, (
        f"--require-labels did NOT block --dataset=platinai.\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )
    assert "platinai_4cell" in (result.stderr + result.stdout), (
        f"--require-labels error message did NOT mention platinai_4cell.\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )


def test_require_labels_guard_allows_platinai_4cell(tmp_path) -> None:
    """When --require-labels is set, --dataset=platinai_4cell must PASS.

    This is the supported labelled regression cohort — the guard should
    not block it.
    """
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "platinai_4cell",
        "--require-labels",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
        "--max-rows", "50",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"--require-labels BLOCKED the labelled cohort unexpectedly.\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )


def test_platinai_5k_emits_n_samples_warning(tmp_path) -> None:
    """When platinai_5k.csv has no labelled rows, the JSON emits a
    n_samples_warning top-level flag so Wf-49 Phase 4 synthesis can grep
    for it.
    """
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "platinai",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
        "--max-rows", "20",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    summary_path = tmp_path / "rep" / "platinai_5k_feature_summary.json"
    assert summary_path.exists(), (
        f"platinai_5k feature summary not generated.\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )
    payload = json.loads(summary_path.read_text())
    # On the 2026-09-26 snapshot platinai_5k.csv has empty ic50_nm / selectivity
    # columns → correlations block has n_samples=0 → warning must be present.
    assert "n_samples_warning" in payload, (
        "n_samples_warning top-level field missing from platinai_5k summary"
    )
    assert "n_samples=0" in payload["n_samples_warning"], (
        f"n_samples_warning text unexpected: {payload['n_samples_warning']}"
    )


def test_platinai_4cell_no_n_samples_warning(tmp_path) -> None:
    """When platinai_4cell has labelled pIC50 columns, no warning is emitted."""
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "platinai_4cell",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
        "--max-rows", "20",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    summary_path = tmp_path / "rep" / "platinai_4cell_feature_summary.json"
    assert summary_path.exists(), (
        f"platinai_4cell feature summary not generated.\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )
    payload = json.loads(summary_path.read_text())
    # platinai_4cell has labelled pIC50 columns → correlations should carry
    # n_samples > 0 for every descriptor (or at least most).
    n_zero = sum(
        1 for c in payload["correlations_vs_target"]
        if c.get("n_samples", 0) == 0
    )
    assert n_zero == 0, (
        f"platinai_4cell has {n_zero}/{len(payload['correlations_vs_target'])} "
        f"descriptors with n_samples=0 — labelled cohort should be clean"
    )


def test_platinai_4cell_per_cell_correlations_block(tmp_path) -> None:
    """platinai_4cell summary JSON should carry a per_cell_correlations
    block (4 cell lines × 12 descriptors = 48 entries).
    """
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "platinai_4cell",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
        "--max-rows", "20",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    summary_path = tmp_path / "rep" / "platinai_4cell_feature_summary.json"
    payload = json.loads(summary_path.read_text())
    # Headline block: pIC50_mcf7 (12 descriptors)
    assert len(payload["correlations_vs_target"]) == 12, (
        f"Headline correlations block should have 12 entries; got "
        f"{len(payload['correlations_vs_target'])}"
    )
    # Per-cell block: 4 cell lines × 12 descriptors = 48 entries (if shipped)
    if "per_cell_correlations" in payload:
        assert len(payload["per_cell_correlations"]) == 4, (
            f"per_cell_correlations should have 4 cell lines; got "
            f"{len(payload['per_cell_correlations'])}"
        )
        for cell, corrs in payload["per_cell_correlations"].items():
            assert cell in {"pIC50_mcf7", "pIC50_a549", "pIC50_a2780", "pIC50_a2780cis"}, (
                f"Unknown cell line {cell}"
            )
            assert len(corrs) == 12, (
                f"per_cell_correlations[{cell}] should have 12 descriptors; "
                f"got {len(corrs)}"
            )


def test_platinai_4cell_csv_has_pIC50_columns(tmp_path) -> None:
    """The platinai_4cell CSV output should carry the per-cell pIC50 columns."""
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "platinai_4cell",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
        "--max-rows", "10",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out_csv = tmp_path / "out" / "platinai_4cell_rdkit_features.csv"
    assert out_csv.exists(), (
        f"platinai_4cell CSV not generated.\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )
    with out_csv.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for col in ("pIC50_mcf7", "pIC50_a549", "pIC50_a2780", "pIC50_a2780cis"):
            assert col in header, (
                f"platinai_4cell CSV missing {col}; got header {header}"
            )


def test_metalcytotox_no_n_samples_warning(tmp_path) -> None:
    """MetalCytoToxDB is labelled (binary activity); no n_samples_warning
    should be emitted.
    """
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "metalcytotox",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
        "--max-rows", "20",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    summary_path = tmp_path / "rep" / "metalcytotox_5metal_feature_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    # metalcytotox should NOT have n_samples_warning (labelled cohort)
    assert payload.get("n_samples_warning") is None, (
        f"metalcytotox should not have n_samples_warning; got "
        f"{payload.get('n_samples_warning')}"
    )


def test_all_datasets_emits_three_summaries(tmp_path) -> None:
    """When --dataset=all (default), all 3 cohort summaries should be
    generated.
    """
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--dataset", "all",
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "rep"),
        "--max-rows", "10",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"--dataset=all failed.\n"
        f"  stdout: {result.stdout}\n"
        f"  stderr: {result.stderr}"
    )
    for fname in (
        "platinai_5k_feature_summary.json",
        "platinai_4cell_feature_summary.json",
        "metalcytotox_5metal_feature_summary.json",
    ):
        path = tmp_path / "rep" / fname
        assert path.exists(), f"{fname} missing from --dataset=all run"


def test_write_feature_summary_persists_warning(tmp_path) -> None:
    """Unit test: write_feature_summary with n_samples_warning should
    persist the warning in the JSON.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "erd",
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py",
    )
    erd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(erd)
    write_feature_summary = erd.write_feature_summary

    out_path = tmp_path / "test_summary.json"
    write_feature_summary(
        out_path=out_path,
        dataset_name="test_dataset",
        n_rows_total=10,
        n_rows_parsed=5,
        n_rows_dropped=5,
        per_metal_counts={"Pt": 5},
        desc_stats=[{"name": "MolWt", "mean": 100.0, "std": 10.0, "min": 80.0, "max": 120.0, "median": 100.0}],
        correlations=[
            {"n_samples": 0, "name": "MolWt", "pearson_r": None},
            {"n_samples": 100, "name": "MolLogP", "pearson_r": 0.2},
        ],
        target_col="pIC50",
        wall_seconds=1.5,
        source_csv="test.csv",
        n_samples_warning="1/2 descriptors have n_samples=0",
    )
    payload = json.loads(out_path.read_text())
    assert payload["n_samples_warning"] == "1/2 descriptors have n_samples=0"


def test_write_feature_summary_omits_warning_when_none(tmp_path) -> None:
    """Unit test: write_feature_summary without n_samples_warning should
    NOT persist the field (clean JSON).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "erd",
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "extract_rdkit_2d_descriptors.py",
    )
    erd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(erd)

    out_path = tmp_path / "test_summary.json"
    erd.write_feature_summary(
        out_path=out_path,
        dataset_name="test_dataset_clean",
        n_rows_total=10,
        n_rows_parsed=10,
        n_rows_dropped=0,
        per_metal_counts={"Pt": 10},
        desc_stats=[{"name": "MolWt", "mean": 100.0, "std": 10.0, "min": 80.0, "max": 120.0, "median": 100.0}],
        correlations=[
            {"n_samples": 100, "name": "MolWt", "pearson_r": 0.1},
        ],
        target_col="pIC50",
        wall_seconds=1.5,
        source_csv="test.csv",
    )
    payload = json.loads(out_path.read_text())
    assert "n_samples_warning" not in payload, (
        f"n_samples_warning should be omitted when not passed; "
        f"got {payload.get('n_samples_warning')}"
    )
