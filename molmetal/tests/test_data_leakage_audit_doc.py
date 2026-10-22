"""The data-leakage document must describe the protocols that are actually run.

Before 2026-10-22 §5 of ``docs/DATA_LEAKAGE_AUDIT.md`` measured only the NCI-60
legacy split, so once the default moved to the coverage split the document
described a protocol no bare run used.  §4 had the mirror-image problem: it
audited ``platinai_4cell.csv`` (synthetic labels, DO-NOT-TRAIN) and called it
"the Platinum 4-cell training cohort", while the cohort the trainer is pointed
at went unaudited.

These tests pin the shape of the measurement, not just its prose:

* both NCI-60 protocols are present and the default is the coverage one;
* the real-label 4-cell cohort is measured, and it has no scaffold spanning
  two splits;
* the scaffold key used by the audit is the splitter's key (canonicalise
  first -- on the real 4-cell cohort the raw-string version disagrees for
  18 % of molecules and invents two "shared" scaffolds);
* the rendered document carries the artifact's verdict verbatim, so a stale
  document is detectable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC = PROJECT_ROOT / "docs" / "DATA_LEAKAGE_AUDIT.md"


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "wf_a51_leakage_audit_doc_under_test",
        PROJECT_ROOT / "molmetal" / "scripts" / "wf_a51_leakage_audit.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load_audit()


@pytest.fixture(scope="module")
def nci60(audit):
    return audit.audit_nci60()


@pytest.fixture(scope="module")
def pt4_real(audit):
    return audit.audit_platinai_4cell_real()


# ---------------------------------------------------------------------------
# NCI-60: both protocols
# ---------------------------------------------------------------------------
def test_nci60_measures_both_protocols(nci60):
    assert set(nci60["protocols"]) == {
        "scaffold_70_10_10",
        "scaffold_coverage_70_10_20",
    }
    assert nci60["default_protocol"] == "scaffold_coverage_70_10_20"


def test_nci60_coverage_protocol_does_not_starve_a_cell(nci60):
    cov = nci60["protocols"]["scaffold_coverage_70_10_20"]
    assert cov["split_sizes_rows"] == {"train": 5371, "val": 920, "test": 1532}
    assert cov["min_test_rows_per_cell"] >= 20
    assert cov["min_val_rows_per_cell"] >= 15
    assert cov["cells_with_zero_val_rows"] == []


def test_nci60_legacy_protocol_is_recorded_as_starving(nci60):
    legacy = nci60["protocols"]["scaffold_70_10_10"]
    assert legacy["split_sizes_unique_smiles"] == {
        "train": 4860,
        "val": 115,
        "test": 232,
    }
    assert len(legacy["cells_below_30_test_rows"]) == 5
    assert legacy["cells_with_zero_val_rows"] == ["SKOV3"]


def test_nci60_neither_protocol_leaks_a_scaffold(nci60):
    for rec in nci60["protocols"].values():
        assert rec["n_scaffolds_spanning_splits"] == 0
        for pair in rec["pairs"].values():
            assert pair["canonical"]["n_intersect_keys"] == 0
            assert pair["bemis_murcko_split_key"]["n_intersect_keys"] == 0


def test_nci60_pre_blk_pa12_claim_is_marked_unreproducible(nci60):
    claim = nci60["pre_blk_pa12_claim"]
    assert claim["split_sizes_unique_smiles"] == {
        "train": 3645,
        "val": 521,
        "test": 1041,
    }
    assert claim["reproducible"] is False


# ---------------------------------------------------------------------------
# Pt 4-cell real-label cohort
# ---------------------------------------------------------------------------
def test_pt4_real_cohort_is_measured(pt4_real):
    assert pt4_real["split_sizes_rows"] == {"train": 1376, "val": 197, "test": 393}
    assert pt4_real["n_compounds"] == 1966
    assert pt4_real["n_unique_smiles"] == 1935


def test_pt4_real_has_no_scaffold_leak(pt4_real):
    assert pt4_real["n_scaffolds_spanning_splits"] == 0
    for pair in pt4_real["pairs"].values():
        assert pair["canonical"]["n_intersect_keys"] == 0
        assert pair["bemis_murcko_split_key"]["n_intersect_keys"] == 0


# ---------------------------------------------------------------------------
# The key must be the splitter's key
# ---------------------------------------------------------------------------
def test_scaffold_key_canonicalises_before_hashing(audit):
    """A raw string and its canonical form must give the same split key.

    ``ScaffoldSplitter`` calls ``_scaffold_for(canonical_smiles)``.  Hashing
    the raw string instead disagreed for 354/1,935 molecules of the real
    4-cell cohort and produced two phantom shared scaffolds.
    """
    from molmetal.data.leakage_utils import canonicalize_smiles

    raw = [
        "CC12CCC3c4ccc(OCCCC[NH]5CC6=N(C=CC=C6)[Pt]5([Cl])[Cl])cc4CCC",
        "O=[Pt]1(Cl)N2=CC=CC=C2C3=CC=CC=C31",
        "C1=CC=C(C=C1)C2=NC=CC=C2",
    ]
    canon = [canonicalize_smiles(s) for s in raw]
    raw_keys = audit._scaffold_split_key(raw).tolist()
    canon_keys = audit._scaffold_split_key(canon).tolist()
    assert raw_keys == canon_keys


# ---------------------------------------------------------------------------
# The document is generated, and it is current
# ---------------------------------------------------------------------------
def test_document_describes_both_nci60_protocols():
    text = DOC.read_text(encoding="utf-8")
    assert "scaffold_70_10_10" in text
    assert "scaffold_coverage_70_10_20" in text
    assert "Two protocols, both measured through the production loader" in text


def test_document_covers_the_real_4cell_cohort():
    text = DOC.read_text(encoding="utf-8")
    assert "platinai_4cell_real_trainer.csv" in text
    assert "### 4.2" in text


def test_document_carries_the_artifact_verdict():
    """A re-run that changes a number must change the document too.

    The artifact lives under ``molmetal/reports/``, which is gitignored, so on
    a fresh clone there is nothing to compare against and the check is
    skipped rather than failed.
    """
    import json

    artifact = (
        PROJECT_ROOT
        / "molmetal"
        / "reports"
        / "wf_a51_leakage_audit"
        / "leakage_audit.json"
    )
    if not artifact.exists():
        pytest.skip("leakage_audit.json is a local artifact (gitignored)")
    verdict = json.loads(artifact.read_text(encoding="utf-8"))["verdict"]
    assert verdict.strip() in DOC.read_text(encoding="utf-8")
