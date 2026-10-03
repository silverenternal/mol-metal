#!/usr/bin/env python3
"""
Wf-313A — m15 Docking RMSD redesign Phase 1 closeout tests (12 tests).

These tests pin the RELINKED Phase 1 reference ligand data shipped by
Wf-313A into the per-pocket subdirectory layout
`molmetal/data/m15_redesign_ref/pocket_<n>/`.

Honest framing per party-twenty-spirit §3.5:
- This is a REFACTOR test surface, NOT a new extraction test surface.
- All data is sourced from Wf-139 P1 MEASURED outputs
  (`molmetal/data/pocket_ref_ligands/ligand_metadata.json`).
- 12 tests cover: 5 per-pocket SMILES extraction + 3 InChIKey
  consistency + 2 schema validation + 2 integration with downstream
  metric JSON.

Run with:
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_wf313a_m15_redesign_phase1.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path("/home/hugo/codes/mol-metal")
SOURCE_DIR = PROJECT_ROOT / "molmetal" / "data" / "pocket_ref_ligands"
SOURCE_METADATA = SOURCE_DIR / "ligand_metadata.json"
TARGET_DIR = PROJECT_ROOT / "molmetal" / "data" / "m15_redesign_ref"
REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf313a_m15_redesign_phase1"
EXTRACTION_SUMMARY = REPORT_DIR / "extraction_summary.json"

# Per TODO-134 §1.3 / Wf-139 P1: 5 pocket CrossDocked + 1 pocket mmp13_real
# (the TODO-134 status field has 6 pockets; Wf-313A covers all 6)
POCKETS = ["830c", "1h36", "3eml", "3ny8", "4aaw", "4rlu"]


# ---- Test scaffolding ----------------------------------------------------


def _read_smi(pocket_id: str) -> str:
    """Read ref_ligand.smi and return the SMILES column (first TSV column)."""
    smi_path = TARGET_DIR / f"pocket_{pocket_id}" / "ref_ligand.smi"
    line = smi_path.read_text().strip()
    parts = line.split("\t")
    assert len(parts) == 3, (
        f"ref_ligand.smi for {pocket_id} must have 3 TSV columns; got {len(parts)}"
    )
    return parts[0]


def _read_inchi_key(pocket_id: str) -> str:
    """Read ref_ligand_inchi.json and return the InChIKey field."""
    inchi_path = TARGET_DIR / f"pocket_{pocket_id}" / "ref_ligand_inchi.json"
    data = json.loads(inchi_path.read_text())
    return data["inchikey"]


def _read_inchi_json(pocket_id: str) -> dict:
    """Read ref_ligand_inchi.json and return the full sidecar dict."""
    inchi_path = TARGET_DIR / f"pocket_{pocket_id}" / "ref_ligand_inchi.json"
    return json.loads(inchi_path.read_text())


def _read_extraction_summary() -> dict:
    return json.loads(EXTRACTION_SUMMARY.read_text())


def _read_source_metadata() -> dict:
    """Wf-139 P1 source-of-truth metadata."""
    return json.loads(SOURCE_METADATA.read_text())


# ---- Tests: per-pocket SMILES extraction (5 tests) ------------------------


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_01_refactor_smiles_file_exists(pocket_id: str) -> None:
    """Each pocket must have a ref_ligand.smi file in the per-pocket subdir."""
    smi_path = TARGET_DIR / f"pocket_{pocket_id}" / "ref_ligand.smi"
    assert smi_path.is_file(), f"missing ref_ligand.smi for {pocket_id}: {smi_path}"
    content = smi_path.read_text()
    assert len(content.strip()) > 0, f"empty ref_ligand.smi for {pocket_id}"


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_02_refactor_smiles_matches_wf139_p1(pocket_id: str) -> None:
    """Relinked SMILES must match Wf-139 P1 MEASURED source verbatim."""
    expected = None
    for p in _read_source_metadata()["pockets"]:
        if p["pocket_id"] == pocket_id:
            expected = p["smiles"]
            break
    assert expected is not None, (
        f"pocket {pocket_id} not found in Wf-139 P1 source metadata"
    )
    actual = _read_smi(pocket_id)
    assert actual == expected, (
        f"SMILES for {pocket_id} mismatch: Wf-139 P1 had {expected!r}, "
        f"Wf-313A relink has {actual!r}"
    )


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_03_refactor_smiles_is_nonempty_and_drug_like(pocket_id: str) -> None:
    """Relinked SMILES must be non-empty and have drug-like length."""
    smiles = _read_smi(pocket_id)
    assert 10 <= len(smiles) <= 200, (
        f"SMILES for {pocket_id} has implausible length "
        f"{len(smiles)} (expected 10-200 chars)"
    )


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_04_refactor_inchi_sidecar_exists(pocket_id: str) -> None:
    """Each pocket must have a ref_ligand_inchi.json sidecar."""
    inchi_path = TARGET_DIR / f"pocket_{pocket_id}" / "ref_ligand_inchi.json"
    assert inchi_path.is_file(), (
        f"missing ref_ligand_inchi.json for {pocket_id}: {inchi_path}"
    )
    data = json.loads(inchi_path.read_text())
    assert data["pocket_id"] == pocket_id
    assert data["smiles"] == _read_smi(pocket_id)
    assert data["ok"] is True


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_05_refactor_provenance_cites_wf139_p1(pocket_id: str) -> None:
    """Sidecar must carry Wf-139 P1 provenance markers (no fabricated data)."""
    data = _read_inchi_json(pocket_id)
    assert data["source_wf"] == "Wf-139 P1", (
        f"sidecar for {pocket_id} must cite Wf-139 P1 as source; "
        f"got {data.get('source_wf')!r}"
    )
    assert data["wf313a_relink_kind"] == "REFACTOR_FROM_FLAT_LAYOUT", (
        f"sidecar for {pocket_id} must mark this as a refactor; "
        f"got {data.get('wf313a_relink_kind')!r}"
    )
    assert data["phase2_redocking_status"] == "BLOCKED", (
        f"sidecar for {pocket_id} must mark Phase 2 as BLOCKED; "
        f"got {data.get('phase2_redocking_status')!r}"
    )


# ---- Tests: InChIKey consistency (3 tests) -------------------------------


def test_06_inchi_keys_unique_across_pockets() -> None:
    """Each pocket's InChIKey must be unique (no collisions across pockets)."""
    inchi_keys = [_read_inchi_key(p) for p in POCKETS]
    assert len(inchi_keys) == len(set(inchi_keys)), (
        f"InChIKey collision detected: {inchi_keys}"
    )


def test_07_inchi_keys_match_wf139_p1_source() -> None:
    """Relinked InChIKeys must match Wf-139 P1 source verbatim (1:1 mapping)."""
    source_metadata = _read_source_metadata()
    for source_pocket in source_metadata["pockets"]:
        pid = source_pocket["pocket_id"]
        expected = source_pocket["inchikey"]
        actual = _read_inchi_key(pid)
        assert actual == expected, (
            f"InChIKey mismatch for {pid}: "
            f"Wf-139 P1 = {expected!r}, Wf-313A = {actual!r}"
        )


def test_08_inchi_keys_14_char_canonical_format() -> None:
    """All InChIKeys must be 14-char canonical InChIKey (with 1 dash separator
    = 14 chars + 1 '-' + 10 chars = 27 chars total)."""
    inchi_keys = [_read_inchi_key(p) for p in POCKETS]
    for ik in inchi_keys:
        # InChIKey format: 14 chars - 10 chars - 1 char (total 27 chars)
        assert len(ik) == 27, f"InChIKey {ik!r} has wrong length {len(ik)}"
        parts = ik.split("-")
        assert len(parts) == 3, f"InChIKey {ik!r} must have 3 dash-separated parts"


# ---- Tests: schema validation (2 tests) ----------------------------------


def test_09_extraction_summary_schema_is_valid() -> None:
    """extraction_summary.json must have the Wf-313A-expected schema."""
    summary = _read_extraction_summary()
    required_keys = {
        "wf_id",
        "wf_kind",
        "source_wf",
        "refactor_only",
        "no_new_smiles_generated",
        "n_pockets_extracted",
        "n_smiles_valid",
        "n_smiles_failed",
        "n_inchi_consistent",
        "n_inchi_collisions",
        "per_pocket",
        "phase2_status",
        "phase2_reason",
        "phase2_recovery_path",
        "preflight_status",
        "generated_utc",
    }
    missing = required_keys - set(summary.keys())
    assert not missing, f"extraction_summary.json missing keys: {missing}"
    assert summary["wf_id"] == "Wf-313A"
    assert summary["source_wf"] == "Wf-139 P1"
    assert summary["refactor_only"] is True
    assert summary["no_new_smiles_generated"] is True
    assert summary["n_pockets_extracted"] == 6
    assert summary["n_smiles_valid"] == 6
    assert summary["n_smiles_failed"] == 0
    assert summary["n_inchi_consistent"] == 6
    assert summary["n_inchi_collisions"] == 0
    assert summary["phase2_status"] == "BLOCKED"
    assert len(summary["per_pocket"]) == 6


def test_10_extraction_summary_per_pocket_schema_matches_metadata() -> None:
    """Per-pocket entries in extraction_summary.json must cite valid .smi +
    .json sidecar paths + match Wf-139 P1 heavy_atom_count."""
    summary = _read_extraction_summary()
    source_metadata = _read_source_metadata()
    source_heavy_atom = {
        p["pocket_id"]: p["heavy_atom_count"] for p in source_metadata["pockets"]
    }

    for entry in summary["per_pocket"]:
        pid = entry["pocket_id"]
        smi_path = PROJECT_ROOT / entry["smiles_path"]
        inchi_path = PROJECT_ROOT / entry["inchi_path"]
        assert smi_path.is_file(), f"{smi_path} missing"
        assert inchi_path.is_file(), f"{inchi_path} missing"
        assert entry["ok"] is True
        assert entry["inchikey"] == _read_inchi_key(pid)
        assert entry["heavy_atom_count"] == source_heavy_atom[pid], (
            f"heavy_atom_count for {pid}: summary={entry['heavy_atom_count']}, "
            f"source={source_heavy_atom[pid]}"
        )


# ---- Tests: integration with downstream (2 tests) ------------------------


def test_11_extraction_summary_lists_all_six_pockets() -> None:
    """extraction_summary.json must list exactly the 6 pockets in TODO-134."""
    summary = _read_extraction_summary()
    listed_pockets = {p["pocket_id"] for p in summary["per_pocket"]}
    expected_pockets = set(POCKETS)
    assert listed_pockets == expected_pockets, (
        f"summary pocket set mismatch: expected {expected_pockets}, "
        f"got {listed_pockets}"
    )


def test_12_integration_with_wf139_p1_synthesis_md() -> None:
    """Wf-313A extraction_summary.json must reference Wf-139 P1
    synthesis.md as a documented evidence source."""
    summary = _read_extraction_summary()
    source_md_rel = summary["source_synthesis_md"]
    source_md_abs = PROJECT_ROOT / source_md_rel
    assert source_md_abs.is_file(), (
        f"Wf-139 P1 synthesis.md not found at {source_md_abs}"
    )
    # Verify the cited synthesis.md mentions all 6 pockets
    content = source_md_abs.read_text()
    missing = [p for p in POCKETS if p not in content]
    assert not missing, (
        f"Wf-139 P1 synthesis.md does not mention pockets {missing}; "
        f"either relink source_wf is wrong or synthesis.md was changed"
    )
    # Verify TODO-134 status field is also referenced (cross-check)
    todo134_abs = PROJECT_ROOT / summary["todo134_status_reference"]
    assert todo134_abs.is_file(), f"TODO-134 not found at {todo134_abs}"
    todo134_content = todo134_abs.read_text()
    assert "Wf-139" in todo134_content or "Phase 1 SHIPPED" in todo134_content, (
        f"TODO-134 should document Wf-139 P1 as the Phase 1 ship source"
    )
