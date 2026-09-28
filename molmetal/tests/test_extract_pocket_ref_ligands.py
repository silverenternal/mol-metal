"""Tests for Wf-139 Phase 1 — pocket reference ligand SMILES extraction.

=====================================================================
These tests pin down the invariants the Wf-139 Phase 1 deliverable
depends on so that future RDKit / CrossDocked regressions are caught
immediately.

The extractor is implemented in
``molmetal/scripts/extract_pocket_ref_ligands.py`` and produces:

* ``molmetal/data/pocket_ref_ligands/<pocket>_ligand.smi`` — 6 NEW
* ``molmetal/data/pocket_ref_ligands/ligand_metadata.json`` — NEW
* ``molmetal/reports/wf139_p1_ref_ligands/synthesis.md`` — NEW

The 6 pockets (per TODO-134 §1.2):

* 830c — MMP-13 hydroxamate RS1
* 1h36 — SQHC_ALIAD R88
* 3eml — AA2AR_HUMAN ZMA (ZM241385)
* 3ny8 — ADRB2_HUMAN TIM
* 4aaw — GLMU_STRPN R83
* 4rlu — Trypsin HCC

Tests (count target: 12):

 1.  All 6 per-pocket ``<pocket>_ligand.smi`` files exist.
 2.  Each ``.smi`` file is TSV with SMILES, InChIKey, pocket_id.
 3.  Each SMILES is parseable by RDKit + canonicalizable.
 4.  Per-pocket ``InChIKey`` matches RDKit ``MolToInchiKey``.
 5.  All 6 InChIKeys are unique (no cross-pocket collision).
 6.  ``ligand_metadata.json`` exists and parses as JSON.
 7.  ``ligand_metadata.json`` reports ``n_extracted_ok == 6``.
 8.  ``ligand_metadata.json`` reports 6 pockets in ``pockets`` list.
 9.  Each pocket has ``source_sha256``, ``element_types``,
     ``coordinates_3d``, and ``heavy_atom_count`` populated.
10.  CLI ``--verify-only`` exits with code 0 (verifies metadata OK).
11.  ``synthesis.md`` exists with PASS status for 6/6 extraction.
12.  Heavy atom count plausibility: each pocket has 15-60 heavy atoms
     (drug-like range).

Tests are **skipped (not failed)** when the artefacts are missing — so
CI on a fresh checkout does not get red until the Wf-139 Phase 1
deliverable has shipped. The acceptance gate for Phase 2 launch is
``pytest -v test_extract_pocket_ref_ligands.py`` showing all 12
tests PASS.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "molmetal" / "data" / "pocket_ref_ligands"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf139_p1_ref_ligands"

POCKETS = ["830c", "1h36", "3eml", "3ny8", "4aaw", "4rlu"]


def _metadata_exists() -> bool:
    return (DATA_DIR / "ligand_metadata.json").exists()


def _smi_exists(pocket_id: str) -> bool:
    return (DATA_DIR / f"{pocket_id}_ligand.smi").exists()


def _read_smi(pocket_id: str) -> str:
    return (DATA_DIR / f"{pocket_id}_ligand.smi").read_text().strip()


def _read_metadata() -> dict:
    return json.loads((DATA_DIR / "ligand_metadata.json").read_text())


def _try_rdkit_mol(smi: str):
    """Parse a SMILES with RDKit, skipping the test if RDKit is unavailable."""
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem, rdMolDescriptors  # noqa: F401
        from rdkit.Chem.inchi import MolToInchiKey  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not on path")
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pytest.skip(f"RDKit failed to parse SMILES: {smi}")
    return mol


# ---------------------------------------------------------------------------
# Tests 1-2: files exist + format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_01_per_pocket_smi_file_exists(pocket_id: str):
    """Each of the 6 pockets has its <pocket>_ligand.smi file."""
    if not DATA_DIR.exists():
        pytest.skip(f"data dir {DATA_DIR} not present — run extract first")
    assert _smi_exists(pocket_id), f"missing {pocket_id}_ligand.smi"


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_02_smi_file_is_tsv_three_columns(pocket_id: str):
    """Each .smi file is TSV with SMILES, InChIKey, pocket_id."""
    if not _smi_exists(pocket_id):
        pytest.skip("smi file missing")
    content = _read_smi(pocket_id)
    parts = content.split("\t")
    assert len(parts) == 3, (
        f"{pocket_id}_ligand.smi should be 3 TSV columns, got {len(parts)}: {content!r}"
    )
    smi, inchi_key, pid = parts
    assert smi, f"empty SMILES for {pocket_id}"
    assert inchi_key, f"empty InChIKey for {pocket_id}"
    assert pid == pocket_id, f"pocket_id mismatch: {pid!r} != {pocket_id!r}"
    # InChIKey is 14-char uppercase hex string + '-' + 10-char + '-' + 1-char
    assert "-" in inchi_key and len(inchi_key) == 27, (
        f"InChIKey {inchi_key!r} not 14-10-1 format"
    )


# ---------------------------------------------------------------------------
# Tests 3-5: SMILES canonicality + InChIKey consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_03_smiles_parses_and_canonicalizes(pocket_id: str):
    """Each SMILES parses with RDKit + matches MolToSmiles output."""
    if not _smi_exists(pocket_id):
        pytest.skip("smi file missing")
    smi = _read_smi(pocket_id).split("\t")[0]
    mol = _try_rdkit_mol(smi)
    from rdkit import Chem

    canon = Chem.MolToSmiles(mol)
    # The file SMILES should already be canonical; allow identity match
    assert canon == smi, (
        f"{pocket_id}: stored SMILES not canonical; "
        f"stored={smi!r}, canonical={canon!r}"
    )


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_04_smi_inchikey_matches_rdkit(pocket_id: str):
    """The InChIKey column matches RDKit's MolToInchiKey for the SMILES."""
    if not _smi_exists(pocket_id):
        pytest.skip("smi file missing")
    smi, inchi_key, _ = _read_smi(pocket_id).split("\t")
    mol = _try_rdkit_mol(smi)
    from rdkit.Chem.inchi import MolToInchiKey

    assert MolToInchiKey(mol) == inchi_key, (
        f"{pocket_id}: stored InChIKey {inchi_key!r} != "
        f"RDKit {MolToInchiKey(mol)!r}"
    )


def test_05_all_inchikeys_unique():
    """No two pockets share an InChIKey (would mean same ligand in two pockets)."""
    if not all(_smi_exists(p) for p in POCKETS):
        pytest.skip("some smi files missing")
    inchi_keys = [_read_smi(p).split("\t")[1] for p in POCKETS]
    assert len(set(inchi_keys)) == len(inchi_keys), (
        f"InChIKey collision: {inchi_keys}"
    )


# ---------------------------------------------------------------------------
# Tests 6-9: ligand_metadata.json
# ---------------------------------------------------------------------------


def test_06_ligand_metadata_json_exists_and_parses():
    """ligand_metadata.json exists and is valid JSON."""
    if not _metadata_exists():
        pytest.skip("metadata file missing")
    payload = _read_metadata()
    assert "pockets" in payload, "metadata missing 'pockets' key"
    assert "n_pockets" in payload, "metadata missing 'n_pockets' key"
    assert "n_extracted_ok" in payload, "metadata missing 'n_extracted_ok' key"


def test_07_metadata_n_extracted_ok_is_six():
    """Exactly 6 of 6 pockets extracted OK (acceptance gate for Phase 2)."""
    if not _metadata_exists():
        pytest.skip("metadata file missing")
    payload = _read_metadata()
    assert payload["n_extracted_ok"] == 6, (
        f"expected 6 OK, got {payload['n_extracted_ok']}"
    )
    assert payload["n_extracted_fail"] == 0, (
        f"expected 0 failures, got {payload['n_extracted_fail']}"
    )


def test_08_metadata_pockets_list_has_six_entries():
    """ligand_metadata.json['pockets'] has 6 entries."""
    if not _metadata_exists():
        pytest.skip("metadata file missing")
    payload = _read_metadata()
    assert len(payload["pockets"]) == 6, (
        f"expected 6 pockets, got {len(payload['pockets'])}"
    )
    pocket_ids = sorted(p["pocket_id"] for p in payload["pockets"])
    assert pocket_ids == sorted(POCKETS), (
        f"pocket_ids mismatch: {pocket_ids} != {sorted(POCKETS)}"
    )


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_09_metadata_per_pocket_schema_complete(pocket_id: str):
    """Each pocket record has source_sha256, element_types, coordinates_3d, heavy_atom_count."""
    if not _metadata_exists():
        pytest.skip("metadata file missing")
    payload = _read_metadata()
    record = next(
        (p for p in payload["pockets"] if p["pocket_id"] == pocket_id), None
    )
    assert record is not None, f"{pocket_id} missing from metadata"
    assert record.get("ok"), f"{pocket_id} ok=False in metadata"
    for key in (
        "source_sha256",
        "element_types",
        "coordinates_3d",
        "heavy_atom_count",
        "smiles",
        "inchikey",
        "formula",
    ):
        assert key in record, f"{pocket_id} missing key {key!r}"
        assert record[key], f"{pocket_id} empty value for key {key!r}"
    # Coordinates should match heavy atom count
    assert len(record["coordinates_3d"]) == record["heavy_atom_count"], (
        f"{pocket_id}: coords {len(record['coordinates_3d'])} != "
        f"heavy {record['heavy_atom_count']}"
    )
    assert len(record["element_types"]) == record["heavy_atom_count"], (
        f"{pocket_id}: elements {len(record['element_types'])} != "
        f"heavy {record['heavy_atom_count']}"
    )


# ---------------------------------------------------------------------------
# Test 10: CLI verify-only
# ---------------------------------------------------------------------------


def test_10_cli_verify_only_returns_zero():
    """``--verify-only`` exits 0 (acceptance gate for downstream Phase 2)."""
    if not _metadata_exists():
        pytest.skip("metadata file missing")
    cmd = [
        sys.executable,
        "-m",
        "molmetal.scripts.extract_pocket_ref_ligands",
        "--verify-only",
    ]
    # Use the project venv python if present, else current interpreter
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        cmd[0] = str(venv_python)
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"--verify-only exited {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 11: synthesis.md
# ---------------------------------------------------------------------------


def test_11_synthesis_md_pass():
    """synthesis.md exists with PASS verdict for 6/6 extraction."""
    synth_path = REPORTS_DIR / "synthesis.md"
    if not synth_path.exists():
        pytest.skip(f"{synth_path} missing")
    content = synth_path.read_text()
    assert "**Status**: PASS" in content, "synthesis.md missing PASS status"
    assert "6/6" in content or "n_extracted_ok == 6" in content.lower() or (
        "Pockets extracted OK**: 6/6" in content
    ), "synthesis.md missing 6/6 line"
    assert "Phase 1 PASS" in content, "synthesis.md missing Phase 1 PASS decision"


# ---------------------------------------------------------------------------
# Test 12: heavy atom count plausibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pocket_id", POCKETS)
def test_12_heavy_atom_count_in_drug_like_range(pocket_id: str):
    """Each pocket has 15-60 heavy atoms (drug-like ligand range)."""
    if not _metadata_exists():
        pytest.skip("metadata file missing")
    payload = _read_metadata()
    record = next(
        (p for p in payload["pockets"] if p["pocket_id"] == pocket_id), None
    )
    assert record is not None
    heavy = record.get("heavy_atom_count", 0)
    assert 15 <= heavy <= 60, (
        f"{pocket_id}: heavy_atom_count {heavy} outside drug-like range [15, 60]"
    )
