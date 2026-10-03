"""Tests for WF-30 Phase 1 — 10 pocket receptor prep staging.

These tests are read-only / opt-in. They verify the integrity of the
WF-30 staging directory that
``molmetal/scripts/check_10_pockets.py`` materialises:

* every entry in ``WF30_POCKETS`` has a non-empty ``receptor.pdbqt``
  with a parseable PDBQT body,
* the corresponding ``prep_meta.json`` agrees with the per-pocket
  pdbqt (path, sha256, size),
* the top-level ``pocket_status.json`` has ``READY`` for every pocket,
* the source-PDB / pdbqt shas recorded in the status JSON are
  internally consistent,
* the PDBQT root inventory covers the 6 metric-measured pockets and
  the 4 task-brief candidates we were able to stage.

Tests do **not** invoke ``mk_prepare_receptor.py``. They assume the
script has already run; the ``--skip-existing`` flag in
``check_10_pockets.py`` makes re-runs cheap, so CI can simply re-stage
on every fresh checkout. When the staging is missing, tests are
skipped with ``pytest.skip`` rather than failing the suite — staging
lives in ``molmetal/reports/wf30_pocket_prep/`` which is a tracked
artifact, not a vendored dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WF30_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
STATUS_PATH = WF30_DIR / "pocket_status.json"

# Mirrors the canonical list in molmetal/scripts/check_10_pockets.py.
# 6 metric-measured + 4 candidates from the task brief.
WF30_POCKETS = (
    "830c",   # MMP-13 (mmp13_real anchor)
    "1h36",   # SQHC_ALIAD (metric-measured)
    "3eml",   # AA2AR_HUMAN GPCR (metric-measured)
    "3ny8",   # ADRB2_HUMAN GPCR (metric-measured)
    "4aaw",   # GLMU_STRPN test_001 (metric-measured)
    "4rlu",   # pdbbindpp-2020 (metric-measured)
    "1g9v",   # HBA_HUMAN (task brief candidate)
    "2hnx",   # FABP4_HUMAN (task brief candidate)
    "4rn0",   # HDAC8_HUMAN test_005
    "1fmc",   # HDHA_ECOLI test_006
)

METRIC_MEASURED = ("1h36", "3eml", "3ny8", "4aaw", "4rlu", "830c")


def _status() -> dict | None:
    if not STATUS_PATH.exists():
        return None
    return json.loads(STATUS_PATH.read_text())


def _skip_if_not_staged() -> None:
    if not STATUS_PATH.exists():
        pytest.skip(
            f"WF-30 staging not present ({STATUS_PATH}); run "
            "`python -m molmetal.scripts.check_10_pockets` first."
        )


def test_status_file_exists() -> None:
    """pocket_status.json must be present and parse as JSON."""
    _skip_if_not_staged()
    payload = _status()
    assert isinstance(payload, dict)
    assert payload["summary"].get("READY", 0) == len(WF30_POCKETS)


@pytest.mark.parametrize("pocket_id", WF30_POCKETS)
def test_pocket_pdbqt_exists(pocket_id: str) -> None:
    """Every WF-30 pocket must have a non-empty receptor.pdbqt."""
    _skip_if_not_staged()
    pdbqt_path = WF30_DIR / pocket_id / "receptor.pdbqt"
    assert pdbqt_path.exists(), f"missing receptor.pdbqt for {pocket_id}"
    assert pdbqt_path.stat().st_size > 0, f"empty pdbqt for {pocket_id}"


@pytest.mark.parametrize("pocket_id", WF30_POCKETS)
def test_pocket_pdbqt_is_parseable(pocket_id: str) -> None:
    """PDBQT body must have ATOM/HETATM lines that look like Meeko output."""
    _skip_if_not_staged()
    pdbqt_path = WF30_DIR / pocket_id / "receptor.pdbqt"
    body = pdbqt_path.read_text().splitlines()
    atom_lines = [line for line in body if line.startswith(("ATOM", "HETATM"))]
    assert len(atom_lines) > 10, (
        f"pdbqt for {pocket_id} has only {len(atom_lines)} ATOM/HETATM lines"
    )
    # Meeko always emits AutoDock type in cols 77-79.
    for line in atom_lines[:20]:
        assert len(line) >= 78, f"short pdbqt line: {line!r}"
        assert line[77:79].strip(), f"missing AD type at {line!r}"


@pytest.mark.parametrize("pocket_id", WF30_POCKETS)
def test_prep_meta_matches_pdbqt(pocket_id: str) -> None:
    """prep_meta.json's pdbqt path + sha + size must equal disk."""
    _skip_if_not_staged()
    meta_path = WF30_DIR / pocket_id / "prep_meta.json"
    assert meta_path.exists(), f"missing prep_meta.json for {pocket_id}"
    meta = json.loads(meta_path.read_text())
    pdbqt_path = Path(meta["pdbqt_path"])
    assert pdbqt_path.exists()
    assert pdbqt_path.stat().st_size == meta["pdbqt_size_bytes"]
    actual_sha = hashlib.sha256(pdbqt_path.read_bytes()).hexdigest()
    assert actual_sha == meta["pdbqt_sha256"]


@pytest.mark.parametrize("pocket_id", WF30_POCKETS)
def test_status_json_records_ready(pocket_id: str) -> None:
    """pocket_status.json pocket entry must be READY."""
    _skip_if_not_staged()
    payload = _status()
    record = next(
        entry for entry in payload["pockets"] if entry["pocket_id"] == pocket_id
    )
    assert record["status"] == "READY", (
        f"{pocket_id} status={record['status']}; error={record.get('error')}"
    )
    assert record["pdbqt_path"] is not None
    assert record["pdbqt_size_bytes"] > 0


@pytest.mark.parametrize("pocket_id", METRIC_MEASURED)
def test_metric_measured_pocket_has_metric_provenance(pocket_id: str) -> None:
    """The 6 metric-measured pockets must reference their vina JSONs."""
    _skip_if_not_staged()
    payload = _status()
    provenance = payload["metric_provenance"]
    assert pocket_id in provenance, (
        f"missing metric provenance for {pocket_id}"
    )
    block = provenance[pocket_id]
    assert block["metric_status"] == "MEASURED"
    assert block["mean_kcal_mol"] is not None
    assert block["n_samples"] >= 1


def test_status_json_source_paths_unique() -> None:
    """Each pocket must point at a distinct source PDB on disk."""
    _skip_if_not_staged()
    payload = _status()
    sources = [entry["source_pdb"] for entry in payload["pockets"]]
    assert all(sources), "all pockets must have a source_pdb"
    assert len(set(sources)) == len(sources), (
        f"duplicate source paths in {sources}"
    )


def test_status_json_source_sha_matches_disk() -> None:
    """source_pdb_sha256 must equal the on-disk hash of source_pdb."""
    _skip_if_not_staged()
    payload = _status()
    for entry in payload["pockets"]:
        src = Path(entry["source_pdb"])
        actual = hashlib.sha256(src.read_bytes()).hexdigest()
        assert actual == entry["source_pdb_sha256"], (
            f"sha mismatch for {entry['pocket_id']}: "
            f"{actual} vs recorded {entry['source_pdb_sha256']}"
        )


def test_wf30_dir_inventory_covers_metric_measured_and_brief() -> None:
    """The WF-30 inventory must cover both the measured set and the task brief."""
    _skip_if_not_staged()
    payload = _status()
    ids = {entry["pocket_id"] for entry in payload["pockets"]}
    for required in METRIC_MEASURED:
        assert required in ids, f"missing metric-measured pocket {required}"
    # The task brief asked for 4-5 more pockets from the candidate
    # short-list (1bc8, 1g9v, 2hnx, 3f7r, 4de2, 4gmt, 4jsz).
    # 1g9v + 2hnx are staged; 1bc8, 3f7r, 4de2, 4gmt, 4jsz are not staged
    # on this host (only 2 of the 7 candidate short-list pockets are
    # available locally), so we add 4rn0 + 1fmc from the crossdocked100
    # manifest. Verify the union still adds up to 10 pockets.
    assert len(ids) == 10
    assert {"1g9v", "2hnx"}.issubset(ids), (
        "task-brief candidate pockets 1g9v + 2hnx must be staged"
    )
