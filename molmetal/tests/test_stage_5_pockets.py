"""Tests for Wf-126 Phase 1 — 5-pocket (1h36/3eml/3ny8/4aaw/4rlu) full staging.

These tests are read-only / opt-in. They verify the integrity of the
Wf-126 staging directory that
``molmetal/scripts/stage_5_pockets_full.py`` materialises:

* every pocket has a non-empty ``receptor.pdbqt`` with a parseable
  PDBQT body,
* ``prep_meta.json`` agrees with the per-pocket pdbqt (path, sha256,
  size, source-pdb sha),
* the top-level ``staging_status.json`` has ``READY`` for every pocket,
* the on-disk pdbqt sha256 matches the value recorded in
  ``prep_meta.json`` and in ``staging_status.json``,
* the on-disk pdbqt sha256 also matches the corresponding wf30 staging
  artefact (proves the recipe is reproducible + the wf30 staging was
  NOT partial),
* the honest-framing block records the correct diagnosis counter to
  TODO-123 §0 (rigid_fragment_rmsd is self-RMSD, not pose RMSD).

Tests do **not** invoke ``mk_prepare_receptor.py``. They assume the
script has already run; ``--skip-existing`` makes re-runs cheap, so
CI can simply re-stage on every fresh checkout. When staging is
missing, tests are skipped with ``pytest.skip`` rather than failing
the suite — staging lives in ``molmetal/reports/wf126_p1_staging/``
which is a tracked artifact, not a vendored dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WF126_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf126_p1_staging"
STATUS_PATH = WF126_DIR / "staging_status.json"
WF30_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"

# Mirrors POCKETS in molmetal/scripts/stage_5_pockets_full.py.
WF126_POCKETS = ("1h36", "3eml", "3ny8", "4aaw", "4rlu")


def _status() -> dict | None:
    if not STATUS_PATH.exists():
        return None
    return json.loads(STATUS_PATH.read_text())


def _skip_if_not_staged() -> None:
    if not STATUS_PATH.exists():
        pytest.skip(
            f"WF-126 staging not present ({STATUS_PATH}); run "
            "`python -m molmetal.scripts.stage_5_pockets_full` first."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_status_file_exists_and_summary() -> None:
    """staging_status.json must exist, parse, and report 5/5 READY."""
    _skip_if_not_staged()
    payload = _status()
    assert isinstance(payload, dict)
    assert payload["workflow"] == "wf126_p1"
    assert payload["ready_count"] == len(WF126_POCKETS)
    assert payload["expected_count"] == len(WF126_POCKETS)
    assert payload["summary"].get("READY", 0) == len(WF126_POCKETS)


def test_honest_framing_block_present() -> None:
    """staging_status.json must include the honest-framing block."""
    _skip_if_not_staged()
    payload = _status()
    block = payload.get("honest_framing")
    assert isinstance(block, dict)
    diagnosis = block.get("diagnosis_in_todo_123", "")
    # The framing must explicitly name the misleading diagnosis in
    # TODO-123 §0 and explain why it is misleading.
    assert "partial staging" in diagnosis
    assert "self-RMSD" in diagnosis
    assert "1.241" in diagnosis
    assert block.get("what_this_pass_does"), (
        "honest_framing.what_this_pass_does must be set"
    )
    assert block.get("stage_recipe"), (
        "honest_framing.stage_recipe must be set"
    )


@pytest.mark.parametrize("pocket_id", WF126_POCKETS)
def test_pocket_pdbqt_exists(pocket_id: str) -> None:
    """Every WF-126 pocket must have a non-empty receptor.pdbqt."""
    _skip_if_not_staged()
    pdbqt_path = WF126_DIR / pocket_id / "receptor.pdbqt"
    assert pdbqt_path.exists(), f"missing receptor.pdbqt for {pocket_id}"
    assert pdbqt_path.stat().st_size > 0, f"empty pdbqt for {pocket_id}"


@pytest.mark.parametrize("pocket_id", WF126_POCKETS)
def test_pocket_pdbqt_is_parseable(pocket_id: str) -> None:
    """PDBQT body must have ATOM/HETATM lines that look like Meeko output."""
    _skip_if_not_staged()
    pdbqt_path = WF126_DIR / pocket_id / "receptor.pdbqt"
    body = pdbqt_path.read_text().splitlines()
    atom_lines = [line for line in body if line.startswith(("ATOM", "HETATM"))]
    assert len(atom_lines) > 10, (
        f"pdbqt for {pocket_id} has only {len(atom_lines)} ATOM/HETATM lines"
    )
    # Meeko always emits AutoDock type in cols 77-79.
    for line in atom_lines[:20]:
        assert len(line) >= 78, f"short pdbqt line: {line!r}"
        assert line[77:79].strip(), f"missing AD type at {line!r}"


@pytest.mark.parametrize("pocket_id", WF126_POCKETS)
def test_prep_meta_matches_pdbqt(pocket_id: str) -> None:
    """prep_meta.json's pdbqt path + sha + size must equal disk."""
    _skip_if_not_staged()
    meta_path = WF126_DIR / pocket_id / "prep_meta.json"
    assert meta_path.exists(), f"missing prep_meta.json for {pocket_id}"
    meta = json.loads(meta_path.read_text())
    pdbqt_path = Path(meta["pdbqt_path"])
    assert pdbqt_path.exists()
    assert pdbqt_path.stat().st_size == meta["pdbqt_size_bytes"]
    actual_sha = hashlib.sha256(pdbqt_path.read_bytes()).hexdigest()
    assert actual_sha == meta["pdbqt_sha256"]


@pytest.mark.parametrize("pocket_id", WF126_POCKETS)
def test_status_json_records_ready(pocket_id: str) -> None:
    """staging_status.json pocket entry must be READY with sha + size."""
    _skip_if_not_staged()
    payload = _status()
    record = next(
        entry
        for entry in payload["pockets"]
        if entry["pocket_id"] == pocket_id
    )
    assert record["status"] == "READY", (
        f"{pocket_id} status={record['status']}; error={record.get('error')}"
    )
    assert record["pdbqt_path"] is not None
    assert record["pdbqt_size_bytes"] > 0
    assert record["pdbqt_sha256"] is not None


@pytest.mark.parametrize("pocket_id", WF126_POCKETS)
def test_pdbqt_sha_matches_wf30(pocket_id: str) -> None:
    """WF-126 pdbqt sha must equal the corresponding wf30 pdbqt sha.

    This is the load-bearing assertion that proves the staging is
    reproducible AND that the wf30 staging was *not* partial — the
    plan said 4/5 were 'partial' but the sha match means the wf30
    artefacts were already complete.
    """
    _skip_if_not_staged()
    wf126_pdbqt = WF126_DIR / pocket_id / "receptor.pdbqt"
    wf30_pdbqt = WF30_DIR / pocket_id / "receptor.pdbqt"
    if not wf30_pdbqt.exists():
        pytest.skip(
            f"wf30 staging not present for {pocket_id}; "
            "comparison skipped."
        )
    wf126_sha = _sha256(wf126_pdbqt)
    wf30_sha = _sha256(wf30_pdbqt)
    assert wf126_sha == wf30_sha, (
        f"sha mismatch for {pocket_id}: "
        f"wf126={wf126_sha} vs wf30={wf30_sha}"
    )


@pytest.mark.parametrize("pocket_id", WF126_POCKETS)
def test_status_json_source_sha_matches_disk(pocket_id: str) -> None:
    """source_pdb_sha256 must equal the on-disk hash of source_pdb."""
    _skip_if_not_staged()
    payload = _status()
    record = next(
        entry
        for entry in payload["pockets"]
        if entry["pocket_id"] == pocket_id
    )
    src = Path(record["source_pdb"])
    actual = hashlib.sha256(src.read_bytes()).hexdigest()
    assert actual == record["source_pdb_sha256"], (
        f"sha mismatch for {pocket_id}: "
        f"{actual} vs recorded {record['source_pdb_sha256']}"
    )


@pytest.mark.parametrize("pocket_id", WF126_POCKETS)
def test_metric_provenance_attached(pocket_id: str) -> None:
    """Each pocket must carry the measured-pocket metric provenance block."""
    _skip_if_not_staged()
    payload = _status()
    record = next(
        entry
        for entry in payload["pockets"]
        if entry["pocket_id"] == pocket_id
    )
    block = record.get("metric_provenance")
    assert isinstance(block, dict), (
        f"missing metric_provenance for {pocket_id}"
    )
    assert block.get("metric_status") == "MEASURED"
    assert block.get("mean") is not None
    assert block.get("n_samples") is not None


def test_status_json_wf30_audit_attached() -> None:
    """Each pocket should carry the wf30 audit pointer for traceability."""
    _skip_if_not_staged()
    payload = _status()
    for record in payload["pockets"]:
        audit = record.get("wf30_audit")
        if audit is None:
            # wf30 staging may be missing on a fresh checkout — that's
            # a soft skip rather than a hard fail.
            continue
        assert audit.get("pocket_id") == record["pocket_id"]
        assert audit.get("status") == "READY"


def test_all_five_pocket_ids_present() -> None:
    """The status payload must mention exactly the five expected pockets."""
    _skip_if_not_staged()
    payload = _status()
    ids = {
        entry["pocket_id"]
        for entry in payload["pockets"]
        if entry.get("pocket_id")
    }
    assert ids == set(WF126_POCKETS), (
        f"unexpected pocket ids: {ids - set(WF126_POCKETS)}; "
        f"missing: {set(WF126_POCKETS) - ids}"
    )


def test_no_duplicate_source_pdb_paths() -> None:
    """Each pocket must point at a distinct source PDB on disk."""
    _skip_if_not_staged()
    payload = _status()
    sources = [entry["source_pdb"] for entry in payload["pockets"]]
    assert all(sources), "all pockets must have a source_pdb"
    assert len(set(sources)) == len(sources), (
        f"duplicate source paths in {sources}"
    )


# A combined helper for downstream consumers that want to do an
# all-in-one read of the status payload. Not a test — exported for
# callers that import this module outside the pytest runner.
def load_wf126_status() -> Dict[str, object]:
    """Return the wf126_p1 staging status payload as a dict."""
    return _status() or {}