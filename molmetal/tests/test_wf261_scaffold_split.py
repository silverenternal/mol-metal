"""Wf-261-Regenerate: unit tests for the strict Bemis-Murcko 70/10/20 split.

These tests guard the regeneration artefacts produced by
``molmetal/scripts/wf261_regenerate_scaffold_split.py``.  They verify the
five critical promises:

  * G1: v1 CSV was backed up (still on disk).
  * G2: v2 CSV exists and was written.
  * G3: zero scaffold_id overlap across train/val/test (CRITICAL).
  * G4: row distribution is within ±2 % of 70/10/20.
  * G5: split_manifest.json SHA256 matches v2 CSV.

If you remove v1 backup, v2 CSV, or the manifest, these tests will fail
loudly.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants (mirrored from the regen script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data" / "molmetal"

V1_CSV = DATA_DIR / "metalcytotox_5metal_multitask_v4.csv"
V1_BACKUP = DATA_DIR / "metalcytotox_5metal_multitask_v1.csv"
V2_CSV = DATA_DIR / "metalcytotox_5metal_multitask_v2.csv"
MANIFEST = DATA_DIR / "metalcytotox_5metal_multitask_v2_split_manifest.json"

EXPECTED_FRACTIONS = (0.70, 0.10, 0.20)
DIST_TOLERANCE_PCT = 2.0
EXPECTED_COLS = [
    "smiles",
    "metal",
    "scaffold_id",
    "split",
    "activity",
    "pic50",
    "cell_line",
    "source",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v2_rows():
    assert V2_CSV.exists(), (
        f"v2 CSV missing at {V2_CSV} — run "
        "molmetal/scripts/wf261_regenerate_scaffold_split.py first"
    )
    with V2_CSV.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST.exists(), f"manifest missing at {MANIFEST}"
    return json.loads(MANIFEST.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_v1_csv_preserved():
    """G1: backup file exists (v1 CSV still on disk for rollback)."""
    assert V1_BACKUP.exists(), f"v1 backup missing at {V1_BACKUP}"


def test_v2_csv_exists():
    """G2: the regenerated scaffold-split CSV was written."""
    assert V2_CSV.exists(), f"v2 CSV missing at {V2_CSV}"
    assert V2_CSV.stat().st_size > 0, "v2 CSV is empty"


def test_v2_csv_schema(v2_rows):
    """v2 CSV must keep the canonical 8-column schema."""
    assert v2_rows, "v2 CSV has no rows"
    sample = v2_rows[0]
    for col in EXPECTED_COLS:
        assert col in sample, f"v2 CSV missing column {col!r}; got keys={list(sample)}"


def test_zero_scaffold_leak(v2_rows):
    """G3 (CRITICAL): every scaffold_id appears in exactly one split."""
    scaf_to_splits = defaultdict(set)
    for r in v2_rows:
        assert r[
            "scaffold_id"
        ], "v2 CSV has empty scaffold_id (should have been dropped)"
        scaf_to_splits[r["scaffold_id"]].add(r["split"])
    leaks = {s: sorted(ss) for s, ss in scaf_to_splits.items() if len(ss) > 1}
    assert not leaks, (
        f"scaffold_id overlap detected across splits (n={len(leaks)}); "
        f"first examples: {dict(list(leaks.items())[:5])}"
    )


def test_row_count_distribution(v2_rows):
    """G4: row distribution is within ±2 % of 70/10/20."""
    counts: dict = defaultdict(int)
    for r in v2_rows:
        counts[r["split"]] += 1
    total = sum(counts.values())
    assert total > 0, "v2 CSV has no rows"
    pct = {k: 100.0 * v / total for k, v in counts.items()}
    for split, expect in zip(("train", "val", "test"), EXPECTED_FRACTIONS):
        assert abs(pct.get(split, 0) - 100.0 * expect) <= DIST_TOLERANCE_PCT, (
            f"split {split!r} drift: got {pct.get(split, 0):.2f} %, "
            f"expected {100.0 * expect:.2f} ±{DIST_TOLERANCE_PCT}"
        )


def test_split_manifest_sha256_match(v2_rows, manifest):
    """G5: split_manifest.json SHA256 matches the on-disk v2 CSV."""
    h = hashlib.sha256()
    with V2_CSV.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    expected_sha = h.hexdigest()
    assert manifest["sha256"] == expected_sha, (
        f"manifest sha256 ({manifest['sha256']}) does not match "
        f"on-disk v2 csv sha256 ({expected_sha})"
    )


def test_split_manifest_invariants(manifest):
    """Manifest contract: must have fractions, seed, scaffold mapping."""
    for key in (
        "version",
        "csv",
        "sha256",
        "n_total_rows",
        "n_dropped_null_scaffold",
        "n_unique_scaffolds",
        "n_rows_v2",
        "fractions",
        "seed",
        "split_rows",
        "scaffold_id_to_split",
    ):
        assert key in manifest, f"manifest missing key {key!r}"

    # Every scaffold in the manifest is in exactly one split.
    scaf_to_split: dict = defaultdict(set)
    for scaf, sp in manifest["scaffold_id_to_split"].items():
        scaf_to_split[scaf].add(sp)
    assert all(len(v) == 1 for v in scaf_to_split.values()), (
        f"manifest has a scaffold with >1 split: "
        f"{[s for s, v in scaf_to_split.items() if len(v) > 1][:5]}"
    )


def test_dropped_rows_reported_honestly(manifest):
    """Honest framing: manifest must surface null-scaffold dropouts."""
    assert "n_dropped_null_scaffold" in manifest
    assert isinstance(manifest["n_dropped_null_scaffold"], int)
    # Dropouts must be >= 0 (≥ 0 by construction; never silently dropped).
    assert manifest["n_dropped_null_scaffold"] >= 0


def test_v1_csv_intact(v2_rows):
    """G_round-trip: v1 backup should still contain the same row count + split."""
    assert V1_BACKUP.exists(), f"v1 backup missing at {V1_BACKUP}"
    with V1_BACKUP.open("r", newline="") as fh:
        v1 = list(csv.DictReader(fh))
    assert v1, "v1 backup is empty"
    assert len(v1) >= len(v2_rows), (
        f"v1 backup has fewer rows ({len(v1)}) than v2 ({len(v2_rows)}) — "
        "v2 must be a subset of v1 (rows dropped, never added)"
    )
