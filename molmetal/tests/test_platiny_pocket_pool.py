"""Tests for the platiny_pocket_pool module (Wf-A31 prep).

Aims
----
1. The CSV round-trips through the loader without loss.
2. The seed (10-entry literature pool) reproduces.
3. The metal_specificity filter narrows the pool correctly.
4. The only_extracted filter is honest (zero entries at A31 ship).
5. The module handles a missing CSV gracefully (returns empty list,
   does not raise) — protects the cross-dataset code path from a
   missing-pool crash when the user has not yet run the seed write.

These tests do NOT require the heavy PocketGen / torch_geometric /
torch_scatter / esm deps; the pool is a pure-Python module that only
depends on the standard library + the existing molmetal test stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data import platiny_pocket_pool as p3  # noqa: E402

CSV_PATH = p3.PLATINY_POOL_CSV


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _ensure_pool_csv_on_disk():
    """The CSV is regenerated for the test session (idempotent)."""
    p3.write_pool_to_csv()
    assert CSV_PATH.exists(), f"platiny_pocket_pool.csv missing at {CSV_PATH}"
    yield


# ------------------------------------------------------------------
# 1. Round-trip
# ------------------------------------------------------------------

def test_csv_roundtrip_has_10_rows():
    rows = p3._read_csv(CSV_PATH)
    assert len(rows) == 10, f"expected 10 rows, got {len(rows)}"


def test_csv_roundtrip_metadata_complete():
    """Every row carries the 4 platiny-specific metadata fields."""
    rows = p3._read_csv(CSV_PATH)
    for r in rows:
        for k in ("pdb_id", "target_name", "metal_specificity", "literature_doi"):
            assert r.get(k, "").strip(), (
                f"row {r.get('target_id')!r} missing {k!r}"
            )


def test_csv_roundtrip_split_role_eval_only():
    """All rows are eval-only — this pool is never used for training."""
    rows = p3._read_csv(CSV_PATH)
    for r in rows:
        assert r.get("split_role") == "eval_only", (
            f"row {r.get('target_id')!r} split_role={r.get('split_role')!r}"
        )


def test_csv_roundtrip_pocket_path_empty_at_seed():
    """At A31 seed time, no row has an extracted pocket.

    The pool is the LITERATURE DECLARATION; the actual pocket files
    are extracted in Wf-A33. Until then, pocket_path must be empty
    so the ``only_extracted`` filter returns [] (honest).
    """
    rows = p3._read_csv(CSV_PATH)
    for r in rows:
        assert r.get("pocket_path", "").strip() == "", (
            f"row {r.get('target_id')!r} unexpectedly has a pocket_path"
        )


# ------------------------------------------------------------------
# 2. list_pocket_pool filter API
# ------------------------------------------------------------------

def test_list_pocket_pool_default_returns_all_10():
    rows = p3.list_pocket_pool()
    assert len(rows) == 10
    assert all(isinstance(r, p3.PocketEntry) for r in rows)


def test_list_pocket_pool_metal_pt_ii_subset():
    """Pt_II filter: must return ≥1 row (we declared 4 Pt_II targets)."""
    rows = p3.list_pocket_pool(metal_specificity="Pt_II")
    assert len(rows) >= 1, "Pt_II filter returned 0 rows; the literature pool should have 4"
    for r in rows:
        # metal_specificity may be a multi-value string like "Pt_II,Au_III,Zn_II"
        assert "Pt_II" in r.metal_specificity, (
            f"row {r.target_id!r} metal_specificity={r.metal_specificity!r}"
        )


def test_list_pocket_pool_metal_au_i_subset():
    rows = p3.list_pocket_pool(metal_specificity="Au_I")
    assert len(rows) >= 1
    for r in rows:
        assert "Au_I" in r.metal_specificity


def test_list_pocket_pool_metal_unknown_returns_empty():
    rows = p3.list_pocket_pool(metal_specificity="NotARealMetal_X")
    assert rows == []


def test_list_pocket_pool_only_extracted_returns_empty_at_seed():
    """Honest framing: at A31 ship, no pocket file has been extracted yet."""
    rows = p3.list_pocket_pool(only_extracted=True)
    assert rows == [], (
        f"only_extracted returned {len(rows)} rows; "
        "this pool is not yet extracted — verify Wf-A33 status"
    )


def test_list_pocket_pool_split_eval_only_matches_default():
    """Default split = 'eval_only' returns same as explicit 'eval_only'."""
    default = p3.list_pocket_pool()
    explicit = p3.list_pocket_pool(split="eval_only")
    assert len(default) == len(explicit) == 10


def test_list_pocket_pool_split_all_includes_eval_only():
    rows = p3.list_pocket_pool(split="all")
    assert len(rows) >= 10  # may include future non-eval roles


# ------------------------------------------------------------------
# 3. Missing-CSV resilience
# ------------------------------------------------------------------

def test_handles_missing_csv_gracefully(tmp_path, monkeypatch):
    """If the CSV is missing, list_pocket_pool returns [] and does not raise.

    This is critical for the cross-dataset / inference code path: the
    platiny pool may not be on disk (e.g. fresh clone) and downstream
    code must not crash.
    """
    # Point the module at a non-existent CSV by monkey-patching the constant
    fake_csv = tmp_path / "no_such_file.csv"
    monkeypatch.setattr(p3, "PLATINY_POOL_CSV", fake_csv)
    # No write_pool_to_csv call — file should stay missing
    assert not fake_csv.exists()
    rows = p3.list_pocket_pool()
    assert rows == []
    # even with a filter
    rows_pt = p3.list_pocket_pool(metal_specificity="Pt_II")
    assert rows_pt == []


# ------------------------------------------------------------------
# 4. Atomic write safety
# ------------------------------------------------------------------

def test_write_pool_to_csv_is_atomic(tmp_path, monkeypatch):
    """A failed write must not leave a half-written CSV behind."""
    fake_csv = tmp_path / "atomic_test.csv"
    monkeypatch.setattr(p3, "PLATINY_POOL_CSV", fake_csv)

    # Successful write
    p3.write_pool_to_csv()
    assert fake_csv.exists()
    size_after = fake_csv.stat().st_size
    assert size_after > 0

    # No stray .tmp file
    tmp_files = list(tmp_path.glob(".platiny_pocket_pool.*.csv.tmp"))
    assert tmp_files == [], f"stray tmp files: {tmp_files}"


# ------------------------------------------------------------------
# 5. Integration with crossdocked_pool schema
# ------------------------------------------------------------------

def test_schema_compatible_with_crossdocked_pool():
    """The 8 crossdocked-pool columns must be present and typed correctly.

    This lets the same downstream reader consume both pools, fixing
    audit finding #6 (MED, no-shared-helper).
    """
    rows = p3._read_csv(CSV_PATH)
    required = {
        "target_id",
        "pocket_path",
        "pocket_sha256",
        "ligand_path",
        "ligand_sha256",
        "split_role",
        "native_ligand_allowed_for_generation",
        "pocket_definition_prove",
    }
    for r in rows:
        missing = required - set(r.keys())
        assert not missing, (
            f"row {r.get('target_id')!r} missing crossdocked-pool columns: {missing}"
        )


# ------------------------------------------------------------------
# 6. Smoke the CLI entry-point
# ------------------------------------------------------------------

def test_module_cli_smoke_returns_0():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "molmetal.data.platiny_pocket_pool"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"smoke failed: rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout
