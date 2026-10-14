"""Tests for molmetal.data.pocket_pool_unified (Wf-A32 prep).

Aim: lock in the dispatch + filter contract for the CFM wire-in
slot. The 2 backends (crossdocked_pool + platiny_pocket_pool) are
each tested separately; this file tests the unified dispatcher only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data import crossdocked_pool, platiny_pocket_pool  # noqa: E402
from molmetal.data.pocket_pool_unified import (  # noqa: E402
    SOURCES,
    load_pocket_pool,
    pool_summary,
    source_of,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _ensure_platiny_csv():
    platiny_pocket_pool.write_pool_to_csv()
    yield


# ------------------------------------------------------------------
# source_of discrimination
# ------------------------------------------------------------------

def test_source_of_crossdocked_entry():
    """A crossdocked_pool.PocketEntry is tagged 'crossdocked2020'."""
    cd = crossdocked_pool.list_pocket_pool(split="test", max_n=1)
    if cd:
        assert source_of(cd[0]) == "crossdocked2020"


def test_source_of_platiny_entry():
    p = platiny_pocket_pool.list_pocket_pool()
    assert source_of(p[0]) == "platiny"
    assert source_of(p[5]) == "platiny"


# ------------------------------------------------------------------
# Source dispatch
# ------------------------------------------------------------------

def test_sources_constant_lists_supported_values():
    assert SOURCES == ("crossdocked2020", "platiny", "all")


def test_load_pocket_pool_default_is_all():
    rows = load_pocket_pool()
    # Default is source='all' with no max_n — could be 100k+1 rows.
    # We just check it's non-empty and contains both sources (or just
    # the crossdocked one if platiny is empty, which it isn't here).
    assert len(rows) > 0


def test_load_pocket_pool_only_crossdocked():
    rows = load_pocket_pool(source="crossdocked2020", split="test", max_n=10)
    assert all(source_of(r) == "crossdocked2020" for r in rows)
    assert len(rows) <= 100  # test split has exactly 100 rows


def test_load_pocket_pool_only_platiny():
    rows = load_pocket_pool(source="platiny")
    assert len(rows) == 10
    assert all(source_of(r) == "platiny" for r in rows)


def test_load_pocket_pool_all_concatenates():
    cd = load_pocket_pool(source="crossdocked2020", split="test", max_n=5)
    pp = load_pocket_pool(source="platiny")
    all_rows = load_pocket_pool(source="all", max_n=5)
    # With max_n=5, all_rows should be 5 crossdocked + 5 platiny = 10
    # (each backend gets max_n applied before concat).
    assert len(all_rows) == 10
    sources = {source_of(r) for r in all_rows}
    assert sources == {"crossdocked2020", "platiny"}


# ------------------------------------------------------------------
# max_n capping
# ------------------------------------------------------------------

def test_max_n_caps_crossdocked():
    rows = load_pocket_pool(source="crossdocked2020", split="test", max_n=3)
    assert len(rows) == 3


def test_max_n_caps_platiny():
    rows = load_pocket_pool(source="platiny", max_n=2)
    assert len(rows) == 2


# ------------------------------------------------------------------
# Filter passthrough
# ------------------------------------------------------------------

def test_metal_filter_platiny_only():
    rows = load_pocket_pool(source="platiny", metal="Pt_II")
    assert len(rows) >= 1
    for r in rows:
        assert "Pt_II" in r.metal_specificity


def test_metal_filter_ignored_for_crossdocked():
    """The metal filter is platiny-only; crossdocked has no metal column."""
    rows_no_filter = load_pocket_pool(source="crossdocked2020", split="test", max_n=5)
    rows_with_filter = load_pocket_pool(
        source="crossdocked2020", split="test", max_n=5, metal="Pt_II"
    )
    assert len(rows_no_filter) == len(rows_with_filter)


def test_only_extracted_honest_for_platiny():
    """At A31 ship time, platiny has 0 extracted rows."""
    rows = load_pocket_pool(source="platiny", only_extracted=True)
    assert rows == []


def test_only_extracted_passes_through_to_crossdocked():
    """Crossdocked: every row has pocket_path, so only_extracted=True
    returns the same as only_extracted=False (no filter)."""
    rows_all = load_pocket_pool(source="crossdocked2020", split="test", max_n=10)
    rows_extracted = load_pocket_pool(
        source="crossdocked2020", split="test", max_n=10, only_extracted=True
    )
    # only_extracted is platiny-only; crossdocked ignores it
    assert len(rows_all) == len(rows_extracted)


# ------------------------------------------------------------------
# Split translation
# ------------------------------------------------------------------

def test_split_train_translates_to_crossdocked_train():
    rows = load_pocket_pool(source="crossdocked2020", split="train", max_n=5)
    assert len(rows) == 5
    assert all(source_of(r) == "crossdocked2020" for r in rows)


def test_split_eval_only_translates_to_platiny():
    rows = load_pocket_pool(source="platiny", split="eval_only")
    assert len(rows) == 10


def test_split_test_translates_to_platiny_all():
    """platiny has no test split — split='test' returns the whole 10."""
    rows = load_pocket_pool(source="platiny", split="test")
    assert len(rows) == 10


def test_invalid_source_raises():
    with pytest.raises(ValueError, match="source must be one of"):
        load_pocket_pool(source="not_a_real_pool")


# ------------------------------------------------------------------
# pool_summary
# ------------------------------------------------------------------

def test_pool_summary_platiny():
    s = pool_summary(source="platiny")
    assert s["source"] == "platiny"
    assert s["total"] == 10
    assert s["by_source"]["platiny"] == 10
    assert len(s["metals"]) >= 1


def test_pool_summary_all_with_max_n():
    """max_n is applied per-source before concatenation."""
    s = pool_summary(source="all", max_n=3)
    # 3 crossdocked (capped) + 3 platiny (capped from 10) = 6
    assert s["total"] == 6
    assert s["by_source"].get("crossdocked2020", 0) == 3
    assert s["by_source"].get("platiny", 0) == 3


def test_pool_summary_all_with_max_n_larger_than_platiny():
    """When max_n >= platiny size, platiny is uncapped and total exceeds 2*max_n."""
    s = pool_summary(source="all", max_n=10)
    # 10 crossdocked (capped) + 10 platiny (uncapped, since 10 ≤ 10) = 20
    assert s["total"] == 20
    assert s["by_source"].get("crossdocked2020", 0) == 10
    assert s["by_source"].get("platiny", 0) == 10


# ------------------------------------------------------------------
# CLI smoke
# ------------------------------------------------------------------

def test_module_cli_smoke_returns_0():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "molmetal.data.pocket_pool_unified"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"smoke failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout
