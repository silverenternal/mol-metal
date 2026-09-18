"""Tests for the tmQM pre-training corpus loader (TODO F2 / P1).

Covers:

1. ``load_tmqm`` succeeds and returns the columns the pre-training script
   needs (SMILES + coordination number + metal Wiberg bond order).
2. ``filter_by_metal`` really restricts the metal centre, and rejects
   non-transition-metal symbols.
3. The Pt slice — the metal the cisplatin story depends on — has > 100
   complexes (it actually has ~7.8k).

The raw tmQM release is ~400 MB of gzipped shards on /mnt/storage; the
tests are skipped (not failed) if it is not mounted.
"""

from __future__ import annotations

import pytest

from molmetal.data.tmqm import (
    PAPER_METALS,
    TRANSITION_METALS,
    filter_by_metal,
    load_tmqm,
    metal_from_stoichiometry,
    summarize,
    tmqm_dir,
)

pytestmark = pytest.mark.skipif(
    not (tmqm_dir() / "tmQM_y.csv").exists(),
    reason=f"tmQM release not available at {tmqm_dir()}",
)


@pytest.fixture(scope="module")
def tmqm_paper_subset():
    """Pt/Ru/Ir slice of tmQM, SMILES-filtered (cached CSV after first call)."""
    return load_tmqm(metals=list(PAPER_METALS), require_smiles=True)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def test_load_tmqm_succeeds(tmqm_paper_subset):
    df = tmqm_paper_subset
    assert len(df) > 10_000, f"expected >10k Pt/Ru/Ir complexes, got {len(df)}"

    required = {"csd_code", "metal", "coord_number", "metal_bo_total", "smiles"}
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"

    # Every retained row must carry the pre-training supervision signals.
    assert df["smiles"].notna().all()
    assert df["coord_number"].notna().all()
    assert df["metal_bo_total"].notna().all()

    # CSD codes are the corpus primary key — no duplicates after the merge.
    assert df["csd_code"].is_unique

    # Physically sane coordination numbers for mononuclear d-block complexes.
    assert df["coord_number"].min() >= 1
    assert df["coord_number"].max() <= 16
    assert (df["metal_bo_total"] > 0).all()

    # Stoichiometry parsing is what identifies the metal.
    assert metal_from_stoichiometry("C40H36LaN2P3Se6") == "La"
    assert metal_from_stoichiometry("C6H6") is None


# ---------------------------------------------------------------------------
# 2. Metal filtering
# ---------------------------------------------------------------------------
def test_filter_by_metal(tmqm_paper_subset):
    df = tmqm_paper_subset
    assert set(df["metal"].unique()) <= set(PAPER_METALS)

    ru = filter_by_metal(df, ["Ru"])
    assert len(ru) > 0
    assert set(ru["metal"].unique()) == {"Ru"}
    assert len(ru) < len(df)

    # A single string is accepted as well as a sequence.
    assert len(filter_by_metal(df, "Ru")) == len(ru)

    # Filtering is partitioning: Pt + Ru + Ir must recover the whole slice.
    n_split = sum(len(filter_by_metal(df, [m])) for m in PAPER_METALS)
    assert n_split == len(df)

    # Non-transition metals are a programming error, not a silent empty frame.
    with pytest.raises(ValueError):
        filter_by_metal(df, ["Na"])
    assert "Pt" in TRANSITION_METALS


# ---------------------------------------------------------------------------
# 3. Pt coverage
# ---------------------------------------------------------------------------
def test_pt_count_above_100(tmqm_paper_subset):
    pt = filter_by_metal(tmqm_paper_subset, ["Pt"])
    assert len(pt) > 100, f"expected >100 Pt complexes, got {len(pt)}"

    stats = summarize(pt)
    assert stats.n_total == len(pt)
    assert set(stats.per_metal) == {"Pt"}

    # tmQM paper: Ni/Pd/Pt strongly prefer square-planar 4-coordination.
    hist = stats.coord_distribution["Pt"]
    assert hist, "empty coordination histogram"
    modal_cn = max(hist, key=hist.get)
    assert modal_cn == 4, f"Pt modal coordination number should be 4, got {modal_cn}"
    assert hist[4] / len(pt) > 0.5
