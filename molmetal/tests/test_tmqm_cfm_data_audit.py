"""Guards the tmQM CFM data audit artifact.

The audit itself streams ~200 MB of gzipped shards and takes minutes, so it is
not run in CI.  These tests check the artifact that run produced: that its
self-hash is intact, that the completeness claims it makes are internally
consistent, and that the numbers a reader would quote (split fractions, val
cohort composition) add up.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "metrics" / "by_metric" / "tmqm_cfm_data_audit.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip(f"{ARTIFACT.name} not generated")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_content_hash_is_intact(audit: dict) -> None:
    payload = dict(audit)
    recorded = payload.pop("content_sha256")
    text = json.dumps(payload, indent=1, sort_keys=False) + "\n"
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == recorded


def test_completeness_claims_are_self_consistent(audit: dict) -> None:
    raw = audit["raw_files"]
    cohort = audit["cohort"]["cohort_size"]
    assert audit["status"] == "MEASURED"
    assert raw["cohort_unique_codes"] == cohort
    assert raw["missing_xyz"] == 0
    assert raw["bo_missing"] == 0
    assert raw["q_missing"] == 0
    assert raw["duplicate_codes"] == 0
    assert raw["nonfinite_coordinate_blocks"] == 0
    assert raw["bo_index_out_of_range"] == 0
    assert raw["q_length_mismatch"] == 0
    assert raw["bo_coverage"] == [cohort, cohort]
    assert raw["q_coverage"] == [cohort, cohort]
    assert raw["metal_atoms_per_complex"] == {"1": cohort}
    assert raw["metal_label_disagrees_with_atom"] == 0


def test_every_complex_lands_in_exactly_one_split(audit: dict) -> None:
    split = audit["scaffold_split"]
    cohort = audit["cohort"]["cohort_size"]
    assert sum(split["rows"].values()) == cohort
    for name, n_rows in split["rows"].items():
        assert sum(split["per_metal"][name].values()) == n_rows
    assert split["scaffolds_spanning_splits"] == 0
    # Fractions are rounded to 4 dp in the artifact, so three of them can
    # sum to 1.0001.
    assert abs(sum(split["row_fraction"].values()) - 1.0) < 5e-4


def test_split_is_not_the_declared_80_10_10(audit: dict) -> None:
    """The scaffold-count budget under-allocates val/test; the artifact says so.

    If a future change makes the split hit 80/10/10, this test fails and the
    artifact must be regenerated -- the docstring and the verdict both quote
    87.7 / 6.2 / 6.2.
    """
    frac = audit["scaffold_split"]["row_fraction"]
    assert frac["train"] > 0.85
    assert 0.05 < frac["val"] < 0.08
    assert audit["scaffold_split"]["val_frac_is_ignored_on_this_branch"] is True


def test_val_cohort_metal_counts_sum_to_the_cohort(audit: dict) -> None:
    runs = audit["val_cohort_at_n192"]["runs"]
    for name, run in runs.items():
        assert sum(run["per_metal"].values()) == run["n_clouds"], name
    assert runs["min_per_metal_0"]["full_pool_scan"] is False
    assert runs["min_per_metal_20"]["full_pool_scan"] is True


def test_val_cohort_is_pt_underweighted_at_the_default(audit: dict) -> None:
    """The claim in the verdict, pinned so it cannot drift silently.

    Pt is 24.8 % of the corpus but 19/192 = 9.9 % of the default val cohort;
    the min_per_metal=20 arm brings it back to 46/192 = 24.0 %.
    """
    block = audit["val_cohort_at_n192"]
    corpus_pt = audit["cohort"]["corpus_metal_share"]["Pt"]
    assert 0.24 < corpus_pt < 0.26
    default_pt = block["runs"]["min_per_metal_0"]["per_metal"]["Pt"] / 192
    fixed_pt = block["runs"]["min_per_metal_20"]["per_metal"]["Pt"] / 192
    assert default_pt < corpus_pt / 2
    assert abs(fixed_pt - corpus_pt) < 0.03


def test_corpus_metal_share_is_a_distribution(audit: dict) -> None:
    share = audit["cohort"]["corpus_metal_share"]
    assert set(share) == {"Pt", "Ru", "Ir", "Rh", "Os", "Re"}
    assert abs(sum(share.values()) - 1.0) < 1e-3
