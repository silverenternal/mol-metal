"""Wf-446 HOLE-5 — nci60 split-leak gate + multi-CSV scaffold gate tests.

CPU-only, no network, no GPU.  Covers:

1. ``_nci60_scaffold_pair_leaks`` detects a synthetic scaffold overlap.
2. ``_nci60_index_triple_violations`` detects index overlaps + coverage gaps.
3. ``gate_scaffold_zero_leak`` FAILs on an injected leak in a synthetic CSV
   and PASSes on the same CSV without the leak (multi-CSV contract).
4. ``gate_nci60_split_no_leak`` FAILs on a missing CSV.
5. (integration, ~30 s) the nci60 gate PASSes on the healthy tree and its
   evidence carries the HOLE-5 honest-negative markers (REAL scaffold
   overlap + degenerate-splitter flag) instead of silently reading
   stronger than it is.

See TODO/completed/h5_nci60_split_gate.md for the full ship surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molmetal.scripts.pre_flight_check import (
    _nci60_index_triple_violations,
    _nci60_scaffold_pair_leaks,
    gate_nci60_split_no_leak,
    gate_scaffold_zero_leak,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NCI60_CSV = REPO_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"


class TestPureHelpers:
    def test_scaffold_pair_leaks_detects_overlap(self) -> None:
        leaks = _nci60_scaffold_pair_leaks(
            {"train": {"a", "b"}, "val": {"c"}, "test": {"b", "d"}}
        )
        assert ("train", "val", 0) in leaks
        assert ("train", "test", 1) in leaks
        assert ("val", "test", 0) in leaks

    def test_scaffold_pair_leaks_clean(self) -> None:
        leaks = _nci60_scaffold_pair_leaks(
            {"train": {"a"}, "val": {"b"}, "test": {"c"}}
        )
        assert all(n == 0 for _a, _b, n in leaks)

    def test_index_triple_violations_detects_overlap(self) -> None:
        bad = _nci60_index_triple_violations(
            [0, 1, 2], [2, 3], [4, 5, 6], n_rows=7
        )
        assert any("index overlap train/val" in v for v in bad)

    def test_index_triple_violations_detects_gap(self) -> None:
        bad = _nci60_index_triple_violations([0, 1], [2], [3], n_rows=9)
        assert any("partition incomplete" in v for v in bad)

    def test_index_triple_violations_clean(self) -> None:
        assert _nci60_index_triple_violations([0, 1], [2], [3], n_rows=4) == []


class TestScaffoldZeroLeakMultiCsv:
    def test_passes_on_clean_synthetic_csv(self, tmp_path: Path) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "c1ccccc1", "CCN", "CO"],
                "split": ["train", "train", "val", "test"],
                "scaffold_id": ["s1", "s2", "s3", "s4"],
            }
        )
        p = tmp_path / "clean.csv"
        df.to_csv(p, index=False)
        r = gate_scaffold_zero_leak("t", "P0", "data", [(p, "synthetic")])
        assert r.status == "PASS", r.evidence
        assert "OK synthetic rows=4" in r.evidence

    def test_fails_on_injected_leak(self, tmp_path: Path) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "c1ccccc1", "CCN", "CO"],
                "split": ["train", "train", "val", "test"],
                # s2 spans train AND test -> injected leak
                "scaffold_id": ["s1", "s2", "s3", "s2"],
            }
        )
        p = tmp_path / "leaky.csv"
        df.to_csv(p, index=False)
        r = gate_scaffold_zero_leak("t", "P0", "data", [(p, "synthetic")])
        assert r.status == "FAIL", r.evidence
        assert "LEAK synthetic" in r.evidence
        assert "('test', 'train', 1)" in r.evidence

    def test_fails_on_missing_file(self, tmp_path: Path) -> None:
        r = gate_scaffold_zero_leak(
            "t", "P0", "data", [(tmp_path / "nope.csv", "missing")]
        )
        assert r.status == "FAIL", r.evidence
        assert "missing" in r.evidence


class TestNci60SplitGate:
    def test_fails_on_missing_csv(self, tmp_path: Path) -> None:
        r = gate_nci60_split_no_leak("t", "P0", "data", tmp_path / "nope.csv")
        assert r.status == "FAIL", r.evidence
        assert "nci60 CSV missing" in r.evidence

    def test_healthy_tree_passes_with_honest_negative(self) -> None:
        """Integration (~30 s): the shipped cohort PASSes + reports the
        HOLE-5 honest-negative (real scaffold overlap, degenerate splitter).
        """
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        pytest.importorskip("rdkit")
        if not NCI60_CSV.is_file():
            pytest.skip(f"nci60 CSV not on disk: {NCI60_CSV}")
        r = gate_nci60_split_no_leak("t", "P0", "data", NCI60_CSV)
        assert r.status == "PASS", r.evidence
        assert "REAL-scaffold-overlap" in r.evidence
        assert "splitter_scaffold_for_degenerate=True" in r.evidence
        assert "HONEST-NEGATIVE" in r.evidence
