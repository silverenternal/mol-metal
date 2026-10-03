"""Tests for the Wf-28 Phase 5 wet-lab data-prep benchmark.

================================================================
Wf-28 Phase 5 (2026-09-26).

End-to-end + unit-level coverage of
:mod:`molmetal.scripts.bench_wetlab_data` -- the bench script that
verifies the Wf-28 wet-lab data prep pipeline (Phase 1 tmQM subset
+ Phase 2 NCI-60 9-cell + PlatinAI 4-cell) is in the same descriptor
neighbourhood as the published PlatinAI 5K r18 reference.

All tests are CPU-only (RDKit + stdlib).  Bench runs under pytest
use the tiny ``--max-rows`` budget so the full sweep fits inside
the standard 5-minute wall-clock CI cap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy import of the subject module (so test file is collectible even
# when the heavier torch stack is broken).
# ---------------------------------------------------------------------------
try:
    from molmetal.scripts import bench_wetlab_data as bench  # noqa: E402

    _HAS_BENCH = True
except Exception:  # pragma: no cover - env guard
    bench = None  # type: ignore
    _HAS_BENCH = False


pytestmark = pytest.mark.skipif(
    not _HAS_BENCH,
    reason="bench_wetlab_data import failed; wetlab bench unavailable",
)


# ---------------------------------------------------------------------------
# Corpus-level CSV fixtures (the four real Wf-28 inputs).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TMQM_CSV = REPO_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
NCI60_CSV = REPO_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
PLATINAI_4CELL_CSV = REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
PLATINAI_5K_CSV = REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"


# ---------------------------------------------------------------------------
# 1. Helper-level: statistics, comparability, dataclasses.
# ---------------------------------------------------------------------------
def test_safe_mean_handles_empty() -> None:
    """An empty list returns NaN, not ZeroDivisionError."""
    import math

    out = bench._safe_mean([])
    assert isinstance(out, float)
    assert math.isnan(out)


def test_safe_mean_ignores_nan() -> None:
    """NaN entries are dropped before averaging."""
    out = bench._safe_mean([float("nan"), 2.0, 4.0])
    assert out == 3.0


def test_safe_std_requires_two_values() -> None:
    """Std with <2 finite values is NaN (sample std, ddof=1)."""
    import math

    assert math.isnan(bench._safe_std([1.0]))
    assert math.isnan(bench._safe_std([float("nan"), 2.0]))


def test_safe_std_basic() -> None:
    """Std of [1,2,3] is 1.0 (sample std)."""
    out = bench._safe_std([1.0, 2.0, 3.0])
    assert abs(out - 1.0) < 1e-9


def test_safe_count_counts_finite() -> None:
    """NaN values are not counted."""
    assert bench._safe_count([float("nan"), 1.0, 2.0]) == 2
    assert bench._safe_count([float("nan"), float("nan")]) == 0


# ---------------------------------------------------------------------------
# 2. Featurise + summarise.
# ---------------------------------------------------------------------------
def test_featurise_smiles_known_drug() -> None:
    """Cisplatin gives finite, sensible 2-D descriptors."""
    row = bench.featurise_smiles("Cl[Pt](Cl)(N)N", bench._RDKit(), *_try_get_sa())
    assert row.mw == row.mw and row.mw > 200
    assert row.logp == row.logp
    assert row.qed == row.qed and 0 <= row.qed <= 1
    assert row.sa_unit == row.sa_unit and 0 <= row.sa_unit <= 1


def test_featurise_smiles_garbage_returns_nan() -> None:
    """Garbage SMILES → NaN fields (no crash)."""
    row = bench.featurise_smiles("not_a_smiles_@@@", bench._RDKit(), *_try_get_sa())
    import math

    assert math.isnan(row.mw)
    assert math.isnan(row.logp)
    assert math.isnan(row.qed)


def test_summarise_corpus_simple() -> None:
    """summarise_corpus returns a 5-stat row set for a tiny corpus."""
    stats = bench.summarise_corpus(
        "tiny",
        ["Cl[Pt](Cl)(N)N", "c1ccccc1"],
    )
    assert stats.name == "tiny"
    assert stats.n_candidates == 2
    assert stats.n_rows_input == 2
    assert stats.mean_mw > 0
    assert 0 < stats.mean_qed < 1
    assert 0 <= stats.mean_sa <= 1


def test_summarise_corpus_dedups_uniques() -> None:
    """n_candidates counts dedup'd SMILES, not duplicates."""
    stats = bench.summarise_corpus(
        "dupes",
        ["Cl[Pt](Cl)(N)N"] * 5 + ["c1ccccc1"],
    )
    assert stats.n_candidates == 2
    assert stats.n_rows_input == 6


# ---------------------------------------------------------------------------
# 3. Comparability verdict.
# ---------------------------------------------------------------------------
def _stats(name: str, **kwargs) -> bench.CorpusStats:
    """Build a minimal :class:`CorpusStats` with sensible defaults."""
    defaults = dict(
        n_candidates=10,
        n_rows_input=10,
        mean_mw=400.0,
        mean_logp=3.0,
        mean_qed=0.4,
        mean_sa=0.7,
        std_mw=100.0,
        std_logp=2.0,
        std_qed=0.2,
        std_sa=0.1,
        parse_failures=0,
        parse_failure_rate=0.0,
        secs=0.1,
    )
    defaults.update(kwargs)
    s = bench.CorpusStats(name=name, **defaults)
    return s


def test_comparability_identical_is_ok() -> None:
    """Two identical corpora are always comparable."""
    cand = _stats("cand", mean_mw=400.0)
    ref = _stats("ref", mean_mw=400.0)
    out = bench.check_comparability(cand, ref)
    assert out.is_comparable is True
    assert out.delta_mw == 0.0


def test_comparability_within_two_sigma_ok() -> None:
    """A candidate within ±2σ of the reference is comparable."""
    cand = _stats("cand", mean_mw=500.0, std_mw=100.0)  # Δ=+100 = 1σ
    ref = _stats("ref", mean_mw=400.0, std_mw=100.0)
    out = bench.check_comparability(cand, ref)
    assert out.is_comparable is True
    assert abs(out.delta_mw - 100.0) < 1e-9


def test_comparability_outside_two_sigma_fails() -> None:
    """Δ > ±2σ on at least one axis = OUT verdict."""
    cand = _stats("cand", mean_qed=0.05, std_qed=0.10)  # Δ=-0.35 > 2σ
    ref = _stats("ref", mean_qed=0.4, std_qed=0.10)
    out = bench.check_comparability(cand, ref)
    assert out.is_comparable is False
    assert "qed" in out.notes


def test_comparability_notes_populated_when_out() -> None:
    """The notes field lists every axis that failed the ±2σ rule."""
    cand = _stats(
        "cand",
        mean_mw=900.0,
        std_mw=50.0,
        mean_logp=-3.0,
        std_logp=0.5,
        mean_qed=0.05,
        std_qed=0.02,
        mean_sa=0.05,
        std_sa=0.05,
    )
    ref = _stats(
        "ref",
        std_mw=100.0,
        std_logp=2.0,
        std_qed=0.10,
        std_sa=0.10,
    )
    out = bench.check_comparability(cand, ref)
    assert out.is_comparable is False
    notes = out.notes
    assert "mw" in notes
    assert "logp" in notes
    assert "qed" in notes
    assert "sa" in notes


# ---------------------------------------------------------------------------
# 4. Loaders + IO smoke.
# ---------------------------------------------------------------------------
def test_read_csv_smiles_missing_returns_empty(tmp_path: Path) -> None:
    """A missing CSV returns an empty list (no exception)."""
    out = bench._read_csv_smiles(tmp_path / "missing.csv", "smiles")
    assert out == []


def test_read_tmqm_returns_strings(tmp_path: Path) -> None:
    """_read_tmqm yields the 'smiles' column of a minimal CSV."""
    (tmp_path / "tmqm.csv").write_text("csd_code,smiles\nA,c1ccccc1\nB,CCO\n")
    out = bench._read_tmqm(tmp_path / "tmqm.csv")
    assert out == ["c1ccccc1", "CCO"]


def test_read_nci60_dedupes(tmp_path: Path) -> None:
    """_read_nci60 deduplicates SMILES but preserves order."""
    (tmp_path / "nci.csv").write_text(
        "smiles,cell_line,pIC50,active\nc1ccccc1,A,5.0,1\nc1ccccc1,B,4.5,0\nCCO,A,6.0,1\n"
    )
    out = bench._read_nci60(tmp_path / "nci.csv")
    assert out == ["c1ccccc1", "CCO"]


def test_read_platinai_4cell_returns_smiles(tmp_path: Path) -> None:
    """_read_platinai_4cell reads the 'smiles' column."""
    (tmp_path / "p4.csv").write_text("complex_id,smiles\np1,Cl[Pt](Cl)(N)N\np2,NCCN\n")
    out = bench._read_platinai_4cell(tmp_path / "p4.csv")
    assert out == ["Cl[Pt](Cl)(N)N", "NCCN"]


def test_read_platinai_5k_returns_smiles(tmp_path: Path) -> None:
    """_read_platinai_5k reads the 'smiles' column."""
    (tmp_path / "p5.csv").write_text("id,smiles\np1,Cl[Pt](Cl)(N)N\np2,NCCN\n")
    out = bench._read_platinai_5k(tmp_path / "p5.csv")
    assert out == ["Cl[Pt](Cl)(N)N", "NCCN"]


# ---------------------------------------------------------------------------
# 5. JSON + Markdown writers.
# ---------------------------------------------------------------------------
def test_write_summary_json_round_trip(tmp_path: Path) -> None:
    """_write_summary_json round-trips a payload."""
    payload = {
        "timestamp_utc": "2026-09-26T00:00:00Z",
        "seed": 20260926,
        "corpora": [],
        "compat": [],
        "inputs": {},
    }
    out = tmp_path / "sub" / "bench.json"
    bench._write_summary_json(out, payload)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["seed"] == 20260926
    assert loaded["corpora"] == []


def test_write_summary_md_contains_corpora_table(tmp_path: Path) -> None:
    """The Markdown must include the 'Per-corpus benchmark metrics' header."""
    out = tmp_path / "bench.md"
    bench._write_summary_md(
        out,
        corpora=[_stats("tmqm_dblock"), _stats("platinai_5k")],
        compat=[],
        inputs={"tmqm_dblock_csv": "/tmp/x.csv", "platinai_5k_csv": "/tmp/y.csv"},
        timestamp="2026-09-26T00:00:00Z",
    )
    text = out.read_text()
    assert "Per-corpus benchmark metrics" in text
    assert "tmqm_dblock" in text
    assert "platinai_5k" in text


def test_write_summary_md_contains_compat_table(tmp_path: Path) -> None:
    """compat rows show up under the comparability header."""
    out = tmp_path / "bench.md"
    cmp = bench.check_comparability(_stats("a"), _stats("b"))
    bench._write_summary_md(
        out,
        corpora=[_stats("a"), _stats("b")],
        compat=[cmp],
        inputs={"a_csv": "/tmp/a.csv", "b_csv": "/tmp/b.csv"},
        timestamp="2026-09-26T00:00:00Z",
    )
    text = out.read_text()
    assert "Comparability" in text
    assert "a → b" in text


# ---------------------------------------------------------------------------
# 6. End-to-end CLI smoke (small --max-rows cap).
# ---------------------------------------------------------------------------
def test_main_runs_end_to_end(tmp_path: Path) -> None:
    """``bench_wetlab_data.main`` runs on tiny synthetic CSVs."""
    (tmp_path / "tmqm.csv").write_text(
        "csd_code,smiles,metal_center\nA,c1ccccc1,C\nB,Cl[Pt](Cl)(N)N,Pt\n"
    )
    (tmp_path / "nci60.csv").write_text("smiles,cell_line,pIC50,active\nCCO,A,5.0,1\n")
    (tmp_path / "p4.csv").write_text("complex_id,smiles\np1,Cl[Pt](Cl)(N)N\n")
    (tmp_path / "p5.csv").write_text("id,smiles\np1,c1ccccc1\n")
    rc = bench.main(
        [
            "--tmqm-csv",
            str(tmp_path / "tmqm.csv"),
            "--nci60-csv",
            str(tmp_path / "nci60.csv"),
            "--platinai-4cell-csv",
            str(tmp_path / "p4.csv"),
            "--platinai-5k-csv",
            str(tmp_path / "p5.csv"),
            "--summary-md",
            str(tmp_path / "bench.md"),
            "--summary-json",
            str(tmp_path / "bench.json"),
            "--max-rows",
            "10",
        ]
    )
    assert rc == 0
    assert (tmp_path / "bench.md").exists()
    assert (tmp_path / "bench.json").exists()
    payload = json.loads((tmp_path / "bench.json").read_text())
    assert "corpora" in payload
    assert len(payload["corpora"]) == 4


def test_main_writes_compat_block(tmp_path: Path) -> None:
    """Compat block is always emitted (Plat 5K is always present)."""
    (tmp_path / "tmqm.csv").write_text("csd_code,smiles\nA,c1ccccc1\n")
    (tmp_path / "nci60.csv").write_text("smiles,cell_line,pIC50,active\nCCO,A,5,1\n")
    (tmp_path / "p4.csv").write_text("complex_id,smiles\np1,Cl[Pt](Cl)(N)N\n")
    (tmp_path / "p5.csv").write_text("id,smiles\np1,c1ccccc1\n")
    rc = bench.main(
        [
            "--tmqm-csv",
            str(tmp_path / "tmqm.csv"),
            "--nci60-csv",
            str(tmp_path / "nci60.csv"),
            "--platinai-4cell-csv",
            str(tmp_path / "p4.csv"),
            "--platinai-5k-csv",
            str(tmp_path / "p5.csv"),
            "--summary-md",
            str(tmp_path / "bench.md"),
            "--summary-json",
            str(tmp_path / "bench.json"),
            "--max-rows",
            "5",
        ]
    )
    assert rc == 0
    payload = json.loads((tmp_path / "bench.json").read_text())
    assert "compat" in payload
    # 3 candidate corpora (all but platinai_5k) should each have 1 compat row.
    assert len(payload["compat"]) == 3
    md = (tmp_path / "bench.md").read_text()
    assert "Comparability" in md
    assert "Verdict" in md


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _try_get_sa():
    """Return (sa_ertl, sa_to_unit) or no-op fallbacks when sascorer is missing."""
    try:
        from molmetal.molmetal_lam.sbdd_env.sa_score import (  # type: ignore
            sa_score_ertl,
            sa_score_to_unit,
        )

        return sa_score_ertl, sa_score_to_unit
    except Exception:  # pragma: no cover - sascorer missing in env

        def _noop(_smi: str) -> float:
            return float("nan")

        return _noop, _noop
