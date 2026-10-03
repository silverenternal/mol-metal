"""Wf-446 L5 — known-leaky v1 CSV visibility gate tests.

CPU-only, no network, no GPU.  Covers the L5 fix gate
``gate_metalcytotox_v1_known_leaky``:

1. PASS on a leaky synthetic CSV (the healthy v1 shape: leaks > 0).
2. FAIL on a clean synthetic CSV (the mutated / re-split artifact shape).
3. SKIP on an absent file (gitignored ``.data/**`` host contract).
4. SKIP on the pandas env-gap probe failure (Wf-345B1 contract).
5. FAIL on unparsable probe output (probe-contract violation).
6. Registration contract: ``p0_metalcytotox_v1_known_leaky`` is
   'all'-only; ``len(build_gates("all")) == EXPECTED_GATE_COUNTS["all"]``
   (the Wf-321D §5.3 same-edit bump invariant).
7. (integration, ~1 s) the live v1 CSV PASSes and its evidence names the
   consumer roster + the frozen sha256 prefix.

See TODO/completed/321d_preflight_gate_audit.md §6.5 for the ship
surface + both-ways MEASURED evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molmetal.scripts import pre_flight_check as pfc
from molmetal.scripts.pre_flight_check import (
    EXPECTED_GATE_COUNTS,
    build_gates,
    gate_metalcytotox_v1_known_leaky,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_CSV = REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"


def _write_csv(p: Path, scaffold_ids, splits) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "smiles": ["CCO"] * len(scaffold_ids),
            "split": splits,
            "scaffold_id": scaffold_ids,
        }
    )
    df.to_csv(p, index=False)


class TestV1KnownLeakyGate:
    def test_pass_on_leaky_synthetic_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "leaky_v1_like.csv"
        # s2 spans train AND test -> the known-leaky v1 shape.
        _write_csv(
            p,
            ["s1", "s2", "s3", "s2"],
            ["train", "train", "val", "test"],
        )
        r = gate_metalcytotox_v1_known_leaky("t", "P0", "data", p)
        assert r.status == "PASS", r.evidence
        assert "still leaky as expected" in r.evidence
        assert "train/test=1" in r.evidence
        assert "metalcytotox_5metal_multitask_v4.csv" in r.evidence

    def test_fail_on_zero_leak_synthetic_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "clean_replacement.csv"
        # Disjoint scaffolds across all three splits -> 0 leaks.
        _write_csv(
            p,
            ["s1", "s2", "s3", "s4"],
            ["train", "train", "val", "test"],
        )
        r = gate_metalcytotox_v1_known_leaky("t", "P0", "data", p)
        assert r.status == "FAIL", r.evidence
        assert "no longer leaks" in r.evidence
        assert "configs/data_checksums.json" in r.evidence
        assert "p0_csv_hash_verify" in r.evidence

    def test_skip_on_missing_file(self, tmp_path: Path) -> None:
        r = gate_metalcytotox_v1_known_leaky("t", "P0", "data", tmp_path / "nope.csv")
        assert r.status == "SKIP", r.evidence
        assert "p0_csv_hash_verify" in r.evidence

    def test_skip_on_pandas_env_gap(self, tmp_path: Path, monkeypatch) -> None:
        p = tmp_path / "present.csv"
        p.write_text("split,scaffold_id\ntrain,s1\n", encoding="utf-8")

        def fake_run(cmd, timeout=None):  # noqa: ANN001, ANN202
            return (
                1,
                "",
                "Traceback (most recent call last):\n"
                "ModuleNotFoundError: No module named 'pandas'\n",
            )

        monkeypatch.setattr(pfc, "_run", fake_run)
        r = gate_metalcytotox_v1_known_leaky("t", "P0", "data", p)
        assert r.status == "SKIP", r.evidence
        assert "pandas" in r.evidence
        assert "env gap" in r.evidence

    def test_fail_on_unparsable_probe_output(self, tmp_path: Path, monkeypatch) -> None:
        p = tmp_path / "present.csv"
        p.write_text("split,scaffold_id\ntrain,s1\n", encoding="utf-8")

        def fake_run(cmd, timeout=None):  # noqa: ANN001, ANN202
            return (0, "garbage output\n", "")

        monkeypatch.setattr(pfc, "_run", fake_run)
        r = gate_metalcytotox_v1_known_leaky("t", "P0", "data", p)
        assert r.status == "FAIL", r.evidence
        assert "unparsable" in r.evidence


class TestRegistrationContract:
    def test_registered_all_only_and_count_synced(self) -> None:
        names_all = [nm for nm, _fn in build_gates("all")]
        assert "p0_metalcytotox_v1_known_leaky" in names_all
        # 'all'-only per the §3.4 design contract (internal pre-retrain
        # check; not a paper-grade claim).
        for pipeline in ("frontier", "chemprop_v2_v3", "cfm"):
            names = [nm for nm, _fn in build_gates(pipeline)]
            assert "p0_metalcytotox_v1_known_leaky" not in names, pipeline
        # Wf-321D §5.3: the constant must move in the same edit.
        assert len(names_all) == EXPECTED_GATE_COUNTS["all"]


class TestLiveTree:
    def test_live_v1_passes_with_evidence(self) -> None:
        """Integration (~1 s): the shipped v1 CSV PASSes with the
        measured overlap counts + consumer roster in the evidence."""
        pytest.importorskip("pandas")
        if not V1_CSV.is_file():
            pytest.skip(f"v1 CSV not on disk: {V1_CSV}")
        r = gate_metalcytotox_v1_known_leaky("t", "P0", "data", V1_CSV)
        assert r.status == "PASS", r.evidence
        assert "rows=26801" in r.evidence
        assert "total=109" in r.evidence
        assert "f167c11931ec" in r.evidence  # frozen sha256 prefix
        assert "train_ru_multitask.py" in r.evidence
