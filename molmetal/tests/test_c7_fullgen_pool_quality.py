"""Tests for the per-epoch pool-quality tabulation (C7 18-epoch pool)."""

from __future__ import annotations

import csv
import json

from molmetal.scripts.c7_fullgen_pool_quality import _epoch_summary, main


def _row(epoch: int, smiles: str, n_frags: int = 1, frac: float = 1.0) -> dict:
    return {
        "epoch": str(epoch),
        "smiles": smiles,
        "n_frags_raw": str(n_frags),
        "largest_frag_frac": str(frac),
    }


def test_single_fragment_counts_only_n_frags_equal_one():
    rows = [
        _row(0, "c1ccccc1", n_frags=1),
        _row(0, "C.C.O", n_frags=3),
        _row(0, "[Na+].[Cl-]", n_frags=2),
    ]
    out = _epoch_summary(rows)
    assert out["n_rows"] == 3
    assert out["single_fragment_rate"] == round(1 / 3, 4)


def test_unparseable_rows_are_excluded_from_every_rate():
    rows = [_row(0, "c1ccccc1", 1), _row(0, "not-a-smiles", 1)]
    out = _epoch_summary(rows)
    assert out["n_parse"] == 1
    assert out["validity"] == 0.5
    # 1 of 1 parsed rows is single-fragment, not 1 of 2 raw rows.
    assert out["single_fragment_rate"] == 1.0


def test_ring_shares_are_fractions_of_all_atom_rings():
    # Benzene: one 6-ring. Cyclopropane: one 3-ring.  -> 3-ring share 0.5
    rows = [_row(0, "c1ccccc1", 1), _row(0, "C1CC1", 1)]
    out = _epoch_summary(rows)
    assert out["ring_size_distribution"] == {"3": 1, "6": 1}
    assert out["three_ring_share"] == 0.5
    assert out["six_ring_share"] == 0.5


def test_first_vs_last_uses_numeric_epoch_order(tmp_path):
    # Regression: max() over string keys picked "9" over "17", so an 18-epoch
    # pool reported epoch 9 as its last one.
    run = tmp_path / "run"
    run.mkdir()
    with (run / "generated_smiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
        )
        writer.writeheader()
        for epoch in range(12):
            writer.writerow(_row(epoch, "c1ccccc1", n_frags=1 if epoch < 9 else 2))
    out_path = tmp_path / "out.json"
    assert main(["--run", str(run), "--out", str(out_path)]) == 0
    payload = json.loads(out_path.read_text())
    assert payload["first_vs_last"]["epoch_first"] == 0
    assert payload["first_vs_last"]["epoch_last"] == 11
    assert payload["first_vs_last"]["single_fragment_rate"] == [1.0, 0.0]


def test_missing_fragment_columns_do_not_crash(tmp_path):
    # A pool from an older trainer has no n_frags_raw column at all.
    run = tmp_path / "run"
    run.mkdir()
    with (run / "generated_smiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "smiles"])
        writer.writeheader()
        writer.writerow({"epoch": "0", "smiles": "c1ccccc1"})
    out_path = tmp_path / "out.json"
    assert main(["--run", str(run), "--out", str(out_path)]) == 0
    payload = json.loads(out_path.read_text())
    assert payload["overall"]["single_fragment_rate"] is None
    assert payload["overall"]["mean_largest_frag_frac"] is None


def test_bond_head_trajectory_is_none_without_a_metrics_file(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with (run / "generated_smiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
        )
        writer.writeheader()
        writer.writerow(_row(0, "c1ccccc1"))
    out_path = tmp_path / "out.json"
    assert main(["--run", str(run), "--out", str(out_path)]) == 0
    assert json.loads(out_path.read_text())["bond_head_trajectory"] is None


def test_bond_head_trajectory_carries_the_head_metrics(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with (run / "generated_smiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
        )
        writer.writeheader()
        writer.writerow(_row(0, "c1ccccc1"))
    (run / "per_epoch_metrics.json").write_text(
        json.dumps(
            {
                "per_epoch_metrics": [
                    {"epoch": 0, "bond_f1": 0.1, "bond_recall": 0.05, "bond_n_scored": 192},
                    {"epoch": 1, "bond_f1": 0.6, "bond_recall": 0.48, "bond_n_scored": 192},
                ]
            }
        )
    )
    out_path = tmp_path / "out.json"
    assert main(["--run", str(run), "--out", str(out_path)]) == 0
    traj = json.loads(out_path.read_text())["bond_head_trajectory"]
    assert [row["bond_f1"] for row in traj] == [0.1, 0.6]
    assert traj[0]["bond_n_scored"] == 192


def test_bonds_per_heavy_atom_divides_by_atoms_not_by_molecules():
    # Regression: the first version divided total bonds by the number of
    # molecules, so a pool of 40-atom molecules scored ~40 instead of ~1.
    rows = [_row(0, "c1ccccc1", 1), _row(0, "C" * 20, 1)]
    out = _epoch_summary(rows)
    # benzene: 6 heavy, 6 bonds; C20: 20 heavy, 19 bonds -> 25 / 26
    assert out["bonds_per_heavy_atom"] == round(25 / 26, 4)


def test_reference_stats_are_none_when_the_corpus_is_absent(tmp_path):
    from molmetal.scripts.c7_fullgen_pool_quality import _reference_stats

    assert _reference_stats(tmp_path / "nope.csv") is None


def test_reference_stats_use_the_train_split_only(tmp_path):
    import pandas as pd

    from molmetal.scripts.c7_fullgen_pool_quality import _reference_stats

    path = tmp_path / "ref.csv"
    pd.DataFrame(
        {
            "smiles": ["c1ccccc1", "C1CC1", "C1CC1"],
            "split": ["train", "test", "test"],
        }
    ).to_csv(path, index=False)
    ref = _reference_stats(path)
    assert ref["n"] == 1
    assert ref["ring_size_distribution"] == {"6": 1}
    assert ref["three_ring_share"] == 0.0
