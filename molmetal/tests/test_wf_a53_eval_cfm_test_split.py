"""Wf-A53 — the held-out-split evaluator's honest-reporting contract.

The evaluator itself runs the trainer's public evaluation functions, which
are already covered by their own tests.  What is new here — and what can
silently lie — is the bookkeeping around them:

* ``_cohort_summary`` must count metals from the CLOUD's atomic numbers, not
  from the CSV.  The loader is supposed to INSERT the metal (Wf-A66); a row
  whose metal never reached the cloud would train a ligand-only model while
  every row-level count still looked correct.
* ``_split_rows`` must report the split's real size independently of how
  many rows happened to embed, so a capped run cannot be read as a
  full-split result.
* The parser must default to ``--split test`` and CPU, and must not require
  any CLI hyperparameter beyond the two paths (§5.11 applies to launches;
  this is an eval reader, but the same "no smuggled knobs" spirit applies).

No model is loaded and no cohort is embedded here — those paths are 2 min
and GPU-bound.  The tests below are sub-second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.wf_a53_eval_cfm_test_split import (  # noqa: E402
    _METAL_Z,
    _build_parser,
    _cohort_summary,
    _resolve,
    _smiles_fragment_stats,
    _split_rows,
    _training_overlap,
)

torch = pytest.importorskip("torch")


class _Cloud:
    """Minimal stand-in: only the three attributes the summary reads."""

    def __init__(self, z, bonds=True):
        self.atomic_numbers = torch.tensor(list(z), dtype=torch.long)
        self.bonds = torch.zeros((2, 1), dtype=torch.long) if bonds else None


def test_parser_defaults_to_test_split_and_cpu():
    args = _build_parser().parse_args(
        ["--ckpt-path", "x.pt", "--out-json", "y.json"]
    )
    assert args.split == "test"
    assert args.device == "cpu"
    assert args.n_clouds is None  # None means "every embeddable row"
    assert args.dry_run is False


def test_parser_requires_both_paths():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--ckpt-path", "x.pt"])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--out-json", "y.json"])


def test_cohort_summary_counts_metals_from_atomic_numbers():
    clouds = [
        _Cloud([6, 6, _METAL_Z["Ru"]]),
        _Cloud([6, _METAL_Z["Ir"]]),
        _Cloud([6, 7, 8]),  # no metal — must be surfaced, not ignored
    ]
    out = _cohort_summary(clouds)
    assert out["n_clouds"] == 3
    assert out["n_with_bond_truth"] == 3
    assert out["clouds_by_metal"]["Ru"] == 1
    assert out["clouds_by_metal"]["Ir"] == 1
    assert out["clouds_by_metal"]["Os"] == 0
    assert out["clouds_with_no_metal"] == 1
    assert out["atoms_min"] == 2
    assert out["atoms_max"] == 3


def test_cohort_summary_reports_missing_bond_truth():
    out = _cohort_summary([_Cloud([6, _METAL_Z["Re"]], bonds=False)])
    assert out["n_with_bond_truth"] == 0


def test_split_rows_is_independent_of_embedding(tmp_path):
    pd = pytest.importorskip("pandas")
    csv = tmp_path / "corpus.csv"
    pd.DataFrame(
        {
            "SMILES_Ligands": ["CCO", "CCO", "c1ccccc1", "CCN"],
            "Metal": ["Ru", "Ru", "Ir", "Os"],
            "split": ["train", "train", "val", "test"],
        }
    ).to_csv(csv, index=False)

    out = _split_rows(csv, "test")
    assert out["available"] is True
    assert out["n_rows"] == 1
    assert out["n_unique_smiles"] == 1
    assert out["n_total_rows"] == 4
    assert out["split_value_counts"] == {"train": 2, "val": 1, "test": 1}
    assert out["rows_by_metal"] == {"Os": 1}


def test_split_rows_survives_a_missing_split_column(tmp_path):
    pd = pytest.importorskip("pandas")
    csv = tmp_path / "no_split.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(csv, index=False)
    out = _split_rows(csv, "test")
    assert out["available"] is False


def test_resolve_anchors_relative_paths_at_repo_root():
    assert _resolve(".data/molmetal/x.csv") == _REPO_ROOT / ".data/molmetal/x.csv"
    assert _resolve("/abs/x.csv") == Path("/abs/x.csv")


def _write_corpus(path, rows):
    pd = pytest.importorskip("pandas")
    pd.DataFrame(rows).to_csv(path, index=False)


def test_training_overlap_says_not_held_out_when_molecules_are_shared(tmp_path):
    train = tmp_path / "train.csv"
    _write_corpus(
        train,
        {
            "smiles": ["CCO", "c1ccccc1"],
            "split": ["train", "train"],
        },
    )
    ev = tmp_path / "eval.csv"
    _write_corpus(
        ev,
        {
            "smiles": ["CCO", "CCN"],
            "split": ["test", "test"],
        },
    )
    out = _training_overlap(
        {"data_csv": str(train), "data_split": "train"}, ev, "test"
    )
    assert out["verdict"] == "NOT_HELD_OUT"
    assert out["n_eval_unique"] == 2
    assert out["n_shared_unique"] == 1
    assert out["frac_eval_seen_in_training"] == pytest.approx(0.5)


def test_training_overlap_says_held_out_when_disjoint(tmp_path):
    train = tmp_path / "train.csv"
    _write_corpus(train, {"smiles": ["CCO"], "split": ["train"]})
    ev = tmp_path / "eval.csv"
    _write_corpus(ev, {"smiles": ["c1ccccc1"], "split": ["test"]})
    out = _training_overlap(
        {"data_csv": str(train), "data_split": "train"}, ev, "test"
    )
    assert out["verdict"] == "HELD_OUT"
    assert out["n_shared_unique"] == 0


def test_training_overlap_is_unknown_without_a_training_csv(tmp_path):
    ev = tmp_path / "eval.csv"
    _write_corpus(ev, {"smiles": ["CCO"], "split": ["test"]})
    out = _training_overlap({}, ev, "test")
    assert out["verdict"] == "UNKNOWN"


def test_fragment_stats_expose_the_loose_atom_artifact():
    """A decode that parses is not a decode that is connected.

    MEASURED on the shipped checkpoint: decode_ratio 256/256 while every
    molecule was 10-18 fragments.  If this test ever reports 1.0 for the
    fragment-heavy list, the report can again read "parses" as "correct".
    """
    pytest.importorskip("rdkit")
    out = _smiles_fragment_stats(
        [
            "CCO",  # one fragment
            "CCO.O.O.[HH]",  # loose atoms
            "c1ccccc1.C[Rh](C)C",  # two fragments, metal present
        ]
    )
    assert out["n_decoded"] == 3
    assert out["n_single_fragment"] == 1
    assert out["single_fragment_ratio"] == pytest.approx(1 / 3)
    assert out["fragments_max"] == 4
    assert out["n_containing_metal"] == 1


def test_fragment_stats_skip_unparseable_smiles():
    pytest.importorskip("rdkit")
    out = _smiles_fragment_stats(["CCO", "not-a-smiles", ""])
    assert out["n_decoded"] == 1
    assert out["single_fragment_ratio"] == 1.0


def test_fragment_stats_report_zero_when_nothing_decoded():
    pytest.importorskip("rdkit")
    out = _smiles_fragment_stats([])
    assert out["n_decoded"] == 0
