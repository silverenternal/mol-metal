"""Integration coverage for the round-11 engine parity harness."""

from __future__ import annotations

import json

from molmetal_lam.benchmarks import compare_engines


def test_compare_engines_returns_pair_metrics(tmp_path, sample_smiles):
    output_dir = tmp_path / "molmetal" / "reports" / "round11_engine_parity"
    summary = compare_engines(
        pocket_names=["1h36"], ligand_smiles=sample_smiles,
        output_dir=output_dir,
    )
    assert set(summary) == {"qvina", "quickvina2"}
    for metrics in summary.values():
        assert {"pearson_r", "mad", "n_samples"} <= set(metrics)
        assert metrics["n_samples"] == len(sample_smiles)
        assert isinstance(metrics["pearson_r"], float)
        assert isinstance(metrics["mad"], float)


def test_mock_engines_are_close_to_vina(tmp_path, sample_smiles):
    summary = compare_engines(
        pocket_names=["1h36", "830c"], ligand_smiles=sample_smiles,
        output_dir=tmp_path / "molmetal" / "reports" / "round11_engine_parity",
    )
    assert summary["qvina"]["mad"] < 2.0
    assert summary["quickvina2"]["mad"] < 2.0


def test_parity_writes_report_files(tmp_path, sample_smiles):
    output_dir = tmp_path / "molmetal" / "reports" / "round11_engine_parity"
    compare_engines(pocket_names=["1h36"], ligand_smiles=sample_smiles, output_dir=output_dir)
    assert output_dir.is_relative_to(tmp_path / "molmetal" / "reports" / "round11_engine_parity")
    assert (output_dir / "results.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    assert isinstance(json.loads((output_dir / "summary.json").read_text()), dict)
