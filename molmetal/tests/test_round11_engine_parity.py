import shutil
import subprocess

import pytest


def test_import():
    from molmetal_lam.benchmarks.engine_parity import compare_engines
    assert callable(compare_engines)


def test_mock_qvina_score_is_noisy():
    from molmetal_lam.benchmarks.engine_parity import mock_qvina
    assert mock_qvina("CC", "1h36")[0] != mock_qvina("CC", "1h36")[0]


def test_mock_quickvina2_score_is_noisy():
    from molmetal_lam.benchmarks.engine_parity import mock_quickvina2
    assert mock_quickvina2("CC", "1h36")[0] != mock_quickvina2("CC", "1h36")[0]


@pytest.mark.slow
def test_compare_engines_produces_files(tmp_path):
    from molmetal_lam.benchmarks.engine_parity import compare_engines
    compare_engines(["1h36", "830c"], ["CC", "CCC"], tmp_path)
    assert (tmp_path / "results.csv").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_pearson_r_computation():
    from molmetal_lam.benchmarks.engine_parity import _pearson
    assert -1 <= _pearson([1, 2], [2, 4]) <= 1


def test_mad_computation():
    from molmetal_lam.benchmarks.engine_parity import compare_engines
    assert compare_engines([], [], "/tmp/round11-test")["qvina"]["mad"] >= 0


@pytest.mark.skipif(
    not shutil.which("vina")
    or subprocess.run(
        [shutil.which("vina") or "vina", "--help"],
        capture_output=True,
        check=False,
    ).returncode != 0,
    reason="working vina CLI binary not available (shared libraries may be missing)",
)
def test_real_docking_available():
    from molmetal_lam.benchmarks.engine_parity import real_vina_score
    assert real_vina_score("CC", "1h36") is not None
