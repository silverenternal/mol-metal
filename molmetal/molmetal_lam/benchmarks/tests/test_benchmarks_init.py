"""Public API checks for :mod:`molmetal_lam.benchmarks`."""

from molmetal_lam.benchmarks import ENGINE_NAMES, compare_engines


def test_benchmarks_public_api():
    assert callable(compare_engines)
    assert ENGINE_NAMES == ("vina", "qvina_mock", "quickvina2_mock")
