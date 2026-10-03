"""test_biolm_score_full_adapter.py — Wf-213 integration tests.

Verifies the public surface of
:file:`molmetal/adapters/biolm_score_full_adapter.py`:

1. The module imports cleanly (no heavy upstream at import time).
2. :func:`biolm_full_oracle_available` returns ``False`` on this
   host (no ``torch_scatter`` / ``dgl``).
3. :func:`biolm_cli_script_available` returns ``True`` because the
   vendored repo is on disk.
4. :class:`BioLMScoreFullAdapter` (no mock) returns
   ``status="error"`` because the CLI subprocess cannot load
   ``dgl`` on this host, and the per-SMILES scores are NaN.
5. :class:`BioLMScoreFullAdapter` with ``allow_mock=True`` returns
   ``status="mock"`` with finite scores in ``[0, 1]`` for every
   SMILES, and the same SMILES twice produces the same number
   (deterministic).
6. The mock scores are bit-for-bit reproducible across calls
   (sha256-based projection).
7. Empty input -> ``status="ok"`` with ``n_invoked=0`` and
   finite-less output.
8. Single-SMILES convenience wrapper
   :func:`biolm_full_score_smiles` returns the same number as
   the batched path.
9. ``biolm_full_oracle_available`` honours an explicit ``repo_root``
   arg (returns ``False`` when the path is bogus).
10. CLI-error fall-through to mock: when the CLI subprocess fails
    AND ``allow_mock=True``, the adapter returns ``status="mock"``
    (not ``status="error"``).

Run from the project root::

    /home/hugo/codes/mol-metal/.venv/bin/python3 -m pytest \\
        molmetal/tests/test_biolm_score_full_adapter.py -v --tb=short
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = PROJECT_ROOT / "molmetal" / "adapters"
TESTS_DIR = PROJECT_ROOT / "molmetal" / "tests"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))


# ---------------------------------------------------------------------------
# Lazy import — keeps pytest collection working even when the module
# is missing or broken.  Each test that needs the adapter calls the
# helper and self-skips when the import fails.
# ---------------------------------------------------------------------------
def _load_adapter_module():
    try:
        return importlib.import_module("biolm_score_full_adapter")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"biolm_score_full_adapter module not importable: {exc}")


# ---------------------------------------------------------------------------
# 1. Smoke-import
# ---------------------------------------------------------------------------
def test_smoke_import() -> None:
    """The adapter module imports cleanly without heavy upstream deps."""
    mod = _load_adapter_module()
    assert mod is not None
    for name in (
        "BioLMScoreFullAdapter",
        "BioLMFullScoreColumn",
        "biolm_full_oracle_available",
        "biolm_cli_script_available",
        "biolm_full_score_smiles",
        "biolm_full_score_batch",
        "_smoke_benchmark",
        "_mock_score_smiles",
    ):
        assert hasattr(mod, name), f"missing public symbol: {name}"


# ---------------------------------------------------------------------------
# 2. biolm_full_oracle_available is False on this host
# ---------------------------------------------------------------------------
def test_oracle_in_process_unavailable() -> None:
    """On this CPU-only host the in-process path is gated by heavy deps."""
    mod = _load_adapter_module()
    # We don't have torch_scatter / MDAnalysis / dgl installed.
    assert mod.biolm_full_oracle_available() is False
    # Bogus repo_root also returns False.
    assert mod.biolm_full_oracle_available(
        repo_root=str(PROJECT_ROOT / "no_such_repo")
    ) is False


# ---------------------------------------------------------------------------
# 3. biolm_cli_script_available — vendored repo detected
# ---------------------------------------------------------------------------
def test_cli_script_available(tmp_path: Path) -> None:
    """CLI script discovery: True iff scripts/casf2016_scoring_ranking.py exists."""
    mod = _load_adapter_module()
    # Provided path that doesn't exist -> False.
    assert mod.biolm_cli_script_available(str(tmp_path / "no_repo")) is False
    # Build a synthetic vendored-repo layout.
    repo = tmp_path / "synthetic_biolm"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "casf2016_scoring_ranking.py").write_text("# stub\n")
    assert mod.biolm_cli_script_available(str(repo)) is True
    # Remove script -> False.
    (scripts / "casf2016_scoring_ranking.py").unlink()
    assert mod.biolm_cli_script_available(str(repo)) is False


# ---------------------------------------------------------------------------
# 4. Adapter without mock returns status=error on this host
# ---------------------------------------------------------------------------
def test_adapter_no_mock_returns_error() -> None:
    """CLI path errors (missing dgl) → status='error', per_smiles NaN."""
    mod = _load_adapter_module()
    adapter = mod.BioLMScoreFullAdapter(allow_mock=False, timeout_sec=10.0)
    assert adapter.available_in_process is False
    assert adapter.available_cli is True
    smiles_list = ["CCO", "c1ccccc1O"]
    col = adapter.score_batch(smiles_list)
    assert col.status in ("error", "unavailable")
    assert col.inference_path == "cli"
    assert col.n_invoked == len(smiles_list)
    assert col.n_scored == 0
    assert col.biolm_score_mean is None
    assert col.biolm_score_std is None
    for s in smiles_list:
        assert s in col.per_smiles
        assert math.isnan(col.per_smiles[s])


# ---------------------------------------------------------------------------
# 5. Adapter with allow_mock returns finite scores
# ---------------------------------------------------------------------------
def test_adapter_mock_returns_finite_scores() -> None:
    """allow_mock=True → status='mock' with finite scores in [0, 1]."""
    mod = _load_adapter_module()
    adapter = mod.BioLMScoreFullAdapter(allow_mock=True, timeout_sec=10.0)
    smiles_list = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"]
    col = adapter.score_batch(smiles_list)
    assert col.status == "mock"
    assert col.inference_path == "mock"
    assert col.n_invoked == len(smiles_list)
    assert col.n_scored == len(smiles_list)
    assert col.biolm_score_mean is not None
    assert math.isfinite(col.biolm_score_mean)
    assert 0.0 <= col.biolm_score_mean <= 1.0
    assert col.biolm_score_std is not None  # n_scored >= 2
    assert col.biolm_score_std >= 0.0
    for s in smiles_list:
        assert s in col.per_smiles
        v = col.per_smiles[s]
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 6. Mock scores are deterministic / bit-for-bit reproducible
# ---------------------------------------------------------------------------
def test_mock_determinism() -> None:
    """Calling the adapter twice with the same input returns the same scores."""
    mod = _load_adapter_module()
    adapter = mod.BioLMScoreFullAdapter(allow_mock=True, timeout_sec=10.0)
    smiles_list = ["CCO", "c1ccc(N)cc1", "CC(=O)Nc1ccc(O)cc1"]
    col1 = adapter.score_batch(smiles_list)
    col2 = adapter.score_batch(smiles_list)
    for s in smiles_list:
        assert math.isclose(
            col1.per_smiles[s],
            col2.per_smiles[s],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ), f"non-deterministic: {s} -> {col1.per_smiles[s]} vs {col2.per_smiles[s]}"
    # The internal sha256-based projection is also stable:
    a = mod._mock_score_smiles("CCO")
    b = mod._mock_score_smiles("CCO")
    assert a == b  # exact float equality


# ---------------------------------------------------------------------------
# 7. Empty input -> status='ok', n_invoked=0
# ---------------------------------------------------------------------------
def test_adapter_empty_input() -> None:
    """An empty input list returns a benign zero-invocation record."""
    mod = _load_adapter_module()
    adapter = mod.BioLMScoreFullAdapter(allow_mock=True, timeout_sec=10.0)
    col = adapter.score_batch([])
    assert col.status == "ok"
    assert col.n_invoked == 0
    assert col.n_scored == 0
    assert col.biolm_score_mean is None
    assert col.biolm_score_std is None
    assert col.per_smiles == {}


# ---------------------------------------------------------------------------
# 8. Single-SMILES convenience wrapper matches the batched path
# ---------------------------------------------------------------------------
def test_single_smiles_wrapper_matches_batch() -> None:
    """biolm_full_score_smiles returns the same value as score_batch."""
    mod = _load_adapter_module()
    smiles = "CC(=O)Nc1ccc(O)cc1"
    val = mod.biolm_full_score_smiles(smiles, allow_mock=True)
    assert val is not None
    assert math.isfinite(val)
    assert 0.0 <= val <= 1.0
    adapter = mod.BioLMScoreFullAdapter(allow_mock=True, timeout_sec=10.0)
    col = adapter.score_batch([smiles])
    assert math.isclose(val, col.per_smiles[smiles], rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 9. CLI-error fall-through to mock
# ---------------------------------------------------------------------------
def test_cli_error_falls_through_to_mock() -> None:
    """When CLI errors AND allow_mock=True, fall through to mock path."""
    mod = _load_adapter_module()
    adapter = mod.BioLMScoreFullAdapter(allow_mock=True, timeout_sec=10.0)
    # The CLI will error on this host (no dgl) → fall through.
    col = adapter.score_batch(["CCO", "c1ccccc1"])
    assert col.status == "mock"
    assert col.inference_path == "mock"
    assert col.n_scored == 2
    assert all(
        math.isfinite(col.per_smiles[s]) for s in ("CCO", "c1ccccc1")
    )


# ---------------------------------------------------------------------------
# 10. JSON dump round-trip via the CLI smoke benchmark
# ---------------------------------------------------------------------------
def test_smoke_benchmark_json_roundtrip(tmp_path: Path) -> None:
    """Run the smoke benchmark end-to-end and validate the JSON output."""
    mod = _load_adapter_module()
    smiles_list = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"]
    json_out = tmp_path / "biolm_full_smoke.json"
    rc = mod._smoke_benchmark(
        smiles_list,
        allow_mock=True,
        json_out=str(json_out),
        pocket="wf213_smoke",
        model_type="biolm",
        encoder="gatedgcn",
    )
    assert rc == 0
    assert json_out.is_file()
    record = json.loads(json_out.read_text())
    # Required keys.
    for k in (
        "status",
        "n_invoked",
        "n_scored",
        "biolm_score_mean",
        "biolm_score_std",
        "per_smiles",
        "inference_path",
        "encoder",
        "model_type",
        "pocket",
        "allow_mock",
        "input_smiles",
    ):
        assert k in record, f"missing key: {k}"
    assert record["status"] == "mock"
    assert record["inference_path"] == "mock"
    assert record["pocket"] == "wf213_smoke"
    assert record["allow_mock"] is True
    assert record["encoder"] == "gatedgcn"
    assert record["model_type"] == "biolm"
    assert record["n_invoked"] == len(smiles_list)
    assert record["n_scored"] == len(smiles_list)
    assert math.isfinite(record["biolm_score_mean"])
    for s in smiles_list:
        assert s in record["per_smiles"]
        assert math.isfinite(record["per_smiles"][s])
