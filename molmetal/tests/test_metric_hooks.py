"""Pytest: 3 measurement hooks (Vina / multi-cell AUC / decode ratio).

Covers the helpers in
:mod:`molmetal.molmetal_lam.sbdd_env.metric_hooks`:

* :func:`compute_vina_min`            — best Vina kcal/mol across ``n_samples``
* :func:`compute_multi_cell_auc`      — per-cell-line AUC for precious-metal D-MPNN
* :func:`compute_decode_ratio`        — decode-ratio probe of a CFM checkpoint

For each hook, we run with small inputs, assert the output is JSON-
serializable, and assert that ``metrics/by_metric/<name>.json`` was
written.  The hooks catch their own exceptions and surface them as an
``error`` field in the envelope, so a missing native binding (no
``vina`` Python module on the host, no checkpoint file at the supplied
path, etc.) does not break the contract — the tests still observe a
JSON dict.

All tests CPU-friendly, <10s on a single core.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helper — small molecule list reused across the three hooks.
# ---------------------------------------------------------------------------
SMILES_LIST = ["CCO", "c1ccccc1", "CCN", "CC(=O)O", "CCCC"]
CELL_LINES = ["MCF-7", "MCF-7", "A549", "A549", "A2780"]


def _is_jsonable(obj) -> bool:
    try:
        json.dumps(obj, default=str)
        return True
    except (TypeError, ValueError):
        return False


def _metric_file(name: str) -> Path:
    from molmetal.molmetal_lam.sbdd_env.metric_hooks import METRIC_OUTPUT_DIR

    return METRIC_OUTPUT_DIR / f"{name}.json"


def _redirect_metric_output(monkeypatch, tmp_path) -> Path:
    """Point the metric hooks' output directory at ``tmp_path``.

    Wf-A123: the hooks default to the PACKAGE tree
    ``molmetal/metrics/by_metric/``, which is a published artifact tree.
    A test run must never write there, so every ``save=True`` test
    redirects the module-level ``METRIC_OUTPUT_DIR`` into ``tmp_path``.
    """
    from molmetal.molmetal_lam.sbdd_env import metric_hooks

    out_dir = tmp_path / "by_metric"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(metric_hooks, "METRIC_OUTPUT_DIR", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Hook 1 — compute_vina_min
# ---------------------------------------------------------------------------
class TestComputeVinaMin:
    def test_vina_min_returns_serializable_dict(self):
        """`compute_vina_min` returns a JSON-serializable dict."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import compute_vina_min

        result = compute_vina_min(
            SMILES_LIST,
            pocket_id="synthetic:CHRM1",
            n_samples=3,
            save=False,  # don't pollute on_metric_dir during test
        )
        assert isinstance(result, dict), f"expected dict, got {type(result)}"
        assert _is_jsonable(result), "result must be JSON-serializable"
        for key in (
            "hook",
            "wall_clock_s",
            "pocket_id",
            "n_samples",
            "best_score",
            "scores",
        ):
            assert key in result, f"missing key {key!r} in {result.keys()}"

    def test_vina_min_best_score_is_float_or_nan(self):
        """`best_score` is a float (or NaN if Vina binding missing)."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import compute_vina_min

        result = compute_vina_min(
            SMILES_LIST,
            pocket_id="synthetic:default",
            n_samples=3,
            save=False,
        )
        bs = result["best_score"]
        assert isinstance(bs, float), f"expected float, got {type(bs)}"
        # NaN is acceptable (no Vina binding); otherwise must be a
        # realistic kcal/mol number.
        import math

        if not math.isnan(bs):
            assert (
                -25.0 <= bs <= 5.0
            ), f"Vina score {bs} outside plausible kcal/mol range"

    def test_vina_min_writes_metric_file(self, monkeypatch, tmp_path):
        """`save=True` (default) writes metrics/by_metric/vina_min.json."""
        _redirect_metric_output(monkeypatch, tmp_path)
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import compute_vina_min

        path = _metric_file("vina_min")
        if path.exists():
            path.unlink()
        compute_vina_min(
            SMILES_LIST,
            pocket_id="synthetic:CHRM1",
            n_samples=3,
        )
        assert path.is_file(), f"{path} was not written"
        # The file is itself JSON.
        data = json.loads(path.read_text())
        assert data["hook"] == "vina_min"
        assert "best_score" in data


# ---------------------------------------------------------------------------
# Hook 2 — compute_multi_cell_auc
# ---------------------------------------------------------------------------
class TestComputeMultiCellAUC:
    def _fresh_dmpnn(self):
        """Build a tiny D-MPNN multi-task model for smoke testing."""
        from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel

        with torch.random.fork_rng():
            torch.manual_seed(41)
            return DMPNNMultiTaskModel(hidden=16, depth=1, dropout=0.0)

    def test_multi_cell_auc_returns_serializable_dict(self):
        """`compute_multi_cell_auc` returns a JSON-serializable dict."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_multi_cell_auc,
        )

        model = self._fresh_dmpnn()
        result = compute_multi_cell_auc(
            model,
            SMILES_LIST,
            CELL_LINES,
            y_true=[0, 1, 1, 0, 1],
            save=False,
        )
        assert isinstance(result, dict), f"expected dict, got {type(result)}"
        assert _is_jsonable(result)
        for key in ("hook", "wall_clock_s", "n_smiles", "per_cell_auc"):
            assert key in result

    def test_multi_cell_auc_per_cell_dict_is_float_only(self):
        """`per_cell_auc` maps cell-line name -> float (or NaN)."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_multi_cell_auc,
        )

        model = self._fresh_dmpnn()
        result = compute_multi_cell_auc(
            model,
            SMILES_LIST,
            CELL_LINES,
            y_true=[0, 1, 1, 0, 1],
            save=False,
        )
        pca = result["per_cell_auc"]
        assert isinstance(pca, dict)
        for cl, auc in pca.items():
            assert isinstance(cl, str)
            assert isinstance(auc, float)

    def test_multi_cell_auc_writes_metric_file(self, monkeypatch, tmp_path):
        """`save=True` writes metrics/by_metric/multi_cell_auc.json."""
        _redirect_metric_output(monkeypatch, tmp_path)
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_multi_cell_auc,
        )

        path = _metric_file("multi_cell_auc")
        if path.exists():
            path.unlink()
        model = self._fresh_dmpnn()
        compute_multi_cell_auc(
            model,
            SMILES_LIST,
            CELL_LINES,
            y_true=[0, 1, 1, 0, 1],
        )
        assert path.is_file(), f"{path} was not written"
        data = json.loads(path.read_text())
        assert data["hook"] == "multi_cell_auc"


# ---------------------------------------------------------------------------
# Hook 3 — compute_decode_ratio
# ---------------------------------------------------------------------------
class TestComputeDecodeRatio:
    def test_decode_ratio_returns_serializable_dict(self):
        """`compute_decode_ratio` returns a JSON-serializable dict."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_decode_ratio,
        )

        # Empty / non-existent checkpoint path → fresh adapter, no load.
        result = compute_decode_ratio(
            "/tmp/__nonexistent_ckpt_for_test__.pt",
            n_samples=4,
            n_steps=8,
            hidden_dim=32,
            n_layers=1,
            save=False,
        )
        assert isinstance(result, dict)
        assert _is_jsonable(result)
        for key in (
            "hook",
            "wall_clock_s",
            "checkpoint_path",
            "n_samples",
            "n_decoded",
            "decode_ratio",
            "checkpoint_loaded",
        ):
            assert key in result, f"missing key {key!r}"

    def test_decode_ratio_value_is_in_unit_interval(self):
        """`decode_ratio` is a float in [0, 1]."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_decode_ratio,
        )

        result = compute_decode_ratio(
            "/tmp/__nonexistent_ckpt_for_test__.pt",
            n_samples=4,
            n_steps=8,
            hidden_dim=32,
            n_layers=1,
            save=False,
        )
        ratio = result["decode_ratio"]
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0, f"decode_ratio {ratio} outside [0, 1]"

    def test_decode_ratio_writes_metric_file(self, monkeypatch, tmp_path):
        """`save=True` writes metrics/by_metric/decode_ratio.json."""
        _redirect_metric_output(monkeypatch, tmp_path)
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_decode_ratio,
        )

        path = _metric_file("decode_ratio")
        if path.exists():
            path.unlink()
        compute_decode_ratio(
            "/tmp/__nonexistent_ckpt_for_test__.pt",
            n_samples=4,
            n_steps=8,
            hidden_dim=32,
            n_layers=1,
        )
        assert path.is_file(), f"{path} was not written"
        data = json.loads(path.read_text())
        assert data["hook"] == "decode_ratio"
        assert data["n_samples"] == 4


# ---------------------------------------------------------------------------
# Helper — resolve_pocket
# ---------------------------------------------------------------------------
class TestResolvePocket:
    def test_synthetic_pocket_has_correct_id(self):
        """`synthetic:<name>` produces a Pocket with the right pdb_id."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import resolve_pocket

        p = resolve_pocket("synthetic:CHRM1", n_atoms=8)
        assert p.pdb_id == "synthetic:CHRM1"
        assert p.n_atoms == 8

    def test_unknown_pocket_falls_back_to_synthetic(self):
        """Unknown pocket_id still produces a valid Pocket."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import resolve_pocket

        p = resolve_pocket("not_a_real_pdb_id", n_atoms=16)
        assert p is not None
        assert p.n_atoms == 16
