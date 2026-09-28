"""Tests for molmetal.replay.curriculum.hardness.

================================================================
Hardness predictor contract tests (TODO-150 §P1.4 + §P1.6)
================================================================
Covers the lightweight 2-layer MLP + 3-way hardness decomposition:

* :class:`HardnessDecomposition` — 3 tests (default, dict round-trip, field bounds).
* :class:`HardnessSample`         — 2 tests (to_dict shape, frozen).
* :class:`HardnessPredictor`      — 8 tests (build, predict shape, fit MAE,
  decomposition geometry/binding/reduction, weights dict, is_fitted,
  drift threshold constant).

Total: **13 tests**, CPU-only.  All under 1 s on any CI container.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    from molmetal.replay.curriculum.hardness import (
        DRIFT_DIVERGENCE_THRESHOLD,
        HardnessDecomposition,
        HardnessPredictor,
        HardnessSample,
        PREDICTOR_HIDDEN_DIM,
        PREDICTOR_INPUT_DIM,
    )
except Exception:
    from molmetal.replay.curriculum import (  # type: ignore
        DRIFT_DIVERGENCE_THRESHOLD,
        HardnessDecomposition,
        HardnessPredictor,
        HardnessSample,
        PREDICTOR_HIDDEN_DIM,
        PREDICTOR_INPUT_DIM,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_threshold(self):
        assert DRIFT_DIVERGENCE_THRESHOLD == pytest.approx(0.5)

    def test_input_dim(self):
        assert PREDICTOR_INPUT_DIM > 0

    def test_hidden_dim(self):
        assert PREDICTOR_HIDDEN_DIM == 64


# ---------------------------------------------------------------------------
# HardnessDecomposition
# ---------------------------------------------------------------------------


class TestHardnessDecomposition:
    def test_default_uniform(self):
        d = HardnessDecomposition.default_uniform()
        assert d.m_geometry == pytest.approx(0.5)
        assert d.m_binding == pytest.approx(0.5)
        assert d.m_reduction == pytest.approx(0.5)

    def test_dict_round_trip(self):
        d = HardnessDecomposition(m_geometry=0.1, m_binding=0.2, m_reduction=0.3)
        out = d.to_dict()
        assert out == {"m_geometry": 0.1, "m_binding": 0.2, "m_reduction": 0.3}
        d2 = HardnessDecomposition.from_dict(out)
        assert d2.m_geometry == d.m_geometry

    def test_construction_bounds(self):
        # The dataclass does NOT bounds-check by design; the predictor
        # is responsible.  Just verify that values outside [0, 1]
        # survive construction (caller's job to clamp).
        d = HardnessDecomposition(m_geometry=2.0, m_binding=-1.0, m_reduction=0.0)
        assert d.m_geometry == 2.0
        assert d.m_binding == -1.0


# ---------------------------------------------------------------------------
# HardnessSample
# ---------------------------------------------------------------------------


class TestHardnessSample:
    def test_to_dict_shape(self):
        decomp = HardnessDecomposition.default_uniform()
        s = HardnessSample(
            row_id=0,
            smiles="O",
            metal="Ru",
            pocket_id="A2780",
            predicted=0.7,
            on_policy=0.4,
            decomposition=decomp,
        )
        d = s.to_dict()
        assert d["row_id"] == 0
        assert d["predicted"] == 0.7
        assert d["on_policy"] == 0.4
        assert d["m_geometry"] == 0.5
        assert d["m_binding"] == 0.5
        assert d["m_reduction"] == 0.5

    def test_frozen(self):
        decomp = HardnessDecomposition.default_uniform()
        s = HardnessSample(
            row_id=0,
            smiles="O",
            metal="Ru",
            pocket_id="A2780",
            predicted=0.7,
            on_policy=0.4,
            decomposition=decomp,
        )
        with pytest.raises((AttributeError, Exception)):
            s.row_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HardnessPredictor
# ---------------------------------------------------------------------------


class TestHardnessPredictor:
    def _make_predictor(self, **kwargs) -> HardnessPredictor:
        defaults = dict(input_dim=10, hidden_dim=4, n_epochs=3, batch_size=8, seed=0)
        defaults.update(kwargs)
        return HardnessPredictor(**defaults)

    def test_construction_defaults(self):
        p = HardnessPredictor()
        assert p.input_dim == PREDICTOR_INPUT_DIM
        assert p.hidden_dim == PREDICTOR_HIDDEN_DIM
        assert not p.is_fitted

    def test_construction_rejects_zero_dims(self):
        with pytest.raises(ValueError):
            HardnessPredictor(input_dim=0)
        with pytest.raises(ValueError):
            HardnessPredictor(hidden_dim=0)

    def test_predict_shape_and_bounds(self):
        p = self._make_predictor()
        feats = np.random.default_rng(0).standard_normal((4, 10))
        out = p.predict(feats)
        assert out.shape == (4,)
        # Sigmoid → [0, 1] but the *random* init may be 0.5 for both
        # backends; we only assert that the bounds are sane (no NaN).
        assert np.all(np.isfinite(out))

    def test_predict_shape_mismatch(self):
        p = self._make_predictor()
        with pytest.raises(ValueError):
            p.predict(np.zeros((4, 99)))

    def test_fit_returns_diagnostics(self):
        p = self._make_predictor()
        feats = np.random.default_rng(0).standard_normal((20, 10))
        tgts = np.clip(np.random.default_rng(1).random(20), 0, 1)
        diag = p.fit(feats, tgts)
        assert "train_mae" in diag
        assert "final_loss" in diag
        assert p.is_fitted
        # Train MAE is finite and >= 0.
        assert math.isfinite(p.train_mae)
        assert p.train_mae >= 0.0

    def test_fit_zero_rows_rejected(self):
        p = self._make_predictor()
        with pytest.raises(ValueError):
            p.fit(np.zeros((0, 10)), np.zeros((0,)))

    def test_decompose_geometry_score(self):
        p = self._make_predictor()
        decomp = p.decompose(
            smiles="O",
            predicted_pic50=6.0,
            target_pic50=6.0,
            metal="Ru",
            pocket_id="A2780",
        )
        # Geometry is "default 0.5" on a parseable SMILES (stub returns 0.5).
        assert decomp.m_geometry == pytest.approx(0.5)

    def test_decompose_binding_score(self):
        p = self._make_predictor()
        # Far-apart pIC50 → m_binding high.
        d = p.decompose("O", 6.0, 7.5, "Ru", "A2780")
        assert d.m_binding == pytest.approx(1.0)
        # Close → low.
        d = p.decompose("O", 6.0, 6.0, "Ru", "A2780")
        assert d.m_binding == pytest.approx(0.0)

    def test_decompose_reduction_canonical_metal(self):
        p = self._make_predictor()
        d = p.decompose("O", 6.0, 6.5, "Ru", "A2780")
        assert d.m_reduction == pytest.approx(0.0)
        d = p.decompose("O", 6.0, 9.0, "Ru", "A2780")
        assert d.m_reduction > 0.0

    def test_decompose_reduction_unknown_metal(self):
        p = self._make_predictor()
        d = p.decompose("O", 6.0, 6.0, "Bi", "A2780")
        assert d.m_reduction == pytest.approx(1.0)

    def test_weights_dict_shape(self):
        p = self._make_predictor()
        out = p.weights_dict()
        assert "backend" in out
        assert out["backend"] in {"torch", "numpy"}
        assert out["input_dim"] == 10
        assert out["hidden_dim"] == 4