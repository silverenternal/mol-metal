"""Tests for :mod:`molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow`.

Wf-247-D TODO-126 §3 M1 master facade — single-API orchestrator for
Wasserstein gradient-flow regularisation.

This module exercises the master facade on all three supported
``loss_type`` selectors (transfer / pic50 / gradient_flow) plus the
shared decay gate and the verification builder.  All tests are
**CPU-only** and use small synthetic tensors (batch ≤ 64, dim ≤ 16)
so the full suite runs in < 2 seconds.

Test surface
------------
1. :class:`MasterWassersteinConfig` — dataclass validation (n_mc ≥ 1).
2. :class:`MasterWassersteinResult` — dataclass shape + finite check.
3. ``MasterWassersteinGradientFlow`` — single ``compute`` API,
   dispatches per ``loss_type``:
   - ``"transfer"``  : Mapping[str, Tensor] → scalar mean pairwise W₂.
   - ``"pic50"``     : 1-D pred vs ref → scalar 1-D W₂.
   - ``"gradient_flow"``: 2-D pred vs ref → scalar sliced-W₂.
4. Loss-type selector validation — unknown selector raises.
5. Decay gate — :meth:`MasterWassersteinGradientFlow.decay_ok`
   delegates to the canonical primitive.
6. Backward-compat: existing
   :mod:`molmetal.molmetal_lam.reward.wasserstein_loss` symbols are
   untouched (sanity import pass; identical primitive values).
7. :func:`build_master_verification` — round-trips; all three keys;
   consistent engine tag.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (  # noqa: E402
    ENGINE_TAG,
    SUPPORTED_LOSS_TYPES,
    MasterWassersteinConfig,
    MasterWassersteinGradientFlow,
    MasterWassersteinResult,
    build_master_verification,
)


# ---------------------------------------------------------------------------
# Backward-compat sanity — canonical primitives still import + work
# ---------------------------------------------------------------------------


def test_canonical_primitives_unchanged():
    """The master is additive; canonical APIs must remain importable."""
    from molmetal.molmetal_lam.reward.wasserstein_loss import (
        WassersteinConfig,
        wasserstein_decay_check,
        wasserstein_loss,
        wasserstein_pic50_loss,
        wasserstein_pocket_loss,
        wasserstein_pose_loss,
        wasserstein_transfer_loss,
    )
    from molmetal.molmetal_lam.reward.wasserstein_transfer import (
        WassersteinTransferLoss,
        WassersteinTransferResult,
        pairwise_w2_matrix,
    )

    assert callable(wasserstein_loss)
    assert callable(wasserstein_transfer_loss)
    assert callable(wasserstein_pic50_loss)
    assert callable(wasserstein_pose_loss)
    assert callable(wasserstein_pocket_loss)
    assert callable(wasserstein_decay_check)
    assert callable(WassersteinConfig)
    assert callable(WassersteinTransferLoss)
    assert callable(WassersteinTransferResult)
    assert callable(pairwise_w2_matrix)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_supported_loss_types_is_tuple_of_three(self):
        assert tuple(SUPPORTED_LOSS_TYPES) == ("transfer", "pic50", "gradient_flow")

    def test_engine_tag(self):
        assert ENGINE_TAG == "wasserstein_reg"


# ---------------------------------------------------------------------------
# 1. MasterWassersteinConfig — dataclass validation
# ---------------------------------------------------------------------------


class TestMasterWassersteinConfig:
    def test_defaults(self):
        cfg = MasterWassersteinConfig()
        assert cfg.n_mc == 1000
        assert cfg.seed == 0

    def test_n_mc_must_be_positive(self):
        with pytest.raises(ValueError, match="n_mc"):
            MasterWassersteinConfig(n_mc=0)
        with pytest.raises(ValueError, match="n_mc"):
            MasterWassersteinConfig(n_mc=-1)

    def test_coerces_int(self):
        cfg = MasterWassersteinConfig(n_mc=10.0, seed=2.5)
        assert cfg.n_mc == 10
        assert cfg.seed == 2


# ---------------------------------------------------------------------------
# 2. MasterWassersteinResult — basic shape
# ---------------------------------------------------------------------------


class TestMasterWassersteinResult:
    def test_construct(self):
        r = MasterWassersteinResult(
            loss_value=1.5,
            loss_type="transfer",
        )
        assert r.loss_value == 1.5
        assert r.loss_type == "transfer"
        assert r.engine == ENGINE_TAG
        assert r.finite is True

    def test_default_engine_tag(self):
        r = MasterWassersteinResult(loss_value=0.0, loss_type="pic50")
        assert r.engine == "wasserstein_reg"


# ---------------------------------------------------------------------------
# 3. MasterWassersteinGradientFlow — selectors
# ---------------------------------------------------------------------------


def _seeded_tensors(batch_size: int = 8, dim: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch_size, dim, generator=g)
    b = torch.randn(batch_size, dim, generator=g) + 0.7
    return g, a, b


class TestTransferSelector:
    def test_returns_scalar_finite(self):
        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=64, seed=0),
        )
        g, a, b = _seeded_tensors(seed=0)
        per_metal = {"Ru": a, "Ir": b, "Os": b + 0.2}
        res = master.compute(per_metal, loss_type="transfer")
        assert isinstance(res, MasterWassersteinResult)
        assert res.loss_type == "transfer"
        assert math.isfinite(res.loss_value)
        assert res.extras["n_metals"] == 3
        assert set(res.extras["per_metal_n"].keys()) == {"Ru", "Ir", "Os"}

    def test_unknown_metal_type_raises_typeerror(self):
        master = MasterWassersteinGradientFlow()
        # Pass a Tensor instead of a Mapping — master must raise TypeError.
        with pytest.raises(TypeError, match="Mapping"):
            master.compute(torch.randn(4, 4), loss_type="transfer")

    def test_matches_canonical_primitive(self):
        """Master selector must equal the canonical primitive value."""
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_transfer_loss,
        )

        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=128, seed=42),
        )
        g, a, b = _seeded_tensors(seed=42)
        per_metal = {"Ru": a, "Ir": b, "Os": b + 0.2, "Rh": b - 0.1}
        master_res = master.compute(per_metal, loss_type="transfer")
        primitive_res = wasserstein_transfer_loss(
            per_metal,
            n_mc=128,
            seed=42,
        )
        assert master_res.loss_value == pytest.approx(
            float(primitive_res.item()),
            rel=1e-5,
        )


class TestPic50Selector:
    def test_returns_scalar_finite(self):
        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=64, seed=0),
        )
        g = torch.Generator().manual_seed(0)
        pred = torch.randn(16, generator=g) * 0.5 + 5.0
        ref = torch.randn(32, generator=g) * 0.5 + 5.0
        res = master.compute(pred, ref, loss_type="pic50")
        assert res.loss_type == "pic50"
        assert math.isfinite(res.loss_value)
        assert res.extras["pred_n"] == 16
        assert res.extras["ref_n"] == 32

    def test_raises_when_ref_missing(self):
        master = MasterWassersteinGradientFlow()
        g = torch.Generator().manual_seed(0)
        pred = torch.randn(16, generator=g)
        with pytest.raises(TypeError, match="ref_distribution"):
            master.compute(pred, None, loss_type="pic50")

    def test_matches_canonical_primitive(self):
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_pic50_loss,
        )

        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=128, seed=7),
        )
        g = torch.Generator().manual_seed(7)
        pred = torch.randn(16, generator=g) * 0.4 + 5.0
        ref = torch.randn(24, generator=g) * 0.4 + 5.0
        master_res = master.compute(pred, ref, loss_type="pic50")
        primitive_res = wasserstein_pic50_loss(pred, ref, n_mc=128, seed=7)
        assert master_res.loss_value == pytest.approx(
            float(primitive_res.item()),
            rel=1e-5,
        )


class TestGradientFlowSelector:
    def test_returns_scalar_finite(self):
        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=64, seed=0),
        )
        g, a, b = _seeded_tensors(seed=0)
        res = master.compute(a, b, loss_type="gradient_flow")
        assert res.loss_type == "gradient_flow"
        assert math.isfinite(res.loss_value)
        assert res.extras["pred_n"] == 8
        assert res.extras["ref_n"] == 8

    def test_raises_when_ref_missing(self):
        master = MasterWassersteinGradientFlow()
        a = torch.randn(4, 4)
        with pytest.raises(TypeError, match="ref_distribution"):
            master.compute(a, None, loss_type="gradient_flow")

    def test_raises_when_pred_wrong_type(self):
        master = MasterWassersteinGradientFlow()
        with pytest.raises(TypeError, match="metal_logits"):
            master.compute(
                {"Ru": torch.randn(4, 4)}, torch.randn(4, 4), loss_type="gradient_flow"
            )

    def test_matches_canonical_primitive(self):
        from molmetal.molmetal_lam.reward.wasserstein_loss import wasserstein_loss

        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=128, seed=11),
        )
        g, a, b = _seeded_tensors(seed=11)
        master_res = master.compute(a, b, loss_type="gradient_flow")
        primitive_res = wasserstein_loss(a, b, n_mc=128, seed=11)
        assert master_res.loss_value == pytest.approx(
            float(primitive_res.item()),
            rel=1e-5,
        )

    def test_zero_on_identical_clouds(self):
        """Identical pred/ref should yield a near-zero W₂."""
        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=128, seed=0),
        )
        g = torch.Generator().manual_seed(0)
        a = torch.randn(16, 4, generator=g)
        res = master.compute(a, a, loss_type="gradient_flow")
        assert res.loss_value == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Unknown loss_type raises
# ---------------------------------------------------------------------------


class TestSelectorValidation:
    def test_unknown_loss_type_raises(self):
        master = MasterWassersteinGradientFlow()
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        with pytest.raises(ValueError, match="loss_type"):
            master.compute(a, b, loss_type="bogus")

    def test_default_loss_type_is_transfer(self):
        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=64, seed=0),
        )
        g = torch.Generator().manual_seed(0)
        per_metal = {
            "Ru": torch.randn(8, 4, generator=g),
            "Ir": torch.randn(8, 4, generator=g),
        }
        res_default = master.compute(per_metal)
        res_explicit = master.compute(per_metal, loss_type="transfer")
        assert res_default.loss_type == res_explicit.loss_type == "transfer"


# ---------------------------------------------------------------------------
# 5. Decay gate (delegates to canonical primitive)
# ---------------------------------------------------------------------------


class TestDecayGate:
    def test_decreasing_history_passes(self):
        master = MasterWassersteinGradientFlow()
        for v in [10.0, 9.5, 9.1, 8.8, 8.5, 8.3, 8.1, 7.95, 7.82, 7.7]:
            master.append_history(v)
        assert master.decay_ok(window=10, tol=1e-3) is True

    def test_growing_history_fails(self):
        master = MasterWassersteinGradientFlow()
        for v in [8.0, 9.0, 9.5, 9.8, 10.0]:
            master.append_history(v)
        assert master.decay_ok(window=5, tol=1e-3) is False

    def test_empty_history_passes(self):
        master = MasterWassersteinGradientFlow()
        assert master.decay_ok() is True

    def test_history_property_returns_copy(self):
        master = MasterWassersteinGradientFlow()
        master.append_history(1.0)
        snap = master.history
        snap.append(99.0)
        assert len(master.history) == 1  # unchanged


# ---------------------------------------------------------------------------
# 6. build_master_verification — round-trip
# ---------------------------------------------------------------------------


class TestBuildMasterVerification:
    def test_returns_expected_shape(self):
        v = build_master_verification(seed=0)
        assert "master_wasserstein_gradient_flow" in v
        inner = v["master_wasserstein_gradient_flow"]
        assert inner["engine"] == ENGINE_TAG
        for k in ("transfer", "pic50", "gradient_flow"):
            assert k in inner
            rec = inner[k]
            assert rec["loss_type"] == k
            assert math.isfinite(rec["loss_value"])
            assert rec["engine"] == ENGINE_TAG
            assert rec["finite"] is True

    def test_cpu_only_marker(self):
        v = build_master_verification()
        assert v["master_wasserstein_gradient_flow"]["cpu_only"] is True


# ---------------------------------------------------------------------------
# 7. to_dict — diagnostic snapshot
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_no_compute_required(self):
        master = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=64, seed=3),
        )
        master.append_history(1.0)
        d = master.to_dict()
        assert d["config"]["n_mc"] == 64
        assert d["config"]["seed"] == 3
        assert d["history_len"] == 1
        assert d["history_tail"] == [1.0]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
