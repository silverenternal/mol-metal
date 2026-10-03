"""test_wf250_apply_fixes.py — tests for Wf-250-Apply CPU-only fixes.

====================================================================
Wf-250-Apply (2026-09-28) — verification driver for the 5 CPU-only
fixes shipped from ../../lab/reports/wf250_research/fix_design.md.
====================================================================

This test module covers:

* Fix #1 (M1) :class:`MasterWassersteinGradientFlow` integration into
  the v3 driver via ``--use-master-wass``.
* Fix #2 (A2) :func:`resolve_per_pocket_config` + per-pocket dispatch
  into :func:`compute_vina_min`.
* Fix #3 (B5) :class:`EnhancedMetalHead` flag plumbing into the v3
  driver via ``--enhanced-head`` and ``--metal-attention``.
* Fix #4 (M2) :class:`MasterCatherineClosureBias` integration into the
  v3 driver via ``--use-master-cath`` (ASTSR-friendly
  ``feature_flag`` selector).
* Fix #5 (B4) Sliced-W2 gradient-flow primitive exposed by the master
  facade (Huang & Malik 2024 W2-FE flavor).

All tests are CPU-only and do NOT require RDKit / chemprop / vina
Python bindings (mocked where needed).  Honest framing: these tests
verify **wiring**, not training lift — the projected L5 R² lift in
``fix_design.md §3.4`` is a literature gap, not a measurement here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fix #2 (A2) — per-pocket Vina dispatch
# ---------------------------------------------------------------------------
class TestResolvePerPocketConfig:
    """Fix #2 (A2) — :func:`resolve_per_pocket_config` per-pocket table."""

    def test_known_zn_pocket_uses_ad4(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
            resolve_per_pocket_config,
        )
        cfg = resolve_per_pocket_config("1h36")
        # Eberhardt 2021 — AD4Zn.dat for Zn coordination.
        assert cfg["scoring"] == "ad4"
        assert cfg["scoring_file"] == "AD4Zn.dat"
        assert cfg["exhaustiveness"] == 32
        assert cfg["n_poses"] == 20

    def test_known_hydrophobic_pocket(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
            resolve_per_pocket_config,
        )
        cfg = resolve_per_pocket_config("3eml")
        assert cfg["scoring"] == "vina"
        assert cfg["exhaustiveness"] == 8
        assert cfg["n_poses"] == 9

    def test_known_medium_pocket(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
            resolve_per_pocket_config,
        )
        for pid in ("3ny8", "4rlu"):
            cfg = resolve_per_pocket_config(pid)
            assert cfg["scoring"] == "vina"
            assert cfg["exhaustiveness"] == 16
            assert cfg["n_poses"] == 15

    def test_metal_binding_pocket_uses_ad4(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
            resolve_per_pocket_config,
        )
        cfg = resolve_per_pocket_config("4aaw")
        assert cfg["scoring"] == "ad4"
        assert cfg["scoring_file"] == "AD4Zn.dat"

    def test_unknown_pocket_falls_back_to_default(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
            resolve_per_pocket_config,
        )
        cfg = resolve_per_pocket_config("unknown_pocket_xyz")
        # Default fallback — exhaustiveness=8, n_poses=9, scoring=vina.
        assert cfg["exhaustiveness"] == 8
        assert cfg["n_poses"] == 9
        assert cfg["scoring"] == "vina"

    def test_synthetic_pocket_id_falls_back_to_default(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
            resolve_per_pocket_config,
        )
        # ``synthetic:CHRM1`` is a synthetic id used by metric_hooks —
        # not in the table, should fall back to default.
        cfg = resolve_per_pocket_config("synthetic:CHRM1")
        assert cfg["exhaustiveness"] == 8
        assert cfg["scoring"] == "vina"

    def test_yaml_load_handles_missing_file(self):
        from molmetal.molmetal_lam.sbdd_env import vina_adapter

        with mock.patch.object(vina_adapter, "_VINA_PER_POCKET_YAML_PATH") as m:
            m.exists.return_value = False
            # Re-call — should fall back to DEFAULT_VINA_PER_POCKET.
            cfg = vina_adapter.resolve_per_pocket_config("1h36")
            # Default fallback table mirrors the YAML block.
            assert cfg["exhaustiveness"] == 32
            assert cfg["scoring"] == "ad4"

    def test_default_table_keys_complete(self):
        from molmetal.molmetal_lam.sbdd_env.vina_adapter import (
            DEFAULT_VINA_PER_POCKET,
        )
        # Every key in the table must have the full dict.
        for k, v in DEFAULT_VINA_PER_POCKET.items():
            assert "exhaustiveness" in v
            assert "n_poses" in v
            assert "scoring" in v


class TestComputeVinaMinPerPocketWiring:
    """Fix #2 (A2) — :func:`compute_vina_min` opt-in dispatch."""

    def test_compute_vina_min_signature_has_use_per_pocket_config(self):
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import compute_vina_min
        import inspect

        sig = inspect.signature(compute_vina_min)
        assert "use_per_pocket_config" in sig.parameters
        # Default must be True so existing callers pick up the new
        # dispatch automatically.
        assert sig.parameters["use_per_pocket_config"].default is True

    def test_compute_vina_min_off_keeps_legacy_behavior(self):
        """Backward compat: ``use_per_pocket_config=False`` preserves the
        caller-supplied ``exhaustiveness`` verbatim."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_vina_min,
        )

        captured = {}

        class _StubAdapter:
            def __init__(self, *a, **kw):
                pass

            def setup(self):
                pass

            def dock(self, mol, pocket, cfg):
                # Capture the exhaustiveness passed via cfg.
                captured["exhaustiveness"] = cfg.exhaustiveness
                return []

        # Stub the vina_adapter module attribute (VinaDockingAdapter is
        # imported lazily inside compute_vina_min via
        # ``from molmetal.molmetal_lam.sbdd_env.vina_adapter import ...``
        # so we patch the symbol on that module instead of on metric_hooks).
        with mock.patch(
            "molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
            _StubAdapter,
        ), mock.patch(
            "molmetal.molmetal_lam.sbdd_env.metric_hooks.resolve_pocket",
            return_value=mock.MagicMock(),
        ):
            out = compute_vina_min(
                ["CCO"], pocket_id="1h36",
                exhaustiveness=4,
                use_per_pocket_config=False,
                save=False,
            )
        # Legacy path: exhaustiveness must be 4 (the caller-supplied
        # value), NOT 32 (the per-pocket override).
        assert captured["exhaustiveness"] == 4
        assert out["exhaustiveness"] == 4
        assert out["per_pocket_used"] is False

    def test_compute_vina_min_on_uses_per_pocket_override(self):
        """Opt-in: ``use_per_pocket_config=True`` swaps in the 1h36
        exhaustiveness=32 override."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_vina_min,
        )

        captured = {}

        class _StubAdapter:
            def __init__(self, *a, **kw):
                pass

            def setup(self):
                pass

            def dock(self, mol, pocket, cfg):
                captured["exhaustiveness"] = cfg.exhaustiveness
                return []

        with mock.patch(
            "molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
            _StubAdapter,
        ), mock.patch(
            "molmetal.molmetal_lam.sbdd_env.metric_hooks.resolve_pocket",
            return_value=mock.MagicMock(),
        ):
            out = compute_vina_min(
                ["CCO"], pocket_id="1h36",
                exhaustiveness=4,
                use_per_pocket_config=True,
                save=False,
            )
        # Per-pocket override must be applied: 1h36 → 32.
        assert captured["exhaustiveness"] == 32
        assert out["exhaustiveness"] == 32
        assert out["per_pocket_used"] is True
        assert out["effective_scoring"] == "ad4"

    def test_compute_vina_min_unknown_pocket_falls_back(self):
        """Unknown pocket id → default fallback in per-pocket dispatch."""
        from molmetal.molmetal_lam.sbdd_env.metric_hooks import (
            compute_vina_min,
        )

        captured = {}

        class _StubAdapter:
            def __init__(self, *a, **kw):
                pass

            def setup(self):
                pass

            def dock(self, mol, pocket, cfg):
                captured["exhaustiveness"] = cfg.exhaustiveness
                return []

        with mock.patch(
            "molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
            _StubAdapter,
        ), mock.patch(
            "molmetal.molmetal_lam.sbdd_env.metric_hooks.resolve_pocket",
            return_value=mock.MagicMock(),
        ):
            out = compute_vina_min(
                ["CCO"], pocket_id="totally_unknown_pocket",
                exhaustiveness=4,
                use_per_pocket_config=True,
                save=False,
            )
        # Unknown pocket → default (8).
        assert captured["exhaustiveness"] == 8
        assert out["per_pocket_used"] is True
        assert out["effective_scoring"] == "vina"


# ---------------------------------------------------------------------------
# Fix #1 (M1) — MasterWassersteinGradientFlow into v3 driver
# ---------------------------------------------------------------------------
class TestMasterWassersteinIntegration:
    """Fix #1 (M1) — MasterWassersteinGradientFlow v3 driver wiring."""

    def test_master_wass_module_importable(self):
        from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
            MasterWassersteinGradientFlow,
            MasterWassersteinConfig,
            MasterWassersteinResult,
            SUPPORTED_LOSS_TYPES,
            ENGINE_TAG,
        )
        assert "transfer" in SUPPORTED_LOSS_TYPES
        assert "pic50" in SUPPORTED_LOSS_TYPES
        assert "gradient_flow" in SUPPORTED_LOSS_TYPES
        assert ENGINE_TAG == "wasserstein_reg"

    def test_master_wass_compute_returns_engine_tag(self):
        """compute(loss_type='transfer') carries engine='wasserstein_reg'."""
        import torch

        from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
            MasterWassersteinGradientFlow,
            MasterWassersteinConfig,
        )
        m = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=32, seed=0),
        )
        per_metal = {
            "Ru": torch.randn(8, 4),
            "Ir": torch.randn(8, 4),
        }
        res = m.compute(metal_logits=per_metal, loss_type="transfer")
        assert res.engine == "wasserstein_reg"
        assert res.loss_type == "transfer"
        assert res.finite is True
        assert res.loss_value >= 0.0  # W₂ is non-negative.

    def test_v3_driver_module_has_master_wass_globals(self):
        """v3 driver exposes ``_HAVE_MASTER_WASS`` and master classes."""
        from molmetal.scripts import train_chemprop_v2_v3 as v3mod
        assert hasattr(v3mod, "_HAVE_MASTER_WASS")
        assert hasattr(v3mod, "_MasterWass")
        assert hasattr(v3mod, "_MasterWassConfig")
        # When torch is available (which it is in the dev venv) the
        # master module is importable.
        assert v3mod._HAVE_MASTER_WASS is True

    def test_v3_driver_has_use_master_wass_flag(self):
        """CLI exposes ``--use-master-wass``.

        Verifies the flag is plumbed via the driver source itself
        (string-grep) — calling ``main()`` actually runs the full
        training pipeline which is too expensive for a unit test.
        """
        from pathlib import Path
        src = Path("/home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py").read_text()
        assert '"--use-master-wass"' in src
        assert '"--master-wass-n-mc"' in src
        # The flag handler must read the value into args.use_master_wass.
        assert "args.use_master_wass" in src
        assert "args.master_wass_n_mc" in src


# ---------------------------------------------------------------------------
# Fix #2 (M2) — MasterCatherineClosureBias into v3 driver
# ---------------------------------------------------------------------------
class TestMasterCatherineIntegration:
    """Fix #2 (M2) — MasterCatherineClosureBias v3 driver wiring."""

    def test_master_cath_module_importable(self):
        from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (
            MasterCatherineClosureBias,
            MasterCatherineBiasResult,
            CatherineClosureFeatureFlag,
            SUPPORTED_BIAS_TYPES,
            ENGINE_FEATURE_FLAG,
        )
        assert "feature_flag" in SUPPORTED_BIAS_TYPES
        assert "oversample" in SUPPORTED_BIAS_TYPES
        assert "generation_bias" in SUPPORTED_BIAS_TYPES
        assert ENGINE_FEATURE_FLAG == "closure_catherine_feature_flag"

    def test_master_cath_feature_flag_returns_engine_tag(self):
        """apply(bias_type='feature_flag') carries the right engine tag."""
        from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (
            MasterCatherineClosureBias,
        )
        m = MasterCatherineClosureBias()
        payload = {
            "term_id": "test-001",
            "metal": "Pt",
            "witness_path": ["CuAAC"],
            "smiles": "",
            "n_visited": 1,
            "max_depth_seen": 1,
            "coord_violations": 0,
        }
        res = m.apply(payload, bias_type="feature_flag")
        assert res.engine == "closure_catherine_feature_flag"
        assert res.bias_type == "feature_flag"
        # Single-payload feature_flag returns a FLAT payload (not nested
        # ``feature_flags``); the per-row :class:`CatherineClosureFeatureFlag`
        # is also attached on ``res.feature_flags`` for batch introspection.
        assert res.payload["term_id"] == "test-001"
        assert res.payload["six_of_six_pass"] is True
        assert "per_property" in res.payload
        # The ``feature_flags`` list is on the result dataclass itself.
        assert isinstance(res.feature_flags, list)
        assert len(res.feature_flags) == 1
        assert res.feature_flags[0].six_of_six_pass is True

    def test_v3_driver_module_has_master_cath_globals(self):
        """v3 driver exposes ``_HAVE_MASTER_CATH`` and master class."""
        from molmetal.scripts import train_chemprop_v2_v3 as v3mod
        assert hasattr(v3mod, "_HAVE_MASTER_CATH")
        assert hasattr(v3mod, "_MasterCath")
        assert hasattr(v3mod, "_CATH_BIAS_TYPES")
        assert v3mod._HAVE_MASTER_CATH is True
        # feature_flag must be in the supported types.
        assert "feature_flag" in v3mod._CATH_BIAS_TYPES

    def test_v3_driver_has_use_master_cath_flag(self):
        """CLI exposes ``--use-master-cath`` and ``--master-cath-bias-type``.

        Verifies the flags are plumbed via source string-grep (calling
        ``main()`` runs the full training pipeline — too expensive for
        a unit test).
        """
        from pathlib import Path
        src = Path("/home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py").read_text()
        assert '"--use-master-cath"' in src
        assert '"--master-cath-bias-type"' in src
        assert "args.use_master_cath" in src
        assert "args.master_cath_bias_type" in src


# ---------------------------------------------------------------------------
# Fix #3 (B5) — EnhancedMetalHead v3 driver wiring (verification only)
# ---------------------------------------------------------------------------
class TestEnhancedMetalHeadWiring:
    """Fix #3 (B5) — verify ``--enhanced-head`` and ``--metal-attention``
    flags are wired into :class:`ChempropV2MTModelV3` construction."""

    def test_v3_driver_has_enhanced_head_flag(self):
        """Verify the source contains the enhanced-head flag plumbing."""
        from pathlib import Path
        src = Path("/home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py").read_text()
        assert '"--enhanced-head"' in src
        assert '"--metal-attention"' in src
        # The flag handler must read the value into args.enhanced_head.
        assert "args.enhanced_head" in src
        assert "args.metal_attention" in src

    def test_metal_attention_requires_enhanced_head_guard_present(self):
        """Backward compat guard: --metal-attention alone raises SystemExit.

        Verify the guard exists in the source — invoking main() runs the
        full training pipeline (too expensive for a unit test).
        """
        from pathlib import Path
        src = Path("/home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py").read_text()
        assert "args.metal_attention and not args.enhanced_head" in src

    def test_chemprop_v2_mt_model_v3_has_enhanced_head(self):
        """The :class:`ChempropV2MTModelV3` accepts ``enhanced_head``."""
        import inspect
        from molmetal.models.chemprop_v2_mt_model_v3 import (
            ChempropV2MTModelV3,
            EnhancedMetalHead,
        )
        # Smoke: class exposes the constructor signature.
        sig_params = list(inspect.signature(ChempropV2MTModelV3.__init__).parameters.keys())
        assert "enhanced_head" in sig_params
        assert "use_metal_attention" in sig_params
        # EnhancedMetalHead is a separate class.
        assert EnhancedMetalHead is not None


def inspect_signature_safe(cls_or_func):
    """Return parameter names for ``cls_or_func.__init__`` or the callable."""
    import inspect
    try:
        sig = inspect.signature(cls_or_func.__init__)
    except (TypeError, ValueError):
        sig = inspect.signature(cls_or_func)
    return list(sig.parameters.keys())


# ---------------------------------------------------------------------------
# Fix #5 (B4) — Wasserstein gradient flow loss (Huang & Malik 2024)
# ---------------------------------------------------------------------------
class TestWassersteinGradientFlow:
    """Fix #5 (B4) — sliced-W2 gradient-flow primitive exposed by M1."""

    def test_master_wass_gradient_flow_selector(self):
        """compute(loss_type='gradient_flow') returns a scalar loss."""
        from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
            MasterWassersteinGradientFlow,
            MasterWassersteinConfig,
        )
        import torch

        m = MasterWassersteinGradientFlow(
            config=MasterWassersteinConfig(n_mc=64, seed=0),
        )
        pred = torch.randn(16, 2)
        ref = torch.randn(16, 2) + 0.5
        res = m.compute(
            metal_logits=pred,
            ref_distribution=ref,
            loss_type="gradient_flow",
        )
        assert res.loss_type == "gradient_flow"
        assert res.engine == "wasserstein_reg"
        assert res.finite is True
        # W₂ between shifted distributions must be positive.
        assert res.loss_value > 0.0


# ---------------------------------------------------------------------------
# Smoke: integration test — both M1 and M2 flags set together
# ---------------------------------------------------------------------------
class TestIntegrationSmoke:
    """Combined smoke: --use-master-wass + --use-master-cath + --enhanced-head
    + --use-per-pocket-config.  CPU-only, no RDKit, no actual training."""

    def test_all_flags_together_in_v3_driver(self):
        """All Wf-250 flags are present in the v3 driver source."""
        from pathlib import Path
        src = Path("/home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py").read_text()
        for needle in (
            "--use-master-wass",
            "--master-wass-n-mc",
            "--use-master-cath",
            "--master-cath-bias-type",
            "--enhanced-head",
            "--metal-attention",
            "wf250_use_master_wass",
            "wf250_use_master_cath",
            "_wf250_master_wass_obj",
            "_wf250_master_cath_obj",
            "_wf250_cath_flags",
        ):
            assert needle in src, f"missing: {needle}"

    def test_master_modules_engine_tags_distinct(self):
        """The two master facades carry distinct engine tags so downstream
        honest reporting can attribute contributions."""
        from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
            ENGINE_TAG as W_TAG,
        )
        from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (
            ENGINE_FEATURE_FLAG as C_TAG,
        )
        assert W_TAG != C_TAG
        assert W_TAG == "wasserstein_reg"
        assert C_TAG == "closure_catherine_feature_flag"
