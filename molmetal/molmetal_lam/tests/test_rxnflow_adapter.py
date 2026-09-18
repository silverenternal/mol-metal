"""Tests for molmetal.adapters.rxnflow_adapter.RxnFlowAdapter.

The adapter is a thin MoleculeGenerator-compatible shim around the cloned
RxnFlow repo.  We test:
  1. Import + class surface matches the MoleculeGenerator protocol
  2. ``is_rxnflow_available()`` returns a bool (correct probe semantics)
  3. Setup is best-effort (no crash if clone present but pip install not run)
  4. Generate returns a list (possibly empty) — never raises
  5. generate fallback is honest: empty list + reason metadata
  6. ``train_step`` is no-op (returns 0.0)
  7. ``get_metadata`` reports coverage honestly (uses_3d=False)
  8. Hidden dim + n_layers kwargs are accepted (Lipman compatibility)
"""

import pytest

from molmetal.adapters.rxnflow_adapter import (
    RxnFlowAdapter,
    is_rxnflow_available,
)
from molmetal.ports import GenerationConfig


def test_class_surface():
    """RxnFlowAdapter exposes the MoleculeGenerator surface."""
    a = RxnFlowAdapter()
    assert a.name == "RxnFlow_v1"
    assert callable(a.setup)
    assert callable(a.generate)
    assert callable(a.train_step)
    assert callable(a.get_metadata)


def test_is_rxnflow_available_returns_bool():
    """Probe returns a bool — never raises even when upstream missing."""
    result = is_rxnflow_available()
    assert isinstance(result, bool)


def test_setup_best_effort_no_crash():
    """setup() must not raise whether or not RxnFlow is importable."""
    a = RxnFlowAdapter(env_dir="/nonexistent/path")
    # Either works (if installed) or falls back cleanly (if not).
    a.setup(device="cpu")
    assert isinstance(a._using_fallback, bool)


def test_generate_returns_list():
    """generate() returns a list, possibly empty (honest fallback)."""
    a = RxnFlowAdapter()
    a.setup(device="cpu")
    cfg = GenerationConfig(n_samples=5, seed=42)
    out = a.generate(pocket=None, config=cfg)
    assert isinstance(out, list)
    assert len(out) <= cfg.n_samples


def test_fallback_returns_empty_list():
    """Without env_dir + checkpoint, generate() returns honest empty list."""
    a = RxnFlowAdapter(env_dir=None, model_path=None)
    a.setup(device="cpu")
    assert a._using_fallback is True
    out = a.generate(pocket=None, config=GenerationConfig(n_samples=3))
    assert out == []


def test_train_step_noop():
    """train_step returns 0.0 (RxnFlow training lives in RxnFlowTrainer)."""
    a = RxnFlowAdapter()
    assert a.train_step(pocket=None, mols=[]) == 0.0


def test_metadata_honest_about_3d():
    """Metadata correctly reports 3D generation as NOT supported."""
    a = RxnFlowAdapter()
    md = a.get_metadata()
    assert md["name"] == "RxnFlow_v1"
    assert md["uses_3d"] is False
    assert md["uses_reaction_templates"] is True
    assert md["uses_pocket_conditioning"] is False  # default
    assert isinstance(md["fallback_active"], bool)  # always bool, default False
    # Paper provenance is recorded for traceability.
    assert "arXiv" in md["paper"]


def test_metadata_pocket_conditional_flag():
    """Metadata reflects pocket_constructor=True constructor flag."""
    a = RxnFlowAdapter(pocket_conditional=True)
    md = a.get_metadata()
    assert md["uses_pocket_conditioning"] is True


def test_lipman_kwarg_compatibility():
    """Adapter accepts the same kwargs as LipmanFlowMatchingAdapter."""
    a = RxnFlowAdapter(
        hidden_dim=64,
        n_layers=2,
        max_atomic_number=54,
        lr=1e-3,
        joint_train=True,    # Lipman-specific kwarg
        bond_loss_weight=0.5,
    )
    # Constructor should not raise; kwargs are surfaced via _extra.
    assert a._hidden_dim == 64
    assert a._n_layers == 2
    assert a._max_atomic_number == 54
    assert a._lr == 1e-3
    # Lipman-specific kwargs should be swallowed silently.
    assert a._extra.get("joint_train") is True
    assert a._extra.get("bond_loss_weight") == 0.5


def test_adapter_is_module_export():
    """Both symbols must be exportable from the adapters package."""
    from molmetal.adapters import rxnflow_adapter  # noqa: F401
    assert hasattr(rxnflow_adapter, "RxnFlowAdapter")
    assert hasattr(rxnflow_adapter, "is_rxnflow_available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])