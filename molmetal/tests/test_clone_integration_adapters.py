"""Smoke tests for the cloned-repo abstract-layer adapters.

Each adapter gets two assertions:

1. The module imports and exposes ``__all__`` with the expected class.
2. The class instance satisfies the relevant :mod:`molmetal.ports`
   protocol by shape (duck-typed attribute check).

These tests are intentionally *fast* and do not pull in the heavy
upstream code paths (FlowDock / FLOWR / REINVENT4 plugins).
"""

from __future__ import annotations

import pytest

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import (
    DockingConfig,
    DockingEngine,
    GenerationConfig,
    MoleculeGenerator,
    PropertyPrediction,
    ScoringFunction,
)
from molmetal_lam.sbdd_env.flowdock_adapter import (
    FlowDockReferenceAdapter,
    is_flowdock_available,
)
from molmetal_lam.sbdd_env.flowr_adapter import (
    FLOWRReferenceAdapter,
    is_flowr_available,
)
from molmetal_lam.sbdd_env.pybind_adapter import (
    Pybind11StubAdapter,
    PybindBindingProtocol,
    is_binding_loaded,
)
from molmetal_lam.sbdd_env.reinvent4_adapter import (
    REINVENT4Adapter,
    discover_reinvent4_tomls,
    is_reinvent4_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_pocket() -> Pocket:
    import torch

    return Pocket(
        pdb_id="test",
        coords=torch.zeros(2, 3, dtype=torch.float32),
        atom_types=torch.tensor([6, 7], dtype=torch.long),
        residue_ids=torch.tensor([0, 0], dtype=torch.long),
        chain_ids=torch.tensor([0, 0], dtype=torch.long),
        mask=torch.tensor([True, True]),
        center=torch.zeros(3, dtype=torch.float32),
        radius=6.0,
    )


def _tiny_mol() -> Molecule:
    return Molecule.from_smiles("CCO")


# ---------------------------------------------------------------------------
# 1. FlowDock
# ---------------------------------------------------------------------------
def test_flowdock_adapter_import_and_protocol() -> None:
    assert FlowDockReferenceAdapter.__name__ == "FlowDockReferenceAdapter"
    assert FlowDockReferenceAdapter.__module__.endswith("flowdock_adapter")
    assert hasattr(FlowDockReferenceAdapter, "name")
    assert hasattr(FlowDockReferenceAdapter, "setup")
    assert hasattr(FlowDockReferenceAdapter, "dock")
    assert hasattr(FlowDockReferenceAdapter, "get_metadata")
    adapter = FlowDockReferenceAdapter()
    adapter.setup()
    meta = adapter.get_metadata()
    assert meta["engine_version"] == "arXiv:2403.04777"
    assert isinstance(adapter, object)
    # Is the cloned repo on disk?
    assert isinstance(is_flowdock_available(), bool)
    # Dock in stub mode returns N Complexes with the right type.
    complexes = adapter.dock(
        molecule=_tiny_mol(),
        pocket=_tiny_pocket(),
        config=DockingConfig(n_poses=3),
    )
    assert len(complexes) == 3
    assert all(isinstance(c, Complex) for c in complexes)
    # Duck-type conformance: required attrs match DockingEngine.
    for attr in ("name", "setup", "dock", "get_metadata"):
        assert hasattr(adapter, attr), f"FlowDockAdapter missing {attr}"
    assert isinstance(adapter, object)  # Protocol is structural; full check below.
    assert callable(getattr(adapter, "dock", None))


# ---------------------------------------------------------------------------
# 2. FLOWR
# ---------------------------------------------------------------------------
def test_flowr_adapter_import_and_protocol() -> None:
    adapter = FLOWRReferenceAdapter()
    adapter.setup()
    assert "FLOWR" in adapter.name
    mols = adapter.generate(
        pocket=_tiny_pocket(),
        config=GenerationConfig(n_samples=4),
    )
    assert len(mols) == 4
    assert all(isinstance(m, Molecule) for m in mols)
    assert adapter.train_step(_tiny_pocket(), mols) == 0.0
    meta = adapter.get_metadata()
    assert meta["engine_version"] == "arXiv:2404.02819"
    # Probe
    assert isinstance(is_flowr_available(), bool)
    # Required attrs match MoleculeGenerator.
    for attr in ("name", "setup", "generate", "train_step", "get_metadata"):
        assert hasattr(adapter, attr), f"FLOWRAdapter missing {attr}"


# ---------------------------------------------------------------------------
# 3. pybind11 stubs
# ---------------------------------------------------------------------------
def test_pybind_stub_import_and_protocol() -> None:
    stub = Pybind11StubAdapter(name="equibind_kernel", version="1.0.0")
    # Protocol attrs
    for attr in ("name", "version", "is_loaded", "setup", "forward", "get_metadata"):
        assert hasattr(stub, attr), f"Pybind11StubAdapter missing {attr}"
    assert stub.is_loaded is False
    stub.setup(device="cuda")
    out = stub.forward(1, 2, 3, foo="bar")
    assert out is None
    assert any(c.startswith("setup") for c in stub.call_log)
    assert any(c.startswith("forward") for c in stub.call_log)
    md = stub.get_metadata()
    assert md["stub"] is True
    # The Protocol class itself is also importable.
    assert PybindBindingProtocol is not None
    # Probe accepts a path and returns bool
    assert isinstance(is_binding_loaded("/nonexistent/path.so"), bool)


# ---------------------------------------------------------------------------
# 4. REINVENT4
# ---------------------------------------------------------------------------
def test_reinvent4_adapter_import_and_protocol() -> None:
    adapter = REINVENT4Adapter()
    adapter.setup()
    assert adapter.is_stub() is True
    # TOML discovery is safe even when the repo is missing.
    tomls = discover_reinvent4_tomls()
    assert isinstance(tomls, list)
    assert isinstance(is_reinvent4_available(), bool)
    # score() accepts the canonical ScoredCandidate shape.
    prop = PropertyPrediction(qed=0.5, sa_score=2.0)
    mol = _tiny_mol()
    candidates: list = [(mol, None, prop)] * 3
    scored = adapter.score(candidates)
    assert len(scored) == 3
    assert all(hasattr(s, "combined_score") for s in scored)
    assert scored[0].rank == 0
    md = adapter.get_metadata()
    assert "REINVENT4" in md["name"]
    # Required attrs match ScoringFunction.
    for attr in ("name", "setup", "score", "get_metadata"):
        assert hasattr(adapter, attr), f"REINVENT4Adapter missing {attr}"


# ---------------------------------------------------------------------------
# 5. Cross-adapter __all__ smoke test
# ---------------------------------------------------------------------------
def test_all_adapters_export_expected_symbols() -> None:
    import molmetal_lam.sbdd_env.flowdock_adapter as fd
    import molmetal_lam.sbdd_env.flowr_adapter as fr
    import molmetal_lam.sbdd_env.pybind_adapter as pb
    import molmetal_lam.sbdd_env.reinvent4_adapter as rv

    assert "FlowDockReferenceAdapter" in fd.__all__
    assert "FLOWRReferenceAdapter" in fr.__all__
    assert "Pybind11StubAdapter" in pb.__all__
    assert "REINVENT4Adapter" in rv.__all__


# ---------------------------------------------------------------------------
# 6. End-to-end: adapters can be instantiated and exercised without crashing
# ---------------------------------------------------------------------------
def test_adapters_instantiate_without_external_state() -> None:
    """All four adapters should be creatable with their default config."""
    adapters = [
        FlowDockReferenceAdapter(),
        FLOWRReferenceAdapter(),
        Pybind11StubAdapter(),
        REINVENT4Adapter(),
    ]
    for a in adapters:
        assert a is not None
        md = a.get_metadata() if hasattr(a, "get_metadata") else None
        # pb stub also exposes get_metadata
        assert md is None or isinstance(md, dict)
