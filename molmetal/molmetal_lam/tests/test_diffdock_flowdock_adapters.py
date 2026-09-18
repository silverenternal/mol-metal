"""Unit tests for the DiffDock-L / FlowDock binding-oracle adapters.

These tests exercise the *adapter plumbing* — they do NOT require the
upstream DiffDock-L or FlowDock backends to be installed.  The whole
point of the adapter layer is to fail soft and let the binding layer
fall back to the fingerprint stub when the heavy backends are missing.
"""

from __future__ import annotations

import os
import sys

import pytest

from molmetal_lam.binding.types import (
    BindingSite,
    BindingTypeCheckResult,
    PROTEASE_GENERIC,
    typecheck,
    _resolve_docking_oracle,
)
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm


# ---------------------------------------------------------------------------
# DiffDockAdapter — unavailable -> AdapterUnavailable
# ---------------------------------------------------------------------------
def test_diffdock_adapter_unavailable_returns_fallback(monkeypatch) -> None:
    """No DiffDock binary -> AdapterUnavailable -> fingerprint fallback."""
    from molmetal_lam.sbdd_env.diffdock_adapter import (
        AdapterUnavailable,
        DiffDockAdapter,
        DockResult,
    )

    # Force every discovery probe to miss.
    monkeypatch.delenv("DIFFDOCK_BIN", raising=False)
    monkeypatch.setattr(
        "molmetal_lam.sbdd_env.diffdock_adapter.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "molmetal_lam.sbdd_env.diffdock_adapter.os.path.isdir",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "molmetal_lam.sbdd_env.diffdock_adapter.os.path.isfile",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *a, **kw: (_ for _ in ()).throw(ImportError(name))
        if name == "diffdock" else __import__(name, *a, **kw),
    )

    adapter = DiffDockAdapter(binary="/nonexistent/diffdock")
    assert adapter.is_available() is False
    with pytest.raises(AdapterUnavailable):
        adapter.dock("CCO", "/tmp/no_such_pocket.pdb")
    # DockResult is the public dataclass.
    dr = DockResult(rmsd_A=1.5, vina_kcal=-8.0, confidence=0.9)
    assert dr.rmsd_A == 1.5 and dr.vina_kcal == -8.0 and dr.confidence == 0.9


# ---------------------------------------------------------------------------
# FlowDockAdapter — unavailable -> AdapterUnavailable
# ---------------------------------------------------------------------------
def test_flowdock_adapter_unavailable_returns_fallback(monkeypatch) -> None:
    """No FlowDock binary -> AdapterUnavailable -> fingerprint fallback."""
    from molmetal_lam.sbdd_env.flowdock_adapter import (
        AdapterUnavailable,
        FlowDockAdapter,
        DEFAULT_ODE_STEPS,
    )

    # Force every discovery probe to miss.
    monkeypatch.delenv("FLOWDOCK_BIN", raising=False)
    monkeypatch.setattr(
        "molmetal_lam.sbdd_env.flowdock_adapter.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "molmetal_lam.sbdd_env.flowdock_adapter.os.path.isdir",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "molmetal_lam.sbdd_env.flowdock_adapter.os.path.isfile",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *a, **kw: (_ for _ in ()).throw(ImportError(name))
        if name == "flowdock" else __import__(name, *a, **kw),
    )

    adapter = FlowDockAdapter(binary="/nonexistent/flowdock")
    assert adapter.is_available() is False
    with pytest.raises(AdapterUnavailable):
        adapter.dock("CCO", "/tmp/no_such_pocket.pdb")
    # CPU-fallback ODE step count is exported as a constant.
    assert DEFAULT_ODE_STEPS == 40


# ---------------------------------------------------------------------------
# typecheck — fingerprint stub still works as fallback
# ---------------------------------------------------------------------------
def test_binding_typecheck_protocol_signature() -> None:
    """typecheck(smiles, binding_site) -> BindingTypeCheckResult, fingerprint stub OK."""
    # Build a tiny molecule via MoleculeClosedTerm; the typecheck
    # must accept it and return a BindingTypeCheckResult.
    mol = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    site = PROTEASE_GENERIC
    result_default = typecheck(mol, site)
    assert isinstance(result_default, BindingTypeCheckResult)
    # Without leaf_oracle_call=True the fingerprint stub is used and
    # the result has no oracle_check field in its details.
    assert "oracle_check" not in result_default.details
    # leaf_oracle_call=True with no oracle available -> fingerprint
    # fallback (the typecheck must NOT raise even with no oracle).
    result_oracle = typecheck(mol, site, leaf_oracle_call=True, oracle=None)
    assert isinstance(result_oracle, BindingTypeCheckResult)
    # When no oracle is reachable, the fingerprint verdict is the
    # final verdict — same outcome as the default path.
    assert result_oracle.success == result_default.success


# ---------------------------------------------------------------------------
# typecheck — leaf_oracle_call=True with a mock oracle
# ---------------------------------------------------------------------------
def test_binding_typecheck_with_mock_oracle() -> None:
    """When a mock oracle is supplied and succeeds, the typecheck ANDs it."""
    from molmetal_lam.sbdd_env.diffdock_adapter import DockResult

    class _MockOracle:
        name = "MockOracle"

        def dock(self, smiles, pocket_pdb, exhaustiveness=16):  # noqa: D401
            return DockResult(rmsd_A=1.0, vina_kcal=-8.5, confidence=0.95)

    # Use a minimal, permissive site (no Lipinski / hbond requirements)
    # so the fingerprint stub passes and the oracle verdict is the
    # binding discriminant.
    site = BindingSite(
        name="MockSite",
        constraints=[],
        geometry_hints={"pdb_path": "/tmp/fake.pdb"},
        description="",
    )
    mol = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    result = typecheck(mol, site, leaf_oracle_call=True, oracle=_MockOracle())
    assert isinstance(result, BindingTypeCheckResult)
    assert "oracle_check" in result.details
    # The mock oracle reports RMSD=1.0Å and Vina=-8.5 -> AND-ed
    # success requires both, so the verdict is True.
    assert result.success is True


def test_binding_typecheck_with_failing_oracle() -> None:
    """A failing oracle AND-ed with the fingerprint stub flips success=False."""
    from molmetal_lam.sbdd_env.diffdock_adapter import DockResult

    class _MockFailingOracle:
        name = "MockFailingOracle"

        def dock(self, smiles, pocket_pdb, exhaustiveness=16):
            return DockResult(rmsd_A=5.0, vina_kcal=-3.0, confidence=0.1)

    site = BindingSite(
        name="MockSite",
        constraints=[],
        geometry_hints={"pdb_path": "/tmp/fake.pdb"},
        description="",
    )
    mol = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    result = typecheck(
        mol, site, leaf_oracle_call=True, oracle=_MockFailingOracle(),
    )
    # The fingerprint stub passes (no constraints violated), but the
    # oracle reports RMSD=5Å and Vina=-3 kcal/mol — both worse than the
    # canonical cutoff, so success is False.
    assert result.success is False
    assert any("oracle" in v for v in result.violated_constraints)


# ---------------------------------------------------------------------------
# _resolve_docking_oracle — returns None when both backends missing
# ---------------------------------------------------------------------------
def test_resolve_docking_oracle_returns_none_when_missing(monkeypatch) -> None:
    """When neither DiffDock nor FlowDock is available, resolve returns None."""
    # Force both adapters to report unavailable.
    from molmetal_lam.sbdd_env import diffdock_adapter, flowdock_adapter

    monkeypatch.setattr(
        diffdock_adapter.DiffDockAdapter, "is_available", lambda self: False,
    )
    monkeypatch.setattr(
        flowdock_adapter.FlowDockAdapter, "is_available", lambda self: False,
    )
    assert _resolve_docking_oracle() is None


# ---------------------------------------------------------------------------
# MCTSProofSearch — leaf_oracle_call_top_k_only flag is exposed
# ---------------------------------------------------------------------------
def test_mcts_leaf_oracle_call_flag_exposed() -> None:
    """MCTSProofSearch exposes leaf_oracle_call_top_k_only=True by default."""
    from molmetal_lam.search_alg.proof_search import MCTSProofSearch

    site = PROTEASE_GENERIC
    # Build a minimal-but-valid MCTSProofSearch (we don't .search() it).
    search_obj = MCTSProofSearch(
        tile_library=[],
        rules={},
        target_predicates=[],
        binding_site=site,
    )
    assert search_obj.leaf_oracle_call_top_k_only is True
    assert search_obj.oracle_top_k == 10
    # And the aggregator carries the flag through.
    agg = search_obj._resolved_reward()
    assert getattr(agg, "_leaf_oracle_call_top_k_only", False) is True
    assert getattr(agg, "_oracle_top_k", 0) == 10
