"""Wf-222 — Click-chemistry building blocks wired into SBDD pipeline.

================================================================
What this test file verifies
================================================================
The new :mod:`molmetal_lam.sbdd_env.click_building_blocks` module
integrates the click-chemistry reaction library
(:mod:`molmetal_lam.reactions.click_reactions`) into the SBDD layer by:

1. Hand-curating a :data:`CLICK_BUILDING_BLOCKS` library — RDKit-
   validated SMILES carrying each of the canonical click handles.
2. Exposing :class:`ClickBuildingBlocks` — a transparent wrapper
   around any :class:`molmetal.ports.MoleculeGenerator` (FLOWR,
   SoftMol, Pocket2Mol, EquiBind, DiffDock …) that *appends* the
   click blocks to the underlying adapter's ``generate()`` output.
3. Exposing :func:`enumerate_click_products` to fire click reactions
   on the building-block library and emit canonical product SMILES.

Acceptance gates (Wf-222):
  G1 — adapter integration: ClickBuildingBlocks wraps an adapter and
       enriches output with click blocks.
  G2 — tests pass: this file passes.
  G3 — smoke runs: the test that enumerates products completes.
  G4 — backward compat: wrapping with ``append_blocks=False`` (or
       never calling :meth:`generate` on the wrapper) leaves the
       underlying adapter's output untouched.

Honest framing
--------------
These tests are CPU-only and use only the *fallback* / stub paths of
the SBDD adapters (FLOWR, Pocket2Mol, SoftMol).  We do not require a
GPU, a checkpoint, or an upstream repo clone.  This is consistent with
the other adapter tests in ``test_softmol_adapter.py``.
"""

from __future__ import annotations

import importlib
from typing import List

import pytest
from rdkit import Chem

from molmetal.domain import Molecule, Pocket
from molmetal.ports import GenerationConfig, MoleculeGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_pocket() -> Pocket:
    import torch

    return Pocket(
        pdb_id="wf222_test",
        coords=torch.zeros(2, 3, dtype=torch.float32),
        atom_types=torch.tensor([6, 7], dtype=torch.long),
        residue_ids=torch.tensor([0, 0], dtype=torch.long),
        chain_ids=torch.tensor([0, 0], dtype=torch.long),
        mask=torch.tensor([True, True]),
        center=torch.zeros(3, dtype=torch.float32),
        radius=6.0,
    )


def _tiny_config(n: int = 5) -> GenerationConfig:
    return GenerationConfig(n_samples=n, seed=42)


# ===========================================================================
# G1 — Library + handle-detection basics
# ===========================================================================
def test_click_building_blocks_library_imports_cleanly() -> None:
    """Module imports, library validates, count is non-empty."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    assert hasattr(mod, "CLICK_BUILDING_BLOCKS")
    assert isinstance(mod.CLICK_BUILDING_BLOCKS, list)
    assert len(mod.CLICK_BUILDING_BLOCKS) >= 16  # at least 4 of each handle
    assert hasattr(mod, "CLICK_BUILDING_BLOCK_COUNT")
    assert mod.CLICK_BUILDING_BLOCK_COUNT == len(mod.CLICK_BUILDING_BLOCKS)


def test_click_building_blocks_library_covers_each_handle() -> None:
    """Every click handle has at least one entry in the library."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    required_handles = {
        "azide",
        "alkyne",
        "thiol",
        "diene",
        "alkene",
        "boronic_acid",
        "bromide",
        "amine",
    }
    present = set()
    for _smi, handles in mod.CLICK_BUILDING_BLOCKS:
        present.update(handles)
    missing = required_handles - present
    assert not missing, f"library missing handles: {missing}"


def test_is_click_block_and_handles_of() -> None:
    """Public SMILES-level helpers correctly flag click handles."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    # An azide SMILES should be flagged as a click block.
    assert mod.is_click_block("CCN=[N+]=[N-]")
    handles = mod.handles_of("CCN=[N+]=[N-]")
    assert "azide" in handles

    # Ethanol (no handles) should NOT be flagged.
    assert not mod.is_click_block("CCO")
    assert mod.handles_of("CCO") == []

    # Malformed SMILES handled gracefully.
    assert not mod.is_click_block("not_a_smiles_!!!")
    assert mod.handles_of("not_a_smiles_!!!") == []


# ===========================================================================
# G1 — Click product enumeration (cartesian product of reactions)
# ===========================================================================
def test_enumerate_click_products_returns_canonical_smiles() -> None:
    """Enumerate click products from the building-block library."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    smiles_pool = [smi for smi, _ in mod.CLICK_BUILDING_BLOCKS]
    products = mod.enumerate_click_products(smiles_pool, max_products=32)
    assert isinstance(products, list)
    assert len(products) >= 1, "click enumeration should fire at least once"
    # Every product must be RDKit-parseable.
    for p in products:
        m = Chem.MolFromSmiles(p)
        assert m is not None, f"product {p!r} failed RDKit parse"
    # Deduplicated.
    assert len(set(products)) == len(products), "products must be deduplicated"


def test_enumerate_click_products_respects_max_products_cap() -> None:
    """The max_products cap is honoured."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    smiles_pool = [smi for smi, _ in mod.CLICK_BUILDING_BLOCKS]
    products = mod.enumerate_click_products(smiles_pool, max_products=4)
    assert len(products) <= 4


def test_enumerate_click_products_deduplicates_on_repeat_calls() -> None:
    """Two calls on the same pool give the same product list."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    smiles_pool = [smi for smi, _ in mod.CLICK_BUILDING_BLOCKS]
    a = mod.enumerate_click_products(smiles_pool, max_products=16)
    b = mod.enumerate_click_products(smiles_pool, max_products=16)
    assert a == b


# ===========================================================================
# G1 — ClickBuildingBlocks wrapper integration
# ===========================================================================
def _stub_adapter(name: str, mols: List[Molecule]) -> MoleculeGenerator:
    """Create a minimal MoleculeGenerator-shaped stub for tests."""

    class _Stub:
        pass

    s = _Stub()
    s.name = name
    s._out = mols

    def _setup(device="cpu"):
        return None

    def _generate(pocket, config):
        return list(s._out)

    def _train_step(pocket, mols):
        return 0.0

    def _get_metadata():
        return {"name": name, "stub": True}

    s.setup = _setup
    s.generate = _generate
    s.train_step = _train_step
    s.get_metadata = _get_metadata
    return s  # type: ignore[return-value]


def test_click_blocks_wrapper_protocol_conformance() -> None:
    """ClickBuildingBlocks exposes name/setup/generate/train_step/get_metadata."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    inner = _stub_adapter("stub_inner", [])
    wrapped = mod.wrap_with_click_blocks(inner)
    assert hasattr(wrapped, "name")
    assert hasattr(wrapped, "setup")
    assert hasattr(wrapped, "generate")
    assert hasattr(wrapped, "train_step")
    assert hasattr(wrapped, "get_metadata")
    assert wrapped.name == "ClickBuildingBlocks_v1"
    md = wrapped.get_metadata()
    assert md["wrapped_adapter"] == "stub_inner"
    assert md["click_block_count"] >= 16


def test_click_blocks_wrapper_appends_click_blocks_to_output() -> None:
    """G1: generate() returns underlying output + click building blocks."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    original = [mod._smiles_to_molecule_safe("CCO")]
    assert original[0] is not None
    inner = _stub_adapter("stub_inner", original)
    wrapped = mod.wrap_with_click_blocks(inner)
    out = wrapped.generate(_tiny_pocket(), _tiny_config(n=2))
    assert isinstance(out, list)
    # Original first.
    assert len(out) >= 1 + len(mod.CLICK_BUILDING_BLOCKS) - 1
    # First molecule IS the original (identity check, since SMILES
    # canonicalisation may differ between the stub path and ours).
    assert out[0] is original[0]
    # Tail of output is all click blocks (RDKit-parseable, with handles).
    tail = out[1:]
    assert all(isinstance(m, Molecule) for m in tail)
    for m in tail:
        assert m.smiles  # non-empty
        # Each appended block should be flagged as click-handle by
        # ``is_click_block`` (since the library carries handles).
        assert mod.is_click_block(m.smiles)


def test_click_blocks_wrapper_handles_empty_underlying_output() -> None:
    """When the underlying adapter returns ``[]`` (fallback SoftMol path),
    the wrapper still emits the click blocks (the documented contract)."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    inner = _stub_adapter("empty_stub", [])
    wrapped = mod.wrap_with_click_blocks(inner)
    out = wrapped.generate(_tiny_pocket(), _tiny_config(n=0))
    assert len(out) >= len(mod.CLICK_BUILDING_BLOCKS)


def test_click_blocks_wrapper_swallows_underlying_exception() -> None:
    """If the underlying adapter raises, the wrapper still emits blocks."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )

    class _Boom:
        name = "boom"

        def setup(self, device="cpu"):
            return None

        def generate(self, pocket, config):
            raise RuntimeError("upstream dead")

        def train_step(self, pocket, mols):
            return 0.0

        def get_metadata(self):
            return {"name": "boom", "stub": True}

    wrapped = mod.wrap_with_click_blocks(_Boom())  # type: ignore[arg-type]
    out = wrapped.generate(_tiny_pocket(), _tiny_config(n=0))
    assert len(out) >= len(mod.CLICK_BUILDING_BLOCKS)


# ===========================================================================
# G4 — Backward compat: passthrough mode + protocol-conformance check
# ===========================================================================
def test_click_blocks_unwrap_returns_original_output() -> None:
    """With append_blocks=False, the wrapper is a transparent passthrough."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    original = [
        mod._smiles_to_molecule_safe("CCO"),
        mod._smiles_to_molecule_safe("c1ccccc1"),
    ]
    inner = _stub_adapter("passthrough", original)
    wrapped = mod.wrap_with_click_blocks(inner, append_blocks=False)
    out = wrapped.generate(_tiny_pocket(), _tiny_config(n=2))
    assert len(out) == len(original)
    # Identity check (not SMILES string comparison, which depends on
    # canonicalisation conventions).
    for a, b in zip(out, original):
        assert a is b


def test_click_blocks_wrapper_does_not_mutate_inner_metadata() -> None:
    """Calling get_metadata() must NOT mutate the inner adapter's metadata."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    inner = _stub_adapter("inner_meta", [])
    md_before = inner.get_metadata()
    wrapped = mod.wrap_with_click_blocks(inner)
    md_after = inner.get_metadata()
    wrapped_md = wrapped.get_metadata()
    # Underlying adapter's metadata is unchanged.
    assert md_before == md_after
    # Wrapper's metadata adds keys.
    assert "wrapped_adapter" in wrapped_md
    assert "click_block_count" in wrapped_md
    # Wrapper's name is its own, not the inner's.
    assert wrapped_md["name"] == "ClickBuildingBlocks_v1"
    assert wrapped_md["wrapped_adapter"] == "inner_meta"


def test_click_blocks_wrapper_train_step_delegates() -> None:
    """train_step() delegates to the inner adapter's train_step()."""
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )

    class _LossStub:
        name = "lossy"

        def setup(self, device="cpu"):
            return None

        def generate(self, pocket, config):
            return []

        def train_step(self, pocket, mols):
            return 0.123

        def get_metadata(self):
            return {"name": "lossy"}

    wrapped = mod.wrap_with_click_blocks(_LossStub())  # type: ignore[arg-type]
    val = wrapped.train_step(_tiny_pocket(), [])
    assert val == pytest.approx(0.123)


# ===========================================================================
# G1 — Wiring with the actual FLOWR/SoftMol reference adapters
# ===========================================================================
def test_click_blocks_wraps_real_flowr_adapter() -> None:
    """The real FLOWRReferenceAdapter can be wrapped (no upstream clone)."""
    try:
        from molmetal.molmetal_lam.sbdd_env.flowr_adapter import (
            FLOWRReferenceAdapter,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"flowr_adapter unavailable: {exc}")
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    adapter = FLOWRReferenceAdapter()
    adapter.setup(device="cpu")  # probe-only
    wrapped = mod.wrap_with_click_blocks(adapter)
    out = wrapped.generate(_tiny_pocket(), _tiny_config(n=2))
    # FLOWR stub returns 2 fallback molecules + click blocks.
    assert len(out) >= 2 + len(mod.CLICK_BUILDING_BLOCKS)
    # The first 2 molecules are the FLOWR stub fallback (smiles derived
    # from CCO / CCN(CC)CC / c1ccccc1 — RDKit canonical form may include
    # explicit Hs after embed_3d).  We check the heavy-atom skeleton: the
    # first molecule's heavy-atom SMILES is one of the three.
    from rdkit import Chem as _Chem

    first_mol = _Chem.MolFromSmiles(out[0].smiles)
    assert first_mol is not None
    heavy_smi = _Chem.MolToSmiles(_Chem.RemoveHs(first_mol))
    assert heavy_smi in {"CCO", "CCN(CC)CC", "c1ccccc1"}


def test_click_blocks_wraps_real_softmol_adapter() -> None:
    """The real SoftMolAdapter can be wrapped (fallback path)."""
    try:
        from molmetal.adapters.softmol_adapter import SoftMolAdapter
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"softmol_adapter unavailable: {exc}")
    mod = importlib.import_module(
        "molmetal.molmetal_lam.sbdd_env.click_building_blocks"
    )
    adapter = SoftMolAdapter()
    adapter.setup(device="cpu")  # may enter fallback mode
    wrapped = mod.wrap_with_click_blocks(adapter)
    # SoftMol fallback returns []; wrapper still emits click blocks.
    out = wrapped.generate(_tiny_pocket(), _tiny_config(n=3))
    assert len(out) >= len(mod.CLICK_BUILDING_BLOCKS)
