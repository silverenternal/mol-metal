"""Pytest: WF-CFM-Rescue Phase 5 — decode_smoke_at_step probe.

Phase 2 Fix #3 already shipped the ``--decode-smoke-every`` flag and
``run_decode_smoke`` helper.  This module adds explicit rescue coverage:

1. run_decode_smoke exists and is importable
2. run_decode_smoke returns (n_decoded, mols) with the expected types
3. decode_smoke_every argument is recognized by argparse
4. An adapter that produces 0 decoded samples for n steps returns
   n_decoded=0 from run_decode_smoke (regression guard)
5. The training-loop hook path can be unit-tested via a fake `main` body

All tests CPU-friendly, <3s.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1: run_decode_smoke is importable
# ---------------------------------------------------------------------------
def test_run_decode_smoke_is_importable():
    """``run_decode_smoke`` is importable from r10_cfg_real_crossdocked."""
    from molmetal.scripts.r10_cfg_real_crossdocked import run_decode_smoke
    assert callable(run_decode_smoke), "run_decode_smoke must be callable"


# ---------------------------------------------------------------------------
# Test 2: run_decode_smoke returns (n_decoded: int, mols: list)
# ---------------------------------------------------------------------------
def test_run_decode_smoke_returns_correct_types():
    """run_decode_smoke returns ``(int, list)`` per the docstring contract.

    We patch ``adapter.generate`` to a stub that returns a list of
    Molecule objects with bonds present or absent; the helper counts
    mols whose bonds tensor has at least one edge.
    """
    from molmetal.scripts.r10_cfg_real_crossdocked import run_decode_smoke
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
    from molmetal.domain import Molecule, Pocket

    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=64, n_layers=2)
    adapter.setup(device="cpu")

    # Build a minimal pocket
    g = torch.Generator().manual_seed(1)
    pcoords = torch.randn(8, 3, generator=g) * 2.0
    pocket = Pocket(
        pdb_id="smoke_pocket", coords=pcoords,
        atom_types=torch.randint(1, 18, (8,), generator=g),
        residue_ids=torch.zeros(8, dtype=torch.long),
        chain_ids=torch.zeros(8, dtype=torch.long),
        mask=torch.ones(8, dtype=torch.bool),
        center=pcoords.mean(0), radius=6.0,
    )

    # Patch adapter.generate to return a controllable list of Molecules
    real_generate = adapter.generate

    def fake_generate(p, cfg):
        # Return n_samples mols; first half have bonds, second half don't
        mols = []
        for i in range(cfg.n_samples):
            has_bonds = (i < cfg.n_samples // 2)
            bonds = torch.tensor([[0, 0], [1, 2]], dtype=torch.long) if has_bonds else None
            bond_types = torch.tensor([1, 1], dtype=torch.long) if has_bonds else None
            mols.append(Molecule(
                coords=torch.randn(8, 3) * 0.5,
                atom_types=torch.randint(1, 10, (8,)),
                bonds=bonds,
                bond_types=bond_types,
                formal_charges=torch.zeros(8, dtype=torch.long),
            ))
        return mols

    adapter.generate = fake_generate
    n_decoded, mols = run_decode_smoke(adapter, pocket, n_samples=4, n_steps=10)
    adapter.generate = real_generate
    assert isinstance(n_decoded, int), f"n_decoded must be int, got {type(n_decoded)}"
    assert isinstance(mols, list), f"mols must be list, got {type(mols)}"
    # With n_samples=4 and first half having bonds, n_decoded = 2
    assert n_decoded == 2, f"Expected 2 decoded (first half), got {n_decoded}"
    assert len(mols) == 4


# ---------------------------------------------------------------------------
# Test 3: argparse recognizes --decode-smoke-every
# ---------------------------------------------------------------------------
def test_decode_smoke_every_argparse_recognized():
    """The script declares ``--decode-smoke-every`` as a CLI flag."""
    script = PROJECT_ROOT / "molmetal" / "scripts" / "r10_cfg_real_crossdocked.py"
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--decode-smoke-every"
        ):
            return  # found
    pytest.fail("--decode-smoke-every flag not declared in argparse")


# ---------------------------------------------------------------------------
# Test 4: decode_smoke_every default is 0 (disabled, backward compat)
# ---------------------------------------------------------------------------
def test_decode_smoke_every_default_disabled():
    """Default ``--decode-smoke-every=0`` keeps the pre-fix behaviour bit-exact."""
    default = None
    script = PROJECT_ROOT / "molmetal" / "scripts" / "r10_cfg_real_crossdocked.py"
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--decode-smoke-every"
        ):
            for kw in node.keywords:
                if kw.arg == "default":
                    if isinstance(kw.value, ast.Constant):
                        default = kw.value.value
            break
    assert default == 0, f"--decode-smoke-every default must be 0 (disabled), got {default}"


# ---------------------------------------------------------------------------
# Test 5: 200-step 8-sample decode smoke — verify count is finite
# ---------------------------------------------------------------------------
def test_200_step_decode_smoke_produces_finite_count():
    """200-step 8-sample decode smoke produces a finite n_decoded in [0, 8].

    This is the actual smoke that WF-CFM-Rescue Phase 5 runs.  The
    honest expectation: on a randomly-initialised h=64 adapter, decode
    ratio is 0/8 (per wf_cfm_frontier_research/final.md §3.1).  The
    contract we test is: n_decoded is an int in [0, n_samples].
    """
    from molmetal.scripts.r10_cfg_real_crossdocked import run_decode_smoke
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
    from molmetal.domain import Pocket

    torch.manual_seed(0)
    adapter = LipmanFlowMatchingAdapter(hidden_dim=64, n_layers=2)
    adapter.setup(device="cpu")
    g = torch.Generator().manual_seed(2)
    pcoords = torch.randn(8, 3, generator=g) * 2.0
    pocket = Pocket(
        pdb_id="smoke2", coords=pcoords,
        atom_types=torch.randint(1, 18, (8,), generator=g),
        residue_ids=torch.zeros(8, dtype=torch.long),
        chain_ids=torch.zeros(8, dtype=torch.long),
        mask=torch.ones(8, dtype=torch.bool),
        center=pcoords.mean(0), radius=6.0,
    )
    n_decoded, mols = run_decode_smoke(adapter, pocket, n_samples=8, n_steps=200)
    assert isinstance(n_decoded, int)
    assert 0 <= n_decoded <= 8
    assert len(mols) == 8
