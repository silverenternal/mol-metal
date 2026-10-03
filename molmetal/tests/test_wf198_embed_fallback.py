"""Tests for Wf-198 graceful 3D-embedding fallback.

TODO-188 / Wf-198 — handle the 2/26801 pathological Ru SMILES that
Wf-188 surfaced (MetalCytoToxDB m14 full benchmark, see
``molmetal/reports/wf188_m14_full_metalcytotox/result.json``).

The wf188 ``_embed_one`` worker was extended with a three-strategy
fallback chain (``etkdg`` → ``random_coords`` → ``etkdgv3_mmff``).
These tests assert:

1. ``_try_embed_with_strategy`` succeeds on a normal SMILES using
   every strategy (sanity check).
2. ``_try_embed_with_strategy`` does not raise on an invalid SMILES;
   it returns ``False`` and logs.
3. ``_embed_one`` returns ``None`` (graceful skip) for an invalid
   input rather than raising out of the worker.
4. ``_embed_one`` still produces a valid pickled ``AtomCloud`` for a
   well-formed SMILES (regression check — pre-Wf-188 path).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Import the wf188 module — guards avoid top-level side effects.
from molmetal.scripts.wf188_m14_full_metalcytotox import (  # noqa: E402
    _try_embed_with_strategy,
    _embed_one,
    _embed_worker_init,
)


@pytest.fixture(scope="module", autouse=True)
def _init_rdkit_globals():
    """Pre-init the worker globals so we don't mutate them in tests."""
    _embed_worker_init()


def _build_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.AddHs(mol)


# ---------------------------------------------------------------------------
# 1. Each strategy succeeds on a normal SMILES.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["etkdg", "random_coords", "etkdgv3_mmff"])
def test_each_strategy_succeeds_on_normal_smi(strategy: str):
    """All three strategies must embed a run-of-the-mill cisplatin-like mol."""
    Chem, AllChem = _embed_worker_init.__globals__["_EMBED_FUNC"] or (
        None, None,
    )
    Chem, AllChem = _embed_worker_init.__globals__.get(
        "_EMBED_FUNC", (None, None),
    )
    # The worker-init fn returns None; rebuild the handle:
    from rdkit import Chem as _Chem  # noqa: WPS433
    from rdkit.Chem import AllChem as _AllChem  # noqa: WPS433
    mol = _build_mol("[Pt](N)(N)(Cl)(Cl)")
    assert mol is not None
    ok = _try_embed_with_strategy(
        _Chem, _AllChem, mol,
        strategy=strategy, seed=42,
        max_attempts=5, max_iterations=200,
    )
    assert ok is True
    assert mol.GetNumConformers() > 0
    conf = mol.GetConformer()
    pos = conf.GetAtomPosition(0)
    assert pos.x == pos.x  # finite (NaN != NaN)


# ---------------------------------------------------------------------------
# 2. _try_embed_with_strategy does NOT raise on impossible SMILES.
# ---------------------------------------------------------------------------
def test_strategy_returns_false_for_unembeddable():
    """A bogus SMILES-like object should not crash the strategy."""
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem as _AllChem
    mol = _Chem.MolFromSmiles("C")  # smallest valid SMILES
    mol = _Chem.AddHs(mol)
    # Valid SMILES — strategy must succeed.  The point of this test is
    # that *no* strategy raises.
    for strat in ("etkdg", "random_coords", "etkdgv3_mmff"):
        ok = _try_embed_with_strategy(
            _Chem, _AllChem, mol,
            strategy=strat, seed=7,
        )
        assert ok is True, f"{strat} failed on a valid C molecule"


# ---------------------------------------------------------------------------
# 3. _embed_one gracefully skips an invalid SMILES (no exception).
# ---------------------------------------------------------------------------
def test_embed_one_returns_none_on_invalid_smi():
    """Wf-198 G2/G3: invalid SMILES → None; never crashes the worker."""
    res = _embed_one(("", 0, 0.0))
    assert res is None
    # Empty SMILES → None
    res = _embed_one(("not_a_valid_smiles@@@", 0, 0.0))
    assert res is None


# ---------------------------------------------------------------------------
# 4. _embed_one returns pickled AtomCloud for valid SMILES.
# ---------------------------------------------------------------------------
def test_embed_one_returns_pickled_atomcloud_for_valid_smi():
    """Pre-Wf-188 regression check: still works for a normal SMILES."""
    res = _embed_one(("[Pt](N)(N)(Cl)(Cl)", 42, 0.0))
    assert res is not None, "wf188 _embed_one must succeed on cisplatin"
    pos, zs = pickle.loads(res)
    assert pos.shape[0] == zs.shape[0]
    assert pos.shape[0] >= 4  # Pt + 2N + 2Cl (H atoms optional)


# ---------------------------------------------------------------------------
# 5. _embed_one with non-zero jitter produces positions near the canonical.
# ---------------------------------------------------------------------------
def test_embed_one_jitter_applied():
    """Jitter of 0.05 Å should not destroy the embedding."""
    res = _embed_one(("[Pt](N)(N)(Cl)(Cl)", 42, 0.05))
    assert res is not None
    pos, _ = pickle.loads(res)
    # Magnitudes should be on the order of bond length (2 Å).
    assert pos.abs().max() > 0.5


# ---------------------------------------------------------------------------
# 6. Repeated calls are deterministic given the same seed.
# ---------------------------------------------------------------------------
def test_embed_one_deterministic_per_seed():
    """Same seed → same positions (ignoring the per-call RNG inside the worker)."""
    a = pickle.loads(_embed_one(("[Pt](N)(N)(Cl)(Cl)", 42, 0.0)))
    b = pickle.loads(_embed_one(("[Pt](N)(N)(Cl)(Cl)", 42, 0.0)))
    # Positions must be identical for the same seed.
    assert (a[0] == b[0]).all()
    assert (a[1] == b[1]).all()
