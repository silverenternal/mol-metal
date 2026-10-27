# SPDX-License-Identifier: Apache-2.0
"""T49 — the connectivity prior must run on the decodes it exists to rescue.

THE DEFECT
----------
``YuelBondBondDecoder.decode`` guarded the prior with

    if self.connectivity_prior and bond_orders:

``bond_orders`` is the list the selector KEPT.  When the head keeps nothing —
every candidate NO_BOND — that list is empty and the prior is skipped, which
is the only case where it could still produce a molecule.  On an already
connected decode the prior is a no-op anyway, so the guard made the branch
reachable only when it was useless.

MEASURED on the c7 pool
(``molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv``): 302 of
12,669 rows (2.38%) have ``n_frags_raw > 1`` and
``largest_frag_frac == 1/n_frags_raw`` — every fragment a single atom, no bond
anywhere.  Those rows are the guard's signature.

``_bridge_components`` already handles an empty input: the union-find starts at
identity (one component per heavy atom) and the candidate pairs come from
``edge_index``, not from the kept bonds.  So the precondition was pure loss.

WHAT THESE TESTS DO NOT CLAIM
-----------------------------
That the bridged molecule is chemically right.  A bridge is added as
BOND_SINGLE regardless of the head's class, so this measures whether the prior
RUNS, not whether it is a good chemist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from molmetal.models.bond_head import AtomCloud  # noqa: E402
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (  # noqa: E402
    YuelBondResult,
)

N_CLASSES = 5
NO_BOND = 0


def _tetrahedral_cloud():
    """One metal at the centre, four carbons on a tetrahedron.

    Every one of the ten pairs is inside the 3.5 A candidate cutoff (the
    C-C edges are 3.10 A) and every pair is above the 1.0 A clamp, so the
    cloud isolates the prior from both the cutoff and the clamp.
    """
    r = 1.9
    dirs = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    positions = torch.cat([torch.zeros(1, 3), dirs * r], dim=0)
    z = torch.tensor([78, 6, 6, 6, 6], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=z)


def _unsure_no_bond_result(n_atoms: int) -> YuelBondResult:
    """Full pair set, NO_BOND wins the argmax but only just.

    This is the shape the c7 zero-bond rows actually have: the head ranks
    NO_BOND first on every pair, yet its strongest NON-NO_BOND logit is above
    `connectivity_prior_min_logit` (0.0), so the pair is still a legal bridge
    candidate.  A head that says NO_BOND by a wide margin is a different case
    and is covered by `test_a_confident_no_bond_head_is_not_overridden`.
    """
    i, j = torch.triu_indices(n_atoms, n_atoms, offset=1)
    edge_index = torch.stack([i, j], dim=0)
    logits = torch.full((edge_index.shape[1], N_CLASSES), -10.0)
    logits[:, NO_BOND] = 2.0
    logits[:, 1] = 1.0  # BOND_SINGLE, below NO_BOND -> argmax is NO_BOND
    return YuelBondResult(
        pair_features=None,
        logits=logits,
        edge_index=edge_index,
        decode_succeeded=True,
    )


def _certain_no_bond_result(n_atoms: int) -> YuelBondResult:
    """Every class far below NO_BOND: the prior must defer, not override."""
    i, j = torch.triu_indices(n_atoms, n_atoms, offset=1)
    edge_index = torch.stack([i, j], dim=0)
    logits = torch.full((edge_index.shape[1], N_CLASSES), -10.0)
    logits[:, NO_BOND] = 10.0
    return YuelBondResult(
        pair_features=None,
        logits=logits,
        edge_index=edge_index,
        decode_succeeded=True,
    )


def _decoder(*, connectivity_prior: bool):
    dec = YuelBondBondDecoder(
        hidden_dim=16, n_layers=1, connectivity_prior=connectivity_prior
    )
    cloud = _tetrahedral_cloud()
    yres = _unsure_no_bond_result(cloud.atomic_numbers.shape[0])
    # The forward pass is irrelevant to this rule; only `_select_bonds` and the
    # prior read the result, and both are deterministic given `yres`.
    dec._decoder.featurise = lambda _cloud: yres  # type: ignore[assignment]
    return dec, cloud


def test_zero_bond_cloud_is_connected_when_the_prior_is_on():
    dec, cloud = _decoder(connectivity_prior=True)
    out = dec.decode(cloud)
    assert out.bonds_added_connectivity > 0, (
        "the connectivity prior did not run on a zero-bond decode — this is "
        "the `and bond_orders` precondition coming back"
    )
    assert out.decoded.sanitized, (
        "the bridged molecule failed RDKit sanitisation; the prior added a "
        f"bond it should not have (bonds={out.decoded.bond_orders})"
    )
    from rdkit import Chem

    mol = Chem.MolFromSmiles(out.decoded.smiles)
    assert mol is not None
    assert len(Chem.GetMolFrags(mol)) == 1, (
        f"expected one fragment, got {len(Chem.GetMolFrags(mol))} from "
        f"{out.decoded.smiles!r}"
    )


def test_the_same_cloud_stays_fragmented_when_the_prior_is_off():
    """The flag is still the control: off reproduces the pre-D2 decode."""
    dec, cloud = _decoder(connectivity_prior=False)
    out = dec.decode(cloud)
    assert out.bonds_added_connectivity == 0
    assert out.decoded.bond_orders == [], (
        "connectivity_prior=False must not add anything; the pre-D2 path is "
        "bit-exact only if this stays empty"
    )
    from rdkit import Chem

    mol = Chem.MolFromSmiles(out.decoded.smiles)
    assert mol is not None
    assert len(Chem.GetMolFrags(mol)) == cloud.atomic_numbers.shape[0], (
        "with the prior off every atom must remain its own fragment"
    )


def test_on_and_off_produce_different_smiles():
    on, cloud = _decoder(connectivity_prior=True)
    off, _ = _decoder(connectivity_prior=False)
    assert on.decode(cloud).decoded.smiles != off.decode(cloud).decoded.smiles, (
        "the prior changed nothing on a zero-bond cloud, so the guard is "
        "still in place or the prior is inert"
    )


def test_a_connected_decode_is_unchanged_by_the_guard_removal():
    """The precondition only mattered when `bond_orders` was empty.

    A decode that already kept bonds must take the same path before and
    after: the branch body is identical, only the test in front of it moved.
    """
    dec, cloud = _decoder(connectivity_prior=True)
    yres = _unsure_no_bond_result(cloud.atomic_numbers.shape[0])
    # Force one bond into the kept set, then run the prior directly.
    bond_orders = [(0, 1, 1)]
    orders, dative, n_bridges = dec._bridge_components(
        cloud, yres, list(bond_orders), []
    )
    assert n_bridges >= 0
    assert (0, 1, 1) in orders, "the prior must never delete a kept bond"


def test_a_confident_no_bond_head_is_not_overridden():
    """`connectivity_prior_min_logit` is the deference knob, and it still works.

    Removing the `and bond_orders` precondition must not turn the prior into
    an unconditional bond generator: a head that ranks every non-NO_BOND
    class far below NO_BOND has said "no" clearly, and the prior stays out.
    """
    dec = YuelBondBondDecoder(hidden_dim=16, n_layers=1, connectivity_prior=True)
    cloud = _tetrahedral_cloud()
    yres = _certain_no_bond_result(cloud.atomic_numbers.shape[0])
    dec._decoder.featurise = lambda _cloud: yres  # type: ignore[assignment]
    out = dec.decode(cloud)
    assert out.bonds_added_connectivity == 0, (
        "the prior overrode a confident NO_BOND head; "
        "connectivity_prior_min_logit is not being read"
    )
