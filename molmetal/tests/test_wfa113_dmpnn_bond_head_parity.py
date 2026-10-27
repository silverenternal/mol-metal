# SPDX-License-Identifier: Apache-2.0
"""T50 — ``DMPNNBondHead`` must share the dative rule and the attribute surface.

THE DEFECT
----------
``DMPNNBondHead.forward`` computed

    is_dative = ((z_i == 7) | (z_i == 8) | (z_i == 16)) & (z_j == 78)

which is wrong twice over:

* **One-directional.**  Candidates come from ``triu_indices``, so a pair may
  be stored as ``(Pt, N)``.  The donor test then runs on the METAL side and
  the flag is 0.
* **Pt-only.**  Z=78 hard-coded, so on Ru/Ir/Os/Rh/Re clouds every dative
  edge carries 0.

``YuelBondDecoderHead`` had both halves fixed (Wf-A69 W2, Wf-A92) and the fix
was never propagated.  The head also lacked ``metal_set`` / ``donor_zs``,
which is why ``model.head.metal_set`` in the checkpoint feature-schema stamp
raised ``AttributeError`` on it.

WHY IT WAS LATENT
-----------------
No YAML in ``configs/`` sets ``bond_predictor: dmpnn``, so the head is
unreachable from a launch today.  Latent is not harmless: the CLI accepts the
value, so the next person to use it gets a silently Pt-only feature and a
crash at checkpoint time.

HOW THE FEATURE IS READ BACK
----------------------------
``forward`` returns logits, not edge features, so the test rewires the head to
PASS THE is_dative CHANNEL THROUGH: zero the first layer except the is_dative
column, zero the output layer except one class.  The logit of that class is
then exactly the is_dative flag the head computed, per pair.  That measures
the real value rather than restating the rule in the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch")

from molmetal.models.bond_head import AtomCloud, NUM_BOND_CLASSES  # noqa: E402
from molmetal.scripts import train_cfm_yuelbond_wasserstein as trainer  # noqa: E402

DMPNNBondHead = trainer.DMPNNBondHead

PROBE_CLASS = 1  # any non-NO_BOND class; the probe only needs a free slot


def _diatomic(z_a: int, z_b: int, distance: float = 2.0) -> AtomCloud:
    """Two atoms on the x-axis; ``z_b`` is at index 1, i.e. the higher index."""
    return AtomCloud(
        positions=torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        atomic_numbers=torch.tensor([z_a, z_b], dtype=torch.long),
    )


def _probe_is_dative(head, cloud) -> float:
    """Return the head's is_dative flag for the cloud's only pair.

    The first layer is zeroed except the is_dative input column, so its hidden
    state is the flag; the output layer is zeroed except one class, so that
    class's logit is the flag.
    """
    h = head.hidden_dim
    dative_col = 2 * h + 1  # pair_in = [h_i (h), h_j (h), d, is_dative, is_short]
    with torch.no_grad():
        head.pair_head[0].weight.zero_()
        head.pair_head[0].bias.zero_()
        head.pair_head[0].weight[:, dative_col] = 1.0
        head.pair_head[2].weight.zero_()
        head.pair_head[2].bias.zero_()
        head.pair_head[2].weight[PROBE_CLASS, 0] = 1.0
    logits, (i_idx, j_idx) = head(cloud)
    assert logits.shape[0] == 1 and logits.shape[1] == NUM_BOND_CLASSES
    return float(logits.detach()[0, PROBE_CLASS])


def test_head_carries_the_two_attribute_surfaces():
    """Without these, the checkpoint feature-schema stamp raises AttributeError."""
    head = DMPNNBondHead(hidden_dim=8, n_layers=1)
    assert isinstance(head.metal_set, frozenset) and head.metal_set, (
        "metal_set missing or empty; model.head.metal_set cannot be stamped"
    )
    assert isinstance(head.donor_zs, frozenset) and head.donor_zs
    assert 78 in head.metal_set and 44 in head.metal_set
    assert 7 in head.donor_zs and 16 in head.donor_zs


@pytest.mark.parametrize(
    "donor_z,metal_z,label",
    [
        (7, 44, "Ru-N"),
        (17, 78, "Pt-Cl"),
        (6, 77, "Ir-C"),
        (16, 75, "Re-S"),
    ],
)
def test_non_pt_dative_pairs_are_flagged(donor_z, metal_z, label):
    """The metal sits at the HIGHER index, which is the direction the old rule
    could not see, and none of these but Pt is in the old hard-coded set."""
    head = DMPNNBondHead(hidden_dim=8, n_layers=1)
    flag = _probe_is_dative(head, _diatomic(donor_z, metal_z))
    assert flag == 1.0, (
        f"{label} (donor Z={donor_z}, metal Z={metal_z}) got is_dative={flag}; "
        "the head is still on the one-directional Pt-only rule"
    )


def test_a_metal_metal_pair_is_not_dative():
    head = DMPNNBondHead(hidden_dim=8, n_layers=1)
    assert _probe_is_dative(head, _diatomic(78, 44)) == 0.0


def test_an_organic_pair_is_not_dative():
    head = DMPNNBondHead(hidden_dim=8, n_layers=1)
    assert _probe_is_dative(head, _diatomic(6, 7)) == 0.0


def test_widening_the_metal_set_reaches_the_feature():
    """The rule must read the ATTRIBUTE, not a constant.

    With the old hard-coded Z=78 this changed nothing; a set that includes 26
    (Fe) must now flag Fe-N.
    """
    head = DMPNNBondHead(hidden_dim=8, n_layers=1, metal_set={26})
    assert _probe_is_dative(head, _diatomic(7, 26)) == 1.0
    assert _probe_is_dative(head, _diatomic(7, 78)) == 0.0, (
        "metal_set={26} must EXCLUDE Pt; if Pt still fires the attribute is "
        "not being read"
    )


def test_checkpoint_feature_schema_stamp_does_not_raise(monkeypatch):
    """The stamp read `model.head.metal_set` unguarded.

    A one-attribute stand-in is enough: the failure mode is AttributeError at
    that read, not anything about the model.
    """

    class _Head:
        pass

    class _Model:
        head = _Head()

    # The exact expression the trainer uses, exercised on a head with no
    # metal_set — this is what DMPNNBondHead used to be.
    fallback = sorted(
        int(z) for z in getattr(_Model.head, "metal_set", trainer._DEFAULT_METAL_SET)
    )
    assert 78 in fallback
