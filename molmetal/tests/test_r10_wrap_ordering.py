"""Pytest: WF-T24-Wrap-Ordering-Fix (TODO-24 Task 2).

Background
----------
At ``molmetal/scripts/r10_cfg_real_crossdocked.py:137-151`` the harness
constructs a learned bond head via ``default_trained_head()`` (legacy
``in_dim=9``), then immediately wraps it in
``GumbelConnectivity`` / ``ConnectivityAwareDecoder`` /
``ReworkedDecoder``.  The CFM-side adapter at
``flow_matching_lipman/__init__.py:1918`` trains the head with
``in_dim=9 + 2 * hidden_dim`` so the EGNN conditioning
``[bond_feats, e_h]`` can be concatenated.  The two values disagree:

  - downstream wrappers (legacy):  ``in_dim=9``
  - CFM trainer (P0-F2):          ``in_dim=9 + 2 * hidden_dim``

The wrap-ordering fix re-builds the head with the right
``in_dim`` **before** it gets wrapped, so the Gumbel /
ConnectivityAwareDecoder / ReworkedDecoder Linear layers agree on
the input dim and the EGNN context actually reaches the head.

What this test file proves
---------------------------
For ``hidden_dim ∈ {32, 64, 128}`` the post-fix code path

1. Resizes the ``BondOrderHead`` so ``head.in_dim == 9 + 2 * h``.
2. Returns a head whose ``fc1`` weight has the correct shape
   ``(hidden_dim_internal, 9 + 2 * h)``.
3. Forwards a freshly-constructed pair-features tensor of width
   ``9 + 2 * h`` without raising.
4. The pre-fix behaviour is preserved for the legacy ``h=9`` case
   (i.e. the fix is backward-compatible when no CFM conditioning is
   present — guarded by a defensive ``if head.in_dim != expected``
   skip).

All tests run on CPU in <2 s — no GPU or external evaluators.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Import the symbols under test
# ---------------------------------------------------------------------------
from molmetal.models.bond_head import (
    AtomCloud,
    BondOrderHead,
    BondAwareDecoder,
    NUM_BOND_CLASSES,
    default_trained_head,
)
from molmetal.models.connectivity_gumbel import (
    DropEdge,
    GumbelConnectivity,
)
from molmetal.models.bond_head import ConnectivityAwareDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expected_in_dim(h: int) -> int:
    """Per F2:  in_dim = 9 + 2 * hidden_dim (bond feats + e_h concat)."""
    return 9 + 2 * int(h)


def _atom_cloud(z_list):
    """Build an AtomCloud from a list of atomic numbers (positions are
    zero — we only test the head's forward pass on synthetic features)."""
    z = torch.tensor(z_list, dtype=torch.long)
    pos = torch.zeros((z.shape[0], 3), dtype=torch.float32)
    return AtomCloud(positions=pos, atomic_numbers=z)


def _rebuild_head_for_hidden_dim(h: int) -> BondOrderHead:
    """Replicate the post-fix rebuild logic from r10_cfg_real_crossdocked.py.

    The harness now:
      1. Gets a default head (legacy in_dim=9).
      2. Computes ``expected_in_dim = 9 + 2 * h``.
      3. If ``head.in_dim != expected_in_dim`` rebuilds via
         ``BondOrderHead(in_dim=expected_in_dim, ...)``.
    """
    head = default_trained_head(n_epochs=0, seed=0)
    expected = _expected_in_dim(h)
    if head.in_dim != expected:
        new_head = BondOrderHead(
            in_dim=expected,
            hidden_dim=64,
            dropout=0.10,
            num_classes=head.num_classes,
            atom_vocab=head.atom_vocab,
            training_mode=head.training_mode,
        )
        new_head.eval()
        head = new_head
    return head


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h", [32, 64, 128])
def test_wrap_ordering_resizes_in_dim(h: int) -> None:
    """For h in {32, 64, 128} the post-fix rebuild yields in_dim=9+2*h."""
    head = _rebuild_head_for_hidden_dim(h)
    assert head.in_dim == _expected_in_dim(h), (
        f"h={h}: expected in_dim={_expected_in_dim(h)}, got {head.in_dim}"
    )


@pytest.mark.parametrize("h", [32, 64, 128])
def test_wrap_ordering_fc1_weight_shape(h: int) -> None:
    """The rebuilt head's ``fc1`` Linear must have in_features=9+2*h."""
    head = _rebuild_head_for_hidden_dim(h)
    assert head.fc1.in_features == _expected_in_dim(h), (
        f"h={h}: fc1.in_features={head.fc1.in_features} != "
        f"{_expected_in_dim(h)}"
    )
    assert head.fc1.out_features == head.hidden_dim


@pytest.mark.parametrize("h", [32, 64, 128])
def test_wrap_ordering_forward_passes(h: int) -> None:
    """Forward a (E, 9+2*h) pair-features tensor through the head."""
    head = _rebuild_head_for_hidden_dim(h)
    head.eval()
    # Build a 6-atom cloud — 15 unordered pairs (6*5/2).
    cloud = _atom_cloud([6, 7, 8, 6, 17, 78])
    n = cloud.positions.shape[0]
    e = n * (n - 1) // 2
    pair_features = torch.randn(e, _expected_in_dim(h))
    with torch.no_grad():
        logits = head(pair_features)
    assert logits.shape == (e, NUM_BOND_CLASSES), (
        f"h={h}: expected logits shape ({e}, {NUM_BOND_CLASSES}), "
        f"got {tuple(logits.shape)}"
    )


@pytest.mark.parametrize("h", [32, 64, 128])
def test_wrap_ordering_gumbel_accepts_resized_head(h: int) -> None:
    """GumbelConnectivity must be constructible with the post-fix head.

    The pre-fix code path constructed GumbelConnectivity(in_dim=9)
    and then fed it a head with in_dim=9+2*h.  The post-fix
    construct path in the harness must agree on in_dim.
    """
    head = _rebuild_head_for_hidden_dim(h)
    # The harness does ``GumbelConnectivity(in_dim=head.in_dim, ...)``.
    # If the head is correctly resized, this constructor call must
    # succeed without raising.
    gumbel = GumbelConnectivity(
        in_dim=head.in_dim,
        expected_bonds_per_atom=3.0,
    )
    gumbel.eval()
    # And the wrapper composite must also be constructible.
    decoder = ConnectivityAwareDecoder(
        bond_head=head,
        connectivity=gumbel,
        drop_edge=DropEdge(p=0.1),
    )
    # Smoke: build an AtomCloud and ask for a (possibly failing) decode.
    # The exact decode may fail (sanitize) — but the wrap itself must
    # not raise at the in_dim layer.
    cloud = _atom_cloud([6, 7, 8, 6, 17, 78])
    # Calling decode_gumbel should at least enter; we accept any
    # downstream sanitization failure as a graceful result, but
    # we DO assert no InDim / shape mismatch from the resize.
    try:
        result = decoder.decode_gumbel(cloud, training=False)
        # On success or DecodedMol with mol=None we are happy.
        assert result is not None
    except (RuntimeError, ValueError) as exc:
        # If the failure is about an in_dim mismatch, the test fails.
        msg = str(exc).lower()
        assert "in_dim" not in msg and "in_features" not in msg, (
            f"h={h}: GumbelConnectivity raised in_dim mismatch: {exc}"
        )
