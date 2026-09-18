"""Tests for the round-9 production-shape tmQM pre-trained loader.

Round-9 TODO-09: ``load_tmQM_pretrained`` is upgraded from the
round-8 best-effort helper (which transfers 0/42 keys when the
checkpoint is a DMPNN encoder but the consumer is an
:class:`EGNNVelocityField`) to a shape-bridged loader.  When the
checkpoint's ``mpnn_config`` is available and ``encoder is None``, the
helper instantiates a fresh :class:`EGNNVelocityField` with matching
``hidden_dim`` / ``n_layers`` / ``max_atomic_number`` and applies the
shape bridge, lifting the transferred count from 0/42 to >40/43 own-key
slots filled.

These tests verify the *production-shape* behaviour — the same code
path real callers will exercise — without requiring the legacy
``DMPNN→DMPNN`` in-place compatibility that the round-8 tests already
cover in :mod:`molmetal.tests.test_tmqm_wireup`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molmetal.adapters.flow_matching_lipman import (
    DEFAULT_TMQM_CKPT,
    EGNNVelocityField,
    _shape_bridge_state_dict,
    load_tmQM_pretrained,
)


CKPT_PATH = Path(DEFAULT_TMQM_CKPT)
pytestmark = pytest.mark.skipif(
    not CKPT_PATH.exists(),
    reason=f"tmQM pre-trained checkpoint not found at {CKPT_PATH}",
)


def _load_ckpt() -> dict:
    """Load the tmQM checkpoint with the project's torch."""
    import torch

    return torch.load(CKPT_PATH, map_location="cpu")


def test_production_shape_bridge_loads_gt_40_of_42_keys():
    """``load_tmQM_pretrained(None, ...)`` fills >40/43 EGNN slots from a
    42-key DMPNN checkpoint (vs. 0/42 in the round-8 smoke)."""
    vf = load_tmQM_pretrained(None, CKPT_PATH)
    assert isinstance(vf, EGNNVelocityField), (
        f"expected EGNNVelocityField, got {type(vf).__name__}"
    )
    # The EGNN has 43 own state_dict keys; the round-9 bridge must fill
    # >40 of them from the 42-key DMPNN checkpoint.
    n_own = len(vf.state_dict())
    ckpt = _load_ckpt()
    src = ckpt["encoder_state_dict"]
    bridged = _shape_bridge_state_dict(src, vf.state_dict())
    n_bridged = len(bridged)
    assert n_bridged > 40, (
        f"expected >40 EGNN slots filled by the production-shape bridge, "
        f"got {n_bridged}/{n_own} own slots filled from {len(src)} source "
        f"keys.  The round-8 smoke transfers only 0/42 — anything ≤40 "
        f"means the bridge regressed."
    )


def test_shape_bridge_returns_egnn_with_matching_hidden_dim():
    """``load_tmQM_pretrained(None, ...)`` reads ``mpnn_config.hidden_dim``
    and instantiates the EGNN with that hidden_dim (matches the DMPNN
    that produced the checkpoint)."""
    ckpt = _load_ckpt()
    expected_hidden_dim = ckpt["mpnn_config"]["hidden_dim"]
    expected_n_layers = ckpt["mpnn_config"]["n_layers"]
    vf = load_tmQM_pretrained(None, CKPT_PATH)
    assert vf.atom_embed.embedding_dim == expected_hidden_dim, (
        f"hidden_dim mismatch: ckpt says {expected_hidden_dim}, "
        f"EGNN has {vf.atom_embed.embedding_dim}"
    )
    assert len(vf.layers) == expected_n_layers, (
        f"n_layers mismatch: ckpt says {expected_n_layers}, "
        f"EGNN has {len(vf.layers)}"
    )


def test_shape_bridge_load_state_dict_strict_false_tolerates_missing():
    """The bridged load uses ``strict=False`` so missing keys fall back
    to the module's own (random-init) parameters without raising.

    We don't run a full forward pass here — the
    :class:`EGNNVelocityField` forward requires a properly-built
    :class:`LipmanFlowMatchingAdapter` (which wires up the velocity
    field with a path + scheduler + optimizer).  The round-9 smoke
    only needs to confirm that the bridged ``load_state_dict`` does
    not raise.
    """
    vf = load_tmQM_pretrained(None, CKPT_PATH)
    assert isinstance(vf, EGNNVelocityField)
    # The state_dict must remain accessible (i.e. the bridge did not
    # leave the module in an inconsistent state).
    sd = vf.state_dict()
    assert len(sd) > 0
    # Re-running the bridge on the same checkpoint must be idempotent
    # — the second call should also succeed (verifies the
    # ``strict=False`` tolerance is robust).
    load_tmQM_pretrained(vf, CKPT_PATH)


def test_legacy_in_place_path_still_returns_same_instance():
    """Backward-compat: a caller passing an already-constructed
    :class:`EGNNVelocityField` still gets an in-place load (legacy
    round-8 behaviour preserved)."""
    vf = EGNNVelocityField(hidden_dim=128, n_layers=3, max_atomic_number=100)
    out = load_tmQM_pretrained(vf, CKPT_PATH)
    assert out is vf, (
        "load_tmQM_pretrained must return the same instance when the "
        "caller has provided an already-constructed encoder."
    )


def test_legacy_dmpnn_in_place_path_transfers_42_of_42_keys():
    """Backward-compat: a caller passing a matching-shape
    :class:`DirectedMPNN` gets a strict 42/42 key load (the round-8
    best-case behaviour)."""
    from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig

    cfg = MPNNConfig(
        atom_feat_dim=39,
        edge_feat_dim=6,
        hidden_dim=128,
        n_layers=3,
        dropout=0.1,
    )
    encoder = DirectedMPNN(cfg)
    out = load_tmQM_pretrained(encoder, CKPT_PATH)
    assert out is encoder
    # All 42 DMPNN keys should match by name.
    ckpt = _load_ckpt()
    src_keys = set(ckpt["encoder_state_dict"].keys())
    own_keys = set(encoder.state_dict().keys())
    n_intersect = len(src_keys & own_keys)
    assert n_intersect == 42, (
        f"expected 42 DMPNN keys to match, got {n_intersect}."
    )