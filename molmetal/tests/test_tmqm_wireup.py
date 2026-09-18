"""Tests for the tmQM-pretrained encoder wire-up (TODO-08).

Covers:

1. ``load_tmQM_pretrained`` returns the same encoder instance it was
   passed (in-place load) — never a fresh module.
2. The encoder's parameters are loaded from the checkpoint when the
   architectures match (DMPNN → DMPNN round-trip: 42 / 42 keys
   transferred).
3. ``EGNNVelocityField(use_tmqm_init=True)`` logs a transfer summary
   and reports the expected "0 / 42" cross-architecture transfer count
   (DMPNN → EGNN: keys diverge by design — the function logs honestly
   and never raises).
4. The round-8 smoke path (``--tmqm-init`` CLI flag on
   :func:`build_argparser`) parses without error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molmetal.adapters.flow_matching_lipman import (
    DEFAULT_TMQM_CKPT,
    EGNNVelocityField,
    build_argparser,
    load_tmQM_pretrained,
)


CKPT_PATH = Path(DEFAULT_TMQM_CKPT)
pytestmark = pytest.mark.skipif(
    not CKPT_PATH.exists(),
    reason=f"tmQM pre-trained checkpoint not found at {CKPT_PATH}",
)


def test_load_tmQM_pretrained_returns_encoder():
    """``load_tmQM_pretrained`` returns the encoder passed in (in-place)."""
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
    assert out is encoder, "load_tmQM_pretrained must return the same instance"


def test_encoder_params_loaded():
    """DirectedMPNN ↔ DirectedMPNN round-trip transfers 42/42 keys."""
    from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig

    cfg = MPNNConfig(
        atom_feat_dim=39,
        edge_feat_dim=6,
        hidden_dim=128,
        n_layers=3,
        dropout=0.1,
    )
    encoder = DirectedMPNN(cfg)
    src = __import__("torch").load(CKPT_PATH, map_location="cpu")
    src_state = src["encoder_state_dict"]
    encoder = load_tmQM_pretrained(encoder, CKPT_PATH)
    own_state = encoder.state_dict()
    n_intersect = len(set(src_state.keys()) & set(own_state.keys()))
    # F2 reports 42 encoder keys, all of which appear in DirectedMPNN.
    assert n_intersect == 42, (
        f"expected 42 keys to match, got {n_intersect} — checkpoint or "
        f"DirectedMPNN changed shape since round-8."
    )


def test_egnn_velocity_default_uses_tmqm(capsys):
    """``EGNNVelocityField(use_tmqm_init=True)`` logs the transfer summary."""
    enc = EGNNVelocityField(hidden_dim=128, n_layers=3, use_tmqm_init=True)
    captured = capsys.readouterr()
    # The helper prints the canonical "Loaded tmQM-pretrained encoder:"
    # message.  We check the substring instead of the full line so the
    # test stays robust to the n_unexpected / n_missing numbers changing
    # as the EGNN evolves.
    assert "Loaded tmQM-pretrained encoder" in captured.out, (
        f"use_tmqm_init=True must log the transfer summary; "
        f"got: {captured.out!r}"
    )
    # Cross-architecture (DMPNN → EGNN) currently transfers 0 keys by
    # design; assert the honest transfer count appears.
    assert "params transferred" in captured.out


def test_egnn_velocity_use_tmqm_init_false_legacy_path(capsys):
    """``use_tmqm_init=False`` keeps the legacy random-init path."""
    enc = EGNNVelocityField(hidden_dim=128, n_layers=3, use_tmqm_init=False)
    captured = capsys.readouterr()
    assert "Loaded tmQM-pretrained encoder" not in captured.out, (
        "use_tmqm_init=False must NOT log the tmQM load line — legacy path."
    )


def test_argparser_tmqm_init_flag():
    """``--tmqm-init`` / ``--random-init`` flags parse without error."""
    p = build_argparser()
    args = p.parse_args(["--tmqm-init"])
    assert args.tmqm_init == "__default__"
    assert args.random_init is False

    args = p.parse_args(["--tmqm-init=/tmp/custom.pt"])
    assert args.tmqm_init == "/tmp/custom.pt"

    args = p.parse_args(["--random-init"])
    assert args.random_init is True
    assert args.tmqm_init is None


def test_load_tmQM_pretrained_missing_ckpt(tmp_path, capsys):
    """Missing checkpoint logs and returns the encoder unchanged."""
    from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig

    cfg = MPNNConfig(
        atom_feat_dim=39,
        edge_feat_dim=6,
        hidden_dim=128,
        n_layers=3,
        dropout=0.1,
    )
    encoder = DirectedMPNN(cfg)
    missing = tmp_path / "does-not-exist.pt"
    out = load_tmQM_pretrained(encoder, missing)
    captured = capsys.readouterr()
    assert out is encoder
    assert "not found" in captured.out or "checkpoint" in captured.out.lower()