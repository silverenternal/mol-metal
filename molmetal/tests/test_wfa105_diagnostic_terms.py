# SPDX-License-Identifier: Apache-2.0
"""Wf-A105 (2026-10-20) - four loss weights never reached the objective.

Wf-A34-1 removed four duplicate class wrappers from the loss sum (their
inline siblings are already there). They kept being computed, kept being
recorded in _LAST_LOSS_TERMS - whose contract is "the exact tensor that
entered loss" - and kept a flag each, documented as if it shaped training.

MEASURED: --consistency-term-weight 1e-9 and 500 give a bit-identical epoch
(train_loss 5.4288, bond_ce 1.1657, grad_norm 80.749). A 5e11x weight change
that moves the loss by zero is not in the objective. Weight 0.0 differs
(5.0606) only because consistency_loss_fn draws torch.randn_like, so enabling
the branch shifts the global RNG stream.

These tests pin the repair: the four moved to _LAST_DIAGNOSTIC_TERMS, the
side-channel gained a completeness audit, and the flag help tells the truth.
The runtime half (the split actually holds, the audit balances) lives in
test_cfm_loss_gradient_liveness.py, which owns the smoke run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

DIAGNOSTIC = (
    "consistency_class_term",
    "so3_class_term",
    "edm_class_term",
    "shortcut_distill_class_term",
)

FLAGS = (
    "--consistency-term-weight",
    "--so3-term-weight",
    "--edm-term-weight",
    "--shortcut-distill-term-weight",
)


def _parse():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    argv = sys.argv
    sys.argv = ["trainer", "--config", os.devnull]
    try:
        return T.parse_args()
    finally:
        sys.argv = argv


def test_the_two_side_channels_exist_and_are_distinct():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    assert hasattr(T, "_LAST_LOSS_TERMS")
    assert hasattr(T, "_LAST_DIAGNOSTIC_TERMS")
    assert T._LAST_LOSS_TERMS is not T._LAST_DIAGNOSTIC_TERMS


def test_the_audit_pair_exists():
    """Without it, 'the dict lists what entered loss' is an unchecked claim."""
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    assert set(T._LAST_LOSS_AUDIT) == {"total", "sum_recorded"}


def _loss_dict_block(src):
    start = src.index("_LAST_LOSS_TERMS.clear()")
    end = src.index("_LAST_DIAGNOSTIC_TERMS.clear()")
    return src[start:end]


def test_the_four_are_not_declared_in_the_loss_dict():
    """Static half: they must not appear in the _LAST_LOSS_TERMS.update
    literal.  The runtime half is in the liveness guard."""
    block = _loss_dict_block(TRAINER.read_text())
    for name in DIAGNOSTIC:
        assert '"%s":' % name not in block, (
            "%s is back in _LAST_LOSS_TERMS. If it was added to the loss sum, "
            "move it to EXPECTED_LIVE in the liveness guard instead of hiding "
            "it here." % name
        )


def test_the_two_previously_unrecorded_terms_are_now_recorded():
    """bond_aux_mse_term and ring_prior_term were IN the sum but absent from
    the side-channel, so the guard could not see them."""
    block = _loss_dict_block(TRAINER.read_text())
    for name in ("bond_aux_mse_term", "ring_prior_term"):
        assert '"%s":' % name in block, "%s is in the loss sum but unrecorded" % name


def test_every_diagnostic_flag_says_it_is_not_in_the_loss():
    """The help text is the only place a user meets these flags."""
    src = TRAINER.read_text()
    for flag in FLAGS:
        i = src.index('"%s"' % flag)
        block = src[i : i + 1400]
        assert "NOT added to the loss" in block, (
            "%s does not say it is not in the objective. That sentence is the "
            "whole repair: before Wf-A105 the help implied these weights "
            "shaped training." % flag
        )


def test_the_flags_still_parse_at_their_old_defaults():
    """The repair changes no run: defaults are untouched."""
    a = _parse()
    assert float(a.consistency_term_weight) == 0.05
    assert float(a.so3_term_weight) == 0.0
    assert float(a.edm_term_weight) == 0.05
    assert float(a.shortcut_distill_term_weight) == 0.0
