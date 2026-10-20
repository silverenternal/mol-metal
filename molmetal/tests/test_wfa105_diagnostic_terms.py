# SPDX-License-Identifier: Apache-2.0
"""Wf-A105/A106 (2026-10-20) - four loss weights never reached the objective,
and the knobs are now gone rather than merely labelled.

Wf-A34-1 removed four duplicate class wrappers (consistency / so3 / edm /
shortcut) from the loss sum; their inline siblings were already there.  The
wrappers kept being computed and recorded, and each kept a flag.

Wf-A105 measured what that meant: --consistency-term-weight 1e-9 and 500 gave a
bit-identical epoch (train_loss 5.4288, bond_ce 1.1657, grad_norm 80.749), so
the value could not be in the objective; weight 0.0 differed only because
consistency_loss_fn draws torch.randn_like, shifting the global RNG stream.

Wf-A106 deleted the wrappers, their flags and their two extra forwards per
step.  These tests pin the deletion and keep the audit invariant that caught it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

REMOVED_FLAGS = (
    "consistency_term_weight",
    "so3_term_weight",
    "edm_term_weight",
    "shortcut_distill_term_weight",
    "shortcut_distill_perturb_sigma",
)

REMOVED_CLASSES = (
    "ConsistencyTerm(",
    "SO3Term(",
    "EDMTerm(",
    "ShortcutDistillLoss(",
)

KEPT = (
    "fm_ode_term_weight",
    "energy_conservation_weight",
    "sboo3_term_weight",
)


def _parse():
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    argv = sys.argv
    sys.argv = ["trainer", "--config", os.devnull]
    try:
        return T.parse_args()
    finally:
        sys.argv = argv


def test_the_five_flags_are_gone():
    a = _parse()
    for dest in REMOVED_FLAGS:
        assert not hasattr(a, dest), (
            dest + " is back. It weighted a tensor that is not summed into "
            "loss; re-adding it without a sum entry recreates 6.148/6.149."
        )


def test_the_four_wrappers_are_gone_from_the_trainer():
    src = TRAINER.read_text()
    for cls in REMOVED_CLASSES:
        assert cls not in src, (
            cls + " reappeared in the trainer. It is a duplicate of an inline "
            "path; if it is genuinely needed, add it to the loss sum AND to "
            "EXPECTED_LIVE in test_cfm_loss_gradient_liveness.py."
        )


def test_the_diagnostic_side_channel_is_gone_too():
    """An empty second dict would be a tombstone, not a fix."""
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    assert not hasattr(T, "_LAST_DIAGNOSTIC_TERMS")


def test_the_three_real_weights_survive():
    a = _parse()
    for dest in KEPT:
        assert hasattr(a, dest), dest + " was deleted by mistake"


def test_the_audit_pair_survives():
    """The invariant that caught the defect must outlive the repair."""
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    assert set(T._LAST_LOSS_AUDIT) == {"total", "sum_recorded"}


def test_the_three_kept_terms_are_in_the_loss_sum():
    """The kept wrappers are the ones with no inline duplicate, so they must
    actually appear in the sum expression."""
    src = TRAINER.read_text()
    # the assignment, not the "loss = (...)" inside the explanatory comment
    start = src.index("\n        loss = (\n")
    end = src.index("\n        )\n", start)
    block = src[start:end]
    for name in ("fm_ode_class_term", "energy_class_term", "sboo3_class_term"):
        assert "+ " + name in block, (
            name + " is computed but not summed - the exact defect this wave "
            "removed, now applied to a term that is supposed to stay"
        )
