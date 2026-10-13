# SPDX-License-Identifier: Apache-2.0
"""The ``class_logit_bias`` operating-point knob on ``bond_decision``.

WHY THIS EXISTS
---------------
BLK-C-M4, measured on 798 real tmQM complexes / 15,851 truth-aromatic bonds
(Wiberg BO 1.3-1.7): the bond head DOES emit ``BOND_AROMATIC`` (precision
0.889) but under-ranks it -- 11,209 of 15,851 (70.7%) collapse to
``BOND_SINGLE``, while single-bond recall is 0.966.  That is a class-4 vs
class-1 RANKING defect, and the smallest thing that can move a ranking
without a retrain is an additive offset on the logits.

Two properties matter, and this file pins both:

1. ``None`` (and an all-zeros vector) is BYTE-IDENTICAL to the historical
   behaviour.  Five call sites pass ``bond_decision`` positionally, so a
   default that changes the arithmetic would silently move every number in
   the repo -- including the metrics JSONs that are cited.
2. The offset moves aromatic and ONLY aromatic.  Index 4 boosts aromatic;
   it must not change any other class's prediction.

RED CHECK
---------
Removing the ``if class_logit_bias is not None:`` block from
``yuelbond_decoder.bond_decision`` makes
``test_bias_moves_aromatic_above_single`` fail (the vector is ignored and the
result is unchanged).  Restoring it returns the suite to green.
"""

from __future__ import annotations

import torch

from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import bond_decision


def _logits(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_default_none_is_byte_identical_to_pre_2026_behaviour():
    """None must build no tensor and run no arithmetic."""
    logits = _logits([1.0, 2.0, 0.0, -1.0, 0.5], [0.0, 0.0, 3.0, 0.0, 1.0])
    assert bond_decision(logits, "argmax") == bond_decision(
        logits, "argmax", 0.5, None
    )
    # And the values are the ones the historical argmax produces.
    assert bond_decision(logits, "argmax", 0.5, None) == [1, 2]


def test_all_zeros_is_a_noop_under_both_rules():
    """An explicitly-supplied zero vector is the same as supplying nothing."""
    logits = _logits([1.0, 2.0, 0.0, -1.0, 0.5], [0.0, 0.0, 3.0, 0.0, 1.0])
    zeros = [0.0] * 5
    assert bond_decision(logits, "argmax", 0.5, zeros) == bond_decision(
        logits, "argmax"
    )
    assert bond_decision(logits, "threshold", 0.5, zeros) == bond_decision(
        logits, "threshold", 0.5
    )


def test_bias_moves_aromatic_above_single():
    """THE lever: single logit wins by 0.5, an aromatic offset of 1.0 flips it.

    Both the un-biased and the biased answers are asserted, so the test
    cannot pass by accident if the default itself were wrong.
    """
    #      none single double triple aromatic
    logits = _logits([0.0, 1.0, 0.0, 0.0, 0.5])
    assert bond_decision(logits, "argmax") == [1], "premise: single must win"
    got = bond_decision(logits, "argmax", 0.5, [0.0, 0.0, 0.0, 0.0, 1.0])
    assert got == [4], f"aromatic offset of 1.0 must flip the class, got {got}"


def test_bias_does_not_disturb_other_classes():
    """Index 4 moves aromatic. No other class's prediction may change."""
    #              none single dbl tri arom
    logits = _logits(
        [5.0, 0.0, 0.0, 0.0, 0.0],   # 0 -> no bond
        [0.0, 3.0, 0.0, 0.0, 0.0],   # 1 -> single
        [0.0, 0.0, 2.0, 0.0, 0.0],   # 2 -> double
        [0.0, 0.0, 0.0, 4.0, 0.0],   # 3 -> triple
        [0.0, 1.0, 0.0, 0.0, 0.5],   # 4 -> single, flips to aromatic
    )
    base = bond_decision(logits, "argmax")
    biased = bond_decision(logits, "argmax", 0.5, [0.0, 0.0, 0.0, 0.0, 1.0])
    assert base == [0, 1, 2, 3, 1]
    assert biased == [0, 1, 2, 3, 4], (
        "a +1.0 offset on class 4 must not move classes 0-3; "
        f"base={base} biased={biased}"
    )


def test_bias_also_applies_under_the_threshold_rule():
    """The offset is applied BEFORE the rule, so it reaches both branches.

    The threshold branch picks the argmax over NON-zero classes; without the
    offset landing first, an aromatic boost would be invisible there.
    """
    logits = _logits([0.0, 1.0, 0.0, 0.0, 0.5])
    assert bond_decision(logits, "threshold", 0.5) == [1]
    got = bond_decision(logits, "threshold", 0.5, [0.0, 0.0, 0.0, 0.0, 1.0])
    assert got == [4], f"threshold rule must see the offset too, got {got}"


def test_wrong_length_bias_raises_instead_of_broadcasting():
    """A short vector would put the aromatic offset on the wrong class."""
    logits = _logits([0.0, 1.0, 0.0, 0.0, 0.5])
    try:
        bond_decision(logits, "argmax", 0.5, [0.0, 0.0, 1.0])
    except ValueError as exc:
        assert "class_logit_bias" in str(exc)
        assert "3" in str(exc) and "5" in str(exc)
    else:
        raise AssertionError(
            "a 3-entry bias against 5 classes must raise, not broadcast"
        )


def test_helper_reads_both_dicts_and_namespaces():
    """MEASURED DEFECT (2026-10-12): the probe hands this helper a plain dict.

    A ``getattr``-only lookup returns the DEFAULT for a mapping --
    ``getattr({"bond_aromatic_logit_bias": 0.5}, ...)`` is always the
    fallback -- so the offset was dropped on the probe path while the
    trainer path worked. The aromatic sweep ran arms 0.0 and 0.5 and
    produced byte-identical SMILES on all 64 clouds because of this.

    Both call styles must resolve, and every absent/zero case must still
    return None so the no-op path stays allocation-free.
    """
    import importlib.util
    import sys as _sys
    from pathlib import Path as _P

    repo = _P(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "_bias_helper_t", str(repo / "molmetal" / "scripts" /
                             "train_cfm_yuelbond_wasserstein.py")
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["_bias_helper_t"] = mod
    spec.loader.exec_module(mod)
    vec = mod._aromatic_bias_vector

    assert vec({"bond_aromatic_logit_bias": 0.5}) == [0.0, 0.0, 0.0, 0.0, 0.5]
    ns = type("A", (), {"bond_aromatic_logit_bias": 1.5})()
    assert vec(ns) == [0.0, 0.0, 0.0, 0.0, 1.5]

    # Every no-op case must still be None, not a zero vector.
    assert vec({"bond_aromatic_logit_bias": 0.0}) is None
    assert vec({"bond_aromatic_logit_bias": None}) is None
    assert vec({}) is None
    assert vec({"hidden_dim": 128}) is None, (
        "the checkpoint args dict has no such key and must not gain one"
    )


def test_negative_bias_suppresses_aromatic():
    """The knob is bidirectional -- a decay arm must be expressible.

    The offset has to exceed the winning margin to change anything: with
    class 4 at 2.0 and every rival at 0.0, a -1.0 offset leaves it at 1.0 and
    still winning. -3.0 is what actually demotes it, and the intermediate
    step is asserted too so the test cannot be satisfied by a clamp.
    """
    logits = _logits([0.0, 0.0, 0.0, 0.0, 2.0])
    assert bond_decision(logits, "argmax") == [4]
    still = bond_decision(logits, "argmax", 0.5, [0.0, 0.0, 0.0, 0.0, -1.0])
    assert still == [4], f"a -1.0 offset cannot beat a 2.0 margin, got {still}"
    got = bond_decision(logits, "argmax", 0.5, [0.0, 0.0, 0.0, 0.0, -3.0])
    assert got == [0], f"a -3.0 aromatic offset must demote it, got {got}"