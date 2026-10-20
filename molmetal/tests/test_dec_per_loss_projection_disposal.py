# SPDX-License-Identifier: Apache-2.0
"""DEC_per_loss_projection_inert - turn the disposal memo into a guard.

The decision record in cfm_todo.json says: "the key MUST BE REMOVED from
DECLARED_INERT_PATHS the first time a model attribute matches the
loss-projection naming convention". That commitment has lived in a comment and
a TODO entry since Wf-A28 step-3 (2026-10-14), which means the only thing that
would notice a `LossProjection` landing is a human reading the comment.

These tests make the trigger mechanical. They fail in BOTH directions:

  * a consumer appears while the key is still INERT  -> remove it, and wire it
  * the key disappears while no consumer exists       -> the reservation was
    dropped by accident and the config key is now unbound

They also assert the negative directly, so the "no consumer exists" half of
the decision is re-verified on every run instead of being assumed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LOSS_DIR = REPO / "molmetal" / "loss"
LOADER = REPO / "molmetal" / "scripts" / "wfA2_config_loader.py"
KEY = "model.per_loss_projection_lr"
DEST = "per-loss-projection-lr"

#: The loss modules the decision names, plus the three added since.  Globbed
#: rather than listed, so a NEW module cannot hide a consumer.
def _loss_modules():
    return sorted(p for p in LOSS_DIR.glob("*.py") if p.name != "__init__.py")


def _inert_paths():
    import molmetal.scripts.wfA2_config_loader as L

    return L.DECLARED_INERT_PATHS


def _projection_classes():
    """Any class whose name reads as a loss projection."""
    out = []
    for p in _loss_modules():
        for m in re.finditer(r"^class\s+(\w*Projection\w*)\b", p.read_text(), re.M):
            out.append((p.name, m.group(1)))
    return out


def _projection_dests():
    """Any argparse dest that would consume the key."""
    out = []
    for p in (REPO / "molmetal" / "scripts").glob("*.py"):
        txt = p.read_text()
        if DEST in txt:
            out.append(p.name)
    return out


def test_the_key_is_still_declared_inert():
    """While no consumer exists the reservation must hold, or the config key
    resolves to nothing and the config-binding gate would have to fail it."""
    assert KEY in _inert_paths(), (
        f"{KEY} is missing from DECLARED_INERT_PATHS. If a consumer now exists, "
        "this is the correct disposal - delete this test. If not, the "
        "reservation was dropped by accident (AGENTS.md 5.11)."
    )


def test_no_loss_projection_class_exists_yet():
    """The negative half of the decision, re-verified rather than assumed."""
    found = _projection_classes()
    assert found == [], (
        f"a projection class landed: {found}. Per DEC_per_loss_projection_inert, "
        f"remove {KEY!r} from DECLARED_INERT_PATHS, wire it to a real dest, and "
        "delete these tests."
    )


def test_no_trainer_consumes_the_key_yet():
    """A dest without a class would be worse than neither: it would pass the
    binding gate while doing nothing."""
    found = _projection_dests()
    assert found == [], (
        f"--{DEST} appeared in {found}. Per DEC_per_loss_projection_inert, "
        f"remove {KEY!r} from DECLARED_INERT_PATHS and wire the dest."
    )


def test_the_two_halves_cannot_disagree():
    """The invariant in one assertion: reserved <-> no consumer."""
    reserved = KEY in _inert_paths()
    consumed = bool(_projection_classes() or _projection_dests())
    assert reserved != consumed, (
        "DEC_per_loss_projection_inert is in an inconsistent state: "
        f"reserved={reserved} consumed={consumed}. Exactly one must hold."
    )
