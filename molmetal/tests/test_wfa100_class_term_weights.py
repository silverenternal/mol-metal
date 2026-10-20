# SPDX-License-Identifier: Apache-2.0
"""Wf-A100 (2026-10-20) — the class-term weights must be reachable from YAML.

Seven Wf-A23/A44 class-term weights were parameters of ``train_one_epoch``
with no argparse dest and no call-site kwarg.  Their values came from the
function defaults, which meant six of them were **ON at 0.05** and shaping
every run's objective while being invisible to the config surface — and a
comment at the wire-in claimed "opt-in via --fm-ode-term-weight /
--energy-conservation-weight / --sboo3-term-weight", flags that did not exist.

AGENTS.md §5.11: a hyperparameter lives in the YAML.  A knob that only exists
as a Python default is a knob nobody audits.

Declaring the flags changes no default, so every existing run stays bit-exact;
these tests pin both halves: the flag default EQUALS the function default, and
the call site actually forwards the dest.
"""
from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

#: flag dest -> (documented default, train_one_epoch parameter name)
WEIGHTS = {
    "consistency_term_weight": 0.05,
    "so3_term_weight": 0.0,
    "edm_term_weight": 0.05,
    "shortcut_distill_term_weight": 0.05,
    "fm_ode_term_weight": 0.05,
    "energy_conservation_weight": 0.05,
    "sboo3_term_weight": 0.05,
}


def _parse(argv_extra=()):
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    argv = sys.argv
    sys.argv = ["trainer", "--config", os.devnull, *argv_extra]
    try:
        return T.parse_args()
    except SystemExit:
        raise AssertionError("parse_args() exited on the new flags")
    finally:
        sys.argv = argv


def test_every_weight_is_declared_with_the_documented_default():
    args = _parse()
    for dest, default in WEIGHTS.items():
        assert hasattr(args, dest), f"--{dest.replace('_', '-')} is not declared"
        assert float(getattr(args, dest)) == pytest.approx(default), dest


def test_the_flag_default_equals_the_train_one_epoch_default():
    """No drift: the flag and the function must agree on the shipped value.

    If someone changes one and not the other, the run silently changes when
    the call site starts forwarding the flag — this is the assertion that
    catches it.
    """
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T

    sig = inspect.signature(T.train_one_epoch)
    for dest, default in WEIGHTS.items():
        assert dest in sig.parameters, f"train_one_epoch lost the {dest} kwarg"
        assert float(sig.parameters[dest].default) == pytest.approx(default), dest


def test_each_flag_reaches_the_objective():
    """The dest must be forwarded at the single train_one_epoch call site."""
    src = TRAINER.read_text()
    for dest in WEIGHTS:
        pattern = re.compile(
            rf"{re.escape(dest)}\s*=\s*float\(\s*getattr\(\s*args\s*,\s*"
            rf"['\"]{re.escape(dest)}['\"]"
        )
        assert pattern.search(src), (
            f"{dest} is declared but never forwarded to train_one_epoch — "
            "the flag would be a silent no-op (§5.11)"
        )


def test_a_non_default_value_round_trips_through_the_yaml():
    """§5.11 round-trip: a YAML value must win over the argparse default."""
    import tempfile

    cfg = Path(tempfile.mkdtemp()) / "wfa100.yaml"
    cfg.write_text(
        "train:\n"
        "  so3_term_weight: 0.02\n"
        "  shortcut_distill_term_weight: 0.0\n",
        encoding="utf-8",
    )
    args = _parse(["--config", str(cfg)])
    assert float(args.so3_term_weight) == pytest.approx(0.02)
    assert float(args.shortcut_distill_term_weight) == 0.0
