# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T70 — the bond-candidate trio is inert; the live set is the head's.

WHAT WAS BELIEVED
-----------------
Wf-A69 W1 (2026-10-17) recorded that ``bond_f1 = 0.265`` came from a
train/eval candidate mismatch: the trainer trained on ``truth_only``
(100% positive) while evaluation used a distance-cutoff set (~95%
NO_BOND).  It added ``--bond-candidate-source`` / ``--bond-candidate-cutoff``
/ ``--subsample-neg-ratio`` to fix that, and 60+ YAMLs set them.

WHAT IS TRUE
------------
Those three flags gate ``LipmanFlowMatchingAdapter.train_step``.  This
trainer's model is ``CFMJointModel``, whose bond target comes from
``_bond_target_for_clean_cloud`` over the pair list ``YuelBondDecoderHead``
already emitted — every pair within ``--edge-cutoff`` (3.5 A), labelled
from corpus truth with BOND_NO_BOND elsewhere.  The negative class was
always present.  MEASURED on 400 tmQM clouds: 52,963 candidate pairs,
30.5% positive (metrics/by_metric/wfa113_t70_candidate_sets.json).

So the flags changed nothing, and the recorded root cause is refuted.  These
tests pin both halves: no consumer outside ``parse_args``, and the real
target contains both classes.

Run:
    .venv/bin/python -m pytest molmetal/tests/test_wfa113_t70_dead_bond_candidate_flags.py -q
"""

from __future__ import annotations

import inspect
import sys

import pytest
import torch

from molmetal.scripts import train_cfm_yuelbond_wasserstein as tr

_DEAD = ("bond_candidate_source", "bond_candidate_cutoff", "subsample_neg_ratio")


# ---------------------------------------------------------------------------
# 1. No consumer outside parse_args
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _DEAD)
def test_flag_has_no_consumer_outside_parse_args(name: str) -> None:
    """The dest is referenced only where it is declared and guarded."""
    module_src = inspect.getsource(tr)
    parser_src = inspect.getsource(tr.parse_args)
    assert name in parser_src, (
        f"{name} must still be declared in parse_args so existing YAMLs "
        "resolve to a dest (AGENTS.md §5.11 config-binding gate)"
    )
    outside = module_src.replace(parser_src, "")
    assert name not in outside, (
        f"{name} has a consumer outside parse_args — Wf-A113 T70 recorded it "
        "as inert.  If it was just wired in, delete this test and the INERT "
        "print, and record the new behaviour in todo.json."
    )


# ---------------------------------------------------------------------------
# 2. A non-default value is rejected, not silently ignored
# ---------------------------------------------------------------------------


def _parse(argv):
    old = sys.argv
    sys.argv = ["train_cfm_yuelbond_wasserstein.py", *argv]
    try:
        return tr.parse_args()
    finally:
        sys.argv = old


def test_defaults_parse_and_report_inert(capsys) -> None:
    args = _parse(["--out-dir", "/tmp/wfa113_t70_parse"])
    assert args.bond_candidate_source == "truth_only"
    assert args.bond_candidate_cutoff == 2.4
    assert args.subsample_neg_ratio == 0
    out = capsys.readouterr().out
    assert "INERT" in out
    assert "edge_cutoff" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--bond-candidate-source", "radius_graph"],
        ["--bond-candidate-cutoff", "3.5"],
        ["--subsample-neg-ratio", "4"],
    ],
)
def test_non_default_value_is_rejected(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        _parse(["--out-dir", "/tmp/wfa113_t70_parse", *argv])
    msg = str(exc.value)
    assert "no consumer in this trainer" in msg
    # The message must name the knob that actually works.
    assert "edge_cutoff" in msg and "bond_sample_balance" in msg


# ---------------------------------------------------------------------------
# 3. The live cutoff is --edge-cutoff, and it reaches the head
# ---------------------------------------------------------------------------


def test_live_cutoff_defaults_to_the_decoder_radius() -> None:
    assert tr.DEFAULT_EDGE_CUTOFF == 3.5
    src = inspect.getsource(tr)
    assert "cutoff=args.edge_cutoff" in src, (
        "the head's candidate radius must come from --edge-cutoff"
    )


# ---------------------------------------------------------------------------
# 4. The production bond target carries both classes
# ---------------------------------------------------------------------------


def test_production_bond_target_has_positives_and_negatives() -> None:
    """A cloud with 3 truth bonds and 4 candidate pairs must yield class 0."""
    from molmetal.models.bond_head import AtomCloud

    cloud = AtomCloud(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        ),
        atomic_numbers=torch.tensor([78, 7, 7, 6]),
        bonds=torch.tensor([[0, 0, 1], [1, 2, 3]]),
        bond_types=torch.tensor([1, 1, 4]),
    )
    # Pairs: (0,1) and (0,2) and (1,3) are truth; (2,3) is not.
    ei = (torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 3, 3]))
    logits = torch.zeros(4, 5)
    _, y = tr._bond_target_for_clean_cloud(cloud, precomputed=(logits, ei))
    assert y is not None
    assert sorted(y.tolist()) == [0, 1, 1, 4], (
        f"expected three truth labels plus one NO_BOND, got {y.tolist()}"
    )


def test_measured_positive_rate_is_not_95_percent_no_bond() -> None:
    """The 0.945 NO_BOND prior the flags were designed around is not this cohort."""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "metrics/by_metric/wfa113_t70_candidate_sets.json"
    if not p.is_file():
        pytest.skip("wfa113_t70 artifact not present")
    d = json.loads(p.read_text())
    rate = d["candidate_sets"]["decode_3p5"]["positive_rate"]
    assert 0.25 < rate < 0.40, (
        f"production candidate positive rate {rate:.4f} is outside the "
        "measured 0.24-0.36 band; re-run the probe before trusting the T70 "
        "conclusion"
    )
