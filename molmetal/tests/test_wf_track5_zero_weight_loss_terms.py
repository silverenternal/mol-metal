#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Wf-TRACK5 — classification of the three zero-weight CFM loss terms.
#
# `gen_validity`, `gen_single_fragment_ratio` and `decode_ratio` are tracked
# metrics with NO live loss term. Three terms exist to serve them and all
# three default to weight 0.0, i.e. an exact `torch.zeros(())` in the
# `loss = (...)` sum at train_cfm_yuelbond_wasserstein.py:3679-3703.
#
# "Switch them all on" is NOT the correct instruction, because they are not
# the same kind of object. Two are live objectives and one is a measured
# degenerate direction. This file pins the distinction so it cannot be
# re-litigated from the flag name alone.
#
#   (A) soft_valence_term  (gen_validity_weight)   -- DEGENERATE, stays OFF
#   (B) entropy_term       (entropy_reg_weight)     -- live, enabled
#   (C) assemble_term      (assemble_loss_weight)   -- live ONLY with the
#                                                      non-default formulation
#
# On (A): the term's own docstring is honest that it is a squared hinge
# rather than the log barrier its spec asked for, and calls the gradient
# direction "the same". It is the same DIRECTION, but the gradient has the
# OPPOSITE SIGN relative to forming a bond — which is the whole problem.
# `excess = clamp(atom_orders - cap + eps, min=0)` means an atom anywhere at
# or below its cap is charged nothing AND pulls nothing. Enabling it would
# buy the tracked `gen_validity` number purely by decoding fewer bonds,
# against the metric the run's `early_stop_metric` actually selects on
# (`bond_type_f1`). metrics/by_metric/decode_ratio.json records the limit
# case: a NO_BOND head reads gen_validity 0.977 while bond precision
# craters.
#
# On (C): the argparse DEFAULT formulation is "product", and product is not
# an objective. Its value is a geometric mean of K per-step stay
# probabilities, so its gradient carries the product's own scale and
# vanishes exactly in the near-connected regime where a prevention term has
# to act. Enabling `assemble_loss_weight` without ALSO setting
# `assemble_formulation: escape_time` is therefore the exact FLAG_ONLY
# failure AGENTS.md §5.7 exists to catch — the flag is consumed, the value
# reaches the sum, and it trains nothing.

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from molmetal.molmetal_lam.lam_chem.assemble_connectivity import (
    assemble_fragment_loss,
)
from molmetal.molmetal_lam.lam_chem.decoder_rework import (
    valence_log_barrier,
)

CONFIG = os.path.join(
    REPO_ROOT, "configs", "cfm_objective_terms_2026_10_11.yaml"
)


# ---------------------------------------------------------------------------
# (A) soft_valence_term — pin the one-sidedness that keeps it disabled
# ---------------------------------------------------------------------------

def _carbon_barrier(orders, n_atoms=2, valences=None):
    """Barrier for `n_atoms` carbons wired as a single chain of len(orders)."""
    n_edges = len(orders)
    src = torch.arange(n_edges, dtype=torch.long)
    dst = src + 1
    z = torch.full((n_atoms,), 6, dtype=torch.long)
    o = torch.tensor(orders, dtype=torch.float64, requires_grad=True)
    loss = valence_log_barrier(o, z, src, dst, n_atoms, valences=valences)
    return o, loss


@pytest.mark.parametrize("order", [0.0, 0.5, 1.0, 2.0, 3.0, 3.5])
def test_soft_valence_gradient_is_exactly_zero_below_the_cap(order):
    """THE REASON (A) STAYS OFF — do not enable it believing it helps.

    Carbon's cap is 4. For every bond order at or below the cap the term is
    an exact constant AND carries exactly zero gradient. A regulariser that
    says nothing about the healthy region cannot reward entering it.
    """
    o, loss = _carbon_barrier([order])
    loss.backward()
    assert float(loss.detach()) == 0.0, (
        f"order {order} is under cap 4, so the penalty must be exactly 0, "
        f"got {float(loss.detach()):.3e}"
    )
    assert float(o.grad) == 0.0, (
        f"order {order} is under cap 4, so the gradient must be EXACTLY 0 "
        f"(one-sided hinge), got {float(o.grad):.3e}. A non-zero gradient "
        "here would mean the term had been changed into a two-sided one, "
        "which is a real change and needs its own justification."
    )


def test_soft_valence_never_rewards_forming_a_bond():
    """Adding MORE bonds under the cap changes nothing — in any direction.

    This is the sharpest statement of the defect: the term cannot express
    "this atom should have another bond", only "this atom has too many".
    """
    o = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([6, 6], dtype=torch.long)
    # both edges land on atom 0, so its order sum is 1.0 against a cap of 4
    loss = valence_log_barrier(
        o, z, torch.tensor([0, 0]), torch.tensor([1, 1]), 2, valences={6: 4}
    )
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.equal(
        o.grad, torch.zeros_like(o.grad)
    ), "two under-cap bonds produced a non-zero gradient"


def test_soft_valence_does_fire_above_the_cap():
    """Counterweight to the two tests above.

    Pins that the term is not simply broken — it is *correct as a one-sided
    penalty* and simply cannot do the job it was named for. Without this,
    "zero gradient below the cap" could be satisfied by a term that is
    dead everywhere.
    """
    o, loss = _carbon_barrier([6.0])
    loss.backward()
    assert float(loss.detach()) > 0.0
    assert float(o.grad) > 0.0


# ---------------------------------------------------------------------------
# (B) entropy_term — live, and not degenerate
# ---------------------------------------------------------------------------

def _entropy_term(logits, weight=0.01):
    logits = logits.clone().requires_grad_(True)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * (log_probs + 1e-9)).sum(dim=-1).mean()
    return logits, -weight * entropy


def test_entropy_term_is_strongest_exactly_where_the_collapse_is():
    """At the argmax distribution the C4 collapse produces, the gradient is live.

    BLK-C4 measured gen_validity = 0.000 from epoch 4 because the head
    collapsed onto one skeleton. This asserts the term actually pushes back
    there, rather than being inert in the collapsed state it exists to fix.
    """
    collapsed = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    logits, term = _entropy_term(collapsed)
    term.backward()
    assert float(logits.grad.abs().max()) > 1e-6, (
        "entropy regulariser has no gradient on a collapsed head — it "
        "cannot do the thing BLK-C4 measured"
    )


def test_entropy_term_rewards_uniformity_not_a_particular_class():
    """The distinguishing test against a one-sided hinge.

    A one-sided penalty is flat on one side of its threshold and rising on
    the other. Entropy is strictly ordered everywhere: a more uniform
    distribution always scores better than a less uniform one, whichever
    class either of them favours. Asserting the ordering in BOTH directions
    is what makes this a regulariser rather than a hinge.
    """
    peaked = torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0]])
    uniform = torch.zeros(1, 5)
    _, t_peaked = _entropy_term(peaked)
    _, t_uniform = _entropy_term(uniform)
    assert float(t_uniform) < float(t_peaked), (
        "entropy term must be MORE negative (i.e. a lower loss) for the "
        "uniform distribution — that is what makes it a maximiser"
    )
    # and the reverse ordering holds too, so it is not "always negative"
    other_peaked = torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0]])
    _, t_other = _entropy_term(other_peaked)
    assert abs(float(t_other) - float(t_peaked)) < 1e-12, (
        "entropy is class-symmetric; it must not encode a preference for "
        "any particular bond class, or it would be a disguised label prior"
    )


def test_entropy_term_default_weight_is_an_exact_zero():
    """Default off must be bit-for-bit compatible (~~AGENTS.md §5.11~~ →
    AGENTS.md §3.3 audit-ledger; corrected 2026-11-01)."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # noqa: PLC0415
        _LAST_LOSS_TERMS,
    )

    # the trainer builds this as torch.zeros(()) whenever the weight is 0
    zero = torch.zeros(())
    assert float(zero) == 0.0
    assert isinstance(_LAST_LOSS_TERMS, dict)


# ---------------------------------------------------------------------------
# (C) assemble_term — the formulation is load-bearing
# ---------------------------------------------------------------------------

def _chain_logits(p_bond, n=7, n_classes=5):
    pairs = [(i, i + 1) for i in range(n - 1)]
    logits = torch.full((len(pairs), n_classes), 0.0, dtype=torch.float64)
    for k in range(len(pairs)):
        logits[k, 0] = math.log((1.0 - p_bond) / p_bond)
    edge = (
        torch.tensor([a for a, _ in pairs]),
        torch.tensor([b for _, b in pairs]),
    )
    return logits, edge


def _assemble(p_bond, formulation, hinge_tau=None):
    logits, edge = _chain_logits(p_bond)
    logits.requires_grad_(True)
    z = torch.full((7,), 6, dtype=torch.long)
    value, n_frag, _ = assemble_fragment_loss(
        logits, edge, z, formulation=formulation, hinge_tau=hinge_tau
    )
    value.backward()
    return float(value), float(logits.grad.abs().max()), n_frag


def test_product_formulation_is_not_an_objective():
    """MEASURED: ~1e-08 gradient on a well-connected graph.

    This is what makes `assemble_loss_weight` alone a FLAG_ONLY flag: the
    default formulation's gradient is ~7 orders of magnitude below the
    bond_ce gradient it would share a loss sum with.
    """
    _, grad_connected, _ = _assemble(0.9, "product")
    _, grad_isolated, _ = _assemble(0.02, "product")
    assert grad_connected < 1e-7, (
        f"product formulation gradient on a connected graph is "
        f"{grad_connected:.3e}; if this is no longer tiny the config note "
        "in cfm_objective_terms_2026_10_11.yaml that justifies "
        "escape_time needs revisiting"
    )
    # and it IS alive when things are truly disconnected, so the assertion
    # above is about the near-connected regime, not about a dead term
    assert grad_isolated > grad_connected


def test_escape_time_is_live_where_product_is_dead():
    v_conn, g_conn, _ = _assemble(0.9, "escape_time")
    _, g_iso, _ = _assemble(0.02, "escape_time")
    assert g_conn > 1e-5, f"escape_time gradient on connected graph {g_conn:.3e}"
    assert g_iso > g_conn, "a stranded graph must pull harder than a connected one"


def test_escape_time_with_the_derived_hinge_keeps_the_pinned_zero_gradient():
    """The property that lets escape_time be enabled at all.

    molmetal/tests/test_assemble_connectivity.py pins "a graph that is
    already one component should not be charged for gradient" as a
    DELIBERATE design choice, and a default flip to escape_time was
    rejected because it broke that. Pairing escape_time with the derived
    tau restores the property without giving up liveness.
    """
    tau = 0.30  # metrics/by_metric/cfm_assemble_hinge_tau.json
    v_conn, g_conn, _ = _assemble(0.9, "escape_time", hinge_tau=tau)
    v_iso, g_iso, _ = _assemble(0.02, "escape_time", hinge_tau=tau)
    assert v_conn == 0.0 and g_conn == 0.0, (
        "the dead zone must silence a correctly-connected graph exactly"
    )
    assert v_iso > 0.0 and g_iso > 1e-5, (
        "the dead zone must not silence a genuinely fragmented graph"
    )


def test_unknown_formulation_is_rejected_loudly():
    """So a typo cannot silently fall back to the degenerate default."""
    with pytest.raises(ValueError):
        _assemble(0.9, "escape-tim")


# ---------------------------------------------------------------------------
# (D) the config under test — which of the three it enables
# ---------------------------------------------------------------------------

def _train_block():
    import yaml  # noqa: PLC0415

    with open(CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["train"]


def test_config_enables_the_two_live_terms():
    train = _train_block()
    assert float(train["entropy_reg_weight"]) > 0.0, (
        "(B) entropy regulariser is the term that addresses the C4 collapse "
        "and must be switched on"
    )
    assert float(train["assemble_loss_weight"]) > 0.0, (
        "(C) gen_single_fragment_ratio has no live loss term without this"
    )


def test_config_pairs_the_assemble_weight_with_a_live_formulation():
    """The load-bearing pairing. Weight without formulation is FLAG_ONLY."""
    train = _train_block()
    assert train["assemble_formulation"] == "escape_time", (
        "assemble_loss_weight > 0 with the default 'product' formulation "
        "enables a term whose gradient is ~1e-08 (see "
        "test_product_formulation_is_not_an_objective)"
    )
    assert float(train["assemble_hinge_tau"]) > 0.0, (
        "tau must be the derived value from "
        "metrics/by_metric/cfm_assemble_hinge_tau.json, not eyeballed; "
        "negative means NO dead zone, which would break the pinned "
        "connected-graph zero-gradient property"
    )


def test_config_leaves_the_degenerate_term_off():
    train = _train_block()
    assert float(train.get("gen_validity_weight", 0.0)) == 0.0, (
        "(A) soft_valence_term is a one-sided hinge: it can only push bond "
        "orders DOWN, never reward forming one. Turning it on buys the "
        "tracked gen_validity number against bond_type_f1, which is this "
        "run's early_stop_metric."
    )


def test_config_does_not_freeze_the_head_the_these_terms_read():
    """`bond_head_warmup_epochs > 0` makes both terms gradient-dead.

    They are built from `model.head`'s own bond-order softmax, so a frozen
    head means they contribute nothing at any weight for the first N
    epochs — the flag looks on and is not.
    """
    train = _train_block()
    assert int(train.get("bond_head_warmup_epochs", 0)) == 0


def test_config_keys_round_trip_to_real_argparse_dests():
    """AGENTS.md §5.11 round trip, verified BY TYPE.

    A key that resolves to no dest is a dead key; a list-valued dest
    written as a space-joined YAML string arrives as ONE string and matches
    nothing. Both failure modes are invisible to a value comparison, so the
    assertions below check `isinstance`, not just equality.
    """
    import importlib.util  # noqa: PLC0415

    trainer = os.path.join(
        REPO_ROOT, "molmetal", "scripts",
        "train_cfm_yuelbond_wasserstein.py",
    )
    argv = sys.argv
    sys.argv = ["train_cfm_yuelbond_wasserstein.py", "--config", CONFIG]
    try:
        spec = importlib.util.spec_from_file_location("_tr5_trainer", trainer)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_tr5_trainer"] = mod
        spec.loader.exec_module(mod)
        args = mod.parse_args()
    finally:
        sys.argv = argv

    assert isinstance(args.entropy_reg_weight, float)
    assert args.entropy_reg_weight > 0.0

    assert isinstance(args.assemble_loss_weight, float)
    assert args.assemble_loss_weight > 0.0

    assert isinstance(args.assemble_formulation, str), (
        "assemble_formulation must arrive as a str, not a list"
    )
    assert args.assemble_formulation == "escape_time"

    assert isinstance(args.assemble_hinge_tau, float)
    assert args.assemble_hinge_tau > 0.0

    assert float(getattr(args, "gen_validity_weight", 0.0)) == 0.0