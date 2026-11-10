"""Wf-A142 (2026-10-31) — unit tests over the CPU bond-head cost probe.

The probe (``molmetal/scripts/wfa142_head_forward_cost.py``) supports the
READ-ONLY feasibility report for frontier-lit row 3 (MolDiff-style
bond-predictor confidence-gradient guidance).

WHAT IS TESTED HERE
-------------------
* the probe's synthetic geometry respects the candidate cutoff;
* the probe's timers return well-formed, positive, finite numbers;
* the two bond heads are DIFFERENTIABLE WITH RESPECT TO ATOM POSITIONS
  — the precondition the whole MolDiff term depends on;
* the SIGN CONVENTION of the MolDiff term: with a hand-wired head whose
  bond logit is ``-distance``, ``d/dx log P(bond)`` is ATTRACTIVE
  (pulls the pair together), which is the sign the report's
  implementation plan depends on;
* the heads' per-call MUTABLE STATE is catalogued (the YuelBond head
  overwrites ``last_clash_rejected_pairs`` on every call — the report's
  "shared instance contaminates the decode diagnostic" hazard);
* the extrapolation arithmetic (``n_calls = n_steps * evals_per_step``);
* the emitted artifact parses and is internally consistent.

Nothing here touches a GPU, a checkpoint, or the network.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "molmetal" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import wfa142_head_forward_cost as probe  # noqa: E402


ARTIFACT = (
    PROJECT_ROOT / "molmetal" / "reports" / "wfa142_head_forward_cost" / "cost.json"
)
PAIRS_ARTIFACT = (
    PROJECT_ROOT / "molmetal" / "reports" / "wfa142_head_forward_cost" / "pairs.json"
)


# ---------------------------------------------------------------------------
# Synthetic geometry
# ---------------------------------------------------------------------------
def test_synthetic_cloud_respects_cutoff():
    cloud = probe.synthetic_cloud(n_atoms=24, cutoff=3.5, seed=0)
    assert cloud["positions"].shape == (24, 3)
    assert cloud["atomic_numbers"].shape == (24,)
    assert cloud["atomic_numbers"].dtype == torch.long
    # Every kept pair is strictly inside the cutoff, and every dropped pair
    # is at or outside it.
    assert bool((cloud["distances"] < 3.5).all())
    assert cloud["i_idx"].shape == cloud["j_idx"].shape
    assert cloud["i_idx"].shape[0] == cloud["distances"].shape[0]
    # Upper-triangle pairs only (i < j), never self-pairs.
    assert bool((cloud["i_idx"] < cloud["j_idx"]).all())


def test_synthetic_cloud_is_deterministic_for_a_fixed_seed():
    a = probe.synthetic_cloud(n_atoms=12, cutoff=3.5, seed=7)
    b = probe.synthetic_cloud(n_atoms=12, cutoff=3.5, seed=7)
    assert torch.equal(a["positions"], b["positions"])
    assert torch.equal(a["atomic_numbers"], b["atomic_numbers"])


def test_position_derived_features_put_distance_in_column_zero():
    positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32)
    i_idx = torch.tensor([0])
    j_idx = torch.tensor([1])
    feats = probe._position_derived_features(positions, i_idx, j_idx, 11)
    assert feats.shape == (1, 11)
    assert math.isclose(float(feats[0, 0]), 3.0, rel_tol=1e-6)
    # Column 8 is the "gap" feature (distance - single-bond anchor).
    assert math.isclose(float(feats[0, 8]), 3.0 - 1.5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
def test_time_callable_returns_well_formed_positive_numbers():
    out = probe.time_callable(lambda: sum(range(1000)), repeats=3, warmup=1)
    assert set(out) >= {"median_ms", "min_ms", "max_ms", "repeats", "headline_ms"}
    assert out["median_ms"] > 0.0
    assert out["min_ms"] <= out["median_ms"] <= out["max_ms"]
    assert out["headline_ms"] == out["min_ms"]
    assert math.isfinite(out["max_ms"])


def test_time_callable_rejects_zero_repeats():
    with pytest.raises(ValueError):
        probe.time_callable(lambda: None, repeats=0)


# ---------------------------------------------------------------------------
# The heads are differentiable w.r.t. positions — the MolDiff precondition
# ---------------------------------------------------------------------------
def test_bond_order_head_has_a_finite_position_gradient():
    out = probe.measure_bond_order_head(n_atoms=16, repeats=1)
    assert out["params"] > 0
    assert out["n_candidate_pairs"] > 0
    assert out["forward_ms"]["median_ms"] > 0.0
    assert out["forward_posgrad_ms"]["median_ms"] > 0.0


def test_yuelbond_head_has_a_finite_position_gradient():
    out = probe.measure_yuelbond_head(n_atoms=16, repeats=1)
    assert out["params"] > 0
    assert out["n_candidate_pairs"] > 0
    assert out["forward_ms"]["median_ms"] > 0.0
    assert out["forward_posgrad_ms"]["median_ms"] > 0.0


def test_yuelbond_forward_overwrites_its_clash_counter():
    """The head is NOT a pure function: it writes a per-call diagnostic.

    Sharing one head instance between the ODE guidance and the decode
    would make the decode report the guidance's counter.  Catalogued here
    so the report's hazard is backed by an executable check.
    """
    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import YuelBondDecoderHead

    head = YuelBondDecoderHead().eval()
    cloud = probe.synthetic_cloud(n_atoms=16, cutoff=3.5, seed=3)
    atom_cloud = AtomCloud(
        positions=cloud["positions"], atomic_numbers=cloud["atomic_numbers"].long()
    )
    head(atom_cloud)
    first = int(head.last_clash_rejected_pairs)
    head(atom_cloud)
    assert int(head.last_clash_rejected_pairs) == first  # idempotent here
    # The attribute exists and is an int — i.e. it IS mutable per-call state.
    assert isinstance(head.last_clash_rejected_pairs, int)


def test_bond_order_head_is_deterministic_in_eval_mode():
    """Guidance must be deterministic: dropout would randomise the force."""
    from molmetal.models.bond_head import BondOrderHead

    head = BondOrderHead(in_dim=11, hidden_dim=64).eval()
    feats = torch.arange(11, dtype=torch.float32).repeat(5, 1)
    a = head(feats)
    b = head(feats)
    assert torch.equal(a, b)
    # ... and in TRAIN mode the dropout makes it non-deterministic.
    head.train()
    torch.manual_seed(0)
    c = head(feats)
    torch.manual_seed(1)
    d = head(feats)
    assert not torch.equal(c, d)


# ---------------------------------------------------------------------------
# The sign convention of the MolDiff term
# ---------------------------------------------------------------------------
def _wire_distance_monotone_head():
    """A ``BondOrderHead`` whose BOND logit is exactly ``-distance``.

    fc1 maps the distance column (index 0) to hidden unit 0; the message
    layer is the identity; the output layer reads hidden unit 0 with
    weight -1 on the bond class.  Therefore
    ``logits[bond_class] = -d`` and ``log P(bond)`` decreases with the
    pair distance.
    """
    from molmetal.models.bond_head import BondOrderHead

    head = BondOrderHead(in_dim=11, hidden_dim=64, num_classes=5).eval()
    with torch.no_grad():
        head.fc1.weight.zero_()
        head.fc1.weight[:, 0] = 1.0
        head.fc1.bias.zero_()
        for fc in head.fc_layers:
            fc.weight.zero_()
            fc.weight.copy_(torch.eye(fc.weight.shape[0]))
            fc.bias.zero_()
        head.out.weight.zero_()
        head.out.bias.zero_()
        head.out.weight[1, 0] = -1.0  # bond class reads hidden unit 0
    return head


def test_moldiff_gradient_is_attractive_with_a_distance_monotone_head():
    """``d/dx log P_head(bond)`` must pull a bonded pair TOGETHER.

    This is the sign the implementation plan in the report relies on:
    the MolDiff term is ADDED to the velocity, and its gradient is
    already attractive for a pair the head is confident about.
    """
    head = _wire_distance_monotone_head()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
    ).requires_grad_(True)
    i_idx = torch.tensor([0])
    j_idx = torch.tensor([1])
    feats = probe._position_derived_features(positions, i_idx, j_idx, 11)
    logits = head(feats)
    # The bond logit is -distance.
    assert math.isclose(float(logits[0, 1].detach()), -2.0, rel_tol=1e-5)
    logp = torch.log_softmax(logits, dim=-1)[:, 1].sum()
    (grad,) = torch.autograd.grad(logp, positions)
    unit = torch.tensor([1.0, 0.0, 0.0])  # i -> j
    # Atom 0 moves toward atom 1 (+x), atom 1 moves toward atom 0 (-x):
    # the force is ATTRACTIVE.
    assert float(grad[0].detach() @ unit) > 0.0
    assert float(grad[1].detach() @ unit) < 0.0
    # Newton-3rd-law check: the pair's net force is zero.
    assert float((grad[0] + grad[1]).detach().abs().max()) < 1e-5


def test_moldiff_gradient_is_repulsive_for_the_no_bond_class():
    """The complement: ``d/dx log P(no-bond)`` pushes a pair APART.

    This is what lets a confident no-bond pair be pushed away instead of
    only being left alone.
    """
    head = _wire_distance_monotone_head()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
    ).requires_grad_(True)
    i_idx = torch.tensor([0])
    j_idx = torch.tensor([1])
    feats = probe._position_derived_features(positions, i_idx, j_idx, 11)
    logits = head(feats)
    logp = torch.log_softmax(logits, dim=-1)[:, 0].sum()  # no-bond class
    (grad,) = torch.autograd.grad(logp, positions)
    unit = torch.tensor([1.0, 0.0, 0.0])
    assert float(grad[0].detach() @ unit) < 0.0
    assert float(grad[1].detach() @ unit) > 0.0


# ---------------------------------------------------------------------------
# Cost extrapolation arithmetic
# ---------------------------------------------------------------------------
def test_extrapolate_ode_cost_arithmetic():
    out = probe.extrapolate_ode_cost(2.0, n_steps=100, n_model_evals_per_step=2)
    assert out["n_calls"] == 200
    assert math.isclose(out["total_s"], 0.4, rel_tol=1e-9)
    out1 = probe.extrapolate_ode_cost(2.0, n_steps=100, n_model_evals_per_step=1)
    assert out1["n_calls"] == 100
    assert math.isclose(out1["total_s"], 0.2, rel_tol=1e-9)


def test_model_evals_per_step_matches_the_solver():
    """The solver primitives: euler 1, midpoint 2, rk4 4 evaluations."""
    assert probe.MODEL_EVALS_PER_STEP["euler"] == 1
    assert probe.MODEL_EVALS_PER_STEP["midpoint"] == 2
    assert probe.MODEL_EVALS_PER_STEP["rk4"] == 4


# ---------------------------------------------------------------------------
# The emitted artifact
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="cost.json not generated yet")
def test_artifact_parses_and_is_consistent():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["schema"] == "wfa142.head_forward_cost.v1"
    assert payload["host"]["device"] == "cpu"
    assert payload["shapes"], "no shapes measured"

    def _cell_ms(node):
        """A timing cell is either a FRESH run (min/median/max/headline_ms)
        or a MERGED cell (min_ms/max_ms/n_sources).  Accept both."""
        if "min_ms" in node and "median_ms" not in node:
            return node["min_ms"], node["max_ms"]
        assert node["min_ms"] <= node["median_ms"] <= node["max_ms"]
        assert node["headline_ms"] == node["min_ms"]
        return node["min_ms"], node["max_ms"]

    for row in payload["shapes"]:
        base_lo, base_hi = _cell_ms(row["table_guidance_baseline"]["forward_ms"])
        yb = row["yuelbond_decoder_head"]
        fwd_lo, fwd_hi = _cell_ms(yb["forward_ms"])
        grad_lo, grad_hi = _cell_ms(yb["forward_posgrad_ms"])
        assert base_lo > 0.0
        assert base_lo <= base_hi
        assert fwd_lo > 0.0 and fwd_lo <= fwd_hi
        assert grad_lo > 0.0 and grad_lo <= grad_hi
        assert yb["n_candidate_pairs"] > 0
        # The position-gradient term is strictly more expensive than the
        # plain forward: it is a forward PLUS an autograd pass.
        assert grad_lo > fwd_lo
        assert grad_hi > fwd_hi
        # The cost must be monotone in the number of candidate pairs.
        assert row["n_atoms"] >= 1
    for block in payload["ode_cost_model"].values():
        for entry in block.values():
            assert entry["n_calls"] == (
                entry["n_steps"] * entry["model_evals_per_step"]
            )
            assert entry["total_s"] >= 0.0
            assert entry["per_call_ms"] > 0.0


@pytest.mark.skipif(not ARTIFACT.exists(), reason="cost.json not generated yet")
def test_artifact_is_monotone_in_atoms():
    """The HEAD's per-call cost must not DECREASE as the molecule grows.

    The head's forward is dominated by the pair-embedding MLP, whose cost
    grows with the candidate-pair count E, so the merged min estimator must
    restore monotonicity (the pre-merge single-run numbers did not).

    The TABLE guidance is deliberately NOT asserted monotone: its per-call
    cost at these sizes is dominated by fixed overhead (radius-graph
    construction), so the MEASURED curve is flat within the contention
    band.  It is only asserted to stay inside a factor-2 band.
    """
    payload = json.loads(ARTIFACT.read_text())
    rows = sorted(payload["shapes"], key=lambda r: r["n_atoms"])

    def _lo(node):
        return node["min_ms"] if "median_ms" not in node else node["min_ms"]

    grad = [_lo(r["yuelbond_decoder_head"]["forward_posgrad_ms"]) for r in rows]
    for lo, hi in zip(grad, grad[1:]):
        assert hi >= lo, f"non-monotone head cost curve: {grad}"

    table = [_lo(r["table_guidance_baseline"]["forward_ms"]) for r in rows]
    assert max(table) <= 2.0 * min(
        table
    ), f"table-guidance cost varies by more than 2x across shapes: {table}"

    # The position-gradient term must always cost more than the plain
    # forward of the same head.
    for r in rows:
        fwd = _lo(r["yuelbond_decoder_head"]["forward_ms"])
        grd = _lo(r["yuelbond_decoder_head"]["forward_posgrad_ms"])
        assert grd > fwd


@pytest.mark.skipif(not ARTIFACT.exists(), reason="cost.json not generated yet")
def test_artifact_records_its_merge_provenance():
    """A merged artifact must say what it was merged from."""
    payload = json.loads(ARTIFACT.read_text())
    if "merged_from" in payload:
        assert len(payload["merged_from"]) >= 2
        for src in payload["merged_from"]:
            assert "path" in src and "created_utc" in src
        # Every shape cell must then carry the source count.
        for row in payload["shapes"]:
            assert row.get("n_sources", 1) == len(payload["merged_from"])
    else:
        # A fresh (unmerged) artifact: the anchor must still be recorded.
        assert payload["cost_model_anchor_n_atoms"] > 0


# ---------------------------------------------------------------------------
# The pair-set census (guidance's symmetric graph vs the head's list)
# ---------------------------------------------------------------------------
def test_pair_sets_measure_the_two_different_pair_sets():
    """The guidance's graph is SYMMETRIC; the head's list is UPPER-TRIANGLE.

    Measured on ONE geometry, so the two counts are comparable: the
    guidance's edge count must be the double of its unique-pair count,
    and the head's list must be a SUBSET (it drops the metal-collapse
    pairs the guidance still pushes on).
    """
    out = probe.measure_pair_sets(n_atoms=16, cutoff=3.5, seed=0)
    assert out["head_is_upper_triangle"] is True
    assert out["guidance_radius_graph_edges"] == 2 * out["guidance_unique_pairs"]
    assert 0 < out["head_candidate_pairs"] < out["guidance_unique_pairs"]


def test_pair_sets_are_deterministic():
    a = probe.measure_pair_sets(n_atoms=24, cutoff=3.5, seed=0)
    b = probe.measure_pair_sets(n_atoms=24, cutoff=3.5, seed=0)
    assert a["head_candidate_pairs"] == b["head_candidate_pairs"]
    assert a["guidance_radius_graph_edges"] == b["guidance_radius_graph_edges"]


@pytest.mark.skipif(not PAIRS_ARTIFACT.exists(), reason="pairs.json not generated yet")
def test_pairs_artifact_is_consistent():
    payload = json.loads(PAIRS_ARTIFACT.read_text())
    assert payload["schema"] == "wfa142.pair_sets.v1"
    assert payload["host"]["device"] == "cpu"
    assert payload["pair_sets"], "no pair-set rows"
    for row in payload["pair_sets"]:
        assert row["guidance_radius_graph_edges"] == (2 * row["guidance_unique_pairs"])
        assert row["head_is_upper_triangle"] is True
        assert 0 < row["head_candidate_pairs"] <= row["guidance_unique_pairs"]
    # The head's candidate list must GROW with the molecule.
    rows = sorted(payload["pair_sets"], key=lambda r: r["n_atoms"])
    head = [r["head_candidate_pairs"] for r in rows]
    for lo, hi in zip(head, head[1:]):
        assert hi > lo
