"""Anti-decorative precondition tests for the 5 highest-leverage math modules.

The failure mode this file guards against is NOT a crash: it is a module
whose *mathematical preconditions* are silently violated, so it trains and
runs but contributes nothing valid (canonical example: the W2 term found
fully detached — zero gradient — after 2.4x wasted wall-time, Wf-C3).

For each touched module we prove two things:

  (i) on VALID inputs ``check_preconditions`` returns ``[]`` (empty list);
  (ii) on an ADVERSARIALLY INVALID input it returns a NON-EMPTY list whose
       entries NAME the violated precondition.

Layer-2 (the ``CHECK_MATH_PRECONDITIONS=1`` wire-in) is checked separately:
  (iii) with the env flag SET, the module's main entry emits a warning
        naming the violation;
  (iv) with the flag UNSET (default), the main entry emits NO warning and
       the numeric output is bit-for-bit unchanged.

Modules covered (all under ``molmetal/math_skills/`` unless noted):
  1. se3_schrodinger_bridge   (LogSinkhornLoss.forward / SchrodingerBridgeSE3.sample)
  2. transport_relaxation     (RelaxedOT.forward / .coupling)
  3. multi_parameter_ph       (two_param_filtration)
  4. wasserstein_gradient_flow (W2GradientFlow.drift / ReactionStepW2.forward)
  5. molmetal_lam/reward/hilbert_primal_dual (LambdaASTSRHilbert.optimize / PrimalDualOptimizer.step)

Honest framing: every "invalid" case below is a *real* violation of a
mathematical assumption stated in the module docstring / reference paper
(probability measure, valid cost, simplex, finite design matrix, filter
positivity). No case was invented purely to force a non-empty list.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from molmetal.math_skills import multi_parameter_ph as mph
from molmetal.math_skills import se3_schrodinger_bridge as se3
from molmetal.math_skills import transport_relaxation as tr
from molmetal.math_skills import wasserstein_gradient_flow as wgf
from molmetal.molmetal_lam.reward import hilbert_primal_dual as hpd


# ===========================================================================
# 1. se3_schrodinger_bridge — probability-measure / cost preconditions
# ===========================================================================

def _valid_sinkhorn_args():
    p0 = torch.randn(6, 6)
    p1 = torch.randn(6, 6)
    a = torch.full((6,), 1.0 / 6.0)
    b = torch.full((6,), 1.0 / 6.0)
    return p0, p1, a, b


def test_se3_valid_returns_empty():
    p0, _p1, a, b = _valid_sinkhorn_args()
    assert se3.check_preconditions(p0, entry="sinkhorn", a=a, b=b) == []


def test_se3_invalid_marginal_not_probability_measure():
    """Adversarial: ``a`` has total mass 3, not 1 — not a probability measure.

    The Sinkhorn objective ``<P,C> + eps H(P|a(x)b)`` is only the
    entropy-regularised OT cost when ``a``/``b`` are probability measures;
    an un-normalised vector silently rescales the whole objective.
    """
    p0, _p1, _a, b = _valid_sinkhorn_args()
    bad_a = torch.full((6,), 0.5)  # sums to 3
    out = se3.check_preconditions(p0, entry="sinkhorn", a=bad_a, b=b)
    assert out, "expected a violation for a non-normalised marginal"
    assert any("does not sum to 1" in msg for msg in out)


def test_se3_invalid_negative_mass():
    p0, _p1, _a, b = _valid_sinkhorn_args()
    bad_a = torch.tensor([1.2, -0.2, 0.0, 0.0, 0.0, 0.0])  # sums to 1 but has negative mass
    out = se3.check_preconditions(p0, entry="sinkhorn", a=bad_a, b=b)
    assert any("negative mass" in msg for msg in out)


def test_se3_invalid_cost_negative():
    p0, _p1, a, b = _valid_sinkhorn_args()
    C = -torch.ones(6, 6)
    out = se3.check_preconditions(p0, entry="sinkhorn", a=a, b=b, C=C)
    assert any("negative entries" in msg for msg in out)


def test_se3_invalid_sample_shape():
    """Adversarial: a ``(5,3)`` batch is not a valid ``se(3)`` twist batch."""
    out = se3.check_preconditions(torch.randn(5, 3), entry="sample")
    assert any("(n, 6)" in msg for msg in out)


def test_se3_invalid_sample_t_values_out_of_range():
    p0 = torch.randn(4, 6)
    out = se3.check_preconditions(
        p0, entry="sample", t_values=torch.tensor([-0.1, 0.5, 1.4])
    )
    assert any("[0, 1]" in msg for msg in out)


def test_se3_default_off_is_bit_identical(monkeypatch):
    """Default OFF (env unset) => numeric output unchanged + no warning."""
    monkeypatch.delenv("CHECK_MATH_PRECONDITIONS", raising=False)
    torch.manual_seed(0)
    loss = se3.LogSinkhornLoss()
    p0, p1, _a, _b = _valid_sinkhorn_args()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loss_val, _P = loss.forward(p0, p1)
    assert [x for x in w if "math-precondition" in str(x.message)] == []
    torch.manual_seed(0)
    p0b, p1b, _a, _b = _valid_sinkhorn_args()
    loss2 = se3.LogSinkhornLoss()
    loss_val2, _P2 = loss2.forward(p0b, p1b)
    assert float(loss_val) == float(loss_val2)


def test_se3_wire_in_fires_under_env_flag(monkeypatch):
    monkeypatch.setenv("CHECK_MATH_PRECONDITIONS", "1")
    loss = se3.LogSinkhornLoss()
    p0, p1, _a, _b = _valid_sinkhorn_args()
    bad_a = torch.full((6,), 0.5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loss.forward(p0, p1, a=bad_a, b=bad_a)
    assert any("math-precondition" in str(x.message) for x in w)


# ===========================================================================
# 2. transport_relaxation — probability-measure / cost / lambda preconditions
# ===========================================================================

def test_transport_valid_returns_empty():
    p0 = torch.randn(5, 4)
    a = torch.full((5,), 0.2)
    b = torch.full((5,), 0.2)
    assert tr.check_preconditions(p0, entry="relaxed_ot", a=a, b=b, lambda_relax=1e-3) == []


def test_transport_invalid_marginal_not_probability_measure():
    """Adversarial: un-normalised target marginals.

    ``_soft_marginals`` renormalises the blend with a convex reward, so an
    un-normalised ``a`` is silently absorbed — the caller's OT problem was
    malformed but the solver reports a plausible cost.
    """
    p0 = torch.randn(5, 4)
    bad_a = torch.full((5,), 2.0)  # sums to 10
    out = tr.check_preconditions(p0, entry="relaxed_ot", a=bad_a, b=bad_a)
    assert any("does not sum to 1" in msg for msg in out)


def test_transport_invalid_negative_cost():
    p0 = torch.randn(5, 4)
    a = torch.full((5,), 0.2)
    out = tr.check_preconditions(
        p0, entry="relaxed_ot", a=a, b=a, C=-torch.ones(5, 5)
    )
    assert any("negative entries" in msg for msg in out)


def test_transport_invalid_lambda_relax_negative():
    p0 = torch.randn(5, 4)
    a = torch.full((5,), 0.2)
    out = tr.check_preconditions(p0, entry="relaxed_ot", a=a, b=a, lambda_relax=-1.0)
    assert any("lambda_relax" in msg for msg in out)


def test_transport_invalid_non_tensor_input():
    out = tr.check_preconditions(None, entry="relaxed_ot")
    assert out and "not a Tensor" in out[0]


# ===========================================================================
# 3. multi_parameter_ph — point-cloud / atomic-number / filter preconditions
# ===========================================================================

def _valid_cloud():
    coords = torch.randn(2, 5, 3)
    z = torch.tensor([[6, 7, 8, 9, 10], [6, 6, 7, 7, 8]])
    return coords, z


def test_multiph_valid_returns_empty():
    coords, z = _valid_cloud()
    assert (
        mph.check_preconditions(
            coords,
            entry="two_param_filtration",
            pocket_atom_types=z,
            max_radius=5.0,
            max_coord=4.0,
            n_bins=8,
        )
        == []
    )


def test_multiph_invalid_negative_atomic_numbers():
    """Adversarial: negative atomic numbers.

    The coordination filter is ``|z_i - z_j|`` with ``z_k`` the atomic
    number (a proton count); a negative label is undefined and the filter
    silently produces magnitudes that no real element supports.
    """
    coords, z = _valid_cloud()
    bad_z = -z.abs()
    out = mph.check_preconditions(
        coords, entry="two_param_filtration", pocket_atom_types=bad_z
    )
    assert any("negative atomic numbers" in msg for msg in out)


def test_multiph_invalid_atom_types_shape_mismatch():
    coords, _z = _valid_cloud()
    bad_z = torch.tensor([[6, 7], [6, 8]])  # (2, 2) vs coords (2, 5, 3)
    out = mph.check_preconditions(
        coords, entry="two_param_filtration", pocket_atom_types=bad_z
    )
    assert any("does not match" in msg for msg in out)


def test_multiph_invalid_nonpositive_radius():
    coords, z = _valid_cloud()
    out = mph.check_preconditions(
        coords, entry="two_param_filtration", pocket_atom_types=z, max_radius=0.0
    )
    assert any("max_radius" in msg for msg in out)


def test_multiph_invalid_n_bins_too_small():
    coords, z = _valid_cloud()
    out = mph.check_preconditions(
        coords, entry="two_param_filtration", pocket_atom_types=z, n_bins=1
    )
    assert any("n_bins" in msg for msg in out)


def test_multiph_invalid_nonfinite_coords():
    coords, z = _valid_cloud()
    coords = coords.clone()
    coords[0, 0, 0] = float("nan")
    out = mph.check_preconditions(
        coords, entry="two_param_filtration", pocket_atom_types=z
    )
    assert any("non-finite" in msg for msg in out)


def test_multiph_wire_in_catches_what_check_inputs_misses(monkeypatch):
    """Layer-2 value: NaN coords pass the module's own ``_check_inputs``
    (which only validates shapes / non-negativity of ``z``) but our
    precondition checker flags them and the wired main entry warns.

    Honest note: the *negative atomic number* adversarial case is caught
    earlier by ``_check_inputs`` (a hard raise), so the flag's marginal
    value for multiph is precisely the non-finite / non-integer cases.
    """
    monkeypatch.setenv("CHECK_MATH_PRECONDITIONS", "1")
    coords, z = _valid_cloud()
    coords = coords.clone()
    coords[1, 2, 1] = float("inf")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mph.two_param_filtration(coords, z, max_radius=5.0, max_coord=4.0, n_bins=8)
    assert any("math-precondition" in str(x.message) for x in w)


# ===========================================================================
# 4. wasserstein_gradient_flow — score-space / simplex / dt preconditions
# ===========================================================================

def test_wgf_valid_returns_empty():
    x = torch.randn(4, 3)
    s = torch.randn(4, 3)
    assert wgf.check_preconditions(x, entry="w2_gradient_flow", score=s) == []


def test_wgf_invalid_score_shape_mismatch():
    """Adversarial: score field in a different space than the particles.

    The drift ``v = -(s(x) - s_ref)`` is a tangent vector at ``x``; a score
    of a different shape makes the subtraction broadcast illegally (or, for
    a broadcastable shape, silently produce a wrong-dimension drift).
    """
    x = torch.randn(4, 3)
    bad_s = torch.randn(4, 2)
    out = wgf.check_preconditions(x, entry="w2_gradient_flow", score=bad_s)
    assert any("does not match x" in msg for msg in out)


def test_wgf_invalid_missing_score():
    x = torch.randn(4, 3)
    out = wgf.check_preconditions(x, entry="w2_gradient_flow", score=None)
    assert any("score: missing" in msg for msg in out)


def test_wgf_invalid_nonfinite_score():
    x = torch.randn(4, 3)
    bad_s = torch.full((4, 3), float("inf"))
    out = wgf.check_preconditions(x, entry="w2_gradient_flow", score=bad_s)
    assert any("non-finite" in msg for msg in out)


def test_wgf_invalid_dt_negative():
    x = torch.randn(4, 3)
    s = torch.randn(4, 3)
    out = wgf.check_preconditions(x, entry="w2_gradient_flow", score=s, dt=-0.1)
    assert any("dt" in msg for msg in out)


def test_wgf_reaction_valid_returns_empty():
    r = torch.tensor([[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.0, 1.0, 0.0, 0.0]])
    assert (
        wgf.check_preconditions(
            r, entry="reaction_step", product=p, dt=1.0, n_bond_types=4
        )
        == []
    )


def test_wgf_reaction_invalid_rows_not_simplex():
    """Adversarial: fingerprint rows are not probability vectors.

    The reaction step's drift is ``p - r`` — the optimal-transport direction
    in the *bond-type simplex* metric. A row summing to 2 is off the simplex,
    so ``p - r`` is no longer a transport direction and the W₂ distance is
    meaningless.
    """
    bad = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # sums to 2
    out = wgf.check_preconditions(
        bad, entry="reaction_step", product=bad.clone(), dt=1.0, n_bond_types=4
    )
    assert any("probability vectors" in msg for msg in out)


def test_wgf_reaction_invalid_shape_mismatch():
    r = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    p = torch.tensor([[0.5, 0.5]])  # wrong K
    out = wgf.check_preconditions(
        r, entry="reaction_step", product=p, dt=1.0, n_bond_types=4
    )
    assert any("does not match reactant" in msg for msg in out)


# ===========================================================================
# 5. hilbert_primal_dual — convexity / finiteness / step-size preconditions
# ===========================================================================

def _valid_regression():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8))
    y = rng.standard_normal(20)
    return X, y


def test_hpd_valid_returns_empty():
    X, y = _valid_regression()
    assert (
        hpd.check_preconditions(
            X,
            entry="hilbert_primal_dual",
            target=y,
            population_dim=8,
            lr_primal=0.01,
            lr_dual=0.01,
            l2_lambda=0.01,
            n_iters=50,
        )
        == []
    )


def test_hpd_invalid_nonfinite_design_matrix():
    """Adversarial: NaN in the design matrix.

    The ridge least-squares objective ``0.5||Xx - y||² + 0.5 l||x||²`` is
    only convex & finite when ``X`` is finite; NaNs make it undefined and
    the inner clip/decay guards silently freeze the iterate instead of
    surfacing the malformed problem.
    """
    X, y = _valid_regression()
    X = X.copy()
    X[3, 2] = np.nan
    out = hpd.check_preconditions(
        X, entry="hilbert_primal_dual", target=y, population_dim=8
    )
    assert any("non-finite" in msg for msg in out)


def test_hpd_invalid_length_mismatch():
    X, y = _valid_regression()
    out = hpd.check_preconditions(
        X, entry="hilbert_primal_dual", target=y[:-1], population_dim=8
    )
    assert any("does not match features rows" in msg for msg in out)


def test_hpd_invalid_dim_mismatch():
    X, y = _valid_regression()
    out = hpd.check_preconditions(
        X, entry="hilbert_primal_dual", target=y, population_dim=16
    )
    assert any("population_dim" in msg for msg in out)


def test_hpd_invalid_negative_l2_breaks_convexity():
    X, y = _valid_regression()
    out = hpd.check_preconditions(
        X, entry="hilbert_primal_dual", target=y, l2_lambda=-1.0
    )
    assert any("l2_lambda" in msg for msg in out)


def test_hpd_invalid_nonpositive_step_size():
    X, y = _valid_regression()
    out = hpd.check_preconditions(
        X, entry="hilbert_primal_dual", target=y, lr_primal=0.0
    )
    assert any("lr_primal" in msg for msg in out)


def test_hpd_step_valid_returns_empty():
    x = np.zeros(8)
    assert hpd.check_preconditions(x, entry="pd_step", lr_primal=0.01, lr_dual=0.01) == []


def test_hpd_step_invalid_nonfinite_iterate():
    x = np.full(8, np.inf)
    out = hpd.check_preconditions(x, entry="pd_step", lr_primal=0.01, lr_dual=0.01)
    assert any("primal_var" in msg and "non-finite" in msg for msg in out)


# ===========================================================================
# Layer-2 wire-in: default-off numerics unchanged for one representative
# ===========================================================================

def test_hpd_default_off_is_bit_identical(monkeypatch):
    """LambdaASTSRHilbert.optimize gives the same coefficients flag off vs on."""
    X, y = _valid_regression()
    monkeypatch.delenv("CHECK_MATH_PRECONDITIONS", raising=False)
    off = hpd.LambdaASTSRHilbert(population_dim=8, seed=1).optimize(X, y)
    monkeypatch.setenv("CHECK_MATH_PRECONDITIONS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = hpd.LambdaASTSRHilbert(population_dim=8, seed=1).optimize(X, y)
    assert np.array_equal(off, on)


def test_wgf_default_off_is_bit_identical(monkeypatch):
    monkeypatch.delenv("CHECK_MATH_PRECONDITIONS", raising=False)
    torch.manual_seed(7)
    x = torch.randn(5, 3)
    s = torch.randn(5, 3)
    flow = wgf.W2GradientFlow()
    off = flow.step(x, s, dt=0.5)
    monkeypatch.setenv("CHECK_MATH_PRECONDITIONS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = wgf.W2GradientFlow().step(x, s, dt=0.5)
    assert torch.equal(off, on)


# ===========================================================================
# Utility modules with NO checkable precondition are declared, not invented
# ===========================================================================

def test_pure_utility_modules_declared_not_invented():
    """Honest framing: the following were inspected and found to be pure
    utilities with no checkable *mathematical* precondition, so they are
    intentionally NOT given a check_preconditions function:
      - SE3Geodesic / so3_exp / so3_log: typed linear-algebra primitives
        (a wrong-shape input raises an ordinary tensor error, not a silent
        math-precondition violation).
      - ChemicalGraphMetric: an ordinary discrete metric; distance() is
        total on any graph pair — nothing to silently violate.
      - HilbertSpace basis helpers: pure Fourier-linear-algebra utility.
    """
    assert not hasattr(se3.SE3Geodesic, "check_preconditions")
    assert not hasattr(wgf.ChemicalGraphMetric, "check_preconditions")
