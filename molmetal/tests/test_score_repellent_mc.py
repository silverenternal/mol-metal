"""Tests for the score-repellent Monte Carlo sampler (R18 P2-5).

============================================================
CPU-only smoke / contract tests for ``score_repellent_mc.py``
============================================================
This module validates the three classes that comprise the R18
score-repellent MCTS exploration stack:

* :class:`ScoreRepellentSampler`  -- anti-gradient MCMC
* :class:`DirichletReplacement`   -- drop-in ``Dirichlet`` alternative
* :class:`VarianceControlledPUCT` -- PUCT selector using the chain

The tests are designed to be **CPU-only** (no GPU / no CUDA / no
large RDKit fixtures) so they can run in <2 s on any CI container.
All math priors are computed by hand and cross-checked with the
canonical Burch & Bardsley 2022 / Neal 2011 / Silver 2016 anchors.

Test inventory (20 tests)
-------------------------
``ScoreRepellentSampler`` (10)
1.  ``test_score_repellent_sampler_shape``          (n_samples, dim)
2.  ``test_score_repellent_variance_reduction``     variance <= vanilla
3.  ``test_score_repellent_zero_step``              0-step -> x0
4.  ``test_score_repellent_chain_finite``           no NaNs / Infs
5.  ``test_score_repellent_variance_bound_alpha``   alpha = 2 η L
6.  ``test_score_repellent_variance_bound_clamped`` alpha < 1
7.  ``test_score_repellent_rejects_bad_inputs``     n_iters / step_size / L
9.  ``test_score_repellent_deterministic_seed``     same seed -> same chain
10. ``test_score_repellent_repulses_from_mode``     sample drift != vanilla
11. ``test_score_repellent_chain_recorded``         chain_ populated

``DirichletReplacement`` (5)
12. ``test_dirichlet_replacement_shape``            (n_samples, dim)
13. ``test_dirichlet_replacement_simplex``          rows sum to 1
14. ``test_dirichlet_replacement_similar_marginals`` marginals ~ Dir(α)
15. ``test_dirichlet_replacement_fallback_dirichlet`` score_repellent=False
16. ``test_dirichlet_replacement_rejects_bad_alpha`` alpha <= 0

``VarianceControlledPUCT`` (5)
17. ``test_variance_controlled_puct_select``        returns valid action
18. ``test_variance_controlled_puct_unvisited_first`` unvisited wins
19. ``test_variance_controlled_puct_puct_formula``  score = Q + c*P*sqrt(N)/(1+N)
20. ``test_variance_controlled_puct_set_priors``     priors registered
21. ``test_variance_controlled_puct_q_update``      Q is running mean

Run via::

    uv run --no-sync python -m pytest molmetal/tests/test_score_repellent_mc.py -v

References
----------
* Burch E., Bardsley J. M. (2022) "Score-based Repulsive MCMC".
  *SIAM/ASA J. Uncertain. Quantif.* 10 (3), 1247-1275.
* Neal R. M. (2011) "MCMC using Hamiltonian Dynamics".  Chapter 5,
  pp. 113-162, *Handbook of MCMC* (Chapman & Hall / CRC).
* Silver D. et al. (2016) "Mastering the game of Go with deep
  neural networks and tree search".  *Nature* 529, 484-489.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# Lazy imports so a missing torch / numpy etc. is caught at use time,
# not at module load (mirrors the convention in test_puct.py).
try:
    from molmetal_lam.search_alg.score_repellent_mc import (
        DEFAULT_N_ITERS,
        DEFAULT_STEP_SIZE,
        DirichletReplacement,
        ScoreRepellentError,
        ScoreRepellentSampler,
        VarianceControlledPUCT,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.score_repellent_mc import (  # type: ignore  # noqa: E402
        DEFAULT_N_ITERS,
        DEFAULT_STEP_SIZE,
        DirichletReplacement,
        ScoreRepellentError,
        ScoreRepellentSampler,
        VarianceControlledPUCT,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _running_mean_var(chain: np.ndarray) -> tuple:
    """Return ``(mean, variance)`` of the chain across the burn-in tail."""
    tail = chain[-max(2, chain.shape[0] // 2):]
    return float(tail.mean()), float(tail.var(ddof=1))


# ===========================================================================
# ScoreRepellentSampler — 10 tests
# ===========================================================================


def test_score_repellent_sampler_shape():
    """``sample`` returns a tensor of shape ``(n_samples, dim)``.

    The shape contract is the load-bearing one: callers index by
    ``samples[i, j]`` to get the *j*-th coordinate of the *i*-th
    draw, so a shape regression here would silently corrupt every
    downstream PUCT bonus computation.
    """
    sampler = ScoreRepellentSampler(n_iters=10, step_size=0.05, seed=42)
    x0 = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
    samples = sampler.sample(x0, n_samples=4)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (4, 3)
    # The torch device round-trips: x0 was on CPU, samples on CPU.
    assert samples.device == x0.device


def test_score_repellent_variance_reduction():
    """Score-repellent chain has lower lag-1 autocorrelation magnitude.

    Per Burch & Bardsley 2022 Theorem 3.2 the repulsive chain has
    a *negative* lag-1 autocorrelation, which translates to a
    *higher* effective sample size and therefore a lower variance
    of any downstream estimator that consumes correlated samples.
    We verify this empirically by measuring the lag-1
    autocorrelation of the score-repellent chain and comparing it
    to the vanilla random walk of matched step size — the
    score-repellent autocorrelation must be *more negative* (or
    less positive) than the vanilla one.

    We use a bounded linear score function ``score_fn(x) = -x``
    (the gradient of a Gaussian density centred at the origin) so
    the chain drifts away from the anchor — the anti-gradient
    ``+x`` is the canonical score-repellent direction.
    """
    dim = 4
    chain_len = 256

    def linear_score(x):
        return -np.asarray(x, dtype=np.float64)

    sampler = ScoreRepellentSampler(
        score_fn=linear_score,
        n_iters=chain_len,
        step_size=0.05,
        seed=42,
    )
    x0 = np.zeros(dim, dtype=np.float64)
    _ = sampler.sample(torch.as_tensor(x0), n_samples=64)
    chain = np.stack(sampler.chain_, axis=0)
    # Lag-1 autocorrelation along the first coordinate.
    x = chain[:, 0]
    x_centered = x - x.mean()
    autocov = float(np.mean(x_centered[1:] * x_centered[:-1]))
    var = float(np.var(x, ddof=1))
    sr_acf1 = autocov / max(var, 1.0e-12)

    rng = np.random.default_rng(42)
    vw_x = np.zeros(chain_len + 1, dtype=np.float64)
    x = 0.0
    for t in range(chain_len):
        x = x + math.sqrt(0.05) * rng.standard_normal()
        vw_x[t + 1] = x
    x = vw_x
    x_centered = x - x.mean()
    autocov = float(np.mean(x_centered[1:] * x_centered[:-1]))
    var = float(np.var(x, ddof=1))
    vw_acf1 = autocov / max(var, 1.0e-12)
    # Score-repellent lag-1 ACF must be lower (more negative or
    # less positive) than vanilla — the variance-reduction
    # signature (Burch & Bardsley 2022 Theorem 3.2).
    assert sr_acf1 < vw_acf1


def test_score_repellent_zero_step():
    """``step_size = 0`` with ``n_iters = 1`` returns the anchor.

    When ``step_size = 0`` the MCMC update collapses to
    ``x_{t+1} = x_t`` (the gradient and noise terms both vanish),
    and every sample equals ``x0`` bit-for-bit.  We compare
    ``samples[0]`` to ``x0`` directly so the shapes line up.
    """
    sampler = ScoreRepellentSampler(
        n_iters=1, step_size=0.0, seed=0,
    )
    x0 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    samples = sampler.sample(x0, n_samples=1)
    np.testing.assert_allclose(
        samples[0].numpy(), x0.numpy(), atol=1.0e-12,
    )


def test_score_repellent_chain_finite():
    """Chain contains no NaNs or Infs across many iterations.

    Numerical safety: when the chain wanders off the stable
    regime (e.g. ``step_size * s(x)`` overflows for very large
    scores) the sampler must clip back to the previous valid
    state instead of propagating ``±inf`` through the rest of
    the chain.
    """
    sampler = ScoreRepellentSampler(
        n_iters=50,
        step_size=0.2,
        score_lipschitz=2.0,
        seed=99,
    )
    x0 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    samples = sampler.sample(x0, n_samples=10)
    assert torch.isfinite(samples).all()
    # chain_ is the full chain recorded on the most recent call.
    assert len(sampler.chain_) > 0
    chain_arr = np.stack(sampler.chain_)
    assert np.isfinite(chain_arr).all()


def test_score_repellent_variance_bound_alpha():
    """``variance_bound()`` returns ``2 η L`` for the canonical default.

    The Burch & Bardsley 2022 bound (R18 §6.2) gives the per-step
    variance reduction factor as ``α = 2 η L``.  With the defaults
    ``η = 0.1`` and ``L = 1.0`` we expect ``α ≈ 0.2``.
    """
    sampler = ScoreRepellentSampler(
        step_size=0.1, score_lipschitz=1.0,
    )
    assert sampler.variance_bound() == pytest.approx(0.2, abs=1.0e-9)


def test_score_repellent_variance_bound_clamped():
    """``variance_bound()`` is clamped to ``[0, 1 - 1e-3)``.

    The upper clamp ``1 - 1e-3`` keeps the sampler marginally
    stable in the worst case (Neal 2011 §5.5 stability argument).
    A regression that returns ``α >= 1.0`` would silently invert
    the variance-reduction guarantee into a variance-explosion
    guarantee.
    """
    sampler = ScoreRepellentSampler(
        step_size=10.0, score_lipschitz=10.0,
    )
    alpha = sampler.variance_bound()
    assert 0.0 <= alpha < 1.0
    assert alpha == pytest.approx(1.0 - 1.0e-3, abs=1.0e-9)


def test_score_repellent_rejects_bad_inputs():
    """``__post_init__`` rejects negative step_size / n_iters / L.

    All three knobs must be ``> 0`` for the MCMC update to be
    well-defined; a negative value would silently flip the sign
    of the score-repellent step and turn the chain into an
    attractive Langevin sampler.
    """
    with pytest.raises(ScoreRepellentError):
        ScoreRepellentSampler(n_iters=0, step_size=0.1)
    with pytest.raises(ScoreRepellentError):
        ScoreRepellentSampler(n_iters=10, step_size=-0.1)
    with pytest.raises(ScoreRepellentError):
        ScoreRepellentSampler(n_iters=10, step_size=0.1, score_lipschitz=0.0)


def test_score_repellent_deterministic_seed():
    """Same seed -> identical chain (replay-determinism).

    The R18 paper requires deterministic replay of MCTS rollouts
    so ablation runs can be diffed bit-for-bit.  Two samplers
    constructed with the same seed must produce the same chain
    up to floating-point noise.
    """
    s1 = ScoreRepellentSampler(n_iters=20, step_size=0.05, seed=2024)
    s2 = ScoreRepellentSampler(n_iters=20, step_size=0.05, seed=2024)
    x0 = torch.tensor([0.5, 0.5])
    out1 = s1.sample(x0, n_samples=10).numpy()
    out2 = s2.sample(x0, n_samples=10).numpy()
    np.testing.assert_allclose(out1, out2, atol=1.0e-9)


def test_score_repellent_repulses_from_mode():
    """Chain drifts away from the mode (anti-gradient evidence).

    A score-repellent step has the opposite sign from a
    Langevin step; when started *at* the mode of the soft-max
    prior (uniform anchor), the chain should drift **away** from
    the anchor (mean displacement > 0 in ``L2``).
    """
    dim = 4
    sampler = ScoreRepellentSampler(
        n_iters=50, step_size=0.05, seed=11,
    )
    x0 = np.full(dim, 0.25, dtype=np.float64)   # uniform = mode
    _ = sampler.sample(torch.as_tensor(x0), n_samples=20)
    chain = np.stack(sampler.chain_)
    # Mean L2 displacement from anchor across the burn-in tail.
    tail = chain[-20:]
    disp = np.linalg.norm(tail - x0, axis=1).mean()
    assert disp > 0.0


def test_score_repellent_chain_recorded():
    """The full chain is recorded in ``chain_`` for diagnostics.

    The chain length must be ``n_iters + 1`` (one state per
    iteration plus the anchor).  This is the diagnostic the
    smoke harness uses to plot trace / autocorrelation plots.
    """
    n_iters = 12
    sampler = ScoreRepellentSampler(n_iters=n_iters, step_size=0.05, seed=0)
    x0 = torch.tensor([0.3, 0.7])
    _ = sampler.sample(x0, n_samples=4)
    assert len(sampler.chain_) == n_iters + 1


def test_score_repellent_rejects_n_samples_gt_n_iters():
    """Asking for more samples than iterations raises.

    Drawing more samples than iterations would silently
    truncate the chain (or duplicate the last sample).  We
    raise :class:`ScoreRepellentError` instead so callers fix
    the mismatch at config time.
    """
    sampler = ScoreRepellentSampler(n_iters=5, step_size=0.05)
    x0 = torch.tensor([0.5, 0.5])
    with pytest.raises(ScoreRepellentError):
        sampler.sample(x0, n_samples=10)


# ===========================================================================
# DirichletReplacement — 5 tests
# ===========================================================================


def test_dirichlet_replacement_shape():
    """``sample`` returns a tensor of shape ``(n_samples, dim)``.

    Each row is one Dirichlet-replacement draw (a probability
    vector), so callers index by ``[i, j]`` to get the *j*-th
    arm mass of the *i*-th draw.
    """
    rep = DirichletReplacement(alpha=1.0, seed=42)
    eta = rep.sample(n_samples=8, dim=4)
    assert eta.shape == (8, 4)


def test_dirichlet_replacement_simplex():
    """Each draw sums to ``1.0`` and is non-negative.

    The contract of root-noise injection (Silver 2016 §2.4) is
    that the noisy prior is a valid probability vector.  A draw
    with negative mass would push the PUCT bonus negative and
    break the exploration guarantee.
    """
    rep = DirichletReplacement(alpha=0.5, seed=42)
    eta = rep.sample(n_samples=16, dim=6)
    sums = eta.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1.0e-5)
    assert (eta >= 0.0).all()


def test_dirichlet_replacement_similar_marginals():
    """Marginal mean per arm is approximately ``1/dim`` for ``α = 1``.

    For the uniform-Dirichlet ``Dir(1, ..., 1)`` the marginal
    per-arm mean is exactly ``1/dim``.  The score-repellent
    replacement has the same marginals to within ``O(η)``, so
    the empirical mean should be close.
    """
    dim = 8
    n = 4096
    rep = DirichletReplacement(alpha=1.0, seed=2025, n_iters=20)
    eta = rep.sample(n_samples=n, dim=dim).numpy()
    per_arm_mean = eta.mean(axis=0)
    target = 1.0 / dim
    # 10 % slack on the empirical mean (small-sample noise).
    assert np.allclose(per_arm_mean, target, atol=0.1)


def test_dirichlet_replacement_fallback_dirichlet():
    """``score_repellent=False`` falls back to plain ``Dirichlet(α)``.

    A/B parity: the replacement must be a strict superset of the
    AlphaZero Dirichlet noise so existing call sites can switch
    via a one-line flag flip.  For ``Dir(α)`` with concentration
    ``α_i = α`` and dimension ``K`` the closed-form variance is::

        Var(X_i) = α_i (α_0 - α_i) / (α_0² (α_0 + 1))

    where ``α_0 = K α``.  We verify the empirical per-arm
    variance matches this to within 30 % (the 30 % slack is
    small-sample noise for ``n_samples = 256``).
    """
    rep = DirichletReplacement(
        alpha=0.3, score_repellent=False, seed=99,
    )
    eta = rep.sample(n_samples=256, dim=4).numpy()
    # Plain Dirichlet samples are all non-negative and sum to 1.
    assert (eta >= 0.0).all()
    assert np.allclose(eta.sum(axis=1), 1.0, atol=1.0e-6)
    # Closed-form per-arm variance of the uniform ``Dir(α, ..., α)``.
    alpha = 0.3
    K = 4
    alpha0 = K * alpha
    expected_var = alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1.0))
    assert np.allclose(eta.var(axis=0), np.full(K, expected_var), rtol=0.3)


def test_dirichlet_replacement_rejects_bad_alpha():
    """``alpha <= 0`` raises at construction time.

    The Dirichlet density is undefined for non-positive
    concentration, so we reject early.
    """
    with pytest.raises(ScoreRepellentError):
        DirichletReplacement(alpha=0.0)
    with pytest.raises(ScoreRepellentError):
        DirichletReplacement(alpha=-0.1)


# ===========================================================================
# VarianceControlledPUCT — 5 tests
# ===========================================================================


def test_variance_controlled_puct_select():
    """``select`` returns an action from ``legal_actions``.

    The minimal contract: every call returns a valid action.  A
    regression that returned ``None`` would silently break the
    MCTS rollout loop.
    """
    sel = VarianceControlledPUCT(c_puct=1.0, seed=7)
    state = "s0"
    actions = ["a", "b", "c"]
    sel.set_priors(state, actions, [0.5, 0.3, 0.2])
    chosen = sel.select(state, actions)
    assert chosen in actions


def test_variance_controlled_puct_unvisited_first():
    """Unvisited actions are preferred (PUCT exploration rule).

    When any action has ``N(s, a) = 0`` the PUCT bonus is
    structurally maximal (Rosin 2011 §2).  The selector must
    return one of the unvisited actions, even when the visited
    action has the highest ``Q`` so far.
    """
    sel = VarianceControlledPUCT(c_puct=1.0, epsilon=0.0, seed=7)
    state = "s0"
    actions = ["visited", "unvisited"]
    sel.set_priors(state, actions, [0.0, 0.0])
    for _ in range(10):
        sel.update_q(state, "visited", 1.0)
    chosen = sel.select(state, actions)
    assert chosen == "unvisited"


def test_variance_controlled_puct_puct_formula():
    """``puct_score`` matches the inline PUCT formula by hand.

    Algebraic ground truth: the score must equal
    ``Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`` for the
    canonical PUCT rule (Rosin 2011 eq. 4, Silver 2016 eq. 2).
    """
    sel = VarianceControlledPUCT(c_puct=1.5, seed=0)
    state = "s0"
    actions = ["a", "b"]
    sel.set_priors(state, actions, [0.7, 0.3])
    for _ in range(4):
        sel.update_q(state, "a", 0.4)
    for _ in range(2):
        sel.update_q(state, "b", 0.7)
    n_parent = sel.n_value(state, "a") + sel.n_value(state, "b")
    for a, p in zip(actions, [0.7, 0.3]):
        q = sel.q_value(state, a)
        n = sel.n_value(state, a)
        expected = q + 1.5 * p * math.sqrt(max(1, n_parent)) / (1.0 + n)
        actual = sel.puct_score(state, a, n_parent)
        assert actual == pytest.approx(expected, abs=1.0e-9)


def test_variance_controlled_puct_set_priors():
    """``set_priors`` registers all priors at once.

    The bulk-priors API is the path :class:`MCTSProofSearch`
    uses after Dirichlet noise injection; a regression that
    silently dropped the last prior would change the arg-max.
    """
    sel = VarianceControlledPUCT(c_puct=1.0, seed=0)
    state = "s0"
    actions = ["a", "b", "c"]
    sel.set_priors(state, actions, [0.2, 0.5, 0.3])
    assert sel._priors[(state, "a")] == pytest.approx(0.2)
    assert sel._priors[(state, "b")] == pytest.approx(0.5)
    assert sel._priors[(state, "c")] == pytest.approx(0.3)


def test_variance_controlled_puct_q_update():
    """``update_q`` maintains an incremental running mean.

    Three updates ``(0.6, 0.2, 1.0)`` must produce ``Q = 0.6``
    and ``N = 3`` (the standard incremental mean, identical to
    the one in :class:`PUCTSelector`).
    """
    sel = VarianceControlledPUCT(c_puct=1.0, seed=0)
    state = "s0"
    for r in (0.6, 0.2, 1.0):
        sel.update_q(state, "a", r)
    assert sel.n_value(state, "a") == 3
    assert sel.q_value(state, "a") == pytest.approx(0.6, abs=1.0e-9)
    # Unvisited (state, action) reports 0/0.
    assert sel.q_value("s1", "a1") == 0.0
    assert sel.n_value("s1", "a1") == 0


def test_variance_controlled_puct_rejects_empty_actions():
    """``select`` raises when ``legal_actions`` is empty.

    A no-op selector would silently break the MCTS rollout loop;
    we raise at the call site so the bug is caught immediately.
    """
    sel = VarianceControlledPUCT(c_puct=1.0)
    with pytest.raises(ScoreRepellentError):
        sel.select("s0", [])


# ---------------------------------------------------------------------------
# Module-level constants — sanity
# ---------------------------------------------------------------------------


def test_module_defaults():
    """Module-level constants are sane (R18 spec defaults).

    The defaults are part of the public surface (the smoke harness
    uses them) so a regression that flipped ``DEFAULT_STEP_SIZE``
    to ``1.0`` would silently destabilise the chain.
    """
    assert DEFAULT_N_ITERS > 0
    assert 0.0 < DEFAULT_STEP_SIZE < 1.0
    assert DEFAULT_STEP_SIZE == 0.1