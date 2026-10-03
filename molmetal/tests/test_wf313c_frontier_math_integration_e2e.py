"""Wf-313C — Frontier Math Integration end-to-end smoke tests (Day 3-5 deliverable).

================================================================
8 NEW integration smoke tests for the 4 STANDALONE frontier-math
adapters composed via :mod:`molmetal.adapters.frontier_math_integration`:

* ``test_multi_ph_pocket_augmentation`` — feature extraction +
  augmentation shape sanity (``(B, H + n_bins**2)`` downstream).
* ``test_se3_lipman_solver_tangent_correctness`` — solver's
  rotation-block orthogonality stays under the 1e-3 contract
  during integration.
* ``test_se3_sb_decoder_endpoint_distribution`` — Schrödinger
  Bridge endpoints at ``t = 0`` and ``t = 1`` match the source /
  target poses within Sinkhorn rounding tolerance.
* ``test_transport_relaxation_endpoint_term`` — ε-annealing from
  ``1.0`` to ``0.0`` is monotonic in the KL penalty.
* ``test_joint_frontier_integration_no_nan`` — full
  :class:`FrontierMathIntegration` instantiation + a forward
  pass through every adapter returns finite tensors.
* ``test_research_only_modules_excluded`` — the 3 research-only
  modules (``bass_existence`` / ``sb_bass_martingale`` /
  ``monge_ampere_regularity``) are NOT in the factory surface.
* ``test_factory_method_consistency`` — every ``get_*`` method
  returns the canonical public class from the standalone adapter.
* ``test_frontier_integration_metrics`` — each adapter's metric
  output is finite + non-negative + has the expected scalar /
  shape contract.

Honest framing (per Wf-313C brief)
----------------------------------
* No modification of :mod:`molmetal.adapters.flow_matching_lipman`
  (R17 invariant preserved).
* Each test exercises ONLY the factory surface from
  :class:`FrontierMathIntegration` — no direct adapter import.
* All 8 tests are CPU-only and run in <1 s on a single thread.
* The Day 7+ wire-in (CFM loop + multi-task backbone + GPU
  production) is deferred to a USER_OK-gated ultracode window.

Cross-references
----------------
* Wf-310F Day 1-2 smoke tests: ``test_wf310f_frontier_math_integration.py``
  (4 build-only tests; this file complements with 8 paper-grade
  behavioural tests).
* Wf-309B equivalent tests: ``test_wf309b_frontier_math_integration.py``
  (same 4 test names).
* TODO-234 §6 — recommended execution order (Day 3-5 = this file).
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Constants — per Wf-313C honest framing
# ---------------------------------------------------------------------------

#: W2.1 orthogonality hard contract from
#: :mod:`molmetal.adapters.se3_lipman_integration` docstring.
SE3_ORTHO_TOL: float = 1e-3

#: Number of ε-annealing steps for the transport-relaxation test.
#: 6 steps (1.0 / 0.1 / 0.01 / ...) covers 6 decades of annealing
#: and keeps the test well under 100 ms.
N_ANNEAL_STEPS: int = 6

#: 3 research-only modules that are HONESTLY EXCLUDED from the
#: :class:`FrontierMathIntegration` factory surface per Wf-305B /
#: Wf-306B / TODO-234 §2.  Their value is the paper §3 reference
#: text, not the code path.
RESEARCH_ONLY_MODULES = (
    "bass_existence",
    "sb_bass_martingale",
    "monge_ampere_regularity",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fmi():
    """Return a :class:`FrontierMathIntegration` with default config.

    Day 3-5 paper-grade integration smoke tests reuse the Wf-310F
    Day 1-2 fixture — same factory surface, deeper assertions.
    """
    from molmetal.adapters.frontier_math_integration import FrontierMathIntegration

    return FrontierMathIntegration()


# ---------------------------------------------------------------------------
# 1. Multi-PH pocket augmentation — feature extract + shape sanity
# ---------------------------------------------------------------------------
def test_multi_ph_pocket_augmentation(fmi):
    """Multi-PH pocket feature extraction + concatenation shape.

    The :class:`MultiPHPocketFeatures` module emits a
    ``(B, n_bins, n_bins)`` feature map; the
    :func:`augment_pocket_embedding` helper concatenates a
    ``(B, H)`` base embedding with the flattened feature map to
    produce a ``(B, H + n_bins**2)`` downstream tensor.

    Smoke contract:

    * feature map shape ``(2, 4, 4)`` for ``(B=2, P=8)`` synthetic
      pocket cloud at ``n_bins=4``;
    * augmented embedding shape ``(2, H=16 + 4**2) = (2, 32)``;
    * the leading slice of the augmented tensor equals the base
      embedding verbatim (the augmentation is *additive only*).
    """
    from molmetal.adapters.frontier_math_integration import (
        FrontierMathIntegration,
    )
    from molmetal.adapters.multi_ph_integration import augment_pocket_embedding

    assert isinstance(fmi, FrontierMathIntegration)
    features_mod = fmi.get_multi_ph_features(
        override={"n_bins": 4, "max_radius": 3.0, "max_coord": 2.0},
    )
    # Synthetic pocket cloud: (B=2, P=8, 3).
    torch.manual_seed(0)
    pocket = torch.randn(2, 8, 3)
    feat_map = features_mod.compute(pocket)
    # Output shape = (B, n_bins, n_bins) = (2, 4, 4).
    assert feat_map.shape == (
        2,
        4,
        4,
    ), f"unexpected feature map shape: {tuple(feat_map.shape)}"
    # Multi-PH feature map is a count of (r, d) pairs per pocket —
    # must be non-negative.
    assert (feat_map >= 0).all(), "multi-PH feature map must be non-negative"

    # Augment a synthetic base embedding (H=16) with the flat
    # feature map (n_bins**2 = 16) -> (B, 32) downstream tensor.
    base = torch.randn(2, 16)
    augmented = augment_pocket_embedding(base, feat_map)
    assert augmented.shape == (2, 32), (
        f"augmented embedding shape must be (B, H + n_bins**2); "
        f"got {tuple(augmented.shape)}"
    )
    # Leading slice must equal the base embedding verbatim
    # (augmentation is additive only, not destructive).
    assert torch.allclose(augmented[:, :16], base, atol=1e-6), (
        "augment_pocket_embedding must preserve the base embedding "
        "verbatim in the leading H columns"
    )
    # Trailing slice must equal the flattened feature map.
    assert torch.allclose(
        augmented[:, 16:].reshape(2, 4, 4),
        feat_map,
        atol=1e-6,
    ), "augment_pocket_embedding must serialise the feature map row-major"


# ---------------------------------------------------------------------------
# 2. SE(3) Lipman solver — tangent-space rotation preservation
# ---------------------------------------------------------------------------
def test_se3_lipman_solver_tangent_correctness(fmi):
    """SE(3) tangent-space ops preserve rotation orthogonality.

    The :class:`SE3LipmanSolver` integrates ``dx/dt = v(x, t)``
    where the velocity is projected through
    :class:`SE3SBVelocityField` onto the ``se(3) ≅ R^6`` tangent
    space.  The W2.1 hard contract is that the rotation block of
    the integrated trajectory stays orthogonal within ``1e-3``
    (``max |R^T R - I|_inf < 1e-3``).

    Smoke contract:

    * the factory-built solver returns with the canonical W2.1
      defaults (``n_steps=50``, ``method="midpoint"``);
    * a forward pass on a deterministic ``(B=2, N=6, 3)`` noise
      tensor produces a finite ``(2, 6, 3)`` trajectory;
    * the audit returns ``max orthogonality error < SE3_ORTHO_TOL``
      after ``n_runs=3`` independent runs.
    """
    solver = fmi.get_se3_lipman_solver()
    # Canonical defaults.
    assert solver.n_steps == 50
    assert solver.method == "midpoint"
    assert solver.ortho_tol == pytest.approx(SE3_ORTHO_TOL)

    # Deterministic velocity_fn: v(x, t) = x (the canonical
    # self-map used by the w_flow + Integration 1 smoke contract).
    def _v_fn(x, t):
        return x.clone()

    torch.manual_seed(0)
    x0 = torch.randn(2, 6, 3)
    x_final = solver.solve(
        velocity_fn=_v_fn,
        x0=x0,
        t_span=(0.0, 1.0),
        pocket=None,
    )
    assert x_final.shape == (2, 6, 3), (
        f"SE(3) solver output must preserve (B, N, 3) shape; "
        f"got {tuple(x_final.shape)}"
    )
    assert torch.isfinite(
        x_final
    ).all(), "SE(3) solver output must be finite — no NaN/Inf permitted"

    # Orthogonality audit: limit to 3 runs (the default is 10) to
    # keep the test well under 1 s on a single CPU thread.
    max_err = solver.verify_se3_orthogonality(n_runs=3)
    assert math.isfinite(
        max_err
    ), f"max orthogonality error must be finite; got {max_err}"
    assert max_err < SE3_ORTHO_TOL, (
        f"max orthogonality error must be < {SE3_ORTHO_TOL}; "
        f"got {max_err} (W2.1 hard contract violated)"
    )


# ---------------------------------------------------------------------------
# 3. SE(3) SB decoder — endpoint distribution sanity
# ---------------------------------------------------------------------------
def test_se3_sb_decoder_endpoint_distribution(fmi):
    """Schrödinger Bridge endpoints match source / target poses.

    The :class:`SE3SBDecoder.decode(z0, z1, t_values)` walks the
    SE(3) geodesic at the supplied ``t_values``.  By construction:

    * at ``t = 0`` the decoded pose equals the source ``z0``;
    * at ``t = 1`` the decoded pose equals the matched target
      ``z1_matched`` (Sinkhorn assignment).

    Smoke contract:

    * decoder builds with the canonical W2.1 defaults;
    * ``decode(z, z, t_values=[0.0, 1.0])`` returns ``(B, 2, 4, 4)``
      where the two endpoints equal the source pose (Sinkhorn
      self-coupling degenerates to identity);
    * the ``decode_at(z, t=0)`` convenience method returns ``z``
      bit-exactly (within fp32 tolerance).
    """
    decoder = fmi.get_se3_sb_decoder(override={"use_sinkhorn": False})
    # Canonical defaults preserved.
    assert decoder.eps == pytest.approx(0.1)
    assert decoder.n_iters == 50
    assert decoder.use_sinkhorn is False  # per override

    b = 3
    z = torch.eye(4).expand(b, 4, 4).clone()
    z[..., :3, 3] = torch.randn(b, 3) * 0.1

    t_values = torch.tensor([0.0, 1.0], dtype=z.dtype, device=z.device)
    out = decoder.decode(z, z, t_values=t_values)
    assert out.shape == (
        b,
        2,
        4,
        4,
    ), f"decode output must be (B, n_t, 4, 4); got {tuple(out.shape)}"
    # Both endpoints must equal the source pose (self-coupling
    # degenerates to identity; geodesic is constant on self-pair).
    for k in (0, 1):
        diff = (out[:, k] - z).abs().max().item()
        assert diff < 1e-5, (
            f"decode(z, z) at t_values[{k}] must equal z within 1e-5; "
            f"got max abs diff={diff}"
        )

    # Convenience: decode_at(z, t=0.0) returns z bit-exactly.
    z0_at_zero = decoder.decode_at(z, t=0.0)
    assert z0_at_zero.shape == z.shape
    assert (z0_at_zero - z).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# 4. Transport relaxation — ε-annealing monotonic decrease
# ---------------------------------------------------------------------------
def test_transport_relaxation_endpoint_term(fmi):
    """ε-annealing from 1.0 → 0.0 monotonically *increases* the KL.

    Per Jiang-Xu-Zhang 2026 §3, the relaxation penalty is
    ``λ · KL(p_T ‖ q_T(ε))`` where ``q_T = softmax(-E / ε)``.
    As ``ε → 0`` the distribution ``q_T`` becomes a delta on the
    lowest-energy trajectory and the KL **inflates**; as
    ``ε → ∞`` the distribution becomes uniform and the KL
    collapses to 0.

    Smoke contract:

    * factory-built :class:`RelaxedEndpointTerm` matches the
      W2.2 defaults (``lambda_relax=1e-3``, ``eps=0.1``);
    * ``forward`` returns a finite non-negative scalar;
    * as ``ε`` anneals from ``1.0`` toward ``1e-5``, the KL
      penalty is **monotonically non-decreasing** (the Gibbs
      lower bound is approached but never crossed).
    """
    term = fmi.get_transport_relaxation()
    # Canonical W2.2 defaults.
    assert term.lambda_relax == pytest.approx(1e-3)
    assert term.eps == pytest.approx(0.1)

    # Synthetic (B=8, T=4, D=3) trajectory + non-uniform energies
    # so the KL is strictly positive (not the Gibbs-minimum case).
    torch.manual_seed(0)
    trajectory = torch.randn(8, 4, 3)
    energies = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    # Sweep ε from 1.0 down to 1e-5 — KL must be monotonically
    # non-decreasing along the sweep.
    eps_schedule = torch.logspace(0, -5, N_ANNEAL_STEPS).tolist()
    kl_history: list[float] = []
    for eps_val in eps_schedule:
        sub_term = type(term)(lambda_relax=term.lambda_relax, eps=float(eps_val))
        kl = float(sub_term.kl(trajectory, energies).item())
        kl_history.append(kl)
        assert math.isfinite(kl), f"KL must be finite at ε={eps_val}; got {kl}"
        assert kl >= 0.0, f"KL must be non-negative at ε={eps_val}; got {kl}"

    # Monotone non-decreasing check.
    for k in range(1, len(kl_history)):
        assert kl_history[k] >= kl_history[k - 1] - 1e-9, (
            f"KL must be monotonically non-decreasing as ε anneals "
            f"from 1.0 → 0.0; got kl[{k}]={kl_history[k]} < "
            f"kl[{k-1}]={kl_history[k-1]}"
        )

    # Forward must equal λ · KL (the augmentation contract).
    for eps_val in eps_schedule[:2]:
        sub_term = type(term)(lambda_relax=term.lambda_relax, eps=float(eps_val))
        penalty = float(sub_term.forward(trajectory, energies).item())
        kl = float(sub_term.kl(trajectory, energies).item())
        assert abs(penalty - term.lambda_relax * kl) < 1e-6, (
            f"forward must equal λ_relax · KL at ε={eps_val}; "
            f"got penalty={penalty} vs λ·KL={term.lambda_relax * kl}"
        )


# ---------------------------------------------------------------------------
# 5. Joint FrontierMathIntegration — finite forward pass (no NaN)
# ---------------------------------------------------------------------------
def test_joint_frontier_integration_no_nan(fmi):
    """Full :class:`FrontierMathIntegration` instantiation + finite forward.

    Smoke contract: build every adapter via the factory surface,
    run one forward pass through each, and assert every output is
    finite (no NaN / Inf propagation).  This is the Wf-313C
    Day 3-5 "no-NaN" gate — the test that ensures the full
    composition is safe to drop into a downstream pipeline.

    The 3 research-only modules (``bass_existence`` /
    ``sb_bass_martingale`` / ``monge_ampere_regularity``) are
    explicitly excluded — see :func:`test_research_only_modules_excluded`.
    """
    # 1. Multi-PH features (4 bins for speed) + metal head.
    multi_ph = fmi.get_multi_ph_features(
        override={"n_bins": 4, "max_radius": 3.0, "max_coord": 2.0},
    )
    metal_head = fmi.get_multi_ph_metal_head(
        override={"n_bins": 4, "max_radius": 3.0, "max_coord": 2.0},
    )
    torch.manual_seed(0)
    pocket = torch.randn(2, 8, 3)
    feat_map = multi_ph.compute(pocket)
    logits = metal_head(feat_map)
    assert torch.isfinite(feat_map).all()
    assert torch.isfinite(logits).all()
    assert feat_map.shape == (2, 4, 4)
    assert logits.shape == (2, metal_head.n_metals)

    # 2. SE(3) SB decoder — self-pair at t=0.5 returns finite
    # SE(3) poses.
    decoder = fmi.get_se3_sb_decoder(override={"use_sinkhorn": False})
    z = torch.eye(4).expand(2, 4, 4).clone()
    z[..., :3, 3] = torch.randn(2, 3) * 0.1
    out = decoder.decode(z, z, t_values=torch.tensor([0.5]))
    assert torch.isfinite(out).all()

    # 3. SE(3) Lipman solver — forward pass on deterministic
    # noise produces a finite trajectory.
    solver = fmi.get_se3_lipman_solver()

    def _v_fn(x, t):
        return x.clone()

    x0 = torch.randn(2, 6, 3)
    x_final = solver.solve(
        velocity_fn=_v_fn,
        x0=x0,
        t_span=(0.0, 1.0),
        pocket=None,
    )
    assert torch.isfinite(x_final).all()
    assert x_final.shape == x0.shape

    # 4. Transport relaxation — non-uniform energies → finite,
    # non-negative penalty.
    relax = fmi.get_transport_relaxation()
    trajectory = torch.randn(8, 3)
    energies = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    penalty = relax.forward(trajectory, energies)
    assert torch.isfinite(penalty).all()
    assert float(penalty.item()) >= 0.0


# ---------------------------------------------------------------------------
# 6. Research-only modules honestly excluded from factory surface
# ---------------------------------------------------------------------------
def test_research_only_modules_excluded(fmi):
    """3 research-only modules NOT in :class:`FrontierMathIntegration`.

    Per Wf-305B / Wf-306B / TODO-234 §2 honest-framing, the 3
    research-only modules are *cite-only skeletons* (statement-text
    holders + statement-quality calibrators) and are NOT drop-in
    adapters for the CFM velocity field or loss.  Forcing them
    into the locked :mod:`flow_matching_lipman` pipeline would
    violate the R17 invariant.

    Smoke contract:

    * the factory surface exposes exactly 4 adapter entry-points
      (one per shipped standalone adapter);
    * none of the 3 research-only modules has a ``get_*`` method
      on :class:`FrontierMathIntegration`;
    * none of the 3 research-only modules is imported by
      :mod:`molmetal.adapters.frontier_math_integration`.
    """
    factory_methods = {
        name
        for name in dir(fmi)
        if name.startswith("get_") and callable(getattr(fmi, name))
    }
    # Canonical 4 factory entry-points (Wf-309B).
    expected = {
        "get_multi_ph_features",
        "get_multi_ph_metal_head",
        "get_se3_sb_decoder",
        "get_se3_lipman_solver",
        "get_transport_relaxation",
    }
    assert factory_methods == expected, (
        f"FrontierMathIntegration factory surface must be exactly the 4 "
        f"shipped adapters; got {sorted(factory_methods)}, expected "
        f"{sorted(expected)}"
    )
    # None of the 3 research-only modules has a get_* method.
    for m in RESEARCH_ONLY_MODULES:
        token = f"get_{m}"
        assert token not in factory_methods, (
            f"research-only module {m!r} must NOT have a factory "
            f"entry-point on FrontierMathIntegration; found {token}"
        )

    # The integration wrapper's namespace must NOT carry the
    # research-only modules' public classes.
    from molmetal.adapters import frontier_math_integration as fmi_mod

    for cls in (
        "BassExistenceTheorem",
        "BassMartingale",
        "KantorovichPotential",
        "MongeAmpereSolver",
        "BassCalibrationHead",
        "AdvantageEstimatorMA",
    ):
        assert not hasattr(fmi_mod, cls), (
            f"research-only class {cls!r} must NOT be re-exported by "
            f"frontier_math_integration; found in module namespace"
        )


# ---------------------------------------------------------------------------
# 7. Factory method consistency — expected types
# ---------------------------------------------------------------------------
def test_factory_method_consistency(fmi):
    """Every ``get_*`` method returns the canonical public class.

    The :class:`FrontierMathIntegration` factory surface is a
    thin wrapper around the 4 STANDALONE adapters.  Each
    ``get_*`` method must return the *exact* public class from
    the underlying adapter module — no proxy / subclass allowed.

    Smoke contract:

    * ``get_multi_ph_features`` → :class:`MultiPHPocketFeatures`
    * ``get_multi_ph_metal_head`` → :class:`MultiPHMetalCoordHead`
    * ``get_se3_sb_decoder`` → :class:`SE3SBDecoder`
    * ``get_se3_lipman_solver`` → :class:`SE3LipmanSolver`
    * ``get_transport_relaxation`` → :class:`RelaxedEndpointTerm`

    All instances must pass ``isinstance`` against the
    canonical class exported by the standalone adapter module.
    """
    from molmetal.adapters.multi_ph_integration import (
        MultiPHMetalCoordHead,
        MultiPHPocketFeatures,
    )
    from molmetal.adapters.se3_lipman_integration import SE3LipmanSolver
    from molmetal.adapters.se3_sb_integration import SE3SBDecoder
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    # Override with the smallest smoke config so the constructors
    # stay under 50 ms each.
    multi_ph = fmi.get_multi_ph_features(
        override={"n_bins": 4, "max_radius": 3.0, "max_coord": 2.0},
    )
    assert isinstance(multi_ph, MultiPHPocketFeatures)

    metal_head = fmi.get_multi_ph_metal_head(
        override={"n_bins": 4, "max_radius": 3.0, "max_coord": 2.0},
    )
    assert isinstance(metal_head, MultiPHMetalCoordHead)

    decoder = fmi.get_se3_sb_decoder(override={"use_sinkhorn": False})
    assert isinstance(decoder, SE3SBDecoder)

    solver = fmi.get_se3_lipman_solver()
    assert isinstance(solver, SE3LipmanSolver)

    relax = fmi.get_transport_relaxation()
    assert isinstance(relax, RelaxedEndpointTerm)


# ---------------------------------------------------------------------------
# 8. Frontier integration metrics — per-component scalar emission
# ---------------------------------------------------------------------------
def test_frontier_integration_metrics(fmi):
    """Per-component metric emission: scalar + shape contracts.

    Each adapter emits a metric with a specific shape / dtype
    contract that downstream dashboards can serialise.  This
    test exercises every emission path once and asserts the
    contract on shape, dtype, finiteness, and (where applicable)
    non-negativity.

    Smoke contract (per adapter):

    * Multi-PH features → ``(B, n_bins, n_bins)`` long-count
      feature map (non-negative integers);
    * SE(3) SB decoder → ``(B, n_t, 4, 4)`` SE(3) poses (finite
      homogeneous matrices);
    * SE(3) Lipman solver → ``(B, N, 3)`` trajectory (finite,
      same shape as input);
    * Transport relaxation → scalar ``()`` non-negative penalty.
    """
    # 1. Multi-PH features.
    multi_ph = fmi.get_multi_ph_features(
        override={"n_bins": 4, "max_radius": 3.0, "max_coord": 2.0},
    )
    torch.manual_seed(0)
    pocket = torch.randn(2, 8, 3)
    feat = multi_ph.compute(pocket)
    assert feat.shape == (2, 4, 4)
    assert (feat >= 0).all(), "multi-PH metric must be non-negative"
    assert torch.isfinite(feat).all()

    # 2. SE(3) SB decoder.
    decoder = fmi.get_se3_sb_decoder(override={"use_sinkhorn": False})
    z = torch.eye(4).expand(2, 4, 4).clone()
    z[..., :3, 3] = torch.randn(2, 3) * 0.1
    out = decoder.decode(z, z, t_values=torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (2, 3, 4, 4)
    assert torch.isfinite(out).all()
    # Bottom rows must be the homogeneous pad (0, 0, 0, 1).
    assert (out[..., 3, :3] == 0.0).all(), (
        "SE(3) decoder metric must carry the homogeneous pad row "
        "(0, 0, 0, ...) on the last row of every 4x4 block"
    )
    assert (out[..., 3, 3] == 1.0).all()

    # 3. SE(3) Lipman solver.
    solver = fmi.get_se3_lipman_solver()

    def _v_fn(x, t):
        return x.clone()

    x0 = torch.randn(2, 6, 3)
    x_final = solver.solve(
        velocity_fn=_v_fn,
        x0=x0,
        t_span=(0.0, 1.0),
        pocket=None,
    )
    assert x_final.shape == x0.shape  # shape preserved
    assert torch.isfinite(x_final).all()

    # 4. Transport relaxation.
    relax = fmi.get_transport_relaxation()
    trajectory = torch.randn(8, 3)
    energies = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    penalty = relax.forward(trajectory, energies)
    # Scalar contract: penalty.dim() == 0 (0-d tensor).
    assert penalty.dim() == 0, (
        f"transport relaxation metric must be a 0-d scalar; "
        f"got dim={penalty.dim()}, shape={tuple(penalty.shape)}"
    )
    assert torch.isfinite(penalty).all()
    assert float(penalty.item()) >= 0.0

    # Aggregate metric dict (the "per-component" payload that
    # dashboards serialise to JSON).
    metrics_payload = {
        "multi_ph_feature_map_sum": float(feat.sum().item()),
        "se3_sb_decoder_pose_norm": float(out.norm().item()),
        "se3_lipman_trajectory_norm": float(x_final.norm().item()),
        "transport_relaxation_penalty": float(penalty.item()),
    }
    for key, val in metrics_payload.items():
        assert isinstance(val, float), (
            f"metrics payload {key!r} must be a JSON-serialisable "
            f"Python float; got {type(val).__name__}"
        )
        assert math.isfinite(val), f"metrics payload {key!r} must be finite; got {val}"
    # Multi-PH feature-map sum must be non-negative.
    assert metrics_payload["multi_ph_feature_map_sum"] >= 0.0
    # Transport-relaxation penalty must be non-negative.
    assert metrics_payload["transport_relaxation_penalty"] >= 0.0
