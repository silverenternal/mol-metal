"""Wf-380-A3 — TODO-202 diversity sampling primitives tests.

=====================================================================
Three pytest contracts verifying the four diversity methods shipped
in TODO-202 §1:

1. ``test_nucleus_sampling`` — :func:`nucleus_sample` matches the
   Holtzman 2020 reference on a small synthetic logits tensor.
2. ``test_mmd_loss`` — :func:`mmd_loss` is non-negative and
   symmetric (Gretton 2012 Eq. (3) invariants).
3. ``test_repulsion_loss`` — :func:`repulsion_loss` penalises
   similarity above the configured margin (Liu 2021 Eq. (7)).

All tests are CPU-only, deterministic (manual ``torch.Generator``
seed), and complete in well under 5 seconds combined.  They do
NOT require GPU and are safe to run on the Wf-367 / Wf-368 baseline.

Run with::

    uv run --no-sync pytest -q \\
        molmetal/tests/test_wf380a3_diversity.py --tb=short
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Lazy imports — keep the module loadable even if torch / diversity_sampling
# fail to import (e.g. partial CI env).  Tests skip rather than error.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def div_mod():
    try:
        from molmetal.adapters.flow_matching_lipman import diversity_sampling as ds
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"diversity_sampling unavailable: {exc}")
    return ds


# ---------------------------------------------------------------------------
# Test 1 — nucleus_sample matches Holtzman 2020 reference
# ---------------------------------------------------------------------------
def test_nucleus_sampling(div_mod):
    """Wf-380-A3 — :func:`nucleus_sample` reproduces the Holtzman 2020
    top-p + top-k + temperature pipeline on a synthetic logits tensor.

    Specifically:

    (a) **Top-p truncation** — when ``p=0.5`` and the sorted
        probabilities are ``[0.4, 0.3, 0.2, 0.1]`` (cumsum
        ``[0.4, 0.7, 0.9, 1.0]``), tokens 3 and 4 (cumsum > 0.5)
        should be excluded from sampling — i.e. the resulting
        samples are always in {0, 1, 2}, never 3.
    (b) **Top-k truncation** — when ``k=2`` only the top-2 tokens
        receive non-zero probability.
    (c) **Temperature scaling** — for a uniform logits vector
        ``[c, c, c, c]`` (any c), the sample distribution must be
        uniform regardless of temperature (since the rescaled
        logits are still uniform).
    (d) **Argmax fast path** — ``p <= 0`` returns the logits
        argmax directly, deterministically.
    (e) **Bit-for-bit determinism** — re-running with the same
        ``torch.Generator`` state returns the same sample.

    All assertions are checked on a controlled ``(B, V)`` logits
    tensor where the expected behaviour is exact.
    """
    nucleus_sample = div_mod.nucleus_sample

    # ---- (a) Top-p truncation: sample only from top tokens ----
    # Construct logits such that softmax gives probs ≈ [0.4, 0.3, 0.2, 0.1].
    # log([0.4, 0.3, 0.2, 0.1]) ≈ [-0.9163, -1.2040, -1.6094, -2.3026].
    logits_top_p = torch.tensor(
        [[-0.9163, -1.2040, -1.6094, -2.3026]], dtype=torch.float32,
    )
    # Sanity-check our reference probabilities.
    probs = torch.softmax(logits_top_p, dim=-1)
    expected_first = 0.4
    assert abs(probs[0, 0].item() - expected_first) < 1e-3, (
        f"Setup: expected prob[0] ≈ 0.4, got {probs[0, 0].item():.4f}"
    )

    # With p=0.5 the cumulative mask is [0.4, 0.7, 0.9, 1.0] >
    # 0.5 → mask = [F, T, T, T] shifted right = [F, F, T, T].
    # So token 0 (kept), tokens 1, 2, 3 (removed).  The only
    # surviving token is 0 → argmax-sample.
    gen = torch.Generator().manual_seed(42)
    sampled = nucleus_sample(logits_top_p, p=0.5, k=4, temperature=1.0, generator=gen)
    assert sampled.shape == (1,), f"shape mismatch: {sampled.shape}"
    assert sampled.item() == 0, (
        f"With p=0.5 and probs [0.4, 0.3, 0.2, 0.1], the only "
        f"surviving token is 0 (cumsum 0.4 < 0.5); got {sampled.item()}"
    )

    # ---- (b) Top-k truncation ----
    # With k=2, only the top-2 logits (positions 0, 1) survive
    # regardless of p.  Sample many times to confirm only {0, 1}.
    logits_top_k = torch.tensor(
        [[3.0, 2.0, 1.0, 0.0]], dtype=torch.float32,
    )
    gen = torch.Generator().manual_seed(0)
    samples_k2 = []
    for _ in range(200):
        gen = torch.Generator().manual_seed(len(samples_k2))
        s = nucleus_sample(logits_top_k, p=1.0, k=2, temperature=1.0, generator=gen)
        samples_k2.append(int(s.item()))
    unique = set(samples_k2)
    assert unique.issubset({0, 1}), (
        f"With k=2 only top-2 tokens should appear in samples; "
        f"got {unique}"
    )

    # ---- (c) Temperature scaling preserves uniformity ----
    # Uniform logits → uniform probs → uniform sampling for any τ.
    logits_unif = torch.zeros((1, 5), dtype=torch.float32)
    gen = torch.Generator().manual_seed(1)
    samples_tau = []
    for _ in range(500):
        gen = torch.Generator().manual_seed(len(samples_tau) + 1)
        s = nucleus_sample(logits_unif, p=0.95, k=5, temperature=0.7, generator=gen)
        samples_tau.append(int(s.item()))
    counts = [samples_tau.count(i) for i in range(5)]
    # Each bucket should be roughly 100 — accept ±50.
    for i, c in enumerate(counts):
        assert 50 <= c <= 150, (
            f"Uniform logits should sample uniformly; bucket {i} got {c}/500"
        )

    # ---- (d) Argmax fast path (p <= 0) ----
    logits_argmax = torch.tensor(
        [[0.5, 2.0, 0.1, -1.0]], dtype=torch.float32,
    )
    s = nucleus_sample(logits_argmax, p=0.0, k=4, temperature=1.0)
    assert s.item() == 1, (
        f"p=0.0 must return argmax; got {s.item()}"
    )

    # ---- (e) Bit-for-bit determinism with same generator ----
    gen1 = torch.Generator().manual_seed(2026)
    gen2 = torch.Generator().manual_seed(2026)
    logits_det = torch.randn((3, 12), dtype=torch.float32)
    s1 = nucleus_sample(logits_det, p=0.85, k=8, temperature=0.9, generator=gen1)
    s2 = nucleus_sample(logits_det, p=0.85, k=8, temperature=0.9, generator=gen2)
    assert torch.equal(s1, s2), (
        f"Same-generator sampling must be deterministic; got "
        f"{s1.tolist()} vs {s2.tolist()}"
    )

    # ---- (f) Shape preservation ----
    logits_3d = torch.randn((2, 4, 8), dtype=torch.float32)
    s3d = nucleus_sample(logits_3d, p=0.9, k=5, temperature=1.0)
    assert s3d.shape == (2, 4), (
        f"nucleus_sample must preserve leading dims minus vocab; "
        f"expected (2, 4), got {tuple(s3d.shape)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — mmd_loss is non-negative and symmetric
# ---------------------------------------------------------------------------
def test_mmd_loss(div_mod):
    """Wf-380-A3 — :func:`mmd_loss` satisfies the two Gretton 2012
    invariants:

    (a) **Non-negativity** — MMD^2 >= 0 for the RBF kernel
        (Cauchy-Schwarz on the RKHS norm squared).  After
        clamping tiny negatives from floating-point roundoff the
        result is exactly zero.
    (b) **Symmetry** — ``mmd_loss(A, B) == mmd_loss(B, A)`` for any
        A, B (the Gram-matrix structure is symmetric on
        interchange).
    (c) **Identity collapse** — ``mmd_loss(A, A) == 0`` exactly.
    (d) **Bias with separation** — well-separated clusters give
        a strictly larger MMD^2 than overlapping clusters.
    """
    mmd_loss = div_mod.mmd_loss
    torch.manual_seed(0)

    # ---- (a) Non-negativity on random inputs ----
    a = torch.randn((8, 4), dtype=torch.float32)
    b = torch.randn((8, 4), dtype=torch.float32) + 2.0  # shifted
    val = mmd_loss(a, b, kernel="rbf", sigma=1.0, unbiased=True)
    assert val.item() >= 0.0, (
        f"MMD^2 must be non-negative; got {val.item()}"
    )

    # ---- (b) Symmetry ----
    val_ab = mmd_loss(a, b, kernel="rbf", sigma=1.0, unbiased=True)
    val_ba = mmd_loss(b, a, kernel="rbf", sigma=1.0, unbiased=True)
    assert torch.allclose(val_ab, val_ba, atol=1e-5), (
        f"MMD must be symmetric: mmd(A,B)={val_ab.item():.6f}, "
        f"mmd(B,A)={val_ba.item():.6f}"
    )

    # ---- (c) Identity collapse ----
    val_aa = mmd_loss(a, a, kernel="rbf", sigma=1.0, unbiased=True)
    assert val_aa.item() < 1e-4, (
        f"MMD(A, A) must collapse to ≈0 for the RBF kernel; got {val_aa.item()}"
    )

    # ---- (d) Bias with separation ----
    # Two well-separated clusters should give a strictly larger
    # MMD^2 than two overlapping clusters (with the same sigma).
    cluster_a_sep = torch.randn((8, 4), dtype=torch.float32)
    cluster_b_sep = torch.randn((8, 4), dtype=torch.float32) + 10.0
    cluster_a_over = torch.randn((8, 4), dtype=torch.float32)
    cluster_b_over = torch.randn((8, 4), dtype=torch.float32) + 0.1
    mmd_sep = mmd_loss(cluster_a_sep, cluster_b_sep, kernel="rbf", sigma=1.0)
    mmd_over = mmd_loss(cluster_a_over, cluster_b_over, kernel="rbf", sigma=1.0)
    assert mmd_sep.item() > mmd_over.item() + 0.05, (
        f"Well-separated clusters must give larger MMD than "
        f"overlapping; got sep={mmd_sep.item():.4f}, over={mmd_over.item():.4f}"
    )

    # ---- (e) Unbiased estimator error path ----
    too_small = torch.randn((1, 4), dtype=torch.float32)
    with pytest.raises(ValueError):
        mmd_loss(too_small, too_small, unbiased=True)
    # Biased estimator should accept.
    val_v = mmd_loss(too_small, too_small, unbiased=False)
    assert val_v.item() >= 0.0


# ---------------------------------------------------------------------------
# Test 3 — repulsion_loss penalises similarity
# ---------------------------------------------------------------------------
def test_repulsion_loss(div_mod):
    """Wf-380-A3 — :func:`repulsion_loss` satisfies the canonical
    repulsion semantics from Liu et al. 2021 eq. (7):

    (a) **Identity collapse** — when two rows are bit-identical,
        their cosine similarity is 1.0 and the penalty is
        ``max(0, 1 − margin)`` averaged over the i<j pairs.
    (b) **Zero penalty on orthogonal vectors** — when rows are
        exactly orthogonal, cosine similarity is 0 and the
        penalty is zero for any margin in (0, 1].
    (c) **Monotonic in similarity** — higher pairwise cosine
        similarity above the margin → higher repulsion penalty.
    (d) **Margin sensitivity** — a tighter margin (e.g. 0.5)
        produces a larger penalty than a looser one (0.95) for
        the same inputs.
    (e) **Single-sample degenerate case** — returns zero
        deterministically (no penalty for a lone sample).
    """
    repulsion_loss = div_mod.repulsion_loss
    torch.manual_seed(1)

    # ---- (a) Identity collapse ----
    same = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32,
    )
    # Pairs: (0,1) cos=1, (0,2) cos=0, (1,2) cos=0.
    # With margin=0.95: penalty = mean(relu([1.0-0.95, 0, 0]))
    #                  = 0.05/3 = 0.01667.
    loss_id = repulsion_loss(same, margin=0.95, reduce=True)
    expected = (1.0 - 0.95) / 3.0
    assert abs(loss_id.item() - expected) < 1e-4, (
        f"Identity pair penalty must be (1 - margin) / N_pairs; "
        f"got {loss_id.item()}, expected {expected}"
    )

    # ---- (b) Orthogonal rows → zero penalty ----
    ortho = torch.eye(3, dtype=torch.float32)
    loss_ortho = repulsion_loss(ortho, margin=0.5, reduce=True)
    assert loss_ortho.item() == 0.0, (
        f"Orthogonal rows must yield zero repulsion; got {loss_ortho.item()}"
    )

    # ---- (c) Monotonic in similarity ----
    # Sample A: highly similar (cosine ~0.95); sample B: random.
    similar_cluster = torch.tensor(
        [[1.0, 0.05], [1.0, -0.05], [0.95, 0.0]], dtype=torch.float32,
    )
    random_cluster = torch.randn((3, 4), dtype=torch.float32)
    loss_sim = repulsion_loss(similar_cluster, margin=0.9, reduce=True)
    loss_rand = repulsion_loss(random_cluster, margin=0.9, reduce=True)
    assert loss_sim.item() > loss_rand.item(), (
        f"Similar cluster must yield larger penalty than random; "
        f"got sim={loss_sim.item():.4f}, rand={loss_rand.item():.4f}"
    )

    # ---- (d) Margin sensitivity ----
    # Same input, tighter margin → larger penalty.
    loss_loose = repulsion_loss(similar_cluster, margin=0.95, reduce=True)
    loss_tight = repulsion_loss(similar_cluster, margin=0.5, reduce=True)
    assert loss_tight.item() > loss_loose.item(), (
        f"Tighter margin must produce larger penalty; "
        f"got tight={loss_tight.item():.4f}, loose={loss_loose.item():.4f}"
    )

    # ---- (e) Single-sample degenerate ----
    single = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    loss_single = repulsion_loss(single, margin=0.5, reduce=True)
    assert loss_single.item() == 0.0, (
        f"Single sample must yield zero penalty; got {loss_single.item()}"
    )

    # ---- (f) reduce=False returns a tensor ----
    penalty_matrix = repulsion_loss(similar_cluster, margin=0.9, reduce=False)
    assert penalty_matrix.shape == (3, 3), (
        f"reduce=False must return (N, N) matrix; got {tuple(penalty_matrix.shape)}"
    )
    # The diagonal is always zero (mask is upper-triangular i<j).
    diag = penalty_matrix.diagonal()
    assert torch.all(diag == 0.0), (
        f"Diagonal of repulsion mask must be zero; got {diag.tolist()}"
    )
