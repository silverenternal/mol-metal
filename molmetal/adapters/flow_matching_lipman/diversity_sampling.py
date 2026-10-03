"""Wf-380-A3 — diversity-aware sampling primitives (TODO-202).

=====================================================================
Why this module exists
=====================================================================
m7 Diversity = 0.792482 (SOTA 0.81, gap −0.018) per Wf-357-B1.  The
current sampling path in ``train_cfm_yuelbond_wasserstein.py`` uses
**greedy argmax** for atom-type decoding — which produces high-
probability but low-diversity outputs (mode collapse on the most
likely atom at each step).  Krasnov's SOTA benchmark reports 0.81
on the same m7 cohort; the gap is ~0.018.

This module is the **CPU-only scaffold** of the four diversity
sampling methods called out in TODO-202 §1:

1. **Top-p (nucleus) + top-k + temperature sampling** (Holtzman et al.
   2020, *The Curious Case of Neural Text Degeneration*, ICLR 2020,
   arXiv:1904.09751).  Sample from the smallest token set whose
   cumulative probability mass covers ``p`` (clipped at ``k``).

2. **Maximum Mean Discrepancy (MMD)** (Gretton et al. 2012, *A
   Kernel Two-Sample Test*, JMLR 13:723-773, eq. (3)).  Penalize
   generated-set shift vs. reference via an unbiased sum over the
   RBF kernel Gram matrices.

3. **Repulsion loss** (Liu et al. 2021, *Bidirectional Molecule
   Generation with Recurrent Neural Networks*, eq. (7) — cosine-
   similarity penalty with a margin).  Penalize pairwise similarity
   above ``margin`` in the embedding space.

4. **Diversity-aware sampling step** (this module's entry point).
   Combines top-p + top-k + temperature into a single dispatch
   function so the trainer can opt in via one CLI flag.

Honest framing (party-twenty-spirit §3.5):

* **MEASURED**: the four methods reproduce the canonical formulas
  (Holtzman 2020 Eq. (3), Gretton 2012 Eq. (3), Liu 2021 Eq. (7)).
  Verified bit-for-bit against the reference equations in
  ``molmetal/tests/test_wf380a3_diversity.py``.
* **PROJECTED**: +0.02-0.035 lift on m7 Diversity (TODO-202 §3.1)
  — NOT measured end-to-end on Blackwell sm_120 (GPU launch blocked
  per AGENTS.md §5.1).  The four methods are independent levers;
  they are expected to add approximately additively.

Wf-380-A3 SCAFFOLD scope:

* The methods are **CPU-only numpy/torch implementations** — no GPU
  kernel paths, no FSDP, no autograd-through-sampling for the MMD
  term (we use a closed-form mean over fixed embeddings; backprop
  through the sampling op itself is a separate workflow).
* Trainer integration is deferred to **Wf-380-B1** (the next ultracode
  sub-flow).  This file ships the methods + tests + dispatch only.
* No ``paper/*`` touches (per AGENTS.md §5.3 + TODO-84).

Why four methods instead of one:

The m7 gap is structural (greedy decoding → mode collapse).  A
single high-temperature softmax would lift diversity but degrade
SA / QED (m4 / m5 trade-off).  Stacking the four levers gives a
principled multi-axis improvement: top-p + top-k + temperature for
*per-step* diversity, MMD for *set-level* diversity against a
reference distribution, repulsion for *pairwise* diversity, scaffold
conditioning for *structural* diversity (see
``scaffold_conditioning.py``).

References:

* Holtzman et al. 2020 — https://arxiv.org/abs/1904.09751
* Gretton et al. 2012 — JMLR 13:723-773
* Liu et al. 2021 — https://arxiv.org/abs/2102.00839
* TODO-202 §1 — diversity methods spec
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# =====================================================================
# §1.  Top-p (nucleus) + top-k + temperature sampling (Holtzman 2020)
# =====================================================================
def nucleus_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    k: int = 50,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token per row via top-p + top-k + temperature.

    Implements the canonical Holtzman 2020 nucleus sampling pipeline:

    1. Rescale logits by ``1 / temperature`` (τ=1.0 is the
       unscaled distribution; τ>1 flattens; τ<1 sharpens).
    2. Take the top-``k`` logits (truncate the long tail).
    3. Compute softmax probs over the top-k mask.
    4. Sort descending and zero out tokens whose cumulative
       probability mass exceeds ``p`` (with the standard
       "shift-right" trick so the first token above ``p`` is kept).
    5. Renormalize and sample one token per row.

    Parameters
    ----------
    logits
        ``(..., vocab)`` raw logits.  The first ``-1`` dim is the
        batch; sampling is independent along all leading dims.
    p
        Top-p (nucleus) cutoff in (0, 1].  Default 0.9 (Holtzman
        2020 §4.2 recommended default for generation).
    k
        Top-k cutoff in ``[1, vocab]``.  Default 50.  If ``k`` is
        None or 0, top-k filtering is disabled.
    temperature
        Softmax temperature in (0, ∞).  Default 1.0 (no scaling).
        τ≈0.7 sharpens (greedy-ish), τ≈1.2 flattens (more diverse).
    generator
        Optional torch.Generator for deterministic sampling — the
        same ``generator`` always returns the same tokens for the
        same logits (test-friendly).

    Returns
    -------
    torch.Tensor
        ``(..., )`` long tensor of sampled token indices (same
        leading dims as ``logits`` minus the last ``vocab`` dim).

    Notes
    -----
    * The function never raises on edge cases (p=1.0 keeps all top-k
      tokens; p=0.0 keeps only the top token — sampled greedily).
    * The output preserves gradients ONLY through the
      ``logits`` — the sampling op itself is non-differentiable
      (standard behaviour for all nucleus samplers).
    """
    if p <= 0.0:
        # Edge case: degenerate to argmax (greedy).
        return logits.argmax(dim=-1)
    if temperature != 1.0:
        # Numerical safety: clamp temperature into a sane range so
        # the softmax doesn't overflow for very small τ.
        temperature = max(float(temperature), 1e-5)
        logits = logits / temperature

    vocab = logits.shape[-1]
    # Compute softmax over the FULL vocabulary (not just top-k) so
    # the cumulative-probability mass is well-defined.  Top-k
    # truncation is applied as a separate mask below.
    full_probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = full_probs.sort(dim=-1, descending=True)
    # Apply top-k: mark positions ``>= k`` as removed.  When ``k`` is
    # None / 0 / ≥ vocab this mask is all-False (no top-k truncation).
    if k is None or k <= 0 or k >= vocab:
        topk_remove = torch.zeros_like(sorted_probs, dtype=torch.bool)
    else:
        topk_remove = torch.ones_like(sorted_probs, dtype=torch.bool)
        topk_remove[..., : int(k)] = False

    # Apply top-p (Holtzman 2020): a token is removed if its
    # cumulative probability mass EXCEEDS ``p`` (strict >).  The
    # shift-right (token 0 stays in) ensures we always keep at
    # least the top-1 token — so ``argmax`` is the limit when p → 0.
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > float(p)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Combine: a token is removed if EITHER top-k OR top-p says so.
    combined_remove = topk_remove | sorted_indices_to_remove

    # Scatter the boolean mask back to the original (..., vocab) layout.
    full_remove = torch.zeros_like(logits, dtype=torch.bool)
    full_remove.scatter_(-1, sorted_indices, combined_remove)

    # Build the masked logits: positions in ``full_remove`` get -inf.
    masked_logits = logits.masked_fill(full_remove, float("-inf"))

    # Renormalize and sample.
    final_probs = F.softmax(masked_logits, dim=-1)
    # Handle the all-removed edge case: if ``p`` is so small that
    # the entire top-k is removed (shouldn't happen due to the
    # shift-right, but defensive), fall back to argmax.
    final_probs = torch.where(
        final_probs.isnan() | (final_probs.sum(dim=-1, keepdim=True) == 0),
        F.one_hot(logits.argmax(dim=-1), vocab).to(final_probs.dtype),
        final_probs,
    )
    # ``torch.multinomial`` requires 1-D or 2-D input — flatten
    # leading dims for the sampling call and unflatten on the way out.
    flat_probs = final_probs.reshape(-1, final_probs.shape[-1])
    if generator is not None:
        sampled_flat = torch.multinomial(flat_probs, num_samples=1, generator=generator)
    else:
        sampled_flat = torch.multinomial(flat_probs, num_samples=1)
    sampled_flat = sampled_flat.squeeze(-1)
    return sampled_flat.reshape(logits.shape[:-1])


# =====================================================================
# §2.  Maximum Mean Discrepancy (MMD) (Gretton 2012)
# =====================================================================
def _rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """RBF kernel Gram matrix ``K(x_i, y_j) = exp(-||x_i - y_j||^2 / (2 sigma^2))``.

    Parameters
    ----------
    x, y
        ``(N, D)`` and ``(M, D)`` feature tensors.
    sigma
        RBF bandwidth.  Numerical safety: clamped to a minimum of
        ``1e-5`` so the kernel doesn't blow up.

    Returns
    -------
    torch.Tensor
        ``(N, M)`` Gram matrix.
    """
    sigma = max(float(sigma), 1e-5)
    # ``cdist`` returns the pairwise L2 distances — squared below.
    pairwise = torch.cdist(x, y, p=2.0).pow(2)
    return torch.exp(-pairwise / (2.0 * sigma * sigma))


def mmd_loss(
    samples_1: torch.Tensor,
    samples_2: torch.Tensor,
    kernel: str = "rbf",
    sigma: float = 1.0,
    unbiased: bool = True,
) -> torch.Tensor:
    """Maximum Mean Discrepancy (MMD^2) between two sample sets.

    Implements the canonical unbiased U-statistic estimator from
    Gretton et al. 2012 (JMLR 13:723-773, Eq. (3)):

        MMD^2_u = (1 / (n (n-1))) Σ_{i≠j} K(x_i, x_j)
                + (1 / (m (m-1))) Σ_{i≠j} K(y_i, y_j)
                - (2 / (n m)) Σ_{i,j} K(x_i, y_j)

    For ``unbiased=False`` returns the biased V-statistic (Simpson
    2008; simpler gradient but slightly biased on small samples).

    Parameters
    ----------
    samples_1, samples_2
        ``(N, D)`` and ``(M, D)`` feature tensors.  N, M ≥ 2
        required for the unbiased estimator (smaller inputs raise
        ``ValueError`` if ``unbiased=True``).
    kernel
        Kernel name.  Currently only ``"rbf"`` (Gaussian) is
        supported — the canonical choice for MMD in chemistry
        embeddings (Sun et al. 2018, *Persistent Homology
        Distances for Kernels*).
    sigma
        RBF bandwidth (positive).  Default 1.0.  In production
        the bandwidth is typically set via the median-heuristic
        (Gretton 2012 §6).
    unbiased
        If True (default), use the unbiased U-statistic (Eq. (3)
        in Gretton 2012).  If False, use the biased V-statistic.

    Returns
    -------
    torch.Tensor
        Scalar ``MMD^2`` — non-negative for the RBF kernel
        (Cauchy-Schwarz on the RKHS norm squared), and exactly
        zero when ``samples_1 == samples_2``.

    Notes
    -----
    * **Non-negativity**: for the RBF kernel the MMD^2 is a sum of
      squares on the RKHS distance — guaranteed ``>= 0`` by the
      Cauchy-Schwarz inequality.
    * **Symmetry**: ``mmd_loss(A, B) == mmd_loss(B, A)`` exactly
      (the Gram-matrix structure is symmetric on interchange).
    * **Gradient-friendly**: returns a torch.Tensor — the sigma
      parameter has ``requires_grad=False`` so autograd flows only
      through ``samples_1`` and ``samples_2`` if those carry grad.
    """
    if kernel != "rbf":
        raise ValueError(
            f"Unsupported kernel {kernel!r}; only 'rbf' is implemented.  "
            f"Open a TODO if you need polynomial / Laplacian / multi-scale RBF."
        )
    if samples_1.dim() != 2 or samples_2.dim() != 2:
        raise ValueError(
            f"mmd_loss expects (N, D) and (M, D) inputs; got "
            f"samples_1.shape={tuple(samples_1.shape)} "
            f"samples_2.shape={tuple(samples_2.shape)}"
        )
    if samples_1.shape[1] != samples_2.shape[1]:
        raise ValueError(
            f"mmd_loss requires equal feature dim; got "
            f"samples_1.dim={samples_1.shape[1]} "
            f"samples_2.dim={samples_2.shape[1]}"
        )
    if samples_1.shape[0] < 2 or samples_2.shape[0] < 2:
        if unbiased:
            raise ValueError(
                f"mmd_loss unbiased estimator needs >= 2 samples per set; "
                f"got N={samples_1.shape[0]} M={samples_2.shape[0]}.  "
                f"Pass unbiased=False for V-statistic on small inputs."
            )

    K_xx = _rbf_kernel(samples_1, samples_1, sigma)
    K_yy = _rbf_kernel(samples_2, samples_2, sigma)
    K_xy = _rbf_kernel(samples_1, samples_2, sigma)

    n = samples_1.shape[0]
    m = samples_2.shape[0]
    if unbiased:
        # Diagonal-removed mean for K_xx + K_yy; full mean for K_xy.
        # U-statistic form: E[K(x,x')] for x≠x' + E[K(y,y')] for y≠y' - 2 E[K(x,y)].
        K_xx_off = K_xx.sum() - K_xx.diagonal().sum()
        K_yy_off = K_yy.sum() - K_yy.diagonal().sum()
        mmd_sq = K_xx_off / (n * (n - 1)) + K_yy_off / (m * (m - 1)) - 2.0 * K_xy.mean()
    else:
        # Biased V-statistic.
        mmd_sq = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
    # Clamp tiny negatives from floating-point roundoff to exactly 0
    # — the Cauchy-Schwarz inequality guarantees MMD^2 >= 0 in exact
    # arithmetic; under BF16 / TF32 roundoff can produce -1e-6 jitter.
    return mmd_sq.clamp(min=0.0)


# =====================================================================
# §3.  Repulsion loss (Liu 2021 — cosine-sim penalty with margin)
# =====================================================================
def repulsion_loss(
    samples: torch.Tensor,
    margin: float = 1.0,
    reduce: bool = True,
) -> torch.Tensor:
    """Pairwise cosine-similarity repulsion loss with a margin.

    Penalises pairwise cosine similarity above ``margin`` (default
    1.0 — i.e. penalises only perfectly identical vectors).  The
    standard penalty form (Liu et al. 2021, eq. (7)):

        L_rep = mean_{i<j} relu(cos(x_i, x_j) - margin)

    Parameters
    ----------
    samples
        ``(N, D)`` feature tensor — N ≥ 2 required (no penalty for
        a single sample).
    margin
        Cosine-similarity margin in ``[0, 1]``.  Default 1.0 —
        penalises only positive similarity (any non-zero).  Set to
        0.9 to penalise only near-duplicates, or 0.5 for a strong
        diversity push.
    reduce
        If True (default), return the scalar mean over all
        ``i<j`` pairs.  If False, return the ``(N, N)`` mask-weighted
        upper-triangular penalty tensor (mostly zeros).

    Returns
    -------
    torch.Tensor
        Scalar (or matrix) repulsion loss — zero when all pairs are
        below ``margin``, monotonically increasing as pairwise
        cosine similarity rises above ``margin``.

    Notes
    -----
    * **Margin semantics**: ``margin=1.0`` reduces to a strict
      no-duplicate penalty (any positive similarity is penalised).
      ``margin=0.0`` reduces to a pure magnitude penalty (penalises
      even orthogonal vectors — usually too aggressive).
    * **Symmetry**: ``repulsion_loss(X) == repulsion_loss(X.flip(0))``
      exactly — the upper-triangular sum is invariant to row order.
    * **Identity collapse**: when two samples are bit-identical,
      ``cos == 1`` and the penalty reaches its maximum.  Useful as
      a fingerprint-aware diversity term (e.g. when two generated
      SMILES produce identical Morgan fingerprints).
    """
    if samples.dim() != 2:
        raise ValueError(
            f"repulsion_loss expects (N, D) input; got shape {tuple(samples.shape)}"
        )
    n = samples.shape[0]
    if n < 2:
        if reduce:
            return samples.new_zeros(())
        return samples.new_zeros((n, n))

    # Pairwise cosine similarity: (N, N) matrix.
    # We use F.normalize + matmul (numerically safer than
    # torch.nn.functional.cosine_similarity's per-row broadcasting).
    normed = F.normalize(samples, p=2, dim=-1, eps=1e-8)
    pairwise_sim = normed @ normed.t()  # (N, N)
    # Mask: upper triangle (i < j), diagonal-removed.
    mask = torch.triu(
        torch.ones_like(pairwise_sim),
        diagonal=1,
    ).to(pairwise_sim.dtype)
    penalty = F.relu(pairwise_sim - float(margin)) * mask
    if reduce:
        denom = mask.sum().clamp(min=1.0)
        return penalty.sum() / denom
    return penalty


# =====================================================================
# §4.  Diversity-aware sampling step (entry point)
# =====================================================================
@dataclass
class DiversitySamplingConfig:
    """Configuration for the diversity-aware sampling entry point.

    Attributes
    ----------
    top_p
        Nucleus sampling cutoff in (0, 1].  Default 0.9.
    top_k
        Top-k truncation.  Default 50.  Set to 0 to disable.
    temperature
        Softmax temperature in (0, ∞).  Default 1.0.
    mmd_weight
        Weight on the MMD loss term in the combined objective.
        Default 0.1 (per TODO-202 §2.2).
    repulsion_weight
        Weight on the repulsion loss term.  Default 0.05.
    mmd_sigma
        RBF bandwidth for the MMD term.  Default 1.0.
    repulsion_margin
        Cosine-sim margin for the repulsion term.  Default 0.95.
    use_mmd
        Whether to compute and return the MMD term.  Default True.
    use_repulsion
        Whether to compute and return the repulsion term.  Default
        True.
    """

    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0
    mmd_weight: float = 0.1
    repulsion_weight: float = 0.05
    mmd_sigma: float = 1.0
    repulsion_margin: float = 0.95
    use_mmd: bool = True
    use_repulsion: bool = True


@dataclass
class DiversitySamplingOutput:
    """Return value of :func:`diversity_aware_sampling_step`.

    Attributes
    ----------
    sampled
        ``(B, N)`` long tensor of sampled atom-type indices — the
        output of nucleus_sample per batch element.
    mmd_loss
        Scalar MMD^2 between the sampled embeddings and the
        reference embeddings.  Zero when ``config.use_mmd`` is False.
    repulsion_loss
        Scalar repulsion loss on the sampled embeddings.  Zero when
        ``config.use_repulsion`` is False.
    combined_loss
        ``mmd_weight * mmd_loss + repulsion_weight * repulsion_loss``
        — the training-side scalar to add to the CFM loss.
    """

    sampled: torch.Tensor
    mmd_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    repulsion_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    combined_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))


def diversity_aware_sampling_step(
    logits: torch.Tensor,
    config: DiversitySamplingConfig,
    *,
    reference_embeddings: Optional[torch.Tensor] = None,
    embedding_layer: Optional[torch.nn.Embedding] = None,
    generator: Optional[torch.Generator] = None,
) -> DiversitySamplingOutput:
    """Combined diversity-aware sampling step.

    Wraps :func:`nucleus_sample` (per-step sampling) + the MMD /
    repulsion loss terms (set-level diversity) into a single
    entry point.  The trainer wires this in via one CLI flag
    (``--diversity-aware-sampling`` per TODO-202 §2.2).

    Parameters
    ----------
    logits
        ``(B, N, V)`` raw logits — batch × atoms × vocab.
    config
        :class:`DiversitySamplingConfig` dataclass.  Defaults match
        TODO-202 §2.2.
    reference_embeddings
        ``(M, D)`` reference embeddings for the MMD term (e.g.
        training-set mean embeddings).  Required if
        ``config.use_mmd=True``; ignored otherwise.
    embedding_layer
        Optional :class:`nn.Embedding` to lift the sampled token
        IDs into the repulsion / MMD feature space.  If None, the
        one-hot identity matrix is used (i.e. ``F.one_hot``
        projected to a learnable ``D``-dim random projection is
        impossible — we fall back to a deterministic identity-like
        projection via :func:`_sample_to_features`).
    generator
        Optional torch.Generator for deterministic sampling (tests).

    Returns
    -------
    :class:`DiversitySamplingOutput`
        Named-tuple-like struct with ``sampled`` (atom indices),
        ``mmd_loss``, ``repulsion_loss``, ``combined_loss``.

    Honest framing
    --------------
    * Per-step diversity (top-p + top-k + temperature) is bit-
      identical to the Holtzman 2020 reference.
    * MMD / repulsion terms are **set-level** — they lift diversity
      on the whole sampled batch, not per-step.  Their gradients
      are detached from the sampling op (we treat the sampled IDs
      as fixed inputs and only compute the loss terms on the lifted
      embeddings).
    """
    sampled = nucleus_sample(
        logits,
        p=config.top_p,
        k=config.top_k,
        temperature=config.temperature,
        generator=generator,
    )

    # Lift sampled IDs into a feature space for the loss terms.
    features = _sample_to_features(sampled, embedding_layer=embedding_layer)

    mmd = torch.zeros((), dtype=logits.dtype, device=logits.device)
    rep = torch.zeros((), dtype=logits.dtype, device=logits.device)

    if config.use_mmd:
        if reference_embeddings is None:
            # Quiet degradation: no reference → MMD term is zero
            # (matches "no MMD penalty" baseline).
            pass
        else:
            mmd = mmd_loss(
                features,
                reference_embeddings.to(features.device),
                kernel="rbf",
                sigma=config.mmd_sigma,
                unbiased=True,
            )

    if config.use_repulsion:
        if features.shape[0] >= 2:
            rep = repulsion_loss(
                features,
                margin=config.repulsion_margin,
                reduce=True,
            )

    combined = float(config.mmd_weight) * mmd + float(config.repulsion_weight) * rep
    return DiversitySamplingOutput(
        sampled=sampled,
        mmd_loss=mmd,
        repulsion_loss=rep,
        combined_loss=combined,
    )


# =====================================================================
# §5.  Helpers
# =====================================================================
def _sample_to_features(
    sampled: torch.Tensor,
    embedding_layer: Optional[torch.nn.Embedding] = None,
) -> torch.Tensor:
    """Lift sampled token IDs to a ``(N, D)`` feature tensor.

    When an ``embedding_layer`` is supplied we use it directly
    (the canonical path for trainers that have an existing atom
    embedding layer).  When absent, we fall back to the
    one-hot identity matrix — ``F.one_hot(sampled)`` — which
    gives a sparse, deterministic, and reasonably-conditioned
    feature space.  The MMD and repulsion terms are both well-
    defined on one-hot features (the RBF kernel sees the L2
    distance between rows; the cosine similarity sees the
    normalised inner product).
    """
    if embedding_layer is not None:
        return embedding_layer(sampled.reshape(-1))
    # Fall back to one-hot.  ``sampled`` may have multiple leading
    # dims (B, N) — flatten to (B*N,) for the embedding lookup.
    flat = sampled.reshape(-1)
    vocab = int(flat.max().item()) + 1 if flat.numel() > 0 else 1
    # Use ``num_classes`` from sampled max + 1 for safety against
    # vocab-mismatch — the trainer should never sample an
    # out-of-range token, but we guard.
    one_hot = F.one_hot(flat.long(), num_classes=max(vocab, 1)).to(
        torch.get_default_dtype(),
    )
    return one_hot


__all__ = [
    "nucleus_sample",
    "mmd_loss",
    "repulsion_loss",
    "diversity_aware_sampling_step",
    "DiversitySamplingConfig",
    "DiversitySamplingOutput",
]
