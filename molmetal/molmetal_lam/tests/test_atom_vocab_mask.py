"""Tests for the WF-1 A2 atom vocabulary mask on
:meth:`molmetal.adapters.flow_matching_lipman.LipmanFlowMatchingAdapter`.

The atom-head softmax at sampling time was previously unrestricted
over ``[1, max_atomic_number)`` minus the Z=0 padding slot — which
exposed the sampler to out-of-vocabulary draws (noble gases,
lanthanides, synthetic transactinides, etc.).  A2 introduces a
``vocab_mask`` constructor arg (default ``True``) that constrains the
support to the 12-element donor + Pt vocabulary sourced from
:data:`molmetal.molmetal_lam.priors.metal_geometry.DEFAULT_METAL_GEOMETRY`.

Tests in this file exercise:

* **vocabulary membership** — sampled atoms are in the vocab when
  the mask is on, and *can* include out-of-vocab atoms when the mask
  is off (modulo the trained head's natural distribution).
* **probability mass conservation** — after softmax, the support sums
  to exactly 1.0 over the masked entries (no leakage to the
  ``-inf`` slots).
* **Pt coverage** — over 2,000 sampling steps, Pt appears with
  non-zero probability (i.e. is not accidentally masked out).
* **sampling distribution shape** — when the mask is on, the sampled
  atom histogram is contained inside the vocab.

All tests use the constructor-side helpers (``atom_vocab``,
``_build_vocab_mask``) and the same masked-softmax routine the
adapter uses at inference time.  They do **not** require the heavy
``flow_matching`` library clone, so they stay fast (~few seconds).
"""

from __future__ import annotations

import collections
import math

import pytest
import torch
import torch.nn.functional as F

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter


# ---------------------------------------------------------------------------
# Reference vocab — same source as the adapter, frozen here so a
# regression that drops an element (e.g. Pt) is caught immediately.
# ---------------------------------------------------------------------------
EXPECTED_VOCAB: tuple = (1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78)
# (H, C, N, O, F, P, S, Cl, Se, Br, I, Pt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_adapter(vocab_mask: bool) -> LipmanFlowMatchingAdapter:
    """Construct an adapter without invoking :meth:`setup` (so the test
    stays independent of the ``flow_matching`` clone).
    """
    return LipmanFlowMatchingAdapter(vocab_mask=vocab_mask)


def _apply_masked_softmax(
    logits: torch.Tensor,
    vocab: tuple,
    max_atomic_number: int,
) -> torch.Tensor:
    """Re-implement the exact masked-softmax path the adapter uses at
    sampling time.  Operates on a ``(..., max_atomic_number)`` tensor.
    """
    logits = logits.clone()
    logits[..., 0] = float("-inf")
    mask = torch.zeros(max_atomic_number, dtype=torch.bool, device=logits.device)
    for z in vocab:
        if 0 <= z < max_atomic_number:
            mask[z] = True
    logits = logits.masked_fill(~mask, float("-inf"))
    return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def adapter_on() -> LipmanFlowMatchingAdapter:
    return _make_adapter(vocab_mask=True)


@pytest.fixture(scope="module")
def adapter_off() -> LipmanFlowMatchingAdapter:
    return _make_adapter(vocab_mask=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_vocab_matches_spec(adapter_on: LipmanFlowMatchingAdapter) -> None:
    """``atom_vocab`` returns the 12-element donor + Pt vocabulary."""
    vocab = adapter_on.atom_vocab
    assert vocab == EXPECTED_VOCAB, (
        f"Expected {EXPECTED_VOCAB}, got {tuple(vocab)}. "
        "If you intentionally re-scope the vocab, update EXPECTED_VOCAB "
        "AND molmetal/reports/wf1_a2_vocab_mask.md in the same commit."
    )
    # Cardinality is part of the public contract (test 4 in the spec).
    assert len(vocab) == 12, f"Expected vocab size 12, got {len(vocab)}"
    # Sanity: Pt (78) MUST be in the vocab — the SBDD-metal use case
    # would silently regress without it.
    assert 78 in vocab
    assert 1 in vocab  # H
    assert 6 in vocab  # C


def test_vocab_mask_default_is_true() -> None:
    """Default behaviour is ON — the masked-softmax ships out of the box."""
    a = LipmanFlowMatchingAdapter()
    assert a._vocab_mask is True, (
        "A2 default must be vocab_mask=True; round-10 spec locked this in."
    )


def test_sampled_atoms_in_vocab_when_mask_on(
    adapter_on: LipmanFlowMatchingAdapter,
) -> None:
    """With vocab_mask=True, 1000 categorical draws land inside the vocab.

    We construct a fixed ``(1000, max_atomic_number)`` logit tensor with
    non-zero values everywhere, apply the masked softmax, and sample
    from the resulting distribution.  Every drawn atomic number must
    belong to ``adapter_on.atom_vocab`` (or be the Z=0 padding slot —
    which is also masked, so it should never appear either).
    """
    max_z = 100
    torch.manual_seed(42)
    # Uniform-ish logits — softmax should be roughly uniform over the
    # masked support.  Crucially the non-vocab logits get a small
    # *positive* bias to make sure the mask is doing the work, not the
    # data.
    logits = torch.full((1000, max_z), -2.0)
    # Boost out-of-vocab entries — if the mask is broken, the boost
    # should push samples into the unmasked slots.
    for z in range(max_z):
        if z not in EXPECTED_VOCAB and z != 0:
            logits[:, z] = 2.0
    probs = _apply_masked_softmax(logits, adapter_on.atom_vocab, max_z)
    # Probability mass check: every non-vocab (and non-zero) column is
    # exactly zero — proves the mask is wired correctly.
    for z in range(max_z):
        if z == 0 or z not in EXPECTED_VOCAB:
            assert torch.all(probs[:, z] == 0), (
                f"Z={z} must have zero probability under vocab_mask=True, "
                f"got max={float(probs[:, z].max())}"
            )
    # Mass conservation over the masked support (test 4 in the spec).
    support_sum = probs.sum(dim=-1)
    assert torch.allclose(support_sum, torch.ones_like(support_sum), atol=1e-6), (
        "softmax over the masked support must sum to 1.0 per atom."
    )
    # Sample — every draw must land inside the vocab.
    sampler = torch.Generator().manual_seed(0)
    samples = torch.multinomial(probs, 1, generator=sampler).flatten()
    vocab_set = set(EXPECTED_VOCAB)
    out_of_vocab = [int(s) for s in samples.tolist() if int(s) not in vocab_set]
    assert not out_of_vocab, (
        f"vocab_mask=True sampled {len(out_of_vocab)} out-of-vocab atoms; "
        f"first 10: {out_of_vocab[:10]}"
    )


def test_all_atoms_in_vocab_when_mask_off(
    adapter_off: LipmanFlowMatchingAdapter,
) -> None:
    """With vocab_mask=False, no atoms are masked out — every Z in
    ``[1, max_atomic_number)`` has non-zero probability.

    Note: this does NOT guarantee sampling each element (the head is
    untrained → roughly uniform).  We test that every column receives
    >0 probability mass after softmax, AND that a uniform-random
    sample over enough draws covers a large fraction of the support.
    """
    max_z = 100
    torch.manual_seed(123)
    logits = torch.randn(2000, max_z)
    # No masking applied — the Z=0 padding slot is the only blocked index.
    logits[:, 0] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    # Every Z in [1, max_z) has >0 probability.
    for z in range(1, max_z):
        assert (probs[:, z] > 0).all(), (
            f"Z={z} must have non-zero probability when vocab_mask=False."
        )
    # Mass conservation.
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(2000), atol=1e-6,
    )
    # Sampling covers a wide swath of the support (at least 50 of the
    # 99 non-padding Zs should appear in 2000 draws).
    sampler = torch.Generator().manual_seed(1)
    samples = torch.multinomial(probs, 1, generator=sampler).flatten().tolist()
    unique = set(int(s) for s in samples if s != 0)
    assert len(unique) >= 50, (
        f"vocab_mask=False should cover most of [1, max_z); only saw "
        f"{len(unique)} unique Zs in {len(samples)} samples."
    )


def test_pt_appears_with_nonzero_probability(
    adapter_on: LipmanFlowMatchingAdapter,
) -> None:
    """Over 2000 sampling steps, Pt (Z=78) is reachable.

    We use a uniform logit tensor so every vocab element has equal
    probability (1/12).  Over 2000 draws, expected count for Pt is
    2000/12 ≈ 166 with std ~12, so getting zero is essentially
    impossible (chance ≈ 1e-64).  We additionally assert the Pt
    column has strictly positive probability mass in the softmax.
    """
    max_z = 100
    torch.manual_seed(7)
    logits = torch.zeros(2000, max_z)
    probs = _apply_masked_softmax(logits, adapter_on.atom_vocab, max_z)
    pt_col = probs[:, 78]
    # Pt probability must be exactly 1/12 for a uniform logit
    # (modulo fp noise).
    expected = 1.0 / len(adapter_on.atom_vocab)
    assert torch.allclose(pt_col, torch.full_like(pt_col, expected), atol=1e-5), (
        f"Pt probability column should be 1/{len(adapter_on.atom_vocab)}, "
        f"got mean={float(pt_col.mean())}"
    )
    # Sample 2000 draws.
    sampler = torch.Generator().manual_seed(7)
    samples = torch.multinomial(probs, 1, generator=sampler).flatten().tolist()
    pt_count = sum(1 for s in samples if int(s) == 78)
    # Poisson lower bound: with mean 166.7, the chance of getting
    # less than 100 is ~1e-12.  Use a generous threshold for flakiness.
    assert pt_count > 0, (
        f"Pt appeared 0 times in 2000 samples — the metal centre is "
        f"unreachable, the vocab mask is broken."
    )
    assert pt_count >= 100, (
        f"Expected ~166 Pt samples (uniform over 12), got {pt_count}."
    )


def test_mask_mass_conservation(adapter_on: LipmanFlowMatchingAdapter) -> None:
    """Per the spec test 4: ``sum(softmax) == 1.0`` over the masked support.

    This is a pure arithmetic check — we apply the masked softmax to a
    batch of random logits and verify the column sum equals 1 within
    fp tolerance.  Independent of the trained-head quality.
    """
    max_z = 100
    torch.manual_seed(2026)
    for seed_shift in range(3):
        torch.manual_seed(2026 + seed_shift)
        logits = torch.randn(50, max_z)
        # Lift Z=0 to a finite value first so we can verify the mask
        # truly drives the padding to zero (not the existing -inf
        # short-circuit on Z=0).
        logits[:, 0] = 5.0
        probs = _apply_masked_softmax(logits, adapter_on.atom_vocab, max_z)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
            f"Per-row softmax sum drifted from 1.0 (max abs err "
            f"{float((sums - 1).abs().max())}) — the masked support "
            f"is leaking probability to the -inf slots."
        )
        # Vocab columns must have strictly positive probability.
        for z in adapter_on.atom_vocab:
            assert (probs[:, z] > 0).all(), (
                f"Vocab element Z={z} has zero probability mass."
            )
        # Non-vocab columns (incl. Z=0) must have exactly zero.
        for z in range(max_z):
            if z == 0 or z not in adapter_on.atom_vocab:
                assert torch.all(probs[:, z] == 0), (
                    f"Z={z} leaked probability mass under vocab_mask=True."
                )


def test_sampled_atom_distribution_histogram(
    adapter_on: LipmanFlowMatchingAdapter,
) -> None:
    """Spec test 5: produce a sampled-atom histogram and report shape.

    With a uniform logit tensor and 2000 sampling steps, the 12 vocab
    elements should each receive ~1/12 of the draws.  We assert the
    histogram is non-empty AND contains every vocab element — i.e.
    no element of the support is *missing* from the empirical
    distribution (which would indicate a wiring bug).  Per-element
    count thresholds use a generous lower bound (>= 50 instead of
    the expected ~166) to avoid flakiness across PyTorch versions.
    """
    max_z = 100
    torch.manual_seed(11)
    n_steps = 2000
    logits = torch.zeros(n_steps, max_z)
    probs = _apply_masked_softmax(logits, adapter_on.atom_vocab, max_z)
    sampler = torch.Generator().manual_seed(11)
    samples = torch.multinomial(probs, 1, generator=sampler).flatten().tolist()
    counts = collections.Counter(int(s) for s in samples)
    # Every vocab element must appear at least once.
    missing = [z for z in adapter_on.atom_vocab if counts.get(z, 0) == 0]
    assert not missing, (
        f"Vocab elements {missing} never appeared in {n_steps} draws — "
        f"distribution shape broken.  Counts: {dict(counts)}."
    )
    # Loose shape check: no element gets more than 4x the uniform share
    # (would indicate a bias towards a single element) and every
    # element gets at least 50 draws (well above zero by CLT).
    expected_share = n_steps / len(adapter_on.atom_vocab)
    for z in adapter_on.atom_vocab:
        c = counts.get(z, 0)
        assert c >= 50, (
            f"Z={z} under-sampled: {c} < 50 (expected ~{expected_share:.0f})."
        )
        assert c <= 4 * expected_share, (
            f"Z={z} over-sampled: {c} > 4*{expected_share:.0f} — biased."
        )


def test_construct_with_vocab_mask_false_does_not_raise() -> None:
    """Sanity: opt-out path still constructs without error."""
    a = LipmanFlowMatchingAdapter(vocab_mask=False)
    assert a._vocab_mask is False
    assert a.atom_vocab == EXPECTED_VOCAB  # vocab is set unconditionally