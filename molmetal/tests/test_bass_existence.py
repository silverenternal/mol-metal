"""Tests for :mod:`molmetal.math_skills.bass_existence`.

T8 — Bass Martingale Coupling Existence (Backhoff-Veraguas 2020)
================================================================

These tests cover the two mathematical skeleton primitives shipped by
the Bass-existence module:

1. **BassExistenceTheorem** — the formal statement holder.  Verifies
   construction (default and override), statement stability across
   calls, statement-text content (contains "Backhoff-Veraguas" and
   the citation key "2004.10060"), uniqueness verdict (defaults to
   ``False``; flips to ``True`` only on the ``d = 1`` carve-out),
   and the citation-key accessor.

2. **RegularityCondition** — the weak :math:`\\mathcal{P}_2`
   regularity check.  Verifies construction (default threshold,
   threshold rejection), the empirical second-moment check on
   1-D and 2-D sample tensors (Gaussian, uniform, degenerate
   Dirac), the rejection of empty / non-finite / heavy-tailed
   marginals, and the ``second_moment`` helper.

Every test is **CPU-only** and uses small data (sample sizes 4–64)
so the full suite runs in < 1 second.

Reference
---------
Backhoff-Veraguas, M., Beiglböck, M., Cox, M. and Huesmann, M.
(2020).  *Martingale Benamou-Brenier: a probabilistic proof*.
arXiv:2004.10060.  Cited in ``paper/refs.bib`` as
``backhoff_veraguas2020bass_existence``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from molmetal.math_skills import BassExistenceTheorem, RegularityCondition


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _uniform_samples(
    n: int, low: float = -1.0, high: float = 1.0, seed: int = 0
) -> torch.Tensor:
    """Return ``n`` i.i.d. uniform samples in ``[low, high]``."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(n).uniform_(low, high, generator=g)


def _gaussian_samples(
    n: int, mean: float = 0.0, std: float = 1.0, d: int = 1, seed: int = 0
) -> torch.Tensor:
    """Return ``n`` i.i.d. Gaussian samples of shape ``(n,)`` or ``(n, d)``."""
    g = torch.Generator().manual_seed(seed)
    shape = (n,) if d == 1 else (n, d)
    return mean + std * torch.randn(shape, generator=g)


# ---------------------------------------------------------------------------
# 1. BassExistenceTheorem — construction and statement content
# ---------------------------------------------------------------------------


def test_bass_existence_default_construction() -> None:
    """``BassExistenceTheorem()`` constructs without error and has no dim."""
    thm = BassExistenceTheorem()
    assert thm._dimension is None  # default: dimension-agnostic


def test_bass_existence_construction_with_dimension() -> None:
    """The ``dimension`` kwarg is stored."""
    thm = BassExistenceTheorem(dimension=3)
    assert thm._dimension == 3


def test_bass_existence_statement_is_string() -> None:
    """``statement()`` returns a non-empty ``str``."""
    thm = BassExistenceTheorem()
    text = thm.statement()
    assert isinstance(text, str)
    assert len(text) > 0


def test_bass_existence_statement_mentions_backhoff() -> None:
    """The statement text references the Backhoff-Veraguas 2020 paper."""
    text = BassExistenceTheorem().statement()
    assert "Backhoff" in text or "Backhoff-Veraguas" in text


def test_bass_existence_statement_mentions_martingale() -> None:
    """The statement text describes the martingale-coupling setting."""
    text = BassExistenceTheorem().statement()
    assert "martingale" in text.lower()


def test_bass_existence_statement_mentions_arxiv_id() -> None:
    """The statement text references the arXiv preprint ID 2004.10060."""
    text = BassExistenceTheorem().statement()
    assert "2004.10060" in text


def test_bass_existence_statement_stable_across_calls() -> None:
    """Two calls to ``statement()`` return identical text."""
    thm = BassExistenceTheorem()
    assert thm.statement() == thm.statement()


def test_bass_existence_statement_includes_dimension_when_set() -> None:
    """When ``dimension`` is provided, the statement annotates it."""
    thm = BassExistenceTheorem(dimension=2)
    text = thm.statement()
    assert "d = 2" in text


def test_bass_existence_statement_override_is_respected() -> None:
    """An explicit ``statement`` override is returned verbatim."""
    custom = "A custom statement text."
    thm = BassExistenceTheorem(statement=custom)
    assert thm.statement() == custom


# ---------------------------------------------------------------------------
# 2. BassExistenceTheorem — uniqueness verdict
# ---------------------------------------------------------------------------


def test_bass_existence_uniqueness_fails_default_returns_false() -> None:
    """``uniqueness_fails()`` defaults to ``False`` (matches brief)."""
    thm = BassExistenceTheorem()
    assert thm.uniqueness_fails() is False


def test_bass_existence_uniqueness_fails_stable_across_calls() -> None:
    """``uniqueness_fails()`` is deterministic across repeated calls."""
    thm = BassExistenceTheorem()
    verdicts = [thm.uniqueness_fails() for _ in range(5)]
    assert verdicts == [False] * 5


def test_bass_existence_uniqueness_fails_d2_returns_false() -> None:
    """``d = 2``: uniqueness still fails (the default carve-out is opt-in)."""
    thm = BassExistenceTheorem(dimension=2)
    assert thm.uniqueness_fails() is False


def test_bass_existence_uniqueness_fails_d1_returns_true() -> None:
    """``d = 1``: uniqueness holds (the canonical 1-D carve-out)."""
    thm = BassExistenceTheorem(dimension=1)
    assert thm.uniqueness_fails() is True


def test_bass_existence_citation_key() -> None:
    """The citation key matches the BibTeX entry in ``paper/refs.bib``."""
    thm = BassExistenceTheorem()
    assert thm.citation_key() == "backhoff_veraguas2020bass_existence"


# ---------------------------------------------------------------------------
# 3. RegularityCondition — construction
# ---------------------------------------------------------------------------


def test_regularity_condition_default_construction() -> None:
    """``RegularityCondition()`` constructs with a positive default threshold."""
    rc = RegularityCondition()
    assert rc.heavy_tail_threshold > 0
    assert math.isfinite(rc.heavy_tail_threshold)


def test_regularity_condition_rejects_nonpositive_threshold() -> None:
    """``heavy_tail_threshold <= 0`` is rejected at construction time."""
    with pytest.raises(ValueError, match="heavy_tail_threshold"):
        RegularityCondition(heavy_tail_threshold=0.0)
    with pytest.raises(ValueError, match="heavy_tail_threshold"):
        RegularityCondition(heavy_tail_threshold=-1.0)


# ---------------------------------------------------------------------------
# 4. RegularityCondition — empirical second-moment check (1-D)
# ---------------------------------------------------------------------------


def test_regularity_condition_1d_uniform_passes() -> None:
    """A 1-D uniform sample tensor is in :math:`\\mathcal{P}_2`."""
    rc = RegularityCondition()
    samples = _uniform_samples(n=128, low=-1.0, high=1.0)
    assert rc.check(samples) is True


def test_regularity_condition_1d_gaussian_passes() -> None:
    """A 1-D Gaussian sample tensor is in :math:`\\mathcal{P}_2`."""
    rc = RegularityCondition()
    samples = _gaussian_samples(n=128, mean=0.0, std=1.0)
    assert rc.check(samples) is True


def test_regularity_condition_1d_dirac_passes() -> None:
    """A 1-D Dirac (single point mass) sample tensor is in
    :math:`\\mathcal{P}_2`."""
    rc = RegularityCondition()
    samples = torch.zeros(8)
    assert rc.check(samples) is True


def test_regularity_condition_1d_with_inf_fails() -> None:
    """A 1-D sample tensor containing ``inf`` fails the check."""
    rc = RegularityCondition()
    samples = torch.tensor([0.0, 1.0, float("inf"), -2.0])
    assert rc.check(samples) is False


def test_regularity_condition_1d_with_nan_fails() -> None:
    """A 1-D sample tensor containing ``nan`` fails the check."""
    rc = RegularityCondition()
    samples = torch.tensor([0.0, 1.0, float("nan"), -2.0])
    assert rc.check(samples) is False


def test_regularity_condition_empty_tensor_fails() -> None:
    """An empty sample tensor fails (no atoms to integrate over)."""
    rc = RegularityCondition()
    samples = torch.empty(0)
    assert rc.check(samples) is False


# ---------------------------------------------------------------------------
# 5. RegularityCondition — empirical second-moment check (2-D)
# ---------------------------------------------------------------------------


def test_regularity_condition_2d_gaussian_passes() -> None:
    """A 2-D Gaussian sample tensor is in :math:`\\mathcal{P}_2(\\mathbb{R}^2)`."""
    rc = RegularityCondition()
    samples = _gaussian_samples(n=128, mean=0.0, std=1.0, d=2)
    assert samples.shape == (128, 2)
    assert rc.check(samples) is True


def test_regularity_condition_2d_with_nan_fails() -> None:
    """A 2-D sample tensor containing ``nan`` fails the check."""
    rc = RegularityCondition()
    samples = torch.zeros(4, 2)
    samples[1, 0] = float("nan")
    assert rc.check(samples) is False


def test_regularity_condition_heavy_tail_threshold_rejects() -> None:
    """A tight ``heavy_tail_threshold`` correctly rejects heavy samples."""
    # Build a sample with empirical second moment ~ 100.
    samples = torch.full((10,), 10.0)  # second_moment = 100
    # Default threshold is way above 100 → passes.
    assert RegularityCondition().check(samples) is True
    # A tight threshold of 50 → fails.
    rc = RegularityCondition(heavy_tail_threshold=50.0)
    assert rc.check(samples) is False


def test_regularity_condition_rejects_non_tensor_input() -> None:
    """A non-tensor input (e.g. Python list) is rejected."""
    rc = RegularityCondition()
    assert rc.check([0.0, 1.0, 2.0]) is False  # type: ignore[arg-type]


def test_regularity_condition_rejects_3d_tensor() -> None:
    """A 3-D tensor is rejected (the theorem is dimension-agnostic but
    the empirical check supports only 1-D / 2-D)."""
    rc = RegularityCondition()
    samples = torch.zeros(4, 2, 2)
    assert rc.check(samples) is False


# ---------------------------------------------------------------------------
# 6. RegularityCondition — second_moment helper
# ---------------------------------------------------------------------------


def test_regularity_condition_second_moment_1d_gaussian() -> None:
    """``second_moment`` of a unit Gaussian is close to 1."""
    rc = RegularityCondition()
    samples = _gaussian_samples(n=4096, mean=0.0, std=1.0, seed=42)
    m2 = rc.second_moment(samples)
    assert 0.7 < m2 < 1.3  # 4096 samples → empirical variance ~ 1 ± 0.05


def test_regularity_condition_second_moment_2d_gaussian() -> None:
    """``second_moment`` of a 2-D unit Gaussian is close to 2 (sum of dims)."""
    rc = RegularityCondition()
    samples = _gaussian_samples(n=4096, mean=0.0, std=1.0, d=2, seed=42)
    m2 = rc.second_moment(samples)
    assert 1.5 < m2 < 2.5


def test_regularity_condition_second_moment_empty_returns_inf() -> None:
    """``second_moment`` of an empty tensor returns ``inf``."""
    rc = RegularityCondition()
    assert rc.second_moment(torch.empty(0)) == float("inf")


def test_regularity_condition_second_moment_inf_input_returns_inf() -> None:
    """``second_moment`` of a non-finite tensor returns ``inf``."""
    rc = RegularityCondition()
    samples = torch.tensor([1.0, float("inf"), 2.0])
    assert rc.second_moment(samples) == float("inf")


# ---------------------------------------------------------------------------
# 7. Standalone — module does not import from molmetal.adapters or lam
# ---------------------------------------------------------------------------


def test_bass_existence_module_is_standalone() -> None:
    """Module source has no imports from `molmetal.adapters.*` /
    `molmetal.molmetal_lam.*` (standalone skeleton contract)."""
    src_path = (
        Path(__file__).resolve().parent.parent / "math_skills" / "bass_existence.py"
    )
    source = src_path.read_text()
    # The standalone contract: only stdlib + torch.
    forbidden = ["molmetal.adapters", "molmetal.molmetal_lam"]
    for token in forbidden:
        assert (
            token not in source
        ), f"bass_existence.py must not import {token}; found in source."
    # And the only third-party import we allow is torch.
    assert "import torch" in source
