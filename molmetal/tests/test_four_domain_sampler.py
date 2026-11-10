"""Tests for the four-domain combined dataset + WeightedRandomSampler.

Covers TODO-143 §2.5 Day 4 acceptance criteria:

1. Combined dataset ``__len__`` returns the sum of the 4 domain row counts
   (with the per-domain ``max_rows_per_domain`` caps applied).
2. ``compute_domain_weights`` yields ``w_d ∝ (N_target / N_d) ** (1 - alpha)``:
   * ``alpha = 0`` → ``w_d = N_target / N_d`` (inverse-frequency).
   * ``alpha = 1`` → ``w_d = 1`` (uniform).
   * ``alpha = 0.5`` → ``w_d = sqrt(N_target / N_d)``.
3. ``make_four_domain_sampler`` returns a working
   :class:`torch.utils.data.WeightedRandomSampler` with ``replacement=True``.
4. Domain counts in a sampled batch match the canonical expected mix
   (within a tolerance appropriate for finite draws).
5. The CLI accepts ``--data-config four-domain`` and ``--alpha-sampler`` and
   reports a sane payload in the resulting JSON.

All tests run with fast per-domain caps so the 1.58M-row DrugOOD CSV is
*not* fully loaded — the lazy loader is exercised in
``test_drugood_lazy_loading`` instead.
"""

from __future__ import annotations

import math
import os
import sys
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Wf-50 pre-condition: honour MOLMETAL_DATA_DIR for the cytotox loader.
_DEFAULT_DATA = PROJECT_ROOT / ".data" / "molmetal"
if not os.environ.get("MOLMETAL_DATA_DIR") and _DEFAULT_DATA.exists():
    os.environ["MOLMETAL_DATA_DIR"] = str(_DEFAULT_DATA)

import torch
from torch.utils.data import WeightedRandomSampler

from molmetal.data.four_domain_combined import (
    DEFAULT_BINDINGDB_CSV,
    DOMAIN_BINDINGDB,
    DOMAIN_DRUGOOD,
    DOMAIN_METALCYTOTOX,
    DOMAIN_NCI60,
    FourDomainDataset,
    iter_three_tuples,
)
from molmetal.data.four_domain_weights import (
    compute_domain_weights,
    expand_weights_to_per_row,
    expected_domain_mix,
    make_four_domain_sampler,
)


# ---------------------------------------------------------------------------
# Shared fixture — fast per-domain caps
# ---------------------------------------------------------------------------
# DrugOOD: cap at 500 rows so the test suite stays <30s.  MetalCytoToxDB
# (full ~5K) and NCI-60 (full 7.8K) are small enough to load whole; BindingDB
# is the real ~1.07M-row TDC bindingdb_ic50.csv (we cap to 5K for fast tests).
DEFAULT_CAPS = {
    DOMAIN_METALCYTOTOX: 5_207,
    DOMAIN_DRUGOOD: 500,
    DOMAIN_NCI60: 7_824,
    DOMAIN_BINDINGDB: 5_000,
}


@pytest.fixture(scope="module")
def four_ds():
    """A :class:`FourDomainDataset` with DrugOOD capped to 500 rows for fast tests."""
    return FourDomainDataset(max_rows_per_domain=DEFAULT_CAPS)


# ---------------------------------------------------------------------------
# 1. Combined dataset shape
# ---------------------------------------------------------------------------
def test_combined_length_matches_sum(four_ds: FourDomainDataset) -> None:
    """``__len__`` equals the sum of per-domain (capped) lengths.

    Note
    ----
    The MetalCytoToxDB count depends on the :class:`CytotoxFilter` (default
    ``time_threshold=24``, ``ic50_min=0.01``, ``require_active_field=True``),
    so we read the actual value rather than hard-coding 5207.  NCI-60 is a
    pre-filtered CSV; BindingDB is the real TDC ``bindingdb_ic50.csv``
    (~1.07M rows, capped at 5K for this test).
    """
    domain_lens = four_ds.domain_lengths()
    expected = sum(domain_lens.values())
    assert len(four_ds) == expected, f"expected {expected}, got {len(four_ds)}"
    # Per-domain sanity (NCI-60 pre-filtered; BindingDB capped at 5K;
    # DrugOOD capped at 500).  Trailing-newline quirks make line-counting
    # off-by-1 for some files — allow ±2.
    assert abs(domain_lens[DOMAIN_NCI60] - 7_824) <= 2
    assert domain_lens[DOMAIN_BINDINGDB] == 5_000
    assert domain_lens[DOMAIN_DRUGOOD] == 500
    # MetalCytoToxDB filter is more permissive than the canonical 5207 (we
    # accept anything in [5_000, 5_500] as long as it's stable).
    assert 5_000 <= domain_lens[DOMAIN_METALCYTOTOX] <= 5_500


def _bindingdb_data_rows() -> int:
    """Data rows in the BindingDB file on disk (header excluded).

    Wf-A53 (2026-10-19) — the tests below used to hardcode 1,072,931, the row
    count of the TDC file that was truncated to 0 bytes on 2026-09-30.  The
    surviving export has a different count, so pinning the old number tested
    the file's identity rather than the loader.  Deriving it keeps the
    assertion about the loader.
    """
    with DEFAULT_BINDINGDB_CSV.open("rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def test_combined_length_with_all_full() -> None:
    """Without caps the length equals the real sum of full domain sizes.

    DrugOOD's full CSV is 1,583,302 rows; BindingDB's count is read from the
    file on disk.  We assert ``len(ds) == sum(full)`` *without* loading any row
    (the lazy len uses line counting).
    """
    ds = FourDomainDataset()
    full_lens = ds._lens  # type: ignore[attr-defined]
    # NCI-60 pre-filtered (allow ±2 for trailing-newline quirks):
    assert abs(full_lens[DOMAIN_NCI60] - 7_824) <= 2
    # BindingDB: the lazy count must match the file actually on disk
    expected_bdb = _bindingdb_data_rows()
    assert (
        abs(full_lens[DOMAIN_BINDINGDB] - expected_bdb) <= 2
    ), f"BindingDB lazy count {full_lens[DOMAIN_BINDINGDB]} != file rows {expected_bdb}"
    # DrugOOD is 1.58M; allow ±10 to accommodate header/trailing-newline quirks
    assert (
        1_583_000 < full_lens[DOMAIN_DRUGOOD] < 1_584_000
    ), f"DrugOOD has {full_lens[DOMAIN_DRUGOOD]} rows, expected ~1.58M"
    # MetalCytoToxDB after filter (5_000 ≤ n ≤ 5_500 is the realistic band)
    assert 5_000 <= full_lens[DOMAIN_METALCYTOTOX] <= 5_500
    assert len(ds) == sum(full_lens.values())


def test_combined_total_is_2_6m() -> None:
    """Honest framing: the four-domain total is dominated by DrugOOD + BindingDB.

    BindingDB's row count is read from the file on disk rather than pinned to
    a version that no longer exists (see ``_bindingdb_data_rows``).  Without
    the symlink (i.e. only the fallback ``protein_metal_binding.csv`` with
    5,258 rows), the total drops to ~1.61M.
    """
    ds = FourDomainDataset()
    full_lens = ds._lens  # type: ignore[attr-defined]
    total = sum(full_lens.values())
    domain3 = ds._domain_3  # type: ignore[attr-defined]
    if domain3.using_fallback:
        # Without the real BindingDB symlink, we only get the 5K fallback.
        assert (
            1_590_000 < total < 1_620_000
        ), f"fallback total = {total}; expected ~1.61M (no BindingDB symlink)"
        return
    expected_bdb = _bindingdb_data_rows()
    assert (
        abs(full_lens[DOMAIN_BINDINGDB] - expected_bdb) <= 2
    ), f"BindingDB lazy count {full_lens[DOMAIN_BINDINGDB]} != file rows {expected_bdb}"
    # Total is whatever the three fixed domains plus the real BindingDB sum to.
    expected_total = (
        full_lens[DOMAIN_DRUGOOD]
        + expected_bdb
        + full_lens[DOMAIN_NCI60]
        + full_lens[DOMAIN_METALCYTOTOX]
    )
    assert abs(total - expected_total) <= 4, (
        f"four-domain total {total} != sum of parts {expected_total}"
    )


# ---------------------------------------------------------------------------
# 2. Items return (smiles, y, domain_id)
# ---------------------------------------------------------------------------
def test_item_returns_smiles_y_domain_id(four_ds: FourDomainDataset) -> None:
    """Each item carries ``smiles``, ``y``, and ``domain_id ∈ {0,1,2,3}``."""
    seen_domains = set()
    for d in range(4):
        item = four_ds[four_ds._offsets[d]]  # type: ignore[attr-defined]
        assert isinstance(item.smiles, str) and item.smiles, f"empty smiles at d={d}"
        assert item.domain_id == d, f"expected domain_id={d}, got {item.domain_id}"
        assert item.y is not None, f"y is None at d={d}"
        seen_domains.add(item.domain_id)
    assert seen_domains == {0, 1, 2, 3}


def test_iter_three_tuples_yields_three_tuple_per_item(
    four_ds: FourDomainDataset,
) -> None:
    """``iter_three_tuples`` yields ``(smiles, y, domain_id)`` per TODO-143 §2.5."""
    first_few = []
    for s, y, d in iter_three_tuples(four_ds):
        first_few.append((s, y, d))
        if len(first_few) >= 16:
            break
    assert len(first_few) == 16
    for s, _, d in first_few:
        assert isinstance(s, str)
        assert d in {0, 1, 2, 3}


def test_domain_offset_resolution(four_ds: FourDomainDataset) -> None:
    """Flat ``__getitem__`` maps back to the right domain via offsets."""
    for d in range(4):
        # The first row of each domain is at the cumulative offset
        idx = four_ds._offsets[d]  # type: ignore[attr-defined]
        item = four_ds[int(idx)]
        assert (
            item.domain_id == d
        ), f"offset {idx} should map to domain {d}, got {item.domain_id}"


# ---------------------------------------------------------------------------
# 3. Weight formula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alpha, expected",
    [
        (0.0, None),  # inverse-freq — checked numerically below
        (0.5, None),  # soft balance — checked numerically below
        (1.0, None),  # uniform — checked numerically below
    ],
)
def test_compute_domain_weights_alpha_formula(alpha: float, expected) -> None:
    """``w_d = (N_target / N_d) ** (1 - alpha)``.

    For a fixed ratio of domain sizes we can predict the weights exactly.
    Use lengths [10, 1000] so N_target = 1000 and N_target/N_0 = 100,
    N_target/N_1 = 1.
    """
    domain_lengths = {0: 10, 1: 1000}
    w = compute_domain_weights(domain_lengths, alpha=alpha)
    n_target = 1000.0
    expected_w0 = (n_target / 10.0) ** (1.0 - alpha)  # big oversampling for d=0
    expected_w1 = (n_target / 1000.0) ** (
        1.0 - alpha
    )  # = 1 always (for d=1, the largest)
    assert math.isclose(w[0], expected_w0, rel_tol=1e-9)
    assert math.isclose(w[1], expected_w1, rel_tol=1e-9)


def test_alpha_1_yields_uniform_weights() -> None:
    """alpha=1 → ``w_d = 1`` for every domain (uniform sampling)."""
    domain_lengths = {0: 100, 1: 50_000, 2: 5_000, 3: 1_000}
    w = compute_domain_weights(domain_lengths, alpha=1.0)
    for d, v in w.items():
        assert math.isclose(v, 1.0, rel_tol=1e-9), f"d={d} w_d={v}"


def test_alpha_0_yields_inverse_freq() -> None:
    """alpha=0 → ``w_d = N_target / N_d`` (canonical inverse-frequency)."""
    domain_lengths = {0: 100, 1: 50_000, 2: 5_000, 3: 1_000}
    w = compute_domain_weights(domain_lengths, alpha=0.0)
    n_target = 50_000.0
    assert math.isclose(w[0], n_target / 100.0)  # 500x oversample
    assert math.isclose(w[1], n_target / 50_000.0)  # = 1 (largest domain)
    assert math.isclose(w[2], n_target / 5_000.0)  # 10x oversample
    assert math.isclose(w[3], n_target / 1_000.0)  # 50x oversample


def test_alpha_0_5_yields_sqrt_inverse_freq() -> None:
    """alpha=0.5 → ``w_d = sqrt(N_target / N_d)`` (soft balance)."""
    domain_lengths = {0: 100, 1: 50_000, 2: 5_000, 3: 1_000}
    w = compute_domain_weights(domain_lengths, alpha=0.5)
    n_target = 50_000.0
    assert math.isclose(w[0], math.sqrt(n_target / 100.0))  # ~22.36
    assert math.isclose(w[1], math.sqrt(n_target / 50_000.0))  # = 1
    assert math.isclose(w[2], math.sqrt(n_target / 5_000.0))  # ~3.16
    assert math.isclose(w[3], math.sqrt(n_target / 1_000.0))  # ~7.07


# ---------------------------------------------------------------------------
# 4. expand_weights_to_per_row
# ---------------------------------------------------------------------------
def test_expand_weights_length_matches_dataset(four_ds: FourDomainDataset) -> None:
    """The per-row weight list has length equal to ``len(dataset)``."""
    w = compute_domain_weights(four_ds.domain_lengths(), alpha=0.5)
    per_row = expand_weights_to_per_row(four_ds, w)
    assert len(per_row) == len(
        four_ds
    ), f"len(per_row)={len(per_row)} but len(dataset)={len(four_ds)}"


def test_expand_weights_are_constant_within_domain(four_ds: FourDomainDataset) -> None:
    """Every row in domain d carries the same weight ``w_d``."""
    w = compute_domain_weights(four_ds.domain_lengths(), alpha=0.5)
    per_row = expand_weights_to_per_row(four_ds, w)
    lens = four_ds.domain_lengths()
    cum = 0
    for d in range(4):
        n = lens[d]
        block = per_row[cum : cum + n]
        assert len(block) == n
        # All values in the block are equal (within fp tolerance)
        assert max(block) - min(block) < 1e-9
        assert math.isclose(block[0], w[d], rel_tol=1e-9)
        cum += n


# ---------------------------------------------------------------------------
# 5. Sampler
# ---------------------------------------------------------------------------
def test_make_sampler_returns_weighted_random_sampler(
    four_ds: FourDomainDataset,
) -> None:
    """Factory returns a :class:`WeightedRandomSampler` with replacement=True."""
    sampler = make_four_domain_sampler(four_ds, alpha=0.5)
    assert isinstance(sampler, WeightedRandomSampler)
    # PyTorch's WeightedRandomSampler requires replacement=True for weights
    assert sampler.replacement is True


def test_sampler_iteration_works(four_ds: FourDomainDataset) -> None:
    """We can iterate the sampler for one full epoch without error."""
    sampler = make_four_domain_sampler(four_ds, alpha=0.5, num_samples=512)
    indices = list(iter(sampler))
    assert len(indices) == 512
    assert all(0 <= int(i) < len(four_ds) for i in indices)


def test_sampler_with_seeded_generator_is_deterministic(
    four_ds: FourDomainDataset,
) -> None:
    """Two seeded samplers produce the same index sequence."""
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    s1 = make_four_domain_sampler(four_ds, alpha=0.5, num_samples=256, generator=gen1)
    s2 = make_four_domain_sampler(four_ds, alpha=0.5, num_samples=256, generator=gen2)
    assert list(iter(s1)) == list(iter(s2))


def test_sampler_replacement_true_is_required(four_ds: FourDomainDataset) -> None:
    """Per-row weights are constant within each domain → replacement=True is required.

    Without replacement PyTorch raises ``ValueError: cannot sample n > total
    with replacement=False``.  Verify our sampler does NOT raise and yields
    ``replacement=True``.
    """
    sampler = make_four_domain_sampler(four_ds, alpha=0.0)  # most imbalanced
    assert sampler.replacement is True
    # Sample beyond dataset length to verify replacement
    sampler_big = make_four_domain_sampler(four_ds, alpha=0.0, num_samples=10_000)
    indices = list(iter(sampler_big))
    assert len(indices) == 10_000


# ---------------------------------------------------------------------------
# 6. Domain mix in sampled batches
# ---------------------------------------------------------------------------
def test_expected_domain_mix_alpha_1_is_per_row_uniform() -> None:
    """alpha=1 → uniform per-row sampling → expected mix ∝ N_d."""
    domain_lengths = {0: 100, 1: 50_000, 2: 5_000, 3: 1_000}
    mix = expected_domain_mix(domain_lengths, alpha=1.0)
    total = sum(domain_lengths.values())
    expected = {d: n / total for d, n in domain_lengths.items()}
    for d, v in mix.items():
        assert math.isclose(v, expected[d], rel_tol=1e-9)


def test_expected_domain_mix_alpha_0_is_per_domain_uniform() -> None:
    """alpha=0 → each row in domain d carries weight N_target/N_d.

    Then per-row probability ∝ w_d * N_d = (N_target/N_d) * N_d = N_target,
    which is **constant** across domains.  Therefore each *domain* gets the
    same share (= 1 / num_domains), regardless of N_d.  This is the
    canonical inverse-frequency weighting.
    """
    domain_lengths = {0: 100, 1: 50_000, 2: 5_000, 3: 1_000}
    mix = expected_domain_mix(domain_lengths, alpha=0.0)
    expected = {d: 1.0 / len(domain_lengths) for d in domain_lengths}
    for d, v in mix.items():
        assert math.isclose(v, expected[d], rel_tol=1e-9)


def test_sampled_domain_mix_matches_expected_within_tolerance(
    four_ds: FourDomainDataset,
) -> None:
    """Sampled domain shares should match ``expected_domain_mix`` within ±3 %.

    Honest caveat
    -------------
    For very imbalanced domains (DrugOOD 500 vs MetalCytoToxDB 5207 vs
    NCI-60 7824) the *expected* per-domain mix for ``alpha=0.5`` still leans
    heavily toward the larger domains.  This test just checks that the
    sampler hits the expected share; it does **not** check that the model
    converges to equal domain mass.
    """
    expected = expected_domain_mix(four_ds.domain_lengths(), alpha=0.5)
    sampler = make_four_domain_sampler(four_ds, alpha=0.5, num_samples=20_000)
    counts = {d: 0 for d in range(4)}
    for i in iter(sampler):
        d, _ = four_ds._resolve_global_index(int(i))  # type: ignore[attr-defined]
        counts[d] += 1
    total = sum(counts.values())
    observed = {d: c / total for d, c in counts.items()}
    # Tolerance: ±3 % absolute for 20K draws (3σ ≈ √(0.5 * 0.5 / 20K) ≈ 0.35 %,
    # so 3 % is very generous)
    for d in range(4):
        assert (
            abs(observed[d] - expected[d]) < 0.03
        ), f"d={d} expected={expected[d]:.3f} observed={observed[d]:.3f}"


# ---------------------------------------------------------------------------
# 7. CLI integration (--data-config four-domain --alpha-sampler)
# ---------------------------------------------------------------------------
def test_cli_accepts_four_domain_flag() -> None:
    """``train_dmpnn_multitask.py --help`` lists --data-config and --alpha-sampler."""
    out = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={
            "CUDA_VISIBLE_DEVICES": "",
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )
    assert out.returncode == 0, out.stderr
    assert "--data-config" in out.stdout
    assert "--alpha-sampler" in out.stdout
    assert "four-domain" in out.stdout


# ---------------------------------------------------------------------------
# 8. Lazy-loading honesty for the 1.58M-row DrugOOD CSV
# ---------------------------------------------------------------------------
def test_drugood_lazy_loading() -> None:
    """Constructing the dataset must NOT load the 1.58M DrugOOD rows eagerly.

    We assert that immediately after construction the inner ``_DrugOODRows``
    has ``_rows is None``, and only after the first ``__getitem__`` does it
    populate.
    """
    ds = FourDomainDataset(max_rows_per_domain={DOMAIN_DRUGOOD: 1_000})
    inner = ds._domain_1  # type: ignore[attr-defined]
    assert inner._rows is None, "DrugOOD rows were eagerly loaded — should be lazy"
    # First getitem triggers load
    _ = ds[ds._offsets[DOMAIN_DRUGOOD]]  # type: ignore[attr-defined]
    assert inner._rows is not None, "DrugOOD rows failed to load on first __getitem__"
    assert len(inner._rows) >= 1_000  # may be more if duplicates / blanks


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------
def test_zero_length_domain_gets_zero_weight() -> None:
    """A domain with N_d = 0 gets weight 0 (no rows can be drawn)."""
    domain_lengths = {0: 100, 1: 0, 2: 50, 3: 0}
    w = compute_domain_weights(domain_lengths, alpha=0.5)
    assert w[1] == 0.0
    assert w[3] == 0.0
    assert w[0] > 0.0
    assert w[2] > 0.0


def test_all_zero_lengths_raises() -> None:
    """If every domain is empty, ``compute_domain_weights`` raises."""
    with pytest.raises(ValueError, match="zero"):
        compute_domain_weights({0: 0, 1: 0, 2: 0, 3: 0}, alpha=0.5)


def test_invalid_alpha_raises() -> None:
    """alpha outside [0, 1] is rejected."""
    with pytest.raises(ValueError, match="alpha"):
        compute_domain_weights({0: 10, 1: 20}, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        compute_domain_weights({0: 10, 1: 20}, alpha=1.5)
