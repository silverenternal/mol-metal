"""Guard for ``load_tmqm_clouds(..., min_per_metal=...)``.

2026-10-12 zero-Pt finding. The seed=0 full-corpus run's val window is
the reservoir sample of 192 of 960 candidates, and that window contained
exactly ZERO Pt complexes::

    [tmqm-clouds] metals=[...] coord_number=None: 192 cloud(s) from
        960 candidate(s); skipped 0 (unmapped element) + 0
        (metal missing from geometry)

while Pt is 24.8% of the underlying six-metal pool (6634 / 26715 in
training; 1641 / 6714 in val full). The model's templates therefore
never include a Pt, so every generated molecule lands in the
Ru/Ir/Rh/Re/Os set with zero cisplatin pharmacophore candidates.

This is a SAMPLING ARTEFACT, not a model defect: the training pool
contains 6634 Pt atoms (the largest single-metal count), so the
checkpoint has seen Pt extensively and a retrain would change nothing
on this axis.  Verdict: explanation (A) -- the loader window is
under-representative.  Explanation (B) -- retraining on a window with
Pt -- is REFUTED on the training-distribution data alone.

The fix is a new keyword ``min_per_metal`` on ``load_tmqm_clouds``.
When ``min_per_metal >= 1`` the loader back-fills reservoir slots so
EVERY requested metal is represented at least that many times, by
re-scanning the geometry shards (not just the streaming-cap window).

THIS TEST IS THE RED CHECK. Without the patch:
  * the loader does not accept ``min_per_metal`` -> ``TypeError``.
  * tests 2 and 3 below go red before the patch lands.

Run on the existing corpus (.data/molmetal/tmQM/) -- it is hermetic
once the data is on disk, which the project README declares to be a
prerequisite for any non-skipped CFM test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMQM_DIR = PROJECT_ROOT / ".data" / "molmetal" / "tmQM"
_HAVE_TMQM = TMQM_DIR.is_dir()
SIX_METALS = ["Pt", "Ru", "Ir", "Rh", "Os", "Re"]

# Mononuclear tmQM -> exactly one metal Z per cloud. Index the metal by Z,
# not by atomic_numbers[0] (file atom order is arbitrary).
Z_OF_METAL = {"Pt": 78, "Ru": 44, "Ir": 77, "Rh": 45, "Os": 76, "Re": 75}


def _metal_counts(clouds) -> dict:
    from collections import Counter
    c: Counter = Counter()
    for cl in clouds:
        for z in cl.atomic_numbers.tolist():
            for sym, zz in Z_OF_METAL.items():
                if z == zz:
                    c[sym] += 1
    return dict(c)


@pytest.mark.skipif(not _HAVE_TMQM, reason="tmQM corpus not present")
def test_min_per_metal_kwarg_is_accepted():
    """Without the patch this raises TypeError -- that IS the red check.

    Backwards compatibility: ``min_per_metal`` must be a keyword on the
    callable loader. If the function signature is unchanged, the call
    raises TypeError and this test fails.
    """
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        str(TMQM_DIR),
        metals=SIX_METALS,
        n_clouds=192,
        seed=0,
        split="val",
        val_frac=0.2,
        min_per_metal=1,
    )
    assert clouds, "loader returned empty list; cannot evaluate"


@pytest.mark.skipif(not _HAVE_TMQM, reason="tmQM corpus not present")
def test_min_per_metal_enforces_one_pt_in_seed0_val_window():
    """The load-bearing test for the zero-Pt finding (red without the patch).

    The seed=0 val window at n_clouds=192 is the cohort reported in
    cfm_todo.json::BLK-C-OBJ::zero_platinum_BLOCKER. Before the patch
    the Pt count is exactly 0. After the patch with min_per_metal=1 it
    must be >= 1 -- and, symmetrically, every other requested metal
    must be >= 1.

    Without the patch this test is a TypeError + assertion failure on
    any node where Pt is missing; with the patch the patch's own
    enforcement makes it green.
    """
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        str(TMQM_DIR),
        metals=SIX_METALS,
        n_clouds=192,
        seed=0,
        split="val",
        val_frac=0.2,
        min_per_metal=1,
    )
    mc = _metal_counts(clouds)
    assert mc.get("Pt", 0) >= 1, (
        f"min_per_metal=1 selected but Pt count = {mc.get('Pt', 0)}; "
        f"full distribution = {mc}. This is the zero-Pt finding restated "
        "in a unit test."
    )
    for m in SIX_METALS:
        assert mc.get(m, 0) >= 1, (
            f"min_per_metal=1 should give every requested metal >= 1; "
            f"got {mc.get(m, 0)} for {m}. full distribution = {mc}."
        )


@pytest.mark.skipif(not _HAVE_TMQM, reason="tmQM corpus not present")
def test_default_window_is_metal_representative():
    """The default (``min_per_metal=0``) cohort must not be single-metal.

    Wf-A113 (T67, 2026-10-28) — this replaces
    ``test_min_per_metal_zero_is_backward_compatible``, which pinned the
    opposite: ``{"Rh": 35, "Re": 38, "Ru": 64, "Ir": 45, "Os": 10}`` with
    ``Pt == 0``.  That "backward compatible" distribution was the defect, not
    the baseline: the loader reservoir-sampled the first ``5 * n_clouds``
    candidates of a METAL-BLOCKED file order, so the seed=0 val window at
    n_clouds=192 was drawn from a Re/Ru/Os-only prefix.  Pinning it meant the
    bug could never be fixed without a red test.

    MEASURED before the fix, on the shipped corpus: the first 512 filtered
    candidates are 100% Re and the first 4096 contain no Pt at all, while Pt
    is 24.5% of the filtered pool (1641/6714 in the val split).

    The invariant asserted here is the one that actually matters: every
    requested metal is present, and the composition tracks the pool.
    """
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        str(TMQM_DIR),
        metals=SIX_METALS,
        n_clouds=192,
        seed=0,
        split="val",
        val_frac=0.2,
    )
    assert len(clouds) == 192
    mc = _metal_counts(clouds)
    for m in SIX_METALS:
        assert mc.get(m, 0) >= 1, (
            f"default cohort is missing {m} entirely: {mc}.  This is the "
            "zero-Pt finding, which the metal-stratified draw exists to "
            "prevent."
        )
    # Pool share in the val split is Pt 1641/6714 = 24.4%; allow the draw to
    # land anywhere in a broad band around it, and require that no single
    # metal swallows the cohort.
    pt = mc.get("Pt", 0)
    assert 0.15 * len(clouds) <= pt <= 0.35 * len(clouds), (
        f"Pt is {pt}/{len(clouds)} of the default cohort; the pool share is "
        f"24.4%. full distribution = {mc}."
    )
    assert max(mc.values()) <= 0.45 * len(clouds), (
        f"one metal dominates the default cohort: {mc}."
    )


@pytest.mark.skipif(not _HAVE_TMQM, reason="tmQM corpus not present")
def test_min_per_metal_underflow_raises():
    """min_per_metal * |metals| > n_clouds is a configuration error.

    The patch refuses to silently truncate the quota -- it raises
    ValueError so a caller passing n_clouds=10 and min_per_metal=5
    gets an immediate signal instead of a partial back-fill.
    """
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    with pytest.raises(ValueError, match=r"min_per_metal="):
        load_tmqm_clouds(
            str(TMQM_DIR),
            metals=SIX_METALS,
            n_clouds=10,
            seed=0,
            split="val",
            val_frac=0.2,
            min_per_metal=5,
        )
