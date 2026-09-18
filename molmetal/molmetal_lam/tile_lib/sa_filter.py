"""Top-10 highest-SA tile blacklist (WF-SA-Fragment-Pool-Optimize).

This module is the implementation artefact of the diagnostic scan
emitted in :mod:`molmetal.reports.wf_sa_fragment_pool_optimize.scan_pool_sa`.

Background
----------
The 200+ tile Phase-1 pool was assembled from ChEMBL reactive-handle
+ ZINC click-chemistry subset (Sterling & Irwin 2015).  Some
fragments in that pool are SMARTS-valid but **synthetically
challenging** -- e.g. an α-azido-alcohol that is unstable to
decomposition, an alkyne-thioether that competes with itself, or a
polycyclic dicyclopentadiene partial that is hard to procure.

Ertl-Schuffenhauer SA-score (range [1, 10], lower = easier) measured
across the full 220-tile pool gives the top-10 listed below.

The MCTS that drives the Lambda generator naturally samples from the
pool, so the highest-SA tiles inflate the leaf-reward SA channel
even when the SA weight is positive.  We therefore exclude the top-10
by default.  The user can re-enable them with the
``--keep-high-sa-tiles`` flag (chemistry baseline or ablation
comparison only).

References
----------
* Ertl & Schuffenhauer, *J. Chem. Inf. Model.* 2009, 49, 1453.
* REINVENT4 paper, Blaschke et al. *J. Chem. Inf. Model.* 2024
  (MOSES benchmark reports mean SA ≈ 3.5 for generative models;
  TargetDiff / Pocket2Mol report SA in [2.65, 2.86] for their
  best models).
* MOSES benchmark, Polykovskiy et al. *Front. Pharmacol.* 2020.
"""

from __future__ import annotations

from typing import FrozenSet, List, Tuple

__all__ = [
    "TOP10_HIGHEST_SA_SMILES",
    "TOP10_HIGHEST_SA_PAIRS",
    "top10_filter_pool",
    "is_top10_highest_sa",
]


# Top-10 highest-SA tiles from the diagnostic scan on the validated
# 220-tile pool (2026-09-15).  These are the SMILES literals emitted
# by the FRAGMENT_POOL_* constants after RDKit validation; copy them
# verbatim so a 1-character difference in the canonicalization does
# not silently drop / retain a tile.
#
# Source data: molmetal/reports/wf_sa_fragment_pool_optimize/sa_pool_scan.csv
#
# (rank, smiles, sa_score, tag_family_hint)
#   1  C1=C[SH]=C1                       5.808  thiol (diene-aromatic, unusual)
#   2  [N-]=[N+]=NC(O)CO                 4.700  azide  (alpha-hydroxy azide, unstable)
#   3  C1=CC2C=CC1C2                     4.699  diene  (dicyclopentadiene partial, hard to procure)
#   4  CC(O)N=[N+]=[N-]                  4.640  azide  (1-azidoethanol, alpha-hydroxy)
#   5  C=Cc1cnn[nH]1                     4.370  diene  (vinyl-pyrazole, vinyl on NH heterocycle)
#   6  C=CCN=[N+]=[N-]                   4.133  azide  (allyl azide)
#   7  CCN=[N+]=[N-]                     4.092  azide  (ethyl azide)
#   8  [N-]=[N+]=Nc1ccn[nH]1             3.964  azide  (4-azidopyrazole)
#   9  CC(N=[N+]=[N-])C(=O)O             3.914  azide  (2-azidopropanoic acid)
#  10  C#CCSC                            3.887  alkyne (propargyl methyl sulfide, thiol-yne competition)
TOP10_HIGHEST_SA_PAIRS: List[Tuple[int, str, float]] = [
    (1,  "C1=C[SH]=C1",                 5.8081076618525955),
    (2,  "[N-]=[N+]=NC(O)CO",           4.70000358052036),
    (3,  "C1=CC2C=CC1C2",               4.698507961190864),
    (4,  "CC(O)N=[N+]=[N-]",            4.6403474653593655),
    (5,  "C=Cc1cnn[nH]1",               4.369602048062541),
    (6,  "C=CCN=[N+]=[N-]",             4.133113240593854),
    (7,  "CCN=[N+]=[N-]",               4.091661518383038),
    (8,  "[N-]=[N+]=Nc1ccn[nH]1",       3.9635812400139168),
    (9,  "CC(N=[N+]=[N-])C(=O)O",       3.9144544781573547),
    (10, "C#CCSC",                      3.8873987964895464),
]

# Convenience: bare SMILES set for O(1) membership tests.
TOP10_HIGHEST_SA_SMILES: FrozenSet[str] = frozenset(p[1] for p in TOP10_HIGHEST_SA_PAIRS)


def is_top10_highest_sa(smiles: str) -> bool:
    """Return True if ``smiles`` is one of the top-10 highest-SA tiles."""
    return smiles in TOP10_HIGHEST_SA_SMILES


def top10_filter_pool(tiles, *, keep_high_sa: bool = False):
    """Filter ``tiles`` to exclude the top-10 highest-SA entries by default.

    Parameters
    ----------
    tiles : Iterable[Tile]
        The pool of Tile instances to filter.  Tiles are matched by
        their ``smiles`` attribute against the top-10 list.
    keep_high_sa : bool, optional
        If True, the top-10 are NOT filtered out (the pool is returned
        unchanged).  Default False so the canonical MCTS path benefits
        from a SA-cleaner pool without a CLI flag.

    Returns
    -------
    List[Tile]
        Filtered pool.  If ``keep_high_sa=False`` the top-10 are
        removed; if the resulting list is empty (all tiles were in
        the blacklist) the function falls back to the unfiltered pool
        so a downstream MCTS never sees an empty tile library.
    """
    # Materialize once so we can both filter and measure in the
    # caller's downstream code.
    materialised = list(tiles)
    if keep_high_sa:
        return materialised
    filtered = [t for t in materialised if not is_top10_highest_sa(getattr(t, "smiles", ""))]
    # Defensive: if every tile was on the blacklist (cannot happen in
    # practice because the pool has 220 tiles and the blacklist has
    # 10, but guard anyway) return the unfiltered pool so downstream
    # search never sees an empty library.
    if not filtered:
        return materialised
    return filtered
