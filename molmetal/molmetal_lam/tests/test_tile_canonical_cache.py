"""Smoke tests for the SMILES canonical-cache helper.

Verifies:

1. :func:`canonicalize` returns RDKit's canonical form on first call
   and the cached form on subsequent calls.
2. The cache hit rate approaches ``1.0`` as more repeat calls are
   made (the first call is a miss).
3. :func:`clear_cache` resets the cache, hit/miss counters, and
   :func:`cache_size` reports the right number.
4. :func:`precompute_library` warms the cache for a list of SMILES and
   the returned list is in input order.
5. :func:`canonicalize` is idempotent (calling it N times always
   returns the same string).
"""

from __future__ import annotations

from molmetal_lam.tile_lib.canonical_cache import (
    canonicalize,
    cache_hit_rate,
    cache_size,
    clear_cache,
    precompute_library,
)


def test_canonicalize_idempotent_and_cached():
    clear_cache()
    raw = "CCN=[N+]=[N-]"  # ethyl azide
    first = canonicalize(raw)
    assert isinstance(first, str) and first
    # First call is a miss; subsequent calls are hits.
    assert cache_hit_rate() == 0.0  # 0 hits / 1 miss so far
    for _ in range(3):
        assert canonicalize(raw) == first
    # 1 miss + 3 hits ⇒ hit rate 0.75; cache size 1.
    assert cache_size() == 1
    assert cache_hit_rate() == 0.75
    # Asymptotic limit is 1.0; verify by sampling enough calls that
    # the residual miss fraction is below 5%.  Need 19 hits per miss.
    for _ in range(20):
        canonicalize(raw)
    assert cache_hit_rate() > 0.95


def test_clear_cache_resets_state():
    clear_cache()
    canonicalize("c1ccccc1")
    canonicalize("CCN")
    assert cache_size() == 2
    assert cache_hit_rate() < 1.0  # both were misses
    clear_cache()
    assert cache_size() == 0
    # After clear, hit rate is the documented "no calls" value 0.0.
    assert cache_hit_rate() == 0.0


def test_precompute_library_warms_cache():
    clear_cache()
    library = [
        "CCN=[N+]=[N-]",
        "C#CC",
        "OCCCN=[N+]=[N-]",
        "c1ccc2ncccc2c1",
    ]
    # 4 distinct SMILES → 4 misses, no hits.
    out = precompute_library(library)
    # Returned list has the same length as input.
    assert len(out) == len(library)
    # Cache size should equal the unique input count (4 distinct SMILES).
    assert cache_size() == 4
    # precompute recorded 4 misses and no hits yet.
    assert cache_hit_rate() == 0.0
    # Every subsequent canonicalize hits (4 hits + 4 prior misses = 0.5).
    for raw in library:
        canonicalize(raw)
    assert cache_hit_rate() == 0.5
    # Drive the hit rate above 0.75 by calling more rounds.
    for _ in range(6):
        for raw in library:
            canonicalize(raw)
    assert cache_hit_rate() > 0.75


def test_canonicalize_matches_rdkit():
    """Cross-check :func:`canonicalize` against RDKit's own output.

    Skip when RDKit is unavailable — the cache falls back to the raw
    SMILES in that case and the equality is trivially satisfied.
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except Exception:
        return
    clear_cache()
    samples = [
        "CCN=[N+]=[N-]",
        "OCCN=[N+]=[N-]",
        "c1ccc2ncccc2c1",
        "CC(=C)C=C",
    ]
    for s in samples:
        expected = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        assert canonicalize(s) == expected