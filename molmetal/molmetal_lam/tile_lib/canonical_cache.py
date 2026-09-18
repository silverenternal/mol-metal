"""Lazy canonical-SMILES cache shared across the tile library and proof
search.

The LamClick / proof-search pipeline calls
``Chem.MolToSmiles(Chem.MolFromSmiles(smi))`` repeatedly with the same
``smi`` (e.g. on every MCTS visit and every reaction-site enumeration).
RDKit's canonicalisation is O(atoms) but not free, and on hot paths it
shows up in profiles.

This module provides a tiny module-level cache keyed by the *raw* input
SMILES string (not by the canonical form — we want the inverse map from
raw -> canonical to be cached, not the other direction).  All public
APIs are safe to call from any thread that is willing to serialise
Python attribute access; the cache itself is a plain ``dict`` because
callers that care about race conditions can hold their own lock.

API
---
* :func:`canonicalize` - lazily canonicalize a SMILES, hitting the
  cache on repeat calls.
* :func:`clear_cache` - drop every cached entry.
* :func:`cache_size` - return the current number of cached entries.
* :func:`cache_hit_rate` - return ``hits / (hits + misses)`` since the
  process started (or since the last :func:`clear_cache`).
* :func:`precompute_library` - warm the cache for a list of raw SMILES
  so subsequent calls are guaranteed hits.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

__all__ = [
    "canonicalize",
    "clear_cache",
    "cache_size",
    "cache_hit_rate",
    "precompute_library",
]


# Module-level cache: raw SMILES -> canonical SMILES.
# ``Dict[str, str]`` is thread-safe enough for our purposes: CPython's
# GIL serialises dict mutations, and concurrent reads always see a
# fully-published entry (either the old value or the new one).
_TILE_CANONICAL_CACHE: Dict[str, str] = {}

# Hit / miss counters.  Kept module-level so :func:`cache_hit_rate` is
# O(1).
_CACHE_HITS: int = 0
_CACHE_MISSES: int = 0


def _rdkit_canonicalize(smi: str) -> str:
    """Return the RDKit-canonical form of ``smi``.

    Falls back to returning the input verbatim if RDKit is unavailable
    or the SMILES fails to parse — :func:`canonicalize` is on hot paths
    so we never want it to raise.
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except Exception:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return smi


def canonicalize(smi: str) -> str:
    """Return the canonical form of ``smi``, hitting the cache when warm.

    The cache is keyed by the raw SMILES string.  Repeat calls are
    O(1) dict lookups.  The returned canonical SMILES is the same one
    RDKit would produce; the cache is just memoisation.
    """
    global _CACHE_HITS, _CACHE_MISSES
    cached = _TILE_CANONICAL_CACHE.get(smi)
    if cached is not None:
        _CACHE_HITS += 1
        return cached
    _CACHE_MISSES += 1
    canon = _rdkit_canonicalize(smi)
    _TILE_CANONICAL_CACHE[smi] = canon
    return canon


def clear_cache() -> None:
    """Drop every cached entry and reset hit/miss counters."""
    global _CACHE_HITS, _CACHE_MISSES
    _TILE_CANONICAL_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def cache_size() -> int:
    """Return the number of cached (raw -> canonical) entries."""
    return len(_TILE_CANONICAL_CACHE)


def cache_hit_rate() -> float:
    """Return ``hits / (hits + misses)`` since the last :func:`clear_cache`.

    Returns ``0.0`` when no calls have been made yet (rather than
    raising ``ZeroDivisionError``) so tests can assert a clean starting
    state.
    """
    total = _CACHE_HITS + _CACHE_MISSES
    if total == 0:
        return 0.0
    return _CACHE_HITS / total


def precompute_library(library_smiles_list: Iterable[str]) -> List[str]:
    """Warm the cache for every SMILES in ``library_smiles_list``.

    Returns the list of canonical SMILES in input order so callers can
    chain ``precompute_library(library)[i]`` against
    ``canonicalize(library[i])`` if they need a side-by-side view.
    """
    return [canonicalize(smi) for smi in library_smiles_list]