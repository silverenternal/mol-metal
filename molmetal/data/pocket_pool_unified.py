"""molmetal/data/pocket_pool_unified.py — single API over the 2 pool sources.

There are two pocket pools in the project:

  1. **crossdocked2020** (100K train + 100 test) — the canonical
     SBDD benchmark, in ``/home/hugo/codes/pocket/data/crossdocked2020/``.
     Every row has an extracted ``pocket_path`` (10 Å sphere around the
     docked ligand) and a ``ligand_path``.

  2. **platiny** (10 literature Pt / Au / Ru targets) — declared in
     ``molmetal/data/platiny_pocket_pool.csv`` (Wf-A31 deliverable).
     At A31 ship time, ALL 10 rows have ``pocket_path=""`` (the
     actual pocket files are extracted in Wf-A33 via P2Rank + Biopython).

For the Wf-A32 wire-in of PocketGenEncoder into the CFM pocket-
conditioning slot, downstream code needs ONE interface. This module
provides it.

Why a thin dispatch wrapper (not a merged CSV / merged dataclass)
------------------------------------------------------------------
The two pools have different semantics:

  * crossdocked2020: 100K real pockets, real ligands, ready-to-use.
  * platiny: 10 literature targets with no 1:1 binding annotation —
    the pool is honest-negative (per audit finding #1) about this.

Merging them into one CSV would lose this distinction. Keeping them
separate and providing a unified query API lets the CFM pipeline
choose per-call: ``load_pocket_pool("crossdocked2020")`` for in-domain
training; ``load_pocket_pool("platiny", only_extracted=False)`` for
honest eval-time framing.

Usage
-----
    from molmetal.data.pocket_pool_unified import load_pocket_pool

    # All crossdocked train pockets (capped at 10k for R21 GPU budget)
    train = load_pocket_pool(source="crossdocked2020", split="train", max_n=10000)

    # All platiny targets (10 literature entries, all with empty
    # pocket_path until Wf-A33 extraction)
    platiny = load_pocket_pool(source="platiny")

    # Only platiny targets that have been extracted
    extracted = load_pocket_pool(source="platiny", only_extracted=True)
    # -> [] at A31 ship time (honest)

    # All sources combined (use with care: the two pools are NOT
    # semantically equivalent)
    all_rows = load_pocket_pool(source="all", max_n=10000)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from molmetal.data import crossdocked_pool
from molmetal.data import platiny_pocket_pool

#: Union of the two PocketEntry dataclasses. We keep them as their
#: native types (NOT a merged dataclass) so the source is preserved
#: at runtime; callers that need a uniform record can branch on
#: ``isinstance(row, platiny_pocket_pool.PocketEntry)`` or read the
#: ``__pool_source__`` attribute (set below).
PocketEntry = Union[crossdocked_pool.PocketEntry, platiny_pocket_pool.PocketEntry]

#: The supported source values for ``load_pocket_pool(source=...)``.
SOURCES: tuple = ("crossdocked2020", "platiny", "all")


def _tag(row, source: str):
    """Attach a ``__pool_source__`` string to the row.

    The crossdocked_pool.PocketEntry is a frozen dataclass; we cannot
    add a true attribute. The platiny PocketEntry is also frozen.
    The convention used here is to wrap the row in a simple namespace
    via the ``__class__.__name__`` discrimination; the source tag is
    surfaced via a helper :func:`source_of` that consults the class.

    Returns the row unchanged; the class is the discriminator.
    """
    return row


def source_of(row: PocketEntry) -> str:
    """Return the pool source of a row, using its concrete class.

    Use this instead of an instance attribute (the dataclasses are
    frozen) so the source is preserved across copies and JSON
    round-trips.
    """
    cls = type(row).__name__
    if cls == "PocketEntry":
        # Disambiguate the two: platiny has metal_specificity,
        # crossdocked does not.
        if hasattr(row, "metal_specificity"):
            return "platiny"
        return "crossdocked2020"
    return f"unknown({cls})"


def load_pocket_pool(
    source: str = "all",
    split: str = "all",
    max_n: Optional[int] = None,
    metal: Optional[str] = None,
    only_extracted: bool = False,
    require_paths_exist: bool = False,
) -> List[PocketEntry]:
    """Unified loader for the crossdocked2020 + platiny pools.

    Parameters
    ----------
    source : str
        One of ``"crossdocked2020"`` (canonical SBDD benchmark),
        ``"platiny"`` (10 literature Pt / Au / Ru targets), or
        ``"all"`` (combined, use with care).
    split : str
        For crossdocked: ``"train"`` (100K) or ``"test"`` (100, DiffDock).
        For platiny: ``"eval_only"`` (default) or ``"all"``.
        For ``source="all"``, the split is interpreted by each backend
        independently.
    max_n : int | None
        Cap on returned rows (applied per-source before concat).
    metal : str | None
        Filter by metal specificity (platiny-only; e.g. ``"Pt_II"``).
        Ignored for crossdocked2020.
    only_extracted : bool
        Return only rows whose ``pocket_path`` is non-empty.
    require_paths_exist : bool
        Return only rows whose ``pocket_path`` and ``ligand_path``
        exist on disk. Crossdocked-only by default; platiny has no
        paths to verify until Wf-A33.

    Returns
    -------
    list[PocketEntry]
        Combined list. Use :func:`source_of` to discriminate.

    Raises
    ------
    ValueError
        If ``source`` is not one of the supported values.
    """
    source = str(source).strip().lower()
    if source not in SOURCES:
        raise ValueError(
            f"source must be one of {SOURCES!r}, got {source!r}"
        )

    out: list = []

    if source in ("crossdocked2020", "all"):
        cd_split = _translate_split_for_crossdocked(split)
        cd_rows = crossdocked_pool.list_pocket_pool(
            split=cd_split,
            max_n=max_n,
            require_paths_exist=require_paths_exist,
        )
        # Only platiny has metal filter; ignore here.
        out.extend(cd_rows)

    if source in ("platiny", "all"):
        pp_split = _translate_split_for_platiny(split)
        pp_rows = platiny_pocket_pool.list_pocket_pool(
            split=pp_split,
            metal_specificity=metal,
            only_extracted=only_extracted,
        )
        if max_n is not None:
            pp_rows = pp_rows[:max_n]
        out.extend(pp_rows)

    return out


def _translate_split_for_crossdocked(split: str) -> str:
    """Map a unified ``split`` value to the crossdocked backend."""
    s = str(split).strip().lower()
    if s in ("train", "test"):
        return s
    if s in ("eval_only", "all", "val"):
        # crossdocked has no "val" / "eval_only"; default to train
        return "train"
    raise ValueError(
        f"split must be one of ('train', 'test', 'eval_only', 'all', 'val'), "
        f"got {split!r}"
    )


def _translate_split_for_platiny(split: str) -> str:
    """Map a unified ``split`` value to the platiny backend."""
    s = str(split).strip().lower()
    if s in ("eval_only", "all"):
        return s
    if s in ("train", "test", "val"):
        # platiny pool is eval-only; we don't have train/test splits.
        # Return empty list instead of raising — the unified API
        # should be safe to call with crossdocked-style split strings
        # when the caller really wants "give me whatever platiny has".
        return "all"  # returns all 10 rows; caller can filter downstream
    raise ValueError(
        f"split must be one of ('train', 'test', 'eval_only', 'all', 'val'), "
        f"got {split!r}"
    )


def pool_summary(source: str = "all", **kwargs) -> dict:
    """Return a summary of one or both pools (size, splits, metals).

    Useful for honest-framing prints in trainer / eval scripts.
    """
    rows = load_pocket_pool(source=source, **kwargs)
    summary = {
        "source": source,
        "total": len(rows),
        "by_source": {},
        "metals": {},
    }
    for r in rows:
        s = source_of(r)
        summary["by_source"][s] = summary["by_source"].get(s, 0) + 1
        if s == "platiny" and hasattr(r, "metal_specificity"):
            m = r.metal_specificity or "(unspecified)"
            summary["metals"][m] = summary["metals"].get(m, 0) + 1
    return summary


# ----------------------------------------------------------------------
# Self-test (smoke)
# ----------------------------------------------------------------------

def _smoke() -> int:
    """Smoke test: dispatch + filter + source discrimination."""
    # 1. Default (source='all') returns both pools concatenated.
    rows = load_pocket_pool(source="all", max_n=5)  # cap each to keep smoke fast
    assert len(rows) > 0, "expected non-empty result for source='all'"
    by_source = {source_of(r) for r in rows}
    print(f"[unified-pool] source='all' max_n=5: {len(rows)} rows, sources={by_source}")

    # 2. source='crossdocked2020' returns only crossdocked rows.
    cd_rows = load_pocket_pool(source="crossdocked2020", split="test", max_n=5)
    cd_sources = {source_of(r) for r in cd_rows}
    assert cd_sources == {"crossdocked2020"} or cd_sources == set(), (
        f"expected only crossdocked2020, got {cd_sources}"
    )
    print(f"[unified-pool] source='crossdocked2020' test max_n=5: {len(cd_rows)} rows")

    # 3. source='platiny' returns the 10 literature targets.
    p_rows = load_pocket_pool(source="platiny")
    assert len(p_rows) == 10, f"expected 10 platiny rows, got {len(p_rows)}"
    p_sources = {source_of(r) for r in p_rows}
    assert p_sources == {"platiny"}, f"expected only platiny, got {p_sources}"
    print(f"[unified-pool] source='platiny': {len(p_rows)} rows (all 10 targets)")

    # 4. metal filter on platiny.
    pt_rows = load_pocket_pool(source="platiny", metal="Pt_II")
    assert len(pt_rows) >= 1, "Pt_II filter returned 0 platiny rows"
    for r in pt_rows:
        assert "Pt_II" in r.metal_specificity
    print(f"[unified-pool] source='platiny' metal=Pt_II: {len(pt_rows)} rows")

    # 5. only_extracted on platiny at A31 ship time -> [].
    extracted = load_pocket_pool(source="platiny", only_extracted=True)
    assert extracted == [], (
        f"platiny only_extracted returned {len(extracted)}; "
        "this pool is not yet extracted — verify Wf-A33 status"
    )
    print("[unified-pool] source='platiny' only_extracted=True: [] (honest)")

    # 6. Bad source.
    try:
        load_pocket_pool(source="not_a_pool")
    except ValueError as exc:
        print(f"[unified-pool] bad source rejected OK: {str(exc)[:60]}")
    else:
        print("[unified-pool] FAIL: bad source did not raise")
        return 1

    # 7. pool_summary.
    s = pool_summary(source="platiny")
    assert s["total"] == 10
    assert s["by_source"].get("platiny", 0) == 10
    assert len(s["metals"]) >= 1
    print(f"[unified-pool] pool_summary(platiny): {s}")

    print("[unified-pool] PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
