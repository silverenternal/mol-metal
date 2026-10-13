"""Coordination-geometry classifier (Wf-A27 step-1, 2026-10-13).

Rule-based lookup table: ``(metal_atomic_number, max_coordination_number)
-> geometry_class`` for the precious-metal / d-block cohort used by the
project's CFM trainer.

This is a deliberate, documented rule-based stub — NOT a learned feature.
It is honest-negative catalog §6.48: the table covers a CURATED subset of
(metal Z, CN) combinations that are well-known in the coordination-chemistry
literature (Housecroft & Sharpe 2018, Miessler 2013); every other query
returns ``"unknown"``.  The xtb GPU optimisation-based classifier is
deferred to Wf-A28 (the GPU path was deliberately NOT invoked per the
2026-10-13 user directive "gpu先别动").

Why a str and not an int
------------------------
Storing the class as a str keeps the JSON side-channel at
``AtomCloud.coord_geom_class`` human-readable and matches the
``molmetal/eval/scoring/pocket2mol_score.py`` vocabulary.  Callers that
need a numerical one-hot should use :func:`one_hot_coord_geometry`.

Backward-compat rule (AGENTS.md §5.11)
--------------------------------------
When :func:`classify_coord_geometry` returns ``"unknown"`` (default for
``coord_number < 2``, ``coord_number > 12``, or any unmapped
(metal_z, coord_number) pair), the :class:`AtomCloud` loader must leave
:attr:`AtomCloud.coord_geom_class` at ``None``.  This preserves the
byte-identical 63-construction-site contract that Wf-A11 / Wf-A15 / Wf-A16
already uphold.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Public constants — vocabulary kept tight so a noisy upstream Smith-style
# 12-category table does not leak into the trainer's hot path.
# ---------------------------------------------------------------------------
COORD_GEOMETRY_VOCAB: Tuple[str, ...] = (
    "linear",
    "trigonal_planar",
    "tetrahedral",
    "trigonal_bipyramidal",
    "octahedral",
    "square_planar",
    "square_pyramidal",
    "pentagonal_bipyramidal",
    "unknown",
)

# Aliases that downstream scoring code may emit (kept import-time safe).
COORD_GEOMETRY_ALIASES: Dict[str, str] = {
    "sp": "square_planar",
    "oct": "octahedral",
    "tet": "tetrahedral",
    "tbp": "trigonal_bipyramidal",
    "sqp": "square_pyramidal",
    "pbp": "pentagonal_bipyramidal",
}

# ---------------------------------------------------------------------------
# Curated hard-coded table (metal atomic number, CN) -> class.  Layout
# rationale: keyed by metal Z for O(1) lookup; values are flat tuples so a
# `Dict[int, Tuple[Dict[int, str], ...]]` is overkill for ~30 rows.  An
# unmapped (Z, CN) -> "unknown" is the safety floor.
# ---------------------------------------------------------------------------
_TABLE: Dict[int, Dict[int, str]] = {
    # Pt Z=78 — most-studied precious metal in this project.
    78: {
        2: "linear",
        4: "square_planar",
        5: "square_pyramidal",
        6: "octahedral",
    },
    # Pd Z=46 — square-planar Pd(II) is the textbook default.
    46: {
        2: "linear",
        4: "square_planar",
        5: "square_pyramidal",
        6: "octahedral",
    },
    # Ir Z=77 — Ir(III) octahedral; Ir(I) linear (Vaska-class).
    77: {
        2: "linear",
        4: "square_planar",
        5: "trigonal_bipyramidal",
        6: "octahedral",
    },
    # Ru Z=44 — Ru(II/III) octahedral dominant; Ru(0) sees trigonal bipyramidal.
    44: {
        4: "tetrahedral",
        5: "trigonal_bipyramidal",
        6: "octahedral",
    },
    # Rh Z=45 — square-planar Rh(I) (Wilkinson), octahedral Rh(III).
    45: {
        4: "square_planar",
        5: "trigonal_bipyramidal",
        6: "octahedral",
    },
    # Os Z=76 — octahedral dominant.
    76: {
        5: "trigonal_bipyramidal",
        6: "octahedral",
    },
    # Re Z=75 — Re(V/VII) oxo complexes are tetrahedral / octahedral.
    75: {
        4: "tetrahedral",
        6: "octahedral",
    },
    # Au Z=79 — Au(I) linear; Au(III) square-planar.
    79: {
        2: "linear",
        4: "square_planar",
        5: "square_pyramidal",
    },
    # Ag Z=47 — Ag(I) linear 2-coordinate dominant; Ag(II) square-planar
    # rare but documented (Housecroft).
    47: {
        2: "linear",
        4: "square_planar",
    },
    # Cu Z=29 — Cu(I) linear / tetrahedral; Cu(II) square-planar / octahedral.
    29: {
        2: "linear",
        4: "square_planar",
        5: "square_pyramidal",
        6: "octahedral",
    },
    # Ni Z=28 — Ni(II) square-planar / octahedral dominant.
    28: {
        4: "square_planar",
        5: "trigonal_bipyramidal",
        6: "octahedral",
    },
    # Fe Z=26 — octahedral dominant; tetrahedral Fe(III) in sulfido clusters.
    26: {
        4: "tetrahedral",
        5: "trigonal_bipyramidal",
        6: "octahedral",
    },
    # Co Z=27 — octahedral / tetrahedral.
    27: {
        4: "tetrahedral",
        6: "octahedral",
    },
    # Mn Z=25 — octahedral dominant.
    25: {
        6: "octahedral",
    },
    # Zn Z=30 — tetrahedral dominant (Zn(II) coordination sphere).
    30: {
        4: "tetrahedral",
        5: "trigonal_bipyramidal",
        6: "octahedral",
    },
}


def _normalize(s: str) -> Optional[str]:
    """Map a free-form string to one of the canonical classes.

    Returns ``None`` for empty / unknown inputs so callers can short-circuit
    to the ``"unknown"`` sentinel rather than embedding the alias map at
    every call site.
    """
    if not s:
        return None
    t = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    if t in COORD_GEOMETRY_VOCAB:
        return t
    return COORD_GEOMETRY_ALIASES.get(t)


def classify_coord_geometry(
    coord_number: int,
    atom_z: int,
) -> Optional[str]:
    """Return the coordination-geometry class for ``atom_z`` at CN ``coord_number``.

    Parameters
    ----------
    coord_number
        Per-atom maximum coordination number.  Caller is responsible for
        computing it (we deliberately do NOT do a graph walk in this
        module so the loader hot path stays O(1)).
    atom_z
        Atomic number of the candidate metal atom.  For non-metal
        ``atom_z`` (e.g. C / N / O / S / halogens) we return ``None``
        even at valid CN — the classifier is metal-only.

    Returns
    -------
    ``"linear" | "trigonal_planar" | "tetrahedral" | "trigonal_bipyramidal"
    | "octahedral" | "square_planar" | "square_pyramidal" |
    "pentagonal_bipyramidal" | None``

    ``None`` is the loader-honest signal: write it as
    ``AtomCloud.coord_geom_class = None`` and the existing trainer path
    sees no change.  See the module docstring for the backward-compat rule.

    Notes
    -----
    Wf-A27 step-1 deliberately does NOT use an xtb GPU optimisation path —
    it is the CPU lookup the §6.48 honest-negative close-out requires.
    Wf-A28 may add a parallel ``classify_coord_geometry_xtb(...)`` that
    consumes an ``AtomCloud`` + an ``xtb_get_geometry`` result; today that
    is out of scope.
    """
    cn = int(coord_number)
    z = int(atom_z)
    # ``coord_number`` outside [2, 12] has no entry in the curated table.
    if cn < 2 or cn > 12:
        return None
    sub = _TABLE.get(z)
    if sub is None:
        return None
    cls = sub.get(cn)
    return cls


def one_hot_coord_geometry(
    cls: Optional[str],
    n_classes: int = len(COORD_GEOMETRY_VOCAB),
    drop_unknown: bool = True,
):
    """Map a str class to its one-hot index.  ``None`` returns ``None``.

    Helper for the (deferred) Wf-A28 wiring where a hot-vector channel
    joins ``AtomCloud.atom_in_dim``.  ``drop_unknown=True`` (the default)
    places the unknown class in the LAST slot so a trained model can
    emit ``"unknown"`` by predicting against the soft-max over all
    ``n_classes``.  ``drop_unknown=False`` swaps ``"unknown"`` to slot 0
    (kept for the audit who prefers the simpler indexing).
    """
    if cls is None:
        return None
    norm = _normalize(cls)
    if norm is None or norm == "unknown":
        norm = "unknown"
    if norm not in COORD_GEOMETRY_VOCAB:
        norm = "unknown"
    if drop_unknown:
        # Index = vocab.index(cls); "unknown" lands on the last slot.
        idx = COORD_GEOMETRY_VOCAB.index(norm)
        if idx == len(COORD_GEOMETRY_VOCAB) - 1:
            # "unknown" -> last slot via drop_unknown contract.
            idx = len(COORD_GEOMETRY_VOCAB) - 1
    else:
        # Drop "unknown" from the vocab so slot 0..n_classes-2 covers the
        # named classes and "unknown" -> slot n_classes-1.
        named = [v for v in COORD_GEOMETRY_VOCAB if v != "unknown"]
        if norm == "unknown":
            idx = len(named)
        else:
            idx = named.index(norm)
    # Lazy import: numpy may not be installed in a CPU-only test env.
    try:
        import numpy as np
    except ImportError:
        return None
    out = np.zeros(int(n_classes), dtype=np.float32)
    out[int(idx)] = 1.0
    return out


__all__ = [
    "COORD_GEOMETRY_VOCAB",
    "COORD_GEOMETRY_ALIASES",
    "classify_coord_geometry",
    "one_hot_coord_geometry",
]
