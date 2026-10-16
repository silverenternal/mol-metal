# Licensed under the Apache License, Version 2.0.
"""Regression test for Wf-A57 W9-P3 — geometry dataclass move.

The decoder-side ``_GeometryConstraint`` dataclass + 3 frozen instances
(``_SQUARE_PLANAR`` / ``_OCTAHEDRAL`` / ``_TETRAHEDRAL``) and the
``_geometry_constraint_for_z`` lookup helper moved from
:mod:`molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond` to
:mod:`molmetal.molmetal_lam.priors.metal_geometry` on 2026-10-16.  The
move preserves dataclass identity (same fields, same instances, same
helper), but the import path changes — the old module now re-exports
the names via

    from molmetal.molmetal_lam.priors.metal_geometry import (
        _GeometryConstraint, _SQUARE_PLANAR, _OCTAHEDRAL,
        _TETRAHEDRAL, _geometry_constraint_for_z,
    )

This test pins BOTH:

  1. The new canonical definition lives in
     :mod:`molmetal.molmetal_lam.priors.metal_geometry` and is exported
     from ``__all__``.
  2. The legacy import path from
     :mod:`molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond` still
     resolves to the SAME class object (identity preserved), so
     ``test_cfm_decoder_geometry_constraints.py`` remains PASS.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.priors import metal_geometry  # noqa: E402


def test_geometry_canonical_module_exports_all_five_names() -> None:
    """The canonical ``priors.metal_geometry`` module exposes all 5 names."""
    assert hasattr(metal_geometry, "_GeometryConstraint")
    assert hasattr(metal_geometry, "_SQUARE_PLANAR")
    assert hasattr(metal_geometry, "_OCTAHEDRAL")
    assert hasattr(metal_geometry, "_TETRAHEDRAL")
    assert hasattr(metal_geometry, "_geometry_constraint_for_z")
    # And they are declared in __all__ so downstream ``from X import *``
    # re-exports stay in sync.
    for name in (
        "_GeometryConstraint",
        "_SQUARE_PLANAR",
        "_OCTAHEDRAL",
        "_TETRAHEDRAL",
        "_geometry_constraint_for_z",
    ):
        assert name in metal_geometry.__all__, (
            f"{name!r} missing from metal_geometry.__all__"
        )


def test_decoder_module_re_exports_canonical_classes() -> None:
    """The decoder-side module re-exports the SAME class object, not a copy."""
    from molmetal.molmetal_lam.lam_chem import cfm_decoder_yuelbond

    canonical_cls = metal_geometry._GeometryConstraint
    decoder_cls = cfm_decoder_yuelbond._GeometryConstraint
    assert canonical_cls is decoder_cls, (
        "W9-P3 regression: decoder module no longer re-exports the "
        "canonical _GeometryConstraint from priors.metal_geometry"
    )


def test_geometry_instances_have_matching_identities() -> None:
    """``_SQUARE_PLANAR`` etc. are identity-equal across both modules."""
    from molmetal.molmetal_lam.lam_chem import cfm_decoder_yuelbond

    assert (
        metal_geometry._SQUARE_PLANAR
        is cfm_decoder_yuelbond._SQUARE_PLANAR
    )
    assert (
        metal_geometry._OCTAHEDRAL
        is cfm_decoder_yuelbond._OCTAHEDRAL
    )
    assert (
        metal_geometry._TETRAHEDRAL
        is cfm_decoder_yuelbond._TETRAHEDRAL
    )


def test_lookup_helper_resolves_canonical_z() -> None:
    """``_geometry_constraint_for_z(78)`` returns the square-planar instance
    with ``max_bonds == 4`` (Pt = Z=78)."""
    from molmetal.molmetal_lam.lam_chem import cfm_decoder_yuelbond

    gc = cfm_decoder_yuelbond._geometry_constraint_for_z(78)
    assert gc is metal_geometry._SQUARE_PLANAR
    assert gc.max_bonds == 4
    assert 90.0 in gc.ideal_angles
