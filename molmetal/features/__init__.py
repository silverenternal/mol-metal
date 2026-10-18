"""Feature blocks for the MSRS response-surface model.

One place where a feature block is defined, so an experiment changes a *spec*,
not a new ad-hoc builder.

Why it exists
-------------
Before this package there were four independent builders
(``msrs_surface_model.base_blocks``, ``msrs_code_levers.rdkit_block``,
``build_172d_features.compute_172d``, ``msrs_172d_rebuild.fragment_features``)
with three different missing-value policies and no shared definition of what a
"ligand" is.  The audit on 2026-10-18 found that every one of them featurized
``ligand_main_fragment`` even though 99.93% of the cohort rows are
multi-component complexes.

Entry points
------------
``build(frame, spec)``
    The single construction site.  Returns a :class:`FeatureBundle` carrying
    the matrix, column names, block slices and the spec hash.
``FeatureSpec`` / ``PRESETS``
    Declarative description of a matrix.  ``PRESETS["legacy_main"]`` reproduces
    the historical builder bit-for-bit.
"""

from molmetal.features.build import FeatureBundle, build, verify_bundle  # noqa: F401
from molmetal.features.ligand import (  # noqa: F401
    ancillary_smiles,
    complex_ligand_blocks,
    component_smiles,
    main_component,
)
from molmetal.features.spec import (  # noqa: F401
    COMPLEX,
    COMPLEX_FULL,
    LEGACY_MAIN,
    PRESETS,
    AssaySpec,
    CellSpec,
    FeatureSpec,
    LigandSpec,
    MetalSpec,
    get_preset,
)

__all__ = [
    "AssaySpec",
    "COMPLEX",
    "COMPLEX_FULL",
    "CellSpec",
    "FeatureBundle",
    "FeatureSpec",
    "LEGACY_MAIN",
    "LigandSpec",
    "MetalSpec",
    "PRESETS",
    "ancillary_smiles",
    "build",
    "complex_ligand_blocks",
    "component_smiles",
    "get_preset",
    "main_component",
    "verify_bundle",
]
