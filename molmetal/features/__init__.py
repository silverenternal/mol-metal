"""Feature blocks for the MSRS response-surface model.

This package is the first slice of FIX-P1: one place where a feature block is
defined, so an experiment changes a *spec*, not a new ad-hoc builder.

Why it exists
-------------
Before this package there were four independent builders
(``msrs_surface_model.base_blocks``, ``msrs_code_levers.rdkit_block``,
``build_172d_features.compute_172d``, ``msrs_172d_rebuild.fragment_features``)
with three different missing-value policies and no shared definition of what a
"ligand" is.  The audit on 2026-10-18 found that every one of them featurized
``ligand_main_fragment`` even though 99.93% of the cohort rows are
multi-component complexes.
"""

from molmetal.features.ligand import (  # noqa: F401
    ancillary_smiles,
    complex_ligand_blocks,
    component_smiles,
)

__all__ = ["ancillary_smiles", "complex_ligand_blocks", "component_smiles"]
