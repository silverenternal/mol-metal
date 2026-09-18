"""molmetal.data — data layer for the Mol-Metal framework.

Provides loaders + featurisers for the three primary data sources:

  cytotox      — MetalCytoToxDB.csv  (Ru/Ir/Rh/Os/Re anticancer screening)
  splits       — random / temporal / chemical Tanimoto train/val/test splits
  featurize    — Morgan (2048-bit ECFP4) + PyG-style atom & edge features
  crossdocked  — CrossDocked2020  (100k protein-ligand pairs)

The layer is deliberately *no-deps on adapters*: it only depends on torch,
pandas, rdkit, and the existing molmetal.domain dataclasses.  Adapters (the
flow-matching generator, docking engine, etc.) consume Molecule objects and
training batches that this layer prepares.
"""

from __future__ import annotations

from molmetal.data.cytotox import MetalCytotoxDataset, CytotoxFilter
from molmetal.data.metal_smiles import (
    coordination_capacity,
    count_donors,
    reconstruct_metal_complex,
)
from molmetal.data.splits import (
    ChemicalSplitter,
    LigandDeduplicatedSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    SplitResult,
    TemporalSplitter,
)
from molmetal.data.featurize import MorganFingerprinter, GraphFeaturizer
from molmetal.data.crossdocked import CrossDockedDataset
from molmetal.data.crossdocked_filter import (
    CrossDockedEntry,
    FilterStats,
    filter_by_pdb,
)
from molmetal.data.mmp_targets import (
    InhibitorEntry,
    MMP2_TARGET,
    MMP9_TARGET,
    MMPTarget,
    MMP_ZN_TRIAD,
    all_targets,
    combined_pdb_ids,
    get_target,
)
from molmetal.data.metalloprotein_targets import (
    CA2_TARGET,
    ACE_TARGET,
    HDAC2_TARGET,
    PKA_TARGET,
    CDK2_TARGET,
    CYP3A4_TARGET,
    ADH_TARGET,
    SOD1_TARGET,
    METALLOPROTEIN_TARGETS,
    MetalloproteinTarget,
    all_targets as metalloprotein_all_targets,
    combined_pdb_ids as metalloprotein_combined_pdb_ids,
    families_by_metal,
    get_target as get_metalloprotein_target,
)
from molmetal.data.tmqm import (
    PAPER_METALS,
    TRANSITION_METALS,
    TmqmStats,
    filter_by_metal,
    load_tmqm,
    summarize,
)

__all__ = [
    "MetalCytotoxDataset",
    "CytotoxFilter",
    "coordination_capacity",
    "count_donors",
    "reconstruct_metal_complex",
    "RandomSplitter",
    "TemporalSplitter",
    "ChemicalSplitter",
    "LigandDeduplicatedSplitter",
    "ScaffoldSplitter",
    "SplitResult",
    "MorganFingerprinter",
    "GraphFeaturizer",
    "CrossDockedDataset",
    "CrossDockedEntry",
    "FilterStats",
    "filter_by_pdb",
    "InhibitorEntry",
    "MMP2_TARGET",
    "MMP9_TARGET",
    "MMPTarget",
    "MMP_ZN_TRIAD",
    "all_targets",
    "combined_pdb_ids",
    "get_target",
    "PAPER_METALS",
    "TRANSITION_METALS",
    "TmqmStats",
    "filter_by_metal",
    "load_tmqm",
    "summarize",
]