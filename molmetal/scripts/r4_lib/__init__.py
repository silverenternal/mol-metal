"""r4_lib — refactored internal package for ``r4_lambda_only_run.py``.

This package is the result of the R1 split (IMPROVEMENT_BACKLOG extension
item R1, 2026-09-25).  The original single-file
``molmetal/scripts/r4_lambda_only_run.py`` was 5270 lines mixing argparse,
Lambda MCTS, click rules, an 84-rule MLE matrix, 10-pocket driver logic
and reporting.  Splitting it into a package improves maintainability.

Module layout
-------------
* :mod:`.constants` — Click-rule aliases, metal-seed SMILES, R17 sweep
  presets.  Pure data + small helpers, no heavy imports.
* :mod:`.scoring`   — Five Lambda-native scoring channels (alpha-eq,
  click-rule bonus, metal-geometry prior, RDKit validity,
  synthesizability).  Lightweight, only RDKit / molmetal_lam.
* :mod:`.metrics`   — All RDKit / numpy / heuristic metric wrappers
  (P0 metrics, P3b V2 metrics, T25 metallodrug proxies, SA / QED,
  per-pocket diversity, RMSD / CoM shift).  Heaviest module — many
  metric_*_mean functions.
* :mod:`.aggregator` — :class:`CellResult` dataclass + the
  ``build_lambda_only_aggregator`` factory used by
  :func:`.runner.run_one_cell`.
* :mod:`.rules`     — Click-rule alias resolver +
  :func:`pocket_derived_metal_seed` (WF-Pocket-Invariance Phase 2C).
* :mod:`.runner`    — Per-cell + per-sweep runners (the largest module
  by line count).  Imports molmetal_lam + RDKit.
* :mod:`.report`    — ``_render_summary_md`` Markdown serialiser.
* :mod:`.cli`       — ``argparse`` definition + ``_resolve_paper_sweep_preset``
  + ``main`` entry point.

Backward-compat re-exports
--------------------------
Every public name previously exposed by ``r4_lambda_only_run.py`` is
re-exported here so the original import surface
(``from molmetal.scripts import r4_lambda_only_run as r4``) keeps
working — see :mod:`molmetal.scripts.r4_lambda_only_run` for the
thin shim that wires this package back to the legacy module name.

Design contract (DO NOT BREAK WITHOUT MIGRATION NOTE)
-----------------------------------------------------
1. No behaviour change.  ``python r4_lambda_only_run.py --smoke`` and
   the library entry points (``run_sweep``, ``run_one_cell``,
   ``CellResult``, ``build_lambda_only_aggregator``, ``METAL_SEED_SMILES``
   etc.) are byte-identical to the pre-R1 file.
2. ``globals().get(\"_R4_PLATINAI_ORACLE\")`` /
   ``globals().get(\"_R4_WETLAB_MEASUREMENTS\")`` lookups from the
   original main() are preserved via module-level globals in
   :mod:`.runner` (so the legacy module globals still resolve when
   the script is invoked via the shim).
"""

from __future__ import annotations

# Re-export every public name from the sub-modules so callers that
# used ``from molmetal.scripts.r4_lambda_only_run import X`` keep
# working after the R1 split.
from .constants import (  # noqa: F401
    CLICK_RULE_ALIASES,
    CLICK_RULE_WEIGHT,
    METAL_PRIOR_WEIGHT,
    METAL_SEED_SMILES,
    POCKET_DERIVED_METAL_SEED_FALLBACK,
    _POCKET_METAL_BUCKETS,
    _R17_LEGACY_DEFAULTS,
    _R17_PAPER_GRADE_PRESET,
    _R17_SMOKE_PRESET,
)
from .scoring import (  # noqa: F401
    alpha_equivalence_uniqueness_score,
    click_rule_match_bonus,
    metal_compliance_truthful,
    metal_geometry_prior_bonus,
    rdkit_validity_score,
    synthesizability_via_lambda_paths,
)
from .metrics import (  # noqa: F401
    _embed_3d_for_rmsd,
    _mol_center_of_mass,
    _safe_mol,
    _tanimoto,
    _import_metrics_v2,
    _import_metallodrug_proxies,
    diversity_alpha,
    diversity_homotype,
    metric_ames_mutagen_mean,
    metric_anticancer_index,
    metric_aqueous_solubility_logS_mean,
    metric_cell_permeability_logPapp_mean,
    metric_com_shift_mean,
    metric_coordination_number_mean,
    metric_dna_kb_proxy,
    metric_gi50_proxy_mean,
    metric_gsh_evasion_score,
    metric_hepatotox_index_mean,
    metric_herg_cardio_risk_mean,
    metric_logp7_4_mean,
    metric_logp_mean,
    metric_metal_coord_compliance_mean,
    metric_metallodrug_lfse_mean,
    metric_metallodrug_pt_dna_crosslink_mean,
    metric_metallodrug_reduction_potential_mean,
    metric_metallodrug_trans_effect_mean,
    metric_monodentate_cl_count,
    metric_oxidation_state_distribution,
    metric_plasma_protein_binding_mean,
    metric_qed_mean,
    metric_rigid_rmsd_mean,
    metric_rotb_mean,
    metric_sa_mean,
    metric_subpocket_diversity_mean,
    metric_tpsa_mean,
    reference_tanimoto,
    training_set_novelty,
)
from .aggregator import (  # noqa: F401
    CellResult,
    build_lambda_only_aggregator,
)
from .rules import (  # noqa: F401
    _resolve_click_rules_aliases,
    pocket_derived_metal_seed,
)
from .runner import (  # noqa: F401
    _DummySite,
    _import_lambda_stack,
    _load_split_training_smiles,
    _state_from_smi,
    load_manifest,
    load_reference_smiles,
    run_one_cell,
    run_sweep,
)
from .report import _render_summary_md  # noqa: F401
from .cli import (  # noqa: F401
    _build_argparser,
    _resolve_paper_sweep_preset,
    main,
)

# Symbols used internally but kept on the public surface for the
# ``run_pb_production.py`` / ``click_rule_effect_size_study.py``
# consumers that import them by string name from the original module.
from .runner import (  # noqa: F401
    _candidate_smi,
    _parse_pareto_weights,
    _METALLO_POOL_DEFAULT,
    _SPLIT_DEFAULT_DIR,
)

__all__ = [
    # constants
    "CLICK_RULE_ALIASES",
    "CLICK_RULE_WEIGHT",
    "METAL_PRIOR_WEIGHT",
    "METAL_SEED_SMILES",
    "POCKET_DERIVED_METAL_SEED_FALLBACK",
    "_POCKET_METAL_BUCKETS",
    "_R17_LEGACY_DEFAULTS",
    "_R17_PAPER_GRADE_PRESET",
    "_R17_SMOKE_PRESET",
    # scoring
    "alpha_equivalence_uniqueness_score",
    "click_rule_match_bonus",
    "metal_compliance_truthful",
    "metal_geometry_prior_bonus",
    "rdkit_validity_score",
    "synthesizability_via_lambda_paths",
    # metrics
    "diversity_alpha",
    "diversity_homotype",
    "metric_logp_mean",
    "metric_tpsa_mean",
    "metric_rotb_mean",
    "metric_oxidation_state_distribution",
    "metric_coordination_number_mean",
    "metric_monodentate_cl_count",
    "metric_gsh_evasion_score",
    "metric_dna_kb_proxy",
    "metric_anticancer_index",
    "metric_metallodrug_reduction_potential_mean",
    "metric_metallodrug_trans_effect_mean",
    "metric_metallodrug_lfse_mean",
    "metric_metallodrug_pt_dna_crosslink_mean",
    "metric_sa_mean",
    "metric_qed_mean",
    "metric_logp7_4_mean",
    "metric_gi50_proxy_mean",
    "metric_cell_permeability_logPapp_mean",
    "metric_herg_cardio_risk_mean",
    "metric_ames_mutagen_mean",
    "metric_hepatotox_index_mean",
    "metric_aqueous_solubility_logS_mean",
    "metric_plasma_protein_binding_mean",
    "metric_subpocket_diversity_mean",
    "metric_metal_coord_compliance_mean",
    "metric_rigid_rmsd_mean",
    "metric_com_shift_mean",
    "training_set_novelty",
    "reference_tanimoto",
    "_tanimoto",
    "_safe_mol",
    "_embed_3d_for_rmsd",
    "_mol_center_of_mass",
    "_import_metrics_v2",
    "_import_metallodrug_proxies",
    # aggregator
    "CellResult",
    "build_lambda_only_aggregator",
    # rules
    "_resolve_click_rules_aliases",
    "pocket_derived_metal_seed",
    # runner
    "_import_lambda_stack",
    "_DummySite",
    "_state_from_smi",
    "_load_split_training_smiles",
    "load_manifest",
    "load_reference_smiles",
    "run_one_cell",
    "run_sweep",
    "_candidate_smi",
    "_parse_pareto_weights",
    "_METALLO_POOL_DEFAULT",
    "_SPLIT_DEFAULT_DIR",
    # report
    "_render_summary_md",
    # cli
    "main",
    "_build_argparser",
    "_resolve_paper_sweep_preset",
]
