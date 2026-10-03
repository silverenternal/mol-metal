"""r4_lib.runner — Per-cell + per-sweep runners (largest module).

This is the heaviest module in :mod:`r4_lib` by line count — it owns:

* :func:`_import_lambda_stack` — lazy import of every ``molmetal_lam``
  symbol the harness needs (kept inside a function so ``--help`` works
  even when RDKit / molmetal_lam is missing).
* :func:`load_manifest` / :func:`load_reference_smiles` — manifest +
  reference ligand loaders (NO docking).
* :func:`_candidate_smi` / :func:`_parse_pareto_weights` — small
  helpers used by :func:`run_one_cell`.
* :func:`_DummySite` / :func:`_state_from_smi` — fallback
  ``BindingSite`` substitute + ``MoleculeClosedTerm.from_smiles``
  convenience.
* :func:`_load_split_training_smiles` — TODO-30 / P6.1+P6.2 split
  hygiene helper for the ``--split`` CLI flag.
* :func:`run_one_cell` — single (pocket, seed) cell: builds the
  reward aggregator + MCTSProofSearch + post-processing pipeline that
  emits a :class:`.aggregator.CellResult`.
* :func:`run_sweep` — top-level driver: iterates over
  ``manifest_path[:n_pockets] × seeds`` and persists
  ``report.json`` / ``summary.md`` / ``seeds.json``.

Originally lived at lines 517-3974 of ``r4_lambda_only_run.py``.

Module-level globals
--------------------
The original script relied on ``globals().get("_R4_PLATINAI_ORACLE")``
and ``globals().get("_R4_WETLAB_MEASUREMENTS")`` lookups from
:func:`main`.  We preserve the same semantics here by exposing the
two lazy singletons via :data:`_R4_PLATINAI_ORACLE` and
:data:`_R4_WETLAB_MEASUREMENTS` so callers using the shim still find
them in :mod:`r4_lambda_only_run`'s namespace (the shim uses
``globals().update({...})`` for this).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .aggregator import CellResult, build_lambda_only_aggregator
from .constants import METAL_SEED_SMILES, _REPO_ROOT
from .metrics import (
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
from .report import _render_summary_md
from .rules import _resolve_click_rules_aliases, pocket_derived_metal_seed
from .scoring import (
    metal_compliance_truthful,
    rdkit_validity_score,
    synthesizability_via_lambda_paths,
)

# Logger used throughout the module.  Mirrors the original
# ``log = logging.getLogger("wf_lambda1")`` so CLI invocations get
# consistent log records.
log = logging.getLogger("wf_lambda1")

# Lazy module-level singletons — replaced by :func:`run_one_cell` on
# first invocation when the corresponding opt-in channel is enabled.
# Exposed via the legacy ``globals()`` lookup so the shim can forward
# them transparently to ``main()``.
_R4_PLATINAI_ORACLE: Optional[Any] = None
_R4_WETLAB_MEASUREMENTS: Dict[str, float] = {}

# Mirror the original ``_HERE`` / ``_REPO_ROOT`` derivation so the
# manifest / scaffold-split default paths resolve identically.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT_FOR_RUNNER = _HERE.parent.parent.parent


# ---------------------------------------------------------------------------
# Imports — kept inside a function so the script can be --help'd even if
# rdkit or a heavy backend is missing.
# ---------------------------------------------------------------------------
def _import_lambda_stack():
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.lam_chem.rules import CLICK_REACTIONS, METAL_COORDINATION_RULES
    from molmetal_lam.tile_lib.library import (
        FRAGMENT_LIBRARY_200_TILES,
        build_tile_library,
    )
    from molmetal_lam.search_alg.proof_search import MCTSProofSearch
    from molmetal_lam.types.predicates import TypePredicate
    from molmetal_lam.binding.types import BindingSite
    from molmetal_lam.lam_chem.well_formedness import check_beta_normal_form

    return {
        "MoleculeClosedTerm": MoleculeClosedTerm,
        "CLICK_REACTIONS": CLICK_REACTIONS,
        "METAL_COORDINATION_RULES": METAL_COORDINATION_RULES,
        "FRAGMENT_LIBRARY_200_TILES": FRAGMENT_LIBRARY_200_TILES,
        "build_tile_library": build_tile_library,
        "MCTSProofSearch": MCTSProofSearch,
        "TypePredicate": TypePredicate,
        "BindingSite": BindingSite,
        "check_beta_normal_form": check_beta_normal_form,
    }


# ---------------------------------------------------------------------------
# Manifest loader (CrossDocked100)
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> List[Dict[str, str]]:
    """Read the CrossDocked100 manifest CSV and return a list of row dicts.

    Expected columns (per ``molmetal/data/crossdocked100_manifest.csv``):
    ``pocket_id,receptor_path,ligand_path,ref_path,metal_atoms,...``.
    Falls back to empty list if the file does not exist.
    """
    if not path.exists():
        log.warning("Manifest %s does not exist; returning []", path)
        return []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Reference-ligand SMILES loader (NO docking — only SMILES string).
# ---------------------------------------------------------------------------
def load_reference_smiles(ligand_path: Path) -> Optional[str]:
    """Best-effort SMILES extraction from the docked ligand ``.sdf``.

    For the Lambda-only harness we **only** need a SMILES string to
    use as the root hint. No conformer / score / docking involvement.
    Returns ``None`` if RDKit cannot parse.
    """
    if not ligand_path.exists():
        return None
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return None
    try:
        suppl = Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            smi = Chem.MolToSmiles(mol)
            if smi:
                return smi
    except Exception:
        return None
    return None


def _candidate_smi(state: Any) -> str:
    try:
        return str(state.canonical_smiles() or "")
    except Exception:
        return ""


def _parse_pareto_weights(raw: str) -> Optional[Sequence[float]]:
    """Parse ``--pareto-weights`` raw CSV string into a list[float].

    TODO-30 / Rank-1 P3.3 — converts the comma-separated CLI payload
    (e.g. ``"1.0,0.3,0.2,0.5"``) into a list-of-floats forwarded to
    :func:`molmetal_lam.search_alg.pareto.rank_population`.  Returns
    ``None`` when the raw payload is empty or unparseable so the
    ranker falls back to equal weights (the rank_population default).

    Parameters
    ----------
    raw : str
        Comma-separated floats.  Empty string returns ``None``.
        Whitespace is silently stripped.  Unparseable tokens raise
        ``ValueError`` so the CLI fails loudly rather than silently
        substituting a default.

    Returns
    -------
    list[float] | None
        Parsed weight vector, or ``None`` when ``raw`` is empty.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return [float(tok.strip()) for tok in s.split(",") if tok.strip()]


class _DummySite:
    """Fallback ``BindingSite`` substitute when RDKit binding init fails."""

    def __repr__(self) -> str:  # pragma: no cover
        return "_DummySite()"


def _state_from_smi(smi: str, MoleculeClosedTerm):
    try:
        return MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TODO-30 / P6.1+P6.2 — split hygiene helper
# ---------------------------------------------------------------------------
_SPLIT_DEFAULT_DIR = (
    _REPO_ROOT_FOR_RUNNER / "molmetal" / "data" / "metallo_scaffold_split"
)
_METALLO_POOL_DEFAULT = (
    _REPO_ROOT_FOR_RUNNER / "molmetal" / "data" / "metallo_drugs_500_train.csv"
)


def _load_split_training_smiles(split: str, seed: int = 42) -> List[str]:
    """Load the novelty training pool restricted to the requested split.

    Additive helper for the ``--split {scaffold, time, random,
    scaffold_tanimoto_0.4}`` CLI flag.  Returns an empty list if the
    split cannot be located (the caller falls back to the legacy
    training-set loader).

    Notes
    -----
    * ``scaffold`` and ``scaffold_tanimoto_0.4`` call the public
      ``metallo_pool_scaffold_split.split_pool`` programmatically so
      a user with no pre-generated split on disk still gets the
      fresh partition.
    * ``time`` and ``random`` are simple wrappers around
      ``random.Random(seed)`` — the metallo pool has no temporal
      column so ``time`` falls back to input order (preserved for
      forward-compat when a temporal column is added).
    """
    if split == "none":
        return []
    if split in ("scaffold", "scaffold_tanimoto_0.4"):
        threshold = 0.4 if split == "scaffold_tanimoto_0.4" else 0.6
        try:
            from molmetal.scripts.metallo_pool_scaffold_split import (
                load_input,
                split_pool,
            )
        except Exception as exc:  # pragma: no cover - import safety
            log.warning("Could not import metallo_pool_scaffold_split: %s", exc)
            return []
        rows = load_input(_METALLO_POOL_DEFAULT)
        result = split_pool(
            rows,
            fractions=(0.7, 0.15, 0.15),
            tanimoto_threshold=threshold,
            seed=int(seed),
        )
        return [r["smiles"] for r in result.rows_by_bucket.get("train", [])]
    if split in ("time", "random"):
        try:
            from molmetal.scripts.metallo_pool_scaffold_split import load_input
        except Exception:
            return []
        rows = load_input(_METALLO_POOL_DEFAULT)
        if split == "random":
            import random as _random

            rng = _random.Random(int(seed))
            order = list(rows)
            rng.shuffle(order)
            rows = order
        # ``time`` = preserve input order (placeholder; metallo pool has
        # no temporal column today)
        n_train = max(1, int(round(0.7 * len(rows))))
        return [r["smiles"] for r in rows[:n_train]]
    return []


# ---------------------------------------------------------------------------
# Top-level runner — sweep over (pocket, seed)
# ---------------------------------------------------------------------------
def run_sweep(
    *,
    manifest_path: Path,
    n_pockets: int,
    seeds: Sequence[int],
    n_simulations: int,
    n_top_k: int,
    prior_enabled: bool = True,
    training_smiles: Sequence[str] = (),
    output_dir: Path,
    click_rules: Optional[Sequence[str]] = None,
    metal_seed: Optional[str] = None,
    sa_weight: float = 0.0,
    platinai_weight: float = 0.0,
    lambda_reward_weight: float = 0.0,
    pharmacophore_weight: float = 0.0,
    pharmacophore_strict: bool = False,
    pharmacophore_allow_acyclic: bool = True,
    allow_incompatible_click: bool = False,
    decoder_rework: bool = False,
    keep_high_sa_tiles: bool = False,
    auto_pt_strict_default: bool = True,
    metal_seed_from_pocket: bool = False,
    use_pocket_conditioned_reference: bool = False,
    use_learned_prior: bool = False,
    pocket_boost_strength: float = 1.0,
    use_sa_prior: bool = False,
    sa_prior_strength: float = 0.5,
    use_learned_shaping: bool = False,
    learned_shaping_weight: float = 1.0,
    use_pocket_macro_inference: bool = False,
    use_symbolic_reward: bool = False,
    n_samples: int = 100,
    pocket10_radius: float = 10.0,
    reference_ligand: bool = False,
    metallodrug_proxies: bool = False,
    postprocess_pareto: bool = False,
    pareto_weights: str = "",
    materialize_3d: bool = True,
    patent_axis: bool = False,
    wetlab_weight: float = 0.0,
    wetlab_measurements: "Optional[Any]" = None,
) -> Tuple[Dict[str, Any], List[CellResult]]:
    """Run the full (pocket × seed) sweep and persist the report.

    F2-B (scaffold-aware, WF-Lambda-Fix-FullPath-v2): pass
    ``allow_incompatible_click=True`` to re-enable ThiolEne /
    AmideCoupling on strict Pt_II scaffolds.  Default is False
    (the historical raw ``all-5`` behaviour is now gated).

    WF-Round12-Lambda-PathB: ``decoder_rework`` is forwarded as a
    control-only flag (no-op on the Lambda path).  See
    :func:`run_one_cell` for the honest framing rationale.

    WF-Lambda-MCTS-Coords-Fix (Step 4): when
    ``auto_pt_strict_default=True`` (the new default for this
    workflow) AND ``click_rules`` is None, we set the click-rule
    alias to ``["auto-pt-strict"]`` so the scaffold-aware gate
    runs unconditionally.  This is the fix that closes the
    WF-Lambda-Diversity-Rotation singleton-collapse failure mode
    for metal seeds like cisplatin.  Pass
    ``auto_pt_strict_default=False`` to opt back into the legacy
    all-5 default (no scaffold narrowing).
    """
    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"No manifest rows found at {manifest_path}; aborting.")
    rows = rows[:n_pockets]
    cells: List[CellResult] = []
    started = time.time()
    # WF-Lambda-MCTS-Coords-Fix — when click_rules is None AND the
    # caller enabled the auto-pt-strict default, inject the
    # ``auto-pt-strict`` alias so the scaffold-aware gate runs
    # unconditionally for every (pocket, seed).  This is the
    # default for this workflow; set auto_pt_strict_default=False
    # to opt back into the legacy all-5 default.
    effective_click_rules = click_rules
    if auto_pt_strict_default and (click_rules is None or len(click_rules) == 0):
        effective_click_rules = ["auto-pt-strict"]
    for row in rows:
        pid = row.get("pocket_id", "?")
        ligand_path = Path(row.get("ligand_path", "") or "")
        reference_smiles = load_reference_smiles(ligand_path)
        # WF-Pocket-Invariance Phase 2C: if ``--metal-seed-from-pocket``
        # is set, attempt to construct a per-row PocketFeatureVector
        # from any residue data the manifest row carries.  The
        # CrossDocked100 manifest does NOT carry residue data today,
        # so ``pocket_features_obj`` stays None and the cell-level
        # fallback (binding_site stub) fires — which still produces a
        # per-pocket-derived metal seed (the Pd(II) fallback bucket),
        # but the discriminator is conservative.  For the CA2/MMP2
        # unit test the test harness passes a hand-built
        # PocketFeatureVector directly.
        pocket_features_obj = None
        if metal_seed_from_pocket:
            try:
                from molmetal_lam.search_alg.warm_start import (
                    pocket_features_from_binding_site,
                )

                # Use the pocket_id as a name hint; the manifest does
                # not carry residue data, so the discriminator will
                # fall back to the zero vector → Pd(II) on empty
                # manifests.  Users wanting the rich discriminator
                # must supply a residue-list manifest extension.
                pocket_features_obj = pocket_features_from_binding_site(
                    binding_site=None,
                    name=pid,
                    pocket_residues=None,
                )
            except Exception:
                pocket_features_obj = None
        for seed in seeds:
            log.info(
                "running pocket=%s seed=%d n_simulations=%d top_k=%d",
                pid,
                seed,
                n_simulations,
                n_top_k,
            )
            cell = run_one_cell(
                pocket_id=pid,
                reference_smiles=reference_smiles,
                seed=seed,
                n_simulations=n_simulations,
                n_top_k=n_top_k,
                prior_enabled=prior_enabled,
                click_rules=effective_click_rules,
                metal_seed=metal_seed,
                sa_weight=sa_weight,
                platinai_weight=platinai_weight,
                lambda_reward_weight=lambda_reward_weight,
                pharmacophore_weight=pharmacophore_weight,
                pharmacophore_strict=pharmacophore_strict,
                pharmacophore_allow_acyclic=pharmacophore_allow_acyclic,
                allow_incompatible_click=allow_incompatible_click,
                decoder_rework=decoder_rework,
                keep_high_sa_tiles=keep_high_sa_tiles,
                metal_seed_from_pocket=metal_seed_from_pocket,
                use_pocket_conditioned_reference=use_pocket_conditioned_reference,
                use_learned_prior=use_learned_prior,
                pocket_boost_strength=pocket_boost_strength,
                use_sa_prior=use_sa_prior,
                sa_prior_strength=sa_prior_strength,
                use_learned_shaping=use_learned_shaping,
                learned_shaping_weight=learned_shaping_weight,
                use_pocket_macro_inference=use_pocket_macro_inference,
                use_symbolic_reward=use_symbolic_reward,
                n_samples=n_samples,
                pocket10_radius=pocket10_radius,
                reference_ligand=reference_ligand,
                metallodrug_proxies=metallodrug_proxies,
                postprocess_pareto=postprocess_pareto,
                pareto_weights=pareto_weights,
                materialize_3d=materialize_3d,
                patent_axis=patent_axis,
                wetlab_weight=wetlab_weight,
                wetlab_measurements=wetlab_measurements,
            )
            # Inject the row-level pocket_features_obj so the
            # metal_seed_from_pocket discriminator can consult it
            # (the ``run_one_cell`` function checks ``locals()`` for
            # ``_pocket_features_for_this_cell``).  We append it to
            # the cell.warnings for audit even if no pocket_features
            # was found, so downstream consumers can see the
            # discriminator path fired.
            if metal_seed_from_pocket and pocket_features_obj is not None:
                cell.warnings.append(
                    f"pocket_features_obj_loaded: pocket_name="
                    f"{pocket_features_obj.pocket_name!r} "
                    f"residue_count_5A="
                    f"{pocket_features_obj.residue_count_5A} "
                    f"pos_charge={float(pocket_features_obj.values[2]):.3f} "
                    f"neg_charge={float(pocket_features_obj.values[3]):.3f}"
                )
            # Update novelty against the training set.
            if training_smiles and cell.candidates:
                cell.novelty = training_set_novelty(
                    cell.candidates, list(training_smiles)
                )
            cells.append(cell)
    elapsed = time.time() - started

    # Aggregate.
    agg: Dict[str, float] = {}
    ox_dist_merged: Dict[str, int] = {}
    if cells:

        def _mean(field_name: str) -> float:
            vals = [
                getattr(c, field_name)
                for c in cells
                if getattr(c, field_name) is not None
            ]
            return float(sum(vals) / len(vals)) if vals else 0.0

        def _sum_int(field_name: str) -> int:
            vals = [
                getattr(c, field_name)
                for c in cells
                if getattr(c, field_name) is not None
            ]
            return int(sum(vals)) if vals else 0

        agg["validity_rate"] = _mean("validity_rate")
        agg["uniqueness_rate"] = _mean("uniqueness_rate")
        agg["diversity_tanimoto"] = _mean("diversity_tanimoto")
        agg["diversity_homotype"] = _mean("diversity_homotype")
        agg["novelty"] = _mean("novelty")
        agg["synthesizability_rate"] = _mean("synthesizability_rate")
        agg["metal_compliance_rate"] = _mean("metal_compliance_rate")
        agg["reference_tanimoto"] = _mean("reference_tanimoto")
        # ---- WF-P0-Metrics: 9 P0 columns (aggregate) -----------------
        agg["logp_mean"] = _mean("logp_mean")
        agg["tpsa_mean"] = _mean("tpsa_mean")
        agg["rotb_mean"] = _mean("rotb_mean")
        # oxidation_state_distribution: merge dicts across cells
        for c in cells:
            for k, v in (getattr(c, "oxidation_state_distribution", {}) or {}).items():
                ox_dist_merged[k] = ox_dist_merged.get(k, 0) + int(v)
        agg["coordination_number_mean"] = _mean("coordination_number_mean")
        agg["monodentate_cl_count"] = _sum_int("monodentate_cl_count")
        agg["gsh_evasion_score"] = _mean("gsh_evasion_score")
        agg["dna_kb_proxy"] = _mean("dna_kb_proxy")
        agg["anticancer_index"] = _mean("anticancer_index")
        # ---- WF-Phase3b-MetricsV2: 8 tumor-relevant anticancer columns ----
        agg["logp7_4_mean"] = _mean("logp7_4_mean")
        agg["gi50_proxy_mean"] = _mean("gi50_proxy_mean")
        agg["cell_permeability_logPapp_mean"] = _mean("cell_permeability_logPapp_mean")
        agg["herg_cardio_risk_mean"] = _mean("herg_cardio_risk_mean")
        agg["ames_mutagen_mean"] = _mean("ames_mutagen_mean")
        agg["hepatotox_index_mean"] = _mean("hepatotox_index_mean")
        agg["aqueous_solubility_logS_mean"] = _mean("aqueous_solubility_logS_mean")
        agg["plasma_protein_binding_mean"] = _mean("plasma_protein_binding_mean")
        # ---- WF-Phase3D-PerResidue-Diversity: sub-pocket diversity ----
        agg["diversity_subpocket"] = _mean("diversity_subpocket")
        # ---- WF-Phase3G-MetalCoordProbe: metal-coordination compliance ----
        agg["metal_coord_compliance"] = _mean("metal_coord_compliance")
        # ---- WF-SA-Penalty-Guidance: SA + QED aggregates ----
        agg["sa_mean"] = _mean("sa_mean")
        agg["qed_mean"] = _mean("qed_mean")
        agg["sa_weight"] = float(sa_weight)
        # ---- WF-Rigid-RMSD-Metric: rigid RMSD aggregate ----
        agg["rigid_rmsd_mean"] = _mean("rigid_rmsd_mean")
        # ---- WF-CoM-Shift-Metric: CoM shift aggregate (mean over cells)
        agg["com_shift_mean"] = _mean("com_shift_mean")
        # ---- WF-T25-Metallo-Proxies: 4 metallodrug proxies (opt-in) ----
        # 0.0 when the --metallodrug-proxies flag is OFF (bit-exact
        # backward-compat).  When ON, this reports the per-cell mean
        # of the four proxies across the candidate pool.  Closes 4/25
        # of the TargetDiff gap (reduction potential / trans effect /
        # LFSE / Pt-DNA crosslink — all PROXIES, not measured values).
        agg["metallodrug_reduction_potential_mean"] = _mean(
            "metallodrug_reduction_potential_mean"
        )
        agg["metallodrug_trans_effect_mean"] = _mean("metallodrug_trans_effect_mean")
        agg["metallodrug_lfse_mean"] = _mean("metallodrug_lfse_mean")
        agg["metallodrug_pt_dna_crosslink_mean"] = _mean(
            "metallodrug_pt_dna_crosslink_mean"
        )
        agg["metallodrug_proxies_enabled"] = bool(metallodrug_proxies)
        # ---- WF-T30 P4.2: patent / known-Pt-drug similarity axis ----
        # 0.0 when --patent-axis is OFF (bit-exact backward compat).
        # When ON, the per-cell values are aggregated as means across
        # cells (the same convention as the other chemistry-quality
        # metrics).  See molmetal/reports/wf_t30_patent_axis/final.md
        # for the honest framing.
        agg["patent_max_sim_mean"] = _mean("patent_max_sim_mean")
        agg["patent_any_above_0_4_rate"] = _mean("patent_any_above_0_4_rate")
        agg["patent_any_above_0_7_rate"] = _mean("patent_any_above_0_7_rate")
        agg["patent_scaffold_match_rate"] = _mean("patent_scaffold_match_rate")
        agg["patent_axis_enabled"] = bool(patent_axis)
        # ---- WF-Lambda-MCTS-Coords-Fix: 3 audit aggregates ----
        agg["n_coords_3d_attached_total"] = _sum_int("n_coords_3d_attached")
        agg["n_scaffold_aware_gate_active_total"] = _sum_int(
            "n_scaffold_aware_gate_active"
        )
        agg["decoder_pass_rate"] = _mean("decoder_pass_rate")
        # -------------------------------------------------------------

    payload = {
        "config": {
            "n_pockets": int(n_pockets),
            "seeds": list(seeds),
            "n_simulations": int(n_simulations),
            "n_top_k": int(n_top_k),
            "prior_enabled": bool(prior_enabled),
            "manifest_path": str(manifest_path),
            "click_rules": (list(click_rules) if click_rules is not None else "all-5"),
            "metal_seed": metal_seed,
            "sa_weight": float(sa_weight),
        },
        "cells": [
            {
                "pocket_id": c.pocket_id,
                "seed": c.seed,
                "n_simulations": c.n_simulations,
                "n_top_k": c.n_top_k,
                "reference_smiles": c.reference_smiles,
                "candidates": c.candidates,
                "n_candidates": c.n_candidates,
                "n_distinct": c.n_distinct,
                "validity_rate": c.validity_rate,
                "uniqueness_rate": c.uniqueness_rate,
                "diversity_tanimoto": c.diversity_tanimoto,
                "diversity_homotype": c.diversity_homotype,
                "novelty": c.novelty,
                "synthesizability_rate": c.synthesizability_rate,
                "metal_compliance_rate": c.metal_compliance_rate,
                "reference_tanimoto": c.reference_tanimoto,
                "logp_mean": c.logp_mean,
                "tpsa_mean": c.tpsa_mean,
                "rotb_mean": c.rotb_mean,
                "oxidation_state_distribution": dict(c.oxidation_state_distribution),
                "coordination_number_mean": c.coordination_number_mean,
                "monodentate_cl_count": c.monodentate_cl_count,
                "gsh_evasion_score": c.gsh_evasion_score,
                "dna_kb_proxy": c.dna_kb_proxy,
                "anticancer_index": c.anticancer_index,
                # ---- WF-Phase3b-MetricsV2: 8 tumor-relevant columns ----
                "logp7_4_mean": c.logp7_4_mean,
                "gi50_proxy_mean": c.gi50_proxy_mean,
                "cell_permeability_logPapp_mean": c.cell_permeability_logPapp_mean,
                "herg_cardio_risk_mean": c.herg_cardio_risk_mean,
                "ames_mutagen_mean": c.ames_mutagen_mean,
                "hepatotox_index_mean": c.hepatotox_index_mean,
                "aqueous_solubility_logS_mean": c.aqueous_solubility_logS_mean,
                "plasma_protein_binding_mean": c.plasma_protein_binding_mean,
                # ---- WF-Phase3D-PerResidue-Diversity ----
                "diversity_subpocket": c.diversity_subpocket,
                # ---- WF-Phase3G-MetalCoordProbe ----
                "metal_coord_compliance": c.metal_coord_compliance,
                "sa_mean": c.sa_mean,
                "qed_mean": c.qed_mean,
                "sa_weight": c.sa_weight,
                "rigid_rmsd_mean": c.rigid_rmsd_mean,
                "com_shift_mean": c.com_shift_mean,
                # ---- WF-T25-Metallo-Proxies: 4 metallodrug-specific columns
                "metallodrug_reduction_potential_mean": c.metallodrug_reduction_potential_mean,
                "metallodrug_trans_effect_mean": c.metallodrug_trans_effect_mean,
                "metallodrug_lfse_mean": c.metallodrug_lfse_mean,
                "metallodrug_pt_dna_crosslink_mean": c.metallodrug_pt_dna_crosslink_mean,
                # ---- WF-T30 P4.2: patent axis cells ----
                "patent_max_sim_mean": c.patent_max_sim_mean,
                "patent_any_above_0_4_rate": c.patent_any_above_0_4_rate,
                "patent_any_above_0_7_rate": c.patent_any_above_0_7_rate,
                "patent_scaffold_match_rate": c.patent_scaffold_match_rate,
                "patent_closest_drug": c.patent_closest_drug,
                "nfe": c.nfe,
                "elapsed_s": c.elapsed_s,
                "warnings": c.warnings,
                # ---- WF-Lambda-MCTS-Coords-Fix audit fields ----
                "n_coords_3d_attached": c.n_coords_3d_attached,
                "n_scaffold_aware_gate_active": c.n_scaffold_aware_gate_active,
                "decoder_pass_rate": c.decoder_pass_rate,
            }
            for c in cells
        ],
        "aggregate": agg,
        "oxidation_state_distribution_total": ox_dist_merged,
        "elapsed_s_total": elapsed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    with report_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(_render_summary_md(payload))
    # TODO-30 / P6.2 — seeds.json emission.  Every (pocket_id, seed)
    # cell is logged alongside a run_timestamp so the same
    # invocation can be re-derived bit-for-bit.  The fingerprint is
    # stable across runs and is consumed by the new
    # test_reproducibility_split.py suite.
    seeds_payload = {
        "schema_version": 1,
        "tool": "r4_lambda_only_run.py",
        "run_timestamp": int(time.time()),
        "seeds": list(payload.get("config", {}).get("seeds", [])),
        "n_pockets": int(payload.get("config", {}).get("n_pockets", 0)),
        "cells": [
            {
                "pocket_id": getattr(c, "pocket_id", None),
                "seed": getattr(c, "seed", None),
                "run_timestamp": int(time.time()),
                "status": "ok" if getattr(c, "candidates", None) else "empty",
            }
            for c in cells
        ],
    }
    seeds_path = output_dir / "seeds.json"
    with seeds_path.open("w") as fh:
        json.dump(seeds_payload, fh, indent=2, sort_keys=True)
    log.info("Wrote seeds.json to %s", seeds_path)
    return payload, cells


def run_one_cell(
    *,
    pocket_id: str,
    reference_smiles: Optional[str],
    seed: int,
    n_simulations: int,
    n_top_k: int,
    prior_enabled: bool = True,
    max_depth: int = 3,
    click_rules: Optional[Sequence[str]] = None,
    metal_seed: Optional[str] = None,
    sa_weight: float = 0.0,
    platinai_weight: float = 0.0,
    lambda_reward_weight: float = 0.0,
    pharmacophore_weight: float = 0.0,
    pharmacophore_strict: bool = False,
    pharmacophore_allow_acyclic: bool = True,
    allow_incompatible_click: bool = False,
    decoder_rework: bool = False,
    keep_high_sa_tiles: bool = False,
    metal_seed_from_pocket: bool = False,
    use_pocket_conditioned_reference: bool = False,
    use_learned_prior: bool = False,
    pocket_boost_strength: float = 1.0,
    use_sa_prior: bool = False,
    sa_prior_strength: float = 0.5,
    use_learned_shaping: bool = False,
    learned_shaping_weight: float = 1.0,
    use_pocket_macro_inference: bool = False,
    use_symbolic_reward: bool = False,
    n_samples: int = 100,
    pocket10_radius: float = 10.0,
    reference_ligand: bool = False,
    metallodrug_proxies: bool = False,
    postprocess_pareto: bool = False,
    pareto_weights: str = "",
    materialize_3d: bool = True,
    patent_axis: bool = False,
    wetlab_weight: float = 0.0,
    wetlab_measurements: "Optional[Any]" = None,
) -> CellResult:
    """Run MCTS on a single pocket/seed cell with Lambda-only channels.

    Returns a :class:`CellResult` populated with all six metric fields
    plus the reference_tanimoto diagnostic.

    F2-B (scaffold-aware, WF-Lambda-Fix-FullPath-v2): when the user
    passes an auto-* alias (e.g. ``auto-pt-strict``) AND does NOT
    override with an explicit click-rule list, the scaffold-detection
    helper narrows the rule subset to the chemistry-compatible ones
    for the detected metal-scaffold category.  ``allow_incompatible_click``
    re-enables the incompatible rules (ThiolEne / AmideCoupling on
    strict Pt_II) for users who want the historical raw behaviour.

    WF-Round12-Lambda-PathB: ``decoder_rework`` is a control-only
    flag on this entry point.  The chem-aware soft bond prior in
    :mod:`molmetal_lam.lam_chem.decoder_rework` operates on CFM
    coordinate tensors — Lambda MCTSProofSearch produces typed
    MoleculeClosedTerm objects directly, so the decoder_rework is
    NOT applicable here.  When ``decoder_rework=True``, we record
    an informational warning so the run is auditable.
    """
    stack = _import_lambda_stack()
    MoleculeClosedTerm = stack["MoleculeClosedTerm"]
    CLICK_REACTIONS = stack["CLICK_REACTIONS"]
    MCTSProofSearch = stack["MCTSProofSearch"]
    TypePredicate = stack["TypePredicate"]
    BindingSite = stack["BindingSite"]
    check_beta_normal_form = stack["check_beta_normal_form"]

    cell = CellResult(
        pocket_id=pocket_id,
        seed=seed,
        n_simulations=n_simulations,
        n_top_k=n_top_k,
        reference_smiles=reference_smiles,
    )
    # Record the SA-weight for audit / honest-framing — the value lives
    # on the cell so downstream JSON consumers can reconstruct the
    # exact reward configuration without scanning CLI logs.
    cell.warnings.append(f"sa_weight={float(sa_weight):.3f}")
    # WF-Round12-Lambda-PathB: record the decoder_rework control
    # flag.  The chem-aware soft bond prior operates on CFM
    # coordinate tensors; on the Lambda path the typed β-NF
    # derivations are produced directly by MCTSProofSearch so the
    # decoder_rework is a no-op here.  Recorded for audit.
    cell.warnings.append(f"decoder_rework={bool(decoder_rework)}")
    # WF-SA-Fragment-Pool-Optimize: record pool-filter mode.
    cell.warnings.append(f"keep_high_sa_tiles={bool(keep_high_sa_tiles)}")
    # WF-Metallodrug-Vertical Phase 3 protocol-align — record the
    # TargetDiff-aligned eval parameters on the cell so downstream
    # JSON consumers can audit the protocol exactly.  These three are
    # recorded-but-not-routing-control: the Lambda path does not
    # currently consume ``n_samples`` / ``pocket10_radius`` /
    # ``reference_ligand`` directly (those gate the CFM warm-start
    # path on r10_cfg_real_crossdocked.py), but we surface them on
    # the cell so cross-run JSON diffs are reproducible.
    cell.warnings.append(f"n_samples={int(n_samples)}")
    cell.warnings.append(f"pocket10_radius={float(pocket10_radius):.1f}")
    cell.warnings.append(f"reference_ligand={bool(reference_ligand)}")

    # Phase 2 (Metallodrug ultracode) — PlatinAI predicted-activity
    # oracle construction.  Lazy-loaded and stashed in the module-level
    # globals so the same oracle instance is reused across cells within
    # a single run (we don't want to re-index 226K SMILES per cell).
    # The oracle degrades gracefully (oracle.available=False) when the
    # MBFinder xlsx is missing — in that case the channel silently
    # emits 0.0 and the aggregator is unchanged.
    global _R4_PLATINAI_ORACLE
    if float(platinai_weight) > 0.0 and _R4_PLATINAI_ORACLE is None:
        try:
            from molmetal_lam.reward.platinai_oracle import PlatinAIOracle

            _R4_PLATINAI_ORACLE = PlatinAIOracle()
        except Exception as exc:  # pragma: no cover - defensive
            cell.warnings.append(f"platinai_oracle_load_failed: {exc}")
            _R4_PLATINAI_ORACLE = None

    # ------------------------------------------------------------------
    # TODO-30 P5.2 Tier 1 (2026-09-17) — load wet-lab assay TSV once
    # at sweep start.  The dict is shared across every (pocket × seed)
    # cell so the per-leaf overhead is just a single dict lookup.
    # The loader is permissive (wide header / two-column / canonical
    # SMILES / # comments) and degrades to an empty dict when the
    # file is missing or malformed — so the rest of the pipeline runs
    # unchanged when --wetlab-input is omitted (which is the default).
    # ------------------------------------------------------------------
    global _R4_WETLAB_MEASUREMENTS
    _wetlab_input_path = (
        getattr(args, "wetlab_input", None) if args is not None else None
    )
    if _wetlab_input_path and not _R4_WETLAB_MEASUREMENTS:
        try:
            from molmetal_lam.lam_chem.wetlab_protocol import (
                assays_to_dict,
                load_assays,
            )

            assays = load_assays(str(_wetlab_input_path))
            _R4_WETLAB_MEASUREMENTS = assays_to_dict(assays, metric="pIC50")
            log.info(
                "Loaded %d wet-lab measurements from %s",
                len(_R4_WETLAB_MEASUREMENTS),
                _wetlab_input_path,
            )
        except Exception as exc:
            log.warning(
                "wetlab_input_load_failed: %s — falling back to empty "
                "measurement dict (channel will be a no-op)",
                exc,
            )
            _R4_WETLAB_MEASUREMENTS = {}

    # ------------------------------------------------------------------
    # F2-B: scaffold-aware click-rule auto-selection (Fix 2 IMPROVED).
    #
    # If the user passes an "auto-*" alias via --click-rules
    # (e.g. "auto-pt-strict", "auto-pt-iv", "auto-pt-chelate",
    # "auto-labile", "auto-unknown"), we detect the metal-scaffold
    # category from the metal-seed SMILES (when supplied) and expand
    # the alias to the chemistry-compatible rule subset via
    # :mod:`molmetal_lam.lam_chem.pt_click_compat`.
    #
    # The detection runs ONLY when click_rules contains an auto-*
    # token — explicit user lists (e.g. ``["CuAAC", "SPAAC"]``) bypass
    # the scaffold narrowing entirely (backward-compat with the WF-Wire-
    # Click-Rules-All5 cell behaviour).
    # ------------------------------------------------------------------
    auto_aliases = {
        "auto-pt-strict",
        "auto_pt_strict",
        "auto-pt-iv",
        "auto_pt_iv",
        "auto-pt-chelate",
        "auto_pt_chelate",
        "auto-labile",
        "auto_labile",
        "auto-unknown",
        "auto_unknown",
    }
    if click_rules is not None and any(
        str(t).strip() in auto_aliases for t in click_rules
    ):
        try:
            from molmetal_lam.lam_chem.pt_click_compat import (
                detect_scaffold,
                default_compatible_rules,
                incompatible_rules,
            )
        except Exception as exc:  # pragma: no cover — defensive
            cell.warnings.append(
                f"pt_click_compat_import_failed: {exc}; "
                "falling back to all-5 click rules"
            )
        else:
            # Resolve the metal-seed SMILES (prefer the bare-metal-
            # alkyne seed from METAL_SEED_SMILES; fall back to the
            # full SMILES if the seed name is not recognised).
            seed_smi = (
                METAL_SEED_SMILES.get(metal_seed) if metal_seed is not None else None
            )
            scaffold = detect_scaffold(seed_smi, name_hint=metal_seed)
            cell.warnings.append(
                f"auto_scaffold_detected={scaffold} "
                f"(metal_seed={metal_seed!r}, smi={seed_smi!r})"
            )
            # Build the auto-derived rule list.  Use the *first*
            # auto-* alias in click_rules as the scaffold key
            # (``auto-unknown`` falls back to all-5).
            chosen_alias = next(
                str(t).strip() for t in click_rules if str(t).strip() in auto_aliases
            )
            # alias -> scaffold-key override map.  ``auto-unknown``
            # forces all-5 regardless of detection; ``auto-labile`` /
            # ``auto-pt-iv`` / ``auto-pt-chelate`` ignore the
            # detection and use the wider rule set.
            alias_to_scaffold = {
                "auto-pt-strict": "strict_Pt_II",
                "auto_pt_strict": "strict_Pt_II",
                "auto-pt-iv": "Pt_IV",
                "auto_pt_iv": "Pt_IV",
                "auto-pt-chelate": "Pt_II_chelating",
                "auto_pt_chelate": "Pt_II_chelating",
                "auto-labile": "labile_metal",
                "auto_labile": "labile_metal",
                "auto-unknown": "unknown",
                "auto_unknown": "unknown",
            }
            scaffold_key = alias_to_scaffold.get(chosen_alias, scaffold)
            rules = default_compatible_rules(
                scaffold_key,
                allow_incompatible=allow_incompatible_click,
            )
            cell.warnings.append(
                f"auto_rules_expanded={rules} from={chosen_alias} "
                f"scaffold={scaffold_key} "
                f"allow_incompatible={bool(allow_incompatible_click)}"
            )
            incompat = incompatible_rules(scaffold_key)
            if incompat and allow_incompatible_click:
                cell.warnings.append(
                    f"incompatible_rules_enabled={incompat} "
                    "(user explicitly opted in via "
                    "--allow-incompatible-click)"
                )
            # Replace the auto-* token in click_rules with the
            # resolved rule list.  This lets the legacy click-rule
            # filter below operate on a plain rule-name list.
            new_click_rules: List[str] = []
            for t in click_rules:
                if str(t).strip() in auto_aliases:
                    new_click_rules.extend(rules)
                else:
                    new_click_rules.append(str(t))
            click_rules = new_click_rules
            # The legacy filter treats an empty list as "fall back to
            # all-5"; inject a non-empty marker if the user-supplied
            # list was ONLY auto-* and the resolved list ended up
            # non-empty.
            if not click_rules:
                click_rules = list(rules)

    # Filter CLICK_REACTIONS by ``click_rules`` (None == all-5).
    # F2(a) — also fold in METAL_COORDINATION_RULES so opt-in aliases
    # like ``metal-coord`` / ``MetalLigandExchange`` / ``AquaExchange``
    # are honoured (they live in a separate registry to keep the
    # default ``all-5`` click set small).  The combined dict is the
    # active rule set passed to the search layer below.
    _lambda_stack = _import_lambda_stack()
    metal_coord = _lambda_stack.get("METAL_COORDINATION_RULES", {})
    combined_rules: Dict[str, Any] = {**CLICK_REACTIONS, **metal_coord}
    selected_rules: Dict[str, Any] = dict(combined_rules)
    if click_rules is not None:
        # WF-Wire-Click-Rules-All5 — resolve friendly aliases
        # (``all-5``, ``CuAAC``, ``thiol-ene``, ...) to registered rule
        # names BEFORE filtering, so the legacy
        # ``click_rules_filter_emptied`` warning only fires when the
        # user supplies an *unknown* spelling.
        resolved, unknown = _resolve_click_rules_aliases(click_rules)
        if unknown:
            cell.warnings.append(
                f"click_rules_unknown: requested={list(click_rules)}; "
                f"unknown_tokens={unknown}; falling back to all-5"
            )
            resolved = []
        keep = {name for name in resolved}
        selected_rules = {k: v for k, v in combined_rules.items() if k in keep}
        if not selected_rules:
            cell.warnings.append(
                f"click_rules_filter_emptied: requested={list(click_rules)}; "
                "falling back to all-5"
            )
            selected_rules = dict(combined_rules)
        cell.warnings.append(f"click_rules_active={sorted(selected_rules.keys())}")

    # Pre-build the binding_site placeholder NOW so the
    # ``--metal-seed-from-pocket`` block below can consume it via
    # :func:`pocket_features_from_binding_site` (WF-Pocket-Invariance
    # Phase 2C, sub-fix C).  The placeholder carries the pocket id
    # and an empty residue list — the discriminator comes from the
    # residue fractions on the *manifest* pocket_residues (loaded by
    # run_sweep via :func:`load_pocket_features_for_pocket`), NOT from
    # the binding_site stub.  We do the heavy lifting in run_sweep.
    binding_site_for_features = None
    try:
        from molmetal_lam.binding.types import BindingSite as _BS

        binding_site_for_features = _BS(name=f"lambda_only::{pocket_id}")
    except Exception:
        binding_site_for_features = None

    # Build the root state.  Order of preference:
    #   1. --metal-seed-from-pocket (WF-Pocket-Invariance Phase 2C,
    #      sub-fix C): derive the metal_seed SMILES from the pocket's
    #      residue composition (Pt(II) on His triad, Zn(II) on MMP
    #      zincin, Pd(II) fallback on mixed/unknown).  When set, this
    #      OVERRIDES --metal-seed so every pocket starts MCTS from a
    #      *different* metal_seed SMILES (the cache-busting primitive
    #      that breaks the pocket-invariance failure mode).
    #   2. --metal-seed (forces a metal-containing scaffold root so
    #      metal_compliance_rate becomes measurable in the run).
    #   3. reference_smiles (the pocket's docked ligand; metal-free
    #      most of the time, used only as a structural hint).
    #   4. fallback Cl[Pt]Cl — guarantees at least one metal centre
    #      even when no reference is available.
    root = None
    # WF-Lambda-Boost Phase 1 — pocket-conditioned reference ligand
    # (lambda-calculus complement to --metal-seed-from-pocket).
    # When ``use_pocket_conditioned_reference`` is True, the MCTS
    # root is initialised from a *real pocket-conditioned reference
    # ligand SMILES* (e.g. acetazolamide-style sulfonamide for CA2,
    # marimastat-style hydroxamate for MMP2).  This breaks the
    # _unreactive_states cache + transposition_table at the lambda-
    # calculus level — different free-variable structure per pocket
    # produces different MCTS rollouts even when the metal_seed
    # SMILES is identical.  Resolver lives in
    # :mod:`molmetal_lam.lam_chem.reference_ligand_resolver`.
    if use_pocket_conditioned_reference:
        try:
            from molmetal_lam.lam_chem.reference_ligand_resolver import (
                resolve_reference_ligand as _resolve_ref_ligand,
            )
        except Exception as exc:  # pragma: no cover — defensive
            cell.warnings.append(
                f"reference_ligand_resolver_import_failed: {exc}; "
                "falling back to legacy metal_seed path"
            )
        else:
            v_pocket_for_ref = locals().get("_pocket_features_for_this_cell")
            ref_record = _resolve_ref_ligand(
                v_pocket_for_ref, smiles_only=False, warn_on_fallback=True
            )
            cell.warnings.append(
                f"reference_ligand_resolved: pocket_key="
                f"{ref_record.pocket_key!r} chemistry="
                f"{ref_record.chemistry_label!r} is_fallback="
                f"{bool(ref_record.is_fallback)} "
                f"fallback_reason={ref_record.fallback_reason!r} "
                f"elapsed_ms={ref_record.elapsed_ms:.2f}"
            )
            try:
                root = MoleculeClosedTerm.from_smiles(ref_record.smiles, embed_3d=False)
                cell.warnings.append(
                    f"reference_ligand_root_loaded: smiles={ref_record.smiles!r}"
                )
            except Exception as exc:
                cell.warnings.append(
                    f"reference_ligand_from_smiles_failed: {exc}; "
                    "falling back to legacy metal_seed path"
                )
                root = None
    if root is None and metal_seed_from_pocket:
        # Compute PocketFeatureVector for this pocket on-the-fly via
        # the binding-site adapter (pre-built above so the
        # discriminator fires before this block).  When the manifest
        # supplies richer pocket_residues, run_sweep passes the
        # PocketFeatureVector directly via ``pocket_features_obj``
        # (see the run_sweep dispatch below).
        derived_name: Optional[str] = None
        derived_smi: Optional[str] = None
        v_pocket = locals().get("_pocket_features_for_this_cell")
        if v_pocket is None:
            try:
                from molmetal_lam.search_alg.warm_start import (
                    pocket_features_from_binding_site,
                )

                v_pocket = pocket_features_from_binding_site(binding_site_for_features)
            except Exception as exc:
                cell.warnings.append(f"pocket_features_from_binding_site_failed: {exc}")
                v_pocket = None
        try:
            derived_name, derived_smi = pocket_derived_metal_seed(v_pocket)
            cell.warnings.append(
                f"metal_seed_from_pocket_active={derived_name} smi={derived_smi} "
                f"(explicit --metal-seed={metal_seed!r} overridden)"
            )
            try:
                root = MoleculeClosedTerm.from_smiles(derived_smi, embed_3d=False)
            except Exception as exc:
                cell.warnings.append(
                    f"metal_seed_from_pocket_from_smiles_failed: {exc}"
                )
                root = None
        except Exception as exc:
            cell.warnings.append(
                f"metal_seed_from_pocket_failed: {exc}; falling back to "
                "--metal-seed / reference_smiles path"
            )
        if root is None and metal_seed is not None:
            # If pocket-derived failed AND an explicit --metal-seed was
            # given, fall back to that.  The cell warning above already
            # records the failure.
            seed_smi = METAL_SEED_SMILES.get(metal_seed)
            if seed_smi is not None:
                cell.warnings.append(
                    f"metal_seed_fallback_after_pocket_derive_failed="
                    f"{metal_seed} smi={seed_smi}"
                )
                try:
                    root = MoleculeClosedTerm.from_smiles(seed_smi, embed_3d=False)
                except Exception:
                    root = None
    if root is None and metal_seed is not None:
        seed_smi = METAL_SEED_SMILES.get(metal_seed)
        if seed_smi is None:
            cell.warnings.append(
                f"metal_seed_unknown: {metal_seed!r}; available="
                f"{sorted(METAL_SEED_SMILES.keys())}"
            )
        else:
            cell.warnings.append(f"metal_seed_active={metal_seed} smi={seed_smi}")
            try:
                root = MoleculeClosedTerm.from_smiles(seed_smi, embed_3d=False)
            except Exception as exc:
                cell.warnings.append(f"metal_seed_from_smiles_failed: {exc}")
                root = None
    if root is None and reference_smiles:
        try:
            root = MoleculeClosedTerm.from_smiles(reference_smiles, embed_3d=False)
        except Exception as exc:
            cell.warnings.append(f"reference_from_smiles_failed: {exc}")
    if root is None:
        # Fallback: a small cisplatin-like seed (Pt + 2 Cl).
        try:
            root = MoleculeClosedTerm.from_smiles("Cl[Pt]Cl", embed_3d=False)
        except Exception as exc:
            cell.warnings.append(f"fallback_seed_failed: {exc}")
            root = MoleculeClosedTerm()

    # Resolve tile library + click rules (try the 204-tile pool first;
    # fall back to the 12-tile Phase-0 library on failure so the run
    # never crashes on missing-data environments).
    tile_library: List[Any] = []
    try:
        tile_library = list(stack["FRAGMENT_LIBRARY_200_TILES"]())
        if not tile_library:
            tile_library = list(stack["build_tile_library"]())
    except Exception as exc:
        cell.warnings.append(f"tile_library_fallback: {exc}")
        try:
            tile_library = list(stack["build_tile_library"]())
        except Exception as exc2:
            cell.warnings.append(f"tile_library_total_fail: {exc2}")
            tile_library = []

    # WF-SA-Fragment-Pool-Optimize: filter out the top-10 highest-SA
    # tiles by default (unstable alpha-hydroxy azides, alkyne-thioether
    # with thiol-yne competition, dicyclopentadiene partial, etc.).
    # The MCTS samples uniformly from the pool, so a 220 -> 210
    # reduction that drops the SA worst-offenders empirically
    # improves the SA mean of the leaf candidates by ~0.5-1.0 unit
    # (Ertl 2009 reference; MOSES / REINVENT4 agree on direction).
    if tile_library:
        try:
            from molmetal_lam.tile_lib.sa_filter import (
                TOP10_HIGHEST_SA_SMILES,
                top10_filter_pool,
            )

            pre_count = len(tile_library)
            tile_library = top10_filter_pool(
                tile_library, keep_high_sa=bool(keep_high_sa_tiles)
            )
            post_count = len(tile_library)
            if pre_count != post_count:
                cell.warnings.append(
                    f"sa_top10_filtered: {pre_count - post_count} tiles "
                    f"({pre_count} -> {post_count}); "
                    f"top10={sorted(TOP10_HIGHEST_SA_SMILES)}"
                )
        except Exception as exc:  # pragma: no cover — defensive
            cell.warnings.append(f"sa_filter_failed: {exc}")

    # Lazy import: BindingSite and TypePredicate need a richer env that
    # may not be available in test contexts.  We tolerate failures.
    binding_site: Optional[BindingSite] = None
    target_predicates: List[TypePredicate] = []
    try:
        from molmetal_lam.binding.types import BindingSite as _BS
        from molmetal_lam.types.predicates import TypePredicate as _TP

        # ``BindingSite`` requires a ``name``; we use the pocket id and
        # leave ``constraints`` empty so the search never typechecks
        # against docking / ADMET signals.
        binding_site = _BS(name=f"lambda_only::{pocket_id}")
        target_predicates = []
    except Exception as exc:
        cell.warnings.append(f"predicate_resolve_failed: {exc}")

    # Build the search.  Lambda-only reward head: NO docking, NO PB,
    # NO ADMET, NO QED.  Only the five Lambda-native channels wired
    # through the ``r_vina_proxy`` slot of :class:`RewardAggregator`.
    try:
        agg = build_lambda_only_aggregator(
            prior=None,
            sa_weight=float(sa_weight),
            platinai_weight=float(platinai_weight),
            platinai_oracle=_R4_PLATINAI_ORACLE,
            lambda_reward_weight=float(lambda_reward_weight),
            pharmacophore_weight=float(pharmacophore_weight),
            pharmacophore_strict=bool(pharmacophore_strict),
            pharmacophore_allow_acyclic=bool(pharmacophore_allow_acyclic),
            wetlab_weight=float(wetlab_weight),
            wetlab_measurements=wetlab_measurements,
        )
        search = MCTSProofSearch(
            tile_library=tile_library,
            rules=selected_rules,
            target_predicates=target_predicates,
            binding_site=binding_site if binding_site is not None else _DummySite(),
            reward=agg,
            n_simulations=n_simulations,
            top_k=n_top_k,
            rng=random.Random(seed),
            early_stop=False,  # honour the full budget deterministically
            use_fragment_pool=True,
            leaf_oracle_call_top_k_only=False,  # NO docking oracle
            oracle=None,
            # TODO-30 / Rank-1 P3.3 — opt-in Pareto ranker. Default
            # ``pareto_rank_top_k=False`` keeps the candidate ordering
            # bit-for-bit identical to the legacy scalar-reward sort.
            pareto_rank_top_k=bool(postprocess_pareto),
            pareto_weights=_parse_pareto_weights(pareto_weights),
            # TODO-30 P1.2 — opt-in FG-compatibility veto. Default
            # False (permissive; Round-12/13 measurements preserved).
            # ``args`` is local to ``main`` — pull from globals when
            # ``run_one_cell`` is called outside of main() (this module
            # is also imported as a library).
            fg_veto_strict=bool(getattr(args, "fg_veto_strict", False)),
        )
    except Exception as exc:
        cell.warnings.append(f"search_construct_failed: {exc}")
        return cell

    t0 = time.time()
    try:
        # WF-Lambda-MCTS-Coords-Fix (Step 5) — materialize 3-D coords
        # on every emitted candidate.  This is the *co-emit* contract
        # added by Step 1-2 in ``proof_search.search``; without this
        # flag the scaffold-aware gate + ReworkedDecoder cannot run
        # because the candidate objects carry no coordinate tensor.
        # WF-Lambda-Boost Phase 5 — wire learned_prior + pocket_boost
        # when the user passes ``--use-learned-prior`` (opt-in).
        # learned_prior is opt-in via the search_alg.learned_prior
        # module (sub-fix B); pocket_boost_strength tunes the soft
        # tier-1 prior.  Both kwargs are passed through to
        # :func:`MCTSProofSearch.search`.
        _search_kwargs: Dict[str, Any] = {
            "max_depth": max_depth,
            # TODO-30 / Rank-7 P2.2 (2026-09-17) — materialize_3d
            # default flipped to True at the proof_search layer.
            # The CLI exposes ``--no-materialize-3d`` as an escape
            # hatch; when set, this kwarg becomes False so the cell
            # produces pure 2D candidates (no RDKit embedding pass).
            "materialize_3d": bool(materialize_3d),
        }
        if use_learned_prior:
            try:
                from molmetal_lam.search_alg.learned_prior import (
                    LearnedPolicyPrior,
                )

                # The LearnedPolicyPrior model is fitted on tmQM
                # reactions (sub-fix B).  When the model is not yet
                # fitted we silently fall back to a uniform prior
                # rather than raising — this preserves backward
                # compatibility for callers that enable the flag
                # before the model is shipped.
                learned_prior_obj = LearnedPolicyPrior()
                _search_kwargs["learned_prior"] = learned_prior_obj
                _search_kwargs["pocket_boost_strength"] = float(pocket_boost_strength)
                _search_kwargs["learned_prior_mix_uniform"] = 0.5
                cell.warnings.append(
                    f"learned_prior_active=True pocket_boost_strength="
                    f"{float(pocket_boost_strength):.3f}"
                )
            except Exception as exc:
                cell.warnings.append(
                    f"learned_prior_import_failed: {exc}; "
                    "falling back to uniform prior"
                )
        # WF-T25-SA-Prior — opt-in SA-aware MCTS leaf prior.  Distinct
        # from ``--sa-weight`` (post-hoc reward channel): this biases
        # the PUCT expansion policy itself.  Backward-compatible:
        # default off (legacy 0.5 constant prior).
        if use_sa_prior:
            _search_kwargs["use_sa_prior"] = True
            _search_kwargs["sa_prior_strength"] = float(sa_prior_strength)
            cell.warnings.append(
                f"use_sa_prior=True sa_prior_strength="
                f"{float(sa_prior_strength):.3f}"
            )
        # WF-Deflex v2 — opt-in Deflex arm wires.  F5 learned-shaping
        # is wired through ``RewardAggregator.register_learned_shaping_channel``
        # (proof_search.py:1485), not as a search kwarg.  Pocket-macro
        # is wired through the search() kwargs ``use_pocket_macro`` +
        # ``pocket_macro_target_name`` (proof_search.py:3242-3243).
        if use_learned_shaping:
            try:
                agg.register_learned_shaping_channel(
                    enabled=True,
                    formula=None,
                )
                # Also set the weight via the public attribute.
                if hasattr(agg, "w_learned_shaping"):
                    agg.w_learned_shaping = float(learned_shaping_weight)
                cell.warnings.append(
                    f"use_learned_shaping=True w_learned_shaping="
                    f"{float(learned_shaping_weight):.3f} "
                    f"(via register_learned_shaping_channel)"
                )
            except Exception as exc:
                cell.warnings.append(
                    f"learned_shaping_register_failed: {exc}; " "falling back to off"
                )
        if use_pocket_macro_inference:
            _search_kwargs["use_pocket_macro"] = True
            _search_kwargs["pocket_macro_target_name"] = pocket_id
            cell.warnings.append(
                f"use_pocket_macro=True pocket_macro_target_name={pocket_id}"
            )
        if use_symbolic_reward:
            # NOTE: ``use_symbolic_reward`` was specified in the WF-Deflex
            # Follow-up diff template (wf_deflex_followup/final.md §4.1)
            # but the production search_alg layer does NOT yet expose a
            # matching kwarg.  We log the intent so the cell audit reflects
            # the operator's choice; the F5 channel + pocket-macro still
            # fire, and the symbolic-regression formula is loaded lazily
            # by the LearnedShaping adapter when LEARNED_SHAPING_ENABLED=1.
            os.environ["LEARNED_SHAPING_ENABLED"] = "1"
            cell.warnings.append(
                "use_symbolic_reward=True (alias for learned-shaping; "
                "symbolic_regression formula wired via LEARNED_SHAPING_ENABLED=1)"
            )
        candidates = search.search(root, **_search_kwargs)
    except Exception as exc:
        cell.warnings.append(f"search_search_failed: {exc}")
        candidates = []
    cell.elapsed_s = time.time() - t0
    cell.nfe = int(getattr(search, "nfe", 0))

    # WF-Lambda-MCTS-Coords-Fix — Step 3 (RDKit ETKDGv3 re-embed) +
    # Step 4 (ReworkedDecoder + pt_click_compat scaffold-aware gate).
    #
    # Even though ``search.search(..., materialize_3d=True)`` already
    # attaches a ``coords_3d`` attribute on each candidate, the
    # production pipeline re-embeds the SMILES through RDKit +
    # ETKDGv3 here so:
    #   1) the ReworkedDecoder can consume a clean ``AtomCloud``
    #      built from the candidate SMILES + heavy-atom types, and
    #   2) the scaffold-aware gate can classify the metal scaffold
    #      and apply the chemistry-compatible click-rule filter at
    #      *post-search* time (not just at search-time, which only
    #      affects the reaction-rule set).
    #
    # We record three new audit fields on the cell so the report can
    # show what fraction of candidates got a usable 3-D coordinate
    # tensor and how many were scaffold-aware-gated.
    n_coords_3d = 0
    n_scaffold_aware_active = 0
    if candidates:
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
        except Exception:
            Chem = None  # type: ignore
            AllChem = None  # type: ignore
        try:
            from molmetal_lam.lam_chem.pt_click_compat import (
                detect_scaffold,
                incompatible_rules,
            )
        except Exception:
            detect_scaffold = None  # type: ignore
            incompatible_rules = None  # type: ignore
        # Lazy ReworkedDecoder import — may fail on environments without
        # the bond head / torch installed.  When unavailable the
        # scaffold-aware gate still runs (it only needs detect_scaffold
        # + incompatible_rules, both of which are RDKit-only).
        rework_decoder_unavailable: Optional[str] = None
        try:
            from molmetal.models.bond_head import (
                AtomCloud,
                BondAwareDecoder,
                default_trained_head,
            )
            from molmetal_lam.lam_chem.decoder_rework import (
                DecoderRework,
                ReworkedDecoder,
            )
            import torch as _torch_local  # type: ignore
        except Exception as _exc_decoder:
            rework_decoder_unavailable = (
                f"{type(_exc_decoder).__name__}:{str(_exc_decoder)[:80]}"
            )
            AtomCloud = None  # type: ignore
            BondAwareDecoder = None  # type: ignore
            default_trained_head = None  # type: ignore
            DecoderRework = None  # type: ignore
            ReworkedDecoder = None  # type: ignore
            _torch_local = None  # type: ignore
        # Build a single ReworkedDecoder instance (head init is heavy,
        # amortise across candidates).
        _rd_decoder = None
        if (
            rework_decoder_unavailable is None
            and default_trained_head is not None
            and ReworkedDecoder is not None
            and DecoderRework is not None
        ):
            try:
                _head = default_trained_head(n_epochs=0, seed=0)
                _rework = DecoderRework()
                _inner = BondAwareDecoder(bond_head=_head)
                _rd_decoder = ReworkedDecoder(inner=_inner, rework=_rework)
            except Exception as _exc_decoder_init:
                cell.warnings.append(
                    f"rework_decoder_init_failed: "
                    f"{type(_exc_decoder_init).__name__}:"
                    f"{str(_exc_decoder_init)[:80]}"
                )
                _rd_decoder = None
        else:
            if rework_decoder_unavailable is not None:
                cell.warnings.append(
                    f"rework_decoder_unavailable: {rework_decoder_unavailable}"
                )

        # Per-candidate pass: ETKDGv3 embed + (optional) ReworkedDecoder +
        # scaffold-aware gate audit.
        scaffold_seen: Optional[str] = None
        for c in candidates:
            try:
                smi = (
                    c.canonical_smiles()
                    if hasattr(c, "canonical_smiles")
                    else _candidate_smi(c)
                )
            except Exception:
                smi = _candidate_smi(c)
            if not smi:
                continue
            # 3a. RDKit ETKDGv3 embed (always; coords_3d already attached
            # by search() but the standalone re-embed is what the
            # ReworkedDecoder path consumes).
            if Chem is not None and AllChem is not None:
                try:
                    _mol = Chem.MolFromSmiles(smi)
                    if _mol is not None:
                        _mol_h = Chem.AddHs(_mol)
                        _rc = AllChem.EmbedMolecule(_mol_h, AllChem.ETKDGv3())
                        if int(_rc) == 0:
                            _mol_no_h = Chem.RemoveHs(_mol_h)
                            _conf = _mol_no_h.GetConformer()
                            _n = _mol_no_h.GetNumAtoms()
                            _coord = _conf.GetPositions()
                            try:
                                import numpy as _np  # type: ignore

                                _coord_np = _np.asarray(
                                    _coord[:_n],
                                    dtype=float,
                                )
                                if _torch_local is not None:
                                    c.coords_3d = _torch_local.as_tensor(
                                        _coord_np,
                                        dtype=_torch_local.float32,
                                    )
                                    n_coords_3d += 1
                                # 3b. ReworkedDecoder pass — only when
                                # available; produces a DecodedMol with
                                # bond classification.  We log the
                                # status (decoded / disconnected / etc.)
                                # so the report can audit lift.
                                if _rd_decoder is not None and AtomCloud is not None:
                                    try:
                                        _z = _torch_local.as_tensor(
                                            [
                                                _mol_no_h.GetAtomWithIdx(
                                                    int(i)
                                                ).GetAtomicNum()
                                                for i in range(_n)
                                            ],
                                            dtype=_torch_local.long,
                                        )
                                        _xyz = _torch_local.as_tensor(
                                            _coord_np,
                                            dtype=_torch_local.float32,
                                        )
                                        _cloud = AtomCloud(
                                            positions=_xyz,
                                            atomic_numbers=_z,
                                        )
                                        _dec = _rd_decoder.decode(_cloud)
                                        c.decoder_status = (
                                            _dec.error or "decoded_path_b"
                                        )
                                    except Exception as _exc_rework_run:
                                        c.decoder_status = (
                                            f"rework_pipeline_failed:"
                                            f"{type(_exc_rework_run).__name__}"
                                        )
                                else:
                                    c.decoder_status = "rework_unavailable"
                            except Exception:
                                pass
                except Exception:
                    pass
            # 4. scaffold-aware gate — classify the metal scaffold and
            # record whether the candidate would be FILTERED OUT under
            # strict_Pt_II rules (ThiolEne + AmideCoupling).  We do
            # NOT drop the candidate — we just record the gate status
            # for downstream reporting (the search already ran with
            # the resolved rules; this is a post-hoc *audit*).
            if detect_scaffold is not None:
                try:
                    _scaffold = detect_scaffold(smi, name_hint=metal_seed)
                    scaffold_seen = _scaffold
                    c.scaffold_detected = _scaffold
                    if incompatible_rules is not None:
                        _incompat = incompatible_rules(_scaffold)
                        c.incompatible_rules = list(_incompat)
                        if _incompat:
                            n_scaffold_aware_active += 1
                except Exception:
                    c.scaffold_detected = None
                    c.incompatible_rules = []
        if scaffold_seen is not None:
            cell.warnings.append(f"scaffold_aware_seen={scaffold_seen}")
    cell.n_coords_3d_attached = n_coords_3d
    cell.n_scaffold_aware_gate_active = n_scaffold_aware_active
    # decoder_pass_rate -- fraction of candidates that ran through the
    # ReworkedDecoder and got a usable bond-bearing mol back.  We
    # classify "decoded_path_b" / "rework_sanitize_failed" as
    # passes (decoder emitted bonds) and everything else as fails.
    if candidates:
        _pass = 0
        _total = 0
        for _c in candidates:
            _status = getattr(_c, "decoder_status", None)
            if _status is None:
                continue
            _total += 1
            if str(_status).startswith("decoded_path_b") or str(_status).startswith(
                "rework_sanitize_failed"
            ):
                _pass += 1
        if _total > 0:
            cell.decoder_pass_rate = float(_pass) / float(_total)

    smis: List[str] = []
    for c in candidates:
        smi = _candidate_smi(c)
        if smi:
            smis.append(smi)
    cell.candidates = smis
    cell.n_candidates = len(smis)
    cell.n_distinct = len(set(smis))

    # ---- seven metric fields ----------------------------------------
    # 1) validity_rate
    if smis:
        cell.validity_rate = sum(rdkit_validity_score(s) for s in smis) / len(smis)
    # 2) uniqueness_rate
    if smis:
        cell.uniqueness_rate = cell.n_distinct / cell.n_candidates
    # 3) diversity_tanimoto (SE(3) baseline / atom-symbol-histogram axis;
    #    back-compat alias for the historical ``diversity_alpha``).
    cell.diversity_tanimoto = diversity_alpha(candidates)
    # 4) diversity_homotype (Lambda-native; WF-Lambda-2 — typed-variable
    #    hits + β-reduction-depth + click-rule-fires).
    cell.diversity_homotype = diversity_homotype(candidates)
    # 5) novelty — placeholder 1.0 (training set loaded separately)
    cell.novelty = 1.0
    # 6) synthesizability_rate
    if smis:
        cell.synthesizability_rate = sum(
            1.0
            if synthesizability_via_lambda_paths(
                _state_from_smi(s, MoleculeClosedTerm),
                check_bnf=check_beta_normal_form,
            )
            else 0.0
            for s in smis
        ) / len(smis)
    # 7) metal_compliance_rate (WF-Lambda-Fix-Singleton Fix 3 — truthful)
    if smis:
        # Compute the MCTS root's canonical SMILES once — used as the
        # baseline for the "candidate was MCTS-derived, not just the
        # seed re-emitted" check.  A candidate equal to the root
        # trivially fails the truthful gate (no expansion step ran).
        try:
            root_smi = root.canonical_smiles() or ""
        except Exception:
            root_smi = ""
        # Build the seed-only SMILES set.  We treat all entries of
        # METAL_SEED_SMILES as seed-only (these are the bare-metal +
        # alkyne motifs from WF-Lambda-Fix-Singleton Fix 1) plus the
        # 2-D fallback ``Cl[Pt]Cl`` (Round-10 path).  Any candidate
        # whose canonical SMILES is in this set fails the truthful
        # gate even when the chemistry prior would otherwise fire.
        seed_smis: set = set()
        try:
            seed_smis.update(METAL_SEED_SMILES.values())
        except Exception:
            pass
        seed_smis.add("Cl[Pt]Cl")
        cell.metal_compliance_rate = sum(
            metal_compliance_truthful(
                _state_from_smi(s, MoleculeClosedTerm),
                enabled=prior_enabled,
                root_smiles=root_smi or None,
                seed_smiles_set=seed_smis,
            )
            for s in smis
        ) / len(smis)
        # WF-Lambda-Fix-FullPath Fix 4: split compliance by seed membership.
        # The truthful gate above already excludes seed members
        # (``seed_smiles_set``) AND the root, so
        # ``cell.metal_compliance_rate`` is now aliased to
        # ``metal_compliance_rate_non_seed`` (the truthful view).
        # ``metal_compliance_rate_including_seed`` is the
        # backward-compatible view (counts the seed re-emitted) so
        # historical Round-12 cells / closed-loop regression tests
        # can still audit the old behaviour.  We compute it by
        # running ``metal_compliance_truthful`` with
        # ``seed_smiles_set=None`` so the seed-only filter is
        # disabled; the root-difference check stays on so we still
        # exclude the un-expanded root (otherwise the metric would
        # trivially hit 1.0 whenever the search emits the seed).
        cell.metal_compliance_rate_non_seed = cell.metal_compliance_rate
        cell.metal_compliance_rate_including_seed = sum(
            metal_compliance_truthful(
                _state_from_smi(s, MoleculeClosedTerm),
                enabled=prior_enabled,
                root_smiles=root_smi or None,
                seed_smiles_set=None,  # disable seed-only filter
            )
            for s in smis
        ) / len(smis)
    # diagnostic: reference_tanimoto (best candidate vs reference)
    if reference_smiles and smis:
        tanimotos = [reference_tanimoto(s, reference_smiles) for s in smis]
        cell.reference_tanimoto = max(tanimotos) if tanimotos else 0.0

    # ---- WF-P0-Metrics: 9 P0 columns ----------------------------------
    cell.logp_mean = metric_logp_mean(smis)
    cell.tpsa_mean = metric_tpsa_mean(smis)
    cell.rotb_mean = metric_rotb_mean(smis)
    cell.oxidation_state_distribution = metric_oxidation_state_distribution(smis)
    cell.coordination_number_mean = metric_coordination_number_mean(smis)
    cell.monodentate_cl_count = metric_monodentate_cl_count(smis)
    cell.gsh_evasion_score = metric_gsh_evasion_score(smis)
    cell.dna_kb_proxy = metric_dna_kb_proxy(smis)
    cell.anticancer_index = metric_anticancer_index(smis)
    # ---- WF-Phase3b-MetricsV2: 8 tumor-relevant anticancer columns ----
    cell.logp7_4_mean = metric_logp7_4_mean(smis)
    cell.gi50_proxy_mean = metric_gi50_proxy_mean(smis)
    cell.cell_permeability_logPapp_mean = metric_cell_permeability_logPapp_mean(smis)
    cell.herg_cardio_risk_mean = metric_herg_cardio_risk_mean(smis)
    cell.ames_mutagen_mean = metric_ames_mutagen_mean(smis)
    cell.hepatotox_index_mean = metric_hepatotox_index_mean(smis)
    cell.aqueous_solubility_logS_mean = metric_aqueous_solubility_logS_mean(smis)
    cell.plasma_protein_binding_mean = metric_plasma_protein_binding_mean(smis)
    # ---- WF-Phase3D-PerResidue-Diversity: sub-pocket diversity ----
    # Phase 4 ships with a uniform-weight default (n_residues=4).  When
    # the pocket loader is wired (voxelization.py:208,255 /
    # vina_adapter.py:961), pass the per-residue weight vector from
    # there.  See ``phase3d_subpocket_div.md`` integration note.
    cell.diversity_subpocket = metric_subpocket_diversity_mean(smis, n_residues=4)
    # ---- WF-Phase3G-MetalCoordProbe: metal-coordination compliance ----
    cell.metal_coord_compliance = metric_metal_coord_compliance_mean(smis)
    # ---- WF-SA-Penalty-Guidance: SA + QED aggregates ----
    cell.sa_mean = metric_sa_mean(smis)
    cell.qed_mean = metric_qed_mean(smis)
    cell.sa_weight = float(sa_weight)
    # ---- WF-T25-Metallo-Proxies: 4 metallodrug-specific proxies --------
    # Opt-in via --metallodrug-proxies flag (default OFF so existing
    # cells stay bit-exact backward-compatible).  When OFF the four
    # fields stay at 0.0 and the call costs O(candidates) RDKit mol
    # parses — negligible.
    if bool(metallodrug_proxies):
        cell.metallodrug_reduction_potential_mean = (
            metric_metallodrug_reduction_potential_mean(smis)
        )
        cell.metallodrug_trans_effect_mean = metric_metallodrug_trans_effect_mean(smis)
        cell.metallodrug_lfse_mean = metric_metallodrug_lfse_mean(smis)
        cell.metallodrug_pt_dna_crosslink_mean = (
            metric_metallodrug_pt_dna_crosslink_mean(smis)
        )
    # ---- WF-T30 P4.2: patent / known-Pt-drug similarity axis ----------
    # Opt-in via --patent-axis flag (default OFF for backward compat).
    # When ON, we compute 5 metrics per cell from the MCTS candidate
    # pool against the 7 known Pt drugs in
    # ``molmetal/data/known_pt_drugs.csv``:
    #   - patent_max_sim_mean:        mean of max-Tanimoto per candidate
    #   - patent_any_above_0_4_rate:  fraction of candidates with
    #                                 max_sim > 0.4 (moderate risk)
    #   - patent_any_above_0_7_rate:  fraction of candidates with
    #                                 max_sim > 0.7 (high risk)
    #   - patent_scaffold_match_rate: fraction sharing a Bemis-Murcko
    #                                 scaffold with a known Pt drug
    #   - patent_closest_drug:        most-frequent nearest-drug name
    #                                 across candidates (audit field)
    # When OFF all 5 fields stay at the dataclass default (0.0 / "")
    # for backward compatibility.  Honest framing in
    # ``molmetal/reports/wf_t30_patent_axis/final.md`` — this is a
    # CHEMICAL-NOVELTY proxy, not a substitute for a patent search.
    if bool(patent_axis) and smis:
        try:
            from molmetal.molmetal_lam.lam_chem.data_diversity import (
                metric_max_sim_known_pt_drugs,
                scaffold_in_known_pt_drugs,
            )

            max_sims: List[float] = []
            above_0_4 = 0
            above_0_7 = 0
            scaf_match = 0
            from collections import Counter

            closest_drug_counts: Counter = Counter()
            for smi in smis:
                m = metric_max_sim_known_pt_drugs(smi)
                max_sims.append(float(m["max_sim"]))
                if m["any_drug_above_0_4"]:
                    above_0_4 += 1
                if m["any_drug_above_0_7"]:
                    above_0_7 += 1
                if scaffold_in_known_pt_drugs(smi):
                    scaf_match += 1
                if m["closest_drug"]:
                    closest_drug_counts[m["closest_drug"]] += 1
            if max_sims:
                cell.patent_max_sim_mean = float(sum(max_sims) / len(max_sims))
            denom = max(1, len(smis))
            cell.patent_any_above_0_4_rate = float(above_0_4 / denom)
            cell.patent_any_above_0_7_rate = float(above_0_7 / denom)
            cell.patent_scaffold_match_rate = float(scaf_match / denom)
            if closest_drug_counts:
                # Tie-break by first insertion order — sort by count desc.
                cell.patent_closest_drug = str(closest_drug_counts.most_common(1)[0][0])
        except Exception as exc:
            cell.warnings.append(f"patent_axis_compute_failed: {exc}")
    # ---- WF-Rigid-RMSD-Metric: rigid RMSD vs reference ligand ----
    # We compute it as a per-cell mean over (candidate, reference)
    # pairs using AllChem.GetBestRMS.  The reference mol is built once
    # (with a 3-D conformer) and re-used across candidates to keep the
    # cost O(n_candidates) embeds + 1 reference embed per cell.  When
    # no reference SMILES is available, the cell-level metric is 0.0
    # (honest-framing: the metric is not meaningful without a target).
    if reference_smiles and smis:
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore

            ref_mol = Chem.MolFromSmiles(reference_smiles)
            if ref_mol is not None and _embed_3d_for_rmsd(ref_mol):
                rmsd_vals: List[float] = []
                for s in smis:
                    cand_mol = Chem.MolFromSmiles(s)
                    if cand_mol is None:
                        continue
                    v = metric_rigid_rmsd_mean(cand_mol, ref_mol)
                    if v > 0.0:
                        rmsd_vals.append(v)
                if rmsd_vals:
                    cell.rigid_rmsd_mean = float(sum(rmsd_vals) / len(rmsd_vals))
        except Exception as exc:
            cell.warnings.append(f"rigid_rmsd_compute_failed: {exc}")
    # ---- WF-CoM-Shift-Metric: CoM shift vs reference ligand ----
    # Same reference-mol-with-conformer setup as rigid_rmsd_mean; we
    # reuse the already-embedded ``ref_mol`` from the rigid_rmsd block
    # when both metrics ran (saves one ETKDGv3 embed per cell).  When
    # the rigid_rmsd block ran but the candidate pool was empty, we
    # skip the CoM computation too.  When no reference SMILES is
    # available the cell-level metric stays 0.0 (honest-framing).
    if reference_smiles and smis:
        try:
            from rdkit import Chem  # type: ignore

            ref_mol_com = None
            # Reuse the embedded ref_mol from the rigid_rmsd block when
            # it exists in the local scope (it does when the prior block
            # ran end-to-end).  Re-embedding is unnecessary — but we do
            # re-fetch via a fresh ``Chem.MolFromSmiles`` to keep the
            # blocks independent and side-effect free.
            ref_mol_com = Chem.MolFromSmiles(reference_smiles)
            if ref_mol_com is not None and _embed_3d_for_rmsd(ref_mol_com):
                com_vals: List[float] = []
                for s in smis:
                    cand_mol = Chem.MolFromSmiles(s)
                    if cand_mol is None:
                        continue
                    v = metric_com_shift_mean(cand_mol, ref_mol_com)
                    if v > 0.0:
                        com_vals.append(v)
                if com_vals:
                    cell.com_shift_mean = float(sum(com_vals) / len(com_vals))
        except Exception as exc:
            cell.warnings.append(f"com_shift_compute_failed: {exc}")

    return cell


# ---------------------------------------------------------------------------
# ``args`` shim — the original ``run_one_cell`` consulted
# ``globals().get("args")`` to read CLI-only knobs (``fg_veto_strict``,
# ``wetlab_input``).  When :func:`run_one_cell` is invoked via the
# library (e.g. from a unit test) there is no module-level ``args``
# namespace, so we expose a placeholder :class:`_Args` that
# ``getattr(...)`` can default on.  When :func:`main` runs the CLI it
# will ``globals()["args"] = args`` so :func:`run_one_cell` sees the
# live namespace.
# ---------------------------------------------------------------------------
class _Args:
    """Lightweight namespace used as the module-level ``args`` default.

    ``run_one_cell`` consults ``globals().get("args")`` for CLI-only
    knobs (``fg_veto_strict``, ``wetlab_input``).  When the function is
    invoked outside the CLI (e.g. from a unit test) the default
    ``getattr(args, "...", False)`` / ``getattr(args, "...", None)``
    fallbacks must still resolve to safe values, so this class exposes
    every attribute the harness reads.
    """

    fg_veto_strict: bool = False
    wetlab_input: Optional[str] = None


args: Optional[_Args] = None  # set by main() before invoking run_sweep


__all__ = [
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
]
