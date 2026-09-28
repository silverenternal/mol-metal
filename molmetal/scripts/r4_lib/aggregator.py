"""r4_lib.aggregator — :class:`CellResult` + the Lambda-only reward factory.

The :class:`CellResult` dataclass is the per-(pocket, seed) cell
output — every metric column that the report serialises.  Originally
lived at lines 2204-2336 of ``r4_lambda_only_run.py``.

The :func:`build_lambda_only_aggregator` factory wires the five
Lambda-native scoring channels (see :mod:`.scoring`) into a
:class:`molmetal_lam.search_alg.proof_search.RewardAggregator`,
plus optional PlatinAI / pharmacophore / wet-lab channels when the
matching CLI weights are > 0.  Originally lived at lines 2017-2198.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .constants import CLICK_RULE_WEIGHT, METAL_PRIOR_WEIGHT
from .scoring import (
    click_rule_match_bonus,
    metal_geometry_prior_bonus,
    rdkit_validity_score,
    synthesizability_via_lambda_paths,
)


@dataclass
class CellResult:
    """All metrics for one (pocket, seed) cell."""

    pocket_id: str
    seed: int
    n_simulations: int
    n_top_k: int
    reference_smiles: Optional[str]
    candidates: List[str] = field(default_factory=list)
    n_candidates: int = 0
    n_distinct: int = 0
    validity_rate: float = 0.0
    uniqueness_rate: float = 0.0
    diversity_tanimoto: float = 0.0
    diversity_homotype: float = 0.0
    novelty: float = 1.0
    synthesizability_rate: float = 0.0
    metal_compliance_rate: float = 0.0
    # ---- WF-Lambda-Fix-FullPath Fix 4: split compliance by seed membership
    # The original ``metal_compliance_rate`` counts ALL candidates
    # returned by MCTS — including the seed re-emitted when the
    # search collapses.  This is the "backward-compatible" view and
    # matches historical Round-12 cells (always 1.0 when the seed is
    # metal-compliant).  ``metal_compliance_rate_non_seed`` is the
    # *truthful* view: it excludes any candidate whose canonical
    # SMILES is in ``seed_smiles_set`` (the same set used by
    # :func:`metal_compliance_truthful`).  Honest framing: the
    # non-seed view is what we should be reporting in §4.6 because
    # it answers "did MCTS produce novel compliant candidates?",
    # while the including-seed view is what the closed-loop
    # regression tests still expect.  Both fields are populated in
    # ``run_one_cell`` so downstream consumers can audit the delta.
    metal_compliance_rate_including_seed: float = 0.0
    metal_compliance_rate_non_seed: float = 0.0
    reference_tanimoto: float = 0.0
    # ---- WF-P0-Metrics: 9 P0 columns (closes 9/25 of TargetDiff gap) ----
    logp_mean: float = 0.0
    tpsa_mean: float = 0.0
    rotb_mean: float = 0.0
    oxidation_state_distribution: Dict[str, int] = field(default_factory=dict)
    coordination_number_mean: float = 0.0
    monodentate_cl_count: int = 0
    gsh_evasion_score: float = 0.0
    dna_kb_proxy: float = 0.0
    anticancer_index: float = 0.0
    # ---- WF-Phase3b-MetricsV2: 8 tumor-relevant anticancer columns ----
    # Drug-likeness / ADMET / safety axes that complement the P0
    # chemistry / metal-coordination columns above.  See
    # ``phase3b_metrics_v2.md`` for the lit basis.  All bounded.
    logp7_4_mean: float = 0.0
    gi50_proxy_mean: float = 0.0
    cell_permeability_logPapp_mean: float = 0.0
    herg_cardio_risk_mean: float = 0.0
    ames_mutagen_mean: float = 0.0
    hepatotox_index_mean: float = 0.0
    aqueous_solubility_logS_mean: float = 0.0
    plasma_protein_binding_mean: float = 0.0
    # ---- WF-Phase3D-PerResidue-Diversity: sub-pocket diversity ----
    # Mean per-residue Tanimoto *distance* over distinct candidates.
    # Complements ``diversity_tanimoto`` (whole-mol) and
    # ``diversity_homotype`` (typed-var Jaccard).  0.0 on < 2 distinct
    # candidates or import failure.
    diversity_subpocket: float = 0.0
    # ---- WF-Phase3G-MetalCoordProbe: metal-coordination compliance ----
    # Fraction of metal-containing molecules whose CN matches the
    # canonical (metal, OS) → CN table (Lippard & Berg 1995).  0.0 on
    # no-metal or import failure.
    metal_coord_compliance: float = 0.0
    # ---- WF-SA-Penalty-Guidance: SA + QED aggregates (per cell) ----
    sa_mean: float = 0.0
    qed_mean: float = 0.0
    sa_weight: float = 0.0
    # ---- WF-Rigid-RMSD-Metric: rigid-fragment RMSD vs reference ligand
    # (Angstroms; lower = better; TargetDiff SOTA target ≈ 1.5).  0.0
    # when the reference or candidate SMILES cannot be embedded in 3-D
    # (RDKit's GetBestRMS requires a conformer).  This is a *conditional*
    # mean — only embeddable candidates contribute.
    rigid_rmsd_mean: float = 0.0
    # ---- WF-CoM-Shift-Metric: center-of-mass shift vs reference ligand
    # (Angstroms; lower = better; TargetDiff SOTA target ≈ 3).  0.0
    # when the reference or candidate SMILES cannot be embedded in 3-D.
    # Conditional mean — only embeddable candidates contribute.  Both
    # mols must already have a conformer attached (see
    # :func:`_embed_3d_for_rmsd`); the metric silently skips mols that
    # cannot be embedded, matching the rigid_rmsd_mean convention.
    com_shift_mean: float = 0.0
    # ---- WF-T25-Metallo-Proxies: 4 metallodrug-specific proxies --------
    # Shriver & Atkins Table 17.7 + Appleton 1997 + Wang 2005.  All
    # values in [0, 1]; 0.0 for non-Pt molecules by construction.
    metallodrug_reduction_potential_mean: float = 0.0
    metallodrug_trans_effect_mean: float = 0.0
    metallodrug_lfse_mean: float = 0.0
    metallodrug_pt_dna_crosslink_mean: float = 0.0
    # ---- WF-T30 P4.2: patent / known-Pt-drug similarity axis ---------
    # Opt-in via --patent-axis flag (default OFF so existing cells
    # stay bit-exact backward-compatible).  Mean max-Tanimoto to
    # the 7 known Pt drugs in molmetal/data/known_pt_drugs.csv.  Plus
    # a Bemis-Murcko scaffold-match rate against the same set.  See
    # ``molmetal/reports/wf_t30_patent_axis/final.md`` for the honest
    # framing — this is a CHEMICAL-NOVELTY proxy, not a substitute
    # for a patent search.
    patent_max_sim_mean: float = 0.0
    patent_any_above_0_4_rate: float = 0.0
    patent_any_above_0_7_rate: float = 0.0
    patent_scaffold_match_rate: float = 0.0
    patent_closest_drug: str = ""
    # ---------------------------------------------------------------------
    nfe: int = 0
    elapsed_s: float = 0.0
    warnings: List[str] = field(default_factory=list)
    # ---- WF-Lambda-MCTS-Coords-Fix: 3 audit fields ---------------------
    # ``n_coords_3d_attached`` — number of candidates whose 3-D
    # coordinate tensor was successfully populated by ETKDGv3 (after
    # the co-emit path on proof_search.search(..., materialize_3d=True)
    # + the explicit re-embed here in run_one_cell).  When 0 the
    # scaffold-aware gate + ReworkedDecoder are no-ops and the
    # downstream report will flag this as "no coords available".
    # ``n_scaffold_aware_gate_active`` — number of candidates whose
    # detected metal-scaffold category has incompatible click rules
    # under ``pt_click_compat``.  This is the count that the
    # scaffold-aware gate *would filter out* if the user had passed
    # ``auto-pt-strict`` at search-time.  0 means no scaffold-
    # incompatible candidate was seen (typical for non-Pt scaffolds).
    # ``decoder_pass_rate`` — fraction of candidates whose
    # ReworkedDecoder returned a usable mol with at least one bond
    # (decoded_path_b / rework_sanitize_failed are counted as
    # "decoder emitted bonds"; disconnected / rework_unavailable
    # are counted as failures).  0.0 when the ReworkedDecoder is
    # unavailable.
    n_coords_3d_attached: int = 0
    n_scaffold_aware_gate_active: int = 0
    decoder_pass_rate: float = 0.0


# ---------------------------------------------------------------------------
# Reward aggregator — purely Lambda-native channels (no docking / PB / ADMET).
# ---------------------------------------------------------------------------
def build_lambda_only_aggregator(
    prior: Any,
    *,
    sa_weight: float = 0.0,
    platinai_weight: float = 0.0,
    platinai_oracle: "Optional[Any]" = None,
    lambda_reward_weight: float = 0.0,
    pharmacophore_weight: float = 0.0,
    pharmacophore_strict: bool = False,
    pharmacophore_allow_acyclic: bool = True,
    wetlab_weight: float = 0.0,
    wetlab_measurements: "Optional[Any]" = None,
) -> Any:
    """Return a :class:`RewardAggregator` with ONLY the Lambda channels.

    WF-SA-Penalty-Guidance: ``sa_weight`` (CLI ``--sa-weight``) wires the
    Ertl-Schuffenhauer synthetic-accessibility channel into the MCTS
    leaf reward.  The default ``sa_weight=0.0`` keeps the historical
    Lambda-only aggregator bit-for-bit identical (SA channel
    contributes zero, since ``w_sa * 0 = 0`` and the SA value is
    inverted from [1, 10] -> [0, 1] so the unit-scale contribution is
    well-defined).  When ``sa_weight > 0`` the aggregator actively
    penalises SA-heavy molecules during MCTS expansion; the leaf reward
    becomes::

        value(s) = w_vina_proxy * r_vina_proxy(s)        # 5-channel Lambda
                 + w_sa          * r_sa(s)               # Ertl SA, inverted

    ``r_sa`` returns the raw Ertl SA in [1, 10] (lower = more
    synthesizable); the aggregator's ``__call__`` applies
    ``1 - (raw - 1)/9`` so the contribution is "higher = better".

    Phase 2 (Metallodrug ultracode): ``platinai_weight`` (CLI
    ``--reward-platinai-weight``) wires the PlatinAI predicted-activity
    oracle channel.  When ``platinai_weight > 0`` AND ``platinai_oracle``
    is provided, the aggregator adds ``w_platinai * r_platinai(s)`` to
    the leaf reward; the oracle itself is a kNN over the
    PlatinAI_MBFinder corpus (226,918 SMILES) + A2780/MCF7
    predicted-activity sheets.
    """
    from molmetal_lam.search_alg.proof_search import RewardAggregator
    from molmetal_lam.lam_chem.well_formedness import check_beta_normal_form

    rules = _import_lambda_stack()["CLICK_REACTIONS"]

    def _lambda_native_score(state: Any) -> float:
        # alpha-equivalence uniqueness across the children of the parent.
        # We do not have the parent's children at hand here, so we use
        # the molecule's own canonical SMILES as a proxy: distinct
        # SMILES -> distinct beta-NF -> +1.0; duplicate -> +0.0.
        try:
            smi = state.canonical_smiles() or ""
        except Exception:
            smi = ""
        aeq = 1.0 if smi else 0.0
        # click rule bonus
        click = click_rule_match_bonus(state, rules)
        # metal prior (always-on in the aggregator; ablations toggle via
        # ``metal_prior_enabled`` passed to the harness).
        metal = metal_geometry_prior_bonus(state, enabled=True)
        # RDKit validity
        valid = rdkit_validity_score(smi)
        # beta-NF synthesis
        syn = synthesizability_via_lambda_paths(state, check_bnf=check_beta_normal_form)
        # WF-Lambda-Fix-Singleton Fix 2: reward rebalance.  The
        # historical per-channel unit weight (1.0 click + 1.0 metal
        # + 1.0 valid + 1.0 syn + 1.0 aeq = max 5.0) over-weighted
        # metal_compliance — the audit showed that on metal-seed
        # pockets the metal channel dominated the leaf value even
        # when no click rule had ever fired, biasing selection back
        # to the seed.  We now apply a configurable weight so that
        # click-rule application gets at least 0.5 weight and the
        # metal channel stays at 0.5 (i.e. click and metal are
        # balanced).  Valid / syn / aeq remain at 1.0 each because
        # they are correctness signals, not preferences.
        w_click = float(CLICK_RULE_WEIGHT)
        w_metal = float(METAL_PRIOR_WEIGHT)
        return float(aeq + w_click * click + w_metal * metal + valid + syn)

    # SA scoring closure: returns raw Ertl SA in [1, 10] (lower = better
    # synthesizability).  The aggregator handles the inversion.  When
    # sascorer cannot be imported the channel returns NaN and the
    # aggregator's graceful-degradation path emits ``0.0`` for the
    # SA contribution.
    def _sa_score(state: Any) -> float:
        try:
            smi = state.canonical_smiles() or ""
        except Exception:
            return float("nan")
        if not smi:
            return float("nan")
        try:
            from molmetal_lam.sbdd_env.sa_score import sa_score_ertl  # type: ignore
            return float(sa_score_ertl(smi))
        except Exception:
            return float("nan")

    agg_kwargs: Dict[str, Any] = dict(
        r_vina_proxy=_lambda_native_score,
        w_vina_proxy=1.0,
        r_sa=_sa_score,
        w_sa=float(sa_weight),
    )
    if platinai_oracle is not None and float(platinai_weight) > 0.0:
        # Phase 2 (Metallodrug ultracode) — wire the PlatinAI oracle
        # into the aggregator.  We expose it as a constructor arg via
        # :meth:`RewardAggregator.register_platinai_oracle_channel`
        # after construction (a fluent-style helper added in this
        # commit).  Default w_platinai = 0.0 keeps the historical
        # reward bit-for-bit identical when the channel is not wired.
        agg = RewardAggregator(**agg_kwargs)
        agg.register_platinai_oracle_channel(platinai_oracle)
        agg.w_platinai = float(platinai_weight)
        if float(lambda_reward_weight) > 0.0:
            # TODO-21 Strategy 1 (2026-09-17) — additive Lambda-as-reward
            # channel wired into the aggregator.  Opt-in (default 0.0)
            # so existing reward is bit-for-bit unchanged when unused.
            from molmetal_lam.lam_chem.lambda_reward_channel import (
                register_lambda_reward_channel,
            )
            register_lambda_reward_channel(agg, weight=float(lambda_reward_weight))
        if float(pharmacophore_weight) > 0.0:
            # TODO-30 P2.5 Phase A (2026-09-17) — wire the pharmacophore
            # reward channel into the aggregator.  Opt-in (default 0.0)
            # so existing reward is bit-for-bit unchanged when unused.
            # We delegate to :func:`register_pharmacophore_channel`
            # which installs r_pharmacophore on the locked filter
            # module without modifying the filter itself.
            from molmetal_lam.lam_chem.pharmacophore_reward_channel import (
                register_pharmacophore_channel,
            )
            register_pharmacophore_channel(
                agg,
                weight=float(pharmacophore_weight),
                strict=bool(pharmacophore_strict),
                allow_acyclic=bool(pharmacophore_allow_acyclic),
            )
        if float(wetlab_weight) > 0.0:
            # TODO-30 P5.2 Tier 1 — wire the wet-lab signed-error
            # channel into the aggregator.  Opt-in (default 0.0) so
            # existing reward is bit-for-bit unchanged when unused.
            from molmetal_lam.lam_chem.wetlab_reward_channel import (
                register_wetlab_channel,
            )
            register_wetlab_channel(
                agg,
                measurement_dict=wetlab_measurements,
                weight=float(wetlab_weight),
            )
        return agg
    agg = RewardAggregator(**agg_kwargs)
    if float(lambda_reward_weight) > 0.0:
        from molmetal_lam.lam_chem.lambda_reward_channel import (
            register_lambda_reward_channel,
        )
        register_lambda_reward_channel(agg, weight=float(lambda_reward_weight))
    if float(pharmacophore_weight) > 0.0:
        # TODO-30 P2.5 Phase A — same wiring as the platinai arm
        # above; keeps the second ``return`` path (no-platinai) in
        # lock-step so the channel is opt-in on both branches.
        from molmetal_lam.lam_chem.pharmacophore_reward_channel import (
            register_pharmacophore_channel,
        )
        register_pharmacophore_channel(
            agg,
            weight=float(pharmacophore_weight),
            strict=bool(pharmacophore_strict),
            allow_acyclic=bool(pharmacophore_allow_acyclic),
        )
    if float(wetlab_weight) > 0.0:
        # TODO-30 P5.2 Tier 1 — same as the platinai arm above;
        # mirrors the second-return path so the channel is opt-in on
        # both branches.
        from molmetal_lam.lam_chem.wetlab_reward_channel import (
            register_wetlab_channel,
        )
        register_wetlab_channel(
            agg,
            measurement_dict=wetlab_measurements,
            weight=float(wetlab_weight),
        )
    return agg


def _import_lambda_stack():
    """Local re-export so ``build_lambda_only_aggregator`` can lazy-import
    the ``CLICK_REACTIONS`` dict without depending on the runner
    module's import order.  Equivalent to
    ``runner._import_lambda_stack()`` — kept here to avoid an import
    cycle.
    """
    from molmetal_lam.lam_chem.rules import CLICK_REACTIONS  # type: ignore
    return {"CLICK_REACTIONS": CLICK_REACTIONS}
