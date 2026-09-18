"""R4-C: Lambda 100-pocket CrossDocked sweep — wire L-3 204-tile library.

GOAL: Build on Lambda round-3 (L-1 DiffDock/FlowDock oracle wired, L-2
persistent tree, L-3 204-tile library, L-4 REINVENT4 fallback) and run a
100-pocket head-to-head sweep on CrossDocked2020. Compare against
TransDiffSBDD (Vina Dock -9.37, success 83.9%), MolCRAFT (-9.25/36.1%),
AlphaDrug (-9.77), Pocket2Mol (-7.15/24.4%), TargetDiff (-7.80/10.5%),
DecompDiff (-8.39/24.5%).

Per-pocket output: 200 MCTS sims (L-3 204-tile, branching factor 1020),
top-20 candidates ranked by Vina proxy + SA + QED. For each pocket we
record: n_candidates_returned, mean_sa, mean_qed, n_lipinski_pass,
mean_vina_proxy, top-1 SMILES.

R4-D defaults (r0 fix — lift n_candidates/pocket to SOTA-comparable):

    * ``top_k=20``       (was 5) — primary n_candidates knob
    * ``max_depth=3``    (was 2) — rollout depth budget
    * ``n_simulations=200`` (was 1000) — keep wall-time bounded
    * ``early_stop=True, patience=30`` — round-0 UCB-converges-fast fix

Expected wall-time per pocket (round-0 measurement, RX 7800 XT, single
thread):
    * 12-tile library, branching=60,   top_k=5,  max_depth=2  →   14s
    * 204-tile library, branching=1020, top_k=5,  max_depth=2  →  298s  (RDKit sanitize dominates)
    * 204-tile library, branching=1020, top_k=20, max_depth=3  →  ~480s (projection: +60% from deeper rollouts + 4x more leaves to score)

For a 100-pocket sweep, this is ~13h.  Use --no-fragments to drop
back to the 12-tile library (and ~70s/pocket = ~2h total).

Network policy (per Round-3 + user directive "装得下就行"):
- CPU-only: Vina docking is the bottleneck and not GPU-bound.
- Triton kernel not exercised here (Lambda MCTS is Python).
- ROCm 7.2 + Triton 3.8 untouched.

Writes:
    molmetal/reports/r4_c_100pocket_sweep.csv
    molmetal/reports/r4_c_100pocket_sweep.json
    molmetal/reports/r4_c_100pocket_sweep.md

Pre-flight: Vina 1.2.x at /usr/bin/vina, Meeko + RDKit + openbabel installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from functools import lru_cache
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from rdkit import Chem

from molmetal_lam.binding.types import PROTEASE_GENERIC
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import REACTION_RULES
from molmetal_lam.scripts._sweep_helpers import (
    BRANCHING_LABELS,
    build_reward,
    build_seed_smiles,
    build_seed_term,
    build_tile_library_for_branching,
    lipinski_pass,
    smi_of,
    vina_proxy,
)
from molmetal_lam.search_alg.proof_search import MCTSProofSearch, RewardAggregator
from molmetal_lam.tile_lib.library import build_tile_library, FRAGMENT_LIBRARY_200_TILES
from molmetal_lam.types.predicates import LIPINSKI


CROSSDOCKED_ROOT = "/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10"


# ---------------------------------------------------------------------------
# Private aliases — preserve the legacy ``_smi_of`` / ``_lipinski_pass`` /
# ``_build_reward`` / ``_sa_proxy`` / ``_qed`` / ``_vina_proxy`` names
# so any external caller (orchestrator, notebook) keeps working.
# ---------------------------------------------------------------------------

_smi_of = smi_of
_lipinski_pass = lipinski_pass
_build_reward = build_reward
_vina_proxy = vina_proxy


def _sa_proxy(smi: str) -> float:
    """1..10 SA proxy via MW + logP + rotatable bonds (Ertl-like).

    Direct computation kept as a module-level helper for callers that
    want a raw SA score without going through the reward aggregator.
    """
    from rdkit.Chem import Descriptors as _D
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float("nan")
    mw = float(_D.MolWt(mol))
    logp = float(_D.MolLogP(mol))
    rotb = float(_D.NumRotatableBonds(mol))
    return max(1.0, min(10.0, 0.02 * mw + 0.5 * abs(logp) + 0.1 * rotb))


def _qed(smi: str) -> float:
    """Raw QED value (0..1, higher = better)."""
    from rdkit.Chem import QED as _QED
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float("nan")
    try:
        return float(_QED.qed(mol))
    except Exception:
        return float("nan")


def _sa_score(smi: str) -> float:
    """Report the reference Ertl SA score, never the search heuristic."""
    try:
        from molmetal_lam.sbdd_env.sa_score import sa_score_ertl
        return float(sa_score_ertl(smi))
    except Exception:
        return float("nan")


def _sa_backend() -> str:
    try:
        from molmetal_lam.sbdd_env.sa_score import _SASCORER_AVAILABLE
        return "rdkit_contrib_ertl" if _SASCORER_AVAILABLE else "unavailable"
    except ImportError:
        return "unavailable"


@lru_cache(maxsize=4096)
def _structure_key(smiles: str) -> Optional[str]:
    """Graph identity excluding atom maps and explicit-H spelling differences."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    except Exception:
        return None


def _observe_search(search, seed_key):
    """Observe existing checks without changing reduction or filtering decisions.

    Counts cover all search calls (including rollouts), not disjoint final-leaf
    populations. The MCTS safe reduction API merges non-matches and exceptions;
    report that ambiguity rather than claiming all empty expansions are errors.
    """
    counters = {"reduction_calls": 0, "reductions_without_products": 0,
                "products_returned": 0}
    states = {"new_products": set(), "typed_rejected": set(), "binding_rejected": set()}
    reduce = search._safe_reduce
    typed = search._satisfies_predicates
    binding = search._binds_target

    def observe_reduce(rule, state, tile):
        products = reduce(rule, state, tile)
        counters["reduction_calls"] += 1
        counters["reductions_without_products"] += int(not products)
        counters["products_returned"] += len(products)
        for product in products:
            key = _structure_key(_smi_of(product))
            if key is not None and key != seed_key:
                states["new_products"].add(key)
        return products

    def observe_typed(state, predicates):
        result = typed(state, predicates)
        if not result:
            key = _structure_key(_smi_of(state))
            if key is not None and key != seed_key:
                states["typed_rejected"].add(key)
        return result

    def observe_binding(state):
        result = binding(state)
        if not result:
            key = _structure_key(_smi_of(state))
            if key is not None and key != seed_key:
                states["binding_rejected"].add(key)
        return result

    search._safe_reduce = observe_reduce
    search._satisfies_predicates = observe_typed
    search._binds_target = observe_binding
    return counters, states


def _choose_click_seed(seed_tiles, expansion_tiles, rules, rng, library_name):
    """Choose a library tile by a witnessed reaction with an expansion partner.

    Rule templates prefilter ordered reactants, then the real reducer verifies
    a new product. Selection uses library/rule chemistry and the experiment RNG
    only: no reference ligand, target-score ranking, or filter relaxation.
    The witness validates reactivity, not acceptance by the binding gate.
    """
    from rdkit.Chem import rdChemReactions

    seed_mols = [Chem.MolFromSmiles(_smi_of(t)) for t in seed_tiles]
    partner_mols = [Chem.MolFromSmiles(_smi_of(t)) for t in expansion_tiles]
    ordered_rules = list(rules.items())
    rng.shuffle(ordered_rules)
    attempts = 0
    unsupported = []
    for name, rule in ordered_rules:
        try:
            if rule.pattern_smiles:
                rxn = rdChemReactions.ReactionFromSmarts(rule.pattern_smiles)
                if rxn is None or rxn.GetNumReactantTemplates() != 2:
                    unsupported.append(name)
                    continue
                left, right = rxn.GetReactantTemplate(0), rxn.GetReactantTemplate(1)
            elif name == "ThiolEne":
                # This rule's implementation edits the first thiol and alkene.
                left, right = Chem.MolFromSmarts("[S;H1]"), Chem.MolFromSmarts("[C]=[C]")
            else:
                unsupported.append(name)
                continue
            seed_indices = [i for i, mol in enumerate(seed_mols)
                            if mol is not None and mol.HasSubstructMatch(left)]
            partner_indices = [i for i, mol in enumerate(partner_mols)
                               if mol is not None and mol.HasSubstructMatch(right)]
        except Exception:
            unsupported.append(name)
            continue
        rng.shuffle(seed_indices)
        rng.shuffle(partner_indices)
        for i in seed_indices:
            for j in partner_indices:
                attempts += 1
                try:
                    products = rule.reduce((seed_tiles[i], expansion_tiles[j]))
                except Exception:
                    continue
                inputs = {_structure_key(_smi_of(seed_tiles[i])),
                          _structure_key(_smi_of(expansion_tiles[j]))}
                new_products = sorted({key for p in products
                                       if (key := _structure_key(_smi_of(p))) is not None
                                       and key not in inputs})
                if new_products:
                    return seed_tiles[i], {
                        "library": library_name, "tile_index": i,
                        "smiles": _smi_of(seed_tiles[i]),
                        "canonical_seed": _structure_key(_smi_of(seed_tiles[i])),
                        "validated_rule": name, "partner_index": j,
                        "partner_smiles": _smi_of(expansion_tiles[j]),
                        "validation_products": new_products,
                        "validation_reduction_calls": attempts,
                        "selection_basis": "configured tile chemistry and experiment seed only",
                        "unsupported_rule_templates": unsupported,
                    }
    raise ValueError(
        "No reactive ordered seed/partner pair in the configured tile library, "
        f"rules and expansion pool ({attempts} reductions validated; "
        f"unsupported templates: {unsupported})"
    )


# ---------------------------------------------------------------------------
# Pocket discovery
# ---------------------------------------------------------------------------


def list_crossdocked_pockets(offset: int = 0, count: int = 100) -> List[str]:
    """List [offset:offset+count] pocket directories under CrossDocked2020."""
    if not os.path.isdir(CROSSDOCKED_ROOT):
        return []
    entries = sorted(os.listdir(CROSSDOCKED_ROOT))
    return [os.path.join(CROSSDOCKED_ROOT, e) for e in entries[offset:offset + count]]


# ---------------------------------------------------------------------------
# Per-pocket MCTS sweep
# ---------------------------------------------------------------------------


def run_one_pocket(
    pocket_dir: str,
    n_simulations: int,
    max_depth: int,
    seed_offset: int,
    include_fragments: bool,
    use_fragment_pool: bool = True,
    *,
    top_k: int = 20,
    early_stop: bool = True,
    patience: int = 30,
    tile_library: Optional[str] = None,
    click_rules: Optional[str] = None,
    branching_target: Optional[int] = None,
    ligand_path: Optional[str] = None,
    seed: Optional[int] = None,
    symbolic_prior: bool | str | None = None,
    synthesis_oracle: bool | str | None = None,
    symbolic_prior_refit_every: Optional[int] = None,
    seed_strategy: str = "reference",
    prior_state: Optional[dict] = None,
    prior_mode: str = "frozen",
    prior_data_split: str = "test",
    synthesis_config_path: Optional[str] = None,
    docking_reward_config: Optional[dict] = None,
) -> Dict[str, object]:
    """Run one Lambda MCTS sweep on one pocket.

    pocket_dir: CrossDocked pocket directory (contains *_pocket10.pdb).
    Explicit ``tile_library`` accepts ``standard_12`` or ``extended_204``.
    ``click_rules`` accepts ``all_5`` or comma-separated registered names.
    ``branching_target`` caps rule/tile attempts per expansion: tiles are
    truncated to floor(target / rule_count), without duplicating fragments.
    This is not the number of successful products, which depends on chemistry.
    ``ligand_path`` selects exactly the first record of that SDF, with no seed
    fallback on failure. ``seed`` overrides the legacy date + offset seed.
    ``seed_strategy='click_tile'`` instead chooses a tile with a validated
    reaction partner in the actual expansion pool; it never reads test-ligand
    chemistry. Its reaction witness does not bypass the type or binding gate.
    Optional prior/oracle requests are reported as not_applied until supported.
    ``symbolic_prior=True`` enables an explicitly labelled linear_descriptor_prior
    (not PySR) from JSON ``prior_state``. Frozen inference is default; updates
    require prior_mode=train with train/development data or an explicitly
    labelled transductive experiment. Refit intervals count pockets, not sims.
    ``synthesis_oracle=True`` requires a real AiZynth config/model; only explicit
    synthesis_oracle='smarts' enables the non-learned heuristic gate.
    Returns dict with: pocket_id, n_simulations, n_candidates,
    candidates (list of dicts), wall_seconds.
    """
    if seed_strategy not in {"reference", "click_tile"}:
        raise ValueError(f"Unknown seed_strategy: {seed_strategy!r}")
    if n_simulations < 1 or max_depth < 1 or top_k < 1 or patience < 1:
        raise ValueError("n_simulations, max_depth, top_k and patience must be positive")
    from molmetal_lam.lam_chem.rules import CLICK_REACTIONS, list_reactions
    if click_rules is None:
        rules = dict(REACTION_RULES)
    else:
        names = list_reactions() if click_rules == "all_5" else click_rules.split(",")
        rules = {}
        for name in names:
            name = name.strip()
            rule = CLICK_REACTIONS.get(name, REACTION_RULES.get(name))
            if rule is None:
                raise ValueError(f"Unknown click rule: {name!r}")
            rules[rule.name] = rule
    if branching_target is not None and (
        isinstance(branching_target, bool) or not isinstance(branching_target, int)
        or branching_target < len(rules)
    ):
        raise ValueError("branching_target must be an integer >= the selected rule count")
    selected_library = tile_library or ("extended_204" if include_fragments else "standard_12")
    if selected_library not in {"extended_204", "standard_12"}:
        raise ValueError(f"Unknown tile library: {selected_library!r}")
    pocket_id = os.path.basename(pocket_dir)
    effective_seed = 20260912 + seed_offset if seed is None else seed
    from molmetal_lam.search_alg import sweep_guidance as guidance
    prior_enabled = symbolic_prior is True or symbolic_prior == "linear_descriptor_prior"
    learned_prior = None
    replay_state = prior_state
    if prior_enabled:
        replay_state, learned_prior = guidance.prepare_prior(
            prior_state, seed=effective_seed, mode=prior_mode, data_split=prior_data_split)
    prior_report = {"requested": symbolic_prior, "backend": "linear_descriptor_prior",
                    "mode": prior_mode, "data_split": prior_data_split,
                    "applied": learned_prior is not None,
                    "status": "fitted_state_loaded" if learned_prior else
                              "awaiting_real_training_observations" if prior_enabled else
                              "disabled" if not symbolic_prior else "unsupported_request",
                    "puct_calls": 0, "pysr_symbolic_regression": False}
    synthesis_checker, synthesis_report = guidance.build_synthesis_gate(synthesis_oracle, synthesis_config_path)
    rng = random.Random(effective_seed)
    if seed is not None:
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    init_started = time.monotonic()
    chosen_tile = None
    seed_smiles = None
    if seed_strategy == "reference":
        # Seed is the first molecule of the pocket ligand file (if exists),
        # falling back to a cyclopentadiene default via the shared helper.
        seed_smiles = None
        if ligand_path is not None:
            try:
                supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False)
                mol = supplier[0] if len(supplier) else None
                if mol is None:
                    raise ValueError("first ligand record is invalid or empty")
                seed_smiles = Chem.MolToSmiles(mol)
            except Exception as exc:
                return {"pocket_id": pocket_id, "status": "seed_parse_fail",
                        "prior_state": replay_state, "prior_report": prior_report,
                        "synthesis_report": synthesis_report,
                        "init_strategy": seed_strategy, "chosen_tile": None,
                        "canonical_seed": None,
                        "ligand_path": str(ligand_path), "seed_smiles": None,
                        "error": str(exc), "n_candidates": 0, "candidates": []}
        else:
            seed_smiles = build_seed_smiles(pocket_dir)
        if seed_smiles is None and ligand_path is None:
            # Fallback: use a default cyclopentadiene (round-1 deeper-seed choice)
            seed_smiles = "C1=CCC=C1"

        try:
            seed = MoleculeClosedTerm.from_smiles(seed_smiles, embed_3d=False)
        except Exception:
            return {
                "pocket_id": pocket_id,
                "status": "seed_parse_fail",
                "prior_state": replay_state, "prior_report": prior_report,
                "synthesis_report": synthesis_report,
                "seed_smiles": seed_smiles,
                "n_candidates": 0,
                "candidates": [],
            }

    # L-3: 204-tile library
    tiles = (
        FRAGMENT_LIBRARY_200_TILES()
        if selected_library == "extended_204"
        else build_tile_library(12, include_fragments=False)
    )
    tile_terms = [MoleculeClosedTerm.from_smiles(t.smiles, embed_3d=False) for t in tiles]
    seed_tile_terms = list(tile_terms)
    if branching_target is not None:
        tile_terms = tile_terms[:branching_target // len(rules)]

    # Reward: SA + QED + Vina_proxy
    reward = _build_reward()
    # Add Vina proxy channel — zero weight by default until L-1 oracle live
    from dataclasses import replace
    reward = replace(reward, r_vina_proxy=_vina_proxy, w_vina_proxy=0.0)
    docking_reward = None
    docking_reward_report = {"requested": docking_reward_config is not None,
                             "status": "disabled", "applied": False}
    if docking_reward_config is not None:
        try:
            from molmetal_lam.sbdd_env.pocket_docking_reward import PocketDockingReward
            reward_options = dict(docking_reward_config)
            reward_weight = float(reward_options.pop("weight", 0.4))
            if not math.isfinite(reward_weight) or reward_weight < 0:
                raise ValueError("Measured docking reward weight must be finite and nonnegative")
            docking_reward = PocketDockingReward(seed=effective_seed, **reward_options)
            reward = replace(reward, r_vina=docking_reward, w_vina=reward_weight,
                             vina_invert=True, w_vina_proxy=0.0)
            docking_reward_report.update(status="configured", applied=False, weight=reward_weight,
                                         energy_feedback_enabled=reward_weight > 0,
                                         transform="weighted negative measured kcal/mol; no reference energy input")
        except Exception as exc:
            # Keep independent chemistry search available, but never label this
            # failed setup as pocket-conditioned execution.
            docking_reward_report.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
    def record_docking_reward():
        if docking_reward is not None:
            execution = docking_reward.report()
            measured = execution.get("n_docked", 0) > 0
            docking_reward_report.update(execution=execution,
                                         applied=measured and docking_reward_report["energy_feedback_enabled"],
                                         status="executed" if measured else "no_measured_rewards")

    search_class = guidance.GuidedMCTS if prior_enabled else MCTSProofSearch
    search = search_class(
        tile_library=tile_terms,
        rules=rules,
        target_predicates=[LIPINSKI],
        binding_site=PROTEASE_GENERIC,
        reward=reward,
        n_simulations=n_simulations,
        c_puct=1.4,
        # R4-D (r0 fix): ``top_k`` controls the n_candidates/pocket
        # column.  Lifted from the Phase-0 default of 5 to 20 (the
        # lower edge of the SOTA 50-100 candidates/pocket band).
        # With ``max_depth=3`` below this gives ~20 leaves per pocket
        # after the LIPINSKI + binding filter; bump to 50-100 in a
        # follow-up to fully match the SOTA evaluation protocol.
        top_k=top_k,
        early_stop=early_stop,
        patience=patience,
        prior=learned_prior,
        # Sweep refits operate on explicit replay state at pocket boundaries.
        # Disable the unrelated in-MCTS simulation-interval refit path.
        prior_refit_every=0,
        rng=rng,
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        # TODO-03 — wire L-3 fragment pool into MCTS expansion so
        # the branching factor grows from |rules| × 12 = 60 (Phase-0)
        # to |rules| × 204 = 1020 when the 204-tile pool loads
        # successfully.  Set ``use_fragment_pool=False`` to fall back
        # to the 12-tile library for backward compat (no 204-tile download).
        # Explicit tile selection/capping must survive MCTS's lazy pool loader.
        use_fragment_pool=bool(use_fragment_pool and include_fragments
                               and tile_library is None and branching_target is None),
    )
    actual_tiles = search._resolve_expand_tile_pool()
    search_config = {
        "n_simulations": n_simulations, "max_depth": max_depth, "top_k": top_k,
        "early_stop": early_stop, "patience": patience,
        "tile_library": selected_library, "click_rules": click_rules,
        "rule_names": list(rules), "tile_count": len(actual_tiles),
        "branching_target": branching_target,
        "branching_attempts": len(rules) * len(actual_tiles),
        "use_fragment_pool": search.use_fragment_pool,
        "seed": effective_seed,
        "seed_strategy": seed_strategy,
        "not_applied": [name for name, value in {
            "symbolic_prior": symbolic_prior,
            "synthesis_oracle": synthesis_oracle,
            "symbolic_prior_refit_every": symbolic_prior_refit_every,
        }.items() if value],
        "disabled": [name for name, value in {
            "symbolic_prior": symbolic_prior, "synthesis_oracle": synthesis_oracle,
            "symbolic_prior_refit_every": symbolic_prior_refit_every,
        }.items() if not value],
        "unsupported_requested": {
            "symbolic_prior": symbolic_prior,
            "synthesis_oracle": synthesis_oracle,
            "symbolic_prior_refit_every": symbolic_prior_refit_every,
        },
    }
    if prior_enabled:
        search_config["not_applied"] = [key for key in search_config["not_applied"]
                                        if key not in {"symbolic_prior", "symbolic_prior_refit_every"}]
        if learned_prior is None:
            search_config["not_applied"].append("symbolic_prior")
        if prior_mode == "frozen" and symbolic_prior_refit_every is not None:
            search_config["not_applied"].append("symbolic_prior_refit_every")
    if synthesis_report["applied"]:
        search_config["not_applied"] = [key for key in search_config["not_applied"]
                                        if key != "synthesis_oracle"]
    search_config["unsupported_requested"] = {
        key: value for key, value in search_config["unsupported_requested"].items()
        if key in search_config["not_applied"]}
    search_config["guidance"] = {"prior": prior_report, "synthesis": synthesis_report,
                                 "docking_reward": docking_reward_report}

    if seed_strategy == "click_tile":
        try:
            seed, chosen_tile = _choose_click_seed(seed_tile_terms, actual_tiles, rules,
                                                   random.Random(effective_seed), selected_library)
            seed_smiles = _smi_of(seed)
        except ValueError as exc:
            return {"pocket_id": pocket_id, "status": "initialization_fail",
                    "prior_state": replay_state, "prior_report": prior_report,
                    "synthesis_report": synthesis_report,
                    "init_strategy": seed_strategy, "chosen_tile": None,
                    "canonical_seed": None, "seed_smiles": None,
                    "ligand_path": ligand_path, "search_config": search_config,
                    "error": str(exc), "n_candidates": 0, "candidates": [],
                    "has_generated_candidates": False, "n_generated_candidates": 0}
    seed_key = _structure_key(seed_smiles)
    initialization_seconds = time.monotonic() - init_started
    observed, observed_states = _observe_search(search, seed_key)
    t0 = time.time()
    try:
        candidates = search.search(seed, max_depth=max_depth)
    except Exception as e:
        record_docking_reward()
        return {
            "pocket_id": pocket_id,
            "status": f"search_fail: {type(e).__name__}",
            "prior_state": replay_state, "prior_report": prior_report,
            "synthesis_report": synthesis_report,
            "seed_smiles": seed_smiles,
            "n_candidates": 0,
            "candidates": [],
            "wall_seconds": time.time() - t0,
            "search_config": search_config,
            "ligand_path": ligand_path,
            "init_strategy": seed_strategy,
            "chosen_tile": chosen_tile,
            "canonical_seed": seed_key,
        }
    wall = time.time() - t0
    record_docking_reward()
    prior_report["puct_calls"] = getattr(search, "learned_prior_calls", 0)
    if prior_enabled:
        observations = (guidance.candidate_observations(candidates, search, pocket_id)
                        if prior_mode != "frozen" else [])
        replay_state, update_info = guidance.update_prior(
            replay_state, observations, mode=prior_mode, data_split=prior_data_split,
            refit_every=symbolic_prior_refit_every if symbolic_prior_refit_every is not None else 5)
        prior_report.update(update_info)
    pre_synthesis_candidates = list(candidates)
    candidates, synthesis_report = guidance.gate_candidates(pre_synthesis_candidates, synthesis_checker, synthesis_report)
    search_config["guidance"]["synthesis"] = synthesis_report
    wall = time.time() - t0

    # Generation denominators precede synthesis selection. Rejected products
    # remain auditable even though only retained top-k proceed downstream.
    synthesis_rows = synthesis_report.get("reports", [])
    retained_indices = [i for i in range(len(pre_synthesis_candidates))
                        if not synthesis_rows or synthesis_rows[i]["passed"]]
    selected_indices = set(retained_indices[:top_k])
    all_cand_records: List[Dict[str, object]] = []
    for index, c in enumerate(pre_synthesis_candidates):
        smi = _smi_of(c)
        structure_key = _structure_key(smi)
        synthesis_row = synthesis_rows[index] if synthesis_rows else {
            "status": "not_applied", "passed": None}
        all_cand_records.append(
            {
                "candidate_id": index,
                "smiles": smi,
                "is_seed": structure_key is not None and structure_key == seed_key,
                "is_generated": structure_key is not None and seed_key is not None and structure_key != seed_key,
                "sa": _sa_score(smi),
                "qed": _qed(smi),
                "lipinski": _lipinski_pass(smi),
                "vina_proxy": float(_vina_proxy(c)),
                "synthesis": synthesis_row,
                "synthesis_passed": synthesis_row["passed"],
                "retained_after_synthesis": index in retained_indices,
                "selected_for_evaluation": index in selected_indices,
                "physical_evaluation_status": "pending_downstream" if index in selected_indices else "not_selected",
            }
        )

    cand_records = [c for c in all_cand_records if c["selected_for_evaluation"]]
    generated = [c for c in all_cand_records if c["is_generated"]]
    retained_generated = [c for c in generated if c["retained_after_synthesis"]]
    n_seed = sum(c["is_seed"] for c in all_cand_records)
    reasons = []
    if not generated:
        if observed["reduction_calls"] and not observed["products_returned"]:
            reasons.append("No products from attempted reductions (non-matching rules or invalid reductions)")
        if observed_states["typed_rejected"]:
            reasons.append("New structures were rejected by existing type predicates")
        if observed_states["binding_rejected"]:
            reasons.append("New structures were rejected by the existing binding gate")
        if not reasons:
            reasons.append("No new structure in returned top-k; available observations do not identify a specific gate")
    return {
        "pocket_id": pocket_id,
        "status": "ok",
        "seed_smiles": seed_smiles,
        "ligand_path": ligand_path,
        "search_config": search_config,
        "init_strategy": seed_strategy,
        "canonical_seed": seed_key,
        "chosen_tile": chosen_tile,
        "prior_state": replay_state,
        "prior_report": prior_report,
        "synthesis_report": synthesis_report,
        "docking_reward_report": docking_reward_report,
        "initialization_seconds": initialization_seconds,
        "metric_backend": {"sa": _sa_backend(), "qed": "rdkit",
                           "vina": "heuristic_proxy_no_docking"},
        "empty_candidates_reason": (None if cand_records else
                                    "No search candidates passed the configured search/filter path"),
        "search_diagnostics": {"nfe": search.nfe,
                               "nfe_reductions": search.nfe_reductions,
                               "nfe_oracle": search.nfe_oracle,
                               "simulations_completed": len(search.history),
                               "early_stopped": bool(search.history and
                                   search.history[-1].get("EARLY_STOPPED", False))},
        "n_sims": n_simulations,
        "n_candidates": len(cand_records),
        "n_pre_synthesis_candidates": len(all_cand_records),
        "n_retained_candidates": len(retained_indices),
        "n_selected_candidates": len(cand_records),
        "n_evaluated_candidates": 0,
        "n_synthesis_rejected_candidates": len(all_cand_records) - len(retained_indices),
        "n_generated_retained_candidates": len(retained_generated),
        "n_generated_selected_candidates": sum(c["is_generated"] for c in cand_records),
        "n_generated_synthesis_rejected_candidates": len(generated) - len(retained_generated),
        "n_generated_candidates": len(generated),
        "n_seed_candidates": n_seed,
        "n_invalid_candidates": len(all_cand_records) - len(generated) - n_seed,
        "n_unique_generated": len({_structure_key(c["smiles"]) for c in generated}),
        "has_generated_candidates": bool(generated),
        "generation_status": ("generated" if generated else "seed_only" if n_seed
                              else "invalid_candidates" if all_cand_records else "no_candidates"),
        "candidate_accounting": {
            "generated_scope": "all search-returned candidates before synthesis filtering and top-k selection",
            "candidates_scope": "retained candidates selected for downstream evaluation",
            "all_candidates_scope": "all search-returned candidates including synthesis rejections",
            "evaluated_scope": "physical evaluation; not performed by this search runner",
        },
        "generation_diagnostics": {
            **observed,
            "unique_new_products_observed": len(observed_states["new_products"]),
            "unique_new_typed_rejections": len(observed_states["typed_rejected"]),
            "unique_new_binding_rejections": len(observed_states["binding_rejected"]),
            "no_generated_reasons": reasons,
            "scope": "all search calls; rejection sets may overlap; not final-leaf counts",
        },
        "wall_seconds": wall,
        "candidates": cand_records,
        "all_candidates": all_cand_records,
        "history_tail": search.history[-3:] if hasattr(search, "history") else [],
    }


# ---------------------------------------------------------------------------
# Aggregation + IO
# ---------------------------------------------------------------------------


def aggregate(results: List[Dict[str, object]]) -> Dict[str, object]:
    ok = [r for r in results if r.get("status") == "ok"]
    n_ok = len(ok)
    n_total = len(results)
    if n_ok == 0:
        return {"n_pockets_total": n_total, "n_pockets_ok": 0}
    sa_vals = [c["sa"] for r in ok for c in r["candidates"] if isinstance(c.get("sa"), float)]
    qed_vals = [c["qed"] for r in ok for c in r["candidates"] if isinstance(c.get("qed"), float)]
    lip_pass = sum(1 for r in ok for c in r["candidates"] if c.get("lipinski"))
    n_total_cands = sum(r["n_candidates"] for r in ok)
    mean_wall = sum(r["wall_seconds"] for r in ok) / n_ok
    return {
        "n_pockets_total": n_total,
        "n_pockets_ok": n_ok,
        "n_pockets_fail": n_total - n_ok,
        "n_candidates_total": n_total_cands,
        "mean_candidates_per_pocket": n_total_cands / n_ok if n_ok else 0,
        "lipinski_pass_count": lip_pass,
        "lipinski_pass_rate": lip_pass / n_total_cands if n_total_cands else 0,
        "sa_mean": sum(sa_vals) / len(sa_vals) if sa_vals else float("nan"),
        "sa_min": min(sa_vals) if sa_vals else float("nan"),
        "sa_max": max(sa_vals) if sa_vals else float("nan"),
        "qed_mean": sum(qed_vals) / len(qed_vals) if qed_vals else float("nan"),
        "wall_seconds_mean_per_pocket": mean_wall,
    }


def write_csv(results: List[Dict[str, object]], path: str) -> None:
    fields = [
        "pocket_id",
        "status",
        "seed_smiles",
        "n_sims",
        "n_candidates",
        "wall_seconds",
        "top1_smiles",
        "top1_sa",
        "top1_qed",
        "top1_lipinski",
        "sa_mean",
        "qed_mean",
        "lipinski_pass_count",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            cands = r.get("candidates") or []
            top1 = cands[0] if cands else {}
            sa_vals = [c["sa"] for c in cands if isinstance(c.get("sa"), float)]
            qed_vals = [c["qed"] for c in cands if isinstance(c.get("qed"), float)]
            w.writerow(
                {
                    "pocket_id": r.get("pocket_id"),
                    "status": r.get("status"),
                    "seed_smiles": r.get("seed_smiles"),
                    "n_sims": r.get("n_sims"),
                    "n_candidates": r.get("n_candidates"),
                    "wall_seconds": r.get("wall_seconds"),
                    "top1_smiles": top1.get("smiles"),
                    "top1_sa": top1.get("sa"),
                    "top1_qed": top1.get("qed"),
                    "top1_lipinski": top1.get("lipinski"),
                    "sa_mean": (sum(sa_vals) / len(sa_vals)) if sa_vals else None,
                    "qed_mean": (sum(qed_vals) / len(qed_vals)) if qed_vals else None,
                    "lipinski_pass_count": sum(1 for c in cands if c.get("lipinski")),
                }
            )


def write_json(results: List[Dict[str, object]], summary: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"summary": summary, "per_pocket": results}, f, indent=2, default=str)


def write_markdown(summary: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# R4-C: Lambda 100-pocket CrossDocked sweep",
        "",
        f"- n_pockets_total: {summary.get('n_pockets_total')}",
        f"- n_pockets_ok: {summary.get('n_pockets_ok')}",
        f"- n_pockets_fail: {summary.get('n_pockets_fail')}",
        f"- n_candidates_total: {summary.get('n_candidates_total')}",
        f"- mean_candidates_per_pocket: {summary.get('mean_candidates_per_pocket', 0):.2f}",
        f"- lipinski_pass_rate: {summary.get('lipinski_pass_rate', 0):.3f}",
        f"- sa_mean (1-10, lower=easier): {summary.get('sa_mean', float('nan')):.3f}",
        f"- qed_mean (0-1, higher=better): {summary.get('qed_mean', float('nan')):.3f}",
        f"- wall_seconds_mean_per_pocket: {summary.get('wall_seconds_mean_per_pocket', 0):.2f}",
        "",
        "## SOTA comparison (Vina Dock + Success rate, CrossDocked2020 100 pockets)",
        "",
        "| Method | Year | Vina Dock (median, kcal/mol) | Success rate |",
        "|---|---:|---:|---:|",
        "| Pocket2Mol | 2022 | -7.15 | 24.4% |",
        "| TargetDiff | 2023 | -7.80 | 10.5% |",
        "| DecompDiff | 2024 | -8.39 | 24.5% |",
        "| MolCRAFT | 2024 | -9.25 | 36.1% |",
        "| AlphaDrug | 2025 | -9.77 | - |",
        "| MolChord | 2025 | -7.62 | 33.2% |",
        "| **TransDiffSBDD** | 2025 | **-9.37** | **83.9%** |",
        "| **Lambda (R4-C)** | 2026 | **see CSV** | **see CSV** |",
        "",
        "*Note*: Round-1 Lambda (1h36 single pocket) reported Vina -5.923; this sweep is the protocol-aligned 100-pocket follow-up.",
        "*Vina proxy is a placeholder until L-1 DiffDock/FlowDock binary is installed.*",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pockets", type=int, default=10, help="Number of pockets (max 100)")
    p.add_argument("--pocket-offset", type=int, default=0, help="Skip this many pockets (for chunked runs)")
    p.add_argument("--n-simulations", type=int, default=200, help="MCTS sims per pocket")
    p.add_argument("--max-depth", type=int, default=3, help="MCTS max depth")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--tile-library", choices=["standard_12", "extended_204"])
    p.add_argument("--click-rules", help="all_5 or comma-separated registered names")
    p.add_argument("--branching-target", type=int,
                   help="Maximum rule/tile attempts per expansion (not successful products)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-strategy", choices=["reference", "click_tile"], default="reference")
    p.add_argument(
        "--use-fragment-pool",
        dest="use_fragment_pool",
        action="store_true",
        default=True,
        help="Use the L-3 204-tile FRAGMENT_LIBRARY_200_TILES pool (default ON; branching 5*204=1020)",
    )
    p.add_argument(
        "--no-fragment-pool",
        dest="use_fragment_pool",
        action="store_false",
        help="Disable the L-3 204-tile pool — fall back to the 12-tile STANDARD library (branching 5*12=60)",
    )
    p.add_argument(
        "--no-fragments",
        dest="use_fragment_pool",
        action="store_false",
        help="Deprecated alias for --no-fragment-pool (backward compat)",
    )
    p.add_argument("--out-prefix", type=str, default="r4_c_100pocket_sweep", help="Output file prefix")
    p.add_argument("--append", action="store_true", help="Append to existing CSV/JSON instead of overwriting")
    p.add_argument("--detach", action="store_true", help="Run as a detached background process (returns immediately)")
    p.add_argument("--detach-pidfile", type=str, default="/tmp/r4_c_2.pid", help="Where to write pid when --detach")
    args = p.parse_args()

    if args.detach:
        _detach_self(args)
        return

    pockets = list_crossdocked_pockets(offset=args.pocket_offset, count=args.n_pockets)
    print(f"[r4_c] offset={args.pocket_offset}, count={args.n_pockets}, found={len(pockets)}")
    if not pockets:
        print("[r4_c] No pockets found at", CROSSDOCKED_ROOT)
        sys.exit(1)

    include_fragments = bool(args.use_fragment_pool)
    use_fragment_pool = bool(args.use_fragment_pool)
    if include_fragments:
        # Branching factor: 5 rules * 204 tiles = 1020 when the L-3
        # pool loads; print so the operator can see the search-space
        # multiplier before any pocket runs.
        try:
            from molmetal_lam.reactions.beta_reductions import REACTION_RULES as _RR
            n_rules = len(dict(_RR))
        except Exception:
            n_rules = 5
        branching = n_rules * 204
        print(
            f"[r4_c] Using 204-tile L-3 fragment pool "
            f"(branching factor = {n_rules} rules x 204 tiles = {branching})"
        )
    else:
        print("[r4_c] Using STANDARD_12_TILES (12-tile fallback, branching = 5 x 12 = 60)")

    out_dir = os.path.abspath(os.path.join(HERE, "..", "reports"))
    csv_path = os.path.join(out_dir, args.out_prefix + ".csv")
    json_path = os.path.join(out_dir, args.out_prefix + ".json")
    md_path = os.path.join(out_dir, args.out_prefix + ".md")

    # Optionally load prior partial results (chunked run support).
    # We keep the prior results for pockets that are NOT in the current
    # chunk (so chunk N keeps all the unique pockets accumulated so far).
    # Pockets already in this chunk's range are re-run (cheap, idempotent).
    existing_results: List[Dict[str, object]] = []
    if args.append and os.path.isfile(json_path):
        try:
            with open(json_path) as f:
                prev = json.load(f)
            prior_per_pocket = prev.get("per_pocket", [])
            current_ids = {os.path.basename(p) for p in pockets}
            existing_results = [
                r for r in prior_per_pocket
                if r.get("pocket_id") not in current_ids
            ]
            print(f"[r4_c] Append mode: keeping {len(existing_results)} prior results outside this chunk")
        except Exception as e:
            print(f"[r4_c] Could not read prior json: {e}")

    results = list(existing_results)
    t_total = time.time()
    for i, pdir in enumerate(pockets):
        idx_global = args.pocket_offset + i
        print(f"[r4_c] [{idx_global+1}/{args.pocket_offset + len(pockets)}] {os.path.basename(pdir)}")
        r = run_one_pocket(
            pdir,
            n_simulations=args.n_simulations,
            max_depth=args.max_depth,
            seed_offset=idx_global,
            include_fragments=include_fragments,
            use_fragment_pool=use_fragment_pool,
            top_k=args.top_k,
            early_stop=args.early_stop,
            patience=args.patience,
            tile_library=args.tile_library,
            click_rules=args.click_rules,
            branching_target=args.branching_target,
            seed=args.seed,
            seed_strategy=args.seed_strategy,
        )
        results.append(r)
        # Periodic flush so partial results survive a crash
        if (i + 1) % max(1, len(pockets) // 5) == 0:
            summary = aggregate(results)
            write_csv(results, csv_path)
            write_json(results, summary, json_path)
            print(f"[r4_c] checkpoint @ {idx_global+1}/{args.pocket_offset + len(pockets)} pockets — {summary}")

    summary = aggregate(results)
    write_csv(results, csv_path)
    write_json(results, summary, json_path)
    write_markdown(summary, md_path)

    total_wall = time.time() - t_total
    print(f"[r4_c] DONE in {total_wall:.1f}s")
    print(f"[r4_c] Summary: {summary}")
    print(f"[r4_c] Wrote: {csv_path}")
    print(f"[r4_c] Wrote: {json_path}")
    print(f"[r4_c] Wrote: {md_path}")


def _detach_self(args):
    """Re-launch the same sweep as a fully detached daemon process.

    Uses os.fork + os.setsid + redirect stdio so the new process is
    detached from the current session and survives shell-tool SIGTERM
    signals. Writes the new pid to args.detach_pidfile.
    """
    log_path = os.path.abspath(args.detach_pidfile + ".log")
    pid = os.fork()
    if pid > 0:
        # Parent — write child pid, return immediately
        with open(args.detach_pidfile, "w") as f:
            f.write(str(pid) + "\n")
        print(f"[r4_c] Detached child pid={pid}, log={log_path}, pidfile={args.detach_pidfile}")
        return
    # Child
    os.setsid()
    # Re-fork to ensure we're not a session leader (fully daemonized)
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)
    # Grandchild — actually run the sweep
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout
    sys.stdin = open(os.devnull, "r")
    # Reconstruct argv without --detach to avoid re-forking
    argv = [
        sys.executable, os.path.abspath(__file__),
        "--n-pockets", str(args.n_pockets),
        "--pocket-offset", str(args.pocket_offset),
        "--n-simulations", str(args.n_simulations),
        "--max-depth", str(args.max_depth),
        "--top-k", str(args.top_k),
        "--seed-strategy", args.seed_strategy,
        "--patience", str(args.patience),
        "--early-stop" if args.early_stop else "--no-early-stop",
        "--out-prefix", args.out_prefix,
    ]
    for flag, value in (("--tile-library", args.tile_library),
                        ("--click-rules", args.click_rules),
                        ("--branching-target", args.branching_target),
                        ("--seed", args.seed)):
        if value is not None:
            argv.extend([flag, str(value)])
    if args.append:
        argv.append("--append")
    if not args.use_fragment_pool:
        argv.append("--no-fragment-pool")
    os.execvp(sys.executable, argv)


if __name__ == "__main__":
    main()
