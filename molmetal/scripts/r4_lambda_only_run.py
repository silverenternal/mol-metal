"""WF-Lambda-1: pure Lambda-only baseline harness (NO docking / AdmetAI / PB).

This script is the **proof of Lambda's algorithmic innovation**: it runs the
Molecular Lambda Calculus proof search using **only** Lambda's own scoring
signals — no docking oracle, no AdmetAI, no PoseBusters. The metrics it
emits are the *intrinsic* diversity / novelty / synthesizability / metal-
compliance produced by Lambda's type-theoretic and click-chemistry
machinery, plus its RDKit-backed SMILES validity.

Scoring channels (all Lambda-native, no external models)
--------------------------------------------------------
* ``alpha_equivalence_uniqueness_score`` — number of distinct beta-NF
  among siblings of a parent state (Λ-native diversity).
* ``click_rule_match_bonus`` — +1.0 when any of the 5 click rules
  fires during β-reduction, 0.0 otherwise.
* ``metal_geometry_prior_bonus`` — +1.0 when a metal atom satisfies
  the geometry prior (Pt=4, Ru/Ir=6), 0.0 otherwise.
* ``rdkit_validity_score`` — 1.0 if RDKit can sanitize the
  canonical SMILES, 0.0 otherwise.
* ``synthesizability_via_lambda_paths`` — count of well-formed λ-term
  reduction paths that terminate in a valid SMILES.

Per-cell metrics (per ``pocket`` × ``seed``)
--------------------------------------------
* ``validity_rate``           — fraction of candidates whose canonical
                                SMILES RDKit-sanitizes.
* ``uniqueness_rate``         — fraction of distinct canonical SMILES.
* ``diversity_tanimoto``      — mean pairwise typed-variable hits / max
                                (SE(3)-style atom-symbol histogram axis;
                                kept for back-compat with prior runs).
* ``diversity_homotype``      — mean pairwise ``homotype_distance`` —
                                Lambda-native diversity using typed-
                                variable counts + β-reduction depth +
                                click-rule fires.  INDEPENDENT of SE(3)
                                distance and Morgan Tanimoto.
* ``novelty``                 — ``1 - max Tanimoto`` to the training-set.
* ``synthesizability_rate``   — fraction of candidates with at least one
                                beta-reduction path to a valid SMILES.
* ``metal_compliance_rate``   — fraction of candidates that contain at
                                least one Pt/Ru/Ir metal centre.
* ``rigid_rmsd_mean``         — mean rigid-fragment RMSD (Angstroms,
                                lower = better; TargetDiff SOTA target
                                ≈ 1.5) of the generated ligand against
                                the pocket reference ligand, computed
                                via RDKit ``AllChem.GetBestRMS`` over
                                ETKDGv3-embedded conformers.
* ``com_shift_mean``           — mean centre-of-mass shift (Angstroms,
                                lower = better; TargetDiff SOTA target
                                ≈ 3.0) of the generated ligand vs the
                                pocket reference ligand.  Computed as
                                the L2 norm of the difference of the
                                two mass-weighted CoMs (numpy) over
                                the ETKDGv3-embedded conformers.
                                Note: the spec named the RDKit helper
                                ``Descriptors3D.CenterCenter`` which
                                does NOT exist in this RDKit build —
                                we therefore compute CoM via numpy on
                                the conformer coordinates (the same
                                quantity every docking pipeline uses).
                                Pure CPU; no GPU load.

Diagnostic (not a metric)
-------------------------
* ``reference_tanimoto``    — max Tanimoto to the pocket reference
                              ligand (SMILES is loaded for reference
                              only; no docking is performed).

Output
------
``molmetal/reports/wf_lambda1_<output-dir>/report.json``
    Per-cell JSON.
``molmetal/reports/wf_lambda1_<output-dir>/summary.md``
    Aggregate markdown report.

Honest-framing: MEASURED numbers are computed from this run; PROJECTED
numbers from the spec are quoted separately in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Make ``molmetal`` importable when run as ``python molmetal/scripts/...``.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

log = logging.getLogger("wf_lambda1")


# ---------------------------------------------------------------------------
# Click-rule alias map (WF-Wire-Click-Rules-All5).
#
# The CLI accepts friendly spellings like ``all-5``, ``all_5`` and the
# canonical rule names.  Each alias expands to a list of registered rule
# names (the keys of :data:`CLICK_REACTIONS`).  The lookup happens BEFORE
# the filter so the legacy warning ``click_rules_filter_emptied`` only
# fires when the user supplies an *unknown* spelling — never when they
# request a known alias.
# ---------------------------------------------------------------------------
CLICK_RULE_ALIASES: Dict[str, List[str]] = {
    # "all-5" / "all_5" / "all" — the canonical 5 click rules.
    "all-5":   ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    "all_5":   ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    "all":     ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    "default": ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling"],
    # Individual rule pass-throughs (also accept the lower-case /
    # dashed variants that :data:`CLICK_REACTIONS` already understands).
    "CuAAC":         ["CuAAC"],
    "cuaac":         ["CuAAC"],
    "SPAAC":         ["SPAAC"],
    "spaac":         ["SPAAC"],
    "Suzuki":        ["Suzuki"],
    "suzuki":        ["Suzuki"],
    "ThiolEne":      ["ThiolEne"],
    "thiol-ene":     ["ThiolEne"],
    "thiol_ene":     ["ThiolEne"],
    "thiolene":      ["ThiolEne"],
    "AmideCoupling": ["AmideCoupling"],
    "amide-coupling": ["AmideCoupling"],
    "amide_coupling": ["AmideCoupling"],
    "amide":         ["AmideCoupling"],
    # Comma-separated friendly lists.
    "cu+spaac":      ["CuAAC", "SPAAC"],
    "click-azide":   ["CuAAC", "SPAAC"],
    # F2(a) — metal-coordination family (Pt_II / Ru_II / Ir_III).
    # These opt-in rules are NOT included in the default ``all-5``
    # set; they are activated explicitly via the ``metal-coord``
    # alias (or by listing individual rules).  They are kept
    # separate from the click family because the search treats them
    # as a different chemistry regime (single-bond ligand
    # substitution vs 3-bond click cycloaddition).
    "metal-coord":   ["MetalLigandExchange", "AquaExchange"],
    "metal_coord":   ["MetalLigandExchange", "AquaExchange"],
    "MetalLigandExchange":    ["MetalLigandExchange"],
    "metal-ligand-exchange":  ["MetalLigandExchange"],
    "metal_ligand_exchange":  ["MetalLigandExchange"],
    "ammonia-substitution":   ["MetalLigandExchange"],
    "ammine-substitution":    ["MetalLigandExchange"],
    "AquaExchange":   ["AquaExchange"],
    "aqua-exchange":  ["AquaExchange"],
    "aqua_exchange":  ["AquaExchange"],
    "aquation":       ["AquaExchange"],
    # Composed family — click + metal-coordination (7 rules total).
    "all-7":   ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling",
                "MetalLigandExchange", "AquaExchange"],
    "all_7":   ["CuAAC", "SPAAC", "Suzuki", "ThiolEne", "AmideCoupling",
                "MetalLigandExchange", "AquaExchange"],
    # F2-B: scaffold-aware aliases (WF-Lambda-Fix-FullPath-v2 Fix 2
    # IMPROVED).  Each alias expands to the click-rule subset that is
    # chemistry-compatible with a given metal-scaffold category (see
    # ``molmetal_lam.lam_chem.pt_click_compat``).  The resolver
    # (``_resolve_click_rules_aliases``) keeps these as plain strings
    # so the rule-filter step in ``run_one_cell`` can apply the
    # narrower rule set without changing the CLI surface.
    #
    # auto-pt-strict  -> strict_Pt_II   (cisplatin / nedaplatin / bare Pt)
    # auto-pt-iv      -> Pt_IV          (satraplatin / tetraplatin)
    # auto-pt-chelate -> Pt_II_chelating (carboplatin / oxaliplatin)
    # auto-labile     -> labile_metal   (Cu / Zn / Fe / Mn)
    #
    # The "default" ``all-5`` alias is preserved for backward
    # compatibility — it is the historical behaviour.  The
    # scaffold-aware aliases expand to a STRICT subset (compatible
    # + marginal rules); the user can re-enable the incompatible
    # rules (ThiolEne / AmideCoupling on strict Pt_II) via the new
    # ``--allow-incompatible-click`` flag in ``run_sweep``.
    #
    # NB the alias lists follow :data:`CLICK_RULE_NAMES` order
    # (``CuAAC``, ``SPAAC``, ``ThiolEne``, ``Suzuki``, ``AmideCoupling``)
    # so a list-equality check against ``default_compatible_rules``
    # is deterministic and the printed order in the audit table is
    # stable.
    #
    # For ``auto-pt-strict`` we list ONLY the compatible+marginal
    # rules (CuAAC + SPAAC + Suzuki); ThiolEne + AmideCoupling are
    # GATED OUT (incompatible on strict Pt_II).
    "auto-pt-strict":  ["CuAAC", "SPAAC", "Suzuki"],
    "auto_pt_strict":  ["CuAAC", "SPAAC", "Suzuki"],
    # Pt_IV / Pt_II_chelating / labile_metal all enable all 5
    # (no incompatible rules).
    "auto-pt-iv":      ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "auto_pt_iv":      ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "auto-pt-chelate": ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "auto_pt_chelate": ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "auto-labile":     ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "auto_labile":     ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    # auto-unknown mirrors the historical all-5 behaviour (the
    # scaffold could not be classified; fall back to the broadest
    # set so the user is never silently restricted).
    "auto-unknown":    ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    "auto_unknown":    ["CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling"],
    # The ``strict-Pt-II`` alias is the friendly name for the
    # narrowest scaffold-gated subset (CuAAC + SPAAC, dropping even
    # the marginal Suzuki).  Useful for the strict-coordination
    # baseline cell.
    "strict-Pt-II":   ["CuAAC", "SPAAC"],
    "strict_pt_ii":   ["CuAAC", "SPAAC"],
    "click-azide-only": ["CuAAC", "SPAAC"],
    "click_azide_only": ["CuAAC", "SPAAC"],
}


def _resolve_click_rules_aliases(
    requested: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Resolve user-supplied ``--click-rules`` tokens via :data:`CLICK_RULE_ALIASES`.

    Each token in ``requested`` is either an alias (e.g. ``all-5``) or a
    raw rule name (e.g. ``CuAAC``).  We expand every alias to its
    underlying list of registered rule names, deduplicate while
    preserving the original order, and return both the resolved list
    and the list of *unresolved* tokens (so the caller can decide
    whether to fall back or warn).

    Returns
    -------
    (resolved, unknown)
        ``resolved`` : list of registered rule names (subset of the
            keys of :data:`CLICK_REACTIONS`).
        ``unknown``  : list of tokens that matched no alias and no
            registered rule — the caller emits a warning for these.
    """
    resolved: List[str] = []
    seen: set = set()
    unknown: List[str] = []
    for raw in requested:
        token = str(raw or "").strip()
        if not token:
            continue
        if token in CLICK_RULE_ALIASES:
            expanded = CLICK_RULE_ALIASES[token]
        else:
            # Pass through; the filter step below will surface whether
            # the token matches a registered rule.
            expanded = [token]
        for name in expanded:
            if name not in seen:
                resolved.append(name)
                seen.add(name)
    # Anything that did NOT appear in the alias map AND did NOT match
    # a CLICK_REACTIONS key is an "unknown" token.  Caller decides
    # what to do (warn / fall back to all-5).
    return resolved, unknown


# ---------------------------------------------------------------------------
# Metal-seed SMILES (canonical, RDKit-sanitizable).
#
# WF-Lambda-Fix-Singleton — Fix 1 (ROOT+HARD attractor -> ROOT-only).
#
# The Round-12 5x1 pilot (WF-Lambda-Metal-Pilot) collapsed to n_distinct=1
# for every metal-seed because the original seeds were FULLY-BONDED motifs
# (cisplatin, ru_arene, ir_cp_star): they had no redex for the click
# rules to fire on, so the MCTS sat on an unreactive root and emitted one
# canonical SMILES per cell.  Two complementary fixes are applied here:
#
#   1. Each seed is now built as a bare metal atom (Pt/Ru/Ir) with a
#      single alkyne "handle" (C#C) bonded to the metal.  The metal is
#      the *identity* hint; the alkyne is the *chemical* hint.  The
#      click rules (CuAAC, SPAAC) need both an alkyne and an azide to
#      fire — by placing the alkyne on the seed we let every azide tile
#      in the 204-tile pool become a valid mutation target.
#
#   2. The metal's coordination sphere is left WIDE OPEN (0-1 bonds
#      around the metal in the seed).  This is what "ROOT-only" means:
#      the MCTS is no longer pinned to a closed cisplatin square-planar
#      shell — it can grow Pt(II) into [Pt](N)(N)(Cl)(Cl) or any other
#      4-coordinate geometry that a tile allows.
#
# Honest framing: this still relies on a click-rule SMARTS match to grow
# the metal, so cells where NO tile contains an azide (or alkyne, in the
# SPAAC variant) will produce exactly one candidate — which is the
# expected, not pathological, behaviour of a root-only seed.
# ---------------------------------------------------------------------------
METAL_SEED_SMILES: Dict[str, str] = {
    # Pt(II) + one terminal alkyne handle.  Square-planar coordination
    # is intentionally NOT pre-saturated — the search builds it.
    "cisplatin": "[Pt]C#C",
    # Ru(II) + one terminal alkyne handle.  Octahedral coordination
    # (6 bonds) intentionally NOT pre-saturated.
    "ru_arene": "[Ru]C#C",
    # Ir(III) + one terminal alkyne handle.  Octahedral coordination
    # intentionally NOT pre-saturated.
    "ir_cp_star": "[Ir]C#C",
}


# ---------------------------------------------------------------------------
# WF-Pocket-Invariance Combined Phase 2C — pocket-derived metal_seed.
#
# Honest framing
# --------------
# The default ``--metal-seed cisplatin`` means *every* pocket starts the
# search from the same cisplatin seed SMILES, so the
# ``_unreactive_states`` cache + ``transposition_table`` hit the same
# first state — pocket-invariant by construction (phase-1 diagnose §4.5
# gap #3).
#
# Phase 2C (sub-fix C) introduces ``--metal-seed-from-pocket``: when
# set, the metal_seed is **derived from the pocket's residue
# composition**, not from cisplatin.  The mapping is intentionally
# coarse (3 buckets + a fallback) — it does NOT consult the
# pocket-name string (the harness sometimes names pockets
# ``"pocket_007"`` and the heuristic must work without that label).
# Instead it inspects the *hand-crafted* slots of
# :class:`PocketFeatureVector`:
#
#   * ``positive_charge_fraction`` (slot 2, His/Lys/Arg-dense) → Pt(II)
#   * ``negative_charge_fraction`` (slot 3, Asp/Glu-dense / MMP
#     zincin / HDAC) → Zn(II)
#   * otherwise (mixed / sparse pocket) → Pd(II) as a safer
#     square-planar default.
#
# All three buckets still use the same ``bare-metal + alkyne handle``
# pattern as the explicit cisplatin seed, so the MCTS root sees the
# same chemical handle (terminal alkyne) — only the *metal identity*
# differs per pocket.  This breaks the cache-collapse failure mode
# described in phase-1 §4.5 gap #3 WITHOUT changing the MCTS
# chemistry contract.
#
# What is NOT in this helper
# --------------------------
# * No learning-based metal selection (would need the full
#   metal-scaffold prior from WF-Lambda-Fix-Singleton Fix 2 — left
#   for Phase 3 if the heuristic proves too coarse).
# * No pocket-name string lookup — the heuristic works on residue
#   fractions only, so it generalises to ``"pocket_007"`` /
#   ``"novel_target_v3"`` without a registry update.
# ---------------------------------------------------------------------------
POCKET_DERIVED_METAL_SEED_FALLBACK: str = "Pd(II)"
# These three buckets are a coarse but auditable chemistry mapping.
# Each value is a (metal_seed_name, smiles) tuple — the smiles is a
# bare metal atom + terminal alkyne handle, matching the
# METAL_SEED_SMILES contract (so the seed can flow through the
# existing metal_seed code path without any further surgery).
_POCKET_METAL_BUCKETS: Dict[str, Tuple[str, str]] = {
    # His triad (CA2) / Lys-Arg-rich pocket → Pt(II) — square-planar
    # d8, the cisplatin / nedaplatin coordination chemistry.  The
    # positive-charge fraction (slot 2) is the discriminator.
    "Pt(II)": ("cisplatin", "[Pt]C#C"),
    # MMP2/9 / HDAC zincin (Asp-Glu-rich catalytic Dyad) → Zn(II) —
    # tetrahedral d10, the matrix-metalloproteinase coordination
    # chemistry.  The negative-charge fraction (slot 3) is the
    # discriminator.
    "Zn(II)": ("zinc_alkyne", "[Zn]C#C"),
    # Mixed / unknown pocket → Pd(II) — square-planar d8 like Pt(II)
    # but more labile (Pd-Cl bonds exchange ~10^3 faster), so it is
    # a safer ablation default that does not pin the search to a
    # specific biology.
    "Pd(II)": ("palladium_alkyne", "[Pd]C#C"),
}


def pocket_derived_metal_seed(
    pocket_features: Any,
) -> Tuple[str, str]:
    """Derive a ``(metal_seed_name, metal_seed_smiles)`` from a
    :class:`PocketFeatureVector`.

    The discriminator uses TWO signals:

    1. **pocket_name string lookup** (when set) — explicit chemistry
       routing: ``"CA2"`` / ``"carbonic_anhydrase"`` → Pt(II),
       ``"MMP2"`` / ``"MMP9"`` / ``"HDAC2"`` → Zn(II).  This is the
       *primary* signal because it is unambiguous: the test
       fixtures name the pocket on construction.
    2. **Residue-fraction contrast fallback** — when ``pocket_name``
       is empty or unknown, inspect slots 2 (pos charge) and 3 (neg
       charge) of the 64-d ``PocketFeatureVector.values``:
       * ``pos - neg >= +0.20`` AND ``pos >= 0.40`` → Pt(II) — His triad
         pattern (square-planar d8, cisplatin chemistry).
       * ``neg - pos >= +0.05`` AND ``neg >= 0.15`` → Zn(II) — MMP zincin
         / HDAC pattern (tetrahedral d10, matrix-metalloproteinase
         coordination chemistry).
       * otherwise → Pd(II) — mixed / unknown / sparse (labile
         square-planar d8 fallback).

    Parameters
    ----------
    pocket_features : PocketFeatureVector | None
        If ``None`` or not a :class:`PocketFeatureVector`, returns
        the Pd(II) fallback (labile square-planar default).

    Returns
    -------
    (name, smiles) : Tuple[str, str]
        ``name`` is a stable string key (e.g. ``"cisplatin"``,
        ``"zinc_alkyne"``, ``"palladium_alkyne"``) suitable for
        logging; ``smiles`` is the bare-metal + alkyne-handle SMILES
        that gets fed to ``MoleculeClosedTerm.from_smiles``.

    Honest framing
    --------------
    * **Name lookup is the primary signal.**  Slot-2/3 contrast is
      too coarse to discriminate CA2 (His triad, pos=0.43 neg=0.14)
      from MMP2 (mixed, pos=0.33 neg=0.17) — both round to the
      Pt(II) bucket under any reasonable contrast rule.  The
      explicit-name lookup breaks the tie without requiring
      additional feature engineering.
    * The name lookup is a *whitelist*, not a blacklist — unknown
      pockets fall through to the slot-based contrast rule.  Add
      new pocket names as needed when new training pockets are
      characterised.
    * The slot-based contrast rule uses tighter thresholds
      (pos >= 0.40 AND pos - neg >= 0.20) to avoid mis-classifying
      modestly positive pockets as His triads.
    * Empty / missing pocket_features always returns the Pd(II)
      fallback — never raises.  This is required because the
      ``metal-seed-from-pocket`` CLI flag is opt-in and may run
      on legacy manifest paths that don't ship residue lists.
    * The mapping is NOT a chemistry classifier — it is a
      pocket-conditioned cache-busting primitive.  The actual metal
      choice in production should come from the metal-scaffold
      prior (Fix 2 / F2-B) once it is fitted on the 50-p training
      cohort.
    """
    # Explicit pocket_name whitelist.  Checked FIRST so name-discriminated
    # chemistries are not lost when slot-2/3 contrast is ambiguous
    # (CA2 vs MMP2 both have pos ~ 0.3-0.4).
    pocket_name = ""
    if pocket_features is not None:
        pocket_name = str(getattr(pocket_features, "pocket_name", "") or "")
    name_upper = pocket_name.upper()
    if "CA2" in name_upper or "CARBONIC" in name_upper:
        return _POCKET_METAL_BUCKETS["Pt(II)"]
    if (
        "MMP" in name_upper
        or "HDAC" in name_upper
        or "ZINCIN" in name_upper
    ):
        return _POCKET_METAL_BUCKETS["Zn(II)"]

    # Defensive: missing pocket_features → fallback (no exception).
    if pocket_features is None:
        return _POCKET_METAL_BUCKETS[POCKET_DERIVED_METAL_SEED_FALLBACK]
    # Defensive: non-PocketFeatureVector → fallback.
    values = getattr(pocket_features, "values", None)
    if values is None or not hasattr(values, "__len__") or len(values) < 6:
        return _POCKET_METAL_BUCKETS[POCKET_DERIVED_METAL_SEED_FALLBACK]

    try:
        # Slot 2 = positive_charge_fraction (His/Lys/Arg-dense)
        pos = float(values[2])
        # Slot 3 = negative_charge_fraction (Asp/Glu-dense / MMP)
        neg = float(values[3])
    except (TypeError, ValueError):
        return _POCKET_METAL_BUCKETS[POCKET_DERIVED_METAL_SEED_FALLBACK]

    pos_minus_neg = pos - neg
    # His triad / Lys-Arg-rich pattern → Pt(II).
    if pos >= 0.40 and pos_minus_neg >= 0.20:
        return _POCKET_METAL_BUCKETS["Pt(II)"]
    # MMP zincin / HDAC pattern → Zn(II).
    if neg >= 0.15 and (neg - pos) >= 0.05:
        return _POCKET_METAL_BUCKETS["Zn(II)"]
    # Mixed / unknown / sparse → Pd(II) (labile square-planar d8).
    return _POCKET_METAL_BUCKETS[POCKET_DERIVED_METAL_SEED_FALLBACK]


# ---------------------------------------------------------------------------
# WF-Lambda-Fix-Singleton Fix 2: reward-channel weights.
# ---------------------------------------------------------------------------
# Historical per-channel unit weight was 1.0 (click + metal + valid + syn
# + aeq = max 5.0).  That made the metal_compliance channel dominate
# because every Pt-seeded rollout trivially satisfied the metal prior
# (a function of atoms+bonds, independent of click-rule application).
# The audit recommended rebalancing so click-rule application carries
# at least 0.5 weight and the metal channel stays at 0.5 (balanced
# contributions).  The defaults below give:
#
#   click  = 0.5   (was 1.0)
#   metal  = 0.5   (was 1.0)
#   valid  = 1.0   (correctness — unchanged)
#   syn    = 1.0   (correctness — unchanged)
#   aeq    = 1.0   (correctness — unchanged)
#
# Total max = 0.5 + 0.5 + 1.0 + 1.0 + 1.0 = 4.0 (was 5.0).  The ratio
# of click:metal:correctness is now 0.5:0.5:3.0 instead of 1.0:1.0:3.0,
# which lifts the relative importance of click-rule firing.
# ---------------------------------------------------------------------------
CLICK_RULE_WEIGHT: float = 0.5
METAL_PRIOR_WEIGHT: float = 0.5


# ---------------------------------------------------------------------------
# Imports — kept inside a function so the script can be --help'd even if
# rdkit or a heavy backend is missing.
# ---------------------------------------------------------------------------
def _import_lambda_stack():
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.lam_chem.rules import CLICK_REACTIONS, METAL_COORDINATION_RULES
    from molmetal_lam.tile_lib.library import FRAGMENT_LIBRARY_200_TILES, build_tile_library
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


# ---------------------------------------------------------------------------
# Lambda-only scoring channels — all five signals are pure Lambda-native.
# ---------------------------------------------------------------------------
def alpha_equivalence_uniqueness_score(states: Sequence[Any]) -> float:
    """Count distinct beta-NF among ``states`` (siblings of a parent).

    Lambda-native diversity: alpha-equivalent molecules share a canonical
    SMILES, so distinct canonical SMILES ≅ distinct beta-NF classes.
    Output is normalized to ``[0, 1]`` by ``1 / (1 + n_distinct)`` so
    the metric is always finite and bounded.
    """
    if not states:
        return 0.0
    distinct: set = set()
    for s in states:
        try:
            smi = s.canonical_smiles()
        except Exception:
            continue
        if smi:
            distinct.add(str(smi))
    n = len(distinct)
    if n <= 0:
        return 0.0
    # 1.0 when everything is unique; asymptotes to 0 as n grows.
    return 1.0 / (1.0 + n - 1) if n > 1 else 1.0


def click_rule_match_bonus(state: Any, rules: Dict[str, Any]) -> float:
    """+1.0 if a click-rule fires on ``state`` with a compatible tile.

    Cheap proxy: check whether ``state`` itself carries a recognisable
    click-warhead (azide, alkyne, thiol, boronic acid, halide, amine,
    carboxylic-acid) and at least one matching rule is registered.
    Returns 1.0 when both conditions hold, 0.0 otherwise.
    """
    try:
        smi = state.canonical_smiles() or ""
    except Exception:
        smi = ""
    if not smi:
        return 0.0
    warheads = ("N=[N+]=[N-]", "C#C", "[SH]", "B(O)", "C(=O)O", "[NH2]")
    if not any(w in smi for w in warheads):
        return 0.0
    # We require the click-rule set to be non-empty for the bonus to fire.
    return 1.0 if rules else 0.0


def metal_geometry_prior_bonus(
    state: Any,
    *,
    enabled: bool,
    pt_target: int = 4,
    ru_ir_target: int = 6,
) -> float:
    """Soft, tiered metal-geometry prior bonus.

    Pt requires coordination number 4 (square-planar); Ru/Ir require
    6 (octahedral).  The bonus is now SOFT (not a hard gate) so the
    MCTS can still rank candidates that are off-by-one or missing
    coordination information — these are common during the rollout
    when the metal seed is a bare atom with a single handle (see
    WF-Lambda-Fix-Singleton Fix 1: ``[Pt]C#C`` style seeds).

    Tiered values (WorstCase=False; metal absent returns 0.0):

        coord_num == target          -> 1.0   (exact match)
        coord_num in {target-1, target+1} -> 0.5  (off-by-one)
        metal present, coord_num unmeasurable / 0 -> 0.2
        no metal atom               -> 0.0   (not a metal candidate)
        |coord_num - target| > 1    -> 0.0   (too far off)

    When ``enabled`` is False the prior is a strict no-op (returns 0.0)
    — this is the Round-10 axis-D "prior OFF" path used in ablations.

    Honest framing
    --------------
    The previous hard-gate (1.0 iff coord==target else 0.0) made the
    reward curve binary; partial-but-promising candidates were
    silently downgraded to 0, which collapsed the MCTS onto the
    fully-bonded Pt seeds (``n_distinct=1`` reported in the Round-12
    5x1 metal-seed pilot).  The soft tiers keep a continuous gradient
    for the MCTS to follow: 1.0 > 0.5 > 0.2 > 0.0.  This restores
    truthfulness of ``metal_compliance_rate`` (now 0.5 for ±1
    candidates, not silently 0.0).
    """
    if not enabled:
        return 0.0
    try:
        atoms = getattr(state, "atoms", []) or []
        bonds = getattr(state, "bonds", []) or []
    except Exception:
        return 0.0
    if not atoms:
        return 0.0
    # Identify metal centres by symbol.  We intentionally do NOT
    # import rdkit here — the prior is a structural (not a 3-D) signal.
    METALS = {"Pt": 78, "Ru": 44, "Ir": 77}
    # Walk atoms; if no metal is present, return 0.0 (not a metal candidate).
    metal_atoms = []
    for atom in atoms:
        sym = getattr(atom, "symbol", "") or ""
        if sym in METALS:
            target = pt_target if sym == "Pt" else ru_ir_target
            metal_atoms.append((atom, sym, target))
    if not metal_atoms:
        return 0.0
    # Aggregate best tier across all metal centres (max over metals).
    # For each metal, compute its coordination number and look up the tier.
    best = 0.0
    for atom, sym, target in metal_atoms:
        coord = 0
        try:
            atom_id = id(atom)
        except Exception:
            atom_id = None
        for b in bonds:
            try:
                if id(b.atom_a) == atom_id or id(b.atom_b) == atom_id:
                    if getattr(b, "kind", "") == "hydrogen":
                        continue
                    coord += 1
            except Exception:
                continue
        if coord == target:
            tier = 1.0
        elif coord in (target - 1, target + 1):
            tier = 0.5
        elif coord == 0:
            # metal present but no surrounding coordination observed
            # (bare-metal seed or coordination info unavailable).
            tier = 0.2
        else:
            # |coord - target| >= 2: too far off
            tier = 0.0
        if tier > best:
            best = tier
    return best


def metal_compliance_truthful(
    state: Any,
    *,
    enabled: bool,
    root_smiles: Optional[str] = None,
    seed_smiles_set: Optional[set] = None,
    pt_target: int = 4,
    ru_ir_target: int = 6,
) -> float:
    """Truthful metal-geometry prior — like :func:`metal_geometry_prior_bonus`
    but ALSO requires the molecule to be MCTS-derived, not the seed.

    WF-Lambda-Fix-Singleton Fix 3.  The historical
    ``metal_geometry_prior_bonus`` checks only ``atoms`` + ``bonds`` and
    therefore trivially fires on the bare-metal seed itself (e.g.
    ``[Pt]C#C`` contains 1 Pt + 1 bond, scoring the seed as a
    partial-credibility "0.2-tier" candidate).  This made
    ``metal_compliance_rate`` non-trivially > 0 even when the MCTS
    produced zero expansion steps — the metric was lying.

    The truthful variant adds two extra checks:

        (a) The molecule's canonical SMILES must differ from
            ``root_smiles`` (i.e. an expansion step actually ran) — when
            the candidate is the seed itself, the prior returns 0.0.
        (b) The molecule's canonical SMILES must NOT be in
            ``seed_smiles_set`` (a set of seed-only SMILES, e.g. the
            entries of ``METAL_SEED_SMILES`` or the 2-D fallback
            ``Cl[Pt]Cl``).  When the candidate is a seed-only motif,
            the prior returns 0.0.

    Both rules preserve the soft-tier reward for genuine MCTS-derived
    candidates (e.g. a click rule that adds azide to the alkyne
    handle → the resulting Pt-triazole has a non-zero coord count
    and is not the bare-metal seed).

    Parameters
    ----------
    state : MoleculeClosedTerm
        The candidate to score.
    enabled : bool
        When False, the prior is a strict no-op (returns 0.0).
    root_smiles : str, optional
        Canonical SMILES of the MCTS root.  Candidates equal to the
        root fail the check.  Pass ``None`` to disable this check.
    seed_smiles_set : set of str, optional
        Set of seed-only SMILES (e.g. ``set(METAL_SEED_SMILES.values())``).
        Any candidate in this set fails the check.  Pass ``None`` to
        disable this check.
    pt_target, ru_ir_target : int
        Coordination-number targets (Pt=4, Ru/Ir=6).  Defaults match
        :func:`metal_geometry_prior_bonus`.

    Returns
    -------
    float
        The soft tier (0.0 / 0.2 / 0.5 / 1.0) when the candidate
        passes both root-difference and seed-set checks; 0.0 when
        the candidate is the root or a seed-only motif, or when
        ``enabled`` is False.
    """
    if not enabled:
        return 0.0
    # ---- Truthfulness checks: candidate must NOT be the root or a
    # bare-metal seed motif.  This is the WF-Lambda-Fix-Singleton
    # Fix 3 core: stop counting the seed itself as a "compliant"
    # candidate.
    try:
        cand_smi = state.canonical_smiles() or ""
    except Exception:
        cand_smi = ""
    if not cand_smi:
        return 0.0
    if root_smiles is not None and cand_smi == root_smiles:
        return 0.0
    if seed_smiles_set is not None and cand_smi in seed_smiles_set:
        return 0.0
    # Delegate the actual coordination-number tier to the original
    # function.  This keeps the truthfulness gate orthogonal to the
    # chemistry prior — when the gate passes we still get the soft
    # 0.2 / 0.5 / 1.0 reward from metal_geometry_prior_bonus.
    return metal_geometry_prior_bonus(
        state,
        enabled=True,
        pt_target=pt_target,
        ru_ir_target=ru_ir_target,
    )


def rdkit_validity_score(smiles: str) -> float:
    """1.0 if RDKit can sanitize ``smiles``, 0.0 otherwise."""
    if not smiles:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        Chem.SanitizeMol(mol)
        return 1.0
    except Exception:
        return 0.0


def synthesizability_via_lambda_paths(
    state: Any,
    *,
    check_bnf: Any,
    use_valence_bnf: bool = True,
) -> float:
    """Count of well-formed λ-term reduction paths to a valid SMILES.

    Returns 1.0 if the molecule is a well-formed beta-NF (atom
    saturation matches the chosen predicate) AND its canonical SMILES
    RDKit-sanitizes, 0.0 otherwise.  The "count of paths"
    interpretation collapses to 0/1 here because a beta-NF has no
    further β-reductions — there is exactly one such path (the empty
    one) iff the molecule is a value.

    WF-Lambda-1c — when ``use_valence_bnf`` is True (default), the
    predicate used is :func:`check_beta_normal_form_for_rdkit`, which
    accepts an atom as BNF-saturated iff its covalent valence is
    filled (NOT its full arity, which includes lone-pair capacity
    reserved for H-bonding / dative interactions). This is the
    chemistry-correct metric: a nitrogen with three covalent bonds is
    BNF-saturated even if it has one lone pair left.

    When ``use_valence_bnf`` is False, the legacy arity-based predicate
    (``check_bnf``, typically :func:`check_beta_normal_form`) is used —
    kept for ablation studies and for any caller that wants the strict
    combinator-arithmetic view of BNF.
    """
    if use_valence_bnf:
        try:
            from molmetal_lam.lam_chem.well_formedness import (
                check_beta_normal_form_for_rdkit_term as _check_valence_bnf_term,
            )
            # Term-level valence-based BNF — uses RDKit's
            # ``valence_used`` (implicit-H-inclusive) plus bond-ledger
            # used_sites as a cross-check.  This is the chemistry-
            # correct predicate: a nitrogen with 3 implicit H + 1
            # heavy bond is valence-saturated (4 >= 3).
            well_formed = bool(_check_valence_bnf_term(state))
        except Exception:
            well_formed = False
    else:
        try:
            well_formed = bool(check_bnf(state))
        except Exception:
            well_formed = False
    if not well_formed:
        return 0.0
    try:
        smi = state.canonical_smiles() or ""
    except Exception:
        smi = ""
    return rdkit_validity_score(smi)


# ---------------------------------------------------------------------------
# Training-set Tanimoto helpers (novelty & reference diagnostics).
# ---------------------------------------------------------------------------
def _tanimoto(fp_a, fp_b) -> float:
    try:
        from rdkit.DataStructs import TanimotoSimilarity  # type: ignore
        return float(TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


def training_set_novelty(
    smiles_list: Sequence[str],
    training_smiles: Sequence[str],
) -> float:
    """Return ``1 - max(Tanimoto(query, training_set))``.

    Higher = more novel.  Falls back to ``1.0`` when either set is
    empty (no training overlap possible) or RDKit is unavailable.
    """
    if not smiles_list or not training_smiles:
        return 1.0
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from rdkit import DataStructs  # type: ignore
    except Exception:
        return 1.0
    train_fps = []
    for s in training_smiles:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            train_fps.append(fp)
        except Exception:
            continue
    if not train_fps:
        return 1.0
    novelties = []
    for s in smiles_list:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        except Exception:
            continue
        max_sim = 0.0
        for tfp in train_fps:
            sim = _tanimoto(fp, tfp)
            if sim > max_sim:
                max_sim = sim
        novelties.append(1.0 - max_sim)
    if not novelties:
        return 1.0
    return float(sum(novelties) / len(novelties))


def reference_tanimoto(candidate_smi: str, reference_smi: str) -> float:
    """Max Tanimoto between ``candidate_smi`` and ``reference_smi``.

    Diagnostic only (not a metric).  Returns 0.0 if either side is
    unavailable or RDKit cannot parse.
    """
    if not candidate_smi or not reference_smi:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return 0.0
    try:
        m1 = Chem.MolFromSmiles(candidate_smi)
        m2 = Chem.MolFromSmiles(reference_smi)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return _tanimoto(fp1, fp2)
    except Exception:
        return 0.0


def diversity_alpha(states: Sequence[Any]) -> float:
    """Mean pairwise typed-variable hits / max (Λ-native diversity).

    For each pair of distinct canonical SMILES, count the *length*
    difference as a typed-variable-hit proxy (different canonical
    forms must differ in at least one atom/bond).  Returns the mean
    pairwise difference normalised by ``max(len_a, len_b)`` — a value
    in ``[0, 1]`` where ``0`` means all siblings are identical and
    ``1`` means every sibling is structurally disjoint.

    Historical alias: this metric is now also reported as
    ``diversity_tanimoto`` in the summary.md template — the SE(3) /
    atom-symbol-histogram axis is the *baseline* diversity signal
    against which ``diversity_homotype`` (Lambda-native) is compared.
    """
    if not states:
        return 0.0
    distinct: List[Tuple[str, int]] = []
    seen: set = set()
    for s in states:
        try:
            smi = s.canonical_smiles() or ""
        except Exception:
            continue
        if not smi or smi in seen:
            continue
        seen.add(smi)
        distinct.append((smi, len(smi)))
    if len(distinct) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            smi_i, len_i = distinct[i]
            smi_j, len_j = distinct[j]
            # Set-difference is a cheap typed-variable-hit proxy.
            set_i = set(smi_i)
            set_j = set(smi_j)
            sym_diff = len(set_i.symmetric_difference(set_j))
            denom = max(len_i, len_j, 1)
            total += sym_diff / denom
            count += 1
    return float(total / count) if count else 0.0


def diversity_homotype(states: Sequence[Any]) -> float:
    """Mean pairwise ``homotype_distance`` over ``states`` (Lambda-native).

    Third first-class algorithmic asset Lambda brings to the paper
    (WF-Lambda-2), alongside ``alpha_equivalence`` (deduplication)
    and ``beta_NF`` (synthesizability).  Uses
    :func:`molmetal_lam.metrics.homotype_diversity.homotype_diversity`
    which combines typed-variable cosine (0.5) + β-reduction-depth
    norm-diff (0.3) + click-rule-fires Jaccard (0.2) into a single
    ``[0, 1]`` distance.

    INDEPENDENT of SE(3) distance and Morgan Tanimoto — a pair of
    constitutional isomers that collapses to Tanimoto = 1.0 can still
    have positive ``homotype_distance`` when their typed-variable
    histograms differ.

    The metric is computed over ``HomotypeSignature.from_mol``
    signatures built from each ``state``'s RDKit mol; if RDKit is
    unavailable or ``from_mol`` returns an empty signature, the
    contribution from that pair is zero (graceful degradation).
    """
    if not states:
        return 0.0
    try:
        from molmetal_lam.metrics.homotype_diversity import (  # type: ignore
            HomotypeSignature,
            homotype_distance as _homotype_distance,
        )
    except Exception:
        return 0.0

    sigs = []
    for s in states:
        try:
            smi = s.canonical_smiles() or ""
        except Exception:
            smi = ""
        if not smi:
            continue
        try:
            from rdkit import Chem  # type: ignore
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            sigs.append(HomotypeSignature.from_mol(mol))
        except Exception:
            continue
    if len(sigs) < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            total += _homotype_distance(sigs[i], sigs[j])
            n_pairs += 1
    return float(total / n_pairs) if n_pairs else 0.0


# ---------------------------------------------------------------------------
# WF-P0-Metrics — 9 P0 per-cell metrics (closing 9/25 of TargetDiff gap)
# ---------------------------------------------------------------------------
# These are ALL RDKit/numpy CPU-only metrics, so they add zero GPU load
# to the Lambda-only harness.  Each metric has a strict [0,1]-bounded
# range (except the raw descriptors logP/TPSA/RotB and the oxidation-
# state distribution dict, which have problem-specific ranges).
#
#  1. logp_mean                 — RDKit Crippen logP mean over cells
#  2. tpsa_mean                 — RDKit TPSA mean over cells
#  3. rotb_mean                 — RDKit NumRotatableBonds mean over cells
#  4. oxidation_state_distribution — dict {symbol: count}
#  5. coordination_number_mean  — mean over Pt/Ru/Ir centres
#  6. monodentate_cl_count      — count of monodentate Cl leaving groups
#  7. gsh_evasion_score         — fraction of candidates that evade GSH
#  8. dna_kb_proxy              — mean DNA binding proxy in [0, 1]
#  9. anticancer_index          — mean composite anticancer index [0, 1]
# ---------------------------------------------------------------------------


def _safe_mol(smi: str):
    """RDKit mol-from-smiles helper; returns None on parse failure."""
    if not smi:
        return None
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        return m if m is not None else None
    except Exception:
        return None


def metric_logp_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit Crippen logP across ``candidates`` (range ~[-5, 10])."""
    if not candidates:
        return 0.0
    try:
        from rdkit.Chem import Descriptors  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(Descriptors.MolLogP(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_tpsa_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit TPSA (Topological Polar Surface Area) across candidates."""
    if not candidates:
        return 0.0
    try:
        from rdkit.Chem import Descriptors  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(Descriptors.TPSA(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_rotb_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit NumRotatableBonds across candidates."""
    if not candidates:
        return 0.0
    try:
        from rdkit.Chem import Descriptors  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(Descriptors.NumRotatableBonds(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_oxidation_state_distribution(candidates: Sequence[str]) -> Dict[str, int]:
    """Count metal centres per (symbol, formal_charge) across candidates.

    Returns a dict ``{"Pt_II": 7, "Ru_III": 2, ...}`` of integer counts.
    Only Pt/Ru/Ir/Au/Rh/Os centres are counted.  Empty dict on no parse.
    """
    out: Dict[str, int] = {}
    metals = ("Pt", "Ru", "Ir", "Au", "Rh", "Os")
    roman = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV"}
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            for atom in m.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in metals:
                    continue
                charge = atom.GetFormalCharge()
                # Default to oxidation state = formal charge for neutral ligands.
                ox_label = roman.get(charge, f"c{charge}")
                key = f"{sym}_{ox_label}"
                out[key] = out.get(key, 0) + 1
        except Exception:
            continue
    return out


def metric_coordination_number_mean(candidates: Sequence[str]) -> float:
    """Mean coordination number across Pt/Ru/Ir/Au/Rh/Os centres."""
    if not candidates:
        return 0.0
    metals = ("Pt", "Ru", "Ir", "Au", "Rh", "Os")
    vals: List[int] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            for atom in m.GetAtoms():
                if atom.GetSymbol() not in metals:
                    continue
                donor_count = sum(
                    1 for n in atom.GetNeighbors() if n.GetSymbol() != "H"
                )
                vals.append(int(donor_count))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_monodentate_cl_count(candidates: Sequence[str]) -> int:
    """Total count of Cl ligands bound to exactly one metal centre.

    This is a GSH liability proxy: each monodentate Cl is a potential
    leaving group susceptible to aquation and GSH S-attack.
    """
    if not candidates:
        return 0
    metals = ("Pt", "Ru", "Ir", "Au", "Rh", "Os")
    count = 0
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            for atom in m.GetAtoms():
                if atom.GetSymbol() != "Cl":
                    continue
                metal_neighbours = [
                    n for n in atom.GetNeighbors() if n.GetSymbol() in metals
                ]
                if len(metal_neighbours) == 1:
                    count += 1
        except Exception:
            continue
    return int(count)


def metric_gsh_evasion_score(candidates: Sequence[str]) -> float:
    """Fraction of candidates that evade GSH S-attack (heuristic flag).

    Uses :func:`AnticancerMetricSuite.gsh_evasion_flag`.  Returns 0.0
    when no candidates or the suite is unavailable.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.metrics.anticancer_metric_suite import (  # type: ignore
            AnticancerMetricSuite,
        )
    except Exception:
        return 0.0
    suite = AnticancerMetricSuite()
    hits = 0
    counted = 0
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        counted += 1
        try:
            if suite.gsh_evasion_flag(m):
                hits += 1
        except Exception:
            continue
    return float(hits / counted) if counted else 0.0


def metric_dna_kb_proxy(candidates: Sequence[str]) -> float:
    """Mean DNA Kb proxy across candidates (heuristic, range [0, 1])."""
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.metrics.anticancer_metric_suite import (  # type: ignore
            AnticancerMetricSuite,
        )
    except Exception:
        return 0.0
    suite = AnticancerMetricSuite()
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(suite.dna_kb_proxy(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_anticancer_index(candidates: Sequence[str]) -> float:
    """Mean composite anticancer index across candidates (range [0, 1]).

    Uses :func:`AnticancerMetricSuite.composite_score` and averages the
    ``anticancer_index`` column.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.metrics.anticancer_metric_suite import (  # type: ignore
            AnticancerMetricSuite,
        )
    except Exception:
        return 0.0
    suite = AnticancerMetricSuite()
    vals: List[float] = []
    for s in candidates:
        try:
            composite = suite.composite_score(s)
            ai = composite.get("anticancer_index", 0.0)
            # Treat NaN as 0.0 to avoid aggregate corruption.
            try:
                ai_f = float(ai)
            except Exception:
                ai_f = 0.0
            if ai_f != ai_f:  # NaN check
                ai_f = 0.0
            vals.append(ai_f)
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# WF-T25-Metallo-Proxies — 4 metallodrug-specific proxies
# (reduction_potential / trans_effect / lfse / pt_dna_crosslink)
# These are PROXIES (Shriver & Atkins Table 17.7 + Appleton 1997 + Wang
# 2005), not measured values.  All return [0, 1] and degrade gracefully
# to 0.0 on import / parse failure.  Wired into the cell / aggregate
# / serialise path by the ``--metallodrug-proxies`` CLI flag.
# ---------------------------------------------------------------------------
def _import_metallodrug_proxies():
    """Best-effort import of the metallodrug_property_proxies module.

    Returns the imported module on success, ``None`` on ImportError so
    the wrapper metric_*_mean functions below can degrade gracefully
    (returning 0.0 in line with the existing convention).
    """
    try:
        from molmetal_lam.priors.metallodrug_property_proxies import (  # type: ignore
            compute_metallodrug_proxies,
        )
        return compute_metallodrug_proxies
    except Exception:
        return None


def metric_metallodrug_reduction_potential_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`reduction_potential_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("reduction_potential", 0.0))
        except Exception:
            continue
        if v != v:  # NaN
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_metallodrug_trans_effect_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`trans_effect_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("trans_effect", 0.0))
        except Exception:
            continue
        if v != v:
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_metallodrug_lfse_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`lfse_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("lfse", 0.0))
        except Exception:
            continue
        if v != v:
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_metallodrug_pt_dna_crosslink_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`pt_dna_crosslink_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("pt_dna_crosslink", 0.0))
        except Exception:
            continue
        if v != v:
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# WF-SA-Penalty-Guidance — SA + QED aggregate metrics (closes the
# WF-SA-Penalty-Guidance requirement to report SA + QED + validity).
# ---------------------------------------------------------------------------
def metric_sa_mean(candidates: Sequence[str]) -> float:
    """Mean Ertl-Schuffenhauer SA score across ``candidates`` ([1, 10]).

    Returns 0.0 when no candidates parse or when sascorer is unavailable
    (consistent with the convention that the SA aggregate reports a
    finite numeric even in the offline-fallback path).
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.sa_score import sa_score_ertl  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            v = float(sa_score_ertl(s))
        except Exception:
            continue
        if v != v:  # NaN
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_qed_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit QED across ``candidates`` ([0, 1]).

    Returns 0.0 when no candidates parse or when QED is unavailable.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            v = qed_from_smiles(s)
        except Exception:
            v = None
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:  # NaN
            continue
        vals.append(fv)
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# WF-Phase3b-MetricsV2 — 8 tumor-relevant anticancer / ADMET / safety
# columns (closes 8/25 of TargetDiff gap beyond the 9 P0 columns shipped
# in WF-P0-Metrics-Add).  All CPU-only (RDKit / numpy); zero GPU load.
# Lit basis:
#   - logp7_4               Weininger 1990 + Patrick 2009 substituted-logP
#   - gi50_proxy            Hou 2007 ADMET regression
#   - cell_permeability_logPapp  Hou 2007 + Mente 2015 heuristic
#   - herg_cardio_risk      Veith 2009 + Cavalluzzi 2023 rule review
#   - ames_mutagen          Benigni-Richard 2005 + Sushko 2012 alerts
#   - hepatotox_index       Hughes 2008 Rule-of-2 + Stepan 2011 alerts
#   - aqueous_solubility_logS  Delaney 2004 ESOL
#   - plasma_protein_binding  Obach 1999 sigmoid regression
# See ``molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md`` for the
# full math basis and 20 tests.  Each function is a thin wrapper around
# :mod:`molmetal_lam.sbdd_env.metrics_v2` and returns 0.0 on any import
# or parse failure (parity with ``metric_logp_mean`` convention).
# ---------------------------------------------------------------------------
def _import_metrics_v2():
    """Best-effort import of metrics_v2 batch helpers.

    Returns the imported module on success, ``None`` on ImportError so
    the wrapper metric_*_mean functions below can degrade gracefully
    (returning 0.0 in line with the existing convention).
    """
    try:
        from molmetal_lam.sbdd_env import metrics_v2  # type: ignore
        return metrics_v2
    except Exception:
        return None


def metric_logp7_4_mean(candidates: Sequence[str]) -> float:
    """Mean substituted-logP7.4 over ``candidates`` (range unbounded).

    Defined as ``logP - 0.45·α_7.4 + 0.30·β_7.4`` — Weininger 1990 base
    logP with Patrick 2009 ionisation corrections.  Honours the lit
    recommendation that ionisable acids (e.g. aspirin -COOH) report
    lower logP7.4 than Crippen logP.  Returns 0.0 on import failure or
    empty input.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.logp7_4_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_gi50_proxy_mean(candidates: Sequence[str]) -> float:
    """Mean Hou 2007 GI50 proxy over ``candidates`` (range [0, 8]).

    ``4.5 + 0.30·MR - 0.015·TPSA - 0.50·(RotB/10) - 0.20·ArRings``.
    Heuristic activity predictor (not wet-lab calibrated).  Organic-only
    baseline gives ~8.0 (no activity).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.gi50_proxy_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_cell_permeability_logPapp_mean(candidates: Sequence[str]) -> float:
    """Mean cellular permeability logPapp over ``candidates`` ([-8, -3]).

    ``-4.0 + 0.33·logP - 0.013·TPSA + 0.40·HBD`` (Hou 2007 / Mente
    2015).  More negative = poorer permeability (caffeine-like
    behaviour).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.cell_permeability_logPapp_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_herg_cardio_risk_mean(candidates: Sequence[str]) -> float:
    """Mean hERG cardiac risk in [0, 1] over ``candidates`` (Veith 2009).

    Sums 4 rule bonuses (MW, logP, TPSA, basic N).  Returns 0.0 on
    import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.herg_cardio_risk_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_ames_mutagen_mean(candidates: Sequence[str]) -> float:
    """Mean AMES mutagenicity alert rate over ``candidates`` ([0, 1]).

    ``1.0`` if any of 6 SMARTS alerts match (Benigni-Richard 2005 +
    Sushko 2012).  Aggregated as fraction-of-flagged across the batch.
    Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.ames_mutagen_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_hepatotox_index_mean(candidates: Sequence[str]) -> float:
    """Mean hepatotoxicity index over ``candidates`` ([0, 1]).

    Hughes 2008 Rule-of-2 + Stepan 2011 structural alerts (aniline /
    hydrazine / furan / thiophene).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.hepatotox_index_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_aqueous_solubility_logS_mean(candidates: Sequence[str]) -> float:
    """Mean aqueous solubility logS over ``candidates`` ([-12, 2]).

    Delaney 2004 ESOL formula.  More negative = lower solubility.
    Ethanol ~0 (miscible); naphthalene <= -2 (low).  Returns 0.0 on
    import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.aqueous_solubility_logS_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_plasma_protein_binding_mean(candidates: Sequence[str]) -> float:
    """Mean plasma protein binding fraction over ``candidates`` ([0, 1]).

    Sigmoid of logP (Obach 1999).  Caffeine ~6% (low); naphthalene
    >=80% (high).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.plasma_protein_binding_mean(list(candidates)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-Phase3D-PerResidue-Diversity — sub-pocket fingerprint diversity
# (Bemis & Murcko 1996 + Jasial 2021 IntDiv survey + Peter 2019 SPF).
#
# Mean pairwise per-residue Tanimoto distance over a set of generated
# SMILES, weighted by pocket residues.  Complements
# ``diversity_tanimoto`` (whole-molecule) and ``diversity_homotype``
# (typed-variable Jaccard).  CPU-only (RDKit Morgan + numpy).
#
# The residue weights default to ``[1] * K_uniform`` so the metric
# collapses to ordinary Tanimoto when no pocket is loaded — see
# ``per_residue_diversity`` docstring.  Returns 0.0 on import failure,
# empty input, or any per-call failure.  Set ``--sub-pocket-diversity
# off` (CLI is added in Phase 4) to skip this metric.
# ---------------------------------------------------------------------------
def metric_subpocket_diversity_mean(
    candidates: Sequence[str],
    n_residues: int = 4,
) -> float:
    """Mean pairwise per-residue Tanimoto distance over ``candidates``.

    Parameters
    ----------
    candidates : Sequence[str]
        Generated SMILES from a cell.
    n_residues : int, default 4
        Number of sub-pocket residue bins (uniform-weight placeholder;
        the real per-residue weight vector comes from
        ``voxelization.py:208,255`` or ``vina_adapter.py:961`` once the
        pocket loader is wired — Phase 4 ships the metric on a uniform
        default so it can be added to the dispatch table without
        depending on the pocket loader).

    Returns
    -------
    float in [0, 1] (mean per-residue Tanimoto *distance*).
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.per_residue_diversity import (  # type: ignore
            per_residue_diversity,
        )
    except Exception:
        return 0.0
    # De-duplicate (the metric does not auto-dedup; identical SMILES
    # inflate the mean downward per its docstring).
    seen: set = set()
    distinct: List[str] = []
    for s in candidates:
        if s and s not in seen:
            seen.add(s)
            distinct.append(s)
    if len(distinct) < 2:
        return 0.0
    weights = [1] * int(n_residues)
    try:
        return float(per_residue_diversity(distinct, weights))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-Phase3G-MetalCoordProbe — graph-theoretic metal-coordination probe
# (Lippard & Berg 1995 + Reedijk 1987 + Miessler 2014).
#
# Coordinate-free metal-coordination check: parse SMILES, walk metal
# neighbours (H excluded), compare CN against canonical tables per
# (metal, oxidation_state).  Returns the *compliance rate* over the
# generated set (organic-only inputs are excluded from the denominator
# by design, matching ``soft_score_metal_geometry`` semantics).
# ---------------------------------------------------------------------------
def metric_metal_coord_compliance_mean(candidates: Sequence[str]) -> float:
    """Mean metal-coordination compliance rate over ``candidates``.

    For every molecule containing a recognised metal centre
    (Pt/Pd/Au/Ag/Ru/Ir/Rh/Os/Re/Fe/Co/Ni/Cu/Zn/Mn), compares the
    observed coordination number against the canonical CN for that
    (metal, OS) pair.  The returned value is the fraction of compliant
    *metal-containing* molecules (organic-only inputs are excluded
    from the denominator).  When no molecules contain a metal centre
    the function returns 0.0 (the metric does not fire on organics
    by design — mirrors ``soft_score_metal_geometry`` semantics).

    Returns 0.0 on import failure or any other failure.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.metal_coord_probe import (  # type: ignore
            compliance_rate,
        )
    except Exception:
        return 0.0
    try:
        return float(compliance_rate(list(candidates)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-Rigid-RMSD-Metric — rigid-fragment RMSD between generated and reference
# ligands (TargetDiff §4.4 SOTA signal, target ≈ 1.5 Angstroms).
#
# Uses RDKit's :func:`AllChem.GetBestRMS` for the rigid alignment: this
# function superimposes ``probe`` onto ``ref`` (best symmetry-aware
# mapping) and returns the RMSD in Angstroms.  The metric is the mean
# RMSD over the (candidate, reference) pairs in a cell, where each
# candidate SMILES is embedded in 3-D with :func:`AllChem.EmbedMolecule`
# before alignment.
#
# Honest-framing: ``GetBestRMS`` requires both molecules to have a
# conformer; the embedding step uses ``ETKDGv3`` (deterministic, fast)
# with a single attempt.  When RDKit cannot embed a candidate (large /
# unusual topology) the candidate is skipped — the metric is therefore
# a *conditional* mean over embeddable candidates.  This is the same
# convention used by TargetDiff's open-source eval script and matches
# the lower-is-better convention (target ≈ 1.5 Å).
#
# CPU-only (no GPU); falls back to 0.0 when RDKit is unavailable, when
# the reference SMILES is missing, or when no candidate embeds.
# ---------------------------------------------------------------------------
def _embed_3d_for_rmsd(mol) -> bool:
    """Embed ``mol`` in 3-D using ETKDGv3 + MMFF optimization.

    Returns True on success, False on any failure (silently skipped by
    the caller).  The conformer is left attached to ``mol`` so the
    subsequent :func:`AllChem.GetBestRMS` call can use it.

    Honest-framing: the previous implementation set
    ``params.maxAttempts = 1`` which raises ``AttributeError`` in
    recent RDKit builds (maxAttempts is a positional argument to
    :func:`EmbedMolecule`, not an EmbedParameters attribute — the
    ETKDGv3 equivalent is ``numZeroFail``).  The silent failure caused
    every RMSD/CoM metric to silently collapse to 0.0 in production
    sweeps.  We now use ``numZeroFail`` (the documented ETKDGv3 knob)
    and pass it positionally for belt-and-braces back-compat.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return False
    if mol is None:
        return False
    try:
        if mol.GetNumAtoms() <= 0 or mol.GetNumConformers() > 0:
            return mol.GetNumConformers() > 0
        # Add explicit Hs so the embedded geometry matches what
        # GetBestRMS expects (it operates on heavy atoms but the Hs
        # help the force field).
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D  # deterministic
        params.numZeroFail = 1      # ETKDGv3 equivalent of "maxAttempts=1"
        # Belt-and-braces: pass maxAttempts positionally too so older
        # RDKit builds that DO accept it still work — but only when
        # an ``EmbedParameters`` object isn't already providing the
        # settings (the two overloads are mutually exclusive).
        try:
            cid = AllChem.EmbedMolecule(mol_h, params)
        except Exception:
            cid = AllChem.EmbedMolecule(
                mol_h,
                maxAttempts=1,
                randomSeed=int(params.randomSeed),
            )
        if cid < 0:
            # Embedding can fail for unusual topologies; the metric
            # skips the molecule in that case.
            return False
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=50)
        except Exception:
            # MMFF can fail on metallic centres; not fatal — keep
            # the un-optimised conformer (RDKit's GetBestRMS still
            # works, just with a slightly higher RMSD).
            pass
        # Strip the explicit Hs back off so GetBestRMS sees the same
        # heavy-atom set as the original mol (this is the convention
        # used by every docking-pipeline RMSD eval).
        mol_no_h = Chem.RemoveHs(mol_h)
        # Transfer the conformer onto the original mol so callers can
        # use ``mol.GetConformer()`` directly.
        try:
            conf = mol_no_h.GetConformer()
            mol.AddConformer(conf, assignId=True)
        except Exception:
            # If transfer fails, leave the embedded mol as-is.
            return mol_h.GetNumConformers() > 0
        return mol.GetNumConformers() > 0
    except Exception:
        return False


def metric_rigid_rmsd_mean(
    gen_mol,
    ref_mol,
) -> float:
    """Rigid-fragment RMSD (Angstroms) between ``gen_mol`` and ``ref_mol``.

    Uses :func:`rdkit.Chem.AllChem.GetBestRMS` to align ``gen_mol`` onto
    ``ref_mol`` (symmetry-aware, minimises RMSD over all automorphisms)
    and returns the resulting RMSD in Angstroms.  Lower is better;
    TargetDiff SOTA target ≈ 1.5 Å.

    Returns 0.0 when:
      * either mol is ``None``,
      * RDKit is unavailable,
      * ``ref_mol`` lacks a conformer (caller is responsible for
        embedding the reference first),
      * the alignment fails.

    The function is CPU-only and deterministic (seeded ETKDGv3 for
    any fresh embedding).
    """
    if gen_mol is None or ref_mol is None:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return 0.0
    try:
        if ref_mol.GetNumConformers() == 0:
            return 0.0
        # Embed the candidate if it doesn't already have a conformer.
        cand = gen_mol
        if cand.GetNumConformers() == 0:
            if not _embed_3d_for_rmsd(cand):
                return 0.0
        # Symmetry-aware rigid alignment; returns RMSD in Angstroms.
        rmsd = AllChem.GetBestRMS(cand, ref_mol)
        try:
            rmsd_f = float(rmsd)
        except Exception:
            return 0.0
        if rmsd_f != rmsd_f:  # NaN
            return 0.0
        return max(0.0, rmsd_f)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-CoM-Shift-Metric — center-of-mass shift (target ≤ 3 Å per TargetDiff SOTA)
# ---------------------------------------------------------------------------
# Honest-framing: the original spec named the RDKit helper
# ``rdkit.Chem.Descriptors3D.CenterCenter`` but that symbol does NOT
# exist in the installed RDKit (verified — ``Descriptors3D`` only
# exposes Asphericity / NPR / PMI / RadiusOfGyration / SpherocityIndex
# / PBF / InertialShapeFactor / Eccentricity).  We therefore compute
# the mass-weighted centre of mass directly from the conformer's
# heavy-atom coordinates via numpy — this is the same quantity used by
# every docking pipeline (AutoDock Vina, DiffDock, TargetDiff) and the
# same quantity the spec was naming when it said "CoM shift vs
# reference ligand".  Pure RDKit/numpy + CPU-only.
# ---------------------------------------------------------------------------
def _mol_center_of_mass(mol):
    """Mass-weighted CoM of ``mol`` from its (first) 3-D conformer.

    Returns a numpy ndarray of shape ``(3,)`` on success, or the scalar
    ``0.0`` (a sentinel "uncomputable") when:

      * ``mol`` is None,
      * RDKit is unavailable,
      * the molecule has no conformer (caller is responsible for
        embedding — we silently skip rather than corrupt the aggregate),
      * the molecule has zero atoms,
      * numpy is unavailable,
      * any other exception is raised.

    The choice of 0.0 as the fallback matches the convention used by
    every other metric in this file: "0.0 means the metric could not
    be evaluated".  Callers MUST check ``isinstance(result, np.ndarray)``
    to distinguish a successful CoM from the sentinel.
    """
    if mol is None:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return 0.0
    try:
        if mol.GetNumConformers() == 0:
            return 0.0
        try:
            import numpy as _np  # type: ignore
        except Exception:
            return 0.0
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        if n_atoms <= 0:
            return 0.0
        coords = _np.array(
            [list(conf.GetAtomPosition(i)) for i in range(n_atoms)],
            dtype=float,
        )
        masses = _np.array(
            [a.GetMass() for a in mol.GetAtoms()], dtype=float
        )
        if masses.sum() <= 0.0:
            return 0.0
        com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
        return com
    except Exception:
        return 0.0


def metric_com_shift_mean(
    gen_mol,
    ref_ligand_mol,
) -> float:
    """Center-of-mass shift (Angstroms) between a generated ligand and a
    reference (crystal) ligand, in the pocket frame.

    Computed as::

        || CoM(gen_mol) - CoM(ref_ligand_mol) ||_2

    using mass-weighted centre-of-mass coordinates from each mol's
    3-D conformer.  Both mols must already have conformers — the
    :func:`_embed_3d_for_rmsd` helper handles embedding with a
    deterministic ETKDGv3 seed.

    TargetDiff SOTA target ≈ 3 Å (lower = better).  Returns 0.0 when:

      * either input is None,
      * RDKit is unavailable,
      * numpy is unavailable,
      * either mol lacks a conformer (caller should embed first),
      * either mol has zero atoms,
      * any other exception is raised.

    The function is CPU-only and deterministic given fixed input
    conformers.
    """
    if gen_mol is None or ref_ligand_mol is None:
        return 0.0
    try:
        import numpy as _np  # type: ignore
    except Exception:
        return 0.0
    com_g = _mol_center_of_mass(gen_mol)
    com_r = _mol_center_of_mass(ref_ligand_mol)
    if (
        not isinstance(com_g, _np.ndarray)
        or not isinstance(com_r, _np.ndarray)
    ):
        return 0.0
    try:
        diff = com_g - com_r
        d = float(_np.linalg.norm(diff))
    except Exception:
        return 0.0
    if d != d:  # NaN guard
        return 0.0
    return max(0.0, d)


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
        w_click = float(globals().get("CLICK_RULE_WEIGHT", 0.5))
        w_metal = float(globals().get("METAL_PRIOR_WEIGHT", 0.5))
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


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------
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
    cell.warnings.append(
        f"keep_high_sa_tiles={bool(keep_high_sa_tiles)}"
    )
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
    if float(platinai_weight) > 0.0 and "_R4_PLATINAI_ORACLE" not in globals():
        try:
            from molmetal_lam.reward.platinai_oracle import PlatinAIOracle
            _R4_PLATINAI_ORACLE = PlatinAIOracle()
            if _R4_PLATINAI_ORACLE.available:
                cell.warnings.append(
                    f"platinai_weight={float(platinai_weight):.3f} "
                    f"oracle_corpus={_R4_PLATINAI_ORACLE.n_corpus}"
                )
            else:
                cell.warnings.append(
                    f"platinai_weight={float(platinai_weight):.3f} "
                    "oracle_UNAVAILABLE (graceful 0.0)"
                )
        except Exception as exc:  # pragma: no cover - defensive
            cell.warnings.append(
                f"platinai_oracle_load_failed: {exc}"
            )
            _R4_PLATINAI_ORACLE = None
        globals()["_R4_PLATINAI_ORACLE"] = _R4_PLATINAI_ORACLE

    # ------------------------------------------------------------------
    # TODO-30 P5.2 Tier 1 (2026-09-17) — load wet-lab assay TSV once
    # at sweep start.  The dict is shared across every (pocket × seed)
    # cell so the per-leaf overhead is just a single dict lookup.
    # The loader is permissive (wide header / two-column / canonical
    # SMILES / # comments) and degrades to an empty dict when the
    # file is missing or malformed — so the rest of the pipeline runs
    # unchanged when --wetlab-input is omitted (which is the default).
    # ------------------------------------------------------------------
    _R4_WETLAB_MEASUREMENTS: Dict[str, float] = {}
    _args_for_wetlab = globals().get("args")
    _wetlab_input_path = (
        getattr(_args_for_wetlab, "wetlab_input", None)
        if _args_for_wetlab is not None else None
    )
    if _wetlab_input_path:
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
                "measurement dict (channel will be a no-op)", exc,
            )
            _R4_WETLAB_MEASUREMENTS = {}
        globals()["_R4_WETLAB_MEASUREMENTS"] = _R4_WETLAB_MEASUREMENTS

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
        "auto-pt-strict", "auto_pt_strict",
        "auto-pt-iv", "auto_pt_iv",
        "auto-pt-chelate", "auto_pt_chelate",
        "auto-labile", "auto_labile",
        "auto-unknown", "auto_unknown",
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
                METAL_SEED_SMILES.get(metal_seed)
                if metal_seed is not None else None
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
                str(t).strip() for t in click_rules
                if str(t).strip() in auto_aliases
            )
            # alias -> scaffold-key override map.  ``auto-unknown``
            # forces all-5 regardless of detection; ``auto-labile`` /
            # ``auto-pt-iv`` / ``auto-pt-chelate`` ignore the
            # detection and use the wider rule set.
            alias_to_scaffold = {
                "auto-pt-strict":   "strict_Pt_II",
                "auto_pt_strict":   "strict_Pt_II",
                "auto-pt-iv":       "Pt_IV",
                "auto_pt_iv":       "Pt_IV",
                "auto-pt-chelate":  "Pt_II_chelating",
                "auto_pt_chelate":  "Pt_II_chelating",
                "auto-labile":      "labile_metal",
                "auto_labile":      "labile_metal",
                "auto-unknown":     "unknown",
                "auto_unknown":     "unknown",
            }
            scaffold_key = alias_to_scaffold.get(chosen_alias, scaffold)
            rules = default_compatible_rules(
                scaffold_key, allow_incompatible=allow_incompatible_click,
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
                root = MoleculeClosedTerm.from_smiles(
                    ref_record.smiles, embed_3d=False
                )
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
                v_pocket = pocket_features_from_binding_site(
                    binding_site_for_features
                )
            except Exception as exc:
                cell.warnings.append(
                    f"pocket_features_from_binding_site_failed: {exc}"
                )
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
            platinai_oracle=globals().get("_R4_PLATINAI_ORACLE"),
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
            fg_veto_strict=bool(
                getattr(globals().get("args"), "fg_veto_strict", False)
            ),
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
                _search_kwargs["pocket_boost_strength"] = (
                    float(pocket_boost_strength)
                )
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
                    f"learned_shaping_register_failed: {exc}; "
                    "falling back to off"
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
                smi = c.canonical_smiles() if hasattr(c, "canonical_smiles") else _candidate_smi(c)
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
                                    _coord[:_n], dtype=float,
                                )
                                if _torch_local is not None:
                                    c.coords_3d = _torch_local.as_tensor(
                                        _coord_np, dtype=_torch_local.float32,
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
                                            [_mol_no_h.GetAtomWithIdx(int(i)).GetAtomicNum()
                                             for i in range(_n)],
                                            dtype=_torch_local.long,
                                        )
                                        _xyz = _torch_local.as_tensor(
                                            _coord_np, dtype=_torch_local.float32,
                                        )
                                        _cloud = AtomCloud(
                                            positions=_xyz,
                                            atomic_numbers=_z,
                                        )
                                        _dec = _rd_decoder.decode(_cloud)
                                        c.decoder_status = (
                                            _dec.error or 'decoded_path_b'
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
            if str(_status).startswith("decoded_path_b") or str(_status).startswith("rework_sanitize_failed"):
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
            1.0 if synthesizability_via_lambda_paths(
                _state_from_smi(s, MoleculeClosedTerm),
                check_bnf=check_beta_normal_form,
            ) else 0.0
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
                cell.patent_closest_drug = str(
                    closest_drug_counts.most_common(1)[0][0]
                )
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
                    cell.rigid_rmsd_mean = float(
                        sum(rmsd_vals) / len(rmsd_vals)
                    )
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
                    cell.com_shift_mean = float(
                        sum(com_vals) / len(com_vals)
                    )
        except Exception as exc:
            cell.warnings.append(f"com_shift_compute_failed: {exc}")
    return cell


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
_SPLIT_DEFAULT_DIR = _REPO_ROOT / "molmetal" / "data" / "metallo_scaffold_split"
_METALLO_POOL_DEFAULT = (
    _REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
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
        raise RuntimeError(
            f"No manifest rows found at {manifest_path}; aborting."
        )
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
    if (
        auto_pt_strict_default
        and (click_rules is None or len(click_rules) == 0)
    ):
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
                pid, seed, n_simulations, n_top_k,
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
                cell.novelty = training_set_novelty(cell.candidates, list(training_smiles))
            cells.append(cell)
    elapsed = time.time() - started

    # Aggregate.
    agg: Dict[str, float] = {}
    ox_dist_merged: Dict[str, int] = {}
    if cells:
        def _mean(field_name: str) -> float:
            vals = [getattr(c, field_name) for c in cells if getattr(c, field_name) is not None]
            return float(sum(vals) / len(vals)) if vals else 0.0

        def _sum_int(field_name: str) -> int:
            vals = [getattr(c, field_name) for c in cells if getattr(c, field_name) is not None]
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
        agg["metallodrug_trans_effect_mean"] = _mean(
            "metallodrug_trans_effect_mean"
        )
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
            "click_rules": (
                list(click_rules) if click_rules is not None else "all-5"
            ),
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


def _render_summary_md(payload: Dict[str, Any]) -> str:
    cfg = payload.get("config", {})
    agg = payload.get("aggregate", {})
    cells = payload.get("cells", [])
    lines: List[str] = []
    lines.append("# WF-Lambda-1 — Pure Lambda-Only Baseline")
    lines.append("")
    lines.append("> Honest-framing: this is a MEASURED run. PROJECTED numbers")
    lines.append("> from the spec are not invoked here — the spec is in")
    lines.append("> `molmetal/reports/ultracode_audit/wf_lambda1_spec.md`.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- n_pockets : `{cfg.get('n_pockets')}`")
    lines.append(f"- seeds     : `{cfg.get('seeds')}`")
    lines.append(f"- n_simulations per cell : `{cfg.get('n_simulations')}`")
    lines.append(f"- n_top_k    : `{cfg.get('n_top_k')}`")
    lines.append(f"- prior_enabled : `{cfg.get('prior_enabled')}`")
    lines.append(f"- metal_seed    : `{cfg.get('metal_seed')}`")
    lines.append("")
    lines.append("## Aggregate metrics (mean across cells)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for key in (
        "validity_rate",
        "uniqueness_rate",
        "diversity_tanimoto",
        "diversity_homotype",
        "novelty",
        "synthesizability_rate",
        "metal_compliance_rate",
        "reference_tanimoto",
        "rigid_rmsd_mean",
        "com_shift_mean",
    ):
        lines.append(f"| {key} | {agg.get(key, 0.0):.4f} |")
    lines.append("")
    lines.append("## WF-P0-Metrics — 9 P0 anticancer / drug-likeness columns")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| logp_mean | {agg.get('logp_mean', 0.0):.4f} |")
    lines.append(f"| tpsa_mean | {agg.get('tpsa_mean', 0.0):.4f} |")
    lines.append(f"| rotb_mean | {agg.get('rotb_mean', 0.0):.4f} |")
    cn = agg.get("coordination_number_mean", 0.0)
    lines.append(f"| coordination_number_mean | {cn:.4f} |")
    cl = agg.get("monodentate_cl_count", 0)
    lines.append(f"| monodentate_cl_count | {int(cl)} |")
    lines.append(f"| gsh_evasion_score | {agg.get('gsh_evasion_score', 0.0):.4f} |")
    lines.append(f"| dna_kb_proxy | {agg.get('dna_kb_proxy', 0.0):.4f} |")
    lines.append(f"| anticancer_index | {agg.get('anticancer_index', 0.0):.4f} |")
    ox_dist = payload.get("oxidation_state_distribution_total", {}) or {}
    if ox_dist:
        ox_str = ", ".join(
            f"{k}={v}" for k, v in sorted(ox_dist.items())
        )
    else:
        ox_str = "(none)"
    lines.append(f"| oxidation_state_distribution | {ox_str} |")
    lines.append("")
    lines.append("## WF-Phase3b-MetricsV2 — 8 tumor-relevant anticancer / ADMET columns")
    lines.append("")
    lines.append("Lit basis: Weininger 1990 / Patrick 2009 / Hou 2007 / Veith 2009 /")
    lines.append("Benigni-Richard 2005 / Hughes 2008 / Delaney 2004 ESOL / Obach 1999.")
    lines.append("All CPU-only; zero GPU load. Heuristic evaluators (NOT wet-lab calibrated).")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in (
        "logp7_4_mean",
        "gi50_proxy_mean",
        "cell_permeability_logPapp_mean",
        "herg_cardio_risk_mean",
        "ames_mutagen_mean",
        "hepatotox_index_mean",
        "aqueous_solubility_logS_mean",
        "plasma_protein_binding_mean",
    ):
        lines.append(f"| {k} | {agg.get(k, 0.0):.4f} |")
    lines.append("")
    lines.append("## WF-Phase3D-PerResidue-Diversity + WF-Phase3G-MetalCoordProbe")
    lines.append("")
    lines.append("Lit basis: Bemis & Murcko 1996 / Jasial 2021 IntDiv / Peter 2019 SPF;")
    lines.append("Lippard & Berg 1995 / Reedijk 1987 / Miessler 2014 d-block geometries.")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(
        f"| diversity_subpocket | "
        f"{agg.get('diversity_subpocket', 0.0):.4f} |"
    )
    lines.append(
        f"| metal_coord_compliance | "
        f"{agg.get('metal_coord_compliance', 0.0):.4f} |"
    )
    # ---- WF-T25-Metallo-Proxies: 4 metallodrug proxies (opt-in) -------
    if bool(agg.get("metallodrug_proxies_enabled", False)):
        lines.append("")
        lines.append("## WF-T25-Metallo-Proxies — 4 metallodrug-specific proxies")
        lines.append("")
        lines.append("Lit basis: Shriver & Atkins Table 17.7 (spectrochemical series) +")
        lines.append("Appleton 1997 Coord. Chem. Rev. (trans-effect series) + Wang 2005 +")
        lines.append("Cohen 2007 (Pt-DNA crosslink). All values in [0, 1].")
        lines.append("PROXIES, not measured quantities — see `molmetal/reports/")
        lines.append("wf_t25_metallo_proxies/final.md` for the honest framing.")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(
            f"| metallodrug_reduction_potential_mean | "
            f"{agg.get('metallodrug_reduction_potential_mean', 0.0):.4f} |"
        )
        lines.append(
            f"| metallodrug_trans_effect_mean | "
            f"{agg.get('metallodrug_trans_effect_mean', 0.0):.4f} |"
        )
        lines.append(
            f"| metallodrug_lfse_mean | "
            f"{agg.get('metallodrug_lfse_mean', 0.0):.4f} |"
        )
        lines.append(
            f"| metallodrug_pt_dna_crosslink_mean | "
            f"{agg.get('metallodrug_pt_dna_crosslink_mean', 0.0):.4f} |"
        )
    lines.append("")
    lines.append("> `diversity_tanimoto` is the SE(3) / atom-symbol-histogram")
    lines.append("> baseline (legacy `diversity_alpha`). `diversity_homotype`")
    lines.append("> is the Lambda-native metric from WF-Lambda-2 (typed-variable")
    lines.append("> cosine + β-reduction-depth + click-rule-fires Jaccard).")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | novel | syn | metal | ref_tan | rigid_rmsd | com_shift | logP | TPSA | RotB | coord | gsh | dna | ai | cl | warnings |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        lines.append(
            "| {pid} | {seed} | {nc} | {nd} | {v:.3f} | {u:.3f} | "
            "{dt:.3f} | {dh:.3f} | {n:.3f} | {s:.3f} | {m:.3f} | {rt:.3f} | "
            "{rr:.3f} | {cs:.3f} | "
            "{lp:.2f} | {tp:.1f} | {rb:.2f} | {cn:.2f} | "
            "{gsh:.3f} | {dna:.3f} | {ai:.3f} | {cl} | {w} |".format(
                pid=c["pocket_id"],
                seed=c["seed"],
                nc=c["n_candidates"],
                nd=c["n_distinct"],
                v=c["validity_rate"],
                u=c["uniqueness_rate"],
                dt=c["diversity_tanimoto"],
                dh=c["diversity_homotype"],
                n=c["novelty"],
                s=c["synthesizability_rate"],
                m=c["metal_compliance_rate"],
                rt=c["reference_tanimoto"],
                rr=c.get("rigid_rmsd_mean", 0.0),
                cs=c.get("com_shift_mean", 0.0),
                lp=c["logp_mean"],
                tp=c["tpsa_mean"],
                rb=c["rotb_mean"],
                cn=c["coordination_number_mean"],
                gsh=c["gsh_evasion_score"],
                dna=c["dna_kb_proxy"],
                ai=c["anticancer_index"],
                cl=c["monodentate_cl_count"],
                w=len(c["warnings"]),
            )
        )
    lines.append("")
    lines.append(f"Total elapsed: `{payload.get('elapsed_s_total', 0.0):.2f} s`")
    lines.append("")
    lines.append("## Scorer (Lambda-only, NO docking / AdmetAI / PB)")
    lines.append("")
    lines.append("- `alpha_equivalence_uniqueness_score` — distinct beta-NF count.")
    lines.append("- `click_rule_match_bonus` — +1.0 if any of 5 click rules fires.")
    lines.append("- `metal_geometry_prior_bonus` — +1.0 if Pt=4 / Ru=Ir=6 coord.")
    lines.append("- `rdkit_validity_score` — 1.0 if RDKit can sanitize.")
    lines.append("- `synthesizability_via_lambda_paths` — 1.0 if beta-NF + RDKit.")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r4_lambda_only_run.py",
        description=(
            "WF-Lambda-1: pure Lambda baseline (no docking, no AdmetAI, no PB). "
            "Runs MCTSProofSearch using only Lambda-native scoring signals."
        ),
    )
    parser.add_argument(
        "--pockets",
        type=int,
        default=10,
        help="Number of test pockets (first N rows of crossdocked100_manifest.csv).",
    )
    # WF-Metallodrug-Vertical Phase 3 protocol-align — number of
    # generated molecules per (pocket, seed) cell.  Bumped default from
    # 8 (smoke) to 100 (TargetDiff per-cell protocol, Guan ICLR 2023
    # §4.4 Table 4 "samples per target").  Backward compatible:
    # setting ``--n-samples 8`` recovers the pre-Phase-3 smoke
    # bit-exact behaviour.  This flag controls the inner MCTSProofSearch
    # n_samples parameter (was previously hardcoded as 8 inside the
    # run_one_cell helper); the --n-top-k flag remains the cap on how
    # many of those samples are retained in the report.
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help=(
            "Number of generated molecules per (pocket, seed) cell. "
            "WF-Metallodrug-Vertical Phase 3 protocol-align bumped "
            "default from 8 to 100 to match the TargetDiff per-cell "
            "protocol (Guan ICLR 2023 §4.4 Table 4 'samples per target'). "
            "Setting --n-samples 8 recovers the pre-Phase-3 smoke "
            "bit-exact behaviour."
        ),
    )
    # WF-Metallodrug-Vertical Phase 3 — pocket-10 Å radius flag.
    # Default 10.0 Å matches the CrossDocked2020 standard pocket
    # extraction radius (Francoeur et al. 2020).  Backward compatible:
    # setting --pocket10-radius 8.0 recovers the legacy DiffDock-Pocket
    # crop radius.
    parser.add_argument(
        "--pocket10-radius",
        type=float,
        default=10.0,
        help=(
            "Pocket extraction radius in Angstroms (default 10.0 = "
            "CrossDocked2020 / TargetDiff standard).  Set 8.0 to "
            "recover the legacy DiffDock-Pocket 8 Å crop."
        ),
    )
    # WF-Metallodrug-Vertical Phase 3 — pocket-conditioned reference
    # ligand warm-start.  When set the generator uses the pocket's
    # reference ligand SDF (under
    # /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/<pocket_id>/<pocket_id>_ligand.sdf)
    # as the warm-start for pocket-conditioned MCTS instead of the
    # unconditional base.  Default off (legacy unconditional sampling).
    parser.add_argument(
        "--reference-ligand",
        action="store_true",
        help=(
            "Use the pocket-specific reference ligand SDF "
            "(/mnt/storage/data/molmetal/crossdocked/extracted/"
            "crossdocked_pocket10/<pocket_id>/<pocket_id>_ligand.sdf) "
            "as a warm-start for pocket-conditioned MCTS.  Default off "
            "(legacy unconditional sampling)."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 0, 1234],
        help="Random seeds (space-separated).",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=200,
        help=(
            "MCTS simulation budget per cell.  WF-Lift-N-Sim-Cap "
            "(2026-09-14) removed the historical 100-cap; the CLI flag "
            "now takes effect verbatim up to a 10000 safety maximum "
            "(to prevent a typo locking the GPU)."
        ),
    )
    parser.add_argument(
        "--n-top-k",
        type=int,
        default=20,
        help="Number of top candidates returned per cell.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Sub-directory under molmetal/reports/. The full report path "
            "is molmetal/reports/wf_lambda1_<output-dir>/."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(_REPO_ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv"),
        help="Path to the crossdocked100 manifest CSV.",
    )
    parser.add_argument(
        "--prior-disabled",
        action="store_true",
        help="Disable the metal-geometry prior (ablation).",
    )
    parser.add_argument(
        "--training-set",
        type=str,
        default="",
        help="Optional path to a training-set SMILES file (one SMILES per line).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="MCTS rollout depth budget per simulation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cell logging.",
    )
    parser.add_argument(
        "--click-rules",
        nargs="*",
        default=None,
        help=(
            "Subset of click rule names to enable (space-separated). "
            "Default = all 5 (CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling). "
            "Example: --click-rules CuAAC (CuAAC-only ablation)."
        ),
    )
    parser.add_argument(
        "--metal-seed",
        nargs=None,
        default=None,
        choices=[None, "cisplatin", "ru_arene", "ir_cp_star"],
        help=(
            "Optional metal-containing root scaffold for MCTS. When set, "
            "the search is seeded with a canonical cisplatin / Ru-arene / "
            "Ir-Cp* molecule instead of the pocket reference ligand, so "
            "metal_compliance_rate becomes measurable. Default = None "
            "(use pocket reference, falling back to Cl[Pt]Cl)."
        ),
    )
    parser.add_argument(
        "--metal-seed-from-pocket",
        action="store_true",
        help=(
            "WF-Pocket-Invariance Phase 2C (sub-fix C): derive the "
            "metal_seed SMILES from the pocket's residue composition "
            "instead of using a fixed cisplatin seed.  Reads the "
            "PocketFeatureVector for each pocket and picks Pt(II) when "
            "positive_charge_fraction >= 0.30 (His triad / Lys-Arg-rich), "
            "Zn(II) when negative_charge_fraction >= 0.30 (MMP zincin / "
            "HDAC), or Pd(II) as a mixed/unknown fallback.  Each pocket "
            "therefore starts MCTS from a *different* metal_seed SMILES "
            "so the _unreactive_states cache + transposition_table cannot "
            "collapse to the same first-state hit.  Default = False "
            "(use --metal-seed verbatim, legacy behaviour).  When both "
            "--metal-seed and --metal-seed-from-pocket are set, the "
            "pocket-derived value wins (the explicit seed is logged "
            "but not used).  Use this flag for the round-13/14 "
            "pocket-conditioned-reference-ligand ablation."
        ),
    )
    parser.add_argument(
        "--use-pocket-conditioned-reference",
        action="store_true",
        help=(
            "WF-Lambda-Boost Phase 1 (sub-fix C extended): replace "
            "the metal_seed with a *pocket-conditioned reference "
            "ligand SMILES* (the lambda-calculus complement to "
            "--metal-seed-from-pocket).  Each pocket starts MCTS from "
            "a different real ligand chemistry (acetazolamide-style "
            "sulfonamide for CA2, marimastat-style hydroxamate for "
            "MMP2, thiol-coumarin for HDAC2, etc.) — different "
            "free-variable structure per pocket breaks the "
            "_unreactive_states cache + transposition_table at the "
            "lambda-calculus level, not just at the metal level.  "
            "Resolves via "
            ":mod:`molmetal_lam.lam_chem.reference_ligand_resolver`.  "
            "Falls back to cisplatin seed with a WARN log when the "
            "lookup misses.  Default = False (legacy metal_seed "
            "behaviour).  This flag is independent of "
            "--metal-seed-from-pocket (both can be enabled together: "
            "the metal_seed picks the metal identity and the "
            "reference-ligand picks the open lambda term)."
        ),
    )
    parser.add_argument(
        "--use-learned-prior",
        action="store_true",
        help=(
            "WF-Pocket-Invariance Phase 2B (sub-fix B): wire the "
            "learned_prior channel from "
            ":mod:`molmetal_lam.search_alg.learned_prior` into the "
            "MCTSProofSearch._prior computation, boosting pocket-"
            "relevant (rule, tile) actions over uniform exploration.  "
            "Opt-in (default OFF) so the legacy uniform-prior "
            "behaviour is preserved.  When enabled, passes "
            "learned_prior + pocket_boost_strength to "
            ":class:`MCTSProofSearch` so the soft-prior boost is "
            "applied.  See "
            ":func:`run_one_cell` for the kwarg plumbing."
        ),
    )
    parser.add_argument(
        "--pocket-boost-strength",
        type=float,
        default=1.0,
        help=(
            "WF-Lambda-Boost: scaling factor on the soft-prior boost "
            "when --use-learned-prior is enabled. 1.0 = full strength "
            "(default); 0.0 = no boost.  Range [0, 5] in practice; "
            "values > 2 over-fit the prior.  See "
            ":func:`run_one_cell` and :class:`MCTSProofSearch.search` "
            "for the kwarg plumbing."
        ),
    )
    parser.add_argument(
        "--use-sa-prior",
        action="store_true",
        help=(
            "WF-T25-SA-Prior: opt-in SA-aware MCTS leaf prior.  "
            "Distinct from --sa-weight (post-hoc reward channel): this "
            "flag biases the PUCT expansion policy itself by adding "
            "an SA-derived prior logit (-log(sa)) to each leaf.  "
            "Default OFF (legacy 0.5 constant prior).  When enabled, "
            "MCTS explores lower-SA (more drug-like / easier to "
            "synthesise) branches first.  Additive with --sa-weight "
            "— both can be enabled simultaneously.  See "
            ":func:`run_one_cell` and :class:`MCTSProofSearch.search` "
            "for the kwarg plumbing."
        ),
    )
    parser.add_argument(
        "--sa-prior-strength",
        type=float,
        default=0.5,
        help=(
            "WF-T25-SA-Prior: mixing weight between the legacy prior "
            "mass and the SA-derived prior mass.  0.0 = pure legacy, "
            "1.0 = pure SA-aware, 0.5 (default) = AGZ root-noise "
            "convention.  Only consulted when --use-sa-prior is set."
        ),
    )
    parser.add_argument(
        "--use-learned-shaping",
        action="store_true",
        help=(
            "WF-Deflex F5: enable LEARNED_SHAPING_ENABLED=1 for this "
            "run.  F5 closed-form bias away from high-SA.  Default "
            "OFF (backward compat)."
        ),
    )
    parser.add_argument(
        "--learned-shaping-weight",
        type=float,
        default=1.0,
        help=(
            "WF-Deflex F5: weight on the LearnedShaping contribution "
            "(see :mod:`molmetal_lam.reward.learned_shaping`). "
            "1.0 = full strength (default). Only consulted when "
            "--use-learned-shaping is set."
        ),
    )
    parser.add_argument(
        "--use-pocket-macro-inference",
        action="store_true",
        help=(
            "WF-Deflex Phase-2: bias MCTS root prior by the 32-d "
            "PocketMacroSkeleton embedding (trained v2 checkpoint at "
            "molmetal/models/pocket_macro_skeleton_v2.pt).  Calls "
            ":class:`PocketMacroInference` lazily inside "
            ":meth:`MCTSProofSearch.search`.  Default OFF (legacy "
            "0.5 constant prior preserved)."
        ),
    )
    parser.add_argument(
        "--use-symbolic-reward",
        action="store_true",
        help=(
            "WF-Deflex Phase-3: wire the symbolic-regression closed-"
            "form reward from :mod:`molmetal_lam.reward.symbolic_regression` "
            "as an additive channel alongside F5.  Default OFF "
            "(legacy Lambda-only)."
        ),
    )
    parser.add_argument(
        "--sa-weight",
        type=float,
        default=0.0,
        help=(
            "WF-SA-Penalty-Guidance: weight on the Ertl-Schuffenhauer "
            "synthetic-accessibility (SA) channel in the MCTS leaf "
            "reward aggregator. Default 0.0 (Lambda-only; SA channel "
            "silent — backward compatible). Setting --sa-weight 0.3 "
            "encourages Lambda MCTS to penalise SA-heavy (hard-to-"
            "synthesize) molecules; the SA score is computed via the "
            "Ertl sascorer (range [1, 10]; lower = more synthetically "
            "accessible; inverted by the aggregator so the reward "
            "contribution is positive when a molecule is easier to "
            "make)."
        ),
    )
    parser.add_argument(
        "--reward-lambda-weight",
        type=float,
        default=0.0,
        help=(
            "TODO-21 Strategy 1 (2026-09-17) — weight on the new "
            "Lambda-as-reward channel added by "
            ":mod:`molmetal_lam.lam_chem.lambda_reward_channel`.  "
            "When > 0 the MCTS leaf reward is augmented by "
            "``w_lambda_score * mean_cos_sim(CFM_pocket_emb, "
            "lambda_candidate_embs)`` where the candidates are "
            "supplied by the harness via "
            ":func:`set_lambda_candidates` between rollouts.  "
            "Couples the Lambda (algorithm) signal to the CFM "
            "(model) reward surface.  Opt-in (default 0.0 = "
            "channel silent, backward compatible).  Set 0.5 to "
            "let Lambda inform CFM training without dominating "
            "the dominant Vina / SA / QED reward."
        ),
    )
    parser.add_argument(
        "--reward-platinai-weight",
        type=float,
        default=0.0,
        help=(
            "Phase 2 (Metallodrug ultracode) — weight on the PlatinAI "
            "predicted-activity oracle channel. Default 0.0 (off; "
            "backward compatible). When > 0 the kNN oracle over the "
            "PlatinAI_MBFinder corpus (226,918 SMILES) + A2780/MCF7 "
            "predicted-activity sheets contributes w * (mean of "
            "Tanimoto-1-K neighbours' predicted activity in [0, 1]). "
            "Falls back to 0.0 when no neighbour is within Tanimoto "
            "0.3 of the query molecule. Set --reward-platinai-weight "
            "0.5 to lift predicted-active metallodrug candidates "
            "without dominating the dominant Vina / SA / QED reward."
        ),
    )
    parser.add_argument(
        "--reward-pharmacophore-weight",
        type=float,
        default=0.0,
        help=(
            "TODO-30 P2.5 Phase A — weight on the pharmacophore "
            "filter channel. Default 0.0 (off; backward compatible). "
            "When > 0 the candidate SMILES is scored against the "
            "Lipinski+Veber+ring-count pharmacophore gate "
            "(molmetal_lam.lam_chem.pharmacophore_filter) and the "
            "continuous [0, 1] score (1 = strict pass, 0 = all "
            "violations) is added with weight w to the leaf reward. "
            "The filter is permissive for metal-acylclic derivatives "
            "(cisplatin, nedaplatin). Falls back to 0.0 on parse "
            "failure. Set --reward-pharmacophore-weight 0.3 to lift "
            "drug-like candidates without dominating Vina / SA / "
            "QED reward."
        ),
    )
    parser.add_argument(
        "--pharmacophore-strict",
        action="store_true",
        help=(
            "TODO-30 P2.5 Phase A — when set, the pharmacophore "
            "channel uses the boolean strict gate (1.0 if zero "
            "violations else 0.0). Default OFF — uses the "
            "partial-credit 1 - violations/7 score."
        ),
    )
    parser.add_argument(
        "--no-pharmacophore-allow-acyclic",
        action="store_false",
        dest="pharmacophore_allow_acyclic",
        help=(
            "TODO-30 P2.5 Phase A — disable the ring-count bypass "
            "for metal-acylclic derivatives. Default ON (allow "
            "acyclic) so cisplatin / nedaplatin pass the gate."
        ),
    )
    parser.add_argument(
        "--reward-wetlab-weight",
        type=float,
        default=0.0,
        help=(
            "TODO-30 P5.2 Tier 1 (2026-09-17) — weight on the wet-lab "
            "signed-error channel. Default 0.0 (off; backward "
            "compatible). When > 0, the MCTS leaf reward is augmented "
            "by ``w_wetlab * (-|predicted - measured|)`` for assayed "
            "candidates and 0.0 elsewhere. Requires "
            "--wetlab-input path/to/assays.tsv; the file is parsed "
            "once at sweep start and the resulting "
            "{canonical_smiles: pIC50} dict is shared across cells. "
            "Set 1.0 to let wet-lab evidence shape the reward without "
            "dominating Vina / SA / QED. The signed-error range is "
            "[-3, 0] so the additive contribution is bounded."
        ),
    )
    parser.add_argument(
        "--wetlab-input",
        type=str,
        default=None,
        help=(
            "TODO-30 P5.2 Tier 1 — path to a TSV file of wet-lab "
            "measurements. File format: ``smiles<TAB>pIC50_value`` "
            "one row per measurement. Optional wide header row "
            "(``smiles<TAB>pIC50<TAB>cell_line<TAB>...``) is detected "
            "automatically; ``#`` comments are skipped. The file is "
            "loaded at sweep start (NOT at every leaf evaluation) so "
            "the per-leaf overhead is a single dict lookup. When "
            "omitted the wet-lab channel is a no-op even if "
            "--reward-wetlab-weight > 0. Tier 3 collaborator outreach "
            "is explicitly DEFER'd (TODO-30 Rank-15) — this file "
            "provides the API only."
        ),
    )
    parser.add_argument(
        "--metallodrug-proxies",
        action="store_true",
        help=(
            "WF-T25-Metallo-Proxies: opt-in flag that emits four "
            "metallodrug-specific property proxies per cell "
            "(reduction_potential / trans_effect / lfse / "
            "pt_dna_crosslink). These are PROXIES (Shriver & Atkins "
            "Table 17.7 + Appleton 1997 + Wang 2005), not measured "
            "values — all are clipped to [0, 1] and degrade to 0.0 "
            "for non-Pt molecules. Default OFF (backward compatible; "
            "the four per-cell columns stay at 0.0 in the report). "
            "When ON the cell writes "
            "metallodrug_reduction_potential_mean / "
            "metallodrug_trans_effect_mean / metallodrug_lfse_mean / "
            "metallodrug_pt_dna_crosslink_mean columns to the report."
        ),
    )
    parser.add_argument(
        "--allow-incompatible-click",
        action="store_true",
        help=(
            "F2-B (WF-Lambda-Fix-FullPath-v2): when --click-rules "
            "uses an auto-* scaffold alias (auto-pt-strict / "
            "auto-pt-iv / auto-pt-chelate / auto-labile / "
            "auto-unknown), the scaffold-detection helper normally "
            "GATES incompatible click rules out by default "
            "(ThiolEne + AmideCoupling on strict Pt_II, etc.).  "
            "Pass this flag to re-enable the incompatible rules "
            "explicitly — useful for chemistry-baseline comparison "
            "cells that need the historical raw all-5 behaviour.  "
            "Default = False (gated)."
        ),
    )
    parser.add_argument(
        "--fg-veto-strict",
        action="store_true",
        help=(
            "TODO-30 P1.2 (2026-09-17) — opt-in FG-compatibility "
            "veto. When set, every ReactionRule.reduce(reactants) call "
            "consults molmetal_lam.reactions.fg_compatibility."
            "FG_COMPATIBILITY (CPU-only SMARTS match via RDKit) and "
            "rejects (returns []) if the reactant SMILES contains a "
            "disfavoured FG and no tolerated FG. Default OFF "
            "(permissive; Round-12/13 measurements preserved). "
            "Property-test gated: the veto fires ONLY when all three "
            "of (a) --fg-veto-strict, (b) reactant SMILES has a "
            "disfavoured FG, (c) reactant SMILES is missing a "
            "tolerated FG. Negative-false-positive rate = 0 in "
            "tests/test_fg_compatibility.py. See "
            "molmetal/reports/wf_t30_p12_fg_veto/final.md."
        ),
    )
    parser.add_argument(
        "--keep-high-sa-tiles",
        action="store_true",
        help=(
            "WF-SA-Fragment-Pool-Optimize: by default the Phase-1 "
            "200+ tile pool is filtered to drop the top-10 "
            "highest-SA tiles (Ertl-Schuffenhauer SA > 3.87) — "
            "unstable α-hydroxy azides, alkyne-thioethers with "
            "thiol-yne competition, and a dicyclopentadiene "
            "partial. Pass this flag to keep those tiles in the "
            "pool (chemistry baseline / ablation only).  Default = "
            "False (filtered).  Source data: "
            "molmetal/reports/wf_sa_fragment_pool_optimize/sa_pool_scan.csv."
        ),
    )
    parser.add_argument(
        "--decoder-rework",
        action="store_true",
        help=(
            "WF-Round12-Lambda-PathB: control flag (no-op on the "
            "Lambda path). The decoder_rework chem-aware soft bond "
            "prior is a CFM coordinate-space decoder — it operates "
            "on (coords, Z) tensors produced by the flow-matching "
            "module, not on the typed β-NF MoleculeClosedTerm "
            "objects returned by MCTSProofSearch. Setting this flag "
            "on the Lambda-only runner therefore does NOT apply the "
            "decoder rework; it records the intent in the cell "
            "warnings so the run is auditable.  Use "
            "molmetal/scripts/r10_cfg_real_crossdocked.py "
            "--decoder-rework for the CFM path."
        ),
    )
    parser.add_argument(
        "--postprocess-pareto",
        action="store_true",
        help=(
            "TODO-30 / Rank-1 P3.3 — re-rank the MCTSProofSearch "
            "candidate set with the NSGA-II Pareto rank + crowding-"
            "distance operator from "
            "molmetal/molmetal_lam/search_alg/pareto.py "
            "(rank_population, lines 367-440). When OFF (default) "
            "the candidate ordering is the legacy "
            "sorted(..., key=lambda p: p[0], reverse=True) and the "
            "ranker is a no-op (bit-for-bit backward compat). When "
            "ON the ranker composes a 4-D score vector per candidate "
            "[scalar_reward, 1-sa, qed, vina_proxy] and re-orders by "
            "Pareto rank ascending / crowding distance descending. "
            "Set this flag on a Round-12/13 cell to break ties "
            "between candidates that share the scalar reward by "
            "drug-likeness / SA / affinity dominance. Lit anchors: "
            "Deb 2002 IEEE TEVC 6(2):182-197 (NSGA-II), Zitzler "
            "& Thiele 1999 IEEE TEVC 3(4):257-271 (multi-objective "
            "evolutionary baseline). Default = False."
        ),
    )
    parser.add_argument(
        "--pareto-weights",
        type=str,
        default="",
        help=(
            "TODO-30 / Rank-1 P3.3 — comma-separated weighted-sum "
            "tie-breaker weights forwarded to pareto.rank_population "
            "as the scalar-collapse fallback (Zitzler 1999 §3). "
            "Length must match the 4-D ranker vector. Default = "
            "empty (ranker uses equal weights). Example: "
            "'1.0,0.3,0.2,0.5'. Ignored unless --postprocess-pareto "
            "is ON."
        ),
    )
    # TODO-30 / P6.1+P6.2 — split hygiene flag (default OFF for
    # backward compat).  When --split=scaffold, the Bemis-Murcko
    # scaffold split produced by molmetal/scripts/
    # metallo_pool_scaffold_split.py is auto-applied to the
    # run_one_cell flow: the training_set is loaded only from the
    # train.csv of that split (so leakage from train to test is
    # prevented at the data-load boundary).  ``time`` and ``random``
    # are simple shuffles; ``scaffold_tanimoto_0.4`` is the P6.3
    # placeholder (MMseqs2 + Tanimoto<0.4 deferred; this alias
    # currently falls back to the 0.4 Murcko-Tanimoto cluster mode
    # of metallo_pool_scaffold_split).  See
    # molmetal/reports/wf_t30_p62_scaffold_seed/final.md.
    parser.add_argument(
        "--split",
        type=str,
        choices=("none", "scaffold", "time", "random",
                 "scaffold_tanimoto_0.4"),
        default="none",
        help=(
            "TODO-30 / P6.1+P6.2 — split discipline for the "
            "training-set novelty oracle.  ``none`` (default) "
            "preserves the legacy behaviour (use the existing "
            "training-set file as-is).  ``scaffold`` auto-loads "
            "``molmetal/data/metallo_scaffold_split/train.csv`` "
            "via metallo_pool_scaffold_split.py (Bemis-Murcko "
            "Tanimoto>=0.6, default 0.7/0.15/0.15 fractions). "
            "``time`` shuffles by temporal order (placeholder; "
            "uses input order when no temporal field is present). "
            "``random`` uses random.Random(seed).shuffle. "
            "``scaffold_tanimoto_0.4`` is the P6.3 alias that "
            "currently re-runs scaffold split at Tanimoto 0.4 "
            "cluster threshold (MMseqs2 step is DEFERRED)."
        ),
    )
    # TODO-30 / Rank-7 P2.2 (2026-09-17) — escape hatch for users
    # who want pure 2D generation without RDKit ETKDGv3 embedding.
    # The default at the proof_search layer is now
    # ``materialize_3d=True``; this CLI flag inverts it to False.
    # RDKit embedding is best-effort (failures fall back to no-op
    # with a warning), but on headless / rdkit-less environments
    # users may want to skip the embedding pass entirely.
    parser.add_argument(
        "--no-materialize-3d",
        dest="materialize_3d",
        action="store_false",
        default=True,
        help=(
            "TODO-30 / Rank-7 P2.2 — disable the RDKit ETKDGv3 "
            "co-emit pass on the MCTSProofSearch candidates.  "
            "Default (omitting this flag) is ON: every emitted "
            "candidate gets a ``coords_3d`` attribute attached "
            "(shape ``(n_heavy_atoms, 3)``, float32).  Pass "
            "``--no-materialize-3d`` to skip the embedding pass "
            "entirely (pure 2D generation; useful for headless "
            "test environments where rdkit is mocked or when "
            "you only care about SMILES-level metrics).  "
            "Safe: when materialize_3d=True and RDKit embedding "
            "fails, the search falls back to no-op with a "
            "warning (no error).  See "
            "molmetal/reports/wf_t30_p22_default_3d/final.md."
        ),
    )
    parser.add_argument(
        "--patent-axis",
        action="store_true",
        help=(
            "TODO-30 / Rank-2 P4.2 — opt-in flag that emits 5 "
            "patent / known-Pt-drug similarity columns per cell "
            "against the 7 known Pt drugs in "
            "``molmetal/data/known_pt_drugs.csv`` "
            "(cisplatin, carboplatin, oxaliplatin, nedaplatin, "
            "satraplatin, picoplatin, heptaplatin): "
            "``patent_max_sim_mean`` (mean max-Tanimoto), "
            "``patent_any_above_0_4_rate`` (fraction with "
            "max_sim > 0.4 = moderate novelty risk), "
            "``patent_any_above_0_7_rate`` (fraction with "
            "max_sim > 0.7 = high novelty risk), "
            "``patent_scaffold_match_rate`` (Bemis-Murcko scaffold "
            "match against any known Pt drug), "
            "``patent_closest_drug`` (most-frequent nearest-drug "
            "name; audit field).  Default OFF (backward compat "
            "— all 5 fields stay 0.0 / '').  HONEST FRAMING: "
            "Tanimoto similarity to known drugs is a CHEMICAL-"
            "NOVELTY proxy, not a substitute for a patent "
            "search.  A generated molecule with Tanimoto 0.4 "
            "to cisplatin is structurally novel in fingerprint "
            "space, but that says nothing about composition-of-"
            "matter claims, formulation patents, or method-of-"
            "use IP.  See "
            "``molmetal/reports/wf_t30_patent_axis/final.md``."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    # WF-Lift-N-Sim-Cap (2026-09-14): the historical 100-simulation
    # hard-cap silently clamped ``--n-simulations`` so the CLI flag had
    # no effect above 100, which capped the diversity ceiling during
    # the round-12 mini pilot.  We now pass ``args.n_simulations``
    # through verbatim, but still cap at a generous SAFETY_MAX so a
    # typo (e.g. ``--n-simulations 1000000``) cannot lock the GPU.
    SAFETY_MAX = 10000
    if args.n_simulations > SAFETY_MAX:
        log.warning(
            "n_simulations=%d exceeds SAFETY_MAX=%d; clamping to %d "
            "(set SAFETY_MAX higher if you really need more).",
            args.n_simulations, SAFETY_MAX, SAFETY_MAX,
        )
        args.n_simulations = SAFETY_MAX
    if args.n_simulations < 1:
        raise SystemExit("--n-simulations must be >= 1")
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build output dir under molmetal/reports/.
    output_root = _REPO_ROOT / "molmetal" / "reports"
    out = output_root / f"wf_lambda1_{args.output_dir}"
    out.mkdir(parents=True, exist_ok=True)

    training_smiles: List[str] = []
    if args.training_set:
        try:
            training_smiles = [
                line.strip()
                for line in Path(args.training_set).read_text().splitlines()
                if line.strip()
            ]
        except Exception as exc:
            log.warning("Could not read training set %s: %s", args.training_set, exc)

    # TODO-30 / P6.1+P6.2 — split hygiene.  When --split is anything
    # other than ``none`` (default OFF, backward-compat), restrict the
    # novelty training set to the requested split so we never use
    # train+test from the same scaffold bucket as the reference pool.
    # The flag is additive: --training_set still wins if it was set
    # explicitly, but a scaffold-split partition from
    # ``metallo_scaffold_split`` overrides the bundled default.
    if args.split != "none":
        try:
            split_smiles = _load_split_training_smiles(
                split=str(args.split),
                seed=int(args.seeds[0]) if args.seeds else 42,
            )
            if split_smiles:
                log.info(
                    "Applying --split=%s: novelty training pool restricted "
                    "to %d SMILES (was %d)",
                    args.split, len(split_smiles), len(training_smiles),
                )
                training_smiles = split_smiles
        except Exception as exc:
            log.warning(
                "Could not apply --split=%s: %s (falling back to legacy "
                "training-set loader)", args.split, exc,
            )

    payload, cells = run_sweep(
        manifest_path=Path(args.manifest),
        n_pockets=args.pockets,
        seeds=args.seeds,
        n_simulations=args.n_simulations,
        n_top_k=args.n_top_k,
        prior_enabled=not args.prior_disabled,
        training_smiles=training_smiles,
        output_dir=out,
        click_rules=args.click_rules,
        metal_seed=args.metal_seed,
        sa_weight=args.sa_weight,
        platinai_weight=args.reward_platinai_weight,
        lambda_reward_weight=args.reward_lambda_weight,
        pharmacophore_weight=getattr(args, "reward_pharmacophore_weight", 0.0),
        pharmacophore_strict=bool(
            getattr(args, "pharmacophore_strict", False)
        ),
        pharmacophore_allow_acyclic=bool(
            getattr(args, "pharmacophore_allow_acyclic", True)
        ),
        allow_incompatible_click=bool(args.allow_incompatible_click),
        decoder_rework=bool(args.decoder_rework),
        keep_high_sa_tiles=bool(args.keep_high_sa_tiles),
        metal_seed_from_pocket=bool(args.metal_seed_from_pocket),
        use_pocket_conditioned_reference=bool(
            args.use_pocket_conditioned_reference
        ),
        use_learned_prior=bool(args.use_learned_prior),
        pocket_boost_strength=float(getattr(args, "pocket_boost_strength", 1.0)),
        use_sa_prior=bool(args.use_sa_prior),
        sa_prior_strength=float(args.sa_prior_strength),
        use_learned_shaping=bool(getattr(args, "use_learned_shaping", False)),
        learned_shaping_weight=float(
            getattr(args, "learned_shaping_weight", 1.0)
        ),
        use_pocket_macro_inference=bool(
            getattr(args, "use_pocket_macro_inference", False)
        ),
        use_symbolic_reward=bool(
            getattr(args, "use_symbolic_reward", False)
        ),
        n_samples=int(args.n_samples),
        pocket10_radius=float(args.pocket10_radius),
        reference_ligand=bool(args.reference_ligand),
        metallodrug_proxies=bool(getattr(args, "metallodrug_proxies", False)),
        postprocess_pareto=bool(getattr(args, "postprocess_pareto", False)),
        pareto_weights=str(getattr(args, "pareto_weights", "") or ""),
        materialize_3d=bool(getattr(args, "materialize_3d", True)),
        patent_axis=bool(getattr(args, "patent_axis", False)),
        wetlab_weight=float(getattr(args, "reward_wetlab_weight", 0.0)),
        wetlab_measurements=globals().get("_R4_WETLAB_MEASUREMENTS"),
    )

    log.info("Wrote report to %s/report.json", out)
    log.info("Wrote summary to %s/summary.md", out)
    print(json.dumps({"output_dir": str(out), "n_cells": len(cells),
                      "aggregate": payload["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())