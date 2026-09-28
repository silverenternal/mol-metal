"""r4_lib.constants — Pure-data module (no heavy imports).

This module bundles every static lookup table that used to live as
module-level constants in the original ``r4_lambda_only_run.py``:

* :data:`CLICK_RULE_ALIASES` — friendly spellings (``all-5``,
  ``CuAAC``, ``thiol-ene``, ...) expanded to canonical click-rule
  names.  Drives the F2-B scaffold-aware alias resolver.
* :data:`METAL_SEED_SMILES` — bare-metal + terminal-alkyne SMILES
  for each canonical metal seed (cisplatin / carboplatin /
  oxaliplatin / nedaplatin / ru_arene / ir_cp_star).  Consumed by
  :mod:`.runner` for ``--metal-seed``.
* :data:`_POCKET_METAL_BUCKETS` +
  :data:`POCKET_DERIVED_METAL_SEED_FALLBACK` — coarse (3-bucket +
  fallback) pocket → metal_seed mapping used by
  :func:`.rules.pocket_derived_metal_seed`.
* :data:`CLICK_RULE_WEIGHT` / :data:`METAL_PRIOR_WEIGHT` —
  per-channel reward weights (WF-Lambda-Fix-Singleton Fix 2; raised
  click weight from 1.0 → 0.5, dropped metal weight from 1.0 → 0.5
  to balance the two contributions).
* :data:`_R17_LEGACY_DEFAULTS` / :data:`_R17_SMOKE_PRESET` /
  :data:`_R17_PAPER_GRADE_PRESET` — the three preset configurations
  that drive ``--smoke`` / ``--r17-paper-grade`` (M-R17-A2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Repo-root helper — used to build default manifest / scaffold-split paths.
# This mirrors the original ``_REPO_ROOT`` derivation in
# r4_lambda_only_run.py: ``molmetal/scripts/r4_lib/`` is 4 levels deep
# from the repo root (mol-metal/), so we walk 3 parents.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent


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
    # Carboplatin (cis-[Pt(NH3)2(cyclobutane-1,1-dicarboxylate)]) —
    # Pt(II) square-planar with the same bare-metal + alkyne-handle
    # contract as cisplatin (the cyclobutane-dicarboxylate is left to
    # the search to assemble via the AquaExchange + MetalLigandExchange
    # SMARTS — see wf_lr17_b2_done.md §1).
    "carboplatin": "[Pt]C#C",
    # Oxaliplatin ((1R,2R)-diaminocyclohexane + oxalate) — Pt(II)
    # square-planar; same bare-metal + alkyne contract.
    "oxaliplatin": "[Pt]C#C",
    # Nedaplatin (cis-[Pt(NH3)2(glycolate)]) — Pt(II) square-planar;
    # same bare-metal + alkyne contract.
    "nedaplatin": "[Pt]C#C",
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
# TODO M-R17-A2 / WF-Paper-Sweep-Spec (2026-09-25) — preset
# constants and the resolution helper.  Both --smoke and
# --r17-paper-grade map to a fixed (pockets, seeds, n_simulations,
# manifest) tuple; explicit user flags take precedence over the
# preset values.  See ``molmetal/reports/wf_paper_sweep_spec/final.md``
# for the full R17 spec (runtime budgets, output paths,
# cross-references).
# ---------------------------------------------------------------------------
_R17_LEGACY_DEFAULTS: Dict[str, Any] = {
    "pockets":        10,
    "seeds":          [42, 0, 1234],
    "n_simulations":  200,
    "manifest":       str(
        _REPO_ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv"
    ),
}
_R17_SMOKE_PRESET: Dict[str, Any] = {
    "pockets":        10,
    "seeds":          [42],
    "n_simulations":  20,
    "manifest":       str(
        _REPO_ROOT / "molmetal" / "data" / "test_010_019_subset_manifest.csv"
    ),
}
_R17_PAPER_GRADE_PRESET: Dict[str, Any] = {
    "pockets":        100,
    "seeds":          [42, 0, 1234],
    "n_simulations":  1000,
    "manifest":       str(
        _REPO_ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv"
    ),
}
