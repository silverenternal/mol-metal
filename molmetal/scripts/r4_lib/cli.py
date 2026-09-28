"""r4_lib.cli — argparse + main entry point for the Lambda-only runner.

This module owns:

* :func:`_build_argparser` — every CLI flag the harness exposes
  (``--smoke`` / ``--r17-paper-grade`` / ``--n-simulations`` / all the
  WF-Phase-X ablation toggles).
* :func:`_resolve_paper_sweep_preset` — applies the
  ``--smoke`` / ``--r17-paper-grade`` preset values to a parsed
  argparse namespace.
* :func:`main` — the entry point invoked by ``r4_lambda_only_run.py``
  (and by ``python -m molmetal.scripts.r4_lib.cli``).

Originally lived at lines 4161-5267 of ``r4_lambda_only_run.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .constants import (
    _R17_LEGACY_DEFAULTS,
    _R17_PAPER_GRADE_PRESET,
    _R17_SMOKE_PRESET,
)
from .runner import _load_split_training_smiles, run_sweep

log = logging.getLogger("wf_lambda1")


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
    # TODO M-R17-A2 / WF-Paper-Sweep-Spec — the default for --pockets
    # is None so the two preset flags (--smoke / --r17-paper-grade)
    # below can resolve them unambiguously.  The legacy default of
    # 10 is preserved as the resolution fallback when neither preset
    # is active.  See _resolve_paper_sweep_preset() in main().
    parser.add_argument(
        "--pockets",
        type=int,
        default=None,
        help=(
            "Number of test pockets (first N rows of the manifest "
            "CSV).  Defaults to 10 when neither --smoke nor "
            "--r17-paper-grade is set.  See those flags for preset "
            "overrides."
        ),
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
        default=None,
        help=(
            "Random seeds (space-separated).  Default = [42, 0, 1234] "
            "when neither --smoke nor --r17-paper-grade is set; the "
            "presets may override (--smoke forces [42], "
            "--r17-paper-grade forces [42, 0, 1234]).  Backward "
            "compat for the dispatch CLI in "
            "TODO-31 §L-R17-B2 verify gate; --seed (singular) is "
            "the friendlier alias added below."
        ),
    )
    # TODO L-R17-B2 / WF-R17-Lambda-Flagfix (2026-09-25) —
    # singular --seed alias for the dispatch CLI (``--seed 42``)
    # which is the form natural to a one-seed smoke / pilot run.
    # When both --seed and --seeds are supplied the --seeds list
    # wins (the more-specific CLI token takes precedence); when
    # only --seed is supplied it is wrapped into a one-element
    # list and forwarded to the rest of the pipeline.  This is
    # strictly additive — existing --seeds behaviour is preserved
    # bit-for-bit when --seed is omitted.
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "TODO L-R17-B2 / WF-R17-Lambda-Flagfix — singular alias "
            "for --seeds (accepts a single int).  When --seeds is "
            "also supplied the list wins; when only --seed is "
            "supplied it is wrapped into [seed] before forwarding "
            "to run_sweep.  Backward compatible: omitting both "
            "preserves the preset-driven default ([42] for --smoke, "
            "[42, 0, 1234] for --r17-paper-grade, [42, 0, 1234] for "
            "the legacy default)."
        ),
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=None,
        help=(
            "MCTS simulation budget per cell.  Default = 200 when "
            "neither --smoke nor --r17-paper-grade is set; the "
            "presets may override (--smoke forces 20, "
            "--r17-paper-grade forces 1000).  WF-Lift-N-Sim-Cap "
            "(2026-09-14) removed the historical 100-cap; the CLI "
            "flag now takes effect verbatim up to a 10000 safety "
            "maximum (to prevent a typo locking the GPU)."
        ),
    )
    parser.add_argument(
        "--n-top-k",
        type=int,
        default=20,
        help="Number of top candidates returned per cell.",
    )
    # TODO L-R17-B2 / WF-R17-Lambda-Flagfix (2026-09-25) — the
    # ``--output-dir`` flag is now OPTIONAL (default None) so users
    # can opt into the new ``--out-dir ABS_PATH`` flag without
    # supplying both.  Exactly one of ``--output-dir NAME`` (the
    # legacy sub-directory prefix) or ``--out-dir ABS_PATH``
    # (bypass the prefix) must be set; if both are supplied
    # ``--out-dir`` wins (the absolute path is the safer, more
    # explicit fallback) and ``--output-dir`` is logged for audit.
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Sub-directory under molmetal/reports/. The full report "
            "path is molmetal/reports/wf_lambda1_<output-dir>/. "
            "Default = None; one of --output-dir / --out-dir must be "
            "supplied.  When both are set, --out-dir wins and "
            "--output-dir is logged for audit."
        ),
    )
    # WF-R17-Lambda-Flagfix — the new ``--out-dir`` flag bypasses
    # the ``wf_lambda1_<NAME>/`` prefix and writes directly to
    # ``ABS_PATH``.  Useful when dispatching to an external report
    # directory that already lives outside the wf_lambda1_* namespace
    # (e.g. wf_paper_grade_300cells/, wf_lr17_b2_pilot/, etc.).
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "TODO L-R17-B2 / WF-R17-Lambda-Flagfix — absolute path to "
            "the report output directory (bypasses the "
            "wf_lambda1_<NAME>/ prefix).  When set, the report is "
            "written directly to <out-dir>/ and the directory is "
            "created if it does not exist.  Wins over --output-dir "
            "when both are supplied; --output-dir is logged for "
            "audit.  Example: --out-dir molmetal/reports/wf_paper_grade_300cells"
        ),
    )
    # TODO M-R17-A2 / WF-Paper-Sweep-Spec — default to None so the
    # --smoke / --r17-paper-grade presets can resolve an alternate
    # manifest (test_010_019_subset_manifest.csv for smoke;
    # crossdocked100_manifest.csv for paper-grade) without colliding
    # with explicit --manifest.  Legacy default = crossdocked100.
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Path to the manifest CSV.  Default = "
            "molmetal/data/crossdocked100_manifest.csv (legacy) when "
            "neither --smoke nor --r17-paper-grade is set; "
            "--smoke forces molmetal/data/test_010_019_subset_manifest.csv "
            "(10 novel pockets); --r17-paper-grade forces "
            "molmetal/data/crossdocked100_manifest.csv (100 pockets)."
        ),
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
    # TODO L-R17-B2 / WF-R17-Lambda-Flagfix (2026-09-25) — expanded
    # --metal-seed choices to include carboplatin / oxaliplatin /
    # nedaplatin (real Pt(II) drugs) so the dispatch CLI in
    # TODO-31 §L-R17-B2 verify gate (``--metal-seed cisplatin``) AND
    # future R17 expansion cells (``--metal-seed carboplatin`` /
    # ``oxaliplatin`` / ``nedaplatin``) all work without code surgery.
    # The expanded METAL_SEED_SMILES table is registered in the
    # _register_metal_seed_aliases() block below; bare-metal + alkyne
    # handle is the canonical pattern so the MCTS root chemistry
    # contract is preserved.
    parser.add_argument(
        "--metal-seed",
        nargs=None,
        default=None,
        choices=[
            None,
            "cisplatin", "carboplatin", "oxaliplatin", "nedaplatin",
            "ru_arene", "ir_cp_star",
        ],
        help=(
            "Optional metal-containing root scaffold for MCTS. When set, "
            "the search is seeded with a canonical cisplatin / "
            "carboplatin / oxaliplatin / nedaplatin / Ru-arene / Ir-Cp* "
            "molecule instead of the pocket reference ligand, so "
            "metal_compliance_rate becomes measurable.  Default = None "
            "(use pocket reference, falling back to Cl[Pt]Cl).  Added "
            "by WF-R17-Lambda-Flagfix: cisplatin + carboplatin + "
            "oxaliplatin + nedaplatin are all Pt(II) square-planar "
            "chemistries with the same bare-metal + alkyne-handle "
            "contract; the MCTS root SMILES is selected from the "
            "METAL_SEED_SMILES table."
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
    # TODO M-R17-A2 / WF-Paper-Sweep-Spec (2026-09-25) — convenience
    # preset flags.  These resolve the (pockets, seeds, n-simulations,
    # manifest) tuple to the two canonical R17 configurations when
    # the user has not already passed an explicit override.  See
    # ``_resolve_paper_sweep_preset()`` in main() for the resolution
    # logic and ``molmetal/reports/wf_paper_sweep_spec/final.md``
    # for the full R17 spec (runtime budgets, output paths,
    # cross-references).
    #
    # --smoke: 10 pockets × 1 seed × n_sim=20 (~5 min CPU).
    # Useful for quick verification during development; explicitly
    # NOT a paper-grade measurement cell.  Loads the
    # test_010..test_019 subset manifest (10 novel pockets, distinct
    # from the test_000..test_009 pilot cohort so singleton-collapse
    # patterns are easy to spot).
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "TODO M-R17-A2 — apply the R17 smoke preset: "
            "10 pockets × 1 seed × n_sim=20 (~5 min CPU).  Loads "
            "the test_010..test_019 subset manifest.  Used for "
            "quick verification during development; explicitly NOT "
            "a paper-grade measurement.  Compatible with explicit "
            "--pockets / --seeds / --n-simulations / --manifest "
            "overrides (any non-None value wins)."
        ),
    )
    # --r17-paper-grade: 100 pockets × 3 seeds × n_sim=1000
    # (~6h on GPU CFM / ~50min on CPU Lambda-only).  Matches the
    # full Round-13 / TODO-14 spec at the top-journal protocol.
    # Outputs to ``molmetal/reports/wf_paper_grade_300cells/`` (300
    # cells = 100 pockets × 3 seeds).  Forces the
    # crossdocked100_manifest.csv; the legacy default --manifest
    # value is also the crossdocked100 path so the flag is
    # idempotent on that axis but pins the choice for audit
    # clarity.
    parser.add_argument(
        "--r17-paper-grade",
        action="store_true",
        help=(
            "TODO M-R17-A2 — apply the R17 paper-grade preset: "
            "100 pockets × 3 seeds × n_sim=1000 (~6h on GPU CFM / "
            "~50min on CPU Lambda-only).  Outputs to "
            "molmetal/reports/wf_paper_grade_300cells/.  Loads "
            "crossdocked100_manifest.csv.  Matches the full "
            "Round-13 / TODO-14 spec at the top-journal protocol. "
            "Compatible with explicit --pockets / --seeds / "
            "--n-simulations / --manifest overrides (any non-None "
            "value wins).  Note: --smoke and --r17-paper-grade are "
            "mutually exclusive; if both are set, --r17-paper-grade "
            "wins (the larger configuration is the safer fallback)."
        ),
    )
    # TODO L-R17-B2 / WF-R17-Lambda-Flagfix (2026-09-25) — engine
    # selector.  r4_lambda_only_run.py is the *Lambda-only* runner,
    # so ``--engine lambda`` is the legacy default (no-op).  The
    # other two tokens are accepted for dispatch-CLI ergonomics
    # (the dispatch in TODO-31 §L-R17-B2 verify gate uses
    # ``--engine both``); they are wired as documented fallbacks:
    # * ``--engine cfm`` — logs a warning and falls back to Lambda
    #   (the CFM runner is r10_cfg_real_crossdocked.py, not this
    #   script).  Future work: wire CFM via TODO-21 coupling adapter.
    # * ``--engine both`` — also logs a warning and falls back to
    #   Lambda; the dual-engine ablation is captured by running
    #   this script AND r10_cfg_real_crossdocked.py back-to-back.
    parser.add_argument(
        "--engine",
        type=str,
        default="lambda",
        choices=["lambda", "cfm", "both"],
        help=(
            "TODO L-R17-B2 / WF-R17-Lambda-Flagfix — engine selector. "
            "Default = 'lambda' (legacy Lambda-only behaviour, this "
            "script's native mode).  'cfm' / 'both' are accepted for "
            "dispatch-CLI ergonomics; this runner falls back to Lambda "
            "in both cases with a warning (the CFM runner is "
            "r10_cfg_real_crossdocked.py, which is dispatched "
            "separately).  Future work: wire CFM via TODO-21 coupling "
            "adapter."
        ),
    )
    return parser


def _resolve_paper_sweep_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply --smoke / --r17-paper-grade preset overrides to ``args``.

    Resolution order (most-specific wins):
      1. Explicit user-supplied flag (any non-None value) wins.
      2. --r17-paper-grade preset value (if --r17-paper-grade is set).
      3. --smoke preset value (if --smoke is set).
      4. Legacy default (preserves pre-R17 behaviour when neither
         preset is active).

    If both --smoke and --r17-paper-grade are set, --r17-paper-grade
    wins (the larger configuration is the safer fallback) and a
    warning is logged.
    """
    if args.r17_paper_grade and args.smoke:
        log.warning(
            "Both --smoke and --r17-paper-grade set; "
            "--r17-paper-grade wins (safer fallback)."
        )

    if args.r17_paper_grade:
        preset = _R17_PAPER_GRADE_PRESET
    elif args.smoke:
        preset = _R17_SMOKE_PRESET
    else:
        preset = _R17_LEGACY_DEFAULTS

    for field_name, preset_value in (
        ("pockets",       preset["pockets"]),
        ("seeds",         preset["seeds"]),
        ("n_simulations", preset["n_simulations"]),
        ("manifest",      preset["manifest"]),
    ):
        current = getattr(args, field_name)
        if current is None:
            setattr(args, field_name, preset_value)
        else:
            log.info(
                "r17-preset: keeping user-supplied --%s=%s "
                "(would have applied preset value %s)",
                field_name.replace("_", "-"),
                current,
                preset_value,
            )
    # Surface the resolved preset for downstream logging.
    args.r17_preset_active = (
        "paper_grade" if args.r17_paper_grade
        else "smoke" if args.smoke
        else "legacy"
    )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    # TODO L-R17-B2 / WF-R17-Lambda-Flagfix (2026-09-25) — wrap the
    # singular --seed into the --seeds list BEFORE the preset
    # resolution runs, so the wrap takes precedence over the legacy
    # default ([42, 0, 1234]).  This is the friendlier dispatch-CLI
    # surface: ``--seed 42`` should produce 1 seed, not 3.
    if args.seed is not None and not args.seeds:
        args.seeds = [args.seed]
        log.info("--seed=%d wrapped into --seeds=%s", args.seed, args.seeds)
    elif args.seed is not None and args.seeds:
        log.info(
            "--seed=%d supplied alongside --seeds=%s; --seeds wins "
            "(the more-specific CLI token takes precedence).",
            args.seed, args.seeds,
        )
    # TODO M-R17-A2 / WF-Paper-Sweep-Spec (2026-09-25) — apply the
    # --smoke / --r17-paper-grade preset overrides BEFORE the
    # n-simulations safety cap, so the resolved value flows through
    # the same clamping path.  See _resolve_paper_sweep_preset().
    args = _resolve_paper_sweep_preset(args)
    log.info(
        "R17 preset: %s (pockets=%d seeds=%s n_simulations=%d manifest=%s)",
        args.r17_preset_active,
        args.pockets,
        args.seeds,
        args.n_simulations, args.manifest,
    )
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

    # TODO L-R17-B2 / WF-R17-Lambda-Flagfix (2026-09-25) —
    # reconcile the remaining new flags into the legacy kwargs
    # surface so run_sweep + the rest of the pipeline never sees
    # them.
    #
    # 1. --engine: log + fall back to lambda when 'cfm' / 'both' is
    #    requested (this runner is Lambda-only; the CFM runner is
    #    r10_cfg_real_crossdocked.py, dispatched separately per the
    #    TODO-31 §L-R17-B2 verify gate).
    if getattr(args, "engine", "lambda") != "lambda":
        log.warning(
            "--engine=%s requested; r4_lambda_only_run.py is the "
            "Lambda-only runner, falling back to Lambda.  Dispatch "
            "r10_cfg_real_crossdocked.py separately for the CFM "
            "side (see TODO-31 §L-R17-B2 verify gate).",
            args.engine,
        )
        args.engine = "lambda"
    # 2. --output-dir / --out-dir mutual exclusion: exactly one must
    #    be supplied.  --out-dir (absolute path) wins over
    #    --output-dir (sub-directory name with wf_lambda1_ prefix).
    if not args.out_dir and not args.output_dir:
        raise SystemExit(
            "One of --output-dir NAME or --out-dir ABS_PATH must be "
            "supplied (see --help).  Use --output-dir for the legacy "
            "wf_lambda1_<NAME>/ prefix; use --out-dir for an "
            "absolute path that bypasses the prefix."
        )
    if args.out_dir:
        out = Path(args.out_dir)
        if args.output_dir:
            log.info(
                "Both --output-dir=%s and --out-dir=%s supplied; "
                "--out-dir wins (absolute path is the safer, more "
                "explicit fallback).  --output-dir=%s logged for "
                "audit.", args.output_dir, args.out_dir, args.output_dir
            )
    else:
        # Build output dir under molmetal/reports/.
        output_root = Path(__file__).resolve().parent.parent.parent.parent / "molmetal" / "reports"
        out = output_root / f"wf_lambda1_{args.output_dir}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out)

    training_smiles: list[str] = []
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

    # Surface the parsed ``args`` namespace into the runner module so
    # :func:`run_one_cell` can read CLI-only knobs (``fg_veto_strict``,
    # ``wetlab_input``) via ``globals().get("args")``.
    from . import runner as _runner_mod
    _runner_mod.args = args  # type: ignore[attr-defined]

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
        wetlab_measurements=_runner_mod._R4_WETLAB_MEASUREMENTS,
    )

    log.info("Wrote report to %s/report.json", out)
    log.info("Wrote summary to %s/summary.md", out)
    print(json.dumps({"output_dir": str(out), "n_cells": len(cells),
                      "aggregate": payload["aggregate"]}, indent=2))
    return 0
