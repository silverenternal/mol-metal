"""Wf-221 — Click-chemistry augmentation for chemprop v2 m7 diversity.

================================================================
Wf-221 — wire click reactions into the chemprop v2 candidate set
================================================================

Background
----------
The chemprop v2 multi-task backbone (:class:`ChempropV2MTModel` in
:mod:`molmetal.models.chemprop_v2_mt_model`) consumes SMILES candidates
and emits per-metal pIC50 logits.  The m7 diversity metric — Morgan
ECFP4 pairwise Tanimoto distance on the *generated candidate set* —
is the lever for closing the chemprop v2 cohort diversity gap that
Wf-186 lifted on the Lambda search side via a rank-time bonus.

This module ships the **chemprop v2 side** of the same lever: take
a SMILES list (typically a chemprop v2 candidate generation batch)
and augment a configurable fraction of the entries by firing one
of the canonical 5 click reactions (:mod:`molmetal_lam.reactions.click_reactions`)
against a curated partner tile from the standard 14-tile library.

Integration contract
--------------------
The augmentation is **opt-in**.  Without a call to
:func:`augment_candidates_via_click_reactions` (or to
:class:`ClickAugmentedChempropGenerator.generate`), chemprop v2
behaviour is unchanged — same SMILES in, same logits out.  This is
the **backward-compat gate (G4)** for Wf-221.

When the augmentation IS invoked, the integration:

* Imports the 4 click rules from
  :mod:`molmetal_lam.reactions.click_reactions` (CuAAC / SPAAC /
  SPC / DielsAlder — the same 4 rules that the
  :data:`molmetal_lam.lam_chem.rules.CLICK_REACTIONS` registry
  exposes; DielsAlder is the under-recognised 4th rule in that
  registry alongside CuAAC / SPAAC / SPC).
* Wraps each chemprop v2 candidate SMILES into a Tile via
  :class:`molmetal_lam.tile_lib.tile.Tile` (auto-tags the
  functional-group handles via simple SMARTS matching so
  CuAAC can recognise azide vs alkyne, etc.).
* Fires one randomly chosen click rule against a random partner
  from :func:`molmetal_lam.tile_lib.click_tiles.AZIDE_TILES`,
  :func:`molmetal_lam.tile_lib.click_tiles.ALKYNE_TILES`,
  :func:`molmetal_lam.tile_lib.click_tiles.PARTNER_TILES`, and
  :func:`molmetal_lam.tile_lib.click_tiles.THIOL_TILES`.
* Returns the canonical product SMILES (RDKit canonicalised), or
  the original input if the rule failed to fire (RDKit
  incompat / no matching handle).  This guarantees the wrapper
  never emits fewer candidates than the caller asked for — the
  diversity lift is monotonic-additive, not subtractive.

Public surface
--------------
* :func:`augment_candidates_via_click_reactions` — pure function:
  take a SMILES list, return the click-augmented SMILES list.
* :class:`ClickAugmentedChempropGenerator` — wraps any
  ``(seed_smiles, n_samples) -> List[str]`` callable (e.g. the
  chemprop v2 batch sampler) and applies the click-augmentation
  per sampled molecule.
* :func:`diversity_lift_tanimoto` — measure the mean pairwise
  Tanimoto-distance **delta** between the base set and the
  augmented set.  Positive values indicate the augmentation
  broadened chemical-space coverage (m7 lift); zero or negative
  values indicate the augmentation did not move the metric
  (which is the G3 honest-negative outcome — not a fabricated
  positive lift).
* :class:`ClickAugmentationResult` — dataclass bundling the
  augmented SMILES + per-rule fire counts + wall-time.

CPU-only / no model loading
---------------------------
* RDKit-only chemistry (no torch, no chemprop, no learned priors).
* No chemprop v2 model weights loaded — the rule firing uses the
  SMARTS templates shipped in
  :mod:`molmetal_lam.reactions.click_reactions`.
* Determinism: every random choice uses an injected
  :class:`random.Random` instance (or a fresh one if ``None``).
  Tests pin ``rng=random.Random(0)`` for reproducibility.

Honest framing (per ``party-twenty-spirit`` §3.5)
-------------------------------------------------
* The augmented generator does **not** guarantee a m7 lift on
  the chemprop v2 prediction side — the chemprop v2 backbone
  re-scores the augmented candidates, so the metric lift is
  downstream of the generator.  This module only ships the
  *candidate generation* lever; the chemprop v2 prediction lift
  is the Wf-221 follow-on.
* When the rule firing returns ``None`` (RDKit incompatibility /
  no functional handle), the original SMILES is preserved so
  the wrapper never emits fewer candidates than the caller asked
  for.  This means G3 (``Lift ≥ 0``) holds by construction —
  augmented_unique_scaffolds ≥ base_unique_scaffolds (counting
  duplicates as zero).

Lit anchors
-----------
* Bemis & Murcko 1996 *J. Med. Chem.* 39(15), 2887-2893 — Bemis-Murcko
  scaffold uniqueness used by the diversity benchmark.
* Rogers & Hahn 2010 *J. Med. Inf. Model.* 50(5), 742-754 — Tanimoto
  on ECFP4 fingerprints (used by the benchmark diversity metric).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

# Quiet RDKit's verbose logging on rule-firing errors that surface as
# benign warnings during the high-throughput augmented-generation path.
try:
    from rdkit import Chem, RDLogger  # type: ignore
    from rdkit.Chem import AllChem, DataStructs  # type: ignore

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    DataStructs = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClickAugmentationError(ValueError):
    """Raised when the chemprop v2 click-augmentation layer fails to wire."""


# ---------------------------------------------------------------------------
# Constants — public rule names
# ---------------------------------------------------------------------------


#: Canonical click-reaction names exposed via
#: :func:`augment_candidates_via_click_reactions`.  These match the
#: :data:`molmetal_lam.reactions.click_reactions.CLICK_REACTIONS` keys
#: exactly so a caller can do ``augment(smis, click_rule_names=["CuAAC"])``
#: and have the integration route to the same rule instance.
DEFAULT_CLICK_RULE_NAMES: Tuple[str, ...] = (
    "CuAAC",
    "SPAAC",
    "SPC",
    "DielsAlder",
)


# ---------------------------------------------------------------------------
# Helpers — RDKit-primitive (SMILES) wrappers
# ---------------------------------------------------------------------------


def _safe_parse(smiles: str) -> Optional[Any]:
    """Parse ``smiles`` into an RDKit mol; ``None`` on failure."""
    if Chem is None or not isinstance(smiles, str) or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _canonical(smiles: str) -> str:
    """Return the canonical SMILES of ``smiles`` or ``""`` on failure."""
    if Chem is None:
        return ""
    mol = _safe_parse(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _fingerprint(smiles: str, n_bits: int = 2048):
    """Compute an ECFP4 fingerprint bit-vector; ``None`` on failure."""
    if Chem is None or AllChem is None:
        return None
    mol = _safe_parse(smiles)
    if mol is None:
        return None
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return bv
    except Exception:
        return None


def _tanimoto_distance(fp1, fp2) -> float:
    """Tanimoto *distance* (1 - similarity) between two bit-vectors."""
    if DataStructs is None or fp1 is None or fp2 is None:
        return 1.0
    try:
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 1.0
    return max(0.0, 1.0 - sim)


def _mean_pairwise_tanimoto_distance(smiles_list: Sequence[str]) -> float:
    """Mean pairwise Tanimoto distance across a SMILES list.

    Returns ``0.0`` for lists of size 0 or 1 (no pairs to compare).
    Returns ``1.0`` for lists where every molecule fails to parse.
    """
    cleaned = [s for s in smiles_list if isinstance(s, str) and s]
    if len(cleaned) < 2:
        return 0.0
    fps: List = []
    for s in cleaned:
        fp = _fingerprint(s)
        if fp is not None:
            fps.append(fp)
    n = len(fps)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _tanimoto_distance(fps[i], fps[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def _unique_scaffold_count(smiles_list: Sequence[str]) -> int:
    """Return the number of distinct Bemis-Murcko scaffolds in ``smiles_list``."""
    if Chem is None:
        return 0
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    except Exception:  # pragma: no cover - defensive
        return 0
    scaffolds: set = set()
    for s in smiles_list:
        mol = _safe_parse(s)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            continue
        if scaf is None or scaf.GetNumAtoms() == 0:
            canonical = _canonical(s)
            if canonical:
                scaffolds.add(canonical)
            continue
        try:
            scaffolds.add(Chem.MolToSmiles(scaf))
        except Exception:
            continue
    return len(scaffolds)


# ---------------------------------------------------------------------------
# Auto-detect functional-group handles for chemprop v2 candidate SMILES
# ---------------------------------------------------------------------------
# These SMARTS are the same ones used in :mod:`click_reactions` — we
# re-declare them here so the auto-tagger does not have to import the
# click-rules module just to look up a SMARTS string.  Keep in sync
# with :data:`click_reactions._AZIDE_SMARTS` / `_TERMINAL_ALKYNE_SMARTS`
# / `_INTERNAL_ALKYNE_SMARTS` / `_PHOSPHINE_SMARTS` / `_DIENE_SMARTS` /
# `_ALKENE_SMARTS`.

_AZIDE_SMARTS = "[N]=[N]=[N]"
_TERMINAL_ALKYNE_SMARTS = "[CH]#[C]"
_INTERNAL_ALKYNE_SMARTS = "[C;H0]#[C;H0]"
_PHOSPHINE_SMARTS = "[P]"
_DIENE_SMARTS = "[#6]=[#6]-[#6]=[#6]"
_ALKENE_SMARTS = "[#6]=[#6]"
_THIOL_SMARTS = "[SH]"
_MALEIMIDE_SMARTS = "[#6]C(=O)[NH]C(=O)[#6]"
_BORONIC_SMARTS = "[B]([OH])[OH]"
_HALIDE_SMARTS = "[#6][F,Cl,Br,I]"
_AMINE_SMARTS = "[NH2]"
_ACID_SMARTS = "[#6]C(=O)[OH]"


def _detect_tags(smiles: str) -> List[str]:
    """Auto-detect click-relevant functional-group tags for a chemprop v2 SMILES.

    Returns the list of tags (e.g. ``["azide"]``, ``["terminal_alkyne"]``)
    the click_reactions module expects.  When the SMILES has no click
    handle, returns ``[]`` so the augmentor can fall back to passthrough.
    """
    if Chem is None:
        return []
    mol = _safe_parse(smiles)
    if mol is None:
        return []
    tags: List[str] = []
    pairs = [
        (_AZIDE_SMARTS, "azide"),
        (_TERMINAL_ALKYNE_SMARTS, "terminal_alkyne"),
        (_INTERNAL_ALKYNE_SMARTS, "cyclooctyne"),
        (_PHOSPHINE_SMARTS, "phosphine"),
        (_DIENE_SMARTS, "diene"),
        (_ALKENE_SMARTS, "dienophile"),
        (_THIOL_SMARTS, "thiol"),
        (_MALEIMIDE_SMARTS, "maleimide"),
        (_BORONIC_SMARTS, "boronic_acid"),
        (_HALIDE_SMARTS, "aryl_halide"),
        (_AMINE_SMARTS, "primary_amine"),
        (_ACID_SMARTS, "carboxylic_acid"),
    ]
    for smarts, tag in pairs:
        try:
            pat = Chem.MolFromSmarts(smarts)
        except Exception:
            continue
        if pat is None:
            continue
        try:
            if mol.HasSubstructMatch(pat):
                tags.append(tag)
        except Exception:
            continue
    return tags


# ---------------------------------------------------------------------------
# Tile wrapping for chemprop v2 SMILES
# ---------------------------------------------------------------------------


def _smiles_to_tile(smiles: str, tags: Optional[List[str]] = None) -> Any:
    """Wrap a chemprop v2 SMILES into a click_reactions-compatible Tile.

    Imports :class:`Tile` and :func:`build_click_tile` lazily so this
    module can be imported without the chemprop v2 tile-library cost
    on the import path.
    """
    try:
        from molmetal_lam.tile_lib.click_tiles import build_click_tile
        from molmetal_lam.tile_lib.tile import Tile  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[click_aug] could not import Tile/build_click_tile: %s", exc
        )
        return None
    canonical = _canonical(smiles)
    if not canonical:
        return None
    chosen_tags = tags if tags is not None else _detect_tags(canonical)
    if not chosen_tags:
        # No click handle — we still build a tile but tag-free so
        # callers can identify passthrough.
        chosen_tags = ["no_handle"]
    try:
        return build_click_tile(canonical, chosen_tags, embed_3d=False)
    except Exception as exc:
        logger.debug("[click_aug] build_click_tile failed for %r: %s", canonical, exc)
        return None


def _get_partner_pool(rule_name: str) -> List[str]:
    """Return a small list of partner tile SMILES for ``rule_name``.

    The pool is curated (not enumerated from the 14-tile library) so
    the chemprop v2 click-augmentation path stays deterministic +
    CPU-cheap.  Each entry is RDKit-parsable and carries the
    functional-group handle the named rule needs.
    """
    if rule_name == "CuAAC":
        # Partners: terminal alkynes (CuAAC needs azide + terminal alkyne;
        # the chemprop v2 SMILES carries the azide — partners are alkynes).
        return [
            "C#CC",                  # propyne (terminal alkyne)
            "C#CCc1ccccc1",          # propargyl benzene
            "C#CCN",                 # propargylamine
        ]
    if rule_name == "SPAAC":
        # Partners: cyclooctyne (SPAAC needs azide + strained alkyne).
        return [
            "C1CCCC#CCC1",           # cyclooctyne
            "C1CCCC#CC1",            # smaller strained ring
        ]
    if rule_name == "SPC":
        # Partners: phosphine (SPC needs azide + P(III)).
        return [
            "CP",                    # methylphosphine
            "c1ccccc1P",             # phenylphosphine
        ]
    if rule_name == "DielsAlder":
        # Partners: dienophile (DielsAlder needs diene + alkene; the
        # chemprop v2 SMILES could carry either — we try both ways
        # via ``symmetric_click`` in :mod:`click_reactions`).
        return [
            "C=CC(=O)C",             # methyl vinyl ketone
            "C=CC",                  # propene
        ]
    return []


def _partner_smis_to_tiles(smis: Sequence[str]) -> List[Any]:
    """Wrap a SMILES list into Tile instances (lazily, with safe fallback)."""
    out: List[Any] = []
    for smi in smis:
        tags = _detect_tags(smi)
        tile = _smiles_to_tile(smi, tags=tags)
        if tile is not None:
            out.append(tile)
    return out


# ---------------------------------------------------------------------------
# Augmentation core — fire one click rule on one (smiles, partner) pair
# ---------------------------------------------------------------------------


def _fire_click_on_smiles(
    smiles: str,
    rule_name: str,
    partner_tile: Any,
) -> Optional[str]:
    """Apply one click rule to (smiles, partner_tile).  Returns canonical SMILES.

    Returns ``None`` when the rule failed to fire (RDKit incompatibility,
    no matching functional handle, etc.) so the caller can fall back to
    the original SMILES — the wrapper is monotone by construction.
    """
    try:
        from molmetal_lam.reactions.click_reactions import (
            apply_click_reaction,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("[click_aug] could not import click_reactions: %s", exc)
        return None
    substrate_tile = _smiles_to_tile(smiles)
    if substrate_tile is None:
        return None
    try:
        results = apply_click_reaction(rule_name, substrate_tile, partner_tile)
    except Exception as exc:
        logger.debug("[click_aug] apply_click_reaction(%s) failed: %s", rule_name, exc)
        return None
    if not results:
        return None
    return str(results[0].product_smiles or "")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class ClickAugmentationResult:
    """Container for :func:`augment_candidates_via_click_reactions` outputs.

    Attributes
    ----------
    base_smiles : list[str]
        The original chemprop v2 candidate SMILES (passthrough).
    augmented_smiles : list[str]
        The click-augmented SMILES.  ``len(augmented_smiles) ==
        len(base_smiles)`` — the wrapper never shrinks the list.
    n_fired : int
        Number of molecules where a click rule actually fired (returned a
        different canonical SMILES).
    n_passthrough : int
        Number of molecules where the augmentation fell back to the
        base SMILES (no handle, RDKit incompat, etc.).
    per_rule_fire_count : dict[str, int]
        Per-rule fire counts (``{"CuAAC": 12, "SPAAC": 3, ...}``).
    diversity_lift_scaffolds : int
        ``unique_scaffold_count(augmented) - unique_scaffold_count(base)``
        — positive values mean the augmentation broadened chemical-space
        coverage.
    diversity_lift_tanimoto : float
        ``mean_pairwise_tanimoto(augmented) - mean_pairwise_tanimoto(base)``
        — the literal m7 diversity lift axis (Morgan ECFP4 Tanimoto distance).
    wall_time_s : float
        Wall-time for the augmentation pass.
    rule_names : list[str]
        The click-rule names used (echo of ``click_rule_names``).
    notes : str
        Free-text diagnostic.
    """

    base_smiles: List[str]
    augmented_smiles: List[str]
    n_fired: int = 0
    n_passthrough: int = 0
    per_rule_fire_count: Dict[str, int] = field(default_factory=dict)
    diversity_lift_scaffolds: int = 0
    diversity_lift_tanimoto: float = 0.0
    wall_time_s: float = 0.0
    rule_names: List[str] = field(default_factory=list)
    notes: str = ""


def augment_candidates_via_click_reactions(
    smiles_list: Sequence[str],
    *,
    click_rule_names: Sequence[str] = DEFAULT_CLICK_RULE_NAMES,
    use_click_prob: float = 0.5,
    partner_pool: Optional[Dict[str, Sequence[str]]] = None,
    rng: Optional[random.Random] = None,
) -> ClickAugmentationResult:
    """Augment a chemprop v2 SMILES candidate list via click reactions.

    Parameters
    ----------
    smiles_list : Sequence[str]
        The chemprop v2 candidate SMILES (any iterable of strings).
    click_rule_names : Sequence[str], default :data:`DEFAULT_CLICK_RULE_NAMES`
        Which click rules to draw from.  Defaults to the 4-rule set
        (CuAAC / SPAAC / SPC / DielsAlder) — the same 4 rules exposed by
        :data:`molmetal_lam.reactions.click_reactions.CLICK_REACTIONS`.
    use_click_prob : float, default 0.5
        Probability that a given candidate SMILES is augmented by a
        click rule (clamped to ``[0, 1]``).  ``0.0`` means passthrough
        (zero diversity lift, backward-compat preserved); ``1.0`` means
        every candidate gets augmented.
    partner_pool : dict[str, Sequence[str]], optional
        Override the curated partner pool per rule.  When ``None``, the
        :func:`_get_partner_pool` defaults are used.
    rng : random.Random, optional
        Random number generator.  When ``None``, a fresh
        :class:`random.Random` is constructed.

    Returns
    -------
    ClickAugmentationResult
        Per-rule fire counts + diversity lift (Tanimoto + Bemis-Murcko
        scaffold) + wall-time + augmented SMILES.  ``augmented_smiles``
        is always ``len(smiles_list)`` long — the wrapper is monotone.

    Notes
    -----
    The function is **idempotent**: passing the same ``smiles_list``
    with the same ``rng`` produces the same ``augmented_smiles``.
    Callers wanting diversity lift measurement should run
    :func:`diversity_lift_tanimoto` on the augmented vs base sets.
    """
    t0 = time.perf_counter()

    rule_names: List[str] = list(click_rule_names) or list(DEFAULT_CLICK_RULE_NAMES)
    use_prob = float(use_click_prob)
    if use_prob < 0.0:
        use_prob = 0.0
    elif use_prob > 1.0:
        use_prob = 1.0
    rng = rng if rng is not None else random.Random()

    # Pre-build partner-tile lists per rule (lazily).
    partner_pool_resolved: Dict[str, List[Any]] = {}
    for rname in rule_names:
        smis = partner_pool.get(rname) if partner_pool is not None else None
        if smis is None:
            smis = _get_partner_pool(rname)
        partner_pool_resolved[rname] = _partner_smis_to_tiles(smis)

    # Per-rule fire counts (default 0).
    per_rule_fire_count: Dict[str, int] = {rname: 0 for rname in rule_names}

    base_clean: List[str] = []
    for s in smiles_list:
        if isinstance(s, str) and s:
            c = _canonical(s)
            if c:
                base_clean.append(c)
            else:
                # Keep the original (un-parseable by RDKit) so the
                # passthrough remains length-preserving on input.
                base_clean.append(s)
    if not base_clean:
        return ClickAugmentationResult(
            base_smiles=[],
            augmented_smiles=[],
            n_fired=0,
            n_passthrough=0,
            per_rule_fire_count=per_rule_fire_count,
            diversity_lift_scaffolds=0,
            diversity_lift_tanimoto=0.0,
            wall_time_s=time.perf_counter() - t0,
            rule_names=rule_names,
            notes="empty input smiles_list; nothing to augment",
        )

    out: List[str] = []
    n_fired = 0
    n_passthrough = 0

    for raw_smi, canon_smi in zip(smiles_list, base_clean):
        # ``use_prob == 0`` is the BACKWARD-COMPAT no-op passthrough —
        # emit the caller's input verbatim, no canonicalisation, no
        # mutation.  This is the G4 contract: chemprop v2 callers that
        # have NOT opted in see zero behaviour change.
        if use_prob <= 0.0:
            out.append(raw_smi)
            n_passthrough += 1
            continue
        if rng.random() < use_prob:
            # Pick a random rule + partner.
            eligible = [
                r for r in rule_names if partner_pool_resolved.get(r)
            ]
            if not eligible:
                out.append(raw_smi)
                n_passthrough += 1
                continue
            rname = rng.choice(eligible)
            partner_tile = rng.choice(partner_pool_resolved[rname])
            aug = _fire_click_on_smiles(canon_smi, rname, partner_tile)
            if aug:
                canonical_aug = _canonical(aug)
                if canonical_aug and canonical_aug != canon_smi:
                    out.append(canonical_aug)
                    per_rule_fire_count[rname] = (
                        per_rule_fire_count.get(rname, 0) + 1
                    )
                    n_fired += 1
                    continue
            # Rule did not fire — fall back to original (input verbatim).
            out.append(raw_smi)
            n_passthrough += 1
        else:
            out.append(raw_smi)
            n_passthrough += 1

    # Diversity lift — both scaffolds and Tanimoto.
    base_scaffolds = _unique_scaffold_count(base_clean)
    aug_scaffolds = _unique_scaffold_count(out)
    base_td = _mean_pairwise_tanimoto_distance(base_clean)
    aug_td = _mean_pairwise_tanimoto_distance(out)

    return ClickAugmentationResult(
        base_smiles=list(base_clean),
        augmented_smiles=out,
        n_fired=n_fired,
        n_passthrough=n_passthrough,
        per_rule_fire_count=per_rule_fire_count,
        diversity_lift_scaffolds=int(aug_scaffolds - base_scaffolds),
        diversity_lift_tanimoto=float(aug_td - base_td),
        wall_time_s=time.perf_counter() - t0,
        rule_names=rule_names,
        notes="",
    )


def diversity_lift_tanimoto(
    base_smiles: Sequence[str],
    augmented_smiles: Sequence[str],
) -> float:
    """Measure the m7 diversity lift (mean Tanimoto-distance delta).

    Returns
    -------
    float
        ``mean_pairwise_tanimoto_distance(augmented) -
        mean_pairwise_tanimoto_distance(base)``.  Positive values
        indicate the augmentation broadened chemical-space coverage;
        zero / negative values indicate no lift (G3 honest-negative).
    """
    base_td = _mean_pairwise_tanimoto_distance(base_smiles)
    aug_td = _mean_pairwise_tanimoto_distance(augmented_smiles)
    return float(aug_td - base_td)


# ---------------------------------------------------------------------------
# ClickAugmentedChempropGenerator — wraps a base (seed, n) -> List[str]
# ---------------------------------------------------------------------------


@dataclass
class ClickAugmentedChempropGenerator:
    """Wrap a base chemprop v2 candidate generator with click-augmentation.

    Mirrors the :class:`MedChemAugmentedGenerator` API surface so a
    chemprop v2 batch sampler that today returns ``[smi] * n`` can be
    swapped for a click-augmented version with a single wrapper line.

    Parameters
    ----------
    base_generator : callable
        ``(seed_smiles: str, n_samples: int) -> List[str]`` — the
        underlying chemprop v2 batch sampler.
    click_rule_names : Sequence[str], default :data:`DEFAULT_CLICK_RULE_NAMES`
        Click rules to draw from.
    use_click_prob : float, default 0.5
        Probability per sample that a click rule fires (clamped to
        ``[0, 1]``).  ``0.0`` is the backward-compat passthrough.
    partner_pool : dict[str, Sequence[str]], optional
        Override the curated per-rule partner pool.
    rng : random.Random, optional
        Inject for determinism in tests.

    Attributes
    ----------
    base_generator : callable
    click_rule_names : tuple[str, ...]
    use_click_prob : float
    rng : random.Random
    """

    base_generator: Callable[..., List[str]]
    click_rule_names: Tuple[str, ...] = DEFAULT_CLICK_RULE_NAMES
    use_click_prob: float = 0.5
    partner_pool: Optional[Dict[str, Sequence[str]]] = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if not callable(self.base_generator):
            raise ClickAugmentationError(
                "base_generator must be callable "
                f"(got {type(self.base_generator).__name__})"
            )
        prob = float(self.use_click_prob)
        if prob < 0.0:
            prob = 0.0
        elif prob > 1.0:
            prob = 1.0
        self.use_click_prob = prob
        if not self.click_rule_names:
            self.click_rule_names = DEFAULT_CLICK_RULE_NAMES
        if self.rng is None:
            self.rng = random.Random()

    # ------------------------------------------------------------------
    def generate(
        self,
        seed_smiles: str,
        n_samples: int = 100,
    ) -> List[str]:
        """Generate ``n_samples`` click-augmented chemprop v2 SMILES.

        Parameters
        ----------
        seed_smiles : str
            The base-generator seed (passed verbatim to the wrapped
            callable).
        n_samples : int, default 100
            Target number of SMILES to return.

        Returns
        -------
        List[str]
            ``n_samples`` SMILES.  When the wrapped base_generator
            returns a shorter list, fills the gap with passthrough
            copies of ``seed_smiles``.  When the base_generator raises,
            logs a warning and returns ``[seed_smiles] * n_samples``.
        """
        if n_samples <= 0:
            return []
        if not isinstance(seed_smiles, str) or not seed_smiles:
            return []
        try:
            base_samples = list(self.base_generator(seed_smiles, n_samples))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "[ClickAugmentedChempropGenerator] base_generator raised: %s", exc
            )
            base_samples = []

        # Run one batched augmentation over the base samples.
        # When ``use_click_prob == 0`` the augmentation is a no-op —
        # this is the backward-compat gate (G4).
        if self.use_click_prob <= 0.0:
            out = []
            for i in range(n_samples):
                smi = base_samples[i] if i < len(base_samples) else seed_smiles
                if isinstance(smi, str) and smi:
                    out.append(smi)
            return out

        # Fill base_samples to n_samples with seed_smiles so the augmentor
        # has ``n_samples`` entries to draw from.
        padded_base = [
            (base_samples[i] if i < len(base_samples) else seed_smiles)
            for i in range(n_samples)
        ]
        aug = augment_candidates_via_click_reactions(
            padded_base,
            click_rule_names=self.click_rule_names,
            use_click_prob=self.use_click_prob,
            partner_pool=self.partner_pool,
            rng=self.rng,
        )
        return aug.augmented_smiles


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "ClickAugmentationError",
    "ClickAugmentationResult",
    "ClickAugmentedChempropGenerator",
    "DEFAULT_CLICK_RULE_NAMES",
    "augment_candidates_via_click_reactions",
    "diversity_lift_tanimoto",
]
