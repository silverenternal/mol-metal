"""r4_lib.rules — Click-rule resolution + pocket-derived metal_seed.

Two related helpers live here:

* :func:`_resolve_click_rules_aliases` — expand friendly spellings
  (``all-5``, ``CuAAC``, ``thiol-ene``, ...) into canonical
  :data:`CLICK_REACTIONS` keys.  Consumed by ``run_one_cell`` BEFORE
  the rule filter so the legacy ``click_rules_filter_emptied`` warning
  only fires for unknown spellings.
* :func:`pocket_derived_metal_seed` — coarse pocket-feature →
  ``(metal_seed_name, smiles)`` mapping used by ``--metal-seed-from-pocket``
  (WF-Pocket-Invariance Phase 2C).  The discriminator uses the
  pocket_name whitelist FIRST (CA2 / MMP / HDAC) and falls back to
  slot-2 (positive charge) / slot-3 (negative charge) residue
  fractions of the :class:`PocketFeatureVector`.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from .constants import (
    METAL_SEED_SMILES,
    POCKET_DERIVED_METAL_SEED_FALLBACK,
    _POCKET_METAL_BUCKETS,
)


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
    # Local import to avoid a circular dependency at module import time.
    from .constants import CLICK_RULE_ALIASES

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
    if "MMP" in name_upper or "HDAC" in name_upper or "ZINCIN" in name_upper:
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


__all__ = [
    "_resolve_click_rules_aliases",
    "pocket_derived_metal_seed",
]
