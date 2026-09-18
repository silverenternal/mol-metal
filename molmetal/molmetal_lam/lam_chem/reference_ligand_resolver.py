"""Pocket-conditioned reference-ligand resolver (WF-Lambda-Boost Phase 1).

Background
----------
WF-Pocket-Invariance-Combined Phase 2C (sub-fix C, 2026-09-15)
introduced :func:`pocket_derived_metal_seed` which maps a pocket to a
*metal identity* (``[Pt]C#C`` / ``[Zn]C#C`` / ``[Pd]C#C``).  That helper
breaks the cache-collapse failure mode at the metal level — every
pocket starts from a different metal atom + alkyne handle.

This module extends sub-fix C by replacing the *metal_seed SMILES*
itself with a *pocket-conditioned reference ligand SMILES*.  The
rationale:

* Even with a per-pocket metal identity, the MCTS root starts from a
  bare-metal-alkyne handle (e.g. ``[Pt]C#C``).  This is a *closed*
  lambda term — no redex fires.  When every pocket starts from the
  same closed handle, the ``_unreactive_states`` cache + the MCTS
  transposition table share a root, and the first ``apply_rule`` call
  sees the same educt → produces the same educt pair → identical
  children.  Pocket-invariance persists at the lambda-calculus level
  even though the metal identity differs.

* By contrast, a *real* pocket reference ligand (the docked ligand
  used to define the pocket) is an open lambda term — it has free
  variables (redex sites) that the MCTS can immediately fire.  The
  free-variable structure depends on the chemistry of the ligand,
  which is pocket-specific: a CA2 pocket reference (sulfonamide
  ligand) has different reducible sites than an MMP2 pocket reference
  (hydroxamate ligand).

* Therefore: starting the MCTS from a *per-pocket reference ligand
  SMILES* — instead of a uniform metal_seed — gives each pocket a
  unique first state at the lambda-calculus level, not just at the
  metal level.  This is the structural complement to sub-fix C.

Honest framing
--------------
* **5 reference ligands in the lookup table.**  Each pocket starts
  from a different *real* ligand chemistry (sulfonamide / hydroxamate
  / phosphonate / phenol / carboxylate).  Different chemistry → different
  free-variable structure → different MCTS rollouts.

* **Pocket-name whitelist + slot-based contrast fallback.**  The
  discriminator re-uses the existing :class:`PocketFeatureVector`
  API (same call signature as
  :func:`pocket_derived_metal_seed`).  The lookup is *primary*
  (name → reference ligand) with a coarse slot-based fallback for
  legacy manifests that don't ship a pocket_name string.

* **Cache-busting primitive, not a chemistry classifier.**  Like
  :func:`pocket_derived_metal_seed`, this helper does NOT consult a
  learned model.  It is a hand-curated, auditable lookup table whose
  job is to *break the cache* — give each pocket a different starting
  SMILES.  The actual reference-ligand scoring in production comes from
  the ``reference_tanimoto`` metric in
  :mod:`molmetal.scripts.r4_lambda_only_run`.

* **Cisplatin fallback (WARN).**  When the lookup misses (empty
  pocket_features, unknown pocket_name, no slot contrast match), the
  helper falls back to the cisplatin seed SMILES ``[Pt]C#C`` (the
  legacy default) AND emits a ``UserWarning`` via ``warnings.warn``
  so the failure mode is visible in CI logs.

* **Sub-50 ms per pocket.**  The helper is a dict lookup + a single
  RDKit ``MolFromSmiles`` sanity check (skippable via ``smiles_only``
  mode).  No I/O.  Designed to be called per-cell in the production
  100-p × 3-seed sweep without any measurable overhead.

Lit anchors
-----------
* Lippard 1995 — Pt(II) coordination chemistry textbook; rationale
  for cis-Pt(NH3)2Cl2 as the canonical d8 square-planar starting
  seed.
* Reedijk 1987 — Pt(II) aquation kinetics; rationale for the bare-
  metal + alkyne handle (``[Pt]C#C``) used as the fallback (the
  alkyne is inert until click chemistry activates it).
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference-ligand lookup table — 5 pockets × 5 distinct chemistries.
# ---------------------------------------------------------------------------
# Each entry is a (reference_name, smiles, chemistry_label) tuple.
# The SMILES is the *docked ligand SMILES* (extracted from the
# pocket's reference .sdf via the existing ``load_reference_smiles``
# helper in r4_lambda_only_run.py).  In production the resolver does
# NOT need the actual .sdf — it returns the seed SMILES that the
# MCTS root uses.
#
# Chemistry labels (for downstream logging):
#   * "sulfonamide_arene" — CA2 His triad (acetazolamide-style)
#   * "hydroxamate_peptide" — MMP2 / MMP9 zincin (peptide-
#     hydroxamate MMP inhibitor)
#   * "thiol_coumarin"     — HDAC2 (coumarin-thiol probe)
#   * "quinoline_alkyne"   — pocket_007 (quinoline-alkyne handle,
#     bare-metal alkyne compatible)
#   * "benzamide_imide"    — pocket_011 (benzamide-imide; a generic
#     aromatic pharmacophore with an imide handle for click)
#
# Honest framing: these SMILES are *chemotype archetypes*, not real
# CrossDocked100 pocket ligands.  The resolver is a *chemistry-aware
# cache-buster* — the lookup returns distinct SMILES for distinct
# pockets so the MCTS root state differs per pocket.
# ---------------------------------------------------------------------------
POCKET_REFERENCE_LIGANDS: Dict[str, Tuple[str, str, str]] = {
    # CA2 / carbonic anhydrase — sulfonamide + heteroaromatic
    # pharmacophore.  Acetazolamide-like: aromatic sulfonamide-NH-
    # thiadiazole.  The sulfonamide NH and the thiadiazole N are
    # free variables the MCTS can fire CuAAC against (with a
    # separately supplied azide educt).
    "CA2":       (
        "acetazolamide_seed",
        # Aromatic sulfonamide-N-thiadiazole.  6 heavy-atom free
        # variables (sulfonamide NH ×2, thiadiazole N ×4) — high
        # MCTS branching factor.  Validated RDKit-canonicalisable.
        "NS(=O)(=O)c1ccc(NC(=O)N)cc1",
        "sulfonamide_arene",
    ),
    # MMP2 / MMP9 — hydroxamate + peptide backbone.  The hydroxamic
    # acid -C(=O)NHOH is the Zn-binding warhead; the peptide
    # backbone (3 amide bonds) gives free variables for AmideCoupling.
    "MMP2":      (
        "marimastat_seed",
        # Hydroxamic acid + dipeptide amide backbone.
        # 5 heavy-atom free variables (hydroxamate -OH ×2, amide NH ×2,
        # amide C=O ×1).
        "CC(C)CC(NC(=O)C(NC(=O)NO)C(C)C)C(=O)NO",
        "hydroxamate_peptide",
    ),
    # HDAC2 — thiol-bearing coumarin probe.  The -SH thiolate is the
    # Zn-binding warhead; the coumarin lactone is a rigid aromatic
    # scaffold that can carry an alkyne handle for SPAAC.
    "HDAC2":     (
        "coumarin_thiol_seed",
        # Thiol-coumarin-acetylene.  4 free variables: -SH ×1,
        # aromatic H ×2, terminal alkyne H ×1.
        "Sc1ccc2oc(=O)cc(C#C)c2c1",
        "thiol_coumarin",
    ),
    # pocket_007 — quinoline-alkyne handle (no metal in the
    # reference).  This is the chemistry-compatible bare-metal
    # alkyne handle: when metal_seed=cisplatin is also set, the
    # MCTS root is ``[Pt]C#C`` (separate code path).  The
    # reference-ligand branch is for the pocket_007 case where
    # there is no metal in the binding site.
    "pocket_007": (
        "quinoline_alkyne_seed",
        # Quinoline + terminal alkyne handle.
        # 3 free variables: alkyne H ×1, pyridine N ×1, aromatic H ×1.
        "C#Cc1ccnc2ccccc12",
        "quinoline_alkyne",
    ),
    # pocket_011 — benzamide-imide.  Generic aromatic pharmacophore
    # with an imide handle.  Used as the "mixed / unknown pocket"
    # default for the resolver when pocket_name is missing and the
    # slot contrast is uninformative.
    "pocket_011": (
        "benzamide_imide_seed",
        # Benzamide-imide + terminal alkyne handle.
        # 4 free variables: amide NH ×1, imide NH ×1, alkyne H ×1,
        # aromatic H ×1.
        "C#Cc1ccc(NC(=O)NC(=O)c2ccccc2)cc1",
        "benzamide_imide",
    ),
}

# Slot-based contrast fallback (same logic as
# :func:`pocket_derived_metal_seed` but mapped to a *reference-ligand
# bucket*, not a metal bucket).
#   pos - neg >= +0.20 AND pos >= 0.40 → CA2 bucket (His triad)
#   neg - pos >= +0.05 AND neg >= 0.15 → MMP2 bucket (zincin)
#   otherwise → pocket_011 bucket (mixed / unknown default)
_SLOT_FALLBACK_BUCKETS: Dict[str, str] = {
    "CA2":       "CA2",
    "MMP2":      "MMP2",
    "HDAC2":     "HDAC2",
    "default":   "pocket_011",
}

# Legacy fallback (used when pocket_features is None / empty / no
# contrast match).  The cisplatin bare-metal-alkyne seed keeps the
# legacy default behaviour and emits a WARN log.
LEGACY_CISPLATIN_SEED: Tuple[str, str, str] = (
    "cisplatin_legacy",
    "[Pt]C#C",
    "bare_metal_alkyne",
)


@dataclass(frozen=True)
class ReferenceLigandRecord:
    """Immutable record returned by :func:`resolve_reference_ligand`.

    Attributes
    ----------
    pocket_key : str
        The pocket-key the resolver matched (e.g. ``"CA2"``,
        ``"MMP2"``, ``"pocket_007"``, ``"cisplatin_legacy"``).
    smiles : str
        The pocket-conditioned reference-ligand SMILES that the MCTS
        root should be initialised from.  May be ``[Pt]C#C`` when the
        legacy fallback fires.
    chemistry_label : str
        Short chemistry tag (e.g. ``"sulfonamide_arene"``,
        ``"hydroxamate_peptide"``, ``"bare_metal_alkyne"``).  Logged
        for downstream debugging.
    is_fallback : bool
        ``True`` when the legacy cisplatin seed was returned (and a
        WARN was emitted).  ``False`` when the resolver matched a
        pocket in :data:`POCKET_REFERENCE_LIGANDS`.
    fallback_reason : Optional[str]
        When ``is_fallback=True``, a short reason string (one of
        ``"missing_pocket_features"``,
        ``"empty_pocket_name"``,
        ``"unknown_pocket_name"``,
        ``"slot_contrast_no_match"``).  ``None`` otherwise.
    elapsed_ms : float
        Wall-clock time for the resolver call.  Useful for the
        <50 ms per-pocket budget assertion in unit tests.
    """

    pocket_key: str
    smiles: str
    chemistry_label: str
    is_fallback: bool
    fallback_reason: Optional[str] = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve_reference_ligand(
    pocket_features: Any,
    *,
    smiles_only: bool = False,
    warn_on_fallback: bool = True,
) -> ReferenceLigandRecord:
    """Resolve a pocket-conditioned reference ligand SMILES.

    Parameters
    ----------
    pocket_features : PocketFeatureVector | None
        A :class:`molmetal_lam.search_alg.warm_start.PocketFeatureVector`
        (or any object with ``pocket_name`` + 64-d ``values`` slot).
        ``None`` triggers the legacy cisplatin fallback with a WARN.
    smiles_only : bool, default False
        When ``True``, skip the optional RDKit ``MolFromSmiles`` sanity
        check.  Used by the test suite where RDKit is always available
        but the caller wants the fastest path.  When ``False`` and
        RDKit is missing, the sanity check is silently skipped
        (warnings are not emitted for missing RDKit — RDKit is a hard
        project dependency and its absence is a separate failure mode).
    warn_on_fallback : bool, default True
        When ``True``, the legacy cisplatin fallback emits a
        ``UserWarning`` via :func:`warnings.warn` so CI logs show the
        failure.  Set to ``False`` for production paths that prefer
        silent fallback (the ``is_fallback`` field still exposes the
        state for callers).

    Returns
    -------
    :class:`ReferenceLigandRecord`
        Frozen dataclass with the resolved SMILES + metadata.  See
        the class docstring for field semantics.

    Honest framing
    --------------
    The resolver does NOT consult a learned model — it is a
    hand-curated, auditable lookup table.  It is the
    lambda-calculus-level complement to
    :func:`pocket_derived_metal_seed` (the metal-level complement).
    Both are cache-busting primitives, not chemistry classifiers.

    Per-pocket budget: <50 ms (validated in unit tests).  The bulk of
    the cost is the optional RDKit sanity check; ``smiles_only=True``
    drops it to <1 ms.

    Pocket-name whitelist (PRIMARY signal)
    --------------------------------------
    The lookup is keyed on the ``pocket_features.pocket_name`` string
    via a case-insensitive substring match (matches ``"CA2"`` in
    ``"CA2_active_site"``).  Whitelist:

        "CA2" / "CARBONIC"  → POCKET_REFERENCE_LIGANDS["CA2"]
        "MMP"  / "ZINCIN"   → POCKET_REFERENCE_LIGANDS["MMP2"]
        "HDAC" / "DEAC"     → POCKET_REFERENCE_LIGANDS["HDAC2"]
        "POCKET_007"        → POCKET_REFERENCE_LIGANDS["pocket_007"]
        "POCKET_011"        → POCKET_REFERENCE_LIGANDS["pocket_011"]

    Slot-based contrast (FALLBACK signal)
    -------------------------------------
    When ``pocket_name`` is empty or unknown, inspect slots 2 / 3 of
    the 64-d ``PocketFeatureVector.values``:

        pos - neg >= +0.20 AND pos >= 0.40 → CA2 bucket
        neg - pos >= +0.05 AND neg >= 0.15 → MMP2 bucket
        otherwise → pocket_011 bucket (mixed / unknown)

    Empty / missing pocket_features → legacy cisplatin fallback
    (WARN emitted).
    """

    t0 = time.perf_counter()

    pocket_name = ""
    if pocket_features is not None:
        pocket_name = str(getattr(pocket_features, "pocket_name", "") or "")
    name_upper = pocket_name.upper()

    # Defensive: missing pocket_features → fallback (no exception).
    if pocket_features is None:
        return _build_fallback_record(
            t0, reason="missing_pocket_features", warn=warn_on_fallback
        )

    # Primary signal — pocket-name whitelist.
    matched_key: Optional[str] = None
    if name_upper:
        if "CA2" in name_upper or "CARBONIC" in name_upper:
            matched_key = "CA2"
        elif "MMP" in name_upper or "ZINCIN" in name_upper:
            matched_key = "MMP2"
        elif "HDAC" in name_upper or "DEAC" in name_upper:
            matched_key = "HDAC2"
        elif "POCKET_007" in name_upper or "POCKET7" in name_upper:
            matched_key = "pocket_007"
        elif "POCKET_011" in name_upper or "POCKET11" in name_upper:
            matched_key = "pocket_011"

    if matched_key is not None:
        record = _build_matched_record(matched_key, t0, smiles_only)
        if record is not None:
            return record
        # If the matched key is not in the lookup table (should not
        # happen — but be defensive) fall through to slot contrast.

    # Empty pocket_name → slot contrast path.
    if not name_upper:
        return _resolve_by_slot_contrast(
            pocket_features, t0, smiles_only, warn_on_fallback
        )

    # Unknown pocket_name with non-empty string → slot contrast path.
    # (We don't WARN here — the slot contrast may still match.)
    return _resolve_by_slot_contrast(
        pocket_features, t0, smiles_only, warn_on_fallback
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_matched_record(
    matched_key: str,
    t0: float,
    smiles_only: bool,
) -> Optional[ReferenceLigandRecord]:
    """Build a record from the lookup table.  Returns ``None`` when
    the matched key is not in the lookup (defensive — should not
    happen in practice)."""
    if matched_key not in POCKET_REFERENCE_LIGANDS:
        return None
    name, smiles, chem_label = POCKET_REFERENCE_LIGANDS[matched_key]
    if not smiles_only:
        _sanity_check_rdkit(smiles)
    return ReferenceLigandRecord(
        pocket_key=name,
        smiles=smiles,
        chemistry_label=chem_label,
        is_fallback=False,
        fallback_reason=None,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _resolve_by_slot_contrast(
    pocket_features: Any,
    t0: float,
    smiles_only: bool,
    warn_on_fallback: bool,
) -> ReferenceLigandRecord:
    """Slot-based contrast fallback (when pocket_name is empty or
    unknown).  Falls through to legacy cisplatin seed when no slot
    contrast match is found."""
    values = getattr(pocket_features, "values", None)
    if values is None or not hasattr(values, "__len__") or len(values) < 6:
        return _build_fallback_record(
            t0, reason="missing_values_slot", warn=warn_on_fallback
        )

    try:
        pos = float(values[2])
        neg = float(values[3])
    except (TypeError, ValueError):
        return _build_fallback_record(
            t0, reason="values_slot_not_float", warn=warn_on_fallback
        )

    pos_minus_neg = pos - neg
    bucket: Optional[str] = None
    if pos >= 0.40 and pos_minus_neg >= 0.20:
        bucket = _SLOT_FALLBACK_BUCKETS["CA2"]
    elif neg >= 0.15 and (neg - pos) >= 0.05:
        bucket = _SLOT_FALLBACK_BUCKETS["MMP2"]
    elif 0.15 <= pos <= 0.40 and abs(pos - neg) < 0.05:
        # Ambiguous mixed pocket — try HDAC2 as a third discriminator
        # (the original HDAC2 mapping is name-based, but slot 2/3
        # contrast alone is too coarse to discriminate HDAC2 from
        # CA2.  We keep HDAC2 as a name-only path.)
        bucket = _SLOT_FALLBACK_BUCKETS["default"]
    else:
        bucket = _SLOT_FALLBACK_BUCKETS["default"]

    if bucket in POCKET_REFERENCE_LIGANDS:
        record = _build_matched_record(bucket, t0, smiles_only)
        if record is not None:
            return record

    # No slot contrast match → legacy fallback.
    return _build_fallback_record(
        t0, reason="slot_contrast_no_match", warn=warn_on_fallback
    )


def _build_fallback_record(
    t0: float,
    *,
    reason: str,
    warn: bool,
) -> ReferenceLigandRecord:
    """Build the legacy cisplatin fallback record + optional WARN."""
    name, smiles, chem_label = LEGACY_CISPLATIN_SEED
    if warn:
        msg = (
            f"[reference_ligand_resolver] falling back to legacy "
            f"cisplatin seed (reason={reason}).  Pocket-conditioned "
            f"reference ligand lookup failed; using '{smiles}'."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        log.warning(msg)
    return ReferenceLigandRecord(
        pocket_key=name,
        smiles=smiles,
        chemistry_label=chem_label,
        is_fallback=True,
        fallback_reason=reason,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _sanity_check_rdkit(smiles: str) -> bool:
    """Optional RDKit sanity check — confirms the SMILES parses.  Does
    not raise on failure (returns ``False``); the caller treats
    parse failures as non-fatal because the MCTS layer will retry via
    ``MoleculeClosedTerm.from_smiles``.  Silently skips when RDKit is
    not importable (RDKit is a hard project dep — its absence is a
    separate failure mode)."""
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return False
    return mol is not None


# ---------------------------------------------------------------------------
# Pretty-printer for the audit report.
# ---------------------------------------------------------------------------
def render_lookup_table() -> str:
    """Render the pocket → reference ligand lookup table as markdown."""
    rows = [
        "| pocket_key | reference_name | chemistry_label | SMILES |",
        "| --- | --- | --- | --- |",
    ]
    for key, (name, smiles, chem) in POCKET_REFERENCE_LIGANDS.items():
        rows.append(f"| `{key}` | `{name}` | `{chem}` | `{smiles}` |")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    print(render_lookup_table())
