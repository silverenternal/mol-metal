"""cfm_tanimoto_ood_filter.py — TODO-98: Tanimoto<0.4 OOD filter for CFM training data.

==============================================================================
Wf-257-Apply TODO-98 — Tanimoto<0.4 OOD filter for the CFM training-data path
==============================================================================

Per ``TODO/completed/98_wf44_closure_ood_phase2.md`` (Wf-44 P2 Closure OOD
Phase 2, L12 cross-domain) + ``TODO/completed/119_decode_ratio_strengthen.md``
§3 P1.3 (real-CFM decode_ratio lift on PlatinAI / DrugOOD / MetalCytoToxDB):

* The Wf-124 P2 ship (TODO-124) achieved ``decode_ratio = 1.000`` (192/192)
  on **synthetic** Pt-click clouds.  Wf-181 (0.9980) + Wf-188 (0.9958) are
  the **real-CFM** counterpart on MetalCytoToxDB.
* The remaining **OOD risk** is that the real-CFM training pool (PlatinAI
  4-cell + DrugOOD IC50 + MetalCytoToxDB) contains scaffolds with
  Tanimoto ≥ 0.4 to the **holdout / test** set — the canonical
  Pocket2Mol / TargetDiff OOD threshold.  These near-duplicates inflate
  decode_ratio on the training-distribution subset while **collapsing
  generalisation** to truly novel chemistry.

This module provides a thin, **CPU-only** filter that wraps
:mod:`molmetal.molmetal_lam.lam_chem.ood_protocol.TanimotoOODProtocol`:

* :func:`apply_tanimoto_ood_filter` — boolean mask over a query SMILES
  list, given a train/reference pool, marking which queries are **OOD**
  relative to the train pool.
* :func:`cfm_ood_filter_smiles_list` — convenience: split a
  ``(query_smiles, query_clouds)`` list into ``(in_domain, ood)``
  buckets, mirroring the Wf-41 P2 / Wf-44 P1 / Wf-44 P2 protocol.
* :func:`summarise_tanimoto_ood` — diagnostic: produce a JSON-friendly
  summary with ``n_total``, ``n_ood``, ``n_in_domain``,
  ``ood_fraction``, and the **max-train-tanimoto** histogram
  sufficient for the §1.2.5-style DIAGNOSTIC-vs-GRADIENT discipline.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* **CPU-only**: pure RDKit Morgan-ECFP4 (radius=2, nBits=2048) —
  consistent with the Wf-57 P1 / Wf-41 P2 OOD protocol stack.
* **DIAGNOSTIC-only**: this filter emits a per-sample OOD flag + a
  cohort-level summary.  It does **NOT** alter the CFM loss surface
  (per checklist §1.2.5 — W₂-pic50 alignment is DIAGNOSTIC-only).
* **Backward-compat**: the filter is a **stand-alone utility**; the
  ``train_cfm_yuelbond_wasserstein.py`` driver does not depend on it.
  Integration with the driver happens at the **data-loader layer**
  via the new ``--use-tanimoto-ood-filter`` CLI flag (added in
  Wf-257-Apply).

Use cases
---------
* Pre-retrain dataset audit: identify OOD samples BEFORE running CFM
  training, so the post-retrain decode_ratio claim is honest about the
  train/test chemistry gap.
* Cross-cohort OOD report: given PlatinAI 4-cell + DrugOOD + MetalCytoTox
  as the train pool and tmQM d-block as the test pool, produce the
  Wf-41 P2 L12 OOD reachability table.
* CFM training-time filter: optionally mask OOD samples from the
  training loop (the ``--use-tanimoto-ood-filter`` path).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

from molmetal.molmetal_lam.lam_chem.ood_protocol import (
    DEFAULT_FP_NBITS,
    DEFAULT_FP_RADIUS,
    DEFAULT_TANIMOTO_THRESHOLD,
    TanimotoOODProtocol,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class TanimotoOODSummary:
    """DIAGNOSTIC summary of a Tanimoto<threshold OOD audit.

    Attributes
    ----------
    threshold : float
        Tanimoto threshold (queries with max-train-Tanimoto < this are
        flagged OOD).  Defaults to 0.4 (Pocket2Mol / TargetDiff).
    n_total : int
        Total queries in the audit cohort.
    n_in_domain : int
        Queries whose max-train-Tanimoto ≥ threshold (NOT OOD).
    n_ood : int
        Queries whose max-train-Tanimoto < threshold (OOD).
    ood_fraction : float
        ``n_ood / n_total`` in [0.0, 1.0].  ``0.0`` if ``n_total == 0``.
    max_train_tanimoto : list[float]
        Per-query max-train-Tanimoto similarity (length = n_total).
        Used for diagnostic histogram plotting; not consumed by the
        filter downstream.
    n_failed_parse : int
        Queries whose SMILES failed RDKit parse (conservatively
        treated as OOD — fail-closed).
    """

    threshold: float = DEFAULT_TANIMOTO_THRESHOLD
    n_total: int = 0
    n_in_domain: int = 0
    n_ood: int = 0
    ood_fraction: float = 0.0
    max_train_tanimoto: List[float] = field(default_factory=list)
    n_failed_parse: int = 0

    def to_dict(self) -> Dict[str, object]:
        """JSON-friendly dict (max_train_tanimoto retained)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Core filter API
# ---------------------------------------------------------------------------
def apply_tanimoto_ood_filter(
    query_smiles: Sequence[str],
    train_smiles: Sequence[str],
    threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_NBITS,
) -> List[bool]:
    """Return a per-query OOD flag.

    Parameters
    ----------
    query_smiles : sequence of str
        SMILES to flag.  Queries that fail RDKit parse are
        conservatively flagged OOD (fail-closed).
    train_smiles : sequence of str
        Train-pool SMILES (the **reference** for OOD comparison).
        An empty train pool returns ``True`` (OOD) for every query
        (mirrors :meth:`TanimotoOODProtocol.is_ood`).
    threshold, radius, n_bits : float, int, int
        Tanimoto threshold + Morgan-ECFP fingerprint params
        (defaults: 0.4 / 2 / 2048, matching the Wf-41 P2 / Wf-57 P1
        OOD protocol stack).

    Returns
    -------
    list[bool]
        ``out[i] = True`` iff ``query_smiles[i]`` is OOD relative to
        the train pool (max-Tanimoto < threshold).
    """
    proto = TanimotoOODProtocol(
        threshold=float(threshold), radius=int(radius), n_bits=int(n_bits),
    )
    return [proto.is_ood(q, list(train_smiles)) for q in query_smiles]


def cfm_ood_filter_smiles_list(
    query_smiles: Sequence[str],
    query_payloads: Sequence[object],
    train_smiles: Sequence[str],
    threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_NBITS,
) -> Tuple[List[object], List[object]]:
    """Partition ``query_payloads`` into (in_domain, ood) by Tanimoto OOD.

    This is the **CFM-specific** filter used at the data-loader
    layer: callers pass the SMILES list and an arbitrary payload
    list (e.g. ``AtomCloud`` objects, dicts with metadata, or
    indices into a larger dataframe); the function returns the
    OOD-filtered subset of payloads.

    The payload contract is intentionally loose (``object``) so
    callers can pass either AtomCloud, dict, or list without
    coupling this filter to a specific CFM data-loader shape.

    Parameters
    ----------
    query_smiles : sequence of str
        Per-query SMILES.  ``len(query_smiles) == len(query_payloads)``.
    query_payloads : sequence of object
        Per-query payloads (length-matched to ``query_smiles``).
    train_smiles : sequence of str
        Train-pool SMILES (the reference for OOD comparison).
    threshold, radius, n_bits : float, int, int
        Tanimoto threshold + Morgan-ECFP fingerprint params.

    Returns
    -------
    (in_domain_payloads, ood_payloads) : tuple of list[object]
        Partitioned payloads — ``in_domain`` first (kept in CFM
        training), ``ood`` second (filtered out).

    Raises
    ------
    ValueError
        If ``len(query_smiles) != len(query_payloads)``.
    """
    if len(query_smiles) != len(query_payloads):
        raise ValueError(
            f"len(query_smiles)={len(query_smiles)} != "
            f"len(query_payloads)={len(query_payloads)}"
        )
    flags = apply_tanimoto_ood_filter(
        query_smiles, train_smiles,
        threshold=threshold, radius=radius, n_bits=n_bits,
    )
    in_domain = [p for p, f in zip(query_payloads, flags) if not f]
    ood = [p for p, f in zip(query_payloads, flags) if f]
    return in_domain, ood


def summarise_tanimoto_ood(
    query_smiles: Sequence[str],
    train_smiles: Sequence[str],
    threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    radius: int = DEFAULT_FP_RADIUS,
    n_bits: int = DEFAULT_FP_NBITS,
) -> TanimotoOODSummary:
    """Compute a TanimotoOODSummary with per-query max-train-Tanimoto.

    In addition to the boolean OOD flag, this returns the full
    ``max_train_tanimoto`` array so callers can produce a
    histogram / per-decile table for honest reporting (per
    ``party-twenty-spirit`` §3.5).

    Parameters
    ----------
    query_smiles : sequence of str
        SMILES to audit.
    train_smiles : sequence of str
        Reference train-pool SMILES.
    threshold, radius, n_bits : float, int, int
        Tanimoto threshold + Morgan-ECFP fingerprint params.

    Returns
    -------
    :class:`TanimotoOODSummary`
        JSON-friendly summary suitable for
        ``metrics/by_metric/tanimoto_ood.json`` reporting.
    """
    proto = TanimotoOODProtocol(
        threshold=float(threshold), radius=int(radius), n_bits=int(n_bits),
    )
    train_list = list(train_smiles)
    n_total = len(query_smiles)
    max_sims: List[float] = []
    n_ood = 0
    n_failed = 0
    for q in query_smiles:
        if not train_list:
            # Empty train pool: every query is OOD by definition
            # (mirrors TanimotoOODProtocol.is_ood semantics).
            max_sims.append(0.0)
            n_ood += 1
            continue
        sim = proto.max_tanimoto(q, train_list)
        max_sims.append(float(sim))
        if sim < float(threshold):
            n_ood += 1
    n_in_domain = n_total - n_ood
    ood_fraction = (n_ood / n_total) if n_total > 0 else 0.0
    return TanimotoOODSummary(
        threshold=float(threshold),
        n_total=n_total,
        n_in_domain=n_in_domain,
        n_ood=n_ood,
        ood_fraction=ood_fraction,
        max_train_tanimoto=max_sims,
        n_failed_parse=n_failed,
    )


# ---------------------------------------------------------------------------
# JSON IO helper (DIAGNOSTIC report)
# ---------------------------------------------------------------------------
def write_tanimoto_ood_summary_json(
    summary: TanimotoOODSummary, path: str,
) -> None:
    """Write a :class:`TanimotoOODSummary` to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary.to_dict(), fh, indent=2)


__all__ = [
    "TanimotoOODSummary",
    "apply_tanimoto_ood_filter",
    "cfm_ood_filter_smiles_list",
    "summarise_tanimoto_ood",
    "write_tanimoto_ood_summary_json",
]