"""wetlab_protocol — assay dataclass + load helpers for P5.2 Tier-1 wet-lab plumbing.

============================================================
TODO-30 P5.2 Tier 1 (2026-09-17)
============================================================
This module is a *plumbing-only* layer for wet-lab assay feedback.  It does
NOT call any external chemistry facility — it is the data-side foundation
that:

1. Loads a ``{smiles: pIC50}`` measurement map from a TSV file (one
   ``smiles<TAB>pIC50_value`` line per measurement, ``#`` comments OK).
2. Exposes :class:`Assay` dataclass + :func:`load_assays` /
   :func:`append_assay` helpers for downstream consumers (the recalibration
   script in :mod:`molmetal.scripts.recalibrate_from_assay` and the reward
   channel in :mod:`molmetal_lam.lam_chem.wetlab_reward_channel`).
3. Normalises SMILES via RDKit canonicalisation when RDKit is available;
   falls back to a whitespace-normalised raw string when not.  This keeps
   the loader usable on bare-CPU hosts where the chem stack is partial.
4. Documents (in the docstrings) what Tier 2/3 would do — actual collaborator
   outreach, IRB paperwork, plate-reader integration.  None of that is
   implemented here.  This file is *pure data plumbing*.

Honest framing
--------------
The channel wired on top of this module
(:mod:`molmetal_lam.lam_chem.wetlab_reward_channel`) is **additive** and
**opt-in** (``w_wetlab=0.0`` default).  It returns a signed-error signal
``-|predicted - measured|`` per assayed SMILES so the MCTS leaf reward can
favour candidates whose dry-lab predictions match in-vitro evidence —
provided that evidence has been loaded via :func:`load_assays`.

This commit ships the *plumbing* only.  When real wet-lab data lands the
file is dropped at ``--wetlab-input path/to/assays.tsv``; no code change
is required.  Tier 3 (collaborator outreach) is explicitly **DEFER'd** to
TODO-30 Rank-15 — see ``molmetal/reports/wf_pitfall_audit/p5_wet_lab.md``.

Public surface
--------------
* :class:`Assay`                       — frozen dataclass per measurement.
* :func:`load_assays`                  — TSV -> List[Assay].
* :func:`append_assay`                 — single-row append with dedup.
* :func:`assays_to_dict`               — flatten to ``{smiles: pIC50}``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


__all__ = [
    "Assay",
    "load_assays",
    "append_assay",
    "assays_to_dict",
]
__version__ = "0.0.1-wetlab-protocol"


# ---------------------------------------------------------------------------
# Assay dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assay:
    """A single wet-lab measurement row.

    Attributes
    ----------
    smiles : str
        The candidate SMILES that was measured.  Stored canonically
        (RDKit) when RDKit is available at load time, otherwise stored
        as the raw input string.
    cell_line : str, default ""
        Cell-line label (e.g. ``"HeLa"``, ``"A2780"``, ``"MCF7"``).
        Empty string when the row did not specify one (synthesis-only
        screening cohorts, etc.).
    target_protein : str, default ""
        Protein target (e.g. ``"DNA"``, ``"PARP1"``).  Empty when the
        assay is whole-cell / cell-viability rather than target-bound.
    outcome_metric : str, default "pIC50"
        One of ``"pIC50"``, ``"IC50_uM"``, ``"GI50_uM"``, ``"AUC"``,
        ``"custom"``.  The reward channel currently only understands
        ``"pIC50"`` (higher = better); other values are stored but the
        channel returns 0.0 for them.
    outcome_value : float
        Numeric outcome.  pIC50 is in log-units (``[0, 12]`` typical),
        IC50_uM / GI50_uM in micromolar (lower = better).  Stored as
        ``float``; ``NaN`` is preserved (the channel silently skips
        NaN values).
    outcome_unit : str, default ""
        Display unit (``"pIC50"``, ``"uM"``, ``"%inhib"``).  Empty when
        the loader could not parse a unit from the header.
    assay_id : str, default ""
        Stable identifier (e.g. ``"A2780-2025-Q3-plate-007"``).  Used
        for provenance + audit when the channel emits a trace.
    collaborator : str, default ""
        Optional collaborator handle (initials / lab code).  Empty for
        in-house / synthetic-placeholder data.  Tier-3 outreach would
        populate this from the registry.
    date : str, default ""
        ISO date (``"YYYY-MM-DD"``) when the assay was performed.
        Empty when unknown.
    extra : dict, default ``{}``
        Free-form key/value bag for cohort / plate / well metadata.
        Preserved verbatim by :func:`load_assays`.
    """

    smiles: str
    cell_line: str = ""
    target_protein: str = ""
    outcome_metric: str = "pIC50"
    outcome_value: float = float("nan")
    outcome_unit: str = ""
    assay_id: str = ""
    collaborator: str = ""
    date: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_canonical(smiles: str) -> str:
    """Return the RDKit canonical SMILES when available; else raw.

    Failure modes (RDKit missing, parse error) all fall back to the
    raw string so :func:`load_assays` is usable on bare-CPU hosts
    that do not have the chem stack installed.
    """
    if not smiles or not isinstance(smiles, str):
        return ""
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles.strip()
        return str(Chem.MolToSmiles(mol))
    except Exception:
        return smiles.strip()


def load_assays(
    path: str,
    *,
    column_smiles: str = "smiles",
    column_value: str = "pIC50",
    sep: str = "\t",
    comment: str = "#",
    canonicalise: bool = True,
) -> List[Assay]:
    """Load a TSV/CSV file of wet-lab measurements into :class:`Assay` rows.

    The loader is permissive — it understands two on-disk shapes:

    1. **Wide schema** (preferred): header row + per-column metadata.
       The required columns are ``column_smiles`` and ``column_value``
       (defaults: ``smiles`` and ``pIC50``).  Optional columns that map
       to :class:`Assay` fields (when present in the header):

       - ``cell_line``           -> :attr:`Assay.cell_line`
       - ``target_protein``      -> :attr:`Assay.target_protein`
       - ``outcome_metric``      -> :attr:`Assay.outcome_metric`
       - ``outcome_unit``        -> :attr:`Assay.outcome_unit`
       - ``assay_id``            -> :attr:`Assay.assay_id`
       - ``collaborator``        -> :attr:`Assay.collaborator`
       - ``date``                -> :attr:`Assay.date`

       Unknown columns are kept in :attr:`Assay.extra`.

    2. **Two-column fallback**: no header.  Each row is parsed as
       ``smiles<sep>value`` with no metadata.  The returned rows
       populate only :attr:`Assay.smiles` and :attr:`Assay.outcome_value`.

    Empty / ``#``-prefixed lines are skipped silently.  The loader
    does NOT raise on parse errors — bad numeric values are stored
    as ``float('nan')`` (which the reward channel ignores).  This
    keeps a wet-lab feed usable even if one row is malformed.

    Parameters
    ----------
    path : str
        Path to the TSV/CSV file.
    column_smiles, column_value : str
        Header names to look up in the wide schema.  Defaults match
        the project-wide convention.
    sep : str, default ``"\\t"``
        Column separator (use ``","`` for CSV).
    comment : str, default ``"#"``
        Lines starting with this prefix are skipped.
    canonicalise : bool, default True
        When True (default), SMILES are RDKit-canonicalised.  Set False
        to preserve the on-disk ordering (useful for ID-based joins).

    Returns
    -------
    list[Assay]
        One :class:`Assay` per non-empty data row.  Empty list when the
        file has no usable rows.
    """
    out: List[Assay] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.readlines()
                     if ln.strip() and not ln.lstrip().startswith(comment)]
    except OSError:
        return out

    if not lines:
        return out

    # Decide schema by sniffing the first line.
    first = lines[0].rstrip("\n").rstrip("\r").split(sep)
    has_header = any(
        tok.strip().lower() in {column_smiles.lower(), column_value.lower()}
        for tok in first
    )

    if has_header:
        header = [tok.strip().lower() for tok in first]
        try:
            i_smiles = header.index(column_smiles.lower())
            i_value = header.index(column_value.lower())
        except ValueError:
            # Header declared but required columns missing — degrade to
            # two-column mode on the data rows below.
            has_header = False
            lines = [lines[0]] + lines[1:]  # keep data rows
        else:
            opt_indices = {
                "cell_line": header.index("cell_line")
                if "cell_line" in header else -1,
                "target_protein": header.index("target_protein")
                if "target_protein" in header else -1,
                "outcome_metric": header.index("outcome_metric")
                if "outcome_metric" in header else -1,
                "outcome_unit": header.index("outcome_unit")
                if "outcome_unit" in header else -1,
                "assay_id": header.index("assay_id")
                if "assay_id" in header else -1,
                "collaborator": header.index("collaborator")
                if "collaborator" in header else -1,
                "date": header.index("date")
                if "date" in header else -1,
            }
            known_cols = {column_smiles.lower(), column_value.lower(),
                          *opt_indices.keys()}
            extra_cols = [tok for tok in header if tok not in known_cols]

    if has_header:
        data_lines = lines[1:]
    else:
        data_lines = lines
        i_smiles = 0
        i_value = 1

    for raw in data_lines:
        toks = raw.rstrip("\n").rstrip("\r").split(sep)
        if len(toks) < 2:
            continue
        smi = toks[i_smiles].strip()
        if not smi:
            continue
        try:
            val = float(toks[i_value].strip())
        except (TypeError, ValueError):
            val = float("nan")
        if canonicalise:
            smi = _safe_canonical(smi)
        else:
            smi = smi.strip()

        kwargs = {"smiles": smi, "outcome_value": val}
        if has_header:
            for key, idx in opt_indices.items():
                if 0 <= idx < len(toks):
                    kwargs[key] = toks[idx].strip()
            extras = {}
            for col in extra_cols:
                ci = header.index(col)
                if 0 <= ci < len(toks):
                    extras[col] = toks[ci].strip()
            if extras:
                kwargs["extra"] = extras

        out.append(Assay(**{k: v for k, v in kwargs.items()
                            if k in Assay.__dataclass_fields__}))

    return out


def append_assay(
    existing: Sequence[Assay],
    new: Assay,
    *,
    dedup_on: str = "smiles",
) -> List[Assay]:
    """Append ``new`` to ``existing``, deduplicating on ``dedup_on``.

    Returns a *new* list; ``existing`` is not mutated.  When ``new.smiles``
    (or whichever field ``dedup_on`` names) is already present, the
    existing row is **kept** and ``new`` is silently dropped — wet-lab
    measurements are immutable so the first-loaded value wins.  Callers
    that want overwrite semantics should filter the list themselves
    before calling :func:`append_assay`.

    The default ``dedup_on="smiles"`` is the project-wide convention —
    the same canonical SMILES is the natural key for a measurement row.
    """
    key = getattr(new, dedup_on, None)
    seen = {getattr(row, dedup_on, None) for row in existing}
    if key in seen:
        return list(existing)
    return [*list(existing), new]


def assays_to_dict(
    assays: Iterable[Assay],
    *,
    metric: str = "pIC50",
) -> Dict[str, float]:
    """Flatten a sequence of :class:`Assay` rows to ``{smiles: value}``.

    Only rows whose ``outcome_metric`` matches the ``metric`` argument
    are kept (defaults to ``"pIC50"`` to match the canonical reward
    channel).  Rows whose value is NaN are skipped.  When multiple
    rows share a SMILES, the **first** one wins; callers wanting
    aggregation should reduce the list before calling this helper.

    Parameters
    ----------
    assays : Iterable[Assay]
        Source rows (e.g. output of :func:`load_assays`).
    metric : str, default ``"pIC50"``
        Outcome-metric filter.  Case-insensitive.

    Returns
    -------
    dict[str, float]
        Mapping from canonical SMILES to outcome value.  Empty dict
        when no rows match.
    """
    out: Dict[str, float] = {}
    for row in assays:
        if not isinstance(row, Assay):
            continue
        if str(row.outcome_metric or "").strip().lower() != metric.lower():
            continue
        val = row.outcome_value
        try:
            if val != val:  # NaN check
                continue
        except Exception:
            continue
        if not row.smiles:
            continue
        if row.smiles in out:
            continue
        out[row.smiles] = float(val)
    return out


# ---------------------------------------------------------------------------
# Optional CLI helper — useful from ``recalibrate_from_assay.py``.
# ---------------------------------------------------------------------------


def describe_assays(assays: Sequence[Assay]) -> Dict[str, object]:
    """Return a small dict summary suitable for JSON serialisation.

    Used by :mod:`molmetal.scripts.recalibrate_from_assay` for its
    calibration report.  The schema is intentionally narrow so the
    JSON consumer does not need to mirror :class:`Assay`.
    """
    metric_counts: Dict[str, int] = {}
    cell_line_counts: Dict[str, int] = {}
    for row in assays:
        m = str(row.outcome_metric or "unknown")
        metric_counts[m] = metric_counts.get(m, 0) + 1
        cl = str(row.cell_line or "unknown")
        cell_line_counts[cl] = cell_line_counts.get(cl, 0) + 1
    return {
        "n_assays": len(assays),
        "metric_counts": metric_counts,
        "cell_line_counts": cell_line_counts,
        "collaborators": sorted({str(r.collaborator)
                                 for r in assays
                                 if r.collaborator}),
    }


# ---------------------------------------------------------------------------
# Backwards-compat alias for callers that import ``WetlabMeasurement``.
# ---------------------------------------------------------------------------


try:
    WetlabMeasurement = Assay  # type: ignore[misc,assignment]
except NameError:  # pragma: no cover
    WetlabMeasurement = None  # type: ignore[assignment]
