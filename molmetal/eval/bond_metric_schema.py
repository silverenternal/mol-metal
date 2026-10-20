"""Single source of truth for what ``bond_pr_auc`` actually MEASURES.

Wf-A99 (2026-10-20).  Two different quantities have been written under the
same field name, and no artefact recorded which one:

* ``no_bond_ap_v1`` (everything before Wf-A97, 2026-10-19) — average
  precision of the **NO_BOND** one-vs-rest problem, i.e. positive class =
  the pair is NOT bonded.  On a corpus that is ~95% NO_BOND this is the
  majority class, so it reads 0.62-0.99 while the head ranks bonds at
  chance.  It was shipped under a "class-invariant" claim that was false.
* ``existence_ap_v2`` (Wf-A97 onward) — average precision of bond
  **EXISTENCE**, i.e. score ``1 - P(NO_BOND)``, positive = the pair IS
  bonded.  MEASURED 0.2659 on 64 tmQM val clouds where the old field read
  0.6209.

Why a module instead of a comment.  Wf-A97 corrected the trainer, but three
consumers (``wf_cfm_run_census.py``, ``wf_blk_c13_epsilon_constraint.py``,
``wf_blk_g1_bond_head_scaling.py``) read ``bond_pr_auc`` out of run
artefacts written across BOTH eras.  A table that puts 0.62 (majority-class
AP) beside 0.27 (existence AP) and calls the difference a regression is
reading two different metrics.  Nothing in the row said which was which, so
no reader could tell.  This module makes the definition a first-class,
stamped field and gives the readers one guard to refuse the mix.

API
---

>>> bond_ap_definition({"bond_pr_auc_definition": "existence_ap_v2"})
'existence_ap_v2'
>>> bond_ap_definition({"bond_pr_auc": 0.6209})   # pre-Wf-A97 artefact
'undeclared'
>>> bond_ap_legend("undeclared")
'no definition stamped; pre-Wf-A97 artefacts measured the NO_BOND AP'

>>> require_single_definition([{"bond_pr_auc_definition": "existence_ap_v2"}])
'existence_ap_v2'
>>> require_single_definition([{"bond_pr_auc_definition": "existence_ap_v2"},
...                            {"bond_pr_auc_definition": "no_bond_ap_v1"}])
Traceback (most recent call last):
    ...
MixedBondApDefinition: ... 2 definitions ...
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

#: Wf-A97 onward: AP over bond EXISTENCE (positive = pair IS bonded).
BOND_AP_EXISTENCE = "existence_ap_v2"

#: Pre-Wf-A97: AP over NO_BOND (positive = pair is NOT bonded) — majority class.
BOND_AP_NO_BOND_LEGACY = "no_bond_ap_v1"

#: No stamp on the row.  Every artefact written before Wf-A99 lands here;
#: in practice those are ``no_bond_ap_v1``, but the module does not guess.
BOND_AP_UNDECLARED = "undeclared"

BOND_AP_LEGEND: Dict[str, str] = {
    BOND_AP_EXISTENCE: (
        "AP over bond EXISTENCE (score = 1 - P(NO_BOND), positive = bonded)"
    ),
    BOND_AP_NO_BOND_LEGACY: (
        "AP over NO_BOND (positive = NOT bonded) — majority class on a "
        "~95%-NO_BOND corpus; NOT comparable with existence AP"
    ),
    BOND_AP_UNDECLARED: (
        "no definition stamped; pre-Wf-A97 artefacts measured the NO_BOND AP"
    ),
}


class MixedBondApDefinition(RuntimeError):
    """Raised when a comparison would put two ``bond_pr_auc`` definitions together."""

    def __init__(self, groups: Mapping[str, Sequence[str]], context: str = "") -> None:
        self.groups = {k: list(v) for k, v in groups.items()}
        self.context = context
        parts = [
            f"{name}: {len(rows)} row(s) {sorted(rows)[:4]}"
            for name, rows in sorted(self.groups.items())
        ]
        where = f" in {context}" if context else ""
        super().__init__(
            f"bond_pr_auc mixes {len(self.groups)} definitions{where} — "
            + "; ".join(parts)
            + ". These numbers are not comparable; split the table by "
            "bond_pr_auc_definition or re-measure the old arms."
        )


def _get(row: Any, key: str) -> Any:
    """Read ``key`` from a Mapping or an attribute-bearing row (dataclass)."""
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def bond_ap_definition(row: Any) -> str:
    """Return the stamped definition for one row, or :data:`BOND_AP_UNDECLARED`."""
    val = _get(row, "bond_pr_auc_definition")
    if val is None:
        return BOND_AP_UNDECLARED
    text = str(val).strip()
    return text or BOND_AP_UNDECLARED


def bond_ap_legend(definition: str) -> str:
    """Human-readable legend for a definition string."""
    return BOND_AP_LEGEND.get(definition, f"unknown definition {definition!r}")


def group_by_definition(rows: Iterable[Any]) -> Dict[str, List[Any]]:
    """Bucket rows by their stamped definition, preserving input order."""
    out: Dict[str, List[Any]] = {}
    for row in rows:
        out.setdefault(bond_ap_definition(row), []).append(row)
    return out


def require_single_definition(rows: Sequence[Any], *, context: str = "") -> str:
    """Return the one definition shared by ``rows``, or raise.

    Raises :class:`MixedBondApDefinition` when more than one definition is
    present.  The error names the offending rows (up to four labels per
    group) so the caller can find the pre-Wf-A97 arm instead of silently
    averaging a majority-class AP with an existence AP.
    """
    groups = group_by_definition(rows)
    if not groups:
        return BOND_AP_UNDECLARED
    if len(groups) == 1:
        return next(iter(groups))
    labelled: Dict[str, List[str]] = {}
    for definition, members in groups.items():
        labels = []
        for i, row in enumerate(members):
            label = _get(row, "run") or _get(row, "name") or f"row#{i}"
            labels.append(str(label))
        labelled[definition] = labels
    raise MixedBondApDefinition(labelled, context=context)


def stamp_counts(rows: Iterable[Any]) -> Dict[str, int]:
    """``{definition: count}`` — for printing an artefact's provenance."""
    return dict(Counter(bond_ap_definition(r) for r in rows))


__all__ = [
    "BOND_AP_EXISTENCE",
    "BOND_AP_NO_BOND_LEGACY",
    "BOND_AP_UNDECLARED",
    "BOND_AP_LEGEND",
    "MixedBondApDefinition",
    "bond_ap_definition",
    "bond_ap_legend",
    "group_by_definition",
    "require_single_definition",
    "stamp_counts",
]
