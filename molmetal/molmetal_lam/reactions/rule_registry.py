"""Public API for reaction-rule registration and vetting (TODO-30 Rank-9 P5.3).

Background
----------
P5.3 in ``TODO/pending/30_pitfall_reinforce_plan.md`` calls for a small
public API so future rules (and tests) can introspect the rule
registry without re-importing the click / metal-coordination modules
or digging through their dataclass internals.

The Mol-Metal codebase ships with two distinct rule registries:

* :data:`molmetal_lam.reactions.beta_reductions.REACTION_RULES` — the
  low-level dict mapping rule name (e.g. ``"CuAAC"``) to the
  :class:`ReactionRule` instance that can be ``reduce``-d on educts.
  Holds 5 click + 2 metal-coordination (MetalLigandExchange /
  AquaExchange) + 13 SyntheMol = 20 rules.
* :data:`molmetal_lam.lam_chem.rules.CLICK_REACTIONS` /
  :data:`molmetal_lam.lam_chem.rules.METAL_COORDINATION_RULES` — the
  public-facing subset that the MCTS expansion path exposes to users,
  with friendly-name aliases (``"cuaac"``, ``"thiol-ene"``,
  ``"metal-ligand-exchange"``).

This module wraps both into a single ``RuleRecord``-shaped view so:

* new rules can be **registered** with metadata (yield range, FG
  compat, partner-tile count) at import time;
* the audit pipeline can **vet** any rule — registered or not — and
  receive a frozen snapshot of its metadata;
* downstream code (audit, paper §5, docking oracle) can ask for
  ``list_rules(scope="Pt")`` to enumerate every rule that fires on
  Pt scaffolds without having to know whether the rule is click,
  metal-coord, or custom.

Design constraints (per the task brief)
---------------------------------------
* Additive only — this module does NOT modify
  ``beta_reductions.py``, ``pt_click_compat.py``, or any existing
  rule file.  We pull metadata at call-time (lazy) or via thin
  helpers wired in by callers (the integration is opt-in).
* CPU-only — no RDKit or ML import at module load.
* Plumbing, not behavior — ``register_rule`` adds a record; it does
  not modify any ``REACTION_RULES`` dict.

RuleRecord shape
----------------
Each registered rule is a :class:`RuleRecord` with 9 fields.  All
fields except ``name``, ``smarts``, and ``scope`` are optional and
default to ``None`` / ``0`` / empty.  ``last_test_result`` is a
``Dict[str, str]`` carrying the most recent pytest result for the
rule (key = test id, value = ``"PASS"`` / ``"FAIL"`` / ``"SKIP"``).

Scope names
-----------
``scope`` is a free-form string but the canonical values are:

    "click"       — azide / alkyne / thiol / boronic-acid cycloadditions
                    (CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling).
    "Pt"          — Pt(II) metal-coordination rules
                    (MetalLigandExchange + AquaExchange).
    "Ru"          — Ru(II) rules (placeholder; the existing rules
                    cover Pt + Au + Pd but not Ru-specific patterns).
    "Au"          — Au(III) rules (placeholder; covered by
                    MetalLigandExchange via the Au_Cl_NH3 pattern).
    "custom"      — anything registered by external callers (research
                    scripts, ad-hoc retrosynthesis experiments).

Multiple scopes can be returned by ``list_rules(scope="Pt")`` —
the match is exact equality on the ``scope`` field.  Use
``list_rules()`` with no argument to enumerate everything.

Honest framing
--------------
* ``vet_rule`` returns metadata *even if the rule has never fired*.
  This is by design — vetting a rule before promoting it to the
  default set is a Tier-1 ship action (P5.3 ETA 11h); we want
  ``vet_rule("new_click_rule")`` to return its declared metadata
  without forcing the caller to call ``reduce`` first.
* The metadata is **declared**, not measured.  ``yield_range`` is
  the literature-cited bracket (0.40-0.95 for CuAAC, 0.60-0.95
  for SPAAC, etc.); ``partner_tiles`` is the *count* of partner
  tiles in the pool (e.g. ``CuAAC`` partners with 36 azide tiles
  and 24 alkyne tiles = 60 partners).  These are populated by
  callers via the registration helper — see ``register_rule``'s
  ``metadata`` dict.
* ``last_test_result`` is populated by the test harness via
  :func:`record_test_result` — *not* by the registration call.
  Without an explicit test record, the field is the empty dict.

Usage
-----
>>> from molmetal_lam.reactions.rule_registry import (
...     register_rule, unregister_rule, list_rules, vet_rule,
... )
>>> register_rule(
...     "CuAAC",
...     smarts="[C:1]#[C:2].[N:3]=[N+:4]=[N-:5]>>[C:1]1=[C:2]..."
...     metadata={"yield_range": (0.40, 0.95),
...               "fg_compat": ["azide", "alkyne"],
...               "partner_tiles": 60},
...     scope="click",
... )
>>> vet_rule("CuAAC")
RuleRecord(name='CuAAC', smarts='[C:1]#[C:2]...', scope='click',
           yield_range=(0.4, 0.95), fg_compat=['azide', 'alkyne'],
           partner_tiles=60, last_test_result={}, registered_at=...)
>>> list_rules(scope="click")
[RuleRecord(name='CuAAC', ...), ...]
>>> unregister_rule("CuAAC")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Public re-exports.
__all__ = [
    "RuleRecord",
    "register_rule",
    "unregister_rule",
    "list_rules",
    "vet_rule",
    "record_test_result",
    "clear_registry",
    "registry_snapshot",
]


# ---------------------------------------------------------------------------
# RuleRecord — the public dataclass.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuleRecord:
    """A single registered rule.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. ``"CuAAC"``.  Unique within the
        global registry.  Case-sensitive (canonical names are
        camelCase: ``"CuAAC"``, ``"SPAAC"``, ``"MetalLigandExchange"``).
    smarts : str
        RDKit reaction SMARTS (with ``>>``) describing the rule.
        Empty string for purely structural rules (ThiolEne) that
        rely on a Python predicate.
    scope : str
        Free-form scope tag.  Canonical values: ``"click"``,
        ``"Pt"``, ``"Ru"``, ``"Au"``, ``"custom"``.  Use a custom
        tag if none of the canonical values fit.
    yield_range : tuple[float, float] or None
        ``(low, high)`` literature-cited isolated yield bracket.
        ``None`` if the rule has no published yield data (e.g.
        purely structural rules, custom research rules).
    fg_compat : list[str]
        Functional groups the rule needs on its educts, in
        canonical form (``"azide"``, ``"alkyne"``,
        ``"primary_amine"``, ``"thiol"``, ...).  Empty list if
        the rule has no FG requirement.
    partner_tiles : int
        Count of partner tiles in the local fragment pool that
        can fire this rule.  ``0`` if the rule cannot be
        expanded via the standard MCTS path (e.g. custom
        retrosynthesis rules that need special educts).
    last_test_result : dict[str, str]
        Mapping ``test_id -> "PASS" | "FAIL" | "SKIP"`` from the
        most recent pytest run for this rule.  Populated by
        :func:`record_test_result`.  Empty dict if the rule has
        never been tested.
    registered_at : float
        Unix timestamp (seconds) of the registration call.
        Set automatically by :func:`register_rule`.

    Notes
    -----
    ``frozen=True`` — the record is immutable.  Mutate state via
    :func:`record_test_result` (which replaces the record), not
    via attribute assignment.
    """

    name: str
    smarts: str
    scope: str
    yield_range: Optional[Tuple[float, float]] = None
    fg_compat: List[str] = field(default_factory=list)
    partner_tiles: int = 0
    last_test_result: Dict[str, str] = field(default_factory=dict)
    registered_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Global registry — process-local dict, intentionally simple.
# ---------------------------------------------------------------------------
#:
#: Process-local registry mapping rule name -> :class:`RuleRecord`.
#: The dict is the single source of truth for what the public
#: ``list_rules`` / ``vet_rule`` API reports.  It is *not* the same
#: as :data:`molmetal_lam.reactions.beta_reductions.REACTION_RULES`
#: — that one holds runnable :class:`ReactionRule` instances; this
#: one holds static metadata for introspection + auditing.
#:
_REGISTRY: Dict[str, RuleRecord] = {}


# ---------------------------------------------------------------------------
# register_rule
# ---------------------------------------------------------------------------
def register_rule(
    name: str,
    smarts: str,
    metadata: Optional[Dict[str, Any]] = None,
    scope: str = "custom",
    *,
    overwrite: bool = False,
) -> RuleRecord:
    """Register a reaction rule in the public registry.

    Parameters
    ----------
    name : str
        Unique rule name.  Must be a non-empty string.
    smarts : str
        RDKit reaction SMARTS (with ``>>``), or empty string for
        purely structural rules.
    metadata : dict, optional
        Static metadata to attach.  Recognised keys (all optional):

        * ``"yield_range"`` — ``(low, high)`` tuple of floats in
          [0, 1].
        * ``"fg_compat"`` — list of functional-group names
          (``"azide"``, ``"alkyne"``, ``"thiol"``, ...).
        * ``"partner_tiles"`` — non-negative int.

        Unrecognised keys are stored in the record's
        ``last_test_result`` (a misnomer — see Notes) and
        ignored at vet time.  Better: only pass the recognised
        keys.
    scope : str, default ``"custom"``
        Scope tag.  See module docstring for canonical values.
    overwrite : bool, default ``False``
        When ``True``, replace an existing record with the same
        name.  When ``False`` (the default), :class:`ValueError`
        is raised on duplicate name.

    Returns
    -------
    RuleRecord
        The freshly-created (or replaced) record.

    Raises
    ------
    ValueError
        If ``name`` is empty, ``scope`` is empty, or a rule with
        ``name`` is already registered and ``overwrite=False``.

    Notes
    -----
    This is the *plumbing* layer — it does not touch
    :data:`molmetal_lam.reactions.beta_reductions.REACTION_RULES`
    or any other runnable registry.  To actually wire a new
    rule into the MCTS, register the ``ReactionRule`` subclass
    in :mod:`beta_reductions` AND call this function with the
    rule's static metadata.  Both calls are additive.
    """
    if not name or not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"register_rule: name must be a non-empty string, got {name!r}"
        )
    if not scope or not isinstance(scope, str):
        raise ValueError(
            f"register_rule: scope must be a non-empty string, got {scope!r}"
        )
    if name in _REGISTRY and not overwrite:
        existing = _REGISTRY[name]
        raise ValueError(
            f"register_rule: rule {name!r} already registered "
            f"(scope={existing.scope!r}, registered_at={existing.registered_at}). "
            f"Pass overwrite=True to replace."
        )
    metadata = metadata or {}
    yield_range = metadata.get("yield_range")
    if yield_range is not None:
        if (not isinstance(yield_range, (tuple, list))
                or len(yield_range) != 2):
            raise ValueError(
                f"register_rule: yield_range must be a (low, high) "
                f"tuple, got {yield_range!r}"
            )
        yield_range = (float(yield_range[0]), float(yield_range[1]))
    fg_compat = list(metadata.get("fg_compat", []))
    partner_tiles = int(metadata.get("partner_tiles", 0))
    if partner_tiles < 0:
        raise ValueError(
            f"register_rule: partner_tiles must be >= 0, got {partner_tiles!r}"
        )
    rec = RuleRecord(
        name=name,
        smarts=smarts or "",
        scope=scope,
        yield_range=yield_range,
        fg_compat=fg_compat,
        partner_tiles=partner_tiles,
        last_test_result={},
        registered_at=time.time(),
    )
    _REGISTRY[name] = rec
    return rec


# ---------------------------------------------------------------------------
# unregister_rule
# ---------------------------------------------------------------------------
def unregister_rule(name: str) -> Optional[RuleRecord]:
    """Remove a rule from the public registry.

    Parameters
    ----------
    name : str
        Rule name to remove.

    Returns
    -------
    RuleRecord or None
        The removed record, or ``None`` if no rule with that name
        was registered.  Callers that need to know whether the
        unregister actually fired should test for ``None``; the
        difference between "never registered" and "already
        removed" is intentionally collapsed.
    """
    return _REGISTRY.pop(name, None)


# ---------------------------------------------------------------------------
# list_rules
# ---------------------------------------------------------------------------
def list_rules(scope: Optional[str] = None) -> List[RuleRecord]:
    """List registered rules, optionally filtered by scope.

    Parameters
    ----------
    scope : str, optional
        When ``None``, return every registered rule.  When a
        string, return only rules whose ``scope`` equals ``scope``
        exactly.  Use a custom string to enumerate e.g.
        ``list_rules(scope="docking-pilot")`` for ad-hoc rules
        registered by a research script.

    Returns
    -------
    list[RuleRecord]
        Snapshot list of records in registration order
        (insertion order, Python 3.7+ dict guarantee).  The list
        is a fresh copy; mutating it does not affect the
        registry.
    """
    if scope is None:
        return list(_REGISTRY.values())
    return [r for r in _REGISTRY.values() if r.scope == scope]


# ---------------------------------------------------------------------------
# vet_rule
# ---------------------------------------------------------------------------
def vet_rule(name: str) -> RuleRecord:
    """Return the metadata record for a rule.

    Parameters
    ----------
    name : str
        Rule name to vet.

    Returns
    -------
    RuleRecord
        The frozen record carrying name, smarts, scope,
        ``yield_range``, ``fg_compat``, ``partner_tiles``,
        ``last_test_result``, ``registered_at``.

    Raises
    ------
    KeyError
        If no rule with that name is registered.  This is
        distinct from ``register_rule``'s duplicate-name error —
        vet is *introspection*, so a missing rule is a hard
        failure (the caller probably mis-spelled the name).

    Notes
    -----
    The return type is the *declared* metadata, not the *measured*
    state.  ``last_test_result`` is the most recent pytest result
    recorded via :func:`record_test_result`; the rule's actual
    chemistry outcomes (e.g. yield on a specific educt pair) are
    the ``rate_predictor``'s job and are NOT exposed here.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"vet_rule: no rule registered under {name!r}.  "
            f"Use list_rules() to see available rules."
        )
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# record_test_result
# ---------------------------------------------------------------------------
def record_test_result(
    name: str,
    test_id: str,
    outcome: str,
) -> RuleRecord:
    """Record a single pytest result for a rule.

    Parameters
    ----------
    name : str
        Rule name.  Must already be registered.
    test_id : str
        Pytest test id, e.g. ``"test_click_reactions.py::test_cuaac_basic"``.
    outcome : str
        One of ``"PASS"``, ``"FAIL"``, ``"SKIP"``, ``"XFAIL"``,
        ``"XPASS"``.  Free-form but the canonical values match
        pytest's long-form outcomes.

    Returns
    -------
    RuleRecord
        A *new* record with the updated ``last_test_result``
        dict.  The original is replaced in the registry
        (the dataclass is frozen, so we cannot mutate in place).

    Raises
    ------
    KeyError
        If no rule with that name is registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"record_test_result: no rule registered under {name!r}."
        )
    if outcome not in ("PASS", "FAIL", "SKIP", "XFAIL", "XPASS"):
        # Soft warn: do not raise — accept any string so callers
        # can record custom outcomes (e.g. "TIMEOUT").
        pass
    existing = _REGISTRY[name]
    merged = dict(existing.last_test_result)
    merged[test_id] = str(outcome)
    new_rec = RuleRecord(
        name=existing.name,
        smarts=existing.smarts,
        scope=existing.scope,
        yield_range=existing.yield_range,
        fg_compat=list(existing.fg_compat),
        partner_tiles=existing.partner_tiles,
        last_test_result=merged,
        registered_at=existing.registered_at,
    )
    _REGISTRY[name] = new_rec
    return new_rec


# ---------------------------------------------------------------------------
# clear_registry
# ---------------------------------------------------------------------------
def clear_registry() -> int:
    """Remove every registered rule.  Test-helper.

    Returns
    -------
    int
        Number of rules removed.
    """
    n = len(_REGISTRY)
    _REGISTRY.clear()
    return n


# ---------------------------------------------------------------------------
# registry_snapshot
# ---------------------------------------------------------------------------
def registry_snapshot() -> Dict[str, Dict[str, Any]]:
    """Return a JSON-serialisable snapshot of the entire registry.

    Returns
    -------
    dict[str, dict]
        Mapping ``rule_name -> record.to_dict()``.  The dict
        shape is stable across Python versions and is safe to
        serialise with ``json.dumps`` (after unwrapping the
        ``yield_range`` tuple if strict JSON is required — by
        default ``json.dumps`` rejects tuples).
    """
    return {name: rec.to_dict() for name, rec in _REGISTRY.items()}


# ---------------------------------------------------------------------------
# Integration helpers — wire into existing rule registries.
# ---------------------------------------------------------------------------
# These helpers pull metadata from the existing rule files at
# call-time and call :func:`register_rule` on the result.  They are
# *opt-in* (callers must invoke them explicitly) so the existing
# registries remain untouched at module import.  This is the
# "wire into existing rule registries (do NOT modify
# beta_reductions.py or pt_click_compat.py themselves; just
# integrate the registry metadata)" requirement from the brief.
# ---------------------------------------------------------------------------
def _safe_import_beta_reductions():
    """Import :mod:`beta_reductions` lazily so this module is importable
    without the full click / metal-coord stack on the path."""
    try:
        from molmetal_lam.reactions import beta_reductions  # type: ignore
        return beta_reductions
    except Exception:
        return None


def _safe_import_pt_click_compat():
    try:
        from molmetal_lam.lam_chem import pt_click_compat  # type: ignore
        return pt_click_compat
    except Exception:
        return None


def wire_click_rules(*, overwrite: bool = False) -> List[RuleRecord]:
    """Register the 5 canonical click rules with default metadata.

    Pulls the rule instances from
    :data:`molmetal_lam.lam_chem.rules.CLICK_REACTIONS` and looks
    up their ``pattern_smiles`` (RDKit reaction SMARTS).  Yield
    brackets and FG-compat are populated from the canonical
    literature values:

        CuAAC          yield 0.40-0.95  FG azide+alkyne
        SPAAC          yield 0.60-0.95  FG azide+cyclooctyne
        ThiolEne       yield 0.50-0.95  FG thiol+alkene
        Suzuki         yield 0.30-0.95  FG boronic_acid+aryl_halide
        AmideCoupling  yield 0.50-0.99  FG acid+amine

    Returns
    -------
    list[RuleRecord]
        Records that were freshly registered (or replaced when
        ``overwrite=True``).  Empty list if the click-rules
        module is not importable.
    """
    meta: Dict[str, Dict[str, Any]] = {
        "CuAAC":         {"yield_range": (0.40, 0.95),
                         "fg_compat":  ["azide", "alkyne"]},
        "SPAAC":         {"yield_range": (0.60, 0.95),
                         "fg_compat":  ["azide", "cyclooctyne"]},
        "ThiolEne":      {"yield_range": (0.50, 0.95),
                         "fg_compat":  ["thiol", "alkene"]},
        "Suzuki":        {"yield_range": (0.30, 0.95),
                         "fg_compat":  ["boronic_acid", "aryl_halide"]},
        "AmideCoupling": {"yield_range": (0.50, 0.99),
                         "fg_compat":  ["acid", "amine"]},
    }
    out: List[RuleRecord] = []
    try:
        from molmetal_lam.lam_chem.rules import CLICK_REACTIONS  # type: ignore
    except Exception:
        return out
    for name, rule in CLICK_REACTIONS.items():
        # CLICK_REACTIONS has duplicate keys (camelCase + lowercase
        # aliases) — only register the canonical names to avoid
        # duplicate-registration errors.
        if name not in meta:
            continue
        smarts = getattr(rule, "pattern_smiles", "") or ""
        out.append(register_rule(
            name=name,
            smarts=smarts,
            metadata=meta[name],
            scope="click",
            overwrite=overwrite,
        ))
    return out


def wire_metal_coordination_rules(
    *, overwrite: bool = False
) -> List[RuleRecord]:
    """Register the 2 metal-coordination rules (MetalLigandExchange +
    AquaExchange) with default metadata.

    Returns
    -------
    list[RuleRecord]
        Records freshly registered (or replaced when
        ``overwrite=True``).
    """
    meta: Dict[str, Dict[str, Any]] = {
        "MetalLigandExchange": {"yield_range": (0.50, 0.95),
                                "fg_compat":  ["leaving_group", "neutral_2e_donor"]},
        "AquaExchange":        {"yield_range": (0.60, 0.99),
                                "fg_compat":  ["leaving_group", "water"]},
    }
    out: List[RuleRecord] = []
    try:
        from molmetal_lam.lam_chem.rules import METAL_COORDINATION_RULES  # type: ignore
    except Exception:
        return out
    for name, rule in METAL_COORDINATION_RULES.items():
        if name not in meta:
            continue
        smarts = getattr(rule, "pattern_smiles", "") or ""
        out.append(register_rule(
            name=name,
            smarts=smarts,
            metadata=meta[name],
            scope="Pt",
            overwrite=overwrite,
        ))
    return out


def wire_all_default_rules(*, overwrite: bool = False) -> List[RuleRecord]:
    """Register both click + metal-coord rules.  Convenience for the
    audit pipeline + smoke tests."""
    return wire_click_rules(overwrite=overwrite) + \
           wire_metal_coordination_rules(overwrite=overwrite)
