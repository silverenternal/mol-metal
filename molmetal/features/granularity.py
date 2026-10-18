"""Granularity consistency gate (FIX-P2 / DEF-6).

The cohort is built per *complex* (``ligand_canonical``), but the historical
feature builder featurized the *main fragment* (``ligand_main_fragment``).  Both
columns exist, both are strings, nothing type-checks the pairing — so the
mismatch was silent for the entire pre-audit history and only surfaced when the
two columns were featurized side by side (5,861 complexes → 2,846 fragments).

This module makes the pairing explicit and checkable:

* :func:`granularity_of_spec` reads the unit off the spec, not off a script name.
* :func:`assert_granularity` is the runtime guard a builder script calls.
* :func:`check_artifact` is the offline rule the pre-flight gate applies to a
  metrics JSON: an artifact that reports a complex-grouped split while
  featurizing the main fragment is internally inconsistent.

The rule is deliberately one-directional.  Fragment-grouped CV on fragment
features is a legitimate (if optimistic) protocol; complex-grouped CV on
fragment features is a claim the matrix cannot support.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

GRANULARITY_FRAGMENT = "fragment"
GRANULARITY_COMPLEX = "complex"

#: Ligand column -> the evaluation unit it defines.
_SOURCE_TO_GRANULARITY = {
    "ligand_main_fragment": GRANULARITY_FRAGMENT,
    "ligand_canonical": GRANULARITY_COMPLEX,
}


def granularity_of_source(source: str) -> str:
    try:
        return _SOURCE_TO_GRANULARITY[source]
    except KeyError:
        raise ValueError(
            f"unknown ligand source {source!r}; expected one of "
            f"{sorted(_SOURCE_TO_GRANULARITY)}"
        )


def granularity_of_spec(spec: Any) -> str:
    """Read the evaluation unit off a :class:`FeatureSpec`."""
    return granularity_of_source(spec.ligand.source)


def assert_granularity(bundle: Any, expected: str) -> None:
    """Raise if the bundle was not built at the unit the caller is scoring."""
    actual = granularity_of_spec(bundle.spec)
    if actual != expected:
        raise ValueError(
            f"granularity mismatch: matrix was built at {actual!r} "
            f"(source={bundle.spec.ligand.source!r}) but the cohort is scored "
            f"at {expected!r}. Featurizing the main fragment while grouping by "
            f"the complex is the DEF-6 mismatch."
        )


def check_artifact(
    payload: Mapping[str, Any], *, path: Optional[str] = None
) -> List[str]:
    """Offline consistency rule for a metrics JSON.

    A report that scores a ``canonical_grouped`` split must say which ligand
    object produced its headline number, and that object must be the complex.
    Ablation reports legitimately contain fragment arms; the rule applies to the
    arm the artifact declares as its result (``best_arm``), not to every row.
    """
    where = f"{path}: " if path else ""
    splits = payload.get("splits")
    if not isinstance(splits, Mapping) or "canonical_grouped" not in splits:
        return []

    design = payload.get("design") if isinstance(payload.get("design"), Mapping) else {}
    headline = (
        payload.get("best_arm")
        or payload.get("best_arm_canonical_split")
        or design.get("headline_arm")
    )

    source = None
    arm_sources = design.get("arm_ligand_source")
    if isinstance(arm_sources, Mapping) and headline in arm_sources:
        source = arm_sources[headline]
    elif design.get("ligand_source"):
        source = design["ligand_source"]
    else:
        block = design.get("ligand_block")
        if isinstance(block, str):
            if "ligand_main_fragment" in block:
                source = "ligand_main_fragment"
            elif "ligand_canonical" in block:
                source = "ligand_canonical"

    if source is None:
        return [
            f"{where}reports a canonical_grouped split but does not declare the "
            f"ligand source of its headline arm; granularity cannot be checked "
            f"(DEF-6)"
        ]
    if granularity_of_source(str(source)) != GRANULARITY_COMPLEX:
        return [
            f"{where}canonical_grouped headline ({headline!r}) scored on "
            f"{source!r} features; grouping by the complex requires featurizing "
            f"the complex (DEF-6)"
        ]
    return []


def requires_check(payload: Mapping[str, Any]) -> bool:
    """In scope when the artifact reports a complex-grouped split."""
    splits = payload.get("splits")
    return isinstance(splits, Mapping) and "canonical_grouped" in splits
