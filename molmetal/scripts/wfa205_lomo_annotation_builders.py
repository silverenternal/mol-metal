# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A205 -- value-agnostic annotation builders for the m19 LOMO metric.

WHY THIS MODULE EXISTS
-----------------------
A published metric artefact is "measurement keys + annotation keys".  The
measurement keys come from a producer script.  The annotation keys (caveats,
honest framing, corrections, provenance) were added later, by hand.  A fresh
execution of the producer therefore drops the whole annotation layer
silently, and a reader of the artefact cannot tell that the file was
hand-augmented.

This module rebuilds the *caveat* part of that layer from the values a
producer just measured, so the caveat can be computed again instead of lost.

THE RULE THIS MODULE FOLLOWS
-----------------------------
A caveat that transcribes a measured number, or a metal name, becomes a
live false claim the moment the producer is executed again on other data --
and a false caveat is worse than a missing one.  So every number and every
name in the emitted text is computed from the payload.  Nothing is copied
across from an earlier artefact.

Two things are NOT properties of a measurement, and so are never asserted
as fact here:

  * the withdrawal of an earlier published claim for this metric.  That is
    a dated editorial act recorded in a document, not a function of a
    measurement, so it is emitted as a clearly-labelled pointer only.
  * the seed-ensemble robustness figure, which belongs to a different
    artefact.  It is emitted as a pointer without a number.

PURITY
------
No file access, no clock, no randomness, no mutable module state.  The
measurement payload is the only input; every other input is a keyword
argument with a documented default.

HOW THE PRODUCER FEEDS IT
-------------------------
``train_metal_lomo_wasserstein.py`` computes, per metal, how many unique
ligands the run's own cohort carries, and publishes that under the payload
key ``per_metal_resolvability`` together with the gate's minimum
unique-ligand count.  The caveat builder then splits the folds into the
resolvable subset and the rest, using only those numbers.  A payload
without that block is handled in a degraded mode: the caveat states that
the resolvability split could not be computed instead of inventing it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # the resolvability rule constant lives in exactly one place
    from molmetal.scripts.metal_activity_transfer_probe import (
        MIN_UNIQUE_FOR_GATE as _PROJECT_MIN_UNIQUE_FOR_GATE,
    )
except Exception:  # pragma: no cover - fail closed (see resolve_min_unique_for_gate)
    _PROJECT_MIN_UNIQUE_FOR_GATE = None

#: Decimal places used for every derived figure in the emitted text.
ROUND_DP = 4

#: Payload key under which the producer publishes the per-metal
#: unique-ligand counts and the gate's minimum unique-ligand count.
RESOLVABILITY_PAYLOAD_KEY = "per_metal_resolvability"

#: Accepted spellings of the per-metal count mapping inside that block.
COUNT_KEYS: Tuple[str, ...] = ("n_unique_ligands", "n_unique")

#: Where the withdrawal log for this metric's earlier published claim
#: lives.  Emitted as a pointer; its CONTENT is never asserted here.
WITHDRAWN_CLAIM_LOG_POINTER = "docs/PREDICTIVE_LINE_INDEX.md section 3"

#: The artefact that carries the seed-ensemble robustness figure.  Emitted
#: as a pointer; the figure itself is not derived from a measurement
#: payload and is therefore never stated here.
SEED_ENSEMBLE_ARTEFACT_POINTER = "metrics/by_metric/lomo_seed_ensemble.json"

#: Provenance string quoted at the head of every emitted caveat.
BUILDER_PROVENANCE = "molmetal/scripts/wfa205_lomo_annotation_builders.py"


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------
def _finite_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("the measurement payload carries a non-finite value")
    return out


def _fmt(value: float, dp: int = ROUND_DP) -> str:
    return "{0:.{1}f}".format(float(value), dp)


def _fmt_count(value: int) -> str:
    return "{0:,}".format(int(value))


def _fmt_plain(value: float) -> str:
    """Compact rendering for a gate constant (no trailing zero padding)."""
    return "{0:g}".format(float(value))


# ---------------------------------------------------------------------------
# payload readers
# ---------------------------------------------------------------------------
def fold_means(payload: Mapping[str, Any]) -> Dict[str, float]:
    """Per-metal fold AUCs, in the payload's own order.

    Raises ``ValueError`` when the payload carries no fold mapping: a
    caveat built from an unreadable payload would be a fabricated caveat.
    """
    raw = payload.get("per_fold_auc_mean")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("payload carries no per_fold_auc_mean mapping")
    return {str(metal): _finite_float(value) for metal, value in raw.items()}


def equal_weight_mean(values: Sequence[float]) -> float:
    """Arithmetic mean, i.e. the mean a reader gets from an equal-weight panel."""
    vals = [_finite_float(value) for value in values]
    if not vals:
        raise ValueError("cannot average an empty sequence")
    return float(sum(vals) / len(vals))


def audit_style_mean(values: Sequence[float], dp: int = ROUND_DP) -> float:
    """Mean computed the way the project's resolvability audit computes it.

    Each fold mean is rounded to ``dp`` decimals first, then the subset
    mean is taken and rounded again.  That is the convention behind
    ``metrics/by_metric/lomo_resolvability_audit.json``, and it is the
    convention the published resolvable-subset figure was built with.
    """
    vals = [_finite_float(value) for value in values]
    if not vals:
        raise ValueError("cannot average an empty sequence")
    return float(round(sum(round(value, dp) for value in vals) / len(vals), dp))


def resolvability_counts(payload: Mapping[str, Any]) -> Dict[str, int]:
    """Per-metal unique-ligand counts, if the payload publishes them."""
    block = payload.get(RESOLVABILITY_PAYLOAD_KEY)
    if not isinstance(block, Mapping):
        return {}
    raw: Any = None
    for key in COUNT_KEYS:
        candidate = block.get(key)
        if isinstance(candidate, Mapping):
            raw = candidate
            break
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, int] = {}
    for metal, count in raw.items():
        try:
            out[str(metal)] = int(count)
        except (TypeError, ValueError):
            continue
    return out


def resolve_min_unique_for_gate(
    payload: Mapping[str, Any],
    min_unique_for_gate: Optional[int] = None,
) -> Optional[int]:
    """The gate's minimum unique-ligand count, or ``None`` when unknown.

    Order of preference:

    1. the explicit ``min_unique_for_gate`` argument;
    2. the value the payload publishes inside its resolvability block;
    3. the project's own rule constant, imported from the single module
       that owns it.

    When none of the three is available the answer is ``None`` and the
    resolvability clause is omitted -- the module never falls back to a
    transcribed threshold.
    """
    if min_unique_for_gate is not None:
        return int(min_unique_for_gate)
    block = payload.get(RESOLVABILITY_PAYLOAD_KEY)
    if isinstance(block, Mapping) and block.get("min_unique_for_gate") is not None:
        return int(block["min_unique_for_gate"])
    if _PROJECT_MIN_UNIQUE_FOR_GATE is not None:
        return int(_PROJECT_MIN_UNIQUE_FOR_GATE)
    return None


# ---------------------------------------------------------------------------
# the structured description
# ---------------------------------------------------------------------------
def describe_lomo_headline(
    payload: Mapping[str, Any],
    *,
    min_unique_for_gate: Optional[int] = None,
) -> Dict[str, Any]:
    """Every derived fact the emitted caveat states, as plain data."""
    folds = fold_means(payload)
    computed_mean = equal_weight_mean(list(folds.values()))
    stated = payload.get("value")
    headline = _finite_float(stated) if stated is not None else computed_mean

    counts = resolvability_counts(payload)
    threshold = resolve_min_unique_for_gate(payload, min_unique_for_gate)
    resolvable: List[str] = []
    unresolvable: List[str] = []
    without_counts: List[str] = []
    if threshold is not None and counts:
        for metal in folds:
            count = counts.get(metal)
            if count is None:
                without_counts.append(metal)
                continue
            if count >= threshold:
                resolvable.append(metal)
            else:
                unresolvable.append(metal)

    subset_mean: Optional[float] = None
    subset_mean_audit_style: Optional[float] = None
    if resolvable:
        subset_values = [folds[metal] for metal in resolvable]
        subset_mean = equal_weight_mean(subset_values)
        subset_mean_audit_style = audit_style_mean(subset_values)

    weakest_metal, weakest_value = min(folds.items(), key=lambda kv: (kv[1], kv[0]))

    real_only = payload.get("real_only_fold_value")
    real_only_value: Optional[float] = None
    real_only_metal: Optional[str] = None
    if real_only is not None:
        real_only_value = _finite_float(real_only)
        for metal, value in folds.items():
            if abs(value - real_only_value) <= 1e-9:
                real_only_metal = metal
                break

    gate_target: Optional[float] = None
    stated_gate = payload.get("gate_target")
    if stated_gate is not None:
        gate_target = _finite_float(stated_gate)

    return {
        "folds": folds,
        "n_folds": len(folds),
        "headline": headline,
        "computed_equal_weight_mean": computed_mean,
        "headline_matches_equal_weight_mean": abs(headline - computed_mean) <= 1e-9,
        "headline_gap_to_computed_mean": headline - computed_mean,
        "gate_target": gate_target,
        "headline_below_gate": (
            None if gate_target is None else bool(headline < gate_target)
        ),
        "min_unique_for_gate": threshold,
        "counts": counts,
        "resolvable_metals": resolvable,
        "unresolvable_metals": unresolvable,
        "metals_without_counts": without_counts,
        "resolvable_subset_mean": subset_mean,
        "resolvable_subset_mean_audit_style": subset_mean_audit_style,
        "weakest_metal": weakest_metal,
        "weakest_value": weakest_value,
        "real_only_fold_value": real_only_value,
        "real_only_fold_metal": real_only_metal,
        "gate_target_provenance_present": "gate_target_provenance" in payload,
    }


# ---------------------------------------------------------------------------
# the emitted caveat
# ---------------------------------------------------------------------------
def build_lomo_headline_caveat(
    payload: Mapping[str, Any],
    *,
    min_unique_for_gate: Optional[int] = None,
    withdrawn_claim_pointer: Optional[str] = WITHDRAWN_CLAIM_LOG_POINTER,
    seed_ensemble_pointer: Optional[str] = SEED_ENSEMBLE_ARTEFACT_POINTER,
) -> str:
    """Build the headline caveat for an m19 LOMO measurement payload.

    Every figure and every metal name in the returned text is derived from
    ``payload``; nothing is transcribed from an earlier artefact.  The
    function is pure: the same payload always yields the same text.
    """
    facts = describe_lomo_headline(
        payload, min_unique_for_gate=min_unique_for_gate
    )
    folds: Dict[str, float] = facts["folds"]
    headline = float(facts["headline"])
    parts: List[str] = []

    parts.append(
        "CAVEAT (computed from this payload by {0}; every figure below is "
        "derived from the payload, none of it is transcribed):".format(
            BUILDER_PROVENANCE
        )
    )

    fold_text = ", ".join(
        "{0} {1}".format(metal, _fmt(value)) for metal, value in folds.items()
    )
    parts.append(
        "The headline {0} is the EQUAL-WEIGHT mean over {1} metal folds: {2}.".format(
            _fmt(headline), facts["n_folds"], fold_text
        )
    )

    if not facts["headline_matches_equal_weight_mean"]:
        parts.append(
            "NOTE: the payload's own value field and the equal-weight mean "
            "over the per-metal fold means differ by {0}.".format(
                _fmt(facts["headline_gap_to_computed_mean"])
            )
        )

    threshold = facts["min_unique_for_gate"]
    counts: Dict[str, int] = facts["counts"]
    resolvable: List[str] = facts["resolvable_metals"]
    unresolvable: List[str] = facts["unresolvable_metals"]

    if threshold is None:
        parts.append(
            "RESOLVABILITY NOT COMPUTED: no minimum unique-ligand count is "
            "available for this payload, so the folds that fall below the "
            "gate's minimum unique-ligand count cannot be identified from it."
        )
    elif not counts:
        parts.append(
            "RESOLVABILITY NOT COMPUTED: this payload carries no per-metal "
            "unique-ligand counts, so the folds that fall below the gate's "
            "minimum unique-ligand count cannot be identified from it, and no "
            "resolvable-subset figure is stated here."
        )
    else:
        if unresolvable:
            unresolvable_text = ", ".join(
                "{0} {1}".format(metal, _fmt_count(counts[metal]))
                for metal in unresolvable
            )
            parts.append(
                "{0} of the {1} folds fall below the gate's minimum "
                "unique-ligand count MIN_UNIQUE_FOR_GATE={2} and are therefore "
                "not resolvable for gating: {3}.".format(
                    len(unresolvable),
                    facts["n_folds"],
                    _fmt_count(threshold),
                    unresolvable_text,
                )
            )
        else:
            all_text = ", ".join(
                "{0} {1}".format(metal, _fmt_count(counts[metal]))
                for metal in folds
                if metal in counts
            )
            parts.append(
                "All {0} folds meet the gate's minimum unique-ligand count "
                "MIN_UNIQUE_FOR_GATE={1} (per-metal counts: {2}), so no fold is "
                "excluded from the headline.".format(
                    facts["n_folds"], _fmt_count(threshold), all_text
                )
            )
        if resolvable:
            resolvable_text = ", ".join(
                "{0} {1}".format(metal, _fmt_count(counts[metal]))
                for metal in resolvable
            )
            subset_mean = facts["resolvable_subset_mean_audit_style"]
            parts.append(
                "The resolvable subset is {0}, whose equal-weight mean is "
                "{1}.".format(resolvable_text, _fmt(subset_mean))
            )
        else:
            parts.append(
                "No fold is resolvable under that rule, so no resolvable-subset "
                "figure exists for this payload."
            )
        if facts["metals_without_counts"]:
            parts.append(
                "Folds without a published unique-ligand count: {0}.".format(
                    ", ".join(facts["metals_without_counts"])
                )
            )

    if unresolvable and resolvable:
        parts.append(
            "Because the equal-weight mean above averages in folds that do not "
            "meet the resolvability rule, it must not be quoted as the gated "
            "headline; quote the resolvable-subset mean ({0}) instead.".format(
                _fmt(facts["resolvable_subset_mean_audit_style"])
            )
        )

    gate_target = facts["gate_target"]
    if gate_target is not None:
        side = "below" if facts["headline_below_gate"] else "at or above"
        parts.append(
            "The headline {0} sits {1} this metric's own gate constant "
            "gate_target={2}.".format(
                _fmt(headline), side, _fmt_plain(gate_target)
            )
        )
        if facts["gate_target_provenance_present"]:
            parts.append(
                "This payload records where that constant comes from under "
                "gate_target_provenance."
            )

    parts.append(
        "The weakest fold in the equal-weight mean is {0} at {1}.".format(
            facts["weakest_metal"], _fmt(facts["weakest_value"])
        )
    )
    real_only_value = facts["real_only_fold_value"]
    if real_only_value is not None:
        if facts["real_only_fold_metal"] is not None:
            parts.append(
                "This payload's real_only_fold_value is {0}, which is the {1} "
                "fold.".format(
                    _fmt(real_only_value), facts["real_only_fold_metal"]
                )
            )
        else:
            parts.append(
                "This payload's real_only_fold_value is {0}, which is not any of "
                "the per-metal fold values above.".format(_fmt(real_only_value))
            )

    if withdrawn_claim_pointer:
        parts.append(
            "Withdrawn-claim pointer (a dated editorial fact about an earlier "
            "published claim; not derived from this measurement and not "
            "asserted by this caveat): {0}.".format(withdrawn_claim_pointer)
        )
    if seed_ensemble_pointer:
        parts.append(
            "The seed-ensemble robustness figure for this metric belongs to a "
            "different artefact and is not derived from this measurement: "
            "{0}.".format(seed_ensemble_pointer)
        )

    return " ".join(parts)


__all__ = [
    "BUILDER_PROVENANCE",
    "COUNT_KEYS",
    "RESOLVABILITY_PAYLOAD_KEY",
    "ROUND_DP",
    "SEED_ENSEMBLE_ARTEFACT_POINTER",
    "WITHDRAWN_CLAIM_LOG_POINTER",
    "audit_style_mean",
    "build_lomo_headline_caveat",
    "describe_lomo_headline",
    "equal_weight_mean",
    "fold_means",
    "resolvability_counts",
    "resolve_min_unique_for_gate",
]
