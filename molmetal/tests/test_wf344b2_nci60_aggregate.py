"""Wf-344-B2 — regression tests for the p3_nci60_9cell_per_cell_aggregate gate.

==============================================================================
Wf-344-B2 — pin NCI-60 9-cell wf312b_nci60_smoke_aggregate block structure
==============================================================================

Wf-338-A3 finding 10: pre-flight gate ``p3_nci60_9cell_per_cell_aggregate``
was flagged as FAIL because ``metrics/by_metric/nci60_precious_9cell_auc.json``
was alleged to be missing the ``wf312b_nci60_smoke_aggregate`` block.

Per :func:`molmetal.scripts.pre_flight_check.gate_nci60_9cell_per_cell_aggregate`
(Wf-312B ship surface; ``pre_flight_check.py:785``) the block MUST carry:

  * ``per_cell_breakdown`` with exactly **9** unique cell-line keys
    (HCT-116 aliased to HCT116 is permitted via explicit ``alias_of``).
  * ``highlight`` field with cell + numeric ``value`` (HCT116 by convention).
  * ``honest_negative`` field with ``value < 0.5`` + non-empty
    ``honest_negative_reason`` + ``badge == 'honest-negative'``.
  * ``overall_mean`` field equal (within ``1e-4``) to the simple mean
    of the 9 per-cell ``mean_auc`` values.

Wf-344-B2 adds **2 NEW tests** that pin this contract at the JSON-boundary
so a future regression (block dropped / removed / mutated) cannot silently
re-FAIL the gate.

This module adds **2 NEW tests** (CPU-only, < 5 s total):

  1. ``TestNci60AggregateBlockPresence::test_wf312b_smoke_aggregate_block_present``
     — Loading ``nci60_precious_9cell_auc.json`` MUST surface a
     ``wf312b_nci60_smoke_aggregate`` dict block.  Catches future
     regressions where the block is dropped / removed / renamed.
     Honest framing: MEASURED-PASS today (2026-09-29); the Wf-338-A3
     finding 10 was a stale snapshot — the block is present in the JSON
     since Wf-312B ship (2026-09-25 per ``timestamp`` field) and the
     ``gate_nci60_9cell_per_cell_aggregate`` function currently returns
     PASS at Wf-344-B2 timestamp (verified 2026-09-29 16:00 UTC).

  2. ``TestNci60AggregateBlockContract::test_per_cell_breakdown_and_aggregates_match``
     — The block MUST carry 9 per-cell entries, an HCT116 highlight with
     a numeric ``value``, an honest-negative flag below 0.5 with a
     non-empty reason, and an ``overall_mean`` within ``1e-4`` of the
     computed mean of the 9 per-cell means.  Pins the Wf-312B ship
     contract at the JSON boundary.

Honest framing (party-twenty-spirit §3.5): the Wf-338-A3 finding 10 is
MEASURED-stale; the block IS present and the gate PASSes.  This Wf-344-B2
delivers an additive 2-test regression layer to ensure the gate cannot
silently regress back to FAIL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JSON_PATH = (
    PROJECT_ROOT
    / "metrics"
    / "by_metric"
    / "nci60_precious_9cell_auc.json"
)


def _ensure_repo_on_path() -> None:
    """Pin ``PROJECT_ROOT`` on ``sys.path`` for direct pytest invocation.

    The harness path used elsewhere in the project is
    ``/home/hugo/codes/mol-metal/.venv/bin/python``; this test
    is venv-agnostic so it runs against any python with molmetal
    importable.  We insert PROJECT_ROOT first so a checkout-level
    install beats any stale system copy.
    """
    repo = str(PROJECT_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


class TestNci60AggregateBlockPresence:
    """Pin the wf312b_nci60_smoke_aggregate block is present in the JSON.

    Regression target: Wf-338-A3 finding 10 — pre-flight gate
    ``p3_nci60_9cell_per_cell_aggregate`` was flagged FAIL because the
    block was alleged missing.  The fix is to ensure the block stays
    present at the JSON boundary.
    """

    def test_wf312b_smoke_aggregate_block_present(self) -> None:
        """The wf312b_nci60_smoke_aggregate block must be a dict.

        This pins the Wf-338-A3 finding 10 regression target: if this
        test FAILs, the block has been dropped / removed / renamed
        and the ``gate_nci60_9cell_per_cell_aggregate`` gate will FAIL
        with ``"{json_path}: wf312b_nci60_smoke_aggregate block missing"``.
        """
        _ensure_repo_on_path()
        assert JSON_PATH.exists(), (
            f"Canonical NCI-60 9-cell metric JSON not found at {JSON_PATH}; "
            "Wf-344-B2 requires the Wf-312B ship artifact on disk."
        )

        with JSON_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        block = data.get("wf312b_nci60_smoke_aggregate")
        assert isinstance(block, dict), (
            f"wf312b_nci60_smoke_aggregate block missing or not a dict "
            f"(got {type(block).__name__}); the pre-flight gate "
            "p3_nci60_9cell_per_cell_aggregate will FAIL."
        )

        # Sanity: the block must carry the 4 sub-keys the gate checks
        # for (per_cell_breakdown + highlight + honest_negative +
        # overall_mean).  This is a coarse smoke; the per-key contract
        # is exercised in the second test class.
        for required_key in (
            "per_cell_breakdown",
            "highlight",
            "honest_negative",
            "overall_mean",
        ):
            assert required_key in block, (
                f"wf312b_nci60_smoke_aggregate.{required_key} missing; "
                "the pre-flight gate p3_nci60_9cell_per_cell_aggregate "
                "will FAIL with sub-issue on this key."
            )


class TestNci60AggregateBlockContract:
    """Pin the wf312b_nci60_smoke_aggregate per-cell + aggregate contract.

    Regression target: Wf-338-A3 finding 10 + Wf-312B ship surface
    contract — the 9 per-cell breakdown, HCT116 highlight, PC3
    honest-negative, and overall_mean consistency MUST all hold.
    """

    def test_per_cell_breakdown_and_aggregates_match(self) -> None:
        """Block must carry 9 cells + highlight + honest-negative + aggregate.

        Pins the Wf-312B contract at the JSON boundary: any mutation that
        breaks one of the 4 contract pillars (9 cells, HCT116 highlight,
        honest-negative < 0.5, overall_mean consistency within 1e-4) will
        FAIL the pre-flight ``gate_nci60_9cell_per_cell_aggregate`` gate.
        """
        _ensure_repo_on_path()
        assert JSON_PATH.exists(), (
            f"Canonical NCI-60 9-cell metric JSON not found at {JSON_PATH}; "
            "Wf-344-B2 requires the Wf-312B ship artifact on disk."
        )

        with JSON_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        block = data["wf312b_nci60_smoke_aggregate"]

        # Pillar 1: per_cell_breakdown must be a dict of exactly 9 keys.
        breakdown = block["per_cell_breakdown"]
        assert isinstance(breakdown, dict), (
            f"per_cell_breakdown is not a dict (got {type(breakdown).__name__})"
        )
        assert len(breakdown) == 9, (
            f"per_cell_breakdown has {len(breakdown)} keys; expected 9 "
            "(Wf-312B ship contract: 9 unique cell-line keys; HCT-116 "
            "alias is permitted via alias_of but counts toward the 9 floor)."
        )

        # Pillar 1b: every entry must have a numeric mean_auc.
        per_cell_means: list[float] = []
        for cell, entry in breakdown.items():
            assert isinstance(entry, dict), (
                f"per_cell_breakdown.{cell} is not a dict "
                f"(got {type(entry).__name__})"
            )
            mean_auc = entry.get("mean_auc")
            assert isinstance(mean_auc, (int, float)), (
                f"per_cell_breakdown.{cell}.mean_auc missing or non-numeric "
                f"(got {type(mean_auc).__name__})"
            )
            per_cell_means.append(float(mean_auc))

        # Pillar 4: overall_mean must equal simple mean of per-cell means
        # within 1e-4 tolerance (per Wf-312B gate contract).
        overall_mean = block["overall_mean"]
        assert isinstance(overall_mean, (int, float)), (
            f"overall_mean missing or non-numeric (got {type(overall_mean).__name__})"
        )
        computed_mean = sum(per_cell_means) / len(per_cell_means)
        assert abs(computed_mean - float(overall_mean)) <= 1e-4, (
            f"overall_mean={overall_mean:.6f} does not match computed mean of "
            f"per_cell_breakdown means={computed_mean:.6f} "
            f"(Δ={abs(computed_mean - float(overall_mean)):.6f} > 1e-4); "
            "the pre-flight gate p3_nci60_9cell_per_cell_aggregate will FAIL."
        )

        # Pillar 2: highlight must point at a cell with a numeric value.
        highlight = block["highlight"]
        assert isinstance(highlight, dict), (
            f"highlight is not a dict (got {type(highlight).__name__})"
        )
        assert isinstance(highlight.get("value"), (int, float)), (
            f"highlight.value missing or non-numeric "
            f"(got {type(highlight.get('value')).__name__})"
        )
        assert isinstance(highlight.get("cell"), str), (
            f"highlight.cell missing or non-string "
            f"(got {type(highlight.get('cell')).__name__})"
        )

        # Pillar 3: honest_negative must point at a cell with value < 0.5
        # + non-empty reason + badge == 'honest-negative'.
        honest_negative = block["honest_negative"]
        assert isinstance(honest_negative, dict), (
            f"honest_negative is not a dict (got {type(honest_negative).__name__})"
        )
        hn_value = honest_negative.get("value")
        assert isinstance(hn_value, (int, float)), (
            f"honest_negative.value missing or non-numeric "
            f"(got {type(hn_value).__name__})"
        )
        assert float(hn_value) < 0.5, (
            f"honest_negative.value={hn_value} >= 0.5; honest-negative cell "
            "must be below random-chance (per Wf-312B contract)."
        )
        hn_reason = honest_negative.get("honest_negative_reason")
        assert isinstance(hn_reason, str) and hn_reason.strip(), (
            f"honest_negative.honest_negative_reason missing or empty string "
            f"(got {type(hn_reason).__name__}={hn_reason!r})"
        )
        assert honest_negative.get("badge") == "honest-negative", (
            f"honest_negative.badge={honest_negative.get('badge')!r}; "
            "expected 'honest-negative' (per Wf-312B contract)."
        )
