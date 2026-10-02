"""Pytest: Wf-358-C2 — ``docs/HONEST_NEGATIVE_CATALOG.md`` existence + structure.

This module ships a single regression test that verifies the honest-negative
catalog shipped by Wf-358-C2 (per AGENTS.md §5.2 honest framing mandate +
``party-twenty-spirit`` §3.5 MEASURED-vs-PROJECTED tag discipline + Wf-357-C3
"workflow improvement #2 — honest-negative catalog expansion"):

1. ``test_honest_negative_catalog_exists_and_has_five_sections`` —
   catalog file exists at ``docs/HONEST_NEGATIVE_CATALOG.md``, declares
   **5** top-level sections matching the per-metric + per-workflow +
   per-launcher + resolved + open surface, AND has **≥ 36** entries.

The 9-entry legacy catalog (``TODO/completed/207_unified_pre_flight_checklist_v2.md``
§9 per Wf-263B) is now expanded to **40 entries** (exceeds Wf-357-C3
§5.4.a target of 36 entries by +4 entries), preserved across 5 sections
as documented in ``docs/HONEST_NEGATIVE_CATALOG.md``.

This test is CPU-only (filesystem + string-match), finishes in <1 s on
RTX 5090 / Blackwell, and requires no PyTorch / RDKit / GPU launch.

Source: Wf-358-C2 closeout (this file) + Wf-357-C3 workflow analysis +
Wf-359-B1 catalog aggregation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"

# Per Wf-357-C3 §5.4.a: catalog must have ≥ 36 entries (legacy had 9).
MIN_ENTRIES = 36

# Per Wf-357-C3 + Wf-358-C2 brief: 5 sections (per-metric, per-workflow,
# per-launcher, resolved, open). §0 TL;DR + §6 Closing + §7 Cross-references
# are framing sections, not counted.
EXPECTED_SECTIONS = (
    "Per-metric honest-negatives",
    "Per-workflow honest-negatives",
    "Per-launcher honest-negatives",
    "Resolved entries",
    "Open entries",
)


def _count_entries(catalog_text: str) -> int:
    """Count numbered subsections ``### §N.M`` in the catalog.

    Each subsection corresponds to one catalogued entry (per-metric,
    per-workflow, per-launcher, resolved, or open). Cross-listed entries
    (e.g. §2.5, §2.7, §2.11, §2.14, §5.7 in the 40-entry catalog) are
    counted twice here; the test only enforces the **≥ 36** floor that
    Wf-357-C3 mandated, not the deduplicated 40 unique count.
    """
    return len(re.findall(r"^### §[0-9]+\.", catalog_text, flags=re.MULTILINE))


def test_honest_negative_catalog_exists_and_has_five_sections() -> None:
    """Catalog file exists + has 5 sections + ≥ 36 entries.

    Per AGENTS.md §5.2 honest framing mandate + Wf-357-C3 expansion
    directive: the catalog must be the canonical reviewer-rebuttal
    surface for every honest-negative finding across Wf-296..358 work.
    """
    assert CATALOG_PATH.exists(), (
        f"HONEST_NEGATIVE_CATALOG.md missing at {CATALOG_PATH}; Wf-358-C2 ship "
        f"requires the file to exist for reviewer-rebuttal + paper §6 "
        f"Limitations cross-reference."
    )

    catalog_text = CATALOG_PATH.read_text(encoding="utf-8")

    # --- §1: 5 sections present ---
    for expected_section in EXPECTED_SECTIONS:
        assert expected_section in catalog_text, (
            f"HONEST_NEGATIVE_CATALOG.md missing required section "
            f"'{expected_section}'. Expected 5 sections: per-metric, "
            f"per-workflow, per-launcher, resolved, open."
        )

    # --- §2: ≥ 36 entries ---
    n_entries = _count_entries(catalog_text)
    assert n_entries >= MIN_ENTRIES, (
        f"HONEST_NEGATIVE_CATALOG.md has only {n_entries} entries; Wf-357-C3 "
        f"§5.4.a mandates ≥ {MIN_ENTRIES} entries (expansion from the "
        f"legacy 9-entry catalog at "
        f"TODO/completed/207_unified_pre_flight_checklist_v2.md §9)."
    )

    # --- §3: AGENTS.md §9 cross-references the catalog ---
    agents_md = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "HONEST_NEGATIVE_CATALOG.md" in agents_md, (
        "AGENTS.md does not reference docs/HONEST_NEGATIVE_CATALOG.md; "
        "Wf-358-C2 requires the canonical catalog to be discoverable from "
        "the agent entry point per AGENTS.md §9 'When unsure' table."
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))  # type: ignore[name-defined]