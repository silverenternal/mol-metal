"""Wf-337-W3 — wet-lab re-scope marker verification tests.

Per AGENTS.md §5.4 (2026-09-29 user directive "我们发二区就可以了，不用规划湿实验"):
- TODO-138 paper v2 → Tier-2 journal target (re-scope from JCIM Tier-1)
- R20-P6 wet-lab IC50 validation → DEPRECATED + re-scope to in-silico metric lift
- Wf-74 v2 wet-lab outreach → DEPRECATED

These tests verify that the 3 main re-scope markers are present in the
canonical close-out files on disk. CPU-only; no GPU launch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path("/home/hugo/codes/mol-metal")

TODO_138_PATH = REPO_ROOT / "TODO" / "completed" / "138_paper_v2_submission_package.md"
TODO_97_PATH = REPO_ROOT / "TODO" / "completed" / "97_r20_p6_wetlab_ic50.md"
WF74_V2_PATH = REPO_ROOT / "TODO" / "completed" / "336x3_wf74_v2.md"


def _read(path: Path) -> str:
    """Read a file as UTF-8 text, fail loudly if missing."""
    assert path.exists(), f"Missing required file: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: TODO-138 Tier-2 marker
# ---------------------------------------------------------------------------


def test_todo138_tier2_marker_present():
    """TODO-138 paper v2 close-out must carry a Tier-2 re-scope marker.

    Per AGENTS.md §5.4: paper v2 downgraded from JCIM Tier-1 to Tier-2
    journal target (JCAMD / WIREs Comput. Mol. Sci. / J. Mol. Graph. Model.
    / Mol. Divers. / Comput. Biol. Chem.).
    """
    text = _read(TODO_138_PATH)
    # Header must reflect Tier-2 re-scope (case-insensitive contains
    # "TIER-2-RE-SCOPED" or "TIER-2 JOURNAL" in the first ~20 lines)
    head = "\n".join(text.splitlines()[:30])
    has_marker = (
        "TIER-2" in head
        or "Tier-2" in head
        or re.search(r"tier[\-\s]*2", head, re.IGNORECASE) is not None
    )
    assert has_marker, (
        f"TODO-138 close-out missing Tier-2 re-scope marker in header. "
        f"First 30 lines:\n{head}"
    )
    # Additionally require §5.4 cross-reference
    assert (
        "§5.4" in text or "5.4" in text
    ), "TODO-138 must cross-reference AGENTS.md §5.4 (Tier-2 directive)."


# ---------------------------------------------------------------------------
# Test 2: R20-P6 DEPRECATED marker
# ---------------------------------------------------------------------------


def test_r20p6_deprecated_marker_present():
    """R20-P6 (TODO-97) wet-lab IC50 must carry a DEPRECATED marker.

    Per AGENTS.md §5.4: R20-P6 wet-lab IC50 validation DEPRECATED for project
    scope; re-scoped to in-silico metric lift.
    """
    text = _read(TODO_97_PATH)
    head = "\n".join(text.splitlines()[:30])
    has_deprecated = "DEPRECATED" in head
    assert has_deprecated, (
        f"TODO-97 R20-P6 close-out missing DEPRECATED marker in header. "
        f"First 30 lines:\n{head}"
    )
    # Additionally require §5.4 cross-reference
    assert (
        "§5.4" in text or "5.4" in text
    ), "TODO-97 must cross-reference AGENTS.md §5.4 (Tier-2 directive)."


# ---------------------------------------------------------------------------
# Test 3: Wf-74 v2 DEPRECATED marker
# ---------------------------------------------------------------------------


def test_wf74v2_deprecated_marker_present():
    """Wf-74 v2 (Wf-336-X3 close-out) must carry DEPRECATED markers.

    Per AGENTS.md §5.4: Wf-74 v2 wet-lab outreach DEPRECATED.
    The Wf-336-X3 close-out (336x3_wf74_v2.md) is the canonical
    Wf-74 v2 lineage re-scope target; it must contain the DEPRECATED
    marker at least 3 times (header + cross-references).
    """
    text = _read(WF74_V2_PATH)
    count = text.count("DEPRECATED")
    assert count >= 3, (
        f"Wf-74 v2 close-out ({WF74_V2_PATH.name}) has only {count} "
        f"DEPRECATED markers; expected >=3."
    )
    # Must cross-reference §5.4
    assert (
        "§5.4" in text or "5.4" in text
    ), "Wf-74 v2 close-out must cross-reference AGENTS.md §5.4."


# ---------------------------------------------------------------------------
# Test 4: in-silico re-scope signal across all 3 items
# ---------------------------------------------------------------------------


def test_insilico_rescope_signal():
    """All 3 re-scoped items must signal in-silico / computational-only path.

    Per AGENTS.md §5.4: re-scope from wet-lab to in-silico metric lift.
    The keyword set below captures the in-silico / computational re-scope
    signal across all 3 close-outs.
    """
    keywords = (
        "in-silico",
        "in_silico",
        "computational-only",
        "Tier-2",
        "TIER-2",
    )
    paths = [TODO_138_PATH, TODO_97_PATH, WF74_V2_PATH]
    failures = []
    for p in paths:
        text = _read(p)
        if not any(kw in text for kw in keywords):
            failures.append(p.name)
    assert not failures, (
        f"Files missing in-silico / Tier-2 / computational-only re-scope "
        f"signal: {failures}. Per AGENTS.md §5.4, all 3 items must "
        f"signal the in-silico re-scope."
    )


if __name__ == "__main__":
    # Allow `python -m molmetal.tests.test_wf337w3_wetlab_rescope` smoke run
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
