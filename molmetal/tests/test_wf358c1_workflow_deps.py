"""Wf-358-C1 — Tests for the new docs/WORKFLOW_DEPENDENCY_GRAPH.md.

=====================================================================
Per Wf-357-C3 §1 (cross-workflow dependency tracking is currently
ad-hoc) + AGENTS.md §6 (recent Wf history) + Wf-358-C1 task brief.

This test file (1 test = structure verification):

  test_01_workflow_deps_structure

Per AGENTS.md §5.1, this is a docs+code audit (no retrain / panel /
ablation launch), so pre-flight gate (TODO-207) is N/A.  Per
AGENTS.md §5.2, all claims use the MEASURED/PROJECTED tag discipline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEPS_PATH = _REPO_ROOT / "docs" / "WORKFLOW_DEPENDENCY_GRAPH.md"


@pytest.fixture(scope="module")
def deps_text() -> str:
    """Read the workflow dependency graph markdown file once per module."""
    assert (
        DEPS_PATH.exists()
    ), f"docs/WORKFLOW_DEPENDENCY_GRAPH.md must exist at {DEPS_PATH}"
    return DEPS_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Group A — Document structure (1 test)
# ---------------------------------------------------------------------------
def test_01_workflow_deps_structure(deps_text: str) -> None:
    """Wf-358-C1 test 01: docs/WORKFLOW_DEPENDENCY_GRAPH.md must exist
    with 5+ sections + cross-workflow dependency entries.

    Per Wf-357-C3 §1 + Wf-358-C1 task brief: document must contain
    §1 linear chains + §2 cross-workflow dependencies + §3
    per-workflow pre-conditions + §4 per-workflow post-conditions +
    §5 critical path analysis. Plus cross-workflow dep entries that
    explicitly cite non-adjacent workflow pairs.

    CPU-only contract: pure file-system read + regex parse; no GPU
    launch; no external network.
    """
    # 1) File exists at the canonical location.
    assert (
        DEPS_PATH.exists()
    ), f"docs/WORKFLOW_DEPENDENCY_GRAPH.md must exist at {DEPS_PATH}"
    file_size = DEPS_PATH.stat().st_size
    assert file_size > 5_000, (
        f"workflow dependency graph must be substantial (>5KB); "
        f"got {file_size} bytes (target ~200 LOC per Wf-358-C1 brief)"
    )

    # 2) 5 main sections (§1..§5 — per Wf-358-C1 task brief).
    main_section_pattern = re.compile(r"^## §(\d+)\s+", re.MULTILINE)
    main_sections = main_section_pattern.findall(deps_text)
    for required_section in ["1", "2", "3", "4", "5"]:
        assert (
            required_section in main_sections
        ), f"§{required_section} must be present (Wf-358-C1 task brief)"

    # 3) Cross-workflow dependency entries: count explicit
    # `Wf-XXX → Wf-YYY` patterns in §2.
    cross_dep_pattern = re.compile(
        r"\*?\*?Wf-\d{3}[A-Za-z0-9\-]*\*?\*?\s*→\s*\*?\*?Wf-\d{3}[A-Za-z0-9\-]*\*?\*?",
    )
    cross_deps = cross_dep_pattern.findall(deps_text)
    assert len(cross_deps) >= 10, (
        f"document must have ≥10 cross-workflow dependency entries "
        f"(Wf-358-C1 task brief target: 12); got {len(cross_deps)}"
    )

    # 4) Honest framing discipline preserved (per `party-twenty-spirit`
    # §3.5).  Document must contain both MEASURED and PROJECTED tags.
    assert (
        "MEASURED" in deps_text
    ), "document must use MEASURED tag discipline (party-twenty-spirit §3.5)"
    assert (
        "PROJECTED" in deps_text
    ), "document must use PROJECTED tag discipline for non-measured claims"

    # 5) Standing directives preserved (per AGENTS.md §5):
    # §5.1 GPU grant, §5.2 honest framing, §5.3 no paper, §5.4 Tier-2.
    assert (
        "§5.1" in deps_text or "5.1" in deps_text
    ), "AGENTS.md §5.1 GPU grant reference must be preserved"
    assert (
        "§5.2" in deps_text or "5.2" in deps_text
    ), "AGENTS.md §5.2 honest framing reference must be preserved"

    # 6) Source citations: document must cite ≥5 source paths on disk.
    source_citation_count = len(
        re.findall(
            r"(TODO/completed/[^\s\)]+|molmetal/reports/[^\s\)]+\.json"
            r"|molmetal/reports/[^\s\)]+/|docs/[^\s\)]+\.md"
            r"|AGENTS\.md|TODO/INDEX\.md|TODO/README\.md)",
            deps_text,
        )
    )
    assert source_citation_count >= 5, (
        f"document must cite ≥5 source paths; " f"got {source_citation_count} citations"
    )

    # 7) Workflow attribution: document must cite at least 10 distinct
    # Wf-XXX workflow references (Wf-340..358).
    wf_refs = set(re.findall(r"Wf-\d{3}[A-Za-z0-9\-]*", deps_text))
    assert len(wf_refs) >= 10, (
        f"document must cite ≥10 distinct workflow references; "
        f"got {len(wf_refs)}: {sorted(wf_refs)[:5]}..."
    )

    # 8) Critical path mention: document must reference the Wf-340 →
    # Wf-356 → Wf-357 critical path explicitly.
    assert "critical path" in deps_text.lower() or (
        "Wf-340" in deps_text and "Wf-356" in deps_text and "Wf-357" in deps_text
    ), "document must reference the Wf-340 → Wf-356 → Wf-357 critical path"

    # 9) Pre-condition + post-condition sections both present.
    assert "pre-condition" in deps_text.lower(), "§3 must reference pre-conditions"
    assert "post-condition" in deps_text.lower(), "§4 must reference post-conditions"

    print(
        f"  [test_01] deps OK: path={DEPS_PATH.name} "
        f"size={file_size}B sections={len(main_sections)} "
        f"cross_deps={len(cross_deps)} "
        f"wf_refs={len(wf_refs)} "
        f"sources_cited={source_citation_count} "
        f"(Wf-358-C1 target: 5 sections + 12 cross-workflow deps)",
        flush=True,
    )
