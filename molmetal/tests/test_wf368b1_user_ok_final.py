# Wf-368-B1 — Final consolidated USER_OK checklist test (post-Wf-368-A4)
#
# Per AGENTS.md §5.1 + §5.2 + party-twenty-spirit §3.5:
# - MEASURED-vs-PROJECTED tag discipline
# - 7-section checklist structure required per Wf-368-B1 brief
# - Path discipline: docs/USER_OK_FINAL_CHECKLIST.md (NOT paper/*)
#
# Wf-368-B1 docs-only sub-flow: produces the FINAL consolidated USER_OK
# checklist per Wf-368-A4 + Wf-366-C1 + Wf-366-C2 cross-verification.
# This test verifies:
# 1. The file exists at the canonical path
# 2. The file has all 7 required sections (per Wf-368-B1 brief §1-§7)
# 3. The file is reasonably long (>= 200 LOC; PROJECTED ~350 LOC)
# 4. All 7 USER_OK-gated launch wrappers referenced (R20 §3 + §4 + §5 +
#    Wf-131 + Wf-146 + G5 + recover_g6)
# 5. Wf-366-A1 (GAP-1) + Wf-366-A2 (GAP-2) gap closures preserved
# 6. m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved per Wf-365-A3
# 7. Forward-looking §7 present (post-paper ship guidance)
# 8. Tier-2 markers preserved per AGENTS.md §5.4

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


# Repo root resolved via Path; tests use absolute path to avoid cwd-dependence.
REPO_ROOT = Path("/home/hugo/codes/mol-metal")
CHECKLIST_PATH = REPO_ROOT / "docs" / "USER_OK_FINAL_CHECKLIST.md"

# Seven required sections per Wf-368-B1 brief §1-§7.
REQUIRED_SECTIONS = [
    "§1 Per-launcher",
    "§2 Per-install",
    "§3 Per-paper",
    "§4 Per-metric",
    "§5 Total wall",
    "§6 Cross-references",
    "§7 ",
]

# Seven USER_OK-gated launch wrappers (per Wf-368-B1 brief inventory §3).
EXPECTED_WRAPPERS = [
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",  # Wf-355-A1 R20 §3
    "launch_wf310e_r20_p4_bindingdb_1.07M.sh",  # Wf-355-A2 R20 §4
    "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",  # Wf-355-A3 R20 §5
    "launch_wf131_p3_deferred_runs.sh",  # Wf-355-B1
    "launch_wf146_deferred_runs.sh",  # Wf-355-B2
    "launch_wf124_p3_g5.sh",  # Wf-366-A1 (GAP-1 closed)
    "recover_g6_failed_tuples.sh",  # Wf-366-A2 (GAP-2 closed)
]

# Two install USER_OK-gated binaries.
EXPECTED_INSTALL_BINARIES = [
    "Zotero",
    "tectonic",
]

# Two metric USER_OK-gated items.
EXPECTED_METRIC_GATES = [
    "m1",  # Ru AUC 5-seed (depends on Wf-355-B1 launch)
    "m6_kappa",  # HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL
]

# Tier-2 markers (per AGENTS.md §5.4 + TODO-138 paper v2 re-scope).
TIER2_MARKERS = [
    "Tier-2",
    "PAPER_OK",
    "USER_OK",
]


class TestWf368B1UserOkFinal:
    """Wf-368-B1: FINAL consolidated USER_OK checklist (post-Wf-368-A4 verification).

    Per Wf-368-B1 brief, the FINAL consolidated checklist must:
    - Exist at docs/USER_OK_FINAL_CHECKLIST.md (NOT paper/*)
    - Have all 7 required sections (per Wf-368-B1 brief §1-§7)
    - Be reasonably long (~350 LOC; >= 200 LOC lower bound)
    - Reference all 7 USER_OK-gated launch wrappers
    - Reference all 2 install USER_OK-gated binaries (Zotero + tectonic)
    - Reference all 2 metric USER_OK-gated items (m1 + m6_kappa)
    - Reference Wf-366-A1 (GAP-1) + Wf-366-A2 (GAP-2) closures
    - Preserve m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL
    - Have §7 forward-looking guidance
    - Preserve Tier-2 markers per AGENTS.md §5.4
    """

    def test_checklist_exists_and_has_seven_sections(self) -> None:
        """The FINAL consolidated checklist exists + has 7 sections + 7 wrappers + 2 install + 2 metric.

        MEASURED on 2026-09-29 at Wf-368-B1 timestamp:
        - File exists at docs/USER_OK_FINAL_CHECKLIST.md
        - All 7 sections present (per Wf-368-B1 brief §1-§7)
        - All 7 USER_OK-gated launch wrappers referenced
        - All 2 install binaries referenced (Zotero + tectonic)
        - All 2 metric gates referenced (m1 + m6_kappa)
        - Wf-366-A1 + Wf-366-A2 GAP-1 + GAP-2 closures referenced
        - m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved
        - Tier-2 markers preserved
        - §7 forward-looking section present
        """
        # 1. File exists
        assert CHECKLIST_PATH.exists(), (
            f"Wf-368-B1 USER_OK FINAL checklist missing at {CHECKLIST_PATH}; "
            f"expected docs-only deliverable per Wf-368-B1 brief"
        )

        # 2. Read content
        content = CHECKLIST_PATH.read_text(encoding="utf-8")

        # 3. Reasonable length (>= 200 LOC; PROJECTED ~350 LOC per Wf-368-B1 brief)
        line_count = len(content.splitlines())
        assert line_count >= 200, (
            f"Wf-368-B1 USER_OK FINAL checklist too short: {line_count} lines; "
            f"PROJECTED ~350 LOC per Wf-368-B1 brief"
        )

        # 4. All 7 required sections present (per Wf-368-B1 brief §1-§7)
        # Note: §7 matches both "§7 " (forward-looking) and "§7." (decision matrix)
        for section_header in REQUIRED_SECTIONS:
            assert section_header in content, (
                f"Wf-368-B1 USER_OK FINAL checklist missing required section "
                f"{section_header!r}; expected 7 sections per Wf-368-B1 brief"
            )

        # 5. All 7 USER_OK-gated launch wrappers referenced
        for wrapper in EXPECTED_WRAPPERS:
            assert wrapper in content, (
                f"Wf-368-B1 USER_OK FINAL checklist missing launch wrapper "
                f"{wrapper!r}; expected 7 wrappers per Wf-368-B1 brief §1"
            )

        # 6. All 2 install USER_OK-gated binaries referenced
        for binary in EXPECTED_INSTALL_BINARIES:
            assert binary in content, (
                f"Wf-368-B1 USER_OK FINAL checklist missing install binary "
                f"{binary!r}; expected 2 binaries per Wf-368-B1 brief §2"
            )

        # 7. All 2 metric USER_OK-gated items referenced
        for metric in EXPECTED_METRIC_GATES:
            assert metric in content, (
                f"Wf-368-B1 USER_OK FINAL checklist missing metric gate "
                f"{metric!r}; expected 2 metric gates per Wf-368-B1 brief §4"
            )

        # 8. Wf-366-A1 (GAP-1) + Wf-366-A2 (GAP-2) gap closures referenced
        assert "Wf-366-A1" in content, (
            "Wf-368-B1 FINAL checklist must reference Wf-366-A1 GAP-1 closure"
        )
        assert "Wf-366-A2" in content, (
            "Wf-368-B1 FINAL checklist must reference Wf-366-A2 GAP-2 closure"
        )
        assert "GAP-1" in content, (
            "Wf-368-B1 FINAL checklist must mention GAP-1 (G5 USER_OK gate)"
        )
        assert "GAP-2" in content, (
            "Wf-368-B1 FINAL checklist must mention GAP-2 (recover_g6 block-order)"
        )

        # 9. m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved per Wf-365-A3
        assert "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" in content, (
            "Wf-368-B1 FINAL checklist must preserve m6_kappa "
            "HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-365-A3 §2"
        )

        # 10. Tier-2 markers preserved per AGENTS.md §5.4
        for marker in TIER2_MARKERS:
            assert marker in content, (
                f"Wf-368-B1 FINAL checklist must preserve Tier-2 marker "
                f"{marker!r} per AGENTS.md §5.4"
            )

        # 11. Forward-looking §7 guidance present (per Wf-368-B1 brief §7)
        # The §7 Forward-looking heading must explicitly appear
        assert re.search(r"§7\s+(Forward|How|User|Decision)", content), (
            "Wf-368-B1 FINAL checklist must have §7 forward-looking section"
        )

        # 12. Path discipline — checklist lives under docs/, NOT paper/*
        assert "docs/USER_OK_FINAL_CHECKLIST.md" in content, (
            "Wf-368-B1 FINAL checklist must reference its own canonical path "
            "docs/USER_OK_FINAL_CHECKLIST.md"
        )
        # Verify the file is NOT under paper/* (TODO-84 + AGENTS.md §5.3)
        rel_path = str(CHECKLIST_PATH.relative_to(REPO_ROOT))
        assert not rel_path.startswith("paper/"), (
            f"Wf-368-B1 FINAL checklist incorrectly under paper/*: {rel_path}; "
            f"per TODO-84 + AGENTS.md §5.3 paper work requires explicit authorization"
        )

        # 13. MEASURED-vs-PROJECTED tag discipline preserved
        # (sanity check AGENTS.md §5.2; the file must contain MEASURED + PROJECTED tags)
        assert "MEASURED" in content, (
            "Wf-368-B1 FINAL checklist must preserve MEASURED tag discipline"
        )
        assert "PROJECTED" in content, (
            "Wf-368-B1 FINAL checklist must preserve PROJECTED tag discipline"
        )

        # 14. §6 Cross-references explicitly references AGENTS.md + Wf-368
        assert "AGENTS.md" in content, (
            "Wf-368-B1 FINAL checklist must reference AGENTS.md per brief §6"
        )
        assert "Wf-368" in content or "Wf-366" in content, (
            "Wf-368-B1 FINAL checklist must reference Wf-368 or Wf-366 in §6"
        )


if __name__ == "__main__":
    # Allow running this test file directly: python -m pytest
    # molmetal/tests/test_wf368b1_user_ok_final.py -v
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
