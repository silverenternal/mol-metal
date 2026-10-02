# Wf-366-C2 — USER_OK FINAL CHECKLIST test (post-Wf-366 gap closures)
#
# Per AGENTS.md §5.1 + §5.2 + party-twenty-spirit §3.5:
# - MEASURED-vs-PROJECTED tag discipline
# - 6-section checklist structure required
# - Path discipline: docs/USER_OK_FINAL_CHECKLIST.md (NOT paper/*)
#
# Wf-366-C2 docs-only sub-flow: produces the consolidated USER_OK final
# checklist at docs/USER_OK_FINAL_CHECKLIST.md. This test verifies:
# 1. The file exists at the canonical path
# 2. The file has all 6 required sections (per Wf-366-C2 brief §1-§6)
# 3. The file is reasonably long (>= 200 LOC; PROJECTED ~300 LOC)
# 4. Wf-366-A1 + Wf-366-A2 gap closures are referenced
# 5. m6_kappa honest-negative is preserved per Wf-365-A3
# 6. Path discipline — checklist lives under docs/, NOT paper/*

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


# Repo root resolved via Path; tests use absolute path to avoid cwd-dependence.
REPO_ROOT = Path("/home/hugo/codes/mol-metal")
CHECKLIST_PATH = REPO_ROOT / "docs" / "USER_OK_FINAL_CHECKLIST.md"

# Six required sections per Wf-366-C2 brief.
REQUIRED_SECTIONS = [
    "§1 Per-launcher",
    "§2 Per-install",
    "§3 Per-paper",
    "§4 Per-metric",
    "§5 Total wall",
    "§6 Cross-references",
]


class TestWf366C2UserOkFinal:
    """Wf-366-C2: USER_OK FINAL CHECKLIST (consolidated post-Wf-366 gap closures).

    Per Wf-366-C2 brief, the consolidated checklist must:
    - Exist at docs/USER_OK_FINAL_CHECKLIST.md (NOT paper/*)
    - Have all 6 required sections
    - Be reasonably long (~300 LOC; >= 200 LOC lower bound)
    - Reference Wf-366-A1 (GAP-1) + Wf-366-A2 (GAP-2) closures
    - Preserve m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL
    """

    def test_checklist_exists_and_has_six_sections(self) -> None:
        """The consolidated checklist exists, has 6 sections, is reasonably long.

        MEASURED on 2026-09-29 at Wf-366-C2 timestamp:
        - File exists at docs/USER_OK_FINAL_CHECKLIST.md
        - All 6 sections present (per Wf-366-C2 brief §1-§6)
        - Length >= 200 LOC (PROJECTED ~300 LOC per task brief)
        - Wf-366-A1 + Wf-366-A2 references present
        - m6_kappa honest-negative preserved
        """
        # 1. File exists
        assert CHECKLIST_PATH.exists(), (
            f"Wf-366-C2 USER_OK final checklist missing at {CHECKLIST_PATH}; "
            f"expected docs-only deliverable per Wf-366-C2 brief"
        )

        # 2. Read content
        content = CHECKLIST_PATH.read_text(encoding="utf-8")

        # 3. Reasonable length (>= 200 LOC; PROJECTED ~300 LOC)
        line_count = len(content.splitlines())
        assert line_count >= 200, (
            f"USER_OK final checklist too short: {line_count} lines; "
            f"PROJECTED ~300 LOC per Wf-366-C2 brief"
        )

        # 4. All 6 required sections present
        for section_header in REQUIRED_SECTIONS:
            assert section_header in content, (
                f"Wf-366-C2 USER_OK final checklist missing required section "
                f"{section_header!r}; expected 6 sections per Wf-366-C2 brief"
            )

        # 5. Wf-366-A1 (GAP-1) + Wf-366-A2 (GAP-2) gap closures referenced
        assert "Wf-366-A1" in content, (
            "Wf-366-C2 final checklist must reference Wf-366-A1 GAP-1 closure"
        )
        assert "Wf-366-A2" in content, (
            "Wf-366-C2 final checklist must reference Wf-366-A2 GAP-2 closure"
        )
        assert "GAP-1" in content, (
            "Wf-366-C2 final checklist must mention GAP-1 (G5 USER_OK gate)"
        )
        assert "GAP-2" in content, (
            "Wf-366-C2 final checklist must mention GAP-2 (recover_g6 block-order)"
        )

        # 6. m6_kappa honest-negative preserved per Wf-365-A3
        assert "HONEST_NEGATIVE_PARTIAL_STRUCTURAL" in content, (
            "Wf-366-C2 final checklist must preserve m6_kappa "
            "HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-365-A3 §2"
        )
        assert "m6_kappa" in content, (
            "Wf-366-C2 final checklist must reference m6_kappa metric gate"
        )

        # 7. Path discipline — checklist lives under docs/, NOT paper/*
        assert "docs/USER_OK_FINAL_CHECKLIST.md" in content, (
            "Wf-366-C2 final checklist must reference its own canonical path "
            "docs/USER_OK_FINAL_CHECKLIST.md"
        )
        # Verify the file is NOT under paper/* (TODO-84 + AGENTS.md §5.3)
        assert "paper/" not in str(CHECKLIST_PATH.relative_to(REPO_ROOT)), (
            f"Wf-366-C2 final checklist incorrectly under paper/*: {CHECKLIST_PATH}; "
            f"per TODO-84 + AGENTS.md §5.3 paper work requires explicit authorization"
        )

        # 8. MEASURED-vs-PROJECTED tag discipline preserved
        # (sanity check AGENTS.md §5.2; the file must contain MEASURED + PROJECTED tags)
        assert "MEASURED" in content, (
            "Wf-366-C2 final checklist must preserve MEASURED tag discipline"
        )
        assert "PROJECTED" in content, (
            "Wf-366-C2 final checklist must preserve PROJECTED tag discipline"
        )

        # 9. USER_OK gate pattern documented for all 7 wrappers
        expected_wrappers = [
            "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",  # Wf-355-A1
            "launch_wf310e_r20_p4_bindingdb_1.07M.sh",  # Wf-355-A2
            "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",  # Wf-355-A3
            "launch_wf131_p3_deferred_runs.sh",  # Wf-355-B1
            "launch_wf146_deferred_runs.sh",  # Wf-355-B2
            "launch_wf124_p3_g5.sh",  # Wf-366-A1 (GAP-1 closed)
            "recover_g6_failed_tuples.sh",  # Wf-366-A2 (GAP-2 closed)
        ]
        for wrapper in expected_wrappers:
            assert wrapper in content, (
                f"Wf-366-C2 final checklist must document USER_OK gate for "
                f"{wrapper}"
            )


if __name__ == "__main__":
    # Allow running this test file directly: python -m pytest
    # molmetal/tests/test_wf366c2_user_ok_final.py -v
    import sys

    sys.exit(pytest.main([__file__, "-v"]))