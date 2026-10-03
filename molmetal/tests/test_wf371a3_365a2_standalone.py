"""Wf-371-A3 — Wf-365-A2 standalone closeout file existence + 6-section check.

================================================================
Wf-371-A3 (2026-09-30) — author the MISSING Wf-365-A2 standalone
closeout (per Wf-369-A1 §22.53.5 verdict; Wf-365 final synthesis
caveat #5). Per the computed task brief, this is a docs-only ship:

1. Author `TODO/completed/365a2_winner_cli_flag.md` with 6 sections
   (TL;DR + 5 numbered: Winning path + augmented_mean comparison +
   CLI flag wired + Backward compat + Cross-references).
2. Verify the file exists + has 6 top-level sections.

Honest framing (party-twenty-spirit §3.5):

* Docs-only; no source-code changes; no GPU launch; deferred to Wf-371-A3.
* All assertions reference MEASURED file-system state (Path.exists + regex section headers).
* Project-wide pre-flight v3 gate PRESERVED at 62 PASS / 0 FAIL / 62 TOTAL
  per Wf-335-SYN baseline (NOT re-run by this Wf-371-A3 docs-only task).
* Wall: PROJECTED 5-10 min CPU-only.

This test file covers 1 axis (per the computed task brief §5):
1. ``test_365a2_standalone_closeout`` — the file exists at
   ``TODO/completed/365a2_winner_cli_flag.md`` + has 6 top-level sections
   (TL;DR + §1-§5).
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLOSEOUT_PATH = REPO_ROOT / "TODO" / "completed" / "365a2_winner_cli_flag.md"


# ---------------------------------------------------------------------------
# Test 1: standalone closeout file exists + has 6 sections
# ---------------------------------------------------------------------------
def test_365a2_standalone_closeout() -> None:
    """Verify Wf-365-A2 standalone closeout exists + has 6 sections.

    Covers:
    * File exists at ``TODO/completed/365a2_winner_cli_flag.md``.
    * File size > 1 KB (sanity: not empty + not stub).
    * First line is the closeout title heading ``# Wf-365-A2``.
    * Has 6 top-level sections:
        - §0 TL;DR
        - §1 Winning path (Neural ODE) + comparison vs MAML
        - §2 augmented_mean comparison (Neural ODE 0.4495 vs MAML 0.4288)
        - §3 CLI flag wired (--neural-ode-flow, default OFF)
        - §4 Backward compat preserved (bit-for-bit)
        - §5 Cross-references (Wf-361-B1 + Wf-361-B2 + Wf-365-A3)
    """
    assert CLOSEOUT_PATH.exists(), (
        f"Wf-365-A2 standalone closeout file MISSING at {CLOSEOUT_PATH} "
        f"(per Wf-369-A1 §22.53.5 verdict; Wf-365 final synthesis caveat #5)"
    )

    file_size = CLOSEOUT_PATH.stat().st_size
    assert file_size > 1024, (
        f"Wf-365-A2 standalone closeout is too small ({file_size} bytes); "
        f"expected > 1 KB to indicate substantive content."
    )

    content = CLOSEOUT_PATH.read_text(encoding="utf-8")

    # Title heading
    first_line = content.splitlines()[0]
    assert first_line.startswith("# Wf-365-A2"), (
        f"First line of Wf-365-A2 closeout should be the title heading "
        f"'# Wf-365-A2 ...' but got: {first_line!r}"
    )

    # Six top-level sections (TL;DR + §1-§5)
    section_pattern = re.compile(r"^## §([0-9]) (.+)$", re.MULTILINE)
    matches = section_pattern.findall(content)
    section_numbers = sorted({int(n) for n, _ in matches})

    expected_sections = {0, 1, 2, 3, 4, 5}
    actual_sections = set(section_numbers)

    missing_sections = expected_sections - actual_sections
    assert not missing_sections, (
        f"Wf-365-A2 closeout missing sections §{sorted(missing_sections)}; "
        f"expected sections {sorted(expected_sections)}, "
        f"got {sorted(actual_sections)}."
    )

    # Section heading content sanity (key phrases per task brief)
    expected_phrases = {
        0: "TL;DR",
        1: "Winning path",
        2: "augmented_mean comparison",
        3: "CLI flag wired",
        4: "Backward compat",
        5: "Cross-references",
    }
    for sec_num, sec_title in matches:
        sec_num_int = int(sec_num)
        if sec_num_int in expected_phrases:
            assert expected_phrases[sec_num_int] in sec_title, (
                f"§{sec_num_int} heading should contain "
                f"{expected_phrases[sec_num_int]!r} but got: {sec_title!r}"
            )

    # Key value presence: Neural ODE 0.4495 vs MAML 0.4288 (per §2 task brief)
    assert (
        "0.4495" in content
    ), "Wf-365-A2 §2 should mention Neural ODE augmented_mean 0.4495"
    assert "0.4288" in content, "Wf-365-A2 §2 should mention MAML augmented_mean 0.4288"

    # CLI flag wired (--neural-ode-flow default OFF) per §3 task brief
    assert (
        "--neural-ode-flow" in content
    ), "Wf-365-A2 §3 should mention --neural-ode-flow CLI flag"
    assert (
        "default OFF" in content
        or "default: False" in content
        or "store_true" in content
    ), "Wf-365-A2 §3 should mention --neural-ode-flow default OFF behaviour"

    # Backward compat per §4 task brief
    assert (
        "bit-for-bit" in content
    ), "Wf-365-A2 §4 should mention bit-for-bit backward compat"

    # Cross-references per §5 task brief
    assert (
        "Wf-361-B1" in content and "Wf-361-B2" in content and "Wf-365-A3" in content
    ), "Wf-365-A2 §5 should cross-reference Wf-361-B1 + Wf-361-B2 + Wf-365-A3"
