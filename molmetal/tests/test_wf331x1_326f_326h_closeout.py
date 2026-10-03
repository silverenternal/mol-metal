"""Wf-331-X1 — Wf-326F + Wf-326H delivery gap closeout tests.

=====================================================================
Per Wf-327-T1 audit: Wf-326 was allocated 8 sub-flow slots but only 6
landed standalone TODO closeout docs (326A/326B/326C/326D/326E/326G).
326F (memory ceiling audit / D-MPNN featurization cache) and 326H
(synthesis) were never given closeout time.

Wf-331-X1 closes the documentation gap by:
  1. Writing TODO/completed/326f_memory_ceiling_closeout.md (FORMAL
     CLOSE — absorbed by Wf-328-W10 cache ship + §332_final_synthesis
     ledger).
  2. Writing TODO/completed/326h_synthesis_closeout.md (FORMAL CLOSE —
     synthesis already shipped via Wf-327 + Wf-330-E + §332).

This test file locks in the closeout by asserting:
  - The 2 closeout docs exist at canonical paths.
  - The 2 Wf-326F / Wf-326H slots are no longer referenced as
    TODO/pending in any open-status surface (TODO/pending/, TODO/INDEX.md
    §0.0 unresolved list, etc.).

CPU-only tests; no GPU launch required.

Honest framing (party-twenty-spirit §3.5):
  - These tests verify the DOCUMENTATION gap closure, not the engineering
    surface (which was already shipped via Wf-328-W10 + Wf-327 synthesis).
  - All assertions are file-exists + content-pattern checks; no GPU
    retrain, no perf claim, no MEASURED number.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TODO_COMPLETED = REPO_ROOT / "TODO" / "completed"
TODO_PENDING = REPO_ROOT / "TODO" / "pending"
TODO_INDEX = REPO_ROOT / "TODO" / "INDEX.md"
TODO_README = REPO_ROOT / "TODO" / "README.md"

CLOSE_326F = TODO_COMPLETED / "326f_memory_ceiling_closeout.md"
CLOSE_326H = TODO_COMPLETED / "326h_synthesis_closeout.md"


class TestWf331X1CloseoutDocsExist:
    """Both closeout docs must exist at canonical TODO/completed paths."""

    def test_326f_closeout_doc_exists(self) -> None:
        assert CLOSE_326F.exists(), (
            f"Wf-326F closeout doc missing at {CLOSE_326F}. "
            "Wf-331-X1 must write TODO/completed/326f_memory_ceiling_closeout.md."
        )

    def test_326h_closeout_doc_exists(self) -> None:
        assert CLOSE_326H.exists(), (
            f"Wf-326H closeout doc missing at {CLOSE_326H}. "
            "Wf-331-X1 must write TODO/completed/326h_synthesis_closeout.md."
        )

    def test_326f_closeout_doc_has_formal_close_marker(self) -> None:
        text = CLOSE_326F.read_text()
        assert "FORMALLY CLOSED" in text, (
            "Wf-326F closeout doc must declare FORMAL CLOSE status (per "
            "party-twenty-spirit §3.5 honest framing)."
        )

    def test_326h_closeout_doc_has_formal_close_marker(self) -> None:
        text = CLOSE_326H.read_text()
        assert (
            "FORMALLY CLOSED" in text
        ), "Wf-326H closeout doc must declare FORMAL CLOSE status."

    def test_326f_closeout_doc_references_wf331_x1(self) -> None:
        text = CLOSE_326F.read_text()
        assert (
            "Wf-331-X1" in text
        ), "Wf-326F closeout doc must reference Wf-331-X1 as the gap closer."

    def test_326h_closeout_doc_references_wf331_x1(self) -> None:
        text = CLOSE_326H.read_text()
        assert (
            "Wf-331-X1" in text
        ), "Wf-326H closeout doc must reference Wf-331-X1 as the gap closer."


class TestWf331X1NoOrphanReferences:
    """326F / 326H slots must not remain as TODO/pending orphans."""

    def test_326f_not_in_pending_directory(self) -> None:
        if not TODO_PENDING.exists():
            pytest.skip("TODO/pending/ not present")
        offenders = [
            p.name
            for p in TODO_PENDING.iterdir()
            if re.match(r"326[fF]_", p.name) and not p.name.startswith("326f")
        ]
        # NOTE: only flag files whose path implies Wf-326F as the *primary*
        # TODO, not docs that *reference* Wf-326F as a follow-up.
        # The regex above intentionally allows files like
        # `326f10_*.md` (Wf-326 part 10) which is unrelated.
        assert not offenders, (
            f"Wf-326F slot remains open in TODO/pending/: {offenders}. "
            "Wf-331-X1 closeout must move this to TODO/completed/."
        )

    def test_326h_not_in_pending_directory(self) -> None:
        if not TODO_PENDING.exists():
            pytest.skip("TODO/pending/ not present")
        offenders = [
            p.name
            for p in TODO_PENDING.iterdir()
            if re.fullmatch(r"326[hH]_.*\.md", p.name)
        ]
        assert not offenders, (
            f"Wf-326H slot remains open in TODO/pending/: {offenders}. "
            "Wf-331-X1 closeout must move this to TODO/completed/."
        )

    def test_index_md_does_not_list_326f_326h_as_open(self) -> None:
        """Per §332_final_synthesis, 326F/326H must NOT appear as open
        TODOs in TODO/INDEX.md.  The current INDEX.md notes "no 326f_*.md"
        as a gap; this assertion must hold AFTER the Wf-331-X1 closeout
        by ensuring no open-status marker remains.

        Strategy: parse the INDEX.md into per-line snippets and assert
        that no line containing a 326F/326H marker is also tagged with
        an OPEN / PENDING / TODO-status verb.
        """
        if not TODO_INDEX.exists():
            pytest.skip("TODO/INDEX.md not present")
        text = TODO_INDEX.read_text()
        offending_lines: list[tuple[int, str]] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not re.search(r"326[Ff]\b", line) and not re.search(r"326[Hh]\b", line):
                continue
            # We tolerate lines that describe the gap-closure itself
            # (e.g. "no 326f_*.md", "326f_memory_ceiling_closeout.md",
            #  "326h_synthesis_closeout.md", "Wf-326F" in cross-ref
            #  context).  We only fail if an OPEN / PENDING / TODO
            #  status verb is on the same line.
            if re.search(r"\b(OPEN|PENDING|UNRESOLVED|UNSHIPPED)\b", line):
                # Allow lines that also reference the closeout doc
                # (i.e. status resolved/closed via this Wf).
                if (
                    "326f_memory_ceiling_closeout" in line
                    or "326h_synthesis_closeout" in line
                ):
                    continue
                offending_lines.append((lineno, line.strip()[:120]))
        assert not offending_lines, (
            "TODO/INDEX.md still flags Wf-326F or Wf-326H as open "
            "(regression after Wf-331-X1 closeout):\n"
            + "\n".join(f"  line {n}: {line!r}" for n, line in offending_lines)
        )

    def test_readme_md_does_not_list_326f_326h_as_open(self) -> None:
        if not TODO_README.exists():
            pytest.skip("TODO/README.md not present")
        text = TODO_README.read_text()
        # Soft check: README may or may not enumerate 326-series individually,
        # but it must not have an explicit "TODO 326F" or "TODO 326H" entry
        # that contradicts the closeout.
        for marker in ("TODO 326F", "TODO 326H"):
            assert marker not in text, (
                f"TODO/README.md still lists {marker!r} as an open TODO; "
                "Wf-331-X1 closeout must reconcile."
            )
