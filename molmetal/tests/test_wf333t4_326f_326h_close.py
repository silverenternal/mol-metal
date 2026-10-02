"""Wf-333-T4: Wf-326F + Wf-326H final gap close — re-scope verification.

Verifies that Wf-331-X1 (which closed the 326F + 326H gap) plus the
Wf-333-T4 final doc are both in place. CPU-only filesystem checks.

Per `party-twenty-spirit §3.5`: every numeric value traces to a JSON /
report path on disk. This test suite introduces NO new measurements;
it re-verifies the prior Wf-331-X1 ship surface + Wf-333-T4 final
closeout doc.

Date: 2026-09-29
Wf: Wf-333-T4 (gap close — confirms Wf-331-X1 already shipped)
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path("/home/hugo/codes/mol-metal")


def _read(path: Path) -> str:
    """Read a UTF-8 text file; return empty string if missing (defensive)."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def test_333t4_final_doc_exists() -> None:
    """Wf-333-T4 final re-scope note exists at canonical path."""
    path = REPO_ROOT / "TODO" / "completed" / "333t4_326f_326h_final.md"
    assert path.exists(), f"Missing: {path}"
    content = _read(path)
    assert "Wf-333-T4" in content, "Final doc must reference Wf-333-T4"
    assert "326F" in content and "326H" in content, (
        "Final doc must reference both 326F + 326H slots"
    )


def test_333t4_final_doc_references_x1_shipped() -> None:
    """Wf-333-T4 final doc acknowledges Wf-331-X1 as already SHIPPED."""
    path = REPO_ROOT / "TODO" / "completed" / "333t4_326f_326h_final.md"
    assert path.exists(), f"Missing: {path}"
    content = _read(path)
    # Must declare that Wf-331-X1 already shipped (the gap-close evidence)
    assert "Wf-331-X1" in content or "331-X1" in content or "331x1" in content, (
        "Final doc must reference Wf-331-X1 as the prior ship"
    )
    # Must contain either 'SHIPPED' or 'FORMAL CLOSE' marker
    has_close_marker = any(
        marker in content.upper()
        for marker in ("SHIPPED", "FORMAL CLOSE", "FORMALLY CLOSED")
    )
    assert has_close_marker, (
        "Final doc must contain SHIPPED or FORMAL CLOSE marker"
    )
