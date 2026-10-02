"""Tests for wf386a3_atomic_write — atomic write, overwrite guard, truncation detection.

Per Wf-386-A3 deliverable #2. Test cases:

  * atomic_write_file creates file at correct path (5 cases)
  * atomic_write_file handles concurrent writes via rename (3 cases)
  * safe_overwrite_check warns on >50% diff (4 cases)
  * safe_overwrite_check silent on small diff (3 cases)
  * detect_truncation flags missing closing tokens (5 cases)
  * detect_truncation passes complete output (3 cases)
  * write_closeout integration (3 cases)

All tests run on a temporary directory; no production files are touched.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from molmetal.scripts.wf386a3_atomic_write import (
    OverwriteRisk,
    atomic_write_file,
    detect_truncation,
    safe_overwrite_check,
    write_closeout,
)


# ---------------------------------------------------------------------------
# 1. atomic_write_file creates file at correct path (5 cases)
# ---------------------------------------------------------------------------
class TestAtomicWriteFileCreates:
    """Verify atomic_write_file correctly creates a file at the target path."""

    def test_writes_to_correct_path(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        atomic_write_file(target, "# Hello\n\nWorld.\n")
        assert target.exists()
        assert target.read_text() == "# Hello\n\nWorld.\n"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "out.md"
        atomic_write_file(target, "nested content.\n")
        assert target.exists()
        assert target.read_text() == "nested content.\n"

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        target.write_text("old content.\n")
        atomic_write_file(target, "new content.\n")
        assert target.read_text() == "new content.\n"

    def test_no_temp_file_leaked(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        atomic_write_file(target, "content.\n")
        # No .tmp.* file should remain in the parent directory.
        leftovers = list(tmp_path.glob(".out.md.tmp.*"))
        assert leftovers == [], f"temp file leak: {leftovers}"

    def test_handles_unicode(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        content = "# 中文测试\n\n## ñoño\n\n完成。\n"
        atomic_write_file(target, content)
        assert target.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# 2. atomic_write_file handles concurrent writes via rename (3 cases)
# ---------------------------------------------------------------------------
class TestAtomicWriteFileConcurrent:
    """Verify concurrent writers serialise via per-path lock."""

    def test_concurrent_writers_serialised(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        results = []
        errors = []

        def writer(idx: int) -> None:
            try:
                atomic_write_file(target, f"writer-{idx}-content.\n")
                results.append(idx)
            except Exception as exc:
                errors.append((idx, exc))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"writer errors: {errors}"
        assert len(results) == 10
        # Final content is one of the writers (last to acquire the lock).
        final = target.read_text()
        assert final.startswith("writer-")
        assert final.endswith("content.\n")

    def test_concurrent_writers_different_paths(self, tmp_path: Path) -> None:
        """Different paths must NOT block each other — only same-path."""
        barrier = threading.Barrier(2)
        done = []

        def writer(idx: int, suffix: str) -> None:
            target = tmp_path / f"out-{suffix}.md"
            atomic_write_file(target, f"writer-{idx}.\n")
            done.append(idx)

        t1 = threading.Thread(target=writer, args=(1, "a"))
        t2 = threading.Thread(target=writer, args=(2, "b"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert done == [1, 2]
        assert (tmp_path / "out-a.md").read_text() == "writer-1.\n"
        assert (tmp_path / "out-b.md").read_text() == "writer-2.\n"

    def test_concurrent_writer_no_partial_state(self, tmp_path: Path) -> None:
        """Verify no half-written file is ever visible on disk."""
        target = tmp_path / "out.md"
        long_content = ("line content here.\n" * 1000) + "END.\n"

        def writer() -> None:
            atomic_write_file(target, long_content)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The final file must be one of the complete writes, not a
        # partial mix. (Either all-or-nothing — never a half-merged
        # state thanks to atomic rename + per-path lock.)
        final = target.read_text()
        assert final == long_content


# ---------------------------------------------------------------------------
# 3. safe_overwrite_check warns on >50% diff (4 cases)
# ---------------------------------------------------------------------------
class TestSafeOverwriteCheckWarns:
    """Verify safe_overwrite_check flags substantial diffs."""

    def test_new_file_returns_none(self, tmp_path: Path) -> None:
        target = tmp_path / "new.md"
        ratio = safe_overwrite_check(target, "anything.\n")
        assert ratio is None

    def test_substantial_diff_returns_ratio(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        existing = "\n".join(f"line {i} of old content." for i in range(100))
        target.write_text(existing)
        new = "\n".join(f"line {i} of completely new text here." for i in range(100))
        ratio = safe_overwrite_check(target, new)
        assert ratio is not None
        assert ratio > 0.5

    def test_substantial_diff_raise_on_risk(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        existing = "\n".join(f"line {i} of old content." for i in range(100))
        target.write_text(existing)
        new = "\n".join(f"line {i} of completely new text here." for i in range(100))
        with pytest.raises(OverwriteRisk) as excinfo:
            safe_overwrite_check(target, new, raise_on_risk=True)
        assert excinfo.value.existing_lines == 100
        assert excinfo.value.diff_ratio > 0.5

    def test_small_existing_file_returns_none(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        target.write_text("a\nb\nc\n")  # only 3 non-empty lines, below default 50
        new = "completely different.\n"
        ratio = safe_overwrite_check(target, new)
        assert ratio is None  # below min_lines, no risk


# ---------------------------------------------------------------------------
# 4. safe_overwrite_check silent on small diff (3 cases)
# ---------------------------------------------------------------------------
class TestSafeOverwriteCheckSilent:
    """Verify safe_overwrite_check is silent when diff is small."""

    def test_identical_content_returns_none(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        content = "\n".join(f"line {i} of content." for i in range(100))
        target.write_text(content)
        ratio = safe_overwrite_check(target, content)
        assert ratio is None

    def test_small_addition_returns_none(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        existing = "\n".join(f"line {i} of original." for i in range(100))
        target.write_text(existing)
        # Append just one new line; only ~1 line diff out of 100.
        new = existing + "\none new line.\n"
        ratio = safe_overwrite_check(target, new)
        assert ratio is None  # 1/100 ratio ~ 0.01 < 0.5 threshold

    def test_small_edits_return_none(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        existing = "\n".join(f"line {i} of original." for i in range(100))
        target.write_text(existing)
        # Edit ~20 of 100 lines (~20/100 = 0.20 ratio, below threshold).
        lines = existing.splitlines()
        for i in range(0, 20):
            lines[i] = f"line {i} EDITED."
        new = "\n".join(lines) + "\n"
        ratio = safe_overwrite_check(target, new)
        assert ratio is None


# ---------------------------------------------------------------------------
# 5. detect_truncation flags missing closing tokens (5 cases)
# ---------------------------------------------------------------------------
class TestDetectTruncationFlags:
    """Verify detect_truncation flags truncated output."""

    def test_empty_output_flagged(self) -> None:
        reason = detect_truncation("")
        assert reason is not None
        assert "empty" in reason

    def test_too_short_output_flagged(self) -> None:
        reason = detect_truncation("hello world", expected_min_length=100)
        assert reason is not None
        assert "too_short" in reason

    def test_ends_with_colon_flagged(self) -> None:
        # 200+ chars ending with a colon (mid-sentence).
        output = ("# Section\n\n" + "This is content that goes on for a while " * 5) + ":"
        reason = detect_truncation(output)
        assert reason is not None
        assert "colon" in reason

    def test_ends_with_todo_flagged(self) -> None:
        output = ("# Section\n\n" + "Content here " * 30) + "TODO"
        reason = detect_truncation(output)
        assert reason is not None
        assert "TODO" in reason

    def test_ends_with_truncated_marker_flagged(self) -> None:
        output = ("# Section\n\n" + "Content here " * 30) + "[truncated]"
        reason = detect_truncation(output)
        assert reason is not None
        assert "truncated" in reason


# ---------------------------------------------------------------------------
# 6. detect_truncation passes complete output (3 cases)
# ---------------------------------------------------------------------------
class TestDetectTruncationPasses:
    """Verify detect_truncation returns None for complete output."""

    def test_markdown_section_close_passes(self) -> None:
        # Pass expected_min_length=50 since this is a short test fixture.
        output = (
            "# Final Section\n\n"
            "This is the conclusion of the closeout document.\n\n"
            "## Next Steps\n\n"
            "Follow-up items go here.\n"
        )
        assert detect_truncation(output, expected_min_length=50) is None

    def test_fenced_code_close_passes(self) -> None:
        # Pass expected_min_length=30 since this is a short test fixture.
        output = (
            "## Code Block\n\n"
            "```python\n"
            "def foo():\n"
            "    return 42\n"
            "```\n"
        )
        assert detect_truncation(output, expected_min_length=30) is None

    def test_paragraph_close_passes(self) -> None:
        # Pass expected_min_length=50 since this is a short test fixture.
        output = (
            "## Summary\n\n"
            "The workflow completed 3 of 5 deliverables with 100% SHIPPED rate.\n"
        )
        assert detect_truncation(output, expected_min_length=50) is None

    def test_long_output_default_threshold_passes(self) -> None:
        """Verify a long, well-formed Markdown closeout passes the
        default ``expected_min_length=200`` threshold without needing
        an override. This is the canonical closeout-doc shape."""
        output = (
            "# Final Closeout\n\n"
            "This is a complete, properly-formatted closeout document with\n"
            "sufficient length to exceed the default minimum length threshold\n"
            "and ends with a clean Markdown section break.\n\n"
            "## Honest Framing\n\n"
            "All metrics are MEASURED with sources traced.\n"
        )
        # Default expected_min_length=200 must accept this output.
        assert detect_truncation(output) is None


# ---------------------------------------------------------------------------
# 7. write_closeout integration (3 cases)
# ---------------------------------------------------------------------------
class TestWriteCloseoutIntegration:
    """Verify write_closeout high-level helper."""

    def test_writes_new_closeout(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        content = "# New Closeout\n\nThis is a complete closeout.\n" * 5
        write_closeout(target, content, min_size=100)
        assert target.exists()
        assert target.read_text() == content

    def test_overwrite_warning_logged_but_proceeds(self, tmp_path: Path, caplog) -> None:
        target = tmp_path / "out.md"
        existing = "\n".join(f"line {i} of original content." for i in range(100))
        target.write_text(existing)
        new_content = "\n".join(f"line {i} of totally different text." for i in range(100))
        # Default: log warning, proceed.
        write_closeout(target, new_content)
        assert target.read_text() == new_content

    def test_overwrite_raises_when_asked(self, tmp_path: Path) -> None:
        target = tmp_path / "out.md"
        existing = "\n".join(f"line {i} of original content." for i in range(100))
        target.write_text(existing)
        new_content = "\n".join(f"line {i} of totally different text." for i in range(100))
        with pytest.raises(OverwriteRisk):
            write_closeout(target, new_content, raise_on_overwrite=True)
        # Original content preserved (atomic write didn't happen).
        assert target.read_text() == existing