#!/usr/bin/env python3
"""Wf-416-A1 migration helper — atomic write protocol migration of 10 closeout-drafting scripts.

This is a one-shot migration utility (CPU-only, idempotent). It applies the
Wf-386-A3 atomic write protocol pattern to the 10 closeout-drafting scripts
selected by Wf-416-A1 audit. Each migrated script:

  1. Imports the shared atomic_write_file + safe_overwrite_check + detect_truncation
     from molmetal.scripts.wf386a3_atomic_write via try/except ImportError fallback.
  2. Replaces direct ``path.write_text(content)`` calls with the migration pattern
     that delegates to atomic_write_file when available.
  3. Calls safe_overwrite_check before any overwrite to surface substantial content
     overwrites before they happen.

The migration is idempotent — re-running the helper on an already-migrated script
is a no-op (the import block check skips reinsertion; the Write sites have already
been replaced).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Import block template (idempotent: _WF416_HAS_ATOMIC guard prevents double-write)
_IMPORT_BLOCK = '''try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF416_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF416_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]
'''

# Migration helper template for replacing path.write_text(content)
_MIGRATE_HELPER = '''def _wf416_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
    """Wf-416-A1 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF416_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")

'''


def already_migrated(content: str) -> bool:
    """Return True if the file has already been migrated (has the marker)."""
    return "_WF416_HAS_ATOMIC" in content


def insert_import_block(content: str) -> Tuple[str, bool]:
    """Insert the import block after the last TOP-LEVEL import statement.
    Only considers single-line imports (`import X` or `from X import Y`) —
    multi-line imports (`from X import (\\n A, B, \\n)`) are detected and
    skipped to avoid inserting inside the parenthesis.
    Returns (new_content, changed).
    """
    if already_migrated(content):
        return content, False

    lines = content.splitlines(keepends=True)
    insert_idx = 0
    open_paren_count = 0
    for i, line in enumerate(lines):
        if open_paren_count > 0:
            # Inside a multi-line import — track paren depth
            open_paren_count += line.count("(") - line.count(")")
            continue
        if line.startswith("import ") or line.startswith("from "):
            # Check if this import line has an unclosed parenthesis
            line_open = line.count("(") - line.count(")")
            if line_open > 0:
                # Multi-line import — skip until close paren
                open_paren_count = line_open
                insert_idx = i  # Place insertion BEFORE the multi-line import
                continue
            insert_idx = i + 1
    # Insert the import block after the last import
    block = "\n" + _IMPORT_BLOCK + "\n"
    new_lines = lines[:insert_idx] + [block] + lines[insert_idx:]
    return "".join(new_lines), True


def insert_helper(content: str) -> Tuple[str, bool]:
    """Insert the _wf416_atomic_write_text helper before the first def/class/main statement.
    Returns (new_content, changed). Skipped if already present.
    """
    if "_wf416_atomic_write_text" in content:
        return content, False

    lines = content.splitlines(keepends=True)
    # Find the first top-level def statement at column 0
    insert_idx = None
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__"):
            insert_idx = i
            break
    if insert_idx is None:
        # Fall back to end of file
        insert_idx = len(lines)
    new_lines = lines[:insert_idx] + [_MIGRATE_HELPER] + lines[insert_idx:]
    return "".join(new_lines), True


# Pattern to match: <var>.write_text(<arg>...)
# Captures the variable (any expression ending just before `.write_text`).
# The prefix class deliberately EXCLUDES `)` to avoid capturing the closing
# paren of a previous `_wf416_atomic_write_text(...)` call. The walker
# below uses a balanced-paren walker to find the matching close paren.
_WRITE_TEXT_PATTERN = re.compile(
    r"(?P<prefix>[a-zA-Z_][a-zA-Z_0-9. ()\[\]\"\'/_]*?)\.write_text\(",
    re.MULTILINE,
)
# Post-process: trim trailing `)` and whitespace from the captured prefix
# (the non-greedy quantifier may have absorbed a closing paren of a previous call).
_WRITE_TEXT_PREFIX_BAD_TRAIL = re.compile(r"\)\s*$")


def _find_balanced_paren(text: str, open_idx: int) -> int:
    """Find the matching closing paren for the opening paren at open_idx.
    Returns the index AFTER the closing paren, or -1 if unbalanced.
    """
    assert text[open_idx] == "(", f"Expected '(' at {open_idx}, got {text[open_idx]!r}"
    depth = 1
    i = open_idx + 1
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def replace_write_text_calls(content: str) -> Tuple[str, int]:
    """Replace all `<var>.write_text(<args>)` calls with `_wf416_atomic_write_text(<var>, <args>)`.
    Returns (new_content, n_replacements).

    NOTE: only replaces calls where the args do NOT include `encoding=` — those
    calls use a non-default encoding (e.g. utf-8) and we keep them as direct
    writes for backward compat (atomic_write_file defaults to utf-8 but we want
    explicit pass-through).
    """
    if "_WF416_HAS_ATOMIC" not in content:
        return content, 0  # not migrated yet

    n_replaced = 0
    out_parts: List[str] = []
    i = 0
    while i < len(content):
        m = _WRITE_TEXT_PATTERN.search(content, i)
        if m is None:
            out_parts.append(content[i:])
            break
        # Append everything before the match
        out_parts.append(content[i:m.start()])
        prefix_raw = m.group("prefix")
        # Trim trailing `)` and whitespace that the non-greedy quantifier
        # may have absorbed from a previous call.
        prefix_trimmed = _WRITE_TEXT_PREFIX_BAD_TRAIL.sub("", prefix_raw).rstrip()
        prefix = prefix_trimmed.strip()
        open_paren_idx = m.end() - 1  # position of the `(` in `.write_text(`
        close_idx = _find_balanced_paren(content, open_paren_idx)
        if close_idx == -1:
            # Unbalanced — bail out
            out_parts.append(content[m.start():])
            break
        args = content[open_paren_idx + 1 : close_idx - 1]
        # Skip if args contain `encoding=` (non-default encoding pass-through)
        if "encoding=" in args:
            out_parts.append(content[m.start() : close_idx])
            i = close_idx
            continue
        n_replaced += 1
        # The trim removed the trailing `)`; we need to keep it as part of
        # the output. We re-insert it after the prefix in the replacement.
        out_parts.append(f"_wf416_atomic_write_text({prefix}{_trailing_chars(prefix_raw, prefix_trimmed)}, {args.strip()})")
        i = close_idx
    return "".join(out_parts), n_replaced


def _trailing_chars(prefix_raw: str, prefix_trimmed: str) -> str:
    """Compute the trailing chars that were stripped from the raw prefix.
    Used to re-insert them after the replacement prefix.
    """
    if len(prefix_raw) > len(prefix_trimmed):
        return prefix_raw[len(prefix_trimmed):]
    return ""


def migrate_one(path: Path) -> Tuple[bool, str]:
    """Migrate a single script. Returns (changed, status_msg)."""
    content = path.read_text(encoding="utf-8")
    if already_migrated(content):
        return False, "already_migrated"

    before = content
    content, _ = insert_import_block(content)
    content, _ = insert_helper(content)
    content, n_replaced = replace_write_text_calls(content)

    if content == before:
        return False, "no_change"

    path.write_text(content, encoding="utf-8")
    return True, f"migrated n_write_sites={n_replaced}"


# 10 closeout-drafting scripts to migrate (selected by Wf-416-A1 audit; not yet
# covered by Wf-406-A3 / Wf-410-A3 / Wf-414-A3 prior waves)
SCRIPTS: List[str] = [
    "molmetal/scripts/m6_wf320h_vina_phase2.py",
    "molmetal/scripts/m15_rmsd_phase2.py",
    "molmetal/scripts/train_nci60_10cell.py",
    "molmetal/scripts/r17_gpu_retrain.py",
    "molmetal/scripts/generate_section4.py",
    "molmetal/scripts/extract_rdkit_2d_descriptors.py",
    "molmetal/scripts/wf146_chemprop_v2_lomo_v2.py",
    "molmetal/scripts/train_platinai_4cell.py",
    "molmetal/scripts/train_metal_lomo.py",
    "molmetal/scripts/train_metal_cytotox.py",
]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Wf-416-A1 atomic migration helper")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without modifying files")
    args = parser.parse_args(argv)

    results = []
    for rel_path in SCRIPTS:
        path = (REPO_ROOT / rel_path).resolve()
        if not path.exists():
            results.append((rel_path, "MISSING_on_disk"))
            continue
        if args.dry_run:
            content = path.read_text(encoding="utf-8")
            n = len(_WRITE_TEXT_PATTERN.findall(content))
            results.append((rel_path, f"would_migrate n_write_sites={n}"))
            continue
        changed, msg = migrate_one(path)
        results.append((rel_path, msg))

    print("=== Wf-416-A1 migration plan ===")
    for path, msg in results:
        flag = "OK " if msg in {"already_migrated", "no_change"} or msg.startswith("migrated") else "?? "
        print(f"  {flag}{path}: {msg}")
    n_changed = sum(1 for _, msg in results if msg.startswith("migrated"))
    n_skipped = sum(1 for _, msg in results if msg == "already_migrated")
    n_missing = sum(1 for _, msg in results if msg == "MISSING_on_disk")
    print(f"\nTotal: {n_changed} migrated, {n_skipped} already migrated, {n_missing} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
