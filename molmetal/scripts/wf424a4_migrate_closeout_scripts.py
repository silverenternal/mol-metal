#!/usr/bin/env python3
"""Wf-424-A4 migration helper — atomic write protocol migration of 10 closeout-drafting scripts.

This is a one-shot migration utility (CPU-only, idempotent). It applies the
Wf-386-A3 atomic write protocol pattern to the 10 closeout-drafting scripts
selected by Wf-424-A4 audit (continuation of Wf-416-A1 / Wf-410-A3 / Wf-414-A3).
Each migrated script:

  1. Imports the shared atomic_write_file + safe_overwrite_check + detect_truncation
     from molmetal.scripts.wf386a3_atomic_write via try/except ImportError fallback.
  2. Replaces direct ``path.write_text(content)`` calls with the migration pattern
     that delegates to atomic_write_file when available.
  3. Calls safe_overwrite_check before any overwrite to surface substantial content
     overwrites before they happen.

The migration is idempotent — re-running the helper on an already-migrated script
is a no-op (the import block check skips reinsertion; the Write sites have already
been replaced).

Cumulative migration count after Wf-424-A4:
  Wf-406-A3 (3) + Wf-410-A3 (5) + Wf-414-A3 (5) + Wf-416-A1 (10) + Wf-424-A4 (10) = 33 scripts.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Import block template (idempotent: _WF424_HAS_ATOMIC guard prevents double-write).
# Coexists with Wf-416-A1's _WF416_HAS_ATOMIC marker — both waves' markers are
# additive, since this script targets DIFFERENT files than Wf-416-A1.
_IMPORT_BLOCK = '''try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF424_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF424_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]
'''

# Migration helper template for replacing path.write_text(content)
_MIGRATE_HELPER = '''def _wf424_atomic_write_text(path, content: str, min_lines: int = 50, expected_min_length: int = 200) -> None:
    """Wf-424-A4 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF424_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk, path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging
                logging.getLogger(__name__).warning(
                    "[Wf-424-A4] detect_truncation reason=%r at %s",
                    _trunc, path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")

'''


def already_migrated(content: str) -> bool:
    """Return True if the file has already been migrated by any prior wave
    (Wf-406 / Wf-410 / Wf-414 / Wf-416 / Wf-424)."""
    return (
        "_WF424_HAS_ATOMIC" in content
        or "_WF416_HAS_ATOMIC" in content
        or "_WF414_HAS_ATOMIC" in content
        or "_WF410_HAS_ATOMIC" in content
        or "_WF406_HAS_ATOMIC" in content
    )


def insert_import_block(content: str) -> Tuple[str, bool]:
    """Insert the import block after the last TOP-LEVEL import statement.
    Only considers single-line imports (`import X` or `from X import Y`).
    Returns (new_content, changed). Skipped if already migrated.
    """
    if already_migrated(content):
        return content, False

    lines = content.splitlines(keepends=True)
    insert_idx = 0
    open_paren_count = 0
    for i, line in enumerate(lines):
        if open_paren_count > 0:
            open_paren_count += line.count("(") - line.count(")")
            continue
        if line.startswith("import ") or line.startswith("from "):
            line_open = line.count("(") - line.count(")")
            if line_open > 0:
                open_paren_count = line_open
                insert_idx = i
                continue
            insert_idx = i + 1
    block = "\n" + _IMPORT_BLOCK + "\n"
    new_lines = lines[:insert_idx] + [block] + lines[insert_idx:]
    return "".join(new_lines), True


def insert_helper(content: str) -> Tuple[str, bool]:
    """Insert the _wf424_atomic_write_text helper before the first def/class/main statement.
    Returns (new_content, changed). Skipped if already present.
    """
    if "_wf424_atomic_write_text" in content:
        return content, False

    lines = content.splitlines(keepends=True)
    insert_idx = None
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__"):
            insert_idx = i
            break
    if insert_idx is None:
        insert_idx = len(lines)
    new_lines = lines[:insert_idx] + [_MIGRATE_HELPER] + lines[insert_idx:]
    return "".join(new_lines), True


# Pattern to match: <var>.write_text(<arg>...)
_WRITE_TEXT_PATTERN = re.compile(
    r"(?P<prefix>[a-zA-Z_][a-zA-Z_0-9. ()\[\]\"\'/_]*?)\.write_text\(",
    re.MULTILINE,
)
_WRITE_TEXT_PREFIX_BAD_TRAIL = re.compile(r"\)\s*$")


def _find_balanced_paren(text: str, open_idx: int) -> int:
    """Find the matching closing paren for the opening paren at open_idx."""
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


def _trailing_chars(prefix_raw: str, prefix_trimmed: str) -> str:
    if len(prefix_raw) > len(prefix_trimmed):
        return prefix_raw[len(prefix_trimmed):]
    return ""


def replace_write_text_calls(content: str) -> Tuple[str, int]:
    """Replace all `<var>.write_text(<args>)` calls with
    `_wf424_atomic_write_text(<var>, <args>)`.

    NOTE: only replaces calls where the args do NOT include `encoding=` — those
    calls use a non-default encoding (e.g. utf-8) and we keep them as direct
    writes for backward compat (atomic_write_file defaults to utf-8 but we want
    explicit pass-through).
    """
    if "_WF424_HAS_ATOMIC" not in content:
        return content, 0  # not migrated by THIS wave yet

    n_replaced = 0
    out_parts: List[str] = []
    i = 0
    while i < len(content):
        m = _WRITE_TEXT_PATTERN.search(content, i)
        if m is None:
            out_parts.append(content[i:])
            break
        out_parts.append(content[i:m.start()])
        prefix_raw = m.group("prefix")
        prefix_trimmed = _WRITE_TEXT_PREFIX_BAD_TRAIL.sub("", prefix_raw).rstrip()
        prefix = prefix_trimmed.strip()
        open_paren_idx = m.end() - 1
        close_idx = _find_balanced_paren(content, open_paren_idx)
        if close_idx == -1:
            out_parts.append(content[m.start():])
            break
        args = content[open_paren_idx + 1 : close_idx - 1]
        if "encoding=" in args:
            out_parts.append(content[m.start() : close_idx])
            i = close_idx
            continue
        n_replaced += 1
        out_parts.append(
            f"_wf424_atomic_write_text({prefix}{_trailing_chars(prefix_raw, prefix_trimmed)}, {args.strip()})"
        )
        i = close_idx
    return "".join(out_parts), n_replaced


def migrate_one(path: Path) -> Tuple[bool, str]:
    """Migrate a single script. Returns (changed, status_msg)."""
    content = path.read_text(encoding="utf-8")
    if already_migrated(content):
        return False, "already_migrated_by_prior_wave"

    before = content
    content, _ = insert_import_block(content)
    content, _ = insert_helper(content)
    content, n_replaced = replace_write_text_calls(content)

    if content == before:
        return False, "no_change"
    # Use atomic_write_file for the migrated script itself to honor §5.6
    if _WF424_HAS_ATOMIC_PRESENT():
        atomic_write_file(str(path), content, encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return True, f"migrated n_write_sites={n_replaced}"


def _WF424_HAS_ATOMIC_PRESENT() -> bool:
    """Check if the canonical atomic primitive is importable in this process."""
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file as _a
        return _a is not None
    except ImportError:
        return False


# 10 closeout-drafting scripts to migrate (selected by Wf-424-A4 audit).
# Each is a top direct-Write site NOT covered by Wf-406-A3 / Wf-410-A3 /
# Wf-414-A3 / Wf-416-A1 prior waves. Cumulative count after Wf-424-A4 = 33.
SCRIPTS: List[str] = [
    "molmetal/scripts/per_wave_checklist.py",
    "molmetal/scripts/wf264_panel_cfm_properties.py",
    "molmetal/scripts/wf179_rescue.py",
    "molmetal/scripts/wf179_fix_rmsd.py",
    "molmetal/scripts/r4_c_full_sweep.py",
    "molmetal/scripts/compute_m15_ligand_vs_pocket.py",
    "molmetal/scripts/bench_wf179_m15_rmsd_p2.py",
    "molmetal/scripts/wf240_ablation_launcher.py",
    "molmetal/scripts/train_hilbert_astsr_end_to_end.py",
    "molmetal/scripts/extract_multiph_pll_end_to_end.py",
]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Wf-424-A4 atomic migration helper")
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

    print("=== Wf-424-A4 migration plan ===")
    for path, msg in results:
        flag = "OK " if msg in {"already_migrated_by_prior_wave", "no_change"} or msg.startswith("migrated") else "?? "
        print(f"  {flag}{path}: {msg}")
    n_changed = sum(1 for _, msg in results if msg.startswith("migrated"))
    n_skipped = sum(1 for _, msg in results if msg == "already_migrated_by_prior_wave")
    n_missing = sum(1 for _, msg in results if msg == "MISSING_on_disk")
    print(f"\nTotal: {n_changed} migrated, {n_skipped} already migrated, {n_missing} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
