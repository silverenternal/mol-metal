# SPDX-License-Identifier: Apache-2.0
"""Refresh NAMED rows of ``registry/metrics.yaml`` without touching the rest.

WHY THIS SCRIPT EXISTS
-----------------------
``molmetal/scripts/registry.py generate`` rebuilds EVERY row from the on-disk
artifacts.  That is the right thing to do after a corpus-wide change, and the
wrong thing to do after a change that touched three artifacts: it moves
``last_refreshed`` on ~490 rows, so the diff becomes unreadable and a reviewer
cannot tell which rows actually changed.  It also silently rewrites rows that
carry a hand-set annotation.

This script does the surgical thing: it re-derives ONLY the named rows with
the registry's OWN builder (``registry.build_metrics`` -- so the row is
exactly the row ``generate`` would have written) and splices them into the
YAML as TEXT, leaving every other byte of the file alone.

A row is replaced in place when it already exists, and APPENDED under
``metrics:`` when it does not.  The result is still valid YAML and still
passes ``registry.py check``.

Usage:
    .venv/bin/python molmetal/scripts/registry_refresh_rows.py cfm_run_census
    .venv/bin/python molmetal/scripts/registry_refresh_rows.py --all-dirty
    .venv/bin/python molmetal/scripts/registry_refresh_rows.py --dry-run <id>

CPU-only.  Reads the registry and the artifacts; writes nothing except the
registry itself (and only with ``atomic_write_file``, AGENTS.md 5.6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from molmetal.scripts.registry import (  # noqa: E402
    METRICS_PATH,
    build_metrics,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: The header the registry file starts with (the schema block).
_HEADER_LINES = 4


def _render_row(metric_id: str, row: Dict[str, Any]) -> List[str]:
    """Render one registry row as YAML lines, indented to sit under ``metrics:``."""
    text = yaml.safe_dump(
        {metric_id: row}, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    lines = text.rstrip("\n").split("\n")
    return [("  " + line) if line else line for line in lines]


def _row_span(lines: Sequence[str], metric_id: str) -> Optional[tuple]:
    """The (start, end) line span of ``metric_id``'s block, or None.

    A metric row is a mapping key at indent 2 followed by its keys at indent
    4.  The block ends at the next line that is not part of it.
    """
    head = f"  {metric_id}:"
    start = None
    for i, line in enumerate(lines):
        if line == head or line.startswith(head + " "):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= 2:
            end = j
            break
    return start, end


def _metrics_block_start(lines: Sequence[str]) -> int:
    for i, line in enumerate(lines):
        if line.rstrip() == "metrics:":
            return i
    raise SystemExit("registry/metrics.yaml has no top-level `metrics:` block")


def _metrics_block_end(lines: Sequence[str], start: int) -> int:
    """The first line AFTER ``metrics:`` that is not part of it.

    The registry file carries top-level keys AFTER the ``metrics:`` mapping
    (e.g. ``uncovered_metric_files:``).  Appending a row at the end of the
    FILE would therefore land it outside the mapping, at a place where the
    YAML parser reads it as a sibling of a scalar key -- which is a parse
    error, not a formatting nit.
    """
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            return j
    return len(lines)


def refresh(metric_ids: Sequence[str], *, dry_run: bool = False) -> Dict[str, Any]:
    """Re-derive and splice the named rows.  Returns a small report."""
    path = Path(METRICS_PATH)
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    built = build_metrics(REPO_ROOT)[0]

    applied: List[str] = []
    skipped: List[str] = []
    for metric_id in metric_ids:
        row = built.get(metric_id)
        if row is None:
            skipped.append(f"{metric_id}: not produced by build_metrics")
            continue
        span = _row_span(lines, metric_id)
        new_lines = _render_row(metric_id, row)
        if span is None:
            # Append: the LAST row of the `metrics:` mapping -- NOT the end
            # of the file, which is a different top-level key.
            start = _metrics_block_start(lines)
            end = _metrics_block_end(lines, start)
            while end > start and not lines[end - 1].strip():
                end -= 1
            lines = lines[:end] + new_lines + lines[end:]
            applied.append(f"{metric_id}: APPENDED")
        else:
            s, e = span
            lines = lines[:s] + new_lines + lines[e:]
            applied.append(f"{metric_id}: replaced lines {s + 1}..{e}")

    out_text = "\n".join(lines)
    # Round-trip check: the spliced file must still be loadable YAML.
    parsed = yaml.safe_load(out_text)
    if not isinstance(parsed, dict) or "metrics" not in parsed:
        raise SystemExit("spliced registry/metrics.yaml is not a valid registry doc")

    if not dry_run:
        atomic_write_file(str(path), out_text)

    return {
        "applied": applied,
        "skipped": skipped,
        "path": str(path),
        "dry_run": dry_run,
        "n_metrics": len(parsed.get("metrics", {})),
    }


def _dirty_ids() -> List[str]:
    """The rows whose source artifact changed since ``last_refreshed``."""
    path = Path(METRICS_PATH)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metrics = doc.get("metrics") or {}
    built = build_metrics(REPO_ROOT)[0]
    dirty: List[str] = []
    for metric_id, row in metrics.items():
        new = built.get(metric_id)
        if new is None:
            continue
        if new.get("source_json_sha256") != row.get("source_json_sha256"):
            dirty.append(metric_id)
    return dirty


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="registry_refresh_rows.py",
        description=(
            "Re-derive and splice NAMED registry rows, leaving every other row "
            "byte-identical.  CPU-only."
        ),
    )
    parser.add_argument("metric_ids", nargs="*", help="metric ids to refresh")
    parser.add_argument(
        "--all-dirty",
        action="store_true",
        help="refresh every row whose source artifact changed on disk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change; write nothing",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.all_dirty:
        ids = _dirty_ids()
        if not ids:
            print("registry: no dirty rows")
            return 0
    else:
        ids = list(args.metric_ids)
        if not ids:
            print("nothing to do: pass metric ids or --all-dirty", file=sys.stderr)
            return 2
    report = refresh(ids, dry_run=args.dry_run)
    for line in report["applied"]:
        print(f"  {line}")
    for line in report["skipped"]:
        print(f"  {line}", file=sys.stderr)
    verb = "would refresh" if args.dry_run else "refreshed"
    print(f"registry: {verb} {len(ids)} row(s) -> {report['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
