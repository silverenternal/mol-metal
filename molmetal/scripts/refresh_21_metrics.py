# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Wf-A25 step 2 — 21-metric report refresh.

Walks :file:`metrics/by_metric/` (244 files as of 2026-10-13) and emits
a JSON report with each metric's:

  * ``mtime``            — POSIX mtime (seconds since epoch).
  * ``value_range``      — min / max over any numeric scalar leaves,
                            including nested ``metrics.<name>.value`` /
                            ``headline.<name>`` / numeric array elements.
  * ``tag``              — ``MEASURED`` / ``PROJECTED`` /
                            ``MEASURED_HONEST_NEGATIVE`` / ``None``.
  * ``tag_explicit``     — ``True`` iff the file declares a top-level
                            ``tag`` key (vs. inferred from provenance).
  * ``file_size``        — bytes on disk.
  * ``path``             — repo-relative path.

Output schema::

    {
      "report_version": "wf-a25.step2.v1",
      "generated_at":   "<ISO 8601>",
      "metrics_dir":    "metrics/by_metric/",
      "n_files":        <int>,
      "tag_counts":     {"MEASURED": <int>, "PROJECTED": <int>, ...},
      "files": [
        {
          "path": "metrics/by_metric/atom_vocab_coverage.json",
          "file_size": <int>,
          "mtime": <float>,
          "mtime_iso": "<ISO 8601>",
          "tag": "MEASURED" | "PROJECTED" | "MEASURED_HONEST_NEGATIVE" | "None",
          "tag_explicit": true | false,
          "value_range": {"min": <float>, "max": <float>, "n_numeric": <int>}
                          | null,
        },
        ...
      ]
    }

Honest scope (per AGENTS.md §5.2 + party-twenty-spirit §3.5):
  * The report NEVER fabricates metrics — every row reads from disk.
  * When ``tag`` is missing the row reports ``"None"`` (so the absence is
    visible at audit time, not hidden behind a defaulted label).
  * The script is CPU-only — no torch, no GPU.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"

# Tag vocabulary we recognise. Anything outside this set is reported as
# the literal string (so e.g. "MEASURED-AND-DECIDED" is preserved rather
# than collapsed to MEASURED) per the "disabled-feature-is-reported, not
# swallowed" mandate.
KNOWN_TAGS = {"MEASURED", "PROJECTED", "MEASURED_HONEST_NEGATIVE"}


def _coerce_float(v: Any) -> Optional[float]:
    """Best-effort float coercion. Returns None when the leaf is not numeric."""
    if isinstance(v, bool):
        # bool is a subclass of int; exclude it explicitly.
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v):  # NaN
            return v
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _walk_numeric_leaves(node: Any, out: List[float], _depth: int = 0) -> None:
    """Recursively gather numeric leaves, with a depth cap for safety."""
    if _depth > 32:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            # Skip the tag key itself; do not let the tag label corrupt the range.
            if k == "tag":
                continue
            _walk_numeric_leaves(v, out, _depth + 1)
    elif isinstance(node, list):
        for v in node:
            _walk_numeric_leaves(v, out, _depth + 1)
    else:
        f = _coerce_float(node)
        if f is not None:
            out.append(f)


def _value_range(payload: Any) -> Optional[Dict[str, Any]]:
    leaves: List[float] = []
    _walk_numeric_leaves(payload, leaves)
    if not leaves:
        return None
    finite = [x for x in leaves if x == x]  # drop NaN
    if not finite:
        return {"min": None, "max": None, "n_numeric": len(leaves)}
    return {
        "min": min(finite),
        "max": max(finite),
        "n_numeric": len(leaves),
    }


def _classify_tag(payload: Any) -> tuple[Optional[str], bool]:
    """Return ``(tag, explicit)`` from the JSON payload.

    The explicit flag is ``True`` only when a top-level ``"tag"`` key
    exists. Otherwise we attempt a provenance inference (the explicit
    tag dominates; inferred tags are recorded as the same string but
    with ``tag_explicit=False``).
    """
    explicit: Optional[str] = None
    if isinstance(payload, dict) and "tag" in payload:
        v = payload["tag"]
        if isinstance(v, str):
            explicit = v.strip()
    inferred: Optional[str] = None
    if explicit is None:
        # Inference from provenance (Wf-389 honesty rule: if a JSON contains
        # a `tag:` field anywhere in nested provenance, prefer that).
        if isinstance(payload, dict):
            for k, v in payload.items():
                if k in ("RETRACTION", "headline", "metric") and isinstance(v, dict):
                    inner = v.get("tag")
                    if isinstance(inner, str):
                        inferred = inner.strip()
                        break
    chosen = explicit if explicit is not None else inferred
    return chosen, explicit is not None


def refresh(metrics_dir: Path) -> Dict[str, Any]:
    """Walk ``metrics_dir`` and return the report dict (NOT yet serialised)."""
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"metrics dir not found: {metrics_dir}")
    files: List[Path] = sorted(
        p for p in metrics_dir.iterdir() if p.is_file() and p.suffix == ".json"
    )

    rows: List[Dict[str, Any]] = []
    tag_counts: Dict[str, int] = {}
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # Honest-negative: a broken file is reported, not skipped silently.
            rows.append({
                "path": str(p.relative_to(REPO_ROOT)),
                "file_size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "mtime_iso": _dt.datetime.fromtimestamp(
                    p.stat().st_mtime, tz=_dt.timezone.utc
                ).isoformat(),
                "tag": None,
                "tag_explicit": False,
                "value_range": None,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        tag, tag_explicit = _classify_tag(payload)
        if tag is not None:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        vr = _value_range(payload)
        stat = p.stat()
        rows.append({
            "path": str(p.relative_to(REPO_ROOT)),
            "file_size": stat.st_size,
            "mtime": stat.st_mtime,
            "mtime_iso": _dt.datetime.fromtimestamp(
                stat.st_mtime, tz=_dt.timezone.utc
            ).isoformat(),
            "tag": tag if tag is not None else "None",
            "tag_explicit": tag_explicit,
            "value_range": vr,
        })

    return {
        "report_version": "wf-a25.step2.v1",
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "metrics_dir": str(metrics_dir.relative_to(REPO_ROOT)),
        "n_files": len(rows),
        "tag_counts": dict(sorted(tag_counts.items())),
        "files": rows,
    }


def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-A25 step 2 — walk metrics/by_metric/ and emit a JSON "
            "report (mtime + value range + MEASURED/PROJECTED tag)."
        )
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(DEFAULT_METRICS_DIR),
        help=f"Path to metrics dir (default: {DEFAULT_METRICS_DIR}).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: stdout).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-file summary table on stderr.",
    )
    args = parser.parse_args(argv)

    md = Path(args.metrics_dir)
    report = refresh(md)

    if not args.quiet:
        sys.stderr.write(
            f"[refresh_21_metrics] n_files={report['n_files']} "
            f"tag_counts={report['tag_counts']}\n"
        )

    serialised = json.dumps(report, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).write_text(serialised, encoding="utf-8")
    else:
        sys.stdout.write(serialised + "\n")
    return 0


__all__ = ["refresh", "_classify_tag", "_value_range", "_walk_numeric_leaves"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
