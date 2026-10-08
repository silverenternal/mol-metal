"""Wf-A6 (BLOCKER D) — list STALE live paper-grade anchors for refresh tooling.

``p3_wf343c4_anchor_freshness_24h`` FAILs when a LIVE anchor matching
``ANCHOR_GLOB_PATTERNS`` is older than 24 h, but until now there was no
companion that enumerates *which* paths are stale so an operator can
regenerate them.  This walks the same globs with the same frozen-path
exemption as the gate (imported, so the two cannot drift) and prints the
stale paths, one per line, newest-agnostic.

Exit codes: 0 = no stale live anchors, 1 = >=1 stale (so callers can gate
on it), 2 = usage error.

CPU-only, < 5 ms.  Reads no torch, launches nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.pre_flight_check import (  # noqa: E402
    ANCHOR_FROZEN_PATHS,
    ANCHOR_GLOB_PATTERNS,
    REPO_ROOT,
)


def stale_live_anchors(
    max_age_sec: float = 24 * 3600.0,
    glob_patterns: Tuple[str, ...] = ANCHOR_GLOB_PATTERNS,
    frozen_paths: Tuple[Tuple[str, str], ...] = ANCHOR_FROZEN_PATHS,
    include_frozen: bool = False,
) -> List[Tuple[str, float]]:
    """Return ``[(rel_path, age_hours)]`` for stale anchors, oldest first.

    LIVE anchors (not in ``frozen_paths``) are checked against
    ``max_age_sec``.  With ``include_frozen`` the frozen historical
    reports are listed too (still subject to the same age budget) so an
    operator can see the full anchor catalogue.
    """
    now = time.time()
    frozen = set()
    for rel, _reason in frozen_paths:
        p = Path(rel)
        frozen.add(str((p if p.is_absolute() else REPO_ROOT / p).resolve()))

    out: List[Tuple[str, float]] = []
    for pattern in glob_patterns:
        if Path(pattern).is_absolute():
            matches = sorted(Path("/").glob(pattern.lstrip("/")))
        else:
            matches = sorted(REPO_ROOT.glob(pattern))
        for path in matches:
            if not path.is_file():
                continue
            if str(path.resolve()) in frozen and not include_frozen:
                continue
            age = now - path.stat().st_mtime
            if age > max_age_sec:
                try:
                    shown = str(path.relative_to(REPO_ROOT))
                except ValueError:
                    shown = str(path)
                out.append((shown, age / 3600.0))
    out.sort(key=lambda t: -t[1])
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--max-age-hours", type=float, default=24.0,
        help="Freshness budget in hours (default 24, matches the gate).",
    )
    p.add_argument(
        "--include-frozen", action="store_true",
        help="Also list frozen historical anchors (exempt in the gate).",
    )
    args = p.parse_args(argv)

    stale = stale_live_anchors(
        max_age_sec=args.max_age_hours * 3600.0,
        include_frozen=args.include_frozen,
    )
    if not stale:
        print("OK: 0 stale anchors")
        return 0
    print(f"STALE: {len(stale)} anchor(s) older than {args.max_age_hours:.0f}h")
    for rel, hours in stale:
        print(f"{hours:6.1f}h  {rel}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
