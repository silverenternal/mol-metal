#!/usr/bin/env python3
"""FIX-P2 — pre-flight gate: feature granularity must match cohort granularity.

Scans ``metrics/by_metric/*.json`` for artifacts that report a
``canonical_grouped`` split and checks the headline arm was featurized on the
complex, not on the main fragment.  See
:mod:`molmetal.features.granularity` for the rule and its one-directional
design.

Exit codes follow the repo convention: 0 PASS, 1 FAIL, 2 SKIP (nothing in
scope).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.features.granularity import check_artifact, requires_check


def check_directory(metrics_dir: Path) -> tuple[int, List[str], List[str]]:
    checked = 0
    passed: List[str] = []
    failed: List[str] = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not requires_check(payload):
            continue
        checked += 1
        problems = check_artifact(payload, path=path.name)
        if problems:
            failed.extend(problems)
        else:
            passed.append(path.name)
    return checked, passed, failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    checked, passed, failed = check_directory(args.metrics_dir)
    if args.json:
        print(json.dumps({"checked": checked, "passed": passed, "failed": failed}, indent=2))
    else:
        for name in passed:
            print(f"PASS  {name}")
        for problem in failed:
            print(f"FAIL  {problem}")

    if failed:
        return 1
    if checked == 0:
        print("SKIP  no complex-grouped artifacts found")
        return 2
    print(f"PASS  {checked} artifact(s) at consistent granularity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
