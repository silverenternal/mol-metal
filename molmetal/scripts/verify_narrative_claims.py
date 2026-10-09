#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify every claim in ``registry/narrative_claims.yaml`` against disk.

WHY
---
AGENTS.md §5.2 says every number in agent output must trace to a JSON or report
path on disk with a MEASURED / PROJECTED tag. That was being enforced by hand,
and hand enforcement failed: a manual audit on 2026-10-10 found that
``ALIGNED_COMPARISON.md`` carried stale per-cell NCI-60 values (it labels PC3's
0.870 as A549) which had been copied verbatim into the pinned narrative, and
that ``n_distinct``'s headline "20" is a saturated ``n_top_k`` cap rather than a
measurement. Both were invisible to a reader who trusted the prose.

This checker closes that loop. Each claim names a source file and a dotted field
path; the value on disk must equal the value the narrative states, within a
tolerance. It is wired into ``pre_flight_check.py`` so drift is a red gate.

WHAT IT DOES NOT DO
-------------------
It does not check prose, and it does not judge whether a claim is *worthy* of
being in the paper. It checks one thing only: does the number still exist on
disk with the value we are quoting. A claim that passes here can still be the
wrong claim to make — that is a judgement, not a provenance problem.

USAGE
-----
::

    .venv/bin/python molmetal/scripts/verify_narrative_claims.py            # human table
    .venv/bin/python molmetal/scripts/verify_narrative_claims.py --strict   # exit 1 on drift
    .venv/bin/python molmetal/scripts/verify_narrative_claims.py --json

Exit codes: 0 = PASS, 1 = DRIFT (a stated value no longer matches disk),
2 = CONFIG (the claims file itself is malformed or unreadable).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLAIMS_PATH = PROJECT_ROOT / "registry" / "narrative_claims.yaml"
DEFAULT_TOL = 5e-4


class ConfigError(Exception):
    """The claims file is malformed — distinct from a value mismatch."""


def _load_claims(path: Path) -> List[Dict[str, Any]]:
    import yaml

    if not path.is_file():
        raise ConfigError(f"claims file missing: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("claims"), list):
        raise ConfigError(f"{path} has no top-level 'claims' list")
    out = []
    seen = set()
    for i, c in enumerate(doc["claims"]):
        if not isinstance(c, dict):
            raise ConfigError(f"claim #{i} is not a mapping")
        for req in ("id", "claim", "value", "source", "field"):
            if req not in c:
                raise ConfigError(f"claim #{i} missing required key {req!r}")
        if c["id"] in seen:
            raise ConfigError(f"duplicate claim id {c['id']!r}")
        seen.add(c["id"])
        out.append(c)
    return out


def _dig(obj: Any, dotted: str) -> Tuple[bool, Any]:
    """Resolve ``a.b.0.c`` against nested dicts/lists. Returns (found, value)."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return False, None
            if not (-len(cur) <= idx < len(cur)):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


def _display_path(path: Path) -> str:
    """Repo-relative when possible; absolute otherwise.

    Tests hand `verify()` a copy under tmp_path, which is outside the repo, so
    a bare ``relative_to(PROJECT_ROOT)`` would raise and mask the drift the
    test is trying to observe.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def verify(path: Path = CLAIMS_PATH) -> Dict[str, Any]:
    claims = _load_claims(path)
    rows: List[Dict[str, Any]] = []
    counts = {"pass": 0, "drift": 0, "unverifiable": 0}

    for c in claims:
        src = PROJECT_ROOT / c["source"]
        row = {
            "id": c["id"],
            "claim": c["claim"],
            "stated": c["value"],
            "source": c["source"],
            "field": c["field"],
        }
        if not src.is_file():
            row.update(status="UNVERIFIABLE", reason=f"source file missing: {c['source']}")
            counts["unverifiable"] += 1
            rows.append(row)
            continue
        try:
            payload = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            row.update(status="UNVERIFIABLE", reason=f"source unreadable: {exc}")
            counts["unverifiable"] += 1
            rows.append(row)
            continue

        found, disk = _dig(payload, c["field"])
        if not found:
            row.update(status="UNVERIFIABLE", reason=f"no such field: {c['field']}")
            counts["unverifiable"] += 1
            rows.append(row)
            continue

        stated = c["value"]
        tol = float(c.get("tol", DEFAULT_TOL))
        if isinstance(stated, (int, float)) and isinstance(disk, (int, float)):
            ok = math.isclose(float(stated), float(disk), rel_tol=0.0, abs_tol=tol)
            row["disk"] = disk
            row["delta"] = float(disk) - float(stated)
            row["status"] = "PASS" if ok else "DRIFT"
        else:
            ok = str(stated) == str(disk)
            row["disk"] = disk
            row["status"] = "PASS" if ok else "DRIFT"
        counts["pass" if ok else "drift"] += 1
        rows.append(row)

    return {
        "metric": "narrative_claims_verification",
        "description": (
            "Every claim in registry/narrative_claims.yaml re-resolved against "
            "the JSON/report it names. Existence + value only; this checker does "
            "not judge whether a claim belongs in the paper."
        ),
        "claims_file": _display_path(path),
        "narrative_doc": "docs/00_MODEL_NARRATIVE.md",
        "n_claims": len(claims),
        "counts": counts,
        "verdict": "PASS" if counts["drift"] == 0 and counts["unverifiable"] == 0
                   else ("DRIFT" if counts["drift"] else "UNVERIFIABLE"),
        "rows": rows,
    }


def _print(rep: Dict[str, Any]) -> None:
    print(f"[narrative-claims] {rep['n_claims']} claims from {rep['claims_file']}")
    for r in rep["rows"]:
        mark = {"PASS": "  ok  ", "DRIFT": " DRIFT", "UNVERIFIABLE": "  ??  "}[r["status"]]
        line = f"[{mark}] {r['id']:<24} stated={r['stated']}"
        if "disk" in r:
            line += f" disk={r['disk']}"
        if r["status"] == "UNVERIFIABLE":
            line += f"  ({r['reason']})"
        elif r["status"] == "DRIFT":
            line += f"  delta={r['delta']:+.6f}"
        print(f"  {line}")
    c = rep["counts"]
    print(
        f"[narrative-claims] pass={c['pass']} drift={c['drift']} "
        f"unverifiable={c['unverifiable']} -> {rep['verdict']}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--claims", type=Path, default=CLAIMS_PATH)
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on drift; unverifiable alone still exits 0")
    args = ap.parse_args(argv)

    try:
        rep = verify(args.claims)
    except ConfigError as exc:
        print(f"[narrative-claims] CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        _print(rep)

    if rep["counts"]["drift"]:
        return 1
    if rep["verdict"] != "PASS":
        return 0 if not args.strict else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
