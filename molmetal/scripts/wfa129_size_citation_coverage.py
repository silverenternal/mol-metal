#!/usr/bin/env python
"""(catalog §6.244) — measure how much of the byte-size citation corpus the
byte-size guard can actually SEE.

**The finding this script measures.**  The §6.243 guard reports its own
scope — "30 sites, 29 resolvable, 29 date-checked, 0 stale" — and that
number is a census taken from INSIDE the guard's own parser.  A census
taken from inside a parser cannot report what that parser cannot see.
This script takes the census from the VALUE side instead, with a parser
deliberately different from the guard's, and then asks the guard which of
the claims it found it recognises.

**Why the two parsers differ.**  The guard recognises a site only when the
path matches ``TODO/(?:completed|pending|archive)/<name>.md`` AND the size
token *immediately* follows the path.  Both are formatting conditions, so
a ``|`` table cell, a ``->`` chain separator or the word ``NEW`` between
path and size defeats the guard while leaving the claim perfectly legible
to a reader.  This script finds claims by SEMANTICS — "the size that
belongs to this path is the first size that appears after the path and
before the next path on the same line" — and then measures the overlap.

**Nothing is dropped silently.**  A size that belongs to a DIFFERENT file
named on the same line, and a number that is a ``total`` over several
files, are counted and reported separately rather than discarded.

**Roles.**  Every claim is classified by what the sentence asserts:

  ``LIVE``        the context carries ``measured YYYY-MM-DD`` — a claim
                  about the file as it stands now.  This is the set the
                  §6.243 guard polices.
  ``HISTORICAL``  the context carries some other date — a transcript of
                  ``ls -la`` at a past wave.  True when written; not a
                  claim about today.
  ``UNDATED``     no date at all — an unfalsifiable bare size.

Read-only against the repository.  Writes only its own report directory.
CPU-only; no GPU, no torch.

Usage::

    python molmetal/scripts/wfa129_size_citation_coverage.py
    python molmetal/scripts/wfa129_size_citation_coverage.py --json OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: The SSOT documents the §6.243 guard polices, and this census measures.
DOCUMENTS: tuple[str, ...] = (
    "docs/HONEST_NEGATIVE_CATALOG.md",
    "TODO/INDEX.md",
    "TODO/README.md",
    "docs/WF_HISTORY.md",
)

#: Any repo-relative path a claim may cite.
ANY_PATH = re.compile(
    r"(?:TODO|docs|molmetal|metrics|configs|scripts|triton_kernels|results)"
    r"/[A-Za-z0-9_./-]+\.(?:md|json|py|yaml|sh|npz|csv)"
)
#: A byte size.  The left boundary keeps "Wf-263B" and "Fix 1B" (wave and
#: fix ids) out; the >= 3-digit floor matches the guard's own ``SITE_RE``.
SIZE = re.compile(
    r"(?<![\w\-])"
    r"(?:\d{1,3}(?:,\d{3})+(?:\s\d{3})?|\d{1,3}(?:\s\d{3})+|\d+)"
    r"\s*B\b"
)
CHAIN_NEXT = re.compile(r"[\s`]*→[\s`]*(?:\d{1,3}(?:,\d{3})+|\d+)\s*B\b")
MEASURED = re.compile(r"measured\s+(\d{4}-\d{2}-\d{2})")
ANY_DATE = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:\s+\d{2}:\d{2})?)"
)
OTHER_FILE = re.compile(r"[A-Za-z0-9_.-]+\.(?:md|json|py|yaml|sh|npz|csv)\b")
TOTAL = re.compile(r"^\s*total\b", re.I)


def _load_guard():
    """Import the §6.243 guard so 'what the guard sees' is asked, not guessed."""
    spec = importlib.util.spec_from_file_location(
        "_szg", REPO / "molmetal" / "tests" / "test_wfa127_size_citation_guard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def census() -> tuple[list[dict], list[dict], dict]:
    """Return (claims, skipped, summary)."""
    guard = _load_guard()
    seen_by_guard = {(rel, ln, path) for rel, ln, path, _q, _c in guard._sites()}
    claims: list[dict] = []
    skipped: list[dict] = []

    for rel in DOCUMENTS:
        text = (REPO / rel).read_text(encoding="utf-8")
        # Text-scoped, like the guard's own parser: a citation may continue
        # on the NEXT line ("(12,898 B -> 13,085 B,\n  measured 2026-10-31)"),
        # so the role context must not be clipped at the line boundary.
        hits = list(ANY_PATH.finditer(text))
        for i, pm in enumerate(hits):
            path = pm.group(0)
            target = REPO / path
            if not target.exists():
                continue
            line_no = text.count("\n", 0, pm.start()) + 1
            line_end = text.find("\n", pm.end())
            line_end = len(text) if line_end < 0 else line_end
            next_start = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            # The SIZE must sit on this path's line; the ROLE may be read from
            # the next couple of lines (a split "old B -> new B," chain).
            size_region = text[pm.end() : min(line_end, next_start)]
            role_region = text[pm.end() : min(line_end + 240, next_start)]
            fm = SIZE.search(size_region)
            if fm is None:
                continue
            if TOTAL.match(size_region[fm.end() :]):
                skipped.append(
                    {
                        "document": rel,
                        "line": line_no,
                        "path": path,
                        "reason": "the number is a total over several files",
                    }
                )
                continue
            other = OTHER_FILE.search(size_region[: fm.start()])
            if other is not None:
                skipped.append(
                    {
                        "document": rel,
                        "line": line_no,
                        "path": path,
                        "reason": f"the size belongs to {other.group(0)}",
                    }
                )
                continue
            claimed = int(re.sub(r"[,\s]", "", fm.group(0)[:-1].strip()))
            if claimed < 100:
                skipped.append(
                    {
                        "document": rel,
                        "line": line_no,
                        "path": path,
                        "reason": f"{claimed} is not a byte size (a wave or fix id)",
                    }
                )
                continue
            end = fm.end()
            while True:
                nxt = CHAIN_NEXT.match(size_region, end)
                if nxt is None:
                    break
                claimed = int(re.sub(r"[^\d]", "", nxt.group(0)))
                end = nxt.end()
            if MEASURED.search(role_region):
                role = "LIVE"
            elif ANY_DATE.search(role_region):
                role = "HISTORICAL"
            else:
                role = "UNDATED"
            actual = target.stat().st_size
            claims.append(
                {
                    "document": rel,
                    "line": line_no,
                    "path": path,
                    "quoted": claimed,
                    "disk": actual,
                    "true": claimed == actual,
                    "role": role,
                    "in_guard_scope": bool(guard.PATH_RE.fullmatch(path)),
                    "seen_by_guard": (rel, line_no, path) in seen_by_guard,
                }
            )

    summary = {
        "documents": len(DOCUMENTS),
        "guard_sites": len(guard._sites()),
        "claims": len(claims),
        "skipped": len(skipped),
        "stale": sum(1 for c in claims if not c["true"]),
        "seen_by_guard": sum(1 for c in claims if c["seen_by_guard"]),
        "by_role": {
            role: {
                "total": sum(1 for c in claims if c["role"] == role),
                "stale": sum(1 for c in claims if c["role"] == role and not c["true"]),
                "seen_by_guard": sum(
                    1 for c in claims if c["role"] == role and c["seen_by_guard"]
                ),
            }
            for role in ("LIVE", "HISTORICAL", "UNDATED")
        },
    }
    return claims, skipped, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the full census to this JSON file")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="print every claim (and every skipped site) with its role",
    )
    args = parser.parse_args()

    claims, skipped, summary = census()

    print("BYTE-SIZE CITATION COVERAGE — value-directed census")
    print(
        f"documents: {summary['documents']}   guard's own site count: {summary['guard_sites']}"
    )
    print(
        f"claims attributed: {summary['claims']}   skipped (attribution): {summary['skipped']}"
    )
    print()
    for role in ("LIVE", "HISTORICAL", "UNDATED"):
        r = summary["by_role"][role]
        print(
            f"  {role:<11} {r['total']:>3} claims   stale {r['stale']:>3}   "
            f"seen by guard {r['seen_by_guard']:>3}"
        )
    print(
        f"  {'TOTAL':<11} {summary['claims']:>3} claims   stale {summary['stale']:>3}   "
        f"seen by guard {summary['seen_by_guard']:>3}"
    )

    if args.detail:
        print()
        for c in claims:
            print(
                f"  [{c['role']:<10}] {'TRUE ' if c['true'] else 'STALE'} "
                f"{'guard' if c['seen_by_guard'] else 'blind'}  "
                f"{c['document']}:{c['line']}  {c['path']}  "
                f"quoted {c['quoted']} B  disk {c['disk']} B"
            )
        for sk in skipped:
            print(
                f"  [SKIPPED   ]          {sk['document']}:{sk['line']}  "
                f"{sk['path']}  -- {sk['reason']}"
            )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_file(
            out,
            json.dumps(
                {"summary": summary, "claims": claims, "skipped": skipped},
                indent=1,
                ensure_ascii=False,
            ),
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
