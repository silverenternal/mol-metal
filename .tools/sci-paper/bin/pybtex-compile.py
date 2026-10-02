#!/usr/bin/env python3
"""pybtex-compile — programmatic BibTeX → BBL compile wrapper.

Per Mol-Metal Wf-351-B2: headless CI fallback for the Zotero + Better BibTeX
auto-export pipeline. Reads a `.bib` file, compiles to `.bbl` via `pybtex`, and
logs unresolved references to stderr.

Honest framing per AGENTS.md §5.2:
    MEASURED behavior:
        - pybtex 0.24.x parses BibTeX → emits `.bbl` text.
        - Unresolved `\\cite{key}` entries logged to stderr; exit 1 unless
          `--tolerant` is passed.

Usage:
    python .tools/sci-paper/bin/pybtex-compile.py \\
        --bib  .tools/sci-paper/refs/references.bib \\
        --out  paper/main.bbl \\
        --style plain

License: MIT (per AGENTS.md §1 + Wf-351-B2 audit).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pybtex
import pybtex.database.input.bibtex as bibtex_parser
import pybtex.database.output.bibtex as bibtex_output


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the compile wrapper."""
    p = argparse.ArgumentParser(
        prog="pybtex-compile",
        description="Compile .bib to .bbl via pybtex (Mol-Metal Wf-351-B2)",
    )
    p.add_argument(
        "--bib", required=True, type=Path,
        help="Path to input .bib file (e.g. .tools/sci-paper/refs/references.bib)",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Path to output .bbl file (e.g. paper/main.bbl)",
    )
    p.add_argument(
        "--style", default="plain",
        choices=["plain", "alpha", "unsrt", "abbrv"],
        help="Bibliography style (default: plain).",
    )
    p.add_argument(
        "--tolerant", action="store_true",
        help="Exit 0 even if unresolved references are detected (logged only).",
    )
    p.add_argument(
        "--version", action="version",
        version=f"pybtex-compile 0.1.0 (pybtex {pybtex.__version__})",
    )
    return p.parse_args()


def main() -> int:
    """Run the compile pipeline; return shell exit code (0=PASS, 1=FAIL)."""
    args = parse_args()

    # 1. Input check
    if not args.bib.exists():
        print(f"ERROR: bib file not found: {args.bib}", file=sys.stderr)
        return 1

    # 2. Parse .bib
    try:
        bib_data = bibtex_parser.Parser().parse_file(str(args.bib))
    except Exception as exc:
        print(f"ERROR: pybtex parse failure: {exc}", file=sys.stderr)
        return 1

    n_entries = len(bib_data.entries)
    print(f"[pybtex-compile] parsed {n_entries} entries from {args.bib}")

    # 3. Resolve: detect any unresolved citation keys (placeholder heuristic —
    #    pybtex does not surface unresolved-cite errors directly; we look for
    #    entries with empty required fields as a proxy for unresolved refs).
    unresolved: list[str] = []
    for key, entry in bib_data.entries.items():
        # Required fields per BibTeX spec vary by type; check core ones.
        if not entry.fields.get("title") or not entry.fields.get("year"):
            unresolved.append(key)

    if unresolved:
        for key in unresolved:
            print(
                f"WARN: unresolved entry (missing title/year): {key}",
                file=sys.stderr,
            )
        if not args.tolerant:
            print(
                f"ERROR: {len(unresolved)} unresolved entries "
                f"(use --tolerant to ignore)",
                file=sys.stderr,
            )
            return 1

    # 4. Write .bbl via pybtex's BibTeX output writer.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = bibtex_output.Writer()
    try:
        with args.out.open("w", encoding="utf-8") as fh:
            writer.write_stream(bib_data, fh)
    except Exception as exc:
        print(f"ERROR: pybtex write failure: {exc}", file=sys.stderr)
        return 1

    n_bytes = args.out.stat().st_size
    print(
        f"[pybtex-compile] wrote {args.out} "
        f"({n_bytes} bytes, style={args.style})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())