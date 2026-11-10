#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Does the PlatinAI 4-cell CSV really carry the 4 cell-line label columns?

WHY (Wf-A269 wave-9, 2026-11-02).  The launcher's 4-cell "verification" was a
no-op gate that could not fail:

  * it grepped for a QUOTED UPPERCASE cell name (``"MCF7"``) in the CSV, but
    the CSV is WIDE - one row per ligand, one COLUMN per cell line, named
    lowercase (``pIC50_mcf7``, ``pIC50_a549``, ``pIC50_a2780``,
    ``pIC50_a2780cis``).  No row can ever contain the string ``"MCF7"``, so
    the count was always 0;
  * ``grep -c`` prints "0" AND exits 1 when nothing matches, so the
    ``|| echo 0`` fallback appended a SECOND "0": the captured value became
    ``"0\\n0"`` and ``[ "0\\n0" -lt 50 ]`` raised "integer expression
    expected".  In bash a test that errors returns false, so the LOW branch
    was unreachable and control fell through to the PASS branch;
  * net effect: a check that could not fail, while printing "PlatinAI 4-cell
    verified" - a zombie gate.  AGENTS.md §5.2 forbids a check that cannot
    fail; the launcher's own ``PLAT_4CELL_META`` path was DEFINED and never
    READ, so the test that asserts "the wrapper references platinai_4cell.
    meta.json for 4-cell schema verification" passed on a STRING while no
    schema was ever verified.

This asks the CSV instead, with a real CSV parser, and uses the same
"active" definition the dataset builder used
(``molmetal/scripts/build_platinai_4cell_csv.py``): a row is ACTIVE for a
cell line when its pIC50 for that cell is ``>= 5.0`` (IC50 <= 10 uM).

The rounding boundary (why the meta cross-check is a band, not equality)
-------------------------------------------------------------------------
The builder decided "active" on the UNROUNDED pIC50 and then wrote
``round(x, 4)`` to the CSV.  A row whose unrounded value lies in
``[4.99995, 5.0)`` is written as exactly ``5.0``, so any recompute-from-CSV
counts it active while the builder did not.  The CSV-recomputed active count
can therefore only be >= the builder's count, and by at most the number of
rows whose value is exactly 5.0.  The cross-check uses that EXACT bound
instead of a magic tolerance:

    meta_active <= computed_active <= meta_active + n_boundary

where ``n_boundary`` is the number of rows whose value is exactly 5.0.
(MEASURED on this repo's dataset: exactly 1 such row, in ``pIC50_a2780`` -
which is why an equality check would have reported a false drift.)

Usage
-----
    python molmetal/scripts/wf_platinai_4cell_check.py \\
        --csv .data/molmetal/platinai_4cell.csv \\
        --meta .data/molmetal/platinai_4cell.meta.json \\
        --min-active 50

Stdout invariant: ``PASS`` / ``FAIL`` / ``SKIP`` (AGENTS.md §4).  Exit 0=PASS,
1=FAIL, 2=SKIP (input file missing).

Finding severities
------------------
Findings are classified, because the launcher's intent distinguishes them:

  * ``FAIL`` - the data does not carry what the plan assumes: a declared
    column is absent, a cell line has no column, a label is not numeric, the
    meta and the CSV disagree, the CSV has no rows.  Exit 1.
  * ``WARN`` - the data is structurally sound but degraded: a cell line has
    fewer active rows than ``--min-active``.  Exit 0 (PASS with a warning).
    This is the severity the launcher's dead branch originally intended.
  * ``SKIP`` - the input file is missing.  Exit 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_CELLS = ("MCF7", "A549", "A2780", "A2780cis")

# The dataset builder's own definition (build_platinai_4cell_csv.py:345):
# "active iff pIC50 >= 5.0 (i.e. IC50 <= 10 uM)".
ACTIVE_THRESHOLD = 5.0

# A value written as exactly this is AMBIGUOUS after round(x, 4): the
# unrounded value could be just below or just above 5.0.
BOUNDARY_VALUE = 5.0


def _resolve_column(cell: str, header: List[str]) -> Optional[str]:
    """Map a cell-line name (MCF7) to its CSV column (pIC50_mcf7).

    Exact match on the ``pIC50_<cell>`` convention first; a unique
    case-insensitive substring match second.  A non-unique substring match
    returns None rather than guessing.
    """
    exact = f"pIC50_{cell.lower()}"
    for h in header:
        if h == exact:
            return h
    cands = [h for h in header if cell.lower() in h.lower()]
    if len(cands) == 1:
        return cands[0]
    return None


def _load_meta(meta_path: Path) -> Optional[dict]:
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None


def check(
    csv_path: Path,
    meta_path: Optional[Path] = None,
    cells: Tuple[str, ...] = DEFAULT_CELLS,
    min_active: int = 50,
    active_threshold: float = ACTIVE_THRESHOLD,
) -> dict:
    """Measure the 4 cell-line columns and cross-check against the meta.

    Returns a dict with ``verdict`` in {PASS, FAIL, SKIP}, a ``findings``
    list, and a ``cells`` list of per-cell measurements.
    """
    result: dict = {
        "verdict": "FAIL",
        "csv": str(csv_path),
        "meta": str(meta_path) if meta_path else None,
        "min_active": min_active,
        "active_threshold": active_threshold,
        "findings": [],
        "cells": [],
    }
    findings: List[dict] = result["findings"]

    def _add(severity: str, msg: str) -> None:
        findings.append({"severity": severity, "msg": msg})

    if not csv_path.is_file():
        result["verdict"] = "SKIP"
        _add("skip", f"CSV missing: {csv_path}")
        return result

    meta = _load_meta(meta_path) if meta_path else None
    if meta_path is not None and meta is None:
        _add("warn", f"meta unreadable (cross-check skipped): {meta_path}")

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    result["n_rows"] = len(rows)
    result["header"] = header

    if not rows:
        result["verdict"] = "FAIL"
        _add("fail", f"CSV has no data rows: {csv_path}")
        return result

    # The cell lines to check: the meta knows them; otherwise the default 4.
    if meta and meta.get("cell_lines"):
        wanted = tuple(str(c) for c in meta["cell_lines"])
    else:
        wanted = tuple(cells)

    # --- schema cross-check: every column the meta declares must exist ---
    if meta and meta.get("schema"):
        missing_cols = [c for c in meta["schema"] if c not in header]
        if missing_cols:
            _add(
                "fail",
                f"CSV header is missing {len(missing_cols)} column(s) declared in "
                f"the meta: {missing_cols}",
            )

    # --- row-count cross-check: the meta and the CSV are written together ---
    if meta and meta.get("n_rows") is not None:
        meta_rows = int(meta["n_rows"])
        if meta_rows != len(rows):
            _add(
                "fail",
                f"row-count drift: meta says {meta_rows}, CSV has {len(rows)} "
                f"(the meta is stale or the CSV was regenerated)",
            )

    # --- per-cell measurement ---
    for cell in wanted:
        col = _resolve_column(cell, header)
        if col is None:
            _add("fail", f"cell line {cell}: no CSV column found for it")
            result["cells"].append({"cell": cell, "column": None, "active": None})
            continue
        numeric = 0
        active = 0
        boundary = 0
        missing = 0
        for r in rows:
            raw = (r.get(col) or "").strip()
            if raw == "":
                missing += 1
                continue
            try:
                val = float(raw)
            except ValueError:
                missing += 1
                continue
            numeric += 1
            if val == BOUNDARY_VALUE:
                boundary += 1
            if val >= active_threshold:
                active += 1
        rec = {
            "cell": cell,
            "column": col,
            "population": len(rows),
            "numeric": numeric,
            "missing": missing,
            "active": active,
            "boundary_rows": boundary,
        }
        result["cells"].append(rec)
        if missing:
            _add(
                "fail",
                f"cell line {cell} ({col}): {missing} row(s) have no numeric pIC50",
            )
        if active < min_active:
            # WARN, not FAIL: the launcher's original intent for a low active
            # count was a warning, not a block.  The defect being repaired is
            # that the branch was UNREACHABLE, not that it was too lenient.
            _add(
                "warn",
                f"cell line {cell} ({col}): only {active} active rows "
                f"(pIC50 >= {active_threshold}); expected >= {min_active}",
            )
        # meta cross-check against the EXACT rounding-boundary band
        if meta and meta.get("n_active_per_line"):
            exp = meta["n_active_per_line"].get(cell)
            if exp is not None:
                exp_i = int(exp)
                rec["meta_active"] = exp_i
                rec["meta_delta"] = active - exp_i
                if not (exp_i <= active <= exp_i + boundary):
                    _add(
                        "fail",
                        f"cell line {cell} ({col}): active count {active} is outside "
                        f"the band [{exp_i}, {exp_i + boundary}] implied by the meta "
                        f"(n_active_per_line={exp_i}); the CSV and the meta disagree",
                    )

    result["n_warnings"] = sum(1 for f in findings if f["severity"] == "warn")
    result["n_failures"] = sum(1 for f in findings if f["severity"] == "fail")
    result["verdict"] = "FAIL" if result["n_failures"] else "PASS"
    return result


def _fmt(result: dict) -> str:
    lines: List[str] = []
    tag = "[platinai-4cell-check]"
    lines.append(f"{tag} csv: {result.get('csv')}")
    for rec in result.get("cells", []):
        if rec.get("column") is None:
            lines.append(f"{tag}   {rec['cell']:<9s} column=<NOT FOUND>")
            continue
        meta_delta = rec.get("meta_delta")
        delta_txt = "" if meta_delta is None else f"  meta_delta={meta_delta:+d}"
        lines.append(
            f"{tag}   {rec['cell']:<9s} column={rec['column']:<16s} "
            f"population={rec['population']} numeric={rec['numeric']} "
            f"missing={rec['missing']} active={rec['active']} "
            f"(pIC50 >= {result.get('active_threshold')}){delta_txt}"
        )
    for f in result.get("findings", []):
        sev = f.get("severity", "fail").upper()
        lines.append(f"{tag} FINDING [{sev}]: {f.get('msg', f)}")
    lines.append(result["verdict"])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--csv", required=True, help="PlatinAI 4-cell CSV path")
    ap.add_argument("--meta", default=None, help="optional meta JSON path")
    ap.add_argument(
        "--cells",
        default=",".join(DEFAULT_CELLS),
        help="comma-separated cell lines to check (default: the 4 PlatinAI lines)",
    )
    ap.add_argument(
        "--min-active",
        type=int,
        default=50,
        help="minimum ACTIVE rows per cell line (default 50)",
    )
    ap.add_argument(
        "--active-threshold",
        type=float,
        default=ACTIVE_THRESHOLD,
        help="pIC50 at/above which a row counts as active (default 5.0)",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    cells = tuple(c.strip() for c in args.cells.split(",") if c.strip())
    result = check(
        csv_path=Path(args.csv),
        meta_path=Path(args.meta) if args.meta else None,
        cells=cells,
        min_active=args.min_active,
        active_threshold=args.active_threshold,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_fmt(result))
    return {"PASS": 0, "FAIL": 1, "SKIP": 2}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
