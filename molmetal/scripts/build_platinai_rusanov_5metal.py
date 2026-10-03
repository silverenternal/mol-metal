#!/usr/bin/env python3
"""Wf-251-F5 — PlatinAI Rusanov-2025 xlsx → 5-metal pIC50 CSV.

=====================================================================
Loader that converts the Rusanov 2025 ChemRxiv V2 PlatinAI xlsx
(17,732 IC50 values across 3,725 Pt complexes; MBFinder free download —
DOI 10.26434/chemrxiv-2025-pp32k V2) into a 5-metal CSV compatible
with the m16 trainer (TODO-130 / Wf-247-C redesign).

The honest framing (per fix_design §F5): ChEMBL29 contains 0 Pt/Ru/Ir
coordination SMILES — the original "ChEMBL redo" task is a misnomer.
The right primary source is the Rusanov 2025 xlsx (real measured
pIC50; no synthetic extrapolation).

Usage::

    # After downloading the xlsx from MBFinder:
    /home/hugo/codes/mol-metal/.venv/bin/python -u \\
        molmetal/scripts/build_platinai_rusanov_5metal.py \\
        --xlsx-path /path/to/rusanov_2025_platinai.xlsx \\
        --out-csv molmetal/data/platinai_rusanov_5metal.csv \\
        --summary molmetal/reports/wf251_apply/platinai_rusanov_summary.json

Notes
-----
* CPU-only; uses openpyxl/pandas (whichever is available).
* Pt coordination bonds use ``[Pt]`` atom-typed SMILES that may be
  intentionally un-Sanitized; we fall back to RDKit relaxed sanitization.
* Per (smiles, cell_line) replicates are aggregated by median.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    p.add_argument(
        "--xlsx-path", type=Path, required=True,
        help="Path to the Rusanov 2025 PlatinAI xlsx file.",
    )
    p.add_argument(
        "--out-csv", type=Path, required=True,
        help="Output CSV path (per-row: smiles, metal, cell_line, pic50).",
    )
    p.add_argument(
        "--summary", type=Path, default=None,
        help="Optional JSON summary path (default <out-csv>.summary.json).",
    )
    p.add_argument(
        "--sheet", type=str, default=None,
        help="Excel sheet name (default: first sheet).",
    )
    p.add_argument(
        "--smiles-col", type=str, default=None,
        help="SMILES column name (auto-detected if None).",
    )
    p.add_argument(
        "--ic50-col", type=str, default=None,
        help="IC50 (nM) column name (auto-detected if None).",
    )
    p.add_argument(
        "--metal-col", type=str, default=None,
        help="Metal column name (auto-detected if None; default 'Pt').",
    )
    p.add_argument(
        "--cell-line-col", type=str, default=None,
        help="Cell line column name (auto-detected if None).",
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Cap on rows (smoke).",
    )
    p.add_argument(
        "--no-rdkit-validate", action="store_true",
        help="Skip RDKit SMILES validation (faster).",
    )
    return p.parse_args(argv)


def _auto_detect_col(header: List[str], candidates: Sequence[str]) -> Optional[str]:
    """Find the first column name in ``header`` that matches any candidate
    (case-insensitive substring match)."""
    lower_map = {h.lower(): h for h in header}
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in lower_map:
            return lower_map[cand_l]
        # substring match
        for hl, h in lower_map.items():
            if cand_l in hl:
                return h
    return None


def _read_xlsx(
    xlsx_path: Path,
    sheet: Optional[str],
) -> List[Dict[str, str]]:
    """Read xlsx into a list of dicts.  Tries pandas first (handles most
    Excel formats cleanly), falls back to openpyxl directly."""
    rows: List[Dict[str, str]] = []
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path, sheet_name=sheet or 0)
        rows = df.to_dict(orient="records")
        # Coerce all values to str for downstream handling.
        rows = [
            {str(k): "" if v is None else str(v)} for k, v in rows
        ]
        return rows
    except Exception as exc:
        print(f"[wf251-F5] pandas read failed ({exc}); falling back to openpyxl",
              flush=True)
    try:
        import openpyxl
        wb = openpyxl.load_workbook(filename=str(xlsx_path), read_only=True,
                                    data_only=True)
        ws = wb[sheet] if sheet else wb.active
        header: List[str] = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                header = [str(c) if c is not None else "" for c in row]
                continue
            d: Dict[str, str] = {}
            for j, v in enumerate(row):
                key = header[j] if j < len(header) else f"col{j}"
                if v is None:
                    d[key] = ""
                else:
                    # Coerce to str so `.strip()` calls work on every value
                    # type (int, float, datetime, etc.).
                    d[key] = str(v)
            rows.append(d)
        return rows
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read xlsx {xlsx_path}: pandas and openpyxl both "
            f"failed. Original error: {exc!r}"
        )


def _validate_smiles_rdkit(smi: str) -> bool:
    """Validate SMILES via RDKit; fall back to relaxed sanitization for
    Pt coordination complexes that intentionally use ``[Pt]`` atom tokens.
    Returns True if the SMILES parses to a non-empty mol."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumHeavyAtoms() >= 1:
            return True
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                 Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                 Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                 catchErrors=True)
                return mol.GetNumHeavyAtoms() >= 1
            except Exception:
                pass
        return False
    except Exception:
        return False


def aggregate_pic50(
    rows: List[Dict[str, str]],
    *,
    smiles_col: str,
    ic50_col: str,
    metal_col: str,
    cell_line_col: str,
    use_rdkit: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Aggregate per-(smiles, cell_line) → median pIC50.

    Returns (out_rows, stats) where ``out_rows`` is the aggregated
    list and ``stats`` summarises parse/aggregate counts.
    """
    out_rows: List[Dict[str, object]] = []
    bucket: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    metals: Dict[Tuple[str, str, str], str] = {}
    smiles_seen: Dict[Tuple[str, str, str], str] = {}
    stats = {
        "n_total": 0,
        "n_no_smiles": 0,
        "n_no_ic50": 0,
        "n_invalid_ic50": 0,
        "n_rdkit_failed": 0,
        "n_aggregated": 0,
    }
    for row in rows:
        stats["n_total"] += 1
        smi_raw = row.get(smiles_col)
        smi = (str(smi_raw) if smi_raw is not None else "").strip()
        if not smi:
            stats["n_no_smiles"] += 1
            continue
        ic50_raw = row.get(ic50_col)
        ic50_str = str(ic50_raw) if ic50_raw is not None else ""
        ic50_str = ic50_str.strip()
        if not ic50_str:
            stats["n_no_ic50"] += 1
            continue
        try:
            ic50_nm = float(ic50_str)
        except ValueError:
            stats["n_invalid_ic50"] += 1
            continue
        if not math.isfinite(ic50_nm) or ic50_nm <= 0:
            stats["n_invalid_ic50"] += 1
            continue
        # nM → µM → pIC50: pIC50 = -log10(M) = 6 - log10(nM)
        pic50 = 6.0 - math.log10(ic50_nm)
        metal_raw = row.get(metal_col)
        metal = (str(metal_raw) if metal_raw is not None else "Pt").strip() or "Pt"
        cell_raw = row.get(cell_line_col)
        cell_line = (str(cell_raw) if cell_raw is not None else "UNK").strip() or "UNK"
        if use_rdkit and not _validate_smiles_rdkit(smi):
            stats["n_rdkit_failed"] += 1
            continue
        key = (smi, metal, cell_line)
        bucket[key].append(float(pic50))
        metals[key] = metal
        smiles_seen[key] = smi
    for key, vals in bucket.items():
        smi, metal, cell_line = key
        if not vals:
            continue
        # Median replicate aggregation
        pic50_med = float(sorted(vals)[len(vals) // 2])
        out_rows.append({
            "smiles": smi,
            "metal": metal,
            "cell_line": cell_line,
            "pic50": pic50_med,
            "n_replicates": int(len(vals)),
        })
        stats["n_aggregated"] += 1
    return out_rows, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.xlsx_path.exists():
        print(f"ERROR: xlsx not found at {args.xlsx_path}", flush=True)
        return 1

    t0 = time.time()
    print(f"[wf251-F5] reading {args.xlsx_path}...", flush=True)
    raw_rows = _read_xlsx(args.xlsx_path, args.sheet)
    print(f"[wf251-F5] read {len(raw_rows)} raw rows", flush=True)
    if not raw_rows:
        print("[wf251-F5] empty xlsx — aborting", flush=True)
        return 1

    # Auto-detect column names
    header = list(raw_rows[0].keys())
    smiles_col = args.smiles_col or _auto_detect_col(
        header, ("smiles", "SMILES", "structure", "Structure"),
    )
    ic50_col = args.ic50_col or _auto_detect_col(
        header, ("ic50_nm", "IC50_nM", "ic50", "IC50", "ic50 (nm)",
                 "pIC50", "pic50"),
    )
    # If we matched pIC50, no need to convert — handled below.
    is_pic50_col = bool(ic50_col and ic50_col.lower().startswith("pic50"))
    metal_col = args.metal_col or _auto_detect_col(
        header, ("metal", "Metal", "metal_id", "ion"),
    )
    cell_line_col = args.cell_line_col or _auto_detect_col(
        header, ("cell_line", "cell", "Cell", "CellLine", "cell line",
                 "tissue"),
    )
    if not smiles_col or not ic50_col:
        print(
            f"ERROR: could not detect SMILES / IC50 columns. "
            f"smiles_col={smiles_col}, ic50_col={ic50_col}, header={header}",
            flush=True,
        )
        return 1
    print(
        f"[wf251-F5] auto-detected columns: smiles={smiles_col!r}, "
        f"ic50/pIC50={ic50_col!r}, metal={metal_col!r}, "
        f"cell_line={cell_line_col!r}",
        flush=True,
    )

    # Subset to max-rows
    if args.max_rows is not None:
        raw_rows = raw_rows[: args.max_rows]

    # Aggregate
    use_rdkit = not args.no_rdkit_validate
    if is_pic50_col:
        # Already pIC50 — no conversion; aggregate directly.
        bucket: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        metals: Dict[Tuple[str, str, str], str] = {}
        out_rows: List[Dict[str, object]] = []
        stats = {"n_total": 0, "n_no_smiles": 0, "n_no_pic50": 0,
                 "n_invalid_pic50": 0, "n_rdkit_failed": 0,
                 "n_aggregated": 0}
        for row in raw_rows:
            stats["n_total"] += 1
            smi_raw = row.get(smiles_col)
            smi = (str(smi_raw) if smi_raw is not None else "").strip()
            if not smi:
                stats["n_no_smiles"] += 1
                continue
            v_raw = row.get(ic50_col)
            v_str = str(v_raw) if v_raw is not None else ""
            v_str = v_str.strip()
            if not v_str:
                stats["n_no_pic50"] += 1
                continue
            try:
                v = float(v_str)
            except ValueError:
                stats["n_invalid_pic50"] += 1
                continue
            if not math.isfinite(v):
                stats["n_invalid_pic50"] += 1
                continue
            if use_rdkit and not _validate_smiles_rdkit(smi):
                stats["n_rdkit_failed"] += 1
                continue
            metal_raw = row.get(metal_col)
            metal = (str(metal_raw) if metal_raw is not None else "Pt").strip() or "Pt"
            cell_raw = row.get(cell_line_col)
            cell_line = (str(cell_raw) if cell_raw is not None else "UNK").strip() or "UNK"
            key = (smi, metal, cell_line)
            bucket[key].append(float(v))
        for key, vals in bucket.items():
            smi, metal, cell_line = key
            pic50_med = float(sorted(vals)[len(vals) // 2])
            out_rows.append({
                "smiles": smi,
                "metal": metal,
                "cell_line": cell_line,
                "pic50": pic50_med,
                "n_replicates": int(len(vals)),
            })
            stats["n_aggregated"] += 1
    else:
        out_rows, stats = aggregate_pic50(
            raw_rows,
            smiles_col=smiles_col,
            ic50_col=ic50_col,
            metal_col=metal_col or "metal",
            cell_line_col=cell_line_col or "cell_line",
            use_rdkit=use_rdkit,
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "metal", "cell_line", "pic50",
                         "n_replicates"])
        for r in out_rows:
            writer.writerow([
                r["smiles"], r["metal"], r["cell_line"],
                f"{float(r['pic50']):.6f}",
                r["n_replicates"],
            ])
    wall = time.time() - t0
    print(
        f"[wf251-F5] wrote {len(out_rows)} aggregated rows to {args.out_csv} "
        f"(wall={wall:.1f}s); stats={stats}",
        flush=True,
    )

    # Summary JSON
    summary = {
        "wf": "Wf-251-F5",
        "xlsx_path": str(args.xlsx_path),
        "out_csv": str(args.out_csv),
        "n_rows_total": stats.get("n_total", 0),
        "n_rows_aggregated": stats.get("n_aggregated", 0),
        "n_rows_dropped_no_smiles": stats.get("n_no_smiles", 0),
        "n_rows_dropped_no_target": stats.get("n_no_ic50",
                                              stats.get("n_no_pic50", 0)),
        "n_rows_dropped_invalid_target": stats.get("n_invalid_ic50",
                                                   stats.get("n_invalid_pic50", 0)),
        "n_rows_dropped_rdkit": stats.get("n_rdkit_failed", 0),
        "wall_seconds": wall,
        "smiles_col": smiles_col,
        "ic50_col": ic50_col,
        "metal_col": metal_col,
        "cell_line_col": cell_line_col,
        "is_pic50_column": is_pic50_col,
        "literature_anchor": (
            "Rusanov 2025 ChemRxiv V2 — 17,732 IC50 values over 3,725 Pt "
            "complexes (MBFinder free download, DOI 10.26434/chemrxiv-2025-pp32k V2)."
        ),
        "honest_framing": (
            "Replaces the Wf-247-C 'ChEMBL redo' task — ChEMBL29 has 0 Pt/Ru/Ir "
            "coordination SMILES so the original task was a misnomer. This "
            "loader produces a real (not synthetic) PlatinAI Rusanov 2025 CSV "
            "suitable for m16 trainer retraining."
        ),
    }
    summary_path = args.summary or args.out_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[wf251-F5] wrote summary to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())