"""Wf-A48: correct metal attribution and emit the Pt-only view.

Two problems this fixes
-----------------------
1. **Every row says ``metal="Pt"``.**  ``_flatten`` used
   ``comp.get("metal") or "Pt"``, but ``get-ceil-line-bio-data`` does not
   return a ``metal`` field, so the fallback fired on every single row.  The
   real metal is the second field of the compound code (``MB-Pt-``/``MB-Cu-``/
   ``MB-Au-``/``MB-Re-``/``MB-Fe-``/``MB-Ru-``).  Measured on the 21-cell
   harvest: 60.7% Pt, 25.8% Cu, 11.8% Au -- so a Pt filter keyed on the
   ``metal`` field silently kept everything.

2. **No Pt-only view existed.**  ``platinai_real_label_loader.py`` filters
   ``imageId.startswith("MB-Pt-")``, but the wf_a37/wf_a46 harvesters do not,
   so the on-disk files mix metals.  Downstream code that expects Pt must
   filter explicitly.

What it writes
--------------
In place, atomic:
  ``platinai_<stem>_biodata.jsonl`` / ``.csv``   metal field corrected

Under ``<out>/pt/``:
  ``platinai_<stem>_pt.jsonl`` / ``.csv``        Pt-only, per cell line
  ``platinai_pt_all_cells.jsonl`` / ``.csv``     Pt-only, all cells combined
  ``pt_only_summary.json``                       counts + per-metal breakdown

CLI
---
  --out-dir   default .data/molmetal/wf_a37_platinai_biodata
  --fix-raw   rewrite the source files' ``metal`` field (default on)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.platinai_real_label_loader import (  # noqa: E402
    metal_from_compound_code,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

CSV_FIELDS = [
    "compound_code",
    "cell_line",
    "metal",
    "value_mean",
    "value_sd",
    "numeric_value_raw",
    "method",
    "incubationTime",
    "smiles",
    "doi",
    "title",
    "imageUrl",
    "biodata_id",
    "stability",
    "in_vivo",
]


def load(path: Path) -> list:
    out: list = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_jsonl(path: Path, records: list) -> None:
    body = "".join(
        json.dumps(r, ensure_ascii=False) + "\n"
        for r in sorted(
            records, key=lambda r: (r.get("_page") or 0, r.get("biodata_id") or 0)
        )
    )
    atomic_write_file(path, body)


def write_csv(path: Path, records: list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    os.replace(tmp, path)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=".data/molmetal/wf_a37_platinai_biodata"
    )
    parser.add_argument(
        "--fix-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rewrite the source files' metal field (default: on)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    pt_dir = out_dir / "pt"
    pt_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(
        p
        for p in out_dir.glob("platinai_*_biodata.jsonl")
        if "_pt" not in p.stem and p.name != "platinai_pt_all_cells.jsonl"
    )
    if not sources:
        print(f"[wf-a48] no source files under {out_dir}", flush=True)
        return 2

    all_pt: list = []
    per_cell: list = []
    global_metals: Counter = Counter()
    global_pt_metals: Counter = Counter()
    raw_fixed = 0

    for src in sources:
        stem = src.name[len("platinai_") : -len("_biodata.jsonl")]
        records = load(src)
        if not records:
            continue

        changed = 0
        for rec in records:
            derived = metal_from_compound_code(rec.get("compound_code"))
            if derived and rec.get("metal") != derived:
                rec["metal"] = derived
                changed += 1
        raw_fixed += changed

        if args.fix_raw and changed:
            write_jsonl(src, records)
            write_csv(out_dir / f"platinai_{stem}_biodata.csv", records)

        pt = [r for r in records if r.get("metal") == "Pt"]
        all_pt.extend(pt)

        global_metals.update(r.get("metal") or "?" for r in records)
        global_pt_metals.update(r.get("metal") or "?" for r in pt)

        if pt:
            write_jsonl(pt_dir / f"platinai_{stem}_pt.jsonl", pt)
            write_csv(pt_dir / f"platinai_{stem}_pt.csv", pt)

        per_cell.append(
            {
                "cell_line": records[0].get("cell_line"),
                "rows_raw": len(records),
                "rows_pt": len(pt),
                "pt_fraction": round(len(pt) / len(records), 4),
                "unique_compounds_raw": len(
                    {r.get("compound_code") for r in records}
                ),
                "unique_compounds_pt": len(
                    {r.get("compound_code") for r in pt}
                ),
                "pt_rows_with_value_mean": sum(
                    1 for r in pt if r.get("value_mean") is not None
                ),
                "pt_rows_with_smiles": sum(
                    1 for r in pt if (r.get("smiles") or "").strip()
                ),
                "metal_field_fixed": changed,
            }
        )

    write_jsonl(pt_dir / "platinai_pt_all_cells.jsonl", all_pt)
    write_csv(pt_dir / "platinai_pt_all_cells.csv", all_pt)

    summary = {
        "source_files": len(sources),
        "metal_field_rows_fixed": raw_fixed,
        "raw_rows_all_metals": sum(c["rows_raw"] for c in per_cell),
        "pt_rows": len(all_pt),
        "pt_unique_compounds": len({r.get("compound_code") for r in all_pt}),
        "pt_rows_with_value_mean": sum(
            1 for r in all_pt if r.get("value_mean") is not None
        ),
        "pt_rows_with_smiles": sum(
            1 for r in all_pt if (r.get("smiles") or "").strip()
        ),
        "raw_rows_by_metal": dict(global_metals.most_common()),
        "per_cell_line": per_cell,
    }
    atomic_write_file(
        pt_dir / "pt_only_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )

    print(
        f"[wf-a48] metal field fixed on {raw_fixed} rows\n"
        f"[wf-a48] raw rows (all metals) = {summary['raw_rows_all_metals']}\n"
        f"[wf-a48] Pt rows = {summary['pt_rows']}  "
        f"unique Pt compounds = {summary['pt_unique_compounds']}  "
        f"with value_mean = {summary['pt_rows_with_value_mean']}  "
        f"with smiles = {summary['pt_rows_with_smiles']}\n"
        f"[wf-a48] by metal: {summary['raw_rows_by_metal']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
