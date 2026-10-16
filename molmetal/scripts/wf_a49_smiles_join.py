"""Wf-A49: join MB-Finder structures onto the Pt-only label view.

Why a separate step
-------------------
Labels and structures live behind two endpoints:

* ``get-ceil-line-bio-data`` -> the IC50 measurement (no structure)
* ``get-substances``         -> ``smiles``/``mw``/``cLogP`` keyed by ``imageId``

``wf_a47`` walks the second endpoint into
``platinai_pt_substances_smiles.jsonl``; ``wf_a48`` produces the Pt-only label
view.  This script joins them on the compound code (``MB-Pt-000010``).

Coverage is reported honestly rather than assumed: if the substances walk is
still incomplete, some Pt compounds will not resolve to a structure and the
summary says so instead of silently dropping rows.

CLI
---
  --out-dir   default .data/molmetal/wf_a37_platinai_biodata
  --require-full  exit 3 if any Pt compound lacks a structure
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

STRUCT_FIELDS = ["smiles", "mw", "cLogP", "casNumber", "substance_id"]

CSV_FIELDS = [
    "compound_code",
    "cell_line",
    "metal",
    "smiles",
    "value_mean",
    "value_sd",
    "numeric_value_raw",
    "method",
    "incubationTime",
    "doi",
    "title",
    "imageUrl",
    "biodata_id",
    "stability",
    "in_vivo",
    "mw",
    "cLogP",
    "casNumber",
    "substance_id",
]


def load_jsonl(path: Path) -> list:
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
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    pt_dir = out_dir / "pt"
    substances = load_jsonl(out_dir / "platinai_pt_substances_smiles.jsonl")
    if not substances:
        print("[wf-a49] no substances table; run wf_a47 first", flush=True)
        return 2

    struct = {}
    for s in substances:
        code = s.get("compound_code")
        if code and s.get("smiles"):
            struct[code] = s

    pt_files = sorted(
        p
        for p in pt_dir.glob("platinai_*_pt.jsonl")
        if "all_cells" not in p.name
    )
    if not pt_files:
        print(f"[wf-a49] no Pt-only files under {pt_dir}; run wf_a48 first", flush=True)
        return 2

    all_rows: list = []
    per_cell: list = []
    for src in pt_files:
        stem = src.name[len("platinai_") : -len("_pt.jsonl")]
        rows = load_jsonl(src)
        if not rows:
            continue
        for r in rows:
            s = struct.get(r.get("compound_code"))
            if s:
                for k in STRUCT_FIELDS:
                    r[k] = s.get(k)
            all_rows.append(r)

        codes = {r.get("compound_code") for r in rows}
        cov = {c for c in codes if c in struct}
        per_cell.append(
            {
                "cell_line": rows[0].get("cell_line"),
                "pt_rows": len(rows),
                "pt_rows_with_smiles": sum(
                    1 for r in rows if (r.get("smiles") or "").strip()
                ),
                "pt_compounds": len(codes),
                "pt_compounds_with_smiles": len(cov),
            }
        )
        write_jsonl(pt_dir / f"platinai_{stem}_pt_with_smiles.jsonl", rows)
        write_csv(pt_dir / f"platinai_{stem}_pt_with_smiles.csv", rows)

    write_jsonl(pt_dir / "platinai_pt_all_cells_with_smiles.jsonl", all_rows)
    write_csv(pt_dir / "platinai_pt_all_cells_with_smiles.csv", all_rows)

    all_codes = {r.get("compound_code") for r in all_rows}
    missing = sorted(c for c in all_codes if c not in struct)
    summary = {
        "substances_in_table": len(substances),
        "substances_with_smiles": len(struct),
        "pt_rows": len(all_rows),
        "pt_rows_with_smiles": sum(
            1 for r in all_rows if (r.get("smiles") or "").strip()
        ),
        "pt_rows_with_value_and_smiles": sum(
            1
            for r in all_rows
            if (r.get("smiles") or "").strip() and r.get("value_mean") is not None
        ),
        "pt_compounds": len(all_codes),
        "pt_compounds_with_smiles": len(all_codes) - len(missing),
        "pt_compounds_missing_smiles": len(missing),
        "missing_examples": missing[:20],
        "per_cell_line": per_cell,
    }
    atomic_write_file(
        pt_dir / "smiles_join_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )

    print(
        f"[wf-a49] Pt compounds with SMILES: "
        f"{summary['pt_compounds_with_smiles']}/{summary['pt_compounds']}\n"
        f"[wf-a49] Pt rows with SMILES: "
        f"{summary['pt_rows_with_smiles']}/{summary['pt_rows']}\n"
        f"[wf-a49] Pt rows with BOTH value_mean + SMILES: "
        f"{summary['pt_rows_with_value_and_smiles']}",
        flush=True,
    )
    if args.require_full and missing:
        print(f"[wf-a49] INCOMPLETE: {len(missing)} Pt compounds lack a structure",
              flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
