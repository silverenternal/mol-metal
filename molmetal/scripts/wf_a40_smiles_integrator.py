"""F2 integration: join smiles cache with real label pool into a new CSV.

Honest framing (AGENTS.md §5.2): if cache is partial, we still write the
integrated file with __no_smiles__ markers and a meta JSON that records the
true coverage. Per task spec: "Accept partial coverage".
"""
import csv
import sys
from pathlib import Path

ROOT = Path("/home/hugo/codes/mol-metal")
POOL = ROOT / ".data/molmetal/platinai_4cell_real_label_pool.csv"
CACHE = ROOT / ".data/molmetal/platinai_smiles_cache.json"
OUT_CSV = ROOT / ".data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv"
META_JSON = ROOT / ".data/molmetal/platinai_4cell_real_label_pool_with_smiles.meta.json"

MISSING_MARKER = "__no_smiles__"


def main() -> int:
    import json
    cache = json.loads(CACHE.read_text())
    with POOL.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("ERROR: pool CSV has no data rows", file=sys.stderr)
        return 1
    fieldnames = list(rows[0].keys())
    if "smiles" not in fieldnames:
        fieldnames.append("smiles")
    n_with_smiles = 0
    n_without = 0
    for row in rows:
        cid = row["compound_code"]
        smi = cache.get(cid, MISSING_MARKER)
        if smi and smi != MISSING_MARKER:
            row["smiles"] = smi
            n_with_smiles += 1
        else:
            row["smiles"] = MISSING_MARKER
            n_without += 1
    with OUT_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    meta = {
        "n_total": len(rows),
        "n_with_smiles": n_with_smiles,
        "n_without_smiles": n_without,
        "smiles_cache_size": len(cache),
        "pool_unique_imageIds": len({r["compound_code"] for r in rows}),
        "cache_unique_imageIds": len(set(cache.keys())),
        "missing_marker": MISSING_MARKER,
        "source_pool": str(POOL.relative_to(ROOT)),
        "source_cache": str(CACHE.relative_to(ROOT)),
    }
    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_CSV.relative_to(ROOT)} rows={len(rows)}")
    print(f"wrote {META_JSON.relative_to(ROOT)}")
    print(f"n_total={meta['n_total']} n_with_smiles={meta['n_with_smiles']} "
          f"n_without_smiles={meta['n_without_smiles']} cache_size={meta['smiles_cache_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())