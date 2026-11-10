"""F2 integration: join smiles cache with real label pool into a new CSV.

Honest framing (AGENTS.md §5.2): if cache is partial, we still write the
integrated file with __no_smiles__ markers and a meta JSON that records the
true coverage. Per task spec: "Accept partial coverage".

AGENTS.md §5.6 atomic-write protocol: the integrated CSV + meta JSON are
written via ``atomic_write_file`` (POSIX ``os.replace`` on a tmp file in
the same directory) so a crash mid-write can never leave the on-disk
file half-merged with stale bytes from a prior run. Falls back to a
plain write only when the atomic helper is not importable.
"""
import csv
import io
import json
import sys
from pathlib import Path

try:
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
    _HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback for direct invocation
    _HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / ".data/molmetal/platinai_4cell_real_label_pool.csv"
CACHE = ROOT / ".data/molmetal/platinai_smiles_cache.json"
OUT_CSV = ROOT / ".data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv"
META_JSON = ROOT / ".data/molmetal/platinai_4cell_real_label_pool_with_smiles.meta.json"

MISSING_MARKER = "__no_smiles__"


def _write_atomic(path: Path, content: str) -> None:
    """§5.6 atomic write — falls back to plain write if helper unavailable."""
    if _HAS_ATOMIC and atomic_write_file is not None:
        atomic_write_file(path, content, encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
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
    # §5.6: build CSV in memory + atomic write so a crash mid-write cannot
    # leave the target half-merged with stale bytes.
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
    _write_atomic(OUT_CSV, buf.getvalue())
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
    _write_atomic(META_JSON, json.dumps(meta, indent=2))
    print(f"wrote {OUT_CSV.relative_to(ROOT)} rows={len(rows)}")
    print(f"wrote {META_JSON.relative_to(ROOT)}")
    print(f"n_total={meta['n_total']} n_with_smiles={meta['n_with_smiles']} "
          f"n_without_smiles={meta['n_without_smiles']} cache_size={meta['smiles_cache_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
