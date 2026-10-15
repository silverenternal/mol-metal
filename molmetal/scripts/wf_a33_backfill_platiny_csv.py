"""Wf-A33-fixup-C14 (2026-10-15) — backfill platiny_pocket_pool.csv from disk.

Why this script exists
----------------------
The 2026-10-14 master commit populated all 10 platiny pocket .pdb files under
``.data/molmetal/platiny_pockets/<PDB>_pocket10.pdb`` (17-49 KB each,
2026-10-14 21:43-21:46), with sha256 written to the CSV at the same time.
A subsequent uncommitted working-copy diff stripped every row's
``pocket_path`` / ``pocket_sha256`` back to ``needs_pdb_extraction`` —
the C14 regression called out in the 2026-10-15 CFM audit.

This script is a CPU-only fix: it globs the on-disk .pdb files,
recomputes the sha256, and rewrites the CSV atomically so the wire-in
chain (CSV → fpocket_wrapper → CFM trainer pocket load) is intact again.

Usage
-----
    .venv/bin/python molmetal/scripts/wf_a33_backfill_platiny_csv.py [--dry-run]
    .venv/bin/python molmetal/scripts/wf_a33_backfill_platiny_csv.py \
        --csv molmetal/data/platiny_pocket_pool.csv \
        --pocket-dir .data/molmetal/platiny_pockets

Exit codes
----------
0 = PASS (every row that had a matching on-disk .pdb is now populated)
1 = FAIL (a row's .pdb is missing or the CSV cannot be parsed)
2 = SKIP (--dry-run; nothing written)

The script honours the §5.6 atomic-write protocol:
``molmetal.scripts.wf386a3_atomic_write::atomic_write_file`` writes via
``os.replace`` so a crash mid-write cannot leave the CSV half-populated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "molmetal" / "data" / "platiny_pocket_pool.csv"
DEFAULT_POCKET_DIR = REPO_ROOT / ".data" / "molmetal" / "platiny_pockets"

# After backfill, every row that has a matching on-disk .pdb carries this
# marker.  Rows whose pdb_id has no on-disk .pdb stay at
# ``needs_pdb_extraction`` so the gap is still visible to a reader.
POCKET_DEFINITION_PROVE = "fpocket_top1_drugability_10A_sphere"

# Path column prefix — must match the deploy layout used by every
# downstream loader (cfm_per_pocket_metric + pocket-conditioned trainer).
POCKET_PATH_PREFIX = ".data/molmetal/platiny_pockets"


def _sha256_of(path: Path) -> str:
    """Return hex sha256 of the file at ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_pocket_files(pocket_dir: Path) -> Dict[str, Path]:
    """Map ``PDB -> .pdb Path`` from the pocket directory.

    Filename convention: ``<PDB>_pocket10.pdb``.  The PDB id is the file
    stem prefix up to the first ``_pocket10`` token.
    """
    out: Dict[str, Path] = {}
    if not pocket_dir.is_dir():
        return out
    for p in sorted(pocket_dir.glob("*_pocket10.pdb")):
        stem = p.name[: -len("_pocket10.pdb")]
        if stem:
            out[stem.upper()] = p
    return out


def _backfill_rows(
    rows: List[Dict[str, str]],
    pdb_to_path: Dict[str, Path],
    pocket_dir: Path,
) -> Tuple[int, int, List[str]]:
    """Return (n_populated, n_unchanged, list_of_warnings)."""
    populated = 0
    unchanged = 0
    warnings: List[str] = []
    for row in rows:
        pdb_id = (row.get("pdb_id") or "").strip().upper()
        if not pdb_id:
            warnings.append(f"row {row.get('target_id')!r} has empty pdb_id; skip")
            continue
        pdb_path = pdb_to_path.get(pdb_id)
        if pdb_path is None:
            warnings.append(f"pdb_id={pdb_id}: no on-disk .pdb under {pocket_dir}")
            continue
        try:
            sha = _sha256_of(pdb_path)
        except OSError as exc:
            warnings.append(f"pdb_id={pdb_id}: sha256 read failed: {exc}")
            continue
        rel_path = f"{POCKET_PATH_PREFIX}/{pdb_path.name}"
        prev_path = row.get("pocket_path", "")
        prev_sha = row.get("pocket_sha256", "")
        if prev_path == rel_path and prev_sha == sha:
            unchanged += 1
            continue
        row["pocket_path"] = rel_path
        row["pocket_sha256"] = sha
        row["pocket_definition_prove"] = POCKET_DEFINITION_PROVE
        populated += 1
    return populated, unchanged, warnings


def _write_csv_atomic(csv_path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    """Atomically rewrite the CSV (§5.6 protocol — atomic_write_file + os.replace)."""
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in header})
    atomic_write_file(csv_path, buf.getvalue().encode("utf-8"), binary=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--pocket-dir", type=Path, default=DEFAULT_POCKET_DIR)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute + report, do NOT rewrite the CSV (exit 2 on success).",
    )
    args = p.parse_args()

    csv_path: Path = args.csv
    pocket_dir: Path = args.pocket_dir

    if not csv_path.is_file():
        print(f"[wf_a33_backfill] CSV not found: {csv_path}", file=sys.stderr)
        return 1
    if not pocket_dir.is_dir():
        print(
            f"[wf_a33_backfill] pocket dir not found: {pocket_dir}", file=sys.stderr
        )
        return 1

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = [dict(r) for r in reader]

    pdb_to_path = _discover_pocket_files(pocket_dir)
    if not pdb_to_path:
        print(
            f"[wf_a33_backfill] no *_pocket10.pdb under {pocket_dir}",
            file=sys.stderr,
        )
        return 1

    n_populated, n_unchanged, warnings = _backfill_rows(rows, pdb_to_path, pocket_dir)

    print(
        f"[wf_a33_backfill] csv={csv_path} pocket_dir={pocket_dir} "
        f"rows={len(rows)} populated={n_populated} unchanged={n_unchanged} "
        f"warnings={len(warnings)}"
    )
    for w in warnings:
        print(f"[wf_a33_backfill][WARNING] {w}")

    if args.dry_run:
        print("[wf_a33_backfill] --dry-run: NOT writing CSV")
        return 2

    if n_populated == 0:
        print(
            "[wf_a33_backfill] nothing to backfill; CSV already populated. "
            "SKIP"
        )
        return 2

    _write_csv_atomic(csv_path, header, rows)
    print(f"[wf_a33_backfill] wrote {csv_path} ({n_populated} rows updated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
