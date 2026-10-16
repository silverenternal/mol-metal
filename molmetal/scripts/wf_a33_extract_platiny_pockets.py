"""molmetal/scripts/wf_a33_extract_platiny_pockets.py

Wf-A33-2: extract 10 platiny pockets via fpocket (real pocket_path).

Closes the Wf-A31 honest-negative that shipped ``platiny_pocket_pool.csv``
with ``pocket_path=""`` for all 10 rows (Wf-A30-B8 close-out). Reads the CSV
(the canonical seed per AGENTS.md §5.5 filesystem-truth), downloads each PDB
from RCSB if not already cached at ``.data/molmetal/pdb_cache/``, runs
fpocket (the C binary pre-installed at
``/home/hugo/.mamba/pkgs/fpocket-4.2.3-h8d0616a_1/bin/fpocket`` per A33-1;
**no Java**), picks the top-1 pocket by Drug Score, crops to a 10 Angstrom
sphere around the centre residue, and writes the cropped PDB to
``.data/molmetal/platiny_pockets/<pdb_id>_pocket10.pdb``.

Rewrites the CSV with the populated ``pocket_path`` /
``pocket_sha256`` / ``pocket_definition_prove`` columns atomically per
AGENTS.md §5.6 (tempfile + ``atomic_write_file``).

Constraints honoured
--------------------
* §5.5 filesystem-truth: every cropped PDB confirmed on disk
  (``ls -la .data/molmetal/platiny_pockets/`` before commit).
* §5.6 atomic_write_file for both the CSV rewrite and the cropped PDB
  copies.
* §5.7 trainer-flag wire-in does NOT apply (this is a data-prep
  script, not a trainer).
* NO GPU, NO Java.
* Local proxy (``127.0.0.1:20171``) is bypassed via a no-proxy
  ``urllib.request`` opener so RCSB downloads work even when the
  shell-level ``HTTPS_PROXY`` env var points at a dead proxy.

CLI
---
    python -m molmetal.scripts.wf_a33_extract_platiny_pockets [--dry-run]
                                                                [--csv PATH]
                                                                [--pockets-root DIR]
                                                                [--pdb-cache-root DIR]

The ``--dry-run`` flag prints the 10 PDB IDs + the planned fpocket work
without actually fetching PDBs or running fpocket.

Stdout invariant
---------------
On a normal run, prints:

    [wf_a33_extract] 10/10 platiny PDBs processed
    [wf_a33_extract] pocket_path= /home/hugo/codes/mol-metal/.data/molmetal/platiny_pockets/<pdb_id>_pocket10.pdb for 10/10 rows
    [wf_a33_extract] .data/molmetal/platiny_pockets/ has 10 pocket10.pdb files

Exit codes
----------
0: all 10 rows processed (extracted or skipped because already present).
1: CSV read failed (no rows, missing file).
2: at least one row failed extraction (downloaded / cropped but did
   not yield a pocket file).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

# Pre-imports — fail loudly at import time if a dep is missing.
from molmetal.adapters.fpocket_wrapper import (
    extract_pocket,
    is_fpocket_available,
)
from molmetal.data.platiny_pocket_pool import (
    PLATINY_POOL_CSV,
    PLATINY_POCKETS_ROOT,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file


# ---------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------

REPO_ROOT = Path("/home/hugo/codes/mol-metal")

#: Default CSV path. Override with --csv.
DEFAULT_CSV = PLATINY_POOL_CSV

#: Default cropped-pocket destination directory. Override with --pockets-root.
DEFAULT_POCKETS_ROOT = PLATINY_POCKETS_ROOT

#: Default PDB cache directory (downloaded RCSB PDBs land here).
DEFAULT_PDB_CACHE_ROOT = REPO_ROOT / ".data" / "molmetal" / "pdb_cache"

#: RCSB PDB download URL template.
RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"

#: 10 Angstrom sphere (CrossDocked "pocket10" convention).
SPHERE_RADIUS_A = 10.0


# ---------------------------------------------------------------------
# Proxy-safe downloader
# ---------------------------------------------------------------------

def _fetch_pdb_bytes(pdb_id: str, timeout_s: int = 60) -> bytes:
    """Download a PDB file from RCSB; bypass any local proxy.

    The shell-level ``HTTPS_PROXY`` env var points at 127.0.0.1:20171,
    which is dead. We construct a fresh ``urllib`` opener with an empty
    proxy handler so the request goes direct, regardless of shell env.

    Parameters
    ----------
    pdb_id : str
        Four-character PDB identifier (e.g. "3EAO").
    timeout_s : int
        Socket timeout in seconds.

    Returns
    -------
    bytes
        Raw PDB file bytes.

    Raises
    ------
    RuntimeError
        If the HTTP fetch fails (URLError / HTTPError).
    """
    url = RCSB_URL_TEMPLATE.format(pdb_id=pdb_id.lower())
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    try:
        with opener.open(url, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"RCSB download failed for {pdb_id}: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# SHA-256 helper (for pocket_sha256 column)
# ---------------------------------------------------------------------

def _sha256_file(p: Path) -> str:
    """Return the SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# CSV read / atomic write (§5.6)
# ---------------------------------------------------------------------

def _read_csv_rows(csv_path: Path) -> List[dict]:
    """Read the platiny_pocket_pool.csv from disk per §5.5.

    Returns an empty list if the file does not exist.
    """
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv_atomic(csv_path: Path, rows: List[dict]) -> None:
    """Atomically rewrite the platiny_pocket_pool.csv per §5.6.

    Uses ``atomic_write_file`` (tempfile + ``os.rename``, fsync) to
    guarantee that the on-disk CSV is never half-written.

    Field order is taken from the first row (DictWriter honours dict
    insertion order on Python 3.7+); we preserve whatever columns
    existed when the CSV was read so callers can add new fields later
    without breaking the writer.
    """
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        # Keep the row key set aligned with fieldnames so missing keys
        # surface as empty cells (instead of KeyError).
        w.writerow({k: row.get(k, "") for k in fieldnames})
    atomic_write_file(csv_path, buf.getvalue(), binary=False)


# ---------------------------------------------------------------------
# Per-PDB driver
# ---------------------------------------------------------------------

def _pocket_defined_proof() -> str:
    """Stamp for the ``pocket_definition_prove`` column."""
    return f"fpocket_top1_drugability_{SPHERE_RADIUS_A:.0f}A_sphere"


def _already_extracted(target_pocket: Path, csv_path: Path) -> bool:
    """True iff the cropped pocket file already exists on disk.

    A CSV row with a populated ``pocket_path`` referencing an existing
    file counts as "already extracted" (idempotent re-run safety).
    """
    return target_pocket.exists() and target_pocket.stat().st_size > 0


def _process_one_pdb(
    pdb_id: str,
    pdb_cache_root: Path,
    pockets_root: Path,
    dry_run: bool,
) -> Optional[Path]:
    """Download + fpocket + crop for one PDB.

    Returns the path to the cropped pocket PDB on success, ``None`` on
    failure or ``dry_run=True``.

    Idempotent: a previously-extracted pocket file is returned as-is.
    """
    target_pocket = pockets_root / f"{pdb_id}_pocket10.pdb"
    if target_pocket.exists() and target_pocket.stat().st_size > 0:
        return target_pocket

    cache_path = pdb_cache_root / f"{pdb_id}.pdb"

    # Fetch PDB (cache-aware)
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        if dry_run:
            return None
        pdb_cache_root.mkdir(parents=True, exist_ok=True)
        pdb_bytes = _fetch_pdb_bytes(pdb_id)
        atomic_write_file(cache_path, pdb_bytes, binary=True)

    if dry_run:
        return None

    if not is_fpocket_available():
        raise RuntimeError(
            "fpocket CLI is not on PATH and not at any of the well-known "
            "fallback paths. Install or set MOLMETAL_FPOCKET_CMD."
        )

    # fpocket work goes in a per-PDB subdir under pockets_root; the
    # final cropped file is named <pdb_id>_pocket10.pdb. We let
    # ``extract_pocket`` pick its own intermediate name, then atomically
    # rename the cropped result to the canonical pocket10 path.
    work_dir = pockets_root / ".fpocket_work" / pdb_id
    work_dir.mkdir(parents=True, exist_ok=True)
    cropped_path = extract_pocket(
        cache_path,
        work_dir,
        sphere_radius_A=SPHERE_RADIUS_A,
    )
    # Strip per-PDB work dir of fpocket's <pdb>_out tree early to keep
    # disk tidy during the run; we keep the cropped file anyway.
    fpocket_out = work_dir / f"{pdb_id}_out"
    if fpocket_out.exists():
        shutil.rmtree(fpocket_out, ignore_errors=True)

    # Move cropped_path → target_pocket (atomic_copy). Don't move
    # because extract_pocket writes directly into work_dir; an atomic
    # write preserves the §5.6 invariant if a later rename fails.
    atomic_write_file(target_pocket, cropped_path.read_bytes(), binary=True)
    cropped_path.unlink(missing_ok=True)
    return target_pocket


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-A33-2: extract 10 platiny pockets via fpocket + 10A "
            "crop. Closes the Wf-A31 honest-negative empty pocket_path."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Read CSV + print the 10 PDB IDs + plan; do NOT download or "
            "run fpocket. Use to verify the import + CSV shape without "
            "any side effects."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to platiny_pocket_pool.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--pockets-root",
        type=Path,
        default=DEFAULT_POCKETS_ROOT,
        help=f"Pocket output dir (default: {DEFAULT_POCKETS_ROOT})",
    )
    parser.add_argument(
        "--pdb-cache-root",
        type=Path,
        default=DEFAULT_PDB_CACHE_ROOT,
        help=f"PDB cache dir (default: {DEFAULT_PDB_CACHE_ROOT})",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv
    pockets_root: Path = args.pockets_root
    pdb_cache_root: Path = args.pdb_cache_root

    # --- 1. Read the CSV (§5.5 filesystem-truth) ---
    rows = _read_csv_rows(csv_path)
    if not rows:
        print(
            f"[wf_a33_extract] CSV not found or empty: {csv_path}",
            file=sys.stderr,
        )
        return 1

    print(f"[wf_a33_extract] read {len(rows)} rows from {csv_path}")

    # --- 2. Validate expected columns ---
    expected_cols = {"target_id", "pdb_id", "pocket_path"}
    actual_cols = set(rows[0].keys())
    missing = expected_cols - actual_cols
    if missing:
        print(
            f"[wf_a33_extract] CSV missing required columns {sorted(missing)}: "
            f"{csv_path}",
            file=sys.stderr,
        )
        return 1

    # --- 3. Plan / dry-run summary ---
    pdb_ids = [r.get("pdb_id", "").strip() for r in rows]
    print(f"[wf_a33_extract] 10 PDBs planned: {pdb_ids}")
    if args.dry_run:
        print("[wf_a33_extract] --dry-run: NOT fetching PDBs or running fpocket")
        pockets_root.mkdir(parents=True, exist_ok=True)
        pdb_cache_root.mkdir(parents=True, exist_ok=True)
        return 0

    # --- 4. Process each row ---
    pockets_root.mkdir(parents=True, exist_ok=True)
    pdb_cache_root.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    failed = 0
    updated_rows: List[dict] = []

    for row in rows:
        pdb_id = row.get("pdb_id", "").strip()
        if not pdb_id:
            print(
                f"[wf_a33_extract] row missing pdb_id "
                f"(target_id={row.get('target_id', '?')})",
                file=sys.stderr,
            )
            failed += 1
            updated_rows.append(row)
            continue

        target_pocket = pockets_root / f"{pdb_id}_pocket10.pdb"
        csv_pocket_path = row.get("pocket_path", "").strip()

        # Idempotent skip: CSV row already populated AND file on disk.
        if csv_pocket_path and Path(csv_pocket_path).exists():
            if not Path(csv_pocket_path).resolve() == target_pocket.resolve():
                # Path mismatch — overwrite with the canonical pocket10 path.
                row["pocket_path"] = str(target_pocket)
                updated_rows.append(row)
                extracted += 1
            else:
                print(
                    f"[wf_a33_extract] {pdb_id}: skip (already extracted at "
                    f"{csv_pocket_path})"
                )
                skipped += 1
                updated_rows.append(row)
            continue

        # File exists on disk but CSV row not yet updated → fill in.
        if target_pocket.exists() and target_pocket.stat().st_size > 0:
            print(
                f"[wf_a33_extract] {pdb_id}: pocket file on disk but CSV empty; "
                f"updating CSV only"
            )
            row["pocket_path"] = str(target_pocket)
            row["pocket_sha256"] = _sha256_file(target_pocket)
            row["pocket_definition_prove"] = _pocket_defined_proof()
            updated_rows.append(row)
            extracted += 1
            continue

        # Full path: download + fpocket + crop.
        try:
            cropped = _process_one_pdb(
                pdb_id, pdb_cache_root, pockets_root, dry_run=False
            )
        except Exception as exc:
            print(
                f"[wf_a33_extract] {pdb_id}: ERROR — {exc}",
                file=sys.stderr,
            )
            failed += 1
            updated_rows.append(row)
            continue

        if cropped is None:
            failed += 1
            updated_rows.append(row)
            continue

        row["pocket_path"] = str(cropped)
        row["pocket_sha256"] = _sha256_file(cropped)
        row["pocket_definition_prove"] = _pocket_defined_proof()
        updated_rows.append(row)
        extracted += 1
        print(f"[wf_a33_extract] {pdb_id}: extracted → {cropped}")

    # --- 5. Atomic CSV rewrite (§5.6) — only on success paths ---
    if extracted > 0 or failed == 0:
        _write_csv_atomic(csv_path, updated_rows)

    # --- 6. Cleanup intermediate work dir ---
    work_dir = pockets_root / ".fpocket_work"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    # --- 7. Closeout stdout (per task schema) ---
    total = len(rows)
    print(
        f"[wf_a33_extract] {total}/{total} platiny PDBs processed "
        f"({extracted} extracted, {skipped} skipped, {failed} failed)"
    )
    pocket_files = sorted(pockets_root.glob("*_pocket10.pdb"))
    print(
        f"[wf_a33_extract] {pockets_root} has {len(pocket_files)} pocket10.pdb files"
    )
    # Verify CSV round-trip
    final_rows = _read_csv_rows(csv_path)
    populated = sum(1 for r in final_rows if r.get("pocket_path", "").strip())
    print(
        f"[wf_a33_extract] CSV pocket_path populated for {populated}/{len(final_rows)} rows"
    )

    # --- 8. Exit code (§5.5 honest framing) ---
    if failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
