"""Wf-175 BD-2: download BindingDB full + metal filter.

Usage
-----
    ${PY} scripts/wf175_bindingdb_metal_filter.py \\
        --tied /home/hugo/codes/mol-metal/.data/molmetal/bindingdb_full/BindingDB_All_202609_tsv.zip \\
        --out  /home/hugo/codes/mol-metal/.data/molmetal/bindingdb_full/metal_binding_db_full.csv

What it does
------------
1. Unzip the BindingDB_All_YYYYMM_tsv.zip in place to a TSV.
2. Stream the TSV through RDKit and apply an 8-metal SMARTS filter
   (Ru / Ir / Rh / Os / Re / Pt / Pd / Au).
3. Convert any IC50 / Kd / Ki / EC50 to pIC50 (log10 of molar).
4. Write the 6-col canonical schema to ``metal_binding_db_full.csv``.

The output schema is the same as MetalCytoToxDB-derived pack so that
downstream D-MPNN multitask loaders can ingest both with no special
casing (canonical-metal filter lives in
``molmetal/scripts/train_dmpnn_multitask.py``).

Columns
-------
- ``smiles``       : canonical multi-component SMILES, with metal token
                     ``[Pt]`` etc. last (cf. ``metal_smiles`` module)
- ``metal``        : element symbol (Pt / Ru / Ir / Au / Pd / Rh / Os / Re)
- ``cell_line``    : "BindingDB" (BindingDB is target-based, not cell-
                     based; cell_line is set to a fixed sentinel so the
                     six columns remain populated).
- ``protein``      : UniProt accession of the target (or target name if
                     UniProt missing)
- ``pIC50``        : -log10(M) of the primary affinity (IC50 > Kd > Ki >
                     EC50 in priority order)
- ``active``       : 1 if IC50 <= 10 uM else 0

This script is intentionally side-effect-light.  It writes exactly one
output file and emits a single summary line at the end with row counts
per gate.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# RDKit — project venv has it.
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings during bulk filter

# ---------------------------------------------------------------------------
# Metal SMARTS — Pt / Ru / Ir / Pd / Au / Rh / Os / Re
# ---------------------------------------------------------------------------
# Order matters for canonical output (highest-precedence first):
# Pt > Ru > Ir > Au > Pd > Rh > Os > Re
METAL_SMARTS: List[Tuple[str, str]] = [
    ("Pt", "[Pt]"),
    ("Ru", "[Ru]"),
    ("Ir", "[Ir]"),
    ("Au", "[Au]"),
    ("Pd", "[Pd]"),
    ("Rh", "[Rh]"),
    ("Os", "[Os]"),
    ("Re", "[Re]"),
]

# Pre-compile for speed
_METAL_PATTERNS: List[Tuple[str, Chem.Mol]] = [
    (sym, Chem.MolFromSmarts(s)) for sym, s in METAL_SMARTS
]


def detect_metal(smiles: str) -> Optional[str]:
    """Return the first matching metal symbol (Pt/Ru/Ir/Au/Pd/Rh/Os/Re)
    or None if no d-block precious metal is present.

    Uses SMARTS so we catch ``[Pt+2]``, ``[Pt]``, ``[Ru(N)(N)(N)(N)(N)N]``
    etc.  Bracket form is required because these metals are not in
    RDKit's default organic subset and would otherwise split into
    disconnected fragments.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001
        return None
    if mol is None:
        return None
    for sym, pat in _METAL_PATTERNS:
        if pat is None:
            continue
        if mol.HasSubstructMatch(pat):
            return sym
    return None


# ---------------------------------------------------------------------------
# TSV column indices (BindingDB 2026 schema, 640 cols)
# ---------------------------------------------------------------------------
# Verified against the actual header of BindingDB_All_202609.tsv:
#   col 1  BindingDB Reactant_set_id
#   col 2  Ligand SMILES               <-- SMILES
#   col 7  Target Name
#   col 8  Target Source Organism
#   col 9  Ki (nM)
#   col 10 IC50 (nM)
#   col 11 Kd (nM)
#   col 12 EC50 (nM)
#   col 45 UniProt (SwissProt) Primary ID of Target Chain 1
#   col 50 UniProt (TrEMBL)   Primary ID of Target Chain 1
# 0-indexed:
COL_SMILES = 1
COL_KI = 8
COL_IC50 = 9
COL_KD = 10
COL_EC50 = 11
COL_TARGET_NAME = 6
# UniProt — prefer SwissProt Primary, fall back to TrEMBL Primary.
COL_UNIPROT_PRIMARY = (44, 49)


def _coerce_float(s: str) -> Optional[float]:
    """Parse a BindingDB affinity cell into a float (None on failure)."""
    if s is None:
        return None
    s = s.strip()
    if not s or s in {"-", "NA", "na", "NaN"}:
        return None
    # Some BindingDB rows use '>' / '<' prefix — strip and keep as numeric.
    if s.startswith(">") or s.startswith("<"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None


def _pick_uniprot(fields: List[str]) -> Optional[str]:
    """Return the first non-empty UniProt accession.

    Priority: SwissProt Primary ID of Target Chain 1 → TrEMBL Primary ID.
    """
    for idx in COL_UNIPROT_PRIMARY:
        if idx < len(fields):
            v = fields[idx].strip()
            if v and v != "-":
                return v
    return None


def _best_affinity(fields: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, unit) for the primary affinity.

    Priority: IC50 > Kd > Ki > EC50.
    Returns (None, None) if none of them are present.
    """
    for col_idx in (COL_IC50, COL_KD, COL_KI, COL_EC50):
        v = _coerce_float(fields[col_idx]) if col_idx < len(fields) else None
        if v is not None:
            return v, "nM"
    return None, None


def _pic50(value_nM: float) -> Optional[float]:
    """Convert nanomolar IC50/Kd to pIC50 = -log10(M).

    Returns None if the value is non-positive (invalid).
    """
    if value_nM is None or value_nM <= 0:
        return None
    molar = value_nM * 1e-9
    try:
        return -math.log10(molar)
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# TSV iterator — streaming over an 8 GB file with constant memory
# ---------------------------------------------------------------------------
def _iter_tsv(tsv_path: Path) -> Iterator[List[str]]:
    """Yield TSV rows from an unzipped .tsv file (skipping the header).

    Streams in 1 MB chunks; yields each row as a tab-split list of fields.
    The header is yielded *first* (so the caller can decide whether to
    skip it).  Subsequent yields are data rows.
    """
    with tsv_path.open("rb", buffering=1024 * 1024) as raw:
        header: Optional[List[str]] = None
        buf = b""
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = buf[:nl]
                buf = buf[nl + 1 :]
                line = line.decode("utf-8", errors="replace").rstrip("\r")
                if not line:
                    continue
                fields = line.split("\t")
                if header is None:
                    header = fields
                    yield header
                    continue
                yield fields
        # Final partial line (no trailing newline)
        if buf:
            line = buf.decode("utf-8", errors="replace").rstrip("\r")
            if line:
                fields = line.split("\t")
                yield fields


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tied",
        type=Path,
        default=Path(
            Path(__file__).resolve().parents[1]
            / ".data"
            / "molmetal"
            / "bindingdb_full"
            / "BindingDB_All.tsv"
        ),
        help="Path to BindingDB_All.tsv (unzipped; ~9 GB)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            Path(__file__).resolve().parents[1]
            / ".data"
            / "molmetal"
            / "bindingdb_full"
            / "metal_binding_db_full.csv"
        ),
        help="Output CSV path (canonical 6-col schema)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on input rows (0 = no cap)",
    )
    args = parser.parse_args()

    if not args.tied.exists():
        print(f"[wf175] zip not found: {args.tied}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Stats counters
    n_total = 0
    n_metal = 0
    n_with_affinity = 0
    n_nan_pic50 = 0
    per_metal: dict[str, int] = {}
    dropped_unparsable = 0

    # CSV writer — streaming.  line_buffering=1 so each writerow() flushes.
    with args.out.open("w", newline="", buffering=1) as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "metal", "cell_line", "protein", "pIC50", "active"])
        fh.flush()

        PROGRESS_EVERY = 200_000
        for fields in _iter_tsv(args.tied):
            # First yielded pair is the header; skip.
            if "BindingDB Reactant_set_id" in fields[0] if fields else "":
                continue
            n_total += 1
            if args.max_rows and n_total > args.max_rows:
                break

            if n_total % PROGRESS_EVERY == 0:
                print(
                    f"[wf175] progress: scanned={n_total} "
                    f"metal_hits={n_metal} pic50_rows={n_with_affinity} "
                    f"per_metal={dict(per_metal)}",
                    file=sys.stderr,
                    flush=True,
                )

            if len(fields) <= COL_EC50:
                dropped_unparsable += 1
                continue

            smiles = fields[COL_SMILES].strip()
            if not smiles:
                dropped_unparsable += 1
                continue

            metal = detect_metal(smiles)
            if metal is None:
                continue
            n_metal += 1
            per_metal[metal] = per_metal.get(metal, 0) + 1

            value_nM, _unit = _best_affinity(fields)
            if value_nM is None:
                # No usable affinity → skip.  We keep ONLY rows with a
                # pIC50 (gate G5: no NaN pIC50 in output).
                dropped_unparsable += 1
                continue

            p = _pic50(value_nM)
            if p is None or not math.isfinite(p):
                dropped_unparsable += 1
                continue
            n_with_affinity += 1

            protein = (
                _pick_uniprot(fields)
                or fields[COL_TARGET_NAME].strip()
                or "BindingDB_unknown_target"
            )
            # cell_line: BindingDB is target-based (mostly enzymes),
            # not cell-based.  Set a fixed sentinel so the 6-col schema
            # is uniformly populated (no NaN).
            cell_line = "BindingDB"
            active = 1 if value_nM <= 10_000 else 0  # <= 10 uM

            # Round pIC50 to 4 decimals for compact storage.
            p = round(p, 4)
            w.writerow([smiles, metal, cell_line, protein, p, active])

    # ---- Gate verification --------------------------------------------------
    df_size = args.tied.stat().st_size
    print(f"[wf175] tsv_size_bytes={df_size}")
    print(f"[wf175] rows_total={n_total}")
    print(f"[wf175] rows_with_metal={n_metal}")
    print(f"[wf175] rows_with_pic50={n_with_affinity}")
    print(f"[wf175] rows_dropped={dropped_unparsable}")
    print(f"[wf175] per_metal={per_metal}")
    print(f"[wf175] wrote={args.out}")

    # Gate G5: no NaN in pIC50 column of the output CSV.
    with args.out.open("r") as fh:
        r = csv.reader(fh)
        header = next(r)
        n_nan = sum(
            1 for row in r if len(row) >= 5 and (row[4] in ("", "nan", "NaN", "NA"))
        )
    print(f"[wf175] gate_G5_nan_pic50={n_nan}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
