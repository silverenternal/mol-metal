"""Wf-A37-fixup-1: COMPLETE REAL-label pool builder.

Wf-A37-4 KNNEvaluator + Wf-A37-5/A38 harvest together drop the
synthetic-label fallback (Rusanov 2025 S4-S5 behind a Cloudflare JS
challenge) for a REAL per-cell-line MB-Finder (Pt + other d-block) pool.

Inputs (read on disk; all REAL MB-Finder rows):

  1. New harvest CSVs / JSONLs
       .data/molmetal/wf_a37_platinai_biodata/platinai_*.csv|.jsonl
       -> compound_code, cell_line, value_mean, value_sd, doi, method,
          incubationTime (REAL MB-Finder per-cell-line measured IC50
          via get-ceil-line-bio-data).
  2. New harvest Pt substances JSONL (smiles join table, when present)
       .data/molmetal/platinai_biodata_full/.../platinai_pt_substances_smiles.jsonl
       -> compound_code -> smiles (REAL MB-Finder substance registry
       via get-substances, populated by Wf-A37-5/A38 orchestrator after
       per-cell-line biodata pages complete).
  3. Older structure index (REAL, Sep 29, 2026)
       .data/molmetal/platinai_4cell_mbfinder.csv
       -> mb_id, imageId, smiles, has_<cell> coverage booleans
       (real MB-Finder Pt compounds with SMILES, no per-cell-line IC50).

Output:

  .data/molmetal/platinai_4cell_real_label_pool.csv
     + .data/molmetal/platinai_4cell_real_label_pool.meta.json

Columns: compound_code, canonical_smiles, cell_line, value_mean,
value_sd, doi, method, incubationTime, label_source.

This is the REAL labelled pool file that KNNEvaluator defaults to when
present (cfm_knn_oracle_smoke.json source), replacing the synthetic
platinai_4cell.csv that was the A37-4 honest-negative fallback.

Per AGENTS.md §1 + §5.2 honest framing: every row's value_mean comes
from REAL MB-Finder per-cell-line measured data (with DOI + method
captured). SMILES comes from get-substances when available; compounds
that pre-date the SMILES index carry an empty canonical_smiles field
(their compound_code + value_mean survive into the JSON provenance but
they're not joinable to a Tanimoto fingerprint unless SMILES is
recovered). The merged pool is Tanimoto-joinable for any compound with
a non-empty canonical_smiles.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# REAL (post-Wf-A37-5/A38) inputs
NEW_BIODATA_DIR: str = ".data/molmetal/wf_a37_platinai_biodata"
NEW_SUBSTANCES_JSONL: str = (
    ".data/molmetal/wf_a37_platinai_biodata/platinai_pt_substances_smiles.jsonl"
)
# Older REAL structure index (Sep 29) — when get-substances join hasn't run
OLDER_STRUCTURE_INDEX: str = ".data/molmetal/platinai_4cell_mbfinder.csv"

# Output — REAL labelled pool that replaces .data/molmetal/platinai_4cell.csv
REAL_POOL_CSV: str = ".data/molmetal/platinai_4cell_real_label_pool.csv"
REAL_POOL_META: str = ".data/molmetal/platinai_4cell_real_label_pool.meta.json"

TARGET_CELL_SHORT = ("mcf7", "a549", "a2780", "a2780cis")
TARGET_CELL_LONG = ("MCF7", "A549", "A2780", "A2780cis")

# Empty filter for "non-Pt" compounds (the harvest may surface Fe-Pt or other
# mixed-metal rows that we shouldn't load into a Pt-only oracle pool).
NON_PT_PREFIXES = ("MB-Fe-", "MB-Au-Fe", "MB-Ir-Fe", "MB-Ni-")


def _try_canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit-canonical SMILES, or None on parse failure."""
    if not smiles:
        return None
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _is_pt_only(compound_code: str) -> bool:
    if not compound_code:
        return False
    for pre in NON_PT_PREFIXES:
        if compound_code.startswith(pre):
            return False
    return compound_code.startswith("MB-Pt-")


def _load_new_biodata() -> List[dict]:
    """Read per-cell-line biodata CSVs + JSONLs in ``NEW_BIODATA_DIR``.

    Returns a flat list of ``{compound_code, cell_line, value_mean, ...}``
    rows.  Reads both the CSV (deterministic, dedup'd) and the JSONL
    (resume-state, all pages) and merges them with CSV winning for
    dedup'd (compound_code, cell_line) keys.
    """
    out: List[dict] = []
    seen: Set[Tuple[str, str]] = set()
    biopath = Path(NEW_BIODATA_DIR)
    if not biopath.is_dir():
        return out
    for cell_long in TARGET_CELL_LONG:
        cl = cell_long.lower()
        csv_path = biopath / f"platinai_{cl}_biodata.csv"
        jsonl_path = biopath / f"platinai_{cl}_biodata.jsonl"
        # CSV first (deterministic), then JSONL (fills remaining gaps).
        if csv_path.is_file():
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = row.get("compound_code", "").strip()
                    if not code or not _is_pt_only(code):
                        continue
                    val = row.get("value_mean")
                    try:
                        val_f = float(val) if val not in (None, "") else None
                    except ValueError:
                        val_f = None
                    sd = row.get("value_sd")
                    try:
                        sd_f = float(sd) if sd not in (None, "") else None
                    except ValueError:
                        sd_f = None
                    key = (code, cl)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {
                            "compound_code": code,
                            "cell_line_long": cell_long,
                            "cell_line_short": cl,
                            "value_mean": val_f,
                            "value_sd": sd_f,
                            "doi": row.get("doi", ""),
                            "method": row.get("method", ""),
                            "incubationTime": row.get("incubationTime", ""),
                            "label_source": "mb-finder/get-ceil-line-bio-data",
                        }
                    )
        if jsonl_path.is_file():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    code = (rec.get("compound_code") or "").strip()
                    if not code or not _is_pt_only(code):
                        continue
                    key = (code, cl)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {
                            "compound_code": code,
                            "cell_line_long": cell_long,
                            "cell_line_short": cl,
                            "value_mean": rec.get("value_mean"),
                            "value_sd": rec.get("value_sd"),
                            "doi": rec.get("doi", ""),
                            "method": rec.get("method", ""),
                            "incubationTime": rec.get("incubationTime", ""),
                            "label_source": "mb-finder/get-ceil-line-bio-data",
                        }
                    )
    return out


def _load_substances_smiles() -> Dict[str, str]:
    """Read the Pt-substances SMILES join table if Wf-A37-5/A38 has emitted it.

    Returns ``{compound_code: smiles}`` — empty dict when the file is
    not yet present (harvest still running) or empty (no rows).  When
    the get-substances JSONL grows, this dict grows; the merged pool
    retains backward compatibility by keeping the JSONL compound_code
    records and only adding the canonical_smiles field when joinable.
    """
    out: Dict[str, str] = {}
    path = Path(NEW_SUBSTANCES_JSONL)
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = (rec.get("compound_code") or "").strip()
            smi = (rec.get("smiles") or "").strip()
            if code and smi:
                out[code] = smi
    return out


def _load_older_structure_index() -> Dict[str, str]:
    """Read the Sep-29 REAL structure index (100 Pt compounds WITH SMILES).

    Used as a fallback SMILES source for compound codes that aren't yet
    covered by the Wf-A37-5/A38 get-substances pass.  The older file's
    compound range (MB-Pt-002120..MB-Pt-002305) overlaps only weakly
    with the new harvest range (MB-Pt-000006..MB-Pt-000046), but we
    still load it for completeness — empty smiles go through.
    """
    out: Dict[str, str] = {}
    path = Path(OLDER_STRUCTURE_INDEX)
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("imageId") or "").strip()
            smi = (row.get("smiles") or "").strip()
            if code and smi:
                out[code] = smi
    return out


def build_real_pool() -> Tuple[List[dict], Dict[str, int]]:
    """Build the merged REAL pool.

    Returns (rows, counts).  ``rows`` is a flat list of dicts; callers
    write them to ``REAL_POOL_CSV``.  Counts surfaces the provenance:
    n_new_biodata_rows, n_with_substance_smiles, n_with_older_index_smiles,
    n_distinct_compound_codes, n_distinct_compound_codes_with_smiles.
    """
    biodata = _load_new_biodata()
    substances = _load_substances_smiles()
    older = _load_older_structure_index()

    # Join: prefer fresh SMILES (newer) over older index.
    n_with_substance = 0
    n_with_older = 0
    n_no_smiles = 0
    seen_codes: Set[str] = set()
    codes_with_smiles: Set[str] = set()

    rows: List[dict] = []
    for rec in biodata:
        code = rec["compound_code"]
        smi = substances.get(code) or older.get(code) or ""
        if smi:
            canon = _try_canonical_smiles(smi) or smi
            if code in substances:
                n_with_substance += 1
            elif code in older:
                n_with_older += 1
            codes_with_smiles.add(code)
        else:
            canon = ""
            n_no_smiles += 1
        seen_codes.add(code)
        out_rec = dict(rec)
        out_rec["canonical_smiles"] = canon
        rows.append(out_rec)

    counts = {
        "n_new_biodata_rows": len(biodata),
        "n_with_substance_smiles": n_with_substance,
        "n_with_older_index_smiles": n_with_older,
        "n_with_no_smiles": n_no_smiles,
        "n_distinct_compound_codes": len(seen_codes),
        "n_distinct_compound_codes_with_smiles": len(codes_with_smiles),
        "n_substance_jsonl_present": 1 if substances else 0,
        "n_substance_jsonl_rows": len(substances),
        "n_older_index_rows": len(older),
    }
    return rows, counts


_FIELDNAMES = [
    "compound_code",
    "canonical_smiles",
    "cell_line_long",
    "cell_line_short",
    "value_mean",
    "value_sd",
    "doi",
    "method",
    "incubationTime",
    "label_source",
]


def write_real_pool(
    rows: List[dict],
    counts: Dict[str, int],
    csv_path: str = REAL_POOL_CSV,
    meta_path: str = REAL_POOL_META,
) -> Dict[str, str]:
    """Write the merged REAL pool CSV + meta JSON."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    per_cell = defaultdict(int)
    for rec in rows:
        per_cell[rec["cell_line_short"]] += 1
    meta = {
        "n_rows": len(rows),
        "counts": counts,
        "per_cell_count": dict(per_cell),
        "fields": list(_FIELDNAMES),
        "source_files": {
            "new_biodata_dir": NEW_BIODATA_DIR,
            "new_substances_jsonl": NEW_SUBSTANCES_JSONL,
            "older_structure_index": OLDER_STRUCTURE_INDEX,
        },
        "endpoint_provenance": {
            "numeric_values": "POST /api/v2/get-ceil-line-bio-data",
            "smiles_join": "POST /api/v2/get-substances",
            "older_smiles": "POST /api/v2/get-substances (Wf-295B Sep 29 seed)",
        },
        "framing": (
            "REAL MB-Finder per-cell-line measured IC50 values (REAL compound_code "
            "+ REAL value_mean + REAL doi + REAL method + REAL incubationTime). "
            "SMILES join is REAL whenever get-substances or the older structure "
            "index carries the compound_code; otherwise canonical_smiles is empty "
            "and the row's value_mean stays queryable by compound_code but not "
            "by Tanimoto fingerprint. Re-runs read whatever the Wf-A37-5/A38 "
            "harvest has written on disk; growing get-substances JSONL lifts "
            "canonical_smiles coverage without any code changes."
        ),
        "timestamp": json.dumps({"iso": None})[10:-2] or None,
    }
    # Use a non-fancy ISO timestamp inline (no extra dep).
    import datetime as _dt

    meta["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"csv": csv_path, "meta": meta_path}


def main() -> Dict[str, object]:
    rows, counts = build_real_pool()
    paths = write_real_pool(rows, counts)
    out = {
        "csv": paths["csv"],
        "meta": paths["meta"],
        "n_rows": len(rows),
        "counts": counts,
    }
    print(
        f"REAL pool rows={len(rows)} "
        f"distinct_codes={counts['n_distinct_compound_codes']} "
        f"with_smiles={counts['n_distinct_compound_codes_with_smiles']} "
        f"(substance_join={counts['n_with_substance_smiles']} "
        f"older_index_join={counts['n_with_older_index_smiles']} "
        f"no_smiles={counts['n_with_no_smiles']})"
    )
    print(f"wrote: {paths['csv']}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
