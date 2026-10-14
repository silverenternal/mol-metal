"""Wf-A37-fixup-1: COMPLETE REAL-label pool builder.

Wf-A37-4 KNNEvaluator + Wf-A37-5/A38 harvest together drop the
synthetic-label fallback (Rusanov 2025 S4-S5 behind a Cloudflare JS
challenge) for a REAL per-cell-line MB-Finder (Pt + other d-block) pool.

Inputs (read on disk; all REAL MB-Finder rows):

  1. New harvest CSVs / JSONLs — A37 orchestrator #1
       .data/molmetal/wf_a37_platinai_biodata/platinai_*.csv|.jsonl
       -> compound_code, cell_line, value_mean, value_sd, doi, method,
          incubationTime (REAL MB-Finder per-cell-line measured IC50
          via get-ceil-line-bio-data).
  2. FULL harvest JSONLs — A38 orchestrator (long-running, 4-child
     parallel, throttling-tolerant backoff)
       .data/molmetal/platinai_biodata_full/platinai_<cell>_biodata.jsonl
       -> imageId (= compound_code), cell_line (long form), value
          (= numeric_value string e.g. "37.7" or "1.7 ± 0.3"),
          measurements[], doi, method, incubationTime.  Schema shim
          converts these to the standard {compound_code,
          cell_line_short, value_mean, value_sd, ...} row shape and
          merges with input #1 (input #1 wins on conflict because
          it's the deterministic CSV).
  3. New harvest Pt substances JSONL (smiles join table, when present)
       .data/molmetal/wf_a37_platinai_biodata/platinai_pt_substances_smiles.jsonl
       -> compound_code -> smiles (REAL MB-Finder substance registry
       via get-substances, populated by Wf-A37-5/A38 orchestrator after
       per-cell-line biodata pages complete).
  4. Older structure index (REAL, Sep 29, 2026)
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

Wf-A38 addition (2026-10-15): per the user directive "要找完整的,不要搞
什么honest negative", the COMPLETE harvest must be on disk, not a
partial subset.  This builder now also reads `.data/molmetal/
platinai_biodata_full/` (input #2 above) — the long-running 4-child
parallel harvest with throttling-tolerant backoff.  Re-runs of this
script pick up incremental rows as the harvest continues, so the pool
grows monotonically toward the COMPLETE count as pages complete.  The
CSV+meta path is unchanged so KNNEvaluator automatically picks up the
COMPLETE pool — no consumer changes required.

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
# Wf-A38: long-running FULL harvest (4-child parallel, throttling-tolerant)
FULL_HARVEST_DIR: str = ".data/molmetal/platinai_biodata_full"
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


def _load_full_harvest_biodata() -> List[dict]:
    """Wf-A38: read the long-running FULL harvest (4-child parallel,
    throttling-tolerant backoff) and schema-shim into the standard
    {compound_code, cell_line_long, cell_line_short, value_mean, ...}
    row shape.

    The full-harvest JSONL schema differs from input #1:
        imageId           -> compound_code (MB-Pt-XXXXXX, identical
                              namespace to input #1's compound_code)
        cell_line         -> long form (e.g. "A2780"); we add the
                              lowercase short form
        value (str)       -> "37.7" or "1.7 ± 0.3"; parsed via the
                              shared ``parse_measurement_value`` helper
        measurements[]    -> optional alternate timings; we use the
                              top-level ``value`` because the orchestrator
                              normalises it (verified 2026-10-15)
        doi, method, incubationTime: pass through unchanged.

    Returns a flat list of dicts with the same schema as
    ``_load_new_biodata()``.  Dedup happens in ``_merge_biodata()``.
    """
    out: List[dict] = []
    biopath = Path(FULL_HARVEST_DIR)
    if not biopath.is_dir():
        return out
    # Local import — soft dep, parser lives in the existing loader
    from molmetal.scripts.platinai_real_label_loader import (  # noqa: E402
        parse_measurement_value,
    )

    long_to_short = {long_: long_.lower() for long_ in TARGET_CELL_LONG}
    short_to_long = {v: k for k, v in long_to_short.items()}

    for jsonl_path in sorted(biopath.glob("platinai_*_biodata.jsonl")):
        # Parse the cell from filename: platinai_a2780_biodata.jsonl -> "a2780"
        stem = jsonl_path.stem  # "platinai_a2780_biodata"
        cl_short = stem.replace("platinai_", "").replace("_biodata", "")
        cell_long = short_to_long.get(cl_short, cl_short.upper())
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = (rec.get("imageId") or "").strip()
                if not code or not _is_pt_only(code):
                    continue
                # Prefer top-level `value` (orchestrator-normalised);
                # fall back to measurements[0].numeric_value if value
                # is missing (defensive — observed values 37.7, "1.2 ± 0.3")
                raw_value = rec.get("value") or ""
                if not raw_value and rec.get("measurements"):
                    raw_value = (
                        rec["measurements"][0].get("numeric_value") or ""
                    )
                value_mean, value_sd = parse_measurement_value(raw_value)
                out.append(
                    {
                        "compound_code": code,
                        "cell_line_long": cell_long,
                        "cell_line_short": cl_short,
                        "value_mean": value_mean,
                        "value_sd": value_sd,
                        "doi": rec.get("doi", "") or "",
                        "method": rec.get("method", "") or "",
                        "incubationTime": rec.get("incubationTime", "") or "",
                        "label_source": "mb-finder/get-ceil-line-bio-data",
                    }
                )
    return out


def _merge_biodata(
    primary: List[dict], secondary: List[dict]
) -> Tuple[List[dict], int, int]:
    """Merge two biodata row lists, dedup by (compound_code, cell_line_short)
    with ``primary`` winning on conflict (primary's deterministic CSV row
    is preferred).

    Returns (merged_rows, n_primary_kept, n_secondary_added).
    """
    seen: Set[Tuple[str, str]] = set()
    merged: List[dict] = []
    n_primary = 0
    for rec in primary:
        key = (rec["compound_code"], rec["cell_line_short"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)
        n_primary += 1
    n_secondary = 0
    for rec in secondary:
        key = (rec["compound_code"], rec["cell_line_short"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)
        n_secondary += 1
    return merged, n_primary, n_secondary


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

    Wf-A38: this now merges TWO biodata sources with ``primary``
    (input #1 — A37 orchestrator CSV-deterministic) winning on conflict:
        primary   = ``_load_new_biodata()``     (Wf-A37-5)
        secondary = ``_load_full_harvest_biodata()`` (Wf-A38 long-running)
    """
    primary = _load_new_biodata()
    secondary = _load_full_harvest_biodata()
    biodata, n_primary_kept, n_secondary_added = _merge_biodata(
        primary, secondary
    )
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
        "n_new_biodata_rows": len(primary),
        "n_full_harvest_rows": len(secondary),
        "n_primary_kept": n_primary_kept,
        "n_secondary_added": n_secondary_added,
        "n_merged_unique_rows": len(biodata),
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
    """Write the merged REAL pool CSV + meta JSON.

    Uses ``wf386a3_atomic_write.atomic_write_file`` for both writes per
    AGENTS.md §5.6 (atomic_write protocol) so a re-run mid-stream
    cannot leave a half-written file as evidence.
    """
    from molmetal.scripts.wf386a3_atomic_write import (  # noqa: E402
        atomic_write_file,
    )

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    import io as _io
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_file(Path(csv_path), buf.getvalue())
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
            "full_harvest_dir": FULL_HARVEST_DIR,
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
    # Non-fancy ISO timestamp (no extra dep).
    import datetime as _dt

    meta["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    atomic_write_file(
        Path(meta_path), json.dumps(meta, indent=2, ensure_ascii=False)
    )
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
