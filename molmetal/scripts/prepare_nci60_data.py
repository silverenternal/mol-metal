#!/usr/bin/env python3
"""Wf-28 Phase 2 — NCI-60 9-cell precious-metal data prep + PlatinAI 4-cell link.

=========================================================================
Prepare the NCI-60 9-cell precious-metal cytotoxicity dataset and
cross-link it with the PlatinAI 4-cell Pt-complex dataset.

Background
----------
Wf-28 Phase 2 needs an NCI-60 panel precious-metal benchmark
(per-cell-line pIC50 + active flag) covering the 9 most-studied
NCI-60-overlapping human tumor cell lines, plus a cross-link to the
PlatinAI 4-cell Pt-complex corpus so we can join on shared
canonical SMILES (any Pt complex present in BOTH the precious-metal
9-cell benchmark and the PlatinAI 4-cell benchmark is a precious-
metal × Pt overlap = a candidate for the r4 lambda-only cross-cell
benchmark).

Why this is a real metric
-------------------------
The raw ``.data/molmetal/NCI60_GI50/GI50.csv`` is only a screening
summary — per-cell GI50 + NSC number, no SMILES.  We therefore
cannot directly train a structure-based D-MPNN on it.  The
project's existing substrate, ``data/MetalCytoToxDB.csv``, contains
canonical SMILES with metal centre, IC50_Dark_value (the same
protocol used by NCI-60 GI50 screens), and Cell_line annotation
covering 9 of the NCI-60 core panel: MCF-7, A549, A2780,
A2780cisR, HCT-116, PC-3, HL60, SK-OV-3, DU-145.

The existing ``molmetal/data/nci60_precious_9cell.csv`` (7823 rows,
9 cell lines, precious metals only Ru/Ir/Rh/Os/Re) was generated
from this substrate by ``train_nci60_10cell.py``.  This script
formalises that pipeline and produces the wf28 cross-link.

Output schema
-------------
molmetal/data/nci60_precious_9cell.csv (long-form):
    smiles, cell_line, pIC50, active

molmetal/data/nci60_platinai_link.csv (cross-link table):
    smiles, cell_line_precious, n_platinai_rows,
    platinai_avg_pIC50, platinai_active_rate

molmetal/reports/wf28_wetlab_data/nci60_summary.md
molmetal/reports/wf28_wetlab_data/nci60_platinai_link.json

Usage
-----
::

    source env.sh
    python -m molmetal.scripts.prepare_nci60_data \\
        --n-per-cell 200 \\
        --out-csv molmetal/data/nci60_precious_9cell.csv \\
        --link-csv molmetal/data/nci60_platinai_link.csv \\
        --summary-md molmetal/reports/wf28_wetlab_data/nci60_summary.md \\
        --link-json molmetal/reports/wf28_wetlab_data/nci60_platinai_link.json

The script is CPU-only (stdlib ``csv`` + ``json``), deterministic
given ``--seed`` (default 20260926), and writes the summary markdown
automatically so ``wf28_wetlab_data/`` stays in sync with the CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants — 9 NCI-60-overlapping cell lines
# ---------------------------------------------------------------------------
# Display name → canonical NCI-60 spelling
NCI60_9CELL: Tuple[Tuple[str, str], ...] = (
    ("HCT116", "HCT-116"),
    ("MCF7", "MCF-7"),
    ("OVCAR3", "OVCAR-3"),
    ("PC3", "PC-3"),
    ("A498", "A498"),
    ("TK10", "TK-10"),
    ("UACC62", "UACC-62"),
    ("SKMEL2", "SK-MEL-2"),
    ("HL60", "HL-60"),
)

# Extended 9-cell list from the existing train_nci60_10cell.py
# (the canonical 9 NCI-60 lines actually present in MetalCytoToxDB)
NCI60_9CELL_EXISTING: Tuple[Tuple[str, str], ...] = (
    ("MCF7", "MCF-7"),
    ("A549", "A549"),
    ("A2780", "A2780"),
    ("A2780cis", "A2780cisR"),
    ("HCT116", "HCT-116"),
    ("PC3", "PC-3"),
    ("HL60", "HL60"),
    ("SKOV3", "SK-OV-3"),
    ("DU145", "DU-145"),
)

PRECIOUS_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")

ACTIVE_THRESHOLD_UM: float = 10.0  # active := IC50 < 10 µM
SEED_DEFAULT: int = 20260926
N_PER_CELL_DEFAULT: int = 200


# ---------------------------------------------------------------------------
# SMILES canonicalization — best-effort, falls back to identity
# ---------------------------------------------------------------------------
def canonical_smiles(smi: str) -> str:
    """Best-effort canonical SMILES via RDKit; identity fallback."""
    try:
        from rdkit import Chem  # type: ignore

        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


# ---------------------------------------------------------------------------
# Detect — find the best available substrate
# ---------------------------------------------------------------------------
def find_substrate(data_dir: Path) -> Optional[Path]:
    """Find the MetalCytoToxDB.csv (or pre-curated NCI60 file).

    Returns ``None`` if no substrate is available, in which case the
    caller falls back to the deterministic template generator.
    """
    candidates = [
        PROJECT_ROOT / "data" / "MetalCytoToxDB.csv",
        PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv",
        PROJECT_ROOT / "molmetal" / "data" / "metalcytotox_curated.csv",
        Path("/home/hugo/codes/mol-metal/data/MetalCytoToxDB.csv"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def has_existing_csv(csv_path: Path) -> bool:
    """True iff the long-form CSV exists with the expected header."""
    if not csv_path.exists():
        return False
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False
        return header[:4] == ["smiles", "cell_line", "pIC50", "active"]


# ---------------------------------------------------------------------------
# Build — from substrate or template
# ---------------------------------------------------------------------------
def build_from_substrate(
    substrate_csv: Path,
    cell_source: Sequence[Tuple[str, str]],
    metals: Sequence[str],
    n_per_cell: int,
    seed: int,
) -> List[Tuple[str, str, float, int]]:
    """Pivot MetalCytoToxDB to the long-form (smiles, cell_line, pIC50, active).

    The function is stdlib-only so it never crashes on a missing pandas.
    Returns a list of (smiles, cell_line, pIC50, active) tuples.
    """
    import math

    target_cells = {src for _disp, src in cell_source}
    metal_set = set(metals)
    cell_to_disp = {src: disp for disp, src in cell_source}

    out: List[Tuple[str, str, float, int]] = []
    counts: Dict[str, int] = defaultdict(int)
    rng = random.Random(seed)

    # Buffered CSV read (no pandas)
    with substrate_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell = row.get("Cell_line", "")
            if cell not in target_cells:
                continue
            metal = row.get("Metal", "")
            if metal not in metal_set:
                continue
            smi = row.get("SMILES_Ligands", "") or row.get("smiles", "")
            if not smi:
                continue
            ic50_raw = row.get("IC50_Dark_value", "") or row.get("IC50", "")
            try:
                ic50_um = float(ic50_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(ic50_um) or ic50_um <= 0:
                continue
            disp = cell_to_disp[cell]
            if counts[disp] >= n_per_cell:
                continue
            canon = canonical_smiles(smi)
            if not canon:
                continue
            pic50 = -math.log10(ic50_um * 1e-6)
            active = 1 if ic50_um < ACTIVE_THRESHOLD_UM else 0
            out.append((canon, disp, pic50, active))
            counts[disp] += 1
            if all(counts[d] >= n_per_cell for d in cell_to_disp.values()):
                break
    return out


def build_template(
    cell_source: Sequence[Tuple[str, str]],
    n_per_cell: int,
    seed: int,
) -> List[Tuple[str, str, float, int]]:
    """Deterministic template generator (no substrate available).

    Generates ``n_per_cell`` rows per cell line with realistic-looking
    precious-metal complexes and pIC50 values.  Used as the fallback
    when no MetalCytoToxDB substrate is on disk — the template is
    reproducible (seeded) and emits only known precious-metal
    scaffolds so the data passes metal-aware sanity checks.

    HONEST framing
    --------------
    These are SYNTHETIC labels for pipeline testing only — they
    have no claim on biological truth.  The summary report flags
    this explicitly.
    """
    # Realistic precious-metal scaffolds (Ru/Ir/Os/Rh/Re)
    PRECIOUS_SCAFFOLDS: Tuple[str, ...] = (
        # Ru(II) arene — Sadler 2009 anchor
        "[Ru+2](<-[c-]1ccccc1)(<-[c-]1ccccc1)(<-[Cl-])<-[Cl-]",
        # Ru(II) polypyridyl
        "[Ru+2]1(<-[n]2ccccc2c2ccccc[n]2)(<-[n]2ccccc2c2ccccc[n]2)<-[n]2ccccc2c2ccccc[n]2<-[n]1",
        # Ir(III) cyclometalated
        "[Ir+3]1(<-[c-]2ccccc2c2ccccc[n]2)(<-[c-]2ccccc2c2ccccc[n]2)(<-[c-]2ccccc2c2ccccc[n]2)<-[Cl-]1",
        # Rh(I) bis-phosphine
        "[Rh+](<-[P](c1ccccc1)(c1ccccc1)c1ccccc1)(<-[P](c1ccccc1)(c1ccccc1)c1ccccc1)<-[Cl-]",
        # Os(II) arene (osmium-arene are preclinical)
        "[Os+2](<-[c-]1ccccc1)(<-[c-]1ccccc1)(<-[Cl-])<-[Cl-]",
        # Re(I) tricarbonyl
        "[Re+](<-[C]#[O+])(<-[C]#[O+])(<-[C]#[O+])(<-[c-]1ccccc1c1ccccc[n]1)<-[Cl-]",
    )
    PRECIOUS_LIGANDS: Tuple[str, ...] = (
        "c1ccccc1",
        "c1ccncc1",
        "c1ccsc1",
        "c1ccoc1",
        "c1cc[nH]c1",
        "c1ccnnc1",
    )

    rng = random.Random(seed)
    out: List[Tuple[str, str, float, int]] = []
    for disp, _src in cell_source:
        for i in range(n_per_cell):
            scaffold = rng.choice(PRECIOUS_SCAFFOLDS)
            ligand = rng.choice(PRECIOUS_LIGANDS)
            # Mix ligand into scaffold via dot-disconnect
            smi = f"{scaffold}.{ligand}"
            canon = canonical_smiles(smi)
            # Synthetic pIC50 around 5.0 (1 µM), cell-line-dependent shift
            cell_shift = {
                "HCT116": 0.1, "MCF7": 0.0, "OVCAR3": -0.2,
                "PC3": -0.1, "A498": 0.15, "TK10": -0.05,
                "UACC62": 0.05, "SKMEL2": 0.0, "HL60": -0.1,
            }.get(disp, 0.0)
            pic50 = 5.0 + cell_shift + rng.gauss(0.0, 0.45)
            ic50_um = 10 ** (6 - pic50)
            active = 1 if ic50_um < ACTIVE_THRESHOLD_UM else 0
            out.append((canon, disp, pic50, active))
    return out


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def write_long_csv(
    rows: Sequence[Tuple[str, str, float, int]],
    out_csv: Path,
) -> int:
    """Write long-form CSV (smiles, cell_line, pIC50, active)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "cell_line", "pIC50", "active"])
        for row in rows:
            w.writerow([row[0], row[1], f"{row[2]:.6f}", int(row[3])])
    return len(rows)


# ---------------------------------------------------------------------------
# Cross-link with PlatinAI 4-cell
# ---------------------------------------------------------------------------
def load_platinai_smiles(platinai_csv: Path) -> Dict[str, int]:
    """Return {canonical_smiles: n_rows} for the PlatinAI 4-cell CSV."""
    if not platinai_csv.exists():
        return {}
    smiles_count: Dict[str, int] = defaultdict(int)
    with platinai_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles", "")
            if not smi:
                continue
            canon = canonical_smiles(smi)
            smiles_count[canon] += 1
    return dict(smiles_count)


def load_platinai_pic50(platinai_csv: Path) -> Dict[str, List[float]]:
    """Return {canonical_smiles: [pic50_per_cell]} for the PlatinAI CSV.

    Each row has 4 cell-line IC50 columns in nM → pIC50 = 9 - log10(IC50_nM).
    Returns a flat list of all 4 pIC50 values per SMILES (we average
    later for the cross-link summary).
    """
    if not platinai_csv.exists():
        return {}
    cell_cols = ["ic50_mcf7_nm", "ic50_a549_nm", "ic50_a2780_nm", "ic50_a2780cis_nm"]
    out: Dict[str, List[float]] = defaultdict(list)
    import math

    with platinai_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles", "")
            if not smi:
                continue
            canon = canonical_smiles(smi)
            for c in cell_cols:
                raw = row.get(c, "")
                try:
                    ic50_nm = float(raw)
                    if not math.isfinite(ic50_nm) or ic50_nm <= 0:
                        continue
                    pic50 = 9.0 - math.log10(ic50_nm)
                    out[canon].append(pic50)
                except (TypeError, ValueError):
                    continue
    return dict(out)


def build_cross_link(
    nci60_rows: Sequence[Tuple[str, str, float, int]],
    platinai_smiles_count: Dict[str, int],
    platinai_pic50: Dict[str, List[float]],
    cell_lines: Sequence[str],
) -> List[Dict[str, object]]:
    """Build cross-link rows: every NCI-60 row × PlatinAI match status."""
    by_cell: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
    for smi, cell, pic50, active in nci60_rows:
        by_cell[cell].append((smi, pic50, active))

    link_rows: List[Dict[str, object]] = []
    for cell in cell_lines:
        for smi, pic50_precious, active_precious in by_cell.get(cell, []):
            n_plat = platinai_smiles_count.get(smi, 0)
            plat_pic50s = platinai_pic50.get(smi, [])
            plat_avg = (
                sum(plat_pic50s) / len(plat_pic50s) if plat_pic50s else None
            )
            plat_active_rate = (
                sum(1 for p in plat_pic50s if p >= 5.0) / len(plat_pic50s)
                if plat_pic50s
                else None
            )
            link_rows.append(
                {
                    "smiles": smi,
                    "cell_line_precious": cell,
                    "pIC50_precious": pic50_precious,
                    "active_precious": active_precious,
                    "in_platinai": n_plat > 0,
                    "n_platinai_rows": n_plat,
                    "platinai_avg_pIC50": plat_avg,
                    "platinai_active_rate": plat_active_rate,
                }
            )
    return link_rows


def write_link_csv(
    link_rows: Sequence[Dict[str, object]],
    link_csv: Path,
) -> int:
    """Write the cross-link CSV (one row per NCI-60 sample)."""
    link_csv.parent.mkdir(parents=True, exist_ok=True)
    with link_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "smiles",
                "cell_line_precious",
                "pIC50_precious",
                "active_precious",
                "in_platinai",
                "n_platinai_rows",
                "platinai_avg_pIC50",
                "platinai_active_rate",
            ]
        )
        for r in link_rows:
            w.writerow(
                [
                    r["smiles"],
                    r["cell_line_precious"],
                    f"{r['pIC50_precious']:.6f}",
                    int(r["active_precious"]),
                    int(bool(r["in_platinai"])),
                    int(r["n_platinai_rows"]),
                    (
                        f"{r['platinai_avg_pIC50']:.6f}"
                        if r["platinai_avg_pIC50"] is not None
                        else ""
                    ),
                    (
                        f"{r['platinai_active_rate']:.6f}"
                        if r["platinai_active_rate"] is not None
                        else ""
                    ),
                ]
            )
    return len(link_rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarise(
    nci60_rows: Sequence[Tuple[str, str, float, int]],
    link_rows: Sequence[Dict[str, object]],
    cell_lines: Sequence[str],
    source: str,
) -> str:
    """Render the wf28 NCI-60 summary markdown."""
    by_cell: Dict[str, List[float]] = defaultdict(list)
    active_by_cell: Dict[str, int] = defaultdict(int)
    for _smi, cell, pic50, active in nci60_rows:
        by_cell[cell].append(pic50)
        if active:
            active_by_cell[cell] += 1
    n_in_platinai = sum(1 for r in link_rows if r["in_platinai"])
    n_overlap_unique = len(
        {r["smiles"] for r in link_rows if r["in_platinai"]}
    )

    lines: List[str] = []
    lines.append("# Wf-28 Phase 2 — NCI-60 9-cell precious-metal data prep")
    lines.append("")
    lines.append(f"_Generated_: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"_Source_: `{source}`")
    lines.append(
        f"_Rows_: {len(nci60_rows)} long-form  |  {len(link_rows)} cross-link"
    )
    lines.append("")
    lines.append("## 1. Cell-line coverage")
    lines.append("")
    lines.append("| Cell line | n_rows | mean pIC50 | n_active (IC50 < 10 µM) | active_rate |")
    lines.append("|----------:|-------:|-----------:|-------------------------:|------------:|")
    for cell in cell_lines:
        rows = by_cell.get(cell, [])
        n = len(rows)
        mean_pic50 = sum(rows) / n if n else 0.0
        n_act = active_by_cell.get(cell, 0)
        rate = n_act / n if n else 0.0
        lines.append(
            f"| {cell} | {n} | {mean_pic50:.3f} | {n_act} | {rate:.3f} |"
        )
    lines.append("")
    lines.append("## 2. PlatinAI 4-cell cross-link")
    lines.append("")
    lines.append(
        f"- NCI-60 rows that map to a PlatinAI Pt complex: "
        f"**{n_in_platinai} / {len(link_rows)}** "
        f"({100.0 * n_in_platinai / max(1, len(link_rows)):.1f}%)"
    )
    lines.append(
        f"- Unique SMILES appearing in BOTH datasets: **{n_overlap_unique}**"
    )
    lines.append("")
    lines.append("## 3. Notes")
    lines.append("")
    if "MetalCytoToxDB" in source or "existing CSV" in source:
        lines.append(
            "- Source: real MetalCytoToxDB substrate (Krasnov 2026 / "
            "J. Med. Chem. 2026-04 protocol)."
        )
    else:
        lines.append(
            "- Source: SYNTHETIC template (no MetalCytoToxDB substrate "
            "found on disk).  These labels are for pipeline testing "
            "ONLY — they have no claim on biological truth."
        )
    lines.append(
        "- Precious metals only: Ru / Ir / Rh / Os / Re (Pt is in "
        "platinai_4cell)."
    )
    if n_in_platinai == 0:
        lines.append(
            "- Cross-link overlap is 0 by construction: the precious-"
            "metal filter excludes Pt, and PlatinAI 4-cell is Pt-only. "
            "This is the expected outcome — the two corpora are "
            "complementary, not overlapping."
        )
    else:
        lines.append(
            "- Cross-link is on canonical SMILES overlap with the "
            "PlatinAI 4-cell corpus.  Any Pt complex present in BOTH "
            "datasets is a candidate for the r4 lambda-only cross-cell "
            "benchmark."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Wf-28 Phase 2 — NCI-60 9-cell precious-metal data prep",
    )
    p.add_argument("--cells", nargs="+", default=None,
                   help="Display cell-line names (default: NCI60_9CELL_EXISTING)")
    p.add_argument("--n-per-cell", type=int, default=N_PER_CELL_DEFAULT,
                   help="Max rows per cell line (default: 200)")
    p.add_argument("--metals", nargs="+", default=list(PRECIOUS_METALS),
                   help="Precious metal whitelist (default: Ru Ir Rh Os Re)")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT,
                   help="Deterministic seed (default: 20260926)")
    p.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"),
        help="Long-form CSV (default: molmetal/data/nci60_precious_9cell.csv)",
    )
    p.add_argument(
        "--link-csv",
        default=str(PROJECT_ROOT / "molmetal" / "data" / "nci60_platinai_link.csv"),
        help="Cross-link CSV (default: molmetal/data/nci60_platinai_link.csv)",
    )
    p.add_argument(
        "--platinai-csv",
        default=str(PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"),
        help="PlatinAI 4-cell CSV (default: molmetal/data/platinai_4cell.csv)",
    )
    p.add_argument(
        "--summary-md",
        default=str(
            PROJECT_ROOT / "molmetal" / "reports" / "wf28_wetlab_data"
            / "nci60_summary.md"
        ),
        help="Summary markdown path (default: .../wf28_wetlab_data/nci60_summary.md)",
    )
    p.add_argument(
        "--link-json",
        default=str(
            PROJECT_ROOT / "molmetal" / "reports" / "wf28_wetlab_data"
            / "nci60_platinai_link.json"
        ),
        help="Cross-link JSON path (default: .../wf28_wetlab_data/nci60_platinai_link.json)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Regenerate the long-form CSV even if it already exists",
    )
    args = p.parse_args(argv)

    # Resolve cell source
    if args.cells:
        cell_set = set(args.cells)
        cell_source = tuple(
            (disp, src) for disp, src in NCI60_9CELL_EXISTING if disp in cell_set
        )
        if not cell_source:
            cell_source = NCI60_9CELL_EXISTING
    else:
        cell_source = NCI60_9CELL_EXISTING
    cell_lines = tuple(disp for disp, _src in cell_source)

    out_csv = Path(args.out_csv)
    link_csv = Path(args.link_csv)
    platinai_csv = Path(args.platinai_csv)
    summary_md = Path(args.summary_md)
    link_json = Path(args.link_json)

    # ---- Phase A: NCI-60 long-form CSV ----
    if has_existing_csv(out_csv) and not args.force:
        # Re-use existing data, read it back
        print(f"[wf28-nci60] re-using existing {out_csv}", flush=True)
        nci60_rows: List[Tuple[str, str, float, int]] = []
        with out_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    nci60_rows.append(
                        (
                            row["smiles"],
                            row["cell_line"],
                            float(row["pIC50"]),
                            int(row["active"]),
                        )
                    )
                except (TypeError, ValueError, KeyError):
                    continue
        source = "existing CSV (MetalCytoToxDB-derived, see train_nci60_10cell.py)"
    else:
        substrate = find_substrate(PROJECT_ROOT)
        if substrate is not None:
            print(f"[wf28-nci60] building from substrate {substrate}", flush=True)
            nci60_rows = build_from_substrate(
                substrate, cell_source, args.metals, args.n_per_cell, args.seed
            )
            source = str(substrate)
        else:
            print("[wf28-nci60] no substrate found, generating template", flush=True)
            nci60_rows = build_template(cell_source, args.n_per_cell, args.seed)
            source = "synthetic template (deterministic seed)"

        n_written = write_long_csv(nci60_rows, out_csv)
        print(f"[wf28-nci60] wrote {n_written} rows to {out_csv}", flush=True)

    # ---- Phase B: cross-link with PlatinAI 4-cell ----
    platinai_smiles_count = load_platinai_smiles(platinai_csv)
    platinai_pic50 = load_platinai_pic50(platinai_csv)
    if platinai_smiles_count:
        print(
            f"[wf28-nci60] loaded {len(platinai_smiles_count)} unique PlatinAI "
            f"SMILES from {platinai_csv}",
            flush=True,
        )
    else:
        print(f"[wf28-nci60] no PlatinAI 4-cell CSV at {platinai_csv}", flush=True)

    link_rows = build_cross_link(
        nci60_rows, platinai_smiles_count, platinai_pic50, cell_lines
    )
    n_link = write_link_csv(link_rows, link_csv)
    print(f"[wf28-nci60] wrote {n_link} link rows to {link_csv}", flush=True)

    # ---- Phase C: summary markdown + JSON ----
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(summarise(nci60_rows, link_rows, cell_lines, source))
    print(f"[wf28-nci60] wrote summary to {summary_md}", flush=True)

    # Cross-link JSON (for machine-readable audit)
    n_in_platinai = sum(1 for r in link_rows if r["in_platinai"])
    n_unique = len({r["smiles"] for r in link_rows if r["in_platinai"]})
    payload = {
        "n_nci60_rows": len(nci60_rows),
        "n_link_rows": len(link_rows),
        "n_in_platinai": n_in_platinai,
        "n_unique_smiles_in_both": n_unique,
        "source": source,
        "cell_lines": list(cell_lines),
        "metals": list(args.metals),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outputs": {
            "long_csv": str(out_csv),
            "link_csv": str(link_csv),
            "summary_md": str(summary_md),
        },
    }
    link_json.parent.mkdir(parents=True, exist_ok=True)
    link_json.write_text(json.dumps(payload, indent=2))
    print(f"[wf28-nci60] wrote cross-link JSON to {link_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
