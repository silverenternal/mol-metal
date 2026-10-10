#!/usr/bin/env python3
"""Build the PlatinAI 4-cell-line dataset.

=====================================================================
PlatinAI 4-cell-line dataset builder (R17 / Metric 2 anchor)
=====================================================================

Background
----------
Rusanov 2025 (PlatinAI, ChemRxiv 2025-06-20 v2,
DOI 10.26434/chemrxiv-2025-pp32k) reports per-cell-line AUC on a corpus
of **4,078 platinum complexes / 18,995 IC50 values** across 4 cell lines
(MCF-7, A549, A2780, A2780cis).  We anchor our Tier-1 metric 2 against
this paper.

The labelled training set is published as ChemRxiv supplementary
xlsx tables (S4-S5) which are behind a Cloudflare JavaScript challenge
and not directly fetchable from this host.  We instead construct a
**representative 4,078-row Pt complex dataset** with all 4 cell-line
columns, sourced from:

* `molmetal/data/platinai_5k.csv`          — 874 Pt rows
* `molmetal/data/known_pt_drugs.csv`        — 7  FDA-approved Pt drugs
* `.data/molmetal/tmQM/tmqm_parsed.csv`     — 8,275 Pt complexes with
                                              DFT HOMO/LUMO/Gap/Polarizability

After de-duplication on canonical SMILES we sub-sample (deterministic
seed=42) to **4,078 unique Pt complexes**.

IC50 label generation (HONEST framing)
---------------------------------------
The original 18,995 IC50 values are NOT directly available.  We
generate **synthetic IC50 labels** in the typical Pt-drug activity
range (0.1 µM – 100 µM, log-normal) using:

    pIC50_cell = base + β_HOMO * HOMO + β_LUMO * LUMO
                          + β_gap  * Gap  + β_P  * Polarizability
                          + cell_offset[line]   # per-cell-line shift
                          + resistance_scale * (line == "A2780cis")
                          + ε ~ N(0, σ)

⚠️ THIS DESCRIPTOR TERM IS ONLY PRESENT UNDER `--descriptor-terms`.
The default path (which produced molmetal/data/platinai_4cell.csv, the
measured provenance of metrics/by_metric/platinai_4cell_auc.json) emits
NO descriptor term at all:

    log10(IC50_nM)_cell = BASE + cell_offset[cell]
                          - RESISTANCE_BOOST * (line == "A2780cis" and n_heavy > 25)
                          + ε ~ N(0, SIGMA)

so for MCF7 / A549 / A2780 every molecule-dependent term is identically
zero and the label is a per-cell-line constant plus independent noise —
a NULL SIGNAL whose population-optimal AUC is 0.5.  Measured
corr(log10 IC50, heavy_atom_count) on the shipped CSV is -0.0095 / -0.0254
/ -0.0324 for those three lines vs -0.2161 for A2780cis.  Only A2780cis
carries a structure-dependent term in the default build.

where:
* `base` = 6.0 (geometric mean around 1 µM — typical Pt IC50)
* `cell_offset` = {MCF7: 0, A549: 0.1, A2780: 0.3, A2780cis: 0.9}
   (A2780cis is cisplatin-resistant → baseline IC50 ~10× higher)
* `resistance_scale` = 0.5 * (heavy_atom_count > 25)
   (larger Pt complexes are more likely to bypass DNA-adduct resistance)
* σ = 0.4 (real assay noise)
* The active threshold is `pIC50 ≥ 5.0` (i.e. IC50 ≤ 10 µM).

This is **synthetic, not real** — the AUC numbers we report are
useful as a *pipeline test* (verifying our 4-cell-line multi-task
D-MPNN learns per-cell-line signal) but are NOT directly comparable
to Rusanov 2025's Table S4–S5.  The honest report explicitly flags
this caveat.

Output schema
-------------
molmetal/data/platinai_4cell.csv (4,078 rows):

    complex_id, smiles, ic50_mcf7_nm, ic50_a549_nm,
                 ic50_a2780_nm, ic50_a2780cis_nm

`ic50_*_nm` is in nanomolar.  Active columns are derived in the
training script (active iff IC50 ≤ 10,000 nM = 10 µM).

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/build_platinai_4cell.py \\
        --out molmetal/data/platinai_4cell.csv \\
        --seed 42 --n-target 4078
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants — cell-line offsets derived from real Pt-drug IC50 literature
# ---------------------------------------------------------------------------
# Cell-line shift on log10(IC50_nM) scale (relative to MCF-7 baseline).
# Literature ranges:
#   MCF-7      : cisplatin IC50 ≈ 1-5 µM  (moderately sensitive)
#   A549       : cisplatin IC50 ≈ 5-10 µM (similar to MCF-7)
#   A2780      : cisplatin IC50 ≈ 1-3 µM  (sensitive)
#   A2780cis   : cisplatin IC50 ≈ 10-30 µM (cisplatin-resistant)
# We translate to pIC50 shifts: pIC50 = -log10(IC50_M).
# Cell-line shift on log10(IC50_nM) scale (relative to MCF-7 baseline).
# Real Pt-drug IC50 ranges (literature):
#   MCF-7      : cisplatin 1-5 µM, carboplatin 30-100 µM (heterogeneous)
#   A549       : cisplatin 5-10 µM (similar to MCF-7, slightly less sensitive)
#   A2780      : cisplatin 1-3 µM (sensitive)
#   A2780cis   : cisplatin 10-30 µM (cisplatin-resistant by definition)
# We choose offsets so that *overall active rate* is ~50 % per cell line
# (a learnable signal exists but no cell line is trivially easy / hard).
CELL_OFFSETS_NM: Dict[str, float] = {
    "MCF7": 0.0,  # baseline
    "A549": +0.15,  # A549 needs ~1.4× more Pt → higher IC50
    "A2780": -0.30,  # A2780 sensitive → lower IC50
    "A2780cis": +0.40,  # cisplatin-resistant → higher IC50
}
CELL_LINES: Tuple[str, ...] = ("MCF7", "A549", "A2780", "A2780cis")
BASE_LOG10_IC50_NM: float = 3.95  # ~9 µM (just below active threshold)
SIGMA: float = 0.45  # log-scale assay noise (real Pt IC50 CV ≈ 30-50 %)
RESISTANCE_BOOST: float = 0.30  # pIC50 boost for large Pt complexes in resistant line

# ---------------------------------------------------------------------------
# --descriptor-terms mode: DFT linear model coefficients
# ---------------------------------------------------------------------------
# Provenance: NOT fitted.  No real per-cell-line IC50 table is joinable to the
# tmQM DFT descriptors on this host — MetalCytoToxDB.csv has 0 Pt rows and 0
# smiles==csd_code matches against tmqm_parsed.csv (MEASURED 2026-10-11), and the
# Rusanov 2025 S4-S5 supplementary xlsx is Cloudflare-blocked.  So these are
# A-PRIORI coefficients chosen for sign and rough magnitude only.  They are
# recorded in the output metadata JSON so any downstream AUC on the descriptor
# dataset is traceable to these exact numbers.
#
# Units are z-scores against the tmQM Pt population (MEASURED from
# .data/molmetal/tmQM/tmqm_parsed.csv, metal == 'Pt', n=3598 matched to
# molmetal/data/platinai_4cell.csv on 2026-10-11; stats recomputed at build
# time by `_descriptor_zscores`, not hard-coded).
#
# Sign conventions chosen:
#   beta_gap   > 0 : a smaller HOMO-LUMO gap = easier intramolecular charge
#                   transfer = softer, more readily activated Pt(II) species =
#                   more cytotoxic => lower IC50 (lower log10_nM). Because
#                   `_generate_ic50_nm` ADDS this term to log10_nm, a POSITIVE
#                   coefficient is what makes a small gap lower the IC50. This
#                   was -0.35, which contradicted the rationale directly above
#                   and inverted the term: MEASURED oracle AUC for "is active"
#                   was 0.1950 on +desc_term and 0.8050 on -desc_term.
#   beta_homo  > 0 : a higher (less negative) HOMO is a weaker donor to the
#                   guanine N7 => weaker DNA adduct => higher IC50.
#   beta_lumo  > 0 : a higher (less negative) LUMO is a weaker acceptor =>
#                   weaker backbonding => higher IC50.
#   beta_pol   > 0 : larger polarizability tracks larger / more lipophilic
#                   complexes, which in this corpus track higher IC50
#                   (the same direction as the heavy-atom term above).
# Magnitudes give a descriptor term with MEASURED sd 0.384 log10 units against
# SIGMA=0.45 (SNR 0.85, not the ~0.45 previously claimed here), i.e. a signal a
# D-MPNN can learn but that cannot be read off a single draw.
DESCRIPTOR_TERMS: Tuple[str, ...] = ("HOMO_Energy", "LUMO_Energy", "HL_Gap", "Polarizability")
BETA_HOMO: float = 0.10
BETA_LUMO: float = 0.10
BETA_GAP: float = 0.35
BETA_POL: float = 0.05
# Descriptor-term linear predictor: BETA_HOMO*z(homo) + ... + BETA_POL*z(pol).
BETA_BY_TERM: Dict[str, float] = {
    "HOMO_Energy": BETA_HOMO,
    "LUMO_Energy": BETA_LUMO,
    "HL_Gap": BETA_GAP,
    "Polarizability": BETA_POL,
}


# ---------------------------------------------------------------------------
# SMILES canonicalisation (RDKit)
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


def _canonicalise(smiles: str) -> Optional[str]:
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        m = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if m is None or m.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _heavy_atom_count(smiles: str) -> Optional[int]:
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        m = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None
    if m is None:
        return None
    return m.GetNumHeavyAtoms()


_PT_RE = re.compile(r"\[Pt[A-Za-z+0-9-]*\]|\[pt[A-Za-z+0-9-]*\]")


def _has_pt(smiles: str) -> bool:
    if not smiles:
        return False
    return bool(_PT_RE.search(smiles))


# ---------------------------------------------------------------------------
# Source collectors
# ---------------------------------------------------------------------------
def _collect_from_platinai_5k() -> List[Tuple[str, str]]:
    """Pt rows from platinai_5k.csv (id, smiles)."""
    p = PROJECT_ROOT / "molmetal" / "data" / "platinai_5k.csv"
    if not p.is_file():
        return []
    out: List[Tuple[str, str]] = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            smi = row.get("smiles", "")
            metal = row.get("metal", "")
            if not smi or metal != "Pt":
                continue
            if not _has_pt(smi):
                continue
            out.append((row.get("id", ""), smi))
    return out


def _collect_from_known_pt_drugs() -> List[Tuple[str, str]]:
    """FDA-approved Pt drugs (name, smiles)."""
    p = PROJECT_ROOT / "molmetal" / "data" / "known_pt_drugs.csv"
    if not p.is_file():
        return []
    out: List[Tuple[str, str]] = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            smi = row.get("smiles", "")
            if not smi:
                continue
            out.append((row.get("name", ""), smi))
    return out


def _collect_from_tmqm(env_dir: Path) -> List[Tuple[str, str]]:
    """Pt rows from tmQM parsed (csd_code, smiles)."""
    p = env_dir / "tmQM" / "tmqm_parsed.csv"
    if not p.is_file():
        return []
    out: List[Tuple[str, str]] = []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("metal", "") != "Pt":
                continue
            smi = row.get("smiles", "")
            if not smi or not _has_pt(smi):
                continue
            out.append((row.get("csd_code", ""), smi))
    return out


# ---------------------------------------------------------------------------
# Synthetic IC50 generation (deterministic per row + per cell line)
# ---------------------------------------------------------------------------
def _row_seed(complex_id: str, smiles: str) -> int:
    """Stable per-row seed derived from id+smiles hash."""
    h = abs(hash((complex_id, smiles))) % (2**31 - 1)
    return int(h)


def _load_tmqm_descriptors(env_dir: Path) -> Dict[str, Dict[str, float]]:
    """csd_code -> {HOMO_Energy, LUMO_Energy, HL_Gap, Polarizability} for Pt rows.

    Reuses the existing tmQM parsed cache (same file `_collect_from_tmqm`
    already reads); no new descriptor code is introduced.
    """
    p = env_dir / "tmQM" / "tmqm_parsed.csv"
    if not p.is_file():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            if row.get("metal", "") != "Pt":
                continue
            code = row.get("csd_code", "")
            if not code:
                continue
            try:
                out[code] = {t: float(row[t]) for t in DESCRIPTOR_TERMS}
            except (TypeError, ValueError, KeyError):
                continue
    return out


def _descriptor_zscores(
    descriptors: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Z-score each descriptor across the population. Deterministic."""
    stats: Dict[str, Tuple[float, float]] = {}
    for term in DESCRIPTOR_TERMS:
        vals = [d[term] for d in descriptors.values()]
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        var = sum((v - mean) ** 2 for v in vals) / n if n else 0.0
        sd = var**0.5 or 1.0
        stats[term] = (mean, sd)
    return {
        code: {t: (d[t] - stats[t][0]) / stats[t][1] for t in DESCRIPTOR_TERMS}
        for code, d in descriptors.items()
    }


def _descriptor_linear_term(desc: Optional[Dict[str, float]]) -> float:
    """beta_HOMO*z(homo) + beta_LUMO*z(lumo) + beta_gap*z(gap) + beta_pol*z(pol).

    Returns 0.0 when a row has no tmQM descriptor record (descriptor absent =>
    no molecule-dependent contribution, never an invented one).
    """
    if not desc:
        return 0.0
    return sum(BETA_BY_TERM[t] * desc[t] for t in DESCRIPTOR_TERMS)


def _generate_ic50_nm(
    complex_id: str,
    smiles: str,
    rng_seed: int,
    n_heavy: int,
    desc: Optional[Dict[str, float]] = None,
    use_descriptor_terms: bool = False,
) -> Dict[str, float]:
    """Return {MCF7, A549, A2780, A2780cis} : IC50 in nM.

    The model:
        log10(IC50_nM)_cell = BASE + cell_offset[cell]
                              + descriptor_term   (only if use_descriptor_terms)
                              + ε
        active = IC50 ≤ 10000 nM (10 µM)

    With use_descriptor_terms=False (the historical default, and what produced
    molmetal/data/platinai_4cell.csv) the descriptor term is identically zero,
    so MCF7/A549/A2780 labels are a per-cell-line constant plus noise.

    Heavy-atom count injects a small bonus for A2780cis (large complexes
    can bypass DNA-adduct resistance) — this gives the model a learnable
    per-cell-line signal without giving away the answer.
    """
    r = random.Random(rng_seed)
    desc_term = _descriptor_linear_term(desc) if use_descriptor_terms else 0.0
    out: Dict[str, float] = {}
    for line in CELL_LINES:
        eps = r.gauss(0.0, SIGMA)
        resistance_bonus = (
            RESISTANCE_BOOST if (line == "A2780cis" and n_heavy > 25) else 0.0
        )
        # log10 nM
        log10_nm = (
            BASE_LOG10_IC50_NM
            + CELL_OFFSETS_NM[line]
            - resistance_bonus
            + desc_term
            + eps
        )
        # Convert to nM, clip to [1, 1e6] nM
        ic50_nm = 10**log10_nm
        ic50_nm = max(1.0, min(1.0e6, ic50_nm))
        out[line] = float(ic50_nm)
    return out


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PlatinAI 4-cell-line dataset")
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"),
        help="Output CSV path (default: molmetal/data/platinai_4cell.csv)",
    )
    p.add_argument(
        "--env-dir",
        default=os.environ.get(
            "MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal")
        ),
        help="MOLMETAL_DATA_DIR for tmQM lookup",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Sampling RNG seed (default 42)"
    )
    p.add_argument(
        "--n-target",
        type=int,
        default=4078,
        help="Target row count (default 4078, matching Rusanov 2025)",
    )
    p.add_argument(
        "--metadata-out",
        default=None,
        help="Optional JSON metadata output path",
    )
    p.add_argument(
        "--descriptor-terms",
        action="store_true",
        help=(
            "Use the HOMO/LUMO/gap/Polarizability linear model the module "
            "docstring promises. Off by default: the OFF path is what produced "
            "molmetal/data/platinai_4cell.csv, whose MCF7/A549/A2780 labels "
            "carry no molecule-dependent term. With this flag, rows whose "
            "source id matches a Pt row in the tmQM parsed cache get "
            "beta_HOMO*z(HOMO) + beta_LUMO*z(LUMO) + beta_gap*z(gap) + "
            "beta_pol*z(Polarizability); unmatched rows get exactly 0.0. "
            "Coefficients are A-PRIORI, not fitted (see BETA_BY_TERM)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    canonical = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
    if args.descriptor_terms and out_path.resolve() == canonical.resolve():
        print(
            "[platinai-4cell] REFUSING to write --descriptor-terms output over "
            "molmetal/data/platinai_4cell.csv. That file is the measured "
            "provenance of metrics/by_metric/platinai_4cell_auc.json. Pass a "
            "versioned --out, e.g. "
            "--out molmetal/data/platinai_4cell_desc_v1.csv",
            file=sys.stderr,
            flush=True,
        )
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env_dir = Path(args.env_dir)

    print("[platinai-4cell] Collecting Pt SMILES from sources...", flush=True)
    src_platinai = _collect_from_platinai_5k()
    src_known = _collect_from_known_pt_drugs()
    src_tmqm = _collect_from_tmqm(env_dir)
    print(
        f"[platinai-4cell]   platinai_5k Pt: {len(src_platinai)}  "
        f"known_pt_drugs: {len(src_known)}  tmQM Pt: {len(src_tmqm)}",
        flush=True,
    )

    # Canonicalise + dedup
    seen: Dict[str, str] = {}  # canon_smiles -> source_id
    sources: Dict[str, str] = {}
    for cid, smi in src_platinai + src_known + src_tmqm:
        canon = _canonicalise(smi)
        if canon is None:
            continue
        n_heavy = _heavy_atom_count(canon)
        if n_heavy is None or n_heavy < 8 or n_heavy > 80:
            continue
        if canon in seen:
            continue
        seen[canon] = cid or f"row_{len(seen)}"
        sources[canon] = cid
    print(
        f"[platinai-4cell] After canonicalisation + dedup: "
        f"{len(seen)} unique Pt complexes",
        flush=True,
    )

    if len(seen) < args.n_target:
        print(
            f"[platinai-4cell] WARNING: only {len(seen)} unique Pt complexes "
            f"available, fewer than n_target={args.n_target}. Using all.",
            flush=True,
        )

    # Deterministic sub-sample
    rng = random.Random(args.seed)
    keys = sorted(seen.keys())
    rng.shuffle(keys)
    keys = keys[: args.n_target]
    print(
        f"[platinai-4cell] Sampling {len(keys)} complexes "
        f"(target={args.n_target}, seed={args.seed})",
        flush=True,
    )

    # Build CSV
    desc_z: Dict[str, Dict[str, float]] = {}
    if args.descriptor_terms:
        raw_desc = _load_tmqm_descriptors(env_dir)
        desc_z = _descriptor_zscores(raw_desc)
        print(
            f"[platinai-4cell] descriptor mode ON: {len(desc_z)} Pt complexes "
            f"with tmQM DFT descriptors; coefficients "
            f"{ {t: BETA_BY_TERM[t] for t in DESCRIPTOR_TERMS} } (a-priori)",
            flush=True,
        )
    else:
        print(
            "[platinai-4cell] descriptor mode OFF: MCF7/A549/A2780 labels are "
            "a per-cell-line constant plus N(0, %.2f) noise (NULL SIGNAL)."
            % SIGMA,
            flush=True,
        )

    n_written = 0
    n_with_desc = 0
    n_active_per_line = {line: 0 for line in CELL_LINES}
    rows: List[Dict[str, object]] = []
    for canon in keys:
        cid = seen[canon]
        n_heavy = _heavy_atom_count(canon) or 0
        # Use the original source SMILES (we canonicalised for dedup)
        # but write the canonical one for downstream consistency.
        seed = _row_seed(cid, canon)
        src_id = sources.get(canon, "unknown")
        desc = desc_z.get(src_id)
        if desc is not None:
            n_with_desc += 1
        ic50 = _generate_ic50_nm(
            cid,
            canon,
            seed,
            n_heavy,
            desc=desc,
            use_descriptor_terms=args.descriptor_terms,
        )
        row = {
            "complex_id": f"pt4_{n_written:04d}",
            "smiles": canon,
            "ic50_mcf7_nm": round(ic50["MCF7"], 2),
            "ic50_a549_nm": round(ic50["A549"], 2),
            "ic50_a2780_nm": round(ic50["A2780"], 2),
            "ic50_a2780cis_nm": round(ic50["A2780cis"], 2),
            "source": src_id,
        }
        for line, val in ic50.items():
            if val <= 10_000.0:
                n_active_per_line[line] += 1
        rows.append(row)
        n_written += 1

    # Write CSV (deterministic order by complex_id)
    rows.sort(key=lambda r: r["complex_id"])
    fieldnames = [
        "complex_id",
        "smiles",
        "ic50_mcf7_nm",
        "ic50_a549_nm",
        "ic50_a2780_nm",
        "ic50_a2780cis_nm",
        "source",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(
        f"[platinai-4cell] Wrote {n_written} rows to {out_path}",
        flush=True,
    )

    # Per-cell-line activity rate
    print("[platinai-4cell] Per-cell-line active rate (IC50 ≤ 10 µM):")
    for line in CELL_LINES:
        n = n_active_per_line[line]
        print(
            f"[platinai-4cell]   {line}: {n}/{n_written} " f"({100*n/n_written:.1f}%)",
            flush=True,
        )

    # Metadata JSON
    meta = {
        "n_rows": n_written,
        "n_target": args.n_target,
        "seed": args.seed,
        "sources": {
            "platinai_5k_pt": len(src_platinai),
            "known_pt_drugs": len(src_known),
            "tmQM_pt": len(src_tmqm),
            "unique_after_dedup": len(seen),
        },
        "cell_lines": list(CELL_LINES),
        "cell_offsets_log10_nm": CELL_OFFSETS_NM,
        "base_log10_ic50_nm": BASE_LOG10_IC50_NM,
        "sigma": SIGMA,
        "descriptor_terms_enabled": bool(args.descriptor_terms),
        "descriptor_coefficients_apriori": (
            {t: BETA_BY_TERM[t] for t in DESCRIPTOR_TERMS}
            if args.descriptor_terms
            else None
        ),
        "descriptor_coefficient_provenance": (
            "A-PRIORI, NOT FITTED. No real per-cell-line IC50 table is joinable "
            "to the tmQM DFT descriptors on this host (MetalCytoToxDB.csv has 0 "
            "Pt rows and 0 smiles==csd_code matches against tmqm_parsed.csv; the "
            "Rusanov 2025 S4-S5 xlsx is Cloudflare-blocked). Signs and "
            "magnitudes are a modelling choice documented at BETA_BY_TERM."
            if args.descriptor_terms
            else None
        ),
        "n_rows_with_tmqm_descriptor": n_with_desc,
        "active_threshold_nm": 10000.0,
        "n_active_per_cell_line": n_active_per_line,
        "ic50_columns": [
            "ic50_mcf7_nm",
            "ic50_a549_nm",
            "ic50_a2780_nm",
            "ic50_a2780cis_nm",
        ],
        "framing": (
            "synthetic IC50 labels for pipeline test; "
            "NOT directly comparable to Rusanov 2025 Table S4-S5"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.metadata_out:
        mp = Path(args.metadata_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta, indent=2))
        print(f"[platinai-4cell] Wrote metadata to {mp}", flush=True)
    else:
        # Sidecar metadata file (handy for reproducibility)
        sidecar = out_path.with_suffix(".meta.json")
        sidecar.write_text(json.dumps(meta, indent=2))
        print(f"[platinai-4cell] Wrote metadata to {sidecar}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
