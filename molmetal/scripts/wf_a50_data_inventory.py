#!/usr/bin/env python
"""Regenerate the training/evaluation data inventory from disk.

Every number in ``docs/DATA_INVENTORY.md`` comes from this script; nothing is
typed by hand.  Re-run after any harvest or cohort change::

    .venv/bin/python molmetal/scripts/wf_a50_data_inventory.py

Emits ``docs/DATA_INVENTORY.md`` (human) and ``docs/DATA_INVENTORY.json``
(machine).  Read-only: it never writes into ``.data/``.

Roles
-----
``predictive_train``  D-MPNN metal-activity cohort (Krasnov-style funnel)
``generative_train``  CFM 5-metal property cohort
``eval_pt``           platinum external evaluation
``control_general``   MoleculeNet general-property controls
``control_ood``       DrugOOD out-of-distribution control
``pretrain``          geometry / structure pretraining
``reference``         published baselines, not trained on
``legacy``            superseded — do not train on
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / ".data" / "molmetal"
MD_OUT = ROOT / "docs" / "DATA_INVENTORY.md"
JSON_OUT = ROOT / "docs" / "DATA_INVENTORY.json"

# (id, path, role, status, note)
DECLARED: List[tuple] = [
    # --- predictive line -------------------------------------------------
    ("metalcytotox_raw", DATA / "MetalCytoToxDB.csv", "predictive_train", "ACTIVE",
     "Raw Krasnov corpus. Predictive cohorts are derived from this by "
     "build_paper_cohort() + collapse_duplicates(tie->active)."),
    # --- generative line -------------------------------------------------
    ("cfm_5metal_v3", DATA / "metalcytotox_5metal_multitask_v3.csv",
     "generative_train", "ACTIVE",
     "Canonical CFM cohort (configs/cfm_5metal_default.yaml). Scaffold split "
     "column is carried in-file."),
    ("cfm_5metal_v4", DATA / "metalcytotox_5metal_multitask_v4.csv",
     "generative_train", "ACTIVE",
     "Same rows as v3 plus lowercase smiles/metal aliases and source. Used by "
     "10 configs; kept because both are wired."),
    ("cfm_5metal_v2", DATA / "metalcytotox_5metal_multitask_v2.csv",
     "generative_train", "SUPERSEDED",
     "Same 26,632 rows as v3, differing only in column naming (lowercase "
     "smiles/metal vs canonical SMILES_Ligands/Metal). Use v3."),
    ("cfm_5metal_v1", DATA / "metalcytotox_5metal_multitask.csv",
     "generative_train", "SUPERSEDED", "Pre-v2 cohort; only legacy configs."),
    # --- platinum --------------------------------------------------------
    ("pt_biodata_all", DATA / "wf_a37_platinai_biodata" / "pt" /
     "platinai_pt_all_cells_with_smiles.jsonl", "eval_pt", "ACTIVE",
     "17,735 real MB-Finder Pt IC50 rows, 100% SMILES-joined (Wf-A46/47/48/49)."),
    ("pt_biodata_raw", DATA / "wf_a37_platinai_biodata" /
     "harvest_summary_2026-10-16.json", "eval_pt", "ACTIVE",
     "All-metal harvest summary: 485 cell lines / 30,185 rows."),
    ("pt_4cell_synthetic", DATA / "platinai_4cell.csv", "legacy", "DO-NOT-TRAIN",
     "SYNTHETIC pIC50 (meta says 'synthetic IC50 labels for pipeline test'). "
     "Its ~0.50 AUC is a null test, not a measurement."),
    ("pt_real_label_pool", DATA / "platinai_4cell_real_label_pool.csv",
     "legacy", "SUPERSEDED",
     "Real labels but 1/2,053 rows carry SMILES; superseded by pt_biodata_all."),
    # --- controls --------------------------------------------------------
    ("moleculenet_bace", DATA / "moleculenet" / "bace.csv",
     "control_general", "ACTIVE", "Scaffold 70/10/20, 5 seeds."),
    ("moleculenet_bbbp", DATA / "moleculenet" / "bbbp.csv",
     "control_general", "ACTIVE", "Scaffold 70/10/20, 5 seeds."),
    ("moleculenet_clintox", DATA / "moleculenet" / "clintox.csv.gz",
     "control_general", "ACTIVE", "Scaffold 70/10/20, 5 seeds."),
    ("moleculenet_hiv", DATA / "moleculenet" / "hiv.csv",
     "control_general", "ACTIVE", "Scaffold 70/10/20, 5 seeds."),
    ("nci60_9cell", ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv",
     "control_general", "ACTIVE",
     "9-cell precious-metal panel. Split is m21's Bemis-Murcko scaffold split "
     "(70/10/20, seed 0), reproduced byte-for-byte."),
    ("nci60_gi50_raw", DATA / "NCI60_GI50" / "GI50.csv", "reference", "UNUSED",
     "Raw NCI-60 GI50 dump; the 9-cell panel above is the cohort actually used."),
    ("drugood_lbap_scaffold", DATA / "drugood" / "ic50" / "csv" /
     "lbap_refined_ic50_scaffold.csv", "control_ood", "ACTIVE",
     "OOD control; iid/ood train/val/test split is a column."),
    # --- pretraining -----------------------------------------------------
    ("tmqm_dblock", ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv",
     "pretrain", "ACTIVE",
     "5,000 d-block complexes with geometry; TMQM init is OFF in the current "
     "CFM config (does not transfer to ligand-only rows)."),
    ("crossdocked100", ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv",
     "pretrain", "ACTIVE", "100-pocket docking manifest for the pocket encoder."),
    # --- reference -------------------------------------------------------
    ("rusanov_s5_ece", DATA / "rusanov_2025" / "parsed" / "s5_ece.json",
     "reference", "REFERENCE", "Published ECE values; compared against, never trained on."),
    ("rusanov_s7_pt_wet_lab", DATA / "rusanov_2025" / "parsed" /
     "s7_pt_wet_lab.json", "reference", "REFERENCE",
     "Published Pt-1..19 wet-lab panel; reference only (no wet lab per §5.4)."),
]


def _rows(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if path.suffix == ".json":
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        # Harvest summaries count their rows instead of holding them.
        for key in ("rows_total_all_cells", "n_rows", "n_total"):
            if isinstance(doc.get(key), int):
                return doc[key]
        return None
    if path.suffix == ".gz":
        return len(pd.read_csv(path, low_memory=False))
    if path.suffix in (".csv", ".xlsx"):
        try:
            return len(pd.read_csv(path, low_memory=False))
        except Exception:
            return None
    return None


def _predictive_cohorts() -> Dict[str, Any]:
    """Per-metal paper_funnel cohort, i.e. what the D-MPNN actually sees."""
    from molmetal.data.cytotox import build_paper_cohort
    from molmetal.scripts.metal_activity_transfer_probe import collapse_duplicates

    raw = pd.read_csv(DATA / "MetalCytoToxDB.csv", low_memory=False)
    out: Dict[str, Any] = {}
    for metal in ("Ru", "Ir", "Rh", "Os", "Re"):
        sub = build_paper_cohort(raw, metal=metal)
        y = (sub["IC50_Dark_value"].astype(float) < 10.0).to_numpy(dtype=int)
        uniq, uy, n_raw = collapse_duplicates(
            sub["SMILES_Ligands"].astype(str).to_numpy(), y, tie_rule="active"
        )
        out[metal] = {
            "funnel_rows": int(len(sub)),
            "unique_smiles": int(len(uniq)),
            "active": int(uy.sum()),
            "inactive": int((1 - uy).sum()),
            "rows_before_collapse": int(n_raw),
        }
    out["_total_unique"] = sum(v["unique_smiles"] for k, v in out.items() if not k.startswith("_"))
    return out


def _generative_cohort() -> Dict[str, Any]:
    df = pd.read_csv(DATA / "metalcytotox_5metal_multitask_v3.csv", low_memory=False)
    return {
        "rows": int(len(df)),
        "unique_smiles": int(df["SMILES_Ligands"].nunique()),
        "metals": {k: int(v) for k, v in df["Metal"].value_counts().items()},
        "split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "activity": {str(k): int(v) for k, v in df["activity"].value_counts().items()},
        "cell_lines": int(df["cell_line"].nunique()),
        "top_cells": {k: int(v) for k, v in df["cell_line"].value_counts().head(5).items()},
    }


def _nci60() -> Dict[str, Any]:
    df = pd.read_csv(ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv")
    return {
        "rows": int(len(df)),
        "unique_smiles": int(df["smiles"].nunique()),
        "cells": int(df["cell_line"].nunique()),
        "per_cell": {k: int(v) for k, v in df["cell_line"].value_counts().items()},
    }


def _platinum() -> Dict[str, Any]:
    pt = DATA / "wf_a37_platinai_biodata" / "pt"
    summary = json.loads((pt / "pt_only_summary.json").read_text(encoding="utf-8"))
    join = json.loads((pt / "smiles_join_summary.json").read_text(encoding="utf-8"))
    return {
        "rows": summary["pt_rows"],
        "compounds": summary["pt_unique_compounds"],
        "cells_with_pt_rows": sum(
            1 for c in summary["per_cell_line"] if c.get("rows_pt", 0) > 0
        ),
        "rows_with_smiles": join["pt_rows_with_smiles"],
        "rows_training_ready": join["pt_rows_with_value_and_smiles"],
    }


def build() -> Dict[str, Any]:
    datasets = []
    for did, path, role, status, note in DECLARED:
        rel = path.relative_to(ROOT) if path.is_absolute() else path
        datasets.append(
            {
                "id": did,
                "path": str(rel),
                "role": role,
                "status": status,
                "rows": _rows(path),
                "bytes": path.stat().st_size if path.exists() else 0,
                "note": note,
            }
        )
    return {
        "generated_by": "molmetal/scripts/wf_a50_data_inventory.py",
        "datasets": datasets,
        "predictive_cohort": _predictive_cohorts(),
        "generative_cohort": _generative_cohort(),
        "nci60": _nci60(),
        "platinum": _platinum(),
    }


def _fmt(n: int | None) -> str:
    return "—" if n is None else f"{n:,}"


def render_md(doc: Dict[str, Any]) -> str:
    p = doc["predictive_cohort"]
    g = doc["generative_cohort"]
    n = doc["nci60"]
    pt = doc["platinum"]
    lines = [
        "# Data inventory — training and evaluation",
        "",
        "> Generated by `molmetal/scripts/wf_a50_data_inventory.py`. Every number",
        "> below is measured from the files on disk; re-run the script rather than",
        "> editing this file. Machine-readable twin: `docs/DATA_INVENTORY.json`.",
        "> `STATUS` values: **ACTIVE** (in use), **SUPERSEDED** (older cohort, do",
        "> not train), **DO-NOT-TRAIN** (labels are synthetic), **UNUSED**,",
        "> **REFERENCE** (published baseline, never trained on).",
        "",
        "## 1. What each model actually trains on",
        "",
        "### 1.1 Predictive — D-MPNN metal activity",
        "",
        "Cohort = `build_paper_cohort(MetalCytoToxDB, metal)` → "
        "`collapse_duplicates(tie→active)`; labels are "
        "`IC50_Dark_value < 10 µM → active`.",
        "",
        "| Metal | Funnel rows | Unique SMILES | Active | Inactive |",
        "|---|---:|---:|---:|---:|",
    ]
    for metal in ("Ru", "Ir", "Rh", "Os", "Re"):
        c = p[metal]
        lines.append(
            f"| {metal} | {_fmt(c['funnel_rows'])} | {_fmt(c['unique_smiles'])} | "
            f"{_fmt(c['active'])} | {_fmt(c['inactive'])} |"
        )
    lines += [
        f"| **Total (per-metal sum)** | | **{_fmt(p['_total_unique'])}** | | |",
        "",
        "A SMILES can appear under more than one metal, so the per-metal sum is not",
        "a count of distinct molecules. Evaluation: `StratifiedKFold(n_splits=5)` ×",
        "seeds `[42,43,44,49,123]`, no fixed holdout — every molecule is tested in",
        "one fold.",
        "",
        "### 1.2 Generative — CFM 5-metal",
        "",
        f"- Rows **{_fmt(g['rows'])}**, unique SMILES **{_fmt(g['unique_smiles'])}**, "
        f"cell lines **{g['cell_lines']}**",
        f"- Metals: {', '.join(f'{k} {v:,}' for k, v in g['metals'].items())}",
        f"- In-file scaffold split: {', '.join(f'{k} {v:,}' for k, v in g['split'].items())}",
        f"- Activity: {', '.join(f'{k} {v:,}' for k, v in g['activity'].items())}",
        "",
        "The CFM is generative: there is no held-out property test set. Quality is",
        "measured by decoding validation clouds (validity, ring profile, novelty,",
        "docking), not by a train/test AUC.",
        "",
        "### 1.3 Platinum external evaluation",
        "",
        f"- Rows **{_fmt(pt['rows'])}**, unique compounds **{_fmt(pt['compounds'])}**, "
        f"cell lines contributing Pt rows **{pt['cells_with_pt_rows']}**",
        f"- Rows with SMILES **{_fmt(pt['rows_with_smiles'])}** / "
        f"{_fmt(pt['rows'])}; training-ready (value + SMILES) "
        f"**{_fmt(pt['rows_training_ready'])}**",
        "",
        "### 1.4 Controls",
        "",
        f"- **NCI-60 precious-metal panel**: {_fmt(n['rows'])} rows / "
        f"{_fmt(n['unique_smiles'])} unique SMILES / {n['cells']} cell lines. "
        "Split is m21's Bemis-Murcko scaffold split (70/10/20, seed 0) on unique "
        "SMILES: train 3,645 / val 521 / test 1,041.",
        "- **MoleculeNet** (bace / bbbp / clintox / hiv): scaffold 70/10/20, 5 seeds.",
        "- **DrugOOD lbap**: 266,522 rows with iid/ood train/val/test as a column; "
        "used as the OOD control, not as a headline result.",
        "",
        "## 2. File register",
        "",
        "| ID | Role | Status | Rows | Path |",
        "|---|---|---|---:|---|",
    ]
    order = [
        "predictive_train", "generative_train", "eval_pt", "control_general",
        "control_ood", "pretrain", "reference", "legacy",
    ]
    for role in order:
        for d in doc["datasets"]:
            if d["role"] != role:
                continue
            lines.append(
                f"| `{d['id']}` | {d['role']} | **{d['status']}** | "
                f"{_fmt(d['rows'])} | `{d['path']}` |"
            )
    lines += ["", "## 3. Notes per dataset", ""]
    for d in doc["datasets"]:
        lines.append(f"- **`{d['id']}`** ({d['status']}) — {d['note']}")
    lines += [
        "",
        "## 4. Do not train on these",
        "",
        "- `platinai_4cell.csv` — synthetic IC50 labels (its ~0.50 AUC is a null",
        "  test proving no leakage, not a result).",
        "- `metalcytotox_5metal_multitask{,_v2}.csv` — superseded by v3/v4.",
        "- `NCI60_GI50/GI50.csv` — raw dump; the 9-cell panel is the cohort.",
        "",
        "## 5. Known gaps",
        "",
        "- `.data/molmetal/bindingdb_ic50.csv` is a **broken symlink** to a 0-byte",
        "  file (`/home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv`); any",
        "  BindingDB path is BLOCKED until that source is restored.",
        "- The platinum set has no scaffold split yet: the 17,735 rows are a pool,",
        "  not a fixed train/val/test partition.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    doc = build()
    JSON_OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_OUT.write_text(render_md(doc), encoding="utf-8")
    print(f"wrote {MD_OUT.relative_to(ROOT)} and {JSON_OUT.relative_to(ROOT)}")
    for d in doc["datasets"]:
        print(f"  {d['id']:26s} {d['status']:13s} rows={_fmt(d['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
