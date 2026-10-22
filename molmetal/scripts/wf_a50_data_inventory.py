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
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / ".data" / "molmetal"
MD_OUT = ROOT / "docs" / "DATA_INVENTORY.md"
JSON_OUT = ROOT / "docs" / "DATA_INVENTORY.json"
LEAKAGE_JSON = (
    ROOT / "molmetal" / "reports" / "wf_a51_leakage_audit" / "leakage_audit.json"
)

# (id, path, role, status, note)
DECLARED: List[tuple] = [
    # --- predictive line -------------------------------------------------
    ("metalcytotox_raw", DATA / "MetalCytoToxDB.csv", "predictive_train", "ACTIVE",
     "Raw Krasnov corpus. Predictive cohorts are derived from this by "
     "`build_paper_cohort()` and `collapse_duplicates(tie -> active)`."),
    # --- generative line -------------------------------------------------
    ("cfm_5metal_v3", DATA / "metalcytotox_5metal_multitask_v3.csv",
     "generative_train", "ACTIVE",
     "Canonical CFM cohort (`configs/cfm_5metal_default.yaml`). The in-file "
     "`split` column is stereo-aware and covers the whole complex; it is not a "
     "desalted ligand-scaffold split (see `docs/DATA_LEAKAGE_AUDIT.md`)."),
    ("cfm_5metal_v4", DATA / "metalcytotox_5metal_multitask_v4.csv",
     "generative_train", "ACTIVE",
     "Same rows as v3, plus lowercase `smiles`/`metal` aliases and a `source` "
     "column. Ten configs read it; kept because both are wired. Same split "
     "caveat as v3."),
    ("cfm_5metal_v5", DATA / "metalcytotox_5metal_multitask_v5.csv",
     "generative_train", "UNUSED",
     "Deduplicated rebuild (24,483 rows) with a desalted stereo-blind "
     "ligand-scaffold split: 17,138/2,448/4,897, 0 shared scaffolds and 0 "
     "shared canonical SMILES across every split pair, all five metals in all "
     "three splits. Built by `wf_a52_rebuild_cfm_split.py`. No training config "
     "reads it yet. Its test split has one consumer: "
     "`wf_a53_eval_cfm_test_split.py` scores a checkpoint on `split='test'` "
     "read-only (4,888 of 4,897 rows embed; 4,888 carry bond truth)."),
    ("cfm_5metal_v2", DATA / "metalcytotox_5metal_multitask_v2.csv",
     "generative_train", "SUPERSEDED",
     "Same 26,632 rows as v3, differing only in column naming (lowercase "
     "`smiles`/`metal` against `SMILES_Ligands`/`Metal`). Use v3."),
    ("cfm_5metal_v1", DATA / "metalcytotox_5metal_multitask.csv",
     "generative_train", "SUPERSEDED", "Pre-v2 cohort; only legacy configs."),
    # --- platinum --------------------------------------------------------
    ("pt_biodata_all", DATA / "wf_a37_platinai_biodata" / "pt" /
     "platinai_pt_all_cells_with_smiles.jsonl", "eval_pt", "ACTIVE",
     "17,735 real MB-Finder Pt IC50 rows, all SMILES-joined (Wf-A46/47/48/49). "
     "No split column; 3,668 duplicate `(code, cell)` rows and 987 exact "
     "duplicate measurements; 296 unique SMILES fail RDKit parsing."),
    ("pt_biodata_raw", DATA / "wf_a37_platinai_biodata" /
     "harvest_summary_2026-10-16.json", "eval_pt", "ACTIVE",
     "All-metal harvest summary: 485 cell lines / 30,185 rows."),
    ("pt_4cell_synthetic", DATA / "platinai_4cell.csv", "legacy", "DO-NOT-TRAIN",
     "Synthetic pIC50 (its meta file says 'synthetic IC50 labels for pipeline "
     "test'). Its ~0.50 AUC is a null test, not a measurement."),
    ("pt_real_label_pool", DATA / "platinai_4cell_real_label_pool.csv",
     "legacy", "SUPERSEDED",
     "Real labels, but only 1 of 2,053 rows carries SMILES; superseded by "
     "`pt_biodata_all`."),
    ("pt_4cell_real", DATA / "platinai_4cell_real_cohort.csv", "eval_pt", "ACTIVE",
     "Real structures and real IC50 labels for the four PlatinAI cell lines, "
     "built from the 2026-10-19 MB-Finder harvest by "
     "`build_platinai_4cell_real_cohort.py`. 1,966 compounds and 3,144 unique "
     "(compound, cell) pairs from 4,370 raw biodata rows (A549 871, A2780 860, "
     "A2780cis 600, MCF-7 813); 953 compounds cover at least two cells and 56 "
     "cover all four. Units are µM, taken from the Rusanov 2025 SI figure "
     "captions; pIC50 = 6 - log10(IC50_uM). The file holds 1,935 unique SMILES "
     "rather than 1,966, because 28 SMILES appear under 2-3 compound codes "
     "(59 rows) with per-cell labels up to 1.45 log units apart. The split "
     "groups by scaffold, so those duplicates cannot cross train/test, but the "
     "compound count overstates chemical diversity by 1.6 %."),
    ("pt_4cell_real_trainer", DATA / "platinai_4cell_real_trainer.csv",
     "eval_pt", "ACTIVE",
     "The same 1,966 compounds projected into the schema "
     "`load_platinai_4cell_split_table` accepts (`SMILES_Ligands`, "
     "`pIC50_{a549,a2780,a2780cis,mcf7}`, `metal=Pt`). No measurement is added, "
     "dropped or imputed. A compound with no IC50 for a cell line has an empty "
     "cell, which the loader turns into NaN and masks out of the loss. The file "
     "exists so the D-MPNN trainer can read real labels without a hand-edited "
     "CSV."),
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
     "9-cell precious-metal panel. The default split is the per-cell-row "
     "coverage split (`scaffold_coverage_70_10_20`, since 2026-10-21): "
     "5,371/920/1,532 rows and 3,636/548/1,023 molecules, with every cell at "
     "least 25 test and 22 val rows. The legacy `scaffold_70_10_10` protocol "
     "gives 4,860/115/232 molecules, which is 7,320/160/343 rows "
     "(93.6/2.0/4.4 %), with five of nine cells under 30 test rows; the shipped "
     "configs pin it. The panel is a subset of MetalCytoToxDB (5,207/5,207 "
     "canonical molecules). The older 3,645/521/1,041 figure is the pre-BLK-PA12 "
     "molecule-level split and no longer reproduces. See section 5."),
    ("nci60_gi50_raw", DATA / "NCI60_GI50" / "GI50.csv", "reference", "UNUSED",
     "Raw NCI-60 GI50 dump; the 9-cell panel is the cohort in use."),
    ("drugood_lbap_scaffold", DATA / "drugood" / "ic50" / "csv" /
     "lbap_refined_ic50_scaffold.csv", "control_ood", "ACTIVE",
     "OOD control; iid/ood train/val/test split is a column."),
    ("drugood_lbap_assay", DATA / "drugood" / "ic50" / "csv" /
     "lbap_refined_ic50_assay.csv", "control_ood", "ACTIVE",
     "The same 266,522 LBAP records as the scaffold split, partitioned by "
     "assay. Read by the wf182/wf174 DrugOOD runs. It had no row in this "
     "inventory, so it carried no lifecycle or label provenance in "
     "`registry/datasets.yaml`."),
    ("drugood_lbap_size", DATA / "drugood" / "ic50" / "csv" /
     "lbap_refined_ic50_size.csv", "control_ood", "ACTIVE",
     "The same 266,522 LBAP records as the scaffold split, partitioned by "
     "size. Same registry gap as `drugood_lbap_assay`."),
    # --- pretraining -----------------------------------------------------
    ("metallo_drugs_500", ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv",
     "pretrain", "ACTIVE",
     "449 metal-drug SMILES with a `source` column and no activity column. A "
     "structure pool for `metallo_pool_scaffold_split.py`, "
     "`r10_cfg_real_crossdocked.py` and `ablation_runner.py`; not a labelled "
     "cohort."),
    # --- pretraining -----------------------------------------------------
    ("tmqm_dblock", ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv",
     "pretrain", "ACTIVE",
     "5,000 d-block complexes with geometry. A small subset, not the CFM "
     "corpus, which is `tmqm_corpus_6metal` below. The tmQM-pretrained encoder "
     "(`models/tmqm_pretrained/tmqm_pretrained_encoder.pt`) is loaded by the "
     "current CFM configs (`use_tmqm_init: true`; 47/51 parameters transfer, "
     "0 unexpected, 4 missing). Before 2026-10-20 `cfm.yaml` alone omitted the "
     "key, so production trained from random init while nine ablations "
     "warm-started; see `TODO/risks.md` R6."),
    ("tmqm_corpus_6metal", DATA / "tmQM" / "tmqm_parsed.csv",
     "generative_train", "ACTIVE",
     "The tmQM corpus the CFM geometry path trains on. The file holds 108,541 "
     "complexes over the whole d block; the CFM cohorts are the six-metal "
     "subset, 33,429 complexes over Pt, Ru, Ir, Rh, Os and Re. Every one of "
     "those carries all three shards a cloud needs (geometry, Wiberg bond "
     "orders, NBO charges): 33,429/33,429 on each. No duplicate CSD codes, no "
     "non-finite coordinates, one metal atom per complex, and the `metal` "
     "column agrees with that atom in 33,429/33,429 rows. 2,277 rows (6.8 %) "
     "have an empty SMILES and fall back to a CSD-prefix scaffold key. Two "
     "protocol values, not corpus defects: the scaffold split is 87.7/6.2/6.2 "
     "% of rows rather than 80/10/10 (fractions are budgeted in scaffold "
     "counts, so `tmqm_val_frac` has no effect on that branch), and the default "
     "`n_val=192` cohort with `tmqm_min_per_metal=0` is Pt 19 / Ru 93 against a "
     "corpus that is Pt 24.8 / Ru 29.3. Source: "
     "`metrics/by_metric/tmqm_cfm_data_audit.json`."),
    ("crossdocked100", ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv",
     "pretrain", "ACTIVE", "100-pocket docking manifest for the pocket encoder."),
    # --- reference -------------------------------------------------------
    ("rusanov_s5_ece", DATA / "rusanov_2025" / "parsed" / "s5_ece.json",
     "reference", "REFERENCE", "Published ECE values; compared against, never trained on."),
    ("rusanov_s7_pt_wet_lab", DATA / "rusanov_2025" / "parsed" /
     "s7_pt_wet_lab.json", "reference", "REFERENCE",
     "Published Pt-1..19 wet-lab panel; reference only (§5.4 bars wet-lab "
     "work)."),
]


def _leakage() -> Dict[str, Any] | None:
    """Measured leakage audit (Wf-A51), if it has been run.

    The inventory must stay regenerable without the audit; a missing file is
    reported in the markdown rather than treated as an error.
    """
    if not LEAKAGE_JSON.exists():
        return None
    return json.loads(LEAKAGE_JSON.read_text(encoding="utf-8"))


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
        "leakage": _leakage(),
    }


def _fmt(n: int | None) -> str:
    return "—" if n is None else f"{n:,}"


def _wrap(text: str, indent: str = "", width: int = 78) -> List[str]:
    """Wrap prose to a fixed column without splitting inline code spans."""
    spans = re.findall(r"`[^`]+`", text)
    order = iter(range(len(spans)))

    def _stash(m: "re.Match") -> str:
        token = f"\x01{next(order)}\x02"
        return token + "\x00" * max(0, len(m.group(0)) - len(token))

    protected = re.sub(r"`[^`]+`", _stash, text)
    filled = textwrap.fill(
        protected,
        width=width,
        initial_indent=indent,
        subsequent_indent=" " * len(indent),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return re.sub(
        r"\x01(\d+)\x02\x00*", lambda m: spans[int(m.group(1))], filled
    ).splitlines()


def render_md(doc: Dict[str, Any]) -> str:
    p = doc["predictive_cohort"]
    g = doc["generative_cohort"]
    n = doc["nci60"]
    pt = doc["platinum"]
    lk = doc.get("leakage") or {}

    # Compact measured-leakage values (absent if wf_a51 has not been run).
    if lk:
        pm = lk["predictive"]["metals"]
        pred_skf_scaf = [
            r["mean_frac_test_rows_scaffold_in_train"]
            for rec in pm.values()
            for r in rec["stratified_kfold"]["per_seed"]
        ]
        pred_skf_min = 100.0 * min(pred_skf_scaf)
        pred_skf_max = 100.0 * max(pred_skf_scaf)
        pred_doi = [
            rec["doi_grouped_10fold"]["mean_frac_test_rows_scaffold_in_train"]
            for rec in pm.values()
        ]
        pred_doi_min = 100.0 * min(pred_doi)
        pred_doi_max = 100.0 * max(pred_doi)
        cfm = lk["cfm"]["v3"]
        cfm_test = cfm["pairs"]["train_vs_test"]["bemis_murcko"]
        cfm_val = cfm["pairs"]["train_vs_val"]["bemis_murcko"]
        cfm_full_ns = cfm["pairs"]["train_vs_test"]["bemis_murcko_full_nostereo"]
        nci_lk = lk["nci60"]
        nci_default = nci_lk["protocols"][nci_lk["default_protocol"]]
        nci_legacy = nci_lk["protocols"]["scaffold_70_10_10"]
        nci_split = nci_default["split_sizes_unique_smiles"]
        pt_lk = lk["pt"]
        pt4_lk = lk.get("platinai_4cell")
        pt4r_lk = lk.get("platinai_4cell_real")
        molnet = lk["moleculenet"]["tasks"]
        molnet_bm_max = max(
            s["bemis_murcko_desalted"]["train_vs_test"]["n_intersect_keys"]
            for rec in molnet.values()
            for s in rec["per_seed"]
        )
        molnet_bm_frac = 100.0 * max(
            s["bemis_murcko_desalted"]["train_vs_test"]["frac_b_rows_seen_in_a"]
            for rec in molnet.values()
            for s in rec["per_seed"]
        )
        molnet_ns_max = max(
            s["bemis_murcko_split_key_nostereo"]["train_vs_test"]["n_intersect_keys"]
            for rec in molnet.values()
            for s in rec["per_seed"]
        )
        molnet_ns_frac = 100.0 * max(
            s["bemis_murcko_split_key_nostereo"]["train_vs_test"]["frac_b_rows_seen_in_a"]
            for rec in molnet.values()
            for s in rec["per_seed"]
        )
        nci_pred = lk["cross"]["nci60_vs_predictive5"]["canonical"]["n_intersect"]
        nci_cfm = lk["cross"]["nci60_vs_cfm_all"]["canonical"]["n_intersect"]
    else:
        pred_skf_min = pred_skf_max = 0.0
        pred_doi_min = pred_doi_max = 0.0
        cfm = cfm_test = cfm_val = cfm_full_ns = None
        nci_split = None
        nci_default = nci_legacy = None
        pt_lk = None
        pt4_lk = None
        pt4r_lk = None
        molnet_bm_max = 0
        molnet_bm_frac = 0.0
        molnet_ns_max = 0
        molnet_ns_frac = 0.0
        nci_pred = nci_cfm = 0
    lines = [
        "# Data inventory: training and evaluation",
        "",
        "> Generated by `molmetal/scripts/wf_a50_data_inventory.py`. Numbers are",
        "> measured from the files on disk. Re-run the script; do not edit this",
        "> file. Machine-readable twin: `docs/DATA_INVENTORY.json`.",
        ">",
        "> `STATUS` values: **ACTIVE** (in use), **SUPERSEDED** (older cohort, do",
        "> not train), **DO-NOT-TRAIN** (labels are synthetic), **UNUSED**,",
        "> **REFERENCE** (published baseline, never trained on).",
        "",
        "## 1. What each model trains on",
        "",
        "### 1.1 Predictive: D-MPNN metal activity",
        "",
        "Cohort: `build_paper_cohort(MetalCytoToxDB, metal)`, then "
        "`collapse_duplicates(tie -> active)`. A row is active when "
        "`IC50_Dark_value < 10 µM`.",
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
        "A SMILES can appear under more than one metal, so the per-metal sum is",
        "not a count of distinct molecules. Evaluation uses",
        "`StratifiedKFold(n_splits=5)` with seeds `[42,43,44,49,123]` and no",
        "fixed holdout: every molecule is tested in one fold.",
        "",
        (
            f"**Leakage.** Raw SMILES are canonical and unique within each metal "
            f"cohort (0 multi-string molecules, 0 exact duplicates after "
            f"collapse), so no molecule appears in both train and test: 0.00 % "
            f"of test rows under both the random and the DOI-grouped protocol. "
            f"The random CV is in-domain by scaffold, with "
            f"{pred_skf_min:.1f}-{pred_skf_max:.1f} % of test rows sharing a "
            f"Bemis-Murcko scaffold with train ({pred_doi_min:.1f}-"
            f"{pred_doi_max:.1f} % under DOI-grouped 10-fold). Report it as a "
            f"random-split AUC, not as a scaffold-generalisation estimate. "
            f"Source: `docs/DATA_LEAKAGE_AUDIT.md`."
            if lk
            else "**Leakage.** Not measured. Run `molmetal/scripts/"
            "wf_a51_leakage_audit.py`."
        ),
        "",
        "### 1.2 Generative: CFM 5-metal",
        "",
        f"- Rows {_fmt(g['rows'])}; unique SMILES {_fmt(g['unique_smiles'])}; "
        f"cell lines {g['cell_lines']}",
        f"- Metals: {', '.join(f'{k} {v:,}' for k, v in g['metals'].items())}",
        f"- In-file scaffold split: {', '.join(f'{k} {v:,}' for k, v in g['split'].items())}",
        f"- Activity: {', '.join(f'{k} {v:,}' for k, v in g['activity'].items())}",
        "",
        "The CFM is generative, so there is no held-out property test set.",
        "Quality is measured by decoding validation clouds (validity, ring",
        "profile, novelty, docking), not by a train/test AUC.",
        "",
        (
            f"**Leakage.** The in-file `scaffold_id` is stereo-aware and covers "
            f"the whole complex, counterions included. It is disjoint across "
            f"train/val/test (0 shared), and no SMILES carries more than one "
            f"split label. On a desalted, stereo-blind ligand scaffold, "
            f"{cfm_test['n_intersect_keys']} scaffolds are shared between train "
            f"and test ({100.0 * cfm_test['frac_b_rows_seen_in_a']:.1f} % of test "
            f"rows) and {cfm_val['n_intersect_keys']} between train and val "
            f"({100.0 * cfm_val['frac_b_rows_seen_in_a']:.1f} % of val rows). "
            f"Removing stereo alone leaves {cfm_full_ns['n_intersect_keys']} "
            f"shared scaffolds "
            f"({100.0 * cfm_full_ns['frac_b_rows_seen_in_a']:.1f} % of test "
            f"rows), so most of the overlap comes from counterion and solvent "
            f"fragments. The split is not a ligand-scaffold OOD split."
            if lk
            else "**Leakage.** Not measured. Run `molmetal/scripts/"
            "wf_a51_leakage_audit.py`."
        ),
        "",
        "**Second data path: tmQM geometry.** CFM configs that train the "
        "velocity field on 3-D clouds read `.data/molmetal/tmQM` instead of a "
        "CSV: 33,429 six-metal complexes with geometry, Wiberg bond orders and "
        "NBO charges on every row (see `tmqm_corpus_6metal` in section 2). The "
        "corpus is complete. Two protocol values affect any generation metric "
        "quoted from it: the scaffold split is 87.7/6.2/6.2 % of rows, and the "
        "default 192-cloud val cohort is Pt 19 / Ru 93 against a corpus that is "
        "Pt 24.8 / Ru 29.3. Source: "
        "`metrics/by_metric/tmqm_cfm_data_audit.json`.",
        "",
        "### 1.3 Platinum external evaluation",
        "",
        f"- Rows {_fmt(pt['rows'])}; unique compounds {_fmt(pt['compounds'])}; "
        f"cell lines contributing Pt rows {pt['cells_with_pt_rows']}",
        f"- Rows with SMILES {_fmt(pt['rows_with_smiles'])} of "
        f"{_fmt(pt['rows'])}; training-ready (value and SMILES) "
        f"{_fmt(pt['rows_training_ready'])}",
        (
            f"- No split column: a pool, not a partition. Duplicate "
            f"`(compound_code, cell_line)` rows "
            f"{pt_lk['n_duplicate_code_cell_rows']:,}; exact duplicate "
            f"`(code, cell, value)` rows "
            f"{pt_lk['n_duplicate_code_cell_value_rows']:,}; "
            f"{pt_lk['n_parse_failures']:,} of "
            f"{pt_lk['n_unique_raw_smiles']:,} unique SMILES fail RDKit parsing."
            if pt_lk
            else "- Split status not measured. Run `wf_a51_leakage_audit.py`."
        ),
        (
            f"- Pt 4-cell synthetic cohort (`.data/molmetal/platinai_4cell.csv`, "
            f"simulated labels, DO-NOT-TRAIN): the loader's `split_strategy` is "
            f"named `scaffold_70_10_10`, but `scaffold_id` is a stable hash of "
            f"the SMILES string, which makes it a molecule-level split. "
            f"{pt4_lk['pairs']['train_vs_test']['bemis_murcko']['n_intersect_keys']} "
            f"desalted scaffolds are shared between train and test "
            f"({100.0 * pt4_lk['pairs']['train_vs_test']['bemis_murcko']['frac_b_rows_seen_in_a']:.1f} % "
            f"of test rows), and "
            f"{100.0 * pt4_lk['pairs']['train_vs_val']['bemis_murcko']['frac_b_rows_seen_in_a']:.1f} % "
            f"of val rows share one with train. Canonical overlap is 0."
            if pt4_lk
            else "- Pt 4-cell synthetic cohort: split not measured. Run "
            "`wf_a51_leakage_audit.py`."
        ),
        (
            f"- Pt 4-cell real cohort "
            f"(`.data/molmetal/platinai_4cell_real_trainer.csv`, the file the "
            f"trainer reads): `scaffold_coverage` split, "
            f"{pt4r_lk['split_sizes_rows']['train']:,} / "
            f"{pt4r_lk['split_sizes_rows']['val']:,} / "
            f"{pt4r_lk['split_sizes_rows']['test']:,} rows, "
            f"{pt4r_lk['n_scaffolds_spanning_splits']} scaffolds spanning two "
            f"splits, canonical train/test overlap "
            f"{pt4r_lk['pairs']['train_vs_test']['canonical']['n_intersect_keys']}."
            if pt4r_lk
            else "- Pt 4-cell real cohort: split not measured. Run "
            "`wf_a51_leakage_audit.py`."
        ),
        "",
        "### 1.4 Controls",
        "",
        (
            f"- NCI-60 precious-metal panel: {_fmt(n['rows'])} rows, "
            f"{_fmt(n['unique_smiles'])} unique SMILES, {n['cells']} cell lines. "
            f"The default split (`{nci_lk['default_protocol']}`, since "
            f"2026-10-21) gives {nci_split['train']:,} / {nci_split['val']:,} / "
            f"{nci_split['test']:,} molecules and "
            f"{nci_default['split_sizes_rows']['train']:,} / "
            f"{nci_default['split_sizes_rows']['val']:,} / "
            f"{nci_default['split_sizes_rows']['test']:,} rows; every cell has "
            f"at least {nci_default['min_test_rows_per_cell']} test and "
            f"{nci_default['min_val_rows_per_cell']} val rows. The legacy "
            f"`scaffold_70_10_10` protocol gives "
            f"{nci_legacy['split_sizes_unique_smiles']['train']:,} / "
            f"{nci_legacy['split_sizes_unique_smiles']['val']:,} / "
            f"{nci_legacy['split_sizes_unique_smiles']['test']:,} molecules "
            f"({100 * nci_legacy['row_fraction']['train']:.1f} / "
            f"{100 * nci_legacy['row_fraction']['val']:.1f} / "
            f"{100 * nci_legacy['row_fraction']['test']:.1f} % of rows), with "
            f"{len(nci_legacy['cells_below_30_test_rows'])} of {nci_lk['n_cells']} "
            f"cells under 30 test rows. The panel is a subset of MetalCytoToxDB: "
            f"{nci_pred:,} of {n['unique_smiles']:,} canonical molecules appear "
            f"in the 5-metal pool and {nci_cfm:,} in CFM. The "
            f"3,645/521/1,041 figure comes from the pre-BLK-PA12 splitter and no "
            f"longer reproduces."
            if lk
            else f"- NCI-60 precious-metal panel: {_fmt(n['rows'])} rows, "
            f"{_fmt(n['unique_smiles'])} unique SMILES, {n['cells']} cell lines. "
            "Split not measured. Run `wf_a51_leakage_audit.py`."
        ),
        (
            f"- MoleculeNet (bace, bbbp, clintox, hiv): scaffold 70/10/20, "
            f"5 seeds. The split key is stereo-aware. At the stereo-blind level "
            f"at most {molnet_ns_max} scaffolds are shared between train and "
            f"test ({molnet_ns_frac:.1f} % of test rows, bace); desalting adds a "
            f"counterion component of up to {molnet_bm_max} scaffolds (hiv), "
            f"with a row-level maximum of {molnet_bm_frac:.1f} % (bbbp). "
            f"Canonical molecule overlap is 0."
            if lk
            else "- MoleculeNet (bace, bbbp, clintox, hiv): scaffold 70/10/20, "
            "5 seeds."
        ),
        "- DrugOOD lbap: 266,522 rows with iid/ood train/val/test as a column. "
        "Used as the OOD control, not as a headline result. Train vs ood_test "
        "shares 0 canonical molecules and 0 scaffold IDs; train vs iid_test "
        "shares scaffold IDs by design.",
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
        lines.append(f"- **`{d['id']}`** ({d['status']}): {d['note']}")
    lines += [
        "",
        "## 4. Do not train on these",
        "",
        "- `platinai_4cell.csv`: synthetic IC50 labels. Its ~0.50 AUC is a null",
        "  test, not a result.",
        "- `metalcytotox_5metal_multitask{,_v2}.csv`: superseded by v3/v4.",
        "- `NCI60_GI50/GI50.csv`: raw dump; the 9-cell panel is the cohort.",
        "",
        "## 5. Known gaps",
        "",
        "Open:",
        "",
        "- No model has been retrained on the NCI-60 coverage split. The default "
        "moved on 2026-10-21 (see Closed below). Until a run lands, the only "
        "9-cell number on disk (0.7294) was measured on the legacy protocol and "
        "is not comparable to anything measured on the new test set, which is "
        "1,532 rows against 343. Source: "
        "`metrics/by_metric/nci60_split_coverage_audit.json`; risk "
        "`TODO/risks.md` R11.",
        "- tmQM's val cohort is not metal-balanced. `n_val` defaults to 192 and "
        "`tmqm_min_per_metal` to 0, so the cohort behind every generation metric "
        "is a 192-of-960 reservoir window: at seed 0 it is Pt 19 / Ru 93 against "
        "a corpus that is Pt 24.8 / Ru 29.3. `tmqm_min_per_metal: 20` gives "
        "Pt 46 / Ru 52 and forces a full 2,061-candidate scan. Source: "
        "`metrics/by_metric/tmqm_cfm_data_audit.json`.",
        "- CFM v3/v4 `scaffold_id` is a whole-complex, stereo-aware key, so "
        "73.7 % of test rows share a desalted scaffold with train. "
        "`cfm_5metal_v5` fixes this (0 shared scaffolds), but no checkpoint has "
        "been trained from it; see section 3.",
        "- `pt_biodata_all` is a pool, not a partition: 17,735 rows, 3,668 "
        "duplicate `(compound_code, cell_line)` rows, 987 exact duplicate "
        "measurements, and 296 unique SMILES that fail RDKit parsing. The 4-cell "
        "cohort filters and median-collapses these (counted in its `.meta.json`); "
        "the raw pool is unchanged.",
        "",
        "Closed (kept for reference):",
        "",
        "- ~~NCI-60 split is not 70/10/20 by rows~~. The legacy "
        "`scaffold_70_10_10` protocol budgeted fractions in *scaffold counts*, so "
        "the largest 70 % of the 1,151 scaffolds (94.3-100 % of every cell's "
        "rows) went to train and test received the singleton leftovers: "
        "7,320 / 160 / 343 rows (93.6 / 2.0 / 4.4 %), five of nine cells under "
        "30 test rows (SKOV3 3, DU145 7, HL60 11, PC3 18, A2780cis 28), and "
        "SKOV3 with 0 val rows. The coverage split "
        "(`scaffold_coverage_70_10_20`: 5,371 / 920 / 1,532 rows, every cell at "
        "least 25 test and 22 val, 0 scaffolds spanning splits) has been the "
        "loader default since 2026-10-21. The configs that reproduce shipped "
        "numbers pin the legacy protocol, and `pre_flight_check.py` verifies "
        "both.",
        "- ~~BindingDB path BLOCKED~~. Fixed 2026-10-19: "
        "`.data/molmetal/bindingdb_ic50.csv` points at the 990,630-row export and "
        "`_load_real_bindingdb` accepts both headers.",
        "- ~~Platinum set has no scaffold split~~. The strategy "
        "`bemis_murcko_70_10_10` gives 13,148 / 1,056 / 2,108 rows with 0.0 % "
        "shared scaffolds. The legacy hash split remains the default for "
        "checkpoint compatibility.",
        "- ~~MoleculeNet `_scaffold_split` is stereo-aware~~. "
        "`_bemis_murcko_scaffold` is now `isomericSmiles=False`, bit-exact with "
        "`MurckoScaffoldSmiles(includeChirality=False)` on all four tasks; the "
        "4-task 5-seed re-run is the live 0.8095 artifact.",
        "",
        "## 6. Leakage audit",
        "",
        "The measured train/val/test audit is `docs/DATA_LEAKAGE_AUDIT.md`",
        "(machine twin `molmetal/reports/wf_a51_leakage_audit/leakage_audit.json`,",
        "generated by `molmetal/scripts/wf_a51_leakage_audit.py`). No active",
        "split shares a canonical molecule between train and test;",
        "scaffold-level caveats are listed in sections 1 and 5.",
        "",
    ]
    # Prose is emitted as one string per paragraph; wrap it here so the
    # rendered document keeps a consistent column width.
    wrapped: List[str] = []
    for ln in lines:
        if not ln or "\n" in ln or len(ln) <= 78 or ln[:1] in ("|", ">", "#", " "):
            wrapped.append(ln)
        elif ln.startswith("- "):
            wrapped.extend(_wrap(ln[2:], indent="- "))
        else:
            wrapped.extend(_wrap(ln))
    return "\n".join(wrapped)


def main() -> int:
    doc = build()
    # The full audit is a separate 120 KB artefact; keep the inventory JSON a
    # pointer, not a copy.  The markdown renderer still sees the full dict.
    out_doc = {k: v for k, v in doc.items() if k != "leakage"}
    out_doc["leakage_audit"] = {
        "path": str(LEAKAGE_JSON.relative_to(ROOT)),
        "generated_by": "molmetal/scripts/wf_a51_leakage_audit.py",
        "present": LEAKAGE_JSON.exists(),
    }
    JSON_OUT.write_text(
        json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    MD_OUT.write_text(render_md(doc), encoding="utf-8")
    print(f"wrote {MD_OUT.relative_to(ROOT)} and {JSON_OUT.relative_to(ROOT)}")
    for d in doc["datasets"]:
        print(f"  {d['id']:26s} {d['status']:13s} rows={_fmt(d['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
