# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T64 — the cohort must not weight a complex by its assay count.

The corpus stores one row per (complex, cell line): 26,632 rows carry only
7,011 distinct (SMILES, Metal, split) keys, and one compound appears 403 times.
That is right for a toxicity model and wrong for a generative one, where the
same 3D structure is then drawn hundreds of times.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load_trainer():
    if "_t64_trainer" in sys.modules:
        return sys.modules["_t64_trainer"]
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_t64_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t64_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["SMILES_Ligands", "Metal", "split", "cell_line"])
        w.writeheader()
        for smi, metal, split, cell in rows:
            w.writerow(
                {"SMILES_Ligands": smi, "Metal": metal, "split": split, "cell_line": cell}
            )


def test_duplicate_rows_collapse_to_one_cloud(tmp_path):
    """Three assay rows of the same complex must build ONE cloud, not three."""
    tr = _load_trainer()
    path = tmp_path / "cohort.csv"
    _write_csv(
        path,
        [
            ("c1cnc2c(c1)ccc1cccnc12", "Ru", "train", "A549"),
            ("c1cnc2c(c1)ccc1cccnc12", "Ru", "train", "HCT116"),
            ("c1cnc2c(c1)ccc1cccnc12", "Ru", "train", "MCF7"),
            ("c1ccc(-c2ccccn2)nc1", "Ir", "train", "A549"),
        ],
    )
    clouds = tr.build_csv_cohort(str(path), split="train", n_clouds=None, seed=0)
    assert len(clouds) == 2
    assert len({c.smiles for c in clouds}) == 2


def test_the_same_ligand_on_a_different_metal_is_kept(tmp_path):
    """Deduplication is by (SMILES, metal), not by SMILES alone."""
    tr = _load_trainer()
    path = tmp_path / "cohort.csv"
    _write_csv(
        path,
        [
            ("c1cnc2c(c1)ccc1cccnc12", "Ru", "train", "A549"),
            ("c1cnc2c(c1)ccc1cccnc12", "Ir", "train", "A549"),
        ],
    )
    clouds = tr.build_csv_cohort(str(path), split="train", n_clouds=None, seed=0)
    assert len(clouds) == 2


def test_dedup_does_not_reduce_a_cohort_below_the_requested_size_when_possible(tmp_path):
    """n_clouds is applied AFTER dedup, so the sample is of distinct complexes."""
    tr = _load_trainer()
    path = tmp_path / "cohort.csv"
    rows = []
    for i in range(12):
        smi = "c1ccncc1" + "C" * i
        for cell in range(5):
            rows.append((smi, "Ru", "train", f"cell{cell}"))
    _write_csv(path, rows)
    clouds = tr.build_csv_cohort(str(path), split="train", n_clouds=10, seed=0)
    assert len(clouds) == 10
    assert len({c.smiles for c in clouds}) == 10
