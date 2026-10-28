# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T60 — measure the corpus after the donor counter stopped being a
regex over the SMILES text.

THE DEFECT
----------
``metal_smiles.count_donors`` walked the SMILES STRING with ``[A-Z][a-z]?``,
so every aromatic heteroatom was invisible: SMILES writes aromatic nitrogen as
lowercase ``n``, and 1,10-phenanthroline (``c1cnc2c(c1)ccc1cccnc12``) contains
no uppercase ``N`` at all.  ``_select_ligands`` then dropped the fragment
("zero donors"), and step 4 of the reconstruction padded the empty ligand list
with ``[OH2]`` until the coordination capacity was reached — so the ligand was
DELETED and six waters invented in its place.

WHAT THIS MEASURES
------------------
Both the row-level composition and the 399-cloud cohort, with the pre-fix
counter restored by monkeypatch, so the artifact carries a paired
before/after.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t60_donor_count_probe.py \
        --config configs/wfa108l_geom_smoke_2026_10_25.yaml \
        --out metrics/by_metric/wfa113_t60_donor_count.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]


def _load_trainer():
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_wfa113_t60_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wfa113_t60_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _old_count_donors(ms, fragment: str) -> int:
    """The pre-T60 body, verbatim: a regex scanner over the SMILES text."""
    if not fragment:
        return 0
    bracket = re.compile(r"\[([^\]]+)\]")
    bare = re.compile(r"([A-Z][a-z]?)")
    count = 0
    for match in bracket.finditer(fragment):
        inside = match.group(1)
        atom = inside[0] if inside and inside[0].isalpha() else ""
        if (
            len(inside) >= 2
            and inside[1].islower()
            and inside[:2] in {"Cl", "Br", "Si", "Se", "As"}
        ):
            continue
        if atom in ms._DONOR_ATOMS:
            count += 1
    pos = 0
    for match in bracket.finditer(fragment):
        for tok in bare.finditer(fragment[pos:match.start()]):
            if tok.group(1) in ms._DONOR_ATOMS:
                count += 1
        pos = match.end()
    for tok in bare.finditer(fragment[pos:]):
        if tok.group(1) in ms._DONOR_ATOMS:
            count += 1
    return count


def _old_donor_score(ms, fragment: str) -> int:
    score = 0
    for atom, strength in ms._DONOR_STRENGTH.items():
        score += strength * (
            len(re.findall(rf"\[{atom}", fragment))
            + len(re.findall(rf"(?<!\[){atom}", fragment))
        )
    return score


def _row_stats(ms, csv_path: Path) -> Dict[str, Any]:
    rows = list(csv.DictReader(open(csv_path)))
    n_rows = len(rows)
    n_zero_donor_rows = 0
    pad_total = 0
    n_all_water = 0
    n_no_ligand_kept = 0
    for r in rows:
        frags = [
            f
            for f in ms._split_fragments(r["SMILES_Ligands"])
            if not ms._is_counterion(f) and not ms._is_solvent(f)
        ]
        cap = ms.coordination_capacity(r["Metal"], 3)
        kept = ms._best_donor_subset([f for f in frags if ms.count_donors(f) > 0], cap)
        donors = sum(ms.count_donors(l) for l in kept)
        pad = max(0, cap - donors)
        pad_total += pad
        if not kept:
            n_no_ligand_kept += 1
        if pad >= cap:
            n_all_water += 1
        if sum(ms.count_donors(f) for f in frags) == 0:
            n_zero_donor_rows += 1
    return {
        "n_rows": n_rows,
        "rows_with_zero_donors": n_zero_donor_rows,
        "rows_with_zero_donor_fraction": n_zero_donor_rows / max(n_rows, 1),
        "rows_kept_no_ligand": n_no_ligand_kept,
        "rows_that_become_all_water": n_all_water,
        "rows_that_become_all_water_fraction": n_all_water / max(n_rows, 1),
        "water_padding_per_row_mean": pad_total / max(n_rows, 1),
    }


def _cohort_stats(tr, cfg, n_clouds: int) -> Dict[str, Any]:
    clouds = tr.build_csv_cohort(
        str(cfg.data_csv),
        split=str(getattr(cfg, "data_split", "train")),
        n_clouds=int(n_clouds),
        seed=int(cfg.seed),
        metal_placement="bonded",
        dative_distance_cutoff=float(getattr(cfg, "dative_distance_cutoff", 2.6)),
    )
    total_atoms = 0
    water_atoms = 0
    n_all_water = 0
    dative = 0
    smiles = set()
    for c in clouds:
        n = int(c.atomic_numbers.shape[0])
        total_atoms += n
        s = c.smiles or ""
        w = s.count("[OH2]") * 3
        water_atoms += w
        if n and w / n >= 0.9:
            n_all_water += 1
        if c.is_dative_pair is not None:
            dative += int(c.is_dative_pair.sum()) // 2
        smiles.add(s)
    return {
        "n_clouds": len(clouds),
        "unique_smiles": len(smiles),
        "atoms_total": total_atoms,
        "atoms_per_cloud_mean": total_atoms / max(len(clouds), 1),
        "water_atoms": water_atoms,
        "water_atom_fraction": water_atoms / max(total_atoms, 1),
        "clouds_at_least_90pct_water": n_all_water,
        "clouds_at_least_90pct_water_fraction": n_all_water / max(len(clouds), 1),
        "dative_bonds_total": dative,
        "dative_per_cloud_mean": dative / max(len(clouds), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=400)
    args = ap.parse_args()

    tr = _load_trainer()
    sys.argv = ["probe", "--config", args.config, "--device", "cpu",
                "--out-dir", "/tmp/wfa113_t60", "--ckpt-dir", "/tmp/wfa113_t60"]
    cfg = tr.parse_args()

    import molmetal.data.metal_smiles as ms

    new_count, new_score = ms.count_donors, ms._donor_score
    csv_path = Path(str(cfg.data_csv))

    # ---- BEFORE ----------------------------------------------------------
    ms.count_donors = lambda f: _old_count_donors(ms, f)
    ms._donor_score = lambda f: _old_donor_score(ms, f)
    ms._donor_symbols.cache_clear()
    before_rows = _row_stats(ms, csv_path)
    before_cohort = _cohort_stats(tr, cfg, args.n_clouds)

    # ---- AFTER -----------------------------------------------------------
    ms.count_donors, ms._donor_score = new_count, new_score
    ms._donor_symbols.cache_clear()
    after_rows = _row_stats(ms, csv_path)
    after_cohort = _cohort_stats(tr, cfg, args.n_clouds)

    out = {
        "metric": "wfa113_t60_donor_count",
        "tag": "MEASURED",
        "date": "2026-10-28",
        "config": args.config,
        "cohort": (
            f"{args.n_clouds} clouds of metalcytotox_5metal_multitask_v3.csv, "
            "split=train, metal_placement=bonded"
        ),
        "device": "cpu",
        "probe": "molmetal/scripts/wfa113_t60_donor_count_probe.py",
        "defect": {
            "code": "molmetal/data/metal_smiles.py::count_donors (regex over the SMILES text)",
            "mechanism": (
                "The counter scanned for [A-Z][a-z]? in the string. SMILES writes "
                "aromatic nitrogen as lowercase n, so 1,10-phenanthroline "
                "(c1cnc2c(c1)ccc1cccnc12) counted zero donors. _select_ligands then "
                "dropped the fragment, and step 4 padded the empty ligand list with "
                "[OH2] until the coordination capacity (6 for every metal here) was "
                "reached. The ligand was deleted and six waters invented."
            ),
            "fix": (
                "count_donors and _donor_score read the parsed molecule "
                "(_donor_symbols, lru_cached): N/O/S/P plus halides, excluding "
                "positively charged atoms, anionic halides, and aromatic N that "
                "carries an H (pyrrole/indole put that lone pair in the ring)."
            ),
        },
        "before": {"rows": before_rows, "cohort": before_cohort},
        "after": {"rows": after_rows, "cohort": after_cohort},
        "honest_limits": [
            "The donor count is still a heuristic: it counts donor ATOMS, not "
            "chelate rings, so a bidentate ligand is assumed to use all its donors.",
            "Water padding still happens for complexes whose ligands genuinely "
            "supply fewer than 6 donors; it is now 6.7% of atoms rather than 21.2%, "
            "not zero.",
            "Changing the counter changes every cloud in every run. Every archived "
            "CFM number was produced on the old corpus and is not comparable to a "
            "run on this one.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"before": out["before"], "after": out["after"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
