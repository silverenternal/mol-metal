#!/usr/bin/env python
"""PAIRED embedder control for cfm_metal_geometry.

WHY THIS FILE EXISTS
--------------------
``metrics/by_metric/cfm_metal_geometry.json`` reports an *embedder floor* as
``|REAL_DFT - REAL_ETKDG|`` and compares the generator gap ``|GENERATED -
REAL_ETKDG|`` against it.  As emitted, that floor is measured on TWO DIFFERENT
POPULATIONS:

  * ``load_real_dft``  draws from the six train metals **that have coordinates**
    (intersected before the cap) -- MEASURED 5,754 molecules with a SMILES.
  * ``load_real_etkdg`` draws from the six train metals **in the csv**
    -- MEASURED 31,152 molecules.

At ``--real-limit 3000`` the two draws share **329** molecules, i.e. 2,671 of
3,000 (89%) of the DFT cohort is absent from the cohort labelled "the same real
molecules".  So the floor mixes an embedder effect with a population effect and
cannot separate them.

THIS SCRIPT MEASURES THE FLOOR THE RIGHT WAY: it pushes the *exact same* DFT
population through the embedder, molecule for molecule, and reports
``|REAL_DFT - REAL_ETKDG_SAME_MOLECULES|``.  That is the only floor against
which a generated-vs-real gap may be judged.

It also reports the CN-definition agreement, because ``coord_number`` in the
csv (tmQM's curated value) and RDKit graph degree are two different
definitions and the previous floor compared across them.

Nothing here trains anything, touches a GPU, or reads ``paper/``.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence

REPO = Path(__file__).resolve().parents[2]
GEOMETRY_SCRIPT = REPO / "molmetal" / "scripts" / "wf_blk_c_obj_metal_geometry.py"


def _load_geometry_module():
    spec = importlib.util.spec_from_file_location("_mg", GEOMETRY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MG = _load_geometry_module()


def paired_embedder_floor(limit: int, seed: int, planarity_cut: float) -> Dict[str, object]:
    """ETKDG geometry on exactly the molecules whose DFT geometry is known."""
    dft_rows = MG.select_dft_population(limit, seed)
    smiles_by_csd = _smiles_for(dft_rows)
    _dft_records, _dft_n = MG.load_real_dft(limit, seed, planarity_cut)
    dft_geoms = {rec["csd_code"]: rec for rec in _dft_records}

    etkdg_records = []
    for csd, _metal, _cn in dft_rows:
        smi = smiles_by_csd.get(csd)
        if not smi:
            continue
        Chem, _ = MG._rdkit()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        probe = MG.embed_geometry(mol) if MG.embeddable(mol) else None
        rec = MG.measure_molecule(mol, probe, planarity_cut)
        rec["csd_code"] = csd
        etkdg_records.append(rec)

    dft_summary = MG.summarise(list(dft_geoms.values()), "REAL_DFT")
    etkdg_summary = MG.summarise(etkdg_records, "REAL_ETKDG_SAME_MOLECULES")
    gen_records, _ = MG.load_generated(MG.GENERATED_CSV, planarity_cut)
    gen_summary = MG.summarise(gen_records, "GENERATED")

    paired = _paired_only(dft_geoms, etkdg_records)
    # COVERAGE is the invariant, not the overlap. The ETKDG loop already walks
    # dft_rows, so overlap is 1.0 by construction and asserting on it asserts
    # nothing (RED CHECK 3, first attempt: a vacuous guard that never fired).
    # What actually breaks is COVERAGE: draw the ETKDG side from a different
    # population and the lookup misses, so etkdg_measured collapses while the
    # note beside it still claims "the SAME csd_code list". RED CHECK 3 asked
    # 40 and got 4 -- a 10x silent shrink.
    coverage = len(etkdg_records) / len(dft_geoms) if dft_geoms else 0.0
    missing = sorted(set(dft_geoms) - {r["csd_code"] for r in etkdg_records})
    if coverage < 0.9:
        raise AssertionError(
            f"pairing broken: only {len(etkdg_records)}/{len(dft_geoms)} DFT molecules "
            f"were reachable on the ETKDG side (coverage {coverage:.3f}); "
            f"{len(missing)} missing, e.g. {missing[:5]}"
        )
    return {
        "population": {
            "dft_requested": len(dft_rows),
            "dft_measured": len(dft_geoms),
            "etkdg_measured": len(etkdg_records),
            "coverage_dft_reachable_on_etkdg_side": round(coverage, 4),
            "paired_scored": len(paired),
            "note": (
                "REAL_ETKDG_SAME_MOLECULES is built from the SAME csd_code list as "
                "REAL_DFT, so this floor is free of population drift; the coverage is "
                "MEASURED above and the script raises if it drops below 0.9, rather "
                "than claiming it in prose. "
                "metrics/by_metric/cfm_metal_geometry.json does NOT have this -- at "
                "--real-limit 3000 its two cohorts share 329 of 3,000."
            ),
        },
        "real_dft": dft_summary,
        "real_etkdg_same_molecules": etkdg_summary,
        "embedder_floor_paired": _floor(dft_summary, etkdg_summary),
        "generator_gap_versus_paired_floor": _verdict(gen_summary, etkdg_summary, dft_summary),
        "paired_cn4_class_table": paired_cn4_table(paired),
        "cn_definition_agreement": cn_definition_agreement(dft_rows, dft_geoms),
        "cn_definition_agreement_dft_vs_etkdg_paired": _cn_agreement_paired(dft_geoms, etkdg_records),
    }


def _verdict(gen: dict, etkdg: dict, dft: dict) -> Dict[str, object]:
    """Is the GENERATED-vs-real gap bigger than what the embedder alone costs?"""
    out = {}
    for key in ("coordination_number_mean", "cn4_planarity_ratio", "metal_bond_length_A"):
        g, e, d = _mean_of(gen.get(key)), _mean_of(etkdg.get(key)), _mean_of(dft.get(key))
        if not all(isinstance(v, (int, float)) for v in (g, e, d)):
            continue
        gap, floor = abs(g - e), abs(d - e)
        out[key] = {
            "generated_minus_real_etkdg_paired": round(g - e, 4),
            "abs_generator_gap": round(gap, 4),
            "abs_paired_embedder_floor": round(floor, 4),
            "verdict": ("generator gap EXCEEDS the embedder floor" if gap > floor
                        else "generator gap is WITHIN the embedder floor -- not a finding"),
        }
    return out


def _floor(dft_summary: dict, etkdg_summary: dict) -> Dict[str, object]:
    """|REAL_DFT - REAL_ETKDG_SAME_MOLECULES| per axis.

    The same molecule set on both sides, so the only thing that differs is the
    coordinate source. This is the number the generator gap must be judged
    against.
    """
    out = {}
    for key in ("coordination_number_mean", "cn4_planarity_ratio", "metal_bond_length_A"):
        a, b = _mean_of(dft_summary.get(key)), _mean_of(etkdg_summary.get(key))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[key] = {
                "real_dft_minus_real_etkdg_paired": round(a - b, 4),
                "abs_floor": round(abs(a - b), 4),
            }
    return out


def _cn_agreement_paired(dft_geoms: Dict[str, dict], etkdg_records: List[dict]) -> Dict[str, object]:
    """REAL_DFT CN (2.6 A donor window) vs REAL_ETKDG CN (RDKit graph degree),
    molecule for molecule. This is the DEFINITION gap that any coordination-number
    comparison across the two cohorts carries, independent of population."""
    by_csd = {rec["csd_code"]: rec for rec in etkdg_records}
    agree = 0
    total = 0
    for csd, dft in dft_geoms.items():
        e = by_csd.get(csd)
        if e is None or dft["coordination_number"] is None or e["coordination_number"] is None:
            continue
        total += 1
        agree += int(dft["coordination_number"] == e["coordination_number"])
    return {
        "n_compared": total,
        "n_agree": agree,
        "fraction": round(agree / total, 4) if total else None,
        "note": (
            "Measured on identical molecules, so this is a pure DEFINITION gap: "
            "DFT counts donors in a 2.6 A window, ETKDG counts RDKit graph degree. "
            "Any coordination-number gap quoted across the two cohorts contains it."
        ),
    }


def _mean_of(node):
    if isinstance(node, dict) and "mean" in node:
        return node["mean"]
    return node


def paired_cn4_table(paired: List[dict]) -> Dict[str, object]:
    """2x2: DFT class vs ETKDG class, on molecules measured by BOTH."""
    table: collections.Counter = collections.Counter()
    n = 0
    for dft_cn4, etkdg_cn4 in paired:
        if dft_cn4 is None or etkdg_cn4 is None:
            continue
        n += 1
        table[(dft_cn4, etkdg_cn4)] += 1
    return {
        "n_paired_cn4": n,
        "table": {f"dft={a}|etkdg={b}": v for (a, b), v in sorted(table.items())},
    }


def cn_definition_agreement(dft_rows, dft_geoms) -> Dict[str, object]:
    """tmQM curated ``coord_number`` vs RDKit graph degree, same molecules."""
    agree = 0
    total = 0
    for csd, _metal, curated_cn in dft_rows:
        rec = dft_geoms.get(csd)
        if rec is None or curated_cn is None:
            continue
        total += 1
        if int(curated_cn) == int(rec["coordination_number"]):
            agree += 1
    return {
        "n_compared": total,
        "n_agree": agree,
        "fraction": round(agree / total, 4) if total else None,
        "note": (
            "A fraction well below 1.0 means 'REAL_DFT CN' and 'generated CN' are "
            "not the same definition, so a coordination-number gap measured across "
            "them is partly a definition gap."
        ),
    }


def _paired_only(dft_geoms: Dict[str, dict], etkdg_records: List[dict]):
    """(dft_cn4_class, etkdg_cn4_class) for molecules ETKDG could score."""
    pairs = []
    for rec in etkdg_records:
        dft = dft_geoms.get(rec["csd_code"])
        if dft is None or dft["coordination_number"] != 4 or rec["coordination_number"] != 4:
            continue
        pairs.append((dft["geometry_class"], rec["geometry_class"]))
    return pairs


def _smiles_for(rows) -> Dict[str, str]:
    want = {csd for csd, _m, _c in rows}
    out: Dict[str, str] = {}
    import csv as _csv
    with open(MG.TMQM_CSV, newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["csd_code"] in want and row.get("smiles"):
                out[row["csd_code"]] = row["smiles"]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-limit", type=int, default=600)
    ap.add_argument("--sample-seed", type=int, default=20261012)
    ap.add_argument("--planarity-cut", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=REPO / "metrics/by_metric/cfm_metal_geometry_pairing_control.json")
    args = ap.parse_args(argv)

    result = paired_embedder_floor(args.real_limit, args.sample_seed, args.planarity_cut)
    result = {"metric": "cfm_metal_geometry_pairing_control", "tag": "MEASURED",
              "date": "2026-10-12", **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["population"], indent=2))
    print(json.dumps(result["embedder_floor_paired"], indent=2))
    print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())