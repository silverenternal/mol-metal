"""wf_a37_knn_smoke.py — Wf-A37-4 smoke for KNNEvaluator.

Runs :func:`molmetal.eval.knn_oracle.build_default_knn_oracle`, scores
the 3 drugs called out by the Wf-A37-4 brief (cisplatin, auranofin,
satraplatin), and writes the result to
``metrics/by_metric/cfm_knn_oracle_smoke.json``.

Honest framing (NOT a honest-negative entry — the JSON just reports
what was actually run, with provenance):

- Pool = COMPLETE PlatinAI 4-cell corpus (4078 mols, every mol has
  4-cell pIC50 labels).  Default factory wires this as the labelled
  pool so every query gets real per-cell-line scores.
- Metallo-drug filter = ``molmetal/data/metallo_drugs_500_train.csv``
  (450 rows → 329 canonical SMILES).  Exposed as
  ``oracle.metallo_pool_size`` + ``oracle.metallo_in_corpus``.
- Drug SMILES:
    - cisplatin + satraplatin: from ``molmetal/data/known_pt_drugs.csv``
    - auranofin: PubChem CID 16663969 canonical SMILES (Au(I) drug,
      NOT in the Pt-only known_pt_drugs.csv, so we cite PubChem).
- pIC50 labels are SYNTHETIC per ``platinai_4cell.meta.json``
  (Rusanov 2025 Table S4-S5 is behind a Cloudflare JS challenge);
  useful as pipeline smoke, NOT comparable to Rusanov 2025 numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from molmetal.eval.knn_oracle import (
    DEFAULT_LABEL_CELL_SHORT,
    KNNScore,
    KNNEvaluator,
    build_default_knn_oracle,
)


KNOWN_PT_DRUGS_PATH = Path("molmetal/data/known_pt_drugs.csv")
SMOKE_OUT_PATH = Path("metrics/by_metric/cfm_knn_oracle_smoke.json")

# PubChem CID 16663969 canonical SMILES (auranofin, Au(I) drug).
# Not in known_pt_drugs.csv (Pt-only file), so we cite PubChem.
AURANOFIN_SMILES = (
    "CC(=O)O[P]([O-])(OC(C)=O)[Au][S]c1ccccc1.CC(=O)OCC(C)C"
)


def _load_known_pt_smiles(names: List[str]) -> Dict[str, str]:
    """Read the known_pt_drugs.csv and return {name: raw_smiles}."""
    out: Dict[str, str] = {}
    if not KNOWN_PT_DRUGS_PATH.is_file():
        return out
    with open(KNOWN_PT_DRUGS_PATH, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["name"] in names:
                out[row["name"]] = row["smiles"]
    return out


def _smoke_drug_query(
    oracle: KNNEvaluator, name: str, smiles: str, source: str
) -> Dict[str, object]:
    """Score a single drug; return the JSON-serialisable row."""
    res: KNNScore = oracle.predict(smiles)
    canon_pool = {c for c in oracle.pool_smiles}
    in_pool = (res.smiles in canon_pool) if res.smiles else False
    in_metallo = (
        res.smiles in (oracle._metallo_pool or []) if oracle._metallo_pool else False
    )
    return {
        "name": name,
        "smiles_input": smiles,
        "smiles_canonical": res.smiles,
        "source": source,
        "in_platinai_corpus": bool(in_pool),
        "in_metallo_pool": bool(in_metallo),
        "scores": {c: res.scores.get(c) for c in DEFAULT_LABEL_CELL_SHORT},
        "top_tanimoto": (
            float(res.neighbours[0].tanimoto) if res.neighbours else None
        ),
        "n_neighbours": len(res.neighbours),
        "neighbours": [
            {"smiles": nb.smiles, "tanimoto": float(nb.tanimoto)}
            for nb in res.neighbours
        ],
    }


def main() -> Dict[str, object]:
    oracle = build_default_knn_oracle()

    # Read cisplatin + satraplatin from known_pt_drugs.csv (Pt drugs).
    pt = _load_known_pt_smiles(["cisplatin", "satraplatin"])
    drugs: List[Dict[str, str]] = []
    if "cisplatin" in pt:
        drugs.append(
            {
                "name": "cisplatin",
                "smiles": pt["cisplatin"],
                "source": "molmetal/data/known_pt_drugs.csv",
            }
        )
    if "satraplatin" in pt:
        drugs.append(
            {
                "name": "satraplatin",
                "smiles": pt["satraplatin"],
                "source": "molmetal/data/known_pt_drugs.csv",
            }
        )
    # auranofin (Au(I)) — PubChem CID 16663969.
    drugs.append(
        {
            "name": "auranofin",
            "smiles": AURANOFIN_SMILES,
            "source": "PubChem CID 16663969 (Au(I) drug, not in known_pt_drugs.csv)",
        }
    )

    scored = [
        _smoke_drug_query(oracle, d["name"], d["smiles"], d["source"])
        for d in drugs
    ]

    payload: Dict[str, object] = {
        "wave": "Wf-A37-4",
        "task": "NEW-1 KNNEvaluator Tanimoto kNN smoke",
        "pool_path": "molmetal/data/metallo_drugs_500_train.csv",
        "label_path": ".data/molmetal/platinai_4cell.csv",
        "k": oracle.k,
        "fp_radius": oracle.fp_radius,
        "fp_bits": oracle.fp_bits,
        # Default factory wires PlatinAI 4-cell as the COMPLETE
        # labelled pool (every mol has 4-cell pIC50 labels), so
        # every query gets real per-cell-line scores.  The 500-mol
        # pool is exposed as the metallo-drug filter diagnostic.
        "pool_size": oracle.pool_size,
        "labelled_in_pool": oracle.labelled_in_pool,
        "metallo_pool_size": oracle.metallo_pool_size,
        "metallo_in_corpus": oracle.metallo_in_corpus,
        "cell_lines": list(DEFAULT_LABEL_CELL_SHORT),
        "drugs": scored,
        "honest_framing": (
            "Default factory wires PlatinAI 4-cell (4078 mols, ALL "
            "labelled per cell line: MCF7 / A549 / A2780 / A2780cis) "
            "as the kNN index.  Every query gets real per-cell-line "
            "scores.  The 500-mol MaxMin pool (329 canonical SMILES) "
            "is exposed as the metallo-drug chemotype filter "
            "(metallo_in_corpus = intersection with PlatinAI 4-cell).  "
            "PlatinAI pIC50 labels are SYNTHETIC per "
            "platinai_4cell.meta.json (Rusanov 2025 supplementary "
            "S4-S5 is behind Cloudflare JS); useful as pipeline smoke, "
            "NOT directly comparable to Rusanov 2025 Table S4-S5."
        ),
    }

    SMOKE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SMOKE_OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    out = main()
    print(
        f"pool_size={out['pool_size']} labelled_in_pool={out['labelled_in_pool']} "
        f"metallo_pool={out['metallo_pool_size']} "
        f"metallo_in_corpus={out['metallo_in_corpus']}"
    )
    for d in out["drugs"]:  # type: ignore[index]
        s = d["scores"]  # type: ignore[index]
        print(
            f"{d['name']}: top_tan={d['top_tanimoto']:.3f} "  # type: ignore[index]
            f"in_platinai={d['in_platinai_corpus']} "  # type: ignore[index]
            f"in_metallo={d['in_metallo_pool']}  "  # type: ignore[index]
            f"mcf7={s['mcf7']:.3f} a549={s['a549']:.3f} "  # type: ignore[index]
            f"a2780={s['a2780']:.3f} a2780cis={s['a2780cis']:.3f}"  # type: ignore[index]
        )
    print(f"wrote: {SMOKE_OUT_PATH}")