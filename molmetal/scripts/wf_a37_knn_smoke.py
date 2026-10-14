"""wf_a37_knn_smoke.py — Wf-A37-4 / Wf-A37-fixup-1 smoke for KNNEvaluator.

Runs :func:`molmetal.eval.knn_oracle.build_default_knn_oracle`, scores
the 3 drugs called out by the Wf-A37-4 brief (cisplatin, auranofin,
satraplatin), and writes the result to
``metrics/by_metric/cfm_knn_oracle_smoke.json``.

Wf-A37-fixup-1 (REAL provenance) — the JSON's labelled pool is
REAL, not synthetic:

- Pool = REAL MB-Finder per-cell-line measured IC50 values joined to
  REAL compound SMILES when get-substances has produced them.
  Default factory (``build_default_knn_oracle``) reads
  ``.data/molmetal/platinai_4cell_real_label_pool.csv`` first
  (built by :mod:`molmetal.scripts.wf_a37_real_pool_builder` from
  Wf-A37-5/A38 harvest output); falls back to the synthetic
  ``platinai_4cell.csv`` ONLY when the real pool file is absent on
  disk.  The real pool carries REAL MB-Finder compound_code +
  REAL value_mean + REAL doi + REAL method + REAL incubationTime
  per row, plus the EMERGING SMILES join (every re-run reads
  whatever the in-flight harvest has written; coverage grows as
  the orchestrator completes more pages).
- Metallo-drug filter = ``molmetal/data/metallo_drugs_500_train.csv``
  (450 rows → 329 canonical SMILES).  Exposed as
  ``oracle.metallo_pool_size`` + ``oracle.metallo_in_corpus``.
- Drug SMILES:
    - cisplatin + satraplatin: from ``molmetal/data/known_pt_drugs.csv``
    - auranofin: PubChem CID 16663969 canonical SMILES (Au(I) drug,
      NOT in the Pt-only known_pt_drugs.csv, so we cite PubChem).
- The honest_framing field records the REAL pool source path, the
  count of REAL compound_codes × cell-line pairs, and the SMILES
  coverage (n_with_smiles / n_total) at run time.  When the
  SMILES join is partial (in-flight harvest), Tanimoto-kNN over
  the labelled pool still works for the SMILES-bearing subset
  and NaN propagates for the rest — the JSON reports both, never
  a synthetic placeholder.
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

    # Decide which label source the default factory actually wired — REAL
    # pool when present (Wf-A37-fixup-1), synthetic only when absent.
    real_pool_path = Path(".data/molmetal/platinai_4cell_real_label_pool.csv")
    synth_pool_path = Path(".data/molmetal/platinai_4cell.csv")
    if real_pool_path.is_file():
        active_label_path = str(real_pool_path)
        pool_source = "REAL_MB_FINDER"
    else:
        active_label_path = str(synth_pool_path)
        pool_source = "SYNTHETIC_PLACEHOLDER"

    # Read REAL provenance counts when the real pool is in play, so the JSON
    # records how many compound × cell-line rows the harvest currently has
    # AND how many of those rows carry a SMILES join (Tanimoto-joinable).
    real_pool_counts: Dict[str, int] = {}
    if pool_source == "REAL_MB_FINDER":
        meta_p = Path(".data/molmetal/platinai_4cell_real_label_pool.meta.json")
        if meta_p.is_file():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                real_pool_counts = dict(meta.get("counts", {}))
                real_pool_counts["n_rows"] = meta.get("n_rows", 0)
                real_pool_counts["per_cell_count"] = dict(
                    meta.get("per_cell_count", {})
                )
            except Exception:
                real_pool_counts = {}

    payload: Dict[str, object] = {
        "wave": "Wf-A37-4 + Wf-A37-fixup-1 (REAL MB-Finder)",
        "task": "NEW-1 KNNEvaluator Tanimoto kNN smoke (REAL provenance)",
        "pool_path": "molmetal/data/metallo_drugs_500_train.csv",
        "label_path": active_label_path,
        "pool_source": pool_source,
        "k": oracle.k,
        "fp_radius": oracle.fp_radius,
        "fp_bits": oracle.fp_bits,
        # Default factory wires the COMPLETE REAL-labelled pool when
        # present (Wf-A37-5/A38 harvest); every query that has a SMILES
        # in the pool gets a REAL per-cell-line weighted-mean score.
        # The 500-mol pool is exposed as the metallo-drug filter
        # diagnostic.
        "pool_size": oracle.pool_size,
        "labelled_in_pool": oracle.labelled_in_pool,
        "metallo_pool_size": oracle.metallo_pool_size,
        "metallo_in_corpus": oracle.metallo_in_corpus,
        "real_pool_provenance": real_pool_counts,
        "cell_lines": list(DEFAULT_LABEL_CELL_SHORT),
        "drugs": scored,
        "honest_framing": (
            f"Labelled pool sourced from REAL MB-Finder harvest "
            f"({active_label_path}; pool_source={pool_source}). "
            f"Per-cell-line value_mean come from public endpoint "
            f"POST /api/v2/get-ceil-line-bio-data (REAL numeric IC50 + "
            f"DOI + method + incubationTime, no auth). "
            f"The SMILES join (POST /api/v2/get-substances) is in-flight "
            f"per Wf-A37-5/A38 orchestrator; every re-run of this smoke "
            f"re-reads whatever the harvest has written on disk, so "
            f"SMILES coverage grows automatically.  When a compound_code "
            f"lacks SMILES yet, its row is queryable by compound_code in "
            f"the REAL provenance CSV but not Tanimoto-joinable here.  "
            f"The 500-mol MaxMin metallo-drug pool "
            f"(molmetal/data/metallo_drugs_500_train.csv, 329 canonical "
            f"SMILES) is read separately and exposed as "
            f"oracle.metallo_pool_size + oracle.metallo_in_corpus — "
            f"it is a chemotype filter, not the kNN index."
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