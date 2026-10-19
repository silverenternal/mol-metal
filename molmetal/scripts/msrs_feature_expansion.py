#!/usr/bin/env python3
"""STR-4 — three one-block feature expansions, evaluated on the four protocols.

The current design matrix is 2,074 columns: ECFP4(2048) + metal/cell codes +
metal state and coordination blocks.  Three hypotheses, one block each so the
effect is attributable:

``ecfp6``        ECFP6(4096) replaces ECFP4(2048) — larger substructures
``maccs_rdkit``  ECFP4 + MACCS(167) + 20 RDKit descriptors
``assay``        ECFP4 + time_h + is_cancer_line

Every arm is scored on the same folds as MET-2 / STR-1 and compared with the
same-seed tree OOF (``results/msrs_seed_ensemble/oof_seed{N}.npz``) using a
paired DOI-cluster bootstrap.

Note: the cohort has **no `method` column**, so the assay arm cannot include
the assay method.  That absence is recorded rather than silently dropped.

Gate (pre-registered): any arm with DOI R2 >= 0.36 and paired CI low > 0.
Kill: every arm DOI R2 < 0.34.

Usage::

    .venv/bin/python molmetal/scripts/msrs_feature_expansion.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.features.cell import cell_vocabulary, native_codes  # noqa: E402
from molmetal.features.spec import LigandSpec  # noqa: E402
from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_fusion_protocols import paired_r2_delta  # noqa: E402
from molmetal.scripts.msrs_generalization import scheme_folds  # noqa: E402
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    ENDPOINT,
    LIGAND_ONLY,
    TOP_CELL_LINES,
    build,
    coordination_block,
    load_descriptors,
    state_block,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
ARMS = ("ecfp6", "maccs_rdkit", "assay")
ARM = "ligand+both"
DOI_GATE = 0.36
DOI_KILL = 0.34
N_BOOT = 2000
RDKIT_DESCRIPTORS = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "MaxPartialCharge",
    "MinPartialCharge",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi1",
    "Kappa1",
)


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
def select_cohort(long_path: Path, limit: int = 0) -> pd.DataFrame:
    long = pd.read_parquet(long_path)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)
    if limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(int(limit))
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)
    return sub


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def maccs_rdkit_block(frame: pd.DataFrame) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, MACCSkeys

    RDLogger.DisableLog("rdApp.*")
    values = frame["ligand_canonical"].astype(str).tolist()
    uniq = list(dict.fromkeys(values))
    table: Dict[str, np.ndarray] = {}
    for smi in uniq:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            table[smi] = np.zeros(167 + len(RDKIT_DESCRIPTORS), dtype=np.float32)
            continue
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float32)
        desc = np.array(
            [float(getattr(Descriptors, name)(mol)) for name in RDKIT_DESCRIPTORS],
            dtype=np.float32,
        )
        vec = np.concatenate([maccs, desc])
        table[smi] = np.where(np.isfinite(vec), vec, 0.0).astype(np.float32)
    return np.stack([table[s] for s in values])


def assay_block(frame: pd.DataFrame) -> np.ndarray:
    time_h = frame["time_h"].to_numpy(dtype=float)
    time_h = np.where(np.isfinite(time_h), np.log1p(np.clip(time_h, 0, None)), -1.0)
    cancer = frame["is_cancer_line"].astype(str).str.lower().isin(("true", "1", "yes"))
    return np.column_stack([time_h, cancer.astype(float)]).astype(np.float32)


def build_arm(
    frame: pd.DataFrame,
    arm: str,
    base_ligand: sparse.csr_matrix,
    state_coord: sparse.csr_matrix,
) -> sparse.csr_matrix:
    if arm == "ecfp6":
        spec = replace(
            LIGAND_ONLY,
            ligand=LigandSpec(
                source="ligand_canonical", n_bits=4096, radius=3, include=("canonical",)
            ),
        )
        lig = build(frame, spec).X
        return sparse.hstack([lig, state_coord], format="csr")
    if arm == "maccs_rdkit":
        extra = sparse.csr_matrix(maccs_rdkit_block(frame))
        return sparse.hstack([base_ligand, extra], format="csr")
    if arm == "assay":
        extra = sparse.csr_matrix(assay_block(frame))
        return sparse.hstack([base_ligand, extra], format="csr")
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        type=Path,
        default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet",
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N ligand fragments")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_feature_expansion.json",
    )
    args = parser.parse_args(argv)

    frame = select_cohort(args.long, limit=args.limit)
    y = frame["value_log10"].to_numpy(dtype=float)
    print(f"[str4] rows={len(frame)}")

    load_descriptors(args.metal_desc)
    vocab = cell_vocabulary(frame["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES)
    codes = native_codes(
        frame["metal"].astype(str).tolist(),
        frame["cell_line_key"].astype(str).tolist(),
        vocab,
        mode="full",
    )
    t0 = time.time()
    base = build(frame, LIGAND_ONLY)
    state, _ = state_block(frame)
    coord, _, _ = coordination_block(frame, include_geometry=True)
    state_coord = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    base_ligand = sparse.hstack([base.X, state_coord], format="csr")
    print(f"[str4] baseline features={base_ligand.shape} in {time.time() - t0:.0f}s")

    oof_path = PROJECT_ROOT / "results" / "msrs_seed_ensemble" / f"oof_seed{args.seed}.npz"
    baseline: Dict[str, np.ndarray] = {}
    if args.limit:
        print("[str4] smoke mode: baseline disabled (row count differs)")
    elif oof_path.exists():
        with np.load(oof_path, allow_pickle=False) as data:
            for protocol in PROTOCOLS:
                baseline[protocol] = data[f"{protocol}__{ARM}"].astype(float)
        print(f"[str4] baseline={oof_path.name}")
    else:
        print(f"[str4] WARNING no baseline OOF at {oof_path}")

    doi = frame["doi"].astype(str).to_numpy()
    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    protocols: Dict[str, object] = {}
    for protocol in args.protocols:
        fold_id, _groups, info = scheme_folds(frame, protocol)
        ev = fold_id >= 0
        protocols[protocol] = {"folds": info, "arms": {}}
        print(f"[str4] {protocol:9s} folds={info['n_folds']} rows={int(ev.sum())}")
        for arm in args.arms:
            t = time.time()
            X = build_arm(frame, arm, base_ligand, state_coord)
            pred = fit_arm(
                X,
                empty,
                empty,
                None,
                0,
                False,
                y,
                fold_id,
                cat_codes=codes,
                params_override={"random_state": int(args.seed)},
            )
            entry = {
                **score(y[ev], pred[ev]),
                "n_features": int(X.shape[1]),
                "wall_s": round(time.time() - t, 1),
            }
            base_pred = baseline.get(protocol)
            if base_pred is not None:
                finite = ev & np.isfinite(base_pred)
                entry["vs_tree"] = paired_r2_delta(
                    y[finite], base_pred[finite], pred[finite], doi[finite],
                    n_boot=N_BOOT,
                )
            protocols[protocol]["arms"][arm] = entry
            print(
                f"[str4]   {arm:12s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f} "
                f"({entry['wall_s']}s)"
            )

    verdicts: Dict[str, object] = {}
    for arm in args.arms:
        doi_entry = protocols.get("doi", {}).get("arms", {}).get(arm)
        if doi_entry is None or "vs_tree" not in doi_entry:
            verdicts[arm] = {"status": "NOT_RUN"}
            continue
        ci = doi_entry["vs_tree"]
        if doi_entry["r2"] >= DOI_GATE and ci["ci_low"] > 0:
            status = "PASS"
        elif doi_entry["r2"] < DOI_KILL:
            status = "KILL"
        else:
            status = "PARTIAL"
        verdicts[arm] = {
            "status": status,
            "doi_r2": doi_entry["r2"],
            "doi_delta": ci["delta_r2"],
            "doi_ci": [ci["ci_low"], ci["ci_high"]],
        }
    all_kill = all(v.get("status") == "KILL" for v in verdicts.values())
    overall = "KILL" if all_kill and verdicts else (
        "PASS" if any(v.get("status") == "PASS" for v in verdicts.values()) else "PARTIAL"
    )

    payload = {
        "artifact": "msrs_feature_expansion",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_feature_expansion.py",
            "git_commit": _git_commit(),
            "long_sha256": _sha256(args.long),
            "baseline_oof": (
                str(oof_path.relative_to(PROJECT_ROOT)) if oof_path.exists() else None
            ),
        },
        "design": {
            "seed": int(args.seed),
            "arms": list(args.arms),
            "protocols": list(args.protocols),
            "baseline_features": int(base_ligand.shape[1]),
            "ecfp6": {"n_bits": 4096, "radius": 3},
            "maccs_rdkit": {
                "maccs_bits": 167,
                "rdkit_descriptors": list(RDKIT_DESCRIPTORS),
            },
            "assay": {
                "columns": ["time_h (log1p)", "is_cancer_line"],
                "method_column_present": bool("method" in frame.columns),
                "note": "the cohort has no `method` column; it is not silently imputed",
            },
            "doi_gate": DOI_GATE,
            "doi_kill": DOI_KILL,
            "n_boot": N_BOOT,
        },
        "protocols": protocols,
        "verdicts": verdicts,
        "verdict": overall,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    for arm, verdict in verdicts.items():
        print(f"[str4] verdict[{arm}]={verdict.get('status')} {verdict}")
    print(f"[str4] verdict={overall} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
