"""WF-Lambda-Core Phase 3 integration smoke.

Combines the 5 newly shipped Lambda core feature modules WITHOUT modifying
``r4_lambda_only_run.py`` (which is locked by workflow w8579x29t):

  L1. conformer_embed       (3D coords from SMILES via ETKDGv3+MMFF94s)
  L2. pharmacophore_filter  (Lipinski/Veber/ring-count quality gate)
  L3. stereo_aware_reduction (CuAAC/SPAAC/Suzuki stereo annotations)
  L4. reactions.confidence  (Laplace-smoothed tmQM scaffold priors)
  L5. search_alg.pareto     (Pareto-front multi-objective ranking)

Pipeline
--------
  1. Read 10 candidate SMILES (a hand-picked curated set spanning
     cisplatin-derivative / Pt_II-chelator / known SOTA ligands / PAINS)
  2. Pharmacophore filter:        Lipinski+Veber+ring >= 1
  3. Conformer embed:             ETKDGv3+MMFF94s -> 3D coords
  4. Stereo annotation:           regio + stereochem for each surviving mol
  5. Reaction confidence:         Laplace estimate of click reliability
  6. Pareto rank:                 (PB_flag, SA, novelty, logp) -> rank
  7. Honest report:               print summary + dump JSON

This script is intended to be run as:

    uv run python molmetal/scripts/wf_lambda_core_phase3_smoke.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make molmetal importable (similar to other scripts under molmetal/scripts/)
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Module L1: conformer embedding
from molmetal.molmetal_lam.lam_chem.conformer_embed import (
    extract_coords,
    generate_conformer,
    has_finite_3d,
)
# Module L2: pharmacophore filter
from molmetal.molmetal_lam.lam_chem.pharmacophore_filter import (
    compute_pharmacophore_report,
    filter_molecules,
    pass_pharmacophore,
)
# Module L3: stereo-aware reduction (we just import annotation; not all
# SMILES are valid click pairs but we report annotations where applicable)
from molmetal.molmetal_lam.lam_chem.stereo_aware_reduction import (
    _annotate_smiles_stereo,
    apply_click,
)
# Module L4: reaction confidence (Laplace-smoothed priors)
from molmetal.molmetal_lam.reactions.confidence import (
    ReactionConfidence,
    laplace_estimate,
    scaffold_key,
)
# Module L5: Pareto multi-objective ranking
from molmetal.molmetal_lam.search_alg.pareto import (
    dominates,
    hypervolume,
    non_dominated_set,
    pareto_front,
    rank_population,
)


# ---------------------------------------------------------------------------
# 10 hand-picked candidate SMILES (cover the spectrum we need to integrate)
# ---------------------------------------------------------------------------
CANDIDATE_SMILES = [
    "C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",  # benzanilide (drug-like)
    "CC(=O)NC1=CC=C(C=C1)O",  # paracetamol (drug-like, passes Ro5)
    "OC(=O)c1ccccc1O",  # salicylic acid (drug-like)
    "[NH2][Pt]([NH2])([Cl])[Cl]",  # cisplatin (Pt_II, acyclic)
    "OC(=O)C1=CC=NC=C1",  # nicotinic acid (drug-like)
    "c1ccc2ccccc2c1",  # naphthalene (Ro5-fail: HBD=0, HBA=0, MW=128 -> passes)
    "CCCCCCCCCCCCCCCC",  # n-hexadecane (acyclic, will fail ring-count gate)
    "CN1CCC[C@H]1c1cccnc1",  # nicotine (drug-like, has stereo)
    "OC1=CC=C(O)C=C1",  # hydroquinone
    "[OH-].[Cu+2]",  # copper(II) hydroxide ion pair (ring count fails)
]


def _try_sa_score(smi: str) -> float:
    """Compute SA score if RDKit SA_Score is available; else 1.0 default.

    Falls back to 1.0 (i.e., neutral) because SA_Score.sascorer.py may not
    be on sys.path.  This is a pure convenience, not a measurement.
    """
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        # pylint: disable=import-outside-toplevel
        import sascorer  # type: ignore
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 1.0
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 1.0


def _try_pb_proxy(smi: str) -> int:
    """Return 1 if RDKit can parse and the molecule has at least 1 ring,
    else 0.  This is a *proxy* for PoseBusters (which needs 3D coords +
    receptor).  The real PB validation runs in the production pipeline
    at molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py:validate_docked.
    """
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0
        ring_info = mol.GetRingInfo()
        return 1 if ring_info.NumRings() >= 1 else 0
    except Exception:
        return 0


def _try_logp(smi: str) -> float:
    """Compute logP if rdMolDescriptors is available; else 0.0."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors  # type: ignore

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return 0.0


def _novelty_proxy(smi: str, training_set: List[str]) -> float:
    """1 - max(Tanimoto(smi, t)) against a tiny training set."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from rdkit.DataStructs import BulkTanimotoSimilarity  # type: ignore

        ref = Chem.MolFromSmiles(smi)
        if ref is None:
            return 0.0
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref, 2, nBits=1024)
        mols = [Chem.MolFromSmiles(t) for t in training_set]
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
            for m in mols
            if m is not None
        ]
        if not fps:
            return 1.0
        sims = BulkTanimotoSimilarity(ref_fp, fps)
        return float(1.0 - max(sims))
    except Exception:
        return 0.0


def _crowding(rank_idx: List[int], pop: List[List[float]]) -> List[float]:
    """Recompute crowding distance from pareto module for the survivors."""
    from molmetal.molmetal_lam.search_alg.pareto import (  # noqa: WPS433
        _crowding_distance,
    )

    n = len(pop)
    if n <= 2:
        return [float("inf")] * n
    # Return crowding only for survivors; others get 0.
    out = [0.0] * n
    by_group: Dict[int, List[int]] = {}
    for i in rank_idx:
        by_group.setdefault(0, []).append(i)
    for r, idxs in by_group.items():
        c = _crowding_distance(idxs, pop)
        for k, i in enumerate(idxs):
            out[i] = c[k]
    return out


def main() -> int:
    t0 = time.perf_counter()
    random.seed(0)
    candidates = list(CANDIDATE_SMILES)
    n_total = len(candidates)
    print(f"[phase3-smoke] start: {n_total} candidate SMILES")
    print(f"[phase3-smoke] candidates = {candidates}")

    # Step 1: pharmacophore filter (L2)
    pharma_filtered = filter_molecules(candidates, strict=True)
    n_pharma = len(pharma_filtered)
    print(f"[phase3-smoke] pharmacophore strict pass: {n_pharma}/{n_total}")

    # Per-mol pharmacophore report (also collected as the new measured metric)
    pharma_per_mol: Dict[str, Dict[str, Any]] = {}
    for smi in candidates:
        try:
            r = compute_pharmacophore_report(smi)
            # Attribute names per PharmacophoreReport dataclass
            v_lip = int(getattr(r, "lipinski_violations", 0))
            v_veb = int(getattr(r, "veber_violations", 0))
            v_ring = bool(getattr(r, "ring_violation", False))
            tot = int(getattr(r, "total_violations", v_lip + v_veb + (1 if v_ring else 0)))
            # Compute strict / lenient from totals
            passes_strict = (tot == 0)
            passes_lenient = (tot <= 1)
            # Descriptors are nested under r.descriptors dict
            desc = getattr(r, "descriptors", {}) or {}
            pharma_per_mol[smi] = {
                "passes_strict": passes_strict,
                "passes_lenient": passes_lenient,
                "violations": tot,
                "lipinski_violations": v_lip,
                "veber_violations": v_veb,
                "ring_violation": v_ring,
                "mw": float(desc.get("MW", 0.0)),
                "logp": float(desc.get("logP", 0.0)),
                "rings": int(desc.get("ring_count", 0)),
            }
        except Exception as exc:  # noqa: BLE001
            pharma_per_mol[smi] = {"error": str(exc)[:80]}

    n_pharma_pass_strict = sum(
        1 for v in pharma_per_mol.values() if v.get("passes_strict")
    )
    pharma_pass_rate_strict = n_pharma_pass_strict / n_total

    # Step 2: conformer embed (L1) for surviving mols
    embedded: Dict[str, Dict[str, Any]] = {}
    for smi in candidates:
        try:
            mol, coords = generate_conformer(smi, max_attempts=5)
            # generate_conformer returns (mol, coords)
            ok = bool(mol is not None and coords is not None and coords.shape[0] > 0)
            if ok:
                embedded[smi] = {
                    "n_atoms": int(coords.shape[0]),
                    "ok": True,
                }
            else:
                embedded[smi] = {"ok": False, "reason": "no_conformer"}
        except Exception as exc:  # noqa: BLE001
            embedded[smi] = {"ok": False, "reason": str(exc)[:80]}

    n_embed_ok = sum(1 for v in embedded.values() if v.get("ok"))
    print(f"[phase3-smoke] conformer embed ok: {n_embed_ok}/{n_total}")

    # Step 3: stereo annotation (L3)
    stereo_per_mol: Dict[str, Dict[str, Any]] = {}
    for smi in candidates:
        try:
            ann = _annotate_smiles_stereo(smi)
            stereo_per_mol[smi] = {
                "n_stereo_atoms": len(ann) if isinstance(ann, dict) else 0,
            }
        except Exception as exc:  # noqa: BLE001
            stereo_per_mol[smi] = {"error": str(exc)[:80]}

    n_stereo = sum(
        1 for v in stereo_per_mol.values() if v.get("n_stereo_atoms", 0) > 0
    )
    print(f"[phase3-smoke] stereo-bearing mols: {n_stereo}/{n_total}")

    # Step 4: reaction confidence (L4) — uniform-prior Laplace estimate
    rc_per_mol: Dict[str, float] = {}
    for smi in candidates:
        try:
            k = scaffold_key(smi)
            laplace_p = laplace_estimate(n_success=0, n_total=0)
            rc_per_mol[smi] = laplace_p
            _ = k  # scaffold_key exercised
        except Exception:
            rc_per_mol[smi] = 0.5

    # Step 5: per-mol objective vector (PB-proxy, -SA, novelty, logp)
    training_set = [
        "CC(=O)NC1=CC=C(C=C1)O",  # paracetamol
        "CN1CCC[C@H]1c1cccnc1",  # nicotine
        "OC(=O)c1ccccc1O",  # salicylic acid
        "C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",  # benzanilide
    ]

    objectives: List[List[float]] = []
    obj_smiles: List[str] = []
    obj_detail: List[Dict[str, Any]] = []
    for smi in candidates:
        pb = float(_try_pb_proxy(smi))
        sa = _try_sa_score(smi)
        nov = _novelty_proxy(smi, training_set)
        logp = _try_logp(smi)
        # PB + novelty + logp larger = better; SA smaller = better -> negate.
        v = [pb, -sa, nov, logp]
        objectives.append(v)
        obj_smiles.append(smi)
        obj_detail.append(
            {
                "smiles": smi,
                "pb_proxy": pb,
                "sa": sa,
                "novelty": round(nov, 3),
                "logp": round(logp, 3),
                "vector": [round(x, 3) for x in v],
            }
        )

    # Step 6: Pareto rank + non-dominated set + front + hypervolume
    sorted_idx = rank_population(objectives)
    pareto_idx = non_dominated_set(objectives)
    front = pareto_front(objectives)
    try:
        # Ref point must be ≥ every front point on every axis (maximisation).
        # Compute it from data: ref[i] = max(obj[i] for obj in front) + small
        # margin.  Axes: [pb, -sa, novelty, logp]
        if front:
            axis_max = [max(obj[i] for obj in front) for i in range(len(front[0]))]
            ref_point = [axis_max[i] + 1.0 for i in range(len(axis_max))]
            hv = hypervolume(front, ref_point)
        else:
            hv = 0.0
        # Cap the value if it's overflow (some Pareto libs give inf)
        if isinstance(hv, float) and (hv == float("inf") or hv != hv):
            hv = float("nan")
    except Exception:
        hv = float("nan")

    # Compute crowding distance for each survivor
    crowding = _crowding(sorted_idx, objectives)

    # Build ranked list
    ranked_full: List[Dict[str, Any]] = []
    for order, i in enumerate(sorted_idx):
        ranked_full.append(
            {
                "smiles": obj_smiles[i],
                "selection_order": order,
                "crowding": round(crowding[i], 4),
                "objective_vector": obj_detail[i]["vector"],
            }
        )

    pareto_top: List[Dict[str, Any]] = []
    for i in pareto_idx:
        pareto_top.append(
            {
                "smiles": obj_smiles[i],
                "crowding": round(crowding[i], 4),
                "objective_vector": obj_detail[i]["vector"],
            }
        )
    # Sort pareto_top by crowding descending
    pareto_top.sort(key=lambda kv: -kv["crowding"])

    elapsed = time.perf_counter() - t0

    summary = {
        "n_candidates": n_total,
        "n_pharma_pass_strict": n_pharma_pass_strict,
        "pharma_pass_rate_strict": round(pharma_pass_rate_strict, 4),
        "n_embed_ok": n_embed_ok,
        "n_stereo": n_stereo,
        "n_pareto_front": len(pareto_idx),
        "hypervolume": (
            round(hv, 4) if not (isinstance(hv, float) and (hv != hv)) else None
        ),
        "wall_seconds": round(elapsed, 3),
        "pareto_top": pareto_top,
        "ranked_full": ranked_full,
        "pharma_per_mol": pharma_per_mol,
        "embedded_per_mol": embedded,
        "stereo_per_mol": stereo_per_mol,
        "rc_per_mol": rc_per_mol,
    }

    out_dir = Path("molmetal/reports/wf_lambda_core")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "phase3_integration.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n[phase3-smoke] ==== SUMMARY ====")
    print(f"  n_candidates              = {n_total}")
    print(f"  n_pharma_pass_strict      = {n_pharma_pass_strict}")
    print(f"  pharma_pass_rate_strict   = {pharma_pass_rate_strict:.4f}")
    print(f"  n_embed_ok                = {n_embed_ok}")
    print(f"  n_stereo                  = {n_stereo}")
    print(f"  n_pareto_front            = {len(pareto_idx)}")
    print(f"  hypervolume               = {hv}")
    print(f"  wall_seconds              = {elapsed:.3f}")
    print(f"  output                    = {out_json}")

    print("\n[phase3-smoke] Pareto front (rank 0, sorted by crowding desc):")
    for entry in pareto_top[:10]:
        print(
            f"  crowding={entry['crowding']:.4f}  "
            f"smi={entry['smiles']!r:40}  vec={entry['objective_vector']}"
        )

    print("\n[phase3-smoke] Full ranking (selection order = rank asc + crowding desc):")
    for entry in ranked_full:
        print(
            f"  ord={entry['selection_order']} crowding={entry['crowding']:.4f}  "
            f"smi={entry['smiles']!r:40}  vec={entry['objective_vector']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())