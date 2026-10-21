#!/usr/bin/env python
"""Wf-A107 — post-hoc ring profile of the BOND HEAD's graph on val clouds.

WHY THIS EXISTS
---------------
The trainer reports the ring profile of the DECODED molecules (96 per
epoch).  That is the metric the C7 verdict is about, but it is downstream
of the ODE sampler and it is noisy at 5 epochs.  The ring PRIOR, however,
acts on the head's graph over the val clouds' own geometry.  If the prior
changes that graph and the decoded profile does not move (or moves the
wrong way), the two facts must be reported together.

This script loads each arm's checkpoint, runs the head on the SAME 64 val
clouds, thresholds p_bond > 0.5, and reports:
  * SSSR ring-size profile of the predicted graph (OPT-2 denominator),
  * bond existence precision / recall / F1 on the same clouds.

No new chemistry is invented: ring perception is RDKit's SSSR on the
predicted edge set (no sanitisation, so an odd valence cannot abort it).

Usage:
  python molmetal/scripts/wfa107_bond_graph_profile.py \
      --arm control=results/wfa107_control/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \
      --arm count=... --arm per_atom=... --arm features=... \
      --out metrics/by_metric/wfa107_bond_graph_profile.json
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def _predicted_edges(head, cloud, threshold: float) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    logits, (i_idx, j_idx) = head(cloud)
    if logits.numel() == 0:
        return [], torch.zeros(0)
    p_bond = 1.0 - torch.softmax(logits.float(), dim=-1)[:, 0]
    keep = p_bond > float(threshold)
    return list(zip(i_idx[keep].tolist(), j_idx[keep].tolist())), p_bond


def _rings_from_edges(z: List[int], edges: List[Tuple[int, int]]) -> List[int]:
    """SSSR ring sizes of an un-sanitised RWMol built from ``edges``."""
    from rdkit import Chem

    rw = Chem.RWMol()
    for zz in z:
        rw.AddAtom(Chem.Atom(int(zz)))
    seen = set()
    for i, j in edges:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        rw.AddBond(int(key[0]), int(key[1]), Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.GetSymmSSSR(mol)
    return sorted(int(len(r)) for r in mol.GetRingInfo().AtomRings())


def _truth_edges(cloud) -> List[Tuple[int, int]]:
    bonds = getattr(cloud, "bonds", None)
    if bonds is None or bonds.numel() == 0:
        return []
    types = getattr(cloud, "bond_types", None)
    if types is not None and types.numel() == bonds.shape[1]:
        bonds = bonds[:, types != 0]
    return list(zip(bonds[0].tolist(), bonds[1].tolist()))


def run_arm(ckpt_path: str, csv_path: str, n_val: int, seed: int,
            threshold: float) -> Dict[str, Any]:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_csv_cohort,
        build_joint_model,
        load_cfm_checkpoint,
        migrate_legacy_vel_state_dict,
        restore_feature_schema,
    )
    from molmetal.eval.ring_profile import summarise_ring_sizes

    ckpt = load_cfm_checkpoint(ckpt_path, map_location="cpu")
    args = Namespace(**ckpt["args"])
    model = build_joint_model(args, device="cpu")
    sd, migrated = migrate_legacy_vel_state_dict(
        model, ckpt["model_state_dict"], verbose=False
    )
    restore_feature_schema(model, None, ckpt, verbose=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    clouds = build_csv_cohort(
        csv_path,
        split="val",
        n_clouds=int(n_val),
        seed=int(seed),
        metal_placement=str(getattr(args, "metal_placement", "bonded")),
        dative_distance_cutoff=float(
            getattr(args, "dative_distance_cutoff", 2.6)
        ),
    )

    per_mol_sizes: List[List[int]] = []
    tp = fp = fn = 0
    n_atoms_total = 0
    n_edges_total = 0
    with torch.no_grad():
        for cloud in clouds:
            z = [int(v) for v in cloud.atomic_numbers.tolist()]
            n_atoms_total += len(z)
            pred, _ = _predicted_edges(model.head, cloud, threshold)
            n_edges_total += len(pred)
            per_mol_sizes.append(_rings_from_edges(z, pred))
            truth = set((min(i, j), max(i, j)) for i, j in _truth_edges(cloud))
            pred_s = set((min(i, j), max(i, j)) for i, j in pred)
            tp += len(pred_s & truth)
            fp += len(pred_s - truth)
            fn += len(truth - pred_s)

    prof = summarise_ring_sizes(per_mol_sizes)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "n_clouds": len(clouds),
        "n_atoms": n_atoms_total,
        "threshold": float(threshold),
        "predicted_edges": n_edges_total,
        "ring_profile": prof,
        "bond": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "migrated_state_keys": bool(migrated),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="name=path/to/checkpoint (repeatable)")
    ap.add_argument("--csv", default=".data/molmetal/metalcytotox_5metal_multitask_v3.csv")
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="metrics/by_metric/wfa107_bond_graph_profile.json")
    args = ap.parse_args()

    arms: Dict[str, Any] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"[wfa107] SKIP {name}: {path} missing")
            continue
        arms[name] = run_arm(path, args.csv, args.n_val, args.seed, args.threshold)
        r = arms[name]
        print(
            f"[wfa107] {name:9s} epoch={r['epoch']} rings/mol={r['ring_profile']['rings_per_mol_mean']:.3f} "
            f"3-share={r['ring_profile']['ring_3_share']:.3f} "
            f"5/6-share={r['ring_profile']['ring_5_6_share']:.3f} "
            f"bondF1={r['bond']['f1']:.3f} (P={r['bond']['precision']:.3f} R={r['bond']['recall']:.3f})"
        )

    doc = {
        "metric": "wfa107_bond_graph_profile",
        "tag": "MEASURED",
        "date": "2026-10-21",
        "what": (
            "SSSR ring-size profile and bond existence P/R/F1 of each Wf-A107 "
            "arm's bond head, evaluated on the same 64 val clouds' own "
            "geometry (threshold 0.5).  This is the graph the ring prior "
            "acts on; it is not the decoded-molecule profile."
        ),
        "arms": arms,
        "honest_limits": [
            "One checkpoint per arm (the last epoch), one seed.",
            "Ring perception runs on an un-sanitised RWMol built from the "
            "predicted edges; it is topological SSSR, not a valence check.",
            "The val clouds are the same 64 the trainer scored; this is a "
            "diagnostic, not a held-out test.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"[wfa107] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
