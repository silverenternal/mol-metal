#!/usr/bin/env python
"""Wf-A107d — decode-path head swap: which half of the pipeline moves the rings?

WHY THIS EXISTS
---------------
Wf-A107c established two facts about the Wf-A107 A/B:

  1. ``vel_field`` weights are BIT-IDENTICAL between the control and the
     edge_ring arm (max |delta| = 0.0), because no loss term in those configs
     depends on the velocity field (``use_fm_objective: false``,
     ``wasserstein_weight: 0.0``).  The sampler therefore emits the same
     clouds for both arms.
  2. On the head's graph over the val clouds' OWN geometry, the two heads
     differ by ~5% of edges (Jaccard 0.95) and the ring effect is not
     consistent across seeds.

Yet the trainer reports ``gen_ring_3_share`` 0.510 (control) vs 0.374
(edge_ring) for seed 0.  The remaining candidate is the path BETWEEN the raw
edge set and the reported number: ``assemble_connectivity`` -> RDKit ->
SMILES -> SSSR.  This script runs that path end to end with the head swapped,
so the 2x2 is measured in the metric's own units instead of a proxy.

Because the geometry is provably shared, the expected signature of a genuine
HEAD effect is: swapping the decoder moves the number, keeping the model
fixed.  If instead the numbers follow the MODEL and not the decoder, the
prior changed something the weights do not show.

Usage:
  python molmetal/scripts/wfa107d_decode_head_swap.py \
      --arm control=results/.../ckpt.pt --arm edge_ring_w1=results/.../ckpt.pt
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _load(path: str, device: torch.device):
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_joint_model,
        load_cfm_checkpoint,
        migrate_legacy_vel_state_dict,
        restore_feature_schema,
    )

    ckpt = load_cfm_checkpoint(path, map_location="cpu")
    args = Namespace(**ckpt["args"])
    model = build_joint_model(args, device=str(device))
    sd, _migrated = migrate_legacy_vel_state_dict(
        model, ckpt["model_state_dict"], verbose=False
    )
    restore_feature_schema(model, None, ckpt, verbose=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, args, ckpt


def _build_decoder(head, args):
    """Mirror of the trainer's construction site (main(), Wf-A35-1 C3)."""
    from molmetal.eval.bond_decision import bond_decision_settings
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    decision, threshold, bias = bond_decision_settings(vars(args))
    return YuelBondBondDecoder(
        head=head,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        decision=decision,
        decision_threshold=threshold,
        connectivity_prior=str(getattr(args, "connectivity_prior", "on")).lower()
        not in ("off", "0", "false", "no"),
        connectivity_prior_min_logit=float(
            getattr(args, "connectivity_prior_min_logit", 0.0)
        ),
        class_logit_bias=bias,
        dative_max_distance=float(getattr(args, "dative_distance_cutoff", 2.6)),
    )


def _decode_block(model, decoder, templates, device, args, epoch, seed_offset):
    """One arm x one head x the trainer's 3 eval seeds."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        evaluate_generation_validity,
    )

    n_samples = int(getattr(args, "gen_eval_n", 32))
    seeds = [
        int(s) for s in str(getattr(args, "gen_eval_seeds", "0 1 2")).split()
    ] or [0]
    rows: List[Dict[str, Any]] = []
    for si, sd in enumerate(seeds):
        chem: Dict[str, Any] = {}
        smiles: List[str] = []
        n_ok, n_tot = evaluate_generation_validity(
            model=model,
            decoder=decoder,
            templates=templates,
            device=device,
            method=str(getattr(args, "method", "euler")),
            n_steps=int(getattr(args, "n_steps", 100)),
            n_samples=n_samples,
            seed=int(seed_offset) * 1_000_000 + int(epoch) * 100 + int(sd),
            template_offset=si * n_samples,
            noise_sigma=float(getattr(args, "fm_noise_sigma", 1.0)),
            sa_target=None,
            chemistry_out=chem,
            smiles_out=smiles,
            pad_groups=bool(getattr(args, "gen_pad_groups", False)),
        )
        rows.append({
            "eval_seed": sd,
            "template_offset": si * n_samples,
            "n_ok": int(n_ok),
            "n_total": int(n_tot),
            "ring_3_share": float(chem.get("ring_3_share", 0.0) or 0.0),
            "ring_5_6_share": float(chem.get("ring_5_6_share", 0.0) or 0.0),
            "rings_per_mol_mean": float(chem.get("rings_per_mol_mean", 0.0) or 0.0),
            "single_fragment_ratio": float(chem.get("single_fragment_ratio", 0.0) or 0.0),
            "n_chemistry_scored": int(chem.get("n_chemistry_scored", 0) or 0),
        })
    # The trainer keeps the LAST seed's chemistry_out as the epoch's
    # `gen_ring_*` value, so that row is the one the archived metric holds.
    last = rows[-1]
    return {
        "per_seed": rows,
        "last_seed": last,
        "mean_over_seeds": {
            k: sum(r[k] for r in rows) / len(rows)
            for k in ("ring_3_share", "ring_5_6_share", "rings_per_mol_mean")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="name=path/to/checkpoint (exactly two)")
    ap.add_argument("--csv", default=".data/molmetal/metalcytotox_5metal_multitask_v3.csv")
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=-1,
                    help="epoch to score; -1 = the checkpoint's own epoch")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="metrics/by_metric/wfa107d_decode_head_swap.json")
    args = ap.parse_args()

    if len(args.arm) != 2:
        raise SystemExit(f"--arm must be given exactly twice, got {len(args.arm)}")
    specs = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            raise SystemExit(f"[wfa107d] missing checkpoint for {name}: {path}")
        specs.append((name, path))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_csv_cohort

    arms = {}
    for name, path in specs:
        model, a, ckpt = _load(path, device)
        arms[name] = {
            "model": model, "args": a, "ckpt": ckpt, "path": path,
            "decoder": _build_decoder(model.head, a),
        }
        print(f"[wfa107d] loaded {name} epoch={ckpt.get('epoch')} on {device}")

    (name_a, _), (name_b, _) = specs
    A, B = arms[name_a], arms[name_b]
    templates = build_csv_cohort(
        args.csv,
        split="val",
        n_clouds=int(args.n_val),
        seed=int(args.val_seed),
        metal_placement=str(getattr(A["args"], "metal_placement", "bonded")),
        dative_distance_cutoff=float(getattr(A["args"], "dative_distance_cutoff", 2.6)),
    )
    epoch = int(A["ckpt"].get("epoch")) if args.epoch < 0 else int(args.epoch)

    table: Dict[str, Any] = {}
    for model_name in (name_a, name_b):
        model_arm = arms[model_name]
        for head_name in (name_a, name_b):
            key = f"model={model_name}|head={head_name}"
            block = _decode_block(
                model_arm["model"], arms[head_name]["decoder"], templates,
                device, model_arm["args"], epoch, int(getattr(model_arm["args"], "seed", 0)),
            )
            table[key] = block
            ls = block["last_seed"]
            print(
                f"[wfa107d] {key:38s} 3-share={ls['ring_3_share']:.3f} "
                f"5/6={ls['ring_5_6_share']:.3f} rings/mol={ls['rings_per_mol_mean']:.2f} "
                f"single_frag={ls['single_fragment_ratio']:.3f} "
                f"ok={ls['n_ok']}/{ls['n_total']}"
            )

    def _g(model_name, head_name, key="ring_3_share"):
        return table[f"model={model_name}|head={head_name}"]["last_seed"][key]

    effects = {}
    for metric in ("ring_3_share", "ring_5_6_share", "rings_per_mol_mean"):
        effects[metric] = {
            "head_effect_with_model_a": _g(name_a, name_b, metric) - _g(name_a, name_a, metric),
            "head_effect_with_model_b": _g(name_b, name_b, metric) - _g(name_b, name_a, metric),
            "model_effect_with_head_a": _g(name_b, name_a, metric) - _g(name_a, name_a, metric),
            "model_effect_with_head_b": _g(name_b, name_b, metric) - _g(name_a, name_b, metric),
        }
        print(
            f"[wfa107d] effect {metric:20s} "
            f"head|modelA={effects[metric]['head_effect_with_model_a']:+.4f} "
            f"head|modelB={effects[metric]['head_effect_with_model_b']:+.4f} "
            f"model|headA={effects[metric]['model_effect_with_head_a']:+.4f} "
            f"model|headB={effects[metric]['model_effect_with_head_b']:+.4f}"
        )

    doc = {
        "metric": "wfa107d_decode_head_swap",
        "tag": "MEASURED",
        "date": "2026-10-21",
        "what": (
            "The trainer's own decode path (sampler -> bond head -> assemble -> "
            "RDKit -> SMILES -> SSSR) run with the two arms' heads crossed "
            "against the two arms' models.  Same val compositions, same eval "
            "seeds, same n_samples as the training loop; the numbers are "
            "directly comparable to `gen_ring_*` in per_epoch_metrics.json."
        ),
        "arms": [name_a, name_b],
        "checkpoints": {n: arms[n]["path"] for n, _ in specs},
        "epoch": epoch,
        "device": str(device),
        "table": table,
        "effects": effects,
        "honest_limits": [
            "Single checkpoint per arm (last epoch); this is a mechanism probe, "
            "not a 3-seed test.",
            "`last_seed` is the value the trainer records as the epoch's "
            "gen_ring_*; the other two eval seeds are reported per-row.",
            "The two arms' velocity fields are bit-identical (Wf-A107c), so the "
            "`model=` dimension should be a no-op.  It is measured, not assumed: "
            "a non-zero model effect means the decode path carries state that "
            "the weights do not show.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"[wfa107d] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
