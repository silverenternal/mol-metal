# SPDX-License-Identifier: Apache-2.0
"""Wf-A107h — can the FM velocity field fit its own training target at all?

WHY THIS EXISTS
---------------
Every generation number the CFM line has reported (``gen_ring_*``,
``gen_validity``, the C7 pool's SA<=4.5 = 0.28%) is a decode of
``generate_clouds_from_noise``'s output.  Wf-A107g MEASURED that output sits on
top of the prior: at ``fm_noise_sigma`` 2.485 the generated clouds have mean
pair distance 5.67 A against the prior's 5.59 A and the truth's 5.00 A.  The
obvious explanation is "5 epochs is too few".  This script tests the other
explanation: that the field CANNOT represent the target, in which case no
budget fixes it.

The test is an overfit test on ONE batch: 200 AdamW steps at lr 1e-3 on a
single 8-cloud val batch, same interpolant the trainer uses
(``x_t = 0.5 x0 + 0.5 x1``, ``target = x1 - x0``).  If the field can fit one
batch, the loss must fall well below the ``v = 0`` baseline.  If it cannot,
the reported ``fm_term`` is the zero-prediction floor and the sampler is a
readout of ``x0``.

The script also reports the RADIAL CEILING.  ``EGNNVelocityField.forward``
composes ``vel = tanh(vel_head(h)) * vel_scale * (x - centroid) + last_v``
(``flow_matching_lipman/__init__.py``), so the dominant term can only move an
atom along the ray from the cloud centroid.  Projecting the target onto that
ray gives the best MSE any radial-only field can reach; a measured MSE at that
value means the field is at its architecture's optimum, and the shortfall is
structural rather than an optimisation problem.

Usage:
  python molmetal/scripts/wfa107h_velocity_capacity.py \
      --ckpt results/wfa107g_fm_sigma_control/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \
      --out metrics/by_metric/wfa107h_velocity_capacity.json

CPU is enough (8 clouds, ~40 s); ``--device cuda:0`` is faster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _radial_best_mse(x0: torch.Tensor, x1: torch.Tensor, sizes: List[int]) -> float:
    """MSE of the best field of the form ``s_i * (x_t - centroid)``.

    Per atom and per cloud, the least-squares scalar along the radial unit
    vector is the projection of the target onto that ray, clipped to the
    gate's (-1, 1) range times ``vel_scale`` — with ``vel_scale`` free the
    clip is not binding, so the projection is the exact optimum.
    """
    xt = 0.5 * x0 + 0.5 * x1
    tgt = x1 - x0
    out = torch.zeros_like(xt)
    start = 0
    for n in sizes:
        sl = slice(start, start + n)
        p = xt[sl]
        rel = p - p.mean(dim=0, keepdim=True)
        r = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit = rel / r
        coeff = (tgt[sl] * unit).sum(dim=-1, keepdim=True)
        out[sl] = coeff * unit
        start += n
    return float(F.mse_loss(out, tgt))


def _stats(v: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    return {
        "mse": float(F.mse_loss(v, tgt)),
        "mean_v_norm": float(v.norm(dim=-1).mean()),
        "mean_target_norm": float(tgt.norm(dim=-1).mean()),
        "cosine": float(
            F.cosine_similarity(v.reshape(-1), tgt.reshape(-1), dim=0)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default=".data/molmetal/metalcytotox_5metal_multitask_v3.csv")
    ap.add_argument("--n-clouds", type=int, default=8)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="metrics/by_metric/wfa107h_velocity_capacity.json")
    args = ap.parse_args()

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_csv_cohort,
        build_joint_model,
        load_cfm_checkpoint,
    )

    ck = load_cfm_checkpoint(args.ckpt, map_location="cpu")
    ck_args = ck["args"]
    import argparse as _ap

    ns = _ap.Namespace(**ck_args)
    dev = torch.device(args.device)
    model = build_joint_model(ns).to(dev)
    model.load_state_dict(ck["model_state_dict"])

    clouds = build_csv_cohort(
        args.csv, split="val", n_clouds=int(args.n_clouds),
        seed=int(args.val_seed),
        metal_placement=str(getattr(ns, "metal_placement", "legacy")),
        dative_distance_cutoff=float(getattr(ns, "dative_distance_cutoff", 2.6)),
    )
    x1 = torch.cat([c.positions.float() for c in clouds], 0).to(dev)
    z = torch.cat([c.atomic_numbers.long() for c in clouds], 0).to(dev)
    sizes = [int(c.atomic_numbers.shape[0]) for c in clouds]

    sigma = float(getattr(ns, "fm_noise_sigma", 1.0))
    g = torch.Generator(device=dev).manual_seed(int(args.noise_seed))
    x0 = torch.randn(x1.shape, generator=g, device=dev) * sigma
    tgt = x1 - x0
    t = float(args.t)
    xt = (1.0 - t) * x0 + t * x1
    t_col = torch.full((x1.shape[0],), t, device=dev)

    zero_baseline = float(F.mse_loss(torch.zeros_like(tgt), tgt))
    radial_ceiling = _radial_best_mse(x0, x1, sizes)

    model.train()
    opt = torch.optim.AdamW(model.vel_field.parameters(), lr=float(args.lr))
    traj: List[Dict[str, Any]] = []
    with torch.no_grad():
        v0 = model.fm_velocity(xt, t_col, z, cloud_sizes=sizes)
    traj.append({"step": 0, **_stats(v0, tgt)})
    print(f"[wfa107h] step   0 {traj[-1]}")
    for step in range(1, int(args.steps) + 1):
        vp = model.fm_velocity(xt, t_col, z, cloud_sizes=sizes)
        loss = F.mse_loss(vp, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 40 == 0:
            with torch.no_grad():
                vp = model.fm_velocity(xt, t_col, z, cloud_sizes=sizes)
                row = {"step": step, **_stats(vp, tgt)}
            traj.append(row)
            print(f"[wfa107h] step {step:3d} {row}")

    last = traj[-1]
    doc = {
        "metric": "wfa107h_velocity_capacity",
        "tag": "MEASURED",
        "date": "2026-10-22",
        "what": (
            "Overfit test of the FM velocity field on ONE 8-cloud batch: 200 "
            "AdamW steps at lr 1e-3, fixed t.  Compares the reachable loss "
            "with the v=0 baseline and with the best radial-only field, which "
            "is the form EGNNVelocityField's dominant term can express."
        ),
        "checkpoint": str(args.ckpt),
        "ckpt_epoch": ck.get("epoch"),
        "noise_sigma": sigma,
        "t": t,
        "n_clouds": len(clouds),
        "n_atoms": int(x1.shape[0]),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "device": str(dev),
        "zero_baseline_mse": zero_baseline,
        "radial_ceiling_mse": radial_ceiling,
        "trajectory": traj,
        "verdict": (
            "CANNOT_FIT"
            if last["mse"] > 0.9 * zero_baseline
            else "FITS_ONE_BATCH"
        ),
        "honest_limits": [
            "One batch, one t (0.5).  A field that fits t=0.5 need not fit the "
            "whole path; a field that cannot fit t=0.5 cannot fit the path.",
            "The radial ceiling is computed on the SAME batch, so it is an "
            "upper bound on the radial term only — `last_v` (the edge-mean "
            "term) can in principle beat it.  A measured MSE at the radial "
            "ceiling means the model is using the radial term alone.",
            "lr 1e-3 is 10x the training lr (1e-4); this is deliberately "
            "generous to the field, so a failure is not a learning-rate "
            "artefact.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(
        f"[wfa107h] zero={zero_baseline:.3f} radial_ceiling={radial_ceiling:.3f} "
        f"reached={last['mse']:.3f} |v|={last['mean_v_norm']:.3f} "
        f"|target|={last['mean_target_norm']:.3f} -> {doc['verdict']}"
    )
    print(f"[wfa107h] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
