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

The test is an overfit test on ONE batch: ``--steps`` AdamW steps (default 200,
Wf-A108 uses 400/800) at lr 1e-3 on a single 8-cloud val batch, same
interpolant the trainer uses
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
    # Wf-A108c — the evaluation cohort.  When the training rotation is
    # `--cycle-batches N` from `--val-seed`, cohort 0 is IN the rotation, so
    # the trajectory is a training-set fit and a wider rotation looks worse
    # only because it sees that batch less often.  --eval-seed separates the
    # two: pick a seed outside [val_seed, val_seed + N) to measure
    # generalisation instead of memorisation.
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--t", type=float, default=0.5)
    # Wf-A108c — cycle through N cohorts (val_seed + k) instead of fitting one.
    # This is the only difference between "the head can fit this batch" and
    # "the trainer's per-step gradient carries the same information".
    ap.add_argument("--cycle-batches", type=int, default=0)
    ap.add_argument("--t-uniform", action="store_true", default=False)
    # Init seed.  Without it the fresh-field arms are not reproducible:
    # three 400-step runs of the SAME config reached 0.548 / 1.060 / 1.352
    # purely from the unseeded model build, which is wider than the effect
    # any one of them was supposed to measure.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--velocity-head",
        choices=("radial", "vector_channels"),
        default=None,
        help=(
            "Wf-A108a: override the coordinate head.  When this differs from "
            "the checkpoint's head the backbone weights cannot be loaded (the "
            "parameter sets differ), so a FRESH field is built from the "
            "checkpoint's architecture args and the run is reported as such."
        ),
    )
    ap.add_argument("--vector-channels", type=int, default=8)
    ap.add_argument("--t-embedding-dim", type=int, default=None)
    ap.add_argument(
        "--metal-frame-channels", action="store_true", default=False
    )
    ap.add_argument(
        "--radial-centroid-masked", action="store_true", default=False
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Build a fresh field instead of loading the checkpoint weights.",
    )
    ap.add_argument("--out", default="metrics/by_metric/wfa107h_velocity_capacity.json")
    # Wf-A108c — dump the trained velocity field so a later probe can ask what
    # the field actually learned (t-sensitivity, channel usage) instead of
    # re-running the whole arm.
    ap.add_argument("--save-field", default=None)
    args = ap.parse_args()

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_csv_cohort,
        build_joint_model,
        load_cfm_checkpoint,
    )

    ck = load_cfm_checkpoint(args.ckpt, map_location="cpu")
    ck_args = ck["args"]
    import argparse as _ap

    torch.manual_seed(int(args.seed))
    ns = _ap.Namespace(**ck_args)
    dev = torch.device(args.device)
    ckpt_head = str(getattr(ns, "velocity_head", "radial"))
    head = str(args.velocity_head or ckpt_head)
    fresh = bool(args.fresh) or head != ckpt_head
    if head == "vector_channels":
        ns.velocity_head = "vector_channels"
        ns.vector_channels = int(args.vector_channels)
        ns.metal_frame_channels = bool(args.metal_frame_channels)
    else:
        ns.velocity_head = "radial"
        ns.radial_centroid_masked = bool(args.radial_centroid_masked)
    # Wf-A108d — the sinusoidal-t surface is already wired end to end
    # (`--t-embedding-dim` on the trainer, `SinusoidalEmbedding` in the field);
    # the gate only needs to be able to turn it on.
    if args.t_embedding_dim is not None:
        ns.t_embedding_dim = int(args.t_embedding_dim)
    model = build_joint_model(ns).to(dev)
    if fresh:
        print(
            f"[wfa107h] FRESH field (head={head}); checkpoint weights not loaded"
        )
    else:
        model.load_state_dict(ck["model_state_dict"])

    _eval_seed = (
        int(args.val_seed) if args.eval_seed is None else int(args.eval_seed)
    )
    clouds = build_csv_cohort(
        args.csv, split="val", n_clouds=int(args.n_clouds),
        seed=_eval_seed,
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

    # Wf-A108c — the single-batch gate above answers "is the target inside the
    # head's function class".  It does NOT answer "does the trainer's per-step
    # gradient carry the same information", because the trainer draws 25
    # different batches per epoch at t ~ U(0,1).  --cycle-batches reproduces
    # that: each step uses the next of N cohorts (val_seed + k), and with
    # --t-uniform the interpolant time is resampled per cloud per step.
    # `--cycle-batches 0` leaves every tensor above untouched, so the
    # single-batch path stays bit-exact.
    batches: List[Dict[str, Any]] = []
    if int(args.cycle_batches) > 0:
        for k in range(int(args.cycle_batches)):
            cs = build_csv_cohort(
                args.csv, split="val", n_clouds=int(args.n_clouds),
                seed=int(args.val_seed) + k,
                metal_placement=str(getattr(ns, "metal_placement", "legacy")),
                dative_distance_cutoff=float(
                    getattr(ns, "dative_distance_cutoff", 2.6)
                ),
            )
            bx1 = torch.cat([c.positions.float() for c in cs], 0).to(dev)
            bz = torch.cat([c.atomic_numbers.long() for c in cs], 0).to(dev)
            bg = torch.Generator(device=dev).manual_seed(int(args.noise_seed) + k)
            bx0 = torch.randn(bx1.shape, generator=bg, device=dev) * sigma
            batches.append(
                {
                    "x0": bx0,
                    "x1": bx1,
                    "z": bz,
                    "sizes": [int(c.atomic_numbers.shape[0]) for c in cs],
                    "gen": bg,
                }
            )
        print(
            f"[wfa107h] CYCLE mode: {len(batches)} cohorts, "
            f"t={'U(0,1) per cloud per step' if args.t_uniform else f'fixed {t}'}"
        )

    model.train()
    opt = torch.optim.AdamW(model.vel_field.parameters(), lr=float(args.lr))
    traj: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        v0 = model.fm_velocity(xt, t_col, z, cloud_sizes=sizes)
    traj.append({"step": 0, **_stats(v0, tgt)})
    print(f"[wfa107h] step   0 {traj[-1]}")
    for step in range(1, int(args.steps) + 1):
        if not batches:
            bxt, bt, bz_, bsz, btgt = xt, t_col, z, sizes, tgt
        else:
            b = batches[(step - 1) % len(batches)]
            if args.t_uniform:
                bt = torch.rand(
                    b["x1"].shape[0], generator=b["gen"], device=dev
                )
            else:
                bt = torch.full((b["x1"].shape[0],), t, device=dev)
            bxt = (
                (1.0 - bt)[:, None] * b["x0"] + bt[:, None] * b["x1"]
            )
            btgt = b["x1"] - b["x0"]
            bz_, bsz = b["z"], b["sizes"]
        vp = model.fm_velocity(bxt, bt, bz_, cloud_sizes=bsz)
        loss = F.mse_loss(vp, btgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 40 == 0:
            # Eval is ALWAYS the fixed batch at the fixed t, so arms with
            # different training distributions remain comparable.
            with torch.no_grad():
                vp = model.fm_velocity(xt, t_col, z, cloud_sizes=sizes)
                row = {"step": step, **_stats(vp, tgt)}
            traj.append(row)
            train_rows.append({"step": step, "train_mse": float(loss)})
            print(
                f"[wfa107h] step {step:3d} {row} "
                f"train_mse={float(loss):.3f}"
            )

    last = traj[-1]
    doc = {
        "metric": "wfa107h_velocity_capacity",
        "tag": "MEASURED",
        "date": "2026-10-22",
        "what": (
            f"Overfit test of the FM velocity field on ONE 8-cloud batch: {int(args.steps)} "
            "AdamW steps at lr 1e-3, fixed t.  Compares the reachable loss "
            "with the v=0 baseline and with the best radial-only field, which "
            "is the form EGNNVelocityField's dominant term can express."
        ),
        "checkpoint": str(args.ckpt),
        "ckpt_epoch": ck.get("epoch"),
        "velocity_head": head,
        "vector_channels": (
            int(args.vector_channels) if head == "vector_channels" else 0
        ),
        "t_embedding_dim": int(getattr(ns, "t_embedding_dim", 0)),
        "metal_frame_channels": bool(args.metal_frame_channels),
        "radial_centroid_masked": bool(args.radial_centroid_masked),
        "fresh_field": fresh,
        "vel_field_params": int(
            sum(p.numel() for p in model.vel_field.parameters())
        ),
        "noise_sigma": sigma,
        "t": t,
        "cycle_batches": int(args.cycle_batches),
        "t_uniform": bool(args.t_uniform),
        "eval_seed": int(_eval_seed),
        "train_rows": train_rows,
        "n_clouds": len(clouds),
        "n_atoms": int(x1.shape[0]),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "seed": int(args.seed),
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
            "The trajectory is ALWAYS evaluated on the fixed first cohort at "
            "t=0.5, whatever the training distribution was, so arms with "
            "different --cycle-batches / --t-uniform settings are comparable.",
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
    if args.save_field:
        sf = Path(args.save_field)
        sf.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vel_field": model.vel_field.state_dict(),
                "velocity_head": head,
                "vector_channels": int(args.vector_channels),
                "t_embedding_dim": int(getattr(ns, "t_embedding_dim", 0)),
                "steps": int(args.steps),
                "cycle_batches": int(args.cycle_batches),
                "t_uniform": bool(args.t_uniform),
                "eval_seed": int(_eval_seed),
            },
            sf,
        )
        print(f"[wfa107h] saved field -> {sf}")
    print(
        f"[wfa107h] zero={zero_baseline:.3f} radial_ceiling={radial_ceiling:.3f} "
        f"reached={last['mse']:.3f} |v|={last['mean_v_norm']:.3f} "
        f"|target|={last['mean_target_norm']:.3f} -> {doc['verdict']}"
    )
    print(f"[wfa107h] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
