# SPDX-License-Identifier: Apache-2.0
"""Wf-A108d — does the trained velocity field actually depend on ``t``?

WHY THIS EXISTS
---------------
Under the trainer's sampling distribution the vector-channel field generalises,
but slowly: on a held-out cohort the zero-prediction MSE 9.008 falls to 7.267
after 500 steps, 7.033 after 2,000 and 6.713 after 10,000, while the best
RADIAL field on the same batch reaches 4.624 (Wf-A108c arms).  Steps are not
free and the channel count is not the limit (C=32 at 2,000 steps: 7.051 vs
C=8: 7.033), so something else bounds the fit.

The interpolant gives one candidate that is cheap to check.  With
``x_t = (1-t) x0 + t x1``, ``x0 ~ N(0, sigma^2 I)`` and the target
``x1 - x0``, the conditional mean flips sign across the path:

  * at ``t -> 0``, ``x_t ~ x0`` and ``E[x1 - x0 | x_t] ~ -x_t``;
  * at ``t -> 1``, ``x_t ~ x1`` and ``E[x1 - x0 | x_t] ~ +x_t``.

So a field that is even approximately correct MUST reverse direction between
the two ends.  A field whose output barely moves with ``t`` can only produce a
compromise in between, which is exactly the shape the arms report
(``|v| ~ 2`` against ``|target| ~ 4.7``, cosine ~ 0.5).

``EGNNVelocityField`` feeds raw ``t`` in ``[0, 1]`` to
``time_mlp = Linear(1, H) -> SiLU -> Linear(H, H)``.  That is a narrow input
range for a linear layer with default init, and this script measures the
consequence rather than assuming it: same coordinates, same atom types, five
values of ``t``, and the pairwise change in the predicted velocity.

Usage:
  python molmetal/scripts/wfa108d_t_sensitivity.py \
      --ckpt results/wfa108_vector_channels/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \
      --field results/wfa108c_fields/vc8_10k.pt \
      --out metrics/by_metric/wfa108d_t_sensitivity.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--csv", default=".data/molmetal/metalcytotox_5metal_multitask_v3.csv")
    ap.add_argument("--n-clouds", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=999)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="metrics/by_metric/wfa108d_t_sensitivity.json")
    args = ap.parse_args()

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_csv_cohort,
        build_joint_model,
        load_cfm_checkpoint,
    )

    ck = load_cfm_checkpoint(args.ckpt, map_location="cpu")
    ns = argparse.Namespace(**ck["args"])
    payload = torch.load(args.field, map_location="cpu", weights_only=False)
    head = str(payload["velocity_head"])
    ns.velocity_head = head
    ns.vector_channels = int(payload["vector_channels"])
    ns.t_embedding_dim = int(payload.get("t_embedding_dim", 0))
    ns.metal_frame_channels = bool(
        getattr(ns, "metal_frame_channels", False)
    )
    model = build_joint_model(ns)
    model.vel_field.load_state_dict(payload["vel_field"])
    dev = torch.device(args.device)
    model.to(dev).eval()

    clouds = build_csv_cohort(
        args.csv, split="val", n_clouds=int(args.n_clouds),
        seed=int(args.eval_seed),
        metal_placement=str(getattr(ns, "metal_placement", "legacy")),
        dative_distance_cutoff=float(getattr(ns, "dative_distance_cutoff", 2.6)),
    )
    x1 = torch.cat([c.positions.float() for c in clouds], 0).to(dev)
    z = torch.cat([c.atomic_numbers.long() for c in clouds], 0).to(dev)
    sizes = [int(c.atomic_numbers.shape[0]) for c in clouds]
    sigma = float(getattr(ns, "fm_noise_sigma", 1.0))
    g = torch.Generator(device=dev).manual_seed(int(args.noise_seed))
    x0 = torch.randn(x1.shape, generator=g, device=dev) * sigma

    t_values = [0.05, 0.25, 0.50, 0.75, 0.95]
    # Per-cloud centroid of x1.  The sign-flip reference is stated around the
    # data mean, not around the origin: at t->0 the conditional mean is
    # ``mu - x_t`` and at t->1 it is ``x_t`` (the prior's mean is 0).  Using
    # the origin for both would bias the low-t cosine by ``mu``.
    mu = torch.zeros_like(x1)
    start = 0
    for n in sizes:
        mu[start:start + n] = x1[start:start + n].mean(dim=0, keepdim=True)
        start += n
    # Closed-form Gaussian reference.  Treating (x0, x1) as jointly Gaussian
    # with x0 ~ N(0, sigma^2 I) and x1 ~ N(mu, s^2 I), the posterior mean of
    # the velocity given x_t is  c(t) (x_t - t mu)  with
    #   c(t) = (t s^2 - (1 - t) sigma^2) / ((1-t)^2 sigma^2 + t^2 s^2).
    # It is one scalar function of t applied to the input, so any field that
    # cannot beat it is losing to a one-parameter model.
    _d = (x1 - mu).reshape(-1)
    s2 = float(_d.var(unbiased=False))
    preds: Dict[float, torch.Tensor] = {}
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for tv in t_values:
            xt = (1.0 - tv) * x0 + tv * x1
            t_col = torch.full((x1.shape[0],), tv, device=dev)
            v = model.fm_velocity(xt, t_col, z, cloud_sizes=sizes)
            preds[tv] = v
            ref = mu - xt if tv < 0.5 else xt
            tgt = x1 - x0
            _num = tv * s2 - (1.0 - tv) * sigma ** 2
            _den = (1.0 - tv) ** 2 * sigma ** 2 + tv ** 2 * s2
            _c = _num / max(_den, 1e-12)
            v_gauss = _c * (xt - tv * mu)
            rows.append(
                {
                    "t": tv,
                    "mean_v_norm": float(v.norm(dim=-1).mean()),
                    "rms_v_per_coord": float(v.pow(2).mean().sqrt()),
                    "cos_to_signflip_ref": float(
                        F.cosine_similarity(
                            v.reshape(-1), ref.reshape(-1), dim=0
                        )
                    ),
                    "mse_zero": float(F.mse_loss(torch.zeros_like(tgt), tgt)),
                    "mse_network": float(F.mse_loss(v, tgt)),
                    "mse_gaussian_closed_form": float(
                        F.mse_loss(v_gauss, tgt)
                    ),
                    "gaussian_coefficient": float(_c),
                }
            )

    pairs: List[Dict[str, Any]] = []
    for i, ta in enumerate(t_values):
        for tb in t_values[i + 1:]:
            va, vb = preds[ta], preds[tb]
            # Scale-free direction change: 0 for a t-blind field, sqrt(2) for
            # two orthogonal fields of equal norm, 2 for exact sign flips.
            scale = 0.5 * (float(va.norm()) + float(vb.norm())) + 1e-12
            pairs.append(
                {
                    "t_a": ta,
                    "t_b": tb,
                    "cosine": float(
                        F.cosine_similarity(
                            va.reshape(-1), vb.reshape(-1), dim=0
                        )
                    ),
                    "relative_l2_change": float((va - vb).norm()) / scale,
                }
            )

    # A t-blind field would have relative_l2_change ~ 0 between every pair.
    # Report the endpoint pair, which is the one the interpolant requires to
    # be most different.
    ends = pairs[0] if pairs else {}
    doc = {
        "metric": "wfa108d_t_sensitivity",
        "tag": "MEASURED",
        "date": "2026-10-22",
        "what": (
            "Same coordinates and atom types, five values of t: how much does "
            "the trained velocity field's output change with t?  The "
            "interpolant's conditional mean flips sign between t->0 (-x_t) "
            "and t->1 (+x_t), so a field that is even approximately right "
            "must change direction across the path."
        ),
        "checkpoint": str(args.ckpt),
        "field": str(args.field),
        "velocity_head": head,
        "vector_channels": int(payload["vector_channels"]),
        "t_embedding_dim": int(payload.get("t_embedding_dim", 0)),
        "field_steps": int(payload.get("steps", -1)),
        "field_cycle_batches": int(payload.get("cycle_batches", -1)),
        "field_t_uniform": bool(payload.get("t_uniform", False)),
        "eval_seed": int(args.eval_seed),
        "noise_sigma": sigma,
        "n_clouds": len(clouds),
        "per_t": rows,
        "pairwise": pairs,
        "endpoint_relative_l2_change": ends.get("relative_l2_change"),
        "endpoint_cosine": ends.get("cosine"),
        "verdict": (
            "T_BLIND"
            if float(ends.get("relative_l2_change", 1.0)) < 0.5
            else "T_SENSITIVE"
        ),
        "sign_flip_achieved": bool(float(ends.get("cosine", 0.0)) < 0.0),
        "honest_limits": [
            "One held-out cohort (8 clouds) and one noise draw.  This "
            "measures the field's t-response on that draw, not a population "
            "average.",
            "relative_l2_change = ||v(t_a) - v(t_b)|| / (||v(t_a)|| ||v(t_b)||) "
            "is a scale-free direction change; it is 0 for a t-blind field and "
            "sqrt(2) for two orthogonal fields of equal norm.",
            "The sign-flip reference is the t->0 / t->1 limit of the "
            "conditional mean for a Gaussian prior, not the exact conditional "
            "mean at intermediate t.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(
        f"[wfa108d] |v| by t: "
        + ", ".join(f"{r['t']:.2f}:{r['mean_v_norm']:.3f}" for r in rows)
    )
    print(
        f"[wfa108d] endpoint t=0.05 vs t=0.95: cosine={ends.get('cosine'):.4f} "
        f"rel_l2={ends.get('relative_l2_change'):.4f} -> {doc['verdict']}"
    )
    print(f"[wfa108d] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
