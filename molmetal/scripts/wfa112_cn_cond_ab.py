# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Wf-A112 (MOD5 / T8): does the CN channel steer, and by how much?

What this measures
------------------
Two different questions, deliberately kept apart.

**1. Does the request reach the field?**  ``velocity_sensitivity`` is the
M2-style statistic: on ONE fixed batch and ONE fixed noise draw, the relative
L2 distance between the velocity at ``cn = r`` and the velocity at the null
token, at five times.  It is a property of the network alone.  Below ~1e-2 the
channel is connected but cannot steer, which is exactly the state MOD1 (typed
edges) was found in.

**2. Does the DECODED molecule obey it?**  ``conditional_success`` is the
share of samples whose decoded metal has exactly the requested degree.  This
is the number T8's ``done_when`` names, and it is a property of the network
AND the decoder AND the training signal, so it is reported per
(request, guidance scale) cell rather than as one headline.

Why the request is swept instead of measured once
-------------------------------------------------
A single ``conditional_success`` at one request cannot distinguish three
states: the channel obeys, the channel is ignored, and the composition forced
the answer.  A monotone ``mean_decoded_cn`` across a request sweep is what
"obeys" looks like; a flat one is "ignored"; and the ``truth`` arm (each
template's own CN) is the reference that tells you what the composition alone
gives you.  The same sweep at several guidance scales separates "the request
is trained in" from "the request is amplified at sampling time".

Why every arm shares one noise seed
-----------------------------------
``generate_clouds_from_noise`` reseeds its generator from ``seed``, and the
grouping is a function of the compositions, which do not change across arms.
A shared seed therefore makes the arms PAIRED: the same x0, the same
integration grid, only the request differs.  Without that, a difference of a
few percentage points on 128 samples is indistinguishable from a different
draw of the noise.

What it does NOT establish
--------------------------
Not that the model should obey, and not that 60% is reachable.  On a
checkpoint trained for a handful of epochs the honest reading is a wiring
number: it shows the request reaches the sampler and the metric can see the
answer, so that the same script on the T3 pool measures obedience rather than
plumbing.

Usage
-----
    .venv/bin/python -m molmetal.scripts.wfa112_cn_cond_ab \
        --ckpt results/wfa112_cn_cond_smoke_on_2026_10_26/checkpoints/\
cfm_yuelbond_wasserstein_seed0.pt \
        --n-clouds 128 --cn-requests truth,2,4,6,8 --cfg-scales 1.0,2.0 \
        --out metrics/by_metric/wfa112_cn_cond_ab.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

#: Times at which the velocity sensitivity is measured.  The endpoints are
#: excluded: at t=0 the field is asked to predict a displacement from pure
#: noise and at t=1 it is asked for zero, and neither is where a structural
#: condition does its work.
_SENSITIVITY_T = (0.1, 0.3, 0.5, 0.7, 0.9)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a trainer checkpoint.

    The trainer writes a plain pickle, not a torch zip, so ``torch.load``
    raises "Invalid magic number" on these files.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _build_model(saved_args: Dict[str, Any], ckpt_state: Dict[str, Any]):
    """Rebuild the model exactly as the run that wrote the checkpoint did.

    Every key the checkpoint recorded is offered to ``CFMJointModel`` and
    filtered by its signature; ``load_state_dict(strict=True)`` is the proof
    that the architecture matches.  A CN-conditioned checkpoint has an extra
    ``vel_field.cn_cond.embed.weight``, so a rebuild that forgot the flag
    fails loudly here instead of silently measuring an unconditioned model.
    """
    import inspect

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    accepted = set(inspect.signature(CFMJointModel.__init__).parameters)
    kwargs: Dict[str, Any] = {}
    for key, val in saved_args.items():
        if key in accepted and key not in ("self", "z_table"):
            kwargs[key] = val
    kwargs["z_table"] = get_z_table(str(saved_args.get("yuelbond_vocab", "pt")))
    kwargs["hidden_dim"] = int(saved_args.get("hidden_dim", 128))
    kwargs["n_layers"] = int(saved_args.get("n_layers", 3))
    kwargs["cutoff"] = float(saved_args.get("edge_cutoff", 3.5))
    kwargs["fm_objective"] = bool(saved_args.get("use_fm_objective", False))
    kwargs["use_egnn_velocity"] = bool(saved_args.get("use_egnn_velocity", False))
    kwargs["max_atomic_number"] = int(saved_args.get("egnn_max_atomic_number", 100))
    model = CFMJointModel(**kwargs)
    model.load_state_dict(ckpt_state, strict=True)
    model.eval()
    return model, sorted(kwargs)


def _mean_from_hist(hist: Dict[str, int]) -> Optional[float]:
    """Mean of an integer histogram keyed by str(int)."""
    n = sum(int(v) for v in hist.values())
    if n <= 0:
        return None
    return sum(int(k) * int(v) for k, v in hist.items()) / float(n)


def _velocity_sensitivity(
    model,
    templates,
    device: torch.device,
    requests: List[int],
    *,
    n_batch: int = 16,
    seed: int = 0,
) -> Dict[str, Any]:
    """``||v(cn=r) - v(null)|| / ||v(null)||`` at each time, on one batch.

    The batch is padded to the largest atom count in the slice and its atoms
    are drawn from N(0, 1) with a fixed generator, so the statistic is a
    function of the weights and the request only.  The radius graph is rebuilt
    once per time and SHARED by the two forwards, so the conditional and the
    null branch see the same edges -- otherwise the difference would include a
    different graph.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    pool = list(templates)[: int(n_batch)]
    if not pool:
        return {"n_batch": 0, "per_request": {}}
    n_atoms = max(int(t.atomic_numbers.shape[0]) for t in pool)
    b = len(pool)
    z_b = torch.zeros(b, n_atoms, dtype=torch.long, device=device)
    node_mask = torch.zeros(b, n_atoms, dtype=torch.bool, device=device)
    for i, t in enumerate(pool):
        n_i = int(t.atomic_numbers.shape[0])
        z_b[i, :n_i] = t.atomic_numbers.long().to(device)
        node_mask[i, :n_i] = True
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(b, n_atoms, 3, device=device, generator=gen)
    edge_index, edge_mask = _block_diag_radius_edges(x, node_mask, model.cutoff)

    out: Dict[str, Any] = {
        "n_batch": int(b),
        "n_atoms_padded": int(n_atoms),
        # ``radius_edge_index`` returns (B, 2, E_per_cloud); reading shape[1]
        # here reported "2 edges" for a graph with 862 per cloud, which would
        # have made a well-connected batch look like a point cloud.
        "edges_per_cloud": int(edge_index.shape[-1]),
        "n_edges_total": int(edge_index.shape[-1]) * int(b),
        "definition": "||v(cn=r) - v(null)|| / ||v(null)||, L2 over the whole batch, one x and one t per row",
        "per_request": {},
    }
    with torch.no_grad():
        for t_val in _SENSITIVITY_T:
            t_b = torch.full((b,), float(t_val), device=device)
            v_null = model.vel_field(
                x, z_b, edge_index, t_b, edge_mask=edge_mask, cond_cn=None
            )["vel"]
            denom = float(v_null.norm())
            row: Dict[str, Any] = {"null_norm": denom}
            for r in requests:
                req = torch.full((b,), int(r), dtype=torch.long, device=device)
                v = model.vel_field(
                    x, z_b, edge_index, t_b, edge_mask=edge_mask, cond_cn=req
                )["vel"]
                row[f"cn_{int(r)}"] = (
                    float((v - v_null).norm()) / denom if denom > 0 else None
                )
            out["per_request"][f"t={float(t_val):.1f}"] = row
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint to load")
    ap.add_argument("--tmqm-dir", default=".data/molmetal/tmQM")
    ap.add_argument("--metals", default="Pt,Ru,Ir,Rh,Os,Re")
    ap.add_argument("--n-clouds", type=int, default=128)
    ap.add_argument("--coord-number", type=int, default=-1)
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cn-requests",
        default="truth,2,4,6,8",
        help=(
            "comma list; 'truth' asks each template for its OWN coordination "
            "number (the reference arm), an integer asks every template for "
            "that number"
        ),
    )
    ap.add_argument(
        "--cfg-scales",
        default="1.0,2.0",
        help="comma list of classifier-free guidance scales; 1.0 is v_cond only",
    )
    ap.add_argument(
        "--n-steps", type=int, default=0, help="0 = the checkpoint's own value"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"FAIL: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    ckpt = _load_checkpoint(ckpt_path)
    saved_args = ckpt.get("args", {}) or {}
    state = ckpt.get("model_state_dict", ckpt)

    try:
        model, used_kwargs = _build_model(saved_args, state)
    except Exception as exc:  # noqa: BLE001
        print(
            f"FAIL: state_dict does not match the rebuilt model ({exc}).\n"
            "Refusing to measure a mismatched model -- the numbers would not "
            "be about the checkpoint on disk.",
            file=sys.stderr,
        )
        return 1

    device = torch.device(
        args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)
    has_cn = bool(getattr(model, "cn_conditioning", False))
    if not has_cn:
        print(
            "FAIL: this checkpoint has no CN conditioner (cn_conditioning is "
            "off), so a request cannot reach the field and every cell would "
            "be the null branch.  Use a checkpoint trained with "
            "--cn-conditioning.",
            file=sys.stderr,
        )
        return 1

    from molmetal.data.tmqm_clouds import load_tmqm_clouds
    from molmetal.adapters.flow_matching_lipman.conditioning import cn_from_cloud
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        evaluate_generation_validity,
    )

    tmqm_root = args.tmqm_dir
    if not Path(tmqm_root).is_dir():
        alt = Path(__file__).resolve().parents[2] / tmqm_root
        if alt.is_dir():
            tmqm_root = str(alt)
        else:
            print(f"FAIL: tmqm dir not found: {args.tmqm_dir}", file=sys.stderr)
            return 1

    templates = load_tmqm_clouds(
        tmqm_root,
        metals=[m.strip() for m in args.metals.split(",") if m.strip()],
        coord_number=args.coord_number if args.coord_number >= 0 else None,
        n_clouds=int(args.n_clouds),
        seed=int(args.seed),
        split=args.split,
    )
    if not templates:
        print("FAIL: no templates loaded", file=sys.stderr)
        return 1
    truth_cn = [cn_from_cloud(c) for c in templates]

    method = str(saved_args.get("method", "euler"))
    n_steps = int(args.n_steps) or int(saved_args.get("n_steps", 100))
    noise_sigma = float(saved_args.get("fm_noise_sigma", 1.0))
    decision = str(saved_args.get("bond_decision", "argmax"))
    threshold = float(saved_args.get("bond_decision_threshold", 0.5))
    conn_prior = str(saved_args.get("connectivity_prior", "on")).lower() not in (
        "off",
        "false",
        "0",
        "no",
    )
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        n_layers=int(saved_args.get("n_layers", 3)),
        decision=decision,
        decision_threshold=threshold,
        connectivity_prior=conn_prior,
        connectivity_prior_min_logit=float(
            saved_args.get("connectivity_prior_min_logit", 0.0) or 0.0
        ),
    )

    request_specs: List[Any] = []
    for tok in str(args.cn_requests).split(","):
        tok = tok.strip()
        if not tok:
            continue
        request_specs.append("truth" if tok == "truth" else int(tok))
    cfg_scales = [float(s) for s in str(args.cfg_scales).split(",") if s.strip()]

    cells: Dict[str, Any] = {}
    for cfg_scale in cfg_scales:
        for spec in request_specs:
            key = f"cfg{cfg_scale:g}__req_{spec}"
            chem: Dict[str, Any] = {}
            rows: List[Optional[Dict[str, Any]]] = []
            smiles: List[str] = []
            per_cloud: List[bool] = []
            with torch.no_grad():
                evaluate_generation_validity(
                    model=model,
                    decoder=decoder,
                    templates=templates,
                    device=device,
                    method=method,
                    n_steps=n_steps,
                    n_samples=len(templates),
                    # The SAME seed for every cell: same x0, same grid, so
                    # the arms are paired and only the request differs.
                    seed=int(args.seed),
                    template_offset=0,
                    noise_sigma=noise_sigma,
                    chemistry_out=chem,
                    chemistry_rows_out=rows,
                    smiles_out=smiles,
                    per_cloud_out=per_cloud,
                    cn_cfg_scale=float(cfg_scale),
                    cn_request_override=None if spec == "truth" else int(spec),
                )
            # ``evaluate_generation_validity`` reports its own success count
            # through chemistry_out; the caller's reasons dict is internal to
            # it, so reading a local one here would report 0 forever.
            decoded_hist = chem.get("decoded_cn_hist", {}) or {}
            cells[key] = {
                "cfg_scale": float(cfg_scale),
                "request": spec,
                "n_clouds": len(templates),
                "n_scored": int(chem.get("conditional_n", 0) or 0),
                "decode_ratio": float(chem.get("decode_ratio", 0.0) or 0.0),
                "metal_retention": float(chem.get("metal_ratio", 0.0) or 0.0),
                "single_fragment_ratio": float(
                    chem.get("single_fragment_ratio", 0.0) or 0.0
                ),
                "conditional_success": float(
                    chem.get("conditional_success", 0.0) or 0.0
                ),
                "requested_cn_hist": chem.get("requested_cn_hist", {}) or {},
                "decoded_cn_hist": decoded_hist,
                "mean_decoded_cn": _mean_from_hist(decoded_hist),
                "failure_reasons": dict(chem.get("failure_reasons", {}) or {}),
            }
            print(
                f"[wfa112] {key}: success={cells[key]['conditional_success']:.3f} "
                f"n={cells[key]['n_scored']} "
                f"mean_cn={cells[key]['mean_decoded_cn']} "
                f"decode_ratio={cells[key]['decode_ratio']:.3f}",
                flush=True,
            )

    # Dose-response: mean decoded CN per request, per guidance scale.  A flat
    # row means the channel is ignored; a monotone one means it steers.
    dose: Dict[str, Any] = {}
    for cfg_scale in cfg_scales:
        row = {}
        for spec in request_specs:
            key = f"cfg{cfg_scale:g}__req_{spec}"
            row[str(spec)] = cells[key]["mean_decoded_cn"]
        vals = [v for v in row.values() if v is not None]
        row["range"] = (max(vals) - min(vals)) if len(vals) >= 2 else None
        dose[f"cfg{cfg_scale:g}"] = row

    sens_requests = sorted(
        {int(s) for s in request_specs if s != "truth"}
        | {int(v) for v in truth_cn if int(v) > 0}
    )
    sensitivity = _velocity_sensitivity(
        model, templates, device, sens_requests, seed=int(args.seed)
    )

    payload: Dict[str, Any] = {
        "metric": "wfa112_cn_cond_ab",
        "tag": "MEASURED",
        "date": "2026-10-26",
        "what": (
            "Wf-A112 (MOD5/T8): does a requested coordination number reach the "
            "velocity field (velocity_sensitivity) and does the decoded metal "
            "obey it (conditional_success), across a request sweep and two "
            "guidance scales, on one paired noise draw."
        ),
        "checkpoint": str(ckpt_path),
        "checkpoint_args_used": used_kwargs,
        "checkpoint_cn_conditioning": has_cn,
        "checkpoint_epoch": ckpt.get("epoch"),
        "sampler": {
            "method": method,
            "n_steps": n_steps,
            "noise_sigma": noise_sigma,
            "seed": int(args.seed),
            "pairing": "every cell shares the generation seed, hence x0 and the integration grid",
        },
        "pool": {
            "split": args.split,
            "metals": args.metals,
            "n_clouds": len(templates),
            "truth_cn_hist": {
                str(k): truth_cn.count(k) for k in sorted(set(truth_cn))
            },
            "truth_cn_mean": (sum(truth_cn) / len(truth_cn)) if truth_cn else None,
        },
        "cells": cells,
        "dose_response": dose,
        "velocity_sensitivity": sensitivity,
        "honest_limits": [
            "conditional_success counts a decode whose metal degree equals the "
            "request; a decode that failed is excluded via conditional_n, so "
            "read n_scored beside it.",
            "metal_retention is the share of decoded molecules that contain a "
            "metal at all; it does not check that it is the REQUESTED metal.",
            "A single conditional_success cell is GAMEABLE.  MEASURED on the "
            "7-epoch smoke checkpoint: request 2 scores 0.750 while requests "
            "4 and 6 score 0.062 and 0.031, and the decoded mean CN is 2.13 "
            "in all three cells.  The model emits a metal with ~2 bonds no "
            "matter what is asked, so 'request the modal CN' passes a 60% bar "
            "on a channel that does not steer.  The gate has to be the "
            "dose-response range across requests, not one cell.",
            "This script does not establish that the model should obey a "
            "request, only whether it does at this checkpoint.",
        ],
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"[wfa112] wrote {out_path}", flush=True)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
