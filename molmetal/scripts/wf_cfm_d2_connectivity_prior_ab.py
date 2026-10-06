# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Round-3 D2 A/B: connectivity prior ON vs OFF on a REAL checkpoint.

What this measures
------------------
The connectivity prior (:meth:`YuelBondBondDecoder._bridge_components`) is
POSTPROCESS in the decoder.  Its entire claim is therefore that an ALREADY
TRAINED checkpoint decodes better with it on, with no retrain.  That claim is
only worth anything if it is measured on a trained checkpoint decoding REAL
geometry -- not on the synthetic Pt-click clouds, where everything is one
fragment already and the prior is a structural no-op.

This script loads a checkpoint, decodes the same validation clouds twice (the
only difference is ``connectivity_prior``), and reports:

  * ``decode_ratio``            — RDKit-sanitised fraction
  * ``single_fragment_ratio``   — fraction of decodes that are ONE component
  * ``largest_fragment_frac``   — mean size of the biggest heavy-atom fragment

and the per-decode count of bonds the prior actually added, so a null result
(prior on, added 0) is distinguishable from a result that fails to move the
metric for a reason other than the prior never firing.

Honest framing
--------------
Every number printed is MEASURED on the run it names, and the script refuses
to print a result for a checkpoint it did not load.  A `single_fragment_ratio`
that does not move is reported as such; this script has no path that turns a
null into a positive.  The checkpoint's own training history (including the
loss values quoted in the D2 rationale) is NOT re-derived here -- this measures
the INFERENCE-TIME effect and nothing else, and says so in its own output.

Usage
-----
    .venv/bin/python -m molmetal.scripts.wf_cfm_d2_connectivity_prior_ab \
        --ckpt results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \
        --n-clouds 128 --out results/cfm_tmqm_all6/d2_connectivity_prior_ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from molmetal.models.bond_head import BOND_NO_BOND  # noqa: E402
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    YuelBondBondDecoder,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (  # noqa: E402
    get_z_table,
)


def _heavy_fragments(atomic_numbers: torch.Tensor, bonds) -> List[int]:
    """Connected-component sizes over heavy atoms, from a bond list.

    Same rule the trainer's `decode_chemistry` uses (H excluded: a stranded H
    is a valence detail, not a separate molecule), so the numbers here are
    comparable with the per-epoch `single_fragment_ratio` in the run's own
    metrics rather than being a second, subtly different definition.
    """
    z = [int(v) for v in atomic_numbers.tolist()]
    parent = list(range(len(z)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _o in bonds:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    sizes: Dict[int, int] = {}
    for i in range(len(z)):
        if z[i] == 1:
            continue
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return list(sizes.values())


def _decode_arm(
    decoder: YuelBondBondDecoder,
    clouds: List[Any],
) -> Dict[str, Any]:
    """Decode every cloud once and summarise the fragment statistics."""
    n_ok = 0
    n_single = 0
    n_heavy_total = 0
    largest_frac_sum = 0.0
    n_scored = 0
    n_added = 0
    n_with_added = 0
    per_cloud = []
    for cloud in clouds:
        res = decoder.decode(cloud)
        decoded = res.decoded
        n_added += int(getattr(res, "bonds_added_connectivity", 0))
        if int(getattr(res, "bonds_added_connectivity", 0)) > 0:
            n_with_added += 1
        frags = _heavy_fragments(cloud.atomic_numbers, decoded.bond_orders)
        n_heavy = len(cloud.atomic_numbers.tolist()) - int(
            (cloud.atomic_numbers == 1).sum().item()
        )
        n_heavy_total += n_heavy
        if frags:
            n_scored += 1
            largest = max(frags)
            largest_frac_sum += (largest / n_heavy) if n_heavy else 0.0
            if len(frags) == 1:
                n_single += 1
        if bool(getattr(decoded, "sanitized", False)):
            n_ok += 1
        per_cloud.append(
            {
                "n_fragments": len(frags),
                "largest_fragment_frac": (max(frags) / n_heavy)
                if (frags and n_heavy)
                else 0.0,
                "sanitized": bool(getattr(decoded, "sanitized", False)),
                "bonds": int(len(decoded.bond_orders)),
                "connectivity_bonds_added": int(
                    getattr(res, "bonds_added_connectivity", 0)
                ),
            }
        )
    n = len(clouds)
    return {
        "n_clouds": n,
        "decode_ratio": (n_ok / n) if n else 0.0,
        "single_fragment_ratio": (n_single / n) if n else 0.0,
        "largest_fragment_frac_mean": (largest_frac_sum / n_scored)
        if n_scored
        else 0.0,
        "mean_heavy_atoms": (n_heavy_total / n) if n else 0.0,
        "connectivity_bonds_added_total": n_added,
        "clouds_with_bonds_added": n_with_added,
        "per_cloud": per_cloud,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint to load")
    ap.add_argument("--tmqm-dir", default=".data/molmetal/tmQM")
    ap.add_argument("--metals", default="Pt,Ru,Ir,Rh,Os,Re")
    ap.add_argument("--n-clouds", type=int, default=128)
    ap.add_argument("--coord-number", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument(
        "--connectivity-prior-min-logit",
        type=float,
        default=0.0,
        help="bridging-candidate logit bar; the A/B is meaningless if the "
        "two arms use different values, so it is recorded in the output",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"FAIL: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {}) or {}
    state = ckpt.get("model_state_dict", ckpt)

    # Rebuild the model EXACTLY as the run that wrote the checkpoint did, so
    # the state_dict loads strict=True -- a silently-mismatched architecture
    # would decode garbage and the A/B would measure the wrong thing.
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    z_table = get_z_table(str(saved_args.get("yuelbond_vocab", "pt")))
    model = CFMJointModel(
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        n_layers=int(saved_args.get("n_layers", 3)),
        cutoff=float(saved_args.get("edge_cutoff", 3.5)),
        fm_objective=bool(saved_args.get("use_fm_objective", False)),
        z_table=z_table,
        use_egnn_velocity=bool(saved_args.get("use_egnn_velocity", False)),
        max_atomic_number=int(saved_args.get("egnn_max_atomic_number", 100)),
    )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"FAIL: state_dict does not match the rebuilt model ({exc}).\n"
            "Refusing to A/B a mismatched model -- the numbers would not be "
            "about the checkpoint on disk.",
            file=sys.stderr,
        )
        return 1
    model.eval()

    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    tmqm_root = args.tmqm_dir
    if not Path(tmqm_root).is_dir():
        alt = Path(__file__).resolve().parents[2] / tmqm_root
        if alt.is_dir():
            tmqm_root = str(alt)
        else:
            print(f"FAIL: tmqm dir not found: {args.tmqm_dir}", file=sys.stderr)
            return 1
    clouds = load_tmqm_clouds(
        tmqm_root,
        metals=[m.strip() for m in args.metals.split(",") if m.strip()],
        coord_number=args.coord_number if args.coord_number >= 0 else None,
        n_clouds=args.n_clouds,
        seed=args.seed,
        split=args.split,
    )
    print(
        f"[d2-ab] checkpoint={ckpt_path} epoch={ckpt.get('epoch')} "
        f"clouds={len(clouds)} split={args.split} metals={args.metals}",
        flush=True,
    )

    decision = str(saved_args.get("bond_decision", "argmax"))
    threshold = float(saved_args.get("bond_decision_threshold", 0.5))

    arms: Dict[str, Any] = {}
    for arm, prior in (("prior_off", False), ("prior_on", True)):
        dec = YuelBondBondDecoder(
            head=model.head,
            hidden_dim=int(saved_args.get("hidden_dim", 128)),
            n_layers=int(saved_args.get("n_layers", 3)),
            decision=decision,
            decision_threshold=threshold,
            connectivity_prior=prior,
            connectivity_prior_min_logit=args.connectivity_prior_min_logit,
        )
        with torch.no_grad():
            arms[arm] = _decode_arm(dec, list(clouds))
        a = arms[arm]
        print(
            f"[d2-ab] {arm:9s} decode_ratio={a['decode_ratio']:.3f} "
            f"single_fragment_ratio={a['single_fragment_ratio']:.3f} "
            f"largest_fragment_frac={a['largest_fragment_frac_mean']:.3f} "
            f"bonds_added={a['connectivity_bonds_added_total']} "
            f"clouds_affected={a['clouds_with_bonds_added']}",
            flush=True,
        )

    off, on = arms["prior_off"], arms["prior_on"]
    report = {
        "experiment": "cfm_round3_d2_connectivity_prior_ab",
        "kind": "MEASURED",
        "claim": (
            "inference-time postprocess; the two arms differ ONLY in "
            "connectivity_prior and share one checkpoint, one decode, one "
            "bond decision rule"
        ),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_decode_ratio": ckpt.get("best_decode_ratio"),
        "corpus": {
            "name": "tmQM",
            "metals": args.metals,
            "split": args.split,
            "n_clouds": len(clouds),
            "coord_number": args.coord_number,
        },
        "bond_decision": decision,
        "bond_decision_threshold": threshold,
        "connectivity_prior_min_logit": args.connectivity_prior_min_logit,
        "arms": arms,
        "delta_on_minus_off": {
            "decode_ratio": on["decode_ratio"] - off["decode_ratio"],
            "single_fragment_ratio": (
                on["single_fragment_ratio"] - off["single_fragment_ratio"]
            ),
            "largest_fragment_frac_mean": (
                on["largest_fragment_frac_mean"]
                - off["largest_fragment_frac_mean"]
            ),
        },
        "not_measured_here": [
            "the effect of the D1 assemble-aware loss (this A/B holds the "
            "checkpoint FIXED, so it isolates D2 alone)",
            "generation from noise (every cloud here is real tmQM geometry; "
            "the prior is an inference-time step either way)",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text[:1200])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[d2-ab] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
