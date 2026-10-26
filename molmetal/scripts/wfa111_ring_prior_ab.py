# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Wf-A111 (MOD4 / T7): does the decode-time ring prior move the ring profile?

What this measures
------------------
The ring prior (:class:`RingSizePrior`) is decode-time postprocess, so its
entire claim is that an ALREADY TRAINED checkpoint produces a better ring-size
profile with it on and no retrain.  This script generates a pool of geometries
from one checkpoint, then decodes that same pool once per arm -- the only
difference between the arms is ``ring_prior`` and its ``min_ring_size`` -- and
reports the OPT-2 ring profile for each.

Why more than one prior arm
---------------------------
``min_ring_size=4`` refuses triangles only.  Real tmQM is 0.49% three-membered
AND 0.50% four-membered, so a prior that leaves four-rings alone refuses the
measured defect and allows a second one of the same size.  The arms are
therefore parameterised by ``min_ring_size`` and the interesting number is
where the freed valence lands: on four-rings (the prior is not a fix), on
five/six-rings (it is), or on nothing (it is a deletion).

Why the pool is generated once and shared
-----------------------------------------
Two calls to ``generate_clouds_from_noise`` with the same seed should produce
the same coordinates, but "should" is not a measurement.  Generating once and
decoding every arm from that one pool makes the pairing a property of the
script rather than of the sampler's determinism.  A fingerprint of the pool is
taken before and after the arms so an in-place mutation by a decoder is caught
rather than silently turning the comparison into two different pools.

Why several generation seeds
----------------------------
One pool is one draw, and a ring SHARE over ~300 rings is a noisy number.
Every seed contributes its own pool and its own paired comparison; the
headline is the ratio-of-sums over all seeds, and the per-seed values travel
with it so the spread is visible instead of being averaged away.

What it does NOT measure
------------------------
The prior can only REMOVE closures.  It cannot put a six-membered ring where
the geometry has none, so a ``ring_6_share`` that does not rise is not
evidence against the prior -- it is evidence that the freed valence did not
land on a hexagon.  The script reports the three numbers separately
(``ring_3_share``, ``ring_6_share``, ``ring_5_6_share``) so that reading is
possible.

Usage
-----
    .venv/bin/python -m molmetal.scripts.wfa111_ring_prior_ab \
        --ckpt results/wfa110_typed_edges_smoke20/checkpoints/\
cfm_yuelbond_wasserstein_seed0.pt \
        --n-clouds 256 --seeds 0,1,2 \
        --out metrics/by_metric/wfa111_ring_prior_ab.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

#: Fields averaged over generation seeds.
_SCALAR_FIELDS = (
    "decode_ratio",
    "single_fragment_ratio",
    "largest_fragment_frac_mean",
    "metal_ratio",
    "rings_per_mol_mean",
    "ring_3_share",
    "ring_6_share",
    "ring_5_6_share",
)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a trainer checkpoint.

    The trainer writes a plain pickle, not a torch zip, so ``torch.load``
    raises "Invalid magic number" on these files.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _pool_fingerprint(clouds) -> str:
    """Stable hash of every cloud's composition and coordinates."""
    h = hashlib.sha256()
    for c in clouds:
        z = c.atomic_numbers
        h.update(z.detach().to(torch.int64).cpu().numpy().tobytes())
        x = c.positions
        h.update(x.detach().to(torch.float32).cpu().numpy().round(6).tobytes())
    return h.hexdigest()[:16]


class _StatsRecordingDecoder:
    """Proxy that records the ring prior's per-decode counters.

    ``decode_clouds_and_classify`` calls only ``.decode(cloud)``, so a proxy
    keeps the trainer's decode loop (and therefore the RDKit-strict
    classification and the per-cloud alignment) untouched while making the
    prior's firing count observable.  The prior's own counters are cumulative
    over the decoder's lifetime, so the per-cloud value is a difference.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.per_decode: List[Dict[str, int]] = []
        self._last = {
            "n_considered": 0,
            "n_refused_too_small": 0,
            "n_refused_too_large": 0,
        }

    def decode(self, cloud):
        res = self._inner.decode(cloud)
        cur = self._inner.last_ring_prior_stats or {}
        self.per_decode.append(
            {
                "n_considered": int(cur.get("n_considered", 0))
                - self._last["n_considered"],
                "n_refused_too_small": int(cur.get("n_refused_too_small", 0))
                - self._last["n_refused_too_small"],
                "n_refused_too_large": int(cur.get("n_refused_too_large", 0))
                - self._last["n_refused_too_large"],
            }
        )
        self._last = {
            "n_considered": int(cur.get("n_considered", 0)),
            "n_refused_too_small": int(cur.get("n_refused_too_small", 0)),
            "n_refused_too_large": int(cur.get("n_refused_too_large", 0)),
        }
        return res


def _ring_block(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The ring numbers, with the six-membered share split out.

    ``summarise_ring_sizes`` reports 5+6 together (the OPT-2 denominator).  T7
    names the six-membered share specifically, and a combined number cannot
    answer it, so the per-size histogram is turned into a six-only share here
    -- derived from the same counts, not a second measurement.
    """
    dist = {
        int(k): int(v)
        for k, v in (summary.get("ring_size_distribution") or {}).items()
    }
    n_rings = sum(dist.values())
    return {
        "ring_3_share": float(summary.get("ring_3_share", 0.0) or 0.0),
        "ring_5_6_share": float(summary.get("ring_5_6_share", 0.0) or 0.0),
        "ring_6_share": float(dist.get(6, 0) / n_rings) if n_rings else 0.0,
        "rings_per_mol_mean": float(summary.get("rings_per_mol_mean", 0.0) or 0.0),
        "n_rings": int(n_rings),
        "ring_size_distribution": {str(k): v for k, v in sorted(dist.items())},
    }


def _shares_from_dist(dist: Dict[int, int]) -> Dict[str, Any]:
    """Ring shares computed from a histogram (ratio of sums, not mean of ratios)."""
    n = int(sum(dist.values()))
    return {
        "ring_3_share": (dist.get(3, 0) / n) if n else 0.0,
        "ring_6_share": (dist.get(6, 0) / n) if n else 0.0,
        "ring_5_6_share": ((dist.get(5, 0) + dist.get(6, 0)) / n) if n else 0.0,
        "n_rings": n,
    }


def _build_model(saved_args: Dict[str, Any], ckpt_state: Dict[str, Any]):
    """Rebuild the model exactly as the run that wrote the checkpoint did.

    Rather than hardcoding the constructor call, every key the checkpoint
    recorded is offered to ``CFMJointModel`` and filtered by its signature.
    ``load_state_dict(strict=True)`` is then the proof: a flag that changes
    the parameter set and was not passed makes the load fail loudly instead of
    decoding a mismatched architecture.
    """
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint to load")
    ap.add_argument("--tmqm-dir", default=".data/molmetal/tmQM")
    ap.add_argument("--metals", default="Pt,Ru,Ir,Rh,Os,Re")
    ap.add_argument("--n-clouds", type=int, default=256)
    ap.add_argument("--coord-number", type=int, default=4)
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument(
        "--seeds",
        default="0,1,2",
        help="comma list of generation seeds; one pool each",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--min-ring-sizes",
        default="4,5",
        help=(
            "comma list; one prior-on arm per value.  ``4`` refuses only "
            "triangles, ``5`` also refuses four-membered rings.  Real tmQM is "
            "0.49%% three-membered and 0.50%% four-membered, so both sizes are "
            "equally unjustified by the data and the question is where the "
            "freed valence lands."
        ),
    )
    ap.add_argument(
        "--max-ring-size",
        type=int,
        default=0,
        help="0 = unbounded (the default the module documents)",
    )
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
            "Refusing to A/B a mismatched model -- the numbers would not be "
            "about the checkpoint on disk.",
            file=sys.stderr,
        )
        return 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    tmqm_root = args.tmqm_dir
    if not Path(tmqm_root).is_dir():
        alt = Path(__file__).resolve().parents[2] / tmqm_root
        if alt.is_dir():
            tmqm_root = str(alt)
        else:
            print(f"FAIL: tmqm dir not found: {args.tmqm_dir}", file=sys.stderr)
            return 1

    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        decode_clouds_and_classify,
        generate_clouds_from_noise,
        summarise_decode_chemistry,
    )

    method = str(saved_args.get("method", "euler"))
    n_steps = int(saved_args.get("n_steps", 100))
    noise_sigma = float(saved_args.get("fm_noise_sigma", 1.0))
    decision = str(saved_args.get("bond_decision", "argmax"))
    threshold = float(saved_args.get("bond_decision_threshold", 0.5))
    conn_prior = str(saved_args.get("connectivity_prior", "on")).lower() not in (
        "off",
        "false",
        "0",
        "no",
    )
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    arm_specs = [("prior_off", None)] + [
        (f"prior_on_min{int(k)}", int(k))
        for k in str(args.min_ring_sizes).split(",")
        if k.strip()
    ]

    per_seed: Dict[str, Dict[str, Any]] = {}
    fingerprints: List[str] = []
    for seed in seeds:
        templates = load_tmqm_clouds(
            tmqm_root,
            metals=[m.strip() for m in args.metals.split(",") if m.strip()],
            coord_number=args.coord_number if args.coord_number >= 0 else None,
            n_clouds=args.n_clouds,
            seed=seed,
            split=args.split,
        )
        if not templates:
            print(f"FAIL: no templates loaded for seed {seed}", file=sys.stderr)
            return 1
        with torch.no_grad():
            clouds = generate_clouds_from_noise(
                model,
                list(templates),
                device,
                method=method,
                n_steps=n_steps,
                seed=seed,
                noise_sigma=noise_sigma,
            )
        fp_before = _pool_fingerprint(clouds)
        fingerprints.append(fp_before)
        print(
            f"[wfa111] seed={seed} clouds={len(clouds)} pool={fp_before}",
            flush=True,
        )

        per_seed[str(seed)] = {}
        for arm, min_size in arm_specs:
            ring_on = min_size is not None
            dec = YuelBondBondDecoder(
                head=model.head,
                hidden_dim=int(saved_args.get("hidden_dim", 128)),
                n_layers=int(saved_args.get("n_layers", 3)),
                decision=decision,
                decision_threshold=threshold,
                connectivity_prior=conn_prior,
                connectivity_prior_min_logit=float(
                    saved_args.get("connectivity_prior_min_logit", 0.0) or 0.0
                ),
                ring_prior=ring_on,
                ring_prior_min_ring_size=int(min_size) if ring_on else 4,
                ring_prior_max_ring_size=(
                    None if int(args.max_ring_size) <= 0 else int(args.max_ring_size)
                ),
            )
            proxy = _StatsRecordingDecoder(dec)
            chem_rows: List[Optional[Dict[str, Any]]] = []
            smiles: List[str] = []
            per_cloud: List[bool] = []
            reasons: Dict[str, int] = {}
            with torch.no_grad():
                decode_clouds_and_classify(
                    proxy, clouds, smiles, None, per_cloud, chem_rows, None, reasons
                )
            summary = summarise_decode_chemistry(chem_rows)
            n_ok = int(reasons.get("ok", 0)) + sum(
                int(v) for k, v in reasons.items() if k.startswith("ok_repaired:")
            )
            per_seed[str(seed)][arm] = {
                "ring_prior": ring_on,
                "min_ring_size": int(min_size) if ring_on else None,
                "n_clouds": len(clouds),
                "decode_ratio": n_ok / max(len(clouds), 1),
                "single_fragment_ratio": float(
                    summary.get("single_fragment_ratio", 0.0) or 0.0
                ),
                "largest_fragment_frac_mean": float(
                    summary.get("largest_fragment_frac_mean", 0.0) or 0.0
                ),
                "metal_ratio": float(summary.get("metal_ratio", 0.0) or 0.0),
                "n_chemistry_scored": int(
                    summary.get("n_chemistry_scored", 0) or 0
                ),
                **_ring_block(summary),
                "ring_prior_stats": {
                    "n_considered": sum(
                        d["n_considered"] for d in proxy.per_decode
                    ),
                    "n_refused_too_small": sum(
                        d["n_refused_too_small"] for d in proxy.per_decode
                    ),
                    "n_refused_too_large": sum(
                        d["n_refused_too_large"] for d in proxy.per_decode
                    ),
                    "clouds_with_a_refusal": sum(
                        1
                        for d in proxy.per_decode
                        if d["n_refused_too_small"] + d["n_refused_too_large"] > 0
                    ),
                },
                "failure_reasons": {
                    k: int(v) for k, v in reasons.items() if not k.startswith("ok")
                },
                "per_cloud_ring_sizes": [
                    list(r.get("ring_sizes", []) or []) if r else []
                    for r in chem_rows
                ],
            }
            a = per_seed[str(seed)][arm]
            print(
                f"[wfa111] seed={seed} {arm:15s} decode={a['decode_ratio']:.3f} "
                f"ring3={a['ring_3_share']:.3f} ring6={a['ring_6_share']:.3f} "
                f"ring56={a['ring_5_6_share']:.3f} "
                f"rings/mol={a['rings_per_mol_mean']:.2f} "
                f"single_frag={a['single_fragment_ratio']:.3f} "
                f"refused={a['ring_prior_stats']['n_refused_too_small']}",
                flush=True,
            )

        fp_after = _pool_fingerprint(clouds)
        if fp_after != fp_before:
            print(
                "FAIL: the generated pool changed across arms "
                f"({fp_before} -> {fp_after}); the comparison is not paired.",
                file=sys.stderr,
            )
            return 1

    # ---------------- pooled across seeds ----------------
    arms: Dict[str, Any] = {}
    for arm, _min_size in arm_specs:
        rows = [per_seed[str(s)][arm] for s in seeds]
        dist: Dict[int, int] = {}
        for r in rows:
            for k, v in r["ring_size_distribution"].items():
                dist[int(k)] = dist.get(int(k), 0) + int(v)
        pooled = {
            "ring_prior": rows[0]["ring_prior"],
            "min_ring_size": rows[0]["min_ring_size"],
            "n_seeds": len(rows),
            "n_clouds": sum(r["n_clouds"] for r in rows),
            "decode_ratio": sum(r["decode_ratio"] for r in rows) / len(rows),
            "single_fragment_ratio": sum(
                r["single_fragment_ratio"] for r in rows
            )
            / len(rows),
            "largest_fragment_frac_mean": sum(
                r["largest_fragment_frac_mean"] for r in rows
            )
            / len(rows),
            "metal_ratio": sum(r["metal_ratio"] for r in rows) / len(rows),
            "rings_per_mol_mean": sum(r["rings_per_mol_mean"] for r in rows)
            / len(rows),
            **_shares_from_dist(dist),
            "ring_size_distribution": {str(k): v for k, v in sorted(dist.items())},
            "ring_prior_stats": {
                key: sum(r["ring_prior_stats"][key] for r in rows)
                for key in (
                    "n_considered",
                    "n_refused_too_small",
                    "n_refused_too_large",
                    "clouds_with_a_refusal",
                )
            },
            "failure_reasons": {
                k: sum(r["failure_reasons"].get(k, 0) for r in rows)
                for k in sorted({k for r in rows for k in r["failure_reasons"]})
            },
            # per-seed values, so the spread is visible rather than averaged.
            "per_seed": {
                f: [round(float(r[f]), 6) for r in rows] for f in _SCALAR_FIELDS
            },
        }
        arms[arm] = pooled

    off = arms["prior_off"]
    on_arms = [k for k in arms if k != "prior_off"]

    paired: Dict[str, Any] = {}
    deltas: Dict[str, Any] = {}
    gates: Dict[str, Any] = {}
    for name in on_arms:
        n_3_down = n_3_up = n_3_same = 0
        n_6_up = n_6_down = 0
        n_rings_off = n_rings_on = 0
        for seed in seeds:
            s_off = per_seed[str(seed)]["prior_off"]["per_cloud_ring_sizes"]
            s_on = per_seed[str(seed)][name]["per_cloud_ring_sizes"]
            for a_off, a_on in zip(s_off, s_on):
                c3_off, c3_on = a_off.count(3), a_on.count(3)
                c6_off, c6_on = a_off.count(6), a_on.count(6)
                n_rings_off += len(a_off)
                n_rings_on += len(a_on)
                if c3_on < c3_off:
                    n_3_down += 1
                elif c3_on > c3_off:
                    n_3_up += 1
                else:
                    n_3_same += 1
                if c6_on > c6_off:
                    n_6_up += 1
                elif c6_on < c6_off:
                    n_6_down += 1
        on = arms[name]
        paired[name] = {
            "n_clouds": n_3_down + n_3_up + n_3_same,
            "three_ring_count_fell": n_3_down,
            "three_ring_count_rose": n_3_up,
            "three_ring_count_unchanged": n_3_same,
            "six_ring_count_rose": n_6_up,
            "six_ring_count_fell": n_6_down,
            "n_rings_off": n_rings_off,
            "n_rings_on": n_rings_on,
            "n_rings_removed": n_rings_off - n_rings_on,
        }
        deltas[name] = {
            "decode_ratio": on["decode_ratio"] - off["decode_ratio"],
            "ring_3_share": on["ring_3_share"] - off["ring_3_share"],
            "ring_6_share": on["ring_6_share"] - off["ring_6_share"],
            "ring_5_6_share": on["ring_5_6_share"] - off["ring_5_6_share"],
            "rings_per_mol_mean": on["rings_per_mol_mean"]
            - off["rings_per_mol_mean"],
            "single_fragment_ratio": on["single_fragment_ratio"]
            - off["single_fragment_ratio"],
            "largest_fragment_frac_mean": on["largest_fragment_frac_mean"]
            - off["largest_fragment_frac_mean"],
        }
        gates[name] = {
            "three_ring_share_le_0_10": bool(on["ring_3_share"] <= 0.10),
            "six_ring_share_ge_0_45": bool(on["ring_6_share"] >= 0.45),
            "single_fragment_not_below_off": bool(
                on["single_fragment_ratio"] >= off["single_fragment_ratio"]
            ),
            "note": (
                "T7's done_when names six_ring_share; the shared module "
                "reports 5+6 together, so both are recorded.  Evaluated on "
                "this pool, this checkpoint, these seeds."
            ),
        }

    report = {
        "experiment": "wfa111_ring_prior_ab",
        "kind": "MEASURED",
        "claim": (
            "inference-time postprocess; the arms differ ONLY in ring_prior "
            "(and its min_ring_size) and decode the SAME generated pool from "
            "one checkpoint with one bond decision rule"
        ),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_decode_ratio": ckpt.get("best_decode_ratio"),
        "checkpoint_provenance": {
            "config": saved_args.get("config"),
            "out_dir": saved_args.get("out_dir"),
            "seed": saved_args.get("seed"),
            "note": (
                "`checkpoints/` is git-ignored, so this artifact's numbers "
                "are reproducible only by re-running that config; the path "
                "above names the file they were measured on, not a file in "
                "the repo."
            ),
        },
        "model_kwargs": used_kwargs,
        "pool": {
            "corpus": "tmQM",
            "metals": args.metals,
            "split": args.split,
            "n_clouds_per_seed": args.n_clouds,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "coord_number": args.coord_number,
            "method": method,
            "n_steps": n_steps,
            "noise_sigma": noise_sigma,
            "fingerprints": fingerprints,
            "fingerprint_unchanged_across_arms": True,
        },
        "decode": {
            "bond_decision": decision,
            "bond_decision_threshold": threshold,
            "connectivity_prior": conn_prior,
            "ring_prior_min_ring_sizes": [
                int(k) for k in str(args.min_ring_sizes).split(",") if k.strip()
            ],
            "ring_prior_max_ring_size": (
                None if int(args.max_ring_size) <= 0 else int(args.max_ring_size)
            ),
        },
        "arms": arms,
        "per_seed": per_seed,
        "delta_on_minus_off": deltas,
        "paired_counts": paired,
        "t7_gate": gates,
        "references": {
            "real_tmqm_opt2": {
                "ring_3_share": 0.004889100882423086,
                "ring_4_share": 0.005008346768423562,
                "ring_5_6_share": 0.9692344383496303,
                "rings_per_mol": 4.193,
                "source": (
                    "metrics/by_metric/cfm_opt2_ring_content.json — 2,000 "
                    "molecules of tmQM_y.csv, 8,386 rings; 42 of them "
                    "four-membered"
                ),
            },
            "real_metalcytotox_v4": {
                "ring_3_share": 0.0001,
                "ring_5_6_share": 0.9957,
                "rings_per_mol": 6.26,
                "source": (
                    "metrics/by_metric/c7_fullgen_pool_quality.json "
                    "'reference' block — 18,137 molecules of "
                    "metalcytotox_5metal_multitask_v4.csv, 113,558 rings"
                ),
                "note": (
                    "The two real references disagree by 40x on the "
                    "three-ring share (0.49% vs 0.01%).  Both are near zero "
                    "and both are three orders of magnitude below the "
                    "generated pool, so the direction of the defect does not "
                    "depend on which is used."
                ),
            },
            "pre_prior_generated_opt2": {
                "ring_3_share": 0.8981481481481481,
                "ring_5_6_share": 0.018518518518518517,
                "source": "metrics/by_metric/wfa96_ring_prior_gate.json",
            },
            "c7_pool": {
                "ring_3_share": 0.6684,
                "ring_6_share": 0.0292,
                "ring_5_6_share": 0.0975,
                "rings_per_mol": 6.31,
                "source": "metrics/by_metric/c7_fullgen_pool_quality.json",
            },
        },
        "honest_limits": [
            "The checkpoint is a 20-epoch smoke arm, not the 18-epoch "
            "full-corpus run that produced the 66.8% c7 pool, whose "
            "checkpoint is no longer on disk.  This pool has "
            f"{off['rings_per_mol_mean']:.2f} rings per molecule against the "
            "c7 pool's 6.31, so the prior has less to refuse here than it "
            "would on that pool; the direction is measured, the magnitude is "
            "not transferable.",
            "The prior can only REMOVE closures.  A ring_6_share that stays "
            "below the gate is not evidence against the prior; it means the "
            "geometry did not offer hexagons to keep, which is the coordinate "
            "field's job (MOD2/MOD3) and the sampler's.",
            "This checkpoint was trained WITHOUT the prior, so the comparison "
            "is prior-on vs prior-off at inference.  A model trained with the "
            "prior in its decode loop could adapt its bond logits to it; that "
            "is a different experiment.",
        ],
    }

    text = json.dumps(report, indent=2)
    print(
        f"[wfa111] pooled over {len(seeds)} seed(s): "
        + " | ".join(
            f"{k} r3={v['ring_3_share']:.3f} r6={v['ring_6_share']:.3f} "
            f"r56={v['ring_5_6_share']:.3f} rings/mol={v['rings_per_mol_mean']:.2f} "
            f"frag={v['single_fragment_ratio']:.3f}"
            for k, v in arms.items()
        )
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[wfa111] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
