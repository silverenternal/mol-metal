#!/usr/bin/env python
"""Wf-A107c — head-swap probe: is the edge_ring ring gain the HEAD or the GEOMETRY?

WHY THIS EXISTS
---------------
Wf-A107b measured the edge_ring arm's bond head on the val clouds' OWN
geometry and found nothing (ring_3 -0.002, p=0.98), while the same arm's
DECODED molecules carry a 3-ring share 0.14 lower than control (3/3 seeds,
p=0.016).  Two explanations survive that pair of facts:

  (a) the gain is real but lives off the data manifold — the head's decision
      boundary changed only where generated distances land, so a probe on
      clean geometry cannot see it;
  (b) the gain is not the head at all — the ring prior leaked into the
      velocity field, the sampler emits different geometry, and the head is
      unchanged.

The ring prior is attached to ``model.head`` only, so (b) should be
impossible by construction — but "should be impossible" is exactly the kind
of claim this project has been wrong about before.  This script settles it
by brute force: sample ONE set of clouds per arm, then run BOTH heads on
BOTH sets.  A 2x2 table where the row effect is zero and the column effect
is not says the geometry did it; the reverse says the head did it.

It also reports why (a) is plausible at all: the fraction of atom pairs
inside the head's 3.5 A cutoff, and the head-vs-head edge Jaccard, on each
geometry.  Generated clouds are denser than data clouds (the sampler starts
at N(0, I)), so the head sees a pair-distance distribution it was never
trained on.

Usage:
  python molmetal/scripts/wfa107c_head_swap_probe.py \
      --pair control=results/wfa107_control/checkpoints/...pt \
      --pair edge_ring_w1=results/wfa107_edge_ring_w1/checkpoints/...pt \
      --out metrics/by_metric/wfa107_head_swap_probe.json
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _load_arm(ckpt_path: str, device: str = "cpu"):
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_joint_model,
        load_cfm_checkpoint,
        migrate_legacy_vel_state_dict,
        restore_feature_schema,
    )

    ckpt = load_cfm_checkpoint(ckpt_path, map_location="cpu")
    args = Namespace(**ckpt["args"])
    model = build_joint_model(args, device=str(device))
    sd, migrated = migrate_legacy_vel_state_dict(
        model, ckpt["model_state_dict"], verbose=False
    )
    restore_feature_schema(model, None, ckpt, verbose=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, args, ckpt, bool(migrated)


def _edges_and_pbond(head, cloud, threshold: float):
    logits, (i_idx, j_idx) = head(cloud)
    if logits.numel() == 0:
        return [], torch.zeros(0)
    p_bond = 1.0 - torch.softmax(logits.float(), dim=-1)[:, 0]
    keep = p_bond > float(threshold)
    return list(zip(i_idx[keep].tolist(), j_idx[keep].tolist())), p_bond


def _rings_from_edges(z: Sequence[int], edges: Sequence[Tuple[int, int]]) -> List[int]:
    from rdkit import Chem

    rw = Chem.RWMol()
    for zz in z:
        rw.AddAtom(Chem.Atom(int(zz)))
    seen = set()
    for i, j in edges:
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key in seen:
            continue
        seen.add(key)
        rw.AddBond(key[0], key[1], Chem.BondType.SINGLE)
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


def _cutoff_stats(cloud, cutoff: float) -> Dict[str, float]:
    """Fraction of atom pairs inside the head's cutoff, and mean distance."""
    return _cutoff_stats_from_pos(cloud.positions.float(), cutoff)


def _cutoff_stats_from_pos(pos: torch.Tensor, cutoff: float) -> Dict[str, float]:
    """Same statistic on a bare ``(N, 3)`` tensor (Wf-A107g: the x0 cloud)."""
    pos = pos.float()
    n = int(pos.shape[0])
    if n < 2:
        return {"frac_pairs_in_cutoff": 0.0, "mean_pair_distance": 0.0}
    i, j = torch.triu_indices(n, n, offset=1)
    d = (pos[i] - pos[j]).norm(dim=-1)
    return {
        "frac_pairs_in_cutoff": float((d < cutoff).float().mean()),
        "mean_pair_distance": float(d.mean()),
    }


def _ckpt_tag(path: str) -> str:
    """Short content hash, so a cache entry can never outlive its weights."""
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def _generate_cached(model, templates, device, tag: str, cache_dir: Optional[str],
                     **kw):
    """Sample from noise, memoising to ``cache_dir`` when one is given.

    Sampling 63 clouds with rk4/100 costs ~6 min on CPU and it is a pure
    function of (weights, templates, seed).  Every probe variant after the
    first (threshold sweeps, extra heads on the same geometry) would pay that
    again, so cache the coordinates — NOT the model — and key by arm tag.
    """
    if cache_dir:
        path = Path(cache_dir) / f"gen_{tag}.pt"
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if blob.get("n_templates") == len(templates):
                print(f"[wfa107c] loaded cached samples {path}")
                return blob["clouds"]
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        generate_clouds_from_noise,
    )

    clouds = generate_clouds_from_noise(model, templates, device, **kw)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"n_templates": len(templates), "clouds": clouds},
            Path(cache_dir) / f"gen_{tag}.pt",
        )
        print(f"[wfa107c] cached samples -> {Path(cache_dir) / f'gen_{tag}.pt'}")
    return clouds


def _match_by_composition(templates, generated) -> List[Tuple[Any, Any]]:
    """Pair each generated cloud with a template of the SAME composition.

    ``generate_clouds_from_noise`` appends its output GROUPED BY ATOM COUNT,
    not in template order — so ``zip(templates, generated)`` silently pairs
    molecule i's predicted graph with molecule j's atom list.  On the first
    run that surfaced as an RDKit ``Range Error`` (edge index 22 on a 19-atom
    mol), i.e. the failure was loud here; a probe that only compared ring
    COUNTS would have reported a number instead.
    """
    buckets: Dict[Tuple[int, ...], List[Any]] = {}
    for g in generated:
        buckets.setdefault(tuple(int(v) for v in g.atomic_numbers.tolist()), []).append(g)
    pairs = []
    for t in templates:
        key = tuple(int(v) for v in t.atomic_numbers.tolist())
        pool = buckets.get(key) or []
        if not pool:
            raise SystemExit(
                f"[wfa107c] no generated cloud with composition {key} "
                f"({len(key)} atoms) — cannot pair"
            )
        pairs.append((t, pool.pop(0)))
    leftover = sum(len(v) for v in buckets.values())
    if leftover:
        print(f"[wfa107c] WARN {leftover} generated cloud(s) left unpaired")
    return pairs


def _profile(zs, pred_edges, truth_sets) -> Dict[str, Any]:
    from molmetal.eval.ring_profile import summarise_ring_sizes

    per_mol = []
    tp = fp = fn = 0
    for z, edges, truth in zip(zs, pred_edges, truth_sets):
        per_mol.append(_rings_from_edges(z, edges))
        pred_s = set((min(i, j), max(i, j)) for i, j in edges)
        tp += len(pred_s & truth)
        fp += len(pred_s - truth)
        fn += len(truth - pred_s)
    prof = summarise_ring_sizes(per_mol)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "ring_profile": prof,
        "bond": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        },
    }


def _edge_jaccard(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> float:
    sa = set((min(i, j), max(i, j)) for i, j in a)
    sb = set((min(i, j), max(i, j)) for i, j in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[],
                    help="name=path/to/checkpoint (exactly two)")
    ap.add_argument("--csv", default=".data/molmetal/metalcytotox_5metal_multitask_v3.csv")
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--gen-seed", type=int, default=0)
    ap.add_argument("--template-offset", type=int, default=0)
    ap.add_argument("--n-gen", type=int, default=0,
                    help="0 = every template after the offset")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cutoff", type=float, default=3.5)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda:0 — the head is tiny, but a TRAINED "
                         "velocity field makes the radius graph large and the "
                         "CPU sampler slow (MEASURED >20 min for 2x32 clouds "
                         "vs ~1 min on GPU0)")
    ap.add_argument("--gen-cache", default="results/wfa107c_headswap/gen_cache")
    ap.add_argument("--out", default="metrics/by_metric/wfa107_head_swap_probe.json")
    args = ap.parse_args()

    if len(args.pair) != 2:
        raise SystemExit(f"--pair must be given exactly twice, got {len(args.pair)}")
    specs = []
    for spec in args.pair:
        if "=" not in spec:
            raise SystemExit(f"--pair expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            raise SystemExit(f"[wfa107c] missing checkpoint for {name}: {path}")
        specs.append((name, path))

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_csv_cohort

    loaded = {}
    for name, path in specs:
        model, a, ckpt, migrated = _load_arm(path, args.device)
        loaded[name] = {
            "model": model, "args": a, "ckpt": ckpt, "migrated": migrated,
            "path": path,
        }
        print(f"[wfa107c] loaded {name} (epoch={ckpt.get('epoch')})")

    (name_a, _), (name_b, _) = specs
    A, B = loaded[name_a], loaded[name_b]

    templates = build_csv_cohort(
        args.csv,
        split="val",
        n_clouds=int(args.n_val),
        seed=int(args.val_seed),
        metal_placement=str(getattr(A["args"], "metal_placement", "bonded")),
        dative_distance_cutoff=float(getattr(A["args"], "dative_distance_cutoff", 2.6)),
    )
    # The trainer scores the LAST eval seed on templates[offset:offset+n].
    # Default (offset 0, n = all) samples the whole val set; pass the
    # trainer's (offset, n, seed) to probe the exact clouds behind a
    # recorded `gen_ring_*` number.  The offset>len fallback is the
    # trainer's own: `evaluate_generation_validity` reuses the first window
    # when the slide runs off the end (63 val clouds, offsets 0/32/64).
    _off = int(args.template_offset)
    _n = int(args.n_gen) if args.n_gen > 0 else len(templates)
    _pool = templates[_off:_off + _n] if _off else templates[:_n]
    if not _pool:
        _pool = templates[:_n]
    templates = _pool
    print(
        f"[wfa107c] sampling {len(templates)} templates "
        f"(offset={args.template_offset}, n={_n})"
    )

    gen_kw = dict(
        method=str(getattr(A["args"], "method", "euler")),
        n_steps=int(getattr(A["args"], "n_steps", 100)),
        seed=int(args.gen_seed),
        noise_sigma=float(getattr(A["args"], "fm_noise_sigma", 1.0)),
    )
    with torch.no_grad():
        gen_a = _generate_cached(
            A["model"], templates, args.device,
            f"{name_a}_{_ckpt_tag(A['path'])}_g{args.gen_seed}_o{args.template_offset}_n{len(templates)}",
            args.gen_cache, **gen_kw,
        )
        gen_b = _generate_cached(
            B["model"], templates, args.device,
            f"{name_b}_{_ckpt_tag(B['path'])}_g{args.gen_seed}_o{args.template_offset}_n{len(templates)}",
            args.gen_cache, **gen_kw,
        )

    # The sampler returns clouds grouped by atom count, so re-order them to
    # template order before anything is zipped with a template's bond table.
    gen_ordered = [g for _, g in _match_by_composition(templates, gen_a)]
    gen_b_ordered = [g for _, g in _match_by_composition(templates, gen_b)]

    # (b)-test: if the ring prior never reached the velocity field, the two
    # arms' samples from the same noise are bit-identical.  Report the max
    # absolute coordinate deviation rather than asserting it.
    max_dev = 0.0
    for ca, cb in zip(gen_ordered, gen_b_ordered):
        if ca.positions.shape == cb.positions.shape:
            max_dev = max(max_dev, float((ca.positions - cb.positions).abs().max()))
    print(f"[wfa107c] sampler deviation between arms: {max_dev:.3e} A")

    zs = [[int(v) for v in c.atomic_numbers.tolist()] for c in templates]
    truth_sets = [
        set((min(i, j), max(i, j)) for i, j in _truth_edges(c)) for c in templates
    ]

    # cutoff statistics on both geometries
    def _agg_cutoff(clouds) -> Dict[str, float]:
        rows = [_cutoff_stats(c, args.cutoff) for c in clouds]
        n = len(rows) or 1
        return {
            "frac_pairs_in_cutoff": sum(r["frac_pairs_in_cutoff"] for r in rows) / n,
            "mean_pair_distance": sum(r["mean_pair_distance"] for r in rows) / n,
        }

    table: Dict[str, Any] = {}
    with torch.no_grad():
        for geom_name, clouds in (
            ("truth_geometry", templates),
            ("generated_geometry", gen_ordered),
        ):
            row = {}
            preds = {}
            for head_name, arm in ((name_a, A), (name_b, B)):
                preds[head_name] = [
                    _edges_and_pbond(arm["model"].head, c, args.threshold)[0]
                    for c in clouds
                ]
                row[head_name] = _profile(zs, preds[head_name], truth_sets)
            jac = [
                _edge_jaccard(pa, pb)
                for pa, pb in zip(preds[name_a], preds[name_b])
            ]
            row["head_edge_jaccard"] = sum(jac) / (len(jac) or 1)
            row["cutoff_stats"] = _agg_cutoff(clouds)
            table[geom_name] = row
            for head_name in (name_a, name_b):
                p = row[head_name]["ring_profile"]
                b = row[head_name]["bond"]
                print(
                    f"[wfa107c] {geom_name:18s} head={head_name:13s} "
                    f"3-share={p['ring_3_share']:.3f} 5/6={p['ring_5_6_share']:.3f} "
                    f"rings/mol={p['rings_per_mol_mean']:.2f} "
                    f"bondF1={b['f1']:.3f} "
                    f"(P={b['precision']:.3f} R={b['recall']:.3f})"
                )
            print(
                f"[wfa107c] {geom_name:18s} head-edge Jaccard={row['head_edge_jaccard']:.3f} "
                f"pairs-in-cutoff={row['cutoff_stats']['frac_pairs_in_cutoff']:.3f} "
                f"mean_d={row['cutoff_stats']['mean_pair_distance']:.2f}"
            )

    # 2x2 effects: row = geometry, column = head.
    def _cell(geom, head, key, sub):
        return table[geom][head][key][sub]

    # Wf-A107g (2026-10-22) — the sampler's STARTING cloud, so the geometry
    # numbers above can be read as a signed transport instead of an absolute
    # distance.  `mean_pair_distance` of a converged sampler should sit near
    # `truth_geometry`; a value near this block means the ODE barely moved the
    # cloud, which is what a prior/data scale mismatch produces.
    #
    # Distributional, not the exact draw `_generate_cached` used: x0 is iid
    # N(0, sigma^2 I) per atom, so the pair-distance statistic depends only on
    # the atom counts and sigma.  The label says so.
    _x0_sigma = float(getattr(A["args"], "fm_noise_sigma", 1.0))
    _g0 = torch.Generator().manual_seed(int(args.gen_seed))
    _x0_rows = []
    for c in templates:
        n = int(c.atomic_numbers.shape[0])
        _p = torch.randn(n, 3, generator=_g0) * _x0_sigma
        _x0_rows.append(_cutoff_stats_from_pos(_p, args.cutoff))
    x0_reference = {
        "noise_sigma": _x0_sigma,
        "frac_pairs_in_cutoff": (
            sum(r["frac_pairs_in_cutoff"] for r in _x0_rows) / (len(_x0_rows) or 1)
        ),
        "mean_pair_distance": (
            sum(r["mean_pair_distance"] for r in _x0_rows) / (len(_x0_rows) or 1)
        ),
        "basis": "iid N(0, sigma^2 I) at the run's fm_noise_sigma; "
                 "distributional, not the exact draw the sampler used",
    }
    print(
        f"[wfa107c] x0_geometry        sigma={_x0_sigma:.3f} "
        f"pairs-in-cutoff={x0_reference['frac_pairs_in_cutoff']:.3f} "
        f"mean_d={x0_reference['mean_pair_distance']:.2f}"
    )

    effects = {}
    for metric, sub in (
        ("ring_3_share", "ring_profile"),
        ("ring_5_6_share", "ring_profile"),
        ("rings_per_mol_mean", "ring_profile"),
        ("bond_f1", "bond"),
    ):
        key = "f1" if metric == "bond_f1" else metric
        block = "bond" if metric == "bond_f1" else "ring_profile"
        def _g(geom, head):
            return table[geom][head][block][key]
        effects[metric] = {
            "head_effect_on_truth": _g("truth_geometry", name_b) - _g("truth_geometry", name_a),
            "head_effect_on_generated": _g("generated_geometry", name_b) - _g("generated_geometry", name_a),
            "geometry_effect_head_a": _g("generated_geometry", name_a) - _g("truth_geometry", name_a),
            "geometry_effect_head_b": _g("generated_geometry", name_b) - _g("truth_geometry", name_b),
        }
        print(
            f"[wfa107c] effect {metric:20s} "
            f"head|truth={effects[metric]['head_effect_on_truth']:+.4f} "
            f"head|gen={effects[metric]['head_effect_on_generated']:+.4f} "
            f"geom|A={effects[metric]['geometry_effect_head_a']:+.4f} "
            f"geom|B={effects[metric]['geometry_effect_head_b']:+.4f}"
        )

    doc = {
        "metric": "wfa107_head_swap_probe",
        "tag": "MEASURED",
        "date": "2026-10-21",
        "what": (
            "2x2 head-swap: both arms' bond heads run on BOTH the val clouds' "
            "own geometry and the sampler's generated geometry (same noise "
            "seed, same compositions, threshold 0.5).  Separates 'the head "
            "changed' from 'the geometry changed'."
        ),
        "arms": [name_a, name_b],
        "checkpoints": {name: loaded[name]["path"] for name, _ in specs},
        "epochs": {name: loaded[name]["ckpt"].get("epoch") for name, _ in specs},
        "sampler": gen_kw,
        "n_clouds": len(templates),
        "threshold": float(args.threshold),
        "sampler_max_abs_deviation": max_dev,
        "table": table,
        "effects": effects,
        # Wf-A107g — the starting cloud, so `generated_geometry` can be read as
        # a signed transport (near x0 = the ODE did not move; near truth = it
        # did) rather than an absolute distance.
        "x0_reference": x0_reference,
        "honest_limits": [
            "Single checkpoint per arm (last epoch); this is a mechanism probe, "
            "not a 3-seed test.",
            "Ring perception is SSSR on an un-sanitised RWMol built from the "
            "predicted edges; no valence or dative-bond handling.",
            "bond precision/recall on generated geometry compares against the "
            "template's bond table, so it is only meaningful if the sampler "
            "reproduced the composition's connectivity — read it next to the "
            "cutoff statistics, not on its own.",
            "The sampler deviation column is the leak test: non-zero means the "
            "ring prior reached the velocity field and the 2x2 is confounded.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"[wfa107c] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
