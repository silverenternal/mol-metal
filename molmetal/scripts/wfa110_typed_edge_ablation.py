"""Wf-A110 (MOD1) — does the typed-edge path actually move the velocity field?

cfm_todo MOD1's ``done_when`` has three parts, and this measures the first:
*an on/off ablation of the typed-edge path changes the velocity field's output
on identical inputs (M2-style perturbation, >= 1e-2)*.

The other two — ``single_fragment_rate`` and ``three_ring_share`` — are
generation metrics and live in the run's own ``gen_eval`` events.

WHY A SEPARATE SCRIPT
---------------------
A zero-initialised projection makes the typed and untyped fields identical at
step 0, so a fresh model cannot answer this question and a unit test cannot
either: it can only prove the wiring, which
``molmetal/tests/test_typed_edges.py`` already does.  The question is whether
TRAINING moved the projection far enough to steer the field, and that needs a
trained checkpoint.

The measurement is M2's, applied to a new channel: hold every input fixed,
change only the bond order the field is shown, and report
``||v(order) - v(no order)|| / ||v(no order)||``.  Below 1e-2 the path is
connected but cannot steer the field by more than a percent, which for a
conditioning channel is the same practical failure M2 exists to catch.

Writes ``metrics/by_metric/wfa110_typed_edge_ablation.json``.

    .venv/bin/python molmetal/scripts/wfa110_typed_edge_ablation.py \\
        --run-dir results/wfa110_typed_edges_smoke \\
        --ckpt results/wfa110_typed_edges_smoke/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "metrics" / "by_metric" / "wfa110_typed_edge_ablation.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def build_lut_from_clouds(
    clouds: Sequence[Any], k: int
) -> Optional[torch.Tensor]:
    """``(n_clouds, A_max, A_max)`` class index per intra-cloud atom pair.

    ``AtomCloud.bonds`` is already in per-cloud local coordinates, so this is
    a direct scatter with no remapping — unlike the trainer's version, which
    has to undo the flat concatenation.
    """
    sizes = [int(c.atomic_numbers.shape[0]) for c in clouds]
    if not sizes:
        return None
    a_max = max(sizes)
    lut = torch.zeros((len(clouds), a_max, a_max), dtype=torch.long)
    wrote = 0
    for ci, c in enumerate(clouds):
        bonds = getattr(c, "bonds", None)
        types = getattr(c, "bond_types", None)
        if bonds is None or types is None or bonds.numel() == 0:
            continue
        s = sizes[ci]
        src, dst = bonds[0].long(), bonds[1].long()
        ok = (src >= 0) & (src < s) & (dst >= 0) & (dst < s)
        src, dst = src[ok], dst[ok]
        cls = types.long()[ok].clamp(0, k - 1)
        lut[ci, src, dst] = cls
        lut[ci, dst, src] = cls
        wrote += int(ok.sum())
    if wrote == 0:
        return None
    return lut


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None, help="run directory (artifact lookup)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-clouds", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    from molmetal.scripts.cfm_module_gates import (
        _build_clouds,
        _build_model,
        _flatten,
        _load_ckpt,
    )

    t0 = time.time()
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"--ckpt {ckpt_path} is not a file")
    device = torch.device(args.device)

    ck = _load_ckpt(ckpt_path)
    model, ns, migrated = _build_model(ck, device)
    clouds = _build_clouds(ns, int(args.n_clouds), int(args.seed))
    if not clouds:
        raise SystemExit("cohort produced 0 clouds")
    x1, z, sizes = _flatten(clouds, device)

    k = int(getattr(getattr(model, "vel_field", None), "_typed_edge_k", 0) or 0)
    if k <= 0:
        raise SystemExit(
            "this checkpoint's velocity field has no typed-edge path "
            "(vel_field._typed_edge_k == 0).  Nothing to ablate."
        )

    lut = build_lut_from_clouds(clouds, k)
    if lut is None:
        raise SystemExit("the cohort carries no bond types")

    # ---- the projection's own magnitude: did it leave zero at all? ---------
    sd = ck.get("model_state_dict") or ck
    proj = [
        (key, sd[key].float())
        for key in sd
        if key.endswith("edge_type_mlp.weight")
    ]
    proj_norm = {key: float(w.norm()) for key, w in proj}
    projection_moved = bool(proj) and all(v > 0.0 for v in proj_norm.values())

    sigma = float(getattr(ns, "fm_noise_sigma", 1.0))
    g = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1)
    x0 = (torch.randn(x1.shape, generator=g) * sigma).to(device)

    rows: List[Dict[str, Any]] = []
    for tv in (0.05, 0.25, 0.50, 0.75, 0.95):
        xt = ((1.0 - tv) * x0 + tv * x1).to(device)
        t_atom = torch.full((int(x1.shape[0]),), float(tv), device=device)
        with torch.no_grad():
            v_none = model.fm_velocity(xt, t_atom, z, cloud_sizes=sizes)
            v_typed = model.fm_velocity(
                xt, t_atom, z, cloud_sizes=sizes,
                bond_type_lut=lut,
                bond_type_mask=torch.ones(len(clouds), device=device),
            )
            # Control for the mask machinery: masking everything must give the
            # untyped field exactly, or the ablation is measuring the mask.
            v_masked = model.fm_velocity(
                xt, t_atom, z, cloud_sizes=sizes,
                bond_type_lut=lut,
                bond_type_mask=torch.zeros(len(clouds), device=device),
            )
        rows.append({
            "t": float(tv),
            "rel_change_typed_vs_none": _rel(v_typed, v_none),
            # Absolute change per atom.  The relative number is diluted by the
            # field's overall magnitude, and at these checkpoints |v| runs 60
            # to 145 against a target norm near 5 — so a constant absolute
            # perturbation reads as a shrinking relative one.  Both are
            # recorded because the done_when asks for the relative form.
            "abs_change_per_atom": float((v_typed - v_none).norm())
            / max(int(x1.shape[0]), 1),
            "rel_change_masked_vs_none": _rel(v_masked, v_none),
            "v_none_norm": float(v_none.norm()),
            "v_typed_norm": float(v_typed.norm()),
        })
        print(
            f"[typed-ablation] t={tv:.2f} typed={rows[-1]['rel_change_typed_vs_none']:.4f} "
            f"masked={rows[-1]['rel_change_masked_vs_none']:.2e} "
            f"|v|={rows[-1]['v_none_norm']:.3f} "
            f"abs/atom={rows[-1]['abs_change_per_atom']:.4f}",
            flush=True,
        )

    best = max(r["rel_change_typed_vs_none"] for r in rows)
    worst_mask = max(r["rel_change_masked_vs_none"] for r in rows)
    threshold = 1e-2

    doc = {
        "script": "molmetal/scripts/wfa110_typed_edge_ablation.py",
        "module": "MOD1",
        "task": "T5",
        "addresses": ["D1"],
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "device": str(device),
        "n_clouds": len(clouds),
        "n_atoms": int(x1.shape[0]),
        "typed_edge_classes": k,
        "migrated_keys": migrated,
        "projection_norm_per_layer": proj_norm,
        "projection_left_zero_during_training": projection_moved,
        "per_t": rows,
        "max_rel_change": best,
        "max_abs_change_per_atom": max(r["abs_change_per_atom"] for r in rows),
        "max_rel_change_masked_control": worst_mask,
        "threshold": threshold,
        "ablation_passes": bool(best >= threshold and projection_moved),
        "reading": (
            "Training moved the typed projection off zero and the ablation "
            f"changes the velocity by up to {best:.1%} on identical inputs, "
            "so the order reaches the coordinates.  This is a WIRING result: "
            "it says the channel is live, not that typed edges improve "
            "generation."
            if best >= threshold and projection_moved else
            "The typed path is present but does not steer the field above the "
            f"1% threshold (max {best:.2%}), or its projection never left "
            "zero.  Either the training budget was too small to move a "
            "zero-init projection, or the order carries nothing the field can "
            "use."
        ),
        "wall_time_s": round(time.time() - t0, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(
        f"[typed-ablation] max={best:.4f} (threshold {threshold}) "
        f"projection_moved={projection_moved} passes={doc['ablation_passes']} "
        f"-> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
