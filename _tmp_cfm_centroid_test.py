"""Centroid hypothesis test: minimal version.

Measures (a) real tmQM nearest-neighbour BEFORE and AFTER centroid centering,
and (b) the same on generated clouds, to test whether the bug is translation.

Does NOT run the bond decoder; that is a separate downstream question and the
hypothesis is fundamentally about the geometry produced by the velocity field.
"""

import sys
import time
import torch

REPO = "/home/hugo/codes/mol-metal"
sys.path.insert(0, REPO)

from molmetal.data.tmqm_clouds import load_tmqm_clouds
from molmetal.models.bond_head import AtomCloud
from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
    CFMJointModel,
    generate_clouds_from_noise,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table


CKPT = (
    "/home/hugo/codes/mol-metal/results/cfm_tmqm_all6/_superseded/"
    "cfm_yuelbond_wasserstein_seed0_ep9_saturating_gate.pt"
)


def nearest_neighbour(pos):
    n = pos.shape[0]
    if n < 2:
        return float("inf")
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    return float(d.min())


def stats_for(clouds, label):
    nns, c_abs, stds, spreads = [], [], [], []
    for c in clouds:
        p = c.positions.float()
        nn = nearest_neighbour(p)
        if nn != float("inf"):
            nns.append(nn)
        c_abs.append(float(p.mean(0).norm()))
        stds.append(float(p.std()))
        spreads.append(float(p.max(0).values.max() - p.min(0).values.min()))
    return {
        "label": label, "n": len(clouds),
        "nn_min": min(nns) if nns else None,
        "nn_mean": sum(nns) / len(nns) if nns else None,
        "nn_max": max(nns) if nns else None,
        "cent_norm_mean": sum(c_abs) / len(c_abs) if c_abs else None,
        "std_mean": sum(stds) / len(stds) if stds else None,
        "spread_mean": sum(spreads) / len(spreads) if spreads else None,
    }


def center_clouds(clouds):
    return [
        AtomCloud(positions=c.positions.float() - c.positions.float().mean(0),
                  atomic_numbers=c.atomic_numbers)
        for c in clouds
    ]


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] loading ckpt", flush=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    print(f"[{time.time()-t0:.1f}s] epoch={ckpt['epoch']}", flush=True)

    model = CFMJointModel(hidden_dim=128, n_layers=3,
                          z_table=get_z_table("all6"),
                          use_egnn_velocity=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[{time.time()-t0:.1f}s] model on {device}", flush=True)

    val_root = "/home/hugo/codes/mol-metal/.data/molmetal/tmQM"
    val_clouds = load_tmqm_clouds(
        val_root, metals=["Pt"], coord_number=None,
        val_frac=float(args["tmqm_val_frac"]), seed=0, split="val",
        n_clouds=32, include_hydrogens=False,
    )
    print(f"[{time.time()-t0:.1f}s] loaded {len(val_clouds)} val templates", flush=True)

    print(f"[{time.time()-t0:.1f}s] generating 32 clouds from noise", flush=True)
    gen_clouds = generate_clouds_from_noise(
        model, val_clouds, device, method="euler", n_steps=100, seed=0,
    )
    print(f"[{time.time()-t0:.1f}s] generated {len(gen_clouds)}", flush=True)

    s_real_raw = stats_for(val_clouds, "REAL_UNCENTERED")
    s_real_cent = stats_for(center_clouds(val_clouds), "REAL_CENTERED")
    s_gen_raw = stats_for(gen_clouds, "GEN_UNCENTERED")
    s_gen_cent = stats_for(center_clouds(gen_clouds), "GEN_CENTERED")

    print()
    print("== STATS ==")
    for s in [s_real_raw, s_real_cent, s_gen_raw, s_gen_cent]:
        print(f"  {s['label']:>16}: "
              f"nn_min={s['nn_min']:.3f}  nn_mean={s['nn_mean']:.3f}  "
              f"nn_max={s['nn_max']:.3f}  "
              f"|cent|={s['cent_norm_mean']:.2f}A  "
              f"std={s['std_mean']:.2f}  spread={s['spread_mean']:.2f}A",
              flush=True)

    print()
    print("== DECISIVE ==")
    print(f"  REAL nn goes from {s_real_raw['nn_mean']:.3f}A -> {s_real_cent['nn_mean']:.3f}A "
          f"after centering (UN-centered raw CSD frame: |cent|={s_real_raw['cent_norm_mean']:.2f}A, "
          f"std={s_real_raw['std_mean']:.2f})", flush=True)
    print(f"  GEN  nn goes from {s_gen_raw['nn_mean']:.3f}A -> {s_gen_cent['nn_mean']:.3f}A "
          f"after centering (UN-centered |cent|={s_gen_raw['cent_norm_mean']:.2f}A, "
          f"std={s_gen_raw['std_mean']:.2f}, min nn={s_gen_raw['nn_min']:.3f}A)", flush=True)
    print(f"  GEN  spread UN-c={s_gen_raw['spread_mean']:.2f}A  CENTERED={s_gen_cent['spread_mean']:.2f}A",
          flush=True)
    print(f"  total wall: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()