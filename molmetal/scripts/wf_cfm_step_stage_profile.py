# SPDX-License-Identifier: Apache-2.0
"""Per-stage wall time INSIDE one real CFM optimizer step.

WHY THIS EXISTS
---------------
`wf_cfm_epoch_budget.py` timed isolated module calls, and the first attempt at
timing the loop itself reported only three buckets — forward+loss, backward,
clip+step. That located the cost but not the cause: forward+loss is ~50% of
an epoch and could be the model forward, the bond-target construction, the
cross-entropy, or the assemble term. Those need four different fixes, so the
three-bucket answer was not actionable.

This calls the REAL `train_one_epoch` on REAL tmQM clouds at cfm.yaml's real
settings, with the per-stage side channel (`set_step_timing`) enabled, and
reports where each sub-stage lands. It runs a bounded number of batches rather
than a whole epoch: the stages being compared are all per-batch costs with low
variance, so 120 batches separates them just as well as 1670 and takes ~2
minutes instead of ~18.

The percentages are the transferable part. Absolute seconds extrapolate a
SHAPE, not a run's wall clock — the once-per-epoch decode eval and any
concurrent GPU work are outside `train_one_epoch` entirely.

SCOPE
-----
A micro-probe. It does not train to a checkpoint, writes no metric, and
touches no report. It reads a config and runs the trainer's own loop.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Sub-stage key -> label. Order is the order they execute in the batch loop.
STAGES = (
    ("model_fwd_s", "model.forward (head + EGNN velocity)"),
    ("bond_target_s", "bond target construction"),
    ("assemble_loss_s", "assemble loss term (walks + components)"),
    ("assemble_stay_diag_s", "assemble stay diagnostic (LOG ONLY)"),
    ("backward_s", "loss.backward()"),
    ("clip_plus_step_s", "clip_grad_norm_ + optimiser.step()"),
)


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "cfm_trainer_stage_profile",
        REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cfm_tmqm_all6.yaml",
                    help="the config whose settings the probe reproduces")
    ap.add_argument("--batches", type=int, default=400,
                    help="batches to time (each is one optimizer step). The "
                         "first run of this probe used 100 and read 3.79 "
                         "s/step against a 0.66 s/step full-epoch rate: too few "
                         "batches to amortise CUDA/cuDNN warmup, so the "
                         "default leaves room for it to be a minority.")
    # cfm_r9_fix_v2 ran with these two off their defaults, per its own
    # launcher line: `--assemble-loss-weight 0.05` on the command line and
    # wasserstein_weight 0.0 from the yaml. Reproducing them matters: the
    # assemble block is one of the stages being attributed.
    ap.add_argument("--assemble-loss-weight", type=float, default=0.05,
                    help="cfm_r9_fix_v2 ran with 0.05 on the command line")
    ap.add_argument("--wasserstein-weight", type=float, default=None,
                    help="default: the config's flow.wasserstein_weight")
    args_cli = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((REPO / args_cli.config).read_text(encoding="utf-8"))
    # Two config layouts are in use — cfm.yaml nests under `train:`,
    # cfm_tmqm_all6.yaml under `protocol:`. Resolving rather than assuming
    # keeps the probe pointed at whatever the named config actually says.
    proto = {**cfg.get("protocol", {}), **cfg.get("train", {})}
    flow = cfg.get("flow", {})
    batch = int(proto["batch_size"])
    n_clouds = batch * args_cli.batches
    w2 = (args_cli.wasserstein_weight if args_cli.wasserstein_weight is not None
          else float(flow.get("wasserstein_weight", 0.0)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = load_trainer()

    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    t0 = time.perf_counter()
    clouds = load_tmqm_clouds(
        cfg["data"]["tmqm_dir"],
        metals=[s.strip() for s in cfg["data"]["tmqm_metals"].split(",")],
        coord_number=None,
        n_clouds=n_clouds,
        seed=0,
        split="train",
        val_frac=float(cfg["data"]["tmqm_val_frac"]),
    )
    clouds = [c.to(device) for c in clouds]
    mean_atoms = sum(int(c.atomic_numbers.shape[0]) for c in clouds) / len(clouds)
    print(f"loaded {len(clouds)} clouds in {time.perf_counter() - t0:.1f}s "
          f"(mean {mean_atoms:.1f} atoms), device={device}")

    model = T.CFMJointModel(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        n_layers=int(cfg["model"]["n_layers"]),
        use_egnn_velocity=bool(cfg["model"].get("use_egnn_velocity", True)),
        # Built from the config, not hard-coded: `fm_velocity` raises at the
        # first step if the model was built without this, so a probe that
        # guessed would measure a different model than the one that ran.
        fm_objective=bool(cfg["model"].get("use_fm_objective", False)),
        max_atomic_number=int(cfg["model"].get("egnn_max_atomic_number", 100)),
        context_dropout=float(cfg["model"].get("egnn_context_dropout", 0.1)),
        bond_predictor="yuelbond",
    ).to(device)
    model.train()

    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    decoder = YuelBondBondDecoder(head=model.head)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

    T.set_step_timing(True)
    wall = time.perf_counter()
    T.train_one_epoch(
        model=model,
        decoder=decoder,
        optimiser=optimiser,
        train_clouds=clouds,
        device=device,
        batch_size=batch,
        wasserstein_weight=w2,
        closure_bounded=bool(flow.get("closure_bounded", False)),
        fm_objective=bool(cfg["model"].get("use_fm_objective", False)),
        assemble_loss_weight=args_cli.assemble_loss_weight,
    )
    wall = time.perf_counter() - wall
    # Read BEFORE disabling. `set_step_timing(False)` clears the dict, so
    # turning the timer off first and reading after returns an empty
    # measurement — which is exactly what the first run of this probe did.
    t: Dict[str, float] = T.get_step_timing()
    T.set_step_timing(False)
    total = t.get("batch_total_s", 0.0)
    n = t.get("n_batches", 0.0)

    print()
    print(f"  {'stage':<44}{'total s':>10}{'ms/step':>10}{'% of step':>11}")
    print("  " + "-" * 75)
    for key, label in STAGES:
        v = t.get(key, 0.0)
        pct = (v / total * 100) if total else 0.0
        print(f"  {label:<44}{v:>10.1f}{(v / n * 1000 if n else 0):>10.1f}{pct:>10.1f}%")
    print("  " + "-" * 75)
    print(f"  {'measured sum of timed stages':<44}{total:>10.1f}"
          f"{(total / n * 1000 if n else 0):>10.1f}{'100.0%':>11}")
    print(f"  {'wall clock of the whole call':<44}{wall:>10.1f}"
          f"{(wall / n * 1000 if n else 0):>10.1f}"
          f"{(wall / total * 100 if total else 0):>10.1f}%")
    print()
    print(f"  batches={int(n)}  batch_size={batch}  clouds={len(clouds)}")
    print("  Untimed remainder is the cross-entropy sum, the per-batch Python")
    print("  bookkeeping, and the permutation — i.e. everything between the")
    print("  stages listed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())