# SPDX-License-Identifier: Apache-2.0
"""Per-component time budget of one CFM epoch, at cfm.yaml's real settings.

WHY
---
cfm_r9_fix_v2 ran 29,681 s for 11 epochs on n_train=26715 at batch_size=16,
i.e. 1670 optimizer steps per epoch — about 45 min/epoch, and that is what
makes the full corpus expensive. Two candidate causes were measured
separately and neither is yet attributed:

  * block-diagonal batching of the D-MPNN message passing: 14.24x, equivalence
    max|delta| = 2.1e-06 (wf_cfm_step_profile.py), and
  * the duplicate head forward that train_one_epoch used to run per cloud,
    now removed: 2.05x on the head alone (test_cfm_bond_target_reuse.py).

Neither number says what fraction of a real epoch the head occupies. If the
head is 4% of a step, 14x on the head is a rounding error and the wall clock
lives somewhere else. This measures the split instead of guessing.

WHAT IT TIMES
-------------
Real modules, real configs, real cloud sizes from the tmQM cohort:

  head          YuelBondDecoderHead.forward — the message passing + pair head
  model         CFMJointModel.forward       — head PLUS the EGNN velocity field
  target        _bond_target_for_clean_cloud — the bond label construction
  eval_cloud    one cloud through evaluate_bond_accuracy's featurise

`model - head` is then the velocity field, which is the residual the head
measurement cannot see. Epoch cost is extrapolated from cfm.yaml's own
n_train / batch_size rather than from a guess.

SCOPE
-----
Micro-probe on real clouds. It does not train and writes no metric. Absolute
seconds extrapolate a single epoch's SHAPE, not a run's wall clock — concurrent
GPU work, dataloading and the once-per-epoch decode eval are all outside it.
The fractions are the transferable part.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# cfm.yaml, read here rather than duplicated so the probe cannot drift from
# the config it claims to describe.
import yaml  # noqa: E402

CFG = yaml.safe_load((REPO / "configs" / "cfm.yaml").open(encoding="utf-8"))
N_TRAIN = int(CFG["train"]["n_train"])
BATCH = int(CFG["train"]["batch_size"])
HIDDEN = int(CFG["model"]["hidden_dim"])
N_LAYERS = int(CFG["model"]["n_layers"])
N_VAL = int(CFG["data"].get("n_val", 192))
WARMUP, ITERS = 2, 6


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "cfm_trainer_profile", REPO / "molmetal" / "scripts"
        / "train_cfm_yuelbond_wasserstein.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def timeit(fn) -> float:
    for _ in range(WARMUP):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = load_trainer()

    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        CFG["data"]["tmqm_dir"],
        metals=[s.strip() for s in CFG["data"]["tmqm_metals"].split(",")],
        coord_number=None,
        n_clouds=BATCH,
        seed=0,
        split="train",
        val_frac=float(CFG["data"]["tmqm_val_frac"]),
    )
    clouds = [c.to(device) for c in clouds]
    mean_atoms = sum(int(c.atomic_numbers.shape[0]) for c in clouds) / len(clouds)

    model = T.CFMJointModel(
        hidden_dim=HIDDEN,
        n_layers=N_LAYERS,
        use_egnn_velocity=bool(getattr(CFG["model"], "use_egnn_velocity", True)),
        bond_predictor="yuelbond",
    ).to(device)
    model.train()

    head = model.head
    t_head = timeit(lambda: [head(c) for c in clouds])
    t_model = timeit(lambda: [model(c) for c in clouds])

    def target_only():
        # MUST pass precomputed. The probe originally called this without it,
        # which measures the RECOMPUTE path — whose cost is a second full head
        # forward, not label construction. That mislabelled the row as
        # "bond target construction 2.24 ms/cloud" when the real label work is
        # orders of magnitude smaller, and sent two rounds of optimisation at
        # the wrong thing (.tolist() syncs, then the decoder object build).
        for c in clouds:
            _logits, _ei = head(c)
            T._bond_target_for_clean_cloud(
                c, head=head, precomputed=(_logits, _ei)
            )

    t_target = timeit(target_only)

    per_cloud_head = t_head / len(clouds)
    per_cloud_model = t_model / len(clouds)
    per_cloud_target = t_target / len(clouds)

    steps_per_epoch = -(-N_TRAIN // BATCH)
    train_s = steps_per_epoch * t_model
    # Once per epoch: bond accuracy over the val cohort. Featurise is the part
    # that scales with cloud count; the accuracy bookkeeping is trivial.
    val_clouds = load_tmqm_clouds(
        CFG["data"]["tmqm_dir"],
        metals=[s.strip() for s in CFG["data"]["tmqm_metals"].split(",")],
        coord_number=None,
        n_clouds=min(N_VAL, 64),
        seed=0,
        split="val",
        val_frac=float(CFG["data"]["tmqm_val_frac"]),
    )
    val_clouds = [c.to(device) for c in val_clouds]
    t_eval = timeit(lambda: T.evaluate_bond_accuracy(model, val_clouds))

    # --- the REAL training step: forward + backward + optimiser ---
    # forward-only accounted for 2.4 min of a 45 min epoch. This measures the
    # thing that actually runs, so the 84% residual stops being a guess.
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def full_step():
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for c in clouds:
            logits, _v, ei = model(c, return_edge_index=True)
            _lg, y = T._bond_target_for_clean_cloud(
                c, head=model.head, precomputed=(logits, ei)
            )
            if y is not None:
                loss = loss + torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()

    t_step = timeit(full_step)

    # --- per-cloud AtomCloud rebuild: the loop reconstructs each cloud ---
    from molmetal.models.bond_head import AtomCloud

    def rebuild():
        for c in clouds:
            AtomCloud(
                positions=c.positions,
                atomic_numbers=c.atomic_numbers,
            )

    t_rebuild = timeit(rebuild)

    # --- host -> device transfer for one batch ---
    cpu_clouds = [c.to("cpu") for c in clouds]

    def h2d():
        for c in cpu_clouds:
            c.to(device)

    t_h2d = timeit(h2d)

    vel = max(per_cloud_model - per_cloud_head, 0.0)
    # target_only() must run the head first, because the reuse path needs its
    # logits — so t_target INCLUDES one head forward. Subtracting it is the
    # only honest reading; the first version of this probe reported the sum and
    # mislabelled the whole second head forward as "target construction".
    per_cloud_target_only = max(per_cloud_target - per_cloud_head, 0.0)
    rows = [
        ("bond head  YuelBondDecoderHead.forward", per_cloud_head),
        ("velocity field (model - head)", vel),
        ("bond target construction (net of head)", per_cloud_target_only),
    ]
    print()
    print(f"  n_train={N_TRAIN}  batch_size={BATCH}  steps/epoch={steps_per_epoch}")
    print(f"  hidden={HIDDEN} layers={N_LAYERS}  clouds/batch={len(clouds)} "
          f"mean_atoms/cloud={mean_atoms:.1f}")
    print()
    print(f"  {'component':<40}{'ms / cloud':>12}{'% of model.forward':>20}")
    print("  " + "-" * 72)
    for name, v in rows:
        pct = (v / (per_cloud_model + per_cloud_target_only) * 100) if per_cloud_model else 0.0
        print(f"  {name:<40}{v*1000:>12.3f}{pct:>19.1f}%")
    print(f"  {'model.forward TOTAL':<40}{per_cloud_model*1000:>12.3f}")
    print()
    print(f"  one epoch, forward only       : {train_s/60:8.1f} min")
    step_s = steps_per_epoch * t_step
    print(f"  one epoch, fwd+bwd+opt        : {step_s/60:8.1f} min   "
          f"({t_step*1000:.1f} ms/step, {steps_per_epoch} steps)")
    print(f"  one epoch, AtomCloud rebuild  : {steps_per_epoch*t_rebuild/60:8.1f} min")
    print(f"  one epoch, host->device       : {steps_per_epoch*t_h2d/60:8.1f} min")
    print(f"  one epoch, val bond eval      : {t_eval/60:8.1f} min "
          f"({len(val_clouds)} clouds)")
    print(f"  one epoch, extrapolated total : {(train_s+t_eval)/60:8.1f} min")
    print()
    print(f"  MEASURED cfm_r9_fix_v2        : {29681/11/60:8.1f} min/epoch")
    print("  (the gap is dataloading, the per-epoch generation/decode eval and "
          "concurrent GPU work — this probe times the model forward only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())