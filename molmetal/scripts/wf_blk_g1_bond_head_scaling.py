# SPDX-License-Identifier: Apache-2.0
"""BLK-G1 — is the CFM bond head data-starved?

WHAT THIS ASKS
-------------
The CFM trainer builds its cohort with ``n_clouds=args.n_train`` and
``DEFAULT_TRAIN_CLOUDS = 256``.  MEASURED on this host: tmQM holds 33,429
complexes across the six target metals (Pt 8275 / Ru 9800 / Ir 5232 / Rh 4640 /
Re 4172 / Os 1310), and the 2026-10-09 production runs record
``"n_train": 256``.  So the bond head trains on **0.77% of the labelled
corpus sitting on disk**, and ``load_tmqm_clouds`` reservoir-samples across the
whole corpus, so the rest is not excluded — it is simply never drawn.

BLK-G1 as originally written proposed to PRETRAIN a separate encoder and
transfer it.  That premise is wrong: the labels are already connected to the
training path (the same ``_bond_target_for_clean_cloud`` reads them every
step).  The only missing quantity is data volume.  So this measures the
learning curve directly instead.

Two blocker claims recorded for BLK-G1 in todo.json are refuted here and the
refutations are re-derived from disk rather than asserted:

  * "class 0 (NO_BOND) appears 0 times" — FALSE.  That count was taken over
    the truth ``.BO`` blocks, which by construction store only bonded pairs.
    The head is scored over CANDIDATE pairs and every unlisted one defaults to
    BOND_NO_BOND; ``metrics/by_metric/cfm_bond_accuracy.json`` records the
    training-cohort counts as [24378, 4460, 421, 28, 5661], i.e. NO_BOND is
    the LARGEST class at ~70%.
  * "severe imbalance, must be decided before training" — already handled.
    The trainer computes sqrt-softened inverse-frequency weights and
    renormalises on ``sum_c p_c w_c = 1``; the exponent was swept MEASURED
    (power 0.5 beats both 0.0 and 1.0 on F1 and PR-AUC).

WHY TRAIN THE HEAD ALONE
------------------------
Because the question is about the head.  Raising ``n_train`` in a CFM run
also enlarges the velocity field's training set, which confounds the
comparison.  Here the head is the only thing training, and the validation
clouds are held IDENTICAL across every arm, so the only variable is the
number of training complexes.

THE METRIC IS NOT REIMPLEMENTED HERE
------------------------------------
Targets come from ``_bond_target_for_clean_cloud`` and scores come from
``evaluate_bond_accuracy`` — both the production functions, imported.  The
only shim is a one-attribute object exposing ``.head``, because
``evaluate_bond_accuracy`` reads ``model.head`` and nothing else.

WHAT WOULD SETTLE IT
--------------------
If val bond F1 is flat from 256 to 4096, the head is capacity-limited, not
data-limited, and BLK-G1 closes as an honest-negative.  If it rises, the lever
is ``n_train`` and no pretraining machinery is needed at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: E402
    NUM_BOND_CLASSES,
    BOND_NO_BOND,
)


class _HeadOnly:
    """``evaluate_bond_accuracy`` reads ``model.head`` and calls ``model.eval``.

    That is the whole of its model coupling — it never touches the velocity
    field.  Rather than fork a 200-line metric to drop one argument, hand it
    an object with the one attribute it reads.
    """

    def __init__(self, head: torch.nn.Module) -> None:
        self.head = head

    def eval(self) -> None:
        self.head.eval()

    def train(self) -> None:
        self.head.train()


def build_head(args: argparse.Namespace) -> torch.nn.Module:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import DMPNNBondHead

    return DMPNNBondHead(
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        cutoff=float(args.edge_cutoff),
    )


def class_weights_from_counts(
    counts: torch.Tensor, power: float
) -> torch.Tensor:
    """The production weighting rule, verbatim.

    Copied rather than imported because the trainer computes it inline inside
    its 9k-line main(); there is no function to call.  The formula and both
    of its guards are reproduced exactly:

      raw = N / (K * freq)          textbook inverse frequency
      w   = raw ** power            1.0 is unusable here (250x on 28 examples)
      w  /= sum_c p_c w_c           SCALE-NEUTRAL invariant, not w.mean()

    The renormaliser matters: ``w.mean() == 1`` divides by 53.9 on these
    counts and silently deletes bond_ce from the objective.
    """
    total = counts.sum().float()
    freq = counts.float().clamp(min=1.0)
    raw = total / (NUM_BOND_CLASSES * freq)
    w = raw.pow(float(power))
    p = counts.float() / total
    w = w / (p * w).sum().clamp(min=1e-12)
    return w


def measure_counts(head: torch.nn.Module, clouds: Sequence, device) -> torch.Tensor:
    """Per-candidate-pair class counts — the same thing the loss sees."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_target_for_clean_cloud,
    )

    counts = torch.zeros(NUM_BOND_CLASSES, dtype=torch.long)
    for cloud in clouds:
        bt = getattr(cloud, "bond_types", None)
        if bt is None:
            continue
        _, target = _bond_target_for_clean_cloud(cloud.to(device), head=head)
        if target is None:
            continue
        counts += torch.bincount(
            torch.as_tensor(target).flatten().cpu(), minlength=NUM_BOND_CLASSES
        )
    return counts


def train_one_arm(
    args: argparse.Namespace,
    n_train: int,
    train_clouds: List,
    val_clouds: List,
    device: torch.device,
) -> Dict:
    """Train the head on ``n_train`` complexes; score on the fixed val set."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_target_for_clean_cloud,
        evaluate_bond_accuracy,
    )

    torch.manual_seed(int(args.seed))
    head = build_head(args).to(device)

    counts = measure_counts(head, train_clouds, device)
    if int(counts.sum()) == 0:
        raise RuntimeError(
            f"n_train={n_train}: zero labelled candidate pairs. The cohort "
            "carries no bond truth, so there is nothing to learn from."
        )
    if bool(args.class_weights):
        w = class_weights_from_counts(counts, args.class_weight_power).to(device)
    else:
        w = None
    print(
        f"[g1] n_train={n_train} class counts={counts.tolist()} "
        f"weights={None if w is None else [round(float(x), 3) for x in w.tolist()]}",
        flush=True,
    )

    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    shim = _HeadOnly(head)

    # Pairs per cloud vary, so accumulate pairs and step on a PAIR budget
    # rather than a cloud count: otherwise every arm takes the same number of
    # gradient steps and the larger arms simply see each complex once.
    pair_budget = int(args.pair_budget)
    history = []
    t0 = time.time()
    step = 0
    seen_pairs = 0
    epoch = 0
    order: List[int] = []
    while seen_pairs < pair_budget:
        if not order:
            order = list(range(len(train_clouds)))
            g = torch.Generator().manual_seed(int(args.seed) + epoch)
            order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]
        head.train()
        for idx in order:
            cloud = train_clouds[idx].to(device)
            if getattr(cloud, "bond_types", None) is None:
                continue
            logits, target = _bond_target_for_clean_cloud(cloud, head=head)
            if logits is None or target is None or logits.shape[0] == 0:
                continue
            loss = torch.nn.functional.cross_entropy(logits, target, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            opt.step()
            seen_pairs += int(target.shape[0])
            step += 1
            if seen_pairs >= pair_budget:
                break
        epoch += 1
        acc = evaluate_bond_accuracy(
            shim, val_clouds, decision=str(args.bond_decision)
        )
        row = {
            "epoch": epoch,
            "grad_steps": step,
            "pairs_seen": seen_pairs,
            "train_loss_last": float(loss.detach()),
            "bond_f1": acc["bond_f1"],
            "bond_precision": acc["bond_precision"],
            "bond_recall": acc["bond_recall"],
            "bond_exact_match": acc["bond_exact_match"],
            "bond_exact_type_rate": acc.get("bond_exact_type_rate", 0.0),
            # bond_type_macro_f1_pooled is the axis that carries the value:
            # `tmqm_bond_perception.json` measures OpenBabel at bond-existence
            # F1 0.985 (free, from geometry) but TYPE macro-F1 0.539.  A head
            # that only learns existence is worth nothing here.
            "bond_type_macro_f1_pooled": acc.get("bond_type_macro_f1_pooled", 0.0),
            "bond_pr_auc": acc.get("bond_pr_auc", 0.0),
            "n_scored": acc.get("n_scored", 0),
            "n_truth_bonds": acc.get("n_truth_bonds", 0),
            "secs": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"[g1] n_train={n_train} ep={epoch} steps={step} "
            f"F1={row['bond_f1']:.4f} P={row['bond_precision']:.4f} "
            f"R={row['bond_recall']:.4f} exact={row['bond_exact_match']:.4f} "
            f"typeF1={row['bond_type_macro_f1_pooled']:.4f} "
            f"pr_auc={row['bond_pr_auc']:.4f} ({row['secs']:.0f}s)",
            flush=True,
        )
        order = []

    return {
        "n_train": n_train,
        "n_val": len(val_clouds),
        "class_counts": counts.tolist(),
        "class_weights": None if w is None else [float(x) for x in w.tolist()],
        "n_pairs_available": len(train_clouds),
        "history": history,
        "final": history[-1] if history else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None)
    # corpus
    p.add_argument("--tmqm-dir", type=str, default=".data/molmetal/tmQM")
    p.add_argument("--tmqm-metals", type=str, default="Pt,Ru,Ir,Rh,Os,Re")
    p.add_argument("--tmqm-val-frac", type=float, default=0.2)
    p.add_argument("--n-val", type=int, default=192)
    p.add_argument("--train-sizes", type=str, default="256,1024,4096")
    # model / optimisation
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--edge-cutoff", type=float, default=3.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--pair-budget", type=int, default=200000)
    p.add_argument("--class-weights", type=int, default=1)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--bond-decision", type=str, default="argmax")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    # output
    p.add_argument("--out-dir", type=str, default="molmetal/reports/wf_blk_g1_head_scaling")
    p.add_argument("--out-metric", type=str, default="")
    args = p.parse_args(argv)
    args = apply_yaml_overrides(p, args, args.config, argv=list(argv) if argv is not None else None)

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device=cuda requested but torch.cuda.is_available() is False")

    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    metals = [s.strip() for s in str(args.tmqm_metals).split(",") if s.strip()]
    sizes = [int(s) for s in str(args.train_sizes).split(",") if s.strip()]
    n_max = max(sizes)

    # The val set is loaded ONCE and reused by every arm.  `split="val"`
    # partitions by CSD-code hash, so it is disjoint from `split="train"` by
    # construction rather than by seed — the seed-overlap leak is documented
    # at the call site in tmqm_clouds.load_tmqm_clouds.
    print(f"[g1] loading {len(metals)} metals, max arm n_train={n_max}", flush=True)
    t0 = time.time()
    val_clouds = load_tmqm_clouds(
        args.tmqm_dir, metals=metals, coord_number=None,
        n_clouds=int(args.n_val), seed=int(args.seed), split="val",
        val_frac=float(args.tmqm_val_frac),
    )
    train_pool = load_tmqm_clouds(
        args.tmqm_dir, metals=metals, coord_number=None,
        n_clouds=n_max, seed=int(args.seed), split="train",
        val_frac=float(args.tmqm_val_frac),
    )
    print(
        f"[g1] loaded val={len(val_clouds)} train_pool={len(train_pool)} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    if not val_clouds or not train_pool:
        raise SystemExit("tmQM loader returned an empty split; refusing to report")

    val_d = [c.to(device) for c in val_clouds]
    # Each arm takes a PREFIX of the same reservoir sample, so the 256-arm is
    # a strict subset of the 4096-arm.  A nested sample is what makes the
    # curve a data-scaling curve rather than five independent draws.
    arms = []
    for n in sizes:
        arms.append(
            train_one_arm(args, n, list(train_pool[:n]), val_d, device)
        )

    report = {
        "metric": "blk_g1_bond_head_scaling",
        "description": (
            "Is the CFM bond head data-starved? Trained alone on increasing "
            "prefixes of one tmQM reservoir sample, scored on one fixed "
            "held-out split. Isolates head data volume from every other "
            "variable, including the velocity field."
        ),
        "unit": "ratio (0-1)",
        "scope": "CFM DMPNNBondHead alone, tmQM six target metals, no CFM training",
        "protocol": {
            "metals": metals,
            "n_val": len(val_clouds),
            "val_frac": float(args.tmqm_val_frac),
            "split": "CSD-code hash (disjoint by construction)",
            "nested_arms": "arm n uses train_pool[:n] — arms are nested prefixes",
            "pair_budget_per_arm": int(args.pair_budget),
            "hidden_dim": int(args.hidden_dim),
            "n_layers": int(args.n_layers),
            "edge_cutoff": float(args.edge_cutoff),
            "lr": float(args.lr),
            "class_weights": bool(args.class_weights),
            "class_weight_power": float(args.class_weight_power),
            "seed": int(args.seed),
            "device": str(device),
            "metric_source": (
                "production evaluate_bond_accuracy + _bond_target_for_clean_"
                "cloud, imported unchanged"
            ),
        },
        "production_baseline": {
            "n_train": 256,
            "source": "molmetal/reports/cfm_fm_ab_2026_10_09/per_epoch_metrics.json",
            "note": (
                "what a real CFM run trains the head on today; the 256 arm "
                "here reproduces that budget with the velocity field removed"
            ),
        },
        "arms": arms,
        "status": "MEASURED_PROBE",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "head_scaling.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[g1] wrote {out_dir / 'head_scaling.json'}", flush=True)
    if args.out_metric:
        Path(args.out_metric).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_metric).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())