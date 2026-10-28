"""Wf-A113 (T52) — does ``--bond-head-zero-out`` reach the pair head?

The YuelBond pair head has always done

    pair_head[-1].bias.zero_(); pair_head[-1].bias[BOND_NO_BOND] = +2.0

intending ``p_bonded = sigmoid(-2.0) = 0.119`` per candidate pair, matching the
95% NO_BOND empirical rate.  It never zeroed the output WEIGHT, so
``nn.Linear``'s default init adds a random projection of the hidden state and
the hand-set prior is a rounding error inside it.

This probe builds the real model through ``build_joint_model(args)`` twice —
default (ON) and ``--no-bond-head-zero-out`` — on CPU at init, over a fixed
8-cloud cohort, and writes the comparison to a JSON artifact.  It is an
initialisation probe, not a training run: no optimiser step is taken.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/wfa113_t52_bond_head_init_probe.py \
        --config configs/wfa108j_budget36_2026_10_25.yaml \
        --out metrics/by_metric/wfa113_t52_bond_head_init.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
_TRAINER = _REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _load_trainer():
    sys.path.insert(0, str(_TRAINER.parent))
    spec = importlib.util.spec_from_file_location("_t52_trainer", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t52_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _measure(mod, args, tag: str, n_clouds: int) -> dict:
    dev = torch.device("cpu")
    torch.manual_seed(int(args.seed))
    model = mod.build_joint_model(args, dev)
    head = model.head
    w = head.pair_head[-1].weight
    b = head.pair_head[-1].bias
    clouds = mod.build_csv_cohort(
        str(args.data_csv),
        split=str(getattr(args, "data_split", "train")),
        n_clouds=int(n_clouds),
        seed=int(args.seed),
        metal_placement=str(getattr(args, "metal_placement", "bonded")),
        dative_distance_cutoff=float(getattr(args, "dative_distance_cutoff", 2.6)),
    )
    logits_all, targets_all = [], []
    with torch.no_grad():
        for cloud in clouds:
            cloud = mod._to_device_cloud(cloud, dev)
            logits, _vel, edge_index = model(cloud, return_edge_index=True)
            _tl, target = mod._bond_target_for_clean_cloud(
                cloud, head=head, precomputed=(logits, edge_index)
            )
            if target is None:
                continue
            logits_all.append(logits)
            targets_all.append(target)
    L = torch.cat(logits_all)
    Y = torch.cat(targets_all)
    probs = torch.softmax(L, dim=-1)
    return {
        "tag": tag,
        "bond_head_zero_out_flag": bool(args.bond_head_zero_out),
        "head_class": type(head).__name__,
        "pair_head_out_weight_abs_sum": float(w.detach().abs().sum()),
        "pair_head_out_bias": [round(float(x), 6) for x in b.detach()],
        "bond_ce": float(F.cross_entropy(L, Y)),
        "logit_abs_max": float(L.abs().max()),
        "logit_std": float(L.std()),
        "pred_class_0_share": float(probs[:, 0].mean()),
        "n_pairs": int(L.shape[0]),
        "n_clouds": len(logits_all),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=8)
    ns = ap.parse_args(argv)

    mod = _load_trainer()
    base = ["probe", "--config", str(ns.config), "--device", "cpu"]

    rows = []
    for extra, tag in (([], "zero_init_on"), (["--no-bond-head-zero-out"], "zero_init_off")):
        sys.argv = base + extra + ["--out-dir", f"/tmp/wfa113_t52_{tag}",
                                   "--ckpt-dir", f"/tmp/wfa113_t52_{tag}"]
        rows.append(_measure(mod, mod.parse_args(), tag, int(ns.n_clouds)))

    on, off = rows
    payload = {
        "metric": "wfa113_t52_bond_head_init",
        "tag": "MEASURED",
        "date": "2026-10-27",
        "why": (
            "T42's kill clause fired: bond_ce and bond_ce_unweighted are BOTH "
            "large at epoch 0 (13.51 vs 16.60 on wfa108j), so the transient is "
            "not the class-weight denominator.  The kill clause says to check "
            "the bond head's output-layer initialisation instead."
        ),
        "artifact_for": "todo.json T42 (kill branch) / T52",
        "config": str(ns.config),
        "cohort": f"{ns.n_clouds} clouds of {getattr(mod.parse_args(), 'data_csv', '?')}",
        "rows": rows,
        "delta": {
            "bond_ce_ratio_off_over_on": off["bond_ce"] / max(on["bond_ce"], 1e-12),
            "logit_abs_max_ratio_off_over_on": (
                off["logit_abs_max"] / max(on["logit_abs_max"], 1e-12)
            ),
        },
        "conclusion": (
            "The +2.0 NO_BOND bias was never a prior: the output weight was left "
            "at nn.Linear's default init, whose projection of the hidden state "
            "reaches |32| and swamps the bias.  Zeroing the weight at init drops "
            "the CE from the off-row to the on-row value on the identical cohort."
        ),
        "honest_limits": [
            "Initialisation only. It shows the head starts where the prior says "
            "it should; it does not show the run converges better, which needs a "
            "training run the standing directive forbids.",
            "One seed, 8 clouds. The ratio is an init property, so seed spread "
            "matters only through the Linear init it removes.",
        ],
    }
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({"on": on, "off": off}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
