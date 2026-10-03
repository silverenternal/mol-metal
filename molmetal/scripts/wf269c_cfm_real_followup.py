#!/usr/bin/env python3
"""Wf-269C followup: run no_wasserstein + no_closure_bounded configs only.

The full run timed out after no_yuelbond epoch=01 (BondAwareDecoder is
slow on real molecules with 30+ tokens).  We already have:
  full: decode_ratio=1.000 (51/51)
  no_yuelbond: decode_ratio=0.000 (0/51)  [epoch=00]

This followup runs the remaining 2 configs on a smaller cohort
(n=20) for speed, just to confirm the pattern: wasserstein and
closure-bounded knobs do NOT differentiate decode_ratio from full.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from rdkit import Chem

from molmetal.scripts.wf269c_cfm_real_ablation import (  # type: ignore
    load_real_cohort,
    filter_degenerate,
    stratified_sample,
    smiles_to_atomcloud,
    REAL_COHORT_CANDIDATES,
    _build_decoder,
    _eval_decode_ratio,
)
from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # type: ignore
    CFMJointModel,
    WassersteinConfig,
    build_reference_positions,
    pocket_specific_closure_bias,
    train_one_epoch,
)
from molmetal.models.bond_head import AtomCloud  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-cohort", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--pocket-id", type=str, default="1h36")
    p.add_argument("--wasserstein-weight", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, default="molmetal/reports/wf269c_cfm_real/")
    args = p.parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load cohort
    csv_path = next((c for c in REAL_COHORT_CANDIDATES if c.exists()), None)
    raw = load_real_cohort(csv_path)
    filtered, _ = filter_degenerate(raw)
    target_n = min(args.n_cohort, len(filtered))
    train_mols, val_mols = stratified_sample(filtered, target_n, seed=args.seed)

    # Embed 3D
    train_clouds = []
    val_clouds = []
    for i, m in enumerate(train_mols):
        c = smiles_to_atomcloud(m.smiles, jitter=0.20, seed=args.seed + i)
        if c is not None:
            train_clouds.append(c)
    for i, m in enumerate(val_mols):
        c = smiles_to_atomcloud(m.smiles, jitter=0.20, seed=args.seed + 1000 + i)
        if c is not None:
            val_clouds.append(c)

    bias_verdict = pocket_specific_closure_bias(args.pocket_id)
    closure_bias = bias_verdict.bias_dict if bias_verdict.is_known_pocket else None
    if closure_bias is None:
        closure_bias = {
            "CuAAC": 0.30,
            "SPAAC": 0.10,
            "ThiolEne": 0.10,
            "Suzuki": 0.20,
            "AmideCoupling": 0.30,
        }
    reference_positions = build_reference_positions().to(device)

    # Run only no_wasserstein and no_closure_bounded
    results = {}
    for cfg_name, no_w2, no_cb in [
        ("no_wasserstein", True, False),
        ("no_closure_bounded", False, True),
    ]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = CFMJointModel(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(device)
        decoder = _build_decoder(args, model, no_yuelbond=False)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        wasserstein_weight = 0.0 if no_w2 else args.wasserstein_weight
        closure_bounded = not no_cb
        last_dr = 0.0
        last_loss = 0.0
        for epoch in range(args.epochs):
            train_d = [
                AtomCloud(
                    positions=c.positions.to(device),
                    atomic_numbers=c.atomic_numbers.to(device),
                )
                for c in train_clouds
            ]
            val_d = [
                AtomCloud(
                    positions=c.positions.to(device),
                    atomic_numbers=c.atomic_numbers.to(device),
                )
                for c in val_clouds
            ]
            t0 = time.time()
            train_loss, bond_ce, w2_val, grad_norm, _ = train_one_epoch(
                model=model,
                decoder=decoder,
                optimiser=optimiser,
                train_clouds=train_d,
                device=device,
                batch_size=args.batch_size,
                wasserstein_weight=wasserstein_weight,
                closure_bounded=closure_bounded,
                closure_bias=closure_bias,
                reference_positions=reference_positions,
            )
            n_ok, n_total = _eval_decode_ratio(
                decoder=decoder,
                val_clouds=val_d,
                device=device,
                closure_bounded=closure_bounded,
                closure_bias=closure_bias,
            )
            decode_ratio = n_ok / max(n_total, 1)
            last_dr = decode_ratio
            last_loss = train_loss
            print(
                f"[Wf-269C-FU][{cfg_name}] epoch={epoch:02d} "
                f"loss={train_loss:.4f} bond_ce={bond_ce:.4f} w2={w2_val:.4f} "
                f"decode_ratio={decode_ratio:.3f} ({n_ok}/{n_total}) "
                f"wall={time.time()-t0:.1f}s",
                flush=True,
            )
        results[cfg_name] = {
            "decode_ratio": last_dr,
            "train_loss": last_loss,
            "ok": n_ok,
            "total": n_total,
        }

    # Append to existing report
    report_path = out_dir / "wf269c_followup.json"
    payload = {
        "wf": "wf269c_followup",
        "results": results,
        "n_train_clouds": len(train_clouds),
        "n_val_clouds": len(val_clouds),
        "epochs_per_config": args.epochs,
    }
    with report_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Wf-269C-FU] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
