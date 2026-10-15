#!/usr/bin/env python3
"""EX1 — metal injection position: single-variable contrast (the paper's core claim).

What this measures
==================
EX1 holds the backbone, protocol, LOMO split, adaptation budget and seeds fixed
and varies ONLY where the metal enters the model:

  * ``none``    — metal-blind baseline (metal is ignored; provably, per METAL_GATE).
  * ``readout`` — C2: metal projected and concatenated to the POOLED READOUT
                  (the PA15 harness family).  NEGATIVE CONTROL.
  * ``message`` — I3: metal projected into the GRU EDGE UPDATE, so it modulates
                  message passing itself.  HYPOTHESIS ARM.

Pre-registered direction: ``message`` > ``readout``.  If it does not hold, the
honest finding is "injection position is not the bottleneck" — still valid.

Protocol (reused, H5)
=====================
Cohort = paper funnel (``cytotox.build_paper_cohort``), per-metal dedup, LOMO
split with the leakage rule (``train_chemprop_lomo_paper_cohort.lomo_split``),
K-shot support drawn label-stratified (``_draw_support``).  Query = held-out
metal test minus support, IDENTICAL across arms (paired).

FSMA adaptation: train encoder+head on the 4 source metals, then FREEZE the
encoder and fine-tune only the readout head on the K support ligands.  Every arm
gets the SAME number of adaptation optimizer steps (H7 budget matching).

Usage
-----
::

    MOLMETAL_DATA_DIR=$PWD/.data/molmetal .venv/bin/python \
        molmetal/scripts/ex1_metal_injection_contrast.py \
        --k 10 --arms none readout message --seeds 42 43 44 45 46 \
        --epochs 20 --adapt-steps 100 --out metrics/by_metric/ex1_metal_injection.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.metal_graph import (  # noqa: E402
    CHEM_METAL_DIM,
    METAL_FEAT_DIM,
    RICH_METAL_DIM,
    build_metal_graph,
)
from molmetal.baselines.dmpnn import DMPNN_DROPOUT, DMPNN_HIDDEN  # noqa: E402
from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN  # noqa: E402

METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_K = (5, 10, 20, 40, 80)


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
def load_cohort(metals: Tuple[str, ...]) -> Dict[str, Dict[str, np.ndarray]]:
    """Paper-funnel cohort per metal WITH metal metadata (OS / charge)."""
    import os

    import pandas as pd

    from molmetal.data.cytotox import build_paper_cohort
    from molmetal.scripts.metal_activity_transfer_probe import collapse_duplicates

    data_dir = Path(
        os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
    )
    raw = pd.read_csv(data_dir / "MetalCytoToxDB.csv", low_memory=False)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for m in metals:
        sub = build_paper_cohort(raw, metal=m)
        ic50 = sub["IC50_Dark_value"].astype(float)
        y_raw = (ic50 < 10.0).to_numpy(dtype=int)
        smi = sub["SMILES_Ligands"].astype(str).to_numpy()
        os_arr = sub["Oxidation_state"].fillna(0).astype(int).to_numpy()
        chg_arr = sub["Charge_complex"].fillna(0).astype(int).to_numpy()
        # dedup on (smiles) keeping first, aligning y/os/chg
        s_u, y_u, _n = collapse_duplicates(smi, y_raw)
        # rebuild metadata for the kept smiles
        first = {}
        for i, s in enumerate(smi):
            if s not in first:
                first[s] = i
        keep = [first[s] for s in s_u]
        out[m] = {
            "smiles": np.asarray(s_u, dtype=object),
            "y": np.asarray(y_u, dtype=np.int64),
            "os": os_arr[keep],
            "chg": chg_arr[keep],
        }
    return out


def lomo_split(cohort: Dict[str, Dict[str, np.ndarray]], held_out: str):
    """LOMO with the probe leakage rule. Returns row dicts for source + held-out."""
    te = cohort[held_out]
    blocked = set(te["smiles"].tolist())
    tr: Dict[str, list] = {"smiles": [], "y": [], "os": [], "chg": [], "metal": []}
    n_leaked = 0
    for m, d in cohort.items():
        if m == held_out:
            continue
        for i, s in enumerate(d["smiles"].tolist()):
            if s in blocked:
                n_leaked += 1
                continue
            tr["smiles"].append(s)
            tr["y"].append(int(d["y"][i]))
            tr["os"].append(int(d["os"][i]))
            tr["chg"].append(int(d["chg"][i]))
            tr["metal"].append(m)
    return tr, te, n_leaked


def featurize(
    rows,
    metal_override: Optional[str],
    use_metal_node: bool,
    rich_metal: bool = False,
    chem_metal: bool = False,
):
    """Build metal-aware graphs.  ``metal_override`` forces the metal (held-out)."""
    graphs = []
    for i in range(len(rows["smiles"])):
        m = metal_override if metal_override is not None else rows["metal"][i]
        g = build_metal_graph(
            rows["smiles"][i], m, int(rows["os"][i]), int(rows["chg"][i]),
            use_metal_node=use_metal_node, rich_metal=rich_metal, chem_metal=chem_metal,
        )
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# Train / adapt / score
# ---------------------------------------------------------------------------
def _valid(graphs, y):
    g = [gi for gi in graphs if gi is not None]
    yy = np.asarray([int(v) for gi, v in zip(graphs, y) if gi is not None], dtype=np.float32)
    return g, yy


def train_source(model, graphs, y, epochs, lr, device, seed):
    torch.manual_seed(seed)
    g, yy = _valid(graphs, y)
    if len(g) == 0:
        return
    yt = torch.from_numpy(yy).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    n = len(g)
    bs = 64
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            sub = [g[i] for i in idx]
            logits = model.forward_graphs(sub, device=device)
            loss = lossf(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def freeze_encoder(model, scope: str = "head"):
    """Select which parameters FSMA adaptation may move (A1 deep-adaptation).

    * ``head``    — readout head only (the WM1 low-K baseline).
    * ``adapter`` — head + the I4 low-rank adapter (requires ``model.adapter``).
    * ``partial`` — head + the metal-conditioning / message-passing layers
                    (``metal_proj`` + the weight-tied ``gru_update``): the
                    encoder's *metal-aware* path adapts, the ligand path stays
                    frozen.
    * ``full``    — everything (upper bound on adaptation capacity).
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.readout_mlp.parameters():
        p.requires_grad = True
    # I4: whenever an adapter is attached it is trainable, independent of scope
    # (the --adapter flag requests it explicitly; scope only widens beyond it).
    if getattr(model, "adapter", None) is not None:
        for p in model.adapter.parameters():
            p.requires_grad = True
    if scope in ("partial", "full"):
        for p in model.metal_proj.parameters():
            p.requires_grad = True
        for p in model.gru_update.parameters():
            p.requires_grad = True
    if scope == "full":
        for p in model.parameters():
            p.requires_grad = True


def adapt_head(model, graphs, y, steps, lr, device, seed, scope: str = "head", l2sp: float = 0.0):
    torch.manual_seed(seed)
    freeze_encoder(model, scope)
    g, yy = _valid(graphs, y)
    if len(g) == 0:
        return
    yt = torch.from_numpy(yy).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # WM1 low-K lever: L2-SP anchor the adapted params to their source-trained
    # values, so the K-shot head stays near the (good) source optimum instead of
    # overfitting the handful of support labels.
    anchor = [p.detach().clone() for p in params] if l2sp > 0 else None
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    n = len(g)
    bs = min(16, max(1, n))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.choice(n, size=bs, replace=n < bs)
        sub = [g[i] for i in idx]
        logits = model.forward_graphs(sub, device=device)
        loss = lossf(logits, yt[idx])
        if l2sp > 0:
            loss = loss + l2sp * sum(((p - a) ** 2).sum() for p, a in zip(params, anchor))
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_control(model, graphs, y, steps, lr, device, seed):
    """C2 control arm: train a *fresh* model on ``graphs`` for exactly ``steps``
    optimizer steps, budget-matched to the K-shot fine-tune (procedure, not
    compute, is the contrast).  Used by ``--control from_scratch|concat_retrain``.
    """
    torch.manual_seed(seed)
    g, yy = _valid(graphs, y)
    if len(g) == 0:
        return
    yt = torch.from_numpy(yy).to(device)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    n = len(g)
    bs = min(16, max(1, n))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.choice(n, size=bs, replace=n < bs)
        logits = model.forward_graphs([g[i] for i in idx], device=device)
        loss = lossf(logits, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def score(model, graphs, device):
    model.eval()
    g, _ = _valid(graphs, np.zeros(len(graphs)))
    if len(g) == 0:
        return np.zeros(0)
    return torch.sigmoid(model.forward_graphs(g, device=device)).cpu().numpy()


def auc(y, p):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ---------------------------------------------------------------------------
# One (metal, arm, seed, k) run
# ---------------------------------------------------------------------------
def run_one(cohort, held_out, arm, k, seed, epochs, adapt_steps, lr, device, rich_metal=False, chem_metal=False, adapter=False, adapt_scope="head", l2sp=0.0, hidden=DMPNN_HIDDEN, dropout=DMPNN_DROPOUT, sampler="stratified", control="kshot"):
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import _draw_support

    # Determinism fix (2026-10-15): the model is constructed below, so the seed
    # MUST be set first — otherwise the init depends on process-global RNG and
    # the run is not reproducible.  This matches train_chemprop_lomo_paper_cohort
    # (seed -> model_factory), which the in-house K-curve was supposed to mirror.
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    tr, te, n_leaked = lomo_split(cohort, held_out)
    use_node = True  # I1: metal node + coordination edges

    tr_graphs = featurize(tr, None, use_node, rich_metal, chem_metal)
    te_graphs = featurize(te, held_out, use_node, rich_metal, chem_metal)
    te_y = te["y"]

    # support / query split on the held-out metal
    scaffolds = None
    if sampler in ("scaffold_diverse", "activity_scaffold"):
        from molmetal.adapters.flow_matching_lipman.scaffold_conditioning import (
            murcko_scaffold_from_smiles,
        )

        scaffolds = [murcko_scaffold_from_smiles(s) for s in te["smiles"]]
    sup_idx = _draw_support(te_y, scaffolds, k, seed, sampler)
    sup_mask = np.zeros(len(te_y), dtype=bool)
    sup_mask[sup_idx] = True
    qry_mask = ~sup_mask

    sup_graphs = [te_graphs[i] for i in range(len(te_graphs)) if sup_mask[i]]
    sup_y = te_y[sup_mask]
    qry_graphs = [te_graphs[i] for i in range(len(te_graphs)) if qry_mask[i]]
    qry_y = te_y[qry_mask]

    use_adapter = bool(adapter) or adapt_scope == "adapter"
    metal_dim = METAL_FEAT_DIM + (RICH_METAL_DIM - METAL_FEAT_DIM if rich_metal else 0) + (CHEM_METAL_DIM if chem_metal else 0)
    model = MetalAwareDMPNN(
        metal_injection=arm,
        metal_feat_dim=metal_dim,
        use_adapter=use_adapter,
        hidden=hidden,
        dropout=dropout,
    ).to(device)
    t0 = time.time()
    if control == "kshot":
        # FSMA: train encoder+head on the source metals, then FREEZE the encoder
        # and fine-tune the readout head on the K support ligands.
        train_source(model, tr_graphs, tr["y"], epochs, lr, device, seed)
        zero_shot = auc(qry_y, score(model, qry_graphs, device))
        adapt_head(model, sup_graphs, sup_y, adapt_steps, lr, device, seed, scope=adapt_scope, l2sp=l2sp)
        adapted = auc(qry_y, score(model, qry_graphs, device))
    else:
        # C2 budget-matched control: a FRESH model trained for exactly the
        # K-shot budget on either the K support rows only (from_scratch) or the
        # source pool ++ K support rows (concat_retrain).  No encoder freezing.
        if control == "from_scratch":
            pool_graphs, pool_y = sup_graphs, sup_y
        elif control == "concat_retrain":
            pool_graphs = list(tr_graphs) + list(sup_graphs)
            pool_y = np.concatenate(
                [np.asarray(tr["y"], dtype=np.float32), np.asarray(sup_y, dtype=np.float32)]
            )
        else:
            raise ValueError(f"unknown control={control!r}")
        train_control(model, pool_graphs, pool_y, adapt_steps, lr, device, seed)
        zero_shot = float("nan")
        adapted = auc(qry_y, score(model, qry_graphs, device))
    return {
        "metal": held_out, "arm": arm, "k": int(k), "seed": int(seed),
        "rich_metal": bool(rich_metal), "chem_metal": bool(chem_metal),
        "adapter": bool(adapter), "adapt_scope": adapt_scope, "l2sp": float(l2sp),
        "hidden": int(hidden), "dropout": float(dropout),
        "sampler": sampler, "control": control,
        "n_leaked": int(n_leaked), "n_support": int(sup_mask.sum()),
        "n_query": int(qry_mask.sum()),
        "zero_shot_auc": zero_shot, "adapted_auc": adapted,
        "wall_s": round(time.time() - t0, 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metals", nargs="+", default=list(METALS))
    ap.add_argument("--arms", nargs="+", default=["none", "readout", "message"])
    ap.add_argument("--k", nargs="+", type=int, default=[10])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--adapt-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--hidden", type=int, default=DMPNN_HIDDEN,
        help="Low-K lever: encoder/head width (default: D-MPNN default).",
    )
    ap.add_argument(
        "--dropout", type=float, default=DMPNN_DROPOUT,
        help="Low-K lever: dropout rate (default: D-MPNN default).",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--rich-metal", action="store_true",
        help="I2: append the RDKit-verifiable periodic/size metal-descriptor block.",
    )
    ap.add_argument(
        "--chem-metal", action="store_true",
        help="I2: append the citable redox/EN metal-descriptor block (metal_chemistry_5.csv).",
    )
    ap.add_argument(
        "--adapter", action="store_true",
        help="I4: attach a residual low-rank adapter (trainable during FSMA adaptation).",
    )
    ap.add_argument(
        "--adapt-scope", choices=["head", "adapter", "partial", "full"], default="head",
        help=(
            "A1 deep-adaptation scope: 'head' (readout only, default), 'adapter' "
            "(head + I4 adapter), 'partial' (head + metal_proj + gru_update), "
            "'full' (all params)."
        ),
    )
    ap.add_argument(
        "--l2sp", type=float, default=0.0,
        help="WM1 low-K lever: L2-SP penalty anchoring adapted params to the source-trained values.",
    )
    ap.add_argument(
        "--sampler", choices=["stratified", "random", "scaffold_diverse", "activity_scaffold"],
        default="stratified",
        help="C3 support-sampler ablation (default: label-stratified, the historical arm).",
    )
    ap.add_argument(
        "--control", choices=["kshot", "from_scratch", "concat_retrain"], default="kshot",
        help=(
            "C2 budget-matched control arms: 'kshot' (default FSMA), 'from_scratch' "
            "(fresh model on the K support rows only), 'concat_retrain' (fresh model "
            "on source ++ K support).  Controls get exactly the K-shot optimizer budget."
        ),
    )
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "ex1_metal_injection.json")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    cohort = load_cohort(tuple(args.metals))
    results = []
    for metal in args.metals:
        for arm in args.arms:
            for k in args.k:
                for seed in args.seeds:
                    r = run_one(
                        cohort, metal, arm, k, seed, args.epochs,
                        args.adapt_steps, args.lr, device,
                        rich_metal=args.rich_metal, chem_metal=args.chem_metal,
                        adapter=args.adapter, adapt_scope=args.adapt_scope,
                        l2sp=args.l2sp, hidden=args.hidden, dropout=args.dropout,
                        sampler=args.sampler, control=args.control,
                    )
                    results.append(r)
                    print(json.dumps(r), flush=True)

    # aggregate: adapted_auc mean per (arm, k)
    agg: Dict[str, Dict[str, float]] = {}
    for arm in args.arms:
        for k in args.k:
            vals = [r["adapted_auc"] for r in results if r["arm"] == arm and r["k"] == k and np.isfinite(r["adapted_auc"])]
            if vals:
                agg[f"{arm}@K{k}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    out = {
        "experiment": "EX1 — metal injection position single-variable contrast",
        "created": "2026-10-14",
        "protocol": "paper_funnel LOMO + FSMA head-only K-shot; encoder frozen; identical query across arms",
        "arms": args.arms, "k": args.k, "seeds": args.seeds,
        "epochs": args.epochs, "adapt_steps": args.adapt_steps, "lr": args.lr,
        "hidden": args.hidden, "dropout": args.dropout,
        "sampler": args.sampler, "control": args.control,
        "rich_metal": bool(args.rich_metal), "chem_metal": bool(args.chem_metal),
        "adapter": bool(args.adapter),
        "adapt_scope": args.adapt_scope, "l2sp": float(args.l2sp),
        "per_run": results, "aggregate": agg,
        "preregistered_direction": "message > readout",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")
    print("aggregate:", json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
