#!/usr/bin/env python3
"""UP-7 — do the auxiliary endpoints regularise the potency head?

Three endpoints share the same ligand–metal structure: potency (21,404 rows),
cisplatin_ratio (14,309) and phototoxicity (2,158).  They are currently modelled
separately.  This script trains one shared encoder with three heads and asks
whether the auxiliary targets help potency on a DOI holdout.

Arms (same folds, same seed, same graphs; only the training task set changes):

    single      potency only
    aux_photo   potency + phototoxicity
    aux_both    potency + phototoxicity + cisplatin_ratio

Gate (frozen before the run, 2026-10-21):

    PASS  if potency_r2 >= single_r2 + 0.02
         and neither auxiliary endpoint's own R2 drops by more than 0.02
    FAIL  otherwise -> auxiliary targets carry no regularisation value here.

**Leakage caveat, measured and reported.**  cisplatin_ratio is derived from the
cisplatin-sensitive / resistant pair, so a row of it can describe the same
(ligand, cell) as a potency row.  ``leak`` in the artifact reports the overlap
share; if it is large, ``aux_both`` is not an independent auxiliary task and the
``aux_photo`` arm is the clean one.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "molmetal" / "scripts"))

from msrs_dmpnn_surface import (  # noqa: E402
    attach_conditioning,
    cell_one_hot,
    conditioning_matrix,
    graph_key,
    load_config,
)
from msrs_surface_model import score  # noqa: E402
from molmetal.data.metal_graph import build_metal_graph  # noqa: E402
from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN  # noqa: E402

OUT = REPO / "metrics" / "by_metric" / "msrs_multitask_aux.json"
TASKS = ("potency", "phototoxicity", "cisplatin_ratio")
GATE = {"potency_delta": 0.02, "aux_drop": 0.02, "frozen_at": "2026-10-21T15:10+08:00"}
# arm -> (task ids trained, endpoints scored)
ARMS = {
    "single": ((0,), ("potency",)),
    "aux_photo": ((0, 1), ("potency", "phototoxicity")),
    "aux_both": ((0, 1, 2), ("potency", "phototoxicity", "cisplatin_ratio")),
    "photo_only": ((1,), ("phototoxicity",)),
    "cis_only": ((2,), ("cisplatin_ratio",)),
}


class MultiHeadDMPNN(MetalAwareDMPNN):
    """Shared metal-aware encoder, ``n_tasks`` regression outputs."""

    def __init__(self, n_tasks: int, **kw):
        super().__init__(metal_injection="message", **kw)
        h = int(kw.get("hidden", 128))
        dropout = float(kw.get("dropout", 0.1))
        self.readout_mlp = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h // 2, n_tasks),
        )


def predict_multi(model, graphs: Sequence, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = graphs[start : start + batch_size]
            if not batch:
                continue
            out.append(model.forward_graphs(batch, device=device).cpu().numpy())
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(out, axis=0)


def train_fold(
    graphs_tr, y_tr, task_tr, graphs_va, y_va, *, cond_dim, model_cfg, train_cfg,
    device, seed, n_tasks=3, eval_task=0, budget=None, n_potency=None,
):
    """Fit one fold.

    ``budget`` caps the rows seen per epoch so that an arm with auxiliary rows
    runs the same number of optimizer steps as the potency-only arm.  Without it
    the multi-task arm simply sees more data per epoch and any gain is a budget
    effect, not a regularisation effect.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiHeadDMPNN(
        n_tasks=n_tasks,
        metal_feat_dim=cond_dim,
        hidden=int(model_cfg["hidden"]),
        depth=int(model_cfg["depth"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    tt = torch.from_numpy(task_tr.astype(np.int64)).to(device)
    bs = int(train_cfg["batch_size"])
    rng = np.random.default_rng(seed)

    pot_rows = np.flatnonzero(task_tr == 0)
    aux_rows = np.flatnonzero(task_tr != 0)
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    bad = 0
    for _epoch in range(int(train_cfg["epochs"])):
        model.train()
        epoch_rows = np.arange(len(graphs_tr))
        if budget is not None and len(aux_rows) and len(pot_rows):
            # Equal optimizer steps across arms: the epoch sees exactly ``budget``
            # rows.  The auxiliary share is the natural one, capped at half, and
            # the potency rows are subsampled to make room for it.
            natural = len(aux_rows) / (len(pot_rows) + len(aux_rows))
            share = min(0.5, natural)
            n_aux = int(round(budget * share))
            n_pot = max(1, int(budget) - n_aux)
            keep_aux = (rng.choice(aux_rows, size=n_aux, replace=False)
                        if n_aux < len(aux_rows) else aux_rows)
            keep_pot = (rng.choice(pot_rows, size=n_pot, replace=False)
                        if n_pot < len(pot_rows) else pot_rows)
            epoch_rows = np.concatenate([keep_pot, keep_aux])
        order = rng.permutation(epoch_rows)
        for start in range(0, len(order), bs):
            idx = order[start : start + bs]
            pred = model.forward_graphs([graphs_tr[i] for i in idx], device=device)
            row = pred[torch.arange(len(idx), device=device), tt[idx]]
            loss = nn.functional.mse_loss(row, yt[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        if len(graphs_va):
            pred_va = predict_multi(model, graphs_va, device)
            col = pred_va[:, eval_task]
            mae = float(np.mean(np.abs(col - y_va)))
            if mae < best_mae - 1e-5:
                best_mae, bad = mae, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= int(train_cfg["patience"]):
                    break

    model.load_state_dict(best_state)
    return model, best_mae


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "msrs_dmpnn_surface_gpu0.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg, compute_cfg = cfg["data"], cfg["compute"]
    if compute_cfg.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(int(compute_cfg.get("num_threads", 1)))
    device = torch.device(str(compute_cfg.get("device", "cpu")))

    long_path = REPO / str(data_cfg["long"])
    long = pd.read_parquet(long_path)
    sub = long[
        long["endpoint"].isin(TASKS)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub.loc[sub["endpoint"] == "potency", "ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)
    sub["task"] = sub["endpoint"].map({t: i for i, t in enumerate(TASKS)}).astype(int)
    print(f"[UP-7] rows={len(sub)} " +
          " ".join(f"{t}={int((sub['endpoint'] == t).sum())}" for t in TASKS))

    # leakage diagnostic: cisplatin_ratio rows describing the same (ligand, cell)
    pot_pairs = set(map(tuple, sub.loc[sub["endpoint"] == "potency",
                                       ["ligand_canonical", "cell_line_key"]].values))
    cis = sub[sub["endpoint"] == "cisplatin_ratio"]
    cis_pairs = list(map(tuple, cis[["ligand_canonical", "cell_line_key"]].values))
    leak = float(np.mean([p in pot_pairs for p in cis_pairs])) if cis_pairs else 0.0
    print(f"[UP-7] cisplatin_ratio rows whose (ligand, cell) also has a potency row: {leak:.3f}")

    t0 = time.time()
    cache = {}
    for row in sub[["ligand_canonical", "metal", "oxidation_state", "charge_complex"]].itertuples(index=False):
        k = graph_key(row)
        if k not in cache:
            cache[k] = build_metal_graph(k[0], k[1], k[2], k[3], use_metal_node=True)
    keys = [
        (str(r.ligand_canonical), str(r.metal), int(r.oxidation_state), int(r.charge_complex))
        for r in sub[["ligand_canonical", "metal", "oxidation_state", "charge_complex"]].itertuples(index=False)
    ]
    ok = np.array([cache[k] is not None for k in keys])
    if not ok.all():
        sub = sub[ok].reset_index(drop=True)
        keys = [k for k, good in zip(keys, ok) if good]
    print(f"[UP-7] graphs={len(cache)} in {time.time() - t0:.1f}s")

    y = sub["value_log10"].to_numpy(dtype=float)
    task = sub["task"].to_numpy(dtype=int)
    vocab = sub["cell_line_key"].value_counts().head(int(data_cfg["top_cell_lines"])).index.tolist()
    cond = conditioning_matrix(sub, "state", cell_one_hot(sub, vocab))
    cond_dim = cond.shape[1]
    graphs = [attach_conditioning(cache[k], cond[i]) for i, k in enumerate(keys)]

    from sklearn.model_selection import GroupKFold
    doi = sub["doi"].astype(str).to_numpy()
    fold_id = np.full(len(sub), -1, dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(sub, groups=doi)):
        fold_id[te] = k

    results: Dict[str, object] = {}
    oof: Dict[str, np.ndarray] = {}
    budget_used: Dict[str, int] = {}
    for arm, (tasks, scored) in ARMS.items():
        t_arm = time.time()
        mask_arm = np.isin(task, tasks)
        pred = np.full((len(sub), 3), np.nan, dtype=np.float64)
        eval_task = int(tasks[0])
        for k in range(5):
            tr = np.flatnonzero((fold_id != k) & mask_arm)
            te = np.flatnonzero(fold_id == k)
            rng = np.random.default_rng(int(cfg["train"]["seed"]) + k)
            if eval_task == 0:
                pot_tr = tr[task[tr] == 0]
                uniq = np.unique(sub["ligand_canonical"].to_numpy()[pot_tr])
                n_val = max(1, int(round(len(uniq) * float(cfg["train"]["inner_val_fraction"]))))
                take = min(n_val, max(1, len(uniq) - 1)) if len(uniq) > 1 else len(uniq)
                val_lig = set(rng.choice(uniq, size=take, replace=False)) if len(uniq) else set()
                is_val = np.array([g in val_lig for g in sub["ligand_canonical"].to_numpy()[pot_tr]])
                fit_idx = pot_tr[~is_val] if is_val.any() else pot_tr
                val_idx = pot_tr[is_val] if is_val.any() else pot_tr
            else:
                # reference arm for an auxiliary endpoint: early-stop on its own target
                fit_idx = tr
                n_val = max(1, int(round(len(tr) * float(cfg["train"]["inner_val_fraction"]))))
                val_idx = rng.choice(tr, size=min(n_val, max(1, len(tr) - 1)), replace=False)
            # auxiliary rows from the same folds join the fit set, but the epoch
            # budget is fixed at the potency-only row count (see train_fold).
            aux_fit = tr[task[tr] != 0] if eval_task == 0 else np.array([], dtype=int)
            fit_all = np.concatenate([fit_idx, aux_fit]) if len(aux_fit) else fit_idx

            model, _mae = train_fold(
                [graphs[i] for i in fit_all], y[fit_all], task[fit_all],
                [graphs[i] for i in val_idx], y[val_idx],
                cond_dim=cond_dim, model_cfg=cfg["model"], train_cfg=cfg["train"],
                device=device, seed=int(cfg["train"]["seed"]) + k,
                eval_task=eval_task, budget=int(len(fit_idx)), n_potency=int(len(fit_idx)),
            )
            pred[te] = predict_multi(model, [graphs[i] for i in te], device)
            budget_used[arm] = int(len(fit_idx))
            print(f"    {arm:10s} fold={k} n_fit={len(fit_all)} "
                  f"n_aux={len(aux_fit)} budget={len(fit_idx)} ({time.time() - t_arm:.0f}s)")

        entry = {}
        for name, tid in (("potency", 0), ("phototoxicity", 1), ("cisplatin_ratio", 2)):
            if name in scored:
                m = task == tid
                entry[name] = score(y[m], pred[m, tid])
        entry["wall_s"] = round(time.time() - t_arm, 1)
        results[arm] = entry
        oof[arm] = pred[:, 0]
        if "potency" in entry:
            print(f"[UP-7] {arm:10s} potency R2={entry['potency']['r2']:.4f} "
                  f"MAE={entry['potency']['mae']:.4f}")
        else:
            print(f"[UP-7] {arm:10s} (reference only) " +
                  " ".join(f"{k} R2={v['r2']:.4f}" for k, v in entry.items()
                           if isinstance(v, dict) and "r2" in v))

    base = results["single"]["potency"]["r2"]
    deltas = {a: round(results[a]["potency"]["r2"] - base, 4)
              for a in ARMS if a != "single" and "potency" in results[a]}
    aux_ok = True
    aux_ref = {"phototoxicity": "photo_only", "cisplatin_ratio": "cis_only"}
    for arm in ("aux_photo", "aux_both"):
        for name, ref_arm in aux_ref.items():
            if name in results[arm] and name in results[ref_arm]:
                if results[arm][name]["r2"] < results[ref_arm][name]["r2"] - GATE["aux_drop"]:
                    aux_ok = False
    best = max(deltas, key=deltas.get)
    verdict = "AUX_HELPS" if (deltas[best] >= GATE["potency_delta"] and aux_ok) else "AUX_DOES_NOT_HELP"

    doc = {
        "artifact": "msrs_multitask_aux",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 multitask / UP-7",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "config": str(Path(args.config).relative_to(REPO)),
            "script": str(Path(__file__).relative_to(REPO)),
            "long": str(long_path.relative_to(REPO)),
            "seed": int(cfg["train"]["seed"]),
            "device": str(device),
        },
        "design": {
            "encoder": "MetalAwareDMPNN (metal_injection=message), shared",
            "heads": "one linear readout with 3 outputs; loss masked to the arm's task set",
            "arms": {a: [TASKS[i] for i in t] for a, (t, _s) in ARMS.items()},
            "folds": "GroupKFold(5) on doi over the union of endpoints",
            "evaluation": "potency rows only, out-of-fold",
            "seeds": 1,
            "budget_matching": "every epoch sees exactly the potency-only row count; "
                               "auxiliary rows are subsampled to fill the remainder",
            "budget_rows_per_epoch": budget_used,
            "gate": GATE,
        },
        "leakage": {
            "cisplatin_ratio_pairs_also_in_potency": round(leak, 4),
            "reading": "cisplatin_ratio is derived from the cisplatin-sensitive / resistant "
                       "pair, so a high share means aux_both is not an independent task",
        },
        "results": results,
        "potency_delta_vs_single": deltas,
        "best_arm": best,
        "verdict": verdict,
        "verdict_rule": "PASS iff best potency delta >= 0.02 AND no auxiliary endpoint drops > 0.02",
        "caveats": [
            "One seed. Optimizer steps per epoch ARE matched (budget_rows_per_epoch), so a "
            "gain cannot come from simply seeing more rows; it can still come from the "
            "gradient mix within a batch.",
            "phototoxicity has 2,158 rows against 21,404 potency rows, so its gradient share "
            "is ~10% even when it is present.",
            "The shared encoder is the only thing regularised; each endpoint still has its own "
            "output column, so a null result means the shared representation did not improve.",
        ],
    }
    if not args.dry_run:
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"json: {args.out}")
    print(f"[UP-7] deltas={deltas} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
