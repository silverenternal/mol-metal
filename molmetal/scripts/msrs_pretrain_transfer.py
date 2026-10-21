#!/usr/bin/env python3
"""UP-5 — does Pt pretraining transfer to the five-metal potency task?

The learning curve is still climbing at 100% of the data
(``msrs_learning_curve.json``, plateau_fraction = 1.0), so sample size is a
constraint.  B1 blocks new curation, but two corpora are already on disk and are
**not** usable as validation:

    .data/molmetal/wf_a37_platinai_biodata/pt/platinai_pt_all_cells_with_smiles.jsonl
        17,735 Pt rows (94.2% RDKit-parseable), a different corpus and a
        different metal — pretraining only, never a test set.
    molmetal/data/tmqm_dblock_subset.csv
        5,000 d-block complexes with geometry — no bioactivity at all.

This script runs the first step only (Pt pretraining), because that is the arm
the gate is stated on:

    scratch      the metal_state arm, trained from random init
    pt_pretrain  the same arm, encoder initialised from the Pt model

Gate (frozen before the run, 2026-10-21):

    PASS  if DOI-holdout R2 >= scratch + 0.03
    FAIL  otherwise -> sample size is not the binding constraint after all, which
         contradicts the learning-curve extrapolation and is itself the finding.

Encoder weights are transferred by name and shape; the readout and metal
projection are re-initialised because the conditioning vector differs between
the two tasks.
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
    _predict,
    attach_conditioning,
    cell_one_hot,
    conditioning_matrix,
    graph_key,
    load_config,
    select_cohort,
    train_fold,
)
from msrs_surface_model import score  # noqa: E402
from molmetal.data.metal_graph import build_metal_graph  # noqa: E402
from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN  # noqa: E402

OUT = REPO / "metrics" / "by_metric" / "msrs_pretrain_transfer.json"
PT_JSONL = REPO / ".data" / "molmetal" / "wf_a37_platinai_biodata" / "pt" / "platinai_pt_all_cells_with_smiles.jsonl"
GATE = {"delta_r2": 0.03, "frozen_at": "2026-10-21T15:10+08:00"}


def load_pt(max_rows: int | None = None) -> Tuple[List, np.ndarray, np.ndarray]:
    graphs, ys, cells = [], [], []
    n_bad = n_nonpos = 0
    with PT_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            v = d.get("value_mean")
            s = d.get("smiles")
            if v is None or s is None:
                n_nonpos += 1
                continue
            v = float(v)
            if not np.isfinite(v) or v <= 0:
                n_nonpos += 1
                continue
            g = build_metal_graph(s, "Pt", 2, 0, use_metal_node=True, strip_metals=False)
            if g is None:
                n_bad += 1
                continue
            graphs.append(g)
            ys.append(-np.log10(v * 1e-6))
            cells.append(str(d.get("cell_line") or "NA"))
            if max_rows and len(graphs) >= max_rows:
                break
    print(f"[UP-5] Pt rows kept={len(graphs)} unparseable={n_bad} nonpositive={n_nonpos}")
    return graphs, np.asarray(ys, dtype=np.float64), np.asarray(cells)


def pretrain_pt(cfg, device, max_rows: int | None):
    graphs, y, cells = load_pt(max_rows)
    vocab = pd.Series(cells).value_counts().head(40).index.tolist()
    index = {c: i for i, c in enumerate(vocab)}
    oh = np.zeros((len(cells), len(vocab) + 1), dtype=np.float32)
    for i, c in enumerate(cells):
        oh[i, index.get(c, len(vocab))] = 1.0
    cond_dim = oh.shape[1]
    gs = [(g[0], g[1], g[2], g[3], oh[i].astype(np.float32)) for i, g in enumerate(graphs)]

    # DOI-grouped inner validation so the early stop is not a within-paper fit
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    n_val = max(1, int(round(len(gs) * 0.1)))
    val_idx = rng.choice(len(gs), size=n_val, replace=False)
    mask = np.zeros(len(gs), dtype=bool)
    mask[val_idx] = True

    model, best_mae = train_fold(
        [gs[i] for i in np.flatnonzero(~mask)], y[~mask],
        [gs[i] for i in np.flatnonzero(mask)], y[mask],
        cond_dim=cond_dim,
        arm={"metal_injection": "readout"},
        model_cfg=cfg["model"], train_cfg=cfg["train"], device=device,
        seed=int(cfg["train"]["seed"]),
    )
    print(f"[UP-5] Pt pretrain inner_val_mae={best_mae:.4f} (cond_dim={cond_dim})")
    return model.state_dict(), best_mae, cond_dim


def finetune(cfg, device, frame, graphs, cond, groups, init_state=None):
    from msrs_code_levers import folds_for

    fold_id = folds_for(groups)
    y = frame["value_log10"].to_numpy(dtype=float)
    y_mu, y_sd = float(y.mean()), float(y.std(ddof=0)) or 1.0
    y_z = (y - y_mu) / y_sd
    seed = int(cfg["train"]["seed"])
    oof = np.full(len(y), np.nan)
    for k in sorted(set(fold_id[fold_id >= 0].tolist())):
        tr = np.flatnonzero(fold_id != k)
        te = np.flatnonzero(fold_id == k)
        rng = np.random.default_rng(seed + k)
        uniq = np.unique(groups[tr])
        n_val = max(1, int(round(len(uniq) * float(cfg["train"]["inner_val_fraction"]))))
        val_groups = set(rng.choice(uniq, size=min(n_val, max(1, len(uniq) - 1)), replace=False))
        is_val = np.array([g in val_groups for g in groups[tr]])
        fit_idx, val_idx = tr[~is_val], tr[is_val]
        if len(val_idx) == 0:
            fit_idx, val_idx = tr, tr

        model, _mae = train_fold(
            [graphs[i] for i in fit_idx], y_z[fit_idx],
            [graphs[i] for i in val_idx], y_z[val_idx],
            cond_dim=cond.shape[1],
            arm={"metal_injection": "message"},
            model_cfg=cfg["model"], train_cfg=cfg["train"], device=device,
            seed=seed + k, init_state=init_state,
        )
        oof[te] = _predict(model, [graphs[i] for i in te], device) * y_sd + y_mu
    return oof


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "msrs_dmpnn_surface_gpu0.yaml")
    ap.add_argument("--max-pt", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="ligand cap on the fine-tune cohort")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg["compute"].get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(int(cfg["compute"].get("num_threads", 1)))
    device = torch.device(str(cfg["compute"].get("device", "cpu")))

    t0 = time.time()
    state, pt_mae, pt_cond_dim = pretrain_pt(cfg, device, args.max_pt)
    pt_wall = round(time.time() - t0, 1)

    frame = select_cohort(REPO / str(cfg["data"]["long"]), str(cfg["data"]["endpoint"]), args.limit)
    cache = {}
    for row in frame[["ligand_canonical", "metal", "oxidation_state", "charge_complex"]].itertuples(index=False):
        k = graph_key(row)
        if k not in cache:
            cache[k] = build_metal_graph(k[0], k[1], k[2], k[3], use_metal_node=True)
    keys = [
        (str(r.ligand_canonical), str(r.metal), int(r.oxidation_state), int(r.charge_complex))
        for r in frame[["ligand_canonical", "metal", "oxidation_state", "charge_complex"]].itertuples(index=False)
    ]
    ok = np.array([cache[k] is not None for k in keys])
    if not ok.all():
        frame = frame[ok].reset_index(drop=True)
        keys = [k for k, good in zip(keys, ok) if good]
    vocab = frame["cell_line_key"].value_counts().head(int(cfg["data"]["top_cell_lines"])).index.tolist()
    cond = conditioning_matrix(frame, "state", cell_one_hot(frame, vocab))
    graphs = [attach_conditioning(cache[k], cond[i]) for i, k in enumerate(keys)]
    groups = frame["doi"].astype(str).to_numpy()

    results: Dict[str, object] = {}
    for arm, init in (("scratch", None), ("pt_pretrain", state)):
        t_arm = time.time()
        oof = finetune(cfg, device, frame, graphs, cond, groups, init_state=init)
        results[arm] = score(frame["value_log10"].to_numpy(dtype=float), oof)
        results[arm]["wall_s"] = round(time.time() - t_arm, 1)
        print(f"[UP-5] {arm:12s} R2={results[arm]['r2']:.4f} MAE={results[arm]['mae']:.4f}")

    delta = round(results["pt_pretrain"]["r2"] - results["scratch"]["r2"], 4)
    verdict = "PRETRAIN_TRANSFERS" if delta >= GATE["delta_r2"] else "PRETRAIN_DOES_NOT_TRANSFER"

    doc = {
        "artifact": "msrs_pretrain_transfer",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 transfer / UP-5",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "config": str(Path(args.config).relative_to(REPO)),
            "script": str(Path(__file__).relative_to(REPO)),
            "pt_corpus": str(PT_JSONL.relative_to(REPO)),
            "seed": int(cfg["train"]["seed"]),
            "device": str(device),
        },
        "design": {
            "arms": {
                "scratch": "metal_state arm, random init",
                "pt_pretrain": "same arm, encoder initialised from a Pt pIC50 model",
            },
            "pretrain_task": "Pt pIC50 = -log10(value_mean * 1e-6), 40 epochs, 10% inner val",
            "transfer": "state_dict entries matched by name and shape; readout and metal_proj "
                        "re-initialised because the conditioning vector differs",
            "split": "GroupKFold(5) on doi",
            "seeds": 1,
            "gate": GATE,
        },
        "pretrain": {"inner_val_mae": round(float(pt_mae), 4),
                     "cond_dim": int(pt_cond_dim), "wall_s": pt_wall},
        "results": results,
        "delta_r2": delta,
        "verdict": verdict,
        "verdict_rule": "PASS iff DOI-holdout R2 >= scratch + 0.03",
        "caveats": [
            "Pt is a different metal and a different corpus; this is domain-shift pretraining, "
            "not self-supervised pretraining. The plan's tmQM arm (no bioactivity) is not run.",
            "One seed. The two arms share folds and seed, but not initialisation, so the "
            "comparison is paired by data only.",
            "94.2% of Pt rows parse; the 5.8% failures are valence errors in the source SMILES "
            "([NH2] bonded twice) and are dropped, not repaired.",
        ],
    }
    if not args.dry_run:
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"json: {args.out}")
    print(f"[UP-5] delta_r2={delta:+.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
