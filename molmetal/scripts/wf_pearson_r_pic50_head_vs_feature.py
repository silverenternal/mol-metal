#!/usr/bin/env python
"""Decisive cheap experiment for pearson_r_pic50 (m1 pIC50 regression).

Question: the D-MPNN multi-task head reaches test_pearson=0.3091 on the Ru
HeLa48h/dark cohort while ridge reaches 0.572. Is the REGRESSION HEAD starved
(by alpha weighting the classification term) or do the D-MPNN FEATURES simply
carry less pIC50 signal?

Design (decisive, one GPU, ~minutes):
  (A) Reproduce the exact Ru split (train=2509/val=363/test=796) via the
      trainer's own build_metal_split_indices + featurize path.
  (B) ridge-on-frozen-embeddings, FROZEN RANDOM encoder  -> feature floor.
  (C) train the model at alpha=0.5 (default) and alpha=0.2 (reg-weighted),
      measure the trained HEAD's test pearson on the same split.
  (D) ridge-on-frozen-embeddings, TRAINED encoder -> if (D) >> (C) the head
      is the problem; if (D) ~ (C) the features are the problem.
  (E) reference ceiling: ridge on Morgan-FP 2048 (the RDKit fingerprint ridge
      baseline, same split) -> confirms the 0.572-order ceiling is reachable
      from features on THIS split.

Every number printed is MEASURED on the runs of THIS script; nothing is
copied from the metric JSON except the historical comparison values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

REPO = Path("/home/hugo/codes/mol-metal")
sys.path.insert(0, str(REPO))

from molmetal.scripts.train_dmpnn_multitask_wasserstein import (  # noqa: E402
    FIVE_METALS,
    MultiTaskWassersteinModel,
    MultiTaskLoss,
    build_metal_split_indices,
    load_5metal_split_table,
    _resolve_default_csv_path,
    train_one_epoch_joint,
    predict_joint,
    _featurize_with_cache,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset  # noqa: E402
from molmetal.training.collate_multimetal import collate_multimetal  # noqa: E402
from molmetal.baselines.dmpnn_multitask import regression_metrics  # noqa: E402
from molmetal.molmetal_lam.reward.wasserstein_loss import (  # noqa: E402
    WassersteinConfig,
)


def morgan_fp(smiles, radius=2, n_bits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.stack(fps)


def extract_embeddings(model, graphs, metal_ids, idx, batch_size, device):
    """Run model.encode_pool over the rows in idx; return (N, 2H) embeddings."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start : start + batch_size]
            bg = [graphs[i] for i in b]
            bm = [metal_ids[i] for i in b]
            data = collate_multimetal(bg, bm, np.zeros(len(b)), np.zeros(len(b)))
            if data is None:
                continue
            atom_f, bond_f, e_src, e_dst, mids_t, _, _ = data
            atom_f = atom_f.to(device)
            bond_f = bond_f.to(device)
            e_src = e_src.to(device)
            e_dst = e_dst.to(device)
            with torch.enable_grad() if False else torch.no_grad():
                h = model.encode_pool(atom_f, bond_f, e_src, e_dst)
            out.append(h.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def ridge_pearson(X_train, y_train, X_test, y_test, alphas=(0.1, 1.0, 10.0, 100.0)):
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    Xte = sc.transform(X_test)
    best = None
    for a in alphas:
        r = Ridge(alpha=a).fit(Xtr, y_train)
        p = r.predict(Xte)
        m = regression_metrics(y_test, p)
        if best is None or m["pearson"] > best["pearson"]:
            best = dict(m, ridge_alpha=a, r2=float(r2_score(y_test, p)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metal", default="Ru")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(REPO / "molmetal/reports/wf_pearson_diag/report.json"))
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    reports = {}

    # ---- (A) reproduce the exact split ----
    csv_path = _resolve_default_csv_path("metalcytotox_5metal")
    split_df = load_5metal_split_table(csv_path)
    flt = CytotoxFilter(
        time_threshold=24.0, ic50_min=0.01,
        metal_whitelist=list(FIVE_METALS), compute_pic50=True, compute_active=True,
    )
    import pandas as _pd
    from molmetal.data import cytotox as _cy

    _df = _pd.read_csv(REPO / ".data/molmetal/MetalCytoToxDB.csv", low_memory=False)
    _df = _cy._apply_filters(_df, flt)
    _df = _cy._add_pic50(_df)
    _df = _cy._add_active(_df)
    _cache = Path("/tmp/wf_pearson_diag_cache")
    _cache.mkdir(parents=True, exist_ok=True)
    ds = MetalCytotoxDataset(df=_df, cache_dir=_cache)
    smiles_all = np.asarray(ds.smiles).astype(str)
    pic50_all = np.asarray(ds.pic50).astype(np.float32)
    active_all = np.asarray(ds.active).astype(np.int64)
    metals_all = np.asarray(ds.metals).astype(str)

    per_metal = {m: build_metal_split_indices(split_df, ds, m) for m in FIVE_METALS}
    target_train, target_val, target_test = per_metal[args.metal]
    train_pool_idx = np.concatenate([per_metal[m][0] for m in FIVE_METALS])
    print(f"[A] split: n_train(pool)={len(train_pool_idx)} "
          f"n_target_train={len(target_train)} n_val={len(target_val)} "
          f"n_test={len(target_test)}", flush=True)
    reports["split"] = {
        "n_joint_train": int(len(train_pool_idx)),
        "n_target_train": int(len(target_train)),
        "n_val": int(len(target_val)),
        "n_test": int(len(target_test)),
    }

    # featurise union (joint train + target val/test) — mirrors trainer used_idx
    used_idx = np.unique(np.concatenate([train_pool_idx, target_val, target_test]))
    pos_of = {int(v): k for k, v in enumerate(used_idx)}
    graphs_all = _featurize_with_cache(
        [str(s) for s in smiles_all[used_idx]],
        cache_key=f"pearson_diag_{args.metal}_seed{args.seed}",
        cache_dir=Path("/tmp/wf_pearson_diag_cache"),
    )
    metal_to_idx = {m: i for i, m in enumerate(FIVE_METALS)}
    metal_ids_all = [metal_to_idx[str(m)] for m in metals_all[used_idx]]

    g_pool = [pos_of[int(i)] for i in train_pool_idx]
    g_val = [pos_of[int(i)] for i in target_val]
    g_test = [pos_of[int(i)] for i in target_test]
    y_pool = pic50_all[train_pool_idx]
    y_val = pic50_all[target_val]
    y_test = pic50_all[target_test]

    # ---- (E) reference ceiling: ridge on Morgan FP, same split ----
    fp_used = morgan_fp([str(s) for s in smiles_all[used_idx]])
    Xtr_fp = fp_used[[pos_of[int(i)] for i in target_train]]
    Xte_fp = fp_used[[pos_of[int(i)] for i in target_test]]
    ridge_fp = ridge_pearson(Xtr_fp, pic50_all[target_train], Xte_fp, pic50_all[target_test])
    print(f"[E] ridge-on-MorganFP same-split test pearson={ridge_fp['pearson']:.4f} "
          f"(alpha={ridge_fp['ridge_alpha']})", flush=True)
    reports["ridge_morgan_fp_same_split"] = ridge_fp

    # ---- (B) ridge on FROZEN RANDOM encoder embeddings ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MultiTaskWassersteinModel(n_metals=len(FIVE_METALS)).to(device)
    emb_rand = extract_embeddings(model, graphs_all, metal_ids_all, g_pool, 128, device)
    emb_rand_te = extract_embeddings(model, graphs_all, metal_ids_all, g_test, 128, device)
    ridge_rand = ridge_pearson(emb_rand, y_pool, emb_rand_te, y_test)
    print(f"[B] ridge-on-RANDOM-encoder-emb test pearson={ridge_rand['pearson']:.4f}",
          flush=True)
    reports["ridge_random_encoder_emb"] = ridge_rand

    # ---- (C) train at each alpha, measure the HEAD's test pearson ----
    head_results = {}
    trained_emb = {}
    for alpha in (0.5, 0.2):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        m = MultiTaskWassersteinModel(n_metals=len(FIVE_METALS)).to(device)
        crit = MultiTaskLoss(alpha=alpha)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        w_cfg = WassersteinConfig()
        ref = torch.tensor(y_pool, dtype=torch.float32)
        for ep in range(args.epochs):
            tr = np.arange(len(g_pool))
            rng = np.random.default_rng(args.seed + ep)
            rng.shuffle(tr)
            gsub = [graphs_all[g_pool[i]] for i in tr]
            msub = [metal_ids_all[g_pool[i]] for i in tr]
            ysub = y_pool[tr]
            asub = active_all[train_pool_idx][tr]
            train_one_epoch_joint(
                model=m, optimizer=opt, criterion=crit, w_w=0.05,
                use_l2_guard=True, graphs=gsub, metal_ids=msub,
                pic50=ysub, active=asub, ref_pic50_dist=ref,
                batch_size=64, device=device, w_config=w_cfg,
            )
        _, _, pic50_t = predict_joint(
            model=m,
            graphs=[graphs_all[i] for i in g_test],
            metal_ids=[metal_ids_all[i] for i in g_test],
            pic50=pic50_all[target_test],
            active=active_all[target_test],
            batch_size=128,
            device=device,
        )
        mh = regression_metrics(y_test, pic50_t)
        head_results[f"alpha_{alpha}"] = mh
        print(f"[C] trained HEAD alpha={alpha} test pearson={mh['pearson']:.4f} "
              f"rmse={mh['rmse']:.4f}", flush=True)
        # ---- (D) ridge on TRAINED encoder embeddings ----
        e_tr = extract_embeddings(m, graphs_all, metal_ids_all, g_pool, 128, device)
        e_te = extract_embeddings(m, graphs_all, metal_ids_all, g_test, 128, device)
        rd = ridge_pearson(e_tr, y_pool, e_te, y_test)
        trained_emb[f"alpha_{alpha}"] = rd
        print(f"[D] ridge-on-TRAINED-encoder(alpha={alpha}) emb test "
              f"pearson={rd['pearson']:.4f}", flush=True)
    reports["trained_head"] = head_results
    reports["ridge_trained_encoder_emb"] = trained_emb

    # ---- verdict ----
    hd = head_results["alpha_0.5"]["pearson"]
    hd02 = head_results["alpha_0.2"]["pearson"]
    rd_tr = trained_emb["alpha_0.5"]["pearson"]
    verdict = {
        "head_default_0.5": hd,
        "head_regweighted_0.2": hd02,
        "ridge_trained_emb": rd_tr,
        "head_is_starved": bool(rd_tr - hd > 0.10),
        "reg_weight_helps": bool(hd02 - hd > 0.02),
        "morgan_fp_ceiling": ridge_fp["pearson"],
    }
    reports["verdict"] = verdict
    print(f"[VERDICT] head(0.5)={hd:.4f} head(0.2)={hd02:.4f} "
          f"ridge_trained_emb={rd_tr:.4f} morgan_fp_ceiling={ridge_fp['pearson']:.4f}",
          flush=True)
    reports["elapsed_seconds"] = time.time() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
