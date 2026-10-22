#!/usr/bin/env python3
"""UP-21 — can a paper's offset be predicted from what the paper *is*?

=====================================================================
Why this exists
=====================================================================
The DOI protocol holds out whole papers. Its R^2 (0.3767 for the
maccs_rdkit arm, 5 seeds) is depressed by something the model cannot
know: each paper carries its own calibration offset, and a held-out
paper's offset is never seen. UP-17 measured that offset as a real
component (paper x metal 0.5484 under batch-first ordering), and UP-14
measured that pairing is a within-paper phenomenon.

A DOI one-hot is useless on a held-out paper -- the column is zero. The
question here is whether the offset is a function of *design* variables
that are known before the assay is read: publication year, which metals
the paper studied, which cell lines, which scaffolds. If it is, the
offset becomes predictable and the DOI number should rise.

=====================================================================
Leakage rule (frozen with the gate)
=====================================================================
A paper's meta-features may use its rows' **metadata** (ligand, metal,
cell line, DOI year) but never its **outcomes**. The primary feature set
is deliberately count-free and outcome-free:

    year                     1 column
    metal composition        5 fractions
    cell-line entropy        1 column
    scaffold bucket shares   16 fractions (Murcko scaffold hashed)

Two ablations are reported and labelled, not folded into the headline:
    +counts    adds n_rows and n_ligands (submission size)
    +outcome   adds mean and deciles of y  -- LEAKY, reference only

=====================================================================
Offset model
=====================================================================
For outer fold k, the training papers get a shrunk offset

    r_p = mean(y - pred | paper p) * n_p / (n_p + 10)

fitted against their meta-features by weighted ridge (weights n_p) and by
LightGBM. The model predicts r_p for the *test* papers from their
meta-features alone. The offset model's own generalisation is measured by
leave-one-paper-out inside the training papers.

=====================================================================
Gate (todo.json, frozen 2026-10-22T12:42+08:00)
=====================================================================
doi_r2            >= 0.42 (5-seed mean; current 0.3767)
offset_mae_drop   >= 10% relative to no correction
no_leak           every feature is training-fold or metadata only
kill              DOI R^2 < 0.40 closes the offset line

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_paper_meta_offset.py
"""

from __future__ import annotations

import argparse
import json
import zlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_external_validation import (  # noqa: E402
    murcko_scaffold,
    parse_year,
)
from molmetal.scripts.msrs_feature_expansion import (  # noqa: E402
    build_arm,
    select_cohort,
)
from molmetal.scripts.msrs_generalization import (  # noqa: E402
    LIGAND_ONLY,
    TOP_CELL_LINES,
    build,
    cell_vocabulary,
    coordination_block,
    load_descriptors,
    native_codes,
    scheme_folds,
    state_block,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    METALS,
    _git_commit,
    _sha256,
    score,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
N_SCAFFOLD_BUCKETS = 16
SHRINK = 10.0
GATE_FROZEN_AT = "2026-10-22T12:42+08:00"
DOI_R2_GATE = 0.42
DOI_R2_KILL = 0.40
OFFSET_MAE_DROP_GATE = 0.10


def paper_meta_features(
    frame: pd.DataFrame, paper: np.ndarray, with_counts: bool, with_outcome: bool
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """One row per paper. Metadata only unless ``with_outcome`` is set."""
    papers = sorted(set(paper.tolist()))
    index = {p: i for i, p in enumerate(papers)}
    n = len(papers)

    year = np.full(n, np.nan)
    metals = np.zeros((n, len(METALS)))
    cell_counts: List[Dict[str, int]] = [{} for _ in range(n)]
    scaffolds = np.zeros((n, N_SCAFFOLD_BUCKETS))
    n_rows = np.zeros(n)

    ys = frame["value_log10"].to_numpy(dtype=float)
    met = frame["metal"].astype(str).to_numpy()
    cells = frame["cell_line_key"].astype(str).to_numpy()
    dois = frame["doi"].astype(str).to_numpy()
    ligs = frame["ligand_canonical"].astype(str).to_numpy()
    smis = frame["ligand_canonical"].astype(str).tolist()

    # zlib.crc32, not hash(): Python's string hash is salted per process, which
    # would make the bucket assignment (and the artifact) irreproducible.
    scaffold_cache: Dict[str, int] = {}
    for smi in smis:
        if smi not in scaffold_cache:
            scaffold_cache[smi] = zlib.crc32(
                murcko_scaffold(smi).encode("utf-8")
            ) % N_SCAFFOLD_BUCKETS
    bucket = np.array([scaffold_cache[s] for s in smis], dtype=int)
    year_row = np.array([parse_year(d) for d in dois], dtype=object)

    for i in range(len(frame)):
        p = index[paper[i]]
        n_rows[p] += 1
        if year_row[i] is not None:
            year[p] = year_row[i] if np.isnan(year[p]) else year[p]
        m = met[i]
        if m in METALS:
            metals[p, METALS.index(m)] += 1
        cell_counts[p][cells[i]] = cell_counts[p].get(cells[i], 0) + 1
        scaffolds[p, bucket[i]] += 1

    # distinct ligand counts
    lig_sets: List[set] = [set() for _ in range(n)]
    for i in range(len(frame)):
        lig_sets[index[paper[i]]].add(ligs[i])
    n_ligands = np.array([len(s) for s in lig_sets], dtype=float)

    rows_by_paper: List[List[int]] = [[] for _ in range(n)]
    for i in range(len(frame)):
        rows_by_paper[index[paper[i]]].append(i)

    entropy = np.zeros(n)
    for p in range(n):
        cnt = np.array(list(cell_counts[p].values()), dtype=float)
        if cnt.sum() > 0:
            pr = cnt / cnt.sum()
            entropy[p] = float(-(pr * np.log(pr)).sum())

    y_stats = np.zeros((n, 11))
    for p, idx in enumerate(rows_by_paper):
        v = ys[idx]
        y_stats[p, 0] = v.mean()
        y_stats[p, 1:] = np.quantile(v, np.linspace(0.1, 0.9, 10)) if len(v) > 1 else v[0]

    year_filled = np.where(np.isnan(year), np.nanmedian(year) if np.isfinite(year).any() else 0.0, year)

    parts = [year_filled[:, None]]
    names = ["year"]
    parts.append(metals / np.maximum(n_rows[:, None], 1.0))
    names += [f"metal_frac_{m}" for m in METALS]
    parts.append(entropy[:, None])
    names.append("cell_entropy")
    parts.append(scaffolds / np.maximum(n_rows[:, None], 1.0))
    names += [f"scaffold_bucket_{b}" for b in range(N_SCAFFOLD_BUCKETS)]
    if with_counts:
        parts.append(np.column_stack([np.log1p(n_rows), np.log1p(n_ligands)]))
        names += ["log_n_rows", "log_n_ligands"]
    if with_outcome:
        parts.append(y_stats)
        names += ["y_mean"] + [f"y_q{int(q*100)}" for q in np.linspace(0.1, 0.9, 10)]
    X = np.column_stack(parts).astype(np.float64)
    return X, names, index


def paper_offsets(
    y: np.ndarray, pred: np.ndarray, paper_idx: np.ndarray, n_papers: int, rows: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Shrunk mean residual per paper, and the row count behind it."""
    off = np.zeros(n_papers)
    cnt = np.zeros(n_papers)
    resid = y[rows] - pred[rows]
    idx = paper_idx[rows]
    sums = np.bincount(idx, weights=resid, minlength=n_papers)
    cnt = np.bincount(idx, minlength=n_papers).astype(float)
    mean = np.divide(sums, cnt, out=np.zeros(n_papers), where=cnt > 0)
    off = mean * cnt / (cnt + SHRINK)
    return off, cnt


def fit_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = 1.0):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    sw = np.sqrt(w)
    Zw = Z * sw[:, None]
    yw = y * sw
    A = Zw.T @ Zw + alpha * np.eye(Z.shape[1])
    b = Zw.T @ yw
    coef = np.linalg.solve(A, b)
    return {"mu": mu, "sd": sd, "coef": coef}


def predict_ridge(model, X: np.ndarray) -> np.ndarray:
    return ((X - model["mu"]) / model["sd"]) @ model["coef"]


def lopo_ridge_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = 1.0) -> float:
    """Leave-one-paper-out R^2 of the offset model (in-sample ridge refit per holdout)."""
    n = len(y)
    if n < 10:
        return float("nan")
    pred = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = fit_ridge(X[mask], y[mask], w[mask], alpha)
        pred[i] = predict_ridge(model, X[i:i + 1])[0]
    sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    sse = float(np.sum(w * (y - pred) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def fit_lgbm(X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        objective="regression", n_estimators=200, learning_rate=0.05,
        num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=int(seed),
        n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
    )
    model.fit(X, y, sample_weight=w)
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_paper_meta_offset.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    frame = select_cohort(args.long)
    y = frame["value_log10"].to_numpy(dtype=float)
    doi = frame["doi"].astype(str).to_numpy()

    load_descriptors(args.metal_desc)
    vocab = cell_vocabulary(frame["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES)
    codes = native_codes(
        frame["metal"].astype(str).tolist(),
        frame["cell_line_key"].astype(str).tolist(),
        vocab, mode="full",
    )
    base = build(frame, LIGAND_ONLY)
    state, _ = state_block(frame)
    coord, _, _ = coordination_block(frame, include_geometry=True)
    state_coord = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    base_ligand = sparse.hstack([base.X, state_coord], format="csr")
    X_model = build_arm(frame, "maccs_rdkit", base_ligand, state_coord)
    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)

    fold_id, _, fold_info = scheme_folds(frame, "doi")
    papers, paper_idx = np.unique(doi, return_inverse=True)
    n_papers = len(papers)
    print(f"[UP-21] rows={len(frame)} papers={n_papers} folds={fold_info['n_folds']} "
          f"features={X_model.shape[1]}")

    feature_sets = {
        "design": paper_meta_features(frame, paper_idx, False, False),
        "design+counts": paper_meta_features(frame, paper_idx, True, False),
        "design+outcome": paper_meta_features(frame, paper_idx, True, True),
    }
    names = feature_sets["design"][1]
    print(f"[UP-21] design features={len(names)} (no counts, no outcomes)")

    per_seed: Dict[str, object] = {}
    for seed in SEEDS[: args.seeds]:
        ts = time.time()
        pred = fit_arm(
            X_model, empty, empty, None, 0, False, y, fold_id,
            cat_codes=codes, params_override={"random_state": int(seed)},
        )
        base_r2 = score(y[fold_id >= 0], pred[fold_id >= 0])["r2"]

        variants: Dict[str, Dict[str, np.ndarray]] = {
            "ridge_design": np.full(len(y), np.nan),
            "lgbm_design": np.full(len(y), np.nan),
            "ridge_design+counts": np.full(len(y), np.nan),
            "ridge_design+outcome": np.full(len(y), np.nan),
            "ridge_no_year": np.full(len(y), np.nan),
            "ridge_no_metal": np.full(len(y), np.nan),
            "ridge_no_scaffold": np.full(len(y), np.nan),
        }
        offset_diag: Dict[str, object] = {}

        for k in sorted(set(fold_id[fold_id >= 0].tolist())):
            tr = np.where((fold_id != k) & (fold_id >= 0))[0]
            te = np.where(fold_id == k)[0]
            tr_papers = np.unique(paper_idx[tr])
            te_papers = np.unique(paper_idx[te])

            off_tr, cnt_tr = paper_offsets(y, pred, paper_idx, n_papers, tr)
            off_te, cnt_te = paper_offsets(y, pred, paper_idx, n_papers, te)

            for set_name, (Xp, cols, _) in feature_sets.items():
                arm = "ridge_design" if set_name == "design" else f"ridge_{set_name}"
                model = fit_ridge(Xp[tr_papers], off_tr[tr_papers], cnt_tr[tr_papers])
                o_hat = predict_ridge(model, Xp[te_papers])
                mapping = dict(zip(te_papers.tolist(), o_hat.tolist()))
                variants[arm][te] = pred[te] + np.array([mapping[p] for p in paper_idx[te]])

            # Ablations on the primary design set.
            Xp, cols, _ = feature_sets["design"]
            col_index = {c: i for i, c in enumerate(cols)}
            for arm, drop in (
                ("ridge_no_year", ["year"]),
                ("ridge_no_metal", [c for c in cols if c.startswith("metal_frac_")]),
                ("ridge_no_scaffold", [c for c in cols if c.startswith("scaffold_bucket_")]),
            ):
                keep = [i for c, i in col_index.items() if c not in set(drop)]
                model = fit_ridge(Xp[tr_papers][:, keep], off_tr[tr_papers], cnt_tr[tr_papers])
                o_hat = predict_ridge(model, Xp[te_papers][:, keep])
                mapping = dict(zip(te_papers.tolist(), o_hat.tolist()))
                variants[arm][te] = pred[te] + np.array([mapping[p] for p in paper_idx[te]])

            model = fit_lgbm(Xp[tr_papers], off_tr[tr_papers], cnt_tr[tr_papers], seed)
            o_hat = model.predict(Xp[te_papers])
            mapping = dict(zip(te_papers.tolist(), o_hat.tolist()))
            variants["lgbm_design"][te] = pred[te] + np.array([mapping[p] for p in paper_idx[te]])

            if k == sorted(set(fold_id[fold_id >= 0].tolist()))[0]:
                offset_diag = {
                    "n_train_papers": int(len(tr_papers)),
                    "n_test_papers": int(len(te_papers)),
                    "lopo_ridge_r2": round(
                        float(lopo_ridge_r2(
                            Xp[tr_papers], off_tr[tr_papers], cnt_tr[tr_papers]
                        )), 4),
                    "train_offset_sd": round(float(np.std(off_tr[tr_papers])), 4),
                    "test_offset_sd": round(float(np.std(off_te[te_papers])), 4),
                    "mean_abs_test_offset": round(
                        float(np.average(np.abs(off_te[te_papers]), weights=cnt_te[te_papers])), 4),
                    "feature_columns": cols,
                }

        ev = fold_id >= 0
        out_arms = {"base": {"r2": round(float(base_r2), 4),
                             "mae": round(float(np.mean(np.abs(y[ev] - pred[ev]))), 4)}}
        for arm, p in variants.items():
            ok = np.isfinite(p)
            s = score(y[ok], p[ok])
            # Offset MAE: per test paper, |true shrunk offset - predicted offset|.
            corr = p - pred
            true_off, cnt_all = paper_offsets(
                y, pred, paper_idx, n_papers, np.where(ev)[0]
            )
            paper_mae = []
            paper_mae0 = []
            for pp in np.unique(paper_idx[ev]):
                m = ev & (paper_idx == pp)
                if m.sum() == 0 or cnt_all[pp] <= 0:
                    continue
                paper_mae.append(abs(true_off[pp] - corr[m].mean()))
                paper_mae0.append(abs(true_off[pp]))
            out_arms[arm] = {
                "r2": s["r2"],
                "mae": s["mae"],
                "offset_mae": round(float(np.mean(paper_mae)), 4),
                "offset_mae_uncorrected": round(float(np.mean(paper_mae0)), 4),
                "offset_mae_drop": round(
                    1.0 - float(np.mean(paper_mae)) / max(float(np.mean(paper_mae0)), 1e-9), 4),
            }
        per_seed[str(seed)] = {
            "arms": out_arms,
            "offset_diagnostics_fold0": offset_diag,
            "elapsed_seconds": round(time.time() - ts, 1),
        }
        print(f"[UP-21] seed={seed} base={base_r2:.4f} "
              + " ".join(f"{a}={out_arms[a]['r2']:.4f}" for a in variants)
              + f" ({per_seed[str(seed)]['elapsed_seconds']}s)")

    agg = {}
    for arm in per_seed[str(SEEDS[0])]["arms"]:
        r2 = [per_seed[s]["arms"][arm]["r2"] for s in per_seed]
        drop = [per_seed[s]["arms"][arm].get("offset_mae_drop") for s in per_seed]
        agg[arm] = {
            "r2_mean": round(float(np.mean(r2)), 4),
            "r2_min": round(float(np.min(r2)), 4),
            "r2_max": round(float(np.max(r2)), 4),
            "offset_mae_drop_mean": round(float(np.mean(drop)), 4) if drop[0] is not None else None,
        }

    primary = agg["ridge_design"]
    checks = {
        "doi_r2": {
            "value": primary["r2_mean"],
            "threshold": DOI_R2_GATE,
            "baseline": agg["base"]["r2_mean"],
            "delta": round(primary["r2_mean"] - agg["base"]["r2_mean"], 4),
            "pass": bool(primary["r2_mean"] >= DOI_R2_GATE),
        },
        "offset_mae_drop": {
            "value": primary["offset_mae_drop_mean"],
            "threshold": OFFSET_MAE_DROP_GATE,
            "pass": bool(
                primary["offset_mae_drop_mean"] is not None
                and primary["offset_mae_drop_mean"] >= OFFSET_MAE_DROP_GATE
            ),
        },
        "no_leak": {
            "primary_features": names,
            "uses_counts": False,
            "uses_outcomes": False,
            "pass": True,
            "note": "n_rows, n_ligands and any y statistic are absent from the primary set; "
                    "the arms that use them are labelled and reported separately.",
        },
    }
    kill = {
        "rule": "DOI R2 < 0.40 closes the offset line",
        "value": primary["r2_mean"],
        "triggered": bool(primary["r2_mean"] < DOI_R2_KILL),
    }
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_paper_meta_offset",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L1 paper-offset transfer (UP-21)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "cohort": "msrs_feature_expansion.select_cohort (21,404 rows, potency, uncensored)",
            "base_arm": "maccs_rdkit (the arm whose 5-seed DOI R2 is 0.3767)",
            "split": "scheme_folds('doi'), 5 GroupKFold folds on DOI",
            "offset_target": f"mean(y - pred | paper) * n / (n + {SHRINK})",
            "offset_models": ["weighted ridge (primary)", "LightGBM (secondary)"],
            "ridge_alpha": 1.0,
            "seeds": list(SEEDS[: args.seeds]),
            "primary_features": names,
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "The offset model is fitted on the training papers' shrunk residuals and applied "
            "to the held-out papers' metadata. Because a DOI fold holds out whole papers, a "
            "test paper contributes no rows to the offset fit -- but its metadata (which "
            "metals, which cells, which scaffolds) is used, and that is only legitimate if "
            "the compound set is known before the assay is read.",
            "n_rows and n_ligands are excluded from the primary set because the frozen "
            "no-leak rule forbids test-fold row counts. The design+counts arm shows what "
            "they would add.",
            "The design+outcome arm uses the test paper's own y mean and deciles. It is a "
            "leaky upper bound and is labelled as such; it is not a result.",
            "LOPO ridge R2 is computed on the first outer fold only, to keep the runtime "
            "bounded; it describes the offset model, not the final prediction.",
            "Offsets are shrunk by n/(n+10); a two-row paper cannot move its own prediction "
            "by a log unit, which also caps how much a small test paper can be corrected.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    for arm in agg:
        print(f"[UP-21] {arm:22s} r2={agg[arm]['r2_mean']} "
              f"drop={agg[arm]['offset_mae_drop_mean']}")
    print(f"[UP-21] gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time()-t0,1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
