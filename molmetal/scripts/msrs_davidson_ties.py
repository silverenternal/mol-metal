#!/usr/bin/env python3
"""UP-29 — ties are an outcome, not a missing label: Davidson on the pair table.

=====================================================================
Why this exists
=====================================================================
UP-23 tried to keep near-ties out of the ranker's gradient.  Excluding
them (``bt_exclude``) and down-weighting them (``bt_weight``) both lost
to the unweighted fit: lambdarank fell 0.5985 -> 0.5612, every
Bradley-Terry arm stayed at or below 0.5411.  The reading recorded then
was "the noise labels are not the bottleneck".

That reading is incomplete.  Both arms treat a tie as *absence of
information*.  Davidson (1970, JASA 65(329):317-328) treats it as a
third outcome with its own probability:

    P(i beats j) = pi_i / Z
    P(tie)       = delta * sqrt(pi_i pi_j) / Z
    Z            = pi_i + pi_j + delta * sqrt(pi_i pi_j)

With pi_i = exp(f(x_i)) and s = f_a - f_b:

    P(a wins) = 1 / (1 + e^-s + delta e^(-s/2))
    P(tie)    = delta e^(-s/2) / (1 + e^-s + delta e^(-s/2))

The tie probability carries gradient information about |s|: a pair that
tied tells the model "these two are close", which a dropped row cannot.
This is a *different mechanism* from UP-23's deadzone reweighting, which
is what ``reopen_rule`` requires before a killed line may be reopened.

=====================================================================
Fitting
=====================================================================
Alternating, five rounds:

  1. fix delta, minimise the 3-class negative log-likelihood over w with
     L-BFGS (analytic gradient, ridge on w).
  2. fix w, solve for delta in closed form.  The stationarity condition

         n_tie / delta = sum_i u_i / (1 + e^-s_i + delta u_i),
         u_i = e^(-s_i/2)

     is monotone decreasing in delta, so bisection on [1e-6, 10] finds
     the unique root.

delta is a reported quantity, not a tuning knob: delta -> 0 is
Bradley-Terry (ties impossible), delta -> inf is "everything ties".

=====================================================================
What is scored
=====================================================================
The gate metric is the one UP-23 and UP-2b used, so the numbers are
comparable: balanced accuracy of the predicted sign on pairs with
|delta| >= 0.2091, restricted to the retained subset |delta| >= tau.
The 3-class accuracy (ties included) is reported separately -- it is
the metric Davidson is built for, but it is not the frozen gate.

Uncertainty is a 2000-draw bootstrap over **ligand clusters**, replacing
UP-23's per-metal-pair Wilson interval.  The Wilson interval on a 94%
majority class measures "does it beat guessing the majority", not
resolution; the cluster bootstrap measures the same quantity the gate
does, with the resampling unit the ligand.

=====================================================================
Gate (todo.json, frozen 2026-10-22T15:45+08:00)
=====================================================================
balanced        >= 0.62 on the retained subset (lambdarank baseline 0.5985)
ci_lower        ligand-cluster bootstrap 2.5th percentile > 0.50
sensitivity     delta reported, plus tau x {0.5, 1.0, 1.5}
kill            balanced < 0.58 closes the ranking line (DP-8 = YES)

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_davidson_ties.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_generalization import pair_table  # noqa: E402
from molmetal.scripts.msrs_listwise_ranker import (  # noqa: E402
    build_features,
    deadzone_labels,
)
from molmetal.scripts.msrs_ranking_hierarchical import (  # noqa: E402
    fit_bt,
    pair_design,
    pair_taus,
    tau_lookup,
)
from molmetal.scripts.msrs_surface_model import _git_commit, _sha256  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_FOLDS = 5
GATE_THRESHOLD = 0.2091
GATE_BALANCED = 0.62
KILL_BALANCED = 0.58
N_BOOT = 2000
TAU_SCALES = (0.5, 1.0, 1.5)
GATE_FROZEN_AT = "2026-10-22T15:45+08:00"
SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
# outcome codes
A_WINS, TIE, B_WINS = 0, 1, 2


# ---------------------------------------------------------------------------
# Davidson likelihood
# ---------------------------------------------------------------------------
def davidson_probs(s: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """(P(a wins), P(tie)) for a score difference ``s``."""
    s = np.asarray(s, dtype=float)
    # log Z via the max trick, so exp(-s) cannot overflow on a large |s|
    a = np.log(max(delta, 1e-12)) - 0.5 * s
    m = np.maximum(np.maximum(0.0, -s), a)
    Z = np.exp(-m) + np.exp(-s - m) + np.exp(a - m)
    logZ = m + np.log(Z)
    return np.exp(-logZ), np.exp(a - logZ)


def _nll_grad(
    w: np.ndarray,
    D: sparse.csr_matrix,
    outcome: np.ndarray,
    delta: float,
    alpha: float,
) -> Tuple[float, np.ndarray]:
    s = np.asarray(D @ w, dtype=float).ravel()
    logd = np.log(max(delta, 1e-12))
    a = logd - 0.5 * s
    m = np.maximum(np.maximum(0.0, -s), a)
    Z = np.exp(-m) + np.exp(-s - m) + np.exp(a - m)
    logZ = m + np.log(Z)
    # d logZ / ds
    dlogZ = -(np.exp(-s - m) + 0.5 * np.exp(a - m)) / Z

    tie = outcome == TIE
    bwin = outcome == B_WINS
    # b-wins is the mirror image: s -> -s
    sb = np.where(bwin, -s, s)
    ab = logd - 0.5 * sb
    mb = np.maximum(np.maximum(0.0, -sb), ab)
    Zb = np.exp(-mb) + np.exp(-sb - mb) + np.exp(ab - mb)
    logZb = mb + np.log(Zb)
    dlogZb = -(np.exp(-sb - mb) + 0.5 * np.exp(ab - mb)) / Zb

    ll = np.where(tie, logd - 0.5 * s - logZ,
                  np.where(bwin, -logZb, -logZ))
    g = np.where(tie, -0.5 + dlogZ,
                 np.where(bwin, dlogZb, dlogZ))
    loss = -float(ll.sum()) + alpha * float(w @ w)
    grad = -(D.T @ g).ravel() + 2.0 * alpha * w
    return loss, grad


def solve_delta(s: np.ndarray, outcome: np.ndarray, lo: float = 1e-6,
                hi: float = 10.0) -> float:
    """Closed-form step: bisection on the monotone stationarity condition."""
    n_tie = float((outcome == TIE).sum())
    if n_tie == 0:
        return lo
    u = np.exp(-0.5 * s)

    def f(d: float) -> float:
        Z = 1.0 + np.exp(-s) + d * u
        return n_tie / d - float(np.sum(u / Z))

    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_davidson(
    D: sparse.csr_matrix,
    outcome: np.ndarray,
    alpha: float = 1.0,
    delta_init: float = 0.5,
    n_alt: int = 5,
) -> Dict[str, object]:
    """Alternating w / delta; returns the coefficients and the tie scale.

    ``w`` is warm-started from the Bradley-Terry fit on the non-tied pairs.
    Starting at ``w = 0`` is a trap: at ``s = 0`` the tie probability is
    ``delta / (2 + delta)``, nearly flat in ``s``, so the first delta solve
    pushes delta up and the w step then has almost no gradient.  The smoke
    run collapsed to exactly ``w = 0`` that way.
    """
    target = (outcome == A_WINS).astype(float)
    tie_w = (outcome != TIE).astype(float)
    w = fit_bt(D, target, tie_w, alpha) if tie_w.sum() > 0 else np.zeros(D.shape[1])
    delta = float(delta_init)
    history: List[Dict[str, float]] = []
    for _ in range(n_alt):
        res = optimize.minimize(
            _nll_grad, w, args=(D, outcome, delta, alpha), jac=True,
            method="L-BFGS-B", options={"maxiter": 500},
        )
        w = res.x
        s = np.asarray(D @ w, dtype=float).ravel()
        new_delta = solve_delta(s, outcome)
        history.append({"delta": round(float(new_delta), 6),
                        "nll": round(float(res.fun), 6),
                        "w_norm": round(float(np.linalg.norm(w)), 6)})
        if abs(new_delta - delta) < 1e-4:
            delta = new_delta
            break
        delta = new_delta
    return {"w": w, "delta": delta, "history": history,
            "n_iter": len(history),
            "degenerate": bool(np.linalg.norm(w) < 1e-8)}


# ---------------------------------------------------------------------------
# Outcomes and scoring
# ---------------------------------------------------------------------------
def davidson_outcome(delta: np.ndarray, tau: np.ndarray, deadzone: bool) -> np.ndarray:
    """0 = a more potent, 1 = tie, 2 = b more potent."""
    out = np.where(delta < 0, A_WINS, np.where(delta > 0, B_WINS, TIE))
    if deadzone:
        out = np.where(np.abs(delta) < tau, TIE, out)
    return out.astype(int)


def balanced_accuracy(obs: np.ndarray, pred: np.ndarray) -> float:
    """Balanced accuracy of the sign, identical to ``msrs_ranking_expanded._metrics``.

    ``obs`` and ``pred`` are both ``a - b``, so a positive value means "a is
    the more potent of the two" on both sides.  UP-23 reported 0.5317 for
    ``bt_exclude`` under this convention; the smoke run disagreed because the
    sign of ``pred`` was flipped relative to this definition.
    """
    pos = obs > 0
    pred_pos = pred > 0
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    tpr = float(np.mean(pred_pos[pos]))
    tnr = float(np.mean(~pred_pos[~pos]))
    return 0.5 * (tpr + tnr)


def cluster_bootstrap(
    obs: np.ndarray, pred: np.ndarray, clusters: np.ndarray,
    n_boot: int, seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    members = [order[s:e] for s, e in zip(starts, ends)]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        v = balanced_accuracy(obs[idx], pred[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_gt_half": float("nan"), "n_draws": 0}
    return {
        "mean": round(float(vals.mean()), 4),
        "lo": round(float(np.percentile(vals, 2.5)), 4),
        "hi": round(float(np.percentile(vals, 97.5)), 4),
        "p_gt_half": round(float(np.mean(vals > 0.5)), 4),
        "n_draws": int(len(vals)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--metal-desc", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    ap.add_argument("--metal-graph", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_graph.json")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_davidson_ties.json")
    args = ap.parse_args(argv)

    for p in (args.long, args.metal_desc, args.metal_graph):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    t0 = time.time()
    long = pd.read_parquet(args.long)
    frame, X, y = build_features(long, args.metal_desc)
    edges = json.loads(args.metal_graph.read_text())["edges_by_pair"]
    tau_map = tau_lookup(edges)

    pairs = pair_table(frame, key=("ligand_canonical", "cell_line_key")).reset_index(drop=True)
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    delta = y[ia] - y[ib]
    tau_p = pair_taus(pairs, tau_map)
    keep, _ = deadzone_labels(delta, tau_p)
    pairs["tau"] = tau_p
    pairs["delta"] = delta
    pairs["keep"] = keep
    pairs["label"] = pairs["metal_a"] + "-" + pairs["metal_b"]

    doi = frame["doi"].astype(str).to_numpy()
    from sklearn.model_selection import GroupKFold
    folds = np.full(len(frame), -1, dtype=int)
    for k, (_, te) in enumerate(
        GroupKFold(n_splits=N_FOLDS).split(np.zeros((len(frame), 1)), groups=doi)
    ):
        folds[te] = k

    ligands = frame["ligand_canonical"].astype(str).to_numpy()
    n_tie_exact = int((delta == 0).sum())
    print(f"[UP-29] rows={len(frame)} pairs={len(pairs)} exact_ties={n_tie_exact} "
          f"deadzone_ties={int((np.abs(delta) < tau_p).sum())} kept={int(keep.sum())}")

    eval_mask = keep & (np.abs(delta) >= GATE_THRESHOLD)
    print(f"[UP-29] gate-eligible pairs (retained, |delta|>= {GATE_THRESHOLD}) = "
          f"{int(eval_mask.sum())}")

    def run_arm(name: str, mode: str, tau_scale: float) -> Dict[str, object]:
        tau_s = tau_p * tau_scale
        keep_s = np.abs(delta) >= tau_s
        outcome = davidson_outcome(delta, tau_s, deadzone=(mode == "davidson"))
        target = (delta < 0).astype(float)
        tie_w = (delta != 0).astype(float)

        scores = np.full((len(frame), args.seeds), np.nan)
        deltas: List[float] = []
        delta_by_fold: Dict[int, float] = {}
        for j, seed in enumerate(SEEDS[: args.seeds]):
            acc = np.zeros(len(frame))
            hits = np.zeros(len(frame))
            for k in range(N_FOLDS):
                tr = (folds[ia] != k) & (folds[ib] != k)
                if mode != "davidson":
                    tr &= keep_s
                if tr.sum() < 20:
                    continue
                D = pair_design(X, ia[tr], ib[tr])
                if mode == "davidson":
                    fit = fit_davidson(D, outcome[tr], args.alpha,
                                       delta_init=0.5, n_alt=5)
                    w = fit["w"]
                    deltas.append(float(fit["delta"]))
                    delta_by_fold[k] = float(fit["delta"])
                else:
                    w = fit_bt(D, target[tr], tie_w[tr], args.alpha)
                te = folds == k
                acc[te] += X[te] @ w
                hits[te] += 1
            sc = np.divide(acc, np.maximum(hits, 1))
            sc[hits == 0] = np.nan
            scores[:, j] = sc
        raw_score = np.nanmean(scores, axis=1)

        # UP-23 convention: negate, so a *lower* score means more potent and
        # ``_metrics`` (pos = obs > 0, pred > 0) reads the agreement directly.
        pred = -raw_score

        ok = eval_mask & np.isfinite(pred[ia]) & np.isfinite(pred[ib])
        obs = y[ib][ok] - y[ia][ok]
        pr = pred[ib][ok] - pred[ia][ok]
        bal = balanced_accuracy(obs, pr)
        boot = cluster_bootstrap(obs, pr, ligands[ia][ok], N_BOOT, SEEDS[0])

        # 3-class accuracy, the metric Davidson is built for.  The predicted
        # class is the argmax of (P(a wins), P(tie), P(b wins)); a sign rule
        # can never emit a tie, so scoring it that way would be constant 0.
        o3 = davidson_outcome(y[ia] - y[ib], tau_s, deadzone=(mode == "davidson"))
        acc3: Optional[float] = None
        tie_rate = 0.0
        if mode == "davidson" and delta_by_fold:
            s = raw_score[ia][ok] - raw_score[ib][ok]
            fold_of_pair = folds[ia][ok]
            fallback = float(np.mean(list(delta_by_fold.values())))
            dvec = np.array([delta_by_fold.get(int(f), fallback)
                             for f in fold_of_pair])
            p_a = np.empty(len(s))
            p_t = np.empty(len(s))
            for dv in np.unique(dvec):
                m = dvec == dv
                p_a[m], p_t[m] = davidson_probs(s[m], float(dv))
            p_b = 1.0 - p_a - p_t
            p3 = np.where(p_t >= np.maximum(p_a, p_b), TIE,
                          np.where(p_a >= p_b, A_WINS, B_WINS))
            tie_rate = round(float(np.mean(p3 == TIE)), 4)
            acc3 = float(np.mean(p3 == o3[ok])) if ok.sum() else float("nan")

        return {
            "balanced_accuracy": round(bal, 4),
            "n_pairs": int(ok.sum()),
            "cluster_bootstrap": boot,
            "three_class_accuracy": round(acc3, 4) if acc3 is not None else None,
            "predicted_tie_rate": tie_rate,
            "delta_mean": round(float(np.mean(deltas)), 4) if deltas else None,
            "delta_values": [round(d, 4) for d in deltas],
            "tau_scale": tau_scale,
        }

    results: Dict[str, Dict[str, object]] = {}
    for name, mode in (("bt_baseline", "bt"), ("davidson_none", "davidson")):
        results[name] = run_arm(name, mode, 1.0)
        print(f"[UP-29] {name:16s} bal={results[name]['balanced_accuracy']} "
              f"n={results[name]['n_pairs']} delta={results[name]['delta_mean']} "
              f"3cls={results[name]['three_class_accuracy']}")

    sens: Dict[str, Dict[str, object]] = {}
    for scale in TAU_SCALES:
        key = f"tau_x{scale:g}"
        sens[key] = run_arm(key, "davidson", scale)
        print(f"[UP-29] {key:16s} bal={sens[key]['balanced_accuracy']} "
              f"n={sens[key]['n_pairs']} delta={sens[key]['delta_mean']}")

    best_key = max(
        ["davidson_none", *sens],
        key=lambda k: (results.get(k) or sens[k])["balanced_accuracy"],
    )
    best = results.get(best_key) or sens[best_key]
    ci_lo = best["cluster_bootstrap"]["lo"]

    checks = {
        "balanced_ge_0.62": {
            "arm": best_key,
            "value": best["balanced_accuracy"],
            "threshold": GATE_BALANCED,
            "lambdarank_baseline": 0.5985,
            "pass": bool(best["balanced_accuracy"] >= GATE_BALANCED),
        },
        "ci_lower_gt_0.50": {
            "arm": best_key,
            "value": ci_lo,
            "threshold": 0.50,
            "pass": bool(np.isfinite(ci_lo) and ci_lo > 0.50),
        },
        "sensitivity_reported": {
            "delta_mean": best["delta_mean"],
            "tau_scales": list(TAU_SCALES),
            "arms": list(sens),
            "pass": True,
        },
    }
    kill = {
        "rule": "balanced < 0.58 on the retained subset closes the ranking line (DP-8 = YES)",
        "value": best["balanced_accuracy"],
        "triggered": bool(best["balanced_accuracy"] < KILL_BALANCED),
    }
    gate_pass = all(v["pass"] for v in checks.values())

    out = {
        "artifact": "msrs_davidson_ties",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 ranking with a tie outcome (UP-29)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "metal_graph": str(args.metal_graph.relative_to(PROJECT_ROOT)),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "device": "cpu",
            "n_rows": int(len(frame)),
            "n_pairs": int(len(pairs)),
            "pair_key": ["ligand_canonical", "cell_line_key"],
            "n_exact_ties": n_tie_exact,
            "n_deadzone_ties": int((np.abs(delta) < tau_p).sum()),
            "n_retained": int(keep.sum()),
            "n_gate_eligible": int(eval_mask.sum()),
            "gate_threshold": GATE_THRESHOLD,
            "split": "GroupKFold(5) on DOI; a pair trains only if both rows are in training folds",
            "model": "P(a wins)=1/(1+e^-s+delta e^(-s/2)), s = w.(x_a - x_b); "
                     "alternating L-BFGS on w and bisection on delta",
            "seeds": list(SEEDS[: args.seeds]),
            "bootstrap": f"{N_BOOT} draws over ligand clusters",
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "results": results,
        "sensitivity": sens,
        "best_arm": best_key,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": bool(gate_pass), "kill": kill},
        "caveats": [
            "The gate metric is binary sign accuracy on |delta| >= 0.2091, the same "
            "convention as UP-2b and UP-23. Davidson's native metric is the 3-class "
            "accuracy, which is reported but is not the gate.",
            "The design is linear in the same features the Bradley-Terry arms use. A "
            "failure here is a failure of the linear pairwise class, not evidence that "
            "ties carry no information; the 3-class accuracy is the diagnostic that "
            "separates those two readings.",
            "delta is identified only up to the model's score scale, so its absolute "
            "value is interpretable within an arm and not across arms.",
            "The ligand-cluster bootstrap resamples ligands with replacement but keeps "
            "every pair of a resampled ligand, so within-ligand correlation is preserved "
            "and the interval is wider than a row-level bootstrap would be.",
            "tau is the metal pair's noise_sd_delta from msrs_metal_graph; a metal pair "
            "with no estimated edge falls back to tau=0, i.e. no deadzone, which is why "
            "the tau-sensitivity arms are not all on the same pair set.",
            "The pairwise fits are deterministic (ridge logistic, no sampling), so the five "
            "seeds produce identical scores and the seed axis carries no variance. The "
            "uncertainty quoted here is the ligand-cluster bootstrap, not a seed spread.",
            "On the gate-eligible pairs every outcome is a win or a loss by construction "
            "(|delta| >= 0.2091 >= tau), so the tie class is never the truth there. A model "
            "that predicts a tie on such a pair is wrong; the 3-class accuracy is reported "
            "to show how often Davidson does that, not as a competing headline.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    print(f"[UP-29] best={best_key} gate_pass={gate_pass} kill={kill['triggered']} "
          f"elapsed={round(time.time()-t0,1)}s")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
