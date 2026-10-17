#!/usr/bin/env python
"""FU3 — metal-conditioned router: is metal's role to ROUTE, not to SCORE?

=====================================================================
Gap G3 (todo.json :: gap_analysis_2026_10_16.gaps[G3])
=====================================================================
The metal-aware injections I1-I4 are negative results as *scorers*:

* ``metrics/by_metric/pa15_inhouse_metalcond_lomo.json`` — token/desc/film all
  fail Holm (``significant_after_holm_0.05 == []``).
* ``metrics/by_metric/ex1_i2_rich_metal_cap512.json`` — ``message - readout``
  is +0.0274, Holm 0.0491, i.e. it only just survives.

G3's remedy is to change the *role* of the metal: from a feature that is
injected into one model, to a **routing condition** that selects among expert
branches (a metal-aware expert vs a metal-blind expert).  Metal-awareness is
then defined as *dispatch*, not as *lift*.

=====================================================================
What this script does
=====================================================================
It consumes the row-level held-out query scores dumped by
``configs/fu3_metal_router_gpu0.yaml`` (four arms: none / readout / message /
film, identical paired query set per ``(metal, k, seed)``) and fits routers on
CPU.  Nothing here needs a GPU; the GPU run only produces the dump.

Routers (all compared against the same per-cell AUC):

* ``single_arm:<arm>``          — one expert alone (the incumbent).
* ``uniform_avg``               — mean of the four expert scores (no fitting).
* ``best_single_lomo``          — for each held-out metal, pick the arm that was
  best on the OTHER four metals.  **This is the honest deployable baseline**: it
  is what you would actually ship if you refused to route per-ligand.
* ``best_single_oracle``        — the arm that is best on the held-out metal
  itself (hindsight; a bound, not a method).
* ``oracle_row``                — per row, take the expert with the lowest BCE
  (label-leaking UPPER BOUND; no method can beat it).
* ``gate_scores``               — per row, learn to pick the expert with the
  lowest BCE from the expert scores alone.  Trained on the other four metals.
* ``gate_scores_metal``         — same gate, plus the rich metal descriptor.
  Within a held-out metal the descriptor is constant, so it cannot change the
  within-metal ranking — but the *training* pool spans four metals, so the gate
  can learn a metal-dependent dispatch policy.  This is the arm that actually
  tests G3's "metal conditions routing" claim.
* ``policy_metal_desc``         — the crudest metal-conditioned policy: one arm
  per metal, chosen by a metal-descriptor regression on the other four metals.

=====================================================================
Pre-registered expectation (from WP3)
=====================================================================
WP3 measured that metal descriptors add +0.003 AUC over ligand chemistry alone
for predicting which matched pairs flip.  If metal conditions routing the same
way it conditions scoring, then ``gate_scores_metal - gate_scores`` will be
inside the seed spread, and no router will beat ``best_single_lomo`` by more
than that spread.  A negative here closes G3 as an honest negative — it is not
re-scoped.

Usage::

    python molmetal/scripts/fu3_metal_router_ablation.py \
        --dump /tmp/fu3_router_test_scores.jsonl \
        --out  metrics/by_metric/metal_router_ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.data.metal_graph import rich_metal_block  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ARMS = ("none", "readout", "message", "film")
EPS = 1e-6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dump(path: Path) -> Dict[Tuple[str, int, int], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """``{(metal, k, seed): {arm: (y, p_adapt)}}`` with a pairing guard."""
    cells: Dict[Tuple[str, int, int], Dict[str, Tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["metal"], int(rec["k"]), int(rec["seed"]))
            cells[key][rec["arm"]] = (
                np.asarray(rec["y"], dtype=np.float64),
                np.asarray(rec["p_adapt"], dtype=np.float64),
            )
    if not cells:
        raise SystemExit(f"empty dump: {path}")

    # The whole router premise is that the four arms are a PAIRED contrast on an
    # identical query set.  If the row order or the labels differ, every join
    # below is silently wrong, so refuse rather than guess.
    for key, arms in cells.items():
        missing = [a for a in ARMS if a not in arms]
        if missing:
            raise SystemExit(f"cell {key} missing arms {missing}")
        ref_y, ref_p = arms[ARMS[0]]
        for a in ARMS[1:]:
            y, p = arms[a]
            if y.shape != ref_y.shape:
                raise SystemExit(f"cell {key}: arm {a} has {y.shape[0]} rows, {ARMS[0]} has {ref_y.shape[0]}")
            if not np.array_equal(y, ref_y):
                raise SystemExit(f"cell {key}: arm {a} label vector differs from {ARMS[0]} — not paired")
            if p.shape != ref_p.shape:
                raise SystemExit(f"cell {key}: arm {a} score vector differs in shape")
    return dict(cells)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _bce(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1.0 - EPS)
    return -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
def _stack(cells, keys) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a list of cells into (X[N,4], y[N], metal[N])."""
    xs, ys, ms = [], [], []
    for key in keys:
        metal, _k, _seed = key
        arms = cells[key]
        y = arms[ARMS[0]][0]
        x = np.column_stack([arms[a][1] for a in ARMS])
        xs.append(x)
        ys.append(y)
        ms.extend([metal] * len(y))
    return np.vstack(xs), np.concatenate(ys), np.asarray(ms)


def _gate_features(x: np.ndarray, metals: np.ndarray, with_metal: bool) -> np.ndarray:
    """Expert scores + shape statistics (+ rich metal descriptor)."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    feats = [x, mean, std, lo, hi]
    if with_metal:
        desc = np.vstack([rich_metal_block(m) for m in metals]).astype(np.float64)
        feats.append(desc)
    return np.hstack(feats)


def _fit_gate_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, m_tr: np.ndarray,
    x_te: np.ndarray, m_te: np.ndarray, with_metal: bool,
) -> np.ndarray:
    """Pick, per test row, the expert with the lowest training BCE.

    The gate is trained on the *other* metals; the held-out metal is never seen
    during fitting (leave-one-metal-out), so the reported AUC is a genuine
    cross-metal generalisation number, not a within-metal CV number.
    """
    from sklearn.linear_model import LogisticRegression

    bce = np.column_stack([_bce(y_tr, x_tr[:, j]) for j in range(len(ARMS))])
    target = bce.argmin(axis=1)
    if len(np.unique(target)) < 2:
        # Degenerate: one expert dominates the training pool.  Nothing to learn.
        return np.full(len(x_te), int(target[0]), dtype=int)

    f_tr = _gate_features(x_tr, m_tr, with_metal)
    f_te = _gate_features(x_te, m_te, with_metal)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(f_tr, target)
    return clf.predict(f_te).astype(int)


def _metal_desc_policy(x_tr, y_tr, m_tr, m_te) -> int:
    """One arm for the held-out metal, from a ridge fit of per-metal mean AUC."""
    from sklearn.linear_model import Ridge

    metals = sorted(set(m_tr.tolist()))
    if len(metals) < 2:
        return 0
    desc = np.vstack([rich_metal_block(m) for m in metals]).astype(np.float64)
    y = np.array([
        [_auc(y_tr[m_tr == m], x_tr[m_tr == m, j]) for j in range(len(ARMS))]
        for m in metals
    ])
    ridge = Ridge(alpha=1.0)
    ridge.fit(desc, y)
    pred = ridge.predict(rich_metal_block(m_te).reshape(1, -1).astype(np.float64))[0]
    return int(np.nanargmax(pred))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(cells) -> Dict[str, object]:
    metals = sorted({k[0] for k in cells})
    cells_by_metal: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for key in cells:
        cells_by_metal[key[0]].append(key)
    for m in cells_by_metal:
        cells_by_metal[m].sort()

    # Per-cell, per-arm AUC (the raw incumbent numbers).
    per_arm: Dict[str, Dict[str, List[float]]] = {a: defaultdict(list) for a in ARMS}
    for key, arms in cells.items():
        metal = key[0]
        for a in ARMS:
            per_arm[a][metal].append(_auc(*arms[a]))

    # Router AUC, keyed by (router, metal) -> list over cells (seeds).
    routed: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for metal in metals:
        others = [k for k in cells if k[0] != metal]
        x_tr, y_tr, m_tr = _stack(cells, others)
        for key in cells_by_metal[metal]:
            arms = cells[key]
            y = arms[ARMS[0]][0]
            x = np.column_stack([arms[a][1] for a in ARMS])

            routed["uniform_avg"][metal].append(_auc(y, x.mean(axis=1)))

            oracle_choice = np.column_stack([_bce(y, x[:, j]) for j in range(len(ARMS))]).argmin(axis=1)
            routed["oracle_row"][metal].append(_auc(y, x[np.arange(len(y)), oracle_choice]))

            m_te = np.array([metal] * len(y))
            for name, with_metal in (("gate_scores", False), ("gate_scores_metal", True)):
                choice = _fit_gate_predict(x_tr, y_tr, m_tr, x, m_te, with_metal)
                routed[name][metal].append(_auc(y, x[np.arange(len(y)), choice]))

            j = _metal_desc_policy(x_tr, y_tr, m_tr, metal)
            routed["policy_metal_desc"][metal].append(_auc(y, x[:, j]))

        # best_single_lomo: the arm best on the other four metals.
        mean_other = {
            a: float(np.nanmean([v for m2 in metals if m2 != metal for v in per_arm[a][m2]]))
            for a in ARMS
        }
        best_a = max(mean_other, key=mean_other.get)
        routed["best_single_lomo"][metal].extend(per_arm[best_a][metal])
        oracle_a = max(ARMS, key=lambda a: float(np.nanmean(per_arm[a][metal])))
        routed["best_single_oracle"][metal].extend(per_arm[oracle_a][metal])

    return {
        "metals": metals,
        "per_arm_auc": {a: {m: float(np.nanmean(v)) for m, v in per_arm[a].items()} for a in ARMS},
        "routed_auc": {r: {m: float(np.nanmean(v)) for m, v in d.items()} for r, d in routed.items()},
        "routed_auc_std": {
            r: {m: float(np.nanstd(v)) for m, v in d.items()}
            for r, d in routed.items() if r != "policy_metal_desc"
        },
    }


def _paired_test(routed, baseline, metals) -> Dict[str, float]:
    """One-sided paired t-test of each router vs ``baseline`` over (metal) cells."""
    from scipy import stats

    base = np.array([routed[baseline][m] for m in metals], dtype=float)
    out = {}
    pvals = {}
    for r in routed:
        if r == baseline:
            continue
        cur = np.array([routed[r][m] for m in metals], dtype=float)
        d = cur - base
        if len(d) < 2 or np.allclose(d, 0.0):
            pvals[r] = 1.0
            out[r] = float(np.nanmean(d))
            continue
        t, p = stats.ttest_rel(cur, base, alternative="greater")
        pvals[r] = float(p) if np.isfinite(p) else 1.0
        out[r] = float(np.nanmean(d))
    # Holm across the router family.
    order = sorted(pvals, key=lambda r: pvals[r])
    m = len(order)
    holm = {}
    running = 0.0
    for i, r in enumerate(order):
        adj = min(1.0, (m - i) * pvals[r])
        running = max(running, adj)
        holm[r] = running
    return {"delta_vs_" + baseline: out, "holm_vs_" + baseline: holm}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, default=Path("/tmp/fu3_router_test_scores.jsonl"))
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "metal_router_ablation.json")
    args = ap.parse_args(argv)

    if not args.dump.exists():
        raise SystemExit(f"dump not found: {args.dump} (run configs/fu3_metal_router_gpu0.yaml first)")
    raw = args.dump.read_bytes()
    sha = hashlib.sha1(raw).hexdigest()

    cells = load_dump(args.dump)
    ev = evaluate(cells)
    metals = ev["metals"]

    stats_lomo = _paired_test(ev["routed_auc"], "best_single_lomo", metals)
    stats_oracle = _paired_test(ev["routed_auc"], "best_single_oracle", metals)
    stats_metal_gate = _paired_test(ev["routed_auc"], "gate_scores", metals)

    # --- Verdict, in three separable parts -----------------------------------
    # A "router" DISPATCHES among experts; `uniform_avg` does not dispatch, it
    # COMBINES.  Keeping them in one bucket conflates G3's question (does metal
    # condition dispatch?) with the ensembling line (FU2).  So they are judged
    # separately, against the honest deployable baseline `best_single_lomo`.
    dispatch = ["gate_scores", "gate_scores_metal", "policy_metal_desc"]
    mean_auc = {r: float(np.nanmean(list(ev["routed_auc"][r].values()))) for r in ev["routed_auc"]}
    lomo_auc = mean_auc["best_single_lomo"]
    oracle_auc = mean_auc["best_single_oracle"]

    best_dispatch = max(dispatch, key=lambda r: mean_auc[r])
    d_disp = stats_lomo["delta_vs_best_single_lomo"][best_dispatch]
    p_disp = stats_lomo["holm_vs_best_single_lomo"][best_dispatch]
    routing_wins = bool(d_disp > 0 and p_disp < 0.05)

    d_comb = stats_lomo["delta_vs_best_single_lomo"]["uniform_avg"]
    p_comb = stats_lomo["holm_vs_best_single_lomo"]["uniform_avg"]
    combination_wins = bool(d_comb > 0 and p_comb < 0.05)
    d_comb_oracle = stats_oracle["delta_vs_best_single_oracle"]["uniform_avg"]
    p_comb_oracle = stats_oracle["holm_vs_best_single_oracle"]["uniform_avg"]
    combination_matches_oracle = bool(not (d_comb_oracle > 0 and p_comb_oracle < 0.05))

    # Metal contribution to the gate: the gate with the metal descriptor vs the
    # gate without it, same cells.
    d_metal = stats_metal_gate["delta_vs_gate_scores"]["gate_scores_metal"]
    p_metal = stats_metal_gate["holm_vs_gate_scores"]["gate_scores_metal"]
    metal_adds = bool(d_metal > 0 and p_metal < 0.05)

    spread = float(np.nanmax([
        v for m in metals for v in ev["routed_auc_std"].get("best_single_lomo", {}).values()
    ] or [0.0]))

    verdict = "ROUTING_ADDS_SIGNAL" if routing_wins else "ROUTING_DOES_NOT_BEAT_BEST_SINGLE"
    combination_verdict = (
        "PLAIN_AVERAGING_MATCHES_ORACLE_SELECTION"
        if (combination_wins and combination_matches_oracle)
        else ("PLAIN_AVERAGING_BEATS_BEST_SINGLE" if combination_wins else "PLAIN_AVERAGING_NO_GAIN")
    )
    metal_verdict = "METAL_CONDITIONS_ROUTING" if metal_adds else "METAL_DOES_NOT_CONDITION_ROUTING"

    rep = {
        "metric": "metal_router_ablation",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "description": (
            "FU3 / gap G3: does metal-awareness help as a ROUTING condition (dispatch among "
            "metal-blind vs metal-aware D-MPNN experts) rather than as a score injection? "
            "Row-level paired scores from four injection arms, leave-one-metal-out gate fitting."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/fu3_metal_router_ablation.py",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "cohort_spec": "paper_funnel",
        "dump": {"path": str(args.dump), "sha1": sha, "bytes": len(raw), "n_cells": len(cells)},
        "arms": list(ARMS),
        "k": sorted({k[1] for k in cells}),
        "seeds": sorted({k[2] for k in cells}),
        "per_arm_auc": ev["per_arm_auc"],
        "routed_auc": ev["routed_auc"],
        "routed_auc_std": ev["routed_auc_std"],
        "paired_vs_best_single_lomo": stats_lomo,
        "paired_vs_best_single_oracle": stats_oracle,
        "paired_gate_metal_vs_gate_scores": stats_metal_gate,
        "router_mean_auc": mean_auc,
        "seed_spread": spread,
        "verdict": verdict,
        "combination_verdict": combination_verdict,
        "metal_verdict": metal_verdict,
        "acceptance": {
            # G3's actual question: does any DISPATCH mechanism beat the honest
            # deployable single-arm baseline?
            "routing_beats_best_single_lomo": routing_wins,
            "best_dispatch_router": best_dispatch,
            "best_dispatch_auc": mean_auc[best_dispatch],
            "best_dispatch_delta": d_disp,
            "best_dispatch_holm_p": p_disp,
            "best_single_lomo_auc": lomo_auc,
            "best_single_oracle_auc": oracle_auc,
            # The ensembling line, kept separate so it is not read as routing.
            "uniform_avg_auc": mean_auc["uniform_avg"],
            "uniform_avg_delta_vs_best_single_lomo": d_comb,
            "uniform_avg_holm_p": p_comb,
            "uniform_avg_matches_best_single_oracle": combination_matches_oracle,
            "uniform_avg_delta_vs_best_single_oracle": d_comb_oracle,
            "uniform_avg_holm_p_vs_oracle": p_comb_oracle,
            # Metal's contribution to the gate.
            "gate_metal_delta": d_metal,
            "gate_metal_holm_p": p_metal,
            "seed_spread": spread,
        },
        "honest_framing": (
            "MEASURED only. oracle_row and best_single_oracle leak the held-out labels and are "
            "reported strictly as UPPER BOUNDS, never as methods. All learned routers are fitted "
            "leave-one-metal-out, so their AUC is cross-metal. K is fixed at 10; the K-curve is FU4. "
            "DISPATCH and COMBINATION are judged separately: `uniform_avg` does not route, it "
            "averages, so crediting it to G3 would conflate the routing question with the "
            "ensembling line (FU2). Significance is the project standard (paired one-sided t, "
            "Holm across the family); the seed spread is reported as context, not as the gate."
        ),
    }
    atomic_write_file(args.out, json.dumps(rep, indent=2, default=float, ensure_ascii=False))
    print(f"[fu3] routing : {verdict}  best_dispatch={best_dispatch} "
          f"{mean_auc[best_dispatch]:.4f} vs lomo {lomo_auc:.4f} "
          f"(delta {d_disp:+.4f}, holm {p_disp:.3g})")
    print(f"[fu3] combine : {combination_verdict}  uniform_avg {mean_auc['uniform_avg']:.4f} "
          f"(vs lomo {d_comb:+.4f} holm {p_comb:.3g}; vs oracle {d_comb_oracle:+.4f} holm {p_comb_oracle:.3g})")
    print(f"[fu3] metal   : {metal_verdict}  gate metal contribution {d_metal:+.4f} (holm {p_metal:.3g})")
    print(f"[fu3] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
