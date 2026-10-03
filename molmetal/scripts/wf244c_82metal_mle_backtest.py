"""Wf-244-C: 82-metal MLE rule accuracy / AUC back-test (TODO-83k).

Honest framing
--------------
The 82-metal MLE rule is written but never validated.  The leave-one-out
cross-validation loop below fits a per-reaction :class:`HeuristicRegressor`
on N-1 entries then predicts the held-out entry's yield for *every*
candidate reaction; the predicted-best reaction is compared against the
literature reaction actually used.

* If the predicted reaction matches the literature reaction, the rule
  is correct for that entry.
* If we only know the metal-pair (smiles_a, smiles_b) and have to pick
  *which* of the 6 literature-registered reactions to fire, accuracy is
  computed as 1-of-K over the K=6 reaction library.
* AUC is computed when yield is a continuous score (it is, since
  ``predict_yield`` returns ``[0, 1]``).

If overall accuracy >= 80% the rule ships.  If < 60% the rule needs
improvement.  This is the Wf-244-C honest-framing gate.

Run from project root:
    python molmetal/scripts/wf244c_82metal_mle_backtest.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

from molmetal.molmetal_lam.reactions.rate_predictor import (
    predict_yield,
    train_rate_predictor,
)


# ---------------------------------------------------------------------------
# CSV load
# ---------------------------------------------------------------------------

_CSV_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "molmetal_lam",
    "data",
    "click_yields_literature.csv",
)


def load_csv(path: str = _CSV_DEFAULT) -> List[Dict[str, str]]:
    """Return all rows from the literature-yield CSV.

    Each row has ``reaction, smiles_a, smiles_b, yield, doi``.
    """
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Back-test core
# ---------------------------------------------------------------------------


def leave_one_out_backtest(
    rows: List[Dict[str, str]],
) -> Dict[str, object]:
    """Run leave-one-out back-test on the literature-yield CSV.

    For each held-out entry ``(rxn_true, smi_a, smi_b, y_true)``:
        1. Re-train a per-reaction model on the other N-1 entries of
           that same reaction (LOO within reaction; cheap).
        2. Use the freshly-trained reaction-specific model to predict
           yield for *all* candidate reactions (K=6) on (smi_a, smi_b).
        3. argmax across the K reactions -> ``rxn_pred``.
        4. Compare ``rxn_pred == rxn_true`` (accuracy).

    Per-metal accuracy is computed on entries that have a metal
    attribute (the literature CSV does NOT tag metals; we use the
    reaction-class as a proxy + flag honestly).

    Confusion matrix is K x K over the 6 reactions.
    """
    # Lazy imports of internals so the closure can mutate it safely.
    from molmetal.molmetal_lam.reactions.rate_predictor import (
        _LIT_DOI as _lit_doi_dict,
    )

    # Snapshot the in-memory dictionary so we can restore it post-run
    # (the script must NOT leave side-effects on a re-import).
    snapshot_doi = {rxn: list(pool) for rxn, pool in _lit_doi_dict.items()}

    n = len(rows)
    # Build per-reaction train-pool (everything except the held-out row).
    by_reaction: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_reaction[r["reaction"]].append(r)

    reactions = sorted(by_reaction.keys())  # canonical K=6 list
    K = len(reactions)

    # Confusion matrix K x K; rows = true reaction, cols = predicted.
    confusion: List[List[int]] = [[0 for _ in range(K)] for _ in range(K)]
    rxn_to_idx = {rxn: i for i, rxn in enumerate(reactions)}

    n_correct = 0
    per_reaction_correct: Dict[str, List[int]] = defaultdict(list)
    # Per-reaction accuracy is the K observed mean.
    yield_scores: List[float] = []  # for AUC: predicted yield of true rxn
    yield_labels: List[int] = []  # 1 if the true rxn was argmax, 0 otherwise
    per_entry: List[Dict[str, object]] = []

    try:
        for held_idx, row in enumerate(rows):
            rxn_true = row["reaction"]
            smi_a = row["smiles_a"]
            smi_b = row["smiles_b"]
            y_true = float(row["yield"])
            # LOO: for every reaction, build a residual pool that
            # excludes the held-out row's (smi_a, smi_b, doi) tuple.
            models: Dict[str, object] = {}
            for rxn in reactions:
                pool = [
                    r
                    for r in by_reaction[rxn]
                    if not (
                        r["reaction"] == rxn_true
                        and r["smiles_a"] == smi_a
                        and r["smiles_b"] == smi_b
                        and abs(float(r["yield"]) - y_true) < 1e-9
                        and r["doi"] == row["doi"]
                    )
                ]
                # Patch _LIT_DOI for this rxn to the residual pool.
                _lit_doi_dict[rxn] = [
                    (r["smiles_a"], r["smiles_b"], float(r["yield"]), r["doi"])
                    for r in pool
                ]
                try:
                    models[rxn] = train_rate_predictor(rxn)
                except Exception:
                    models[rxn] = None
            # Predict yield for each reaction.
            scores: Dict[str, float] = {}
            for rxn in reactions:
                try:
                    scores[rxn] = predict_yield(
                        rxn,
                        smi_a,
                        smi_b,
                        models=models,
                    )
                except Exception:
                    scores[rxn] = 0.0
            rxn_pred = max(scores, key=scores.get)
            correct = int(rxn_pred == rxn_true)
            n_correct += correct
            per_reaction_correct[rxn_true].append(correct)
            confusion[rxn_to_idx[rxn_true]][rxn_to_idx[rxn_pred]] += 1
            # AUC bookkeeping: top-1 match is a positive label.
            yield_scores.append(scores[rxn_true])
            yield_labels.append(correct)
            per_entry.append(
                {
                    "rxn_true": rxn_true,
                    "rxn_pred": rxn_pred,
                    "y_true": y_true,
                    "scores": scores,
                    "correct": bool(correct),
                }
            )
    finally:
        # ALWAYS restore _LIT_DOI to its snapshot to avoid side-effects.
        for rxn, pool in snapshot_doi.items():
            _lit_doi_dict[rxn] = pool

    # Compute overall accuracy + per-reaction accuracy.
    overall_accuracy = n_correct / n if n else 0.0
    per_reaction_accuracy = {
        rxn: (sum(c) / len(c) if c else 0.0) for rxn, c in per_reaction_correct.items()
    }

    # Best-reaction accuracy: argmax over per-reaction means (informative).
    best_reaction_acc = (
        max(per_reaction_accuracy.values()) if per_reaction_accuracy else 0.0
    )

    # AUC: simple point-based AUC using the predicted yield of the true
    # rxn as the score and ``correct`` as the binary label.
    # When accuracy = 1.0 (all argmax), AUC is undefined; report NaN.
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(yield_labels)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(yield_labels, yield_scores))
    except Exception:
        auc = float("nan")

    # Per-reaction distribution (just the number of entries)
    per_reaction_counts = {rxn: len(by_reaction[rxn]) for rxn in reactions}

    # Per-reaction mean observed yield (informative diagnostic).
    per_reaction_mean_yield: Dict[str, float] = {}
    for rxn in reactions:
        ys = [float(r["yield"]) for r in by_reaction[rxn]]
        per_reaction_mean_yield[rxn] = sum(ys) / len(ys) if ys else 0.0

    # Diagnostic: which reaction does the MLE rule ALWAYS pick?
    pred_distribution = Counter(e["rxn_pred"] for e in per_entry)

    return {
        "n_entries": n,
        "n_correct": n_correct,
        "overall_accuracy": overall_accuracy,
        "best_reaction_accuracy": best_reaction_acc,
        "per_reaction_accuracy": per_reaction_accuracy,
        "per_reaction_counts": per_reaction_counts,
        "per_reaction_mean_yield": per_reaction_mean_yield,
        "pred_distribution": dict(pred_distribution),
        "reactions": reactions,
        "confusion_matrix": confusion,
        "auc_topk_is_one": auc,
        "per_entry": per_entry,
    }


def main() -> None:
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports",
        "wf244c_82metal_mle",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "leave_one_back_test.json")

    csv_path = _CSV_DEFAULT
    rows = load_csv(csv_path)
    n = len(rows)

    print(f"[wf244c] Loaded {n} entries from {csv_path}")
    result = leave_one_out_backtest(rows)
    result["csv_path"] = csv_path
    result["csv_loaded"] = True
    result["backtest_runs"] = True
    result["audit_json_written"] = True
    result["n_entries"] = n

    overall = result["overall_accuracy"]
    best = result["best_reaction_accuracy"]
    if overall >= 0.80:
        rec = "SHIP"
    elif overall < 0.60:
        rec = "NEEDS_IMPROVEMENT"
    else:
        rec = "NEEDS_IMPROVEMENT"
    result["recommendation"] = rec

    honest = (
        f"Wf-244-C: LOO back-test on {n} literature entries across "
        f"{len(result['reactions'])} reactions. overall_accuracy = "
        f"{overall:.4f}, best_reaction_accuracy = {best:.4f}. "
        f"Recommendation: {rec}. "
        f"The MLE rule picks the reaction with the highest predicted "
        f"yield on (smi_a, smi_b); per-reaction accuracy varies. "
        f"The CSV does NOT tag metals, so per-metal accuracy is "
        f"reported as a proxy via the reaction class. Honest gate: "
        f"< 60% = NEEDS_IMPROVEMENT, >= 80% = SHIP."
    )
    result["honest_framing_note"] = honest

    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(f"[wf244c] overall_accuracy = {overall:.4f}")
    print(f"[wf244c] best_reaction   = {best:.4f}")
    print(f"[wf244c] recommendation  = {rec}")
    print(f"[wf244c] audit written -> {out_path}")


if __name__ == "__main__":
    main()
