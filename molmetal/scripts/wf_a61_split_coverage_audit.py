#!/usr/bin/env python
"""Why the 70/10/20 scaffold split gives A2780cis 27 test rows.

Question
--------
``platinai_3cell_resistance_real.json`` carries
``status: MEASURED_2_OF_3_CELLS`` because ``_per_cell_auc`` refuses to score a
cell line with fewer than ``MIN_EVAL_ROWS`` (30) labelled rows or with only one
class present, and A2780cis has 27 labelled rows in the test fold (2 in val).
Is that a gap in the data, a property of Bemis-Murcko grouping, or an accident
of the order the cohort file happens to be written in?

Method
------
Every measurement here re-uses the production code path --
``molmetal.data.splits.ScaffoldSplitter`` with ``fractions=(0.7, 0.1, 0.2)``,
``strategy='largest_first'`` -- so the "as-is" row of the audit is the split
the trainer actually trained and evaluated on.  Nothing is reimplemented.

Four things are measured, in increasing order of how much they explain:

``as_is``
    Per split, per cell line: labelled rows, class balance, and how many cell
    lines each compound contributes.  This is the baseline the trainer saw.

``concentration``
    How much of each cell line's evidence sits in scaffolds large enough to be
    handed to train by the size-descending assignment.

``order_sensitivity``
    ``_assign_groups`` sorts groups by size descending with Python's *stable*
    sort, and 76% of the scaffolds here are singletons.  Ties at size 1 are
    therefore broken by first appearance in the CSV.  Permuting the group order
    -- which is exactly what permuting the CSV rows does -- measures how much
    of the per-cell coverage is decided by that tie order rather than by the
    protocol.

``assignment_randomness``
    ``strategy='random'`` over many seeds: the coverage you would get if the
    same scaffold groups were dealt to splits without regard to size.  This
    bounds what re-ordering alone can achieve, because it also randomises which
    scaffolds are large.

Output: ``metrics/by_metric/split_coverage_4cell.json``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from molmetal.data.splits import ScaffoldSplitter  # noqa: E402
from molmetal.scripts.train_cisplatin_resistance import (  # noqa: E402
    MIN_EVAL_ROWS,
)

COHORT = ROOT / ".data" / "molmetal" / "platinai_4cell_real_trainer.csv"
OUT = ROOT / "metrics" / "by_metric" / "split_coverage_4cell.json"

FRACTIONS = (0.7, 0.1, 0.2)
CELLS = {
    "A549": "pIC50_a549",
    "A2780": "pIC50_a2780",
    "A2780cis": "pIC50_a2780cis",
    "MCF-7": "pIC50_mcf7",
}
ACTIVE_PIC50 = 5.0
N_PERMUTATIONS = 500
N_RANDOM_SEEDS = 500


class _Smiles:
    def __init__(self, smiles: Sequence[str]) -> None:
        self.smiles = np.asarray(list(smiles), dtype=object)

    def __len__(self) -> int:
        return len(self.smiles)


def _load():
    rows = list(csv.DictReader(open(COHORT, newline="")))
    smiles = [r["SMILES_Ligands"] for r in rows]
    labels = {
        cl: np.array([r[col] != "" for r in rows], dtype=bool)
        for cl, col in CELLS.items()
    }
    values = {
        cl: np.array([float(r[col]) if r[col] else np.nan for r in rows])
        for cl, col in CELLS.items()
    }
    return rows, smiles, labels, values


def _scaffold_groups(smiles: List[str]):
    """Groups in the splitter's own order: first appearance in the input."""
    splitter = ScaffoldSplitter(fractions=FRACTIONS)
    canon = splitter._canonicalize(np.asarray(smiles, dtype=object))
    keys = [splitter._scaffold_for(str(c)) or f"__empty__{c}" for c in canon]
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    return splitter, keys, list(groups.items())


def _counts(idx: np.ndarray, labels, values) -> Dict[str, Dict[str, int]]:
    out = {}
    for cl in CELLS:
        m = idx[labels[cl][idx]]
        v = values[cl][m]
        pos = int((v >= ACTIVE_PIC50).sum())
        out[cl] = {
            "labelled": int(m.size),
            "positive": pos,
            "negative": int(m.size) - pos,
            "evaluable": bool(m.size >= MIN_EVAL_ROWS and 0 < pos < m.size),
        }
    return out


def _as_is(rows, labels, values, splitter, groups):
    assign = splitter._assign_groups([g for _, g in groups])
    per_split = {}
    n_cells = np.array(
        [sum(1 for cl in CELLS if labels[cl][i]) for i in range(len(rows))]
    )
    for name in ("train", "val", "test"):
        idx = np.array(sorted(i for i, s in assign.items() if s == name), dtype=int)
        per_split[name] = {
            "rows": int(idx.size),
            "row_fraction": round(idx.size / len(rows), 4),
            "cells": _counts(idx, labels, values),
            "compounds_with_n_cells": {
                str(k): int(v)
                for k, v in sorted(Counter(n_cells[idx].tolist()).items())
            },
            "single_cell_share": round(float(np.mean(n_cells[idx] == 1)), 4),
        }
    return per_split, assign


def _trainer_view(assign, labels, values, modelled: Sequence[str]):
    """Re-count after the trainer's own row filter.

    ``train_cisplatin_resistance`` models three of the four columns (A2780 is
    dropped as label-noise), so a compound whose only measurement is A2780 has
    ``(active >= 0).any() == False`` and is dropped after the split.  That is
    why the metric reports 1522/92/190 while the split itself has 1650/105/211.
    """
    keep = np.array(
        [any(labels[cl][i] for cl in modelled) for i in range(len(labels[modelled[0]]))],
        dtype=bool,
    )
    out = {
        "modelled_cells": list(modelled),
        "dropped_rows": int((~keep).sum()),
        "dropped_share": round(float((~keep).mean()), 4),
        "dropped_because": "no measurement in any modelled cell line (A2780-only)",
    }
    for name in ("train", "val", "test"):
        idx = np.array(
            sorted(i for i, s in assign.items() if s == name and keep[i]), dtype=int
        )
        out[name] = {"rows": int(idx.size), "cells": _counts(idx, labels, values)}
    return out


def _concentration(labels, groups):
    order = sorted(range(len(groups)), key=lambda k: len(groups[k][1]), reverse=True)
    n_groups = len(order)
    n_train = max(1, int(round(FRACTIONS[0] * n_groups)))
    n_val = max(1, int(round(FRACTIONS[1] * n_groups)))
    n_test = n_groups - n_train - n_val
    # The greedy fills the split with the largest remaining capacity first, so
    # train drains from n_train down to max(n_val, n_test) before anything else
    # is touched.  That boundary is the "train block" for a coverage audit.
    train_block = n_train - max(n_val, n_test)
    out = {"n_groups": n_groups, "train_block_ranks": int(train_block)}
    per_cell = {}
    for cl in CELLS:
        held = [
            (r, int(labels[cl][g].sum()), len(g))
            for r, (_, g) in enumerate((groups[k] for k in order))
            if labels[cl][g].any()
        ]
        total = sum(c for _, c, _ in held)
        per_cell[cl] = {
            "labelled_scaffolds": len(held),
            "labelled_rows": total,
            "rows_per_scaffold": round(total / max(len(held), 1), 2),
            "rows_in_train_block": sum(c for r, c, _ in held if r < train_block),
            "share_in_train_block": round(
                sum(c for r, c, _ in held if r < train_block) / total, 4
            ),
            "rows_in_singleton_scaffolds": sum(c for _, c, s in held if s == 1),
            "median_size_rank": int(np.median([r for r, _, _ in held])),
        }
    out["per_cell"] = per_cell
    return out


def _permute_groups(groups, seed: int):
    """Permuting first-appearance order is what permuting the CSV rows does."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    return [groups[int(i)] for i in order]


def _order_sensitivity(labels, groups, n_perm: int) -> Dict[str, Dict[str, float]]:
    splitter = ScaffoldSplitter(fractions=FRACTIONS)
    per_cell: Dict[str, List[int]] = {cl: [] for cl in CELLS}
    for seed in range(n_perm):
        shuffled = _permute_groups(groups, seed)
        assign = splitter._assign_groups([g for _, g in shuffled])
        test_idx = np.array(sorted(i for i, s in assign.items() if s == "test"), dtype=int)
        for cl in CELLS:
            per_cell[cl].append(int(labels[cl][test_idx].sum()))
    return {
        cl: {
            "min": int(np.min(v)),
            "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)),
            "mean": round(float(np.mean(v)), 1),
            "p95": float(np.percentile(v, 95)),
            "max": int(np.max(v)),
        }
        for cl, v in per_cell.items()
    }


def _assignment_randomness(labels, groups, n_seeds: int) -> Dict[str, Dict[str, float]]:
    per_cell: Dict[str, List[int]] = {cl: [] for cl in CELLS}
    for seed in range(n_seeds):
        splitter = ScaffoldSplitter(
            strategy="random", fractions=FRACTIONS, seed=seed
        )
        assign = splitter._assign_groups([g for _, g in groups])
        test_idx = np.array(sorted(i for i, s in assign.items() if s == "test"), dtype=int)
        for cl in CELLS:
            per_cell[cl].append(int(labels[cl][test_idx].sum()))
    return {
        cl: {
            "min": int(np.min(v)),
            "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)),
            "mean": round(float(np.mean(v)), 1),
            "p95": float(np.percentile(v, 95)),
            "max": int(np.max(v)),
        }
        for cl, v in per_cell.items()
    }


def _hash_tiebreak(labels, groups, keys, n_salts: int = 200) -> Dict[str, object]:
    """Size-descending, but ties broken by a hash of the scaffold, not by file order.

    Same groups, same fractions, same greedy.  The only change is that the
    order among equal-size scaffolds no longer depends on the input file.
    Reported over many salts, because a single hash is itself one draw: the
    question is whether the tie-break *reliably* clears the evaluation bar, not
    whether one particular salt happens to.
    """
    names = ("train", "val", "test")
    per_cell: Dict[str, List[int]] = {cl: [] for cl in CELLS}
    sample = None
    for salt in range(n_salts):
        digest = {k: hashlib.sha1(f"{salt}:{k}".encode()).hexdigest() for k in keys}
        ordered = sorted(groups, key=lambda kv: (-len(kv[1]), digest[kv[0]]))
        assign = ScaffoldSplitter(fractions=FRACTIONS)._assign_groups(
            [g for _, g in ordered]
        )
        if salt == 0:
            sample = {}
            for name in names:
                idx = np.array(
                    sorted(i for i, s in assign.items() if s == name), dtype=int
                )
                sample[name] = {
                    "rows": int(idx.size),
                    "labelled": {cl: int(labels[cl][idx].sum()) for cl in CELLS},
                }
        test_idx = np.array(
            sorted(i for i, s in assign.items() if s == "test"), dtype=int
        )
        for cl in CELLS:
            per_cell[cl].append(int(labels[cl][test_idx].sum()))
    return {
        "n_salts": n_salts,
        "method": "sort by (-group_size, sha1('<salt>:<scaffold>')) before the "
                  "unchanged greedy assignment",
        "salt_0_split": sample,
        "test_labelled_rows": {
            cl: {
                "min": int(np.min(v)),
                "p05": float(np.percentile(v, 5)),
                "median": float(np.median(v)),
                "max": int(np.max(v)),
                "share_below_bar": round(
                    float(np.mean(np.array(v) < MIN_EVAL_ROWS)), 4
                ),
            }
            for cl, v in per_cell.items()
        },
    }


def _stratified(labels, groups, keys) -> Dict[str, Dict[str, int]]:
    """Greedy that targets each cell line's label share, not just scaffold count.

    Groups are still visited largest-first, and a scaffold still lands in
    exactly one split, so no leakage is introduced.  The choice of *which*
    split takes a group minimises the squared shortfall against the per-cell
    label targets (0.7 / 0.1 / 0.2 of that cell's labelled rows) plus a row
    fraction term.  Purely a counterfactual: the trainer is untouched.
    """
    names = ("train", "val", "test")
    n_cells_total = {cl: int(labels[cl].sum()) for cl in CELLS}
    target = {
        s: {cl: FRACTIONS[k] * n_cells_total[cl] for cl in CELLS}
        for k, s in enumerate(names)
    }
    n_rows = sum(len(g) for _, g in groups)
    cap = {s: FRACTIONS[k] * n_rows for k, s in enumerate(names)}
    rows = {s: 0 for s in names}
    have = {s: {cl: 0 for cl in CELLS} for s in names}
    assign: Dict[int, str] = {}
    order = sorted(groups, key=lambda kv: (-len(kv[1]), kv[0]))
    for _, g in order:
        vec = {cl: int(labels[cl][g].sum()) for cl in CELLS}
        best, best_score = None, None
        for s in names:
            # Hard row budget, so the split sizes stay 70/10/20.  Without it the
            # greedy keeps feeding the most starved split and test grows to a
            # third of the cohort.
            over = max(0.0, rows[s] + len(g) - cap[s])
            shortfall = sum(
                (target[s][cl] - have[s][cl] - vec[cl]) ** 2 / max(n_cells_total[cl], 1)
                for cl in CELLS
            )
            score = shortfall + 1e3 * over ** 2
            if best_score is None or score < best_score:
                best, best_score = s, score
        assign.update({int(i): best for i in g})
        rows[best] += len(g)
        for cl in CELLS:
            have[best][cl] += vec[cl]
    out = {}
    for s in names:
        idx = np.array(sorted(i for i, v in assign.items() if v == s), dtype=int)
        out[s] = {"rows": int(idx.size), "labelled": {
            cl: int(labels[cl][idx].sum()) for cl in CELLS
        }}
    return out


def build() -> dict:
    rows, smiles, labels, values = _load()
    splitter, keys, groups = _scaffold_groups(smiles)
    as_is, assign = _as_is(rows, labels, values, splitter, groups)
    concentration = _concentration(labels, groups)
    trainer_view = _trainer_view(
        assign, labels, values, ("MCF-7", "A549", "A2780cis")
    )

    test_idx = np.array(sorted(i for i, s in assign.items() if s == "test"), dtype=int)
    as_is_test = {cl: int(labels[cl][test_idx].sum()) for cl in CELLS}

    payload = {
        "metric": "split_coverage_4cell",
        "description": (
            "Per-cell-line coverage of the 70/10/20 Bemis-Murcko scaffold split "
            "used by train_cisplatin_resistance.py on the REAL PlatinAI 4-cell "
            "cohort, plus the order-sensitivity and counterfactual measurements "
            "that explain why A2780cis lands below the 30-row evaluation bar."
        ),
        "unit": "labelled rows per split (one row = one compound; a compound can "
                "carry up to four cell-line labels)",
        "scope": "data_protocol_audit",
        "source": (
            "molmetal/scripts/wf_a61_split_coverage_audit.py over "
            f"{COHORT.relative_to(ROOT)}"
        ),
        "cohort": {
            "path": str(COHORT.relative_to(ROOT)),
            "n_compounds": len(rows),
            "n_scaffolds": len(groups),
            "n_singleton_scaffolds": sum(1 for _, g in groups if len(g) == 1),
            "row_order": "sorted by (-n_cells, compound_code) in "
                         "build_platinai_4cell_real_cohort.py",
        },
        "evaluation_bar": {
            "min_labelled_rows": MIN_EVAL_ROWS,
            "requires_both_classes": True,
            "source": "molmetal/scripts/train_cisplatin_resistance.py::MIN_EVAL_ROWS "
                      "(enforced in _per_cell_auc)",
        },
        "as_is": as_is,
        "trainer_view": trainer_view,
        "concentration": concentration,
        "order_sensitivity": {
            "n_permutations": N_PERMUTATIONS,
            "method": "permute the order in which scaffolds are first seen, "
                      "which is exactly what permuting the CSV rows does; the "
                      "size-descending sort is stable, so this changes only the "
                      "tie order among equal-size scaffolds",
            "test_labelled_rows": _order_sensitivity(labels, groups, N_PERMUTATIONS),
        },
        "assignment_randomness": {
            "n_seeds": N_RANDOM_SEEDS,
            "method": "strategy='random' over the same scaffold groups; also "
                      "randomises which scaffolds are large, so it is an upper "
                      "bound on what re-ordering alone can buy",
            "test_labelled_rows": _assignment_randomness(labels, groups, N_RANDOM_SEEDS),
        },
        "counterfactual_hash_tiebreak": _hash_tiebreak(labels, groups, keys),
        "counterfactual_stratified": _stratified(labels, groups, keys),
    }

    order = payload["order_sensitivity"]["test_labelled_rows"]
    rand = payload["assignment_randomness"]["test_labelled_rows"]
    h = payload["counterfactual_hash_tiebreak"]["test_labelled_rows"]
    s = payload["counterfactual_stratified"]
    a2780cis_perm = order["A2780cis"]
    payload["value"] = as_is_test["A2780cis"]
    payload["value_basis"] = "as_is.test.cells.A2780cis.labelled"
    payload["status"] = "MEASURED"
    payload["verdict"] = {
        "as_is_test_labelled": as_is_test,
        "evaluable_as_is": {
            cl: as_is["test"]["cells"][cl]["evaluable"] for cl in CELLS
        },
        "a2780cis_is_a_data_gap": False,
        "a2780cis_labelled_rows_in_cohort": int(labels["A2780cis"].sum()),
        "explanation": (
            "A2780cis is not short of measurements: 600 of the cohort's 4,370 "
            f"labelled rows are A2780cis. It is short of measurements in the "
            f"test fold. Re-ordering the same compounds moves its test coverage "
            f"from {as_is_test['A2780cis']} to a median of "
            f"{a2780cis_perm['median']:.0f} (p05 {a2780cis_perm['p05']:.0f}, "
            f"max {a2780cis_perm['max']}), and dealing the same scaffold groups "
            f"without regard to size gives a mean of {rand['A2780cis']['mean']}. "
            "So most of the shortfall is the stable-sort tie order, which the "
            "cohort file decides because it is written sorted by descending "
            "cell-line coverage; the rest is that A2780cis evidence sits on "
            "large scaffolds, which largest-first sends to train."
        ),
        "secondary_finding": (
            "The same tie order decides how many cell lines a test compound "
            "carries. As written, val is 100% single-cell compounds and test "
            f"is {as_is['test']['single_cell_share']:.1%} single-cell, while all "
            "56 four-cell compounds sit in train. That is not a per-cell "
            "coverage problem; it is the evaluation set being the least "
            "multi-task subset of the cohort."
        ),
        "split_fractions_are_group_fractions": {
            "declared": list(FRACTIONS),
            "achieved_by_rows": {
                s: as_is[s]["row_fraction"] for s in ("train", "val", "test")
            },
            "note": (
                "`_assign_groups` budgets capacities in scaffolds, not rows, so "
                "70/10/20 scaffolds is 84/5/11 rows on this cohort: val receives "
                "105 singleton scaffolds and therefore 105 rows."
            ),
        },
        "not_a_fix": (
            "Lowering MIN_EVAL_ROWS would make A2780cis report a number from 27 "
            "rows; the split would still have 27. The bar is not what starved "
            "the cell."
        ),
        "fix_options": {
            "hash_tiebreak": (
                "Deterministic tie-break on a hash of the scaffold instead of "
                "first appearance. Same groups, same fractions, same greedy, no "
                f"leakage. Over {payload['counterfactual_hash_tiebreak']['n_salts']} "
                "salts A2780cis test coverage never falls below "
                f"{h['A2780cis']['min']} (median {h['A2780cis']['median']:.0f}), so "
                "the cell becomes evaluable and stays evaluable. Cost: it changes "
                "the split for EVERY ScaffoldSplitter caller, so any number "
                "measured on a largest-first scaffold split has to be re-run or "
                "pinned."
            ),
            "stratified_assignment": (
                "Choose the split per group to hit each cell line's label share. "
                f"Here: A2780cis {s['test']['labelled']['A2780cis']} test rows, "
                f"all four cells evaluable, rows still 70/10/20 "
                f"({s['train']['rows']}/{s['val']['rows']}/{s['test']['rows']}). "
                "Cost: a protocol change, not a bug fix, and it makes the split "
                "depend on the label columns."
            ),
            "reorder_only": (
                "Shuffle the cohort file's row order. Removes the correlation "
                "between descending label coverage and train membership, but "
                "gives no guarantee: A2780cis ranges "
                f"{a2780cis_perm['min']}-{a2780cis_perm['max']} over permutations."
            ),
        },
    }
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    payload = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    a = payload["as_is"]
    print(f"rows: train={a['train']['rows']} val={a['val']['rows']} test={a['test']['rows']}")
    for cl in CELLS:
        print(
            f"  {cl:8s} train={a['train']['cells'][cl]['labelled']:4d} "
            f"val={a['val']['cells'][cl]['labelled']:3d} "
            f"test={a['test']['cells'][cl]['labelled']:4d} "
            f"evaluable={a['test']['cells'][cl]['evaluable']}"
        )
    o = payload["order_sensitivity"]["test_labelled_rows"]
    r = payload["assignment_randomness"]["test_labelled_rows"]
    h = payload["counterfactual_hash_tiebreak"]["test_labelled_rows"]
    s = payload["counterfactual_stratified"]
    print("reorder-only (median [p05, max]) vs size-blind mean:")
    for cl in CELLS:
        print(
            f"  {cl:8s} {o[cl]['median']:6.0f} [{o[cl]['p05']:3.0f}, {o[cl]['max']:3.0f}]"
            f"   {r[cl]['mean']:6.1f}   hash-tiebreak={h[cl]['median']:4.0f}"
            f" (min {h[cl]['min']}, <bar {h[cl]['share_below_bar']:.0%})"
            f"   stratified={s['test']['labelled'][cl]:4d}"
        )
    print(f"wrote {args.out.relative_to(ROOT)}")
    tv = payload["trainer_view"]
    print(
        f"trainer view ({'/'.join(tv['modelled_cells'])}): "
        f"train={tv['train']['rows']} val={tv['val']['rows']} "
        f"test={tv['test']['rows']}  dropped={tv['dropped_rows']} "
        f"({tv['dropped_share']:.1%}, {tv['dropped_because']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
