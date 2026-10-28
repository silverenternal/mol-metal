#!/usr/bin/env python
"""Wf-A113 (T76) — do the trainer and the bench score against the same truth?

Both tables report an ``exact_type``-style number, and the trainer's docstring
says the two are "directly comparable".  They are not, because the two truth
mappers disagree on the aromatic band:

* corpus / trainer — :func:`molmetal.data.tmqm_clouds.bo_to_bond_class` uses
  BANDS ``<1.25 single | 1.25-1.75 AROMATIC | 1.75-2.50 double | >=2.50 triple``
  (5 classes counting NO_BOND).
* bench — :func:`molmetal.scripts.bench_tmqm_bond_perception.bo_to_type` takes
  the NEAREST of {single 1.0, double 2.0, triple 3.0}; there is no aromatic
  centre, so every 1.25-1.75 bond is labelled ``single``.

Consequence: a bond at BO 1.45 is AROMATIC in the trainer's truth and
``single`` in the bench's.  A head that calls it aromatic is correct for the
trainer and WRONG for the bench; a head that calls it single is wrong for the
trainer and correct for the bench.  The two ``exact_type`` numbers therefore
cannot be placed side by side.

This probe measures how large that band is on the real corpus, so the decision
(unify the truth, or keep two truths and stop claiming comparability) rests on
a number rather than on an argument.

Writes ``metrics/by_metric/wfa113_t76_truth_label_divergence.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t76_truth_label_divergence_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t76_truth_label_divergence.json"

BO_PATHS = [
    ROOT / ".data/molmetal/tmQM/tmQM_X1.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X2.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X3.BO.gz",
]

#: Mirrors ``bench_tmqm_bond_perception._BO_TYPE_CENTRES``.
_BENCH_CENTRES = (("single", 1.0), ("double", 2.0), ("triple", 3.0))


def bench_type(bo: float) -> str:
    return min(_BENCH_CENTRES, key=lambda kv: abs(kv[1] - float(bo)))[0]


def corpus_type(bo: float) -> str:
    from molmetal.data.tmqm_clouds import bo_to_bond_class

    return {1: "single", 2: "double", 3: "triple", 4: "aromatic"}[
        bo_to_bond_class(bo)
    ]


def _iter_blocks(paths: List[Path], max_complexes: int):
    """Yield ``(code, {pair: max_bo})`` from the .BO shards, deduped per pair."""
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks

    n = 0
    for code, atoms in iter_bo_blocks([p for p in paths if p.is_file()]):
        bonds: Dict[Tuple[int, int], float] = {}
        for idx, _sym, nbrs in atoms:
            for nbr_idx, bo in nbrs:
                if nbr_idx == idx:
                    continue
                key = (min(idx, nbr_idx), max(idx, nbr_idx))
                if key not in bonds or bo > bonds[key]:
                    bonds[key] = bo
        yield code, bonds
        n += 1
        if max_complexes and n >= max_complexes:
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-complexes",
        type=int,
        default=4000,
        help="per shard; 0 = all",
    )
    args = ap.parse_args(argv)

    per_shard = []
    total_pairs = 0
    band_counts: Counter = Counter()
    disagreements: Counter = Counter()
    examples = []
    for path in BO_PATHS:
        if not path.is_file():
            continue
        shard_pairs = 0
        shard_disagree = 0
        for code, bonds in _iter_blocks([path], args.max_complexes):
            for key, bo in bonds.items():
                if bo < 0.3:
                    continue
                c = corpus_type(bo)
                b = bench_type(bo)
                band_counts[c] += 1
                shard_pairs += 1
                if c != b:
                    shard_disagree += 1
                    disagreements[(c, b)] += 1
                    if len(examples) < 8:
                        examples.append(
                            {"csd": code, "pair": list(key), "bo": round(bo, 3),
                             "corpus": c, "bench": b}
                        )
        per_shard.append(
            {
                "shard": path.name,
                "n_pairs_ge_0.3": shard_pairs,
                "n_disagree": shard_disagree,
                "disagree_frac": (shard_disagree / shard_pairs) if shard_pairs else 0.0,
            }
        )
        total_pairs += shard_pairs

    n_disagree = sum(v for v in disagreements.values())
    payload = {
        "wave": "Wf-A113",
        "task": "T76",
        "date": "2026-10-28",
        "question": (
            "is the trainer's bond_exact_type_rate computed against the same "
            "truth as the bench's exact_type?"
        ),
        "definitions": {
            "corpus": "tmqm_clouds.bo_to_bond_class — <1.25 single | 1.25-1.75 aromatic | 1.75-2.50 double | >=2.50 triple",
            "bench": "bench_tmqm_bond_perception.bo_to_type — nearest of {1.0 single, 2.0 double, 3.0 triple}",
        },
        "measured": {
            "max_complexes_per_shard": args.max_complexes,
            "n_pairs": total_pairs,
            "n_label_disagreements": n_disagree,
            "disagreement_frac": (n_disagree / total_pairs) if total_pairs else 0.0,
            "corpus_label_counts": dict(band_counts),
            "disagreement_pairs": {
                f"{c}->{b}": n for (c, b), n in disagreements.most_common()
            },
            "per_shard": per_shard,
            "examples": examples,
        },
        "verdict": (
            "the two exact_type numbers are computed against different truth "
            "label sets on every bond in the aromatic band, so they are not "
            "comparable as printed"
        ),
        "script": "molmetal/scripts/wfa113_t76_truth_label_divergence_probe.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["measured"], indent=2)[:1600])
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
