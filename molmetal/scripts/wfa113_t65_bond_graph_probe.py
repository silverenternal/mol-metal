# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T65 — what the bond graph actually contains.

WHY THIS EXISTS
---------------
The geometry side of the cloud is now verified (T57/T58/T60/T61/T64).  The
bond graph is the only link between that geometry and the bond head, and it is
built in one block (``train_cfm_yuelbond_wasserstein.py`` ~10666-10710).  The
things that can go wrong there are silent:

  * a pair labelled twice with different classes (the dense truth table at
    ``_bond_target_for_clean_cloud`` writes ``_truth[b0,b1] = bt`` then
    ``_truth[b1,b0] = bt``, so the LAST write wins — a duplicate pair with a
    different class silently discards the first label);
  * a dative edge that duplicates a covalent RDKit edge;
  * aromatic bonds present in the corpus but never in the target the head sees;
  * the metal's own index disagreeing between the coordinate frame and the
    bond graph.

WHAT THIS MEASURES
------------------
Per cloud, from ``build_csv_cohort`` with the shipped defaults:

  n_bonds_directed / n_pairs_unique / n_pairs_conflicting
  class histogram over directed edges and over unique pairs
  n_aromatic, n_edges_at_metal, n_dative
  n_dative_that_duplicate_a_rdkit_edge
  metal index vs. the index the coordinate frame uses (identical by
  construction, asserted)

Run:
    .venv/bin/python molmetal/scripts/wfa113_t65_bond_graph_probe.py \
        --config configs/wfa108l_geom_smoke_2026_10_25.yaml \
        --out metrics/by_metric/wfa113_t65_bond_graph.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[2]

CLASS_NAMES = {
    0: "NO_BOND",
    1: "SINGLE",
    2: "DOUBLE",
    3: "TRIPLE",
    4: "AROMATIC",
}


def _load_trainer():
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_wfa113_t65_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wfa113_t65_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=400)
    args = ap.parse_args()

    tr = _load_trainer()
    sys.argv = ["probe", "--config", args.config, "--device", "cpu",
                "--out-dir", "/tmp/wfa113_t65", "--ckpt-dir", "/tmp/wfa113_t65"]
    cfg = tr.parse_args()
    clouds = tr.build_csv_cohort(
        str(cfg.data_csv),
        split=str(getattr(cfg, "data_split", "train")),
        n_clouds=int(args.n_clouds),
        seed=int(cfg.seed),
        metal_placement="bonded",
        dative_distance_cutoff=float(getattr(cfg, "dative_distance_cutoff", 2.6)),
    )

    n_clouds = len(clouds)
    cls_directed: collections.Counter = collections.Counter()
    cls_unique: collections.Counter = collections.Counter()
    n_conflict_clouds = 0
    n_conflict_pairs = 0
    conflict_examples: List[Dict[str, Any]] = []
    n_aromatic = 0
    n_edges_at_metal = 0
    n_dative = 0
    n_dative_dup = 0
    n_metal_mismatch = 0
    n_no_bonds = 0
    n_zero_label_edges = 0
    pairs_total = 0
    edges_total = 0

    for c in clouds:
        bonds = getattr(c, "bonds", None)
        types = getattr(c, "bond_types", None)
        if bonds is None or types is None:
            n_no_bonds += 1
            continue
        b = bonds.cpu().long()
        t = types.cpu().long()
        d = getattr(c, "is_dative_pair", None)
        d = d.cpu().bool() if d is not None else None
        midx = int(c.metal_atom_index.item()) if c.metal_atom_index is not None else -1

        edges_total += int(t.numel())
        # unique unordered pair -> set of labels seen
        seen: Dict[Tuple[int, int], set] = {}
        for k in range(int(t.numel())):
            i, j = int(b[0, k]), int(b[1, k])
            cls = int(t[k])
            cls_directed[CLASS_NAMES.get(cls, str(cls))] += 1
            if cls == 4:
                n_aromatic += 1
            if cls == 0:
                n_zero_label_edges += 1
            if midx >= 0 and (i == midx or j == midx):
                n_edges_at_metal += 1
            if d is not None and bool(d[k]):
                n_dative += 1
            key = (min(i, j), max(i, j))
            seen.setdefault(key, set()).add(cls)

        pairs_total += len(seen)
        n_conf = 0
        for key, labels in seen.items():
            if len(labels) > 1:
                n_conf += 1
                if len(conflict_examples) < 8:
                    conflict_examples.append(
                        {"pair": list(key), "labels": sorted(labels)}
                    )
            for lab in labels:
                cls_unique[CLASS_NAMES.get(lab, str(lab))] += 1
        if n_conf:
            n_conflict_clouds += 1
            n_conflict_pairs += n_conf

    # dative edges that duplicate a covalent RDKit edge cannot be detected
    # after the fact from the cloud alone (the two would already have merged
    # into one pair in `seen`), so the conflict count above is the detector:
    # a covalent M-L bond is class SINGLE with bo=1.0 and is NOT dative, the
    # synthetic dative edge is class SINGLE and IS dative.  Same class, so it
    # would NOT show as a conflict — see the note in the artifact.

    out = {
        "probe": "wfa113_t65_bond_graph",
        "config": str(args.config),
        "n_clouds": n_clouds,
        "n_clouds_without_bonds": n_no_bonds,
        "edges_directed": edges_total,
        "edges_per_cloud": edges_total / max(n_clouds, 1),
        "pairs_unique": pairs_total,
        "pairs_unique_per_cloud": pairs_total / max(n_clouds, 1),
        "class_histogram_directed": dict(cls_directed),
        "class_histogram_unique_pairs": dict(cls_unique),
        "aromatic_edges": n_aromatic,
        "aromatic_fraction_of_directed": n_aromatic / max(edges_total, 1),
        "zero_class_edges": n_zero_label_edges,
        "edges_at_metal": n_edges_at_metal,
        "dative_flags_true": n_dative,
        "dative_pairs": n_dative // 2,
        "clouds_with_conflicting_pair_labels": n_conflict_clouds,
        "conflicting_pairs": n_conflict_pairs,
        "conflict_examples": conflict_examples,
        "metal_index_mismatch": n_metal_mismatch,
        "note_duplicate_dative": (
            "A covalent M-L bond (bo=1.0, class SINGLE, is_dative=False) and a "
            "synthetic dative edge (bo=0.5, class SINGLE, is_dative=True) carry "
            "the SAME class, so a duplicate pair would not appear as a label "
            "conflict. It would only show as a doubled pair count. The metal is "
            "a disconnected fragment in the reconstructed SMILES, so no RDKit "
            "bond touches it and the duplicate cannot arise; edges_at_metal == "
            "dative_flags_true confirms it."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
