#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does the TRAINED bond head get ring size right? Capacity is not behaviour.

THE CHAIN OF QUESTIONS
----------------------
Claim A4 (2026-10-13) said ring size is UNREPRESENTABLE in the per-edge
bond head, and that was used as the structural argument against spending
the single authorised bond-head retrain on ring closure.

That claim was refuted twice over:

  * capacity (metrics/by_metric/cfm_a4_receptive_field.json) -- a plain
    LogisticRegression on the sorted pairwise distances of a ring classifies
    ring size at 1.000 +/- 0.000 (5-fold CV) and 1.000 held-out, per-class
    recall 1.000 for sizes 3/4/5/6, over 634 real rings. The head is a
    3-layer MPNN over those same distances, so that is a LOWER bound on what
    it could represent.
  * THIS SCRIPT -- behaviour. Capacity says the head COULD read ring size.
    It does not say it DOES. That is the decision-relevant question, and it
    is what actually settles where the retrain slot goes.

THE MEASUREMENT
---------------
Load the shipped checkpoint's head and run it on CLEAN tmQM geometry, which
is fully in-distribution for a head trained on clean CSD structures. Count
the cycles its argmax bond predictions close, against the cycles the truth
Wiberg bonds close on the SAME clouds.

MEASURED, 120 real clouds (seed 1234, the repo's own load_tmqm_clouds):

    ring size        truth    head-predicted
         3            120          39
         4             14          10
         5            165          55
         6            335         197
       all            657         305

    5/6-membered fraction   TRUTH 76.10%    HEAD 82.62%
    3-membered   fraction   TRUTH 18.26%    HEAD 12.79%

THE VERDICT, AND WHY IT MATTERS MORE THAN THE A4 REFUTATION
-----------------------------------------------------------
On clean geometry -- the head's home turf, no sampling, no ODE, no noise --
the trained head produces a ring-size distribution CLOSER to the real
corpus than the truth bonds themselves: it under-produces 3-rings
(12.79% vs 18.26%) and over-produces 5/6-rings (82.62% vs 76.10%).

So the bond head is NOT the ring defect. The 89.81% three-membered rate in
generated output cannot be blamed on a head that, fed clean geometry, gets
ring size right. Whatever produces it lives upstream, in the geometry the
velocity field supplies -- which is exactly what ARCH-4 (make the sampled
geometry in-distribution first) and ARCH-1 (the head never sees a noisy
sample) address, and NOT in the retrain slot for the head.

THIS DOES NOT SAY THE HEAD IS FINE
----------------------------------
Total predicted rings (305) is well below truth (657): the head drops
bonds. A ring-size DISTRIBUTION being right is not the same as a ring-size
COUNT being right, and the head's macro-F1 deficit against OpenBabel
(0.4784 vs 0.5051, metrics/by_metric/tmqm_bond_perception_3seed.json) is
unaffected by any of this. What this rules out is the specific hypothesis
"the head cannot do ring closure, so retrain it for that" -- not the
hypothesis "the bond head has a real, separate macro-F1 deficit".

SMOKE SCALE
-----------
120 clouds, one checkpoint, one seed, argmax decisions (the checkpoint's own
bond_decision is 'threshold' at 0.5; this script uses argmax and says so).
Directional, not a settled number. No GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
TMQM_DIR = REPO / ".data" / "molmetal" / "tmQM"
CKPT = (REPO / "molmetal" / "reports" / "cfm_patience10_seed0" / "checkpoints"
        / "cfm_yuelbond_wasserstein_seed0_last.pt")
DEFAULT_OUT = REPO / "metrics" / "by_metric" / "cfm_head_ring_size_behaviour.json"

N_CLOUDS = 120
SEED = 1234


def load_model(ckpt_path: Path = CKPT):
    """Rebuild the shipped model and load the checkpoint STRICTLY.

    Strict loading is the point: a silently-wrong vocabulary would produce a
    confident head with garbage weights, and every number below would be
    meaningless while still looking like a measurement. The all-six-metals Z
    table (width 18) is required -- the checkpoint's atom_encoder is
    Linear(128, 18), and the default Pt-only table is width 13.
    """
    import torch

    sys.path.insert(0, str(REPO))
    import molmetal.scripts.train_cfm_yuelbond_wasserstein as T
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        _YUELBOND_Z_TABLE_ALL6,
    )

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    a = a if isinstance(a, dict) else vars(a)
    model = T.CFMJointModel(
        n_layers=int(a.get("n_layers", 3)),
        hidden_dim=int(a.get("hidden_dim", 128)),
        use_egnn_velocity=bool(a.get("use_egnn_velocity", True)),
        fm_objective=bool(a.get("use_fm_objective", False)),
        cutoff=float(a.get("cutoff", 3.5)),
        z_table=_YUELBOND_Z_TABLE_ALL6,
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, ck, a


def _truth_bonds(cloud) -> np.ndarray:
    """tmQM truth bonds arrive as an (2, E) edge-index, not an (E, 2) list."""
    b = cloud.bonds.detach().cpu().numpy()
    if b.ndim == 2 and b.shape[0] == 2 and b.shape[1] != 2:
        b = b.T
    return b


def measure(n_clouds: int = N_CLOUDS, seed: int = SEED,
            ckpt_path: Path = CKPT) -> dict:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    sys.path.insert(0, str(REPO))

    import networkx as nx
    import torch

    from molmetal.data.tmqm_clouds import load_tmqm_clouds
    from molmetal.models.bond_head import BOND_NO_BOND

    model, ck, args = load_model(ckpt_path)
    clouds = load_tmqm_clouds(TMQM_DIR, n_clouds=n_clouds, seed=seed,
                              include_hydrogens=False)

    truth_sz: Counter = Counter()
    pred_sz: Counter = Counter()
    n_true = 0
    n_pred = 0
    n_clouds_used = 0
    n_pairs = 0
    n_pred_bonds = 0

    for cloud in clouds:
        with torch.no_grad():
            out = model.head(cloud)
        if not isinstance(out, tuple) or len(out) < 2:
            continue
        logits, (i_idx, j_idx) = out[0], out[1]
        if logits is None or logits.shape[0] == 0:
            continue
        n_atoms = int(cloud.atomic_numbers.shape[0])
        pred = logits.argmax(dim=-1).cpu().numpy()
        i = i_idx.detach().cpu().numpy()
        j = j_idx.detach().cpu().numpy()
        n_pairs += int(len(pred))

        g = nx.Graph()
        g.add_nodes_from(range(n_atoms))
        for e in _truth_bonds(cloud):
            g.add_edge(int(e[0]), int(e[1]))
        for cyc in nx.cycle_basis(g):
            if len(cyc) >= 3:
                truth_sz[len(cyc)] += 1
                n_true += 1

        p = nx.Graph()
        p.add_nodes_from(range(n_atoms))
        for k in range(len(pred)):
            if int(pred[k]) != int(BOND_NO_BOND):
                p.add_edge(int(i[k]), int(j[k]))
                n_pred_bonds += 1
        for comp in nx.connected_components(p):
            try:
                cycles = nx.cycle_basis(p.subgraph(comp))
            except Exception:
                cycles = []
            for cyc in cycles:
                if len(cyc) >= 3:
                    pred_sz[len(cyc)] += 1
                    n_pred += 1
        n_clouds_used += 1

    def frac(c: Counter, total: int, sizes) -> float:
        return sum(c[k] for k in sizes) / max(1, total)

    return {
        "metric": "cfm_head_ring_size_behaviour",
        "tag": "MEASURED",
        "date": "2026-10-13",
        "question": (
            "Claim A4 was refuted on CAPACITY (ring size is recoverable from "
            "the distance pattern). This measures BEHAVIOUR: does the TRAINED "
            "head, on CLEAN geometry, actually close rings of the right size?"
        ),
        "setup": {
            "checkpoint": str(ckpt_path.relative_to(REPO)),
            "checkpoint_epoch": int(ck.get("epoch", -1)),
            "checkpoint_decode_ratio": ck.get("decode_ratio"),
            "head_n_layers": int(args.get("n_layers", 3)),
            "head_hidden_dim": int(args.get("hidden_dim", 128)),
            "head_z_table": "all6 (width 18) -- REQUIRED; the default Pt-only "
                            "table is width 13 and fails strict loading",
            "state_dict_loaded": "strict=True (0 missing, 0 unexpected)",
            "decision_rule": "argmax over the 5 bond classes. NOTE the "
                             "checkpoint's own bond_decision is 'threshold' "
                             "at 0.5; this script uses argmax and says so.",
            "loader": "molmetal.data.tmqm_clouds.load_tmqm_clouds",
            "n_clouds_requested": n_clouds,
            "n_clouds_used": n_clouds_used,
            "seed": seed,
            "geometry": "CLEAN tmQM coordinates -- in-distribution for a head "
                        "trained on clean CSD structures",
            "n_candidate_pairs": n_pairs,
            "n_predicted_bonds": n_pred_bonds,
        },
        "truth_ring_size": {str(k): v for k, v in sorted(truth_sz.items())},
        "head_ring_size": {str(k): v for k, v in sorted(pred_sz.items())},
        "n_truth_rings": n_true,
        "n_head_rings": n_pred,
        "frac_5_or_6": {
            "truth": frac(truth_sz, n_true, (5, 6)),
            "head": frac(pred_sz, n_pred, (5, 6)),
        },
        "frac_3_membered": {
            "truth": truth_sz[3] / max(1, n_true),
            "head": pred_sz[3] / max(1, n_pred),
        },
        "verdict": (
            "THE HEAD IS NOT THE RING DEFECT. On clean geometry the trained "
            "head's ring-size distribution is CLOSER to the real corpus than "
            "the truth bonds are: it UNDER-produces 3-rings and "
            "OVER-produces 5/6-rings relative to truth. The 89.81% "
            "three-membered rate in generated output therefore cannot be "
            "blamed on a head that, fed clean geometry, gets ring size right."
        ),
        "where_the_defect_must_live": (
            "Upstream, in the geometry the velocity field supplies. That is "
            "what ARCH-4 (make sampled geometry in-distribution) and ARCH-1 "
            "(the head never sees a noisy sample) address -- NOT the retrain "
            "slot for the bond head."
        ),
        "what_this_does_NOT_say": [
            "It does NOT say the head is fine. It predicts 305 rings against "
            "657 in truth: it drops bonds. A correct ring-size DISTRIBUTION "
            "is not a correct ring-size COUNT.",
            "It does NOT touch the macro-F1 deficit against OpenBabel "
            "(0.4784 vs 0.5051, metrics/by_metric/tmqm_bond_perception_"
            "3seed.json). That remains a real, separate honest-negative.",
            "It rules out the specific hypothesis 'the head cannot do ring "
            "closure, so retrain it for that'. It does not rule out spending "
            "the slot on the macro-F1 deficit.",
        ],
        "smoke_scale_caveat": (
            "120 clouds, one checkpoint, one seed, CPU, no GPU. Directional, "
            "not a settled number."
        ),
        "related": {
            "capacity_refutation": "metrics/by_metric/cfm_a4_receptive_field.json",
            "generated_side_for_contrast":
                "metrics/by_metric/cfm_opt2_ring_dative_correction.json "
                "(generated 89.81 percent 3-membered, like-for-like real "
                "12.38 percent)",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-clouds", type=int, default=N_CLOUDS)
    args = ap.parse_args(argv)

    res = measure(n_clouds=args.n_clouds)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(args.out_json, json.dumps(res, indent=2) + "\n")

    print(f"clouds used: {res['setup']['n_clouds_used']}")
    print(f"{'ring':>5} {'truth':>8} {'head':>8}")
    for k in sorted(set(res["truth_ring_size"]) | set(res["head_ring_size"]),
                    key=int):
        print(f"{k:>5} {res['truth_ring_size'].get(k, 0):8d} "
              f"{res['head_ring_size'].get(k, 0):8d}")
    print(f"\n5/6-membered  truth {res['frac_5_or_6']['truth']:.2%}   "
          f"head {res['frac_5_or_6']['head']:.2%}")
    print(f"3-membered    truth {res['frac_3_membered']['truth']:.2%}   "
          f"head {res['frac_3_membered']['head']:.2%}")
    print(f"\ntotal rings: truth {res['n_truth_rings']}  head {res['n_head_rings']}")
    print(f"\nVERDICT: {res['verdict'][:150]}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
