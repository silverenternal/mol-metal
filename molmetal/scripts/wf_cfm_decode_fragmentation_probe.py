# SPDX-License-Identifier: Apache-2.0
"""Which decode-time filter turns a connected soft graph into fragments?

WHY THIS EXISTS
---------------
Three recorded CFM runs collapse to zero single-fragment molecules around
epoch 5 and do not recover (cfm_fm_ab 0.53 -> 0.00 at ep5, cfm_fm_control
0.57 -> 0.00, cfm_blk_c5_repair 0.39 -> 0.00 by ep3). The decoded "molecules"
are bags of ~10 disconnected atoms, which is what drags m5 QED to 0.198
against a 0.50 target and m4 SA to 7.37.

The obvious suspect — the assemble/connectivity term — measures the wrong
thing. It scores the head's SOFT graph, and the recorded r9 log shows that
graph is essentially single-fragment the whole way (`assemble_fragments`
1.943 at ep0, ~1.009 thereafter, `assemble_largest_fragment_frac` 0.9957).
So the loss term sees a connected molecule while the decoder emits ten
pieces, and no threshold on that term can close the gap. Before rewiring
anything, the transition has to be located.

This replays the REAL decode on a REAL checkpoint and measures the component
count at each stage, so the stage that loses the bonds is named rather than
guessed.

STAGES MEASURED
---------------
  1. head          — the candidate pair list the head emits (cutoff-limited)
  2. soft          — components of the graph where P(bond) >= 0.5, i.e. what
                     the assemble term approximates
  3. selected      — after `_select_bonds`: the hard decision rule AND the
                     per-atom valence budget, which can reject a bond the
                     head asked for
  4. bridged       — after `_bridge_components`, the connectivity prior

A drop between two consecutive stages is that filter's doing.

SCOPE
-----
Replays decode only. It does not train, does not write a metric, and does not
touch any report. It reads a checkpoint and the tmQM val split.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "cfm_decode_frag_probe",
        REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _components(n_atoms: int, pairs: List[tuple]) -> int:
    """Union-find over heavy atoms; returns the number of components."""
    parent = list(range(n_atoms))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    return len({find(i) for i in range(n_atoms)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cfm_fm_ab_2026_10_09")
    ap.add_argument("--n-clouds", type=int, default=192)
    ap.add_argument("--bond-threshold", type=float, default=0.5)
    args_cli = ap.parse_args()

    import yaml

    T = load_trainer()
    run_dir = REPO / "molmetal" / "reports" / args_cli.run
    ckpt_path = run_dir / "checkpoints" / "cfm_yuelbond_wasserstein_seed0.pt"
    if not ckpt_path.is_file():
        print(f"no checkpoint at {ckpt_path}")
        return 1
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = T.CFMJointModel(
        hidden_dim=int(a["hidden_dim"]),
        n_layers=int(a["n_layers"]),
        use_egnn_velocity=bool(a.get("use_egnn_velocity", True)),
        fm_objective=bool(a.get("use_fm_objective", False)),
        bond_predictor=str(a.get("bond_predictor", "yuelbond")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
        YuelBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import bond_decision as _bond_decision

    dec = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=int(a["hidden_dim"]),
        n_layers=int(a["n_layers"]),
        decision=str(a.get("bond_decision", "threshold")),
        decision_threshold=float(a.get("bond_decision_threshold", 0.5)),
        connectivity_prior=str(a.get("connectivity_prior", "on")) == "on",
        connectivity_prior_min_logit=float(a.get("connectivity_prior_min_logit", 0.0)),
    )

    cfg = yaml.safe_load((REPO / "configs" / "cfm_tmqm_all6.yaml").read_text())
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        cfg["data"]["tmqm_dir"],
        metals=[s.strip() for s in cfg["data"]["tmqm_metals"].split(",")],
        coord_number=None,
        n_clouds=args_cli.n_clouds,
        seed=int(a.get("seed", 0)),
        split="val",
        val_frac=float(cfg["data"]["tmqm_val_frac"]),
    )

    counts: Dict[str, List[int]] = defaultdict(list)
    dropped_by_valence = 0
    dropped_total = 0
    with torch.no_grad():
        for c in clouds:
            cloud = T._to_device_cloud(c, device)
            yres = YuelBondDecoder(
                head=model.head, hidden_dim=int(a["hidden_dim"]),
                n_layers=int(a["n_layers"]),
            ).featurise(cloud)
            if yres.logits is None or yres.logits.shape[0] == 0:
                continue
            n = int(cloud.atomic_numbers.shape[0])
            ei = yres.edge_index
            src = ei[0].tolist() if not isinstance(ei, tuple) else ei[0].tolist()
            dst = ei[1].tolist() if not isinstance(ei, tuple) else ei[1].tolist()
            counts["1_head_candidates"].append(_components(n, list(zip(src, dst))))

            p_bond = 1.0 - torch.softmax(yres.logits.float(), dim=-1)[:, 0]
            soft = [
                (i, j) for i, j, k in zip(src, dst, p_bond.tolist())
                if k >= args_cli.bond_threshold
            ]
            counts["2_soft_p>=thr"].append(_components(n, soft))

            pred = _bond_decision(
                yres.logits, dec.decision, dec.decision_threshold
            )
            pt = pred.tolist() if hasattr(pred, "tolist") else list(pred)
            cand = [(i, j, o) for i, j, o in zip(src, dst, pt) if o != 0]
            dropped_total += len(cand)
            sel, dative, n_kept, n_over = dec._select_bonds(cloud, yres)
            dropped_by_valence += int(n_over)
            counts["3_after_select"].append(
                _components(n, [(int(i), int(j)) for i, j, _ in sel])
            )
            sel2, dat2, n_added = dec._bridge_components(cloud, yres, list(sel), list(dative))
            counts["4_after_bridge"].append(
                _components(n, [(int(i), int(j)) for i, j, _ in sel2])
            )

    print()
    print(f"  run {args_cli.run}  (checkpoint epoch {ckpt.get('epoch')}), "
          f"{len(counts['1_head_candidates'])} val clouds, device={device}")
    print()
    print(f"  {'stage':<24}{'mean comps':>12}{'single-frac':>13}")
    print("  " + "-" * 49)
    for stage in ("1_head_candidates", "2_soft_p>=thr", "3_after_select", "4_after_bridge"):
        v = counts[stage]
        if not v:
            continue
        print(f"  {stage:<24}{sum(v)/len(v):>12.3f}{sum(1 for x in v if x == 1)/len(v):>13.3f}")
    print()
    print(f"  candidate pairs offered to the selector : {dropped_total}")
    print(f"  rejected as OVERVALENT by the budget    : {dropped_by_valence}"
          f"  ({dropped_by_valence/max(dropped_total,1):.3f})")
    print()
    print("  A single-fragment fraction that FALLS at one transition names that")
    print("  filter. If stage 2 is already ~1.0 and stage 3 is ~0.0, the hard")
    print("  decision rule plus the valence budget is what fragments the graph,")
    print("  and the assemble term cannot see it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())