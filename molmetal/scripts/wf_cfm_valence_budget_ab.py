# SPDX-License-Identifier: Apache-2.0
"""Do the valence-budget variants trade sanitisation for connectivity?

WHY THIS EXISTS
---------------
`wf_cfm_decode_fragmentation_probe.py` MEASURED, on 64 tmQM val clouds
against a real checkpoint, that the graph is connected before selection
(stage 2: 1.016 components, single-fragment 0.984) and 2.3-fragmented
after it (stage 3: 2.312, single-fragment 0.172). 224 of 2763 candidate
bonds (8.1%) are rejected as overvalent, and they are the ring-closure
and metal-ligand edges.

Two arithmetic causes were identified:

  * ``_CLASS_VALENCE_COST[BOND_AROMATIC] = 1.5``. A benzene carbon with two
    aromatic neighbours has 1.0 left against a cap of 4, and the third ring
    bond costs 1.5 > 1.0, so it is dropped and the ring opens.
  * metal caps (Ru/Ir/Rh/Os/Re) of 6 are exceeded by 6-coordinate complexes.

An earlier measurement showed metal caps 6->8 lifts single-fragment
0.172 -> 0.422. It ALSO showed `decode_ratio` FALLING 0.203 -> 0.188,
i.e. relaxing the cap costs sanitisation. That is the whole reason the
metal change was reverted.

THE GAP THIS SCRIPT FILLS: that decode_ratio sweep covered only the three
GLOBAL variants. The aromatic-only variant had a connectivity number and
no sanitisation number, so nobody knows whether it is a free win or the
same trade in a new place. A connectivity gain that quietly costs
sanitisation is not a fix.

MEASURED BOTH, per variant, on the same clouds and the same checkpoint:
  single_fragment_ratio  - connectivity, after the bridge repair
  decode_ratio           - fraction RDKit accepts, i.e. sanitisation

Read the two columns together. A variant that lifts the first and drops
the second is NOT a win and must be reported as such.

Run with the venv python. Replays decode only; trains nothing.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _components(n_atoms: int, pairs: List[Tuple[int, int]]) -> int:
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


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "cfm_valence_budget_ab",
        REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_METALS6 = {78: 4, 44: 6, 30: 4, 77: 6, 29: 4, 79: 4, 46: 4, 26: 4, 28: 4, 45: 6, 75: 6, 76: 6}
_METALS8 = {78: 4, 44: 8, 30: 4, 77: 8, 29: 4, 79: 4, 46: 4, 26: 4, 28: 4, 45: 8, 75: 8, 76: 8}

#: The variants under test. Each is a full replacement for the two tables that
#: carry the budget arithmetic, so a variant is fully described by this dict.
VARIANTS: Dict[str, Dict[str, object]] = {
    # What ships today.
    "as_shipped": {
        "aromatic_cost": 1.5,
        "metal_caps": _METALS6,
    },
    # Charge an aromatic bond as a single bond. A Kekule benzene ring is
    # 3 single + 3 double over 6 carbons, but the budget is PER ATOM: a ring
    # carbon carrying two aromatic bonds and one more is charged 1.5*3 = 4.5
    # against a cap of 4 purely from the 1.5, which is an over-count.
    "aromatic_1.0": {
        "aromatic_cost": 1.0,
        "metal_caps": _METALS6,
    },
    # Metal caps raised. Already known to cost sanitisation; measured here
    # again so the two columns are comparable across all four variants.
    "metals_8": {
        "aromatic_cost": 1.5,
        "metal_caps": _METALS8,
    },
    "both": {
        "aromatic_cost": 1.0,
        "metal_caps": _METALS8,
    },
    # Reserve budget for the bridge so a repair is always affordable. This is
    # NOT a cap relaxation: the selector is forbidden from spending the last
    # `reserve` units, so the total is unchanged and sanitisation should not
    # move. It is the one remaining candidate that could raise connectivity
    # without trading decode_ratio away.
    "reserve_0.5": {"aromatic_cost": 1.5, "reserve": 0.5, "metal_caps": _METALS6},
    "reserve_1.0": {"aromatic_cost": 1.5, "reserve": 1.0, "metal_caps": _METALS6},
    "reserve_2.0": {"aromatic_cost": 1.5, "reserve": 2.0, "metal_caps": _METALS6},
    # The degenerate direction, measured so it can be named as gaming rather
    # than merely asserted against.
    "no_budget": None,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cfm_objective_terms_2026_10_11")
    ap.add_argument("--n-clouds", type=int, default=64)
    ap.add_argument("--out-json", default=None)
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

    device = torch.device("cpu")
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    model = T.CFMJointModel(
        hidden_dim=int(a["hidden_dim"]),
        n_layers=int(a["n_layers"]),
        cutoff=float(a.get("cutoff", 3.5)),
        use_egnn_velocity=bool(a.get("use_egnn_velocity", True)),
        fm_objective=bool(a.get("use_fm_objective", False)),
        max_atomic_number=int(a.get("max_atomic_number", 100)),
        context_dropout=float(a.get("context_dropout", 0.1)),
        z_table=tuple(get_z_table(str(a.get("yuelbond_vocab", "pt")))),
        bond_predictor=str(a.get("bond_predictor", "yuelbond")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from molmetal.molmetal_lam.lam_chem import cfm_decoder_yuelbond as C

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

    # Featurise ONCE so every variant sees identical logits.
    print("[ab] featurising clouds ...", flush=True)
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondDecoder

    head = YuelBondDecoder(
        head=model.head, hidden_dim=int(a["hidden_dim"]), n_layers=int(a["n_layers"])
    )
    feats = []
    with torch.no_grad():
        for c in clouds:
            cloud = T._to_device_cloud(c, device)
            feats.append((cloud, head.featurise(cloud)))

    saved_cost = copy.deepcopy(C._CLASS_VALENCE_COST)
    saved_val = copy.deepcopy(C._COVALENT_VALENCE)

    results = {}
    for name, spec in VARIANTS.items():
        C._CLASS_VALENCE_COST.clear()
        C._CLASS_VALENCE_COST.update(saved_cost)
        C._COVALENT_VALENCE.clear()
        C._COVALENT_VALENCE.update(saved_val)
        if spec is not None:
            C._CLASS_VALENCE_COST[4] = float(spec["aromatic_cost"])
            for z, cap in dict(spec["metal_caps"]).items():
                C._COVALENT_VALENCE[int(z)] = int(cap)
        # One decoder for the variant. The tables above are module globals, so
        # the decoder picks them up at construction; there is no per-variant
        # constructor knob and the shipped decoder source stays untouched.
        #
        # The reserve acts by shrinking the caps the SELECTOR reads, because
        # that is the only place budget can be withheld from selection while
        # still leaving the bridge its full cap.
        _reserve = float((spec or {}).get("reserve", 0.0) or 0.0)
        if _reserve:
            for _z, _c in list(C._COVALENT_VALENCE.items()):
                C._COVALENT_VALENCE[_z] = max(0, int(_c) - int(round(_reserve)))
        dec = C.YuelBondBondDecoder(
            head=model.head,
            hidden_dim=int(a["hidden_dim"]),
            n_layers=int(a["n_layers"]),
            decision=str(a.get("bond_decision", "threshold")),
            decision_threshold=float(a.get("bond_decision_threshold", 0.5)),
            connectivity_prior=str(a.get("connectivity_prior", "on")) == "on",
            connectivity_prior_min_logit=float(a.get("connectivity_prior_min_logit", 0.0)),
        )
        if spec is None:
            # Degenerate control: every bond is free.
            C._CLASS_VALENCE_COST.update({k: 0.0 for k in saved_cost})
            dec = C.YuelBondBondDecoder(
                head=model.head,
                hidden_dim=int(a["hidden_dim"]),
                n_layers=int(a["n_layers"]),
                decision=str(a.get("bond_decision", "threshold")),
                decision_threshold=float(a.get("bond_decision_threshold", 0.5)),
                connectivity_prior=str(a.get("connectivity_prior", "on")) == "on",
                connectivity_prior_min_logit=float(a.get("connectivity_prior_min_logit", 0.0)),
            )

        single, decoded_ok, n = 0, 0, 0
        with torch.no_grad():
            for cloud, yres in feats:
                if yres.logits is None or yres.logits.shape[0] == 0:
                    continue
                sel, dative, _k, _o = dec._select_bonds(cloud, yres)
                sel2, _d2, _added = dec._bridge_components(
                    cloud, yres, list(sel), list(dative)
                )
                res = dec._assemble_mol(cloud, sel2, [])
                n += 1
                if getattr(res, "sanitized", False) and getattr(res, "smiles", ""):
                    decoded_ok += 1
                    from rdkit import Chem

                    m = Chem.MolFromSmiles(res.smiles)
                    if m is not None and len(Chem.GetMolFrags(m)) == 1:
                        single += 1
        results[name] = {
            "n": n,
            "single_fragment_ratio": round(single / n, 4) if n else None,
            "decode_ratio": round(decoded_ok / n, 4) if n else None,
        }
        print(
            f"  {name:14} single_frag={results[name]['single_fragment_ratio']}"
            f"  decode_ratio={results[name]['decode_ratio']}  (n={n})",
            flush=True,
        )

    C._CLASS_VALENCE_COST.clear()
    C._CLASS_VALENCE_COST.update(saved_cost)
    C._COVALENT_VALENCE.clear()
    C._COVALENT_VALENCE.update(saved_val)

    print()
    print("  READ THE TWO COLUMNS TOGETHER. A variant that lifts")
    print("  single_fragment_ratio while dropping decode_ratio is NOT a fix:")
    print("  it trades sanitisation for connectivity, which is the same")
    print("  trade that got the metal-cap change reverted.")
    if args_cli.out_json:
        import json

        out = {
            "metric": "cfm_valence_budget_variant_ab",
            "tag": "MEASURED",
            "run": args_cli.run,
            "checkpoint": str(ckpt_path.relative_to(REPO)),
            "checkpoint_epoch": ckpt.get("epoch"),
            "n_clouds": args_cli.n_clouds,
            "split": "val (real tmQM clouds, NOT ODE-from-noise)",
            "variants": results,
            "note": (
                "'no_budget' is the degenerate control, not a proposal. "
                "Metric-gaming direction: raising single_fragment_ratio by "
                "dropping or ignoring valence enforcement without reporting "
                "the decode_ratio cost."
            ),
        }
        Path(args_cli.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args_cli.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())