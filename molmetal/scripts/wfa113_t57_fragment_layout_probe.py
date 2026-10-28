# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T57/T58 — measure the corpus geometry after fragment separation.

WHAT THIS MEASURES
------------------
The CSV loader embeds a multi-component SMILES with RDKit, then hands the
positions straight to the flow-matching objective.  Two defects made those
positions teach the model the wrong thing:

  T57  ``EmbedMolecule`` embeds each disconnected fragment independently and
       translates them all to the origin, so ``[OH2]x6.[Ru]`` puts six
       oxygens on one point.  Fixed by ``molmetal/data/fragment_layout.py``.
  T58  ``_place_metal_at_donor`` pushed the metal out until every HEAVY atom
       was clear, ignoring hydrogens, so the metal landed on a C-H hydrogen.
       Fixed by tilting at the same M-L distance instead of pushing.

Both are measured here in the same pass, with the pre-fix behaviour restored
by monkeypatch, so the artifact carries a paired before/after rather than a
claim.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t57_fragment_layout_probe.py \
        --config configs/wfa108l_geom_smoke_2026_10_25.yaml \
        --out metrics/by_metric/wfa113_t57_fragment_layout.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]


def _load_trainer():
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_wfa113_t57_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wfa113_t57_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _old_place_metal_at_donor(
    pos: torch.Tensor,
    z: torch.Tensor,
    metal_idx: int,
    donor_idx: int,
    bond_length: float = 2.05,
    min_clearance: float = 1.2,
    step: float = 0.1,
    max_steps: int = 40,
) -> Tuple[torch.Tensor, bool]:
    """The pre-T58 body: hydrogens are NOT in the clearance set."""
    p = pos.clone()
    d = p[int(donor_idx)]
    heavy = (z != 1).clone()
    heavy[int(metal_idx)] = False
    heavy[int(donor_idx)] = False
    centroid = p[heavy].mean(dim=0) if int(heavy.sum().item()) > 0 else torch.zeros_like(d)
    u = d - centroid
    if float(u.norm().item()) < 1e-6:
        u = torch.tensor([1.0, 0.0, 0.0], dtype=p.dtype)
    u = u / u.norm()
    cand = d + float(bond_length) * u
    others = p[heavy]
    for _ in range(int(max_steps)):
        if int(others.shape[0]) == 0:
            p[int(metal_idx)] = cand
            return p, True
        if float((others - cand).norm(dim=-1).min().item()) >= float(min_clearance):
            p[int(metal_idx)] = cand
            return p, True
        cand = cand + float(step) * u
    p[int(metal_idx)] = cand
    return p, False


def _bond_set_including_h(smiles: str) -> set:
    """RDKit bond pairs for ``smiles`` with explicit H, as a set of tuples."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return set()
    mol_h = Chem.AddHs(mol)
    return {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol_h.GetBonds()
    }


def _measure(clouds: List[Any]) -> Dict[str, Any]:
    n_coincident_pairs = 0
    n_clouds_coincident = 0
    nonbonded: List[float] = []
    metal_h: List[float] = []
    metal_h_clash = 0
    dative_counts: List[int] = []
    n_metal_clouds = 0

    for c in clouds:
        z = c.atomic_numbers.long()
        x = c.positions.float()
        n = int(x.shape[0])
        dd = torch.cdist(x, x) + torch.eye(n) * 1e9

        k = int((dd < 1e-3).sum()) // 2
        n_coincident_pairs += k
        n_clouds_coincident += int(k > 0)

        bonded = _bond_set_including_h(c.smiles or "")
        mask = torch.zeros((n, n), dtype=torch.bool)
        for i, j in bonded:
            if i < n and j < n:
                mask[i, j] = True
                mask[j, i] = True
        nb = dd.masked_fill(mask, 9.0)
        nonbonded.append(float(nb.min()))

        mi = int(c.metal_atom_index)
        if mi < 0:
            continue
        n_metal_clouds += 1
        d = (x - x[mi]).norm(dim=-1)
        d[mi] = 9.0
        h = z == 1
        if bool(h.any()):
            v = float(d[h].min())
            metal_h.append(v)
            metal_h_clash += int(v < 1.17)  # 0.7 * (r_Pt + r_H) = 1.17 A
        if c.is_dative_pair is not None:
            dative_counts.append(int(c.is_dative_pair.sum()) // 2)

    def _q(v: List[float], q: float) -> float:
        return float(torch.quantile(torch.tensor(v), q)) if v else float("nan")

    nb_t = torch.tensor(nonbonded)
    mh_t = torch.tensor(metal_h)
    dat_t = torch.tensor(dative_counts, dtype=torch.float)
    return {
        "n_clouds": len(clouds),
        "n_metal_clouds": n_metal_clouds,
        "coincident_pairs": n_coincident_pairs,
        "clouds_with_coincident_atoms": n_clouds_coincident,
        "clouds_with_coincident_fraction": n_clouds_coincident / max(len(clouds), 1),
        "nonbonded_min_median_A": float(nb_t.median()),
        "nonbonded_min_p10_A": _q(nonbonded, 0.10),
        "nonbonded_min_A": float(nb_t.min()),
        "nonbonded_below_1A": int((nb_t < 1.0).sum()),
        "metal_h_min_median_A": float(mh_t.median()) if metal_h else float("nan"),
        "metal_h_min_p10_A": _q(metal_h, 0.10),
        "metal_h_min_A": float(mh_t.min()) if metal_h else float("nan"),
        "metal_h_below_1p17A": metal_h_clash,
        "metal_h_below_1p17A_fraction": metal_h_clash / max(n_metal_clouds, 1),
        "dative_per_cloud_mean": float(dat_t.mean()),
        "clouds_with_zero_dative": int((dat_t == 0).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=400)
    args = ap.parse_args()

    tr = _load_trainer()
    sys.argv = ["probe", "--config", args.config, "--device", "cpu",
                "--out-dir", "/tmp/wfa113_t57", "--ckpt-dir", "/tmp/wfa113_t57"]
    cfg = tr.parse_args()

    def build():
        return tr.build_csv_cohort(
            str(cfg.data_csv),
            split=str(getattr(cfg, "data_split", "train")),
            n_clouds=int(args.n_clouds),
            seed=int(cfg.seed),
            metal_placement="bonded",
            dative_distance_cutoff=float(getattr(cfg, "dative_distance_cutoff", 2.6)),
        )

    new_place = tr._place_metal_at_donor
    from molmetal.data.fragment_layout import separate_fragments as new_sep

    # ---- BEFORE: identity separation + heavy-only clearance --------------
    tr._place_metal_at_donor = _old_place_metal_at_donor
    import molmetal.data.fragment_layout as _fl

    _fl.separate_fragments = lambda coords, fragments, **kw: coords.clone()
    before = _measure(build())

    # ---- AFTER -----------------------------------------------------------
    _fl.separate_fragments = new_sep
    tr._place_metal_at_donor = new_place
    after = _measure(build())

    out = {
        "metric": "wfa113_t57_fragment_layout",
        "tag": "MEASURED",
        "date": "2026-10-28",
        "config": args.config,
        "cohort": (
            f"{args.n_clouds} clouds of metalcytotox_5metal_multitask_v3.csv, "
            "split=train, metal_placement=bonded"
        ),
        "device": "cpu",
        "probe": "molmetal/scripts/wfa113_t57_fragment_layout_probe.py",
        "defects": {
            "T57": {
                "code": (
                    "molmetal/scripts/train_cfm_yuelbond_wasserstein.py "
                    ":: build_cloud_from_csv_row (no fragment separation)"
                ),
                "mechanism": (
                    "RDKit EmbedMolecule embeds each disconnected fragment "
                    "independently and translates them all to the same point. "
                    "Two waters of identical topology get the SAME conformer at "
                    "the SAME coordinates; [OH2]x6.[Ru] puts six oxygens on one "
                    "point and the twelve hydrogens in six coincident pairs."
                ),
                "fix": (
                    "molmetal/data/fragment_layout.py::separate_fragments keeps the "
                    "largest fragment and the metal's fragment fixed, then translates "
                    "each other fragment along a deterministic Fibonacci direction "
                    "until its closest contact with the placed atoms is >= 2.0 A."
                ),
            },
            "T58": {
                "code": (
                    "molmetal/scripts/train_cfm_yuelbond_wasserstein.py "
                    ":: _place_metal_at_donor (heavy-only clearance set)"
                ),
                "mechanism": (
                    "The metal was pushed along donor-minus-centroid until every "
                    "HEAVY atom was >= 1.2 A away, with hydrogens excluded. A C-H "
                    "or N-H sits exactly where 'straight out from the donor' lands, "
                    "so the metal was placed on top of it. Pushing further out also "
                    "walked the metal beyond the 2.6 A dative cutoff, leaving 21 "
                    "clouds with a metal bonded to nothing."
                ),
                "fix": (
                    "_place_metal_at_donor now tests clearance against every atom "
                    "(hydrogens included) and, when the straight-out position is "
                    "blocked, searches tilted directions at the SAME 2.05 A bond "
                    "length before falling back to pushing outward."
                ),
            },
        },
        "before": before,
        "after": after,
        "honest_limits": [
            "The separated solvent layout is not physical: a solvent molecule's "
            "position in a crystal is a fact about the crystal, and this loader "
            "has no crystal. The claim is only that the layout no longer teaches "
            "the model that atoms overlap.",
            "The 2.0 A inter-fragment clearance and the 1.2 A metal clearance are "
            "chosen, not fitted. 1.2 A is above the geometry term's M-H clash "
            "threshold (0.7 * (1.36 + 0.31) = 1.17 A) so the truth stops being a "
            "clash at the same threshold the loss uses.",
            "nonbonded_min is measured against the full RDKit bond set with "
            "explicit hydrogens; the trainer's own bond graph drops H bonds, so "
            "an O-H at 0.96 A is a bond there and is not counted as a clash here.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"before": before, "after": after}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
