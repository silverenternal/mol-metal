# SPDX-License-Identifier: Apache-2.0
"""Decode a finished CFM checkpoint and persist its SMILES.

WHY THIS EXISTS
---------------
`--save-generated-smiles` has existed in the trainer all along
(`train_cfm_yuelbond_wasserstein.py`, the flag is declared around :5629 and
consumed at :9069/9095), but none of the configs used on 2026-10-11 set it.
So three completed runs --

    cfm_fullcorpus_2026_10_10   seed 0, bond_type_f1 0.9194, early-stopped ep13
    cfm_objective_terms_2026_10_11   bond_type_f1 0.9305, ep9 -- NOT attributable:
        that arm differs from baseline only by two knobs measured inert (see
        configs/cfm_objective_terms_2026_10_11.yaml's RETRACTION block). Listed
        here as a checkpoint to replay, not as evidence for a mechanism.
    cfm_fullcorpus_b64_2026_10_11              bond_type_f1 0.8772, ep12

-- hold a checkpoint and an events.jsonl but NO generated_smiles.csv. Every
decoded SMILES string was computed and dropped on the floor. That leaves
`cfm_property_panel.py` with nothing to score, which is exactly why m4 (QED),
m5 (SA) and m7 (diversity) are UNMEASURED rather than bad: they were never fed.

This replays the BOND HEAD from each checkpoint and writes the CSV those tools
want, so the panel and any downstream classifier can finally run on decoded
molecules instead of on nothing.

It does NOT retrain and does NOT change any metric already on disk. It writes
one new CSV per run.

!! THESE ARE NOT GENERATED MOLECULES. MEASURED 2026-10-11. !!
    This script loads REAL tmQM geometries via load_tmqm_clouds (:142) and runs
    only the bond head: featurise -> _select_bonds -> _bridge_components ->
    _assemble_mol. It never calls the sampler. The trainer's real generation path
    is evaluate_generation_validity -> generate_clouds_from_noise
    (train_cfm_yuelbond_wasserstein.py:4683), which integrates the ODE from
    noise; this script calls none of it.

    So every SMILES it writes is a BOND-HEAD RECONSTRUCTION OF A REAL CORPUS
    MOLECULE, not a de-novo structure. Any panel metric computed from these CSVs
    (QED, SA, single-fragment ratio, diversity) describes the bond head
    reconstructing known chemistry -- NOT the generator. The proof is that real
    tmQM itself scores SA 4.9781 mean over 31150 complexes, against 5.0146 over
    these three CSVs: a gap of 0.14 SA, because the geometry was supplied.

    Independent confirmation that the gap is population, not batch: per-run
    means 4.974 / 5.033 / 5.032 (eta^2 = 0.0006, corr(SA, epoch) = -0.021), and
    size-matched to 192 the real corpus is 4.8781 -- 1.5-2.8 SA BETTER than these
    at every heavy-atom bucket.

    Reporting these as generator quality would be the same class of error as
    `gen_validity` itself, which is 0.23-0.38 for free generation and ~1.0 for
    seeded reconstruction.

COLUMNS
    epoch,seed,smiles,sanitized,n_fragments,largest_fragment_frac,metal_z

`epoch` is the checkpoint's own epoch so the panel's `--last-epoch` works, and
`n_fragments` is carried alongside the SMILES so the connectivity story can be
recomputed downstream without re-decoding.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "cfm_decode_dump",
        REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run directory names under molmetal/reports/",
    )
    ap.add_argument("--n-clouds", type=int, default=192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    args_cli = ap.parse_args()

    import yaml

    T = load_trainer()
    cfg = yaml.safe_load((REPO / "configs" / "cfm_tmqm_all6.yaml").read_text())
    from molmetal.data.tmqm_clouds import load_tmqm_clouds
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
        YuelBondDecoder,
    )
    from rdkit import Chem

    device = torch.device(args_cli.device)
    written = []
    for run in args_cli.runs:
        run_dir = REPO / "molmetal" / "reports" / run
        ckpt_path = run_dir / "checkpoints" / "cfm_yuelbond_wasserstein_seed0.pt"
        if not ckpt_path.is_file():
            alt = run_dir / "checkpoints" / "cfm_yuelbond_wasserstein_seed0_last.pt"
            ckpt_path = alt if alt.is_file() else ckpt_path
        if not ckpt_path.is_file():
            print(f"  {run}: no checkpoint, skipped")
            continue
        out_csv = run_dir / "generated_smiles.csv"
        if out_csv.is_file() and out_csv.stat().st_size > 0:
            print(f"  {run}: already has {out_csv.name}, skipped")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ckpt["args"]
        epoch = int(ckpt.get("epoch") or 0)

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

        head = YuelBondDecoder(
            head=model.head,
            hidden_dim=int(a["hidden_dim"]),
            n_layers=int(a["n_layers"]),
        )
        dec = YuelBondBondDecoder(
            head=model.head,
            hidden_dim=int(a["hidden_dim"]),
            n_layers=int(a["n_layers"]),
            decision=str(a.get("bond_decision", "threshold")),
            decision_threshold=float(a.get("bond_decision_threshold", 0.5)),
            connectivity_prior=str(a.get("connectivity_prior", "on")) == "on",
            connectivity_prior_min_logit=float(
                a.get("connectivity_prior_min_logit", 0.0)
            ),
        )

        clouds = load_tmqm_clouds(
            cfg["data"]["tmqm_dir"],
            metals=[s.strip() for s in cfg["data"]["tmqm_metals"].split(",")],
            coord_number=None,
            n_clouds=args_cli.n_clouds,
            seed=int(a.get("seed", 0)),
            split=args_cli.split,
            val_frac=float(cfg["data"]["tmqm_val_frac"]),
        )

        rows: List[Dict[str, object]] = []
        n_ok = n_sanitized = 0
        with torch.no_grad():
            for cloud in clouds:
                c = T._to_device_cloud(cloud, device)
                yres = head.featurise(c)
                if yres.logits is None or yres.logits.shape[0] == 0:
                    continue
                sel, dative, _k, _o = dec._select_bonds(c, yres)
                sel2, _d2, _added = dec._bridge_components(
                    c, yres, list(sel), list(dative)
                )
                res = dec._assemble_mol(c, sel2, [])
                # `_assemble_mol` returns DecodedMol DIRECTLY (it carries
                # .smiles/.sanitized itself); it is not wrapped in a
                # YuelBondDecodeResult, so reading `res.decoded` yields None
                # and silently writes zero rows.
                dec_mol = res
                smi = getattr(dec_mol, "smiles", None) or ""
                sanitized = bool(getattr(dec_mol, "sanitized", False))
                nfrag, largest = 0, 0.0
                if smi:
                    n_ok += 1
                    n_sanitized += int(sanitized)
                    m = Chem.MolFromSmiles(smi)
                    if m is not None:
                        frags = Chem.GetMolFrags(m, asMols=False)
                        nfrag = len(frags)
                        sizes = [len(f) for f in frags]
                        largest = (
                            max(sizes) / sum(sizes) if sum(sizes) else 0.0
                        )
                    metal_z = ""
                    for z in c.atomic_numbers.tolist():
                        if int(z) in (78, 44, 77, 45, 76, 75, 46, 29, 79, 28, 26, 30):
                            metal_z = int(z)
                            break
                    rows.append(
                        {
                            "epoch": epoch,
                            "seed": int(a.get("seed", 0)),
                            "smiles": smi,
                            "sanitized": int(sanitized),
                            "n_fragments": nfrag,
                            "largest_fragment_frac": round(largest, 4),
                            "metal_z": metal_z,
                        }
                    )

        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "epoch",
                    "seed",
                    "smiles",
                    "sanitized",
                    "n_fragments",
                    "largest_fragment_frac",
                    "metal_z",
                ],
            )
            w.writeheader()
            w.writerows(rows)

        print(
            f"  {run}: ckpt_epoch={epoch} decoded={len(rows)} "
            f"with_smiles={n_ok} sanitized={n_sanitized} "
            f"({n_sanitized / max(n_ok, 1):.4f}) -> {out_csv.relative_to(REPO)}"
        )
        written.append(str(out_csv.relative_to(REPO)))
        del model

    print()
    if written:
        print("  next: cfm_property_panel.py --smiles-csv <csv> --last-epoch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())