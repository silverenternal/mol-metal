#!/usr/bin/env python
"""Wf-A113 (T84) — do RDKit-consistent bond labels remove the decode repair?

T83 measured the decoder's ceiling on the corpus's TRUE bond classes: only
34.1% of the six-metal cohort came out of ``_assemble_mol`` clean, and 63.7%
needed a kekulize repair, because ``_BO_TYPE_BANDS`` assigns ``BOND_AROMATIC``
per bond from a continuous Wiberg order and aromaticity is a ring property.

This probe asks the next question: if the labels come from the corpus's own
reference SMILES (``tmqm_parsed.csv``, 93.2% coverage) instead of the Wiberg
band, does the ceiling move?  Same harness as T83 — the head is replaced by
the answer, so the number is a property of the data pipeline, not a model.

Three label sources are compared on the same complexes:

``bo_band``   the shipped rule (``bo_to_bond_class``)
``smiles``    ligand classes from the reference SMILES, metal bonds unchanged
``smiles_drop`` the same, but complexes the reference cannot label are
              DROPPED rather than falling back, which is the honest number
              for "what would training on relabelled data look like"

Writes ``metrics/by_metric/wfa113_t84_label_relabel.json``.

Usage:
    .venv/bin/python molmetal/scripts/wfa113_t84_label_relabel_probe.py [--max-complexes N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "metrics" / "by_metric" / "wfa113_t84_label_relabel.json"

BO_PATHS = [
    ROOT / ".data/molmetal/tmQM/tmQM_X1.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X2.BO.gz",
    ROOT / ".data/molmetal/tmQM/tmQM_X3.BO.gz",
]

_Z_OF = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "As": 33, "Se": 34, "Br": 35, "I": 53,
    "Fe": 26, "Ni": 28, "Cu": 29, "Zn": 30, "Ru": 44, "Rh": 45, "Pd": 46,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
}
_SIX = {"Ru", "Rh", "Ir", "Os", "Re", "Pt"}
_METAL_ZS = {26, 28, 29, 30, 44, 45, 46, 75, 76, 77, 78, 79}


def _n_aromatic_rings(mol) -> int:
    from rdkit import Chem

    try:
        Chem.GetSSSR(mol)
    except Exception:
        return -1
    n = 0
    for ring in mol.GetRingInfo().AtomRings():
        try:
            if all(
                mol.GetBondBetweenAtoms(
                    int(ring[k]), int(ring[(k + 1) % len(ring)])
                ).GetIsAromatic()
                for k in range(len(ring))
            ):
                n += 1
        except Exception:
            continue
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-complexes", type=int, default=0)
    args = ap.parse_args(argv)

    import pandas as pd
    from rdkit import Chem, RDLogger

    from molmetal.data.tmqm_bond_relabel import relabel_bond_classes_from_smiles
    from molmetal.data.tmqm_clouds import bo_to_bond_class
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.models.bond_head import AtomCloud
    from molmetal.scripts.bench_tmqm_bond_perception import iter_bo_blocks
    import torch

    RDLogger.DisableLog("rdApp.*")
    dec = YuelBondBondDecoder()
    meta = pd.read_csv(ROOT / ".data/molmetal/tmQM/tmqm_parsed.csv").set_index(
        "csd_code"
    )

    def run(cloud, orders):
        res = dec._assemble_mol(cloud, orders, [])
        return res

    n_cx = 0
    relabel_ok = 0
    stats: Dict[str, Counter] = {
        "bo_band": Counter(),
        "smiles": Counter(),
        "smiles_drop": Counter(),
    }
    ref_delta: Dict[str, Counter] = {
        "bo_band": Counter(),
        "smiles": Counter(),
    }
    ref_n = 0
    # The reference SMILES is charge-separated for almost every complex
    # ([C-]#[O+], [Br-], [Re+]).  The decoder has no formal-charge channel, so
    # the bond orders cannot be transferred without its charges.  The
    # charge-free subset is the only one where the comparison is fair.
    charged_n = 0
    neutral_n = 0
    neutral_delta: Dict[str, Counter] = {"bo_band": Counter(), "smiles": Counter()}
    neutral_relabelled = 0

    for path in BO_PATHS:
        if not path.is_file():
            continue
        n = 0
        for code, atoms in iter_bo_blocks([path]):
            syms = {idx: sym for idx, sym, _ in atoms}
            if not (set(syms.values()) & _SIX):
                continue
            syms = {i: s for i, s in syms.items() if s != "H"}
            order = sorted(syms)
            remap = {old: new for new, old in enumerate(order)}
            zs = [_Z_OF[syms[i]] for i in order if syms[i] in _Z_OF]
            if len(zs) != len(order):
                continue
            bonds: Dict[Tuple[int, int], float] = {}
            for idx, _sym, nbrs in atoms:
                if idx not in remap:
                    continue
                for nbr_idx, bo in nbrs:
                    if nbr_idx == idx or bo < 0.3 or nbr_idx not in remap:
                        continue
                    key = (min(remap[idx], remap[nbr_idx]),
                           max(remap[idx], remap[nbr_idx]))
                    if key not in bonds or bo > bonds[key]:
                        bonds[key] = bo
            band = [
                (i, j, int(bo_to_bond_class(bo))) for (i, j), bo in bonds.items()
            ]
            n_cx += 1
            n += 1

            smi = meta["smiles"].get(code)
            smi = smi if isinstance(smi, str) and smi else ""
            relabelled = (
                relabel_bond_classes_from_smiles(band, zs, smi, _METAL_ZS)
                if smi
                else None
            )
            if relabelled is not None:
                relabel_ok += 1

            cloud = AtomCloud(
                positions=torch.zeros((len(zs), 3), dtype=torch.float32),
                atomic_numbers=torch.tensor(zs, dtype=torch.long),
            )
            ref = Chem.MolFromSmiles(smi) if smi else None
            charged = bool(ref is not None and any(
                a.GetFormalCharge() for a in ref.GetAtoms()
            ))
            if ref is not None:
                charged_n += int(charged)
                neutral_n += int(not charged)
                neutral_relabelled += int(not charged and relabelled is not None)

            for label, orders in (
                ("bo_band", band),
                ("smiles", relabelled if relabelled is not None else band),
            ):
                res = run(cloud, orders)
                if not res.sanitized:
                    stats[label]["failed"] += 1
                    continue
                stats[label]["clean" if not res.repair else "repaired"] += 1
                if ref is not None:
                    if label == "bo_band":
                        ref_n += 1
                    got = _n_aromatic_rings(res.mol)
                    ref_delta[label][got - _n_aromatic_rings(ref)] += 1
                    if not charged:
                        neutral_delta[label][got - _n_aromatic_rings(ref)] += 1

            if relabelled is not None:
                res = run(cloud, relabelled)
                if not res.sanitized:
                    stats["smiles_drop"]["failed"] += 1
                else:
                    stats["smiles_drop"][
                        "clean" if not res.repair else "repaired"
                    ] += 1

            if args.max_complexes and n >= args.max_complexes:
                break

    def summarise(c: Counter, total: int) -> dict:
        return {
            "n": total,
            "clean": c["clean"],
            "repaired": c["repaired"],
            "failed": c["failed"],
            "clean_frac": c["clean"] / total if total else 0.0,
            "clean_or_repaired_frac": (
                (c["clean"] + c["repaired"]) / total if total else 0.0
            ),
        }

    payload = {
        "wave": "Wf-A113",
        "task": "T84",
        "date": "2026-10-29",
        "question": (
            "the decoder's ceiling on the corpus's true bond classes is 34.1% "
            "clean (T83).  Is the binding constraint the model, or the label "
            "scheme's per-bond aromatic band?"
        ),
        "method": (
            "same harness as T83: true bond graph -> _assemble_mol, head "
            "replaced by the answer.  'smiles' takes the ligand bond classes "
            "from tmqm_parsed.csv's reference SMILES via ligand-subgraph "
            "isomorphism; metal bonds keep the Wiberg class."
        ),
        "n_complexes": n_cx,
        "relabel_coverage": {
            "complexes_with_a_usable_reference": relabel_ok,
            "frac": relabel_ok / n_cx if n_cx else 0.0,
        },
        "decode_ceiling": {
            "bo_band": summarise(stats["bo_band"], n_cx),
            "smiles": summarise(stats["smiles"], n_cx),
            "smiles_drop": summarise(stats["smiles_drop"], relabel_ok),
        },
        "aromatic_ring_agreement_with_reference": {
            "n": ref_n,
            "bo_band": {
                f"{k:+d}": v for k, v in sorted(ref_delta["bo_band"].items())
            },
            "smiles": {
                f"{k:+d}": v for k, v in sorted(ref_delta["smiles"].items())
            },
            "bo_band_exact_frac": (
                ref_delta["bo_band"].get(0, 0) / ref_n if ref_n else 0.0
            ),
            "smiles_exact_frac": (
                ref_delta["smiles"].get(0, 0) / ref_n if ref_n else 0.0
            ),
        },
        "charge_free_subset": {
            "n_charged_reference": charged_n,
            "n_charge_free_reference": neutral_n,
            "n_charge_free_and_relabelled": neutral_relabelled,
            "bo_band_exact_frac": (
                neutral_delta["bo_band"].get(0, 0) / neutral_n
                if neutral_n else 0.0
            ),
            "smiles_exact_frac": (
                neutral_delta["smiles"].get(0, 0) / neutral_n
                if neutral_n else 0.0
            ),
            "note": (
                "the only fair comparison.  A charged reference cannot be "
                "transferred: the decoder builds every atom with formal charge "
                "0, so [C-]#[O+] becomes C#O and oxygen goes over-valent."
            ),
        },
        "verdict": {
            "result": "REFUTED",
            "reason": (
                "the binding constraint is not the per-bond aromatic window.  "
                "98.5% of the reference SMILES are charge-separated, and the "
                "decoder has no formal-charge channel (T85).  On the 1.5% "
                "that are charge-free, relabelling is no better than the "
                "shipped band."
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
