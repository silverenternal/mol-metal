#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Benchmark a CFM checkpoint on the properties a generative model is judged by.

WHY THIS EXISTS
---------------
The project has bond-perception numbers (EXISTENCE F1 0.9962, macro-F1(type)
0.4784 +/- 0.0055 over 3 seeds) but has never measured the properties that
decide whether generated molecules are any use: drug-likeness, synthesability,
permeability, mutagenicity, solubility. The user directive 2026-10-12 is that
these must at least reach the last-two-years literature average BEFORE any
retraining, and that the loss / architecture work should be driven by what the
benchmark says is missing -- not guessed at.

The one free-generation measurement on disk (metrics/by_metric/
cfm_free_generation_panel_seed0.json) covers QED and SA only, on one seed. This
script adds the ADMET triple and, crucially, always reports the SAME metrics on
a REAL tmQM cohort so there is a reference band to read the generated numbers
against. Without that band a number like QED 0.094 is uninterpretable.

It MEASURES. It does not train, does not modify the checkpoint, and does not
write to metrics/by_metric/ -- it writes to the caller-specified --out-json so
the numbers can be reviewed before anything is claimed.

USAGE
    python scripts/cfm_checkpoint_benchmark.py \
        --smiles-csv molmetal/reports/<run>/generated_smiles.csv \
        --out-json /tmp/bench.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TARGETS = {
    "qed": {
        "direction": "higher",
        "target": 0.5,
        "source": "metrics/by_metric/qed.json",
    },
    "sa": {
        "direction": "lower",
        "target": 3.0,
        "source": "Ertl 2009; SA <= 3 is the common synthesability threshold",
    },
    "ames_mutagen": {
        "direction": "lower",
        "target": 0.30,
        "source": "Wf-297D band; Benegini-Bossa 2012 / Kazius 2005 via RDKit FilterCatalog",
    },
    "cell_permeability_logPapp": {
        "direction": "higher",
        "target": -6.0,
        "source": "Wf-297D band; Veber 2002 / Hou 2007",
    },
    "aqueous_solubility_logS": {
        "direction": "higher",
        "target": -4.0,
        "source": "Wf-297D band; Delaney 2004 ESOL",
    },
    # 2026-10-12 (BLK-C-M4) — aromatic bond fraction, the axis the aromatic
    # logit offset moves. Added because the lever was being aimed with no
    # metric that could see it: the offset changes which bonds are drawn
    # AROMATIC, and nothing in this panel recorded aromaticity at all.
    #
    # The target is the MEASURED real tmQM value, not a literature gate. On
    # real tmQM complexes (n=31,150, six train metals) aromatic_bond_frac is
    # 0.454591 -- see metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json.
    # A generated cohort sits far below it (0.001229 measured pre-fix), so
    # "higher" is the direction and 0.454591 is the reference, not a promise.
    "aromatic_bond_frac": {
        "direction": "higher",
        "target": 0.454591,
        "source": (
            "MEASURED real tmQM six-metal band, n=31,150 "
            "(metrics/by_metric/cfm_real_tmqm_sixmetal_baseline.json)"
        ),
    },
}


def load_smiles(path: Path, last_epoch: bool) -> List[str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [r for r in rows if (r.get("smiles") or "").strip()]
    if last_epoch and rows and "epoch" in rows[0]:
        top = max(int(r["epoch"]) for r in rows if str(r.get("epoch", "")).strip())
        rows = [r for r in rows if int(r["epoch"]) == top]
    return [r["smiles"] for r in rows]


def _safe(fn, smi):
    try:
        return fn(smi)
    except Exception:
        return None


def score_cohort(name: str, smiles: List[str]) -> Dict[str, object]:
    from molmetal.molmetal_lam.sbdd_env import metrics_v2 as M

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    parsed, failed = 0, 0
    vals: Dict[str, List[float]] = {k: [] for k in TARGETS}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            failed += 1
            continue
        parsed += 1
        for key, fn in (
            ("ames_mutagen", M.ames_mutagen),
            ("cell_permeability_logPapp", M.cell_permeability_logPapp),
            ("aqueous_solubility_logS", M.aqueous_solubility_logS),
        ):
            v = _safe(fn, s)
            if v is not None:
                vals[key].append(float(v))
        try:
            from rdkit.Chem import QED

            vals["qed"].append(float(QED.qed(m)))
        except Exception:
            pass
        # 2026-10-12 (BLK-C-M4) — aromatic bond fraction, the axis the
        # aromatic logit offset actually moves.
        #
        # Per MOLECULE, then averaged. Deliberately not the pooled ratio
        # sum(aromatic)/sum(bonds): that is dominated by whichever molecules
        # happen to be large, so a cohort could raise it purely by emitting a
        # few big aromatic molecules while its typical molecule stayed flat.
        # A molecule with no bonds scores 0.0 rather than being skipped --
        # skipping would let a cohort IMPROVE by decoding fewer real
        # molecules, which is the survivor-conditioning defect the L3 sweep
        # already documents.
        try:
            n_bonds = m.GetNumBonds()
            vals["aromatic_bond_frac"].append(
                (
                    sum(1 for b in m.GetBonds() if b.GetIsAromatic()) / n_bonds
                )
                if n_bonds
                else 0.0
            )
        except Exception:
            pass
        try:
            # Same late-import the panel uses (cfm_property_panel.py:117):
            # RDKit ships the upstream SA_Score package in its Contrib dir,
            # which is NOT on sys.path by default. Importing it from
            # molmetal.molmetal_lam.reward.sa_score does not work -- that
            # module does not export calculateScore -- and the failure was
            # swallowed, leaving SA as NA for every cohort.
            import os as _os

            from rdkit.Chem import RDConfig

            _sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
            if _sa_dir not in sys.path:
                sys.path.append(_sa_dir)
            import sascorer  # type: ignore[import-not-found]

            vals["sa"].append(float(sascorer.calculateScore(m)))
        except Exception:
            pass

    out: Dict[str, object] = {
        "cohort": name,
        "n_input": len(smiles),
        "n_parsed": parsed,
        "n_failed": failed,
    }
    for k, arr in vals.items():
        if arr:
            spec = TARGETS[k]
            mean = st.mean(arr)
            ok = mean >= spec["target"] if spec["direction"] == "higher" else mean <= spec["target"]
            out[k] = {
                "mean": round(mean, 4),
                "std": round(st.stdev(arr), 4) if len(arr) > 1 else None,
                "n": len(arr),
                "target": spec["target"],
                "direction": spec["direction"],
                "meets_target": bool(ok),
                "target_source": spec["source"],
            }
        else:
            out[k] = {"mean": None, "note": "no value could be computed"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smiles-csv", required=True, type=Path,
                    help="generated_smiles.csv from a CFM run")
    ap.add_argument("--reference-csv", type=Path, default=None,
                    help="optional real-molecule CSV, e.g. a tmQM decode")
    ap.add_argument("--last-epoch", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()

    gen = load_smiles(a.smiles_csv, a.last_epoch)
    if a.limit:
        gen = gen[: a.limit]
    if not gen:
        print("no SMILES to score", file=sys.stderr)
        return 1

    cohorts = [score_cohort("generated", gen)]
    if a.reference_csv and a.reference_csv.is_file():
        ref = load_smiles(a.reference_csv, a.last_epoch)
        if a.limit:
            ref = ref[: a.limit]
        if ref:
            cohorts.append(score_cohort("reference", ref))

    print(f"{'cohort':12} {'n':>4} " + " ".join(f"{k:>9}" for k in TARGETS))
    for c in cohorts:
        cells = []
        for k in TARGETS:
            v = c[k].get("mean")
            flag = ""
            if v is not None:
                flag = "*" if c[k].get("meets_target") else "!"
            cells.append(f"{v:9.4f}{flag}" if v is not None else f"{'NA':>9}")
        print(f"{c['cohort']:12} {c['n_parsed']:>4} " + " ".join(cells))
    print("\n* meets target, ! misses target; targets in TARGETS (see --out-json)")

    if a.out_json:
        a.out_json.write_text(json.dumps(
            {"cohorts": cohorts, "targets": TARGETS,
             "source_csv": str(a.smiles_csv)}, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())