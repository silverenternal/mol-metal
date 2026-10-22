#!/usr/bin/env python3
"""UP-31 — pairwise logistic inside the D-MPNN, against a matched BCE arm.

=====================================================================
Why this exists
=====================================================================
UP-24 measured +0.0080 AUC for a pairwise loss and was killed anyway, because
the gate it was measured against (0.63) had been frozen for the D-MPNN while
UP-24's learner was a linear ECFP4 model.  That is a learner mismatch, not a
result about the loss.  This package reruns the comparison on the D-MPNN
itself, where the gate was written.

The two arms are the same script, the same cohort, the same folds, the same
seed list, the same 30-epoch budget, and the same Wasserstein setting.  They
differ in one YAML key: ``protocol.pairwise_weight`` (0.0 vs 1.0).  Any AUC
gap is therefore attributable to the term.

=====================================================================
What the term is
=====================================================================
    loss_pairwise = -log sigmoid((f(x+) - f(x-) - margin) / T)

over every (active, inactive) pair inside a batch, added to the existing BCE
plus regression loss with weight ``pairwise_weight``.  ``f`` is the active-class
logit.  The first run of this package fed the whole 2-vector into the term and
scored 0.4377 — worse than chance — because pushing both class scores apart
fights the classifier.  The shipped code takes column 1; the 0.4377 run is
recorded in the caveats rather than deleted.

=====================================================================
Gate (todo.json, frozen 2026-10-22T15:45+08:00)
=====================================================================
primary   : pairwise arm, Ru/Ir mean AUC >= 0.63
reproduce : BCE arm reproduces 0.5974 +/- 0.01 on the same code path, else the
            whole package is void
per_metal : Ru >= 0.60 and Ir >= 0.60
kill      : pairwise < 0.61 with the BCE arm reproduced closes the LOMO line;
            the paper reports 0.5974 only.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_lomo_pairwise_dmpnn.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

GATE_FROZEN_AT = "2026-10-22T15:45+08:00"
PAIRWISE_GATE = 0.63
PAIRWISE_KILL = 0.61
PER_METAL_GATE = 0.60
REPRODUCE_TARGET = 0.5974
REPRODUCE_TOL = 0.01


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def _load(path: Path) -> Dict[str, object]:
    doc = json.loads(path.read_text())
    return {
        "value": float(doc["value"]),
        "value_std": float(doc["value_std"]),
        "per_metal": {k: float(v) for k, v in doc["per_fold_auc_mean"].items()},
        "per_run": [float(v) for v in doc["per_run_auc"]],
        "n_runs": int(doc["n_runs"]),
        "metals": list(doc["metals"]),
        "seeds": list(doc["seeds"]),
        "epochs": int(doc["epochs"]),
        "source": _rel(path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairwise", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "reports"
                    / "up31_lomo_pairwise" / "metrics.json")
    ap.add_argument("--bce", type=Path,
                    default=PROJECT_ROOT / "molmetal" / "reports"
                    / "up31_lomo_pairwise_bce" / "metrics.json")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric"
                    / "msrs_lomo_pairwise_dmpnn.json")
    args = ap.parse_args(argv)

    for p in (args.pairwise, args.bce):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1

    pair = _load(args.pairwise)
    bce = _load(args.bce)

    checks = {
        "primary": {
            "arm": "pairwise",
            "value": round(pair["value"], 4),
            "threshold": PAIRWISE_GATE,
            "pass": bool(pair["value"] >= PAIRWISE_GATE),
        },
        "reproduce": {
            "arm": "bce",
            "value": round(bce["value"], 4),
            "target": REPRODUCE_TARGET,
            "tolerance": REPRODUCE_TOL,
            "pass": bool(abs(bce["value"] - REPRODUCE_TARGET) <= REPRODUCE_TOL),
        },
        "per_metal": {
            "pairwise": {k: round(v, 4) for k, v in pair["per_metal"].items()},
            "threshold": PER_METAL_GATE,
            "pass": bool(all(v >= PER_METAL_GATE for v in pair["per_metal"].values())),
        },
    }
    gate_pass = all(v["pass"] for v in checks.values())
    kill = {
        "rule": "pairwise < 0.61 with the BCE arm reproduced closes the LOMO line",
        "value": round(pair["value"], 4),
        "triggered": bool(
            pair["value"] < PAIRWISE_KILL and checks["reproduce"]["pass"]
        ),
    }
    delta = round(pair["value"] - bce["value"], 4)

    out = {
        "artifact": "msrs_lomo_pairwise_dmpnn",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L3 loss (UP-31)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "git_commit": _git_commit(),
            "pairwise_config": "configs/msrs_lomo_pairwise_dmpnn.yaml",
            "bce_config": "configs/msrs_lomo_pairwise_bce.yaml",
            "script": _rel(Path(__file__)),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "cohort": ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
            "protocol": "leave-one-metal-out; Ru and Ir only, the two folds "
                        "lomo_resolvability_audit calls resolvable",
            "single_variable": "protocol.pairwise_weight 0.0 (bce) vs 1.0 (pairwise); "
                               "every other YAML key is identical",
            "pairwise_loss": "-log sigmoid((f(x+) - f(x-) - margin) / T) over every "
                             "(active, inactive) pair in a batch, f = active-class logit",
            "margin": 0.0,
            "temperature": 1.0,
            "epochs": pair["epochs"],
            "seeds": pair["seeds"],
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "arms": {"pairwise": pair, "bce": bce},
        "delta_pairwise_minus_bce": delta,
        "gate": {"frozen_at": GATE_FROZEN_AT, "checks": checks,
                 "gate_pass": gate_pass, "kill": kill},
        "caveats": [
            "The first pairwise run fed the whole (B, 2) logit vector into the term "
            "and scored 0.4377. That run is a bug, not an arm: pushing both class "
            "scores apart fights the classifier. The shipped term takes column 1.",
            "Two metals, five seeds. The per-metal split (Ru 0.6191, Ir 0.5665) moves "
            "in opposite directions, so the mean is not evidence that the term is "
            "neutral for every fold.",
            "The term pairs every active row with every inactive row in the batch, "
            "not within (ligand, cell) groups as UP-24 did. That is a different "
            "pairing rule and the two deltas are not directly comparable.",
            "margin and temperature are at their neutral values (0.0 / 1.0); no "
            "sweep was run, so this is one point in that space, not a best case.",
            "The BCE arm lands at 0.5924 against the 0.5974 the resolvability audit "
            "reports. The 0.005 gap is inside the frozen tolerance but is not zero; "
            "the audit aggregated a stored artifact while this arm retrained.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[UP-31] pairwise={pair['value']:.4f} bce={bce['value']:.4f} "
          f"delta={delta:+.4f} gate_pass={gate_pass} kill={kill['triggered']}")
    print(f"json: {args.out}")
    print("PASS" if gate_pass else "FAIL")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
