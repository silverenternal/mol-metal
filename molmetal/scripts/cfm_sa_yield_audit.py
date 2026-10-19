#!/usr/bin/env python3
"""STR-6 — SA yield audit: what fraction of generated molecules is synthesizable?

The plan asked for "3 seeds x 200, SA penalty on/off, gate SA <= 4.5 and mean
novelty >= 0.90".  Before spending GPU on that A/B, three facts rule it out:

1. ``--sa-penalty`` is inert.  ``sa_penalty.py`` says so in its own docstring:
   sascorer is a fragment-frequency lookup with no gradient, and the trainer's
   proxy is wrapped in ``torch.tensor(float(...))`` (``requires_grad=False``).
   Parameter updates with the flag on are bit-for-bit identical to off.
2. There is no differentiable SA surrogate to guide decoding with.  The one
   that existed was deleted on 2026-10-04 because it was fake.
3. Only one checkpoint exists per run (``..._seed0.pt``), so a 3-seed sweep
   would first require training two more seeds.

What remains measurable is the yield: generate an unguided pool, score SA per
molecule, and report the fraction that clears 4.5.  That is what this script
does, over the pool of record plus every large dump on disk.

The gate is only meaningful with a yield attached.  "SA <= 4.5" over a 0.7%
hand-picked subset is a property of the filter, not of the generator.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/cfm_sa_yield_audit.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SA_GATE = 4.5
NOVELTY_GATE = 0.90
THRESHOLDS = (4.0, 4.5, 5.0, 5.5, 6.0)
DEFAULT_POOLS = (
    ("panel_of_record", "molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv"),
    ("pcgrad_on", "molmetal/reports/cfm_pcgrad_on_2026_10_14/generated_smiles.csv"),
    ("pcgrad_off", "molmetal/reports/cfm_pcgrad_off_2026_10_15/generated_smiles.csv"),
)


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score interval; well behaved at k = 0, where the normal one is not."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def novelty_of(smiles: Sequence[str], train_path: Path) -> Optional[Dict[str, object]]:
    """Mean 1 - max Tanimoto to the training corpus, for a selected subset."""
    if not smiles or not train_path.exists():
        return None
    import pandas as pd

    from molmetal.scripts.cfm_generation_metrics import (
        max_tanimoto_to_reference,
        morgan,
    )

    train = pd.read_csv(train_path, low_memory=False)
    if "split" in train.columns:
        train = train[train["split"] == "train"]
    ref = [f for f in (morgan(s) for s in train["smiles"].dropna().astype(str)) if f is not None]
    fps = [f for f in (morgan(s) for s in smiles) if f is not None]
    if not fps or not ref:
        return None
    sims = max_tanimoto_to_reference(fps, ref)
    novelty = 1.0 - sims
    return {
        "n": int(len(novelty)),
        "mean_novelty": round(float(novelty.mean()), 4),
        "median_novelty": round(float(sorted(novelty)[len(novelty) // 2]), 4),
        "underpowered": bool(len(novelty) < 20),
        "train_reference": str(train_path.relative_to(PROJECT_ROOT)),
        "fingerprint": "Morgan radius 2, 1024 bits",
    }


def score_smiles(smiles: Sequence[str]) -> Dict[str, object]:
    """Per-molecule SA / QED.  Unparseable rows are counted, never dropped."""
    from rdkit import Chem, RDLogger

    from molmetal.adapters.flow_matching_lipman.sa_penalty import compute_sa_score
    from molmetal.adapters.flow_matching_lipman.qed_penalty import compute_qed_score

    RDLogger.DisableLog("rdApp.*")
    sas: List[float] = []
    qeds: List[float] = []
    gate_smiles: List[str] = []
    n_bad = 0
    for text in smiles:
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) else None
        if mol is None:
            n_bad += 1
            continue
        sa = compute_sa_score(mol)
        if sa is None:
            n_bad += 1
            continue
        sas.append(float(sa))
        if float(sa) <= SA_GATE:
            gate_smiles.append(Chem.MolToSmiles(mol))
        qed = compute_qed_score(mol)
        if qed is not None:
            qeds.append(float(qed))
    return {"sas": sas, "qeds": qeds, "n_unscored": n_bad, "gate_smiles": gate_smiles}


def pool_report(label: str, path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = score_smiles([r.get("smiles", "") for r in rows])
    sas = scored["sas"]
    if not sas:
        return {
            "label": label,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_rows": len(rows),
            "n_scored": 0,
        }
    ordered = sorted(sas)
    return {
        "label": label,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_rows": len(rows),
        "n_scored": len(sas),
        "n_unscored": int(scored["n_unscored"]),
        "sa_mean": round(sum(sas) / len(sas), 4),
        "sa_median": round(ordered[len(ordered) // 2], 4),
        "sa_min": round(ordered[0], 4),
        "sa_max": round(ordered[-1], 4),
        "sa_counts_at_or_below": {str(t): sum(1 for s in sas if s <= t) for t in THRESHOLDS},
        "qed_mean": round(sum(scored["qeds"]) / len(scored["qeds"]), 4) if scored["qeds"] else None,
        "gate_smiles": list(scored["gate_smiles"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "cfm_sa_guided.json",
    )
    args = parser.parse_args(argv)

    pools: List[Dict[str, object]] = []
    for label, rel in DEFAULT_POOLS:
        report = pool_report(label, PROJECT_ROOT / rel)
        if report is not None:
            pools.append(report)
            print(
                f"[str6] {label:16s} n={report['n_scored']:5d} "
                f"sa_mean={report.get('sa_mean')} "
                f"<=4.5={report.get('sa_counts_at_or_below', {}).get('4.5')}"
            )

    scored = [p for p in pools if p.get("n_scored")]
    n_total = sum(int(p["n_scored"]) for p in scored)
    n_gate = sum(int(p["sa_counts_at_or_below"]["4.5"]) for p in scored)
    rate = (n_gate / n_total) if n_total else 0.0
    ci = wilson(n_gate, n_total)

    gate_smiles = [s for p in scored for s in p.get("gate_smiles", [])]
    novelty = novelty_of(
        gate_smiles, PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
    )
    verdict = "KILL" if (n_total and n_gate / n_total < 0.05) else "PARTIAL"

    payload = {
        "artifact": "cfm_sa_guided",
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/cfm_sa_yield_audit.py",
            "git_commit": _git_commit(),
        },
        "design": {
            "sa_gate": SA_GATE,
            "novelty_gate": NOVELTY_GATE,
            "thresholds": list(THRESHOLDS),
            "guided_decoding": "UNAVAILABLE",
            "blockers": [
                "--sa-penalty is inert: sascorer has no gradient and the trainer proxy is wrapped in torch.tensor(float(...)) with requires_grad=False",
                "no differentiable SA surrogate exists; the previous one was deleted on 2026-10-04 as fake",
                "one checkpoint per run (..._seed0.pt), so a 3-seed sweep needs two more training runs",
            ],
            "pool_of_record": "cfm_free_generation_panel_seed0.json (66 molecules, 192 attempts)",
        },
        "pools": [{k: v for k, v in p.items() if k != "gate_smiles"} for p in pools],
        "combined": {
            "n_scored": n_total,
            "n_sa_le_gate": n_gate,
            "rate_sa_le_gate": round(rate, 4),
            "wilson_ci95": ci,
        },
        "gate_subset": {
            "n": len(gate_smiles),
            "novelty": novelty,
            "note": "the gate asks for SA <= 4.5 AND mean novelty >= 0.90; both can hold for a "
            "0.7% hand-picked subset while the generator itself is unchanged",
        },
        "verdict": verdict,
        "verdict_detail": {
            "gate_met": bool(n_gate / n_total >= 0.05) if n_total else False,
            "note": "the gate is a property of the generator only if the yield is reported next to it",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        f"[str6] combined n={n_total} <=4.5={n_gate} rate={rate:.4f} "
        f"ci={ci} verdict={verdict}"
    )
    print(f"[str6] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
