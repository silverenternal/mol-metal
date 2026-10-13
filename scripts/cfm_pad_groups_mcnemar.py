#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""BLK-C-PERF: does pad_groups change the result? McNemar on PAIRED outcomes.

WHY THIS SCRIPT AND NOT THE ONE THAT CRASHED
--------------------------------------------
The first attempt (cfm_pad_groups_dist.py, log at $JOB/tmp/pad_dist.txt)
died of CUDA OOM at 04:57:04 on the 31 GB card while the grouped arm was
still holding 30.65 GB. Its output was aggregates only:

    SEED  path      n_out  decoded  single_frag  n_frags
    0     grouped     37       37      0.8919     1.108   [1892.7s]

which cannot answer the question, for two independent reasons.

1. AGGREGATES CANNOT BE PAIRED. The A/B is paired by construction -- same
   192 val clouds, same noise seed, only the batching differs -- so the whole
   answer lives in the DISCORDANT CELLS. ``n_out 37 vs n_out 38`` could be one
   cloud flipping or thirty clouds flipping in opposite directions, and a
   mean difference cannot tell those apart.

2. ``smiles_out`` CANNOT BE PAIRED. ``evaluate_generation_validity`` appends
   to it only on success (``if smiles_out is not None and smi``), so failures
   are compressed out and its indices stop corresponding to ``clouds``. The
   old script derived ``single_frag`` and ``n_frags`` from that list, so its
   denominators were the SUCCESSFUL decodes -- the survivor-conditioning
   defect the Vina sweep already documents.

So this script records a per-cloud outcome VECTOR for each arm (via the new
``per_cloud_out`` parameter, which is index-aligned including failures) and
applies McNemar's exact test to the discordant cells.

WHY McNemar
-----------
The two arms are paired, so the difference is within-cloud flips, not two
independent binomial draws. An unpaired difference-in-proportions test would
ignore the pairing and be strictly looser whenever many clouds agree. The
same reasoning that produced cfm_blkc_m3_noise_floor.json.

Also, and this is the measurement that matters for a SPEEDUP claim: the
question is not whether padding is "correct" but whether it changes the
answer. Wilson intervals on the two arms are also reported, but the verdict
comes from the paired test.

ONE ARM PER PROCESS
-------------------
The OOM happened because two 30 GB integrations shared one card. Each arm
therefore runs in its OWN process (``--arm``), launched sequentially by
``launch_cfm_pad_groups_mcnemar.sh``, so the previous arm's memory is
released by process exit rather than by a cleanup call that can be skipped.

Coordinates CANNOT be compared
------------------------------
padded draws randn(192, 72, 3); grouped draws randn(b, n_atoms, 3) per group.
Same seed, different shapes, unrelated samples. The 7.58 A max coordinate
difference in equiv2.txt measures the NOISE DRAW, not the batching. That
criterion is impossible to satisfy and its DIVERGES verdict is a false
negative; this script does not repeat it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

# 2026-10-12 FIX. This file lives in <repo>/scripts/, so parent.parent IS the
# repo root. The `.parent.parent.parent` used by modules inside
# <repo>/molmetal/scripts/ resolved to /home/hugo/codes, and every path built
# from it failed with FileNotFoundError -- both arms died in under a second and
# the aggregate then failed on a missing file, which is how a launcher can look
# like it ran an experiment when it never loaded a model.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c.

    ``b`` = grouped passed / padded failed, ``c`` = the reverse. Under H0
    each discordant pair is a fair coin, so the exact two-sided p is
    ``2 * P(X <= min(b, c))`` with ``X ~ Binom(b + c, 1/2)``, capped at 1.

    Computed as an EXACT RATIONAL and divided once at the end. An earlier
    version computed ``sum(comb(n,i)) / 2**n`` in float, which silently
    underflows to 0.0/0.0 = nan or rounds a genuinely tiny p to exactly 0 or
    1: with b=c=32, n=64, the float tail is smaller than the machine epsilon
    and the result came back as 1.0 -- i.e. a total disagreement between the
    arms reported as NO EVIDENCE, which is the most dangerous possible
    direction for this test to fail in. ``2**64`` exceeds float64's exact
    integer range, so the division must happen after the ratio is formed as a
    Fraction.
    """
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = Fraction(sum(comb(n, i) for i in range(m + 1)), 2 ** n)
    return min(1.0, float(2 * tail))


def paired_table(grouped: List[bool], padded: List[bool]) -> Dict[str, object]:
    """The 2x2 of paired outcomes, plus the exact test and Wilson intervals."""
    if len(grouped) != len(padded):
        raise ValueError(
            f"unpaired vectors: grouped has {len(grouped)} entries, padded "
            f"{len(padded)}. McNemar is only defined on matched pairs, and a "
            "length mismatch means one vector lost or gained a cloud -- which "
            "is exactly the alignment bug per_cloud_out exists to prevent."
        )
    both = sum(1 for g, p in zip(grouped, padded) if g and p)
    only_g = sum(1 for g, p in zip(grouped, padded) if g and not p)
    only_p = sum(1 for g, p in zip(grouped, padded) if not g and p)
    neither = sum(1 for g, p in zip(grouped, padded) if not g and not p)
    n_disc = only_g + only_p
    p_exact = mcnemar_exact_p(only_g, only_p)

    def wilson(k: int, n: int) -> Dict[str, Optional[float]]:
        if n == 0:
            return {"n": 0, "mean": None, "lo": None, "hi": None}
        import math

        phat = k / n
        z = 1.959963984540054
        denom = 1 + z * z / n
        centre = (phat + z * z / (2 * n)) / denom
        half = (
            z
            * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
            / denom
        )
        return {
            "n": n,
            "mean": round(phat, 6),
            "lo": round(max(0.0, centre - half), 6),
            "hi": round(min(1.0, centre + half), 6),
        }

    ng, npd = len(grouped), len(padded)
    kg = sum(1 for x in grouped if x)
    kp = sum(1 for x in padded if x)
    return {
        "n_clouds": ng,
        "contingency": {
            "both_pass": both,
            "grouped_only": only_g,
            "padded_only": only_p,
            "neither": neither,
        },
        "mcnemar": {
            "n_discordant": n_disc,
            "net": only_g - only_p,
            "exact_two_sided_p": p_exact,
            "verdict": (
                "CLEAR" if p_exact < 0.05 and n_disc > 0
                else ("SUB-FLOOR (no discordant pairs)" if n_disc == 0
                      else "SUB-FLOOR (p >= 0.05)")
            ),
        },
        "wilson_independently": {
            "grouped": wilson(kg, ng),
            "padded": wilson(kp, npd),
        },
        "interpretation": (
            "McNemar is the licensed test here: the arms share clouds AND the "
            "noise seed, so the difference is within-cloud flips. The Wilson "
            "intervals are shown for reference only and are strictly looser "
            "because they discard the pairing."
        ),
    }


def run_arm(arm: str, seed: int, n_clouds: int, device: str, out: Path) -> int:
    """Run ONE arm in this process and write its per-cloud vector."""
    import importlib.util

    import torch
    import yaml

    spec = importlib.util.spec_from_file_location(
        "cfm_trainer_arm", str(REPO / "molmetal" / "scripts" /
                               "train_cfm_yuelbond_wasserstein.py")
    )
    T = importlib.util.module_from_spec(spec)
    sys.modules["cfm_trainer_arm"] = T
    spec.loader.exec_module(T)

    from molmetal.data.tmqm_clouds import load_tmqm_clouds
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    ckpt = REPO / (
        "molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/"
        "cfm_yuelbond_wasserstein_seed0.pt"
    )
    st = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    a = st["args"]
    dev = torch.device(device)
    model = T.CFMJointModel(
        hidden_dim=int(a["hidden_dim"]),
        n_layers=int(a["n_layers"]),
        cutoff=float(a.get("cutoff", 3.5)),
        fm_objective=bool(a.get("use_fm_objective", False)),
        z_table=tuple(get_z_table(str(a.get("yuelbond_vocab", "pt")))),
        use_egnn_velocity=bool(a.get("use_egnn_velocity", False)),
        max_atomic_number=int(a.get("max_atomic_number", 100)),
        context_dropout=float(a.get("context_dropout", 0.1)),
        bond_predictor=str(a.get("bond_predictor", "yuelbond")),
    ).to(dev)
    model.load_state_dict(st["model_state_dict"])
    model.eval()
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

    cfg = yaml.safe_load(
        (REPO / "configs" / "cfm_tmqm_all6.yaml").read_text(encoding="utf-8")
    )
    clouds = load_tmqm_clouds(
        cfg["data"]["tmqm_dir"],
        metals=[s.strip() for s in cfg["data"]["tmqm_metals"].split(",")],
        coord_number=None, n_clouds=n_clouds, seed=0, split="val",
        val_frac=float(cfg["data"]["tmqm_val_frac"]),
    )
    print(f"[arm {arm}] pad_groups={arm == 'padded'} clouds={len(clouds)} "
          f"seed={seed}", flush=True)

    per_cloud: List[bool] = []
    t0 = time.time()
    n_ok, n_total = T.evaluate_generation_validity(
        model, dec, clouds, dev, method="midpoint", n_steps=50, seed=seed,
        n_samples=len(clouds), per_cloud_out=per_cloud,
        pad_groups=(arm == "padded"),
    )
    wall = time.time() - t0
    payload = {
        "arm": arm,
        "pad_groups": arm == "padded",
        "seed": seed,
        "n_clouds": len(clouds),
        "n_ok": n_ok,
        "n_total": n_total,
        "decode_rate": n_ok / max(n_total, 1),
        "wall_seconds": round(wall, 2),
        # The vector is the POINT: it is what makes the arms pairable.
        "per_cloud": per_cloud,
        "per_cloud_len": len(per_cloud),
        "alignment_check": (
            "len(per_cloud) == n_clouds" if len(per_cloud) == len(clouds)
            else f"MISALIGNED: {len(per_cloud)} vs {len(clouds)}"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[arm {arm}] n_ok={n_ok}/{n_total} decode_rate={n_ok/max(n_total,1):.4f} "
          f"wall={wall:.1f}s vector_len={len(per_cloud)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("grouped", "padded"), default=None,
                    help="run ONE arm and exit (the launcher runs two, in "
                         "separate processes, so the first one's VRAM is "
                         "released by process exit rather than a cleanup call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clouds", type=int, default=192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--aggregate", action="store_true",
                    help="read both arm files and emit the McNemar verdict")
    a = ap.parse_args()

    out_dir = REPO / "molmetal" / "reports" / "cfm_pad_groups_mcnemar"

    if a.aggregate:
        g = json.loads((out_dir / "arm_grouped.json").read_text(encoding="utf-8"))
        p = json.loads((out_dir / "arm_padded.json").read_text(encoding="utf-8"))
        table = paired_table(g["per_cloud"], p["per_cloud"])
        speedup = (
            g["wall_seconds"] / p["wall_seconds"]
            if p["wall_seconds"] else None
        )
        verdict = {
            "metric": "cfm_pad_groups_mcnemar",
            "tag": "MEASURED",
            "date": "2026-10-12",
            "question": (
                "Does pad_groups change the GENERATED RESULT, or only the "
                "wall time? Paired: same clouds, same noise seed, only the "
                "batching differs."
            ),
            "speedup_measured_x": round(speedup, 2) if speedup else None,
            "wall_seconds": {
                "grouped": g["wall_seconds"], "padded": p["wall_seconds"],
            },
            "decode_rate": {
                "grouped": g["decode_rate"], "padded": p["decode_rate"],
            },
            "paired_test": table,
            "verdict": (
                "IDENTICAL" if table["mcnemar"]["n_discordant"] == 0
                else (
                    "DIFFERS -- padding changes the result"
                    if table["mcnemar"]["verdict"] == "CLEAR"
                    else "NO CLEAR DIFFERENCE -- the discordant flips do not "
                         "clear the exact test"
                )
            ),
            "what_this_does_not_establish": [
                "One seed. The discordant count is the evidence; a second "
                "seed would bound the between-seed spread.",
                "Coordinates are NOT compared and cannot be: the two paths "
                "draw differently-shaped noise, so the 7.58 A max difference "
                "in equiv2.txt measures the noise draw, not the batching.",
                "This tests the GENERATION path only. It says nothing about "
                "whether a trained model under pad_groups would train the "
                "same, which is the question that matters for deployment.",
            ],
            "coordinate_verdict_status": (
                "FALSE NEGATIVE -- mis-specified criterion, superseded"
            ),
        }
        dest = REPO / "metrics" / "by_metric" / "cfm_pad_groups_mcnemar.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(verdict, indent=2))
        print(f"\nwrote {dest}")
        return 0

    if a.arm is None:
        ap.error("pass --arm {grouped,padded} or --aggregate")
    out = a.out or (out_dir / f"arm_{a.arm}.json")
    return run_arm(a.arm, a.seed, a.n_clouds, a.device, out)


if __name__ == "__main__":
    raise SystemExit(main())