# SPDX-License-Identifier: Apache-2.0
"""Wf-A103 (2026-10-20) - calibrate the ShortcutDistillLoss perturbation scale.

Wf-A102 wired the second view with a hard-coded 0.01 A perturbation and
MEASURED the resulting term at 1.24e-08 against a bond_ce of 1.68 - live, but
three orders of magnitude below observability.  The constant was never
calibrated, so "0.01 is too small" was an inference, not a measurement.

This script measures the curve.  For each sigma it runs the same 1-epoch
32-cloud CPU smoke the liveness guard uses (same flags, same cohort), with
--shortcut-distill-term-weight 0.05, and reads the term out of the trainer's
own _LAST_LOSS_TERMS side-channel.

Why a curve instead of a "better default".  The term is a mean squared
difference between two velocity fields; for a smooth field it grows like
sigma^2, so any single number can be made to look "observable" by picking a
large enough sigma.  What matters is where the growth stops describing
self-consistency and starts describing robustness to a different input - a
perturbation comparable to the CFM path's own noise_sigma = 1.0 A is no
longer the same objective.

Output: metrics/by_metric/wfa103_shortcut_sigma_sweep.json.

Usage
-----
    .venv/bin/python molmetal/scripts/wfa103_shortcut_sigma_sweep.py
    .venv/bin/python molmetal/scripts/wfa103_shortcut_sigma_sweep.py --sigmas 0.01 0.1 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
OUT = REPO / "metrics" / "by_metric" / "wfa103_shortcut_sigma_sweep.json"

#: The same cohort and budget the liveness guard uses, so the numbers here are
#: comparable with the 1.24e-08 that Wf-A102 measured.
SMOKE_ARGS: List[str] = [
    "--device", "cpu",
    "--n-train", "32",
    "--n-val", "16",
    "--batch-size", "8",
    "--use-fm-objective",
    "--resume-from", "none",
]

#: Wf-A102's measurement point plus a ladder up to the CFM noise scale.
DEFAULT_SIGMAS: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

_PROBE = """
import sys, json, importlib.util
sys.argv = ["train_cfm_yuelbond_wasserstein.py"] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("_t", {trainer!r})
m = importlib.util.module_from_spec(spec); sys.modules["_t"] = m
spec.loader.exec_module(m)
m.main()
# Wf-A105 moved the shortcut class term out of _LAST_LOSS_TERMS: it is
# computed but not summed into the loss.  Read the diagnostic dict, and merge
# it so the lookup below keeps working; bond_ce stays in the loss dict.
out = {{k: {{"requires_grad": bool(v.requires_grad),
            "value": float(v.detach())}}
       for k, v in m._LAST_LOSS_TERMS.items()}}
out.update({{k: {{"requires_grad": bool(v.requires_grad),
                 "value": float(v.detach())}}
            for k, v in m._LAST_DIAGNOSTIC_TERMS.items()}})
print("LOSS_TERMS_JSON=" + json.dumps(out))
"""


def run_one(sigma: float, workdir: Path, epochs: int = 1) -> Dict[str, Any]:
    """One smoke at (sigma, epochs); returns the two terms of interest."""
    tag = "s" + repr(float(sigma)) + "_e" + str(int(epochs))
    probe = workdir / ("_probe_" + tag + ".py")
    probe.write_text(_PROBE.format(trainer=str(TRAINER)), encoding="utf-8")
    cmd = [
        sys.executable,
        str(probe),
        *SMOKE_ARGS,
        "--epochs", str(int(epochs)),
        "--shortcut-distill-term-weight", "0.05",
        "--shortcut-distill-perturb-sigma", repr(float(sigma)),
        "--out-dir", str(workdir / ("run_" + tag)),
        "--ckpt-dir", str(workdir / ("ck_" + tag)),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO), timeout=900
    )
    marker = "LOSS_TERMS_JSON="
    for line in reversed(proc.stdout.splitlines()):
        if marker in line:
            terms = json.loads(line.split(marker, 1)[1])
            sd = terms.get("shortcut_distill_class_term", {})
            bc = terms.get("bond_ce", {})
            return {
                "sigma": float(sigma),
                "shortcut_distill_class_term": sd.get("value"),
                "shortcut_requires_grad": sd.get("requires_grad"),
                "bond_ce": bc.get("value"),
                "ratio_to_bond_ce": (
                    sd.get("value") / bc.get("value")
                    if sd.get("value") is not None and bc.get("value")
                    else None
                ),
            }
    raise RuntimeError(
        "trainer did not emit LOSS_TERMS_JSON at sigma=" + repr(sigma)
        + " rc=" + str(proc.returncode)
        + " STDOUT tail: " + proc.stdout[-2000:]
        + " STDERR tail: " + proc.stderr[-2000:]
    )


def derive_sigma_rows(rows):
    """Add the sigma-normalised column and summarise it.

    The term is a mean squared difference of two velocity fields evaluated at
    inputs sigma apart, so for a smooth field it is C * sigma^2.  Dividing by
    sigma^2 leaves C = E[||grad v||^2] - a property of the FIELD, not of the
    probe.  That is the scale-free statistic the Wf-A102 follow-up asked for:
    if C is constant across the ladder, sigma is a free parameter of the probe
    and the weight becomes interpretable.
    """
    for r in rows:
        s = r.get("sigma")
        v = r.get("shortcut_distill_class_term")
        r["normalized_by_sigma2"] = (v / (s * s)) if (v is not None and s) else None

    vals = [r["normalized_by_sigma2"] for r in rows if r.get("normalized_by_sigma2")]
    summary = {"n": len(vals)}
    if vals:
        lo, hi = min(vals), max(vals)
        summary.update({
            "min": lo,
            "max": hi,
            "spread_over_min": (hi - lo) / lo if lo else None,
            "mean": sum(vals) / len(vals),
            "interpretation": (
                "E[||grad v||^2] on this cohort. A spread close to 0 across "
                "the ladder means sigma is a free parameter of the probe and "
                "the term's magnitude is a field property."
            ),
        })
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="epochs for the sigma sweep (the liveness guard uses 1)",
    )
    ap.add_argument(
        "--ladder-sigma",
        type=float,
        default=0.1,
        help="fixed sigma for the epoch ladder",
    )
    ap.add_argument(
        "--ladder-epochs", type=int, nargs="+", default=[1, 2, 4, 8],
        help="epoch ladder at --ladder-sigma; tests whether the field's "
             "smoothness is a 1-epoch artefact",
    )
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument(
        "--derive-only",
        action="store_true",
        help="recompute only the derived columns from the existing "
             "artefact (they are a pure function of rows); used so a "
             "15-min sweep need not be repeated to add a column",
    )
    args = ap.parse_args()

    if args.derive_only:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        payload = json.loads(OUT.read_text(encoding="utf-8"))
        payload["sigma2_normalised"] = derive_sigma_rows(payload["rows"])
        atomic_write_file(str(OUT), json.dumps(payload, indent=2))
        print("rewrote " + str(OUT) + " (derived only)")
        return 0

    rows: List[Dict[str, Any]] = []
    ladder: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="wfa103_") as td:
        workdir = Path(td)
        for sigma in args.sigmas:
            row = run_one(sigma, workdir, args.epochs)
            rows.append(row)
            print(
                "[wfa103] sigma=" + str(row["sigma"])
                + " term=" + format(row["shortcut_distill_class_term"], ".3e")
                + " bond_ce=" + format(row["bond_ce"], ".4f")
                + " ratio=" + format(row["ratio_to_bond_ce"], ".3e")
                + " grad=" + str(row["shortcut_requires_grad"]),
                flush=True,
            )

        if not args.skip_ladder:
            for ep in args.ladder_epochs:
                row = run_one(args.ladder_sigma, workdir, ep)
                row["epochs"] = int(ep)
                ladder.append(row)
                print(
                    "[wfa103] ladder epochs=" + str(ep)
                    + " term=" + format(row["shortcut_distill_class_term"], ".3e")
                    + " bond_ce=" + format(row["bond_ce"], ".4f")
                    + " grad=" + str(row["shortcut_requires_grad"]),
                    flush=True,
                )

    # Growth exponent: d log(term) / d log(sigma).  A smooth field gives ~2.
    exponents = []
    for a, b in zip(rows, rows[1:]):
        if (
            a["shortcut_distill_class_term"]
            and b["shortcut_distill_class_term"]
            and a["sigma"] > 0
            and b["sigma"] > 0
        ):
            exponents.append({
                "from": a["sigma"],
                "to": b["sigma"],
                "exponent": math.log(
                    b["shortcut_distill_class_term"]
                    / a["shortcut_distill_class_term"]
                ) / math.log(b["sigma"] / a["sigma"]),
            })

    sigma_norm = derive_sigma_rows(rows)

    payload = {
        "metric": "wfa103_shortcut_sigma_sweep",
        "purpose": (
            "Calibrate the ShortcutDistillLoss second-view perturbation scale. "
            "The term is a mean squared difference of two velocity fields, so "
            "its size is a property of the FIELD, not of the weight."
        ),
        "probe": {
            "cohort": "same 32-cloud / 1-epoch CPU smoke as test_cfm_loss_gradient_liveness",
            "flags": SMOKE_ARGS + ["--shortcut-distill-term-weight 0.05"],
            "read_from": (
                "trainer _LAST_DIAGNOSTIC_TERMS (the shortcut term is computed "
                "but NOT summed into the loss since Wf-A34-1; Wf-A105 made that "
                "explicit, see HONEST_NEGATIVE_CATALOG.md §6.148). bond_ce is "
                "read from _LAST_LOSS_TERMS."
            ),
        },
        "rows": rows,
        "growth_exponent": exponents,
        "sigma2_normalised": sigma_norm,
        "epoch_ladder": {
            "sigma": args.ladder_sigma,
            "rows": ladder,
            "purpose": (
                "The sigma sweep is run on a 1-epoch model, whose velocity "
                "field is barely trained. If the term stays this small as "
                "training proceeds, the smoothness is a property of the "
                "architecture; if it grows, the sigma-sweep magnitude is a "
                "lower bound only valid at initialisation."
            ),
        },
        "MEASURED_vs_PROJECTED": "MEASURED",
        "not_established": (
            "No claim that a larger sigma helps quality. This measures the "
            "term magnitude only; whether the smoothness penalty helps or "
            "hurts bond_type_f1 / gen_validity is unmeasured and needs a GPU A/B."
        ),
    }
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(str(OUT), json.dumps(payload, indent=2))
    print("wrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())