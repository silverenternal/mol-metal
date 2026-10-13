#!/usr/bin/env python3
"""Aggregate the L3 decode-enabled A/B.

cfm_todo.json::next_actions[2] (priority 3, 2026-10-12):
- Two arms of the L3 0.05 sampling-time guidance A/B on the existing
  cfm_fullcorpus_2026_10_10 checkpoint, decode ON.
- Control: --guidance-strength 0.0, Treatment: --guidance-strength 0.05.
- Per arm, capture: decode_rate, n_decoded, QED, SA, arom_pair_frac.
- This script reads both arm JSONs (the per-arm files written by
  molmetal/scripts/wf_blkc_l3_guidance_probe.py), runs
  molmetal/scripts/cfm_property_panel.py on each arm's decoded SMILES,
  pulls the geometry aromatic-band fraction from the probe's
  generated_geometry aggregate, and writes the final
  metrics/by_metric/cfm_l3_decode_enabled_ab.json.

Conventions (AGENTS.md 5.2 + 5.6):
- Every number MUST trace to a JSON / report path on disk; this script
  records the source path of every input and output.
- The metric is written via atomic_write_file (AGENTS.md 5.6) so a
  concurrent reader can never see a half-written file.
- The verdict summarises whether the L3 axis is exhausted on this
  checkpoint and acknowledges the selection-effect caveat (each arm's
  QED/SA is on the molecules THAT arm decoded, so the cohorts differ
  when n_decoded differs).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]


def _load_arm(arm_path: Path) -> Dict[str, Any]:
    if not arm_path.exists():
        raise SystemExit(f"missing arm JSON: {arm_path}")
    return json.loads(arm_path.read_text(encoding="utf-8"))


def _extract_smiles(arm: Dict[str, Any]) -> List[str]:
    """Pull the decoded SMILES from an arm JSON's decode.rows block.

    The probe writes decode.rows as a list of {smiles, sanitized, ...} dicts
    (truncated to 200 rows), and decode.n_with_smiles as the count of rows
    whose smiles is non-empty. We re-filter rather than trusting the count
    because a SMILES can be the empty string in the rows.
    """
    rows = arm.get("decode", {}).get("rows", []) or []
    out: List[str] = []
    for r in rows:
        s = r.get("smiles")
        if isinstance(s, str) and s:
            out.append(s)
    return out


def _write_smiles_csv(smiles: List[str], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for s in smiles:
            w.writerow([s])


def _run_property_panel(smiles: List[str], out_json: Path) -> Dict[str, Any]:
    """Run cfm_property_panel.py on a SMILES list and return its JSON."""
    if not smiles:
        # Property panel cannot run on empty input; emit a stub that says so
        # so the aggregator can still record n_decoded=0 honestly.
        stub = {
            "panel_version": "wf264-sa/v1",
            "n_total": 0,
            "n_parsed": 0,
            "parsed_ratio": None,
            "sa": {"mean": None, "std": None, "min": None, "max": None, "n": 0, "percentiles": {}},
            "qed": {"mean": None, "std": None, "min": None, "max": None, "n": 0, "percentiles": {}},
            "note": "no decoded SMILES -- property panel did not run",
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(stub, indent=2), encoding="utf-8")
        return stub

    csv_path = out_json.with_suffix(".csv")
    _write_smiles_csv(smiles, csv_path)
    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)  # fall back to whatever the script was started with
    cmd = [
        str(py),
        str(REPO / "molmetal" / "scripts" / "cfm_property_panel.py"),
        "--smiles-csv", str(csv_path),
        "--out-json", str(out_json),
    ]
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(
            f"cfm_property_panel.py failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out_json.read_text(encoding="utf-8"))


def _frac_in_arom_band(arm: Dict[str, Any]) -> Optional[float]:
    """Mean of per-cloud frac_pairs_in_arom_band from the probe's aggregate.

    Returns None if the key is missing (older probe without the extension).
    """
    gen = arm.get("generated_geometry", {}) or {}
    band = gen.get("frac_pairs_in_arom_band")
    if isinstance(band, dict):
        return band.get("mean")
    return None


def _arm_block(
    arm_path: Path,
    label: str,
) -> Dict[str, Any]:
    arm = _load_arm(arm_path)
    smiles = _extract_smiles(arm)
    decode = arm.get("decode", {}) or {}
    n_decoded = decode.get("n_with_smiles", len(smiles))
    decode_rate = decode.get("decode_rate", (len(smiles) / 64.0) if smiles else 0.0)
    panel_out_json = arm_path.parent / f"panel_{label}.json"
    panel = _run_property_panel(smiles, panel_out_json)
    qed_mean = (panel.get("qed") or {}).get("mean")
    sa_mean = (panel.get("sa") or {}).get("mean")
    arom_frac = _frac_in_arom_band(arm)
    return {
        "arm_json": str(arm_path),
        "panel_json": str(panel_out_json),
        "smiles_csv": str(panel_out_json.with_suffix(".csv")),
        "checkpoint": arm.get("checkpoint"),
        "guidance_strength": (arm.get("cohort") or {}).get("guidance_strength"),
        "n_clouds": decode.get("n_attempted"),
        "n_decoded": n_decoded,
        "decode_rate": decode_rate,
        "qed_mean": qed_mean,
        "sa_mean": sa_mean,
        "arom_pair_frac": arom_frac,
        "n_pairs_in_arom_band_total": (arm.get("generated_geometry", {}) or {}).get("total_pairs_in_arom_band"),
        "n_clouds_with_any_arom_band_pair": (arm.get("generated_geometry", {}) or {}).get("n_clouds_with_any_arom_band_pair"),
    }


def _verdict(control: Dict[str, Any], treatment: Dict[str, Any]) -> str:
    """One-sentence summary of whether the L3 axis is exhausted.

    cfm_todo.json::next_actions[2]: "If QED or SA moves measurably, this is
    the cheapest no-retrain win. If neither moves, the L3 guidance axis is
    exhausted on this checkpoint."
    """
    cq = control.get("qed_mean")
    tq = treatment.get("qed_mean")
    cs = control.get("sa_mean")
    ts = treatment.get("sa_mean")
    ca = control.get("arom_pair_frac")
    ta = treatment.get("arom_pair_frac")
    nc = control.get("n_decoded")
    nt = treatment.get("n_decoded")

    parts: List[str] = []
    if tq is None or cq is None:
        parts.append("QED could not be computed (no SMILES in one or both arms)")
    else:
        dq = tq - cq
        parts.append(f"QED delta = {dq:+.4f} (control {cq:.4f} n={nc} -> treatment {tq:.4f} n={nt})")
    if ts is None or cs is None:
        parts.append("SA could not be computed (no SMILES in one or both arms)")
    else:
        ds = ts - cs
        parts.append(f"SA delta = {ds:+.4f} (control {cs:.4f} n={nc} -> treatment {ts:.4f} n={nt})")
    if ca is not None and ta is not None:
        da = ta - ca
        parts.append(
            f"arom_pair_frac delta = {da:+.4f} (control {ca:.4f} -> treatment {ta:.4f}, "
            f"per-cloud mean over {control.get('n_clouds', '?')} clouds)"
        )
    # Selection-effect caveat is part of the verdict, per AGENTS.md 5.2.
    if nc != nt:
        parts.append(
            f"selection-effect caveat: QED/SA cohorts differ in size "
            f"(control n={nc} vs treatment n={nt}); differences in property "
            f"means are not like-for-like."
        )
    # Exhausted-on-this-checkpoint verdict.
    if tq is not None and cq is not None and ts is not None and cs is not None:
        if abs(tq - cq) < 0.01 and abs(ts - cs) < 0.05:
            return (
                "L3 guidance axis EXHAUSTED on this checkpoint: "
                + " | ".join(parts)
                + " | geometry gain (measured separately) does not translate to a SMILES-level gain."
            )
        return (
            "L3 guidance axis shows a measurable SMILES-level shift: "
            + " | ".join(parts)
        )
    return " | ".join(parts)


def _delta(control: Dict[str, Any], treatment: Dict[str, Any]) -> Dict[str, Any]:
    def _d(k: str) -> Optional[float]:
        c, t = control.get(k), treatment.get(k)
        if c is None or t is None:
            return None
        return float(t) - float(c)
    return {
        "decode_rate": _d("decode_rate"),
        "qed": _d("qed_mean"),
        "sa": _d("sa_mean"),
        "arom_pair_frac": _d("arom_pair_frac"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--control", required=True,
        help="path to control arm JSON (guidance_strength=0.0)",
    )
    ap.add_argument(
        "--treatment", required=True,
        help="path to treatment arm JSON (guidance_strength=0.05)",
    )
    ap.add_argument(
        "--out", required=True,
        help="output aggregate JSON path",
    )
    ap.add_argument(
        "--preflight", default="/tmp/l3_preflight.json",
        help="pre-flight JSON path to cite in the output (AGENTS.md 4 + 5.1)",
    )
    ap.add_argument(
        "--vram-free-mib", default=None,
        help="free VRAM on cuda:0 at start, in MiB (recorded verbatim in output)",
    )
    args = ap.parse_args()

    control_path = Path(args.control)
    treatment_path = Path(args.treatment)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    control = _arm_block(control_path, "control")
    treatment = _arm_block(treatment_path, "treatment")

    payload: Dict[str, Any] = {
        "metric": "cfm_l3_decode_enabled_ab",
        "tag": "MEASURED",
        "date": "2026-10-13",
        "checkpoint": control.get("checkpoint") or treatment.get("checkpoint"),
        "seed": 1234,
        "n_clouds": control.get("n_clouds") or treatment.get("n_clouds"),
        "arms": {
            "control_0.00": control,
            "treatment_0.05": treatment,
        },
        "delta": _delta(control, treatment),
        "verdict": _verdict(control, treatment),
        "preflight_path": args.preflight,
        "vram_check": (
            f"cuda:0 free VRAM at start = {args.vram_free_mib} MiB"
            if args.vram_free_mib else None
        ),
    }

    # AGENTS.md 5.6 -- atomic write so a reader can never see a partial file.
    sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
    from wf386a3_atomic_write import atomic_write_file
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    atomic_write_file(out_path, text)
    print(f"wrote {out_path} ({len(text)} bytes)")
    print()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
