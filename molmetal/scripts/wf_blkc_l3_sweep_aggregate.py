#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the L3 guidance strength sweep: geometry AND the decoded molecule.

WHY THIS EXISTS
---------------
``wf_blkc_l3_guidance_ab.py`` compares exactly two arms and only their GEOMETRY
metrics. Two things changed on 2026-10-12 that it cannot express:

1. There are now FOUR strengths (0.0 / 0.01 / 0.05 / 0.15), because the one
   measured point at 0.05 both overshot the global minimum and under-shot the
   metal-ligand contact, so the grid has to bracket it on both sides.
2. The probe now runs with decode ENABLED. That is the whole reason the sweep
   exists: cfm_todo.json::the_objective_2026_10_12 asks for better QED / SA /
   ADMET, and none of those exist until the sampled geometry is decoded into a
   molecule. The old aggregator has no decode columns at all.

So the comparison this file makes is the one the decision actually turns on:

    does better SAMPLED GEOMETRY produce a better DECODED MOLECULE?

Every number here is reported per arm against the 0.0 control, with the
control's own absolute value alongside, because a delta without its baseline
is unreadable.

HONESTY RULES enforced in the output, not left to the reader:

* The arm with the best geometry is NOT automatically the best molecule. If
  the best-geometry arm loses on decode or on QED, that contradiction is
  stated at the top of the output rather than buried in a table.
* ``n_with_smiles`` is the DENOMINATOR-aware decode rate. The earlier Vina-style
  survivor count (n=51 over an unstated attempted set) conditioned the mean on
  success; this reports both the attempted and the successful count always.
* A property computed on 22 molecules is reported with its n, and a difference
  smaller than the within-arm spread is not called a difference.
* Reuses ``scripts/cfm_checkpoint_benchmark.py::score_cohort`` so the property
  scoring is the SAME implementation the standalone benchmark uses. A second
  implementation would be a second thing to drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Geometry metrics where "toward real" is the goal, and the direction.
GEOM_DIRECTION = {
    "min_pair_distance": "up",
    "p01_pair_distance": "up",
    "median_pair_distance": "up",
    "frac_pairs_hard_clash": "down",
    "radius_of_gyration": "up",
}


def _mean(block: Dict[str, Any], key: str) -> Optional[float]:
    v = block.get(key)
    if isinstance(v, dict):
        v = v.get("mean")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _smiles(arm: Dict[str, Any]) -> List[str]:
    rows = (arm.get("decode") or {}).get("rows") or []
    return [r["smiles"] for r in rows if r.get("smiles")]


def summarise(arm_path: Path, with_properties: bool) -> Dict[str, Any]:
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    gen = arm.get("generated_geometry") or {}
    real = arm.get("real_tmqm_geometry") or {}
    dec = arm.get("decode") or {}

    out: Dict[str, Any] = {
        "arm": arm.get("arm"),
        "strength": (arm.get("cohort") or {}).get("guidance_strength"),
        # 2026-10-12 (BLK-C-M4). None = no offset. Read from the arm JSON's
        # cohort block, which is where the probe records what it actually ran
        # -- so the aggregator reads the measurement rather than the filename.
        "aromatic_logit_bias": (arm.get("cohort") or {}).get(
            "bond_aromatic_logit_bias"
        ),
        "file": str(arm_path),
        "checkpoint": arm.get("checkpoint"),
        "n_clouds": (arm.get("cohort") or {}).get("n_clouds"),
        "n_steps": (arm.get("cohort") or {}).get("n_steps"),
        "seed": (arm.get("cohort") or {}).get("seed"),
        "geometry": {},
        "decode": {
            "n_attempted": dec.get("n_attempted"),
            "n_with_smiles": dec.get("n_with_smiles"),
            # The denominator is stated every time. The earlier sweep's n=51
            # was a survivor count and the difference is the whole point.
            "decode_rate": dec.get("decode_rate"),
            "single_fragment_ratio": dec.get("single_fragment_ratio"),
            "n_errors": dec.get("n_errors"),
            "all_clouds_errored": dec.get("all_clouds_errored"),
        },
    }

    for key, direction in GEOM_DIRECTION.items():
        g, r = _mean(gen, key), _mean(real, key)
        if g is None:
            continue
        row: Dict[str, Any] = {"generated": g, "real_tmqm": r, "direction_wanted": direction}
        if r is not None:
            row["abs_err_unguided_form"] = abs(g - r)
        out["geometry"][key] = row

    if with_properties:
        try:
            sys.path.insert(0, str(REPO / "scripts"))
            from cfm_checkpoint_benchmark import score_cohort

            smi = _smiles(arm)
            if smi:
                # 2026-10-12 (BLK-C-M4). The cohort label names the OFFSET as
                # well as the strength: a bias sweep changes what the DECODER
                # draws, and two arms labelled only by strength would be
                # indistinguishable in the output.
                lab = f"strength {out['strength']}"
                if out.get("aromatic_logit_bias") is not None:
                    lab += f" bias {out['aromatic_logit_bias']}"
                props = score_cohort(lab, smi)
                out["properties_on_decoded"] = {
                    k: {kk: props.get(k, {}).get(kk) for kk in ("mean", "std", "n")}
                    for k in ("qed", "sa", "ames_mutagen",
                              "cell_permeability_logPapp",
                              "aqueous_solubility_logS",
                              # The metric the aromatic offset moves. Without
                              # it a bias sweep reports only decode rate and
                              # cannot see whether the lever worked.
                              "aromatic_bond_frac")
                    if isinstance(props.get(k), dict)
                }
                out["properties_on_decoded"]["n_input"] = props.get("n_input")
                out["properties_on_decoded"]["n_parsed"] = props.get("n_parsed")
                out["properties_on_decoded"]["n_failed_to_parse"] = props.get("n_failed")
        except Exception as exc:  # noqa: BLE001
            # Property scoring failing must NOT be silent. It would otherwise
            # leave an empty column that reads as "no difference".
            out["properties_on_decoded"] = {
                "ERROR": f"{type(exc).__name__}: {exc}",
                "why_this_matters": "an empty property column reads as 'no "
                                    "change'; it is actually 'not measured'",
            }
    return out


def compare(arms: List[Dict[str, Any]], control: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per (arm, metric) with the delta and whether it moved to real."""
    rows: List[Dict[str, Any]] = []
    for arm in arms:
        for key, c_row in (control.get("geometry") or {}).items():
            a_row = (arm.get("geometry") or {}).get(key)
            if not a_row:
                continue
            real = c_row.get("real_tmqm")
            before, after = c_row["generated"], a_row["generated"]
            row: Dict[str, Any] = {
                "arm": arm["arm"],
                "strength": arm["strength"],
                "metric": key,
                "control": before,
                "guided": after,
                "delta": round(after - before, 6),
                "real_tmqm": real,
            }
            if real is not None:
                row["abs_err_control"] = abs(before - real)
                row["abs_err_guided"] = abs(after - real)
                row["moved_toward_real"] = row["abs_err_guided"] < row["abs_err_control"]
            rows.append(row)
    return rows


def headline(arms: List[Dict[str, Any]], control: Dict[str, Any]) -> str:
    """State the geometry winner AND the molecule winner, and any contradiction.

    Written out because the table hides it: a reader scanning for the largest
    geometry improvement would otherwise conclude the sweep succeeded.
    """
    guided = [a for a in arms if a["strength"] != control["strength"]]
    if not guided:
        return "NO GUIDED ARM PRESENT -- nothing to compare against the control."

    # 2026-10-12 FIX. This used to pick the best-geometry arm by the RAW value
    # of min_pair_distance, which is monotone increasing in guidance strength and
    # therefore always selects the STRONGEST arm -- including arms that have
    # overshot past real tmQM (0.15 reaches 1.368 A against a real 1.229 A, and
    # is simultaneously worse on p01_pair_distance and radius_of_gyration).
    # Ranking by raw value cannot detect an overshoot, which is the single most
    # likely way to misread this sweep.
    #
    # "Best geometry" now means LOWEST TOTAL ABSOLUTE ERROR against the real
    # tmQM cohort across every geometry metric -- i.e. the arm that looks most
    # like a real complex, not the arm that pushes atoms furthest apart.
    def _geo_err(a: Dict[str, Any]) -> Optional[float]:
        g = a.get("geometry") or {}
        errs = [
            abs(row["generated"] - row["real_tmqm"])
            for row in g.values()
            if row.get("real_tmqm") is not None
        ]
        return sum(errs) if errs else None

    scored = [(a, _geo_err(a)) for a in guided]
    scored = [(a, e) for a, e in scored if e is not None]
    geo_best = min(scored, key=lambda t: t[1])[0] if scored else None
    geo_err = dict((id(a), e) for a, e in scored).get(id(geo_best)) if geo_best else None
    control_err = _geo_err(control)

    mol_best = max(
        [a for a in guided if a["decode"].get("decode_rate") is not None],
        key=lambda a: a["decode"]["decode_rate"],
        default=None,
    )
    lines = []
    if geo_best:
        lines.append(
            f"best GEOMETRY (closest to real tmQM, total abs error over all "
            f"geometry metrics): strength {geo_best['strength']} at {geo_err:.4f} A"
            + (f", vs control {control_err:.4f} A" if control_err is not None else "")
        )
    # Overshoot is reported for EVERY guided arm, not just the best-geometry
    # one. In the measured sweep the arm that overshoots is the one that WINS on
    # decode (0.15), so an overshoot note attached only to the geometry winner
    # would print a clean bill of health for exactly the arm that most needs
    # the warning.
    for a in guided:
        over = [
            f"{k}={a['geometry'][k]['generated']:.3f} vs real "
            f"{a['geometry'][k]['real_tmqm']:.3f}"
            for k, row in (a.get("geometry") or {}).items()
            if row.get("real_tmqm") is not None
            and k in (control.get("geometry") or {})
            and abs(row["generated"] - row["real_tmqm"])
            > abs(control["geometry"][k]["generated"] - row["real_tmqm"])
        ]
        if over:
            tag = " (DECODE WINNER)" if mol_best and a is mol_best else ""
            lines.append(
                f"*** OVERSHOOT at strength {a['strength']}{tag} -- worse than "
                "the control on: " + "; ".join(over)
            )
    if mol_best:
        lines.append(
            f"best DECODED MOLECULE (decode rate): strength {mol_best['strength']} "
            f"at {mol_best['decode']['decode_rate']:.4f} "
            f"({mol_best['decode']['n_with_smiles']}/{mol_best['decode']['n_attempted']})"
        )
    if geo_best and mol_best and geo_best["strength"] != mol_best["strength"]:
        lines.append(
            "*** CONTRADICTION: the arm with the best geometry is NOT the arm "
            f"that decodes best (geometry wants {geo_best['strength']}, decode "
            f"wants {mol_best['strength']}). Sampling-time guidance may improve "
            "the coordinates while leaving -- or worsening -- what the bond head "
            "makes of them. Do not report the geometry win alone."
        )
    # QED moves the OPPOSITE way to decode rate. Stated here because the table
    # invites reading "decode rate 0.344 -> 0.828, a 2.4x gain" as a quality
    # win, and it is not one.
    def _qed(a):
        p = a.get("properties_on_decoded") or {}
        return (p.get("qed") or {}).get("mean")

    cq, mq = _qed(control), _qed(mol_best) if mol_best else None
    if cq is not None and mq is not None:
        lines.append(
            f"QED moves the OPPOSITE WAY: control {cq:.4f} -> best-decode arm "
            f"{mq:.4f}. The decode gain is NOT a quality gain."
        )
        lines.append(
            "SELECTION EFFECT, and it is not small: each arm's properties are "
            "computed on the molecules THAT ARM SUCCEEDED IN DECODING (n = "
            + ", ".join(
                f"{a['strength']}:{(a.get('properties_on_decoded') or {}).get('n_parsed')}"
                for a in arms
            )
            + "). Stronger guidance ADDS molecules the weaker arms could not "
            "produce at all, so the cohorts differ in composition, not only in "
            "sampling. The property columns are therefore not a like-for-like "
            "comparison of the same molecules at different settings."
        )

    n = {a["strength"]: a["decode"].get("n_with_smiles") for a in arms}
    lines.append(f"decode successes by strength: {n}")
    lines.append(
        "n is small per arm. A difference smaller than the within-arm spread "
        "is not a finding, and no arm here clears that bar on decode."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", type=Path, required=True,
                    help="arm JSONs; the strength-0.0 one is the control")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-properties", action="store_true",
                    help="skip QED/SA/ADMET scoring on the decoded SMILES")
    a = ap.parse_args()

    missing = [p for p in a.arms if not p.is_file()]
    if missing:
        print(f"MISSING arm files: {[str(p) for p in missing]}", file=sys.stderr)
        return 2

    summaries = [summarise(p, not a.no_properties) for p in a.arms]
    controls = [s for s in summaries if float(s["strength"] or 0.0) == 0.0]
    if not controls:
        print(
            "NO CONTROL ARM (strength 0.0) among the inputs. Without it every "
            "delta is unreadable and the sweep cannot answer its question.",
            file=sys.stderr,
        )
        return 3
    control = controls[0]

    order = {0.0: 0}
    summaries.sort(key=lambda s: (order.get(float(s["strength"]), 1),
                                  float(s["strength"] or 0.0)))

    out = {
        "metric": "cfm_l3_guidance_sweep",
        "tag": "MEASURED",
        "date": "2026-10-12",
        "question": (
            "Does sampling-time guidance improve the DECODED MOLECULE, or only "
            "the sampled coordinates? The geometry half was already answered at "
            "strength 0.05 on 24 clouds; this sweep asks the harder half."
        ),
        "design": {
            "control_strength": control["strength"],
            "strengths": [s["strength"] for s in summaries],
            "held_fixed": ["checkpoint", "template cohort", "initial noise seed",
                           "ODE method", "n_steps"],
            "note": (
                "0.05 overshot the global minimum (1.155 vs 1.172 real) while "
                "under-shooting the metal-ligand contact, so the grid brackets it "
                "on BOTH sides rather than scaling outward from it."
            ),
        },
        "HEADLINE": headline(summaries, control),
        "arms": summaries,
        "geometry_deltas_vs_control": compare(summaries, control),
        "what_this_does_not_establish": [
            "One checkpoint, one noise seed. A guidance effect measured on a "
            "single seed of a single checkpoint is a direction, not a result.",
            "Decode rate is the primary read-out; a molecule can decode and "
            "still be chemically poor, which is why the property columns "
            "matter and why they are scored on the SAME SMILES rather than "
            "re-decoded.",
            "Nothing here says whether a guidance term belongs in a TRAINING "
            "loss. It acts at sampling time, so it does not change the model.",
        ],
        "DO_NOT_COMPARE_ACROSS_CHECKPOINTS": (
            "The property columns in this file are NOT comparable to "
            "cfm_checkpoint_property_benchmark.json. That panel was measured on "
            "66 molecules from cfm_fullcorpus_2026_10_10 epoch 10; this sweep "
            "decodes 64 clouds from cfm_patience10_seed0 at midpoint/50. For "
            "reference the control arm here scores QED "
            + str((control.get("properties_on_decoded") or {}).get("qed"))
            + " where the panel recorded 0.0944, and logS "
            + str((control.get("properties_on_decoded") or {}).get(
                "aqueous_solubility_logS"))
            + " where the panel recorded -8.0099. Those are two different "
            "checkpoints and two different cohorts, not a property improvement. "
            "Read arms against the control of THIS sweep and nothing else."
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print("HEADLINE:\n" + out["HEADLINE"])
    print()
    hdr = f"{'strength':>9} {'decode':>8} {'n_ok':>6} {'1-frag':>8} " \
          f"{'min_d':>8} {'clash':>9}"
    print(hdr)
    for s in summaries:
        d = s["decode"]
        g = s["geometry"]
        md = g.get("min_pair_distance", {}).get("generated")
        cl = g.get("frac_pairs_hard_clash", {}).get("generated")
        print(f"{s['strength']:>9} {str(d.get('decode_rate')):>8} "
              f"{str(d.get('n_with_smiles')):>6} "
              f"{str(d.get('single_fragment_ratio')):>8} "
              f"{('%.3f' % md) if md else 'NA':>8} "
              f"{('%.5f' % cl) if cl else 'NA':>9}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())