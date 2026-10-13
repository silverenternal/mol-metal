# SPDX-License-Identifier: Apache-2.0
"""Re-run the 10-pocket Vina sweep over the generated cohort, with a DENOMINATOR.

WHY THIS SCRIPT EXISTS
======================
The 2026-10-12 sweep (``vina_gen2.txt``) died in pocket 2 of 10 with no
traceback. Nine of ten pockets are NOT ON DISK, so no generated-vs-control
docking statement exists. The boron atype bug that plausibly caused the hard
failures is fixed (21 of 44 tmQM elements accepted by Vina 1.2.7, 23 rejected;
``metrics/by_metric/vina_127_element_atypes.json``) and the surrogate
mislabelling is fixed (``MEASURED_SURROGATE_ATOM``), so the sweep is runnable.

WHAT THE DEAD SWEEP GOT WRONG STATISTICALLY
===========================================
It printed ``generated 1fmc: n=51 mean=-6.407`` and that ``n`` is a SURVIVOR
count over an unstated attempted set. The molecules that failed were precisely
the ones that failed to become a PDBQT or to parse, so the mean is conditioned
on success — and the failures are not a random sample of the cohort, they are
selected by the embeddability of the molecule. A mean over survivors is a mean
over a different population than the cohort it is quoted for.

This script therefore records, per pocket:

* ``n_attempted``  — every SMILES handed to the pipeline, unconditionally;
* ``n_docked``     — how many produced a finite kcal/mol;
* ``errors_by_type``— failures split by the exception TYPE, so
  ``MetalPDBQTUnavailable`` (RDKit embed) is never pooled with
  ``VinaAtomTypeRejected`` (a harness defect) or ``DockError``.

and the aggregate mean is reported alongside the attempted denominator, with
the survivor-conditioned mean explicitly labelled as such.

HONEST FRAMING
==============
* kcal/mol is real Vina 1.2.7 output. It is NOT an affinity measurement for a
  metal complex: the six train metals (Pt Ru Ir Rh Os Re) are all in Vina's
  rejected-element set, so every one of them is scored as ``Mg`` (metal branch)
  or as a covalent surrogate (B->C, As->P), carrying ``MEASURED_GENERIC_ATOM`` or
  ``MEASURED_SURROGATE_ATOM``. The number describes the stand-in, not the metal.
* The control is NOT like-for-like. Cisplatin is a 4-heavy-atom complex; the
  generated cohort is ~40. Vina scores larger ligands more favourably, so any
  generated-vs-cisplatin difference is dominated by SIZE. ``n_atoms_ligand`` is
  recorded per record so this is checkable rather than asserted.
* The defensible use of this sweep is a pose-feasibility filter, not affinity
  ranking. Cinaroglu (JCIM 2019) measured 73% pose success over 213
  metal-protein complexes with ~0 scoring power (r^2); a 2026 Springer review
  states Vina "systematically underestimates binding energies and generates
  wrong poses" for metal ligands and accepts no custom metal parameters.

Usage
-----
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/cfm_vina_generated_sweep.py \\
        --out-dir molmetal/reports/cfm_vina_sweep_2026_10_12
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.bench_cisplatin_real_vina import (  # noqa: E402
    CISPLATIN_SMILES,
    DEFAULT_POCKETS,
    _dock_one_python_binding,
    _load_receptor_geometry,
    _probe_vina_accepts_chemotype_atype,
    _rewrite_metal_atype_for_vina,
    _vina_record_status,
)

LOG = logging.getLogger("cfm_vina_generated_sweep")

DEFAULT_COHORT = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "cfm_fullcorpus_2026_10_10"
    / "generated_smiles.csv"
)
DEFAULT_PREP_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf30_pocket_prep"
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "cfm_vina_sweep_2026_10_12"

DEFAULT_SEED = 42
DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_N_POSES = 3


# ---------------------------------------------------------------------------
# Exception taxonomy — the reason errors_by_type exists
# ---------------------------------------------------------------------------
def classify_exception(exc: BaseException) -> str:
    """Map an exception to a stable, non-pooling error type.

    The whole point of splitting by type is that these three are NOT the same
    event and must never be averaged into one failure count:

    ``MetalPDBQTUnavailable``
        RDKit/meeko could not build a PDBQT (usually ETKDG embed). A property
        of the MOLECULE.
    ``VinaAtomTypeRejected``
        Vina's parser rejected an atype. A property of the HARNESS — the boron
        class that killed the previous sweep. Should be ZERO; a non-zero count
        means a regression, not a property of the cohort.
    ``DockError``
        Anything else raised inside the Vina call (box, maps, numerics).
    ``Other``
        Unclassified. Kept separate precisely so it cannot hide inside a pool.
    """
    name = type(exc).__name__
    if name == "MetalPDBQTUnavailable":
        return "MetalPDBQTUnavailable"
    text = str(exc)
    if "is not a valid AutoDock type" in text or "PDBQT parsing error" in text:
        return "VinaAtomTypeRejected"
    return "DockError" if name in {"RuntimeError", "ValueError"} else "Other"


# ---------------------------------------------------------------------------
# One (pocket, molecule) dock
# ---------------------------------------------------------------------------
def dock_one(
    pocket_id: str,
    smiles: str,
    receptor_pdbqt: Path,
    center_xyz: List[float],
    work_dir: Path,
    seed: int,
    exhaustiveness: int,
    n_poses: int,
) -> Dict[str, Any]:
    """Dock one SMILES into one pocket. NEVER raises — returns a record.

    A raised exception here would take down the pocket sweep and lose every
    pocket after it, which is exactly how the previous sweep lost nine of ten.
    """
    from molmetal.molmetal_lam.sbdd_env.vina_adapter import _ligand_to_pdbqt

    rec: Dict[str, Any] = {
        "pocket_id": pocket_id,
        "smiles": smiles,
        "seed": int(seed),
        "kcal_mol": None,
        "n_atoms_ligand": None,
        "status": "NOT_DOCKED",
        "error_type": None,
        "error": None,
        "atype_substitutions": [],
        "vina_accepted_atype": None,
    }
    lig_pdbqt: Optional[Path] = None
    v = None
    try:
        lig_pdbqt = _ligand_to_pdbqt(smiles, work_dir, name=f"lig_{pocket_id}")
        rec["n_atoms_ligand"] = sum(
            1
            for line in lig_pdbqt.read_text().splitlines()
            if line.startswith("ATOM")
        )

        subs: List[Tuple[str, str, str]] = []
        lig_pdbqt = _rewrite_metal_atype_for_vina(
            lig_pdbqt, chemotype_vdw=True, record=subs
        )
        rec["atype_substitutions"] = [list(s) for s in subs]

        accepted = _probe_vina_accepts_chemotype_atype(lig_pdbqt)
        rec["vina_accepted_atype"] = accepted
        rec["status"] = _vina_record_status(subs, accepted)

        kcal, _elapsed, rmsd_lb, rmsd_ub = _dock_one_python_binding(
            receptor_pdbqt,
            lig_pdbqt,
            center_xyz,
            seed,
            exhaustiveness,
            n_poses,
        )
        if not math.isfinite(kcal):
            rec["error_type"] = "DockError"
            rec["error"] = "non-finite kcal/mol"
            rec["status"] = "DOCK_FAILED"
        else:
            rec["kcal_mol"] = float(kcal)
            rec["best_pose_rmsd_lb"] = rmsd_lb
            rec["best_pose_rmsd_ub"] = rmsd_ub
    except BaseException as exc:  # noqa: BLE001 — see docstring: must not escape
        rec["error_type"] = classify_exception(exc)
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rec["status"] = "DOCK_FAILED"
        rec["traceback"] = traceback.format_exc()[-800:]
        LOG.warning(
            "  %s: FAIL (%s) %s",
            pocket_id,
            rec["error_type"],
            smiles[:70],
        )
    finally:
        # The previous run died with no traceback. Memory pressure is the one
        # cause that produces exactly that signature, so the Vina object (which
        # holds the full 25 A map grid) is released explicitly per molecule.
        del v
        gc.collect()
    return rec


# ---------------------------------------------------------------------------
# Summary — the pure function the test pins
# ---------------------------------------------------------------------------
def summarise_pocket(pocket_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one pocket's records WITH the attempted denominator.

    The dead sweep reported ``n=51 mean=-6.407``. That is ``n_docked`` here, and
    ``n_attempted`` is reported next to it so the two can never be confused.
    """
    n_attempted = len(records)
    docked = [r for r in records if r.get("kcal_mol") is not None]
    n_docked = len(docked)
    errors_by_type = Counter(
        r["error_type"] for r in records if r.get("error_type")
    )
    statuses = Counter(r["status"] for r in records)

    kcal = sorted(float(r["kcal_mol"]) for r in docked)
    out: Dict[str, Any] = {
        "pocket_id": pocket_id,
        "n_attempted": n_attempted,
        "n_docked": n_docked,
        "n_failed": n_attempted - n_docked,
        "docked_fraction": (n_docked / n_attempted) if n_attempted else None,
        "errors_by_type": dict(errors_by_type),
        "status_counts": dict(statuses),
    }
    if kcal:
        out["kcal_mean_over_docked"] = sum(kcal) / len(kcal)
        out["kcal_median_over_docked"] = kcal[len(kcal) // 2]
        out["kcal_min"] = kcal[0]
        out["kcal_max"] = kcal[-1]
        # Explicitly named so it can never be quoted without the denominator.
        out["_survivor_conditioned"] = True
        out["_mean_excludes"] = n_attempted - n_docked
        # 2026-10-12. The survivor label was already here; the SIZE confound was
        # not. MEASURED on the 128 docked records of
        # molmetal/reports/cfm_vina_sweep_2026_10_12/per_record.jsonl:
        #
        #   r(kcal_mol, n_atoms_ligand) = +0.2953
        #   slope                         = +0.0538 kcal/mol per atom
        #   mean n_atoms_ligand           = 37.56
        #
        # Bare cisplatin is FOUR heavy atoms; the generated cohort averages 37.6.
        # A flat 32-atom size shift alone predicts +1.72 kcal/mol of difference.
        # So "generated vs cisplatin" is dominated by SIZE, not chemistry, and a
        # single pooled mean cannot separate the two. The number stays (it is
        # what was measured) but it can no longer be read as affinity.
        out["_kcal_is_size_confounded"] = True
        out["_size_confound"] = (
            "r(kcal_mol, n_atoms_ligand)=+0.2953, slope=+0.0538 kcal/mol/atom, "
            "mean n_atoms_ligand=37.56 vs cisplatin's 4. A flat 32-atom size "
            "shift predicts +1.72 kcal/mol. Quote this as a FEASIBILITY "
            "signal (docked_fraction), never as an affinity comparison."
        )
        out["_citable"] = (
            "docked_fraction and errors_by_type only. Not citable: any "
            "absolute kcal/mol, any generated-vs-cisplatin comparison, any "
            "pocket ranking by mean kcal. Two of ten pockets completed, which "
            "is a sanity check that the protocol runs, not a sample."
        )
    else:
        out["kcal_mean_over_docked"] = None
        out["kcal_median_over_docked"] = None
        out["kcal_min"] = None
        out["kcal_max"] = None
        out["_survivor_conditioned"] = None
    atoms = [
        r["n_atoms_ligand"] for r in docked if r.get("n_atoms_ligand") is not None
    ]
    out["mean_n_atoms_ligand_over_docked"] = (
        sum(atoms) / len(atoms) if atoms else None
    )
    return out


def summarise_cohort(
    records: List[Dict[str, Any]],
    pockets: List[Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:
    """Whole-cohort aggregate. Still survivor-conditioned, still says so."""
    n_attempted = sum(p["n_attempted"] for p in pockets)
    n_docked = sum(p["n_docked"] for p in pockets)
    errors: Counter = Counter()
    statuses: Counter = Counter()
    for p in pockets:
        errors.update(p["errors_by_type"])
        statuses.update(p["status_counts"])

    kcal = [float(r["kcal_mol"]) for r in records if r.get("kcal_mol") is not None]
    kcal.sort()
    out: Dict[str, Any] = {
        "label": label,
        "n_attempted": n_attempted,
        "n_docked": n_docked,
        "n_failed": n_attempted - n_docked,
        "docked_fraction": (n_docked / n_attempted) if n_attempted else None,
        "errors_by_type": dict(errors),
        "status_counts": dict(statuses),
        "n_pockets": len(pockets),
    }
    if kcal:
        out["kcal_mean_over_docked"] = sum(kcal) / len(kcal)
        out["kcal_median_over_docked"] = kcal[len(kcal) // 2]
        out["_survivor_conditioned"] = True
        out["_mean_excludes"] = n_attempted - n_docked
    else:
        out["kcal_mean_over_docked"] = None
        out["kcal_median_over_docked"] = None
        out["_survivor_conditioned"] = None
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_cohort(csv_path: Path) -> List[str]:
    import csv

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"cohort CSV has no rows: {csv_path}")
    return [r["smiles"] for r in rows if (r.get("smiles") or "").strip()]


def sweep_pocket(
    pocket_id: str,
    smiles_list: List[str],
    receptor: Path,
    work_root: Path,
    seed: int,
    exhaustiveness: int,
    n_poses: int,
    jsonl: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Dock the whole cohort into one pocket, appending each record to JSONL.

    The JSONL append happens INSIDE the loop: a pocket that dies mid-way leaves
    its completed records on disk instead of losing the pocket entirely, which
    is what made 9 of 10 pockets NOT ON DISK last time.
    """
    center_xyz, _n_heavy = _load_receptor_geometry(receptor)
    if center_xyz is None:
        center_xyz = [0.0, 0.0, 0.0]
        LOG.warning("%s: receptor centroid unavailable; using origin", pocket_id)

    work_dir = work_root / pocket_id
    work_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    n = len(smiles_list)
    for i, smi in enumerate(smiles_list, 1):
        rec = dock_one(
            pocket_id,
            smi,
            receptor,
            center_xyz,
            work_dir,
            seed,
            exhaustiveness,
            n_poses,
        )
        records.append(rec)
        with jsonl.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if i % 10 == 0 or i == n:
            LOG.info(
                "  %s: %d/%d done, %d docked so far",
                pocket_id,
                i,
                n,
                sum(1 for r in records if r.get("kcal_mol") is not None),
            )
    return summarise_pocket(pocket_id, records), records


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="cfm_vina_generated_sweep",
        description=(
            "Re-run the 10-pocket Vina sweep over the generated cohort and the "
            "cisplatin control, reporting the attempted DENOMINATOR and "
            "per-pocket error counts split by exception type."
        ),
    )
    p.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--pockets", nargs="+", default=list(DEFAULT_POCKETS))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS)
    p.add_argument("--n-poses", type=int, default=DEFAULT_N_POSES)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the cohort size (smoke test only; 0 = full cohort).",
    )
    p.add_argument(
        "--skip-control",
        action="store_true",
        help="Skip the cisplatin positive control arm.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "_work"
    work_root.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(args.cohort_csv)
    if args.limit:
        cohort = cohort[: args.limit]
    LOG.info("cohort: %d molecules from %s", len(cohort), args.cohort_csv)

    jsonl = out_dir / "per_record.jsonl"
    jsonl.write_text("")  # fresh run; the file is append-only WITHIN a run

    import vina  # noqa: F401 — pre-flight, fail loudly before any pocket

    arms: Dict[str, Dict[str, Any]] = {}
    arm_defs: List[Tuple[str, List[str]]] = [("generated", cohort)]
    if not args.skip_control:
        arm_defs.append(("cisplatin_control", [CISPLATIN_SMILES]))

    for label, arm_cohort in arm_defs:
        arm_jsonl = jsonl if label == "generated" else (out_dir / "per_record_control.jsonl")
        if arm_jsonl != jsonl:
            arm_jsonl.write_text("")
        LOG.info("=== arm: %s (%d molecules) ===", label, len(arm_cohort))
        all_records: List[Dict[str, Any]] = []
        pocket_summaries: List[Dict[str, Any]] = []
        for pocket_id in args.pockets:
            receptor = args.prep_dir / pocket_id / "receptor.pdbqt"
            if not receptor.exists():
                LOG.error(
                    "%s: receptor.pdbqt MISSING at %s — recording as an "
                    "attempted pocket with 0 attempts, not skipping silently",
                    pocket_id,
                    receptor,
                )
                pocket_summaries.append(
                    {
                        "pocket_id": pocket_id,
                        "n_attempted": 0,
                        "n_docked": 0,
                        "n_failed": 0,
                        "docked_fraction": None,
                        "errors_by_type": {"MissingReceptor": 1},
                        "status_counts": {},
                        "kcal_mean_over_docked": None,
                        "kcal_median_over_docked": None,
                        "kcal_min": None,
                        "kcal_max": None,
                        "_survivor_conditioned": None,
                        "mean_n_atoms_ligand_over_docked": None,
                    }
                )
                continue
            LOG.info("--- %s ---", pocket_id)
            summ, recs = sweep_pocket(
                pocket_id,
                arm_cohort,
                receptor,
                work_root,
                args.seed,
                args.exhaustiveness,
                args.n_poses,
                arm_jsonl,
            )
            pocket_summaries.append(summ)
            all_records.extend(recs)
            LOG.info(
                "%s: attempted=%d docked=%d failed=%d errors=%s "
                "mean(kcal|over docked)=%s",
                pocket_id,
                summ["n_attempted"],
                summ["n_docked"],
                summ["n_failed"],
                summ["errors_by_type"],
                summ["kcal_mean_over_docked"],
            )
            # Persist after EVERY pocket, so a death costs one pocket, not nine.
            (out_dir / "partial_summary.json").write_text(
                json.dumps(
                    {
                        "_generated_utc": datetime.now(timezone.utc).isoformat(),
                        "_state": "PARTIAL",
                        "arm": label,
                        "per_pocket": pocket_summaries,
                    },
                    indent=2,
                )
            )

        arms[label] = {
            "cohort": summarise_cohort(all_records, pocket_summaries, label),
            "per_pocket": pocket_summaries,
        }

    summary = {
        "_metric": "cfm_vina_generated_sweep_2026_10_12",
        "_generated_utc": datetime.now(timezone.utc).isoformat(),
        "_state": "COMPLETE",
        "_engine": f"vina {getattr(vina, '__version__', 'unknown')} (Python binding)",
        "_seed": int(args.seed),
        "_exhaustiveness": int(args.exhaustiveness),
        "_n_poses": int(args.n_poses),
        "_pockets": list(args.pockets),
        "_cohort_csv": str(args.cohort_csv),
        "_cohort_size": len(cohort),
        "_arms": arms,
        "_honest_framing": {
            "denominator": (
                "n_attempted is reported beside every mean. Any mean over "
                "kcal_mean_over_docked is conditioned on the molecules that "
                "docked; failures are not a random sample (they are selected by "
                "RDKit embeddability), so the conditioned mean does not describe "
                "the cohort."
            ),
            "atype": (
                "All six train metals (Pt Ru Ir Rh Os Re) are in Vina 1.2.7's "
                "rejected-element set and are scored as Mg or as a covalent "
                "surrogate. Records carry MEASURED_GENERIC_ATOM or "
                "MEASURED_SURROGATE_ATOM; the kcal/mol describes the stand-in."
            ),
            "size_confound": (
                "Cisplatin is 4 heavy atoms; the generated cohort is ~40. Vina "
                "scores larger ligands more favourably, so the generated-vs-"
                "control difference is dominated by SIZE, not affinity. "
                "mean_n_atoms_ligand_over_docked is recorded so this is "
                "checkable."
            ),
            "what_vina_supports": (
                "Pose feasibility, not affinity. Cinaroglu (JCIM 2019): 73% pose "
                "success over 213 metal-protein complexes with ~0 scoring power. "
                "A 2026 Springer review: Vina systematically underestimates "
                "binding energies and generates wrong poses for metal ligands "
                "and accepts no custom metal parameters."
            ),
        },
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "partial_summary.json").unlink(missing_ok=True)
    LOG.info("summary written: %s", out_dir / "sweep_summary.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
