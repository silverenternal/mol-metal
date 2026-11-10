#!/usr/bin/env python3
"""Per-epoch quality of a free-generation pool: fragments, rings, validity.

WHY THIS EXISTS
---------------
``c7_scale_matched_pools.py`` answers the C7 gate (SA <= 4.5, novelty) but says
nothing about whether the molecules got *worse* while the SA rate stayed flat.
The 18-epoch full-corpus run emits 704 molecules per epoch, and the epoch
column is preserved in ``generated_smiles.csv``, so the trajectory is already
on disk — it just was never tabulated.

WHAT IT MEASURES, PER EPOCH
---------------------------
  * ``n_rows`` / ``n_parse`` / ``validity`` — RDKit-parsed fraction,
  * ``mean_n_frags`` / ``share_n_frags_2_to_6`` / ``share_n_frags_ge21`` — how many
    connected components the molecule came out as. **The corpus is multi-component
    by convention** (the metal is written as its own atom: 69.1% of training rows
    have 3 components, 0.07% have one), so ONE component is not "a molecule".
  * ``second_frag_is_single_atom`` / ``mean_second_frag_size`` — the composition
    behind the count. One big fused blob plus leftover atoms has the same
    component count as three ligands, and only this separates them.
  * ``single_fragment_rate`` — the trainer's own ``n_frags_raw == 1`` column.
    Kept for cross-checking against ``connected_rate`` (they agree exactly on the
    generation pool); it is NOT a quality metric for this dataset.
  * ``mean_largest_frag_frac`` — the trainer's own connectivity summary,
  * ``ring_size_distribution`` — atom rings by size, plus the 3-ring and
    6-ring shares of all rings,
  * ``n_unique_smiles`` / ``mean_heavy_atoms``.

Read-only. No model, no GPU. Numbers land in a JSON with MEASURED tags.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    KIND_RECONSTRUCTION,
    classify_pool,
)
from molmetal.scripts.cfm_pool_epoch_rule import census_row  # noqa: E402
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: The run census -- the SSOT of which run wrote what.
CENSUS_PATH = PROJECT_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"


def _epoch_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    n_rows = len(rows)
    n_parse = 0
    n_single = 0
    n_frag_present = 0
    n_frac_present = 0
    frac_sum = 0.0
    heavy_sum = 0
    bond_sum = 0
    n_bond_eligible = 0
    n_connected = 0
    frag_counts: List[int] = []
    second_sizes: List[int] = []
    unique = set()
    ring_sizes: Counter = Counter()
    for row in rows:
        text = row.get("smiles", "")
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) and text else None
        if mol is None:
            continue
        n_parse += 1
        unique.add(Chem.MolToSmiles(mol))
        heavy = mol.GetNumHeavyAtoms()
        heavy_sum += heavy
        if heavy >= 2:
            bond_sum += mol.GetNumBonds()
            n_bond_eligible += 1
        # Connectivity read off the molecule, not off the trainer's
        # n_frags_raw column: reconstruction CSVs have no such column, and
        # without this the head's pools would score None on the one measure
        # that makes them comparable to a generation pool.
        frags = sorted((len(f) for f in Chem.GetMolFrags(mol)), reverse=True)
        frag_counts.append(len(frags))
        if len(frags) == 1:
            n_connected += 1
        if len(frags) >= 2:
            # The corpus writes the metal as its own component, so a correct
            # molecule has several SUBSTANTIAL components. A second component
            # of one atom means the molecule is one fused blob plus leftovers,
            # which the fragment count alone cannot distinguish.
            second_sizes.append(frags[1])
        frag = row.get("n_frags_raw", "")
        try:
            if int(float(frag)) == 1:
                n_single += 1
            n_frag_present += 1
        except (TypeError, ValueError):
            pass
        frac = row.get("largest_frag_frac", "")
        try:
            frac_sum += float(frac)
            n_frac_present += 1
        except (TypeError, ValueError):
            pass
        for ring in mol.GetRingInfo().AtomRings():
            ring_sizes[len(ring)] += 1
    total_rings = sum(ring_sizes.values())
    return {
        "n_rows": n_rows,
        "n_parse": n_parse,
        "validity": round(n_parse / n_rows, 4) if n_rows else None,
        # A pool written before the trainer emitted n_frags_raw has no
        # connectivity column at all; report None, not a flattering 0%.
        "single_fragment_rate": (
            round(n_single / n_frag_present, 4) if n_frag_present else None
        ),
        "connected_rate": round(n_connected / n_parse, 4) if n_parse else None,
        "mean_n_frags": round(sum(frag_counts) / len(frag_counts), 3) if frag_counts else None,
        "share_n_frags_2_to_6": (
            round(sum(1 for c in frag_counts if 2 <= c <= 6) / len(frag_counts), 4)
            if frag_counts else None
        ),
        "share_n_frags_ge21": (
            round(sum(1 for c in frag_counts if c >= 21) / len(frag_counts), 4)
            if frag_counts else None
        ),
        "second_frag_is_single_atom": (
            round(sum(1 for s in second_sizes if s == 1) / len(second_sizes), 4)
            if second_sizes else None
        ),
        "mean_second_frag_size": (
            round(sum(second_sizes) / len(second_sizes), 3) if second_sizes else None
        ),
        "mean_largest_frag_frac": round(frac_sum / n_frac_present, 4) if n_frac_present else None,
        "n_unique_smiles": len(unique),
        "mean_heavy_atoms": round(heavy_sum / n_parse, 3) if n_parse else None,
        # Bonds per heavy atom separates "how many bonds" from "which bonds".
        # A pool can match the real bond density and still put every bond in a
        # 3-membered ring.
        "bonds_per_heavy_atom": (
            round(bond_sum / heavy_sum, 4) if heavy_sum else None
        ),
        "n_rings": total_rings,
        "ring_size_distribution": {str(k): v for k, v in sorted(ring_sizes.items())},
        "three_ring_share": round(ring_sizes.get(3, 0) / total_rings, 4) if total_rings else None,
        "six_ring_share": round(ring_sizes.get(6, 0) / total_rings, 4) if total_rings else None,
    }


def _bond_head_trajectory(run_dir: Path) -> Dict[str, object] | None:
    """Teacher-forced bond metrics per epoch, from the run's own metrics file.

    These are computed on real held-out geometries (the head is given the true
    atoms and asked for the bonds), which is a different question from whether
    the sampler can connect a cloud. Putting them side by side is the point:
    if the head improves while the generated pools fragment more, the failure
    is in the sampling path, not in the head.
    """
    path = run_dir / "per_epoch_metrics.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    rows = payload.get("per_epoch_metrics") or []
    keep = (
        "epoch",
        "bond_precision",
        "bond_recall",
        "bond_f1",
        "bond_pr_auc",
        "bond_n_scored",
    )
    out = [{k: row.get(k) for k in keep if k in row} for row in rows]
    return out or None


def _event_stream_epochs(run_dir: Path) -> List[int]:
    """Every epoch index the run's append-only event stream carries.

    The stream is the record of EVERY population that ever wrote into the
    directory, so it can hold MORE epochs than the run of record ran: the
    leading ones are a killed pre-resume attempt.
    """
    path = run_dir / "events.jsonl"
    epochs: set[int] = set()
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("event") != "train_epoch":
                    continue
                try:
                    epochs.add(int(row.get("epoch")))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return []
    return sorted(epochs)


def _done_event(run_dir: Path) -> Dict[str, object] | None:
    """The run's ``done`` event, or None when the stream carries none."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("event") == "done":
                    return row
    except OSError:
        return None
    return None


def _census_for(run_name: str) -> Dict[str, object] | None:
    """The run census's row for ``run_name``, or None when it carries none."""
    if not CENSUS_PATH.exists():
        return None
    try:
        census = json.loads(CENSUS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return census_row(census, run_name)


def _bond_head_trajectory_meta(
    run_dir: Path, trajectory: List[Dict[str, object]] | None
) -> Dict[str, object] | None:
    """WHAT the published bond-head trajectory covers -- read off disk.

    The trajectory list says which epochs carry a bond-head row; it does not
    say why those epochs and no others.  MEASURED on disk (Wf-A121 / I3):
    ``cfm_c7_fullgen_2026_10_20``'s trajectory starts at epoch 4, because the
    run RESUMED into the same directory and epochs 0..3 belong to a killed
    pre-resume attempt.  A reader who takes "epochs 4..17" for "the run's
    first four epochs were skipped" is wrong.

    Every field here is derived:
      * the covered epochs are the epochs of the rows that were read;
      * the run-of-record epochs are the census's own answer (or, when the
        census does not carry the run, the population the run's own ``done``
        event describes);
      * the verdict is the census's own ``population_verdict`` -- never
        invented: when the census does not carry the run, the meta SAYS so.
    """
    if not trajectory:
        return None
    covered = sorted(
        {int(r["epoch"]) for r in trajectory if r.get("epoch") is not None}
    )
    if not covered:
        return None

    metrics_path = run_dir / "per_epoch_metrics.json"
    try:
        source = str(metrics_path.relative_to(PROJECT_ROOT))
    except ValueError:
        source = str(metrics_path)

    stream_epochs = _event_stream_epochs(run_dir)
    done = _done_event(run_dir) or {}
    try:
        n_epochs_run = int(done.get("n_epochs_run"))
    except (TypeError, ValueError):
        n_epochs_run = None

    # The population's OWN epochs are the LAST ``n_epochs_run`` epochs of the
    # stream: a segment can hold more epochs than it ran, and the leading
    # extras are the killed pre-resume attempt (the census's own rule).
    own_epochs: List[int] = list(stream_epochs)
    if n_epochs_run is not None and 0 < n_epochs_run < len(stream_epochs):
        own_epochs = stream_epochs[-n_epochs_run:]

    row = _census_for(run_dir.name)
    integrity = (row or {}).get("integrity") or {}
    census_epochs = integrity.get("run_of_record_epoch_indices")
    if isinstance(census_epochs, list) and census_epochs:
        run_of_record_epochs = sorted(int(e) for e in census_epochs)
        run_of_record_source = "census:run_of_record_epoch_indices"
    else:
        run_of_record_epochs = list(own_epochs)
        run_of_record_source = (
            "events.jsonl:done(n_epochs_run) -- the census does not carry "
            "this run"
        )

    verdict = integrity.get("population_verdict")
    if row is None:
        source_verdict = None
        verdict_note = (
            f"the run census ({CENSUS_PATH.relative_to(PROJECT_ROOT)}) does "
            f"not carry this run, so no population verdict is published; the "
            f"run-of-record epochs below are the run's own `done` event"
        )
    else:
        source_verdict = str(verdict) if verdict is not None else None
        verdict_note = (
            f"the verdict is the run census's own "
            f"`integrity.population_verdict` for {run_dir.name}"
        )

    # The leading epochs of the stream that the trajectory does NOT cover: a
    # killed pre-resume attempt, when they are exactly a leading prefix.
    covered_set = set(covered)
    missing = [e for e in stream_epochs if e not in covered_set]
    is_resume = bool(missing) and missing == stream_epochs[: len(missing)]
    if is_resume:
        resume_note = (
            f"the trajectory starts at epoch {covered[0]} because the run "
            f"RESUMED: epochs {missing[0]}..{missing[-1]} belong to a killed "
            f"pre-resume attempt in the same directory"
        )
    elif missing:
        resume_note = (
            f"the trajectory covers epochs {covered[0]}..{covered[-1]} while "
            f"the event stream also carries {missing}; the epochs it does "
            f"not cover are not a leading prefix, so no resume is claimed"
        )
    else:
        resume_note = (
            f"the trajectory covers every epoch the event stream carries "
            f"({covered[0]}..{covered[-1]}); no pre-resume history on disk"
        )

    return {
        "epochs_covered": covered,
        "n_epochs": len(covered),
        "first_epoch": covered[0],
        "last_epoch": covered[-1],
        "run_of_record_epochs": run_of_record_epochs,
        "resume_note": resume_note,
        "source_verdict": source_verdict,
        "source": source,
        "run_of_record_source": run_of_record_source,
        "verdict_note": verdict_note,
        "verdict_source": str(CENSUS_PATH.relative_to(PROJECT_ROOT)),
        "done_event_n_epochs_run": n_epochs_run,
        "event_stream_epochs": stream_epochs,
    }


def _reference_stats(path: Path) -> Dict[str, object] | None:
    """The same numbers for real chemistry, so the pool has something to be wrong against.

    Without this, ``three_ring_share = 0.67`` reads as "lots of small rings",
    which sounds like a stylistic quirk. Against the corpus it is a different
    statement: the corpus has essentially no 3-membered rings at all.
    """
    if not path.exists():
        return None
    import pandas as pd

    frame = pd.read_csv(path, low_memory=False)
    if "split" in frame.columns:
        frame = frame[frame["split"] == "train"]
    if "smiles" not in frame.columns:
        return None
    # Reuse the pool summariser so the two sides cannot drift apart in
    # definition. The corpus has no n_frags_raw / largest_frag_frac columns,
    # so those two come back None and the fragment-count measures carry the
    # comparison.
    summary = _epoch_summary([{"smiles": s} for s in frame["smiles"].dropna().astype(str)])
    if not summary.get("n_parse"):
        return None
    summary["source"] = str(path)
    summary["n"] = summary["n_parse"]
    return summary


def _profile_pools(paths: Sequence[Path]) -> List[Dict[str, object]]:
    """Summarise bond-head reconstruction pools with the same measures.

    A reconstruction pool is written by ``wf_cfm_decode_dump_smiles.py`` in its
    FIRST era: real tmQM atoms, bonds predicted by the head, no sampler. It
    answers a different question from the generation pool — does the HEAD
    already get the topology wrong? — and without it the generation numbers
    cannot be attributed.
    """
    out: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        summary = _epoch_summary(rows)
        summary["label"] = path.parent.name
        summary["path"] = str(path)
        out.append(summary)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
        help="real corpus to compare the pool against; skipped when absent",
    )
    parser.add_argument(
        "--recon-pool",
        type=Path,
        action="append",
        help=(
            "bond-head reconstruction pool (a run dir or its generated_smiles.csv); "
            "repeatable. Defaults to the two full-corpus reconstruction pools on disk."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "c7_fullgen_pool_quality.json",
    )
    args = parser.parse_args(argv)

    recon_paths = args.recon_pool or [
        PROJECT_ROOT / "molmetal" / "reports" / name / "generated_smiles.csv"
        for name in (
            "cfm_fullcorpus_b64_2026_10_11",
            "cfm_objective_terms_2026_10_11",
        )
    ]
    recon_paths = [p / "generated_smiles.csv" if p.is_dir() else p for p in recon_paths]

    # The reconstruction pools are declared by the FLAG, but the KIND is
    # decided by the shared rule, not by the flag and not by the header
    # (one script wrote one header in two eras).  A pool that the rule does
    # not classify as a reconstruction is REFUSED here rather than profiled
    # as "the bond head on real geometries" -- the two kinds mean different
    # things (see molmetal/scripts/cfm_pool_kind.py).  This refuses an input;
    # it never changes a number.
    recon_pool_kinds: Dict[str, str] = {}
    for recon_pool in recon_paths:
        kind = classify_pool(recon_pool)
        recon_pool_kinds[str(recon_pool)] = kind.kind
        if kind.kind != KIND_RECONSTRUCTION:
            print(
                f"[c7_fullgen] REFUSING a --recon-pool that is not a "
                f"reconstruction: {recon_pool} (kind={kind.kind}; reason: "
                f"{kind.reason})",
                file=sys.stderr,
            )
            return 2

    path = args.run / "generated_smiles.csv"
    try:
        pool_label = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        # A pool outside the repo (a test fixture, an archived run).
        pool_label = str(path)

    # Wf-A114 verify:B D1 -- the MAIN pool is classified too, not only the
    # --recon-pool inputs.  The payload already carried `pool_kind_source`
    # next to `recon_pool_kinds`, which read as "the pool kinds are labelled"
    # while the main pool's numbers were unlabelled.  MEASURED (Wf-A114D):
    # the main pool's kind is decided by the CONTENT, not by the header --
    # `cfm_fullcorpus_2026_10_10/generated_smiles.csv` carries `metal_z` but
    # every row has `sanitized = 0` and an empty `metal_z`, so it is the
    # 2026-10-12+ ODE path and therefore a GENERATION pool: its numbers ARE
    # the generator's quality.  A main pool that the rule classifies as a
    # RECONSTRUCTION still must not be read as the generator's quality.  This
    # LABELS; it never changes a number.
    main_pool_kind = classify_pool(path)

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_epoch: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", "")))
        except (TypeError, ValueError):
            continue
        by_epoch.setdefault(epoch, []).append(row)

    per_epoch = {str(e): _epoch_summary(by_epoch[e]) for e in sorted(by_epoch)}
    overall = _epoch_summary(rows)
    order = sorted(per_epoch, key=int)
    first = per_epoch[order[0]] if order else {}
    last = per_epoch[order[-1]] if order else {}
    trend = None
    if first and last:
        trend = {
            "epoch_first": int(order[0]),
            "epoch_last": int(order[-1]),
            "validity": [first["validity"], last["validity"]],
            "single_fragment_rate": [first["single_fragment_rate"], last["single_fragment_rate"]],
            "three_ring_share": [first["three_ring_share"], last["three_ring_share"]],
        }

    bond_trajectory = _bond_head_trajectory(args.run)

    payload = {
        "artifact": "c7_fullgen_pool_quality",
        "tag": "MEASURED",
        "generated_utc": None,
        "provenance": {
            "script": "molmetal/scripts/c7_fullgen_pool_quality.py",
            "git_commit": _git_commit(),
            "pool": pool_label,
            "n_rows": len(rows),
            "epochs": len(per_epoch),
            "recon_pool_kinds": recon_pool_kinds,
            "pool_kind": main_pool_kind.kind,
            "pool_kind_reason": main_pool_kind.reason,
            "pool_kind_source": "molmetal/scripts/cfm_pool_kind.py",
        },
        "overall": overall,
        "per_epoch": per_epoch,
        "first_vs_last": trend,
        "bond_head_trajectory": bond_trajectory,
        "bond_head_trajectory_meta": _bond_head_trajectory_meta(
            args.run, bond_trajectory
        ),
        "reference": _reference_stats(args.reference),
        "head_on_real_geometries": _profile_pools(recon_paths),
        "reading": (
            f"The MAIN pool ({pool_label}) is classified as "
            f"'{main_pool_kind.kind}' by molmetal/scripts/cfm_pool_kind.py: "
            f"{main_pool_kind.reason}.  Its numbers are that kind's numbers. "
            "The corpus writes the metal as its own component (69.1% of training rows have 3 "
            "components, 0.07% have one), so a correct generation is multi-component and "
            "single_fragment_rate is not a quality metric here. Compare mean_n_frags against the "
            "reference, then second_frag_is_single_atom: matching the count while the second "
            "component is a lone atom means one fused blob plus leftovers, not several ligands. "
            "three_ring_share is the fraction of all atom rings that are 3-membered. "
            "bond_head_trajectory is teacher-forced (true atoms, predicted bonds) and measures the "
            "head; head_on_real_geometries measures the same head's bond graph as molecules."
        ),
    }
    from datetime import datetime, timezone

    payload["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[c7q] rows={len(rows)} epochs={len(per_epoch)} out={args.out}")
    for e in sorted(per_epoch, key=int):
        row = per_epoch[e]
        print(
            f"[c7q] ep{e:>2} valid={row['validity']} single={row['single_fragment_rate']} "
            f"r3={row['three_ring_share']} r6={row['six_ring_share']} "
            f"heavy={row['mean_heavy_atoms']} uniq={row['n_unique_smiles']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
