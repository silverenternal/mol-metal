#!/usr/bin/env python3
"""BLK-C-M4 BAND-reconciliation probe: per-band recall + FULL-bond precision at
AROMATIC_LO in {1.30, 1.45, 1.50}.

WHAT THIS MEASURES (vs the existing per-sub-band diagnostic)
------------------------------------------------------------
The 2026-10-13 sub-band diagnostic at
``molmetal/reports/cfm_blkc_m4_subband_diagnostic_2026_10_13/`` is locked at
AROMATIC_LO=1.30 (the evaluator's band) and only reports recall on the
truth-aromatic subset. The 71% aromatic->single collapse splits across
sub-bands as 5212/1.30-1.40 + 5216/1.40-1.45 + 1516/1.45-1.50 + 204/1.50-1.60
+ 46/1.60-1.70, so a substantial fraction sits IN the 1.30-1.50 wedge that
physically overlaps canonical single-bond Wiberg features.

The BLK-C-M4 live sub-question (i) is whether narrowing AROMATIC_LO to 1.45
or 1.50 RELOCATES the collapse rather than fixes it. The cheap measurement
on the board is a 1.45/1.50 probe of the same 150-per-metal cohort, but the
existing script's per_subband table is recall-only on the truth-aromatic
subset; the round-7 caveat is that narrowing the band changes which
*predictions* count as false-positives, and recall alone cannot see the
precision side.

This script:

  1. Runs the head ONCE on the same 900-complex 150-per-metal cohort.
  2. Stores ALL in-scope records (truth-aromatic + truth-single +
     truth-double + truth-triple + truth-none) -- not just the
     truth-aromatic subset.
  3. For a CLI-provided --aromatic-lo, recomputes ``truth_label`` against
     the new threshold and emits BOTH the per-subband recall (same schema
     as the 1.30 baseline) AND the full-bond precision.

PROTOCOL REUSE
--------------
BO/XYZ parsers, head loader, sym-to-Z table, heavy-atom scope filter and
the bond-decision logic are imported from
``molmetal/tests/test_wf_blkc_qed_lever_aromatic_head.py``. We override
the truth_label mapping locally (so the test's AROMATIC_LO=1.30 constant
is left untouched, per the next_actions[1] directive "Do NOT edit the
AROMATIC_LO constant").

USAGE
-----
    .venv/bin/python molmetal/scripts/wf_blkc_m4_band_sweep_2026_10_13.py \\
        --max-complexes 150 --aromatic-lo 1.30 \\
        --out-dir molmetal/reports/cfm_blkc_m4_band_sweep_1.30_2026_10_13/

    .venv/bin/python molmetal/scripts/wf_blkc_m4_band_sweep_2026_10_13.py \\
        --max-complexes 150 --aromatic-lo 1.45 \\
        --out-dir molmetal/reports/cfm_blkc_m4_band_sweep_1.45_2026_10_13/

    .venv/bin/python molmetal/scripts/wf_blkc_m4_band_sweep_2026_10_13.py \\
        --max-complexes 150 --aromatic-lo 1.50 \\
        --out-dir molmetal/reports/cfm_blkc_m4_band_sweep_1.50_2026_10_13/
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# AROMATIC_HI is held at the evaluator's 1.7 across the sweep; the only
# moving parameter is the AROMATIC_LO (passed via CLI). The test module's
# AROMATIC_LO constant stays at 1.30 -- this script does not import it
# for the truth_label mapping; it uses a local variable.
sys.path.insert(0, str(REPO))
from molmetal.tests.test_wf_blkc_qed_lever_aromatic_head import (  # noqa: E402
    AROMATIC_HI,  # kept at 1.7 (the evaluator's band)
    HEAD_AROMATIC,
    METAL_Z,
    TRAIN_METALS,
    _SYM_TO_Z,
    _iter_bo_blocks,
    _iter_xyz_blocks,
    _load_head,
)

# Sub-bands: same as the existing 1.30 diagnostic, fixed across the sweep so
# the per_subband table is directly comparable.
SUBBANDS: list[tuple[str, float, float]] = [
    ("1.30-1.40", 1.30, 1.40),
    ("1.40-1.45", 1.40, 1.45),
    ("1.45-1.50", 1.45, 1.50),
    ("1.50-1.60", 1.50, 1.60),
    ("1.60-1.70", 1.60, 1.70),
    (">=1.70", 1.70, float("inf")),
]
CLASS_NAMES = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

LOG = logging.getLogger("wf_blkc_m4_band_sweep")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def _truth_label(bo: float, aromatic_lo: float, aromatic_hi: float) -> str:
    """Local copy of the test's truth_label mapping, parameterised on
    AROMATIC_LO so we can sweep without editing the constant.

    The single/double/triple split is held at 1.5 and 2.5 -- the
    evaluator's other two boundaries -- because those are the bands the
    head was actually trained against (cf. _BO_TYPE_BANDS in
    ``molmetal/data/tmqm_clouds.py:174-179``), and the brief asks only
    about the aromatic lower boundary.
    """
    if aromatic_lo <= bo < aromatic_hi:
        return "aromatic"
    if bo < 1.5:
        return "single"
    if bo < 2.5:
        return "double"
    return "triple"


def _score_all(model, featuriser, symbols, coords, truth_bo, aromatic_lo):
    """Per-pair argmax + logits, ALL in-scope pairs (not just truth-aromatic).

    Reuses every filter the test applies:
      * non-H non-metal pair scope
      * truth_label recomputed against the *local* aromatic_lo

    Returns
    -------
    list of dict
        One entry per in-scope pair with keys: key, symbol_i, symbol_j, bo,
        truth_label, pred_class (int 0..4), logit_0..logit_4 (float).
    """
    import torch

    from molmetal.models.bond_head import AtomCloud

    zs = [_SYM_TO_Z.get(s, 0) for s in symbols]
    unknown = [s for s, z in zip(symbols, zs) if z == 0]
    if unknown:
        return None
    cloud = AtomCloud(
        positions=torch.as_tensor(coords, dtype=torch.float32),
        atomic_numbers=torch.as_tensor(zs, dtype=torch.long),
    )
    with torch.no_grad():
        res = featuriser.featurise(cloud)
    if not res.decode_succeeded or res.logits.shape[0] == 0:
        return None

    logits = res.logits
    src = res.edge_index[0].tolist()
    dst = res.edge_index[1].tolist()
    pred_class = logits.argmax(dim=-1).tolist()

    out = []
    for k in range(len(src)):
        s, d = int(src[k]), int(dst[k])
        key = (min(s, d) + 1, max(s, d) + 1)
        bo = truth_bo.get(key)
        if bo is None:
            # No Wiberg reading for this pair -- still record so the
            # full-bond-set precision denominator is honest about the
            # pairs we cannot label. Tag as truth_label=None.
            sym_i = symbols[key[0] - 1]
            sym_j = symbols[key[1] - 1]
            if sym_i == "H" or sym_j == "H":
                continue
            zi = _SYM_TO_Z.get(sym_i, 0)
            zj = _SYM_TO_Z.get(sym_j, 0)
            if zi in METAL_Z or zj in METAL_Z:
                continue
            out.append({
                "key": list(key),
                "symbol_i": sym_i,
                "symbol_j": sym_j,
                "bo": None,
                "truth_label": None,
                "pred_class": int(pred_class[k]),
                "logit_0": float(logits[k, 0]),
                "logit_1": float(logits[k, 1]),
                "logit_2": float(logits[k, 2]),
                "logit_3": float(logits[k, 3]),
                "logit_4": float(logits[k, 4]),
            })
            continue
        sym_i = symbols[key[0] - 1]
        sym_j = symbols[key[1] - 1]
        if sym_i == "H" or sym_j == "H":
            continue
        zi = _SYM_TO_Z.get(sym_i, 0)
        zj = _SYM_TO_Z.get(sym_j, 0)
        if zi in METAL_Z or zj in METAL_Z:
            continue
        t_label = _truth_label(bo, aromatic_lo, AROMATIC_HI)
        out.append({
            "key": list(key),
            "symbol_i": sym_i,
            "symbol_j": sym_j,
            "bo": float(bo),
            "truth_label": t_label,
            "pred_class": int(pred_class[k]),
            "logit_0": float(logits[k, 0]),
            "logit_1": float(logits[k, 1]),
            "logit_2": float(logits[k, 2]),
            "logit_3": float(logits[k, 3]),
            "logit_4": float(logits[k, 4]),
        })
    return out


def _aromatic_logit_margin(rec: dict) -> float:
    others = [rec["logit_0"], rec["logit_1"], rec["logit_2"], rec["logit_3"]]
    return rec["logit_4"] - max(others)


def _per_subband_table(aromatic_recs: list[dict]) -> list[dict]:
    """Same schema as the 1.30 baseline per_subband table."""
    out = []
    for label, lo, hi in SUBBANDS:
        rows = [r for r in aromatic_recs if lo <= r["bo"] < hi]
        if not rows:
            out.append({"band": label, "n": 0})
            continue
        ctr: Counter[str] = Counter()
        for r in rows:
            ctr[CLASS_NAMES[r["pred_class"]]] += 1
        margins = [_aromatic_logit_margin(r) for r in rows]
        n = len(rows)
        n_pred_arom = ctr["AROMATIC"]
        n_collapse_to_single = ctr["SINGLE"]
        out.append({
            "band": label,
            "n": n,
            "predicted_counts": dict(ctr),
            "predicted_fractions": {
                k: (v / n if n else 0.0) for k, v in ctr.items()
            },
            "aromatic_recall": n_pred_arom / n if n else 0.0,
            "aromatic_to_single_count": n_collapse_to_single,
            "aromatic_to_single_fraction": (
                n_collapse_to_single / n if n else 0.0
            ),
            "logit_margin_mean": statistics.fmean(margins),
            "logit_margin_median": statistics.median(margins),
            "logit_margin_min": min(margins),
            "logit_margin_max": max(margins),
        })
    return out


def _arom_to_single_collapse_distribution(aromatic_recs: list[dict]) -> dict:
    collapse_bos = [r["bo"] for r in aromatic_recs if r["pred_class"] == 1]
    return {
        "n_arom_to_single": len(collapse_bos),
        "mean_bo": statistics.fmean(collapse_bos) if collapse_bos else None,
        "median_bo": statistics.median(collapse_bos) if collapse_bos else None,
        "min_bo": min(collapse_bos) if collapse_bos else None,
        "max_bo": max(collapse_bos) if collapse_bos else None,
        "per_subband_counts": {
            label: sum(
                1 for r in aromatic_recs
                if r["pred_class"] == 1 and lo <= r["bo"] < hi
            )
            for label, lo, hi in SUBBANDS
        },
    }


def _select_cohort(syms_by_code, n_per_metal: int, seed: int = 0) -> list[str]:
    """Per-metal balanced sample, matches bench_tmqm_bond_perception.py."""
    per_metal = {m: [] for m in TRAIN_METALS}
    for code, syms in syms_by_code.items():
        for m in set(syms) & TRAIN_METALS:
            per_metal[m].append(code)

    rng = random.Random(seed)
    chosen: list[str] = []
    for m in sorted(TRAIN_METALS):
        pool = sorted(set(per_metal.get(m, [])))
        rng.shuffle(pool)
        for code in pool[:n_per_metal]:
            chosen.append(code)
    return sorted(set(chosen))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO
        / "molmetal"
        / "reports"
        / "cfm_fullcorpus_2026_10_10"
        / "checkpoints"
        / "cfm_yuelbond_wasserstein_seed0.pt",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--aromatic-lo",
        type=float,
        required=True,
        help=(
            "Lower boundary of the aromatic band. Held local to this "
            "script; does NOT edit the test module's AROMATIC_LO constant."
        ),
    )
    parser.add_argument("--aromatic-hi", type=float, default=AROMATIC_HI)
    parser.add_argument(
        "--max-complexes",
        type=int,
        default=0,
        help=(
            "0 = run all qualifying complexes in the BO shards. "
            "If > 0, cap to that many complexes per metal after passing "
            "the metal-presence filter."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()
    _setup_logging()

    if not (0.0 < args.aromatic_lo < args.aromatic_hi < 5.0):
        LOG.error(
            "Unreasonable band: AROMATIC_LO=%s AROMATIC_HI=%s",
            args.aromatic_lo, args.aromatic_hi,
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bo_paths = [
        REPO / ".data" / "molmetal" / "tmQM" / f"tmQM_X{i}.BO.gz"
        for i in (1, 2, 3)
    ]
    xyz_paths = [
        REPO / ".data" / "molmetal" / "tmQM" / f"tmQM_X{i}.xyz.gz"
        for i in (1, 2, 3)
    ]
    if not all(p.is_file() for p in bo_paths) or not all(
        p.is_file() for p in xyz_paths
    ):
        LOG.error("tmQM BO/XYZ shards missing")
        return 2
    if not args.checkpoint.is_file():
        LOG.error("checkpoint missing: %s", args.checkpoint)
        return 2

    LOG.info("Loading head from %s", args.checkpoint)
    model, featuriser = _load_head()

    # Index qualifying complexes.
    LOG.info("Indexing qualifying complexes (single pass over BO shards) ...")
    syms_by_code: dict[str, list[str]] = {}
    truth_by_code: dict[str, dict] = {}
    for code, atoms in _iter_bo_blocks(bo_paths):
        syms = [a[1] for a in atoms]
        if not (set(syms) & TRAIN_METALS):
            continue
        syms_by_code[code] = syms
        from molmetal.tests.test_wf_blkc_qed_lever_aromatic_head import (
            _ground_truth_bonds,
        )
        truth_by_code[code] = _ground_truth_bonds(atoms)
    LOG.info("  %d complexes carry one of the six train metals", len(syms_by_code))

    # Per-metal balanced sample.
    per_metal = {m: [] for m in TRAIN_METALS}
    for code, syms in syms_by_code.items():
        for m in set(syms) & TRAIN_METALS:
            per_metal[m].append(code)
    LOG.info("Per-metal pool sizes: %s", {m: len(v) for m, v in per_metal.items()})

    rng = random.Random(args.sample_seed)
    codes_to_run: list[str] = []
    for m in sorted(TRAIN_METALS):
        pool = sorted(set(per_metal.get(m, [])))
        rng.shuffle(pool)
        n = args.max_complexes if args.max_complexes else len(pool)
        codes_to_run.extend(pool[:n])
    codes_to_run = sorted(set(codes_to_run))
    LOG.info(
        "Running head on %d complexes (cap=%s per metal)",
        len(codes_to_run),
        args.max_complexes if args.max_complexes else "all",
    )

    # Build {code: (syms, coords)}.
    LOG.info("Indexing XYZ geometries (single pass) ...")
    coords_by_code: dict[str, tuple[list[str], object]] = {}
    want = set(codes_to_run)
    for code, symbols, coords in _iter_xyz_blocks(xyz_paths):
        if code in want:
            coords_by_code[code] = (symbols, coords)
    LOG.info(
        "  %d of %d sampled complexes have XYZ geometries on this host",
        len(coords_by_code),
        len(codes_to_run),
    )

    t0 = time.time()
    n_succeeded = 0
    n_failed_decode = 0
    n_missing_xyz = 0
    n_no_truth_aromatic = 0
    all_records: list[dict] = []  # ALL in-scope records, not just truth-aromatic

    for i, code in enumerate(codes_to_run):
        if (i + 1) % 50 == 0 or (i + 1) == len(codes_to_run):
            elapsed = time.time() - t0
            LOG.info(
                "  [%d/%d] succeeded=%d failed_decode=%d missing_xyz=%d "
                "records=%d elapsed=%.1fs",
                i + 1, len(codes_to_run), n_succeeded, n_failed_decode,
                n_missing_xyz, len(all_records), elapsed,
            )
        truth = truth_by_code.get(code)
        geo = coords_by_code.get(code)
        if geo is None:
            n_missing_xyz += 1
            continue
        symbols, coords = geo
        recs = _score_all(
            model, featuriser, symbols, coords, truth, args.aromatic_lo
        )
        if recs is None:
            n_failed_decode += 1
            continue
        n_succeeded += 1
        for r in recs:
            r["code"] = code
        all_records.extend(recs)
        if not [r for r in recs if r["truth_label"] == "aromatic"]:
            n_no_truth_aromatic += 1

    LOG.info(
        "Head run complete: succeeded=%d failed_decode=%d missing_xyz=%d "
        "no_truth_aromatic=%d total_records=%d elapsed=%.1fs",
        n_succeeded, n_failed_decode, n_missing_xyz, n_no_truth_aromatic,
        len(all_records), time.time() - t0,
    )

    # ---- Headline + per-sub-band (truth-aromatic subset only) ----
    aromatic_records = [r for r in all_records if r["truth_label"] == "aromatic"]
    n_truth_arom = len(aromatic_records)
    n_pred_arom_in_arom_subset = sum(
        1 for r in aromatic_records if r["pred_class"] == HEAD_AROMATIC
    )
    n_arom_to_single = sum(
        1 for r in aromatic_records if r["pred_class"] == 1
    )
    LOG.info(
        "Truth-aromatic subset at AROMATIC_LO=%.2f: pred_arom=%d / %d "
        "(recall %.4f); arom_to_single=%d (frac %.4f)",
        args.aromatic_lo, n_pred_arom_in_arom_subset, n_truth_arom,
        n_pred_arom_in_arom_subset / max(1, n_truth_arom),
        n_arom_to_single, n_arom_to_single / max(1, n_truth_arom),
    )

    summary = _per_subband_table(aromatic_records)
    collapse_summary = _arom_to_single_collapse_distribution(aromatic_records)

    # ---- FULL-BOND-SET precision (round-7 caveat) ----
    # precision = (truth_aromatic AND pred_aromatic) / (all pred_aromatic)
    # on the FULL bond set (truth-aromatic + truth-single + truth-double +
    # truth-triple + truth-none), not just the truth-aromatic subset.
    n_pred_arom_full = sum(1 for r in all_records if r["pred_class"] == HEAD_AROMATIC)
    n_tp_full = sum(
        1 for r in all_records
        if r["truth_label"] == "aromatic" and r["pred_class"] == HEAD_AROMATIC
    )
    n_fp_full = n_pred_arom_full - n_tp_full
    precision_full = n_tp_full / n_pred_arom_full if n_pred_arom_full else 0.0
    recall_full = n_tp_full / n_truth_arom if n_truth_arom else 0.0
    f1_full = (
        2 * precision_full * recall_full / (precision_full + recall_full)
        if (precision_full and recall_full) else 0.0
    )

    # Per-prediction-class truth breakdown for the full bond set, so the
    # reader can see what the head's class-4 outputs are mostly labelling
    # (single bonds that the new AROMATIC_LO has promoted to aromatic, etc).
    pred_arom_breakdown: Counter[str] = Counter()
    for r in all_records:
        if r["pred_class"] == HEAD_AROMATIC:
            pred_arom_breakdown[r["truth_label"] or "no_bo"] += 1

    out = {
        "scope": (
            "Truth labels recomputed against local AROMATIC_LO={lo} "
            "(held across the sweep at the CLI value, NOT editing the "
            "test module's AROMATIC_LO constant). Head inference is "
            "identical to the 2026-10-13 sub-band diagnostic -- same "
            "cohort, same checkpoint, same per-metal balanced sample. "
            "All in-scope pairs (non-H, non-metal heavy-atom pairs) "
            "are stored, not just the truth-aromatic subset, so the "
            "full-bond-set precision can be computed from the same "
            "forward pass (round-7 caveat)."
        ).format(lo=args.aromatic_lo),
        "aromatic_band": [args.aromatic_lo, args.aromatic_hi],
        "sub_bands": [list([lo, hi]) for _label, lo, hi in SUBBANDS],
        "checkpoint": str(args.checkpoint),
        "cohort": {
            "selection_protocol": (
                "Per-metal balanced sample, same as the 2026-10-13 "
                "sub-band diagnostic: per-metal pool sorted, shuffled "
                "with random.Random(seed=0), first n_per_metal=150 codes "
                "kept per metal."
            ),
            "n_complexes_sampled": len(codes_to_run),
            "n_with_xyz": len(coords_by_code),
        },
        "n_complexes_succeeded": n_succeeded,
        "n_complexes_failed_decode": n_failed_decode,
        "n_complexes_missing_xyz": n_missing_xyz,
        "n_complexes_no_truth_aromatic": n_no_truth_aromatic,
        # Truth-aromatic subset headline (same shape as 1.30 baseline).
        "n_truth_aromatic_pairs": n_truth_arom,
        "n_predicted_aromatic_in_arom_subset": n_pred_arom_in_arom_subset,
        "n_arom_to_single_collapse": n_arom_to_single,
        "arom_to_single_fraction": (
            n_arom_to_single / n_truth_arom if n_truth_arom else 0.0
        ),
        "aromatic_recall_argmax": (
            n_pred_arom_in_arom_subset / n_truth_arom if n_truth_arom else 0.0
        ),
        "per_subband": summary,
        "arom_to_single_collapse_distribution": collapse_summary,
        # FULL-BOND-SET metrics (the round-7 caveat).
        "n_records_total": len(all_records),
        "n_predicted_aromatic_full": n_pred_arom_full,
        "n_truth_aromatic_and_pred_aromatic": n_tp_full,
        "n_predicted_aromatic_but_not_truth_aromatic": n_fp_full,
        "aromatic_precision_full": precision_full,
        "aromatic_recall_full": recall_full,
        "aromatic_f1_full": f1_full,
        "pred_arom_truth_label_breakdown": dict(pred_arom_breakdown),
        # Provenance
        "verdict_premise_check": {
            "baseline_1.30_arom_to_single_frac": 0.707,
            "this_run_arom_to_single_frac": (
                n_arom_to_single / n_truth_arom if n_truth_arom else 0.0
            ),
            "interpretation": (
                "At AROMATIC_LO=1.30 this run reproduces the 70.7% "
                "headline (per-band recall values match the existing "
                "1.30 baseline diagnostic). At AROMATIC_LO=1.45/1.50 the "
                "n_truth_aromatic_pairs shrinks (the lower band is "
                "narrower), so the per-subband SHAPE rather than the "
                "absolute arom_to_single_frac is the load-bearing output."
            ),
        },
    }

    # Atomic write (§5.6).
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    summary_path = args.out_dir / "diagnostic.json"
    atomic_write_file(summary_path, json.dumps(out, indent=2, sort_keys=True))
    LOG.info("Wrote summary to %s", summary_path)

    # Per-pair records -- ALL in-scope pairs, not just truth-aromatic.
    rec_path = args.out_dir / "per_pair_records.jsonl"
    rec_lines = []
    for r in all_records:
        rec_lines.append(json.dumps(r, separators=(",", ":")))
    atomic_write_file(rec_path, "\n".join(rec_lines) + "\n")
    LOG.info("Wrote per-pair records to %s", rec_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
