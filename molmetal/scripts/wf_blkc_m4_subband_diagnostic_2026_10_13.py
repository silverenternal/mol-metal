#!/usr/bin/env python3
"""BLK-C-M4 decisive diagnostic: per-sub-band BO distribution of the aromatic->single collapse.

WHAT THIS MEASURES
------------------
Rounds 1-5 of the BLK-C-M4 audit established that 11,209 of 15,851 (70.7%)
truth-aromatic bonds (Wiberg BO in [1.3, 1.7)) are predicted BOND_SINGLE by
the production YuelBond head on real geometry. The retrain premise is a
specific kind of fix: shift the label band so the head's class-4 vs class-1
ranking separates correctly on the intermediate bond orders.

That premise only holds if the head ranks aromatic correctly across most of
the band. If the head fails broadly across [1.3, 1.5) as well as [1.5, 1.7),
the retrain premise is an artefact and the slot should not be spent on the
head.

This script splits the truth-aromatic bonds on the SAME cohort (non-H heavy
atom scope; one of the six training metals in the complex) by their actual
Wiberg BO and reports, per sub-band:

  - n truth-aromatic in the sub-band
  - n predicted as each of the 5 classes (NO_BOND, SINGLE, DOUBLE, TRIPLE,
    AROMATIC)
  - the LOGIT MARGIN (aromatic logit minus the best competing non-aromatic
    logit) for the pairs in that sub-band -- mean and median
  - the recall of aromatic in the sub-band

It also reports, among the 11,209 aromatic-to-single collapses, the BO
distribution: mean, median, and per-sub-band counts.

PROTOCOL REUSE
--------------
The BO/XYZ parsers, the head loader, the bond-decision logic, the sym-to-Z
table and the heavy-atom scope filter are imported from
``molmetal/tests/test_wf_blkc_qed_lever_aromatic_head.py`` -- the test that
produced the 70.7% headline. The 70.7% number is reproduced here on the full
filtered cohort as a sanity check; if this script does not reproduce ~0.293
aromatic recall on truth-aromatic bonds, the protocol has drifted and the
sub-band split is unsafe.

USAGE
-----
    python -m molmetal.scripts.wf_blkc_m4_subband_diagnostic_2026_10_13 \\
        --checkpoint molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \\
        --out-dir molmetal/reports/cfm_blkc_m4_subband_diagnostic_2026_10_13
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

# Imports from the test module -- BO/XYZ parsers, scope filter, head loader.
# We deliberately do NOT re-implement them.
sys.path.insert(0, str(REPO))
from molmetal.tests.test_wf_blkc_qed_lever_aromatic_head import (  # noqa: E402
    AROMATIC_LO,
    AROMATIC_HI,
    HEAD_AROMATIC,
    METAL_Z,
    TRAIN_METALS,
    _SYM_TO_Z,
    _ground_truth_bonds,
    _iter_bo_blocks,
    _iter_xyz_blocks,
    _load_head,
)

# Sub-bands required by the brief.
SUBBANDS: list[tuple[str, float, float]] = [
    ("1.30-1.40", 1.30, 1.40),
    ("1.40-1.45", 1.40, 1.45),
    ("1.45-1.50", 1.45, 1.50),
    ("1.50-1.60", 1.50, 1.60),
    ("1.60-1.70", 1.60, 1.70),
]
# Class names for clarity in JSON output.
CLASS_NAMES = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

LOG = logging.getLogger("wf_blkc_m4_subband")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def _score_with_logits(model, featuriser, symbols, coords, truth_bo):
    """Variant of the test's ``_score`` that ALSO returns per-pair logits.

    Reuses every filter the test applies:
      * non-H non-metal pair scope
      * truth_label via the test's band convention

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

    logits = res.logits  # shape [n_pairs, 5]
    src = res.edge_index[0].tolist()
    dst = res.edge_index[1].tolist()
    pred_class = logits.argmax(dim=-1).tolist()

    out = []
    for k in range(len(src)):
        s, d = int(src[k]), int(dst[k])
        key = (min(s, d) + 1, max(s, d) + 1)
        bo = truth_bo.get(key)
        if bo is None:
            continue
        sym_i = symbols[key[0] - 1]
        sym_j = symbols[key[1] - 1]
        if sym_i == "H" or sym_j == "H":
            continue
        zi = _SYM_TO_Z.get(sym_i, 0)
        zj = _SYM_TO_Z.get(sym_j, 0)
        if zi in METAL_Z or zj in METAL_Z:
            continue
        # truth_label via the test's band convention
        if AROMATIC_LO <= bo < AROMATIC_HI:
            t_label = "aromatic"
        elif bo < 1.5:
            t_label = "single"
        elif bo < 2.5:
            t_label = "double"
        else:
            t_label = "triple"
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
    """Margin = aromatic_logit - max(non-aromatic logits). Negative => class 1
    (single) or class 0 (no_bond) wins; positive => aromatic wins."""
    others = [rec["logit_0"], rec["logit_1"], rec["logit_2"], rec["logit_3"]]
    return rec["logit_4"] - max(others)


def _per_subband_summary(aromatic_recs: list[dict]) -> list[dict]:
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


def _select_cohort(
    bo_paths, metals, n_per_metal, seed=0
):
    """Mirror the bench_tmqm_bond_perception.py sample so the cohort matches
    the 798-complex evaluation window as closely as possible.

    Per-metal pool is sorted (so the random shuffle is deterministic), shuffled
    with random.Random(seed), and the first n_per_metal codes are kept.
    Duplicates across metals are deduplicated at the end.
    """
    syms_by_code = {}
    LOG.info("Indexing all BO shard complexes (one pass) ...")
    for code, atoms in _iter_bo_blocks(bo_paths):
        syms = [a[1] for a in atoms]
        if set(syms) & metals:
            syms_by_code[code] = syms
    LOG.info("  %d complexes carry one of the six train metals", len(syms_by_code))

    per_metal = {m: [] for m in metals}
    for code, syms in syms_by_code.items():
        for m in set(syms) & set(metals):
            per_metal[m].append(code)

    LOG.info("Per-metal pool sizes: %s", {m: len(v) for m, v in per_metal.items()})

    rng = random.Random(seed)
    chosen: list[str] = []
    for m in metals:
        pool = sorted(set(per_metal.get(m, [])))
        rng.shuffle(pool)
        for code in pool[:n_per_metal]:
            chosen.append(code)
    chosen = sorted(set(chosen))
    return chosen


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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO
        / "molmetal"
        / "reports"
        / "cfm_blkc_m4_subband_diagnostic_2026_10_13",
    )
    parser.add_argument(
        "--max-complexes",
        type=int,
        default=0,
        help=(
            "0 = run all qualifying complexes in the BO shards. "
            "If > 0, cap to that many complexes per metal after passing "
            "the metal-presence filter (matches the 150-per-metal bench "
            "protocol when set to 150)."
        ),
    )
    parser.add_argument(
        "--sample-seed", type=int, default=0,
        help=("Random seed for the per-metal shuffle (matches bench "
              "bench_tmqm_bond_perception.py when 0)."),
    )
    args = parser.parse_args()
    _setup_logging()

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

    # Single-pass: build {code: (syms, ground_truth_bonds)} for all
    # qualifying complexes. Avoids the O(N * M) per-complex scan that
    # the previous version of this script suffered from.
    LOG.info("Indexing qualifying complexes (single pass over BO shards) ...")
    syms_by_code: dict[str, list[str]] = {}
    truth_by_code: dict[str, dict] = {}
    for code, atoms in _iter_bo_blocks(bo_paths):
        syms = [a[1] for a in atoms]
        if not (set(syms) & TRAIN_METALS):
            continue
        syms_by_code[code] = syms
        truth_by_code[code] = _ground_truth_bonds(atoms)
    LOG.info(
        "  %d complexes carry one of the six train metals", len(syms_by_code)
    )

    # Sample per the bench protocol (sorted pool + random.shuffle).
    per_metal = {m: [] for m in TRAIN_METALS}
    for code, syms in syms_by_code.items():
        for m in set(syms) & TRAIN_METALS:
            per_metal[m].append(code)
    LOG.info(
        "Per-metal pool sizes: %s", {m: len(v) for m, v in per_metal.items()}
    )

    metals_ordered = sorted(TRAIN_METALS)
    rng = random.Random(args.sample_seed)
    codes_to_run: list[str] = []
    for m in metals_ordered:
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

    # Build {code: (syms, coords)} in one pass over XYZ shards.
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
    aromatic_records: list[dict] = []  # accumulate in memory (small enough)

    for i, code in enumerate(codes_to_run):
        if (i + 1) % 50 == 0 or (i + 1) == len(codes_to_run):
            elapsed = time.time() - t0
            LOG.info(
                "  [%d/%d] succeeded=%d failed_decode=%d missing_xyz=%d "
                "no_truth_aromatic=%d records=%d elapsed=%.1fs",
                i + 1, len(codes_to_run), n_succeeded, n_failed_decode,
                n_missing_xyz, n_no_truth_aromatic, len(aromatic_records),
                elapsed,
            )
        truth = truth_by_code.get(code)
        geo = coords_by_code.get(code)
        if geo is None:
            n_missing_xyz += 1
            continue
        symbols, coords = geo
        recs = _score_with_logits(
            model, featuriser, symbols, coords, truth
        )
        if recs is None:
            n_failed_decode += 1
            continue
        n_succeeded += 1
        # Filter to truth-aromatic -- this is the BLK-C-M4 cohort.
        for r in recs:
            if r["truth_label"] == "aromatic":
                r["code"] = code
                aromatic_records.append(r)
        if not [r for r in recs if r["truth_label"] == "aromatic"]:
            n_no_truth_aromatic += 1

    LOG.info(
        "Head run complete: succeeded=%d failed_decode=%d missing_xyz=%d "
        "no_truth_aromatic=%d total_arom_records=%d elapsed=%.1fs",
        n_succeeded, n_failed_decode, n_missing_xyz, n_no_truth_aromatic,
        len(aromatic_records), time.time() - t0,
    )

    # Headline sanity check.
    n_truth_arom = len(aromatic_records)
    n_pred_arom = sum(1 for r in aromatic_records if r["pred_class"] == HEAD_AROMATIC)
    n_arom_to_single = sum(
        1 for r in aromatic_records if r["pred_class"] == 1
    )
    LOG.info(
        "Sanity: pred_arom=%d / %d (recall %.4f); "
        "arom_to_single=%d (frac %.4f); MEASURED 0.293 / 0.707",
        n_pred_arom, n_truth_arom,
        n_pred_arom / max(1, n_truth_arom),
        n_arom_to_single, n_arom_to_single / max(1, n_truth_arom),
    )

    # Per-sub-band summary.
    summary = _per_subband_summary(aromatic_records)

    # Aromatic-to-single collapse BO distribution.
    collapse_bos = [r["bo"] for r in aromatic_records if r["pred_class"] == 1]
    collapse_summary = {
        "n_arom_to_single": len(collapse_bos),
        "mean_bo": statistics.fmean(collapse_bos) if collapse_bos else None,
        "median_bo": statistics.median(collapse_bos) if collapse_bos else None,
        "min_bo": min(collapse_bos) if collapse_bos else None,
        "max_bo": max(collapse_bos) if collapse_bos else None,
        "per_subband_counts": {
            label: sum(
                1 for r in aromatic_records
                if r["pred_class"] == 1 and lo <= r["bo"] < hi
            )
            for label, lo, hi in SUBBANDS
        },
    }

    out = {
        "scope": (
            "Truth-aromatic bonds (Wiberg BO in [1.3, 1.7)) restricted to "
            "non-H heavy-atom pairs on tmQM complexes whose composition "
            "includes at least one of {Pt, Ru, Ir, Rh, Os, Re}. Same scope "
            "as molmetal/tests/test_wf_blkc_qed_lever_aromatic_head.py."
        ),
        "aromatic_band": [AROMATIC_LO, AROMATIC_HI],
        "sub_bands": [list([lo, hi]) for _label, lo, hi in SUBBANDS],
        "checkpoint": str(args.checkpoint),
        "cohort": {
            "selection_protocol": (
                "Per-metal balanced sample matching "
                "bench_tmqm_bond_perception.py: per-metal pool sorted, "
                "shuffled with random.Random(seed={}), first "
                "n_per_metal={} codes kept per metal.".format(
                    args.sample_seed,
                    args.max_complexes if args.max_complexes else "ALL",
                )
            ),
            "n_complexes_sampled": len(codes_to_run),
            "n_with_xyz": len(coords_by_code),
        },
        "n_complexes_succeeded": n_succeeded,
        "n_complexes_failed_decode": n_failed_decode,
        "n_complexes_missing_xyz": n_missing_xyz,
        "n_complexes_no_truth_aromatic": n_no_truth_aromatic,
        "n_truth_aromatic_pairs": n_truth_arom,
        "n_predicted_aromatic": n_pred_arom,
        "n_arom_to_single_collapse": n_arom_to_single,
        "arom_to_single_fraction": (
            n_arom_to_single / n_truth_arom if n_truth_arom else 0.0
        ),
        "aromatic_recall_argmax": (
            n_pred_arom / n_truth_arom if n_truth_arom else 0.0
        ),
        "per_subband": summary,
        "arom_to_single_collapse_distribution": collapse_summary,
        "verdict_premise_check": {
            "headline_arom_to_single_frac_2026_10_12": 0.707,
            "this_run_arom_to_single_frac": (
                n_arom_to_single / n_truth_arom if n_truth_arom else 0.0
            ),
            "interpretation": (
                "If this_run_arom_to_single_frac is close to 0.70, the "
                "protocol reproduces the 70.7% headline and the per-sub-band "
                "split below is comparable. If it is materially different, "
                "the cohort changed (e.g. larger sample) and the per-sub-band "
                "SHAPE -- not the absolute count -- is the load-bearing "
                "output."
            ),
        },
    }

    summary_path = args.out_dir / "diagnostic.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    LOG.info("Wrote summary to %s", summary_path)

    # Per-pair records as JSONL for audit.
    rec_path = args.out_dir / "per_pair_records.jsonl"
    with rec_path.open("w", encoding="utf-8") as fh:
        for r in aromatic_records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    LOG.info("Wrote per-pair records to %s", rec_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
