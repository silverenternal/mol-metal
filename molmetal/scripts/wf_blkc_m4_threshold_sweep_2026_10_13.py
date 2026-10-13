#!/usr/bin/env python3
"""BLK-C-M4 BOND-DECISION THRESHOLD sweep on the existing seed0 checkpoint.

WHAT THIS MEASURES (vs the existing per-band diagnostic)
--------------------------------------------------------
The 2026-10-13 sub-band diagnostic at
``molmetal/reports/cfm_blkc_m4_subband_diagnostic_2026_10_13/`` and the
follow-up band-reconciliation sweep at
``molmetal/reports/cfm_blkc_m4_band_sweep_1.30_2026_10_13/`` both use the
5-CLASS argmax rule to derive per-pair predictions, so the head's
intrinsic aromatic prediction count is locked at 5539 pairs on the
900-complex cohort, INDEPENDENT of the AROMATIC_LO truth boundary.

The deployed ``bond_decision`` rule is ``threshold`` at 0.5
(see ``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::bond_decision``
lines 235-241 and the ARCH-1 ``checkpoint_settings_read_from_disk``
convention cited in
``metrics/by_metric/cfm_arch1_t_degradation.json::convention``): a pair
is bonded iff ``1 - softmax(logits)[0] >= threshold``; if bonded, the
type is the argmax over the NON-zero classes (1..4).  This decouples
"is there a bond" from "which bond" and so CALIBRATES the aromatic count
on the softmax of the BOND_NO_BOND column rather than the raw argmax.

The round-9 finding (per ``cfm_todo.json::round9_finding_2026_10_13``) is
that ``n_predicted_aromatic_full = 5539`` is INVARIANT under AROMATIC_LO
change.  That finding is about the argmax axis, where band change cannot
move predictions; the deployed threshold rule is a DIFFERENT axis the
band sweep did not vary.  This script sweeps bond_decision threshold and
answers: does a calibrated threshold raise F1 above the 0.4338 baseline
(reference: AROMATIC_LO=1.30 argmax diagnostic, recall 0.2872,
precision_full 0.8868, F1 0.4338, n_predicted_aromatic_full 5539)?

PROTOCOL REUSE
--------------
BO/XYZ parsers, head loader, sym-to-Z table, heavy-atom scope filter and
the in-scope truth_label mapping are imported from
``molmetal/tests/test_wf_blkc_qed_lever_aromatic_head.py`` -- the same
pattern as ``wf_blkc_m4_band_sweep_2026_10_13.py``.  We override the
prediction rule locally (do NOT edit the test module's constants).

USAGE
-----
    .venv/bin/python molmetal/scripts/wf_blkc_m4_threshold_sweep_2026_10_13.py \\
        --max-complexes 150 --aromatic-lo 1.30 --sample-seed 0 \\
        --bond-thresholds 0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 \\
        --out-dir molmetal/reports/cfm_blkc_m4_threshold_sweep_2026_10_13
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
# moving parameter is the AROMATIC_LO (passed via CLI).  The test module's
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

CLASS_NAMES = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

LOG = logging.getLogger("wf_blkc_m4_threshold_sweep")

# Default grid per the brief: must include 0.50 (deployed) so the
# deployed-vs-sweep comparison is direct.
DEFAULT_THRESHOLDS = [
    0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
]


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
    ``molmetal/data/tmqm_clouds.py:174-179``).
    """
    if aromatic_lo <= bo < aromatic_hi:
        return "aromatic"
    if bo < 1.5:
        return "single"
    if bo < 2.5:
        return "double"
    return "triple"


def _score_all(model, featuriser, symbols, coords, truth_bo, aromatic_lo):
    """Per-pair forward pass + logit storage, ALL in-scope pairs (not just
    truth-aromatic).  Reuses the heavy-atom scope filter and the truth
    label mapping of the band-sweep script.

    Returns
    -------
    list of dict
        One entry per in-scope pair with keys: key, symbol_i, symbol_j,
        bo, truth_label, p_bonded, p_class_0..p_class_4 (softmax
        probabilities -- the threshold rule operates on p_bonded =
        1 - p_class_0 and the type argmax over p_class_1..p_class_4).
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
    probs = torch.softmax(logits.float(), dim=-1)
    p_bonded = 1.0 - probs[:, 0]  # column 0 is BOND_NO_BOND

    src = res.edge_index[0].tolist()
    dst = res.edge_index[1].tolist()

    out = []
    for k in range(len(src)):
        s, d = int(src[k]), int(dst[k])
        key = (min(s, d) + 1, max(s, d) + 1)
        bo = truth_bo.get(key)
        if bo is None:
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
                "p_bonded": float(p_bonded[k]),
                "p_class_0": float(probs[k, 0]),
                "p_class_1": float(probs[k, 1]),
                "p_class_2": float(probs[k, 2]),
                "p_class_3": float(probs[k, 3]),
                "p_class_4": float(probs[k, 4]),
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
            "p_bonded": float(p_bonded[k]),
            "p_class_0": float(probs[k, 0]),
            "p_class_1": float(probs[k, 1]),
            "p_class_2": float(probs[k, 2]),
            "p_class_3": float(probs[k, 3]),
            "p_class_4": float(probs[k, 4]),
        })
    return out


def _apply_threshold(records: list[dict], threshold: float) -> list[int]:
    """Apply the DEPLOYED bond_decision="threshold" rule at the given
    threshold to a pre-computed list of in-scope pair records (which carry
    p_bonded + p_class_1..p_class_4 softmax probabilities).

    Mirrors ``yuelbond_decoder.bond_decision`` lines 235-241:

        p_bonded = 1.0 - prob[:, 0]
        types = prob[:, 1:].argmax(dim=-1) + 1
        return [int(types[k]) if float(p_bonded[k]) >= threshold else 0 ...]

    Returns the per-pair predicted class list (0=NO_BOND, 1..4=BOND types).
    """
    out = []
    for r in records:
        if r["p_bonded"] >= threshold:
            # argmax over the four BOND types (1..4) -> 0-indexed offset + 1
            p_types = (
                r["p_class_1"],
                r["p_class_2"],
                r["p_class_3"],
                r["p_class_4"],
            )
            types_idx = max(range(4), key=lambda i: p_types[i])
            out.append(int(types_idx) + 1)
        else:
            out.append(0)
    return out


def _per_threshold_metrics(
    records: list[dict],
    pred_classes: list[int],
    aromatic_lo: float,
    threshold: float,
) -> dict:
    """Compute the headline metrics at one threshold:

      - n_predicted_aromatic_full (pred_class == 4 across full bond set)
      - aromatic_recall at the AROMATIC_LO..AROMATIC_HI eval band
      - aromatic_precision_full on the FULL bond set
      - aromatic_f1_full
      - n_arom_to_single_collapse (truth-aromatic subset predicted SINGLE)
    """
    n_pred_arom_full = sum(1 for c in pred_classes if c == HEAD_AROMATIC)
    n_truth_arom = sum(1 for r in records if r["truth_label"] == "aromatic")
    n_tp_full = sum(
        1 for r, c in zip(records, pred_classes)
        if r["truth_label"] == "aromatic" and c == HEAD_AROMATIC
    )
    n_fp_full = n_pred_arom_full - n_tp_full
    precision_full = n_tp_full / n_pred_arom_full if n_pred_arom_full else 0.0
    recall_full = n_tp_full / n_truth_arom if n_truth_arom else 0.0
    f1_full = (
        2 * precision_full * recall_full / (precision_full + recall_full)
        if (precision_full and recall_full) else 0.0
    )

    # arom-to-single collapse on the truth-aromatic subset (same
    # definition as the round-6 sub-band diagnostic).
    n_arom_to_single = sum(
        1 for r, c in zip(records, pred_classes)
        if r["truth_label"] == "aromatic" and c == 1
    )
    pred_arom_breakdown: Counter[str] = Counter()
    for r, c in zip(records, pred_classes):
        if c == HEAD_AROMATIC:
            pred_arom_breakdown[r["truth_label"] or "no_bo"] += 1
    pred_class_counter: Counter[int] = Counter(pred_classes)
    return {
        "threshold": threshold,
        "n_records_total": len(records),
        "n_predicted_aromatic_full": n_pred_arom_full,
        "n_truth_aromatic_pairs": n_truth_arom,
        "n_truth_aromatic_and_pred_aromatic": n_tp_full,
        "n_predicted_aromatic_but_not_truth_aromatic": n_fp_full,
        "aromatic_precision_full": precision_full,
        "aromatic_recall_full": recall_full,
        "aromatic_f1_full": f1_full,
        "n_arom_to_single_collapse": n_arom_to_single,
        "arom_to_single_fraction": (
            n_arom_to_single / n_truth_arom if n_truth_arom else 0.0
        ),
        "pred_arom_truth_label_breakdown": dict(pred_arom_breakdown),
        "pred_class_counts": {
            CLASS_NAMES[c]: int(pred_class_counter.get(c, 0))
            for c in range(5)
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


def _parse_thresholds(raw: str) -> list[float]:
    """Parse a comma-separated list of bond_decision thresholds into a
    strictly ascending list of floats in (0, 1).

    Reject duplicates (would silently double-write the same dir) and any
    value outside (0, 1) (the bond_decision rule uses p_bonded >= T, so
    0 and 1 are degenerate).
    """
    out: list[float] = []
    seen: set[float] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError as exc:
            raise SystemExit(
                f"could not parse --bond-thresholds token {tok!r}: {exc}"
            ) from exc
        if not (0.0 < v < 1.0):
            raise SystemExit(
                f"--bond-thresholds value {v} is outside (0, 1); the "
                f"bond_decision rule uses p_bonded >= threshold so 0 and "
                f"1 are degenerate."
            )
        if v in seen:
            raise SystemExit(
                f"--bond-thresholds has duplicate value {v}; deduplicate "
                f"the list to avoid overwriting the same output dir."
            )
        seen.add(v)
        out.append(v)
    out.sort()
    if not out:
        raise SystemExit("--bond-thresholds must contain at least one value")
    return out


def _dir_for_threshold(out_dir: Path, threshold: float) -> Path:
    """Stable, sortable per-threshold sub-dir name like 'threshold_0.50'."""
    return out_dir / f"threshold_{threshold:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO
        / "molmetal"
        / "reports"
        / "cfm_patience10_seed0"
        / "checkpoints"
        / "cfm_yuelbond_wasserstein_seed0_last.pt",
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
    parser.add_argument(
        "--bond-thresholds",
        type=str,
        default=",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLDS),
        help=(
            "Comma-separated list of bond_decision thresholds to sweep. "
            "Must be strictly ascending, unique, in (0, 1). Default: "
            + ",".join(f"{t:.2f}" for t in DEFAULT_THRESHOLDS)
        ),
    )
    parser.add_argument(
        "--bond-threshold",
        type=float,
        default=None,
        help=(
            "Single-threshold shorthand: equivalent to passing "
            "--bond-thresholds <value>. Useful for one-shot runs."
        ),
    )
    args = parser.parse_args()
    _setup_logging()

    if args.bond_threshold is not None:
        # The single-flag shorthand takes precedence over the list.
        args.bond_thresholds = f"{args.bond_threshold:.2f}"
    thresholds = _parse_thresholds(args.bond_thresholds)

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

    # Per-metal balanced sample (same protocol as band-sweep script).
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
    all_records: list[dict] = []  # ALL in-scope records, with probs

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

    # ---- Sweep thresholds ----
    per_threshold: dict[str, dict] = {}
    for T in thresholds:
        pred_classes = _apply_threshold(all_records, T)
        m = _per_threshold_metrics(all_records, pred_classes, args.aromatic_lo, T)
        per_threshold[f"{T:.2f}"] = m

        # Atomic write (§5.6): per-threshold sub-dir.
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
        sub_dir = _dir_for_threshold(args.out_dir, T)
        sub_dir.mkdir(parents=True, exist_ok=True)

        # diagnostic.json for this threshold
        diag_path = sub_dir / "diagnostic.json"
        atomic_write_file(diag_path, json.dumps(m, indent=2, sort_keys=True))

        # per_pair_records.jsonl with the per-pair decision at this threshold
        rec_lines = []
        for r, c in zip(all_records, pred_classes):
            rec_lines.append(json.dumps({
                "code": r["code"],
                "key": r["key"],
                "symbol_i": r["symbol_i"],
                "symbol_j": r["symbol_j"],
                "bo": r["bo"],
                "truth_label": r["truth_label"],
                "p_bonded": r["p_bonded"],
                "p_class_4": r["p_class_4"],
                "pred_class": int(c),
                "pred_is_aromatic": int(c) == HEAD_AROMATIC,
                "threshold": T,
            }, separators=(",", ":")))
        atomic_write_file(
            sub_dir / "per_pair_records.jsonl",
            "\n".join(rec_lines) + "\n",
        )

    # ---- Headline ----
    f1_at_050 = per_threshold.get("0.50", {}).get("aromatic_f1_full", 0.0)
    f1_values = [
        (T, per_threshold[f"{T:.2f}"]["aromatic_f1_full"])
        for T in thresholds
    ]
    max_T, max_f1 = max(f1_values, key=lambda kv: kv[1])
    f1_lift_vs_deployed = max_f1 - f1_at_050
    best_count = per_threshold[f"{max_T:.2f}"]["n_predicted_aromatic_full"]
    n_truth_arom = sum(1 for r in all_records if r["truth_label"] == "aromatic")
    # Reference argmax-baseline 5539 (from band-sweep AROMATIC_LO=1.30).
    ARGMAX_AROM_COUNT_REF = 5539
    # "Calibrated" if any threshold brings the aromatic prediction count
    # within +/-10% of the truth count (i.e. the head stops grossly
    # over- or under-calling aromatic).
    tolerance = 0.10
    if n_truth_arom == 0:
        calibrates = False
    else:
        lower = (1.0 - tolerance) * n_truth_arom
        upper = (1.0 + tolerance) * n_truth_arom
        calibrates = any(
            lower <= per_threshold[f"{T:.2f}"]["n_predicted_aromatic_full"] <= upper
            for T in thresholds
        )

    headline = {
        "f1_at_deployed_0.5": f1_at_050,
        "max_f1_in_sweep": max_f1,
        "threshold_at_max_f1": max_T,
        "f1_lift_vs_deployed": f1_lift_vs_deployed,
        "calibrates_5539_count": calibrates,
        "best_threshold_arom_count": best_count,
        "n_truth_aromatic_pairs": n_truth_arom,
        "argmax_baseline_count_ref": ARGMAX_AROM_COUNT_REF,
        "argmax_baseline_f1_ref": 0.43382645175535434,
        "argmax_baseline_precision_ref": 0.8868026719624481,
        "argmax_baseline_recall_ref": 0.2871507073541447,
    }

    # ---- Master summary diagnostic ----
    master = {
        "scope": (
            "Per-pair softmax probabilities (p_class_0..p_class_4) from "
            "the same 150-per-metal 900-complex cohort and the same "
            "seed0 checkpoint the band-sweep used.  The deployed "
            "bond_decision='threshold' rule (cf. "
            "yuelbond_decoder.py::bond_decision lines 235-241) is applied "
            "to those probabilities for each threshold in the grid; the "
            "round-9 finding (n_predicted_aromatic_full = 5539 invariant "
            "under AROMATIC_LO change) is about the 5-class argmax axis, "
            "which this script does NOT use.  Round-9 invariant-count is "
            "held as argmax_baseline_count_ref=5539 for direct comparison."
        ),
        "aromatic_band": [args.aromatic_lo, args.aromatic_hi],
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
        "n_records_total": len(all_records),
        "decision_rule": (
            "bond_decision='threshold' (the DEPLOYED rule, NOT the "
            "5-class argmax used by the band-sweep script).  Pair is "
            "bonded iff 1 - softmax(logits)[BOND_NO_BOND] >= threshold; "
            "if bonded, type = argmax over classes 1..4."
        ),
        "thresholds_swept": [float(T) for T in thresholds],
        "deployed_threshold": 0.5,
        "per_threshold": per_threshold,
        "headline": headline,
        "verdict_premise_check": {
            "band_sweep_argmax_baseline_5539": (
                "The band-sweep diagnostic at AROMATIC_LO=1.30 reports "
                "n_predicted_aromatic_full=5539, F1=0.4338, recall=0.2872, "
                "precision=0.8868 -- this is the 5-class argmax rule.  The "
                "deployed bond_decision='threshold' rule is a different "
                "decision rule, so its predictions at any threshold T != "
                "argmax can (and usually do) differ in count.  The round-9 "
                "INVARIANT finding applies to the argmax axis only."
            ),
            "this_run_n_predicted_aromatic_full_at_0.50": (
                per_threshold.get("0.50", {}).get(
                    "n_predicted_aromatic_full", 0
                )
            ),
        },
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file
    master_path = args.out_dir / "diagnostic.json"
    atomic_write_file(master_path, json.dumps(master, indent=2, sort_keys=True))
    LOG.info("Wrote master summary to %s", master_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())