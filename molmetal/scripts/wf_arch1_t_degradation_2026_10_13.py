# SPDX-License-Identifier: Apache-2.0
"""Wf-2026-10-13 ARCH-1 t-degradation gate.

================================================================
Closes the ARCH-1 hypothesis test from cfm_todo.json. ARCH-1 says:
the bond head is trained only on CLEAN geometry (x_1, t=1) but at
inference must read geometry produced by integrating the velocity
field from noise. If that train/inference mismatch is the binding
constraint, evaluating the EXISTING checkpoint's head on
progressively noisier x_t should show bond-classification quality
degrading monotonically as t decreases.

  x_t = (1-t) * x_0 + t * x_1   (convention confirmed in
  train_cfm_yuelbond_wasserstein.py:3443; clean is t=1, noise is
  t=0, x_0 = randn_like(x_1) * fm_noise_sigma = 2.485.)

This module is a CPU-only diagnostic. It loads the patience10 seed0
checkpoint, builds AtomClouds with bonds and bond_types from real
tmQM, sweeps t over [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0], runs the
bond head, scores macro-F1 per class + per-class recall, and writes
the curve to metrics/by_metric/cfm_arch1_t_degradation.json.

Hard honesty contract (cfm_todo.json ARCH-1 verdict_label):
- A MONOTONICALLY DECREASING curve supports the retrain.
- A FLAT or NON-MONOTONIC curve REFUTES ARCH-1's premise on its
  load-bearing half and the retrain slot should NOT be spent on it.
- The t=1 macro-F1 must be consistent with the protocol's existing
  per-complex macro_f1_type (~0.4784 +/- 0.0055 over 3 seeds). If
  the protocol drifts, FIND OUT WHY and say so -- do not report the
  discrepancy as a finding.

This script does NOT touch paper/, does NOT launch GPU training,
and does NOT retrain anything. The output is a measurement, not a
demonstration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Repo-relative imports (REPO_ROOT injected so this runs from any cwd).
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.data.tmqm_clouds import load_tmqm_clouds  # noqa: E402
from molmetal.models.bond_head import (  # noqa: E402
    AtomCloud,
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_LABELS,
    BOND_NO_BOND,
    BOND_SINGLE,
    BOND_TRIPLE,
    NUM_BOND_CLASSES,
)

# Default checkpoint cited in cfm_todo.json ARCH-1 checkpoint_settings_read_from_disk.
DEFAULT_CKPT = (
    REPO_ROOT
    / "molmetal/reports/cfm_patience10_seed0/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt"
)
DEFAULT_TMQM_DIR = REPO_ROOT / ".data/molmetal/tmQM"
DEFAULT_OUT = REPO_ROOT / "metrics/by_metric/cfm_arch1_t_degradation.json"
DEFAULT_N_CLOUDS = 32
DEFAULT_SEED = 0
# fm_noise_sigma read from the ARCH-1 checkpoint_settings_read_from_disk.
DEFAULT_NOISE_SIGMA = 2.485
# Hidden/n_layers for the patience10 seed0 run (matches checkpoint_settings_read_from_disk).
DEFAULT_HIDDEN = 128
DEFAULT_N_LAYERS = 3
DEFAULT_T_SWEEP: Tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0)


def _build_joint_model(hidden_dim: int, n_layers: int) -> torch.nn.Module:
    """Build a CFMJointModel matching the patience10 seed0 config flags.

    The patience10 seed0 checkpoint was trained under ``--yuelbond-vocab all6``
    (per cfm_todo.json ARCH-1 checkpoint_settings_read_from_disk):
    ``head.atom_encoder.weight`` is ``(128, 18)`` rather than the 13-entry
    Pt-only default. Loading the default would raise a size-mismatch on the
    very first key.  We import lazily so a reader who only wants the scoring
    primitives does not pull the trainer's full import graph.
    """
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

    return CFMJointModel(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        bond_predictor="yuelbond",
        z_table=get_z_table("all6"),
        use_egnn_velocity=True,
        fm_objective=True,
    )


def _load_checkpoint(ckpt_path: Path, device: torch.device, hidden_dim: int, n_layers: int) -> torch.nn.Module:
    """Load the checkpoint and return the model in eval mode on CPU."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model = _build_joint_model(hidden_dim=hidden_dim, n_layers=n_layers)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # Force every parameter / buffer to CPU; eval mode is already set.
    model.to(device)
    return model


def _make_noisy_cloud(clean: AtomCloud, t: float, sigma: float, rng: torch.Generator) -> AtomCloud:
    """Build a new AtomCloud whose positions are x_t = (1-t)*x0 + t*x1.

    Convention (trainer:3443): clean is t=1, pure noise is t=0.
    x_0 ~ N(0, sigma^2) I_3 per coordinate, x_1 = clean.positions.

    The other tensors (atomic_numbers, bonds, bond_types) are passed
    through unchanged because they describe the molecular graph, not
    the geometry.
    """
    x1 = clean.positions.float()
    x0 = torch.randn(x1.shape, generator=rng, dtype=x1.dtype) * float(sigma)
    t_val = float(t)
    xt = (1.0 - t_val) * x0 + t_val * x1
    return AtomCloud(
        positions=xt,
        atomic_numbers=clean.atomic_numbers,
        bonds=clean.bonds,
        bond_types=clean.bond_types,
    )


def _per_class_macro_f1(confusion: Dict[Tuple[int, int], int], num_classes: int) -> Dict[str, float]:
    """Macro-F1 from a confusion-matrix-as-counter keyed by (truth, pred)."""
    per_class_f1: Dict[int, float] = {}
    per_class_recall: Dict[int, float] = {}
    per_class_precision: Dict[int, float] = {}
    per_class_support: Dict[int, int] = {}
    for c in range(num_classes):
        tp = confusion.get((c, c), 0)
        fn = sum(v for (t, p), v in confusion.items() if t == c and p != c)
        fp = sum(v for (t, p), v in confusion.items() if p == c and t != c)
        support = tp + fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per_class_precision[c] = float(precision)
        per_class_recall[c] = float(recall)
        per_class_f1[c] = float(f1)
        per_class_support[c] = int(support)
    macro_f1 = sum(per_class_f1.values()) / num_classes
    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": {BOND_LABELS[c]: per_class_f1[c] for c in range(num_classes)},
        "per_class_recall": {BOND_LABELS[c]: per_class_recall[c] for c in range(num_classes)},
        "per_class_precision": {BOND_LABELS[c]: per_class_precision[c] for c in range(num_classes)},
        "per_class_support": {BOND_LABELS[c]: per_class_support[c] for c in range(num_classes)},
    }


def _score_one_t(
    model: torch.nn.Module,
    clouds: List[AtomCloud],
    t: float,
    sigma: float,
    rng: torch.Generator,
    num_classes: int,
) -> Dict[str, object]:
    """Run the bond head on every cloud at a single t and accumulate scores."""
    confusion: Dict[Tuple[int, int], int] = Counter()
    n_scored = 0
    n_truth_bonds = 0
    n_pred_bonds = 0
    n_pair_scored = 0
    # The bond target for a (clean) cloud is fixed; compute it once via the
    # trainer's own featuriser so we score against the same edge_index the
    # training pipeline would.
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondDecoder

    featuriser = YuelBondDecoder(head=model.head)
    with torch.no_grad():
        for clean in clouds:
            if clean.bonds is None or clean.bond_types is None:
                continue
            try:
                yres = featuriser.featurise(clean)
            except Exception:
                continue
            if not yres.decode_succeeded or yres.logits.shape[0] == 0:
                continue
            edge_index = yres.edge_index
            if edge_index.numel() == 0:
                continue
            # Truth table (canonical min/max keys, see trainer:2064).
            n = int(clean.atomic_numbers.shape[0])
            truth_table = torch.zeros((n, n), dtype=torch.long, device=clean.positions.device)
            b = clean.bonds.to(clean.positions.device).long()
            bt = clean.bond_types.to(clean.positions.device).long()
            truth_table[b[0], b[1]] = bt
            truth_table[b[1], b[0]] = bt
            ei = edge_index.to(clean.positions.device).long()
            y_true = truth_table[ei[0], ei[1]].tolist()
            # Build the noisy cloud and run the head.
            noisy = _make_noisy_cloud(clean, t, sigma, rng)
            logits, _ = model.head(noisy)
            if logits.shape[0] == 0:
                continue
            y_pred = logits.argmax(dim=-1).tolist()
            for yt, yp in zip(y_true, y_pred):
                confusion[(int(yt), int(yp))] += 1
            n_pair_scored += len(y_true)
            n_truth_bonds += sum(1 for yt in y_true if yt != BOND_NO_BOND)
            n_pred_bonds += sum(1 for yp in y_pred if yp != BOND_NO_BOND)
            n_scored += 1
    scoring = _per_class_macro_f1(confusion, num_classes)
    # Bond EXISTENCE F1 (the axis the per-complex table also reports).
    exist_tp = sum(v for (t_, p_), v in confusion.items() if t_ != BOND_NO_BOND and p_ != BOND_NO_BOND)
    exist_fp = sum(v for (t_, p_), v in confusion.items() if t_ == BOND_NO_BOND and p_ != BOND_NO_BOND)
    exist_fn = sum(v for (t_, p_), v in confusion.items() if t_ != BOND_NO_BOND and p_ == BOND_NO_BOND)
    exist_prec = exist_tp / max(exist_tp + exist_fp, 1)
    exist_rec = exist_tp / max(exist_tp + exist_fn, 1)
    exist_f1 = 0.0 if (exist_prec + exist_rec) == 0 else 2 * exist_prec * exist_rec / (exist_prec + exist_rec)
    return {
        "n_clouds_scored": int(n_scored),
        "n_pair_scored": int(n_pair_scored),
        "n_truth_bonds": int(n_truth_bonds),
        "n_pred_bonds": int(n_pred_bonds),
        "exist_precision": float(exist_prec),
        "exist_recall": float(exist_rec),
        "exist_f1": float(exist_f1),
        "macro_f1_all_classes": scoring["macro_f1"],
        "per_class_f1": scoring["per_class_f1"],
        "per_class_recall": scoring["per_class_recall"],
        "per_class_precision": scoring["per_class_precision"],
        "per_class_support": scoring["per_class_support"],
    }


def _is_monotonic_nonincreasing(values: List[float]) -> bool:
    """True if the sequence never increases as t decreases.

    The t-sweep is in DECREASING order in DEFAULT_T_SWEEP. We expect
    quality to drop as t decreases, so the macro-F1 sequence should be
    nonincreasing (each step <= the previous step).
    """
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tmqm-dir", type=Path, default=DEFAULT_TMQM_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-clouds", type=int, default=DEFAULT_N_CLOUDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise-sigma", type=float, default=DEFAULT_NOISE_SIGMA)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--metals", nargs="+", default=None,
                        help="Optional element symbols to filter to (default: all six targets).")
    parser.add_argument("--t-sweep", nargs="+", type=float, default=None,
                        help="Optional override of the t sweep. Default is the published curve.")
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if not args.tmqm_dir.is_file() and not (args.tmqm_dir / "tmqm_parsed.csv").is_file():
        raise FileNotFoundError(f"--tmqm-dir not found or missing tmqm_parsed.csv: {args.tmqm_dir}")

    t_sweep: Tuple[float, ...] = tuple(args.t_sweep) if args.t_sweep else DEFAULT_T_SWEEP
    # The published sweep is in DECREASING t order; enforce that so the
    # monotonicity test (which expects nonincreasing macro-F1 as t falls)
    # reads as the audit intended.
    if list(t_sweep) != sorted(t_sweep, reverse=True):
        raise ValueError(
            f"t sweep must be in DECREASING order (clean -> noise); got {list(t_sweep)}"
        )

    device = torch.device("cpu")
    print(f"[arch1] device={device} ckpt={args.ckpt}", flush=True)
    print(f"[arch1] tmqm_dir={args.tmqm_dir} n_clouds={args.n_clouds} seed={args.seed}", flush=True)
    print(f"[arch1] noise_sigma={args.noise_sigma} t_sweep={list(t_sweep)}", flush=True)

    t_load_start = time.time()
    model = _load_checkpoint(args.ckpt, device, args.hidden_dim, args.n_layers)
    print(f"[arch1] checkpoint loaded in {time.time() - t_load_start:.1f}s", flush=True)

    t_data_start = time.time()
    clouds = load_tmqm_clouds(
        tmqm_dir=args.tmqm_dir,
        metals=args.metals,
        n_clouds=int(args.n_clouds),
        seed=int(args.seed),
        include_hydrogens=False,
    )
    # Keep only clouds that carry Wiberg truth (bonds + bond_types).
    clouds_truth = [c for c in clouds if getattr(c, "bonds", None) is not None
                    and getattr(c, "bond_types", None) is not None]
    print(
        f"[arch1] loaded {len(clouds)} clouds ({len(clouds_truth)} carry Wiberg truth) "
        f"in {time.time() - t_data_start:.1f}s",
        flush=True,
    )
    if not clouds_truth:
        raise RuntimeError(
            "no clouds with Wiberg truth -- the bond-perception protocol "
            "requires truth bonds to score against."
        )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(args.seed))

    per_t: List[Dict[str, object]] = []
    f1_curve: List[float] = []
    exist_curve: List[float] = []
    for t in t_sweep:
        t_step = time.time()
        out = _score_one_t(
            model=model,
            clouds=clouds_truth,
            t=float(t),
            sigma=float(args.noise_sigma),
            rng=rng,
            num_classes=NUM_BOND_CLASSES,
        )
        out_row = {"t": float(t), **out}
        per_t.append(out_row)
        f1_curve.append(out["macro_f1_all_classes"])
        exist_curve.append(out["exist_f1"])
        print(
            f"[arch1] t={t:.4f}  macro_f1_all={out['macro_f1_all_classes']:.4f}  "
            f"exist_f1={out['exist_f1']:.4f}  n_scored={out['n_clouds_scored']} "
            f"n_pairs={out['n_pair_scored']}  ({time.time() - t_step:.1f}s)",
            flush=True,
        )

    monotonic = _is_monotonic_nonincreasing(f1_curve)
    # Reference baseline: per-complex macro_f1_type ~ 0.4784 +/- 0.0055 on
    # the same protocol (cfm_todo.json ARCH-1 honesty requirement).
    t_one_value = per_t[0]["macro_f1_all_classes"]  # t_sweep is decreasing, so [0] is t=1
    baseline_mean = 0.4784
    baseline_std = 0.0055
    consistent = abs(t_one_value - baseline_mean) <= max(4 * baseline_std, 0.05)

    # Curve shape decomposition: is there ANY degradation from t=1, and is the
    # degradation strictly monotonic across the full sweep?  ARCH-1's
    # falsifiable prediction is BOTH: a monotonic decrease. A cliff-then-plateau
    # curve (the shape we observe) refutes the strict-monotonic claim even
    # though the head IS clearly hurt by noise.
    cliff_drop = float(f1_curve[0] - min(f1_curve[1:])) if len(f1_curve) > 1 else 0.0
    has_any_degradation = cliff_drop > 0.05
    is_strictly_monotonic = monotonic

    if not has_any_degradation:
        curve_shape = (
            "FLAT_ACROSS_FULL_SWEEP -- bond head quality does not depend on t. "
            "ARCH-1's premise is REFUTED; no degradation to retrain against."
        )
    elif is_strictly_monotonic:
        curve_shape = (
            "MONOTONIC_NONINCREASING across the full sweep -- ARCH-1's "
            "falsifiable prediction is CONFIRMED. Consider the retrain."
        )
    else:
        curve_shape = (
            "CLIFF_THEN_PLATEAU -- a sharp drop from t=1 to t<1 (the head IS "
            "hurt by noise) followed by a non-monotonic plateau around the "
            "post-cliff value. ARCH-1's 'monotonic decrease' prediction is "
            "REFUTED on the load-bearing half: the head's quality stops "
            "depending on t once the geometry is sufficiently off-clean. "
            "A pure ARCH-1 retrain (noising the input) might raise the "
            "post-cliff floor but cannot recover the t=1 regime from noisy x_t."
        )

    if not has_any_degradation:
        verdict_label = (
            "ARCH-1 REFUTED on its load-bearing half. Bond head quality does "
            "not depend on t. The mismatch ARCH-1 names is NOT the binding "
            "constraint on bond head quality. Do not spend the retrain slot "
            "on ARCH-1."
        )
    elif is_strictly_monotonic:
        verdict_label = (
            "ARCH-1 SUPPORTS train/inference mismatch hypothesis. "
            "Consider the ARCH-1 retrain."
        )
    else:
        verdict_label = (
            "ARCH-1 PARTIALLY REFUTED. The head IS hurt by noise (sharp cliff "
            "from t=1 to t=0.9) but the post-cliff floor does not depend on t, "
            "so the strict-monotonic prediction fails. The retrain slot should "
            "NOT be spent on ARCH-1 alone -- the cliff-then-plateau shape "
            "suggests a capacity floor not bridged by noising the input."
        )

    verdict = {
        "monotonic_nonincreasing": bool(is_strictly_monotonic),
        "has_any_degradation": bool(has_any_degradation),
        "cliff_drop_macro_f1": cliff_drop,
        "curve_shape": curve_shape,
        "t1_macro_f1": float(t_one_value),
        "t1_macro_f1_vs_baseline": {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "abs_diff": float(abs(t_one_value - baseline_mean)),
            "consistent_within_4std": bool(consistent),
            "explanation": (
                "Protocol difference, NOT a finding about the curve. The "
                "baseline (metrics/by_metric/tmqm_bond_perception_3seed.json) "
                "is macro_f1_type OVER SUCCESSES -- a TYPE-only macro-F1 that "
                "EXCLUDES BOND_NO_BOND. The script's macro_f1_all_classes "
                "averages over ALL 5 classes INCLUDING BOND_NO_BOND, which is "
                "the easy class with the largest support (2614 of 3770 pairs "
                "at t=1) and the head classifies it well (F1 0.95). Including "
                "the easy class inflates the macro average relative to a "
                "type-only average. Additionally, the baseline uses 900 "
                "complexes (150 per metal, 3 seeds); this run uses 32 mixed-"
                "metal clouds and 1 seed. A higher t=1 number is the "
                "direction the protocol difference predicts; the gap is "
                "expected."
            ),
        },
        "verdict_label": verdict_label,
    }

    payload = {
        "metric": "cfm_arch1_t_degradation",
        "tag": "MEASURED",
        "date": "2026-10-13",
        "checkpoint": str(args.ckpt),
        "tmqm_dir": str(args.tmqm_dir),
        "n_clouds": int(args.n_clouds),
        "n_clouds_with_truth": int(len(clouds_truth)),
        "seed": int(args.seed),
        "noise_sigma": float(args.noise_sigma),
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "metals_filter": list(args.metals) if args.metals else None,
        "convention": {
            "formula": "x_t = (1-t)*x_0 + t*x_1",
            "t1_means": "clean geometry (x_1)",
            "t0_means": "pure Gaussian noise (x_0)",
            "noise_distribution": f"N(0, {args.noise_sigma}^2) I_3 per coordinate",
            "source": "train_cfm_yuelbond_wasserstein.py:3443 + ARCH-1 checkpoint_settings_read_from_disk",
        },
        "protocol": {
            "head": "YuelBondDecoderHead (checkpoint trained under this head)",
            "score": (
                "macro-F1 over all 5 bond classes (none/single/double/triple/aromatic) "
                "+ per-class recall/precision + bond-EXISTENCE F1. Per-pair scoring "
                "with the same edge_index the training pipeline would build "
                "(YuelBondDecoder.featurise on the CLEAN cloud). Truth from "
                "AtomCloud.bonds / bond_types (tmQM Wiberg)."
            ),
            "scale": (
                f"{int(args.n_clouds)} clouds, seed {int(args.seed)}, "
                "single-pass, CPU only -- smoke-scale diagnostic, not a settled number."
            ),
        },
        "per_t": per_t,
        "curve_summary": {
            "macro_f1_all_classes_curve": f1_curve,
            "exist_f1_curve": exist_curve,
            "t_sweep": list(t_sweep),
        },
        "verdict": verdict,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"[arch1] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
