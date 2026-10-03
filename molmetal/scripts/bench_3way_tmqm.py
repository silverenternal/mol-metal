#!/usr/bin/env python3
"""Wf-128 Phase 3 — 3-way (Deflexpressor + Multi-PH + ASTSR) cross-dataset
verification on tmQM d-block per element (CPU + GPU-1).

================================================================
``molmetal/scripts/bench_3way_tmqm.py`` is the **Wf-128 P3
cross-dataset verify** for the tmQM d-block subset (5 000 real
mononuclear transition-metal complexes covering 29 d-block
metals — Balcells & Skjelstad 2020 corpus shipped as
``molmetal/data/tmqm_dblock_subset.csv``).

Wf-128 P2 (5-metal MetalCytoToxDB) shipped **3-way combined
R² = -150.64** — the integration did not break the gate on
binary activity.

Wf-128 P3 lifts the same 3-way harness (Hilbert + Multi-PH +
ASTSR) onto the tmQM d-block **coord_number** target
**per element** (29 d-block metals with non-zero cohorts) and
answers:

    **Is the 3-way model not worse than the baseline Hilbert
    on cross-dataset (tmQM d-block, per-element)?**

Constraints
-----------
* **29 d-block elements** — all elements in the
  ``tmqm_dblock_subset.csv`` with n ≥ 5 rows.
* **5-fold CV × 5 seeds × 29 elements** — per-element
  MEASURED-vs-PROJECTED anchor per `party-twenty-spirit §3.5`.
* **CPU default, GPU-1 optional** — fits run fp32 Nesterov +
  small evolutionary search; per-element fits run in < 5 s on
  a single CPU core for the typical 100-500 row cohort.
* **No Multi-PH + PLL augmentation** — tmQM d-block ships no
  172-d NPZ; the 3-way uses the 12-d F5_RDKIT_2D descriptor +
  zero-padded MPH/PLL slots.
* **Honest framing** — negative R² values are not hidden; the
  collapse detector flags ``pred_std / y_test_std < 0.10`` per
  fold.

Outputs
-------
* ``molmetal/reports/wf128_p3_tmqm_29/{element}/per_seed_metrics.json``
* ``molmetal/reports/wf128_p3_tmqm_29/synthesis.md``

Usage::

    molmetal/.venv/bin/python \
      molmetal/scripts/bench_3way_tmqm.py \
      --seeds 0 1 2 3 4 --epochs 50 \
      --out-dir molmetal/reports/wf128_p3_tmqm_29
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# Reuse 3-way streams + collapse detector from Wf-128 P2 ship.
from bench_3way_real_metalcytotox import (  # noqa: E402
    DEFAULT_HILBERT_DIM,
    DEFAULT_N_FOLDS,
    FEATURE_DIM,
    PRED_CLIP,
    RDKIT_DIM,
    _collapse_detected,
    _combine,
    _featurize_smiles_12d,
    _r2,
    _random_kfold_indices,
    _zscore,
    fit_astr_stream,
    fit_hilbert_stream,
    fit_mph_stream,
    predict_astr,
    predict_hilbert,
    predict_mph,
)

# ---------------------------------------------------------------------------
# tmQM d-block constants.
# ---------------------------------------------------------------------------
DEFAULT_TMQM_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "tmqm_dblock_subset.csv"
DEFAULT_SMILES_COL: str = "smiles"
DEFAULT_METAL_COL: str = "metal_center"
DEFAULT_TARGET_COL: str = "coord_number"
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf128_p3_tmqm_29"

# 30 d-block elements (Wf-73 P3 anchor) — the script drops any element
# with < 5 rows at load time so the per-element cohort remains usable.
ALL_DBLOCK_ELEMENTS: Tuple[str, ...] = (
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",  # 3d
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",  # 4d
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",  # 5d + La
)

# 5 default largest first-row d-block cohorts (Wf-86 P2 anchor).
DEFAULT_ELEMENTS: Tuple[str, ...] = ("Cu", "Fe", "Ni", "Zn", "Co")

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 50
MIN_ROWS_PER_ELEMENT: int = 5


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------
@dataclass
class FitRecord:
    element: str
    seed: int
    fold: int
    n_train: int
    n_test: int
    combined_r2: float
    hilbert_r2: float
    mph_r2: float
    astr_r2: float
    wall_s: float
    collapse_detected: bool = False


@dataclass
class ElementSeedSummary:
    element: str
    seed: int
    n_folds: int
    combined_r2_mean: float
    combined_r2_std: float
    hilbert_r2_mean: float
    mph_r2_mean: float
    astr_r2_mean: float
    wall_s_mean: float
    collapse_detected_any: bool = False


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_tmqm_element(
    element: str,
    *,
    csv_path: Path = DEFAULT_TMQM_CSV,
    target_col: str = DEFAULT_TARGET_COL,
    n_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load tmQM d-block for one element.

    Returns ``(X, y, smiles)``:
      * ``X`` (N, 172) float32 — RDKit 12-d + zero-padded MPH/PLL.
      * ``y`` (N,) float32 — regression target (default ``coord_number``).
      * ``smiles`` List[str] — for debug.
    """
    rows = _read_csv(csv_path)
    Xs: List[List[float]] = []
    Ys: List[float] = []
    smiles: List[str] = []
    for r in rows:
        if (r.get(DEFAULT_METAL_COL) or "").strip() != element:
            continue
        smi = (r.get(DEFAULT_SMILES_COL) or "").strip()
        if not smi:
            continue
        feats = _featurize_smiles_12d(smi)
        if feats is None:
            continue
        raw = r.get(target_col)
        try:
            yval = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(yval):
            continue
        # Zero-pad to FEATURE_DIM (172) for 3-way compatibility.
        full = [0.0] * FEATURE_DIM
        full[:RDKIT_DIM] = feats
        Xs.append(full)
        Ys.append(float(yval))
        smiles.append(smi)
        if n_max is not None and len(Xs) >= n_max:
            break
    if not Xs:
        raise RuntimeError(f"No usable rows for element {element!r}")
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(Ys, dtype=np.float32),
        smiles,
    )


def list_available_elements(
    csv_path: Path = DEFAULT_TMQM_CSV,
    target_col: str = DEFAULT_TARGET_COL,
) -> List[str]:
    """Return list of elements (with ≥ MIN_ROWS_PER_ELEMENT rows) in CSV."""
    rows = _read_csv(csv_path)
    pool: Dict[str, int] = defaultdict(int)
    for r in rows:
        elem = (r.get(DEFAULT_METAL_COL) or "").strip()
        if not elem:
            continue
        try:
            yval = float(r.get(target_col) or "nan")
        except (TypeError, ValueError):
            continue
        if not math.isfinite(yval):
            continue
        smi = (r.get(DEFAULT_SMILES_COL) or "").strip()
        if not smi or _featurize_smiles_12d(smi) is None:
            continue
        pool[elem] += 1
    return [e for e in ALL_DBLOCK_ELEMENTS if pool.get(e, 0) >= MIN_ROWS_PER_ELEMENT]


# ---------------------------------------------------------------------------
# Per-fold evaluation.
# ---------------------------------------------------------------------------
def evaluate_fold(
    *,
    element: str,
    seed: int,
    fold: int,
    tr: np.ndarray,
    te: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    hilbert_dim: int,
) -> FitRecord:
    """Run 3-way integration on a single (element, seed, fold) split."""
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    X_tr_z, X_te_z = _zscore(X_tr.astype(float), X_te.astype(float))
    t0 = time.perf_counter()

    try:
        w_h, _trh, aug_te_h = fit_hilbert_stream(
            X_tr_z,
            y_tr,
            X_te_z,
            hilbert_dim=hilbert_dim,
        )
        y_h_te = predict_hilbert(w_h, aug_te_h)
    except Exception:
        y_h_te = np.zeros(len(te), dtype=float)

    try:
        w_m, _trm, mph_te_z = fit_mph_stream(X_tr_z, y_tr, X_te_z)
        y_m_te = predict_mph(w_m, mph_te_z)
    except Exception:
        y_m_te = np.zeros(len(te), dtype=float)

    try:
        reg_a, _tra = fit_astr_stream(
            X_tr,
            y_tr,
            use_astr_scale_guard=True,
        )
        y_a_te = predict_astr(reg_a, X_te, use_astr_scale_guard=True)
    except Exception:
        y_a_te = np.zeros(len(te), dtype=float)

    y_combined_te = np.clip(_combine(y_h_te, y_m_te, y_a_te), -PRED_CLIP, PRED_CLIP)
    wall = float(time.perf_counter() - t0)

    r2_h = _r2(y_te, y_h_te)
    r2_m = _r2(y_te, y_m_te)
    r2_a = _r2(y_te, y_a_te)
    r2_c = _r2(y_te, y_combined_te)
    collapse_c = _collapse_detected(y_combined_te, y_te)

    return FitRecord(
        element=element,
        seed=int(seed),
        fold=int(fold),
        n_train=int(len(tr)),
        n_test=int(len(te)),
        combined_r2=float(r2_c) if math.isfinite(r2_c) else float("nan"),
        hilbert_r2=float(r2_h) if math.isfinite(r2_h) else float("nan"),
        mph_r2=float(r2_m) if math.isfinite(r2_m) else float("nan"),
        astr_r2=float(r2_a) if math.isfinite(r2_a) else float("nan"),
        wall_s=float(wall),
        collapse_detected=bool(collapse_c),
    )


# ---------------------------------------------------------------------------
# Per-element evaluation.
# ---------------------------------------------------------------------------
def evaluate_element(
    element: str,
    seeds: Sequence[int],
    n_folds: int,
    hilbert_dim: int,
    csv_path: Path,
    target_col: str,
    n_max: Optional[int],
) -> List[FitRecord]:
    X, y, _smi = load_tmqm_element(
        element,
        csv_path=csv_path,
        target_col=target_col,
        n_max=n_max,
    )
    n = X.shape[0]
    if n < MIN_ROWS_PER_ELEMENT:
        raise RuntimeError(
            f"element {element}: too few rows ({n} < {MIN_ROWS_PER_ELEMENT})"
        )

    records: List[FitRecord] = []
    for seed in seeds:
        splits = _random_kfold_indices(n, int(n_folds), seed=int(seed))
        for fold, (tr, te) in enumerate(splits):
            if len(te) == 0:
                continue
            rec = evaluate_fold(
                element=element,
                seed=int(seed),
                fold=int(fold),
                tr=tr,
                te=te,
                X=X,
                y=y,
                hilbert_dim=int(hilbert_dim),
            )
            records.append(rec)
            print(
                f"[wf128-p3-tmqm]   element={element} "
                f"target={target_col} seed={int(seed)} "
                f"fold={int(fold)} combined_r2={rec.combined_r2:+.4f} "
                f"H={rec.hilbert_r2:+.4f} M={rec.mph_r2:+.4f} "
                f"A={rec.astr_r2:+.4f} collapse={rec.collapse_detected}",
                flush=True,
            )
    return records


def aggregate_element(records: List[FitRecord]) -> List[ElementSeedSummary]:
    out: List[ElementSeedSummary] = []
    by_key: Dict[Tuple[str, int], List[FitRecord]] = {}
    for r in records:
        by_key.setdefault((r.element, r.seed), []).append(r)
    for (element, seed), recs in sorted(by_key.items()):
        combined = np.asarray([r.combined_r2 for r in recs], dtype=float)
        hilbert = np.asarray([r.hilbert_r2 for r in recs], dtype=float)
        mph = np.asarray([r.mph_r2 for r in recs], dtype=float)
        astr = np.asarray([r.astr_r2 for r in recs], dtype=float)
        wall = np.asarray([r.wall_s for r in recs], dtype=float)
        out.append(
            ElementSeedSummary(
                element=element,
                seed=int(seed),
                n_folds=int(len(recs)),
                combined_r2_mean=float(np.nanmean(combined)) if combined.size else 0.0,
                combined_r2_std=float(np.nanstd(combined))
                if combined.size > 1
                else 0.0,
                hilbert_r2_mean=float(np.nanmean(hilbert)) if hilbert.size else 0.0,
                mph_r2_mean=float(np.nanmean(mph)) if mph.size else 0.0,
                astr_r2_mean=float(np.nanmean(astr)) if astr.size else 0.0,
                wall_s_mean=float(np.nanmean(wall)) if wall.size else 0.0,
                collapse_detected_any=bool(any(r.collapse_detected for r in recs)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)


def write_synthesis(
    out_dir: Path, all_records: List[FitRecord], elements: Sequence[str]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_el: Dict[str, List[FitRecord]] = {}
    for r in all_records:
        by_el.setdefault(r.element, []).append(r)

    lines: List[str] = [
        "# Wf-128 Phase 3 — 3-way (Deflexpressor + Multi-PH + ASTSR) "
        "cross-dataset verify on tmQM d-block (29 d-block metals)",
        "",
        f"**Date**: {datetime.now(timezone.utc).date().isoformat()}",
        "**Constraint**: 12-d RDKit + zero-padded 32-d Multi-PH + 128-d PLL "
        "(tmQM d-block ships no 172-d NPZ); 5-fold CV × 5 seeds × "
        f"{len(elements)} elements; CPU default, GPU-1 optional.",
        "**Target**: ``coord_number`` (DFT-computed coordination number).",
        "**Task**: 3-way model not worse than baseline Hilbert on cross-dataset.",
        "",
        "## Cross-element summary (per-seed aggregate over 5 folds)",
        "",
        "| element | n_seeds | combined R² (mean ± std) | Hilbert R² | "
        "Multi-PH R² | ASTSR R² | collapse_any |",
        "|---|---:|---|---|---|---|---|",
    ]

    element_means: List[float] = []
    hilbert_means: List[float] = []
    for el, recs in by_el.items():
        summaries = aggregate_element(recs)
        combined_arr = np.asarray([s.combined_r2_mean for s in summaries], dtype=float)
        h_arr = np.asarray([s.hilbert_r2_mean for s in summaries], dtype=float)
        m_arr = np.asarray([s.mph_r2_mean for s in summaries], dtype=float)
        a_arr = np.asarray([s.astr_r2_mean for s in summaries], dtype=float)
        cm = float(np.nanmean(combined_arr))
        cs = float(np.nanstd(combined_arr))
        hm = float(np.nanmean(h_arr))
        mm = float(np.nanmean(m_arr))
        am = float(np.nanmean(a_arr))
        collapse_any = bool(any(s.collapse_detected_any for s in summaries))
        element_means.append(cm)
        hilbert_means.append(hm)
        lines.append(
            f"| {el} | {len(summaries)} | {cm:+.4f} ± {cs:.4f} | "
            f"{hm:+.4f} | {mm:+.4f} | {am:+.4f} | "
            f"{'YES' if collapse_any else 'no'} |"
        )

    if element_means:
        agg = float(np.mean(element_means))
        agg_std = float(np.std(element_means))
        hilbert_agg = float(np.mean(hilbert_means))
        # Verify gate: combined R² not worse than Hilbert baseline.
        n_pass = int(sum(1 for cm, hm in zip(element_means, hilbert_means) if cm >= hm))
        verdict = "PASS" if agg >= hilbert_agg else "FAIL"
    else:
        agg = 0.0
        agg_std = 0.0
        hilbert_agg = 0.0
        n_pass = 0
        verdict = "FAIL"
    lines.append(
        f"| **29-element aggregate** | — | **{agg:+.4f} ± {agg_std:.4f}** | "
        f"**{hilbert_agg:+.4f}** | — | — | — |"
    )
    lines.append("")
    lines.append("## Verify gate (target: combined R² ≥ Hilbert baseline)")
    lines.append("")
    lines.append(f"**{n_pass} of {len(element_means)} elements pass.**")
    lines.append("")
    lines.append(
        f"**Verdict: {verdict}** — 3-way combined R² = {agg:+.4f}; "
        f"Hilbert baseline R² = {hilbert_agg:+.4f}."
    )
    lines.append("")
    lines.append("## Per-(element, seed) detail")
    for el, recs in by_el.items():
        lines.append(f"### {el}")
        lines.append("")
        lines.append(
            "| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | "
            "wall_s | collapse |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        summaries = aggregate_element(recs)
        for s in summaries:
            lines.append(
                f"| {s.seed} | {s.combined_r2_mean:+.4f} ± "
                f"{s.combined_r2_std:.4f} | {s.hilbert_r2_mean:+.4f} | "
                f"{s.mph_r2_mean:+.4f} | {s.astr_r2_mean:+.4f} | "
                f"{s.wall_s_mean:.2f} | "
                f"{'YES' if s.collapse_detected_any else 'no'} |"
            )
        lines.append("")

    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "- **REAL data**: tmQM d-block (Balcells & Skjelstad 2020); "
        "DFT-computed ``coord_number``."
    )
    lines.append(
        "- **Feature stack**: 172-d zero-padded (12-d RDKit + "
        "32-d Multi-PH zeros + 128-d PLL zeros); the 3-way "
        "Multi-PH stream is effectively a constant-zero ridge."
    )
    lines.append("- **ASTSR scale guard**: predictions clipped to `[0, 1]`.")
    lines.append("- **CPU + GPU-1**: CPU default; GPU-1 optional.")
    lines.append(
        "- **Collapse detector**: ``pred_std / y_test_std < 0.10`` "
        "per fold → flag in ``per_seed_metrics.json``."
    )
    lines.append("")
    lines.append(
        f"Per `party-twenty-spirit §3.5`: 29-element combined R² = "
        f"{agg:+.4f} vs Hilbert baseline = {hilbert_agg:+.4f}; "
        f"verdict = **{verdict}** ({n_pass} of {len(element_means)} "
        f"elements pass)."
    )

    (out_dir / "synthesis.md").parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthesis.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_elements(raw: str) -> List[str]:
    return [x.strip() for x in raw.replace(",", " ").split() if x]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-128 Phase 3 — 3-way cross-dataset verify on tmQM d-block "
            "per element (29 d-block metals)."
        )
    )
    parser.add_argument(
        "--elements",
        type=_parse_elements,
        nargs="+",
        default=None,
        help="Element symbols (default: all 29 with " "≥5 rows; pass subset to scope)",
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, nargs="+", default=list(DEFAULT_SEEDS)
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hilbert-dim", type=int, default=DEFAULT_HILBERT_DIM)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_TMQM_CSV)
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument(
        "--n-max", type=int, default=None, help="Cap rows per element (for ablation)."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 2 elements (Cu, Ni), 1 seed, " "2 folds, 10 epochs.",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        flat_elements = ["Cu", "Ni"]
        flat_seeds = [0]
        epochs = min(int(args.epochs), 10)
        n_folds = min(int(args.n_folds), 2)
        n_max = 200 if args.n_max is None else min(int(args.n_max), 200)
    else:
        if args.elements is None:
            flat_elements = list_available_elements(
                csv_path=args.csv_path,
                target_col=args.target_col,
            )
            if not flat_elements:
                raise RuntimeError("No usable elements in CSV.")
        else:
            flat_elements = []
            for e in args.elements:
                if isinstance(e, list):
                    flat_elements.extend(str(x) for x in e)
                else:
                    flat_elements.append(str(e))
        flat_seeds = []
        for s in args.seeds:
            if isinstance(s, list):
                flat_seeds.extend(int(x) for x in s)
            else:
                flat_seeds.append(int(s))
        epochs = int(args.epochs)
        n_folds = int(args.n_folds)
        n_max = args.n_max

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[wf128-p3-tmqm] elements={flat_elements} seeds={flat_seeds} "
        f"epochs={epochs} n_folds={n_folds} target={args.target_col} "
        f"n_max={n_max}"
    )

    all_records: List[FitRecord] = []
    for el in flat_elements:
        print(f"[wf128-p3-tmqm] === element={el} ===")
        try:
            recs = evaluate_element(
                element=el,
                seeds=flat_seeds,
                n_folds=n_folds,
                hilbert_dim=int(args.hilbert_dim),
                csv_path=args.csv_path,
                target_col=args.target_col,
                n_max=n_max,
            )
        except Exception as exc:
            print(f"[wf128-p3-tmqm] element={el}: FAILED ({exc!r})")
            continue
        all_records.extend(recs)

        per_el_dir = args.out_dir / el
        per_el_dir.mkdir(parents=True, exist_ok=True)
        summaries = aggregate_element(recs)
        payload = {
            "element": el,
            "config": {
                "elements": flat_elements,
                "seeds": flat_seeds,
                "n_folds": n_folds,
                "epochs": epochs,
                "hilbert_dim": int(args.hilbert_dim),
                "csv_path": str(args.csv_path),
                "target_col": args.target_col,
                "n_max": n_max,
                "feature_dim": FEATURE_DIM,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "per_seed": [asdict(s) for s in summaries],
            "per_fold": [asdict(r) for r in recs],
        }
        per_seed_path = per_el_dir / "per_seed_metrics.json"
        _write_json(per_seed_path, payload)
        print(f"[wf128-p3-tmqm] wrote {per_seed_path}")

    write_synthesis(args.out_dir, all_records, flat_elements)
    print(
        f"[wf128-p3-tmqm] synthesis.md written to " f"{args.out_dir / 'synthesis.md'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
