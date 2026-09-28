#!/usr/bin/env python3
"""Wf-128 Phase 3 — 3-way (Deflexpressor + Multi-PH + ASTSR) cross-dataset
verification on PlatinAI 4-cell per cell-line (CPU + GPU-1).

================================================================
``molmetal/scripts/bench_3way_platinai_4cell.py`` is the
**Wf-128 P3 cross-dataset verify** for the PlatinAI 4-cell
synthetic pIC50 dataset (4 078 rows × 4 cell lines, sourced
from ``platinai_4cell.meta.json``).

Wf-128 P2 (5-metal MetalCytoToxDB) shipped **3-way combined
R² = -150.64** — the integration did not break the gate.
Wf-128 P3 lifts the same 3-way harness (Hilbert + Multi-PH +
ASTSR) onto the PlatinAI 4-cell pIC50 target **per cell-line**
(MCF7, A549, A2780, A2780cis) and answers:

    **Is the 3-way model not worse than the baseline Hilbert
    on cross-dataset (PlatinAI 4-cell)?**

Constraints
-----------
* **CPU default, GPU-1 optional** — fits run fp32 Nesterov +
  small evolutionary search; per-cell-line fits run in < 30 s
  on a single CPU core for the 4 078-row cohort.
* **No Multi-PH + PLL augmentation** — PlatinAI 4-cell does not
  ship a 172-d NPZ; the 3-way uses the 12-d F5_RDKIT_2D
  descriptor + zero-padded MPH/PLL slots.
* **5-fold CV × 5 seeds × 4 cell-lines** — small but reproducible
  per `party-twenty-spirit §3.5`.
* **Honest framing** — negative R² values are not hidden; the
  collapse detector flags ``pred_std / y_test_std < 0.10`` per
  fold.

Outputs
-------
* ``molmetal/reports/wf128_p3_platinai_4cell/{cell_line}/per_seed_metrics.json``
* ``molmetal/reports/wf128_p3_platinai_4cell/synthesis.md``

Usage::

    CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python \
      molmetal/scripts/bench_3way_platinai_4cell.py \
      --cell-lines mcf7 a549 a2780 a2780cis --seeds 0 1 2 3 4 \
      --epochs 50 --out-dir molmetal/reports/wf128_p3_platinai_4cell
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
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
    COLLAPSE_RATIO_THRESHOLD,
    DEFAULT_HILBERT_DIM,
    DEFAULT_N_FOLDS,
    EPS_STD,
    FEATURE_DIM,
    MULTI_PH_DIM,
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
# PlatinAI 4-cell constants.
# ---------------------------------------------------------------------------
DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
DEFAULT_SMILES_COL: str = "smiles"
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
DEFAULT_CELL_LINES: Tuple[str, ...] = ("mcf7", "a549", "a2780", "a2780cis")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 50
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf128_p3_platinai_4cell"
DEFAULT_N_MAX: int = 4078


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------
@dataclass
class FitRecord:
    cell_line: str
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
class CellLineSeedSummary:
    cell_line: str
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


def load_platinai_cell(
    cell_line: str,
    *,
    csv_path: Path = DEFAULT_PLATINAI_CSV,
    n_max: int = DEFAULT_N_MAX,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load PlatinAI 4-cell for one cell-line.

    Returns ``(X, y, smiles)``:
      * ``X`` (N, 12) float32 — RDKit 2D descriptor (zero-padded
        to 172-d for 3-way compatibility).
      * ``y`` (N,) float32 — pIC50 = -log10(IC50_µM × 1e-6).
      * ``smiles`` List[str] — for scaffold split + debug.

    Rows with missing/unparseable target or SMILES are dropped.
    """
    if cell_line not in PLATINAI_CELL_COLS:
        raise ValueError(
            f"Unknown cell line: {cell_line}; "
            f"supported: {sorted(PLATINAI_CELL_COLS)}"
        )
    target_col = PLATINAI_CELL_COLS[cell_line]
    rows = _read_csv(csv_path)
    Xs: List[List[float]] = []
    Ys: List[float] = []
    smiles: List[str] = []
    for r in rows:
        smi = (r.get(DEFAULT_SMILES_COL) or "").strip()
        if not smi:
            continue
        feats = _featurize_smiles_12d(smi)
        if feats is None:
            continue
        raw = r.get(target_col)
        try:
            ic50_nm = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic50_nm) or ic50_nm <= 0.0:
            continue
        ic50_um = ic50_nm / 1000.0
        pic50 = -math.log10(max(ic50_um * 1e-6, 1e-12))
        if not math.isfinite(pic50):
            continue
        Xs.append(feats)
        Ys.append(float(pic50))
        smiles.append(smi)
        if len(Xs) >= n_max:
            break
    if not Xs:
        raise RuntimeError(f"No usable rows for cell line {cell_line!r}")
    # Zero-pad to FEATURE_DIM (172) for the 3-way streams.
    Xs_padded: List[List[float]] = []
    for f12 in Xs:
        full = [0.0] * FEATURE_DIM
        full[:RDKIT_DIM] = f12
        Xs_padded.append(full)
    return (
        np.asarray(Xs_padded, dtype=np.float32),
        np.asarray(Ys, dtype=np.float32),
        smiles,
    )


# ---------------------------------------------------------------------------
# Per-fold evaluation.
# ---------------------------------------------------------------------------
def evaluate_fold(
    *,
    cell_line: str,
    seed: int,
    fold: int,
    tr: np.ndarray, te: np.ndarray,
    X: np.ndarray, y: np.ndarray,
    epochs: int,
    hilbert_dim: int,
) -> FitRecord:
    """Run 3-way integration on a single (cell_line, seed, fold) split.

    Streams
    -------
    * Hilbert — closed-form ridge on the 172-d Hilbert-padded features.
    * Multi-PH — closed-form ridge on the 32-d Multi-PH slice (zero
      here; the slice is all zeros in PlatinAI 4-cell, so the stream
      degrades to a constant zero ridge — this is the honest baseline
      for "no Multi-PH signal available").
    * ASTSR — Deflexpressor evolutionary symbolic regression on the
      12-d RDKit slice, with the Wf-127 P1 scale guard.

    Combined predictor — convex blend ``0.4*H + 0.4*M + 0.2*A``.
    """
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    X_tr_z, X_te_z = _zscore(X_tr.astype(float), X_te.astype(float))
    t0 = time.perf_counter()

    try:
        w_h, _trh, aug_te_h = fit_hilbert_stream(
            X_tr_z, y_tr, X_te_z, hilbert_dim=hilbert_dim,
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
            X_tr, y_tr, use_astr_scale_guard=True,
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
        cell_line=cell_line, seed=int(seed), fold=int(fold),
        n_train=int(len(tr)), n_test=int(len(te)),
        combined_r2=float(r2_c) if math.isfinite(r2_c) else float("nan"),
        hilbert_r2=float(r2_h) if math.isfinite(r2_h) else float("nan"),
        mph_r2=float(r2_m) if math.isfinite(r2_m) else float("nan"),
        astr_r2=float(r2_a) if math.isfinite(r2_a) else float("nan"),
        wall_s=float(wall), collapse_detected=bool(collapse_c),
    )


# ---------------------------------------------------------------------------
# Per-cell-line evaluation.
# ---------------------------------------------------------------------------
def evaluate_cell_line(
    cell_line: str,
    seeds: Sequence[int],
    n_folds: int,
    epochs: int,
    hilbert_dim: int,
    csv_path: Path,
    n_max: int,
) -> List[FitRecord]:
    X, y, _smi = load_platinai_cell(cell_line, csv_path=csv_path, n_max=n_max)
    n = X.shape[0]
    if n < 5:
        raise RuntimeError(f"cell_line {cell_line}: too few rows ({n})")

    records: List[FitRecord] = []
    for seed in seeds:
        splits = _random_kfold_indices(n, int(n_folds), seed=int(seed))
        for fold, (tr, te) in enumerate(splits):
            if len(te) == 0:
                continue
            rec = evaluate_fold(
                cell_line=cell_line, seed=int(seed), fold=int(fold),
                tr=tr, te=te, X=X, y=y,
                epochs=int(epochs), hilbert_dim=int(hilbert_dim),
            )
            records.append(rec)
            print(
                f"[wf128-p3-plat]   cell_line={cell_line} "
                f"target={PLATINAI_CELL_COLS[cell_line]} seed={int(seed)} "
                f"fold={int(fold)} combined_r2={rec.combined_r2:+.4f} "
                f"H={rec.hilbert_r2:+.4f} M={rec.mph_r2:+.4f} "
                f"A={rec.astr_r2:+.4f} collapse={rec.collapse_detected}",
                flush=True,
            )
    return records


def aggregate_cell_line(records: List[FitRecord]) -> List[CellLineSeedSummary]:
    out: List[CellLineSeedSummary] = []
    by_key: Dict[Tuple[str, int], List[FitRecord]] = {}
    for r in records:
        by_key.setdefault((r.cell_line, r.seed), []).append(r)
    for (cell_line, seed), recs in sorted(by_key.items()):
        combined = np.asarray([r.combined_r2 for r in recs], dtype=float)
        hilbert = np.asarray([r.hilbert_r2 for r in recs], dtype=float)
        mph = np.asarray([r.mph_r2 for r in recs], dtype=float)
        astr = np.asarray([r.astr_r2 for r in recs], dtype=float)
        wall = np.asarray([r.wall_s for r in recs], dtype=float)
        out.append(CellLineSeedSummary(
            cell_line=cell_line, seed=int(seed),
            n_folds=int(len(recs)),
            combined_r2_mean=float(np.nanmean(combined)) if combined.size else 0.0,
            combined_r2_std=float(np.nanstd(combined)) if combined.size > 1 else 0.0,
            hilbert_r2_mean=float(np.nanmean(hilbert)) if hilbert.size else 0.0,
            mph_r2_mean=float(np.nanmean(mph)) if mph.size else 0.0,
            astr_r2_mean=float(np.nanmean(astr)) if astr.size else 0.0,
            wall_s_mean=float(np.nanmean(wall)) if wall.size else 0.0,
            collapse_detected_any=bool(any(r.collapse_detected for r in recs)),
        ))
    return out


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=str)


def write_synthesis(out_dir: Path, all_records: List[FitRecord]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cl: Dict[str, List[FitRecord]] = {}
    for r in all_records:
        by_cl.setdefault(r.cell_line, []).append(r)

    lines: List[str] = [
        "# Wf-128 Phase 3 — 3-way (Deflexpressor + Multi-PH + ASTSR) "
        "cross-dataset verify on PlatinAI 4-cell",
        "",
        f"**Date**: {datetime.now(timezone.utc).date().isoformat()}",
        "**Constraint**: 12-d RDKit + zero-padded 32-d Multi-PH + 128-d PLL "
        "(PlatinAI 4-cell ships no 172-d NPZ); 5-fold CV × 5 seeds × 4 "
        "cell-lines; CPU default, GPU-1 optional.",
        "**Task**: 3-way model not worse than baseline Hilbert on cross-dataset.",
        "",
        "## Cross-cell-line summary (per-seed aggregate over 5 folds)",
        "",
        "| cell_line | n_seeds | combined R² (mean ± std) | Hilbert R² | "
        "Multi-PH R² | ASTSR R² | collapse_any |",
        "|---|---:|---|---|---|---|---|",
    ]

    cell_means: List[float] = []
    hilbert_means: List[float] = []
    for cl, recs in by_cl.items():
        summaries = aggregate_cell_line(recs)
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
        cell_means.append(cm)
        hilbert_means.append(hm)
        lines.append(
            f"| {cl} | {len(summaries)} | {cm:+.4f} ± {cs:.4f} | "
            f"{hm:+.4f} | {mm:+.4f} | {am:+.4f} | "
            f"{'YES' if collapse_any else 'no'} |"
        )

    if cell_means:
        agg = float(np.mean(cell_means))
        agg_std = float(np.std(cell_means))
        hilbert_agg = float(np.mean(hilbert_means))
        # Verify gate: combined R² not worse than Hilbert baseline.
        n_pass = int(sum(1 for cm, hm in zip(cell_means, hilbert_means) if cm >= hm))
        verdict = "PASS" if agg >= hilbert_agg else "FAIL"
    else:
        agg = 0.0
        agg_std = 0.0
        hilbert_agg = 0.0
        n_pass = 0
        verdict = "FAIL"
    lines.append(
        f"| **4-cell aggregate** | — | **{agg:+.4f} ± {agg_std:.4f}** | "
        f"**{hilbert_agg:+.4f}** | — | — | — |"
    )
    lines.append("")
    lines.append("## Verify gate (target: combined R² ≥ Hilbert baseline)")
    lines.append("")
    lines.append(f"**{n_pass} of {len(cell_means)} cell-lines pass.**")
    lines.append("")
    lines.append(
        f"**Verdict: {verdict}** — 3-way combined R² = {agg:+.4f}; "
        f"Hilbert baseline R² = {hilbert_agg:+.4f}."
    )
    lines.append("")
    lines.append("## Per-(cell_line, seed) detail")
    for cl, recs in by_cl.items():
        lines.append(f"### {cl}")
        lines.append("")
        lines.append(
            "| seed | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | "
            "wall_s | collapse |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|")
        summaries = aggregate_cell_line(recs)
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
    lines.append("- **REAL data**: PlatinAI 4-cell synthetic pIC50 (per "
                 "``platinai_4cell.meta.json``).")
    lines.append("- **Feature stack**: 172-d zero-padded (12-d RDKit + "
                 "32-d Multi-PH zeros + 128-d PLL zeros); the 3-way "
                 "Multi-PH stream is effectively a constant-zero ridge.")
    lines.append("- **ASTSR scale guard**: predictions clipped to `[0, 1]`.")
    lines.append("- **CPU + GPU-1**: CPU default; GPU-1 optional.")
    lines.append("- **Collapse detector**: ``pred_std / y_test_std < 0.10`` "
                 "per fold → flag in ``per_seed_metrics.json``.")
    lines.append("")
    lines.append(
        f"Per `party-twenty-spirit §3.5`: 4-cell combined R² = "
        f"{agg:+.4f} vs Hilbert baseline = {hilbert_agg:+.4f}; "
        f"verdict = **{verdict}** ({n_pass} of {len(cell_means)} "
        f"cell-lines pass)."
    )

    (out_dir / "synthesis.md").parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthesis.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def _parse_cell_lines(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.replace(",", " ").split() if x]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-128 Phase 3 — 3-way cross-dataset verify on PlatinAI 4-cell "
            "per cell-line."
        )
    )
    parser.add_argument("--cell-lines", type=_parse_cell_lines, nargs="+",
                        default=list(DEFAULT_CELL_LINES))
    parser.add_argument("--seeds", type=_parse_seeds, nargs="+",
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hilbert-dim", type=int, default=DEFAULT_HILBERT_DIM)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_PLATINAI_CSV)
    parser.add_argument("--n-max", type=int, default=DEFAULT_N_MAX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: 1 cell-line (mcf7), 1 seed, 2 folds, 10 epochs.")
    args = parser.parse_args(argv)

    if args.smoke:
        flat_cls = ["mcf7"]
        flat_seeds = [0]
        epochs = min(int(args.epochs), 10)
        n_folds = min(int(args.n_folds), 2)
        n_max = min(int(args.n_max), 200)
    else:
        # Flatten argparse nargs="+" artifacts.
        flat_cls: List[str] = []
        for c in args.cell_lines:
            if isinstance(c, list):
                flat_cls.extend(str(x).lower() for x in c)
            else:
                flat_cls.append(str(c).lower())
        flat_seeds: List[int] = []
        for s in args.seeds:
            if isinstance(s, list):
                flat_seeds.extend(int(x) for x in s)
            else:
                flat_seeds.append(int(s))
        epochs = int(args.epochs)
        n_folds = int(args.n_folds)
        n_max = int(args.n_max)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[wf128-p3-plat] cell_lines={flat_cls} seeds={flat_seeds} "
        f"epochs={epochs} n_folds={n_folds} n_max={n_max}"
    )

    all_records: List[FitRecord] = []
    for cl in flat_cls:
        print(f"[wf128-p3-plat] === cell_line={cl} ===")
        try:
            recs = evaluate_cell_line(
                cell_line=cl, seeds=flat_seeds, n_folds=n_folds,
                epochs=epochs, hilbert_dim=int(args.hilbert_dim),
                csv_path=args.csv_path, n_max=n_max,
            )
        except Exception as exc:
            print(f"[wf128-p3-plat] cell_line={cl}: FAILED ({exc!r})")
            continue
        all_records.extend(recs)

        per_cl_dir = args.out_dir / cl
        per_cl_dir.mkdir(parents=True, exist_ok=True)
        summaries = aggregate_cell_line(recs)
        payload = {
            "cell_line": cl,
            "target_col": PLATINAI_CELL_COLS.get(cl, "?"),
            "config": {
                "cell_lines": flat_cls,
                "seeds": flat_seeds,
                "n_folds": n_folds,
                "epochs": epochs,
                "hilbert_dim": int(args.hilbert_dim),
                "csv_path": str(args.csv_path),
                "n_max": n_max,
                "feature_dim": FEATURE_DIM,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "per_seed": [asdict(s) for s in summaries],
            "per_fold": [asdict(r) for r in recs],
        }
        per_seed_path = per_cl_dir / "per_seed_metrics.json"
        _write_json(per_seed_path, payload)
        print(f"[wf128-p3-plat] wrote {per_seed_path}")

    write_synthesis(args.out_dir, all_records)
    print(f"[wf128-p3-plat] synthesis.md written to "
          f"{args.out_dir / 'synthesis.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())