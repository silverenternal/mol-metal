#!/usr/bin/env python3
"""Wf-77 Phase 2 — ASTSR-only R² baseline (no Hilbert, no Multi-PH).

================================================================
``molmetal/scripts/bench_astr_only_baseline.py`` is the **ASTSR-only
counterpart** to Wf-77 Phase 1 Hilbert-only baseline
(``molmetal/scripts/bench_hilbert_only_baseline.py``).  It runs the
**ASTSR-only** Deflexpressor-style evolutionary symbolic regressor on
real labelled pIC50 data (PlatinAI 4-cell + MetalCytoToxDB 5-metal
multitask) WITHOUT the Hilbert primal-dual regressor and WITHOUT
Multi-PH / PLL augmentations.  Future composite R² numbers from Wf-37
/ Wf-49 / Wf-56 can therefore be **honestly attributed** to the
Hilbert dual / Multi-PH augmentation, not to the ASTSR search alone.

Why this script matters
-----------------------
* Wf-37 / Wf-44 / Wf-49 ship composite Hilbert+ASTSR R² = +0.16 to
  +0.18 on PlatinAI 4-cell (per ``molmetal/reports/wf37_hilbert_astsr``
  and ``wf49_astr_rdkit``) — but **the ASTSR-only baseline** has never
  been run on the same data with the same F5_RDKIT_2D feature set, so
  the lift attribution is not rigorous.
* Wf-56 R20-P2 plans a 4-week GPU retrain that targets R² ≥ 0.85 on
  PlatinAI 4-cell.  Without an ASTSR-only anchor, the composite R²
  attribution chain is broken.
* Per ``party-twenty-spirit §3.5`` we report MEASURED-vs-PROJECTED;
  this script is the **MEASURED** ASTSR-only anchor.

Constraints honored
-------------------
* **No Hilbert** — the Hilbert primal-dual regressor is deliberately
  absent.  The only learner is the ``LambdaASTSR`` evolutionary
  symbolic regressor on the F5_RDKIT_2D 12-d RDKit feature set.
* **CPU-only** — the evolutionary search is fp32 tournament selection
  on a fixed-size population; 5 seeds × 5-fold CV runs in < 5 min per
  dataset on a single CPU core.  GPU is **not** required.
* **Honest framing** — the script reports mean ± std per seed, per
  cohort (cell-line / metal), and per-fold.  Negative R² values are
  not hidden — they are tagged ``collapse_detected`` when
  ``pred_std / y_test_std < 0.1``.

Inputs
------
* ``--datasets``: ``platinai_4cell`` (4078 rows × 4 cell lines) and/or
  ``metalcytotox_5metal`` (26801 rows × 5 metals, binary activity).
* ``--seeds``: list of integers (default ``0 1 2 3 4``).
* ``--n-folds``: 5 by default.
* ``--population``: ASTSR population size (default 32).
* ``--generations``: ASTSR generations (default 30).
* ``--depth``: ASTSR max AST depth (default 4).
* ``--output``: ``molmetal/reports/wf77_p2_astr_baseline/``.

Outputs
-------
* ``<output>/{dataset}/per_seed_metrics.json`` — per-seed mean
  R² ± std, per-cohort R², per-fold breakdown, collapse flag,
  ``astr_wall_time_s`` per seed + best AST per seed (lisp).
* ``<output>/{dataset}/run.log`` — stdout mirror.
* ``<output>/synthesis.md`` — cross-dataset summary table.

Usage
-----

::

    .venv/bin/python -u molmetal/scripts/bench_astr_only_baseline.py \\
        --datasets platinai_4cell metalcytotox_5metal \\
        --seeds 0 1 2 3 4 --n-folds 5 \\
        --population 32 --generations 30 --depth 4 \\
        --output molmetal/reports/wf77_p2_astr_baseline

Pytest
~~~~~~

::

    pytest molmetal/tests/test_astr_only_baseline.py -v
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root import bootstrap.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Feature column registry — F5_RDKIT_2D (Wf-49 Phase 2 ship).
# ---------------------------------------------------------------------------
# The 12-d RDKit 2D descriptor set is the canonical Lipinski-style
# baseline (MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms,
# RingCount, NumAromaticRings, FractionCSP3, MolMR, BalabanJ).
# Wf-49 §Cause 2 (CRITICAL) confirms that the legacy F5 features are
# NOT informative on pIC50 (every F5 feature has |Pearson r| < 0.02
# on PlatinAI 4-cell), so the ASTSR-only anchor uses F5_RDKIT_2D.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)
COLLAPSE_RATIO_THRESHOLD: float = 0.1


# ---------------------------------------------------------------------------
# Dataset loaders — featurise SMILES into the F5_RDKIT_2D 12-d vector.
# ---------------------------------------------------------------------------
def _featurize_smiles(smiles: str) -> Optional[List[float]]:
    """Return 12-d RDKit 2D descriptor vector; ``None`` if SMILES fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        feats: List[float] = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(Descriptors.MolMR(mol)),
            float(Descriptors.BalabanJ(mol)),
        ]
        if any((not math.isfinite(v)) for v in feats):
            return None
        return feats
    except Exception:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


@dataclass
class DatasetSpec:
    name: str
    csv_path: Path
    smiles_col: str
    target_cols: Tuple[str, ...]
    cohort_col: Optional[str] = None
    cohort_values: Optional[Tuple[str, ...]] = None
    n_max: int = 10000


def _load_dataset(spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return ``(X, Y, cohort_idx, cohort_names)`` for the given dataset spec.

    ``X`` is (n_samples, 12) float32; rows with failed SMILES are dropped.
    ``Y`` is (n_samples, n_targets) float32; rows with any NaN target are dropped.
    """
    rows = _read_csv(spec.csv_path)
    Xs: List[List[float]] = []
    Ys: List[List[float]] = []
    cohorts: List[str] = []
    n_targets = len(spec.target_cols)
    for r in rows:
        smi = (r.get(spec.smiles_col) or "").strip()
        if not smi:
            continue
        feats = _featurize_smiles(smi)
        if feats is None:
            continue
        targets: List[float] = []
        ok = True
        for c in spec.target_cols:
            v = r.get(c)
            if v is None or v == "":
                ok = False
                break
            try:
                f = float(v)
            except (TypeError, ValueError):
                ok = False
                break
            if not math.isfinite(f):
                ok = False
                break
            targets.append(f)
        if not ok:
            continue
        cohort_val = ""
        if spec.cohort_col and spec.cohort_values:
            raw = (r.get(spec.cohort_col) or "").strip()
            for cv in spec.cohort_values:
                if cv.lower() in raw.lower():
                    cohort_val = cv
                    break
            if not cohort_val:
                cohort_val = spec.cohort_values[0]
        elif spec.cohort_col:
            cohort_val = (r.get(spec.cohort_col) or "").strip()
        Xs.append(feats)
        Ys.append(targets)
        cohorts.append(cohort_val or "all")
        if len(Xs) >= spec.n_max:
            break

    if not Xs:
        raise RuntimeError(f"No usable rows in {spec.csv_path}")
    X = np.asarray(Xs, dtype=np.float32)
    Y = np.asarray(Ys, dtype=np.float32)
    unique_cohorts = sorted(set(cohorts))
    name_to_idx = {n: i for i, n in enumerate(unique_cohorts)}
    cohort_idx = np.asarray([name_to_idx[c] for c in cohorts], dtype=np.int64)
    return X, Y, cohort_idx, unique_cohorts


def _platinai_4cell_spec(repo_root: Path) -> DatasetSpec:
    return DatasetSpec(
        name="platinai_4cell",
        csv_path=repo_root / "molmetal" / "data" / "platinai_4cell.csv",
        smiles_col="smiles",
        target_cols=(
            "ic50_mcf7_nm",
            "ic50_a549_nm",
            "ic50_a2780_nm",
            "ic50_a2780cis_nm",
        ),
        cohort_col="source",
        cohort_values=("KULGUU", "HILLUL", "FAFYEU", "NURGAI"),
        n_max=4078,
    )


def _metalcytotox_5metal_spec(repo_root: Path) -> DatasetSpec:
    candidates = [
        repo_root / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
        repo_root / "data" / "metalcytotox_5metal_multitask_v4.csv",
        repo_root / "molmetal" / "data" / "metalcytotox_5metal_multitask_v4.csv",
    ]
    for c in candidates:
        if c.exists():
            return DatasetSpec(
                name="metalcytotox_5metal",
                csv_path=c,
                smiles_col="smiles",
                target_cols=("activity",),
                cohort_col="metal",
                cohort_values=("Pt", "Pd", "Au", "Ru", "Ir"),
                n_max=20000,
            )
    raise FileNotFoundError(
        "metalcytotox_5metal_multitask_v4.csv not found in "
        + ", ".join(str(c) for c in candidates)
    )


def _build_dataset_specs(repo_root: Path, names: Sequence[str]) -> List[DatasetSpec]:
    builders = {
        "platinai_4cell": _platinai_4cell_spec,
        "metalcytotox_5metal": _metalcytotox_5metal_spec,
    }
    out: List[DatasetSpec] = []
    for n in names:
        b = builders.get(n)
        if n not in builders:
            raise ValueError(f"Unknown dataset: {n}; supported: {sorted(builders)}")
        out.append(b(repo_root))
    return out


# ---------------------------------------------------------------------------
# ASTSR-only regressor — wraps LambdaASTSR for this benchmark.
# ---------------------------------------------------------------------------
@dataclass
class ASTSRRegressor:
    """Minimal ASTSR-only wrapper.

    We deliberately avoid pulling in :class:`HilbertASTSRComposite`
    (which by default pairs ASTSR with the Hilbert primal-dual
    regressor) — this script explicitly wants **ASTSR only** for the
    R² anchor.
    """

    population: int = 32
    generations: int = 30
    depth: int = 4
    p_var: float = 0.45
    seed: int = 0

    def _build(self):
        from molmetal.molmetal_lam.reward.lambda_astsr import LambdaASTSR

        return LambdaASTSR(
            depth=self.depth,
            population=self.population,
            generations=self.generations,
            seed=self.seed,
            p_var=self.p_var,
            feature_names=tuple(FEATURE_COLUMNS),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ASTSRRegressor":
        """Run :meth:`LambdaASTSR.fit` then store the regressor."""
        opt = self._build()
        # LambdaASTSR.fit takes (X, y); feature_names is a constructor arg.
        opt.fit(X, y)
        self._opt = opt
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._opt.predict(X), dtype=np.float64)

    def best_ast_lisp(self) -> str:
        """Return the best AST in lisp form for transparency."""
        best = getattr(self._opt, "best_ast_", None)
        if best is None:
            return "<no_ast>"
        try:
            return best.to_lisp()
        except Exception as e:  # pragma: no cover
            return f"<ast_print_error:{type(e).__name__}>"


# ---------------------------------------------------------------------------
# K-fold + metrics (mirrors Wf-77 Phase 1 to keep numbers comparable).
# ---------------------------------------------------------------------------
def _kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _collapse_detected(y_pred: np.ndarray, y_true: np.ndarray) -> bool:
    if y_true.std() < 1e-12:
        return True
    return float(y_pred.std() / max(y_true.std(), 1e-12)) < COLLAPSE_RATIO_THRESHOLD


# ---------------------------------------------------------------------------
# Per-dataset evaluation.
# ---------------------------------------------------------------------------
def _evaluate_dataset(
    spec: DatasetSpec,
    seeds: Sequence[int],
    n_folds: int,
    population: int,
    generations: int,
    depth: int,
) -> Dict[str, Any]:
    """Run ASTSR-only 5-fold × N-seed on a single dataset."""
    X, Y, cohort_idx, cohort_names = _load_dataset(spec)
    n, d_feat = X.shape
    n_targets = Y.shape[1]

    per_seed: List[Dict[str, Any]] = []
    per_target_aggregate: Dict[str, List[float]] = {c: [] for c in spec.target_cols}

    for seed in seeds:
        splits = _kfold_indices(n, n_folds, seed=seed)
        per_fold: List[Dict[str, Any]] = []
        fold_r2_per_target: Dict[str, List[float]] = {c: [] for c in spec.target_cols}
        cohort_r2_per_target: Dict[str, Dict[str, List[float]]] = {
            t: {c: [] for c in cohort_names} for t in spec.target_cols
        }
        all_y_true: List[float] = []
        all_y_pred: List[float] = []
        t0 = time.perf_counter()
        for fold_idx, (tr, te) in enumerate(splits):
            X_tr, X_te = X[tr], X[te]
            # Per-fold z-score.
            mu = X_tr.mean(axis=0)
            sigma = X_tr.std(axis=0)
            sigma = np.where(sigma < 1e-9, 1.0, sigma)
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma

            for t_idx, t_name in enumerate(spec.target_cols):
                y_tr = Y[tr, t_idx]
                y_te = Y[te, t_idx]
                if y_tr.std() < 1e-9:
                    fold_r2_per_target[t_name].append(0.0)
                    per_fold.append(
                        {
                            "fold": fold_idx,
                            "target": t_name,
                            "n_train": int(len(tr)),
                            "n_test": int(len(te)),
                            "r2": 0.0,
                            "collapse_detected": True,
                            "wall_s": 0.0,
                        }
                    )
                    continue
                reg = ASTSRRegressor(
                    population=population,
                    generations=generations,
                    depth=depth,
                    seed=seed,
                ).fit(X_tr_n, y_tr)
                y_hat = reg.predict(X_te_n)
                # ASTSR can produce extreme values from ill-conditioned
                # symbolic ops; clip to a sane bound.
                y_hat = np.clip(y_hat, -1e6, 1e6)
                r2 = _compute_r2(y_te, y_hat)
                collapse = _collapse_detected(y_hat, y_te)
                fold_r2_per_target[t_name].append(r2)
                per_fold.append(
                    {
                        "fold": fold_idx,
                        "target": t_name,
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                        "r2": float(r2),
                        "collapse_detected": bool(collapse),
                        "wall_s": float(time.perf_counter() - t0) / max(fold_idx + 1, 1)
                        / max(t_idx + 1, 1),
                    }
                )
                all_y_true.extend(y_te.tolist())
                all_y_pred.extend(y_hat.tolist())
                for ci, cohort_name in enumerate(cohort_names):
                    mask = cohort_idx[te] == ci
                    if mask.sum() < 5:
                        continue
                    cohort_r2_per_target[t_name][cohort_name].append(
                        _compute_r2(y_te[mask], y_hat[mask])
                    )

        # Aggregate across folds for this seed.
        seed_overall_r2 = _compute_r2(
            np.asarray(all_y_true, dtype=np.float64),
            np.asarray(all_y_pred, dtype=np.float64),
        )
        per_target_mean = {
            t: float(np.mean(v)) if v else 0.0
            for t, v in fold_r2_per_target.items()
        }
        per_target_std = {
            t: float(np.std(v)) if len(v) > 1 else 0.0
            for t, v in fold_r2_per_target.items()
        }
        for t, v in per_target_mean.items():
            per_target_aggregate[t].append(v)

        # Re-fit the best AST on the full training set to capture the
        # symbolic form (for downstream explainability).
        try:
            best_reg = ASTSRRegressor(
                population=population,
                generations=generations,
                depth=depth,
                seed=seed,
            ).fit(X, Y.mean(axis=1) if n_targets > 1 else Y[:, 0])
            best_ast_lisp = best_reg.best_ast_lisp()
        except Exception:
            best_ast_lisp = "<fit_failed>"

        per_seed.append(
            {
                "seed": int(seed),
                "overall_r2": float(seed_overall_r2),
                "per_target_mean_r2": per_target_mean,
                "per_target_std_r2": per_target_std,
                "per_fold": per_fold,
                "wall_time_s": float(time.perf_counter() - t0),
                "n_samples": int(n),
                "n_folds": int(n_folds),
                "collapse_detected_overall": bool(
                    _collapse_detected(
                        np.asarray(all_y_pred, dtype=np.float64),
                        np.asarray(all_y_true, dtype=np.float64),
                    )
                ),
                "per_target_per_cohort_mean_r2": {
                    t: {
                        c: float(np.mean(vs)) if vs else 0.0
                        for c, vs in cohort_r2_per_target[t].items()
                    }
                    for t in spec.target_cols
                },
                "best_ast_lisp": best_ast_lisp,
            }
        )

    summary_per_target = {
        t: {
            "mean": float(np.mean(vs)) if vs else 0.0,
            "std": float(np.std(vs)) if len(vs) > 1 else 0.0,
            "n_seeds": len(vs),
        }
        for t, vs in per_target_aggregate.items()
    }
    overall_r2_per_seed = [s["overall_r2"] for s in per_seed]
    summary = {
        "dataset": spec.name,
        "n_samples": int(n),
        "n_features": int(d_feat),
        "n_targets": int(n_targets),
        "cohorts": cohort_names,
        "feature_columns": list(FEATURE_COLUMNS),
        "n_seeds": len(seeds),
        "astr_population": int(population),
        "astr_generations": int(generations),
        "astr_depth": int(depth),
        "overall_r2_mean": float(np.mean(overall_r2_per_seed))
        if overall_r2_per_seed
        else 0.0,
        "overall_r2_std": float(np.std(overall_r2_per_seed))
        if len(overall_r2_per_seed) > 1
        else 0.0,
        "per_target_summary": summary_per_target,
        "per_seed": per_seed,
    }
    return summary


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def _write_run_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_synthesis(out_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Wf-77 Phase 2 — ASTSR-only R² baseline synthesis",
        "",
        "**Date**: 2026-09-26",
        "**Constraint**: ASTSR-only (no Hilbert, no Multi-PH); per party-twenty-spirit §3.5.",
        "**Feature set**: F5_RDKIT_2D 12-d RDKit 2D descriptors (Wf-49 Phase 2 ship).",
        "",
        "## Cross-dataset summary",
        "",
        "| Dataset | n_samples | overall R² mean ± std | per-target (mean ± std) | collapse? |",
        "|---|---:|---|---|---|",
    ]
    for r in all_results:
        pt_parts = []
        for t, s in r["per_target_summary"].items():
            pt_parts.append(f"{t}: {s['mean']:+.4f}±{s['std']:.4f}")
        collapse_any = any(
            s.get("collapse_detected_overall", False) for s in r["per_seed"]
        )
        lines.append(
            "| {ds} | {n} | {m:+.4f}±{sd:.4f} | {pt} | {c} |".format(
                ds=r["dataset"],
                n=r["n_samples"],
                m=r["overall_r2_mean"],
                sd=r["overall_r2_std"],
                pt="<br>".join(pt_parts),
                c="yes" if collapse_any else "no",
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **ANCHOR role**: this ASTSR-only R² is the Wf-56 R20-P2 "
        "complement of Wf-77 P1 Hilbert-only.  Any composite R² lift "
        "on the same data is attributed to the Hilbert dual / Multi-PH "
        "augmentation rather than to the ASTSR search alone."
    )
    lines.append(
        "- **Feature set**: F5_RDKIT_2D 12-d RDKit 2D descriptors "
        "(Wf-49 Phase 2 ship).  The legacy F5 5-feature set is NOT "
        "used here because Wf-49 §Cause 2 (CRITICAL) confirmed it is "
        "non-informative on pIC50."
    )
    lines.append(
        "- **CPU-only**: the ASTSR evolutionary search runs in fp32 "
        "tournament selection; no GPU acceleration is required or used."
    )
    lines.append(
        "- **Honest framing**: per party-twenty-spirit §3.5, negative "
        "R² values are NOT hidden.  If `overall R² < 0` consistently "
        "across 5 seeds, the composite stack must lift *above* this "
        "number to claim any attribution."
    )
    lines.append("")
    lines.append("## Per-seed R² detail")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['dataset']}")
        lines.append("")
        lines.append("| seed | overall R² | wall_s | collapse |")
        lines.append("|---:|---:|---:|:---:|")
        for s in r["per_seed"]:
            lines.append(
                "| {seed} | {r2:+.4f} | {w:.3f} | {c} |".format(
                    seed=s["seed"],
                    r2=s["overall_r2"],
                    w=s["wall_time_s"],
                    c="yes" if s["collapse_detected_overall"] else "no",
                )
            )
        lines.append("")
    lines.append("## Best AST (lisp form, full-data refit per seed)")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r['dataset']}")
        lines.append("")
        for s in r["per_seed"]:
            lines.append(f"- seed {s['seed']}: `{s.get('best_ast_lisp', '<n/a>')}`")
        lines.append("")
    _write_run_log(out_dir / "synthesis.md", lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-77 Phase 2 — ASTSR-only R² baseline. "
            "Anchors Wf-56 R20-P2 Hilbert-ASTSR composite R²."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["platinai_4cell", "metalcytotox_5metal"],
        choices=["platinai_4cell", "metalcytotox_5metal"],
        help="Datasets to benchmark (default: both).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for K-fold splitting (default: 0..4).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="K-fold count (default: 5).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=32,
        help="ASTSR population size (default: 32).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="ASTSR generations (default: 30).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="ASTSR max AST depth (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("molmetal/reports/wf77_p2_astr_baseline"),
        help="Output directory (default: molmetal/reports/wf77_p2_astr_baseline).",
    )
    args = parser.parse_args(argv)

    specs = _build_dataset_specs(_REPO_ROOT, args.datasets)
    all_results: List[Dict[str, Any]] = []
    for spec in specs:
        print(
            f"[wf77-p2] {spec.name}: loading {spec.csv_path}"
        )
        result = _evaluate_dataset(
            spec,
            args.seeds,
            args.n_folds,
            args.population,
            args.generations,
            args.depth,
        )
        per_seed_path = args.output / spec.name / "per_seed_metrics.json"
        _write_json(per_seed_path, result)
        log_path = args.output / spec.name / "run.log"
        log_lines = [
            f"dataset={spec.name}",
            f"n_samples={result['n_samples']}",
            f"astr_population={result['astr_population']}",
            f"astr_generations={result['astr_generations']}",
            f"astr_depth={result['astr_depth']}",
            f"overall_r2_mean={result['overall_r2_mean']:+.4f}",
            f"overall_r2_std={result['overall_r2_std']:.4f}",
            f"per_target_summary={result['per_target_summary']}",
        ]
        _write_run_log(log_path, log_lines)
        print(
            f"[wf77-p2] {spec.name}: overall R² = "
            f"{result['overall_r2_mean']:+.4f} ± {result['overall_r2_std']:.4f}"
        )
        all_results.append(result)
    _write_synthesis(args.output, all_results)
    print(
        f"[wf77-p2] synthesis.md written to {args.output / 'synthesis.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
