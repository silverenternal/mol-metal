#!/usr/bin/env python3
"""Regression tests for ``merge_within_metal_slices.py``.

The within-metal paper-funnel run is split into one process per
(metal, split_seed) slice; this script unions the slices back into the canonical
artifact.  Two correctness properties are easy to get wrong and both were real
bugs before being fixed here:

  * the aggregates must equal a direct recomputation (mean over per-fold means,
    population std, paired XGB delta) — mirroring ``_run_metal`` exactly;
  * ``honest_negative`` must be recomputed over the FULL merged table, not
    copied from one slice (a single slice only knows its own metal) — AGENTS.md
    §5.2 (MISSING != silently dropped).

Plus the metal-level SKIPPED reason must survive the merge, and metadata
disagreement across slices must fail loudly rather than silently trusting the
first file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "merge_within_metal_slices.py"

KRASNOV = {"Ru": 0.81, "Ir": 0.73, "Rh": 0.61, "Os": 0.55, "Re": 0.72}
_XGB_FOLDS = [0.80, 0.75, 0.70, 0.65, 0.60]


def _slice(metal: str, seed: int, dmpnn_fold_aucs) -> dict:
    per_fold = []
    for k, (dm, xg) in enumerate(zip(dmpnn_fold_aucs, _XGB_FOLDS)):
        per_fold.append(
            {
                "fold": k, "status": "MEASURED", "n_train": 100, "n_test": 10 * (k + 1),
                "pos_rate_test": 0.5, "dmpnn_seed_aucs": [dm], "dmpnn_auc": dm,
                "dmpnn_acc": 0.7, "xgb_auc": xg, "delta_dmpnn_minus_xgb": dm - xg,
            }
        )
    d = float(np.mean(dmpnn_fold_aucs))
    x = float(np.mean(_XGB_FOLDS))
    row = {
        "split_seed": seed, "dmpnn_auc": d,
        "dmpnn_auc_fold_std": float(np.std(dmpnn_fold_aucs)), "xgb_auc": x,
        "delta_dmpnn_minus_xgb": d - x, "per_fold": per_fold,
    }
    entry = {
        "n_rows_raw": 1000, "n_unique": 500, "pos_rate": 0.5, "resolvable": True,
        "krasnov_reported_auc": KRASNOV.get(metal), "status": "MEASURED",
        "n_splits": 5, "split_seeds": [seed], "n_folds_measured": len(dmpnn_fold_aucs),
        "dmpnn_auc": d, "dmpnn_auc_across_folds_std": row["dmpnn_auc_fold_std"],
        "xgb_auc_same_folds": x, "delta_dmpnn_minus_xgb": d - x,
        "dmpnn_beats_xgb": bool(d > x),
        "dmpnn_beats_krasnov": bool(d > KRASNOV[metal]),
        "paired_split_seed_42": (
            {"dmpnn_auc": d, "xgb_auc": x, "delta_dmpnn_minus_xgb": d - x}
            if seed == 42 else None
        ),
        "per_split": [row], "wall_sec": 1.0,
    }
    return {
        "metric": "dmpnn_within_metal_auc", "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED", "unit": "ROC-AUC", "scope": "s",
        "source": "molmetal/scripts/dmpnn_within_metal_auc.py",
        "data_source": "MetalCytoToxDB", "cohort": "c", "cohort_spec": "paper_funnel",
        "fold_protocol": "f", "dmpnn_arch": {"hidden": 128},
        "xgb_reference": {"model": "m"}, "split_seeds": [seed], "device": "cpu",
        "krasnov_within_metal_auc": KRASNOV, "min_unique_for_gate": 40,
        "per_metal": {metal: entry},
        "honest_negative": {
            "pooled_mean_reported": False, "why": "w",
            "unresolvable_metals": [], "dmpnn_below_xgb": [],
            "fallback_if_dmpnn_loses": "fb",
        },
    }


def _run_merge(tmp_path: Path, slices: dict, name="m.json"):
    sd = tmp_path / "slices"
    sd.mkdir(parents=True, exist_ok=True)
    for (metal, seed), rep in slices.items():
        (sd / f"{metal}_s{seed}.json").write_text(json.dumps(rep))
    out = tmp_path / name
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--slices-dir", str(sd), "--out-metric", str(out)],
        capture_output=True, text=True,
    )
    return r, (json.loads(out.read_text()) if out.exists() else None)


def test_aggregates_match_recomputation(tmp_path):
    slices = {
        ("Ru", 42): _slice("Ru", 42, [0.80, 0.78, 0.76, 0.74, 0.72]),
        ("Ru", 43): _slice("Ru", 43, [0.81, 0.79, 0.77, 0.75, 0.73]),
        ("Ir", 42): _slice("Ir", 42, [0.70, 0.68, 0.66, 0.64, 0.62]),
        ("Ir", 43): _slice("Ir", 43, [0.71, 0.69, 0.67, 0.65, 0.63]),
    }
    r, mg = _run_merge(tmp_path, slices)
    assert r.returncode == 0, r.stderr
    assert mg["split_seeds"] == [42, 43]
    assert mg["merged_from_slices"]["n_slice_files"] == 4
    for metal in ("Ru", "Ir"):
        all_d = []
        for s in (42, 43):
            pf = slices[(metal, s)]["per_metal"][metal]["per_split"][0]["per_fold"]
            all_d.extend(f["dmpnn_auc"] for f in pf)
        e = mg["per_metal"][metal]
        assert abs(e["dmpnn_auc"] - float(np.mean(all_d))) < 1e-12
        assert abs(e["dmpnn_auc_across_folds_std"] - float(np.std(all_d))) < 1e-12
        assert abs(e["xgb_auc_same_folds"] - float(np.mean(_XGB_FOLDS * 2))) < 1e-12
        assert e["n_folds_measured"] == 10
        assert e["split_seeds"] == [42, 43]
        # paired seed-42 record must equal the seed-42 per_split row
        assert e["paired_split_seed_42"]["dmpnn_auc"] == e["per_split"][0]["dmpnn_auc"]
        assert e["dmpnn_beats_krasnov"] == (e["dmpnn_auc"] > KRASNOV[metal])
        assert "wall_sec" not in e


def test_honest_negative_recomputed_over_full_table(tmp_path):
    """A slice only knows its own metal; the merged block must span all metals."""
    slices = {
        ("Ru", 42): _slice("Ru", 42, [0.80, 0.78, 0.76, 0.74, 0.72]),
        ("Ir", 42): _slice("Ir", 42, [0.70, 0.68, 0.66, 0.64, 0.62]),
    }
    r, mg = _run_merge(tmp_path, slices)
    assert r.returncode == 0, r.stderr
    # Ir dmpnn 0.665 < xgb 0.700; Ru 0.76 > 0.70
    assert mg["honest_negative"]["dmpnn_below_xgb"] == ["Ir"], mg["honest_negative"]
    assert mg["honest_negative"]["unresolvable_metals"] == []
    # prose fields carried through unchanged
    assert mg["honest_negative"]["pooled_mean_reported"] is False
    assert mg["honest_negative"]["fallback_if_dmpnn_loses"] == "fb"


def test_metal_level_skip_reason_preserved(tmp_path):
    skip = _slice("Os", 42, [0.60, 0.58, 0.56, 0.54, 0.52])
    os_entry = skip["per_metal"]["Os"]
    os_entry.clear()
    os_entry.update({
        "n_rows_raw": 5, "n_unique": 5, "pos_rate": 0.4, "resolvable": False,
        "krasnov_reported_auc": 0.55, "status": "SKIPPED",
        "reason": "n=5 too small or single-class",
    })
    slices = {("Os", 42): skip, ("Ru", 42): _slice("Ru", 42, [0.80, 0.78, 0.76, 0.74, 0.72])}
    r, mg = _run_merge(tmp_path, slices)
    assert r.returncode == 0, r.stderr
    assert mg["per_metal"]["Os"]["status"] == "SKIPPED"
    assert mg["per_metal"]["Os"]["reason"] == "n=5 too small or single-class"
    assert mg["honest_negative"]["unresolvable_metals"] == ["Os"]
    assert mg["per_metal"]["Ru"]["status"] == "MEASURED"


def test_metadata_disagreement_fails(tmp_path):
    a = _slice("Ru", 42, [0.80, 0.78, 0.76, 0.74, 0.72])
    b = _slice("Ru", 43, [0.81, 0.79, 0.77, 0.75, 0.73])
    b["cohort_spec"] = "24h"  # stale slice from a different run
    r, _ = _run_merge(tmp_path, {("Ru", 42): a, ("Ru", 43): b})
    assert r.returncode != 0
    assert "disagreement" in (r.stdout + r.stderr)


def test_empty_dir_raises(tmp_path):
    sd = tmp_path / "empty"
    sd.mkdir()
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--slices-dir", str(sd),
         "--out-metric", str(tmp_path / "m.json")],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "no slice JSONs" in (r.stdout + r.stderr)
