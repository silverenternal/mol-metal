"""Wf-329-W4 — Multi-PH/PLL feature extraction smoke script.

================================================================
Wf-329-W4 (2026-09-29) — CPU-only smoke entry point for the
32-bin Fourier + Multi-PH + PLL feature extraction wire-in
(module :mod:`molmetal.molmetal_lam.reward.multiph_pll_feature`).

Run on the metallo cohort by default.  Writes ``result.json`` to
the supplied ``--output-dir`` (default
``molmetal/reports/wf329w4_multiph_pll``).

Honest framing per AGENTS.md §5.2:
* No GPU launch; CPU-only feature extraction.
* No retrain; downstream regressor is Ridge (CPU sklearn).
* MEASURED-vs-PROJECTED tags in the result JSON.

Run with::

    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf329w4_multiph_pll_smoke.py \\
        --output-dir molmetal/reports/wf329w4_multiph_pll
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# gfx1101 env vars applied by __init__ if present
try:
    import molmetal  # noqa: F401
except Exception:
    pass

# Quiet RDKit warnings — the metallo cohort has many non-standard
# atom types (Sc2+3, Pt2+, Y_5 etc.) that RDKit logs UFFTYPER
# errors for.
try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_cohort(csv_path: Path, max_samples: int) -> list:
    """Load SMILES from the metallo cohort CSV."""
    import csv

    smiles: list = []
    if not csv_path.exists():
        return smiles
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                smiles.append(smi)
            if len(smiles) >= max_samples:
                break
    return smiles


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"),
        help="Path to metallo_drugs_500_train.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "molmetal" / "reports" / "wf329w4_multiph_pll"),
        help="Where to write result.json + per-cohort csvs",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Cap cohort size for the smoke run (CPU-only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260929,
        help="RNG seed for the m7 sample_size selector",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Wf-329-W4] Loading cohort from {csv_path} (max {args.max_samples}) ...")
    t0 = time.time()
    smiles = _load_cohort(csv_path, args.max_samples)
    t_load = time.time() - t0
    print(f"[Wf-329-W4] Loaded {len(smiles)} SMILES in {t_load:.2f}s")

    if not smiles:
        # Fallback to a tiny synthetic cohort so the smoke doesn't
        # crash on an empty environment.
        smiles = ["CCO", "c1ccccc1", "[Pt](N)(N)(Cl)Cl"]
        print(f"[Wf-329-W4] CSV empty; using synthetic fallback ({len(smiles)} SMILES)")

    # Late import so the gfx1101 + RDKit env vars are applied first.
    from molmetal.molmetal_lam.reward.multiph_pll_feature import (
        MultiPHPLLFeatureExtractor,
        MultiPHPLLRecoveryReport,
        wire_m6_kappa_recovery,
        wire_m7_diversity_recovery,
        wire_r2_recovery,
    )

    extractor = MultiPHPLLFeatureExtractor()
    print(
        f"[Wf-329-W4] Extractor: n_bins={extractor.n_bins} "
        f"(Fourier={extractor.fourier_dim}, "
        f"multi-PH={extractor.multi_ph_dim}, "
        f"PLL={extractor.pll_dim}, joint={extractor.joint_dim})"
    )

    # ----- m6_kappa recovery -----
    print("[Wf-329-W4] Running m6_kappa recovery wire-in ...")
    t0 = time.time()
    m6 = wire_m6_kappa_recovery(smiles, extractor)
    t_m6 = time.time() - t0
    print(
        f"[Wf-329-W4] m6_kappa: baseline={m6['m6_kappa_baseline']:.4f} "
        f"-> augmented={m6['m6_kappa_augmented']:.4f} "
        f"(delta={m6['m6_kappa_delta']:+.4f}, "
        f"r2={m6.get('augmented_r2', float('nan')):.4f}, "
        f"n={m6['n_samples']}, {t_m6:.2f}s)"
    )

    # ----- m7 diversity recovery -----
    print("[Wf-329-W4] Running m7 diversity recovery wire-in (PLL distance) ...")
    t0 = time.time()
    m7 = wire_m7_diversity_recovery(
        smiles,
        extractor,
        sample_size=min(30, len(smiles)),
        seed=args.seed,
    )
    t_m7 = time.time() - t0
    print(
        f"[Wf-329-W4] m7: Tanimoto={m7['m7_diversity_baseline']:.4f} "
        f"-> PLL={m7['m7_diversity_pll']:.4f} "
        f"(delta={m7['m7_diversity_delta']:+.4f}, "
        f"n={m7['n_samples']}, {t_m7:.2f}s)"
    )

    # ----- R² recovery -----
    print("[Wf-329-W4] Running R² recovery wire-in (joint design matrix + Ridge) ...")
    t0 = time.time()
    r2 = wire_r2_recovery(smiles, extractor)
    t_r2 = time.time() - t0
    print(
        f"[Wf-329-W4] R²: baseline={r2['r2_baseline']:.4f} "
        f"-> joint={r2['r2_joint']:.4f} "
        f"(delta={r2['r2_delta']:+.4f}, "
        f"joint_dim={r2['joint_dim']}, "
        f"n={r2['n_samples']}, {t_r2:.2f}s)"
    )

    # ----- Aggregated report -----
    report = MultiPHPLLRecoveryReport(
        m6_kappa_augmented=m6["m6_kappa_augmented"],
        m6_kappa_baseline=m6["m6_kappa_baseline"],
        m7_diversity_pll=m7["m7_diversity_pll"],
        m7_diversity_baseline=m7["m7_diversity_baseline"],
        r2_joint=r2["r2_joint"],
        r2_baseline=r2["r2_baseline"],
        wall_seconds=t_m6 + t_m7 + t_r2,
        notes={
            "m6": (
                f"baseline {m6['m6_kappa_baseline']:.4f} -> "
                f"augmented {m6['m6_kappa_augmented']:.4f} "
                f"(delta={m6['m6_kappa_delta']:+.4f}, n={m6['n_samples']})"
            ),
            "m7": (
                f"Tanimoto {m7['m7_diversity_baseline']:.4f} -> "
                f"PLL {m7['m7_diversity_pll']:.4f} "
                f"(delta={m7['m7_diversity_delta']:+.4f}, n={m7['n_samples']})"
            ),
            "r2": (
                f"baseline {r2['r2_baseline']:.4f} -> joint "
                f"{r2['r2_joint']:.4f} (delta={r2['r2_delta']:+.4f}, "
                f"n={r2['n_samples']}, joint_dim={r2['joint_dim']})"
            ),
            "fourier_features": (
                "32-bin radially-binned magnitude spectrum of multi-PH "
                "feature map (Kim-Jung 2026 §4 acceleration)"
            ),
            "multi_ph_features": (
                f"TwoParameterFiltration n_bins={extractor.n_bins} "
                f"(W2.3 R18 P3 surface)"
            ),
            "pll_features": (f"PLL top-{extractor.k} eigenvalues (Liu-Feng 2026)"),
        },
    )

    result = report.as_dict()
    result["n_smiles_loaded"] = len(smiles)
    result["t_load_seconds"] = round(t_load, 4)
    result["t_m6_seconds"] = round(t_m6, 4)
    result["t_m7_seconds"] = round(t_m7, 4)
    result["t_r2_seconds"] = round(t_r2, 4)
    result["csv_path"] = str(csv_path)
    result["max_samples"] = args.max_samples
    result["extractor"] = {
        "n_bins": extractor.n_bins,
        "k": extractor.k,
        "knn": extractor.knn,
        "max_radius": extractor.max_radius,
        "max_coord": extractor.max_coord,
        "fourier_dim": extractor.fourier_dim,
        "multi_ph_dim": extractor.multi_ph_dim,
        "pll_dim": extractor.pll_dim,
        "joint_dim": extractor.joint_dim,
    }

    out_path = output_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[Wf-329-W4] Wrote {out_path}")

    # Also dump per-axis csv for downstream consumers.
    import csv as _csv

    csv_path_out = output_dir / "per_axis.csv"
    with open(csv_path_out, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=[
                "axis",
                "baseline",
                "recovered",
                "delta",
                "n_samples",
                "wall_seconds",
                "tag",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "axis": "m6_kappa",
                "baseline": round(m6["m6_kappa_baseline"], 6),
                "recovered": round(m6["m6_kappa_augmented"], 6),
                "delta": round(m6["m6_kappa_delta"], 6),
                "n_samples": m6["n_samples"],
                "wall_seconds": round(t_m6, 4),
                "tag": "MEASURED",
            }
        )
        writer.writerow(
            {
                "axis": "m7_diversity",
                "baseline": round(m7["m7_diversity_baseline"], 6),
                "recovered": round(m7["m7_diversity_pll"], 6),
                "delta": round(m7["m7_diversity_delta"], 6),
                "n_samples": m7["n_samples"],
                "wall_seconds": round(t_m7, 4),
                "tag": "MEASURED",
            }
        )
        writer.writerow(
            {
                "axis": "r2",
                "baseline": round(r2["r2_baseline"], 6),
                "recovered": round(r2["r2_joint"], 6),
                "delta": round(r2["r2_delta"], 6),
                "n_samples": r2["n_samples"],
                "wall_seconds": round(t_r2, 4),
                "tag": "MEASURED",
            }
        )
    print(f"[Wf-329-W4] Wrote {csv_path_out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
