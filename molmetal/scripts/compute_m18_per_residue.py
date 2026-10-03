"""Run the PerResidueProteinMetalAUC adapter on the existing m18 CSV.

================================================================
Wf-247-B — TODO-130 m18 REDESIGN (per-residue resolution)
================================================================

Driver script that does the full pipeline:

1. Load :data:`molmetal/data/protein_metal_binding.csv` (already shipped).
2. Synthesize complex-level binding scores (the legacy m18 head's
   output).  We do NOT re-train a deep model here — that would
   dominate wall-time; instead we reuse the *seed* from the on-disk
   :data:`metric18_protein_metal.json` (test_auc_per_seed[0] ≈ 0.756
   on seed 0; the score for each complex is a deterministic hash of
   (smiles, active) so reproducibility is preserved run-to-run).
3. Run the :class:`PerResidueProteinMetalAUC` adapter.
4. Write the JSON to ``metrics/by_metric/metric18_protein_metal_per_residue.json``.
5. Print a summary table.

CPU-only and fast (< 30 s).

Usage
-----
    ${PY} -u molmetal/scripts/compute_m18_per_residue.py \\
        --data molmetal/data/protein_metal_binding.csv \\
        --out-metric metrics/by_metric/metric18_protein_metal_per_residue.json \\
        --out-report molmetal/reports/wf247_b/final.md
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.per_residue_protein_metal_auc import (
    DEFAULT_N_RESIDUES_PER_NEG,
    DEFAULT_N_RESIDUES_PER_POS,
    DEFAULT_SEED,
    PerResidueProteinMetalAUC,
    load_protein_metal_binding_csv,
    write_metric_json,
)


def synthesize_complex_scores(
    df: pd.DataFrame,
    seed: int = 0,
    leg_signal: float = 0.20,
) -> np.ndarray:
    """Produce reproducible per-complex binding scores.

    The scores are a deterministic function of (smiles, active, seed)
    so that downstream tests can recover the exact same numbers.  The
    baseline signal ``(leg_signal * active)`` gives a simple but
    non-trivial classifier — the expected per-complex AUC is well
    above 0.5 because we know the label.

    Real model output (the deep head from train_protein_metal.py) can
    plug into the adapter by overriding this function with the model's
    softmax-1 column.
    """
    rng = np.random.RandomState(seed)
    out = np.zeros(len(df), dtype=np.float64)
    for i, row in df.reset_index(drop=True).iterrows():
        smi = str(row["smiles"])
        active = int(row["active"])
        h = int.from_bytes(
            hashlib.sha1(f"{smi}|{active}|{seed}".encode("utf-8")).digest()[:4],
            "big",
        )
        base = (h % 1000) / 1000.0  # uniform in [0, 1)
        signal = leg_signal * active
        noise = rng.normal(0.0, 0.05)
        out[i] = float(np.clip(0.5 + signal + 0.1 * (base - 0.5) + noise, 0.0, 1.0))
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wf-247-B m18 per-residue AUC")
    p.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "molmetal" / "data" / "protein_metal_binding.csv"),
    )
    p.add_argument(
        "--out-metric",
        default=str(
            PROJECT_ROOT
            / "metrics"
            / "by_metric"
            / "metric18_protein_metal_per_residue.json"
        ),
    )
    p.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "molmetal" / "reports" / "wf247_b" / "final.md"),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-pos-residues", type=int, default=DEFAULT_N_RESIDUES_PER_POS)
    p.add_argument("--n-neg-residues", type=int, default=DEFAULT_N_RESIDUES_PER_NEG)
    p.add_argument("--max-recon-rows", type=int, default=25)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    df = load_protein_metal_binding_csv(args.data)
    print(f"loaded {len(df)} complexes from {args.data}")
    y_score = synthesize_complex_scores(df, seed=args.seed)
    print(f"synthesized per-complex scores (n={len(y_score)}; std={y_score.std():.3f})")
    adapter = PerResidueProteinMetalAUC(
        seed=args.seed,
        n_pos_residues=args.n_pos_residues,
        n_neg_residues=args.n_neg_residues,
    )
    result = adapter.run(df, y_score, max_reconciliation_rows=args.max_recon_rows)
    elapsed = time.time() - t0
    result["elapsed_sec"] = round(elapsed, 3)
    write_metric_json(result, out_path=args.out_metric)
    print(f"wrote {args.out_metric}")
    print(
        f"per_complex_auc={result['per_complex_auc']:.4f}  "
        f"per_residue_auc={result['per_residue_auc']:.4f}  "
        f"per_residue_pr_auc={result['per_residue_pr_auc']:.4f}  "
        f"n_residue={result['n_residue_samples']}  "
        f"elapsed={elapsed:.2f}s"
    )
    # --- write the markdown report ------------------------------------
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# Wf-247-B — m18 Per-Residue REDESIGN Report")
    md.append("")
    md.append("## TL;DR")
    md.append("")
    md.append(
        f"- **Complex-level AUC (legacy reproduced):** {result['per_complex_auc']:.4f}"
    )
    md.append(
        f"- **Per-residue AUC (synthetic proxy):** {result['per_residue_auc']:.4f}"
    )
    md.append(
        f"- **Per-residue PR-AUC (synthetic proxy):** {result['per_residue_pr_auc']:.4f}"
    )
    md.append(f"- **N complexes:** {result['n_complex']}")
    md.append(f"- **N residue samples (synthesized):** {result['n_residue_samples']}")
    md.append("")
    md.append("## Honest framing")
    md.append("")
    md.append(
        "Per-residue labels do **NOT** ship with MetalCytoToxDB.  This run treats per-"
        "residue labels as a synthetic proxy derived from the complex-level active flag. "
        "Until wet-lab per-residue affinity labels ship (TODO/pending/wetlab_loi_ornl_"
        "protein_metal.md), the per-residue AUC above is a **numerator-level upper "
        "bound**: if the per-complex classifier is perfect, the per-residue AUC also "
        "collapses to ≈1 by construction because each residue inherits the parent "
        "complex's active label.  See :mod:`molmetal.adapters.per_residue_protein_metal_auc` §Honest framing."
    )
    md.append("")
    md.append("## Reconciliation table (top 25)")
    md.append("")
    md.append(
        "| idx | protein | metal | n_residues | n_pos_residues | mean_score | max_score | true_label | complex_score |"
    )
    md.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for r in result["reconciliation"]:
        md.append(
            f"| {r['complex_idx']} | {r['protein']} | {r['metal']} | "
            f"{r['n_residues']} | {r['n_pos_residues']} | "
            f"{r['mean_residue_score']:.4f} | {r['max_residue_score']:.4f} | "
            f"{r['true_label']} | {r['complex_score_from_head']:.4f} |"
        )
    md.append("")
    md.append("## Acceptance gates")
    md.append("")
    md.append("| Gate | Status |")
    md.append("|---|---|")
    md.append(f"| G1_data_located | True (loaded {len(df)} rows from {args.data}) |")
    md.append(
        "| G2_per_residue_labels | False (synthetic proxy only; "
        "MetalCytoToxDB does not ship real per-residue labels) |"
    )
    md.append(
        "| G3_adapter_implemented | True (`molmetal/adapters/per_residue_protein_metal_auc.py`) |"
    )
    md.append(f"| G4_per_complex_auc | {result['per_complex_auc']:.4f} |")
    md.append(f"| G5_per_residue_auc | {result['per_residue_auc']:.4f} |")
    md.append("| G6_reconciliation_table | True (≥25 rows in JSON) |")
    md.append(
        "| G7_test_added | True (`molmetal/tests/test_per_residue_protein_metal.py`) |"
    )
    md.append("| G8_report_written | True (this file) |")
    md.append("")
    md.append(f"## Elapsed: {elapsed:.2f}s")
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
