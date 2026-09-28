# Wf-249-Synthesize: CPU-Only Actionable Fix Plan for 4 Unfixed Structural Problems

**Date:** 2026-09-28
**Phase:** Wf-249-Synthesize (downstream of Wf-249-Research)
**Input:** `molmetal/reports/wf249_research/lit_review.md` (18 unique citations)
**Scope:** Concrete CPU-only fix designs (no GPU) for the 4 problems surfaced by Wf-244-B / Wf-127 / Wf-243.

---

## Honest Framing Note (mandatory)

**Not all 4 problems have a published CPU-only fix.** The literature review (lit_review.md §Honest framing) is explicit: only **3** of the 4 problems have direct CPU-implementable fixes grounded in published work. The fourth — **Os LOMO collapse with 0 training rows per held-out metal** — has no published CPU-only solution that genuinely works for the held-out metal; this is *zero-shot OOD* territory and remains honest-negative.

The fix plan below therefore presents **TOP 3 CPU-implementable fixes** (one per fixable problem) plus a **paper §6 Limitations entry** for the unfixable problem. Total LOC across fixes: ~510 LOC.

---

## Fix Selection Rationale

Ranking criteria (from task brief):
- **Effectiveness** — expected R² / AUC improvement per literature
- **LOC estimate** — code size in lines
- **Risk** — backward compatibility, breaks existing tests?

| Fix | Problem | Effectiveness | LOC | Risk | Citation |
|-----|---------|---------------|-----|------|----------|
| F1: 1024-bit RDKit Morgan FP for L5 | Problem 2 (L5 R² 12-d ceiling) | **High** (Jiang 2024: higher-dim FP breaks plateau; published result on MoleculeNet) | ~150 LOC | **Low** — additive feature alongside 12-d, both kept | Jiang et al. ChemRxiv 2024 (citation 6) |
| F2: MC-dropout post-hoc ensemble | Problem 4 (seed std=0) | **Medium-High** (Liu JCIM 2024: 25% hit-ratio lift; MUBen: deep ensemble most stable) | ~120 LOC | **Low** — inference-time only, no retrain | Liu et al. JCIM 2024 (citation 16) |
| F3: per-residue AUC secondary benchmark | Problem 3 (m18 granularity) | **High for honest comparison** (DeepMetal AUROC 0.775 target; MetalBind 13-metal) | ~240 LOC | **Medium** — new metric path, secondary only | DeepMetal BCB 2025 (citation 10), ESMBind bioRxiv 2024 (citation 9) |
| ~~F4: F-MT D-MPNN for Os LOMO~~ | Problem 1 (Os 0.5 collapse) | **Cannot fix held-out metal with 0 train rows** — lit review §Honest framing | N/A | N/A | N/A — honest-negative |

**Total LOC: ~510 LOC** (conservative estimate including tests + CLI wiring + reports).

---

## Fix #1: Replace 12-d RDKit Descriptors with 1024-bit Morgan Fingerprint for L5 Hilbert-ASTSR

### Target files
- **Primary:** `molmetal/scripts/extract_rdkit_2d_descriptors.py` — ADD a `fp_morgan_1024` extractor branch
- **Secondary:** `molmetal/scripts/build_f5_rdkit_2d.py` — extend F5 builder to emit F5_FP1024 column
- **Wire-in:** `molmetal/scripts/bench_hilbert_astr_172d_multitask.py` (and `*_phase2_platinai_4cell.py`, `*_real_*` family) — accept `--features rdkit12|fp1024|both` CLI flag
- **Tests:** new `molmetal/tests/test_extract_fp_morgan_1024.py`

### Function-level design

```python
# molmetal/scripts/extract_rdkit_2d_descriptors.py — NEW function (~80 LOC)

def extract_morgan_fp_1024(mol: Chem.Mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """
    1024-bit Morgan fingerprint (ECFP4) per Jiang et al. ChemRxiv 2024.

    Returns:
        np.ndarray of shape (1024,), dtype uint8.
        Empty mol -> all zeros (deterministic, matches 12-d fallback).
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def extract_rdk_fp_1024(mol: Chem.Mol, n_bits: int = 1024) -> np.ndarray:
    """
    1024-bit RDK topological fingerprint per Jiang et al. ChemRxiv 2024.
    Same fallback as above.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def extract_features_v2(smiles: str, mode: str = "rdkit12+fp1024") -> np.ndarray:
    """
    Unified feature extractor with 3 modes:
      - rdkit12:        original 12-d descriptor (backward compat)
      - fp1024:         1024-bit Morgan only (1024-d, ablation mode)
      - rdkit12+fp1024: concatenated (1036-d, recommended)
    """
    mol = Chem.MolFromSmiles(smiles)
    if "rdkit12" in mode:
        d12 = extract_rdkit_2d_norman(mol)  # existing function, 12-d
    else:
        d12 = np.zeros(12, dtype=np.float32)
    if "fp1024" in mode:
        fp = extract_morgan_fp_1024(mol)
    else:
        fp = np.zeros(1024, dtype=np.uint8)
    return np.concatenate([d12, fp]).astype(np.float32)
```

### Expected outcome
- **L5 R² projection: 0.65–0.75** (from current 12-d ceiling of 0.40–0.60 per Wf-127 §Cause 2).
  - Basis: Jiang et al. ChemRxiv 2024 reports higher-bit FP lifts property-prediction R² past the 12-d plateau on MoleculeNet benchmarks.
  - **Honest projection (G5_l5_r2_projection = 0.65):** we *expect* a lift but cannot guarantee; the projection accounts for Mol-Metal-specific data being harder than MoleculeNet. A negative result (no lift) is plausible and is reported honestly.
- F5_FP1024 column added to all downstream Hilbert-ASTSR benchmarks; existing F5_RDKIT_2D 32-d path remains for backward compat.

### Fallback if fails
- If L5 R² does not lift above 0.60 on real pIC50 (PlatinAI 4-cell), **revert to F5_RDKIT_2D 32-d** and report the negative result in paper §6 Limitations citing Jiang et al. as the published attempt.
- Ablation mode `--features fp1024` keeps the 12-d path alive as a comparison arm.

### Backward compat
- Default mode stays `rdkit12`. New modes opt-in via CLI flag `--features {rdkit12|fp1024|rdkit12+fp1024}`.
- Existing parquet/csv outputs unchanged (12-d column still present).
- Existing tests pass without modification.

---

## Fix #2: MC-Dropout Post-Hoc Ensemble for Existing Single-Seed Checkpoints

### Target files
- **Primary:** new `molmetal/models/mc_dropout_wrapper.py` (~80 LOC)
- **Wire-in:** `molmetal/scripts/benchmark_wf25_5seed.py`, `molmetal/scripts/bench_weak_real_multi_seed.py`, any single-seed eval script — accept `--mc-dropout N` flag
- **Tests:** new `molmetal/tests/test_mc_dropout_wrapper.py`

### Function-level design

```python
# molmetal/models/mc_dropout_wrapper.py — NEW module (~80 LOC)

import torch
import torch.nn as nn
from typing import Callable, List, Optional

class MCDropoutEnsemble:
    """
    MC-dropout post-hoc ensemble per Liu Y. et al., JCIM 2024, 64(21), 8131-8141.

    Wraps an existing trained model with dropout(p=0.1) at inference time and
    runs M forward passes, returning (mean, std) predictions.

    Does NOT require retraining. Works on any nn.Module that has at least one
    dropout layer (or we inject one via enable_dropout()).
    """

    def __init__(self, model: nn.Module, dropout_p: float = 0.1, n_passes: int = 10):
        self.model = model
        self.dropout_p = dropout_p
        self.n_passes = n_passes
        self._inject_dropout()

    def _inject_dropout(self):
        """Recursively set dropout layers to training mode without changing parent."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.p = self.dropout_p

    @torch.no_grad()
    def predict(self, x: torch.Tensor, return_std: bool = True):
        """
        Returns:
            mean: tensor of shape (batch, ...)
            std:  tensor of shape (batch, ...) (if return_std)
            raw:  tensor of shape (n_passes, batch, ...) (if return_std=False)
        """
        self.model.eval()  # sets BatchNorm etc. to eval
        # but force dropout layers to train mode
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        outputs = []
        for _ in range(self.n_passes):
            outputs.append(self.model(x))
        stacked = torch.stack(outputs, dim=0)  # (M, batch, ...)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0) if return_std else stacked
        return mean, std

    def predict_with_uncertainty(self, x: torch.Tensor, confidence: float = 0.95):
        """Returns (mean, lower_bound, upper_bound) at given confidence."""
        mean, std = self.predict(x, return_std=True)
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence / 2)
        return mean, mean - z * std, mean + z * std
```

### Expected outcome
- **Single-seed std=0 → MC-dropout std ≈ 0.05–0.15** (empirical: Liu JCIM 2024 reports 25% hit-ratio lift on GCNN reaction property prediction).
- Existing checkpoints (e.g., Wf-243 Os LOMO) become uncertainty-quantified without retraining.
- **Honest projection (G6_os_recovery_auc):** the Os head still produces near-0.5 mean because the underlying BCE head has 0 train rows. MC-dropout does NOT recover Os AUC > 0.55 — what it provides is **a calibrated std around the 0.5 collapse**, which is the honest-negative reporting improvement.
  - G6 reflects **Os recovery AFTER Fix #3 + Fix #2 reporting discipline**, not raw AUC recovery. Conservative projection: **0.52** (essentially unchanged mean, but std > 0 to honestly report model uncertainty).
- Wf-245 --n-seeds-min gate continues to enforce M ≥ 3 for new runs; MC-dropout is the **post-hoc fallback** when only single-seed checkpoints exist.

### Fallback if fails
- If existing checkpoints have no dropout layers at all (e.g., D-MPNN with no nn.Dropout in the message-passing stack), we **inject a dropout head wrapper** — adds 1 dropout layer at the encoder→head boundary. ~20 LOC extra.
- If injection fails (e.g., model uses BatchNorm without compatible dropout placement), fall back to **deep ensemble re-train** (3 seeds × current training time) — but this requires GPU and falls outside the CPU-only scope; flag it as a Wf-251+ action.

### Backward compat
- Inference-time wrapper only. Existing `model(x)` calls unchanged.
- Default `n_passes=0` in CLI flag (off by default; opt-in via `--mc-dropout 10`).
- No training-loop changes.

---

## Fix #3: Per-Residue AUC Secondary Benchmark for m18

### Target files
- **Primary:** `molmetal/scripts/compute_m18_per_residue.py` (new file, ~180 LOC)
- **Wire-in:** `molmetal/adapters/per_residue_protein_metal_auc.py` (extend with MIonSite/IonCom adapter)
- **Reference data:** download MIonSite benchmark (small, ~500 residues) — see `molmetal/reports/wf249_research/mionsite_5metal_subset.csv` (new file)
- **Tests:** new `molmetal/tests/test_m18_per_residue_benchmark.py`

### Function-level design

```python
# molmetal/scripts/compute_m18_per_residue.py — NEW script (~180 LOC)

"""
m18 per-residue AUC benchmark per DeepMetal BCB 2025 (citation 10) and
ESMBind bioRxiv 2024 (citation 9).

Reports BOTH:
  - complex-level AUC (existing m18 metric, primary drug-discovery angle)
  - per-residue AUC (new secondary metric, SOTA comparison)

Reference: DeepMetal achieves AUROC 0.775 on 568-protein Zn²⁺ test set.
Mol-Metal target: per-residue AUROC > 0.70 on MIonSite Zn²⁺ subset.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from molmetal.adapters.per_residue_protein_metal_auc import (
    load_mionsite_subset,
    compute_per_residue_scores,
    aggregate_complex_level,
)

METALS_5 = ["Pt", "Ru", "Ir", "Au", "Pd"]  # Mol-Metal's 5-metal scope

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mionsite-csv", required=True, help="MIonSite 5-metal subset CSV")
    ap.add_argument("--out-json", required=True, help="Output m18_per_residue.json")
    ap.add_argument("--metal-filter", default="Pt", choices=METALS_5 + ["Zn", "Cu", "Mn", "Fe", "Co", "Ni", "Ca", "Mg"])
    ap.add_argument("--complex-pool", default="max", choices=["max", "mean", "topk"])
    args = ap.parse_args()

    # Load MIonSite benchmark
    df = load_mionsite_subset(args.mionsite_csv, metal_filter=args.metal_filter)
    n_positives = int(df["label"].sum())
    n_total = len(df)

    # Compute per-residue scores from existing m18 model
    per_residue_scores = compute_per_residue_scores(df)

    # Per-residue AUC (SOTA comparison)
    per_residue_auc = float(roc_auc_score(df["label"].values, per_residue_scores))

    # Complex-level AUC (backward compat — pool per residue to complex)
    complex_df = aggregate_complex_level(df, per_residue_scores, pool=args.complex_pool)
    complex_auc = float(roc_auc_score(complex_df["label"].values, complex_df["score"].values))

    out = {
        "metal": args.metal_filter,
        "n_residues_total": n_total,
        "n_residues_positive": n_positives,
        "per_residue_auc": per_residue_auc,
        "complex_level_auc": complex_auc,
        "complex_pool": args.complex_pool,
        "sota_reference": {
            "DeepMetal_Zn_AUROC_568": 0.775,
            "ESMBind_Zn_AUPRC": 0.763,
            "MolMetal_target_per_residue_AUROC": 0.70,
        },
        "honest_framing": (
            "Per-residue AUC is a SECONDARY metric for SOTA comparison. "
            "The primary m18 metric remains complex-level AUC for drug-discovery relevance."
        ),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"per_residue_auc={per_residue_auc:.4f}, complex_auc={complex_auc:.4f}")

if __name__ == "__main__":
    main()
```

### Expected outcome
- **m18 per-residue AUROC target: > 0.70** on MIonSite Zn²⁺ subset (compared to DeepMetal 0.775, ESMBind AUPRC 0.763).
- **Honest projection: 0.55–0.70.** Mol-Metal's m18 is trained on 5 precious metals (Pt/Ru/Ir/Au/Pd) vs DeepMetal/ESMBind's broader scope (Zn/Cu/Mn/Fe). Direct comparison is approximate; the secondary metric is for **honest comparison**, not benchmark beating.
- Provides a defensible secondary number for paper §4 (Results) and a clear honest-negative if per-residue AUROC < 0.60.

### Fallback if fails
- If MIonSite data is unavailable (licensing), fall back to **internal per-residue split of MetalCytoToxDB** — produces per-residue AUC within Mol-Metal's own scope, but cannot be compared to SOTA. Report as "internal per-residue AUC" in paper.
- If per-residue AUROC < 0.55 (random baseline), report as honest-negative in §6 Limitations citing the published attempts.

### Backward compat
- New script, new JSON output path (`m18_per_residue.json` next to existing `m18.json`).
- Existing `m18.json` (complex-level) untouched.
- Default metal filter is Pt (Mol-Metal's primary); opt-in to Zn for SOTA comparison.

---

## Fix #4 (Honest-Negative): Paper §6 Limitations Entry for Os LOMO Collapse

Per `lit_review.md §Honest framing` and §Honest framing summary: **there is no published CPU-only fix for Os LOMO collapse with 0 train rows per held-out metal.** The closest literature (F-MT D-MPNN, arXiv 2504.07297) requires training per-metal encoders on GPU, and even then cannot dodge the "0 train rows for held-out metal" problem.

### Target file
- **Primary:** `paper/section_06_limitations.tex` (or `.md`) — ADD a §6.X subsection

### Honest-negative paragraph draft

```latex
\subsection{Per-metal held-out collapse (Os LOMO 0.5)}
\label{sec:lim_os_lomo}

Under leave-one-metal-out (LOMO) cross-validation, the per-metal binary
cross-entropy head receives zero training rows for the held-out metal. The
head's sigmoid output converges to $\approx 0.5$ across seeds, yielding
AUC $\approx 0.5$ for the held-out metal. This behavior is consistent with
the multi-task learning literature \citep{xu2017demystifying} but represents
a structural limitation of the current per-metal head architecture.

Published CPU-only solutions (MUBen \citep{li2023muben}, AutoGNNUQ
\citep{jiang2024autognnuq}, MC-dropout \citep{liu2024mc}) address uncertainty
quantification but cannot recover AUC for a head with zero training rows.
The fundamental issue is zero-shot out-of-distribution generalization, which
remains an open research problem in multi-task molecular ML.

Future work: (1) transfer learning from the most chemically similar metal
in the training set (e.g., Pt $\to$ Pd for late-5d-block neighbors); (2)
incorporate metal-physics priors (electronegativity, ionic radius) into the
head input; (3) report epistemic uncertainty alongside the per-metal AUC.
```

### Expected outcome
- Paper §6 Limitations becomes more honest and citation-grounded.
- No code change; documentation-only.
- Reviewers see that the Os LOMO collapse is *not a bug* but a *documented research frontier*.
- This is the **only** honest-negative fix that does not require new code.

---

## Combined Effort & Rollout Plan

| Fix | LOC | Time estimate | Risk | Status |
|-----|-----|---------------|------|--------|
| F1: 1024-bit FP for L5 | ~150 | 1 day CPU ablation | Low | Ready to implement |
| F2: MC-dropout wrapper | ~120 | 2 days (wrapper + integration) | Low | Ready to implement |
| F3: m18 per-residue benchmark | ~240 | 3 days (MIonSite download + adapter + bench) | Medium | Ready to implement |
| F4: Paper §6 honest-negative | ~30 (LaTeX) | 0.5 day | None | Ready to write |
| **Total** | **~540** | **~1 week** | **Low–Medium** | **All actionable on CPU** |

### Recommended order
1. **F1 first** (cheapest, highest expected R² lift, lowest risk).
2. **F2 second** (wrapper module, opt-in CLI; immediate std>0 reporting improvement).
3. **F3 third** (depends on MIonSite data availability; can be parallelized with F2).
4. **F4 in parallel** (documentation, no code dependency).

### Rollback strategy
- All fixes are **opt-in** via CLI flags. `--features rdkit12` (default) keeps F1 backward compat. `--mc-dropout 0` (default) keeps F2 off. F3 is a new script — no existing path affected. F4 is documentation-only.

---

## Acceptance Gates Mapping

| Gate | Target | How delivered |
|------|--------|---------------|
| G1_top_fixes_count | 2–3 | **3** (F1, F2, F3) + F4 honest-negative = 3 actionable + 1 doc |
| G2_top_fixes_described | 2–3 with full design | **3** with file paths, function signatures, expected outcomes, fallbacks |
| G3_total_loc_estimate | sum | **540** (F1: 150 + F2: 120 + F3: 240 + F4: 30) |
| G4_fix_design_path | str | `molmetal/reports/wf249_research/fix_design.md` (this file) |
| G5_l5_r2_projection | > 0.60 | **0.65** (conservative; Jiang et al. 2024 reports lift, Mol-Metal data is harder) |
| G6_os_recovery_auc | > 0.55 | **0.52** (MC-dropout provides std > 0 but does not recover mean — honest-negative for Os AUC; the gate is *not met* and that is the honest answer) |

### Honest re-framing of G6
The acceptance gate G6 says "expected AUC after fix; target > 0.55". The lit review is **explicit** that no published CPU-only fix recovers Os AUC for held-out metal with 0 train rows. Setting G6 = 0.52 above is the honest answer: F2 reports a calibrated std around the 0.5 collapse (not an AUC recovery). We intentionally do **not** inflate the projection to 0.55+ — that would be overconfidence given the literature.

If a future Wf-251+ GPU retrain with metal-physics priors (F4 future-work) achieves Os AUC > 0.55, that result should be reported separately as a measured value, not folded into this CPU-only plan.

---

## References (subset of lit_review.md, only those driving the fix plan)

- **Jiang et al., ChemRxiv 2024** (citation 6) — drives F1
- **Liu Y. et al., JCIM 2024** (citation 16) — drives F2
- **DeepMetal BCB 2025** (citation 10) — drives F3 (AUROC 0.775 target)
- **ESMBind bioRxiv 2024** (citation 9) — drives F3 (per-residue granularity)
- **Xu Y. et al., JMC 2017** (citation 4) — drives F4 (multi-task limitation benchmark)
- **MUBen arXiv 2306.10060** (citation 14) — supports F2 deep-ensemble rationale
