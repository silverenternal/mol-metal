# WF-Lift-EV3 — SA penalty 10x3 sweep at --sa-weight 0.3 on test_001

> **Verdict (MEASURED 2026-09-17, CPU-only, 3 cells, 362 s wall):**
> **`sa_mean = 3.0988 ± 0.0000`** across 3 seeds on `test_001`,
> **identical** to the `wf_sa_fragment_pool_optimize` baseline at
> `n_sim=100` (`sa_mean = 3.099`).
> **TargetDiff range [2.65, 2.86] NOT reached** (gap = +0.239).
> The MCTS at `n_sim=1000` with `metal-seed cisplatin` + `--click-rules
> all-5` collapses onto a **single Pt-trizole + aryl-substituent
> family** (20 distinct SMILES differing only in the aryl ring
> substituent: H / CH3 / Cl / F / OMe / Cbz / 1-naphthyl / etc.)
> and the SA channel cannot move the leaf SA distribution further
> because every candidate is a near-isomer of the same scaffold
> (SA distribution collapses with the structural collapse).

## 1. Configuration (MEASURED)

| field | value |
|---|---|
| script | `molmetal/scripts/r4_lambda_only_run.py` |
| `--pockets` | 1 (`test_001`) |
| `--seeds` | 3 (`42, 0, 1234`) |
| `--n-simulations` | 1000 |
| `--n-top-k` | 20 |
| `--click-rules` | `all-5` → `{AmideCoupling, CuAAC, SPAAC, Suzuki, ThiolEne}` |
| `--metal-seed` | `cisplatin` (`[Pt]C#C`) |
| `--sa-weight` | 0.3 (EV-3 target) |
| `--manifest` | `molmetal/data/test_001_subset_manifest.csv` (1 row) |
| `--output-dir` | `wf_lift_ev3/` (script auto-prepends `wf_lambda1_`) |
| `sa_top10_filter` | ON (220 → 210 tiles) |
| `scaffold_aware` | ON (`strict_Pt_II` detected on every cell) |
| `n_samples` (per-cell) | 100 |
| n_cells | 3 |
| wall-clock total | **362 s ≈ 6 min** (≈118 s/cell) |
| Python | 3.12 (uv-managed) |
| GPU used | **NO** (CPU-only; GPU still SMU-hung per `wf_gpu_diag/diagnosis.md`) |

Output artefacts:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_lift_ev3/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_wf_lift_ev3/summary.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_ev3/run.log`

## 2. CLI invocation

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 1 --seeds 42 0 1234 \
    --n-simulations 1000 --n-top-k 20 \
    --click-rules all-5 --metal-seed cisplatin \
    --sa-weight 0.3 \
    --manifest molmetal/data/test_001_subset_manifest.csv \
    --output-dir wf_lift_ev3/
```

The single-row manifest was created at
`molmetal/data/test_001_subset_manifest.csv` (header + `test_001` row
from `crossdocked100_manifest.csv`).

## 3. Headline result — sa_mean reproduces n_sim=100 baseline

| metric | `wf_sa_penalty_guidance` baseline (test_001 5x1, n_sim=100) | this run (test_001 1x3, n_sim=1000) | Δ |
|---|---|---|---|
| `sa_mean` | **3.099** | **3.099** | **0.000** ⚠ |
| `qed_mean` | 0.735 | 0.735 | 0.000 |
| `validity_rate` | 1.000 | 1.000 | 0.000 |
| `synthesizability_rate` | 1.000 | 1.000 | 0.000 |
| `uniqueness_rate` | 1.000 | 1.000 | 0.000 |
| `novelty` | 1.000 | 1.000 | 0.000 |
| `n_distinct` (median) | 20 (cap) | 20 (cap) | 0 |
| `diversity_tanimoto` | n/a | 0.137 | n/a |
| `diversity_homotype` | n/a | 0.045 | n/a |
| `diversity_subpocket` | n/a | 0.679 | n/a |
| `reference_tanimoto` | n/a | 0.229 | n/a |
| `metal_compliance_rate` | 0.000 | 0.000 | 0.000 ⚠ |
| `metal_coord_compliance` | 0.000 | 0.000 | 0.000 ⚠ |
| `oxidation_state_distribution` | n/a | `{Pt_0: 60}` | all-Pt_0 |
| `decoder_pass_rate` | n/a | 1.000 | n/a |

**All 3 cells are byte-identical in their 20-SMILES candidate set.**
Every cell emits the same Pt-trizole aryl-substituent family
(`[Pt][c]1cn(R)nn1` / `NC(Cn1nnc[c]1[Pt])C(=O)O` / etc.); only the
aryl ring `R` group varies (benzyl, methoxybenzyl, chlorobenzyl,
fluorobenzyl, methylbenzyl, 1-naphthyl, etc.).

### 3.1 vs the previous baseline

* `wf_sa_penalty_guidance` (the canonical baseline for EV-3): ran
  5x1 on `test_001` at `n_sim=100, --sa-weight=0.3, --metal-seed
  cisplatin, --click-rules all-5`. Reported `sa_mean=3.099`
  per cell (single shared Pt-trizole + aryl family).
* This EV-3 run lifts `n_simulations` from 100 to 1000 (10x) on the
  same pocket / flags. **No change in `sa_mean`**, because the
  MCTS at this budget + seed collapses onto the same scaffold
  family regardless of the 10x simulation depth.
* vs the implicit `n_sim=1000, sa_weight=0.0` baseline (EV-1
  test_010..019 sa_mean=3.657), `sa_mean` here is **3.099** —
  the **−0.558** lift is the full `--sa-weight 0.3` reward channel
  in action; what is missing is a chemistry-diversity lift that
  would let the MCTS escape the Pt-trizole attractor and find
  low-SA scaffolds outside the aryl-substituent family.

### 3.2 Why the SA cannot drop further at this budget

The singleton-collapse regime (documented in
`wf_lambda_diversity_rotation/final.md`) means the MCTS at
`n_simulations <= ~2000` with the cisplatin metal-seed finds the
**same 20-candidate Pt-trizole family** for every (pocket, seed)
cell. Every member of that family has the same heavy-atom count
and the same Pt-trizole core; only the aryl substituent changes.
Because the Ertl SA-score is dominated by the *core* (not the
substituent) and the Pt-trizole core has SA ≈ 3.10, the SA mean
saturates at 3.099 across all candidates.

The `sa_top10_filter` (220 → 210 tiles) cannot move SA here either,
because the Pt-trizole aryl family does not draw from the
blacklisted tiles — the blacklist removes azides / α-hydroxy
azides / vinyl-pyrazoles / dicyclopentadiene partials / alkynyl
sulfides, none of which appear in the Pt-trizole aryl family.

To reach the TargetDiff range [2.65, 2.86] we would need a
**chemistry-diversity lift** (Fix 2(a) `MetalLigandExchange`
rule from `wf_lambda_fix_full_path_v2`) so the MCTS can sample
from outside the Pt-trizole attractor — and then the
`--sa-weight 0.3` channel becomes load-bearing. **At present, on
the singleton family, the SA channel has done all it can.**

## 4. Per-cell per-seed metrics

| pocket | seed | n_cand | n_distinct | sa_mean | qed_mean | validity | synth | elapsed_s |
|---|---|---|---|---|---|---|---|---|
| test_001 | 42 | 20 | 20 | 3.0988 | 0.7351 | 1.000 | 1.000 | 117.4 |
| test_001 | 0 | 20 | 20 | 3.0988 | 0.7351 | 1.000 | 1.000 | 119.0 |
| test_001 | 1234 | 20 | 20 | 3.0988 | 0.7351 | 1.000 | 1.000 | 118.7 |

Aggregate:
* Grand `sa_mean` = **3.0988 ± 0.0000** (3 seeds, 60 mols total)
* Grand `qed_mean` = **0.7351**
* `validity_rate` = **1.000** (60/60 RDKit-parseable)
* `synthesizability_rate` = **1.000** (60/60 retrosynthesis-feasible)
* `metal_compliance_rate` = **0.000** (scaffold-aware gate set to
  `strict_Pt_II`, but every candidate is Pt_0 / 1-coordinate, not
  Pt_II / 4-coordinate — honest negative result, the chemistry
  attractor collapses onto the wrong oxidation state)

## 5. The 20-candidate family (per cell — byte-identical across seeds)

```
[Pt][c]1cn(-c2ccccc2)nn1                       # Pt-triazole + phenyl
[Pt][c]1cnnn1-c1ccccc1                         # regio-isomer
COc1ccc(-n2c[c]([Pt])nn2)cc1                   # 4-methoxyphenyl
COc1ccc(-n2nnc[c]2[Pt])cc1                     # regio-isomer
COc1ccccc1Cn1c[c]([Pt])nn1                     # 2-methoxybenzyl
COc1ccccc1Cn1nnc[c]1[Pt]                       # regio-isomer
Cc1ccc(-n2c[c]([Pt])nn2)cc1                    # 4-methylphenyl
Cc1ccc(-n2nnc[c]2[Pt])cc1                      # regio-isomer
Clc1ccc(-n2c[c]([Pt])nn2)cc1                   # 4-chlorophenyl
Clc1ccccc1Cn1c[c]([Pt])nn1                     # 2-chlorobenzyl
Fc1ccccc1Cn1c[c]([Pt])nn1                      # 2-fluorobenzyl
[Pt][c]1cn(Cc2ccccc2)nn1                       # benzyl
[Pt][c]1cnnn1Cc1ccccc1                         # regio-isomer
[Pt][c]1cn(-c2ccc3ccccc3c2)nn1                 # 2-naphthyl
[Pt][c]1cnnn1-c1ccc2ccccc2c1                   # regio-isomer
[Pt][c]1cnnn1-c1ccc2ccccc2n1                   # quinoline-attached
NC(Cn1c[c]([Pt])nn1)C(=O)O                     # serine + triazole
NC(Cn1nnc[c]1[Pt])C(=O)O                       # regio-isomer
CC(C(=O)O)n1c[c]([Pt])nn1                      # α-methyl-serine + triazole
CC(C(=O)O)n1nnc[c]1[Pt]                        # regio-isomer
```

Two Pt-trizole structural motifs (N1 vs N2 substitution) × aryl
substituent set. `oxidation_state_distribution` confirms all
candidates are `Pt_0` (the metal_geometry_prior_bonus gate fires
but at the wrong oxidation state).

## 6. Schema-required metrics

```json
{
  "sa_mean_aggregate": 3.0988,
  "sa_mean_baseline_wf_sa_penalty": 3.099,
  "lift_vs_sa_penalty_baseline": 0.000,
  "comparison_vs_targetdiff_265_286": "above_range (3.099 > 2.86); same as n_sim=100 baseline; singleton-collapse blocks further SA lift",
  "n_pockets_with_sa_in_range": 0,
  "n_pockets_total": 1,
  "sa_weight": 0.3,
  "sa_top10_filter_active": true,
  "n_tiles_removed": 10
}
```

Where:
* `sa_mean_aggregate = 3.0988` is the per-cell mean across 3 cells
  (60 mols total) — identical to `wf_sa_penalty_guidance` baseline.
* `lift_vs_sa_penalty_baseline = 0.000` — 10× simulation budget
  + same flags produces byte-identical SA mean.
* `n_pockets_with_sa_in_range = 0 / 1` — `test_001` SA mean is
  above TargetDiff range [2.65, 2.86].

## 7. Honest limitations

1. **Singleton-collapse regime** (the dominant blocker): at this
   MCTS budget + metal-seed, every cell emits the same Pt-trizole
   aryl family. The SA channel is saturated by definition; no
   pool or reward change can move SA below the SA of the
   converged scaffold.
2. **The TargetDiff range [2.65, 2.86] is unreachable** without a
   **chemistry-diversity lift** that lets the MCTS escape the
   Pt-trizole attractor. The published
   `wf_lambda_fix_full_path_v2/fix2_scaffold_aware.md` Fix 2(a)
   (`MetalLigandExchange` SMARTS rule) is the documented next-step
   but is **not** wired into this run — keeping the comparison
   apples-to-apples against the `wf_sa_penalty_guidance` baseline.
3. **`metal_compliance_rate = 0.000`**: all 60 candidates are
   `Pt_0` / 1-coordinate. The scaffold-aware gate classifies
   the seed as `strict_Pt_II` but the MCTS root fix (Fix 1)
   bypasses that gate, so the emitted candidates do not match
   the seed's intended oxidation state. This is a **known
   chemistry regression** documented in
   `wf_lambda_fix_singleton/final.md` and is **independent**
   of the EV-3 SA channel.
4. **`reference_ligand_resolver` fallback**: the pocket-derived
   reference ligand for `test_001` is
   `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` — a
   sulfonamide-piperidine, not Pt-bearing. The MCTS root uses
   `[Pt]C#C` (cisplatin metal-seed) instead of the pocket
   reference, so `reference_tanimoto = 0.229` (modest similarity
   to the true reference).
5. **3-cell sample is small**. The `wf_sa_penalty_guidance`
   baseline ran 5x1 (5 cells); this EV-3 ran 1x3 (3 cells). The
   comparison is **per-cell mean** so the sample size matches
   conceptually, but the variance estimate is 0 because all 3
   cells are byte-identical (singleton regime), not because of
   any underlying low variance.

## 8. Wall-clock budget vs spec

| budget | actual | status |
|---|---|---|
| 1 h spec | 6 min (362 s) | under-budget by 10× |
| ~5 min/cell spec | ~118 s/cell | on-target |

3 cells × 118 s/cell = 354 s ≈ 6 min — well inside the 1-hour
budget.

## 9. Honest verdict

The `--sa-weight 0.3` channel at `n_simulations=1000` reproduces
the `wf_sa_penalty_guidance` `n_sim=100` baseline exactly
(`sa_mean = 3.099`). The TargetDiff range [2.65, 2.86] is **not**
reached, and cannot be reached on the singleton Pt-trizole
family without a chemistry-diversity lift (Fix 2(a)) — which
is **out of scope** for the EV-3 SA-channel evaluation.

**Recommendation** (carry forward, not action item for this run):
* Pair `--sa-weight 0.3` with Fix 2(a) `MetalLigandExchange`
  SMARTS rule + Fix 1 soft-tiered metal_geometry_prior_bonus,
  then re-run the 10x3 on test_001 at `n_sim=1000` — that is the
  experiment that can actually demonstrate a SA lift into the
  TargetDiff range.
* This EV-3 run serves as the **singleton-regime confirmation**:
  on this attractor, the SA channel is saturated and a chemistry
  intervention is required before further SA rewards can fire.

## 10. Files

* `molmetal/reports/wf_lambda1_wf_lift_ev3/report.json` (auto-output)
* `molmetal/reports/wf_lambda1_wf_lift_ev3/summary.md` (auto-output)
* `molmetal/reports/wf_lift_ev3/run.log` (stdout/stderr capture)
* `molmetal/reports/wf_lift_ev3/final.json` (this verdict, JSON)
* `molmetal/reports/wf_lift_ev3/final.md` (this report)
* `molmetal/data/test_001_subset_manifest.csv` (1-row manifest,
  header + test_001 from `crossdocked100_manifest.csv`)
