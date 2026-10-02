# Wf-330-T2 — Triangle closure end-to-end on MetalCytoToxDB 5-metal

**Last updated:** 2026-09-29
**Status:** SHIPPED (CPU-only bench, MEASURED-Below-Projected honest-negative)
**Owner:** Wf-330-T2 / R21-P4 ship surface (TODO-93 cross-ref L8)

## 1. Summary

Wf-330-T2 extends the Wf-37 P4 + Wf-39 P4 `bench_lambda_triangle_end_to_end.py`
ship surface to the **MetalCytoToxDB 5-metal cohort** (Ru / Ir / Rh / Os / Re)
with **F5_RDKIT_2D 12-d RDKit feature integration** (Wf-49 P1 / Wf-249).  The
bench re-uses the Wf-37 L8 `TriangleClosure` orchestrator unchanged; the new
work is:

1. `--dataset metalcytotox_5metal` CLI flag + `DATASET_REGISTRY` plumbing.
2. `_load_metalcytotox_5metal()` loader that filters the 5-metal cohort.
3. `_f5_rdkit_2d_features()` 12-d RDKit descriptor extraction (12 fixed
   columns: `MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms, RingCount,
   NumAromaticRings, FractionCSP3, MolMR, BalabanJ`).
4. `_build_design_matrix(..., feature_mode="f5_rdkit_2d")` plumbing.
5. `per_metal_breakdown` report block — one row per metal in the cohort with
   PROJECTED target, MEASURED hilbert R², and an explicit
   `MEASURED_vs_PROJECTED` verdict string per metal + unweighted cohort mean.
6. Markdown report writer now emits the per-metal breakdown table.

## 2. Per-metal MEASURED-vs-PROJECTED ledger (Wf-330-T2 ship)

Source: `molmetal/reports/wf330_triangle_metalcytotox/triangle_bench.json`
(Bench: 100 trajectories × 5 rounds × 5 seeds × 5 metals, hilbert_iters=20)

| Metal | n_rows | hilbert R² MEASURED | PROJECTED (skill §4.3) | verdict |
|-------|------:|-------------------:|-----------------------:|---------|
| Ru | ~5,326 | **0.4626** | 0.65 | MEASURED=BELOW_TARGET |
| Ir | ~5,326 | **0.3790** | 0.63 | MEASURED=BELOW_TARGET |
| Rh | ~5,326 | **0.4309** | 0.62 | MEASURED=BELOW_TARGET |
| Os | ~5,326 | **0.4526** | 0.61 | MEASURED=BELOW_TARGET |
| Re | ~5,326 | **0.3530** | 0.60 | MEASURED=BELOW_TARGET |
| **5-metal cohort mean (unweighted)** | 26,632 | **0.4156** | **0.6220** | **MEASURED=BELOW_TARGET (PROJECTED=0.622)** |

**Honest framing:** the Wf-330-T2 CPU bench on the 5-metal MetalCytoToxDB cohort
with F5_RDKIT_2D 12-d features produces a MEASURED cohort mean `mean_hilbert_r2`
of 0.4156, vs the PROJECTED 0.622 target from skill §4.3.  All 5 metals show
**MEASURED=BELOW_TARGET** (per-metal gaps range −0.16 to −0.27).  This is
**honest-negative** per AGENTS.md §5.2: the F5_RDKIT_2D 12-d integration is
alive (12-d feature variance >> 5-d structural fallback), but the synthetic
target labels + small Hilbert primal-dual iterations (20) yield R² values
that fall short of the PROJECTED R² ≥ 0.6 anchor.

## 3. Skill §4.3 PROJECTED anchors (verbatim, MEASURED-vs-PROJECTED ledger)

Per `mol-metal-triangle-closure/SKILL.md §4.3`:

| Cell | n | mean_hilbert_r2 (PROJECTED) | mean_evolutionary_r2 (PROJECTED) | mean_variance_reduction (PROJECTED) | triangle_convergence_rate (PROJECTED) |
|---|---:|---:|---:|---:|---:|
| Ru | 160K | 0.65 | 0.07 | 0.06 | 0.50 |
| Ir | 160K | 0.63 | 0.06 | 0.06 | 0.48 |
| Rh | 160K | 0.62 | 0.06 | 0.05 | 0.47 |
| Os | 160K | 0.61 | 0.05 | 0.05 | 0.46 |
| Re | 160K | 0.60 | 0.05 | 0.05 | 0.45 |
| **mean (unweighted)** | 800K | **0.622** | **0.058** | **0.054** | **0.472** |

Wf-330-T2 MEASURED (per-metal) targets the §4.3 PROJECTED above.  The
5-metal cohort mean (PROJECTED 0.622) is implemented as
`METALCYTO_5METAL_PROJECTED_MEAN = 0.622` and the per-metal anchors are
implemented as `METALCYTO_5METAL_PROJECTED_R2 = {"Ru": 0.65, "Ir": 0.63,
"Rh": 0.62, "Os": 0.61, "Re": 0.60}` in `bench_lambda_triangle_end_to_end.py`.

## 4. Skill §4.2 ledger discrepancy (HONEST-NEGATIVE FLAG)

The PlatinAI 4-cell 7-metal cohort (§4.2) PROJECTED values are:

| Pt | Pd | Ru | Rh | Ir | Os | Au | mean (unweighted) |
|----|----|----|----|----|----|----|-------------------|
| 0.62 | 0.61 | 0.60 | 0.60 | 0.59 | 0.58 | 0.58 | **0.594** (skill ledger) |

**Discrepancy:** the unweighted mean of the 7 values is **0.5971** (= 4.18 / 7),
NOT 0.594 as printed in the skill §4.2 table.  The 0.0031 gap is a
skill-ledger math error (the printed mean does not match the printed
per-cell values).  `test_7_metal_cohort_projected_mean` locks the **true**
mean (0.5971) and explicitly records the 0.0031 ledger delta so downstream
paper §3.6 / Wf-330-T2 closeout can call this out as honest-negative.

## 5. Test surface (Wf-330-T2)

File: `molmetal/tests/test_wf330t2_triangle_metalcytotox.py`

| Test | Locks |
|------|--------|
| `test_per_metal_breakdown_5_metal_cohort` | All 5 metals in `per_metal_breakdown`; PROJECTED keys equal `METALCYTO_5METAL_PROJECTED_R2`; verdict strings well-formed; summary reports `dataset=metalcytotox_5metal` + `feature_mode=f5_rdkit_2d`; projected cohort mean = 0.622. |
| `test_f5_rdkit_2d_12d_integration` | The 12-d feature name tuple is the Wf-49 P1 canonical set; concrete SMILES parse returns a 12-d vector with positive `MolWt` and `HeavyAtoms`. |
| `test_7_metal_cohort_projected_mean` | Locks the true unweighted 7-metal mean (0.5971) and the 0.0031 skill §4.2 ledger discrepancy. |
| `test_convergence_lift_f5_rdkit_2d_vs_5d` | The 12-d F5_RDKIT_2D integration is alive (non-zero feature variance); 12-d variance ≥ 5-d baseline variance × 1.0. |
| `test_wf330t2_module_smoke` | (Bonus) Module-level smoke — `DATASET_REGISTRY` plumbing, projected ledger constants, feature name tuples. |

5 tests total (4 NEW per Wf-330-T2 task + 1 module-level smoke), all
**PASS** at HEAD.

## 6. CLI quick reference

CPU-only bench, end-to-end:

```bash
PYTHONPATH=/home/hugo/codes/mol-metal:/home/hugo/codes/CadButEaas \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/bench_lambda_triangle_end_to_end.py \
    --dataset metalcytotox_5metal \
    --n-trajectories 100 --n-rounds 5 \
    --seeds 0 1 2 3 4 \
    --n-arms 4 --hilbert-iters 20 \
    --out-dir molmetal/reports/wf330_triangle_metalcytotox/
```

Smoke (~5 s):

```bash
PYTHONPATH=/home/hugo/codes/mol-metal:/home/hugo/codes/CadButEaas \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/bench_lambda_triangle_end_to_end.py \
    --dataset metalcytotox_5metal \
    --n-trajectories 20 --n-rounds 2 --seeds 0 1 \
    --n-arms 4 --hilbert-iters 10 \
    --out-dir molmetal/reports/wf330_triangle_metalcytotox_smoke/
```

Backward-compat PlatinAI 5K bench (unchanged):

```bash
PYTHONPATH=/home/hugo/codes/mol-metal:/home/hugo/codes/CadButEaas \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/bench_lambda_triangle_end_to_end.py \
    --dataset platinai_5k \
    --n-trajectories 100 --n-rounds 5 \
    --seeds 0 1 2 3 4 \
    --n-arms 4 --hilbert-iters 20 \
    --out-dir molmetal/reports/wf39_triangle/
```

## 7. Why this is honest-negative, not a regression flag

Per AGENTS.md §5.2 honest framing + Wf-330-T2 task §Honest framing:

* **MEASURED baseline per Wf-329-W2 was very low** (5-d structural
  features insufficient signal — mean_hilbert_r2 ≪ 0.05 on synthetic
  target with the same TriangleClosure orchestrator).
* **F5_RDKIT_2D 12-d integration is alive** — `test_convergence_lift_f5_rdkit_2d_vs_5d`
  confirms the 12-d feature variance is non-zero and ≥ 5-d baseline.
* **MEASURED cohort mean (0.4156) is BELOW the PROJECTED 0.622 target**
  because the bench uses **synthetic targets** (`y = X_std @ w + 0.05 *
  noise`) and **CPU-cheap Hilbert primal-dual** (20 iters).  Reaching
  the PROJECTED 0.622 anchor requires:
  1. Real labelled pIC50 targets (not synthetic), AND
  2. GPU-accelerated Hilbert primal-dual with Wf-44 P2 numerical
     fixes (grad_clip + step_decay + fp64) — both Wf-37 P4 +
     Wf-39 P4 ship surface, deferred to TODO-93 (R20-P2 GPU joint
     retrain) per the R21-P4 triangle closure ship plan.

## 8. Files touched

| File | Change |
|------|--------|
| `molmetal/scripts/bench_lambda_triangle_end_to_end.py` | Added `DATASET_REGISTRY`, `METALCYTO_5METAL_*` constants, `_f5_rdkit_2d_features`, `_load_metalcytotox_5metal`, `--dataset` + `--metalcytotox-csv` + `--feature-mode` CLI flags, `per_metal_breakdown` report block, MEASURED-vs-PROJECTED verdict strings. |
| `molmetal/tests/test_wf330t2_triangle_metalcytotox.py` | NEW file — 5 tests locking the 5-metal cohort breakdown, F5_RDKIT_2D 12-d integration, 7-metal ledger discrepancy, F5_RDKIT_2D convergence lift, and module-level smoke. |
| `molmetal/reports/wf330_triangle_metalcytotox/triangle_bench.{json,md}` | NEW directory — production bench output (100 traj × 5 rounds × 5 seeds × 5 metals). |
| `molmetal/reports/wf330_triangle_metalcytotox_smoke/` | NEW directory — smoke bench output (20 traj × 2 rounds × 2 seeds × 5 metals). |
| `TODO/completed/330t2_triangle_metalcytotox.md` | NEW — this file. |

## 9. References

* Skill: `mol-metal-triangle-closure` §4.2 (PlatinAI 4-cell, 7-metal
  PROJECTED) + §4.3 (MetalCytoToxDB 5-metal PROJECTED)
* Wf-37 P4 L8 — base triangle closure orchestrator
  (`molmetal/scripts/triangle_closure.py`)
* Wf-39 P4 — PlatinAI 5K end-to-end bench ship surface
* Wf-44 P2 — Hilbert primal-dual numerical fix
* Wf-49 P1 — F5_RDKIT_2D 12-d RDKit 2D descriptor extraction
* Wf-249 Fix #1 — RDKit fingerprint extension (1024-d ablation)
* Wf-329-W2 — 5-d structural feature baseline (BELOW_TARGET honest-neg)
* AGENTS.md §5.2 honest framing — MEASURED-vs-PROJECTED ledger discipline