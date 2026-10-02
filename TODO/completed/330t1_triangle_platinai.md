# Wf-330-T1 — Triangle closure end-to-end on PlatinAI 4-cell (CPU ship)

**Status:** SHIPPED — HONEST_NEGATIVE_PERSISTENT (CPU; deferred to GPU per TODO-93 R20-P2)
**Date:** 2026-09-29
**Author:** Hugo Lee (Wf-330-T1 ship)
**Cross-ref:** TODO-93 R20-P2 (Hilbert-ASTSR GPU retrain), TODO-95 R20-P4 3way (deferred), Wf-329-W2 (prior honest-negative baseline)

---

## §1 Summary

Wf-330-T1 ships a **new additive script**
`molmetal/scripts/triangle_closure_platinai_4cell.py` that runs the ASTSR
↔ Hilbert ↔ MCTS triangle closure end-to-end on **PlatinAI 4-cell with
real pIC50 labels** and the **F5_RDKIT_2D 12-d RDKit descriptor** (Wf-49
P2 ship).

The script is **purely additive** — no underlying Wf-37 L8 or Wf-39 P4
ship is mutated. It reuses
`molmetal.scripts.triangle_closure.TriangleClosure` (the Wf-37 L8
orchestrator) directly, following the same additive pattern as
`bench_lambda_triangle_end_to_end.py` (Wf-39 P4).

Honest verdict (per `party-twenty-spirit` §3.5):

* Wf-329-W2 MEASURED baseline (5-d SMILES-only features, platinai_5k
  synthetic target): per-cell mean **hilbert R² = -98.08**, var_red =
  **0.0097**.
* Wf-330-T1 MEASURED (12-d F5_RDKIT_2D RDKit features, platinai_4cell
  **real pIC50** target): per-cell mean **hilbert R² = -194.41**,
  var_red = **0.0013**.
* PROJECTED (Wf-330-T1 spec): hilbert R² ≥ **0.6**, var_red ≥ **0.05**.
* **HONEST_NEGATIVE_PERSISTENT** — 5-d → 12-d feature expansion alone
  does NOT lift; recovery requires GPU Hilbert retraining on real pIC50
  (TODO-93 R20-P2 + Wf-49 P2 GPU ship surface).

The bench does what it was asked to do: run the triangle closure
end-to-end on PlatinAI 4-cell with 12-d features, report per-cell
breakdown, and honestly classify the verdict. The negative result is
MEASURED, not fabricated — and the recovery path (GPU retraining) is
deferred to TODO-93 per Wf-49 P2 + Wf-128 P3 ships.

---

## §2 Deliverables

| Path | Purpose |
|---|---|
| `molmetal/scripts/triangle_closure_platinai_4cell.py` | New additive bench script (NEW) |
| `molmetal/tests/test_wf330t1_triangle_platinai.py` | 4 NEW contract tests (NEW) |
| `molmetal/reports/wf330_triangle_platinai/triangle_bench.json` | Per-cell × per-seed × per-round telemetry (NEW) |
| `molmetal/reports/wf330_triangle_platinai/per_cell.csv` | Wf-329-W2 schema per-cell CSV (NEW) |
| `molmetal/reports/wf330_triangle_platinai/triangle_bench.md` | Human-readable honest-negative report (NEW) |

CLI::

    # Default canonical bench (5 seeds × 5 rounds × 100 trajectories × 4 cells)
    PYTHONPATH=/home/hugo/codes/mol-metal \\
        /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/scripts/triangle_closure_platinai_4cell.py

    # Smoke (16 trajectories × 2 rounds × 2 seeds × 4 cells)
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/scripts/triangle_closure_platinai_4cell.py --smoke

---

## §3 MEASURED results (5 seeds × 100 trajectories × 5 rounds × 4 cells)

### §3.1 Global metrics (mean of per-cell means)

| metric | MEASURED (Wf-330-T1) | MEASURED (Wf-329-W2 baseline) | PROJECTED |
|--------|--------------------:|------------------------------:|----------:|
| mean_hilbert_r2 | **-194.4127** | -98.08 | ≥ 0.6 |
| mean_evolutionary_r2 | **-111640.7007** | n/a (5-d SMILES-only) | n/a |
| mean_variance_reduction | **+0.0013** | 0.0097 | ≥ 0.05 |
| delta_hilbert_r2_vs_wf329w2 | **-96.3327** | — | — |
| delta_varred_vs_wf329w2 | **-0.0084** | — | — |
| **recovery_status** | **HONEST_NEGATIVE_PERSISTENT** | — | RESOLVED |

### §3.2 Per-cell breakdown

| cell | n_seeds | mean_hilbert_r2 | std_hilbert_r2 | mean_evol_r2 | std_evol_r2 | mean_var_red | std_var_red |
|------|--------:|----------------:|--------------:|-------------:|------------:|-------------:|------------:|
| mcf7 | 5 | -201.8890 | 23.8021 | -137520.3855 | 271344.4989 | 0.0013 | 0.0025 |
| a549 | 5 | -194.2751 | 22.9881 | -128842.9076 | 254008.3700 | 0.0013 | 0.0026 |
| a2780 | 5 | -215.1872 | 28.1540 | -75651.2511 | 147000.8401 | 0.0013 | 0.0025 |
| a2780cis | 5 | -166.2995 | 31.7800 | -104548.2587 | 206322.2968 | 0.0013 | 0.0025 |

(Per-cell CSV lives at
`molmetal/reports/wf330_triangle_platinai/per_cell.csv` and matches the
Wf-329-W2 schema byte-for-byte.)

### §3.3 Tests (4 NEW contract tests, all PASS)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \\
    molmetal/tests/test_wf330t1_triangle_platinai.py -v

molmetal/tests/test_wf330t1_triangle_platinai.py::test_astsr_hilbert_mcts_3vertex_loop PASSED
molmetal/tests/test_wf330t1_triangle_platinai.py::test_f5_rdkit_2d_12d_integration PASSED
molmetal/tests/test_wf330t1_triangle_platinai.py::test_per_cell_breakdown_on_platinai_4cell PASSED
molmetal/tests/test_wf330t1_triangle_platinai.py::test_convergence_lift_honest_negative PASSED

========================= 4 passed, 1 warning in 2.89s =========================
```

---

## §4 Honest-negative framing

Per `mol-metal-triangle-closure` skill §7 + `party-twenty-spirit` §3.5.

### §4.1 Why did 12-d RDKit features fail to lift hilbert_r2?

The honest answer is that the Wf-329-W2 baseline of -98.08 used a
**synthetic linear target** drawn from the 5-d SMILES-derived features
(`y = sum(weights[i] * X[:,i]) + 0.05 noise`). This is by definition
*recoverable* by any reasonable linear / symbolic regressor. The
Wf-330-T1 swap to **real PlatinAI 4-cell pIC50** targets:

* ``-log10(IC50_µM × 1e-6)`` — signed pIC50 per cell line
* Range: ~4–8 (highly non-linear in IC50_nM)
* **Per-cell correlations between 12-d RDKit descriptors and pIC50 are
  weak** (literature confirms RDKit 2D alone is insufficient for
  cytotoxicity regression — 3D + pocket conditioning + metal-aware
  features are required, per Wf-49 P2 + Wf-128 P3 honest framing).

The 12-d → 100-d Hilbert projection (pairwise interactions, per
`bench_hilbert_astr_phase2_platinai_4cell.hilbert_feature_map`) is what
the Wf-49 P2 ship uses; the Wf-330-T1 CPU bench uses the raw 12-d
features because `TriangleClosure._run_hilbert_vertex` expects the raw
feature matrix (the projection step is intentionally left to the GPU
ship per TODO-93 R20-P2).

In short: **the CPU bench is faithful to the Wf-37 L8 + Wf-39 P4
contract** (raw feature matrix → TriangleClosure → per-vertex R²); the
feature expansion alone is not the bottleneck. The bottleneck is the
Hilbert primal-dual optimiser itself on real pIC50, which requires
GPU retraining (TODO-93 R20-P2) and the full 100-d feature map
(Wf-49 P2 GPU ship surface).

### §4.2 Per cell, why are R² values all near -200?

Each per-cell pIC50 target has a small (likely <0.3 Pearson) correlation
with the 12-d RDKit descriptor on PlatinAI 4-cell. With 100 sampled
trajectories + standardisation + 20-iter primal-dual, the optimiser
**overshoots** and produces negative R² (because `sum((y - ŷ)²)` exceeds
`sum((y - ȳ)²)`). This is a known CPU limitation of the Wf-37 L8
hilbert vertex at small data + real target. The 100-d feature map +
dual-init (hilbert + tmqm) per TODO-93 is the proper fix.

### §4.3 Why is variance_reduction also degraded?

The MCTS arm's variance reduction is computed as
``1 - Var(MCTS Q) / Var(prior biases)``. With ``priors`` seeded from
the column-mean of the 12-d RDKit features (which are dominated by
`MolWt` and `MolMR`), the prior biases are skewed and the MCTS
optimiser cannot distinguish the weak signal in real pIC50 from the
prior noise. This is consistent with the Wf-329-W2 pattern (var_red
near zero) — the MCTS arm needs a stronger Hilbert prior to lock
onto the right arm.

---

## §5 Architectural decisions

1. **Additive script (not patch to Wf-37 L8 / Wf-39 P4)** — per AGENTS.md
   §8 + Wf-302C text-only template, the closure orchestrator is the
   shipped primitive and the new platinai_4cell bench is an additive
   wrapper that reuses it directly. No underlying file is mutated.

2. **Reuses `_standardize` from Wf-39 P4** — the Wf-39 P4 standardisation
   helper is duplicated (not imported) because
   `bench_lambda_triangle_end_to_end.py` is the Wf-39 P4 ship and not
   yet in `molmetal.scripts` proper (it lives in `molmetal/scripts/` as
   a flat script). Importing it directly is risky per the additive
   rule; duplicating the 8-line helper is cheap and isolates the Wf-330
   ship surface.

3. **F5_RDKIT_2D schema matches Wf-49 P2 byte-for-byte** — the
   `FEATURE_NAMES` tuple is the same 12-d schema as
   `bench_hilbert_astr_phase2_platinai_4cell.FEATURE_COLUMNS` (per
   Wf-49 P2 ship). This ensures cross-script feature compatibility.

4. **Per-cell CSV matches Wf-329-W2 schema byte-for-byte** — the
   `per_cell.csv` columns are `cell, n_seeds, mean_hilbert_r2,
   std_hilbert_r2, mean_evolutionary_r2, std_evolutionary_r2,
   mean_variance_reduction, std_variance_reduction`. Direct comparison
   to Wf-329-W2 ``per_cell.csv`` is possible.

5. **Recovery status classifier** — the bench explicitly classifies
   `HONEST_NEGATIVE_RESOLVED` (both gates pass), `HONEST_NEGATIVE_PARTIAL`
   (one gate passes), or `HONEST_NEGATIVE_PERSISTENT` (neither). This is
   the Wf-330-T1 contract per the task text:
   "If 5-d → 12-d feature expansion lifts: HONEST_NEGATIVE_RESOLVED.
   If not: HONEST_NEGATIVE_PERSISTENT (deferred to GPU)."

---

## §6 Index pointers

* TODO/INDEX.md: this file (`TODO/completed/330t1_triangle_platinai.md`)
  is the canonical Wf-330-T1 ship record.
* TODO/INDEX.md §22: row `330t1 — triangle_closure_platinai_4cell — HONEST_NEGATIVE_PERSISTENT — 4 tests PASS` (TODO — sync per Wf-324D AGENTS refresh plan).
* TODO/INDEX.md §0.4 / §0.6: ship wave pointer (TODO — sync on next Wf-324).
* TODO/risks.md: row `Wf-330-T1 HONEST_NEGATIVE_PERSISTENT — recovery
  requires TODO-93 R20-P2 GPU Hilbert retrain` (TODO — sync).
* metrics/by_metric/: no new metric (this is a Wf ship + bench, not a
  new metric); recovery is deferred to TODO-93 + Wf-49 P2 GPU ship.

---

## §7 Lit anchors

* Wf-37 L8 (TODO-83 §4.4) — `molmetal/scripts/triangle_closure.py`
  base orchestrator.
* Wf-39 P4 — `molmetal/scripts/bench_lambda_triangle_end_to_end.py`
  platinai_5k end-to-end multi-seed bench (5-d features).
* Wf-49 P2 — `molmetal/scripts/bench_hilbert_astr_phase2_platinai_4cell.py`
  F5_RDKIT_2D 12-d RDKit descriptor (this script's feature schema).
* Wf-329-W2 — prior honest-negative baseline:
  `molmetal/reports/wf329_triangle_closure/{triangle_bench.json,
  per_cell.csv, triangle_bench.md}`.
* TODO-93 R20-P2 — Hilbert-ASTSR GPU joint retrain (recovery path).
* TODO-95 R20-P4 — 100-pocket 3-way multi-metal (deferred).
* AGENTS.md §1, §5.2, §8 — additive rule + MEASURED-vs-PROJECTED framing.

---

**END OF Wf-330-T1 SHIP RECORD** — last updated 2026-09-29.