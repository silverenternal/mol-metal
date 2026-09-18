# DesignLoop Phase-0 Report (W1 / TODO/11)

**Author**: molmetal orchestration pipeline
**Date**: 2026-09-11
**Pipeline**: `DesignLoop_v1` (generate → pre-filter → dock → predict → score)
**Status**: Phase-0 OPEN-LOOP — single round, no real docking, no refinement

---

## 1. What was built

Five files added under `molmetal/` and `molmetal/scripts/`:

| Path | Role |
|------|------|
| `molmetal/orchestration/design_loop.py` | `DesignLoop` orchestrator + `LoopResult` + `CandidateProvenance` dataclasses |
| `molmetal/scripts/run_design_loop.py` | CLI driver (mock generator + DiffDock STUB + RDKit + WeightedSumScorer) |
| `molmetal/tests/test_design_loop.py` | 7 unit tests (smoke / top-K / provenance / refinement) |
| `molmetal/reports/design_loop_phase0.md` | This report |
| `molmetal/reports/design_loop_phase0.json` | Full LoopResult dump (top-5 candidates + metadata) |

### 1.1 Adapter stack used

| Port | Adapter | State |
|------|---------|-------|
| `MoleculeGenerator` | `MockGenerator` wrapped by `SMILESInjectingGenerator` | random 8-atom mols + heuristic SMILES |
| `DockingEngine`    | `DiffDockAdapter(checkpoint_path=None)`            | STUB mode — random SE(3) + realistic Vina range |
| `PropertyPredictor`| `RDKitPropertyPredictor`                          | real QED / logP / MW / TPSA / SA-score |
| `ScoringFunction`  | `WeightedSumScorer`                               | multi-objective weighted sum (qed 0.5, sa 0.3, pIC50 0.8, vina -1.0) |

The `SMILESInjectingGenerator` wrapper is a Phase-0 convenience that
attaches canonical SMILES to the mock generator's output so RDKit can
compute real 2D descriptors.  The real Lipman flow-matching adapter will
emit SMILES natively and this wrapper goes away in Phase 1.

### 1.2 Pipeline contract

```python
result = loop.run(
    pocket,
    n_generate=100,
    n_dock=100,            # Phase-0: dock everyone to populate Vina
    n_top=5,
    refinement_iters=0,    # Phase-1 will set this to 2-3
    seed=42,
)
# result.candidates — list[CandidateProvenance] (rank, smiles, qed, mw,
#   sa, vina, pose_confidence, pose_coords, score_breakdown,
#   generator_metadata, docker_metadata, predictor_metadata, raw_values)
```

Per-iteration summary is recorded in `result.per_iteration_summary`
(n_generated, n_docked, n_dock_failures, best_score, mean_score,
elapsed_s).  This is what the Phase-1 convergence logic will key off.

---

## 2. Phase-0 run

Command:

```bash
source .venv/bin/activate
python -m molmetal.scripts.run_design_loop \
    --metal Ru --pdb-id 1HOV \
    --n-generate 100 --n-dock 100 --n-top 5
```

(Note: `--n-dock` is bumped to 100 for the demo so the Vina column is
always populated; in production this is the cost-control knob.)

### 2.1 Top-5 candidates

| rank | score   | qed   | mw    | sa   | vina     | smiles          |
|------|---------|-------|-------|------|----------|-----------------|
| 1    | 0.3854  | 0.565 | 118.18| 1.000| -11.241  | CCC(O)CCCO      |
| 2    | 0.3427  | 0.405 | 116.21| 1.000| -11.764  | CCCNCNCC        |
| 3    | 0.3195  | 0.366 | 124.10| 1.000| -11.709  | ON(O)CCN(O)O    |
| 4    | 0.2866  | 0.267 | 120.15| 1.000| -11.946  | CCNNC(O)CO      |
| 5    | 0.2592  | 0.219 | 122.13| 1.000| -11.893  | NCN(O)NCNO      |

### 2.2 Per-iteration summary

```
iter=0 gen=100 dock=100 best=0.3854 mean=-0.0175 (0.06s)
```

### 2.3 Wall-clock

* **Loop only**: 0.06 s (CPU, mock adapters)
* **End-to-end including venv import + JSON dump**: ~1.5 s

A realistic Phase-1 run with the Lipman flow-matching adapter on a
single ROCm GPU is expected to take **30-90 s** for 100 molecules
(generation dominates — docking with DiffDock STUB is <1 s/mol,
RDKit descriptors are <1 ms/mol).

---

## 3. Test coverage

`pytest molmetal/tests/test_design_loop.py -v`:

```
TestDesignLoopSmoke::test_loop_smoke                       PASSED
TestDesignLoopSmoke::test_loop_wall_clock_recorded         PASSED
TestDesignLoopSmoke::test_loop_metadata_present            PASSED
TestDesignLoopTopKOrdering::test_loop_topk_monotone        PASSED
TestDesignLoopTopKOrdering::test_loop_topk_respects_smaller_n_top  PASSED
TestDesignLoopProvenance::test_loop_provenance_present     PASSED
TestDesignLoopRefinement::test_loop_refinement_iters       PASSED

7 passed in 1.05s
```

Coverage:

* **Smoke** — `LoopResult` shape, wall-clock recorded, adapter metadata snapshot
* **Top-K** — strictly non-increasing scores, rank 1..n_top, capped at n_top
* **Provenance** — generator/docker/predictor metadata + raw values + smiles populated
* **Refinement** — `refinement_iters=1` triggers a second iteration, the seed-SMILES branch executes, and the per-iter record count differs from the baseline

---

## 4. Observations & known limitations

### 4.1 What works end-to-end

1. **Adapter composition** — `DesignLoop(generator, docker, predictor, scorer)` wires four duck-typed ports with no concrete-class coupling.
2. **Per-candidate failure isolation** — adapter exceptions are caught per (mol, cplx) so a single bad sample does not abort a multi-hour round.
3. **Full provenance** — every top candidate carries the generator + docker + predictor metadata snapshots plus raw values for every scorer term, so results are reproducible from `molmetal/reports/design_loop_phase0.json`.
4. **Top-K + refinement** — final ranking is over all iterations, so refinement cannot regress previously-discovered good candidates.

### 4.2 Phase-0 limitations

* **Mock generator** — coordinates are random, atom types are in {C,N,O}, the heuristic SMILES reconstruction is brittle (RDKit rejects some N/O valence patterns and the candidate then ranks poorly).  Real chemistry arrives in Phase 1 with the Lipman flow-matching adapter.
* **STUB docking** — DiffDockAdapter returns random SE(3) poses with realistic Vina ranges but no pocket awareness.  Phase 2 will load the real checkpoint from `molmetal/references/DiffDock`.
* **No refinement in the demo run** — `--refinement-iters 0`.  The pipeline branch is exercised by `test_loop_refinement_iters` but the demo run does not demonstrate it visually.
* **No real pocket** — `pdb_id=1HOV` is recorded in metadata but the loop uses a synthetic 64-atom pocket with `center = mean(coords)` because no real PDB parse happens here.  Phase 1 will wire `molmetal.domain.io.pdb_loader._load_pocket` for the MMP2/MMP9 CrossDocked subsets.
* **Synthetic SA-score = 1.0** — all five candidates have SA=1.0 because the mock generator's atom pool is tiny; real SA scores will spread once we plug in Lipman generations.

### 4.3 Why the demo run ranks `CCC(O)CCCO` highest

The WeightedSumScorer normalises each term to [0, 1] and weights:

| term      | weight | raw         | normalised | contribution |
|-----------|--------|-------------|-----------|--------------|
| qed       | +0.5   | 0.565       | 0.565     | +0.283       |
| sa_score  | +0.3   | 1.0 (synth) | 1.000     | +0.300       |
| binding_pic50 | 0.0 | None        | —         | 0            |
| vina_score| -1.0   | -11.241     | 0.197     | -0.197       |
| **total** |        |             |           | **+0.386**   |

qed and sa dominate because vina_normalised ∈ [0, 1] stays high (the
STUB vina range is U(-12, -4) which maps to [0.14, 0.71] — close to
"good" already).  In Phase 1 the Real Vina range (-14, 0) will be used
and this will spread the vina contribution.

---

## 5. Next steps — Phase 1 (closed-loop refine)

Per `TODO/11_design_loop/closed_loop_design.md`:

1. **Wire the real generator** — load `molmetal/checkpoints/fm_pocket.pt`
   into a `LipmanFlowMatchingAdapter` and replace `MockGenerator`.  The
   `SMILESInjectingGenerator` wrapper becomes obsolete because Lipman
   emits SMILES natively.
2. **Wire the real pocket** — load `molmetal/data/crossdocked/MMP2`
   PDB files via `molmetal.domain.io.pdb_loader._load_pocket`.
3. **Wire real SA-score** — `RDKitPropertyPredictor` already uses
   `rdkit.Contrib.SA_Score.sascorer` when available; the SA=1.0 in the
   demo is a mock-generator artefact.
4. **Run with `--refinement-iters 2-3`** — the seed-SMILES branch will
   pass top-10 SMILES from iter N to iter N+1 via
   `GenerationConfig.conditioning["seed_smiles"]`.  The
   `WeightedSumScorer` will keep the global best across iterations.
5. **Metal conditioning** — record the `--metal Ru` (or Pt/Ir/Os) flag
   in `LoopResult.loop_metadata["metal"]` so the report can break down
   per-metal hit rate.  This unblocks the W3 metal-specialised loop.
6. **Convergence detection** — `per_iteration_summary[*]["converged"]`
   will trip when `|best_score − prev_best_score| < 0.01`, terminating
   the loop early.

### 5.1 Phase-1 success criteria

* `n_generate=1000, n_dock=100, n_top=10, refinement_iters=3`
* wall-clock < 5 min on ROCm
* top-10 hit-rate @ `qed ≥ 0.6` ≥ 30 %
* top-10 hit-rate @ `vina ≤ -8.0` ≥ 50 %
* baseline (open-loop) comparison shows ≥ 10 % improvement in mean top-10 score

### 5.2 Phase-2 plans

* DiffDock real model (replace STUB)
* `EGNNPropertyPredictor` for real pIC50
* Metal-binding-aware SA score (`metal_smiles_parser` from W1 A1)
* D-MPNN + EGNN fusion predictor (`TODO/04_architecture/model_design.md`)

---

## 6. File index

| File | Lines | Purpose |
|------|-------|---------|
| `molmetal/orchestration/design_loop.py` | ~430 | DesignLoop + LoopResult + CandidateProvenance |
| `molmetal/scripts/run_design_loop.py` | ~290 | CLI driver + SMILESInjectingGenerator wrapper |
| `molmetal/tests/test_design_loop.py` | ~210 | 7 unit tests |
| `molmetal/reports/design_loop_phase0.md` | this | Phase-0 report |
| `molmetal/reports/design_loop_phase0.json` | ~6 KB | Full LoopResult dump |

End of Phase-0 report.
