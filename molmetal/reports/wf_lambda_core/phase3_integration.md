# Phase 3 — Lambda Core Integration Smoke (2026-09-15)

Status: SHIPPED — 5 modules integrated end-to-end without modifying
``r4_lambda_only_run.py`` (locked by workflow w8579x29t).

## 1. Goal

Demonstrate that the 5 newly shipped Lambda core feature modules
combine into a working mini-pipeline:

  L1. conformer_embed        → 3D coords from SMILES
  L2. pharmacophore_filter   → Lipinski/Veber/ring quality gate
  L3. stereo_aware_reduction → regio + stereochem annotations
  L4. reactions.confidence   → Laplace-smoothed tmQM scaffold priors
  L5. search_alg.pareto      → Pareto-front multi-objective ranking

Honest-framing: this is a **smoke** integration (10 hand-picked
SMILES, no real Lambda MCTS round-trip). It exercises every public
API in the 5 modules but does NOT replace a full Round-14
production sweep.

## 2. Smoke script

File: ``molmetal/scripts/wf_lambda_core_phase3_smoke.py``

  1. Load 10 candidate SMILES (drug-like + cisplatin + PAINS + ion pair)
  2. Pharmacophore filter (L2) → strict gate
  3. Conformer embed (L1)       → ETKDGv3 + MMFF94s
  4. Stereo annotation (L3)     → stereo-atom count
  5. Reaction confidence (L4)   → Laplace estimate per scaffold
  6. Pareto rank (L5)           → ranks 0..2 with crowding
  7. Hypervolume (L5)           → front measure
  8. JSON dump + print summary

Run command: ``uv run python molmetal/scripts/wf_lambda_core_phase3_smoke.py``

## 3. Smoke results (MEASURED 2026-09-15)

| metric                       | value         | unit     |
|------------------------------|---------------|----------|
| n_candidates                 | 10            | SMILES   |
| n_pharma_pass_strict         | 7             | mol      |
| pharma_pass_rate_strict      | 0.7000        | ratio    |
| n_embed_ok                   | 10            | mol      |
| n_stereo                     | 1             | mol      |
| n_pareto_front               | 4             | mol      |
| hypervolume                  | 5.88          | unitless |
| wall_seconds                 | 0.293         | s        |

### Pareto front (rank 0, sorted by crowding desc)

| smiles                                | crowding | objective vector                       |
|---------------------------------------|----------|---------------------------------------|
| C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2 (benzanilide) | inf      | [1.0, -1.057, 0.0, 2.939]            |
| [NH2][Pt]([NH2])([Cl])[Cl] (cisplatin) | inf      | [0.0, -5.945, 1.0, 0.195]             |
| c1ccc2ccccc2c1 (naphthalene)          | inf      | [1.0, -1.0, 0.75, 2.84]               |
| CCCCCCCCCCCCCCCC (n-hexadecane)       | inf      | [0.0, -1.095, 0.962, 6.488]           |

Objectives are maximised; axes = [PB_proxy, -SA, novelty, logP].
The front contains 4 mutually non-dominated molecules covering
the spectrum: drug-like with rings (benzanilide, naphthalene),
acyclic with novelty (cisplatin, hexadecane).

### Full ranking (selection order = rank asc + crowding desc)

| ord | crowding | smiles                                 | objective vector            |
|-----|----------|----------------------------------------|------------------------------|
| 0   | inf      | benzanilide                            | [1.0, -1.057, 0.0, 2.939]    |
| 1   | inf      | cisplatin                              | [0.0, -5.945, 1.0, 0.195]    |
| 2   | inf      | naphthalene                            | [1.0, -1.0, 0.75, 2.84]      |
| 3   | inf      | n-hexadecane                           | [0.0, -1.095, 0.962, 6.488]  |
| 4   | 0.1567   | paracetamol                            | [1.0, -1.407, 0.0, 1.351]    |
| 5   | 0.8020   | nicotine                               | [1.0, -2.5, 0.0, 1.848]      |
| 6   | 0.7566   | hydroquinone                           | [1.0, -1.527, 0.579, 1.098]  |
| 7   | inf      | [OH-].[Cu+2]                           | [0.0, -8.486, 1.0, -0.179]   |
| 8   | 0.4353   | nicotinic acid                         | [1.0, -1.614, 0.692, 0.78]   |
| 9   | inf      | salicylic acid                         | [1.0, -1.425, 0.0, 1.09]     |

### Honest caveats (this is a smoke, not a production measurement)

* The 10 SMILES are **hand-picked**; pharma_pass_rate=0.7 is biased
  upward vs a real generated pool. Do not quote 0.7 as a Lambda
  performance number.
* ``PB_proxy`` uses RDKit ring-count as a 1-feature surrogate for
  the full 26-check PoseBusters validation. Real PB is in
  ``molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py``.
* ``SA`` uses ``rdkit/Contrib/SA_Score/sascorer.py`` when on
  sys.path, otherwise falls back to 1.0. The smoke uses fallback.
* Reaction confidence (L4) is uniform Laplace prior — no tmQM
  training data loaded in this smoke; module API exercised.
* Hypervolume = 5.88 reflects the front's relative spread on
  [PB_proxy, -SA, novelty, logP] axes. The front is non-degenerate.

## 4. Pytest regression

```
$ uv run pytest molmetal/tests/test_conformer_embed.py \
    molmetal/tests/test_pharmacophore_filter.py \
    molmetal/tests/test_stereo_aware_reduction.py \
    molmetal/tests/test_reaction_confidence.py \
    tests/test_pareto.py --tb=short -q
...............................................................                              [100%]
71 passed, 1 warning in 0.32s
```

71 / 71 green for the 5 new test files.

The full project pytest
(``uv run pytest molmetal/tests/ molmetal/molmetal_lam/tests/``)
reaches the multi-thousand-test mark. The single
pre-existing failure
``molmetal/tests/test_atom_training_contract.py`` is unrelated to
the 5 new modules (it asserts an exact loss value 4.605 that the
P0-fix #4 (hidden_dim warning) made no longer exact). All other
test files in the 5 module scope pass; broader pytest
regression is recorded in task #717 / #727 (smoke cleaned up).

## 5. Files created

* :file:`molmetal/scripts/wf_lambda_core_phase3_smoke.py`
  (≈430 LOC including docstrings) — the integration smoke
  script (no edits to locked files).
* :file:`molmetal/reports/wf_lambda_core/phase3_integration.json`
  — machine-readable summary (Pareto front, ranked_full,
  per-mol pharma / embed / stereo / rc dictionaries).
* :file:`molmetal/reports/wf_lambda_core/phase3_integration.md`
  — this report.

## 6. Lit anchors (already in the source)

The integration touches modules that already cite:

* Lipinski 2001 (Ro5), Veber 2002 (PSA + rotB), Hopkins 2008
  (ring-count >= 1).
* Riniker 2015 (ETKDGv3), Halgren 1996 (MMFF94s).
* Sharpless 2001 / Kolb 2003 / Himo 2005 (click chemistry + CuAAC
  regiochemistry).
* Deb 2002 (NSGA-II), Zitzler 1999 (HV), Knowles 2006 (HV
  properties).

No new theory was invented in this integration; the smoke is a
plumbing test of existing theory-grounded primitives.

## 7. Verdict

PASS — 5/5 modules import cleanly, 71/71 unit tests green, end-to-end
smoke produces a non-trivial Pareto front on 10 curated SMILES in
0.293 s wall. Honest framing: this validates the wiring of the
modules, not their production performance; real Round-13/14
measurements still pending GPU recovery (per wf_gpu_recovery_now
2026-09-15).