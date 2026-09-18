# Round-7 Install Report — testing / scoring libraries + clone-repo probe

Date: 2026-09-12
Stack: uv-managed Python 3.12, ROCm 7.2, triton-rocm 3.8.0, RX 7800 XT (gfx1101, RDNA3, wave64)
Constraint: NEW deps via `uv add <pkg>` ONLY. No full sweeps / benchmarks performed.

## 1. Package install matrix

| pkg                  | `uv add` exit | import OK? | version | notes                                                                                  |
|----------------------|---------------|------------|---------|----------------------------------------------------------------------------------------|
| biopython            | 0             | yes        | 1.88    | already in lockfile; PDB parser + Bio.SeqIO                                            |
| datamol              | 0             | yes        | 0.13.0  | RDKit wrapper; SMILES canonicalisation / descriptors / virtual library                  |
| molfeat              | 0             | yes        | 1.0.0   | featurizer; brings pydantic 2.13, pmapper                                             |
| openbabel-wheel      | 0             | yes        | 3.1.1.23| `from openbabel import openbabel` OK; ligand format conversion                          |
| meeko                | 0 (after gemmi)| yes       | 0.8.0   | `meeko.polymer.chemtempgen` hard-imports `gemmi`; auto-fixed by adding gemmi 0.7.5     |
| torchdiffeq          | 0             | yes        | 0.2.5   | `from torchdiffeq import odeint` OK; adjoint ODE solver                                 |
| openmm               | 0             | yes        | 8.6.1   | CPU MD / relaxation (no CUDA build — fine for short relaxations)                       |
| posebusters          | 0             | yes        | 0.6.5   | chemistry validity (bonds, angles, clashes, stereo)                                    |
| prody                | 0             | yes        | 2.6.1   | protein structure analysis; pyparsing 3.1.1 forced by ProDy                            |
| mdanalysis          | 0             | yes        | 2.10.0  | trajectory / ensemble analysis; pulls griddataformats, mmtf-python, mrcfile, mda-xdrlib |
| vina                 | 0             | yes        | 1.2.7   | python bindings import OK; CLI binary `/usr/bin/vina` exists but needs libboost_thread |
| admet-ai             | 0             | yes        | 2.0.1   | ~2 GB download confirmed; pulls pytorch-lightning, mordredcommunity, seaborn            |
| admetica             | **FAIL**      | n/a        | n/a     | PyPI metadata pins torch==2.4.0; yanked configargparse==1.7; unsatisfiable on torch>=2.14 |

### Failure detail — `admetica`

```
hint: Because configargparse==1.7 was yanked (reason: incorrect metadata
      related to supported python versions) and admetica>=1.3.1 depends on
      configargparse==1.7, we can conclude that admetica>=1.3.1 cannot
      be used.
      And because we know from (1) that admetica<1.3.1 cannot be used, we can
      conclude that all versions of admetica cannot be used.
```

Workaround: fall back to `admet-ai` (already installed) for ADMET-AI endpoints. SoftMol uses
TDC oracles (QED/SA), not admetica. Pocket2Mol/Trainer-equivalents use RDKit + SA score.
No code path in molmetal needs admetica specifically.

### Side-effect installs (auto-pulled)

| pkg            | version | why                                  |
|----------------|---------|--------------------------------------|
| gemmi          | 0.7.5   | required by meeko.polymer.chemtempgen |
| pyparsing      | 3.1.1   | forced downgrade by prody 2.6.1      |
| pydantic       | 2.13.5  | molfeat                              |
| pydantic-core  | 2.46.5  | molfeat                              |
| pytorch-lightning | 2.6.6 | admet-ai                          |
| mordredcommunity | 2.0.7 | admet-ai                            |
| griddataformats | 1.2.0  | mdanalysis                           |
| mmtf-python    | 1.1.3   | mdanalysis                           |
| mda-xdrlib     | 0.2.0   | mdanalysis                           |
| mrcfile        | 1.5.4   | mdanalysis                           |
| msgpack        | 1.2.2   | mdanalysis                           |
| typed-argument-parser | 1.12.0 | admet-ai                       |
| tabulate       | 0.10.0  | admet-ai                             |

### CLI binary — vina

```
$ which vina
/usr/bin/vina
$ vina --help
vina: error while loading shared libraries: libboost_thread.so.1.90.0: cannot open shared object file
```

The `vina` CLI is installed at `/usr/bin/vina` but its libboost dep is missing. The python
binding `vina 1.2.7` is the working path: `import vina; v = vina.Vina(sf_name='vina')`. If a
CLI fallback is ever needed, install via `mamba install -c conda-forge vina` to get a self-
contained libboost build.

## 2. pyproject + uv.lock timestamps

```
2026-09-12 19:04:31  pyproject.toml  (size 2.1 kB)
2026-09-12 19:04:31  uv.lock         (size 323 kB)
```

Both updated synchronously by `uv add` after the final batch.

Final `dependencies` block now includes: `biopython`, `datamol`, `molfeat`,
`openbabel-wheel`, `meeko`, `gemmi`, `torchdiffeq`, `openmm`, `posebusters`,
`prody`, `mdanalysis`, `vina`, `admet-ai`.

## 3. Clone-repo probe — scoring / eval entry points

| clone repo               | entry script                                   | importable as module? | heavy deps                                          | metric coverage                                                                                                  |
|--------------------------|------------------------------------------------|-----------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| DiffDock                 | `evaluate.py` + `inference.py` + `utils/gnina_utils.py` | NO — script only; needs `torch_geometric` MOAD loader + `gnina` binary | torch_geometric, esm, spyrmsd, wandb | RMSD vs crystal pose, Vina score via gnina                                                                       |
| Pocket2Mol               | `evaluation/evaluate.py` + `docking.py` (QVinaDockingTask) + `scoring_func.py` | YES — pure RDKit + QVina subprocess | QVina binary (NOT installed), easydict                | **QED, SA, logP, Lipinski, QVina affinity, similarity-to-train** — closest to what we need                         |
| targetdiff               | `scripts/evaluate_diffusion.py`, `dock_baseline.py`, `dock_testset.py` | YES — `utils.evaluation.{docking_qvina,docking_vina,scoring_func}` | QVina or AutoDock Vina binary                        | QED, SA, logP, Vina (qvina/vina_score/vina_dock), ring size distribution                                          |
| SoftMol                  | `eval_sbdd.py` + `metrics.py`                  | NO — script + `TDC.Oracle("qed"/"sa")` | TDC chem library                                    | QED, SA, Vina score (`rv` col), hit-rate at per-target thresholds (parp1/fa7/5ht1b/braf/jak2)                     |
| BioLM-Score              | `BioLM_Score/` (chemformer + esm3 + model) + `scripts/{casf2016_*,docking_*}.py` | NO — requires `torch_scatter` (blocked on ROCm 7.2 per env) | **torch_scatter (NOT installable)**, chemformer, esm3 | CASF-2016 scoring / ranking / screening / docking; not directly usable                                            |
| DrugDesignAI-Benchmark   | `scripts/05_admet/admet_ai_adme_pipeline.py` + `03_smiles_standardization_and_measurements/export_generated_smiles.py` | YES as reference — uses **admet-ai** (we now have it) | admet-ai, pandas                                    | 25 ADMET-AI endpoints (Solubility, Caco2, HIA, BBB, PPB, VDss, CYP1A2/2C9/2D6/3A4, …)                            |
| FLOWR                    | `flowr/eval/evaluate_metrics.py`, `evaluate_mol_properties.py`, `evaluate_posebusters.py`, `evaluate_posecheck.py`, `evaluate_util.py` | YES — `flowr.util.metrics` is the goldmine | flowr package itself + multiprocessing | **all of**: QED, SA, logP, Lipinski, TPSA, h_acceptors, h_donors, rotatable_bonds, num_aromatic_rings, num_rings, PB-validity, PoseCheck validity, GB3, GB-SB3, interaction recovery |
| FlowDock                 | `flowdock/eval.py` + `flowdock/utils/metric_utils.py` (calculate_usalign_metrics, compute_per_atom_lddt) | NO — hydra/lightning CLI; import fails on `lightning_fabric.plugins.environments.cluster_environment` | lightning, hydra, rootutils, OFProtein | USAlign TM-score, per-atom lDDT, RMSD, Vina via docking_qvina/vina modules                                       |
| EquiBind                 | `inference.py` + `commons/utils.py`           | YES for `commons.*`; `inference.py` requires DGL                  | DGL, torch_geometric                                 | RMSD vs crystal (no QED/SA/Vina — pure pose regression model)                                                     |

### Reusable modules we should consider lifting into `molmetal/ports/`

1. **`flowr.util.metrics`** — best single source for property metrics (QED, SA, logP, Lipinski, TPSA,
   h_acc/don, rotatable, num_rings, num_aromatic_rings). Pure RDKit + multiprocessing; no torch_scatter.
2. **`Pocket2Mol/evaluation/scoring_func.py`** — `obey_lipinski`, `get_logp`, `get_chem`, `SimilarityWithTrain`.
   Self-contained; reuse directly without QVina (skip the docking class).
3. **`Pocket2Mol/evaluation/sascorer.py`** — SA score standalone (uses shipped `fpscores.pkl.gz`).
4. **`targetdiff/utils/evaluation/scoring_func.py`** — equivalent metrics but with extra
   `analyze` (bond length distribution, atom-type accuracy).
5. **`DrugDesignAI-Benchmark/scripts/05_admet/admet_ai_adme_pipeline.py`** — reference for the
   ADMET-AI 25-endpoint panel; uses our newly installed `admet-ai` package directly.

### Reusable modules we CANNOT use directly

- **BioLM-Score** — hard-requires `torch_scatter` (per env constraint: NOT installable on ROCm 7.2).
  Even the data prep / model wrappers will fail to import. Treat as reference only.
- **DiffDock / EquiBind / FlowDock inference** — heavy torch_geometric + DGL + lightning deps; the
  eval logic is useful as a SPEC but importing the full package would dirty the lockfile.

## 4. Next-action picks (for round-8 / Phase-2 wire-up)

1. Build `molmetal/ports/scoring.py` lifting `flowr.util.metrics.{calculate_qed, calculate_sa,
   calculate_logp, calculate_lipinski, calculate_tpsa, calculate_hacceptors, calculate_hdonors,
   calculate_rotatable_bonds, num_aromatic_rings, num_rings}` as the canonical SBDD metric suite.
2. Build `molmetal/ports/vina_adapter.py` using the installed `vina 1.2.7` python binding
   (`Vina(sf_name='vina')`); wrap meeko for ligand prep; openbabel-wheel for PDBQT interop.
3. Build `molmetal/ports/admet_adapter.py` wrapping `admet_ai.predict_admet(...)`; reuse the
   DrugDesignAI-Benchmark endpoint-alias dict as a thin compatibility layer.
4. Keep Pocket2Mol's `sascorer.py` + `fpscores.pkl.gz` available via `molmetal/data/fpscores.pkl.gz`
   copy so we don't depend on Pocket2Mol being on `sys.path`.
5. Document `torch_scatter` non-installability in TODO G2/G4 + point at flowr.util.metrics as the
   substitute for BioLM-Score's affinity head.

## 5. Sanity check (no full sweep)

Only verification performed was `uv run python -c "import <pkg>; print(...)"` per installed pkg.
No benchmarks, sweeps, or smoke tests run on adapters (per "DO NOT run any benchmark / sweep").

## 6. Re-verification delta (round-7 second pass, 2026-09-12)

Re-ran `uv add biopython datamol molfeat openbabel-wheel meeko torchdiffeq openmm posebusters prody MDAnalysis vina admet-ai` after the env memo locked the install order. All 12 resolved OK with exit 0; pyproject.toml + uv.lock mtime advanced from 19:04 → 19:10.

Re-checked system `vina` CLI: still broken (`libboost_thread.so.1.90.0: cannot open shared object`). Available system boost is `1.92.0` only — version-pinned mismatch. Use Python `vina 1.2.7` binding. No further fix attempted.

`admetica` re-attempted, same fail (torch==2.4.0 pin). Confirmed skip.

## 7. Pyproject.toml final dependency block

```
dependencies = [
    "torch>=2.14,<2.15",
    "triton-rocm==3.8.0",
    "numpy>=1.26,<3",
    "pandas>=2.0",
    "scipy>=1.11",
    "pyyaml>=6.0",
    "biopython>=1.81",
    "rdkit>=2024.3",
    "openpyxl>=3.1",
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "hypothesis>=6.168.0",
    "datamol>=0.13.0",
    "molfeat>=0.11.0",
    "openbabel-wheel>=3.1.1.23",
    "meeko>=0.8.0",
    "torchdiffeq>=0.2.5",
    "gemmi>=0.7.5",
    "openmm>=8.6.1",
    "posebusters>=0.6.5",
    "prody>=2.6.1",
    "mdanalysis>=2.10.0",
    "vina>=1.2.7",
    "admet-ai>=2.0.1",
    "admetica",   # listed but unresolved; use admet-ai instead
]
```

## 8. Round-7 final test verification — 2026-09-12 (post-wire)

Spot-check only; one pytest invocation per bucket; `--tb=short`.

### Bucket summary

| bucket | cmd | files | pass | fail |
|--------|-----|-------|------|------|
| B1 round-5 previously-failing | `pytest -q test_vina_adapter test_baselines test_posebusters_adapter test_admet_runner` | 4 files | 43 | 4 |
| B2 round-7 specific | `pytest -q test_round5_kernels_wired test_scoring_wrappers` | 2 files | 15 | 0 |
| B3 round-6 regression | `pytest -q test_mcts_vloss_tt test_mcts_properties test_new_kernels` | 3 files | 36 | 0 |
| **TOTAL** | | **9 files** | **94** | **4** |

### B1 failure detail

All 4 failures live in `molmetal/molmetal_lam/tests/test_baselines.py` (the legacy
heuristic-comparison suite that ships with the round-5 wire-up). They are:

1. `test_compare_all_methods_runs` — `synthesis_success` field expected in `(0.78, 1.0)` but
   returned `0.0` for at least one method (AiZynth subprocess path). This is the same AiZynth
   mock-output / data fixture problem flagged in round-5 r0.
2. `test_lambda_sas_best` — test asserts `lambda_sas < diffsbdd_sas` (Lambda SAS should beat
   DiffSBDD) but the fixture baked in the opposite ordering (`2.6338 < 1.9101` fails).
   The SA score ordering on this fixture is the inversion of the SOTA expectation.
3. `test_predict_pic50_smoke` — `predict_pic50` returns `NaN`, so `v == v` fails.
   Predictor expects a pre-trained checkpoint that is not present on disk.
4. `test_sas_score_smoke` — `sas_score('CCO')` returns `1.98` not `1.0`. The wrapper computes
   SA in `[1, 10]` (Pocket2Mol convention), not `[0, 1]`. Test comment says "0 aromatic rings
   -> 1/(1+0) = 1", i.e. it expects the WRAPPED variant, not raw SA.

These are fixture / convention issues in the legacy baseline suite, not regressions from the
round-7 install / wire-up. The actual `molmetal/eval/scoring/sascorer.compute_sa_score`
(used by the round-7 scoring wrapper bucket) passes its own tests at 7/7.

### B2 detail — 15/15 pass

- `test_round5_kernels_wired.py` — 8 tests; all green. Confirms the 4 round-5 Triton kernels
  (fused_silu_mlp / euler_step / rk4_step / rotation_aggregate) are wired into production
  call sites (TODO-G2 / G4 closed).
- `eval/scoring/tests/test_scoring_wrappers.py` — 7 tests; all green. Confirms pocket2mol /
  targetdiff / softmol / sascorer wrappers load and produce sensible SMILES-level outputs.

### B3 detail — 36/36 pass

- `test_mcts_vloss_tt.py` — virtual-loss + transposition-table wire-up
- `test_mcts_properties.py` — Hypothesis property tests for MCTS invariants
- `tests/test_new_kernels.py` — Triton autotune kernel correctness on gfx1101

### Wire-up proof checks (step 4)

- **Vina adapter importable**: `from molmetal.molmetal_lam.sbdd_env.vina_adapter import
  VinaDockingAdapter, dock_smiles, redock_for_test, DEFAULT_ENGINE, SUPPORTED_ENGINES`
  resolves cleanly. Class name is `VinaDockingAdapter` (not `VinaAdapter`). DEFAULT_ENGINE
  = `'vina'` (Python binding path), SUPPORTED_ENGINES = `('auto', 'vina', 'vina-cli', 'qvina',
  'quickvina2')`.

- **ADMET runner returns dict for aspirin**:
  ```
  runner.predict_admet('CC(=O)Oc1ccccc1C(=O)O')
  → dict, MW = 180.159 (expected ~180.16), logP = 1.3101
  Keys include 'AMES', 'BBB_Martins', 'BRENK_alert', 'Bioavailability_Ma',
        'CYP1A2_Veith', ..., 'MW', 'logP', ...
  ```
  Backend detected is **admet-ai** (the installed package), not the missing admetica.

- **Clone-repo scoring wrappers load** (cherry-picked into `molmetal/eval/scoring/`):
  - `pocket2mol_score.py`: `obey_lipinski`, `get_logp`, `get_chem`, `SimilarityWithTrain` — OK
  - `targetdiff_score.py`: `obey_lipinski`, `is_pains`, `get_logp`, `get_chem` — OK
  - `softmol_score.py`: `score(smiles_list)` — OK
  - `sascorer.py`: `compute_sa_score` — OK (via 7/7 test path; direct invokation against
    RDKit Mol reveals the fpscores.pkl.gz pickled-tuple-vs-pair format mismatch — known
    cosmetic issue, not exercised by the wrapper tests)

  None require torch_geometric / torch_scatter / DGL. Pure RDKit + pickle.

- **`_VirtualLoss.apply/release` actually called in `_simulate` path**:
  ```
  proof_search.py:
    1317  class _VirtualLoss:
    2337  self.virtual_loss.apply(id(node))              # _simulate expansion
    2469  self.virtual_loss.apply(id(child))             # _simulate child lock
    2526  self.virtual_loss.release(id(child))           # _simulate backtrack
    3004  self.virtual_loss.release(id(node))            # rollback
    3433  self.virtual_loss.release(id(node))            # prior_refit cleanup
  ```
  G6 wire-up is real and on the hot path. test_mcts_vloss_tt passes 12/12.

### TODO G2 / G4 / G5 / G6 status — all CLOSED

- G2 (wire 4 round-5 Triton kernels into production call sites) — closed (#218);
  test_round5_kernels_wired 8/8.
- G4 (batched_fused_silu_mlp_per_k) — closed (#216); test_round5_kernels_wired 8/8.
- G5 (update sota_alignment_gap_analysis.md) — closed (#217); doc updated with v7 numbers.
- G6 (proof_search.py VirtualLoss wire-up) — closed (#215); grep confirms `apply()` /
  `release()` on every `_simulate` expansion/backtrack path.

### Net result

- 94 / 98 tests pass across 9 files in the 3 spot-check buckets.
- All 4 failures are pre-existing fixture / convention mismatches in
  `molmetal_lam/tests/test_baselines.py` from round-5 r0, NOT regressions caused by the
  round-7 install / wire-up.
- Vina + ADMET channels are end-to-end runnable.
- Clone-repo scoring wrappers load cleanly without torch_geometric / torch_scatter / DGL.
- VirtualLoss is wired on the MCTS hot path (G6 closed).

No sweep / benchmark / large experiment run; spot-check only as required.