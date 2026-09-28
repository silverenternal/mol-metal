# Code Availability Statement — Mol-Metal (R17 / R18 close-out)

**Date:** 2026-09-26 (Wf-26 Phase 4 ship)
**Scope:** Every line of code, configuration, benchmark script, and
reproducibility artefact shipped as part of this submission.
**License:** **MIT** (permissive, redistribution + modification allowed
with attribution; full text in `LICENSE` at the repo root).

---

## 1. Repository

- **Repository:** `https://github.com/.../mol-metal` (placeholder pending
  release; tracked in `TODO/release_plan.md`)
- **Default branch:** `master`
- **Tag for this paper:** `r18-paper-final` (planned at submission)
- **Archive:** GitHub release + Zenodo mirror (DOI assigned at submission)
- **License:** MIT (`LICENSE` at repo root)

The repository is a `uv`-managed Python 3.12 project. `uv sync` from a
clean clone reproduces the full environment bit-for-bit (REPRODUCIBILITY.md § 2).

---

## 2. Source-tree layout

```
mol-metal/
├── molmetal/                 # core library (~282 Python modules)
│   ├── adapters/             # flow-matching, EGNN, SE(3) wrappers
│   ├── data/                 # dataset loaders + tokenisers
│   ├── molmetal_lam/         # λ-calculus / typed-β-NF engine
│   │   ├── atoms/            # atom-typed BNF
│   │   ├── bonds/            # bond-typed BNF
│   │   ├── lam_chem/         # MetalClick typing + combinators
│   │   ├── reactions/        # click-rule library
│   │   ├── reward/           # soft tiered reward + learned shaping
│   │   ├── sbdd_env/         # SynFlowNet env wrappers
│   │   ├── search_alg/       # proof search / MCTS
│   │   ├── tile_lib/         # tiled λ-terms
│   │   └── training/         # proof-search trainer
│   ├── benchmarks/           # benchmark drivers
│   ├── pipeline/             # full end-to-end pipelines
│   ├── priors/               # tmQM, ML-CPM, FG priors
│   ├── configs/              # YAML experiment configs
│   └── metrics/              # shared metric utilities
├── molmetal/scripts/         # 190 benchmark / utility scripts
├── molmetal/tests/           # 342 test files
├── .claude/skills/           # 8 SKILL.md files (see § 4)
├── triton_kernels/           # 13 Triton ROCm kernels
├── data/                     # PlatinAI, tmQM, ASTSR corpora
├── configs/                  # top-level experiment YAMLs
├── flow_matching/            # CFM training + checkpoints
├── environments/             # CUDA / ROCm env specs
├── paper/                    # manuscript + supplementary
│   ├── supplementary/        # S1–S8 + DATA / CODE / ETHICS
│   ├── figures/              # Figure 1–6 + appendix figs
│   └── reports/              # 21 wf-XX reports (see § 5)
├── TODO/                     # planning + audit trail
├── pyproject.toml            # uv-managed dependency list
├── uv.lock                   # pinned dependency lockfile
├── env.sh                    # activation script
├── Makefile                  # top-level make targets
├── README.md / readme.md     # project overview
├── DATA_STATUS.md            # data status tracker
├── CHANGELOG.md              # round-by-round history
└── LICENSE                   # MIT
```

---

## 3. R18 modules shipped (32 modules)

These are the modules specifically added or substantially refactored in
the R18 close-out pass. All ship in the public release under MIT.

| # | Module | Path | Status |
|---|---|---|---|
| 1 | Multi-PH Hilbert primal-dual | `molmetal/adapters/multi_ph_integration.py` | shipped |
| 2 | Transport relaxation prior | `molmetal/adapters/transport_relaxation_integration.py` | shipped |
| 3 | SE(3) Lipman affine path | `molmetal/adapters/se3_lipman_integration.py` | shipped |
| 4 | Flow-matching Lipman (Lipman α-schedule) | `molmetal/adapters/flow_matching_lipman/` (4 files) | shipped |
| 5 | Quotient-space diffusion | `molmetal/adapters/quotient_space_diffusion.py` | shipped |
| 6 | YuelBond decoder | `molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` | shipped |
| 7 | W-flow CFM integration | `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` | shipped |
| 8 | Fractional ODE | `molmetal/adapters/flow_matching_lipman/fractional_ode.py` | shipped |
| 9 | M5 schedulers | `molmetal/adapters/flow_matching_lipman/m5_schedulers.py` | shipped |
| 10 | W-flow sampler | `molmetal/adapters/flow_matching_lipman/w_flow.py` | shipped |
| 11 | AMP schedule (mixed precision) | `molmetal/adapters/flow_matching_lipman/amp.py` | shipped |
| 12 | Coordination GNN | `molmetal/adapters/coordination_gnn.py` | shipped |
| 13 | Coordination graph printer | `molmetal/adapters/coordination_gnn.py::print_graph` | shipped |
| 14 | EGNN coord + scatter (Triton-fused) | `molmetal/adapters/egnn_rocm.py` | shipped |
| 15 | EGNN predictor | `molmetal/adapters/egnn_predictor.py` | shipped |
| 16 | EGNN ROCm checkpoint | `molmetal/adapters/egnn_rocm_checkpoint.py` | shipped |
| 17 | Meeko receptor prep | (vendored via `molmetal/adapters/`) | shipped |
| 18 | RDKit predictor | `molmetal/adapters/rdkit_predictor.py` | shipped |
| 19 | RxnFlow adapter | `molmetal/adapters/rxnflow_adapter.py` | shipped |
| 20 | Facebook FM wrapper | `molmetal/adapters/facebook_fm_wrapper.py` | shipped |
| 21 | Facebook FM smoke | `molmetal/adapters/facebook_fm_smoke.py` | shipped |
| 22 | SE(3)-SB integration | `molmetal/adapters/se3_sb_integration.py` | shipped |
| 23 | DiffDock adapter | `molmetal/adapters/diffdock.py` | shipped |
| 24 | EquiBind adapter | `molmetal/adapters/equibind.py` | shipped |
| 25 | Mock adapter (CI / smoke) | `molmetal/adapters/mock.py` | shipped |
| 26 | Metal-λ coupling | `molmetal/molmetal_lam/lam_chem/coupling_adapter.py` | shipped |
| 27 | Lambda combinators | `molmetal/molmetal_lam/lam_chem/lambda_combinators.py` | shipped |
| 28 | PlatinAI dataset loader | `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` | shipped |
| 29 | Pt click compatibility | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | shipped |
| 30 | Pt metal-ligand exchange | `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` | shipped |
| 31 | Cisplatin builder | `molmetal/molmetal_lam/lam_chem/cisplatin_builder.py` | shipped |
| 32 | SyntheMol reactions | `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py` | shipped |
| 33 | Learned reward shaping | `molmetal/molmetal_lam/reward/learned_shaping.py` | shipped |
| 34 | Soft tiered reward aggregator | `molmetal/molmetal_lam/reward/` (10 files) | shipped |
| 35 | Proof-search engine | `molmetal/molmetal_lam/search_alg/proof_search.py` | shipped |
| 36 | CrossDocked loader | `molmetal/data/crossdocked.py` | shipped |
| 37 | MetalCytoTox loader | `molmetal/data/cytotox.py` | shipped |

(More than 32 modules touch R18; the list above is the named-R18 set
that the manuscript text calls out by name.)

---

## 4. Skills shipped (8 SKILL.md files)

The repo ships a `.claude/skills/` directory with 8 SKILL.md
domain-knowledge packs that encode the project's conventions and
reproducibility gates. They are not required to run the code; they are
shipped so that future Claude instances (and human contributors) can
pick up the project's conventions without re-deriving them.

| # | Skill | Path | Purpose |
|---|---|---|---|
| 1 | `mol-metal-paper-writing` | `.claude/skills/mol-metal-paper-writing/SKILL.md` | paper-writing workflow + figure / table conventions |
| 2 | `mol-metal-r18-math` | `.claude/skills/mol-metal-r18-math/SKILL.md` | typed β-NF + λ-calculus math conventions |
| 3 | `mol-metal-lambda-calculus` | `.claude/skills/mol-metal-lambda-calculus/SKILL.md` | λ-calculus semantics + soundness proof |
| 4 | `mol-metal-cfm-training` | `.claude/skills/mol-metal-cfm-training/SKILL.md` | CFM training pipeline + GPU-stack conventions |
| 5 | `mol-metal-click-chemistry` | `.claude/skills/mol-metal-click-chemistry/SKILL.md` | 5 click reactions + MetalLigandExchange |
| 6 | `mol-metal-anticancer` | `.claude/skills/mol-metal-anticancer/SKILL.md` | anticancer oracles + metal identity conventions |
| 7 | `mol-metal-mechanism-mapping` | `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` | mechanism-of-action mapping (DNA / GSH / TOP1) |
| 8 | `party-twenty-spirit` | `.claude/skills/party-twenty-spirit/SKILL.md` | internal workflow rituals + wf report conventions |

---

## 5. Bench scripts (10+ scripts)

The repo ships more than 10 standalone bench / smoke scripts under
`molmetal/scripts/`. A representative subset called out in the
manuscript or supplementary:

| # | Script | Purpose |
|---|---|---|
| 1 | `benchmark_r18_end_to_end.py` | full R18 pipeline on PlatinAI + MetalCytoTox |
| 2 | `benchmark_r18_metalcytotox.py` | MetalCytoToxDB 5-metal scaffold split |
| 3 | `benchmark_r18_platinai.py` | PlatinAI 5K scaffold split |
| 4 | `benchmark_lambda_cfm_coupling.py` | λ + CFM coupling ablation |
| 5 | `benchmark_mcts_real.py` | MCTS on real (not synthetic) ligands |
| 6 | `benchmark_metalcytotox_e2e.py` | MetalCytoToxDB end-to-end |
| 7 | `benchmark_astsr_extended.py` | ASTSR oracle bench |
| 8 | `benchmark_wf25_5seed.py` | wf-25 5-seed reproducibility bench |
| 9 | `metallo_drug_smoke_retrain.py` | CFM retrain smoke (CI) |
| 10 | `r4_lambda_only_run.py` | λ-only (no CFM) baseline |
| 11 | `ablation_counterion.py` | counterion ablation |
| 12 | `ablation_linear_descriptor_prior.py` | linear-descriptor prior ablation |
| 13 | `ablation_temporal_grid.py` | temporal-grid ablation |
| 14 | `baseline_lomo_cv.py` | leave-one-metal-out CV baseline |
| 15 | `baseline_temporal_holdout.py` | temporal holdout baseline |
| 16 | `calibrate_conditioned_pic50_baseline.py` | pIC50 calibration baseline |
| 17 | `calibrate_pic50_uncertainty.py` | pIC50 uncertainty calibration |
| 18 | `click_rule_effect_size_study.py` | click-rule effect-size study |
| 19 | `close_loop_1_deeper_seed.py` | R18 close-out 5-seed deep benchmark |
| 20 | `audit_pic50_assay_data.py` | pIC50 assay-data audit |
| 21 | `analyze_measured_reward_control.py` | measured-reward-control analysis |
| 22 | `bench_rocm_throughput.py` | ROCm throughput bench |
| 23 | `check_blackwell_env.py` | Blackwell env verification |
| 24 | `check_3d_embed.py` | 3-D embed sanity check |
| 25 | `check_submission_ready.py` | pre-submission readiness check |

---

## 6. Tests

342 test files under `molmetal/tests/` (excluding vendor tests).
Representative subsets:

| Suite | Coverage |
|---|---|
| `test_round10_metrics_harness.py` | R10 metrics harness |
| `test_atom_training_contract.py` | atom-typed BNF training contract |
| `test_generate_atom_types.py` | atom-type generator |
| `test_lipman_spatial_contract.py` | Lipman spatial-path contract |
| `tests/test_atom_training_contract.py` | (legacy test path) |
| `tests/test_lipman_spatial_contract.py` | (legacy test path) |

All tests are runnable via `make test` (or `uv run pytest molmetal/tests/`)
on a clean clone.

---

## 7. Workflow reports (21 wf reports)

The project ships 21 numbered workflow reports (`wf-01` … `wf-26`) under
`paper/reports/` and `TODO/archive/`. Each captures one end-to-end
audit pass over the paper / codebase:

| Report | Phase |
|---|---|
| wf-01 … wf-09 | R10 / R11 baseline + early ablation |
| wf-10 … wf-15 | R12 honest-negative reframe + data-gap alignment |
| wf-16 … wf-20 | R13 lit-grounded + CFM redo + Lambda coupling |
| wf-21 … wf-24 | R14 weak-to-strong + complete plan |
| wf-25 | 5-seed reproducibility bench |
| wf-26 | paper-writing close-out + ship |

---

## 8. Top-level commands

```bash
# 1. Clone + enter environment
git clone https://github.com/.../mol-metal
cd mol-metal
uv sync                           # reproduces Python 3.12 env from uv.lock
source env.sh                     # sets CUDA / ROCm env vars

# 2. Reproduce a single number
make bench-r18-metalcytotox       # runs benchmark_r18_metalcytotox.py

# 3. Run the full test suite
make test                         # uv run pytest molmetal/tests/

# 4. Build the paper
make paper                        # pdflatex over paper/main.tex

# 5. Pre-submission readiness check
make submission-check             # runs check_submission_ready.py
```

See `Makefile` for the full target list. Every target that ships a
number in the manuscript has a 1:1 mapping to a `bench_*` or
`check_*` script.

---

## 9. Honest caveats

- **GitHub URL:** the URL above is a placeholder pending the public
  release. The actual URL + release tag + Zenodo DOI will be inserted
  at submission time.
- **CUDA / Blackwell primary:** primary hardware stack is NVIDIA
  Blackwell (sm_120); AMD ROCm 7.2 is a documented fallback. Both paths
  reproduce the shipped numbers within float32 ε; see REPRODUCIBILITY.md
  § 1 + § 3.
- **Skill files:** the `.claude/skills/` directory is project-internal
  documentation; it is not required to run any code. Removing it does
  not affect any test or benchmark.
- **TODO/ directory:** `TODO/` contains planning + audit notes from
  every round. It is shipped for transparency but is not part of the
  reproducibility surface; do not depend on its content for
  reproducibility.

---

## 10. License

```
MIT License

Copyright (c) 2026 Mol-Metal authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Full text at `LICENSE` in repo root.

---

## 11. Contact

For code-access questions not answered above, contact the corresponding
author via the address on the manuscript's title page. Bugs and
reproducibility issues go through the GitHub issue tracker.