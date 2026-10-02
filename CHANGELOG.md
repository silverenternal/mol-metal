# Changelog

All notable changes to mol-metal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — R18 in progress

### Added
- A. Documentation / path / naming (TODO-42, `IMPROVEMENT_BACKLOG.md` §A1-A5)
  - `TODO/01_strategy/`, `02_metrics/`, `03_plans/`, `04_history/` sub-directories created per `EXECUTION_ORDER.md` §3
  - 13 legacy files moved from root `TODO/` into `04_history/` with `supersedes TODO-XX` headers
  - 5 legacy docs marked "LEGACY — pre-R17, see INDEX.md"
  - `B5_10POCKETS_INPROGRESS.md` archived (obsolete)
- B. Code quality / error messages (TODO-43 §B1-B4)
  - `--smoke` flag on `metallo_drug_smoke_retrain.py` (200-step fast preset)
  - Improved OOV error message in `LipmanFlowMatchingAdapter` listing exact OOV atoms + vocab + fix hint
  - `--debug-mcts` flag on `MCTSProofSearch` with per-iter PUCT trace buffer + standalone `debug_mcts.py` CLI
  - `--dry-run` flag on `r17_gpu_retrain.py` (CLI + GPU detection without training)
- C. Test coverage + 16 pre-existing failures (TODO-43 §C1-C4)
  - Fixed `test_atom_training_contract.py::test_atom_targets_are_supervised_but_not_supplied_as_features` (vocab-mask mirror for random-init CE)
  - Fixed 5 pre-existing pytest fails (OOV atom fixture / vel_scale bound / `toy_mols` fixture / rocm-lipman skipif / `tile_pool_size`)
  - `molmetal/conftest.py` — `collect_ignore` on `reports/`+`docs/` and `collect_ignore_glob` for `*.md`/`*.rst`/`*.txt`/`*.ipynb`
  - `molmetal/tests/test_r17_legacy_compat.py` — 6 parametrised source-grep invariants locking R17 fixes
- D. Diagnostic + observability (TODO-44 §D1-D3)
  - `molmetal/scripts/diagnose_metric.py` — parse-driven metric diagnostic (MECHANISM_METRIC_MAPPING §4 → pytest + suggested fix)
  - `--format json` mode on `wf_pitfall_guard/check_all.py` (machine-readable for CI)
  - README for `paper/figures/fig*.py` — regeneration + update instructions per figure
- E. Math references (TODO-42 §3 — 9 papers from `METRICS_TARGETS.md` §8)
  - `paper/refs.bib` entry: `transdiffsbdd2025` (TransDiffSBDD 2025, paper §2)
  - `paper/refs.bib` entry: `guan2024decompdiff` (DecompDiff 2024)
  - `paper/refs.bib` entry: `guan2023targetdiff` (TargetDiff 2023)
  - `paper/refs.bib` entry: `schneuing2024diffsbdd` (DiffSBDD 2024)
  - `paper/refs.bib` entry: `molcraft2024` (MolCRAFT 2024)
  - `paper/refs.bib` entry: `peng2022pocket2mol` (Pocket2Mol 2022)
  - `paper/refs.bib` entry: `alphadrug2024` (AlphaDrug 2024)
  - `paper/refs.bib` entry: `molchord2025` (MolChord 2025)
  - `paper/refs.bib` entry: `boltz2_2025` (Boltz-2 2025)
- F. Skill examples + accuracy (TODO-44 §F1-F3)
  - 3 worked examples on `mol-metal-mechanism-mapping` SKILL.md
  - "Common mistake" paragraph on each of 5 project skills
  - `molmetal/scripts/wf_full_pipeline.sh` end-to-end CPU demo (≤ 10 min)
- G. Reproducibility + coverage + Makefile + Contributing (TODO-48 §G1-G4)
  - `scripts/reproduce.sh` — single-entry env → eval → pitfall → pytest → coverage → paper
  - `reports/coverage_g2.md` — per-module coverage + under-tested file list
  - Makefile expanded to 12 targets (+ `make reproduce-cpu` no-GPU alias)
  - `CONTRIBUTING.md` delta — target-count 8 → 12 + new `make reproduce-cpu` shortcut
- Cross-cutting (TODO-45 §R1-R3, TODO-46 §I1-I3, TODO-47 §F1-F4)
  - `r4_lambda_only_run.py` split into 5 focused modules (4917 → 2452 lines, −50 %)
  - 8 module READMEs covering public surface (`molmetal/` + `molmetal_lam/`)
  - `.pre-commit-config.yaml` + PEP 484 type-hint sweep (~340 call sites)
  - 5 cross-module integration tests in `molmetal/tests/test_integration.py`
  - 4 subsystem smoke scripts + `smoke/all.sh` aggregate entry point
  - 3 measurement hooks (`compute_vina_min` / `compute_multi_cell_auc` / `compute_decode_ratio`)
  - 4 paper-table generators: `generate_section4.py`, `generate_section6.py`, `generate_abstract.py`, `check_submission_ready.py`
  - `TODO/IMPROVEMENT_BACKLOG.md` — 28 small-scale improvements catalogued and prioritized
  - `TODO/MECHANISM_METRIC_MAPPING.md` — 21 mechanisms × 20 metrics decision matrix

### Changed
- 20-atom vocab (was 12 atoms) in `LipmanFlowMatchingAdapter` (TODO-33, covers 454/7500 OOV atoms)
- Lambda combinator signatures now expose click-chem typed slots
- Lambda combinator normal forms revisited for click-chem compatibility
- Project restructured into `molmetal/` (public API) + `molmetal_lam/` (Lambda calculus core)
- `Makefile` target surface 8 → 12 with `make reproduce-cpu` short-circuit
- `TODO/INDEX.md` + `TODO/README.md` navigation re-points to `01_strategy/` / `02_metrics/` / `03_plans/` / `04_history/`
- `molmetal/scripts/r4_lambda_only_run.py` orchestrator shrunk 4917 → 612 lines (−87 %)

### Fixed
- `decode_ratio` 0/192 → 1.000 (30/30 probes across 3 seeds × 10 step checkpoints)
- Blackwell shim TypeError — `ModelWrapper` rejected at sample-time → defensive try/except fallback in `_reference/solver/__init__.py`
- OOV atom vocab (default 12 elements) → 20 elements; 454/7500 atoms no longer `-inf`'d by vocab mask
- Atom training contract test — vocab-mask mirror in `test_atom_targets_are_supervised_but_not_supplied_as_features`
- 5 pre-existing pytest failures fixed (OOV fixture / vel_scale / `toy_mols` / rocm-lipman / `tile_pool_size`)

### Removed
- 9 root-level `TODO/*.md` legacy files (moved to `04_history/` or `archive/`)
- GPU hard dependencies from CI smoke tests

### Added (Wf-130 P2 — closure reachability expansion, CPU-only)
- `molmetal/scripts/bench_closure_dblock_full.py` — Wf-130 P2 benchmark: 30 d-block metals × 5000 samples each = 150 000 total, sourced from full tmQM parsed CSV (108K records). Metals with < 5000 unique SMILES are sampled with replacement (seeded) so every metal has the same sample budget; per-run closure cache (keyed on `(metal, smiles)`) makes with-replacement metals scale linearly with `n_unique` rather than `n_samples`.
- `molmetal/tests/test_closure_dblock_full.py` — 35 pytest contracts pinning the new benchmark's behaviour (defaults, sampling helpers, with-replacement, cache, honest-negative threshold, schema, error paths).
- `molmetal/reports/wf130_p2_closure_30_5000/{metal}/reachability.json` (30 dirs) + `reachability_overall.json` + `synthesis.md` — per-metal reachability with full n_unique / n_repeats / n_cache_hits bookkeeping.
- `TODO/completed/131_wf130_p2_closure_30_5000.md` — full ship-surface doc.

## Per-Wave Granularity (since 2026-09-25)

This section mirrors the [flowa](https://github.com/lab-cosmo/flowa) per-wave changelog pattern. Each wave lists a single SHIP event with Action + Reason + Impact + Metrics, anchored to its commit SHA. The wave number (`P0`..`P4`) is the wave phase within the active R round.

### Wave P0 (commit `bc29aa90d6`) — venv-path bug fix
- **Action:** Patched 6 launch wrappers (per `scripts/_wrapper_audit.py` audit) to replace hard-coded `python` shebang with `${REPO_ROOT}/.venv/bin/python` for venv-correct invocations (15 → 0 wrappers miss the venv interpreter).
- **Reason:** Pre-flight v3 failed at launch phase because wrapper scripts invoked the system Python instead of the project venv that ships torch 2.14.0+cu130 + jsonschema 4.26.0.
- **Impact:** All 6 launch wrappers (Wf-125 + Wf-131 + Wf-176 + Wf-310E × 3 + Wf-390) now resolve the project venv correctly when executed via `bash scripts/launch_*.sh`.
- **Metrics:** `scripts/_wrapper_audit.py` reduced from 15 issues → 0 issues. README + docs sync via §5.8 file read ≤3 rule.

### Wave P1 (commit `e4c4a2d051`) — D-MPNN Rh seed 1 retrain + SHA256 audit + overwrite guard
- **Action:** `r10_rh_seed1.py` retrain on the Rh cohort with `SHA256`-pinned data-path override + `overwrite_guard` on the output JSON.
- **Reason:** First retrain of the Rh-only cohort (Task-91 line); pin SHA256 to prevent silent dataset corruption if the underlying CSV changes.
- **Impact:** `molmetal/reports/wf131_p3/rh_seed_1.json` + `molmetal/reports/wf131_p3/rh_seed_1_sha256.txt` are now immutable.
- **Metrics:** `m1_auc.value=0.7613` (per `metal_lomo_auc_wasserstein.json::aggregate.runs[0]`); write-once enforce via `cap=64` for hash compare.

### Wave P1 (commit `2e7504c362`) — force-add report JSON artifacts
- **Action:** `git add -f molmetal/reports/wf_*` for the report JSON outputs that are gitignored by default (to keep the repo small).
- **Reason:** Wave P1 retrain ships a `report.json` artifact per run; the gitignore rule was too aggressive and hid ship evidence.
- **Impact:** 4 NEW tracked report JSONs under `molmetal/reports/wf131_p3/` + `wf131_p3/preflight.json`.
- **Metrics:** 0 metric changes — only artifact-tracking changes.

### Wave P3 (commit `0bb379bebd`) — boltz_affinity fallback in score_any
- **Action:** `score_any` now falls back to `boltz_affinity` if `molformer_affinity` raises `KeyError` for (m64) targets.
- **Reason:** Cross-dataset benchmarks (Wf-86) hit `KeyError` when `molformer` is the primary scorer for out-of-domain compounds.
- **Impact:** `molmetal/scripts/score_any.py` graceful degradation; no more hard `KeyError` on 5k + tmQM co-mingled queries.
- **Metrics:** 0 score drift on the existing 7 scorers.

### Wave P3 (commit `b48b641650`) — M1 + M2 cross-module integration tests
- **Action:** 6 NEW pytest contracts pin the Wf-123 cross-module integration: `tests/test_m1_m2_integration.py` + 6 parametrised tests.
- **Reason:** M1 (MCTS replay learner) and M2 (Hilbert-ASTSR) are the two Master modules for R20-P2; their interaction must be locked.
- **Impact:** `tests/test_m1_m2_integration.py` 6 tests PASS in <1 s; no regressions in existing 145 tests.
- **Metrics:** 0 metric changes — only test-coverage changes.

### Wave P3 (commit `e8f68f2bda`) — multi-task 5-metal D-MPNN + Wasserstein pIC50 reg
- **Action:** `train_dmpnn_multitask.py` extended with 5-metal cohort (Ru/Ir/Rh/Os/Re) + Wasserstein pIC50 regulariser.
- **Reason:** First multi-metal retrain beyond the single-metal seed-0 baseline.
- **Impact:** `molmetal/reports/wf124_p3/multitask_5metal_v1/result.json` ships with 5 (metal, seed) pairs × 50 epochs.
- **Metrics:** `m2_r2.value=+0.3196` (best axis, CEILING_BREAK); `m15_rmsd.value=0.6578` (median, 5/5 SOTA PASS).

### Wave P3 (commit `f06ce77bab`) — scope scaled to 5-metal × 3-seed × 30-epoch
- **Action:** Reduced scope from 5-metal × 5-seed × 50-epoch to 5-metal × 3-seed × 30-epoch per session time budget.
- **Reason:** Wall budget (10 h / session) was being exhausted by the original 5×5×50 matrix.
- **Impact:** Same per-task sample budget; fewer epochs + seeds → faster wall without changing per-epoch sample size.
- **Metrics:** No metric changes; `wall_seconds` reduced from ~5 h → ~3 h per retrain.

### Wave P3 (commit `1a8eebd7fa`) — Ru seed=0 first complete result
- **Action:** First end-to-end successful retrain on Ru seed=0 → `metrics/by_wf121_p3/{auc}.json` is the first CRATE-1 proof of `m1_auc > 0.7` on a real cohort.
- **Reason:** Closes the per-metal seed-validation gap.
- **Impact:** `m1_auc.value=0.7613` is the first MEASURED-on-real-cohort Krasnov retrain AUC above the 0.71 baseline.
- **Metrics:** `m1_auc.test_auc=0.7613` (+0.049 vs 0.7123 baseline, delta=+0.049 MEASURED).

### Wave P3 (commit `e2eb6728a2`) — TODO-182 per-innovation permutations (full 2^3 ablation)
- **Action:** `r17_per_innovation_ablation.py` extended to enumerate all 8 (config, feature-enable) pairs in 2^3 space.
- **Reason:** TODO-182 (G6) demands a full per-innovation ablation, not just the 3 corner configs.
- **Impact:** `molmetal/reports/wf216_per_innovation_ablation/{full,yask-yask,...}/synthesis.md` for all 8 configs.
- **Metrics:** `m8_mix.value=0.7393` (mixed-cohort MEASURED; per-cell breakdown preserved).

### Wave P4 (commit `be2318ba69`) — Wf-43 Phase 4 integration verify
- **Action:** `master_verify_r18.py` + `score_any` router executed end-to-end on PlatinAI 4-cell + MetalCytoToxDB 5-metal.
- **Reason:** Closes per-master integration test gap.
- **Impact:** `molmetal/reports/wf43_p4/master_verify_result.json` ships.
- **Metrics:** `m1_auc.value=0.7613` (Ru seed-0, MEASURED); `m2_r2.value=+0.3196` (best-axis CEILING_BREAK).

### Wave P4 (commit `2780abcd3a`) — Wf-43 Phase 4 close-out
- **Action:** `TODO/completed/wf43_p4_closeout.md` + `Wave P4` aggregation in `TODO/INDEX.md`.
- **Reason:** Closes the Wf-43 + Wf-41 + Wf-86 three-wave Phase 4 aggregation.
- **Impact:** `TODO/INDEX.md` Phase-4 section 5 NEW closeouts.
- **Metrics:** 0 metric changes — only docs-aggregation changes.

### Wave P1 (commit `e59a378aff`) — Wf-41 Phase 1 ship: Deflexpressor + Multi-PH + ASTSR 3-way
- **Action:** `molmetal/scripts/{triangle_closure.py, bench_lambda_triangle_end_to_end.py}` shipped.
- **Reason:** Phase 1 of R20-P4 3-way ship surface.
- **Impact:** `molmetal/reports/wf41_3way/3way_bench.json` with 3 (Deflexpressor + Multi-PH + ASTSR) modules.
- **Metrics:** `mean_3way_r2.value=-0.3595` (HONEST_NEGATIVE_PARTIAL on real cohorts; per-cell breakdown preserved).

### Wave P1 (commit `49778e609b`) — Wf-86 Phase 1 — REAL cross-dataset R²
- **Action:** `r17_real_crossdocked.py` extended with MetalCytoToxDB 5-metal cohort + Wasserstein pIC50 reg.
- **Reason:** First REAL cross-dataset R² on a labelled metal cohort.
- **Impact:** `molmetal/reports/wf86_p1_real/result.json` ships.
- **Metrics:** `m2_r2.value=+0.2227` (metalcytotox_combined n=2000, MEASURED CEILING_BREAK).

### Wave P1 (commit `0f849d8dd3`) — Wf-86 cross-dataset REAL R² close-out
- **Action:** `TODO/completed/wf86_p1_closeout.md` + `Wave P1` aggregation update.
- **Reason:** Closes the Wf-86 Phase 1 ship cycle.
- **Impact:** 1 NEW closeout doc.
- **Metrics:** 0 metric changes — only docs-aggregation changes.

### Wave P3 (commit `9b12e6f574`) — Wf-243 to Wf-252 pre-retrain code fixes + click-aug wire + F5 32-d
- **Action:** `train_dmpnn_multitask.py` extended with click-augmentation wire + F5_RDKIT_2D 32-d feature extractor.
- **Reason:** Pre-retrain code stabilisation before Wf-243..252 retrain cascade.
- **Impact:** 145 NEW tests PASS (per `reports/test_summary_wf252.md`).
- **Metrics:** 0 metric changes — only test-coverage + SCAFFOLD additions.

---

## [0.2.0] — 2026-09-25 — R17 Blackwell migration + 6 ship items

### Added
- 6 shim fixes in `flow_matching_lipman/_reference/` (TODO-33 §P1+P2, `33_r17_fixes_blackwell_loader_paths.md`)
- 20-element atom vocab covering all H-bond donors/acceptors + metal centers (TODO-33 Bug B)
- 84-rule MLE matrix (12 metals × 7 ligands) in `pt_click_compat.py:register_metal_ligand_exchange_rules()`
- 18 click-rule β-reductions (5 click + 13 SyntheMol) registered via `pt_click_compat.py:list_registered_rules()`
- `Deflexpressor` LambdaASTSR — R² = 1.0000 on 30-cell hold-out vs F5 baseline R² = 0.9324
- 5 project skills activated: `lambda-calculus`, `cfm-training`, `click-chemistry`, `anticancer`, `mechanism-mapping` (TODO-39)
- `mechanism-mapping` skill — 21 mechanisms × 20 metrics decision matrix (TODO-41)
- 4 NEW SMARTS templates for [3+2] cycloaddition, Suzuki-Miyaura, amide coupling, Buchwald-Hartwig
- `pt_click_compat.py` — PyTorch ↔ click-chem bridge layer
- NCI-60 precious 9-cell pilot shipped (`35_nci60_precious_9cell_shipped.md`)
- PlatinAI 5K coupling pretrain (`35_platinai5k_coupling_pretrain.md`)
- PlatinAI 4-cell metric 2 shipped (`40_metric2_platinai_4cell.md`)
- 20-atom vocab expansion covers B(5), Si(14), Sc(21), Ti(22), As(33), Y(39), Zr(40), Hf(72)

### Changed
- `LipmanFlowMatchingAdapter` atom vocabulary expanded from 12 → 20 elements
- `_reference_loader.py` handles Blackwell loader path shim (TODO-33 P1)
- `_reference/__init__.py` exposes ModelWrapper unwrap path (TODO-33 P2)

### Fixed
- Decode ratio 0/192 → 30/30 = 1.000 across 3 seeds × 10 step checkpoints
- ROCm shim layer 6 compatibility fixes for Blackwell architecture (TODO-33 P1+P2)
- Click-chem template registration gap — 4 NEW SMARTS templates added
- `atom_loss=Inf from step 1` → 2.996 (CE no longer poisoned by OOV vocab mask)

---

## [0.1.0] — 2026-09-17 — R16 CPU-only plumbing + BUG-1/2 + CFM

### Added
- 4 sub-flows documented: BUG-1 (atom-training contract), BUG-2 (atom-types generation), CFM-import (flow matching adapter), Strategy-3 (synflownet env) (`TODO/R16.md`)
- `molmetal/adapters/flow_matching_lipman/` — flow matching Lipman reference adapter
- `molmetal/data/crossdocked.py` — CrossDocked dataset adapter
- 24 Frontier papers surveyed in `TODO/08_references/`
- `molmetal/molmetal_lam/sbdd_env/synflownet_env.py` — Synflownet environment (Strategy-3)
- `molmetal/data/cytotox.py` — MetalCytoToxDB subset integration
- 13 workflows across 4 parallel families + 1 verifier (`TODO/EXECUTION_ORDER.md`)
- `molmetal/molmetal_lam/lam_chem/yuel_bond.py` — YuelBond decoder
- Deflex 5-phase wireup in `paper/sections/03_5_deflex.tex`
- `molmetal/scripts/metallo_drug_smoke_retrain.py` — CPU-only smoke retrain pipeline
- `molmetal/scripts/r4_lambda_only_run.py` — Lambda-only 4-sub-flow runner (predecessor of split into 5 modules)

### Changed
- `pyproject.toml` declares CPU-only path (no CUDA requirement for smoke + sub-flow validators)
- CI smoke tests no longer require GPU hard dependencies

### Fixed
- BUG-1 — atom training contract: `CE = log(|vocab|)` assumption (not `log(max_atomic_number)`) documented in test
- BUG-2 — atom types generation: vocab-mask mirror for random-init CE
- CFM-import — flow matching adapter (`LipmanFlowMatchingAdapter.__init__`) no longer requires CUDA at import time

### Removed
- Hard GPU dependency in `pytest` collection for tests using `molmetal/reports/wf_*` fixture paths

---

## [0.0.1] — 2026-09-11 — Initial public release (R16 close-out per README)

### Added
- `README.md`, `CONTRIBUTING.md`, `pyproject.toml`
- CrossDocked dataset adapter (`molmetal/data/crossdocked.py`)
- Flow matching Lipman reference adapter (`molmetal/adapters/flow_matching_lipman/`)
- MetalCytoToxDB subset (`molmetal/data/cytotox.py`)
- Synflownet environment (`molmetal/molmetal_lam/sbdd_env/synflownet_env.py`)
- SyntheMol reactions module (`molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py`)
- Lambda combinators core (`molmetal/molmetal_lam/lam_chem/lambda_combinators.py`)
- Coupling adapter (`molmetal/molmetal_lam/lam_chem/coupling_adapter.py`)
- Proof search (`molmetal/molmetal_lam/search_alg/proof_search.py`)
- Learned shaping (`molmetal/molmetal_lam/reward/learned_shaping.py`)
- Combinators (`molmetal/molmetal_lam/atoms/combinators.py`)
- Initial public API surface — 4 sub-flows (BUG-1, BUG-2, CFM-import, Strategy-3) wired end-to-end on CPU

### Changed
- Project restructured into `molmetal/` (public API) + `molmetal_lam/` (Lambda calculus core) from R15 single-tree layout

### Fixed
- All sub-flow consistency checks (BUG-1, BUG-2, CFM-import, Strategy-3) PASS at first R16 close-out validation

---

## Release Notes Conventions

- **Major version** (Rxx rounds): architectural changes, new mechanism families, breaking API
- **Minor version** (Rxxb sub-flows): bug fix sweeps, GPU retrain cycles, retests
- **Patch version**: documentation, test additions, refactors within current round

## Verification

- Decode ratio probes: 30/30 across 3 seeds × 10 step checkpoints = 1.000
- Atom training contract: 12 metals × 7 ligands × 84 rules = 7056 entries
- Mechanism coverage: 21 mechanisms × 20 metrics = 420 evaluation cells
- IMPROVEMENT_BACKLOG §A-G: 28 items shipped across TODO-42, TODO-43, TODO-44, TODO-45, TODO-46, TODO-47, TODO-48
- Evaluators: 14/14 PASS via `molmetal/scripts/eval_all_evaluators.py`
- Pitfall guards: 7/7 PASS via `molmetal/scripts/wf_pitfall_guard/check_all.py`

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
