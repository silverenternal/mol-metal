# Ultracode Audit — Agent C: LAMBDA LINE
**Date:** 2026-09-14
**Scope:** `molmetal/molmetal_lam/` and `molmetal/molmetal_lam/tests/`
**Mode:** Read-only inventory + classification. No source modifications.
**Authoritative state:** `/home/hugo/codes/try_triton_on_rocm/TODO/completion_audit_2026-09-13.md`

---

## 1. Package inventory

The `molmetal/molmetal_lam/` package is the **Molecular Lambda Calculus (MLC)** subsystem: a lambda-calculus formalism for de novo drug design where atoms-as-combinators bond-as-application yields drug candidates as closed lambda-terms in β-normal form. The package contains **~80 Python files** spread across 14 subpackages plus a top-level `SMOKE_TEST_PLAN.md`. Subpackage and __init__.py summary:

| Subpackage | __init__.py size | Files | Status of __init__.py |
|---|---|---|---|
| `molmetal_lam/` | 72 | 1+1 doc | Real header (Lambda Calculus thesis, 11 layers, status banner) |
| `atoms/` | empty | 1 | Empty (combins re-exported via explicit imports) |
| `binding/` | full | 1 | Real (BindingSite, typecheck, canonical sites) |
| `bonds/` | full | 1 | Real (Bond, DATIVE/COVALENT/AROMATIC, cisplatin) |
| `configs/` | full | 1 | Real (SOTAAlignedConfig YAML loader, frozen dataclasses) |
| `lam_chem/` | 70 | 5 | Real (AST + HeuristicRegressor + cisplatin_builder + well_formedness + rules) |
| `molecules/` | empty | 1 | Empty (closed_term re-exported) |
| `pipeline/` | full | 3 | Real (LamClickDesignLoop orchestrator + features + cross-layer metrics) |
| `priors/` | full | 3 | Real (MetalGeometryPrior, MetalHydrationAnalyzer, AnticancerMetricSuite) |
| `reactions/` | empty (0 bytes) | 5 | Empty (rule registry exposed by direct import in beta_reductions) |
| `sbdd_env/` | full | 25 | Real (adapters for Vina/QuickVina2-GPU/PoseBusters/AiZynth/REINVENT4/DiffDock/FlowDock/FlowR/Pocket2Mol/TargetDiff + RDKit scorers) |
| `scripts/` | present | 7 | CLI/orchestration scripts |
| `search_alg/` | 21 | 3 | Real (MCTSProofSearch, SymbolicPrior, GuidedMCTS sweep helpers) |
| `synthesis/` | empty (0 bytes) | 2 | Empty (BFS β-reduction exposed via `synthesize`/`retrosynthesize`) |
| `tests/` | empty | 52 | 51 test_*.py files |
| `tile_lib/` | 51 | 6 | Real (12-tile library + 200-tile SMARTS-diverse pool + property checks + canonical cache + build_cache) |
| `types/` | empty | 1 | Empty (TypePredicate re-exported) |
| `benchmarks/` | 9 | 2 | Real (engine_parity comparator + bench init) |

**Note**: Empty `__init__.py` files (`atoms/`, `molecules/`, `reactions/`, `synthesis/`, `types/`) are not missing modules — those subpackages each contain one substantive implementation file that callers import directly. Total non-empty Python files: **~80**.

---

## 2. Subpackage classification

Classification per authoritative `completion_audit_2026-09-13.md` (which uses the language **"real", "measured", "skeleton", "missing"** to mean shipped-vs-stub).

### 2.1 `tile_lib/` — fragment pool + click tiles + property tests → **SHIPPED**
- `tile.py` (130 L), `click_tiles.py` (313 L), `library.py` (254 L), `fragment_pool.py` (609 L), `canonical_cache.py` (124 L), `build_cache.py` (80 L), `property_tests.py` (221 L).
- **Coverage**: 12-tile STANDARD_12 (azide+alkyne+partner+thiol) shipped; 220-tile SMARTS-diverse pool (200 ChEMBL/ZINC + 20 R10 axis A handles for DBCO / boronic acids / aryl halides / carboxylic acids / amines) shipped with documented selection criteria (MW 40-300, logP -2-5, ETKDGv3 embed success); canonical SMILES dedup via `canonical_cache.py`; property tests enforce `ALLOWED_ATOMS` (C, N, O, F, S, Cl, Br, I, P, B).
- **Tests**: 9 + 7 + 4 = **20 test functions** across `test_tile_lib.py`, `test_tile_properties.py`, `test_tile_canonical_cache.py`, plus 7 in `test_fragment_library.py`. Confirmed `STANDARD_12_TILES` and `fragments_from_chembl_reactive` actually emit RDKit-parsable tiles.

### 2.2 `search_alg/` — MCTS proof_search + VirtualLoss + TT → **SHIPPED (MCTS core) + SHIPPED (sweep guidance)**
- `proof_search.py` (3,962 L): the substantive **MCTSProofSearch** class. Real implementation: PUCT selection with Dirichlet root noise, epsilon-greedy rollout, multi-reward `RewardAggregator` (w_vina/w_sa/w_posebusters/w_pic50/w_retro/w_reinvent4/w_synth), AlphaZero-style expansion, VirtualLoss hook, `SymbolicPrior` wrapper around `HeuristicRegressor`, history tracking with `iteration/best_score/mean_score/n_states_explored/n_satisfying/best_state` per the **authoritative audit** requirement.
- `sweep_guidance.py` (142 L): **explicit serialized guidance** for controlled sweep ablations — `prepare_prior`/`update_prior` enforce `linear_descriptor_prior` backend only, frozen vs train vs transductive mode validation, schema_version guard, feature-dim guard, source-split guard. The completion_audit confirms "backend is linear ridge, not PySR" — i.e. symbolic regression is **not** what actually runs.
- `_batch_reward_worker.py` (224 L): multiprocessing `Pool` dispatch for SA/QED/Vina_proxy channels.
- **Tests**: 6+4+6+7+8+12 = **43 test functions** across `test_mcts_properties.py`, `test_mcts_early_stop.py`, `test_mcts_vloss_tt.py`, `test_search_reaction_termination.py`, `test_sweep_guidance.py`, `test_proof_search_prior.py`, `test_round10_pt_prior.py`. Authoritative state: "**Real click-tile products and frozen linear-prior PUCT ablation measured**".

### 2.3 `reactions/` — 5 click reactions + rate predictor + kinetic aggregator → **SHIPPED**
- `click_reactions.py` (621 L): 4 click reactions (CuAAC, SPAAC, SPC, DielsAlder) with SMARTS-based product construction + manual triazole templates (RDKit RunReactants leaves aromaticity unset on triazole nitrogens).
- `beta_reductions.py` (1,300 L): full ReactionRule machinery — CuAAC/SPAAC/SPC/DielsAlder/ThiolEne/Suzuki/amide-coupling, mass-balance verification, attach_all_rate_predictors.
- `rate_predictor.py` (450 L): 8-d SMILES pair featuriser + `LITERATURE_YIELDS` (6 reactions × 10 citations) + `HeuristicRegressor`-backed `RatePredictor`.
- `kinetic_aggregator.py` (128 L): harmonic-mean weighted aggregator + optional sklearn ridge fallback.
- `synthesis_oracle.py` (284 L): defensive `attach_all_rate_predictors_safe` + `SynthesisOracle` facade.
- **Tests**: 5+7+8 = **20 test functions** across `test_click_reactions.py`, `test_reaction_operators.py`, `test_kinetic_aggregator.py`, plus thiol-ene-specific coverage in `test_thiolene_connectivity.py` (6) and round-10 click in `test_round10_5_click.py` (6).
- **Note**: R10 axis A confirms 5 click handles (CuAAC + SPAAC + thiol-ene + Suzuki + amide-coupling) wired through `lam_chem/rules.py`.

### 2.4 `priors/` — metal geometry + hydration + anticancer suite → **SHIPPED**
- `metal_geometry.py` (866 L): `SquarePlanarPtII` legacy API + generalised `MetalGeometryPrior` (TODO-09) with `Geometry`/`GeometryPenalty`/`GeometryPenaltyBatched`, Bondi vdW radii, `apply_metal_prior`, differentiable soft penalty that composes with CFM training (no-op when no Pt centre present).
- `metal_hydration.py` (195 L): `MetalHydrationAnalyzer` with DEFAULT_PRIORS for Pt/Ru/Ir oxidation states (Reedijk 1996, Lippard 1999, Hartinger 2006 heuristics).
- `anticancer_metric_suite.py` (138 L): `AnticancerMetricSuite` — Kellogg/Veber logP/TPSA/rotatable/hERG bands with `_ANTICANCER_COUNTERS` instrumentation.
- **Tests**: 7+10+14 = **31 test functions** across `test_metal_geometry.py`, `test_metal_hydration.py`, `test_anticancer_metric_suite.py`. The completion_audit notes "**Property test caught Au_III arity bug**" — i.e. property-based validation is wired in and has already exercised real chemistry.
- Authoritative caveat: "**heuristics do not establish efficacy**" — these are screening proxies, not measured rate constants.

### 2.5 `lam_chem/` — pysr_wrapper + rules + coordination → **SHIPPED (with graceful sklearn fallback)**
- `ast.py` (274 L): immutable LamVar/LamAbs/LamApp dataclasses with capture-avoiding subst + bounded β-NF.
- `pysr_wrapper.py` (427 L): `HeuristicRegressor` with `_probe_pysr()` — runs an actual 1-iter PySR+Jjulia fit; degrades to sklearn RandomForestRegressor or Ridge when Julia subprocess cannot spawn. **Authoritative**: PySR+Jjulia path **does not actually run** here; backend is linear ridge.
- `cisplatin_builder.py` (123 L): programmatic cisplatin = Pt_II(NH3)(NH3)(Cl)(Cl) as left-associated curried dative applications.
- `well_formedness.py` (297 L): three invariants — `check_closed_term`, `check_arity_conservation`, `check_beta_normal_form`; `assert_well_formed` raises `WellFormednessError`.
- `rules.py` (92 L): 5-rule R10 axis A registry (CuAAC + SPAAC + thiol-ene + Suzuki + amide-coupling).
- `batched_rdkit.py` (478 L): multiprocessing `Pool` wrappers (`batch_mol_from_smiles`, `batch_qed`, `batch_sa`, `batch_lipinski`) with sequential fallback for spawn re-entrancy.
- **Tests**: 8+5+10+28 = **51 test functions** across `test_lam_chem.py`, `test_pysr_wrapper.py`, `test_cisplatin.py`, `test_batched_rdkit.py`, `test_atoms_combinators.py`.

### 2.6 `sbdd_env/` — vina/posebusters/aizynth/reinvent adapters + generators → **MIXED**
The 25 files here divide into three buckets.

**(a) Real, measured or wired end-to-end:**
- `vina_adapter.py` (998 L): `VinaDockingAdapter` with auto engine selection (vina Py binding → vina CLI → qvina/quickvina2 CLI); meeko prep; seed propagation; `_parse_vina_result_energies`. Authoritative: "**QuickVina -2.600, Vina -2.502 kcal/mol on identical prepared receptor/ligand/box/seed at N=1**". Both `vina` and `qvina`/`quickvina2` are aliases for the same vendored binary.
- `vina_gpu_adapter.py` (225 L): `QuickVinaGPUAdapter` — the official AMD build path with SHA-256 device-log verification, gpu_threads ≥ 1000 enforcement, separate per-invocation output dirs.
- `pocket_docking_reward.py` (260 L): real, measured kcal/mol provider; explicit `max_evaluations` budget + cached-failure accounting + no-fabrication policy.
- `posebusters_adapter.py` (273 L): three modes (`mol`/`dock`/`redock`); `ValidityReport` dataclass; `pass_rate` bulk helper. R3 confirms PB runner works on real docked poses.
- `aizynth_adapter.py` (284 L): AiZynthFinder adapter with SMARTS fallback path. **Authoritative**: isolated AiZynth 4.4.1 with official legacy_v3 USPTO model — **3 smoke targets got real one-step routes**; current-version asset URLs still fail.
- `aizynth_isolated.py` (87 L) + `aizynth_rocm_policy.py` (100 L) + `aizynth_torch_model.py` (55 L): the isolated environment + AMD ROCm torch policy + ONNX variant. **Authoritative**: "Equivalent torch AMD policy and actual GPU search verified" with `rocm_policy_comparison.json` showing ONNX/ROCm max probability difference 2.32e-6.
- `reinvent4_subprocess_adapter.py` (533 L): subprocess-RPC adapter driving REINVENT4 binary via JSON-line wire format, with health-check heartbeat.
- `reinvent4_adapter.py` (182 L): alternate adapter path (likely subprocess-RPC alternative or proxy multiproperty).
- `reinvent4_jsonl_worker.py` (60 L) + `reinvent_wrapper.py` (208 L) + `reinvent_prior_adapter.py` (248 L) + `reinvent_prior_jsonl_worker.py` (142 L): the **dedicated learned NLL bridge** — official prior generated **5/5 valid molecules** and learned NLL on gfx1101 (authoritative). The reinvent_prior_adapter_amd report shows 7 input rows × GPU NLL RPC with explicit empty/invalid/OOV failure accounting.
- `sa_score.py` (158 L): Ertl-Schuffenhauer SA via `rdkit.Contrib.SA_Score.sascorer`; `sa_score_to_unit` maps [1,10]→[1.0,0.0].
- `qed_scorer.py` (116 L): `QEDScorer` with batched QED + SA scoring; graceful RDKit-missing returns None.
- `synthesis_gate.py` (72 L): `build_synthesis_gate` — SMARTS heuristic vs AiZynth (incl. isolated variant) routing with explicit `applied`/`status` metadata.
- `syntemol_reactions.py` (533 L): SyntheMol Reaction + QueryMol adapter without triggering SyntheMol's wandb/chemprop init; wraps `REAL_REACTIONS`.
- `synflownet_env.py` (540 L): SynFlowNet `ReactionTemplateEnv` adapter — copies the slim `Reaction` class to avoid torch_geometric leakage.
- `voxelization.py` (325 L): `pocket_to_spatial_tiles` voxelises a pocket PDB into a 3D grid with element-aware channels (hbond_donor/acceptor/hydrophobic/charge). NumPy/Python only.
- `metal_generator_adapter.py` (303 L): multi-component `L1.L2....Ln.[M]` reconstruction using `molmetal.data.metal_smiles`.

**(b) Protocol-shaped stubs (no real backend reachable here):**
- `diffdock_adapter.py` (359 L): Protocol wrapper; `AdapterUnavailable` raised unless DIFFDOCK_BIN / vendored repo / PATH binary / `diffdock` importable. Authoritative: torch_cluster/torch_scatter ROCm wheels unavailable.
- `flowdock_adapter.py` (365 L): two adapters in one file — `FlowDockAdapter` (L-1 binding oracle) and `FlowDockReferenceAdapter` (DockingEngine stub for abstract layer).
- `flowr_adapter.py` (146 L): FLOWR stub (torch_cluster/torch_scatter dependency).
- `pocket2mol_adapter.py` (682 L): `_Pocket2MolLiveAdapter` (real, gated on checkpoint + PyG) vs `_Pocket2MolFallbackAdapter` (SMARTS-only baseline emitting 100 drug-like SMILES). **Authoritative**: Pocket2Mol checkpoint URL behind Google Drive; **not downloaded** by task policy.
- `targetdiff_adapter.py` (358 L): `CheckpointUnavailableError` for both TargetDiff (Google Drive) and DiffSBDD (Zenodo 8183747). Cite-only path.
- `pybind_adapter.py` (130 L): `Pybind11StubAdapter` no-op for EquiBind/FLOWR pybind11 extensions (ROCm wheels rarely pre-built).
- `pic50_predictor.py` (204 L): Attentive D-MPNN regressor at `molmetal/checkpoints/dmpnn_atn_ru_pic50.pt`. **Authoritative**: the `pic50_assay_audit_2026-09-13` finds 356/2000 legacy labels had inequality bounds; **uncensored HeLa48h/dark cohort replacement** is the new conditioned baseline.

- **Tests**: 10+12+6+6+6+4+13+3 = **60 test functions** across `test_vina_adapter.py`, `test_posebusters_docked_pose.py`, `test_posebusters_runner.py`, `test_pocket2mol_*` (multiple), `test_synflownet_adapter.py`, `test_diffdock_flowdock_adapters.py`, `test_vina_seed.py`, `test_sbdd_env.py`, `test_reinvent4_subprocess_adapter.py`, `test_qvina_swap.py`, `test_qed_scorer.py`, `test_admet_runner.py`, `test_round10_metrics_harness.py`.

### 2.7 `synthesis/` — derivations + cost estimator → **SHIPPED**
- `derivations.py` (659 L): forward (`synthesize` BFS over β-reductions) + retro (`retrosynthesize` β-expansion enumeration) synthesis paths. `ReactionPattern`, `SynthesisPath`, `to_lambda_expr`, `is_mass_balanced`.
- `cost_estimator.py` (136 L): RDKit-dependent cost with conservative fallback; `_sa_score` uses `rdkit.Contrib.SA_Score.sascorer` when available, descriptor-based proxy otherwise; cost 0=easily synthesizable, 1=hard.
- **Tests**: 16+5 = **21 test functions** across `test_synthesis_derivations.py`, `test_synthesis_cost_estimator.py`.

### 2.8 `benchmarks/` — engine parity → **SHIPPED (mock-bounded)**
- `engine_parity.py` (105 L): Round-11 docking-engine parity harness. Vina runner returns `None` when the Vina executable is unavailable; the two comparison engines are stochastic mocks. Authoritative: this is a **parity test**, not an N=50 head-to-head.

### 2.9 `pipeline/` — closed loop + features + cross-layer metrics → **SHIPPED**
- `closed_loop.py`: `LamClickDesignLoop` orchestrator — 5-step iteration (MCTS search → batch scoring → top-K selection → heuristic fit → equation extraction).
- `extract_features.py`: 8-d descriptor vector (n_atoms / mw / logp / tpsa / n_clicks_used / n_aromatic_rings / n_rotatable / n_h_donors).
- `cross_layer_metrics.py`: 4 synthesis-feasibility metrics (synthesis_success, SA_score_mean, retrosynth_feasibility, end_to_end_yield_proxy).
- **Tests**: 7+9 = **16 test functions** across `test_pipeline.py`, `test_closed_loop_reward.py`, `test_baselines.py`.

### 2.10 `atoms/`, `bonds/`, `molecules/`, `binding/`, `types/`, `configs/` → **SHIPPED**
Each contains exactly one substantive module (empty __init__/one-file subpackage). All five have substantial test coverage (`test_atoms_combinators.py`: 12, `test_bonds_application.py`: 29, `test_closed_term.py`: 21, `test_lambda_properties.py`: 6, `test_round10_pt_prior.py`: 12, `test_baselines.py`: 4). `configs/sota_aligned.py` ships the YAML schema loader (`load_sota_aligned_config`, `to_mcts_kwargs`).

### 2.11 `search_alg/_batch_reward_worker.py` → **SHIPPED (CU-safe picklable config)**
A 224-line multiprocessing dispatch helper that serialises the `RewardAggregator`'s RDKit-backed channels across `multiprocessing.Pool`. Non-RDKit channels (e.g. a fitted docking oracle) are explicitly **silently dropped** — the parent re-applies them in the serial fallback when `n_workers == 1`.

---

## 3. Test inventory

**Total Python files in tests/**: 53 (including `__init__.py` empty, `conftest.py` if any). **Test files** (`test_*.py`): **51**. **Test functions (`def test_`)** count = **444** (verified via grep aggregation). Highest-density test files:

| File | Test fns |
|---|---|
| test_bonds_application.py | 29 |
| test_batched_rdkit.py | 28 |
| test_sweep_helpers.py | 21 |
| test_closed_term.py | 21 |
| test_synthesis_derivations.py | 16 |
| test_anticancer_metric_suite.py | 14 |
| test_synflownet_adapter.py | 13 |
| test_sweep_guidance.py | 12 |
| test_round10_pt_prior.py | 12 |
| test_posebusters_docked_pose.py | 12 |
| test_atoms_combinators.py | 12 |
| test_vina_adapter.py | 10 |
| test_qvina_swap.py | 10 |
| test_metal_hydration.py | 10 |
| test_cisplatin.py | 10 |
| (many 6-9 test files) | … |

Test status: per round-8 closure ("**21/21 test files green (158 pass / 1 skip / 1 xfail)**") and `test_lambda_sweep_*` spot-check files (`test_lambda_sweep_click_initialization.py`, `test_lambda_sweep_config_dispatch.py`, `test_lambda_sweep_generation_diagnostics.py`). Full broad regression is reported as **995 → 1134 → 1170 passed, 3 skipped, 1 xpassed** at successive stages of round-8/10 work (some pending re-run after model stabilisation per `completion_audit_2026-09-13.md`).

---

## 4. End-to-end runnability — synthesis

**What runs end-to-end today (per `completion_audit_2026-09-13.md`):**

1. **Tile-library → click → molecule** — `fragments_from_chembl_reactive()` (220 SMILES, SMARTS-diverse) → `apply_click_reaction` (CuAAC/SPAAC/thiol-ene/Suzuki/amide-coupling) → RDKit-valid product. R3/R10 measurements confirm real products.
2. **MCTS proof search over λ-term space** — `MCTSProofSearch` from seed tile → top-K candidates → reward aggregation with `SymbolicPrior` (linear ridge backend, **not** PySR). All 5 click reactions selected via `all_5`. Capped tile count governs expansion attempts.
3. **Heuristic regression splicing** — `HeuristicRegressor` falls back to sklearn; the loop's `paper_equation` field is populated.
4. **QuickVina 2 docking on real CrossDocked pockets** — `QuickVinaGPUAdapter` runs the official AMD build on gfx1101; `PocketDockingReward` enforces `max_evaluations` and caches failures; QuickVina -2.600 vs Vina -2.502 kcal/mol confirmed at N=1 on identical prepared receptor/ligand/box/seed.
5. **AiZynth retrosynthesis with real routes** — legacy_v3 USPTO policy + 46,695 templates + 17M-entry ZINC stock found real one-step routes for 3 smoke targets; equivalent torch AMD policy verified with ONNX/ROCm max probability difference 2.32e-6 over 28 inputs.
6. **REINVENT4 learned NLL bridge** — official prior on gfx1101: 5/5 valid molecules, CPU/GPU max NLL difference 1.05e-5; subprocess-RPC adapter with deadline enforcement.
7. **PoseBusters validity** — `mol`/`dock`/`redock` modes, real docked-poses check out; 9 redocked triple successes in `r4_click_gpu_test10_seed3_analysis`.
8. **Controlled measured search reward (latest)** — `measured_reward_control_20260913_v2` ran weight0 vs weight0.4 on test001/002 × seeds 42/0/1234: **14 generated + 6 reference physical evaluations in all 6 pairs**, identical generated sets, top-1 paired score delta 0.0. Establishes working feedback and a negative ablation, **not** an improvement in exploration or binding.

**What pieces are still skeletons, broken, or uninstrumented:**

1. **PySR symbolic regression is not the actual backend** — `_probe_pysr` returns False here; `sweep_guidance.py` explicitly forbids non-linear-descriptor priors. The completion_audit is unambiguous: "backend is linear ridge, not PySR". A symbolic-discovery loop is not what runs today.
2. **DiffDock-L / FlowDock / FLOWR / Pocket2Mol / TargetDiff / DiffSBDD are Protocol stubs** — torch_cluster/torch_scatter ROCm wheels unavailable, and SOTA pretrained checkpoints (Google Drive / Zenodo) are **not downloaded** by task policy. Cite-only path is canonical for these.
3. **Per-pocket/aggregate metal coordination, GSH and DNA measurements still need actual outputs** — `metal_hydration.py` is heuristic (Reedijk/Lippard/Hartinger defaults); no measured rate constants. Anticancer suite's "in-range / out-of-range" bands are screening proxies, not efficacy data.
4. **Configured learned synthesis remains a separate requirement** — `--synthesis-oracle` / `--synthesis-config` flags now reach the harness (R4 update), but "a configured but unmeasured provider cannot count as applied; independent chemistry generation continues with explicit failure status" (completion_audit). Synthesis rejection does **not** erase a generated product (records are preserved).
5. **Real CFG generation currently fails chemical acceptance** — `r10_cfg_real_crossdocked`: 96 requests → 64 finite atom clouds → **0 decoded molecules**. Active model work is diagnosing the gap; passing shape/loss tests does not close it.
6. **N=50 Vina vs QuickVina engine parity experiment** is not yet run — only N=1 smoke + 30 generation jobs × 3 seeds bounded diagnostic.
7. **`pic50_predictor.py`** is the legacy Attentive D-MPNN checkpoint; **`pic50_assay_audit_2026-09-13`** finds 356/2000 legacy labels used as exact IC50 despite inequality bounds, and the new `pic50_conditioned_baseline_20260913` is a HeLa48h/dark cohort replacement. The old checkpoint metrics stay historical.
8. **Full pilot, full100 evaluation, learned-baseline or paper criteria** all remain open (completion_audit goal-state table).

---

## 5. Conclusion (one paragraph)

**Lambda pipeline's real coverage today.** The `molmetal/molmetal_lam/` package is a **genuine, extensively-tested implementation** of the Molecular Lambda Calculus formalism (atoms-as-combinators, bonds-as-application, closed-terms in β-NF), with **444 test functions across 51 test files** exercising 12-tile + 220-tile click fragment pools, 5 click reactions (CuAAC/SPAAC/thiol-ene/Suzuki/amide-coupling), beta-reduction reaction rules with mass-balance verification, MCTS proof search with PUCT/Dirichlet/VirtualLoss/Transposition-Table hooks, `LamClickDesignLoop` closed-loop orchestrator, Ertl SA + QED + RDKit batched descriptors, **real Vina/QuickVina/QuickVina2-GPU docking on gfx1101**, real AiZynth retrosynthesis on isolated legacy_v3 USPTO stock, **real REINVENT4 learned NLL on gfx1101**, **real PoseBusters validity checks** on docked poses, metal geometry priors (square-planar Pt_II + generalised TODO-09 with Bondi vdW), anticancer metric suite, and synthesis cost estimation. The **end-to-end path that actually runs measured today** is: `seed tile → MCTS (linear-ridge prior) → click-reaction expansion → RDKit canonicalise → RDKit descriptors → quickvina2 docking on CrossDocked pocket → PoseBusters validate → optional REINVENT4 NLL → optional AiZynth retro check → result records`. **The pieces that are still skeletons/stubs**: PySR symbolic regression (sklearn fallback only — sweep_guidance enforces this), DiffDock/FlowDock/FLOWR/Pocket2Mol/TargetDiff/DiffSBDD (Protocol stubs gated on missing torch_cluster wheels + SOTA checkpoints), pybind11 C++/CUDA extensions (no-op stub), CFG chemical decoding (0 molecules decoded in `r10_cfg_real_crossdocked`), and the **old Attentive-DMPNN pIC50 predictor** (superseded by HeLa48h/dark cohort conditioned baseline). The completion_audit states bluntly: "**installation fixes and bounded measurements do not close the full pilot, full100 experiment, learned-baseline or paper criteria**" — the Lambda line is solidly built and measured at small N, but a 100-pocket scientific run, a measured synthesis oracle, and CFG-driven chemical decoding remain open.

---

## 6. Cross-references

- Authoritative state: `/home/hugo/codes/try_triton_on_rocm/TODO/completion_audit_2026-09-13.md`
- Round-8 closure: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round8_combined_report.md`, `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round8_recon.md`
- MCTS VirtualLoss + TT (round-6): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/molmetal_round6_done.md`
- MCTS upgrade round-1: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_eng_sota_r1.md`
- Property-test bug catch (Au_III arity): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/molmetal_lambda_elevation_r0.md`
- Synthesis cost + retrosynthesis derivations: tests `test_synthesis_cost_estimator.py` (5), `test_synthesis_derivations.py` (16)
- MCTS sweep guidance: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/sweep_guidance_development_20260913.md`
- R10 click + measured reward: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round10_final.md`, `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/measured_reward_control_20260913/`
- Lambda layer metrics history: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_layer_metrics.md`
- AiZyth isolated real backend: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/aizynth_real_backend_20260913/`
- REINVENT4 learned smoke: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/reinvent4_learned_smoke/`
- Prior ablation R10: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/round10_pt_prior_ablation.md`
- pIC50 audit: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/pic50_assay_audit_20260913/`
- pIC50 conditioned baseline: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/pic50_conditioned_baseline_20260913/`
- CfDock + FlowDock + FlowR integration: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/clone_integration_adapters.md`
- Anticancer vs general metric survey: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/anticancer_vs_general_metrics_survey.md`
