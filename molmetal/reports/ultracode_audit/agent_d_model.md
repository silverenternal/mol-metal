# Agent D — MODEL line + GPU + orchestration audit

Authoritative reference for current state: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/completion_audit_2026-09-13.md`.
Historical reports are evidence of original runs, not current behaviour. This audit is READ-ONLY: no source/config/test/TODO was modified.

Conventions: **SHIPPED** = real GPU/ROCm path runs end-to-end today. **FALLBACK-ONLY** = Triton kernel exists but production path falls back to PyTorch / cuBLAS. **SKELETON/STUB** = port satisfied with random / no-op output. **MISSING** = nothing on disk.

---

## 1. `triton_kernels/` (16 .py modules, 4863 LOC)

| File | LOC | Classification | Notes |
|---|---|---|---|
| `autotune.py` | 96 | **SHIPPED** | Canonical 9-config grid restricted to `num_warps∈{2,4,8}` / `num_stages∈{2,3,4}` per TODO/environment.md §4 (no `waves_per_eu` on gfx1101). |
| `matmul.py` | 286 | **SHIPPED** | FP32 reference matmul; 12 trimmed (BLOCK_M, BLOCK_N, BLOCK_K) tiles; stride-aware wrapper; `_check_cuda_pointers` host guard. |
| `fused_norm.py` | 289 | **SHIPPED** | LayerNorm + RMSNorm, autotune over (num_warps, num_stages). |
| `fused_rmsnorm_residual.py` | 97 | **FALLBACK-ONLY** | Triton path runs **only in inference** (`requires_grad=False`); training routes through pure-PyTorch `summed * rsqrt(...)`. Documented in the source. |
| `fused_residual_add.py` | 314 | **SHIPPED** | `out = α·x + β·residual`, broadcast-aware, autograd-wrapped. |
| `fused_dropout.py` | 313 | **SHIPPED** | Philox4x32 mask fused with residual add; standard tutorial pattern. |
| `fused_softmax.py` | 269 | **SHIPPED (CPU fallback)** | `softmax_last_dim` falls back to `torch.softmax` on CPU; GPU Triton path otherwise. |
| `fused_cross_entropy.py` | 338 | **SHIPPED (CPU fallback)** | `fused_cross_entropy` falls back to `F.cross_entropy` on CPU; indices-flavour only (FlagGems cherry-pick). |
| `fused_gelu_mlp.py` | 86 | **SHIPPED** (separate from `fused_mlp.py`) | `FusedGeluMLP` module (d_model, d_hidden); has its own Triton kernel; CPU fallback inside `forward()` when `not is_cuda`. |
| `fused_swiglu_mlp.py` | 87 | **SHIPPED** | SiLU-gated MLP Triton kernel; raises on CPU (no fallback in this wrapper). |
| `fused_mlp.py` | 826 | **FALLBACK-ONLY (production)** | Public `fused_silu_mlp` / `fused_gelu_mlp` API exists and is autograd-wrapped, but `_fused_mlp_forward` (line 663–669) **explicitly routes through `torch.nn.functional.silu`/`gelu` + `torch.matmul`** because `_fused_mlp_act_proj_kernel` has "a multi-tile correctness bug on gfx1101" (per TODO/fused_mlp_task.md). Triton kernel retained for inspection only. |
| `batched_mlp.py` | 771 | **FALLBACK-ONLY** | Public `_BatchedMLPFunction.forward` (line 378–389) uses `torch.bmm` + `F.silu` + `torch.bmm`; the fused `_batched_mlp_act_proj_kernel` is exposed but not wired into production (same rationale as `fused_mlp.py`). `batched_fused_*_per_k` expose CPU fallbacks. |
| `ode_solver.py` | 264 | **SHIPPED** | Euler step + RK4 combine; elementwise, autotuned; CPU guard. |
| `equivariant_ops.py` | 339 | **SHIPPED** | `aggregate_vectors` (scatter-sum with mask, autotuned) + `rotation_from_axis_angle` (3×3 Rodrigues, fixed `BLOCK_BATCH=1`). Both wired into EGNN coord path. |
| `config.py` | 361 | **SHIPPED** | `triton_config.should_use_fused(tensor, op=...)` dispatch gate with `TRITON_USE_FUSED` / `TRITON_USE_FUSED_EVAL` env vars, `MAX_FEAT_DIM=4096`, `MIN_BYTES=1 MB`. **Defaults: ON in training, OFF in eval.** |
| `__init__.py` | 127 | n/a | Re-exports the above. |

GPU verification: `triton_kernels/tests/test_kernel_suite.py`, `test_fused_gelu_mlp.py`, `test_fused_rmsnorm_residual.py`, `test_autotune.py` exist; MEMORY note R5 confirms `T1/T2 Triton fix` shipped (silu_mlp wire, gelu kernel). `molmetal/tests/test_round5_kernels_wired.py` is the wiring regression. Task #292 still pending: `gradcheck_gpu + silu_mlp shape` test bugs.

**Bottom line**: 11 of 14 leaf kernels are real GPU paths; **3** (fused_mlp, batched_mlp, fused_rmsnorm_residual) ship a kernel but route the public path through PyTorch. The `triton_config` gate keeps eval cheap by default.

---

## 2. `molmetal/adapters/`

| File | Classification | Notes |
|---|---|---|
| `flow_matching_lipman/__init__.py` | **SHIPPED (Lipman)** | `LipmanFlowMatchingAdapter(MoleculeGenerator)` + `EGNNVelocityField` (T5 pocket conditioning, T9 dative edges, R10-axis-C CFG + context_dropout, square-planar Pt(II) prior). Imports `triton_kernels.fused_silu_mlp` via local `_MaybeFusedSiLUMLP` (gated by `triton_config.use_fused_mlp`). Setup uses `verify_rocm_active()` + `get_device()` → ROCm/CUDA first, CPU fallback. `load_tmQM_pretrained` does best-effort shape-bridge from DMPNN→EGNN ckpt. `PocketEncoder` reuses `models.velocity_net.EGNNLayer`. |
| `egnn_predictor.py` | **SKELETON** (`EGNNPropertyPredictor`) | Per its own docstring: "This file is a **stub**. When a pretrained checkpoint is supplied ... loads weights; without one ... returns None pIC50." Training script (`scripts/train_property_predictor.py`) exists as scaffolding only. |
| `egnn_rocm.py` | **SHIPPED** | `EquivariantGraphConv` + `EGNN` with `EDGE_TYPE_SINGLE/DOUBLE/DATIVE` (T9). Imports `triton_kernels.rotation_from_axis_angle` and `fused_silu_mlp`. `_compute_coord_msg` defaults to Triton axis-angle rotation; `try/except` falls back to `dir_vec * coord_scale` on failure. `_MaybeFusedSiLUMLP` wrapper honours `triton_config`. Scatter-sum routed through `models._scatter.scatter_sum_legacy` (T1-gated backend). |
| `rdkit_predictor.py` | **SHIPPED (CPU)** | RDKit 2D descriptors + SA score. CPU only. Falls back to `1/(1+n_aromatic)` SA proxy if `sascorer` data file is missing. |
| `equibind.py` | **SKELETON/STUB** | Docstring: "Phase-1 skeleton adapter ... returns a STUB: `n_poses` copies of the molecule with small random SE(3) perturbations." Phase 2 (real EquiBind model) deferred. |
| `diffdock.py` | **SKELETON/STUB** | Same pattern as EquiBind; random poses + random vina_score. |
| `mock.py` | **SHIPPED (test-only)** | `MockGenerator`, `MockDocker`, `MockPredictor`, `MockScorer`. Deterministic via local `torch.Generator`. |
| `flow_matching_lipman/` sub-pkg | (empty `__init__.py` shown above; implementation lives in `__init__.py`) | Adapter + `PocketEncoder` + `EGNNVelocityField` + `LipmanFlowMatchingAdapter` + `load_tmQM_pretrained`. No separate `flow_matching_lipman/*.py` files beyond `__init__.py` (no `_reference/` or `_reference_loader.py` until the cloned library lands; `_reference_loader` is imported lazily). |

GPU wiring truth table:
- **Lipman FM** is the only generator port that runs on GPU today. `_MaybeFusedSiLUMLP` chooses between Triton fused-silu-mlp kernel and PyTorch `nn.Sequential` based on `triton_config.use_fused_mlp`.
- **EGNNPropertyPredictor** is a stub — never actually returns binding affinity in production; orchestrators get `None`.
- **DiffDock / EquiBind** docking adapters are stubs that do not call any model. Real docking today is **external Vina/QuickVina2-GPU** via `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` (wired in `r4_c_full_sweep.py` — see §5 below).

---

## 3. `molmetal/models/`

All `.py` present are **SHIPPED** for their advertised purpose (no stubs). No `posebusters_check` model lives here (pose validation is in `validation/posebusters_runner.py`).

| File | Classification | Notes |
|---|---|---|
| `egnn_predict.py` | SHIPPED | EGNN model module used by EGNNVelocityField + egnn_rocm. |
| `dmpnn.py` | SHIPPED | D-MPNN encoder (used by `DMPNNPropertyPredictor` and `load_tmQM_pretrained` source). |
| `metal_hybrid{,_v2,_v3,_v4}.py` | SHIPPED (CPU-fallback-friendly) | Metal-binding hybrid heads used as training targets, not inference runtime. |
| `cross_attention_fusion{,_v3}.py`, `fusion.py`, `loss.py`, `_scatter.py` | SHIPPED | Backbone utilities. `_scatter.py` exposes `scatter_sum_legacy` + a backend selector (T1) so RDNA3 at small shapes uses `torch_scatter` fallback rather than the Triton kernel. |

Triton usage inside `models/` is mediated through `triton_kernels.config.triton_config` (see `fused_silu_mlp` wiring).

---

## 4. `molmetal/ports/` + `ports/generators.py`

| File | Classification | Notes |
|---|---|---|
| `__init__.py` | **SHIPPED** | Five PEP-544 Protocols: `MoleculeGenerator`, `DockingEngine`, `PropertyPredictor`, `ScoringFunction`, `DesignLoop`. Frozen dataclasses `GenerationConfig`, `DockingConfig`, `PropertyPrediction`, `ScoredCandidate`, `DesignLoopConfig`. |
| `generators.py` | **SHIPPED** | `MetalLigandGenerator` Protocol + `MetalLigandConfig` (oxidation state, embed_3d flag, extra_metadata). Targets the metal-complex multi-component `L1.L2....Ln.[M]` SMILES reconstruction. Real implementation lives at `molmetal/molmetal_lam/sbdd_env/metal_generator_adapter.py` (not in `molmetal/adapters/`). |
| **MISSING ports** | `docking.py`, `predictor.py`, `scorer.py`, `design_loop.py`, `validator.py`, `retrosynthesis_checker.py` | None of these exist as separate files in `molmetal/ports/`. The task brief listed them but only `__init__.py` and `generators.py` are present. The corresponding **adapters** for docking/synthesis/predictor live under `molmetal/molmetal_lam/sbdd_env/` (Lambda side), not under `molmetal/`. |

---

## 5. `molmetal/scripts/` (orchestration scripts)

### `r4_c_full_sweep.py` (603 LOC) — the canonical end-to-end driver
**Classification: SHIPPED, Lambda-only pipeline; the model line (Triton+EGNN) is not on the hot path.**

Top-level structure (lines 1–80):
- Docstring declares the script is a "Lambda search on explicit CrossDocked test receptor/ligand pairs"; "no SOTA-equivalence is claimed"; `--dry-run` validates without generating.
- `PocketResult` dataclass records `top1_smiles`, `top1_sa`, `top1_qed`, `top1_lipinski`, `top1_vina_proxy`, `n_candidates`, `physical` (separate from search proxies).
- `load_manifest` validates receptor/ligand pair identity + files; `file_digest` produces SHA-256; `runtime_fingerprints` fingerprints `molmetal/molmetal_lam`, `molmetal/domain`, `molmetal/validation` plus the orchestrator scripts.

Per-pocket runner (lines 128–169): `run_one_pocket` **delegates to `lambda_100pocket_sweep.run_one_pocket`** (the Lambda MCTS script). It does **not** import `LipmanFlowMatchingAdapter`, `EGNNVelocityField`, or any Triton kernel.

CLI surface (lines 415+):
- `--search-docking-engine {auto,vina,quickvina2,quickvina2-gpu}` and `--physical-engine {auto,vina,quickvina2,quickvina2-gpu}` (default `auto`).
- `--gpu-config` (JSON) selects QuickVina2-GPU-2.1 binary path; **default config is `molmetal/configs/amd_gpu_docking.json`** pointing at `/mnt/storage/tools/vina_gpu21_source/.../QuickVina2-GPU-2-1`.
- `--physical-docking` toggles actual Vina/QuickVina2-GPU rerun after search; `--symbolic-prior`, `--prior-state`, `--synthesis-oracle`, `--synthesis-config` wire the Lambda guidance channels.
- `manifest`, `n_pockets`, `seeds`, `job_timeout`, `output_prefix` plus `--dry-run`, `--append` (resume).

Pre-flight + execution loop (lines 444–599):
1. Loads `sota_aligned_targetdiff.yaml` + projected lambdas; builds the `search` dict.
2. Hashes `vina_adapter.py`, `posebusters_runner.py`, `prepare_crossdocked_receptor.py`, `receptor_preparation_for_evaluation.py` into `metadata["physical_implementation_sha256"]` so any in-flight change invalidates the run.
3. Iterates `jobs = [(pocket, seed) for pocket in pairs for seed in args.seeds]`; each job calls `execute_job` → `run_one_pocket` → `lambda_100pocket_sweep.run_one_pocket`.
4. Aggregates per-pocket rows into `.csv`, `.json`, `.md`; supports resume via SHA-pinned metadata.

**What `r4_c_full_sweep.py` actually does end-to-end**:
1. Resolve a test manifest of (receptor, ligand) pairs from `molmetal/data/crossdocked100_manifest.csv` (or a user-supplied CSV); SHA-256 every input file.
2. For each pocket × seed: invoke `lambda_100pocket_sweep.run_one_pocket` — this is **Lambda MCTS over a 204-tile fragment library + 5 click rules** (NOT the Lipman FM generator). It uses RDKit + Meeko for SMILES embedding and Vina/QuickVina2-GPU for the docking reward channel (when `--search-docking-reward` is on).
3. Optionally rerun physical docking + PoseBusters (`--physical-docking`) via `molmetal/scripts/evaluate_generated_poses.py` + `molmetal/validation/posebusters_runner.py`.
4. Emit per-pocket CSV/JSON/MD reports.

**No Triton, no EGNN, no LipmanFlowMatchingAdapter is invoked by `r4_c_full_sweep.py` today.** Per the Lambda-side sweep docstring, "Triton kernel not exercised here (Lambda MCTS is Python); ROCm 7.2 + Triton 3.8 untouched." CPU is the Lambda hot path; GPU is the docking reward + physical-docking binary only.

### Other scripts (selected, READ-ONLY inventory)

| Script | LOC-class hint | Role |
|---|---|---|
| `lambda_100pocket_sweep.py` | ~830 | Lambda MCTS over fragment library + click rules; per-pocket runner imported by `r4_c_full_sweep.py`. CPU-only. |
| `close_loop_{1,2,3}_*.py` | — | Closed-loop design experiments (Lambda side). |
| `train_metal_hybrid{,_v4}.py`, `train_property_predictor.py`, `train_dmpnn_multitask.py` | — | Training scripts; not exercised by the r4_c sweep. |
| `evaluate_generated_poses.py`, `prepare_crossdocked_receptor.py`, `receptor_preparation_for_evaluation.py` | — | Physical-eval pipeline. |
| `r10_*.py`, `r10_cfg_*.py`, `r10_ot_*.py`, `r10_tmqm_*.py`, `r10_pt_prior_*.py` | — | Round-10 axis ablations (Triton kernel fixes, OT, CFG, metal prior on/off). Used for benchmarking kernels, NOT for the sweep driver. |
| `bench_rocm_throughput.py`, `verify_amd_eval_capabilities.py`, `verify_docking_adapter_smoke.py`, `smoke_vina_gpu_generated.py`, `smoke_test.py` | — | Smoke + benchmark scripts. |
| `r3_drugood_benchmark.py`, `r10_ot_qm9_3seed.py`, `r10_ot_qm9_convergence.py` | — | SOTA-comparison benchmarks (cited-only path per R4 / R10 final notes). |
| `aizynth_*.py`, `verify_aizynth_*.py`, `verify_reinvent_*.py`, `verify_reinvent_learned_inference.py` | — | AiZynthFinder + REINVENT4 retrofit wrappers (round-7 install ladder). |
| `pretrain_coordination.py`, `finetune_tmqm_metacytotox.py`, `train_fm_cytotox.py`, `train_fm_pocket.py` | — | EGNN/FM training scripts (off the r4_c sweep path). |

---

## 6. `molmetal/configs/`

| File | Classification | Notes |
|---|---|---|
| `sota_aligned_targetdiff.yaml` | **SHIPPED** | Canonical TargetDiff-style protocol: `test_set=crossdocked_pocket10`, `n_test_pockets=100`, `engine=vina 1.2.7`, `exhaustiveness=8`, `n_poses=9`, SA/QED/Lipinski thresholds, PoseBusters `pass_all`, MCTS `top_k=100 / max_depth=3 / branching_target=1020`, Lambda knobs (`extended_204`, `all_5` click rules). This is the file `r4_c_full_sweep.py` reads by default. |
| `amd_gpu_docking.json` | **SHIPPED** | QuickVina2-GPU-2.1 binary path + `gpu_threads=1000`, `search_depth=1`, `opencl_device=opencl:0`, `expected_device=gfx1101`. |
| `aizynth_legacy_v3_cpu.yml`, `aizynth_legacy_v3_rocm.yml` | **SHIPPED** | AiZynthFinder config variants (retro-synthesis oracle wiring). |
| `reinvent_prior_amd.json` | **SHIPPED** | REINVENT4 prior config (round-7 install ladder). |
| `lambda_click_physical_smoke.yaml` | **SHIPPED** | Smaller smoke config for the closed-loop pipeline. |

---

## 7. `molmetal/tests/` (selected — task brief)

| Test | What it asserts |
|---|---|
| `test_r4c_full_sweep.py`, `test_r4c_pair_execution.py` | Driver + per-pocket runner contracts. |
| `test_validate_generate_metrics.py` | Metric aggregation. |
| `test_qvina_swap.py`, `test_vina_adapter.py`, `test_vina_gpu_adapter.py`, `test_docking_adapters.py`, `test_docking_pose_chemistry.py` | Vina/QuickVina2 / Vina-GPU adapter behaviour + pose-chemistry. |
| `test_posebusters_adapter.py`, `test_posebusters_doc.py` | PoseBusters outer-gate runner. |
| `test_round5_kernels_wired.py`, `test_fused_silu_mlp_wired.py`, `test_dmpnn_scatter_wiring.py`, `test_egnn_coord_scatter_wired.py`, `test_scatter_backend_selector.py`, `test_layer_metrics_l{1,4,7,9}*.py` | Kernel/scatter/coordinate wiring regressions. |
| `test_rocm_lipman.py`, `test_rocm_throughput.py`, `test_round11_engine_parity.py` | ROCm-device smoke tests for Lipman FM + throughput parity. |
| `test_minibatch_ot.py`, `test_real_qm9_ot_protocol.py`, `test_ot_effective_backend.py` | OT/mini-batch OT coupling (SOTA-comparison lane). |
| `test_lipman_*.py` (5 files), `test_pocket_conditioned_lipman.py`, `test_egnn_velocity_cfg.py`, `test_atom_training_contract.py` | EGNN/FM contract tests. |
| `test_pic50_predictor.py`, `test_posebusters_adapter.py`, `test_reinvent_prior_adapter.py`, `test_aizynth_*.py` (4 files), `test_clone_integration_adapters.py` | Property / pose / synthesis clone adapters. |
| `test_design_loop.py`, `test_explicit_metal_prior_sampling.py`, `test_lambda_measured_reward_dispatch.py`, `test_pocket_docking_reward.py`, `test_persistent_mcts_tree.py` | Loop orchestration + Lambda integration. |
| 30+ others (`test_hybrid*`, `test_metal_hybrid*`, `test_metal_smiles`, `test_metal_generator_adapter`, `test_dmpnn*`, `test_cross_attention*`, `test_gpu_molecular_metrics`, `test_metalloprotein_targets`, `test_mmp_targets`, `test_physical_sweep_reporting`, `test_synflownet_env`, `test_syntemol_reactions`, `test_sa_score`, `test_retrosynthesis`, `test_tmqm_*`) | Module-level coverage. |

MEMORY/R4 reports: 158 tests pass / 1 skip / 1 xfail across 21 test files; the round-5 wiring and round-8 retro tests are green.

---

## 8. `molmetal/validation/`

| File | Classification | Notes |
|---|---|---|
| `posebusters_runner.py` | **SHIPPED (graceful skip)** | `posebusters_available()` probe + `check_posebusters(smiles)`, `check_docked_pose(mol, receptor_path)`, `check_posebusters_batch(list)`, `pb_pass_rate(results)`. Per-call dict return; never raises when `posebusters` package missing. |
| `admet_runner.py` | **SHIPPED (3-tier fallback)** | `predict_admet(smiles) → dict`; backend chain `admet-ai` → `datamol+molfeat` → `RDKit-only`. Memoised at module level. Layer-8 reward channel for `RewardAggregator`. |
| `gpu_molecular_metrics.py` | **SHIPPED** | GPU-backed molecular metrics (used by orchestrator + tests). |
| `training_novelty.py` | **SHIPPED** | Training-set novelty metric. |

---

## 9. End-to-end pipeline that `r4_c_full_sweep.py` actually runs today

Per the Lambda MCTS sweep docstring + the orchestrator wiring:

1. **Pocket discovery** — `load_manifest(crossdocked100_manifest.csv)` → list of (pocket_id, receptor_path, ligand_path) triples; SHA-256 every file.
2. **Per pocket × seed** — call `lambda_100pocket_sweep.run_one_pocket`, which runs **Lambda MCTS** (CPU) over a 204-tile fragment library + 5 click-rule reactions. Optional symbolic prior, optional retro-synthesis oracle (`aizynth_legacy_v3_cpu.yml` / `_rocm.yml`), optional `--search-docking-reward` calling Vina 1.2.x or **QuickVina2-GPU-2.1** (`molmetal/configs/amd_gpu_docking.json`) as the docking reward channel.
3. **Optional physical eval** — when `--physical-docking` is set, `molmetal/scripts/evaluate_generated_poses.py` re-docks with the same GPU/CPU Vina config + runs `molmetal/validation/posebusters_runner.py:check_posebusters` / `pb_pass_rate`.
4. **Reporting** — `r4_c_full_sweep.py` emits `.csv`, `.json`, `.md` per pocket + summary; SHA-pinned metadata guards against in-flight drift.

**The model line (Triton kernels + LipmanFlowMatchingAdapter / EGNNVelocityField) is NOT on the `r4_c_full_sweep.py` hot path.** The Lipman FM adapter exists and is GPU-capable (`fused_silu_mlp` + axis-angle coord update + CFG + metal-prior), but the r4_c sweep uses Lambda MCTS only. The EGNN-based `EGNNPropertyPredictor` remains a stub (returns `None` binding affinity in production); docking comes from external Vina/QuickVina2-GPU. Pose validity comes from RDKit+MMFF94 + the `posebusters` package (or graceful skip).

**GPU paths that genuinely run today** (not just "wired"):
- QuickVina2-GPU-2.1 binary (when `molmetal/configs/amd_gpu_docking.json` is reachable).
- Triton kernels used by `LipmanFlowMatchingAdapter` and `egnn_rocm` (`rotation_from_axis_angle`, `fused_silu_mlp` via `_MaybeFusedSiLUMLP` + `triton_config` gate; `_compute_coord_msg` axis-angle rotation).
- ROCm-resident `EGNNVelocityField` / `PocketEncoder` if a caller invokes `LipmanFlowMatchingAdapter.train_step()` or `generate()` — exercised by `test_rocm_lipman.py` and `test_rocm_throughput.py`, **not** by `r4_c_full_sweep.py`.
- `_scatter.scatter_sum_legacy` Triton path (when `models._scatter._backend_for()` selects it; RDNA3 small-shape fallback to `torch_scatter`/`index_add_` is automatic).

**CPU-fallback paths that dominate `r4_c_full_sweep.py`**:
- Lambda MCTS Python loop.
- Lipman FM `fused_mlp` + `batched_mlp` production path (Triton kernel exists but is unsafe on gfx1101 multi-tile → routed through `torch.matmul`/`torch.bmm`).
- `fused_rmsnorm_residual` training path (Triton kept for inference only).
- RDKit descriptor predictors + SA score.

**Missing / skeleton ports not used by r4_c**: real DiffDock, real EquiBind (Phase-2 deferred), EGNN pIC50 predictor (stub), `molmetal/ports/{docking,predictor,scorer,design_loop,validator,retrosynthesis_checker}.py` (don't exist as separate files — the corresponding protocols are all in `molmetal/ports/__init__.py` and their adapters live under `molmetal/molmetal_lam/sbdd_env/`).

---

## 10. 1-paragraph synthesis

**Coverage:** The model line is a SHIPPED Triton kernel library (11 real GPU kernels; 3 kernels kept on disk but routed through PyTorch due to gfx1101 multi-tile correctness bugs) plus a SHIPPED Lipman-FM adapter (`LipmanFlowMatchingAdapter` + `EGNNVelocityField` + `PocketEncoder`) that exercises `fused_silu_mlp` and `rotation_from_axis_angle` on ROCm via `_MaybeFusedSiLUMLP` and `_compute_coord_msg`. Two docking adapters (DiffDock, EquiBind) and one binding-affinity predictor (EGNN) are STUBS. The orchestrator `r4_c_full_sweep.py` (603 LOC) is a Lambda-side driver that loads `sota_aligned_targetdiff.yaml`, walks the test manifest, and per pocket calls `lambda_100pocket_sweep.run_one_pocket` (CPU Lambda MCTS over 204-tile + 5 click rules); the optional `--physical-docking` path reruns docking via `molmetal/configs/amd_gpu_docking.json` (QuickVina2-GPU-2.1 binary) and `molmetal/validation/posebusters_runner.py`. **Today's end-to-end r4_c sweep pipeline is: SHA-pinned manifest → Lambda MCTS (CPU) → optional Vina/QuickVina2-GPU reward + physical Vina + PoseBusters gate → CSV/JSON/MD reports — Triton kernels and Lipman FM are not invoked.** The GPU pieces ship and are test-covered (`test_rocm_lipman.py`, `test_round5_kernels_wired.py`, `test_round11_engine_parity.py`), but the round-4 sweep canonically runs Lambda on CPU and uses QuickVina2-GPU only as the docking reward / physical-eval binary.
