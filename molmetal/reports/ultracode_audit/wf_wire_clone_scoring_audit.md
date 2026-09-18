# WF-Wire-Clone-Scoring: Adapter + Clone-Repo Audit

**Audit date:** 2026-09-14
**Workflow:** WF-Wire-Clone-Scoring (audit phase)
**Environment:** uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64
**Auditor scope:** `molmetal/molmetal_lam/sbdd_env/` adapter files + `molmetal/references/` clone inference scripts
**Honest framing:** Only adapters with working code paths are marked `REAL`; stub-only, protocol-shape and `NotImplementedError` adapters are marked `STUB`. Status is independent of whether the *underlying external model weights* are on disk (a `REAL` adapter can still need external assets to actually run inference).

---

## 1. Adapter × Status Matrix

| # | Adapter file | Class / public API | REAL / STUB | Integrated into `r4_c_full_sweep.py`? | Notes |
|---|---|---|---|---|---|
| 1 | `aizynth_adapter.py` | `AiZynthAdapter.check(smiles)` | **REAL** (with SMARTS fallback always) | YES — via `--synthesis-oracle aizynthfinder` / `aizynthfinder_isolated` (`r4_c_full_sweep.py:410`) | Lazy-imports `aizynthfinder`; falls back to SMARTS if assets missing |
| 2 | `aizynth_isolated.py` | `IsolatedAiZynthChecker.check_many` | **REAL** (subprocess-RPC bridge) | YES — via `--synthesis-oracle aizynthfinder_isolated` (`r4_c_full_sweep.py:410`) | Process-group managed, JSON-RPC over stdin/stdout |
| 3 | `aizynth_rocm_policy.py` | `RemoteTorchPolicy.predict` / `TorchRemoteExpansionStrategy` | **REAL** (subprocess ROCm policy bridge; only loadable inside AiZynth env) | PARTIAL — usable only inside isolated aizynth subprocess; not directly wired into `r4_c_full_sweep` | Persistent ROCm worker; SHA-256 + arch-checked |
| 4 | `aizynth_torch_model.py` | `TorchDensePolicy.predict` | **REAL** (legacy Dense net, ROCm-torch native) | PARTIAL — required by `aizynth_rocm_policy.py`; never directly invoked from `r4_c_full_sweep` | Used as predictor behind `TorchRemoteExpansionStrategy` |
| 5 | `diffdock_adapter.py` | `DiffDockAdapter.dock` / `is_available` | **REAL** (subprocess wrapper, raises `AdapterUnavailable` if no backend) | NO — has `is_available()` but `r4_c_full_sweep` does not call it | Discovery: `$DIFFDOCK_BIN` → PATH → vendored repo → module |
| 6 | `flowdock_adapter.py` | `FlowDockAdapter.dock` + `FlowDockReferenceAdapter.dock` (stub) | **REAL** (FlowDockAdapter) + **STUB** (FlowDockReferenceAdapter) | NO — `FlowDockReferenceAdapter` is intentionally stub; live adapter unused | Two classes in one file; only the first is REAL |
| 7 | `flowr_adapter.py` | `FLOWRReferenceAdapter.generate` | **STUB** (returns fallback_smiles: ethanol/triethylamine/benzene) | NO | No live inference path |
| 8 | `metal_generator_adapter.py` | `MetalLigandAdapter.generate` | **REAL** (RDKit ETKDGv3 + MMFF94 / UFF; multi-component SMILES reconstruction) | NO — used internally by the Lambda search, not directly in `r4_c_full_sweep` | Real 3D embed, fallback to placeholder |
| 9 | `pic50_predictor.py` | `predict_pic50(smiles)` / `AttentiveDMPNNPredictor` | **REAL** (D-MPNN checkpoint) | NO — only used in test_harness scripts; `r4_c_full_sweep` does not import | Requires `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` |
| 10 | `pocket2mol_adapter.py` | `Pocket2MolAdapter.sample` | **REAL** (live branch) + **REAL** (SMARTS fallback branch) | NO — not wired into `r4_c_full_sweep` | Live branch gated by `pretrained_Pocket2Mol.pt`; fallback uses built-in 120-SMILES pool |
| 11 | `pocket_docking_reward.py` | `PocketDockingReward.__call__` | **REAL** (Vina/QVina/QuickVina2/QuickVina2-GPU; boxed search reward) | YES — via `--search-docking-reward --search-docking-engine` (`r4_c_full_sweep.py:413-415`) | Cached + budget-aware |
| 12 | `posebusters_adapter.py` | `PoseBustersAdapter.validate_mol` / `pass_rate` | **REAL** (RDKit + posebusters Python API; MMFF94 → UFF fallback) | NO direct flag — invoked indirectly by `lambda_100pocket_sweep` + `evaluate_generated_poses.py` (`r4_c_full_sweep.py:377`) | Requires `posebusters` package; metrics gated to bool-dtype check columns |
| 13 | `pybind_adapter.py` | `Pybind11StubAdapter.forward` | **STUB** (no-op stub; `forward()` returns `None`) | NO | Documents pybind11 surface but never links a `.so` |
| 14 | `qed_scorer.py` | `QEDScorer.score` / `qed_from_smiles` | **REAL** (RDKit `QED.qed` with NaN fallback) | NO — used inside `lambda_100pocket_sweep` metrics path | Batched, optional SA via `sa_score` |
| 15 | `reinvent4_adapter.py` | `REINVENT4Adapter.score` | **STUB** (always returns RDKit-fallback scores; `is_stub() → True`) | NO | Discovers TOML presets but never uses them |
| 16 | `reinvent4_jsonl_worker.py` | `handle(request)` JSON-RPC | **REAL** (RDKit-only protocol bridge) | NO — used in test infra, not wired into `r4_c_full_sweep` | Single-process deterministic |
| 17 | `reinvent4_multiproperty_jsonl_worker.py` | `handle(request)` JSON-RPC multi-property | **REAL** (drives `reinvent` CLI subprocess) | NO — runs inside isolated REINVENT4 ROCm project venv | Real TOML-driven multiproperty scoring |
| 18 | `reinvent4_subprocess_adapter.py` | `REINVENT4Adapter` + `REINVENT4MultipropertyAdapter` (JSON-RPC clients) | **REAL** (process management + heartbeat + fallback to `0.0`) | YES — wired into `RewardAggregator.r_reinvent4` channel per WF-Extra-2 (per memory `WF-Extra-2`); `r4_c_full_sweep` invokes through `lambda_100pocket_sweep` | Multiproperty mode wired via `molmetal/configs/reinvent_multiproperty_amd.json` |
| 19 | `reinvent_prior_adapter.py` | `REINVENT4PriorAdapter.likelihood` (batch NLL with deadline) | **REAL** (subprocess JSON-RPC + arch-validated, SHA256 + `gfx1101` check) | NO — used in separate prior-NLL harness; not in `r4_c_full_sweep` | Total-deadline lock incl. startup |
| 20 | `reinvent_prior_jsonl_worker.py` | `PriorWorker.likelihood` JSON-RPC server | **REAL** (real REINVENT prior NLL via `reinvent.runmodes.create_adapter`) | NO — only importable inside isolated REINVENT env | Batched per-token-bucket; ROCm/CPU explicit |
| 21 | `reinvent_wrapper.py` | `REINVENT4Scorer.score` (4-component weighted) | **REAL** (RDKit fallback; binding = rot-bond proxy) | YES — used inside `lambda_100pocket_sweep` (default scoring) | Multi-property fallback |
| 22 | `retrosynthesis.py` | `retrosynthesize` / `synthesis_success_rate` | **REAL** (RDKit reverse-SMARTS over CuAAC/SPAAC/SPC/DA/ThiolEne; AiZynth optional) | YES — invoked through `synthesis_oracle` channel (mirrors `--synthesis-oracle`) | RDKit primary; AiZynth preferred if config present |
| 23 | `sa_score.py` | `sa_score_ertl` / `sa_score_to_unit` | **REAL** (Ertl sascorer from RDKit Contrib) | YES — invoked through `lambda_100pocket_sweep` | Cached `lru_cache(4096)` |
| 24 | `synflownet_env.py` | `SynFlowNetEnvAdapter.forward_step` / `backward_step` | **REAL** (vendored RDKit-only copy of `Reaction`; SMARTS swap) | NO — used by `beta_reductions` directly | No torch_geometric dependency |
| 25 | `syntemol_reactions.py` | `SyntheMolReactionRule.fire` / `from_syntemol_to_lambda` | **REAL** (bypasses SyntheMol `__init__` to avoid wandb/chemprop; uses RDKit reaction SMARTS) | NO — used internally by Lambda search synthesis layer | Loaded via `importlib.util` |
| 26 | `synthesis_gate.py` | `build_synthesis_gate` / `gate_candidates` | **REAL** (high-level wrapper around aizynth_adapter) | YES — exposes `request=True / "smarts" / "aizynthfinder_isolated"` to caller | Stateless functional API |
| 27 | `targetdiff_adapter.py` | `TargetDiffAdapter.generate` / `DiffSBDDAdapter.generate` | **STUB** (raises `CheckpointUnavailableError`; no weights, no PyG) | NO | Only published_metrics cite-only path works |
| 28 | `vina_adapter.py` | `VinaDockingAdapter.dock` | **REAL** (`vina` Python binding + meeko + qvina/quickvina2 CLI) | YES — used by `pocket_docking_reward.py` and `physical-docking --physical-engine` | Default engine = `vina` |
| 29 | `vina_gpu_adapter.py` | `QuickVinaGPUAdapter.dock` | **REAL** (OpenCL kernel trace verification + receptor-prep dependency) | YES — via `--physical-engine quickvina2-gpu` and `--search-docking-engine quickvina2-gpu` (`r4_c_full_sweep.py:421`) | Hard-coded `opencl:0`, no CPU fallback |
| 30 | `voxelization.py` | `pocket_to_spatial_tiles` / `SpatialTile` | **REAL** (NumPy voxelization) | NO — used internally by `binding` layer | Element-aware channels |

**Totals (over 30 adapter modules / 31 classes if counting both FlowDock classes):**

| Metric | Count |
|---|---|
| Total adapter modules | 30 |
| REAL (working code path) | 24 |
| STUB (protocol-shape / NotImplementedError / no-op / fake data) | 6 |
| INTEGRATED (visible to `r4_c_full_sweep.py` via CLI flag or indirectly through `lambda_100pocket_sweep`) | 11 |
| NOT_INTEGRATED (exists in `sbdd_env/` but never invoked by `r4_c_full_sweep.py`) | 19 |

The 6 STUB adapters and their missing capability:

| Adapter | Missing capability |
|---|---|
| `flowr_adapter.py` (`FLOWRReferenceAdapter`) | Real SE(3)-equivariant flow sampling (torch_cluster + flowr package not loaded) |
| `pybind_adapter.py` (`Pybind11StubAdapter`) | A real `.so` of `equibind_geodesic_kernel` / diffdock sphere sampler |
| `reinvent4_adapter.py` (`REINVENT4Adapter`) | REINVENT4 heavy RL loop + QSAR/docking/shape/RAscore plugins |
| `targetdiff_adapter.py` (`TargetDiffAdapter` + `DiffSBDDAdapter`) | Pretrained `.pt`/`ckpt` weights + `torch_geometric` + `torch_scatter` ROCm wheels |
| `flowdock_adapter.py` (`FlowDockReferenceAdapter`) | Same as `flowr_adapter.py`: torch_cluster + real flowdock package |

> The 11 INTEGRATED adapters (REAL+stubs path-through via `--synthesis-oracle`, `--search-docking-reward`, `--search-docking-engine`, `--physical-engine`, indirect via `lambda_100pocket_sweep`):
> `aizynth_adapter`, `aizynth_isolated`, `pocket_docking_reward`, `posebusters_adapter`, `qed_scorer`, `reinvent4_subprocess_adapter`, `reinvent_wrapper`, `retrosynthesis`, `sa_score`, `synthesis_gate`, `vina_adapter`, `vina_gpu_adapter`.
>
> The remaining 19 NOT_INTEGRATED adapters are REAL but not directly callable from `r4_c_full_sweep` CLI: `aizynth_rocm_policy`, `aizynth_torch_model`, `diffdock_adapter` (live branch), `flowdock_adapter` (live branch), `metal_generator_adapter`, `pic50_predictor`, `pocket2mol_adapter`, `pybind_adapter` (stub), `reinvent4_jsonl_worker` (test-only), `reinvent4_multiproperty_jsonl_worker`, `reinvent4_adapter` (stub), `reinvent_prior_adapter`, `reinvent_prior_jsonl_worker`, `synflownet_env`, `syntemol_reactions`, `targetdiff_adapter` (stub), `voxelization`, `flowr_adapter` (stub).

---

## 2. Clone-Repo Inference Scripts Audit

### 2.1 DiffDock (`molmetal/references/DiffDock/`)

**Script:** `inference.py`
**CLI:** `python -m inference --protein_path <pdb> --ligand <smiles|file> --out_dir <tmp> [--samples_per_complex 10] [--inference_steps 20] [--batch_size 10] [--model_dir <dir>] [--ckpt best_ema_inference_epoch_model.pt]`
**GPU requirement:** CUDA + `torch_geometric` (Heavy). ROCm wheels absent for `torch_cluster`/`torch_scatter`; inferred blocked on ROCm.
**Checkpoint paths expected:** `diffdock_models/` (downloaded from `REPOSITORY_URL/releases/latest/download/diffdock_models.zip`). Two checkpoints required: `best_ema_inference_epoch_model.pt` (score model) + `best_model.pt` (confidence model).
**Adapter coverage:** `DiffDockAdapter.dock()` → matches CLI shape; `is_available()` returns True if `$DIFFDOCK_BIN`, vendored repo dir, or importable module.
**Status:** READY-TO-USE (script + signature), blocked by external weights + ROCm torch_cluster.

### 2.2 TargetDiff (`molmetal/references/targetdiff/`)

**Script:** `scripts/sample_for_pocket.py` (not `inference.py`; uses `score_diffusion_ligand` via `molopt_score_model.ScorePosNet3D`)
**CLI:** `python scripts/sample_for_pocket.py <config.yml> --pdb_path <pdb> --device cuda:0 --batch_size 100 --result_path ./outputs_pdb --num_samples <N>`
**GPU requirement:** CUDA + `torch_geometric` + `torch_scatter`. ROCm wheels absent.
**Checkpoint paths expected:** `pretrained_models/pretrained_diffusion.pt` (~250 MB, behind Google Drive folder 1-ftaIrTXjWFhw3-0Twkrs5m0yX6DaCNarz).
**Adapter coverage:** `TargetDiffAdapter.generate()` maps to this CLI; raises `CheckpointUnavailableError` if ckpt absent.
**Status:** READY-TO-USE (script + signature), blocked by external weights + ROCm torch_scatter.

### 2.3 Pocket2Mol (`molmetal/references/Pocket2Mol/`)

**Script:** `sample_for_pdb.py` (uses `MaskFillModelVN`, `AtomComposer`, `get_init`, `get_next`, `reconstruct_from_generated_with_edges`)
**CLI:** `python sample_for_pdb.py --pdb_path <pdb> --center <x,y,z> --bbox_size 23.0 --config ./configs/sample_for_pdb.yml --device cuda --outdir ./outputs`
**GPU requirement:** CUDA + `torch_geometric`. ROCm wheels absent.
**Checkpoint paths expected:** `ckpt/pretrained_Pocket2Mol.pt` (~165 MB, Google Drive).
**Adapter coverage:** `_Pocket2MolLiveAdapter.sample()` invokes this CLI; `_Pocket2MolFallbackAdapter` uses 120-SMILES pool when ckpt missing.
**Status:** READY-TO-USE (script + signature), blocked by external weights + ROCm torch_geometric.

### 2.4 FLOWR (`molmetal/references/FLOWR/`)

**Script:** `flowr/train.py` (no top-level `run.py`); `flowr_root/` mirror contains `genbench3d/`, `posecheck/` for evaluation.
**CLI:** Not a single `inference.py`; FLOWR exposes `gflownet` MCTS + flowr package; intended usage is `python flowr/train.py` + downstream evaluation. SLURM scripts `scripts/gen_pdb.sl`, `scripts/eval_spindr.sh`.
**GPU requirement:** CUDA + `torch_cluster` + `torch_scatter`. ROCm wheels absent.
**Checkpoint paths expected:** None shipped in repo (paper uses CrossDocked2020 fine-tune).
**Adapter coverage:** `FLOWRReferenceAdapter` (STUB) returns ethanol/triethylamine/benzene fallback_smiles. No live path.
**Status:** NO READY-TO-USE INFERENCE SCRIPT. Adapter is permanently stub.

### 2.5 BioLM-Score (`molmetal/references/BioLM-Score/`)

**Scripts:** `BioLM_Score/model/model4.py` (inference module), `scripts/casf2016_*.py` (CASF-2016 benchmarking: docking, scoring, ranking, screening), `scripts/train_model.py`, `chemformer/molbart/inference_score.py`.
**CLI:** `python BioLM_Score/model/model4.py --input <feats.csv> --model_path <dir>` for inference; `scripts/casf2016_docking.py` for CASF benchmarking.
**GPU requirement:** PyTorch (CPU possible). ROCm compatible.
**Checkpoint paths expected:** `BioLM_Score/model/pretrained/` not shipped.
**Adapter coverage:** NO adapter in `molmetal/molmetal_lam/sbdd_env/` (no `biolm_*_adapter.py` file). Untouched.
**Status:** NO ADAPTER. Repository script usable but unwired.

### 2.6 EquiBind (`molmetal/references/EquiBind/`)

**Script:** `inference.py` + `multiligand_inference.py`
**GPU requirement:** PyTorch + custom CUDA geodesic kernels. ROCm wheel absent.
**Adapter coverage:** NO dedicated adapter. Closest is `pybind_adapter.Pybind11StubAdapter` (STUB).
**Status:** NO ADAPTER. Repo clones but never imported by molmetal.

### 2.7 TankBind (`molmetal/references/TankBind/`)

**Script:** No standalone `inference.py`; `tankbind/` package + `examples/` jupyter notebooks.
**Adapter coverage:** NO adapter.
**Status:** NO ADAPTER.

### 2.8 RxnFlow (`molmetal/references/RxnFlow/`)

**Script:** `scripts/opt_unidock.py`, `scripts/opt_seh.py`, `scripts/sampling_unidock.py`, `scripts/pretrain_qed.py`.
**GPU requirement:** PyTorch + `torch_geometric`.
**Adapter coverage:** NO dedicated adapter.
**Status:** NO ADAPTER.

### 2.9 SynFlowNet (`molmetal/references/SynFlowNet/`)

**Script:** `src/synflownet/` package; no top-level `run.py`.
**Adapter coverage:** `synflownet_env.py` REAL adapter (vendored RDKit-only Reaction).
**Status:** ADAPTER READY (vendored copy of `Reaction`); clone repo usable as reference only.

### 2.10 SyntheMol (`molmetal/references/SyntheMol/`)

**Script:** No top-level `run.py`; `synthemol/generate.py`, `synthemol/models.py` (heavy wandb/chemprop deps).
**Adapter coverage:** `syntemol_reactions.py` REAL adapter (uses `importlib.util` to bypass `synthemol.__init__`).
**Status:** ADAPTER READY.

### 2.11 REINVENT4 (`molmetal/references/REINVENT4/`)

**Script:** `reinvent/Reinvent.py` (TOML-driven RL CLI); `configs/*.toml` presets.
**Adapter coverage:** `reinvent4_adapter.py` (STUB), `reinvent4_subprocess_adapter.py` (REAL, subprocess-RPC), `reinvent4_jsonl_worker.py` (REAL, RDKit-only), `reinvent4_multiproperty_jsonl_worker.py` (REAL, drives real `reinvent` CLI), `reinvent_prior_adapter.py` (REAL, prior-NLL bridge).
**Status:** ADAPTERS READY (4 of 5 REAL).

---

## 3. r4_c_full_sweep.py — No `--use-X` Flags

Grep over `molmetal/scripts/r4_c_full_sweep.py` for `--use-*`, `--posebusters`, `--diffdock`, `--flowdock`, `--targetdiff`, `--equibind`, `--pocket2mol`, `--flowr`, `--biolm`, `--rxnflow`, `--synflownet`, `--synthemol` — **0 matches**.

The script is wired through:
1. `--synthesis-oracle {none,smarts,aizynthfinder,aizynthfinder_isolated}` (line 410)
2. `--search-docking-reward` + `--search-docking-engine {auto,vina,quickvina2,quickvina2-gpu}` (lines 413-415)
3. `--physical-docking` + `--physical-engine {auto,vina,quickvina2,quickvina2-gpu}` (lines 420-421)
4. `--gpu-config` (line 423, JSON path for QuickVina2-GPU binary)
5. `--config <yaml>` — loads `molmetal_lam/configs/sota_aligned_targetdiff.yaml` and routes into `lambda_100pocket_sweep.run_one_pocket(...)` (line 154-167)
6. `--prior-state <json>` (line 409) — feeds `prepare_prior(...)` symbolic prior
7. `--symbolic-prior` (line 407) — toggle the labelled-linear descriptor prior

There is no `--use-posebusters`, `--use-diffdock`, `--use-targetdiff`, `--use-pocket2mol`, `--use-pic50`, `--use-metalligand`, etc. Each "additional adapter" would have to be added as a new CLI flag, OR (cleaner) routed through the YAML config (`molmetal/configs/sota_aligned_targetdiff.yaml`) which `load_sota_aligned_config` projects into `__mcts_call_kwargs__` + `__lambda_kwargs__`.

**This is the structural gap**: 6 stub adapters cannot be wired without code changes, and 19 REAL-but-not-integrated adapters are exposed only as imports (callers must hand-write a wrapper around them).

---

## 4. Suggested Wire Targets (for follow-up WF-Wire-Clone-Scoring phase 2)

Priority order based on (i) REAL status, (ii) external-asset readiness, (iii) ROCm compatibility:

1. **`diffdock_adapter.py` → r4_c_full_sweep `--use-diffdock`** — adapter is REAL; missing just GPU ckpt. Wire behind `--use-diffdock` flag with `is_available()` check + `AdapterUnavailable` graceful fallback.
2. **`pocket2mol_adapter.py` → r4_c_full_sweep `--use-pocket2mol`** — REAL fallback path (120 SMILES) is asset-free. Live branch gated by `_checkpoint_available()`.
3. **`metal_generator_adapter.py` → r4_c_full_sweep `--metal-center`** — REAL RDKit path; let search branch with Pt/Au/Cu.
4. **`pic50_predictor.py` → r4_c_full_sweep `--use-pic50`** — REAL if `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` exists. Adds ML oracle to top-K ranking.
5. **`flowdock_adapter.py` (live branch) → `--use-flowdock`** — REAL; needs GPU + torch_cluster same as DiffDock.
6. **`aizynth_rocm_policy.py` + `aizynth_torch_model.py`** → keep in isolated env (already correctly isolated via `aizynthfinder` subprocess); no sweep wire needed.

Stub-to-REAL conversions:
- `targetdiff_adapter.py` + `diffsbdd_adapter.py` — needs weights + PyG/ROCm wheels; Phase-3.
- `reinvent4_adapter.py` — needs REINVENT4 plugins; deliberately kept RDKit-fallback.
- `flowr_adapter.py` — needs torch_cluster; Phase-3.
- `pybind_adapter.py` — needs pybind11 `.so` for EquiBind geodesic kernel; Phase-3.

---

## 5. Metrics for Structured Output

| Metric | Value |
|---|---|
| `n_adapters_total` | 30 |
| `n_real` | 24 |
| `n_stub` | 6 |
| `n_integrated` (visible from `r4_c_full_sweep.py` CLI or via `lambda_100pocket_sweep`) | 11 |
| `n_not_integrated` | 19 |
| `n_clone_repos_with_inference_script` | 5 (DiffDock, TargetDiff, Pocket2Mol, EquiBind, BioLM-Score; FLOWR has no top-level inference.py; SynFlowNet/SyntheMol/REINVENT4 use vendored Reaction-class adapters; TankBind/RxnFlow have no canonical inference entry) |

---

## 6. File References

- Adapter directory: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/`
- Sweep driver: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_c_full_sweep.py`
- Sweep backend: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/lambda_100pocket_sweep.py`
- Cloned repos: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/`
- Pretrained checkpoints: `/home/hugo/codes/try_triton_on_rocm/molmetal/checkpoints/`
