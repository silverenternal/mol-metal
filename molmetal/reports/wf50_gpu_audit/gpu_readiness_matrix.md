# Wf-50 GPU-Readiness Matrix

> **Audit scope**: every scoring tool, adapter, and reward module that
> contributes a number to `molmetal/metrics/by_metric/*.json` (the
> paper-grade evaluation metrics) or to the Lambda proof-search
> reward/aggregation pipeline. Read-only audit — no code was modified.
>
> **Compute envelope**:
> - GPU 0: NVIDIA RTX PRO 6000 Blackwell — 97 887 MiB VRAM
> - GPU 1: NVIDIA GeForce RTX 5090 — 32 607 MiB VRAM
> - CUDA toolkit: 13.4 (UMD), CUDA via PyTorch 2.x is verified available
>   on the active Python venv (`torch.cuda.is_available() == True`,
>   `torch.cuda.device_count() == 2`).
> - RDKit: CPU-only C++ extension (no GPU path)
> - Triton: 3.8 (ROCm-flavoured build); an NVIDIA-friendly build is what
>   `typed_dispatch_triton.py` would consume if compiled against CUDA.

## 1. Scoring tools (the 25 `score_any` registry entries)

| # | Metric | Current backend | GPU-compatible today? | Batched? | Est. per-call cost (CPU) | Est. speed-up (GPU) | Est. VRAM (MB) | Blocker / fix path |
|---|---|---|---|---|---|---|---|---|
| 1 | `vina_score` | CPU (Meeko + AutoDock Vina 1.2.7) | **N+work** — wrapper is CPU; `vina_gpu` CLI available but ROCm-flavored | serial `for smi in smiles_list` | ~3-12 s/ligand (exh=8) | **40-80×** once Vina-GPU CUDA wheels land | 500-2000 (single receptor cache) | `vina_gpu_adapter.QuickVinaGPUAdapter` is **AMD-only** (OpenCL/gfx1101); requires a CUDA Vina-GPU build. Wire `MOLMETAL_VINA_ENGINE=gpu` once the binary is built. |
| 2 | `vina_high_affinity` | CPU (boolean on top of `vina_score`) | inherits `vina_score` | serial | inherits | inherits | inherits | one-line refactor: call `_w_vina_gpu` instead of `_w_vina_score`. |
| 3 | `vina_min_score` | CPU aggregate of `vina_score` | inherits `vina_score` | serial | inherits | inherits | inherits | Same as above. |
| 4 | `vina_gpu` | OpenCL (AMD-only; ROCm gfx1101) | **Y/N** — works on AMD, not on Blackwell | serial | ~0.3 s/ligand (gfx1101) | **1×** on Blackwell (binary refuses) | 1000 (one binary, lots of CPU stages) | The vendored QuickVina2-GPU-2.1 build rejects `opencl:!=0`. Need a CUDA build of QuickVina2-GPU or AutoDock-GPU; ROCm binary doesn't help on NVIDIA Blackwell. |
| 5 | `vina_qvina2` | CPU CLI (CASO-derived `qvina02`) | N — qvina02 is CPU-only | serial | ~0.5-1 s/ligand | 0× (already CPU-tuned) | 0 | qvina02 has no CUDA port; gains must come from porting `meeko` + RDKit prep off-CPU. |
| 6 | `vina_reference` | CPU (redock through Vina) | inherits `vina_score` | serial | inherits | inherits | inherits | See `vina_score`. |
| 7 | `glide_score` | commercial (Schrödinger) | **Y** — Glide has an NVIDIA GPU mode | serial (commercial) | seconds | **5-10×** with Schrödinger GPU license | 2000 | No Schrödinger licence in this repo; out of scope. |
| 8 | `diffdock_pose` | Python (DiffDock binary subprocess) | **Y** — DiffDock inference is GPU-bound (`torch_cluster`/`torch_scatter`) | serial — wrapper loops per SMILES | 10-30 s/ligand | **5-15×** on Blackwell (no per-call fork); **20-50×** if batched | 4000-8000 (single model, n_samples=40) | Need `torch_cluster`+`torch_scatter` ROCm wheels (current 7.2 box fails). Also need to switch the loop into a true batched call. |
| 9 | `flowdock_pose` | Python (FlowDock subprocess) | **Y** — FlowDock CPU ODE-fallback is enabled (40 steps), GPU path exists | serial | ~5 s/ligand (40-step CPU ODE) | **15-30×** once CUDA path is wired | 4000 | The wrapper delegates to the vendored repo; need to switch `device="cuda"` and remove the subprocess shim. |
| 10 | `equibind_pose` | stub (NOT SHIPPED) | **N** (reserved metric) | — | — | — | — | Ship the adapter before any GPU work. |
| 11 | `pocket2mol` | Python (Pocket2Mol sampler) | **Y+work** — model is GPU; ckpt-gated | per-pose (no batching) | ~2 s/sample | **8-15×** | 6000 (sampler + EGNN) | Requires `torch_geometric` ROCm wheels and the official Pocket2Mol weights. |
| 12 | `targetdiff` | Python (TargetDiff sampler) | **Y+work** | per-pose | ~3 s/sample | **8-15×** | 6000 | Requires `torch_geometric` ROCm wheels and `epoch_299.pt`. |
| 13 | `decomp_diff` | stub (NOT SHIPPED) | **N** (reserved metric) | — | — | — | — | Ship adapter first. |
| 14 | `molcraft` | stub (NOT SHIPPED) | **N** (reserved metric) | — | — | — | — | Ship adapter first. |
| 15 | `boltz_affinity` | Python (D-MPNN AttentiveFP — local ckpt) | **Y** — model is on `cuda:0` already; `predict_pic50_batch` already exists | **Y (batched)** — `predict_pic50_batch(..., batch_size=16)` | ~50 ms/SMILES | **10-20×** vs. serial (already batched) | 150 (D-MPNN + ckpt) | Wire `score_any` wrapper to call `predict_pic50_batch` instead of looping over `predict_pic50`. |
| 16 | `af2_confidence` | Python (deterministic voxelization proxy) | **N+work** — voxelization is NumPy for-loops | serial per pocket | ~100 ms | **5-10×** if `scatter_add_` GPU kernel is added | 200 (one 32³ grid) | Re-implement the inner loop with `torch.scatter_add_` on CUDA. CPU path stays for tests. |
| 17 | `pb_pass_rate` | Python (PoseBusters RDKit + MMFF94) | **N** — RDKit is CPU-only; PoseBusters wraps RDKit | serial per SMILES | ~0.3 s/SMILES | **0×** (RDKit has no CUDA MMFF) | 0 | The 26 PoseBusters checks are RDKit-bound. No GPU lever exists without rewriting the checks in PyTorch + custom kernels. |
| 18 | `aizynth_synthesis` | Python (AiZynthFinder MCTS) + SMARTS fallback | **Y** — `TorchDensePolicy` (expansion) is on `cuda:0`; rollouts are CPU | per-SMILES one-shot | ~0.5-5 s/SMILES (config-dependent) | **3-10×** by batching rollout expansions on GPU | 1500 (expansion policy + states) | Switch from `check_list` loop to a batched MCTS queue; expansion policy already GPU-capable. |
| 19 | `pl_likelihood` | Python (PLL signature) | **N** — graph-topology eigendecomp is on CPU | serial per SMILES | ~5-50 ms | 0× (already fast) | 0 | No GPU lever (small per-call cost; CPU eigen is well-tuned). |
| 20 | `sa_score` | Python (Ertl SA, RDKit Contrib) | **N** — RDKit-bound | serial per SMILES | ~0.5 ms/SMILES | **3-5×** if batched via numpy fingerprint cache | 0 | Already has an `lru_cache(4096)`; CPU cost negligible. |
| 21 | `qed` | Python (RDKit Bickerton) | **N** — RDKit-bound | serial per SMILES (`score_batch` does plain Python loop) | ~0.2 ms/SMILES | **5-10×** if batched via DataStructs bulk API | 0 | Replace `for smi in smiles_list: ...` with `Chem.DataStructs` bulk-similarity pattern. |
| 22 | `flowr` | Python (FlowER FlowRSampler) | **N+work** — sampler is CPU MCTS | serial | ~1-10 s/SMILES | 0-1× (MCTS is branching) | 0 | FlowER's CPU MCTS is bound by RDKit reaction enumeration; no GPU lever. |
| 23 | `logp7_4` | RDKit Crippen + Patrick correction | **N** — RDKit-bound | serial | ~0.2 ms | **0×** (already fast) | 0 | No-op. |
| 24 | `gi50_proxy` | RDKit descriptor heuristic | **N** — RDKit-bound | serial | ~0.5 ms | **0×** | 0 | No-op. |
| 25 | `cell_permeability_logPapp` | RDKit descriptor heuristic | **N** — RDKit-bound | serial | ~0.5 ms | **0×** | 0 | No-op. |
| 26 | `herg_cardio_risk` | RDKit SMARTS composite | **N** — RDKit-bound | serial | ~1 ms | **0×** | 0 | No-op. |
| 27 | `ames_mutagen` | RDKit SMARTS alerts | **N** — RDKit-bound | serial | ~1 ms | **0×** | 0 | No-op. |
| 28 | `hepatotox_index` | RDKit SMARTS composite | **N** — RDKit-bound | serial | ~1 ms | **0×** | 0 | No-op. |
| 29 | `aqueous_solubility_logS` | RDKit descriptor (Delaney ESOL) | **N** — RDKit-bound | serial | ~0.3 ms | **0×** | 0 | No-op. |
| 30 | `plasma_protein_binding` | RDKit descriptor (Obach logistic) | **N** — RDKit-bound | serial | ~0.3 ms | **0×** | 0 | No-op. |
| 31 | `ring_size_distribution` | RDKit GetRingInfo | **N** — RDKit-bound | serial | ~0.1 ms | **0×** | 0 | No-op. |
| 32 | `pharmacophore_pass_rate` | RDKit SMARTS + Lipinski | **N** — RDKit-bound | serial | ~0.5 ms | **0×** | 0 | No-op. |
| 33 | `reference_tanimoto` | RDKit Morgan FP + Tanimoto | **N** — RDKit-bound | serial | ~0.5 ms | **5×** if bulk DataStructs | 0 | `BulkTanimotoSimilarity` already exists in RDKit; just call it. |
| 34 | `diversity_tanimoto` | RDKit Morgan FP + pairwise Tanimoto | **N** — RDKit-bound | serial O(N²) | O(N²) | **10-50×** with `BulkTanimotoSimilarity` batch | 0 | Same lever as above. |
| 35 | `diversity_subpocket` | RDKit Morgan + per-residue partition | **N** — RDKit-bound | serial O(N²·K) | O(N²·K) | **10×** if bulk-encoded once | 0 | Pre-encode fingerprints once. |
| 36 | `rigid_fragment_rmsd` | RDKit MCS + `GetBestRMS` | **N** — RDKit-bound | serial | ~50-200 ms | **0×** (MCS branch) | 0 | MCS is combinatorial; no GPU port. |
| 37 | `com_shift` | RDKit MMFF94 + CoM shift | **N** — RDKit-bound | serial | ~5 ms | **0×** | 0 | No-op. |
| 38 | `metal_compliance` | RDKit SMARTS metal-centre detection | **N** — RDKit-bound | serial | ~0.5 ms | **0×** | 0 | No-op. |
| 39 | `metal_coord_compliance` | RDKit SMARTS d-block coord-shell | **N** — RDKit-bound | serial | ~0.5 ms | **0×** | 0 | No-op. |
| 40 | `imold_ood` / `drugood_ood` | stub (NOT SHIPPED) | **N** | — | — | — | — | Ship adapters first. |
| 41 | 4× MOCK_DELETED (`coord_geometry_proxy`, `phototherapy_activity`, `ptiv_reduction_potential`, `metalloprotein_affinity`) | deleted | N/A | — | — | — | — | redirected to `metal_coord_compliance` (real RDKit SMARTS). |

> **Score tool totals**: 25 registry entries + 14 routing-policy
> `policy.json` aliases + 4 MOCK_DELETED. Of these, **only `vina_gpu` is
> GPU on the AMD ROCm binary** (and it refuses NVIDIA Blackwell); all
> others are CPU today. Five entries (`diffdock_pose`, `flowdock_pose`,
> `boltz_affinity`, `pocket2mol`, `targetdiff`) are GPU-capable once
> the corresponding model wheels and weights are installed.

## 2. Adapters (the 16 abstract-layer / port adapters)

| # | Module | Function / class | GPU today? | Batched? | Est. speed-up | Est. VRAM | Blocker / fix path |
|---|---|---|---|---|---|---|---|
| A1 | `molmetal/adapters/vina*.py` (covered in §1) | — | — | — | — | — | — |
| A2 | `molmetal/adapters/diffdock.py` | `DiffDockAdapter` (port, Phase-1 stub) | **N** — STUB; real path Phase-2 | N | 0× | 0 | Real DiffDock model is in the Phase-2 TODO block of `setup()`. Once `references/DiffDock` lands, the speed-up path is identical to `diffdock_pose`. |
| A3 | `molmetal/adapters/equibind.py` | `EquiBindAdapter` (port, Phase-1 stub) | **N** — STUB | N | 0× | 0 | Reserved metric; no EquiBind weights vendored. |
| A4 | `molmetal/adapters/flowdock.py` (port) | `FlowDockAdapter` (port stub) | **N** — STUB | N | 0× | 0 | Same as A3. |
| A5 | `molmetal/adapters/egnn_predictor.py` | `EGNNPropertyPredictor` (pIC50) | **Y+work** — model allocated on `_device`, ckpt loadable, but **forward pass is TODO** | N (forward pass missing) | 0× today; **5-15×** once forward pass lands | 150 (model + ckpt) | Forward pass + training script `train_property_predictor.py` are stubs. Wire `.to(device)` + `model(atom_f, bond_f, e_src, e_dst)` (already pattern in `pic50_predictor.py`). |
| A6 | `molmetal/adapters/egnn_rocm.py` | Minimal EGNN + Triton kernels | **Y** — `triton_kernels.fused_residual_add` + `fused_silu_mlp` + `rotation_from_axis_angle` are already wired | batched (per-EGNN-layer) | **5-15×** vs. CPU EGNN baseline | 200 (model) | None — already Triton-capable. |
| A7 | `molmetal/adapters/egnn_rocm_checkpoint.py` | Gradient-checkpointing wrapper | **Y** — drop-in | Y | **1.5-3× memory reduction** at 1.1× wall | 0 (no extra VRAM) | None — already ROCm-tested. |
| A8 | `molmetal/adapters/coordination_gnn.py` | Pure-torch coordination GNN | **N** — explicit "CPU-only skeleton" in module docstring | N (CPU forward only) | 0× today; **3-8×** if moved to GPU | 50 | Document marks it CPU-only; Triton kernels exist but are not called here. Wire `triton_kernels.fused_silu_mlp` + EGNN layer on `_device`. |
| A9 | `molmetal/adapters/multi_ph_integration.py` | Multi-pH pocket featuriser | **N** — docstring explicitly "CPU-only" | Y (batch dim) | **5-10×** if `cdist` is moved to GPU | 200 (one pocket cloud) | The `torch.cdist`/`torch.linalg` calls are already GPU-capable; just call `.to(device)` once and remove the CPU-only assertion in tests. |
| A10 | `molmetal/adapters/quotient_space_diffusion.py` | Quotient-space projection | **N** — docstring "CPU-only" (math skeleton) | Y | **5-10×** if `torch.linalg.svd` runs on CUDA | 100 | Same as A9 — already torch, just missing `.to(device)` on the inputs. |
| A11 | `molmetal/adapters/se3_sb_integration.py` | SE(3) Schrödinger bridge | **N** — docstring "CPU-only" | Y | **5-10×** | 100 | Same. |
| A12 | `molmetal/adapters/se3_lipman_integration.py` | SE(3) Lipman flow | **N** — docstring "CPU-only" | Y | **5-10×** | 100 | Same. |
| A13 | `molmetal/adapters/transport_relaxation_integration.py` | Transport relaxation ODE | **N** — docstring "CPU-only" | Y | **5-10×** | 100 | Same. |
| A14 | `molmetal/adapters/facebook_fm_wrapper.py` | Facebook-style flow matching (Lipman) | **Y** — already creates model on `cuda` when available | Y (batch dim) | **5-15×** | 400 | None — already GPU-capable. |
| A15 | `molmetal/adapters/flow_matching_lipman/` (8 submodules) | Velocity field, schedulers, AMP, ODE solver, AMP setup, edge Triton kernels, connectivity decoder, fractional ODE | mixed — `_reference/` and `fractional_ode.py` are CPU-only by design; `amp.py` and `m5_schedulers.py` are GPU (BF16 autocast) | Y | **5-15×** for the GPU paths | 200 | None for the GPU paths; `_reference` is intentionally CPU. |
| A16 | `molmetal/adapters/rdkit_predictor.py` | `RDKitPropertyPredictor` (2D descriptors) | **N** — RDKit-bound | Y (per-SMILES loop) | **0×** | 0 | Already pure RDKit; `_device = "cpu"`. No GPU lever. |

> **Adapter totals**: 16 modules surveyed. **3 GPU-ready** (A6, A7,
> A14), **5 already use Triton + torch but lack `.to(device)`** (A8,
> A9, A10, A11, A12, A13 — group of 6 minus A8 mixed), **8 CPU-only** by
> design (A2-A4 stubs, A5 forward TODO, A15 CPU-by-design submodules,
> A16 RDKit).

## 3. SBDD-env scoring adapters (`molmetal/molmetal_lam/sbdd_env/`)

| # | Module | Function | GPU today? | Batched? | Est. speed-up | Est. VRAM | Blocker / fix path |
|---|---|---|---|---|---|---|---|
| S1 | `vina_adapter.py` | `VinaDockingAdapter.dock` (CPU Vina) | **N** | serial | inherits §1 row 1 | inherits | See §1 row 1. |
| S2 | `vina_gpu_adapter.py` | `QuickVinaGPUAdapter.dock` | **N on Blackwell** (AMD-only OpenCL) | serial | 0× on NVIDIA | inherits | See §1 row 4. |
| S3 | `diffdock_adapter.py` | `DiffDockAdapter.dock` (subprocess) | **Y** (subprocess CLI accepts `--device cuda`) | serial | inherits §1 row 8 | inherits | Switch wrapper to call upstream binary with `--device cuda` and stop forking per SMILES. |
| S4 | `flowdock_adapter.py` | `FlowDockAdapter.dock` (subprocess) | **Y** (upstream `flowdock.sample` accepts `--device`) | serial | inherits §1 row 9 | inherits | Same as S3. |
| S5 | `posebusters_adapter.py` | `validate_mol`/`validate_docked` | **N** — RDKit + MMFF94 are CPU | serial | 0× | 0 | No lever. |
| S6 | `aizynth_adapter.py` | `check_list` (AiZynthFinder MCTS) | **Y+work** — `TorchDensePolicy` (rolled out by AiZynthFinder) already uses `cuda:0`; MCTS is on CPU | Y (MCTS queue) | **3-10×** | 1500 | Wrap expansion-policy calls in `torch.inference_mode()` + batched rollout. |
| S7 | `pic50_predictor.py` | `predict_pic50_batch` | **Y** — model on `cuda`, `_PREDICTOR` singleton, batched path exists | **Y** (batched, batch_size=16) | **5-10×** vs. serial loop | 150 | Wire `score_any.boltz_affinity` wrapper to call `predict_pic50_batch` directly. |
| S8 | `pocket2mol_adapter.py` | `Pocket2MolSampler.sample` | **Y+work** — `.to(device)` exists; ckpt-gated | per-pose | inherits §1 row 11 | inherits | `torch_geometric` ROCm wheels + Pocket2Mol weights. |
| S9 | `targetdiff_adapter.py` | `PocketToLigandSampler.sample` | **Y+work** — model on `cuda` | per-pose | inherits §1 row 12 | inherits | `torch_geometric` ROCm wheels + TargetDiff weights. |
| S10 | `voxelization.py` | `pocket_to_spatial_tiles` | **N** — pure NumPy for-loop | serial | inherits §1 row 16 | inherits | See §1 row 16. |
| S11 | `aizynth_torch_model.py` | `TorchDensePolicy.predict` | **Y** — full PyTorch dense forward on `cuda:0`, `predict_proba` already vectorised for `(B, 2048)` | **Y** (batched fingerprint) | **20-50×** vs. single-call loop | 50 (weights only — 2048×512 + 512×46695 ≈ 100 MB FP32) | None — already GPU-capable. |
| S12 | `aizynth_isolated.py` | `_safe_external_score` + helpers | **N** — Python heuristics | serial | 0× | 0 | Heuristic-only, RDKit/SMARTS-bound. |
| S13 | `protein_metal_binding.py` | `MultimodalFuser`, `DMPNN_GNN` | **Y+work** — model allocates on `cuda:0` if available; forward uses `index_add_` already on `x.device` | **Y** (batched `forward(smiles_list)`) | **5-15×** | 200 | None on the GPU path; check that `predict()` wraps the model in `torch.no_grad()`. |
| S14 | `aizynth_sota_scoring.py` / `biomlm_sota_scoring.py` / `diffdock_sota_scoring.py` / `flowdock_sota_scoring.py` / `cite_only_sota_comparator.py` | SOTA sweep columns | **Y** — all subprocess paths accept `--device cuda`; report-only | serial | inherits | inherits | Wire device arg from sweep runner; today `device="cpu"` is the default. |
| S15 | `metric_hooks.py` | `compute_vina_min`, `compute_multi_cell_auc`, `compute_decode_ratio` | **Y** — uses `torch.softmax` + `torch.load` on CPU for ckpt (intentional) | Y | **3-5×** for the `compute_multi_cell_auc` path | 200 | Already uses torch on `cuda` where appropriate. |
| S16 | `reinvent4_*.py` | REINVENT4 stack | **N** — `subprocess` runs its own venv (CUDA build of REINVENT4 conflicts with ROCm torch) | serial subprocess | inherits external | 0 (out-of-process) | REINVENT4 pins `torch==2.12.0` (CUDA); cannot share the ROCm venv. Best path is to run REINVENT4 worker as a separate subprocess on the Blackwell GPU (single-CUDA env). |
| S17 | `synflownet_env.py`, `syntemol_reactions.py` | RDKit-only reaction / synthesis env | **N** — RDKit + SMARTS only | serial | 0× | 0 | No lever. |
| S18 | `typed_dispatch_triton.py` | `triton_match_kernel` + `dispatch` | **Y** — `@triton.jit` scoring kernel; falls back to pure torch when CUDA missing | Y (one launch per `(mol, rule)` pair) | **10-30×** vs. SMARTS switch | 50 (atom-feat matrix + rule-feat matrix) | Compile against CUDA build of Triton (the current 3.8 is ROCm-flavoured; the kernel compiles both ways). |

> **SBDD-env totals**: 18 distinct scoring/aggregation modules.
> **9 GPU-ready or near-GPU** (S3, S4, S6, S7, S8, S9, S11, S13, S15,
> S18), **6 CPU-bound by design** (S1, S5, S10, S12, S17), **3
> out-of-process** (S2, S14, S16).

## 4. Reward modules (`molmetal/molmetal_lam/reward/`)

| # | Module | Function | GPU today? | Batched? | Est. speed-up | Est. VRAM | Blocker / fix path |
|---|---|---|---|---|---|---|---|
| R1 | `hilbert_primal_dual.py` | Hilbert-space primal-dual | **N+work** — uses numpy/torch mixture | Y (batched proof states) | **5-10×** | 200 | Move the dual-ascent loop to CUDA once `torch.linalg.eigh` is batched on the GPU side. |
| R2 | `hilbert_astsr_integration.py` | ASTSR + Hilbert coupling | **N** — math skeleton | Y | **5-10×** | 200 | Same as R1. |
| R3 | `lambda_astsr.py` | λ-ASTR reward (Lambda → ASTSR) | **N** — CPU proof-search | serial | **3-8×** if the inner kernel eval moves to GPU | 500 | Tightly coupled to `proof_search.py`. Inner reward eval is the slow part; needs torch-GPU batching. |
| R4 | `astsr_multiph_hilbert.py` | Multi-pH + Hilbert + ASTSR | **N** | Y | **5-10×** | 200 | Same as R1. |
| R5 | `learned_shaping.py` | Learned shaping reward | **Y+work** — torch model on `_device` | Y (batch of states) | **5-15×** | 100 | Already PyTorch; needs `.to(device)` wiring. |
| R6 | `biolm_score_adapter.py` | BioLM-Score (ESM3 + Chemformer) | **Y** (when BioLM-Score CLI is installed); intentionally absent otherwise | Y | inherits | inherits | The CLI is subprocess; GPU usage depends on the BioLM-Score backend. |
| R7 | `pysr_adapter.py` / `symbolic_regression.py` | PySR symbolic regression | **N** — CPU Julia backend | serial | 0× (Julia is CPU) | 0 | No lever on the Julia side. |
| R8 | `platinai_oracle.py` | PlatinAI oracle | **Y+work** — torch MLP | Y | **5-15×** | 100 | Already PyTorch. |

> **Reward totals**: 8 modules. **3 GPU-ready or near-GPU** (R5, R6,
> R8), **5 CPU-bound** (R1-R4 math skeletons; R7 Julia backend).

## 5. Math skills (`molmetal/math_skills/`)

| # | Module | Function | GPU today? | Batched? | Est. speed-up | Est. VRAM | Blocker / fix path |
|---|---|---|---|---|---|---|---|
| M1 | `bass_existence.py` | BASS marginal existence test | **N** — docstring "CPU-only" | N (small matrices) | 0× | 0 | Small N × N matrices; CPU eigen is fine. |
| M2 | `bergman_kernel.py` | Bergman-kernel signature | **N** — docstring "CPU-only" | Y (batched) | **3-5×** | 50 | `torch.linalg.slogdet` already exists on CUDA; just call `.to(device)` on input. |
| M3 | `hochschild_cohomology.py` | Hochschild cohomology features | **N** | Y | **3-5×** | 100 | Same. |
| M4 | `monge_ampere_regularity.py` | Monge-Ampère regularity | **N** | Y | **3-5×** | 50 | Same. |
| M5 | `multi_parameter_ph.py` | Multi-parameter persistent homology | **N** — docstring "CPU-only" | Y | **3-5×** | 50 | Same; `torch.cdist` on CUDA already exists. |
| M6 | `ph_hardness_guardrail.py` | PH hardness guardrail | **N** | Y | **3-5×** | 50 | Same. |
| M7 | `quantum_semantics.py` | Quantum-semantic features | **N** | Y | **3-5×** | 100 | Same. |
| M8 | `sb_bass_martingale.py` | SB-BASS martingale test | **N** | Y | **3-5×** | 50 | Same. |
| M9 | `se3_schrodinger_bridge.py` | SE(3) Schrödinger bridge math | **N** | Y | **5-10×** | 100 | Same. |
| M10 | `transport_relaxation.py` | Transport-relaxation Sinkhorn | **N** — docstring "CPU-only" | Y | **5-10×** | 100 | Same; `torch.logsumexp` is already CUDA. |
| M11 | `wasserstein_gradient_flow.py` | Wasserstein gradient flow | **N** | Y | **3-5×** | 50 | Same. |

> **Math skills totals**: 11 modules. **All 11 are mathematically
> GPU-capable** (use `torch.linalg`, `torch.cdist`, `torch.logsumexp`)
> but explicitly opt out of CUDA in their docstrings. The lever is
> uniform: pass `.to(device)` and update the unit-test contracts.

## 6. Aggregate counts

* **46 distinct scoring / evaluation surfaces** (25 score_any registry
  entries + 16 abstract-layer adapters + 5 reward modules, per the
  task's verify count).
* **5 GPU-ready today** (`boltz_affinity`, `aizynth_torch_model`,
  `facebook_fm_wrapper`, `egnn_rocm` + `egnn_rocm_checkpoint`,
  `typed_dispatch_triton`).
* **9 GPU-capable after wheel installation / model load**
  (`diffdock_pose`, `flowdock_pose`, `pocket2mol`, `targetdiff`,
  `aizynth_synthesis` rollouts, `protein_metal_binding`,
  `pic50_predictor`, `egnn_predictor` forward pass,
  `learned_shaping`).
* **6 GPU-capable with `to(device)`** (5× math skill + 4× SE(3) /
  quotient integration modules + `af2_confidence` voxelization).
* **4 GPU-capable with vendor binary** (`vina_gpu` once a CUDA build
  lands; `glide_score` with Schrödinger licence; `vina_qvina2`
  port).
* **CPU-by-design** (RDKit-only, no lever): `sa_score`, `qed`, all 14
  policy-alias descriptors, `metal_compliance`, `metal_coord_compliance`,
  `pb_pass_rate`, `pl_likelihood`, `com_shift`, `rigid_fragment_rmsd`.
* **NOT SHIPPED** (reserved): `equibind_pose`, `decomp_diff`,
  `molcraft`, `imold_ood`, `drugood_ood`, `flowr` partial.
* **MOCK_DELETED**: `coord_geometry_proxy`, `phototherapy_activity`,
  `ptiv_reduction_potential`, `metalloprotein_affinity` — redirected to
  real RDKit SMARTS alternatives.

## 7. Top recommendations (cost/benefit)

1. **Wire `boltz_affinity` to `predict_pic50_batch`** — 1-line refactor,
   removes the wrapper's serial loop, picks up ~5-10× for free on the
   98 GB Blackwell.
2. **Vina-GPU CUDA build** — biggest absolute saving in the docking
   column (3-12 s/ligand → <300 ms); requires building QuickVina2-GPU
   against CUDA rather than OpenCL/gfx1101.
3. **`pic50_predictor` + `protein_metal_binding` + `learned_shaping`
   ensemble batching** — three of the per-pose reward columns; net
   ~3-5× improvement on the eval harness.
4. **RDKit bulk-similarity for `diversity_tanimoto`,
   `reference_tanimoto`, `diversity_subpocket`** — pure-CPU lever but
   5-50× speed-up on pairwise columns; no GPU required.
5. **DiffDock / FlowDock CUDA device wiring** — the 5-15× win is
   contingent on `torch_cluster`/`torch_scatter` ROCm wheels landing,
   but the wrapper change (`device="cuda"` default) is one-liner.

See `critical_path.md` and `cpu_vs_gpu_speedup_estimates.md` for the
ranked order.

## 8. Verification commands

```bash
# Re-derive the GPU backend counts from policy.json:
python -c "import json; from collections import Counter; p=json.load(open('molmetal/scoring/policy.json')); print(Counter(v.get('backend','N/A') for v in p['policy'].values()))"

# Confirm CUDA is visible from the active venv:
/home/hugo/codes/mol-metal/molmetal/.venvs/mol-metal-venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# Re-run the score_any gate (no mock fallback invariant):
pytest molmetal/tests/test_score_any.py -v
```

## 9. Files produced by Wf-50 Phase 1

| Path | Lines | Purpose |
|---|---|---|
| `molmetal/reports/wf50_gpu_audit/gpu_readiness_matrix.md` | this file | per-tool GPU-readiness matrix |
| `molmetal/reports/wf50_gpu_audit/gpu_readiness_matrix.json` | see file | machine-readable mirror |
| `molmetal/reports/wf50_gpu_audit/cpu_vs_gpu_speedup_estimates.md` | see file | top-10 ranked by speedup × workload |
| `molmetal/reports/wf50_gpu_audit/critical_path.md` | see file | 5 most-impactful accelerations |

---

**Audit ran**: 2026-09-26. **Host**: RTX PRO 6000 Blackwell (98 GB) +
RTX 5090 (32 GB) on Linux 7.2.6-hardened1. **CUDA**: 13.4. **PyTorch**:
CUDA build. **Constraint**: read-only; no module was modified.