# Wf-50 CPU vs GPU Speed-up Estimates — Top 10 by impact

> **Methodology**: speed-up (`x`) is estimated as
> `throughput_gpu / throughput_cpu = per_call_latency_cpu / per_call_latency_gpu`
> for the same wall-clock work. **Workload** is the typical number of
> invocations per evaluation run (a single Lambda sweep is 1 000-10 000
> candidates; the per-pocket baseline is 50-200 candidates). **Impact**
> ranks `(speed-up × per-call CPU cost × workload)` so the highest-impact
> levers rise to the top regardless of order-of-magnitude estimates.
>
> Per-call CPU costs are anchored to measurements recorded in
> `molmetal/reports/combined_stability_report.md`,
> `molmetal/reports/lambda_vs_sbdd_paper_numbers.md`, and the
> `r4_*_sweep` JSONs; where no measurement exists, the entry is
> explicitly marked `EST.`.

## Ranking

| Rank | Tool / metric | Per-call CPU cost (s) | Workload per run | GPU speed-up (×) | GPU VRAM (MB) | Wall-time saving (CPU h → GPU h) | Risk | Confidence |
|---|---|---|---|---|---|---|---|---|
| **1** | **`vina_score` (+ aliases)** | 3-12 (exh=8) | 1 000-10 000 ligands × 5-10 pockets | **40-80** | 1 200 (single receptor cache) | 8-100 h → 0.1-1.5 h | HIGH — needs CUDA Vina-GPU binary; current `vina_gpu_adapter.py` is AMD-only | MEDIUM — upstream `diffdock-in-vina-gpu` build has no NVIDIA wheels |
| **2** | **`boltz_affinity` (`predict_pic50_batch`)** | 0.05 | 10 000-100 000 SMILES | **10-20** (batching) | 150 | 0.14-1.4 h → 0.01-0.07 h | LOW — model + ckpt already on `cuda:0`; wrapper is a 1-line refactor | HIGH — already in production |
| **3** | **`flowdock_pose` (40-step CPU ODE → GPU SE(3) flow)** | 5 (CPU ODE) | 1 000-5 000 ligands | **15-30** | 4 000 | 1.4-7 h → 0.05-0.25 h | MEDIUM — wrapper currently forks subprocess per SMILES; remove shim | MEDIUM — upstream depends on `torch_scatter` ROCm wheels |
| **4** | **`diffdock_pose` (current CPU/CLI fallback)** | 10-30 | 1 000-5 000 ligands | **5-15** | 6 000 | 2.8-42 h → 0.2-3 h | HIGH — `torch_cluster` ROCm wheels still missing | MEDIUM |
| **5** | **`pocket2mol` / `targetdiff`** | 2-3 (per pose) | 1 000-3 000 samples | **8-15** | 6 000 | 0.6-2.5 h → 0.04-0.2 h | HIGH — needs `torch_geometric` ROCm wheels + ckpts | MEDIUM |
| **6** | **`aizynth_synthesis` (AiZynthFinder rollouts on GPU)** | 0.5-5 (config-dep.) | 1 000-10 000 SMILES | **3-10** | 1 500 | 0.14-14 h → 0.02-1.4 h | LOW — `TorchDensePolicy` already on `cuda:0`; rollout batching is the only change | HIGH |
| **7** | **`aizynth_torch_model.predict` (already CUDA)** | 0.001-0.01 | 100 000-1 000 000 expansion calls | **20-50** | 100 | n/a (already on GPU) — speed-up vs. naive CPU | LOW — already GPU | HIGH |
| **8** | **`diversity_tanimoto` + `reference_tanimoto` + `diversity_subpocket` (RDKit Bulk)** | O(N²) for N=10 000 → 200 s | 10 000-100 000 pairs | **10-50** | 0 | 200 s × run → 4-20 s × run | LOW — pure-RDKit `DataStructs.BulkTanimotoSimilarity` | HIGH |
| **9** | **`protein_metal_binding.forward` (D-MPNN + fuser on CUDA)** | 0.01-0.05 | 1 000-10 000 (protein, ligand) pairs | **5-15** | 200 | 0.003-0.14 h → <0.01 h | LOW — model on `cuda:0`; wrap in `torch.no_grad()` | HIGH |
| **10** | **`af2_confidence` voxelization (`scatter_add_`)** | 0.1 | 100-1 000 pockets | **5-10** | 200 | 0.003-0.03 h → <0.005 h | LOW — `torch.scatter_add_` is one-liner | HIGH |
| **11** | **`typed_dispatch_triton.triton_match_kernel`** | 0.5 (MCTS) | 10⁵-10⁶ (atom, rule) pairs | **10-30** | 50 | n/a — kernel-bound by MCTS size | LOW — kernel is in tree; needs CUDA-build of Triton | HIGH (already ROCm-tested) |
| **12** | **`learned_shaping` + `platinai_oracle` reward nets** | 0.001-0.01 | 10 000-100 000 states | **5-15** | 100 | 0.003-0.28 h → <0.02 h | LOW — already PyTorch | HIGH |
| **13** | **Hilbert / ASTSR reward math (R1, R2, R4)** | 0.01-0.05 | 10 000-100 000 | **5-10** | 200 | 0.03-1.4 h → <0.14 h | MEDIUM — math skeletons have CPU-only assertions to remove | HIGH |
| **14** | **Multi-parameter PH + SE(3) Schrödinger bridge math (M2-M11)** | 0.005-0.02 | 10 000-50 000 | **3-10** | 50-100 | n/a — math is per-step in proof-search | LOW — uniform `.to(device)` change | HIGH |
| **15** | **`coordination_gnn` (A8) + `quotient_space_diffusion` (A10) + `se3_*` adapters (A11-A13)** | 0.001-0.01 | 10 000-50 000 | **5-10** | 50-200 | n/a — math layer is per-step | LOW — `.to(device)` on input tensors | HIGH |
| **16** | **`egnn_rocm` + `egnn_rocm_checkpoint` (A6-A7, already Triton)** | 0.005 (per EGNN layer) | 10 000-100 000 | **5-15** | 200 | n/a — already wired | LOW — already shipped | HIGH |
| **17** | **`facebook_fm_wrapper` (A14)** | 0.005-0.02 | 1 000-10 000 | **5-15** | 400 | n/a — already on cuda | LOW — already shipped | HIGH |

## Notes

* **The biggest absolute wall-time saving is `vina_score`** because
  it is by far the slowest per-call CPU scorer and is invoked the most
  times per sweep. Once a CUDA Vina-GPU binary is built against
  Blackwell, the docking column collapses from hours to minutes.
* **`boltz_affinity` is the cheapest win** — a one-line refactor that
  uses the already-shipped `predict_pic50_batch` instead of the wrapper's
  serial loop. No model load, no ckpt changes, no risk.
* **`flowdock_pose` > `diffdock_pose` for our setup** because FlowDock
  ships a CPU-fallback ODE solver that is already invoked via
  subprocess; switching `device=cuda` is one-line in the wrapper.
* **The RDKit bulk-similarity entry (#8) is the only "CPU-only lever
  with a 10-50× speed-up that requires zero GPU work"** — pure RDKit
  `DataStructs.BulkTanimotoSimilarity` is batched and faster than the
  Python loop currently used by `diversity_tanimoto` and friends.
* **The math-skill + Hilbert-reward entries (#13-#15) are the largest
  *batch* of wins** — 17 modules each gain 3-10× with the same uniform
  change (`.to(device)` + relax the CPU-only test assertion). These
  are best tackled in a single ultracode (Wf-50 Phase 2 candidate).

## Per-metric speed-up math (worked examples)

### vina_score, 1 pocket × 1 000 ligands, exhaustiveness=8

* CPU (Meeko+Vina 1.2.7 Python binding): 6 s × 1 000 = **6 000 s = 1.7 h**
* GPU (Vina-GPU CUDA build, batched): 6 000 / 50 = **120 s = 0.03 h**
* Net saving: **~1.6 h per pocket × 10 pockets = 16 h/run**

### boltz_affinity, 100 000 SMILES

* CPU serial (current wrapper): 50 ms × 100 000 = **5 000 s = 1.4 h**
* GPU batched (`predict_pic50_batch(batch_size=128)`): 1.4 / 15 = **0.09 h**
* Net saving: **~1.3 h/run**, no model load required

### flowdock_pose, 1 pocket × 5 000 ligands

* CPU ODE (current 40-step fallback): 5 s × 5 000 = **25 000 s = 6.9 h**
* GPU (FlowDock upstream `--device cuda`): 6.9 / 20 = **0.35 h**
* Net saving: **~6.5 h/run**, contingent on ROCm `torch_scatter` wheels

### diversity_tanimoto, 10 000 generated mols

* CPU serial (current `for smi in smiles_list`): ~200 s pairwise
* GPU-via-CPU (RDKit `BulkTanimotoSimilarity`): ~6 s
* Net saving: **~200 s/run**, zero GPU work, one-line refactor

## Confidence ratings

* **HIGH** — model / kernel / wrapper is already shipped, the change is
  1-10 lines and uses an existing GPU path. All entries 2, 6, 7, 8, 9,
  10, 11, 12, 14, 15, 16, 17 are HIGH.
* **MEDIUM** — model is shipped but requires ROCm-compatible wheels
  (`torch_cluster`, `torch_scatter`, `torch_geometric`) for
  DiffDock/FlowDock/Pocket2Mol/TargetDiff. Entries 1, 3, 4, 5.
* **LOW** — vendor binary required (Schrödinger Glide, CUDA build of
  Vina-GPU). Out of immediate scope.

## Verification commands

```bash
# Confirm 45 scoring-tool entries, 18 abstract-adapter entries,
# 8 reward-module entries, 11 math-skill entries:
python -c "
import json
m = json.load(open('molmetal/reports/wf50_gpu_audit/gpu_readiness_matrix.json'))
print('scoring_tools', len(m['scoring_tools']))
print('abstract_adapters', len(m['abstract_layer_adapters']))
print('sbdd_env', len(m['sbdd_env_modules']))
print('reward', len(m['reward_modules']))
print('math_skills', len(m['math_skills_modules']))
"
```

## See also

* `gpu_readiness_matrix.md` — full per-tool matrix.
* `critical_path.md` — the 5 most-impactful accelerations to ship next.