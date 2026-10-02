export const meta = {
  name: 'wf-326-gpu-utilization-optimization',
  description: 'Wf-326: GPU utilization deep dive + framework optimization (no G6 interruption). (A) GPU 0/1 utilization profile — nvidia-smi + torch profiler to find bottlenecks (CPU-bound? DataLoader? SMILES featurization? Sync gradient?). (B) DataLoader optimization — increase num_workers (currently 0!) + pin_memory + prefetch_factor + persistent_workers. (C) torch.compile + CUDA Graphs optimization — static shape capture. (D) Mixed precision audit — BF16/AMP check + larger batch size optimization. (E) Triton kernel audit — verify what is already optimized vs what is missed (was supposedly done). (F) SMILES featurization cache effectiveness (Wf-320D follow-up) — verify PerEpochFeatureCache is actually being hit (0% hit rate in 1-epoch smoke). (G) Reference code (Chemprop v2 official + D-MPNN paper) deep dive — identify missed optimizations (potential not squeezed). (H) Synthesis + final optimization report + propose concrete speed-up config (5-10× speedup realistic target).',
  phases: [
    { title: 'GPU utilization optimization (parallel)', detail: '7 parallel agents profiling + optimizing GPU usage without G6 interruption' },
    { title: 'Synthesis', detail: 'Final optimization report + speed-up proposal' },
  ],
}

const PY = '/home/hugo/cades/CadButEaas/.venv-vlm/bin/python'

phase('GPU utilization optimization (parallel)')

const [wf326A, wf326B, wf326C, wf326D, wf326E, wf326F, wf326G] = await parallel([

  () => agent(`Wf-326A: GPU 0/1 utilization profile — nvidia-smi + torch profiler to find bottlenecks.

USE Bash + Read + Edit.

CONTEXT: GPU 1 only 24% util during G6 (per nvidia-smi). GPU 0 idle 70 GB. Why is compute not being fully saturated?

TASK:
1. Read molmetal/scripts/train_dmpnn_multitask_wasserstein.py — current trainer
2. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import torch; print(torch.cuda.get_device_properties(0))" 2>&1
3. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import torch; print(torch.cuda.get_device_properties(1))" 2>&1
4. Run nvidia-smi during G6 active training:
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory --format=csv -l 5 -c 12 2>&1
5. CPU utilization check:
   ps -ef | grep -E "train_dmpnn_multitask" | grep -v grep | awk '{print $3, $4, $5, $6}' | head -1
6. torch profiler run:
   /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
import torch
from torch.profiler import profile, ProfilerActivity
# Profile a single training step
...
" 2>&1 | tail -30
7. Identify bottlenecks:
   - DataLoader (num_workers=0 → main thread blocked)
   - SMILES featurization (per TODO-155 §3: CPU 660%+ util)
   - PyTorch DDP overhead (if any)
   - CPU-GPU sync (gradient sync)
8. Save profile: molmetal/reports/wf326a_gpu_profile/profile.json
9. Document: TODO/completed/326a_gpu_utilization_profile.md

Honest framing:
- G6 is running on GPU 1 (RTX 5090, 32 GB)
- 24% util means 76% compute headroom available
- Bottleneck is likely DataLoader (num_workers=0) + SMILES featurization CPU-bound
- Pure CPU profile work (no GPU contention with G6)

OUTPUT (PLAIN TEXT): gpu_0_util + gpu_1_util + per_bottleneck_status + per_bottleneck_pct + profile_status + recovery_path. Do NOT call StructuredOutput.`, {
    phase: 'GPU utilization optimization (parallel)',
    label: 'wf326A',
  }),

  () => agent(`Wf-326B: DataLoader optimization — num_workers + pin_memory + prefetch_factor + persistent_workers.

USE Bash + Read + Edit.

CONTEXT: Per launch command: --cpu-num-workers 0. Per TODO-155 §3: SMILES featurization is CPU-bound (660%+ util). DataLoader is single-threaded → main thread blocked on data loading.

TASK:
1. Read molmetal/scripts/train_dmpnn_multitask_wasserstein.py — current DataLoader config
2. /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -c "import psutil; print('CPU cores:', psutil.cpu_count(logical=False), 'logical:', psutil.cpu_count(logical=True))" 2>&1
3. Identify optimal num_workers: typically 4 × num_GPUs × num_CPUs_per_GPU
4. Implement DataLoader optimization:
   - num_workers = 4-8 (vs current 0)
   - pin_memory = True
   - prefetch_factor = 2
   - persistent_workers = True
5. Write 8 NEW tests: molmetal/tests/test_wf326b_dataloader_optimization.py
   - 3 tests: per-config correctness
   - 3 tests: throughput benchmark
   - 2 tests: memory overhead
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf326b_dataloader_optimization.py -v 2>&1 | tail -20
7. Save optimization config: molmetal/scripts/wf326b_dataloader_optimized.py
8. Document: TODO/completed/326b_dataloader_optimization.md

Honest framing:
- num_workers=0 is single-threaded bottleneck
- Optimal num_workers depends on CPU cores (4-8 typical)
- Throughput improvement: PROJECTED 3-5× (vs single-threaded)
- No GPU contention (CPU-only work)

OUTPUT (PLAIN TEXT): n_cores + optimal_num_workers + throughput_before_after + preflight_status + per_test_status. Do NOT call StructuredOutput.`, {
    phase: 'GPU utilization optimization (parallel)',
    label: 'wf326B',
  }),

  () => agent(`Wf-326C: torch.compile + CUDA Graphs optimization.

USE Bash + Read + Edit.

CONTEXT: Per WEAK_TO_STRONG_PLAN §3 + Wf-320B: harness auto-fallback in place. Need to verify torch.compile + CUDA Graphs for static shape capture.

TASK:
1. /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('compile supported:', hasattr(torch, 'compile'))
" 2>&1
2. Read molmetal/scripts/train_dmpnn_multitask_wasserstein.py — check if torch.compile is used
3. Read molmetal/adapters/flow_matching_lipman/ — check if torch.compile is used
4. Identify candidates for torch.compile:
   - D-MPNN message passing (static shape)
   - Wasserstein loss (static shape)
   - Bond head (static shape)
   - Aggregate (static shape)
5. Write 8 NEW tests: molmetal/tests/test_wf326c_torch_compile.py
   - 3 tests: per-component compile correctness
   - 2 tests: CUDA Graphs capture
   - 2 tests: speedup measurement (baseline vs compiled)
   - 1 test: integration with downstream consumer
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf326c_torch_compile.py -v 2>&1 | tail -20
7. Save optimized trainer: molmetal/scripts/wf326c_torch_compile_optimized.py
8. Document: TODO/completed/326c_torch_compile_optimization.md

Honest framing:
- torch.compile can give 1.5-3× speedup for static-shape models
- CUDA Graphs capture for repeated small batch sizes
- D-MPNN message passing is static shape (good candidate)
- CPU-only profile (no GPU contention with G6)

OUTPUT (PLAIN TEXT): per_component_compile_status + per_component_speedup + total_speedup + preflight_status + per_test_status. Do NOT call StructuredOutput.`, {
    phase: 'GPU utilization optimization (parallel)',
    label: 'wf326C',
  }),

  () => agent(`Wf-326D: Mixed precision (BF16) + larger batch size optimization.

USE Bash + Read + Edit.

CONTEXT: Per molmetal-gpu-execution: MOLMETAL_CFM_AMP_DTYPE=bfloat16 is default. Need to verify D-MPNN trainer uses BF16 + check if larger batch size is feasible.

TASK:
1. Read molmetal/scripts/train_dmpnn_multitask_wasserstein.py — check AMP / BF16 usage
2. /home/hugo/cades/CadButEaas/.venv-vlm/bin/python -c "
import torch
# Test BF16 capability
print('BF16 supported:', torch.cuda.is_bf16_supported())
print('GPU 1 memory total:', torch.cuda.get_device_properties(1).total_memory / 1e9, 'GB')
print('GPU 1 memory free:', torch.cuda.mem_get_info(1)[0] / 1e9, 'GB')
" 2>&1
3. Identify AMP / BF16 status:
   - Per MOLMETAL_CFM_AMP env var
   - Per script AMP_FLAG
4. Larger batch size optimization:
   - Current batch_size=256
   - Try 512 / 1024 / 2048 on RTX 5090 (32 GB)
   - Find max batch size that fits
5. Write 8 NEW tests: molmetal/tests/test_wf326d_amp_batch_optimization.py
   - 3 tests: BF16 forward/backward correctness
   - 2 tests: batch size sweep (256/512/1024/2048)
   - 2 tests: memory utilization
   - 1 test: integration with downstream consumer
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf326d_amp_batch_optimization.py -v 2>&1 | tail -20
7. Save optimization script: molmetal/scripts/wf326d_amp_batch_optimized.py
8. Document: TODO/completed/326d_amp_batch_optimization.md

Honest framing:
- BF16 typically gives 1.5-2× speedup with minimal accuracy loss
- Larger batch size reduces step overhead but increases memory
- RTX 5090 32 GB: PROJECTED max batch ~2048 for hidden=300/depth=3
- CPU-only profile (no GPU contention with G6)

OUTPUT (PLAIN TEXT): bf16_status + max_batch_size + throughput_before_after + preflight_status + per_test_status. Do NOT call StructuredOutput.`, {
    phase: 'GPU utilization optimization (parallel)',
    label: 'wf326D',
  }),

  () => agent(`Wf-326E: Triton kernel audit — verify what is already optimized vs what is missed.

USE Bash + Read + Edit.

CONTEXT: Per user feedback: "之前参考的那些源代码的潜力估计还是没有被榨干" + "本来就做过triton优化了". Need to audit Triton kernel coverage.

TASK:
1. Read molmetal/adapters/flow_matching_lipman/_reference/ — find Triton kernels
   grep -rn "triton" /home/hugo/codes/mol-metal/molmetal/adapters/ 2>&1 | head -20
2. /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
import triton
print('Triton:', triton.__version__)
" 2>&1
3. Identify Triton kernel coverage:
   - Which kernels are present?
   - Which hot-path operations are MISSING Triton kernels?
   - Where are the gaps?
4. Identify candidates for NEW Triton kernels:
   - D-MPNN message passing edge updates
   - Wasserstein sliced distance (per Wf-310C TODO-179)
   - Bond head aggregation
   - SMILES featurization (if GPU-compatible)
5. Write 8 NEW tests: molmetal/tests/test_wf326e_triton_audit.py
   - 3 tests: existing Triton kernel coverage verification
   - 3 tests: missing Triton candidate analysis
   - 2 tests: speedup measurement (CPU vs Triton)
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf326e_triton_audit.py -v 2>&1 | tail -20
7. Save audit: molmetal/reports/wf326e_triton_audit/audit.json
8. Document: TODO/completed/326e_triton_audit.md

Honest framing per party-twenty-spirit §3.5:
- Audit identifies WHICH kernels exist + WHICH are missing
- Per-user feedback: even with Triton optimization done, gaps remain
- Triton kernel speedup: PROJECTED 2-10× for hot-path operations
- CPU-only audit (no GPU contention with G6)

OUTPUT (PLAIN TEXT): n_existing_kernels + n_missing_candidates + per_kernel_status + projected_speedup + preflight_status. Do NOT call StructuredOutput.`, {
    phase: 'GPU utilization optimization (parallel)',
    label: 'wf326E',
  }),

  () => agent(`Wf-326F: SMILES featurization cache effectiveness — verify PerEpochFeatureCache is actually being hit (Wf-320D follow-up).

USE Bash + Read + Edit.

CONTEXT: Per Wf-320D: TODO-155 §3 chemprop v2 featurization fix. PerEpochFeatureCache + FastAtomFeaturizer implemented. Per Wf-320D smoke: cache hit rate 0% on 1-epoch (no cross-epoch reuse). PROJECTED 98% on 50-epoch run.

TASK:
1. Read molmetal/scripts/wf320d_featurization_fix.py — PerEpochFeatureCache impl
2. Read molmetal/scripts/train_chemprop_v2_v2_fastfeaturize.py — smoke trainer
3. Read molmetal/reports/wf320d_featurization_fix_smoke/Ru_seed0.json — smoke result
5. Verify cache effectiveness via G6 training (if G6 uses chemprop v2):
   - G6 uses D-MPNN (train_dmpnn_multitask_wasserstein.py) NOT chemprop v2
   - Chemprop v2 featurization is for chemprop trainer only
   - D-MPNN has its own featurization
6. Implement PerEpochFeatureCache for D-MPNN featurization (CPU-bound currently):
   - Add cache layer for D-MPNN atom/bond features
   - Verify cache hit on cross-epoch reuse
7. Write 8 NEW tests: molmetal/tests/test_wf326f_dmpnn_featurization_cache.py
   - 3 tests: cache hit/miss
   - 2 tests: per-epoch reset
   - 2 tests: integration with D-MPNN trainer
   - 1 test: speedup benchmark
8. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf326f_dmpnn_featurization_cache.py -v 2>&1 | tail -20
9. Save: molmetal/reports/wf326f_dmpnn_featurization_cache/result.json
10. Document: TODO/completed/326f_dmpnn_featurization_cache.md

Honest framing:
- Wf-320D cache was for chemprop v2 only
- D-MPNN (G6 trainer) has separate featurization — also CPU-bound
- Adding D-MPNN cache could lift GPU util 24% → 50%+
- CPU-only work (no G6 interference)

OUTPUT (PLAIN TEXT): cache_hit_rate + throughput_before_after + gpu_util_estimate + preflight_status + per_test_status. Do NOT call StructuredOutput.`, {
    phase: 'GPU utilization optimization (parallel)',
    label: 'wf326F',
  }),

  () => agent(`Wf-326G: Reference code (Chemprop v2 official + D-MPNN paper) deep dive — identify missed optimizations.

USE Bash + Read + Edit.

CONTEXT: Per user feedback: "之前参考的那些源代码的潜力估计还是没有被榨干". Need to identify optimizations in reference code that haven't been ported.

TASK:
1. Read Chemprop v2 official repo (if accessible):
   find / -name "chemprop" -type d 2>/dev/null | head -5
2. Read D-MPNN paper implementation notes:
   molmetal/models/dmpnn.py — current D-MPNN impl
3. Identify missed optimizations:
   - Atom-level feature caching (vs molecule-level)
   - Edge update batching
   - GraphSAGE-style neighbor sampling (vs full batch)
   - DistributedDataParallel (DDP) for multi-GPU
   - Mixed precision autocast (vs manual AMP)
   - Gradient checkpointing for memory savings
4. Write NEW optimization plan: molmetal/reports/wf326g_missed_optimizations/plan.md
   - §1: Chemprop v2 official repo optimizations
   - §2: D-MPNN paper optimizations
   - §3: PyTorch 2.x native optimizations
   - §4: Per-optimization speedup estimate
5. Write 8 NEW tests: molmetal/tests/test_wf326g_missed_optimizations.py
   - 3 tests: per-optimization correctness
   - 3 tests: speedup benchmark
   - 2 tests: integration
6. Run: PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf326g_missed_optimizations.py -v 2>&1 | tail -20
7. Document: TODO/completed/326g_missed_optimizations_plan.md

Honest framing:
- Reference code has optimizations not yet ported
- Per optimization: PROJECTED speedup estimate
- Some optimizations may require GPU verification (out of scope for CPU)
- This is AUDIT work, not implementation work

OUTPUT (PLAIN TEXT): n_optimizations_identified + per_optimization_status + total_speedup_estimate + preflight_status + per_optimization_priority. Do NOT call StructuredOutput.`, {
    phase: 'GPU utilization optimization (parallel)',
    label: 'wf326G',
  }),
])

phase('Synthesis')

const wf326H = await agent(`Wf-326H: Aggregate Wf-326 GPU utilization optimization outcomes — final optimization report + speed-up proposal.

USE Read only.

UPSTREAM RESULTS:
- Wf-326A: GPU utilization profile
- Wf-326B: DataLoader optimization
- Wf-326C: torch.compile + CUDA Graphs
- Wf-326D: Mixed precision + batch size
- Wf-326E: Triton kernel audit
- Wf-326F: D-MPNN featurization cache
- Wf-326G: Reference code missed optimizations

TASK:
1. Read all 7 outputs
2. Aggregate: completed + failed + partial + projected speedup per optimization
3. Final optimization report: molmetal/reports/wf326_final/optimization_report.md
   - §1: Current state (24% GPU util, 70% idle capacity)
   - §2: Bottleneck decomposition
   - §3: Proposed optimizations + projected speedup per item
   - §4: Implementation roadmap (CPU-only vs GPU portion)
   - §5: 5-10× speedup realistic target
4. Verify G6 still running
5. Document: TODO/completed/326_final_synthesis.md

OUTPUT (PLAIN TEXT): per_optimization_status + per_optimization_projected_speedup + total_projected_speedup + current_gpu_util_24pct_to_target + G6_status + implementation_roadmap (12-18 lines). Do NOT call StructuredOutput.`, {
  phase: 'Synthesis',
  label: 'wf326H',
})

return {
  ultracode: 'wf-326-gpu-utilization-optimization',
  wf326a: wf326A,
  wf326b: wf326B,
  wf326c: wf326C,
  wf326d: wf326D,
  wf326e: wf326E,
  wf326f: wf326F,
  wf326g: wf326G,
  wf326h: wf326H,
}