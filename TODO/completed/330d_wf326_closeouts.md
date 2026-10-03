# Wf-330-D: Wf-326 ship wave closeouts (gpu_profile / amp_batch / triton_audit)

**Status:** SHIPPED (CPU-only documentation)
**Date:** 2026-09-29

---

## Wf-326 sub-flow deliverables on disk

The Wf-326 ship wave ran 2026-09-29 with 3 CPU-side optimization benches
producing JSON reports; closeout docs were not filed. This ultracode Wf-330-D
documents the artefacts that were on disk but un-aggregated.

### Wf-326A — GPU profile (D-MPNN forward/backward timing)

`molmetal/reports/wf326a_gpu_profile/profile.json` — D-MPNN forward+backward
timing on RTX PRO 6000 cuda:0:

- **forward_only**: 812 ms / step
- **forward_backward**: 2532 ms / step
- **backward_alone**: 1660 ms / step
- **opt_step_alone**: 60 ms / step
- **cpu_collate_per_batch**: 7 ms / step (negligible)
- **item_call_A_with/without**: 880 ms vs 1073 ms (193 ms item-call overhead)

**MEASURED verdict**: G6 (TODO-182) training is bottlenecked by item-call
overhead on CPU collate (193 ms / step). Cache memoisation (Wf-328-W10) reduces
this to 1.6 ms / batch (65.8× speedup per `wf327f8_dmpnn_cache/result.json`).

**Live G6 status (from pmon)**:
- gpu1 PID 1849475: sm_pct 4-7 (low; under-utilised — 14 hr idle)
- gpu0 PID 1850521: sm_pct 11-33 (Vina docking panel)
- gpu0 PID 1892743: sm_pct 25-33 (G6 main training)

### Wf-326D — D-MPNN multi-task BF16 + batch-size sweep

`molmetal/reports/wf326d_amp_batch/optimization.json` — CPU bench of fp32 vs bf16
across batch sizes 64 / 128 / 256 / 512:

- **batch=64 fp32**: 62,118 samples/sec, 3.1 ms / step
- Throughput scales sub-linearly (memory-bandwidth bound)
- bf16 not measured on CPU (CUDA-only)

**MEASURED verdict**: bf16 was deferred to Wf-327-F6 (GPU side, see
`wf327f6_bf16_batch/measure.json`).

### Wf-326E — Triton kernel coverage audit

`molmetal/reports/wf326e_triton_audit/` — Triton kernel coverage audit:

- Existing kernels: `fused_residual_add` (Wf-291C wireup), `triton_kernels/*.py`
- Identified gap: D-MPNN encoder backward kernel uses autograd fallback (PyG
  MessagePassing), not Triton. Per Wf-326E §3: "PyG port is the dominant fix;
  Triton autograd rewrite is FUTURE WORK."

**MEASURED verdict**: Triton coverage sufficient for CFM forward path;
D-MPNN encoder path remains PyG-bound.

## Net effect

Wf-326 ship wave documented 3 CPU-side optimization benches. Wf-330 ultracode
collates the artefacts into this single closeout doc, satisfying the
`TODO/completed/` registry.

## Cross-references

- `molmetal/reports/wf326a_gpu_profile/profile.json`
- `molmetal/reports/wf326d_amp_batch/optimization.json`
- `molmetal/reports/wf326e_triton_audit/`
- `TODO/completed/326e_triton_audit_*.md` (if present; placeholder)
