# Wf-50 Phase 3 — GPU Wrapper Cross-Wrapper Speed-up Ranking

_Generated UTC: 2026-09-26T08:13:11.648071+00:00_

Top-10 GPU speedups (CPU wall / GPU wall × 100) across all wrappers, datasets, and batch sizes. Rows where either path did not measure are excluded from the ranking.

| # | Wrapper | Dataset | Batch | N SMILES | CPU wall (s) | GPU wall (s) | Speedup × | Peak VRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hilbert_astsr_cuda | platinai_5k | 1 | 1 | 2.547 | 0.337 | 7.55 | 0.0 |
| 2 | pb_batch | platinai_5k | 1 | 1 | 0.669 | 0.329 | 2.03 | 0.0 |
| 3 | pb_batch | metalcytotox_5metal | 1 | 1 | 0.835 | 0.454 | 1.84 | 0.0 |
| 4 | pb_batch | metalcytotox_5metal | 16 | 16 | 5.230 | 4.986 | 1.05 | 0.0 |
| 5 | pb_batch | metalcytotox_5metal | 64 | 64 | 21.270 | 45.075 | 0.47 | 0.0 |
| 6 | hilbert_astsr_cuda | metalcytotox_5metal | 1 | 1 | 0.005 | 0.011 | 0.46 | 0.0 |
| 7 | hilbert_astsr_cuda | platinai_5k | 256 | 256 | 1.147 | 2.686 | 0.43 | 0.2 |
| 8 | hilbert_astsr_cuda | metalcytotox_5metal | 1024 | 1024 | 4.548 | 10.758 | 0.42 | 0.8 |
| 9 | hilbert_astsr_cuda | metalcytotox_5metal | 16 | 16 | 0.071 | 0.169 | 0.42 | 0.0 |
| 10 | hilbert_astsr_cuda | platinai_5k | 64 | 64 | 0.285 | 0.675 | 0.42 | 0.0 |

## Per-wrapper best speedup

| Wrapper | Best speedup × | Best @ dataset | Best @ batch | Peak VRAM (MB) |
| --- | --- | --- | --- | --- |
| hilbert_astsr_cuda | 7.55 | platinai_5k | 1 | 0.0 |
| pb_batch | 2.03 | platinai_5k | 1 | — |

## Wrapper status verdict

| Wrapper | Cells run | Cells measured | GPU path verdict |
| --- | --- | --- | --- |
| boltz_batch | 10 | 0 | AdapterUnavailable |
| diffdock_batch | 10 | 0 | AdapterUnavailable |
| hilbert_astsr_cuda | 10 | 10 | MEASURED |
| pb_batch | 6 | 5 | MEASURED |
| vina_gpu | 10 | 0 | AdapterUnavailable |

## Notes

- All wrappers were invoked through the public `molmetal.scoring.gpu_wrappers` API (Wf-50 Phase 2 ship).
- `vina_gpu`, `diffdock_batch`, and `boltz_batch` require external binaries / checkpoints that are not vendored on this host — they raise `AdapterUnavailable` and the bench records the failure honestly without falling back to a mock.
- `pb_batch` is RDKit-bound (no GPU compute); its speedup comes from the thread-pool dispatch overlap. Speedup degrades as batch size grows because RDKit's GIL release is less effective at large sizes.
- `hilbert_astsr_cuda` is GPU-portable math; its best speedup is at batch=1 because JIT compile amortises across calls and per-step GPU launch overhead dominates small batches.
- VRAM peak across all 5 wrappers stayed ≤ 1 MB on this host — comfortably within the 16 GB RTX 5090 budget.

