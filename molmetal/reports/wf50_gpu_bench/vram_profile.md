# Wf-50 Phase 3 — GPU Wrapper VRAM Profile

_Generated UTC: 2026-09-26T08:13:11.648071+00:00_

Per-wrapper VRAM ceiling (max peak across all datasets + batch sizes) and per-dataset VRAM peak. **VRAM ceiling must stay ≤ 16 GB for RTX 5090 fit.** All wrappers honour the audit's expected_vram_mb cap; rows where peak_vram_mb is 0.0 mean the wrapper did not successfully load on this host (AdapterUnavailable).

| Wrapper | Metric | Peak VRAM (MB) | Per-dataset peak (MB) | Batch sizes covered |
| --- | --- | --- | --- | --- |
| boltz_batch | boltz_affinity | n/a | metalcytotox_5metal (0.0 MB), platinai_5k (0.0 MB) | [1, 16, 64, 256, 1024] |
| diffdock_batch | diffdock_pose | n/a | metalcytotox_5metal (0.0 MB), platinai_5k (0.0 MB) | [1, 16, 64, 256, 1024] |
| hilbert_astsr_cuda | hilbert_astsr_optimize | 0.8 | metalcytotox_5metal (0.8 MB), platinai_5k (0.8 MB) | [1, 16, 64, 256, 1024] |
| pb_batch | pb_pass_rate | n/a | metalcytotox_5metal (0.0 MB), platinai_5k (0.0 MB) | [1, 16, 64] |
| vina_gpu | vina_score | n/a | metalcytotox_5metal (0.0 MB), platinai_5k (0.0 MB) | [1, 16, 64, 256, 1024] |

