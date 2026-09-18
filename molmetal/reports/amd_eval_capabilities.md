# Actual AMD evaluation capability — 2026-09-13

Commands and raw results are in `amd_eval_capabilities.json` and can be
reproduced with:

```bash
HIP_VISIBLE_DEVICES=0 uv run python -m molmetal.scripts.verify_amd_eval_capabilities
```

No dependencies were installed for this audit. Each native backend ran in
a separate process, and all three processes exited 0.

| Path | Executed evidence | Conclusion |
|---|---|---|
| Project ADMET-AI runner | CCO produced 104 properties; forward hooks on all 10 neural ensemble calls recorded parameters on `cuda:0`; HIP `7.2.53211`, architecture `gfx1101` | Actual AMD ROCm learned evaluation works |
| OpenMM 8.6.1 | Explicit OpenCL context, AMD Accelerated Parallel Processing platform, device `gfx1101`; five integration steps changed harmonic potential from 0.50000006 to 0.49987501 kJ/mol | Actual AMD OpenCL dynamics and energy execution works |
| Installed GNINA 1.3.2 | Real test_001 receptor/reference-ligand score-only run returned affinity −7.99492 and CNN scores, while explicitly printing `WARNING: No GPU detected` | Working CPU scoring; no AMD GPU acceleration established |
| Vendored QuickVina 2 | Static CPU binary, already validated with real receptor/ligand docking | Working CPU docking, not GPU docking |
| AutoDock-GPU/Vina-GPU/QuickVina2-GPU | No executable found on PATH or inspected reference/storage locations; SynFlowNet has a wrapper pointing to absent `bin/QuickVina2-GPU-2-1` | No installed GPU docking implementation validated |

The selected PyTorch device has `gcnArchName=gfx1101` and 17,163,091,968 bytes
of VRAM. This matches the RX 7800 XT shown by ROCm SMI. PyTorch's generic
`AMD Radeon Graphics` name is not evidence that this is the 780M iGPU;
the separate 780M is gfx1103. Older reports conflating 780M and gfx1101
should not be used as device provenance.

OpenMM exposes Reference, CPU and OpenCL here, with no plugin-load failures.
Its OpenCL capability does not imply a validated molecular force field,
metal parameterization, receptor minimization, or GPU docking pipeline.
The harmonic-system result intentionally tests only backend execution.

The GNINA launcher configures a local CUDA runtime for `gnina.real`; the
underlying binary links CUDA/cuDNN, not HIP. Successful `--version` or its
CPU scoring path is not evidence of ROCm compatibility. The probe did not
use `--no_gpu`; the program itself detected no supported GPU and fell back.
GNINA's reference-ligand score-only result is not compared with generated
CCO docking or presented as an independent Vina parity measurement.

The current practical AMD paths are learned PyTorch evaluation and OpenMM
OpenCL energy/dynamics. Physical docking should retain the verified CPU
QuickVina path until a separate GPU docking implementation is installed,
tested on this AMD device, and validated against the evaluation protocol.
