# Actual AMD OpenCL docking smoke (2026-09-13)

## Goal

Build an official OpenCL docking engine outside the core project environment and run the exact test_001 CrossDocked receptor with a real CCO ligand. Verify device kernel execution, output coordinates and pose checks. This follows the earlier installed-capability inventory, which found no AMD GPU docking binary.

## Outcome

**PASS: QuickVina2-GPU 2.1 performs real GPU grid/search on AMD gfx1101.** The fixed official source is `DeltaGroupNJUPT/Vina-GPU-2.1` commit `180272b8a5265d6ed9664178345933cebe2cd349`. Archive SHA256: `c5d8ca922f6ca4130dcc809aa620e1120be24306ca475046faef5ab7ccbf6aa5`. Its C++ host code builds with system Boost 1.92 and OpenCL, without installing CUDA or changing the project dependency lock.

Actual run: seed 42, 1,000 search lanes, depth 1, one output mode, original test_001 box center/size, strictly Meeko-prepared original receptor. Exit code 0; CCO affinity **−2.3 kcal/mol**; three finite heavy-atom coordinates. The actual output was reconstructed with Meeko into `CCO_gpu_out.sdf`, preserving its conformer. PoseBusters 0.6.5 `config='dock'` passed all 22 reported checks with the original manifest receptor (no missing or skipped checks).

An independent `LD_PRELOAD` observer wraps the real `clEnqueueNDRangeKernel`, waits for its returned event and queries the command queue's device. It confirms:

| Kernel | Queue device | Enqueue | Wait | Completion status |
|---|---|---:|---:|---:|
| kernel1 (grid) | gfx1101 | 0 | 0 | CL_COMPLETE (0) |
| kernel2 (search) | gfx1101 | 0 | 0 | CL_COMPLETE (0) |

Full evidence: `run_traced.log`, `report.json`, `CCO_gpu_out.pdbqt`, `CCO_gpu_out.sdf`, `posebusters.json`. The native C++ engine performs CPU input processing and final refinement after GPU grid/search; this is a mixed CPU/GPU docking pipeline.

## Caveats

The AMD driver's event profiling timestamps are invalid for both completed kernels (`end < start`). Upstream unsigned subtraction consequently prints an impossible ~18.4-billion-second runtime. We retain that failure in `gpu_runtime_invalid.log` and explicitly reject it as performance evidence. An untraced startup-inclusive host run took about 2.238 seconds, but one CCO smoke cannot establish throughput, acceleration or druglike docking quality. The observer synchronizes kernel launches and is for execution evidence only.

The GPU engine's `thread`/`search_depth` settings are distinct from native Vina `exhaustiveness`; equal affinity in this N=1 case establishes no parity or score correlation. Native CPU benchmarks are unchanged. GNINA remains a CUDA build that falls back to CPU on this machine; this successful official OpenCL engine is separate.

Two minimal compatibility changes are retained in `source_compatibility.patch`: replace a removed Boost convenience header with the umbrella header, and exclude C-string terminators from the OpenCL concatenated source length. The latter fixes an actual overread/extra-NUL build error rejected by AMD clang with `-Werror`. No scoring/search code or kernel math was modified. All failed logs are retained, including the initial unsupported `cpu` option.

## Reproduce

The currently validated executable is:

```text
/mnt/storage/tools/vina_gpu21_source/Vina-GPU-2.1-180272b8a5265d6ed9664178345933cebe2cd349/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1
```

Build independently from the pinned official archive (download, checksum, patch, compilation and optional trace library):

```bash
bash molmetal/reports/amd_gpu_docking_build/reproduce_build.sh /mnt/storage/tools/vina_gpu21_reproduce
```

The commands within this script were executed successfully during the recorded build; `bash -n` also passed. The script defaults to a separate destination to avoid replacing an in-use executable. Its final source build uses `AMD_PLATFORM`, `OPENCL_2_0`, `CL_TARGET_OPENCL_VERSION=200`, `DISPLAY_ADDITION_INFO`, and `TIME_ANALYSIS`. Build commands, errors and warnings are in the three `build_*.log` files.

Run the validated binary from its own directory:

```bash
cd /mnt/storage/tools/vina_gpu21_source/Vina-GPU-2.1-180272b8a5265d6ed9664178345933cebe2cd349/QuickVina2-GPU-2.1
ulimit -s 8192
GPU_DEVICE_ORDINAL=0 HIP_VISIBLE_DEVICES=0 ./QuickVina2-GPU-2-1 \
  --config /home/hugo/codes/try_triton_on_rocm/molmetal/reports/amd_gpu_docking_build/smoke_config.txt
```

To capture independent kernel evidence, prepend `LD_PRELOAD=/mnt/storage/tools/vina_gpu21_source/trace_opencl.so` to the environment assignments. The C source and its build command are included in this directory and `reproduce_build.sh`.

The source build reads `./OpenCL` and writes `Kernel1_Opt.bin` / `Kernel2_Opt.bin`; `opencl_binary_path` must point to that cache directory. Concurrent invocations need isolated working directories (each containing an OpenCL source directory or symlink) or a lock. Seeds, complete configuration, input/output/kernel/binary hashes are in `seed.json` and `report.json`.
