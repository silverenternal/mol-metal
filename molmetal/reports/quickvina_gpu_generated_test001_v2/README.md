# Generated-product QuickVina2-GPU adapter smoke

2026-09-13. The independent `QuickVinaGPUAdapter` docks the first generated
`test_001`, seed 42 click product, `OCCOCCn1nncc1Cc1ccccc1` (18 heavy atoms),
against its exact CrossDocked manifest receptor. Generation is reused from the
recorded click-tile search and is independent of this pocket; this is a bounded
adapter integration check, not evidence of pocket-conditioned generation or a
SOTA benchmark.

The saved GPU pose scores **−6.2 kcal/mol**, retains the product's complete
molecular graph and the engine's coordinates, and passes all **22 PoseBusters
`dock` checks**. The domain molecule, adapter RDKit pose and saved SDF coordinates
agree within the relevant float32/SDF serialization tolerances. `report.json`
records input, adapter/runtime, binary, OpenCL source and trace-library hashes.

The engine runs QuickVina2-GPU 2.1, official commit
`180272b8a5265d6ed9664178345933cebe2cd349`, with the environment agent's documented
Boost-header and source-length compatibility fixes. The executable SHA256 is
`5bd58c41f2c6e5bc77a9ef47e7a7daf2b7edd17c8924d80d2e3dd6408f1101ca`.
Build provenance and the reproducible build script are in
`../amd_gpu_docking_build/`.

GPU execution is observed on AMD OpenCL device `gfx1101`: both `kernel1` and
`kernel2` have `enqueue=0 wait=0 status=0` in the retained `engine.log`.
The explicit budget is **1000 computing lanes, search_depth 1, one output pose**.
The native `DockingConfig.exhaustiveness` field is ignored and recorded as such;
there is no claim of equivalent search effort to native Vina/QuickVina.
The seed is requested 42, effective 42 and verified against the engine log.

RDKit/Meeko preparation, final engine pose refinement and PoseBusters retain their
native CPU chemistry. This adapter has no CPU docking fallback. Each invocation
has a unique working directory and kernel cache, so shared OpenCL source is not
mutated. It requires a passed prepared receptor and does not synthesize receptor
chemistry from pocket atom types.

The complete second smoke took **3.680 s** wall time. This is not a performance
comparison: the OpenCL tracer synchronizes each kernel, and this driver's
profiling end/start timestamps are invalid. Reported adapter timing uses Python
monotonic wall time. The first successful smoke remains in
`../quickvina_gpu_generated_test001/`; this second run additionally records
OpenCL source and trace-library hashes after the metadata addition.

Reproduce from the repository root:

```sh
uv run python -m molmetal.scripts.smoke_vina_gpu_generated \
  --binary /mnt/storage/tools/vina_gpu21_source/Vina-GPU-2.1-180272b8a5265d6ed9664178345933cebe2cd349/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1 \
  --trace-library /mnt/storage/tools/vina_gpu21_source/trace_opencl.so \
  --output-dir molmetal/reports/quickvina_gpu_generated_test001_reproduce
```

Validation: **17 focused tests passed**, covering GPU failure/timeout/device
verification, seed logs and zero-seed mapping, independent GPU budgets, cache
isolation, required prepared receptor, completed kernel traces, and inherited
pose chemistry/energy alignment:

```sh
uv run pytest -q molmetal/tests/test_vina_gpu_adapter.py \
  molmetal/tests/test_docking_pose_chemistry.py \
  molmetal/molmetal_lam/tests/test_vina_seed.py
```

Remaining limits: one product and one pocket, low search depth, one GPU model;
the adapter does not establish population validity, native-engine parity,
statistical improvement, experimentally measured affinity or a GPU speedup.
