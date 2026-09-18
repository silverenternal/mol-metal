# Project development workflow

This repository has two implementation lines that share a core ROCm environment;
optional tools can use separately locked environments when their dependencies differ:

- **MolFlow/model line:** `models/`, `flow_matching/`, `triton_kernels/`, root `data/` and training scripts.
- **molmetal/Lambda line:** `molmetal/`, metal priors, SBDD adapters, scoring and evaluation.

## Before changing code

1. Use `uv run` so Python 3.12 and the locked dependencies are selected.
2. Run `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip)"`.
3. Inspect `/mnt/storage/data/molmetal` before declaring a dataset or checkpoint missing. The CrossDocked2020 archive and 100/100,000 pair split are staged there.
4. Read the relevant pending task, `pending/risks.md`, `pending/decisions.md`, and `engineering_practices.md`.
5. Classify the work as model, Lambda, environment, or external-data blocked.

## Validation scopes

- Model smoke: `uv run pytest -q tests triton_kernels/tests` and, when GPU is available, `uv run python main.py`.
- Lambda unit scope: target `molmetal/tests`, `molmetal/eval/scoring/tests`, and the relevant package tests. Exclude cloned reference tests unless their optional dependencies are installed.
- Environment: `uv lock --check`, import checks, and the smallest failing command reproduced under `uv run`.

Record whether a result is a smoke test, unit test, or paper-grade evaluation, including device, seed, split, and external-tool availability.

## GPU policy

Torch workloads must resolve their device through the project helper or an
explicit `--device`; the default is ROCm/CUDA `cuda:0` when visible. Run with
`HIP_VISIBLE_DEVICES=0` on the RX 7800 XT when multiple adapters are present.
Use GPU implementations where they preserve the intended computation. Verified paths include torch/POT Sinkhorn, learned ADMET inference, chunked pairwise Tanimoto, OpenMM OpenCL dynamics, and official REINVENT prior generation/NLL on gfx1101. RDKit fingerprint/chemistry preparation and native Vina/QuickVina remain CPU; test alternative AMD backends explicitly before claiming GPU support. GNINA's installed CUDA build reports no GPU and its CPU fallback does not count as AMD acceleration.

## REINVENT environment and scoring boundaries

- Installation and real pretrained prior generation/NLL are complete. Use the independent managed project `/mnt/storage/env-projects/reinvent4-rocm`, which explicitly overrides upstream's torch 2.12 pin with ROCm torch 2.14 and matching torchvision 0.29. Its lock also includes the omitted SciPy dependency. Keep the main project and original `/mnt/storage/envs/reinvent4` CUDA environment separate.
- The verified official prior is `/mnt/storage/models/reinvent4/reinvent_v4.4.22.prior`. Read [the evidence and reproducible commands](../molmetal/reports/reinvent4_learned_smoke/README.md) for its fixed official commit/blob/SHA256 and environment manifests. Do not declare the prior missing solely because current upstream `priors/` is absent or Zenodo is unavailable.
- Run `HIP_VISIBLE_DEVICES=0 uv run --frozen --project /mnt/storage/env-projects/reinvent4-rocm python /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/verify_reinvent_learned_inference.py --device cuda:0 --out-dir /home/hugo/codes/try_triton_on_rocm/molmetal/reports/NEW_REINVENT_PREFIX` for a bounded five-molecule official-CLI/likelihood check. Record neural parameter/input devices and actual gfx1101 architecture.
- The legacy prior emits an internal metadata xxhash warning in the current runtime; retain it and verify external file hashes. The validated file was unchanged and matched the official Git blob. Do not rewrite metadata to hide the warning.
- Learned prior NLL describes the model's SMILES sequence likelihood, including sequence-length/representation dependence. It does not prove activity, synthesis, novelty or calibrated multiproperty desirability. The existing JSONL worker remains an explicitly labelled RDKit proxy, and task-specific learned multiproperty integration remains pending in [task 05](pending/05_reinvent4_install.md).
- Use the independent `REINVENT4PriorAdapter` with `molmetal/configs/reinvent_prior_amd.json` for learned RPC (`mode='prior_nll'`); [its report](../molmetal/reports/reinvent_prior_adapter_amd/README.md) records total request deadlines/process-group cleanup and the real AMD batch. It preserves raw NLL, token count, all invalid rows, checkpoint SHA and actual neural device; equal-length buckets avoid upstream EOS-padding dependence. Do not place NLL into the existing clipped `[0,1]` `qed/sa/binding/novelty` fields or silently change experiment reward semantics. The standard TOML CLI is not a JSONL RPC worker.

## TODO transitions

An item moves from `TODO/pending/` to `TODO/completed/` only after its acceptance criteria and focused validation are recorded in a dated completion note. Keep unresolved external blockers in `pending/risks.md` with the failing command and fallback. Update `pending/roadmap.md` when a phase status changes.

## Workstream boundary

Do not mix model-line and Lambda-line refactors in one task. Shared interfaces may be changed only with tests on both sides. Optional external references must not be imported during ordinary test collection.

## Paired diagnostic execution and recovery (2026-09-13 correction)

- Validate all test pairs with `uv run python molmetal/scripts/stage_crossdocked100.py --dry-run`; this must not modify the manifest. Regenerate only from verified local files.
- `r4_c_full_sweep.py` selects explicit manifest pairs, never alphabetically sorted data directories. It accepts `--seeds 42 0 1234` and logs real effective expansion counts.
- Use `--job-timeout` (maximum 600 seconds) for isolated jobs. A timeout kills the worker process group and produces a failure row while subsequent independent jobs proceed.
- Every job checkpoints JSON before CSV/Markdown. `--append` preserves all metrics and skips completed pocket/seed keys only when config, source, manifest and input hashes match. Use a new output prefix for a changed protocol.
- Inspect `n_generated_candidates`, `n_seed_candidates`, and reduction diagnostics. Returning the reference ligand is `seed_only`, not a generated molecule.
- Select `--seed-strategy click_tile` for declared tile initialization without paired-ligand chemistry; `reference` remains a labelled diagnostic. `--physical-docking` runs actual docking/PoseBusters separately from search proxies. The search is still pocket-independent until a real pocket-conditioned reward is connected. JSON distinguishes requested and executed guidance settings.
- QuickVina 2 is the verified vendored `qvina02` executable; `qvina` and `quickvina2` can name the same physical engine. Use the real adapter smoke script for environment checks, and compare against Vina as a separate engine only with matched inputs and settings.

## Physical evaluation and agent coordination

- Freeze runtime code while a multi-job experiment runs. The driver records runtime hashes and stops before the next job if dependencies change; keep partial evidence and choose a new prefix for a changed implementation.
- Workers checkpoint generated candidates before physical evaluation and after each completed pose. A physical timeout retains completed records, marks the stage interrupted, and allows independent jobs to continue.
- `prepare_receptor_for_evaluation` preserves source PDBs. Its report distinguishes strict preparation, explicit residue-level altloc selection and verified equivalent O/OXT naming repair. PoseBusters reads the returned `effective_pdb`.
- `check_docked_pose` consumes the actual single docked conformer with config `dock`; save that same RDKit pose as SDF. Never establish docking validity from newly embedded SMILES.
- Include all generated products in coverage/pass denominators; finite-value descriptor/energy means additionally report their measured sample count. Show redocked-reference thresholds explicitly, and never test statistical significance against cite-only aggregate scores.
- The original test_005 / 4RN0 ASP B101 lacks CG/OD1/OD2 in the crystal (REMARK 470). Any computational completion is a separate modeled-receptor protocol with new-atom provenance, not observed structure recovery.

Bounded physical integration example (not canonical SOTA parameters):

```sh
HIP_VISIBLE_DEVICES=0 uv run python molmetal/scripts/r4_c_full_sweep.py --config molmetal/configs/lambda_click_physical_smoke.yaml --n-pockets 10 --seeds 42 0 1234 --seed-strategy click_tile --physical-docking --physical-engine quickvina2 --physical-exhaustiveness 1 --physical-n-poses 1 --physical-top-k 3 --job-timeout 120 --output-prefix molmetal/reports/NEW_UNIQUE_PREFIX
HIP_VISIBLE_DEVICES=0 uv run python -m molmetal.scripts.summarize_physical_sweep molmetal/reports/NEW_UNIQUE_PREFIX.json --output-prefix molmetal/reports/NEW_UNIQUE_PREFIX_analysis --device cuda:0
```

## Frozen experiment snapshots and measured reward controls

- When agents continue editing unrelated runtime modules, create a separate
  experiment copy with `uv run python molmetal/scripts/snapshot_experiment_runtime.py /mnt/storage/experiments/NEW_RUN/runtime`.
  The manifest verifies copied code against the live source before/after copy.
  Shared upstream references, checkpoints and the uv environment are explicit;
  experiment-specific assets still need their own hashes. Do not edit the copy
  after execution begins; changes require a new copy and output prefix.
- Start orchestration through `HIP_VISIBLE_DEVICES=0 uv run python ...` so its
  children inherit the uv CLI PATH as well as the interpreter. Calling the
  `.venv/bin/python` path alone does not expose `mk_prepare_receptor.py`.
  The initial `measured_reward_control_20260913` run preserves this launcher
  failure; the corrected run uses the separate `_v2` prefix.
- `--search-docking-reward` connects actual energy to MCTS; default weight0.4
  contributes negative measured kcal/mol. `--search-docking-weight 0` still
  calls/caches the oracle but excludes its reward contribution for a controlled
  ablation. Record actual attempts, cache hits, failed calls and budgets for
  both treatments; equal limits do not automatically mean equal actual cost.
- The in-search provider saves raw energy and original docked poses with GPU
  traces. Failure/budget exhaustion keeps null energy, never a fabricated
  value. The aggregator's failed-channel contribution is zero and this policy
  must remain explicit. A requested provider with zero real measurements is
  labelled unavailable while independent chemical generation continues.
- Physical evaluation after search is a separate stage. Preserve all generated
  candidates before synthesis/top-k filtering as the coverage denominator;
  analyse selection quality and all-generated quality separately.
