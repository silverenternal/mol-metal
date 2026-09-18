# REINVENT4 installation, learned prior and multiproperty bridge

**Status:** PARTIAL — installation, actual AMD learned prior generation/NLL and independent likelihood RPC complete; learned multiproperty bridge pending
**Priority:** high
**Effort:** 0.5d
**Owner:** (unset)
**Depends on:** none
**Blockers:** the current JSONL bridge is still RDKit proxy scoring. Its four
`[0,1]` component fields do not represent learned prior sequence NLL; a real
task-specific multiproperty scoring worker remains unconfigured.
**Created:** 2026-09-12

## Goal
Keep REINVENT4 dependencies isolated, verify official pretrained generation
and learned likelihood on the AMD GPU, and provide a truthful JSON-lines
bridge for the project adapter. The upstream `reinvent` CLI is TOML/RL
oriented and is intentionally not launched as a JSONL worker. Prior NLL and
task-specific multiproperty scoring are separate capabilities.

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py  (verify backend flag)
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/reinvent4_adapter.py  (if pipx-style install path is used)

## Success criterion
`/mnt/storage/envs/reinvent4/bin/reinvent --version` exits 0 in the isolated
environment; the JSONL bridge returns complete `ScoreResult` values for valid
SMILES and graceful incomplete results for invalid SMILES; the adapter reports
`cli_protocol_mismatch` when pointed at the standard CLI. Learned-plugin
scores remain **OBS** until a compatible scoring worker is configured.

Current acceptance status:

- [x] Original isolated REINVENT 4.8.24 CLI executes.
- [x] Existing proxy bridge handles valid and invalid inputs with its declared protocol.
- [x] Official pretrained prior retrieved and verified against its exact Git blob.
- [x] Official learned generation and prior NLL execute on AMD gfx1101 with observed neural forwards.
- [x] Independent explicit learned prior NLL RPC adapter is validated; the existing four-component bridge remains proxy-only.
- [ ] Task-specific learned multiproperty scoring is configured and verified; NLL does not satisfy this criterion.

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_L4_reinvent4_scorer.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_round3_combined_report.md
- [Official CPU/AMD learned smoke and reproduction](../../molmetal/reports/reinvent4_learned_smoke/README.md)
- [Independent AMD likelihood RPC validation](../../molmetal/reports/reinvent_prior_adapter_amd/README.md)

## 2026-09-13 environment update

`/mnt/storage/envs/reinvent4/bin/reinvent --version` succeeds after installing
REINVENT4 4.8.24 plus its isolated dependencies. The project adapter discovers
this path (or `REINVENT4_BIN`) while keeping it outside the ROCm environment,
because REINVENT4 pins CUDA Torch 2.12. A separate RDKit-only
`reinvent4_jsonl_worker.py` now satisfies the adapter protocol and is covered
by tests; its scores are proxy protocol validation and do not claim learned
REINVENT4 results.

## 2026-09-13 verified AMD learned update

Installation and real prior inference are now complete in a separate managed
uv project, `/mnt/storage/env-projects/reinvent4-rocm`. It uses REINVENT 4.8.24,
torch 2.14.0+rocm7.2 and torchvision 0.29.0+rocm7.2, explicitly overrides the
upstream torch 2.12 pin, and adds the omitted SciPy runtime dependency.
All 576 upstream Python files are unchanged. The original CUDA installation
and the main project environment are retained.

The real ChEMBL 25 prior was recovered from official tag `v4.4.22` after
Zenodo timed out: `/mnt/storage/models/reinvent4/reinvent_v4.4.22.prior`,
23,226,277 bytes, Git blob `0dee328238b3d413b5a34c80e3ddab49e4e0af28`.
Its SHA256 is `b6513ec6dbc54c87ea45cdbf9b4aaefadd7652548b74175366b27f12ec5732fe`.
The legacy internal metadata xxhash warning is retained; external file bytes
match the official blob exactly and neither weights nor metadata were rewritten.

Seed-42 official sampling generates 5/5 RDKit-valid molecules on CPU and GPU.
The AMD run observes 80 RNN and 80 LSTM sampling forwards, plus one each for
five fixed SMILES likelihoods; parameters and inputs are all on `cuda:0`,
with actual architecture gfx1101. Fixed-input CPU/GPU NLL maximum difference
is 1.0491e-5. Logs, manifests/lock, checkpoints hashes and the observer script
are linked in the report above. These are N=5 capability checks, not training,
throughput or activity-quality results.

The independent `REINVENT4PriorAdapter` / `reinvent_prior_jsonl_worker.py`
now return raw NLL, predicted-token counts, model SHA and actual backend under
an explicit `prior_nll` contract. Eight subprocess tests cover request
deadlines/process-group cleanup and malformed/invalid outcomes. The real AMD
integration retains 7/7 input rows (3 scored, 4 intentional errors); duplicate
and singleton CCO NLLs agree. It groups by token length to avoid the upstream
padding-dependent batch sum. Config: `molmetal/configs/reinvent_prior_amd.json`.

Remaining work is actual task-specific learned multiproperty scoring, whose
components require their own assets, calibration and verification. The old
proxy bridge and multiproperty reward are unchanged; raw NLL is not clipped
into `qed/sa/binding/novelty` or automatically connected to search.

## 2026-09-14 WF-Extra-2 verify — 10-SMILES batch test (MEASURED)

End-to-end batch test of the new
`reinvent4_multiproperty_jsonl_worker.py` driving the real REINVENT4
binary at `/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent`.
10/10 SMILES succeeded, all scores in `[0, 1]`, learned-vs-proxy
Pearson r = 0.676, mean |Δ| = 0.107, max |Δ| = 0.32 (n-octane).
`r_reinvent4` channel is functional and is now wired into
`RewardAggregator` (see `molmetal/reports/wf_extra2_wire.md`).
Per-SMILES wall-clock ≈ 4.0 s (cold-start dominated; production
batch amortises to ~0.3 s/SMILES). Reproduction:
`uv run python molmetal/scripts/test_reinvent4_multiproperty_batch.py
--output-dir molmetal/reports/wf_extra2_batch/`. Report:
`molmetal/reports/wf_extra2_batch/final.md`. Not a search-loop run —
that belongs to WF-3.
