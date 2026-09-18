# Real AiZynth CPU and ROCm backend — 2026-09-13

The project now executes genuine learned AiZynth retrosynthesis through an
isolated chemistry environment. An explicitly labelled **official legacy v3
USPTO policy** runs on CPU ONNX or on the RX 7800 XT through an equivalent
Dense/ELU/Softmax predictor. No SMARTS fallback counts as learned success.
Current upstream Zenodo assets remain unavailable; their failed probes are
preserved in `asset_manifest.json`, `adapter_load.json`, and `gate_smoke.json`.

## Measured evidence

| Evidence | Result |
|---|---|
| Real ZINC stock | 17,422,831 unique InChIKeys, loaded by official AiZynth query |
| Legacy model | Official Figshare file 23086454; 300,068,076 bytes; whole-file MD5 matches S3 ETag |
| Matching templates | Official Figshare file 23086457; 46,695 templates, unchanged original ordering |
| Keras graph to ONNX | Original weights preserved; 28 inputs versus independent NumPy graph equations; max probability error 3.28e-6, top-1 agreement 100% |
| ROCm versus ONNX | Same 28 inputs; max probability error 2.32e-6; top-1/10/50 sets all agree |
| Real CPU learned smoke | 3/3 declared targets found one-step routes to stock |
| Real ROCm learned smoke | Same 3/3 targets found the same final routes; 10 actual GPU predictor calls |
| Actual Lambda products | 4-simulation standard-12 CuAAC search generated 3 products; learned CPU gate retained 3/3, depths 6/5/5 |
| ROCm on those products | 3/3 solved with depths 6/5/5; 31 actual GPU predictor calls; leaf SMILES retained |

`legacy_learned_smoke.json` and `rocm_learned_smoke.json` retain seed,
search statistics, selected policy/stock, configuration hashes and full leaf
SMILES. The three declared smoke targets are aspirin, N-methyl triazole and
an aryl amide. Target exclusion from stock is enabled: a target's own stock
membership is not counted as a solved zero-step route in these probes.
`lambda_generated_gate_smoke.json` and `rocm_lambda_products_gate.json`
record actual generated products; these are development functional probes,
not CrossDocked, efficacy, synthesis-rate benchmark or physical-docking claims.

The NumPy conversion check evaluates the saved Keras inference equations;
it is not a comparison against a TensorFlow runtime. ROCm parameters and
outputs are confirmed on `cuda:0`, `gfx1101`, torch `2.14.0+rocm7.2`, HIP
`7.2.53211`, 17,163,091,968 bytes device memory. RDKit fingerprints, template
application, stock queries, MCTS and route analysis remain official CPU code.
The GPU extension replaces only the predictor. Small floating-point/ranking
differences change intermediate MCTS trajectories (triazole: 16 GPU nodes
versus 19 CPU nodes); no trajectory identity or end-to-end speedup is claimed.

## Configuration and commands

- `molmetal/configs/aizynth_legacy_v3_cpu.yml`: official legacy assets, CPU ONNX.
- `molmetal/configs/aizynth_legacy_v3_rocm.yml`: same assets and search, remote ROCm Dense predictor.
- `environments/aizynth/`: separate Python 3.12 uv lock and exact cached runtime.
  RDKit 2023.9.6 / NumPy 1.26.4 do not replace the root chemistry environment.
- The GPU worker uses the existing root ROCm environment through an explicit
  subprocess. The chemistry process never adds the root site-packages to its
  import path. Model, stock, converted weights, configs and source snapshot
  hashes are in `provenance.json` and `conversion_validation.json`.

```bash
bash environments/aizynth/run.sh python molmetal/scripts/aizynth_learned_smoke.py \
  --config molmetal/configs/aizynth_legacy_v3_cpu.yml \
  --output /tmp/aizynth_cpu_smoke.json
HIP_VISIBLE_DEVICES=0 bash environments/aizynth/run.sh python molmetal/scripts/aizynth_learned_smoke.py \
  --config molmetal/configs/aizynth_legacy_v3_rocm.yml \
  --protocol-label 'official legacy v3; ROCm policy; CPU chemistry and MCTS' \
  --output /tmp/aizynth_rocm_smoke.json
HIP_VISIBLE_DEVICES=0 uv run --no-sync python molmetal/scripts/verify_aizynth_torch_policy.py
```

From the root environment, `build_synthesis_gate("aizynthfinder_isolated",
config_path)` probes the real isolated backend and returns a checker only
when policy and stock load. `gate_candidates` sends one batch so policy/stock
load once per batch. Failed imports, missing models, subprocess errors and
SMARTS fallback cannot count as learned passes. The checker has a total
process deadline and terminates its complete process group, including GPU
children, on timeout/cancellation.

`download_aizynth_legacy_model.py` reproduces the range-based download through
the exact bucket/object returned by the official Figshare redirect. Byte
ranges and lengths are checked, partial chunks are journaled, ETags must
agree, and the whole-file MD5 is verified before publication. The official
v3 downloader source snapshot is saved with SHA256/Git blob identity; upstream
Git tag-commit lookup failed with a TLS error and is not claimed verified.
`convert_aizynth_legacy_onnx.py` accepts only the audited original model SHA256.

## Correctness and denominator changes

The adapter now selects policies/stocks, applies real iteration/time budgets,
uses the 4.x reaction-tree API, accepts only `is_solved`, and reports longest
root-to-leaf reaction depth. Partial routes are not synthesis successes.
Source and model load errors remain visible.

Generation counts now precede synthesis filtering and top-k selection.
`all_candidates` preserves every search-returned candidate with its synthesis
verdict, retention and selection flags; `candidates` contains only retained
candidates selected for downstream evaluation. Rejected generated products
remain in `n_generated_candidates` and `n_unique_generated`. Retained,
selected and physically evaluated counts are separate; this search runner
performs zero physical evaluations. A unit test that rejects all three
search-returned candidates still records both generated products.

Focused validation: 50 search/generation/guidance/adapter tests passed after
the denominator change; the later adapter/bridge/guidance suite passed
25 tests, including a real launcher/child timeout proving descendant cleanup.
CPU and ROCm probes are standalone measured evidence, not optional mocks.


Final bridge validation also exercised the real nested uv/Python GPU workers.
`real_gpu_descendant_timeout.json` records their actual process-group IDs and
confirms both terminated on a 15-second deadline. The monitor checks children
of all uv threads; earlier main-thread-only probes did not establish GPU
cleanup and are explicitly documented. `final_process_group_gate_smoke.json`
confirms the final normal path still solves aspirin and publishes the real
post-batch GPU predict-call count. No AiZynth workers remained afterward.

`TODO/environment.md` now documents both audited configurations. The AMD
skill preserves the existing REINVENT GPU/RPC guidance and adds the verified
AiZynth boundary and matched-asset requirements; skill-creator's
`quick_validate.py` returned `Skill is valid!`. The isolated uv lock and shell
wrapper checks also pass.
