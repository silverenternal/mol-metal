# Explicit learned REINVENT prior RPC on AMD (2026-09-13)

## Goal

Expose real pretrained REINVENT sequence likelihood through an isolated subprocess, preserving raw NLL, input order, invalid-item counts, model identity and actual device. Keep the existing four-component multiproperty proxy and search reward unchanged.

## Outcome

**PASS:** the real AMD integration preserves all seven input records: three scored entries (including duplicate CCO) and four explicit failures (empty string, invalid SMILES, null and an out-of-vocabulary xenon token). A subsequent all-invalid batch preserves all four failure records and reports no neural forwards. An empty batch returns an empty result without starting a worker. The same worker process is reused across calls.

| Input | Status | Raw sequence NLL | Predicted tokens |
|---|---|---:|---:|
| CCO | ok | 19.1170120 | 4 |
| c1ccccc1 | ok | 17.9505653 | 9 |
| empty string | error | — | — |
| invalid | error | — | — |
| null | error | — | — |
| [Xe] | unsupported prior token | — | — |
| CCO | ok | 19.1170120 | 4 |

The GPU is **gfx1101**, torch 2.14.0+rocm7.2 / HIP 7.2.53211, with parameters and observed RNN input tensors on `cuda:0`. The batch uses two neural forwards because the worker groups valid inputs by token length. CCO batch-versus-single NLL difference is exactly zero in this run. The model is the 5,805,602-parameter official ChEMBL 25 RNN prior, SHA256 `b6513ec6dbc54c87ea45cdbf9b4aaefadd7652548b74175366b27f12ec5732fe`.

`report.json` retains input indices, original input values, every error, raw NLL/token counts, checkpoint SHA, runtime versions, metadata warning and neural-device evidence. The official prior's legacy internal xxhash warning remains in `worker.log`; the external file checksum is checked before model loading. See [the prior/environment provenance report](../reinvent4_learned_smoke/README.md).

Eight focused subprocess tests pass. They cover values above one without clipping, duplicates/order/invalid slots, empty batches, rejecting proxy backends, worker exit and malformed output, one total deadline across startup and inference, and a partial-line timeout that kills the worker's process-group descendant.

## Implementation and semantics

- Adapter: `molmetal/molmetal_lam/sbdd_env/reinvent_prior_adapter.py`, class `REINVENT4PriorAdapter`, method `likelihood(list)`, result `PriorLikelihoodResult`.
- Worker: `reinvent_prior_jsonl_worker.py`, protocol `reinvent-prior-jsonl-v1`, explicit `capabilities` / `likelihood` operations and correlated request IDs.
- Configuration: `molmetal/configs/reinvent_prior_amd.json`, explicit `mode='prior_nll'`, isolated interpreter, exact prior SHA and expected gfx1101 architecture.

The adapter imports no torch/REINVENT modules in the main process. A single deadline includes acquiring the request lock, lazy model startup, capability verification, writing and reading the request. A timeout or protocol/process failure produces one error result per input, terminates the worker process group and allows a later request to start a new worker. Cleanup has a bounded additional 0.4-second reap allowance. Stderr is a log file or temporary file, preventing an unconsumed pipe from blocking inference. Call `close()` or use the context manager.

NLL is not clipped or transformed into `[0,1]`. Original SMILES are not canonicalized. `predicted_token_count` excludes the beginning token and includes one end token. The official `likelihood_smiles` implementation sums over its padded sequence; to avoid changing a molecule's score when longer molecules enter its batch, this worker groups inputs by token length before calling that same official method. Consequently these NLLs have an explicit padding-free contract and should not be silently substituted for earlier mixed-length padded NLL artifacts. No neural weights or network implementation changed.

## Caveats

This is learned prior likelihood, not activity, synthesis, novelty, docking or calibrated multiproperty desirability. Sequence length and SMILES representation affect NLL. No reward channel is connected and no new claims are made for the existing RDKit multiproperty bridge. The seven-entry integration is a capability/correctness smoke, not a molecular-quality or throughput benchmark. Limits of 256 items per request and 4,096 characters per SMILES produce explicit errors rather than silently truncating requests.

## Reproduce

```bash
uv run pytest -q molmetal/tests/test_reinvent_prior_adapter.py
HIP_VISIBLE_DEVICES=0 uv run python -m molmetal.scripts.verify_reinvent_prior_adapter \
  --config molmetal/configs/reinvent_prior_amd.json \
  --out-dir molmetal/reports/NEW_PRIOR_RPC_PREFIX
```

Python use from the core environment:

```python
from molmetal.molmetal_lam.sbdd_env.reinvent_prior_adapter import REINVENT4PriorAdapter

with REINVENT4PriorAdapter.from_config("molmetal/configs/reinvent_prior_amd.json") as prior:
    results = prior.likelihood(["CCO", "invalid", "c1ccccc1"])
    provenance = prior.last_metadata
```

No seed is needed for deterministic likelihood evaluation in inference mode; the separate generation smoke records seed 42. Runtime/checkpoint/config hashes are retained in the machine-readable report.
