# WF-Extra-2 — REINVENT4 multiproperty scoring surface audit

**Date:** 2026-09-14
**Scope:** audit-only — no code edits shipped. Task #395 (WF-Extra-2: TODO-05 REINVENT4
multiproperty bridge) is the upstream owner of the actual implementation.
**Honest framing:** every line below is annotated `[MEASURED]` (read straight
from source on disk or from a CLI invocation) vs `[PROJECTED]` (planned
behaviour for the not-yet-written worker). Mixed pairs are spelled out
explicitly so a reader can grep for the marker.

---

## 1-paragraph audit

The current REINVENT4 path is bifurcated and **r_reinvent4 is
silently off by default**: the subprocess adapter
(`molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py`) refuses to
launch the upstream `reinvent` CLI because it speaks TOML-RL, not
JSONL, and falls back to a tiny RDKit proxy
(`reinvent4_jsonl_worker.py`) whose four `[0,1]` fields are toy
heuristics (`QED.qed`, `1 − MW/600`, `0.5 − MolLogP/10`,
`1 − NumHeavyAtoms/80`), not learned REINVENT4 components — those
labels `(qed, sa, binding, novelty)` are protocol placeholders, not
real REINVENT4 outputs. The downstream `RewardAggregator` carries a
fully plumbed `r_reinvent4` channel with default weight
`w_reinvent4 = 1.0` and an explicit `_safe` degradation to `0.0`,
but **no caller in `orchestration/closed_loop.py` ever sets
`r_reinvent4` to a callable** (only `closed_loop.py:1020` reports
its `None`-ness), so the closed-loop silently records `r_reinvent4 = 0.0`
every iteration. The neighbouring `REINVENT4PriorAdapter`
(`reinvent_prior_jsonl_worker.py`) already runs the real
`reinvent.runmodes.create_adapter` on ROCm at `/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent`
4.8.24 + torch 2.14.0+rocm7.2 + gfx1101 and ships an explicit
`prior_nll` JSONL protocol; the same harness can host a learned
**multiproperty** worker because REINVENT4 also supports
`run_type = "scoring"` driven by a TOML config (`run_type`, `parameters`,
`[scoring] type = "geometric_mean"` (or `"arithmetic_mean"`),
`[[scoring.component]]` blocks with `weight` + `transform.type` —
`double_sigmoid` / `reverse_sigmoid` / `step` / `sigmoid` /
`value_mapping` — already shipped in upstream examples
`stage1_scoring.toml`, `scoring_components_example.toml`,
`upstream/configs/scoring.toml`). The plan is therefore to add a
second isolated worker
(`reinvent4_multiproperty_jsonl_worker.py`) that mirrors the prior
worker’s ROCm-aware lifecycle (architecture verify via `gcnArchName`,
`parameter_device` consistency check, forward-hook observation, no
RDKit proxy), consume the existing four-`[0,1]` JSONL contract so the
adapter wire format does not change, and pre-clamp each component via
the TOML `transform.type` so the returned scalar is already in `[0,1]`
and the aggregator’s existing `ScoreAggregator` / `RewardAggregator`
wiring lights up `r_reinvent4` for the first time. Honest-framing
caveat: the upstream CLI takes a TOML `FILE` and writes to
`output_csv` + `json_out_config`; it is **not** a JSONL RPC, so the
worker must drive REINVENT4 in-process via `reinvent.runmodes.create_adapter`
+ `reinvent.scoring.Score...` (same trick the prior worker already
uses), then expose the resulting per-component table on a JSONL
socket — the four-`[0,1]` protocol returned to the parent adapter is
the **only** REINVENT4-native guarantee; the inner scoring component
list is configurable per TOML and remains the user’s choice (default
profile below uses QED + SAScore + MolecularWeight + TPSA, all
arithmetic mean).

---

## file:line list

### Subprocess adapter (JSONL contract, error policy)

- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:35-38` —
  JSONL wire format documentation:
  `{"op": "score", "smiles": [...]}` → `{"results": [{...}, ...]}`.
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:109-134` —
  `is_reinvent4_binary_available` probes `REINVENT4_BIN`, then
  `/mnt/storage/envs/reinvent4/bin/reinvent`, then `shutil.which("reinvent")`
  and the cloned reference checkout at
  `molmetal/references/REINVENT4/reinvent/Reinvent.py`. [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:137-144` —
  `_is_jsonl_worker(binary)` accepts only workers whose basename
  contains `jsonl_worker` (defensive — the upstream CLI is
  TOML-RL, not RPC). [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:212-264` —
  `__post_init__` resolves binary, refuses the upstream CLI
  (`last_error = "cli_protocol_mismatch"`), points the default at the
  in-tree RDKit proxy
  `molmetal_lam/sbdd_env/reinvent4_jsonl_worker.py`. [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:316-337` —
  `_start_worker` builds the command: `<binary>` if the binary already
  ends with `reinvent4_jsonl_worker.py`; otherwise
  `<binary> --scoring-config <toml>` (the `reinvent` CLI supports
  `--scoring-config`, see upstream `reinvent/runmodes/RL/reward.py`).
  [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:374-421` —
  `score(smiles_batch)` builds `{"op": "score", "smiles": [...]}`,
  parses `results[i]` into a `ScoreResult(qed, sa, binding, novelty)`,
  clips each to `[0,1]` via `_clip01`; any failure mode returns `None`
  per-element (no raise). [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:423-454` —
  `_rpc_roundtrip` does `json.dumps(request) + "\n"` →
  `proc.stdout.readline()` → `json.loads`; empty line, JSON
  `DecodeError`, or any IO error flips `self.dead = True`. [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:476-509` —
  `ScoreAggregator(w_qed=0.3, w_sa=0.3, w_binding=0.3, w_novelty=0.1)`
  weighted sum; missing component → `0.0`; `total_weight()` exposed for
  diagnostics. [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py:79-103` —
  `ScoreResult` dataclass with `is_complete()` validator (finite
  `[0,1]` for all four fields) and `to_dict()`. [MEASURED]

### RDKit proxy worker (the current default backend)

- `molmetal/molmetal_lam/sbdd_env/reinvent4_jsonl_worker.py:17-32` —
  `_score(smiles)` is RDKit-only, never imports REINVENT4 or PyTorch.
  Four deterministic heuristics: [MEASURED]
  - `qed = clip(QED.qed(mol))` (real QED, [0,1]).
  - `sa = clip(1 − MW/600)` — toy MW-derived placeholder, **not** the
    Ertl SA score.
  - `binding = clip(0.5 − MolLogP/10)` — toy logP placeholder, **not**
    a docking or learned binder score.
  - `novelty = clip(1 − NumHeavyAtoms/80)` — toy size-only placeholder.
  Module docstring at lines 1-9 explicitly states these are
  *protocol-level proxies* and the real scorer can replace the worker
  without changing the wire format. [MEASURED — honest-framing caveat]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_jsonl_worker.py:35-44` —
  `handle(request)` dispatches on `op` (`ping`/`score`/else), `score`
  expects `smiles: list[str]` and returns
  `{"results": [{"qed":..,"sa":..,"binding":..,"novelty":..}, ...]}`. [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent4_jsonl_worker.py:47-60` —
  `main()` reads line-delimited JSON from stdin, writes
  `json.dumps(..., separators=(",", ":")) + "\n"` to stdout with
  `flush()` per response. [MEASURED]

### RewardAggregator — r_reinvent4 channel wiring

- `molmetal/molmetal_lam/search_alg/proof_search.py:633-640` —
  `r_reinvent4: Optional[Callable[[MoleculeClosedTerm], float]] = None`.
  Docstring says when adapter is unavailable, set to `None` or a
  callable returning `0.0`. [MEASURED]
- `molmetal/molmetal_lam/search_alg/proof_search.py:685-688` —
  `w_reinvent4: float = 1.0` (default ON at the weight layer, so
  *if* the callable were wired the contribution would be 1.0 × value). [MEASURED]
- `molmetal/molmetal_lam/search_alg/proof_search.py:799-801` —
  `aggregate()` recognises the channel name `"r_reinvent4"` in the
  per-channel `channels` dict. [MEASURED]
- `molmetal/molmetal_lam/search_alg/proof_search.py:860-873` —
  `weight_for_channel` lookup table maps `r_reinvent4 → w_reinvent4`. [MEASURED]
- `molmetal/molmetal_lam/search_alg/proof_search.py:928` —
  `v_reinvent4 = _safe(self.r_reinvent4)` — the `_safe` helper at
  line 907-913 returns `0.0` for `None` callables (the silent-off
  path). [MEASURED]
- `molmetal/molmetal_lam/search_alg/proof_search.py:956-958` —
  `value += self.w_reinvent4 * v_reinvent4` — the actual sum. [MEASURED]
- `molmetal/molmetal_lam/search_alg/proof_search.py:1865` —
  Per-channel weight docstring lists `r_reinvent4` alongside the other
  opt-in channels. [MEASURED]
- `molmetal/molmetal_lam/search_alg/proof_search.py:3293` —
  Closed-loop channel registration mention; not a wiring site. [MEASURED]
- `molmetal/molmetal_lam/search_alg/_batch_reward_worker.py:69` —
  `aggregator_to_config` already serialises `w_reinvent4` into the
  picklable worker config (silent channel — the worker cannot reach
  the subprocess, so it returns `0.0`). [MEASURED]
- `molmetal/orchestration/closed_loop.py:99` —
  Comment: `r_sa, r_qed, r_pic50 and r_reinvent4. Round-3 evidence`. [MEASURED]
- `molmetal/orchestration/closed_loop.py:507, 521` —
  `breakdown["r_reinvent4"] = aggregator.w_reinvent4 * v_reinvent4` —
  reports `0.0` whenever the callable is `None`. [MEASURED]
- `molmetal/orchestration/closed_loop.py:1020` —
  `"r_reinvent4": agg.r_reinvent4 is not None` — closed-loop only
  records whether the channel is wired, never wires it itself.
  [MEASURED — confirms `r_reinvent4` is **never** set to a callable
  by `closed_loop.py`; this is the silent-off gap.]

### Existing prior_nll config (model for the new multiproperty config)

- `molmetal/configs/reinvent_prior_amd.json:1-9` — full file:
  ```json
  {
    "mode": "prior_nll",
    "worker_python": "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/python",
    "prior_path": "/mnt/storage/models/reinvent4/reinvent_v4.4.22.prior",
    "prior_sha256": "b6513ec6dbc54c87ea45cdbf9b4aaefadd7652548b74175366b27f12ec5732fe",
    "device": "cuda:0",
    "expected_architecture": "gfx1101",
    "timeout": 60.0
  }
  ```
  Mode is `prior_nll` (NLL RPC, **not** multiproperty). [MEASURED]

### Upstream REINVENT4 CLI + scoring TOML format (PROJECTED plan)

- `/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent --help` —
  CLI is `reinvent [-f toml|json|yaml] [-d cuda|cpu] [-s N] [FILE]`;
  the runtime mode (`run_type`) is decided by `FILE`, not by a CLI
  subcommand. `run_type = "scoring"` is selected by the TOML key.
  [MEASURED]
- `/mnt/storage/env-projects/reinvent4-rocm/upstream/configs/stage1_scoring.toml:1-41` —
  Example of a single-component TOML:
  ```toml
  [[component]]
  [component.MolecularWeight]
  [[component.MolecularWeight.endpoint]]
  name = "Molecular weight"
  weight = 0.342
  transform.type = "double_sigmoid"
  transform.high = 500.0
  transform.low = 200.0
  transform.coef_div = 500.0
  transform.coef_si = 20.0
  transform.coef_se = 20.0
  ```
  Same TOML also drives the in-process
  `reinvent.runmodes.create_adapter` + `reinvent.scoring` path that
  the existing `reinvent_prior_jsonl_worker.py` already invokes for
  NLL. [MEASURED — TOML format; in-process RPC is PROJECTED because
  the upstream CLI is non-RPC and would need a separate driver.]
- `/mnt/storage/env-projects/reinvent4-rocm/upstream/configs/scoring_components_example.toml:1-358` —
  Canonical component catalogue: `QED`, `SlogP`, `MolecularWeight`,
  `TPSA`, `GraphLength`, `NumAtomStereoCenters`, `HBondAcceptors`,
  `HBondDonors`, `NumRotBond`, `Csp3`, `numsp/sp2/sp3`,
  `NumHeavyAtoms`, `NumHeteroAtoms`, `NumRings`, `NumAromaticRings`,
  `NumAliphaticRings`, `LargestRingSize`, `pmi`, `MolVolume`,
  `GroupCount`, `MatchingSubstructure`, `custom_alerts`,
  `TanimotoDistance`, `MMP`, `RingPrecedence`, `rocssimilarity`,
  `DockStream`, `Maize`, `SAScore`. Top-level key
  `run_type = "scoring"` + `[parameters] smiles_file/output_csv` +
  `[scoring] type = "geometric_mean"` (or `arithmetic_mean`) +
  `[[scoring.component]]` blocks (notice the plural — *scoring*
  namespace, not the bare `[[component]]` shown in
  `stage1_scoring.toml` which is the first-stage scaffolding file).
  [MEASURED — both formats coexist in upstream; the worker should
  accept whichever the caller points at.]
- `/mnt/storage/env-projects/reinvent4-rocm/upstream/contrib/reinvent_plugins/components/config/scoring.toml` —
  Re-export of the same scoring TOML inside the plugins package. [MEASURED]

### Test coverage (sanity)

- `molmetal/molmetal_lam/tests/test_reinvent4_subprocess_adapter.py:131, 150, 157` —
  Three test cases build `RewardAggregator(r_reinvent4=stub_channel, w_reinvent4=1.0)`
  to exercise the channel end-to-end. Means the aggregator path is
  already covered, **only** the subprocess wiring is missing. [MEASURED]

### Adjacent worker that already proves the ROCm RPC pattern works

- `molmetal/molmetal_lam/sbdd_env/reinvent_prior_jsonl_worker.py:14-46` —
  `PriorWorker.__init__` performs:
  - SHA-256 check of `prior_path` against `prior_sha256`.
  - `reinvent.runmodes.create_adapter(str(args.prior), "inference",
    torch.device(args.device))` (in-process, not CLI subprocess).
  - `model_type == "Reinvent"` assertion.
  - `torch.cuda.get_device_properties(0)` → `architecture=gcnArchName`,
    `gpu_name`, `gpu_memory_bytes`.
  - `parameter_devices` consistency check against `args.device`
    (raises on mismatch).
  This is the **template** the multiproperty worker should copy —
  same SHA/architecture/device checks, same in-process adapter
  construction (only swapping the run-mode to `scoring`). [MEASURED]
- `molmetal/molmetal_lam/sbdd_env/reinvent_prior_jsonl_worker.py:47-100` —
  `PriorWorker.likelihood(smiles)` is a per-SMILES bucket-by-token-length
  forward with a `register_forward_pre_hook` that records `(param_device,
  input_device, count)` neural-forward counts. The multiproperty worker
  should keep the same observer hook (it’s how the round-10/11 reports
  prove *measured* GPU activity). [MEASURED]

### Planned new config (PROJECTED — not yet written)

- `molmetal/configs/reinvent_multiproperty_amd.json` (PROJECTED):
  ```json
  {
    "mode": "multiproperty",
    "worker_python": "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/python",
    "scoring_toml": "/home/hugo/codes/try_triton_on_rocm/molmetal/configs/reinvent_multiproperty_default.toml",
    "device": "cuda:0",
    "expected_architecture": "gfx1101",
    "timeout": 60.0,
    "aggregate": "arithmetic_mean",
    "components": [
      {"name": "QED", "weight": 0.4},
      {"name": "SAScore", "weight": 0.2},
      {"name": "MolecularWeight", "weight": 0.2,
       "transform": {"type": "double_sigmoid", "high": 500.0, "low": 200.0}},
      {"name": "TPSA", "weight": 0.2,
       "transform": {"type": "double_sigmoid", "high": 140.0, "low": 0.0}}
    ]
  }
  ```
  Rationale: mirrors `reinvent_prior_amd.json`’s shape so the parent
  loader is a one-line switch on `mode`; the **wire** to
  `REINVENT4Adapter` is unchanged (the four `[0,1]` fields
  `qed/sa/binding/novelty` are produced from the *aggregated* REINVENT4
  scoring output — `qed` ← QED component, `sa` ← SAScore component,
  `binding` ← MolecularWeight-in-drug-window proxy, `novelty` ←
  TPSA-window proxy — so the existing `ScoreAggregator(w_qed, w_sa,
  w_binding, w_novelty)` weights still apply and no caller edits are
  required beyond the new worker file and the new TOML config). [PROJECTED]

### Wire-up checklist for the actual WF-Extra-2 implementation

1. Add `molmetal/molmetal_lam/sbdd_env/reinvent4_multiproperty_jsonl_worker.py`
   — in-process `reinvent.runmodes` `scoring` adapter + the four
   `[0,1]` JSONL contract. Mirror `PriorWorker`’s SHA/architecture/
   device checks. [PROJECTED]
2. Write `molmetal/configs/reinvent_multiproperty_default.toml`
   (the actual REINVENT4 scoring TOML the worker loads). [PROJECTED]
3. Write `molmetal/configs/reinvent_multiproperty_amd.json` (see
   block above). [PROJECTED]
4. In `molmetal/orchestration/closed_loop.py` (around line 1020, the
   `r_reinvent4` registration point), add the learned callable when
   the new config exists and the binary is available — gated on
   `REINVENT4_MULTIPROPERTY=1` env so the silent-off behaviour is
   preserved unless the user opts in. [PROJECTED]
5. Add a unit test that boots the new worker against a 3-SMILES
   fixture and asserts the JSONL `results` field contains four
   finite floats in `[0,1]` per row. [PROJECTED]
6. Honest-framing caveat for the round-12/13 paper: until the new
   worker is run on the AMD GPU with measured `neural_forwards` and
   the four-component dict values match REINVENT4’s `scoring.json`,
   the channel must be reported as `[PROJECTED, not yet measured]`.
   This matches the framing already in `TODO/pending/05_reinvent4_install.md:38`
   (`Learned-plugin scores remain OBS until a compatible scoring worker is configured`).
   [MEASURED — i.e. the reporting policy is already in place.]

---

## Honest-framing summary

- **MEASURED today (no edits required to confirm):**
  - `r_reinvent4` is plumbed in `RewardAggregator` with weight 1.0
    default, but every caller in `molmetal/orchestration/` leaves the
    callable as `None` → `r_reinvent4 = 0.0` always.
  - The default backend in
    `reinvent4_subprocess_adapter.__post_init__` is the in-tree RDKit
    proxy `reinvent4_jsonl_worker.py`, whose four fields are toy
    proxies (not learned REINVENT4 outputs).
  - REINVENT4 4.8.24 is installed at
    `/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent`,
    with prior NLL already validated end-to-end (round-9/10 reports).
  - The upstream TOML format for `run_type = "scoring"` is fully
    documented in `stage1_scoring.toml` and
    `scoring_components_example.toml`.

- **PROJECTED (requires edits to ship):**
  - New multiproperty worker file + new TOML config + new AMD config.
  - New `orchestration/closed_loop.py` registration gated on
    `REINVENT4_MULTIPROPERTY=1`.
  - First measured-on-AMD run to lift `[PROJECTED]` → `[MEASURED]` in
    the paper draft.
