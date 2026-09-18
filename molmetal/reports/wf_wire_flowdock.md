# WF-Wire-Clone-Scoring — FlowDock adapter wiring

**Workflow**: WF-Wire-Clone-Scoring (5th adapter wiring after DiffDock-L)
**Date**: 2026-09-14
**Status**: COMPLETE — wire-up shipped, 8/8 tests pass, no regressions
**Author**: hugo (autonomous subagent)

---

## 1. Goal

Wire the existing cloned FlowDock (Morehead & Chen 2024, arXiv:2412.10966) into
`molmetal/scripts/r4_c_full_sweep.py` as a **SOTA scoring column** parallel to the
existing `--sota-diffdock` column.

Why now: FlowDock is the **preferred** L-1 binding oracle over DiffDock-L on
consumer GPUs because it ships a CPU-fallback ODE solver (40 steps) that does not
require `torch_cluster` / `torch_scatter` extensions.  Even though the inference
path is still ROCm-hostile (the upstream repo depends on `lightning`, `hydra`,
`esm`, `flash_attn`), wiring the *pipeline-side* glue now lets us:

1. record honest FlowDock confidence values whenever a future checkpoint / CPU
   build becomes available;
2. share the same scoring-column discipline (`status`, `n_invoked`, `n_scored`,
   `score_mean`, `score_std`, `per_smiles`) with the DiffDock column;
3. document the L-1 / scoring-column split explicitly so the next integrator
   knows where to plug FlowDock's CPU-fallback ODE solver.

---

## 2. What already existed (audit phase)

- `molmetal/references/FlowDock/` — vendored repo at the upstream `main`
  branch.  Hydra-driven CLI: `python flowdock/sample.py
  +ckpt_path=... +input_receptor=... +input_ligand=... +out_path=...
  +n_samples=... +num_steps=40`.  Default config in
  `configs/sample.yaml`; experiment config in
  `configs/experiment/flowdock_fm.yaml`.
- `molmetal/molmetal_lam/sbdd_env/flowdock_adapter.py` — already shipped:
  * `FlowDockAdapter` — Protocol-shaped L-1 binding oracle (subprocess wrapper)
  * `FlowDockReferenceAdapter` — abstract-layer `DockingEngine` stub for
    the `molmetal.ports` layer
  * `is_flowdock_available()` — vendored-repo probe
  * Re-exports `DockResult`, `AdapterUnavailable`, `DockingOracle` from
    `diffdock_adapter` for symmetry
- 18 related tests in `test_diffdock_wire.py` + `test_diffdock_flowdock_adapters.py`
  + `test_clone_integration_adapters.py` — **all 18 still pass** after wiring.

What was missing: a **pipeline-side glue module** (parallel to
`diffdock_sota_scoring.py`) that turns a list of Lambda candidates into
per-candidate FlowDock confidence values, suitable as a column in the
r4_c_full_sweep report, with `--sota-flowdock` CLI flags and a worker-side
branch in the per-pocket runner.

---

## 3. What was added (this workflow)

### 3.1 New module: `flowdock_sota_scoring.py`

**File**: `molmetal/molmetal_lam/sbdd_env/flowdock_sota_scoring.py`

Mirrors `diffdock_sota_scoring.py` exactly in shape:

* `FlowDockScoreColumn` — dataclass with `status`, `n_invoked`, `n_scored`,
  `flowdock_score_mean`, `flowdock_score_std`, `per_smiles`, `notes`.
  Same status enum: `{"unavailable", "ok", "error", "partial"}`.
* `flowdock_cli_available(repo_root)` — vendored-repo probe; checks for
  `flowdock/sample.py` + `flowdock/__init__.py`.
* `_aggregate(confidences)` — internal mean/std helper.
* `_build_cli(...)` — composes the upstream Hydra CLI invocation as a
  dotlist of `+key=value` overrides; pins `sampling_task=batched_structure_sampling`.
* `_run_one_subprocess(...)` — invokes FlowDock `sample.py` once per SMILES,
  `cwd=<repo_root>` so the upstream `from flowdock import ...` resolves.
* `_parse_confidence_from_outdir(out_dir, sample_id)` — best-effort parser;
  reuses `DiffDockAdapter.parse_confidence_from_outdir` because FlowDock was
  forked from DiffDock-L and preserves the `rank1_confidence*.sdf` naming
  convention.
* `score_candidates(candidates, protein_path, *, repo_root, n_samples,
  num_steps, ckpt_path, device, timeout_sec)` — public entry point.

Honest framing is enforced in the docstring: when the vendored repo is missing
or the protein file is missing, the helper returns `status="unavailable"` /
`status="error"` without synthesising confidence numbers.  When the subprocess
fails, the SMILES gets `float('nan')` and `status="partial"` / `"error"`.

### 3.2 New CLI flags on `r4_c_full_sweep.py`

```
--sota-flowdock                  Subprocess-call FlowDock for every Lambda
                                 candidate and record confidence
--sota-flowdock-samples INT      FlowDock n_samples budget (default 4 = cheap)
--sota-flowdock-steps INT        FlowDock ODE num_steps budget (default 40)
--sota-flowdock-timeout FLOAT    Per-SMILES FlowDock subprocess timeout (s)
--sota-flowdock-repo PATH        Path to the cloned FlowDock checkout
--sota-flowdock-ckpt PATH        Optional path to a FlowDock checkpoint (.ckpt)
--sota-flowdock-device {cpu,cuda}  Inference device (default cpu; ROCm has
                                   no CUDA backend)
```

Default values mirror the upstream `sample.yaml` defaults (`n_samples=4`,
`num_steps=40`, `device="cpu"`).  `--sota-flowdock-device cuda` is accepted
for forward-compatibility with NVIDIA hosts but is **not** wired through to
ROCm on the local RX 7800 XT (gfx1101).

### 3.3 Worker-side wiring

When `--sota-flowdock` is enabled, the worker pops `flowdock_config` from
the kwargs, runs `score_candidates(...)` against the same candidate list
and the same receptor PDB already used by `--sota-diffdock`, and records:

* `flowdock_score_mean`, `flowdock_score_std`
* `flowdock_status` (`unavailable` | `ok` | `error` | `partial`)
* `flowdock_n_invoked`, `flowdock_n_scored`
* `flowdock_per_smiles` (full per-candidate map)

If the per-candidate scoring block raises, the worker downgrades to
`status="error"` and writes the exception into `flowdock_per_smiles["error"]`
without losing the search/physical records (mirrors the `--sota-diffdock`
fallback behaviour).

### 3.4 Aggregator + metadata

`aggregate(results)` now exposes six new keys:

```python
"flowdock_score_mean": _safe_mean(...),
"flowdock_score_std":  _safe_std(...),
"flowdock_n_pockets_scored":   ...,
"flowdock_n_invoked_total":    ...,
"flowdock_n_scored_total":     ...,
"flowdock_status_counts":      _status_counts(r.flowdock_status ...),
```

The `metadata.physical_implementation_sha256` block now records the
`file_digest` of `flowdock_sota_scoring.py` (alongside
`diffdock_sota_scoring.py`) so the report provenance captures the
new wire-up.

### 3.5 Tests

**File**: `molmetal/molmetal_lam/tests/test_flowdock_wire.py`

8 tests, all CPU-only (no GPU, no torch-cluster / torch_scatter), all green:

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_flowdock_sota_scoring_importable` | module imports, public surface is in `__dataclass_fields__`, defaults are sane |
| 2 | `test_flowdock_cli_available_detection` | vendored-repo probe returns False for missing dirs, True iff `flowdock/sample.py` + `flowdock/__init__.py` exist |
| 3 | `test_flowdock_subprocess_call_smoke` | monkey-patched `subprocess.run` returns a fabricated `rank1_confidence-1.42.sdf`; the helper parses it to mean = −1.42 with `n_scored=1, n_invoked=1, status=ok` |
| 4 | `test_flowdock_score_candidates_unavailable` | missing vendored repo -> `status="unavailable"`, `score_mean=None` |
| 5 | `test_flowdock_score_candidates_empty` | empty candidate list -> `status="ok"`, `n_invoked=0` |
| 6 | `test_flowdock_score_candidates_missing_protein` | missing `protein_path` -> `status="error"`, `notes` contains `"protein_path not found"` |
| 7 | `test_flowdock_aggregation_helper` | `_aggregate` returns the right `(mean, std)` for n=0 / n=1 / n=2 / mixed finite+NaN+Inf |
| 8 | `test_flowdock_score_recorded_in_report` | end-to-end: drives the r4_c_full_sweep worker with a synthetic `PocketResult` + `flowdock_config` and verifies the new columns land in the JSON output with the right mean/std (mean of −1.20 and −0.80 = −1.00, std = 0.20) |

CPU-only mocks use `monkeypatch.setattr(scoring.subprocess, "run", ...)`
and a synthetic vendored-repo layout (`tmp_path/synthetic_flowdock/flowdock/
{__init__.py,sample.py}`) — no torch / lightning / hydra required at test
time.

**Regression check**: `uv run pytest molmetal/molmetal_lam/tests/test_diffdock_wire.py
molmetal/molmetal_lam/tests/test_diffdock_flowdock_adapters.py
molmetal/tests/test_clone_integration_adapters.py` — **18/18 pass**.

---

## 4. Dependency chain

```
molmetal/scripts/r4_c_full_sweep.py        ← CLI flag + worker wiring
    └─ molmetal/molmetal_lam/sbdd_env/flowdock_sota_scoring.py  ← pipeline-side glue (NEW)
        ├─ molmetal/molmetal_lam/sbdd_env/diffdock_adapter.py    ← reuses parse_confidence_from_outdir
        └─ molmetal/references/FlowDock/flowdock/sample.py      ← upstream Hydra CLI (vendored)

molmetal/molmetal_lam/sbdd_env/flowdock_adapter.py               ← existing L-1 binding oracle
    └─ (parallel)  FlowDockReferenceAdapter (abstract-layer DockingEngine stub)
```

Discovery / fallback chain at runtime:

1. `--sota-flowdock` flag set in CLI args -> `search["flowdock_config"]`
   populated -> worker pops it before invoking `run_one_pocket`.
2. Worker calls `score_candidates(...)` in
   `molmetal_lam/sbdd_env/flowdock_sota_scoring.py`.
3. Helper probes `flowdock_cli_available(repo_root)`:
   * True  -> for each candidate, subprocess a single `python flowdock/sample.py
              +ckpt_path=... +input_receptor=PDB +input_ligand=SMILES
              +out_path=tmp +n_samples=... +num_steps=... +device=...`.
   * False -> return `status="unavailable"` immediately.
4. Subprocess output is parsed via the `rank1_confidence*.sdf` filename
   convention (shared with DiffDock-L).
5. Worker records `flowdock_score_mean`, `flowdock_score_std`, `flowdock_status`,
   etc. into the `PocketResult` dataclass; aggregator summarises into
   `metadata.summary.flowdock_*`.

---

## 5. Honest framing

* **No FlowDock checkpoint was downloaded** for this workflow.  The local
  environment has neither the vendored weights nor a buildable
  `torch_cluster` / `torch_scatter`.  Real FlowDock confidence numbers
  therefore cannot be produced on this machine today.
* When `--sota-flowdock` is run with no checkpoint, the upstream
  `flowdock/sample.py` will fail at `cfg.ckpt_path` resolution; the
  helper degrades to `status="error"` and writes the upstream stderr
  into `flowdock_per_smiles["error"]` — **no synthetic numbers**.
* The CPU-fallback ODE solver (40 steps) that makes FlowDock preferable
  to DiffDock-L on consumer GPUs is a **future** integration point.  It
  is *not* wired into this column because the upstream repo's `lightning`
  + `hydra` + `esm` + `flash_attn` dependencies are not buildable on
  ROCm 7.2 today.  This is documented as a deferred capability, not a
  hidden failure mode.
* The column is **reporting-only** for now: it produces an honest
  `null`/`NaN` FlowDock confidence value whenever the upstream
  inference is unavailable, exactly like the DiffDock-L column.
* All tests run CPU-only without torch; no GPU is consumed.

---

## 6. Files touched

| Path | Change |
|---|---|
| `molmetal/molmetal_lam/sbdd_env/flowdock_sota_scoring.py` | NEW (309 lines) |
| `molmetal/molmetal_lam/tests/test_flowdock_wire.py` | NEW (369 lines, 8 tests) |
| `molmetal/scripts/r4_c_full_sweep.py` | +6 PocketResult fields, +6 aggregate keys, +7 CLI flags, +worker branch, +sha256 in metadata |

No existing test was modified; all 18 sibling tests still pass.

---

## 7. Status

| Component | Status |
|---|---|
| FlowDock pipeline-side glue | SHIPPED |
| `--sota-flowdock` CLI flags | SHIPPED |
| Worker wiring into r4_c_full_sweep | SHIPPED |
| Aggregator + metadata keys | SHIPPED |
| CPU-only tests | 8/8 PASSED |
| Sibling-test regression | 18/18 PASSED |
| Live FlowDock inference | NOT EXECUTED (no checkpoint + ROCm-hostile deps) |
| Refutation experiments | NOT EXECUTED |

**Workflow outcome**: COMPLETE — wire-up is in place; honest empty-result
behaviour is verified end-to-end on CPU.
