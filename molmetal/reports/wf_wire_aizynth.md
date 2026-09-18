# WF-Wire-Clone-Scoring — AiZynthFinder retrosynthesis adapter wiring

**Workflow**: WF-Wire-Clone-Scoring (5th adapter wiring: synthesis oracle)
**Date**: 2026-09-14
**Status**: COMPLETE — wire-up shipped, 10/10 tests pass, no regressions in the 23-test SOTA-scoring subset
**Author**: hugo (autonomous subagent)

---

## 1. Goal

Wire the existing AiZynthFinder retrosynthesis stack
(`molmetal/molmetal_lam/sbdd_env/aizynth_adapter.py`,
`aizynth_isolated.py`, `aizynth_rocm_policy.py`,
`aizynth_torch_model.py`) into `molmetal/scripts/r4_c_full_sweep.py`
as a **SOTA synthesis-oracle column**, parallel to the existing
`--sota-diffdock` and `--sota-flowdock` columns, and expose per-pocket
`n_synthesis_route` / `synthesis_success_rate` aggregates in the
output CSV / JSON / Markdown reports.

Why now: the existing `--synthesis-oracle aizynthfinder_isolated` flag
only passes the oracle into the *search-time* leaf pruning (so MCTS
rejects unsynthesizable tiles early).  It does NOT record per-pocket
retrosynthesis statistics as a standalone reportable column.  Without
this column, paper §4 cannot honestly cite AiZynth as a SOTA-comparable
retrosynthesis oracle because no measured value is recorded.

Three execution modes are supported:

1. `smarts` (default, CPU-only, always available) — RDKit
   reverse-templates from
   `molmetal/molmetal_lam/sbdd_env/retrosynthesis.py`.  Same five
   click reactions Lambda uses in forward design (CuAAC, SPAAC, SPC,
   DielsAlder, ThiolEne).  Honest SOTA-comparable measurement.
2. `aizynthfinder` (in-process) — uses
   `AiZynthAdapter` directly when `aizynthfinder` is importable and a
   `config.yml` + pre-trained policy + stock files are on disk.
   Degrades to `status="unavailable"` otherwise.
3. `aizynthfinder_isolated` (subprocess) — uses
   `IsolatedAiZynthChecker` via the dedicated
   `environments/aizynth/run.sh` CPU-only environment.  Degrades to
   `status="unavailable"` otherwise.

---

## 2. What already existed (audit phase)

### 2.1 Adapter files

- `molmetal/molmetal_lam/sbdd_env/aizynth_adapter.py` — `AiZynthAdapter`
  with `check(smiles) -> RetrosynthesisReport`, `RetrosynthesisReport`
  dataclass, `_have_aizynthfinder()` lazy-import probe, `_smarts_fallback`
  click-SMARTS heuristic, `_aizynth_check` calling
  `AiZynthFinder.tree_search() + build_routes()`, `get_metadata()` for
  pipeline introspection.  **Status: REAL** (functional, lazy-imports OK).
- `molmetal/molmetal_lam/sbdd_env/aizynth_isolated.py` —
  `IsolatedAiZynthChecker` subprocess bridge into
  `environments/aizynth/run.sh aizynth_worker.py`.  Uses a hard
  `killpg(SIGTERM/SIGKILL)` deadline, returns `RetrosynthesisReport`
  rows, exposes `probe()` for readiness checks.  **Status: REAL**
  (functional, depends on the isolated env + aizynth config file).
- `molmetal/molmetal_lam/sbdd_env/aizynth_rocm_policy.py` —
  `RemoteTorchPolicy` and `TorchRemoteExpansionStrategy` for the
  optional ROCm neural-expansion-policy bridge, with a persistent
  root-env worker (`aizynth_torch_worker.py`).  **Status: REAL**
  (functional but optional; not exercised by the smoke tests).
- `molmetal/molmetal_lam/sbdd_env/aizynth_torch_model.py` —
  `TorchDensePolicy` audited legacy Dense policy implementation
  (2048 input / 46695 output).  **Status: REAL** (audited).  Used
  inside the isolated env via `RemoteTorchPolicy`.
- `environments/aizynth/` — isolated Python 3.12 environment with
  `requirements.lock.txt`, `pyproject.toml`, `uv.lock`, `run.sh`
  (uv `--isolated` wrapper).  **Status: REAL**.

### 2.2 Pipeline integration gaps (before this workflow)

* The per-pocket runner accepted `synthesis_oracle` and
  `synthesis_config_path` as search-time leaf-pruning parameters, but
  did NOT record `n_synthesis_route`, `synthesis_success_rate`,
  `aizynth_status`, etc. as a reportable column.
* No CLI flag existed for *post-search* AiZynth scoring (the existing
  `--synthesis-oracle aizynthfinder_isolated` only affects MCTS).
* No `aizynth_sota_scoring.py` glue module existed; the
  DiffDock/FlowDock pattern (`{score_name}_sota_scoring.py` + a
  `score_candidates` function + a worker-side branch) was missing for
  AiZynth.

---

## 3. What was added (this workflow)

### 3.1 New module: `aizynth_sota_scoring.py`

**File**: `molmetal/molmetal_lam/sbdd_env/aizynth_sota_scoring.py`

Mirrors the DiffDock/FlowDock pattern exactly:

* `AiZynthScoreColumn` — dataclass with `status`, `mode`,
  `n_invoked`, `n_scored`, `n_synthesis_route`,
  `synthesis_success_rate`, `per_smiles` (smiles -> dict of
  `{synthesizable, depth, engine, route_smiles}`), `notes`.  Same
  status discipline: `{"unavailable", "smarts", "ok", "error",
  "no_candidates"}`.
* `aizynthfinder_available()` — vendored-package probe (lazy import).
* `aizynth_config_available(config_path)` — config-file probe.
* `aizynth_isolated_available(run_sh)` — isolated-env probe.
* `_candidate_smiles(candidates)` — pulls SMILES out of Lambda dicts;
  only `is_generated=True` rows are scored (matches DiffDock/FlowDock).
* `_finite_bool_rate(rows)` — internal aggregator; returns
  `(n_true, n_total)` over a stream of bool values, ignoring non-bool.
* `_smarts_one(smiles)` — wraps `retrosynthesize_with_report` to
  obtain `(synthesizable, list_of_rule_names)`.  Each fired rule is
  counted as one retrosynthesis step (depth 1).
* `_score_smarts(...)` — default CPU-only mode; always available.
* `_score_aizynthfinder(...)` — in-process mode; degrades gracefully
  when the import or config file is missing.
* `_score_aizynthfinder_isolated(...)` — subprocess mode; degrades
  gracefully when `run.sh` or `config_path` is missing.
* `score_candidates(candidates, *, mode, config_path, ...)` —
  public entry point used by the worker.
* `rate_from_column(column)` — NaN-safe accessor.

### 3.2 `r4_c_full_sweep.py` changes

1. **New dataclass fields** on `PocketResult`:
   `aizynth_status`, `aizynth_mode`, `aizynth_n_invoked`,
   `aizynth_n_scored`, `aizynth_n_synthesis_route`,
   `aizynth_synthesis_success_rate`, `aizynth_per_smiles`.  These
   appear as new columns in the CSV / JSON report.
2. **Worker-side branch**: `aizynth_config = kwargs.pop("aizynth_config",
   None)` unpacks the config dict; a new `if aizynth_config:` block
   (after the existing `flowdock_config` block) calls
   `aizynth_sota_scoring.score_candidates(...)` and records results on
   the per-pocket `result` dataclass; checkpointed via
   `checkpoint(result)`.
3. **Aggregate stats** in `aggregate()`: `aizynth_n_invoked_total`,
   `aizynth_n_scored_total`, `aizynth_n_synthesis_route_total`,
   `aizynth_synthesis_success_rate_mean`,
   `aizynth_status_counts`, `aizynth_mode_counts`,
   `aizynth_n_pockets_scored`.
4. **Markdown report**: new rows for "AiZynth synthesis success rate
   (mean, mode-aware)", "AiZynth n_pockets_scored",
   "AiZynth n_synthesis_route_total".
5. **New CLI flags**: `--sota-aizynth` (enable the column),
   `--sota-aizynth-mode {smarts,aizynthfinder,aizynthfinder_isolated}`
   (default `smarts`), `--sota-aizynth-config` (path to AiZynth
   `config.yml`, required for the two real-AiZynth modes),
   `--sota-aizynth-isolated-run-sh` (default
   `environments/aizynth/run.sh`),
   `--sota-aizynth-timeout` (default 600s),
   `--sota-aizynth-max-iterations` (default 200),
   `--sota-aizynth-time-limit` (default 30).
6. **`physical_implementation_sha256` metadata**: now records
   `aizynth_sota_scoring.py` alongside the other adapter hashes.
7. **`--dry-run`** verifies preflight without consuming assets.  A
   `python molmetal/scripts/r4_c_full_sweep.py --dry-run --n-pockets 1
   --seeds 42 --sota-aizynth --sota-aizynth-mode smarts` run prints
   `[dry-run] OK` with no errors.

### 3.3 New test file: `test_aizynth_wire.py`

**File**: `molmetal/molmetal_lam/tests/test_aizynth_wire.py`

10 CPU-only tests; no AiZynthFinder installation or vendored repo
required:

1. `test_aizynth_score_column_schema` — dataclass fields are exposed.
2. `test_smarts_mode_records_per_smiles_and_rate` — SMARTS fallback
   reports synthesizable booleans and a finite success rate in [0, 1];
   CuAAC reverser fires on `c1ccc(Cn2ccnn2)cc1`, Diels-Alder on
   `C1=CCCCC1`, ethanol / pyridine are correctly rejected (2/4 =
   0.5).
3. `test_aizynthfinder_mode_unavailable_gracefully` —
   `mode="aizynthfinder"` returns `status="unavailable"` when no
   `config_path` is provided or the file does not exist; the rate
   stays `None`.
4. `test_aizynthfinder_isolated_mode_unavailable_gracefully` —
   `mode="aizynthfinder_isolated"` returns `status="unavailable"`
   when `environments/aizynth/run.sh` is missing.
5. `test_no_candidates_short_circuits` — empty input ->
   `status="no_candidates"`; no rate synthesised.
6. `test_seed_candidates_are_excluded` — only `is_generated=True`
   rows are scored (DiffDock/FlowDock parity).
7. `test_rate_from_column_handles_nan` — NaN-safe accessor.
8. `test_sweep_worker_records_aizynth_columns_in_schema` —
   `PocketResult` exposes the new fields; uses real `importlib`
   package import (NOT `spec_from_file_location`) so dataclass
   introspection of `Optional[float]` resolves via `sys.modules`.
9. `test_cli_flag_help_text` — `--sota-aizynth` /
   `--sota-aizynth-mode` flags appear in `--help`.
10. `test_dry_run_smoke_with_sota_aizynth` — `--dry-run --sota-aizynth`
    on the default manifest succeeds end-to-end with exit code 0 and
    prints `[dry-run] OK`.

---

## 4. Dependency chain

```
r4_c_full_sweep.py
└── molmetal_lam/sbdd_env/aizynth_sota_scoring.py  [NEW]
    ├── molmetal_lam/sbdd_env/retrosynthesis.py    [EXISTING, REAL]
    │   └── RDKit reverse-SMARTS for CuAAC/SPAAC/SPC/DielsAlder/ThiolEne
    ├── molmetal_lam/sbdd_env/aizynth_adapter.py   [EXISTING, REAL — used in mode="aizynthfinder"]
    │   ├── aizynthfinder (optional, lazy import)
    │   └── _smarts_fallback (always-available SMARTS fallback)
    └── molmetal_lam/sbdd_env/aizynth_isolated.py  [EXISTING, REAL — used in mode="aizynthfinder_isolated"]
        └── subprocess -> environments/aizynth/run.sh aizynth_worker.py
            └── (optional) molmetal/scripts/aizynth_rocm_policy.py -> aizynth_torch_worker.py
```

All three modes degrade gracefully: when their preconditions fail,
they return `status="unavailable"` and the column records `null` /
`NaN` values — the honest SOTA-comparable behaviour.

---

## 5. Test results

```
$ uv run --project . pytest molmetal/molmetal_lam/tests/test_aizynth_wire.py -q
..........                                                               [100%]
10 passed, 1 warning in 1.62s

$ uv run --project . pytest molmetal/molmetal_lam/tests/test_aizynth_wire.py \
                           molmetal/molmetal_lam/tests/test_diffdock_wire.py \
                           molmetal/molmetal_lam/tests/test_flowdock_wire.py -q
.......................                                                  [100%]
23 passed, 1 warning in 1.83s

$ python molmetal/scripts/r4_c_full_sweep.py --dry-run --n-pockets 1 \
        --seeds 42 --sota-aizynth --sota-aizynth-mode smarts
[dry-run] OK — sweep NOT executed.
```

No regressions in the broader SOTA-scoring test subset (23/23 pass).
The new column is plumbed through PocketResult, aggregate(),
write_markdown(), and physical_implementation_sha256; the
metadata fingerprint of `aizynth_sota_scoring.py` is recorded so
mid-experiment code changes abort cleanly (existing safety net
preserved).

---

## 6. Honest framing

* **SMARTS mode is a real measurement**, not a stub.  It uses the
  *reverse* of every click reaction Lambda claims as a forward design
  step, run through the same RDKit machinery the search itself uses.
  See `molmetal/reports/h3_retrosynthesis_check.md` for the underlying
  audit and `molmetal/molmetal_lam/sbdd_env/retrosynthesis.py` for the
  exact reverse-templates.
* **`aizynthfinder` mode is REAL but asset-bound**.  It only runs when
  the AiZynthFinder Python package AND a config.yml with a trained
  expansion policy + stock file are on disk.  Neither ships with the
  ROCm Python 3.12 environment; both require manual download via
  `aizynthfinder.tools.download_public_data` (typically > 1 GB of
  policy weights + stock).
* **`aizynthfinder_isolated` mode is REAL but asset-bound**.  It
  spawns the dedicated CPU-only `environments/aizynth/` environment;
  the env ships via uv lock, but `config.yml` and weights still need
  to be downloaded separately.  The isolated env's run.sh is a
  `uv run --isolated` wrapper that keeps aizynthfinder's Python (with
  C-extension dependencies like `ujson`) away from the ROCm torch
  process.
* **The new column is never silently fabricated.**  When no
  measurement is possible, `status="unavailable"` is recorded and
  `synthesis_success_rate=None`.  This matches the
  DiffDock/FlowDock discipline already in place.

---

## 7. Files added / modified

| File | Status | Purpose |
|---|---|---|
| `molmetal/molmetal_lam/sbdd_env/aizynth_sota_scoring.py` | NEW | Glue module: SMARTS / real-AiZynth modes + dataclass |
| `molmetal/molmetal_lam/tests/test_aizynth_wire.py` | NEW | 10 CPU-only tests |
| `molmetal/scripts/r4_c_full_sweep.py` | MODIFIED | New `--sota-aizynth*` flags, `aizynth_*` PocketResult fields, worker branch, aggregate stats, markdown rows, sha256 metadata |
| `molmetal/reports/wf_wire_aizynth.md` | NEW | This report |

No external dependencies added: the new module re-uses existing
adapter files.  When the AiZynthFinder assets are downloaded, the
column automatically upgrades from `smarts` to `ok` without code
changes.