# WF-Wire-Clone-Scoring: DiffDock adapter as SOTA scoring column

**Date:** 2026-09-14
**Workflow:** WF-Wire-Clone-Scoring (Round-12 continuation)
**Goal:** Wire the existing DiffDock-L adapter (`molmetal_lam/sbdd_env/diffdock_adapter.py`)
as a SOTA scoring column in `molmetal/scripts/r4_c_full_sweep.py`.

---

## 1. Diff

### 1.1 New module — `molmetal/molmetal_lam/sbdd_env/diffdock_sota_scoring.py`

A pipeline-side glue module that turns a list of Lambda candidates
into per-candidate DiffDock-L confidence scores.  Public API:

* `DiffDockScoreColumn` dataclass — per-pocket aggregate with
  `status`, `n_invoked`, `n_scored`, `diffdock_score_mean`,
  `diffdock_score_std`, `per_smiles`.
* `diffdock_cli_available(repo_root)` — discovers the vendored
  `inference.py` and the `utils/` package.
* `score_candidates(candidates, protein_path, ...)` — main entry
  point.  Subprocess-calls `inference.py` per SMILES, parses the
  rank1 confidence from the resulting `rank1_confidence{X.XX}.sdf`
  filename, aggregates.

The helper degrades gracefully:

* Vendored repo missing → `status="unavailable"`.
* Protein path missing → `status="error"`.
* Subprocess rc != 0 → entry recorded as `NaN`; aggregate
  `status="partial"` if any candidate scored, otherwise
  `"error"`.

### 1.2 Extension — `molmetal/molmetal_lam/sbdd_env/diffdock_adapter.py`

Added `DiffDockAdapter.parse_confidence_from_outdir(out_dir, complex_name)`
helper.  Pure stdlib `Path.glob` + filename parsing — no torch,
no upstream imports.

### 1.3 Wire-up — `molmetal/scripts/r4_c_full_sweep.py`

* New CLI flags:

  ```
  --sota-diffdock                      enable DiffDock SOTA column
  --sota-diffdock-samples N            samples_per_complex (default 4)
  --sota-diffdock-timeout SEC          per-SMILES timeout (default 600)
  --sota-diffdock-repo PATH            vendored DiffDock checkout
  --sota-diffdock-config PATH          default_inference_args.yaml override
  ```

* New `PocketResult` columns:
  `diffdock_score_mean`, `diffdock_score_std`, `diffdock_status`,
  `diffdock_n_invoked`, `diffdock_n_scored`, `diffdock_per_smiles`.

* Worker-side hook (after `physical_config`, before exit):
  if `diffdock_config` is supplied, run `score_candidates` and
  update the checkpointed JSON between iterations.

* `aggregate()` now reports
  `diffdock_score_mean`/`std` across pockets plus
  `diffdock_status_counts` so the summary makes the
  unavailable/ok/partial distribution explicit.

### 1.4 New tests — `molmetal/molmetal_lam/tests/test_diffdock_wire.py`

Five tests, all CPU-only, all passing:

| Test | Purpose |
|---|---|
| `test_diffdock_adapter_importable` | API surface (Adapter, DockResult, Protocol, Exception) |
| `test_diffdock_subprocess_call_smoke` | End-to-end subprocess path with mocked `subprocess.run`; verifies confidence parsed from `rank1_confidence{X.XX}.sdf` filename |
| `test_diffdock_score_recorded_in_report` | Drives the r4 worker with `diffdock_config`; verifies the new JSON columns are present and the mean/std are correct |
| `test_diffdock_score_candidates_unavailable` | Graceful fallback when vendored repo missing |
| `test_diffdock_parse_confidence_from_outdir` | Outdir parser round-trip + NaN handling |

---

## 2. Test results

```
$ uv run python -m pytest molmetal/molmetal_lam/tests/test_diffdock_wire.py -v
... collected 5 items
molmetal/molmetal_lam/tests/test_diffdock_wire.py::test_diffdock_adapter_importable PASSED
molmetal/molmetal_lam/tests/test_diffdock_wire.py::test_diffdock_subprocess_call_smoke PASSED
molmetal/molmetal_lam/tests/test_diffdock_wire.py::test_diffdock_score_recorded_in_report PASSED
molmetal/molmetal_lam/tests/test_diffdock_wire.py::test_diffdock_score_candidates_unavailable PASSED
molmetal/molmetal_lam/tests/test_diffdock_wire.py::test_diffdock_parse_confidence_from_outdir PASSED
========================= 5 passed in 1.41s ==========================
```

Combined run with the existing diffdock/flowdock adapter tests:

```
$ uv run python -m pytest molmetal/molmetal_lam/tests/test_diffdock_flowdock_adapters.py \
                            molmetal/molmetal_lam/tests/test_diffdock_wire.py -v
... 12 passed in 1.62s
```

CLI wiring sanity check:

```
$ uv run python molmetal/scripts/r4_c_full_sweep.py --help | grep diffdock
  --sota-diffdock       Subprocess-call DiffDock-L for every Lambda candidate
  --sota-diffdock-samples N
  --sota-diffdock-timeout SEC
  --sota-diffdock-repo PATH
  --sota-diffdock-config PATH
```

---

## 3. Limitations & honest framing

1. **No live DiffDock inference in this environment.**  The upstream
   `molmetal/references/DiffDock` is vendored but its `inference.py`
   imports `torch_geometric`, `esm`, `torch_scatter`, and other
   GPU-only extensions that we have not built for ROCm 7.2 / RX 7800 XT
   (gfx1101).  When the tests run they detect the vendored repo at
   `molmetal/references/DiffDock/inference.py` and the `utils/`
   package, so the subprocess path is *available*; the only thing
   preventing a real end-to-end run is the missing torch_cluster /
   torch_scatter ROCm wheels.  This is an environment limitation,
   not a wire-up defect.

2. **CPU-only smoke tests.**  All five tests are designed to run
   without a GPU.  `test_diffdock_subprocess_call_smoke` and
   `test_diffdock_score_recorded_in_report` monkeypatch
   `subprocess.run` to fabricate `rank1_confidence{X.XX}.sdf` files
   rather than invoke the heavy model.  The wire-up itself (CLI →
   JSON → worker → `score_candidates` → JSON) is exercised
   end-to-end.

3. **Per-SMILES timeout.**  Each candidate is subprocess-isolated
   with its own `--out_dir` and `--complex_name r4_complex`.  Default
   timeout is 600 s, samples_per_complex is 4 (vs. upstream default
   10).  In a real run with 50 pockets × 3 seeds × 10 candidates,
   this is still O(hours) — not a smoke budget.

4. **Honest fallback semantics.**  When DiffDock is unavailable,
   `diffdock_status="unavailable"`, `diffdock_score_mean=None`,
   `diffdock_score_std=None`, and the aggregate reports
   `diffdock_n_pockets_scored=0`.  We do **not** synthesise
   confidence values.  The column is honest-NA by construction.

5. **CrossDocked receptor PDB may need preprocessing.**  The
   upstream `inference.py` expects pre-processed protein files
   (e.g. `examples/*_protein_processed.pdb`).  If the manifest's
   `receptor_path` is a raw PDB the upstream loader will fail.  In
   that case each candidate gets `NaN` and the aggregate reports
   `status="partial"` or `"error"`.  A future task could wire
   `prepare_crossdocked_receptor.py` ahead of the DiffDock pass.

6. **No retry logic.**  A single transient failure (network, OOM,
   missing checkpoint weight) marks that candidate NaN.  No retry
   counter, no exponential backoff.  This is intentional — we want
   the report to honestly reflect transient failures, not paper over
   them.

---

## 4. Files touched

| File | Change |
|---|---|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/diffdock_adapter.py` | Added `parse_confidence_from_outdir` static helper |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/sbdd_env/diffdock_sota_scoring.py` | **NEW** — pipeline-side glue module |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_c_full_sweep.py` | Added `--sota-diffdock` family of flags; new `PocketResult` columns; worker hook; aggregate summary fields |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_diffdock_wire.py` | **NEW** — 5 tests, all CPU-only |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_wire_diffdock.md` | **NEW** — this report |

---

## 5. Recommended next steps (out of scope here)

1. Build the ROCm-flavoured `torch_cluster` / `torch_scatter`
   wheels for gfx1101 so the wire-up can run live.
2. Wire `prepare_crossdocked_receptor.py` ahead of the DiffDock
   pass so raw PDB manifest entries work out-of-the-box.
3. Add per-pocket confidence histograms to the report.
4. Promote DiffDock confidence to a *search reward* (not just a
   post-hoc column) by feeding it into the MCTS aggregator — but
   only if the GPU build lands; otherwise wall-time blows up.
