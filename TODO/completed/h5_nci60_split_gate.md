# H5 closeout — split-leak coverage beyond the single MetalCytoTox gate

**Wave:** Wf-446 HOLE-5 (pre-flight split-leak coverage extension)
**Date:** 2026-10-03
**Status:** SHIPPED (filesystem-verified; see §7 `ls -la` evidence)
**Owned files:** `molmetal/scripts/pre_flight_check.py`; `molmetal/tests/test_wf446_h5_nci60_split_gate.py`; `TODO/completed/321d_preflight_gate_audit.md`; this file.

## §1 What changed

### 1.1 `gate_scaffold_zero_leak` — widened from 1 CSV to v2 + v3 + v4

- `molmetal/scripts/pre_flight_check.py` L665: signature is now
  `(name, priority, pipeline, csv_specs: List[Tuple[Path, str]])`.  A single
  subprocess probe iterates the specs and emits one `OK <label> rows=N` /
  `LEAK <label> rows=N leaks=[...]` line per CSV (probe timeout 20 s +
  20 s per CSV).  Missing-file check names each `label=path`.  Env-aware
  SKIP on probe-side pandas ImportError kept (and generalized).
- Registration L8293-8305 passes
  `[(v2, "metalcytotox_v2"), (v3, "metalcytotox_v3"), (v4, "metalcytotox_v4")]`.
  v1 stays EXCLUDED — the pre-Wf-261 leaky artifact (MEASURED 29-48 leaks;
  not a trainer input) — honest-negative note lives in the registration
  comment.
- Docstring now states exactly what is and is not verified (CSV-level
  `scaffold_id` column only — see §5 non-fixes).

### 1.2 NEW `gate_nci60_split_no_leak` (all-pipeline-only)

- Pure helpers: `_nci60_index_triple_violations` L820 (pairwise index
  overlap + partition coverage), `_nci60_scaffold_pair_leaks` L848 (all
  three pair counts, zeros included).
- Gate `gate_nci60_split_no_leak` L868, registered L8307-8317.
  `NCI60_SPLIT_STRICT_SCAFFOLD=1` (env) or `strict_scaffold=True` (arg)
  turns real-scaffold overlap into a hard FAIL.
- Part A: loads `molmetal/data/nci60_precious_9cell.csv` through the REAL
  Wf-A10/m21 loader (`load_nci60_9cell_split_table`), hard-asserts
  cross-cell leakage == 0, unique-size triple == frozen reference
  (3645, 521, 1041), row-index disjointness; then measures REAL
  Bemis-Murcko overlap (`MurckoScaffoldSmiles(..., includeChirality=False)`)
  and reports it.
- Part B: re-runs the chemprop scaffold helper
  (`_scaffold_split_indices_nci60` + `compute_per_cell_n_test_min`) for
  every cell x seed (0,1,2); FAILs on a `None` fallback; asserts index
  disjointness + coverage and test-vs-{train,val} scaffold disjointness.
  train<->val sharing is REPORTED only (documented: non-test pool split at
  molecule level; affects early stopping only).
- Lazy imports -> ImportError becomes SKIP with an honest env-gap note.

### 1.3 Gate-count sync 101 -> 103 + audit-doc sync

- `EXPECTED_GATE_COUNTS["all"]` L3600: 101 -> 103, MEASURED via
  `len(build_gates("all"))`.  Comment records the full history: the H4 fix
  (this run, Wf-446) added the all-only `p3_dmpnn_loss_wire_in_smoke`
  (101 -> 102 measured; constant left at 101 — pre-existing drift the HOLE-5
  edit absorbs), then HOLE-5 added `p0_nci60_split_no_leak` (102 -> 103).
  `cfm` NOT moved (HOLE-1 canary is cfm-only; `all` runs the identical smoke
  under `p3_wf385_scaffold_loss_wire_in_verified`).
- `TODO/completed/321d_preflight_gate_audit.md`: §2.1 header + Total row +
  the `EXPECTED_GATE_COUNTS` code block + §6 `n_gates_total` row synced to
  103; NEW §6.3 changelog entry.
- `molmetal/tests/test_wf327f11_gate_count_consistency.py`: 3/3 PASS
  (was RED at 101 vs measured 102 before this edit).

## §2 FAIL proofs (deliberately broken input)

| # | Input | Result (captured) |
|---|---|---|
| F1 | synthetic CSV with `scaffold_id` s2 on train+test rows | `FAIL` — `LEAK leaky_synthetic rows=4 leaks=[('test', 'train', 1)]` |
| F2 | missing nci60 CSV path | `FAIL` — `nci60 CSV missing: /tmp/nci60_missing.csv` |
| F3 | truncated nci60 copy (head 4000 rows) | `FAIL` — `4 problem(s): m21 split unique-molecule sizes (1898, 271, 543) != frozen reference (3645, 521, 1041) (m21 comparability broken); HL60/seed0..2: scaffold helper returned None -> random 80/10/20 fallback (scaffold protection OFF; n_test_min=5)` |
| F4 | real tree, `strict_scaffold=True` | `FAIL` — `1 problem(s): m21 split REAL scaffold leaks (strict mode): train/val=165, train/test=282, val/test=121` |
| F5 | missing file inside a multi-CSV spec list | `FAIL` (covered by test `test_fails_on_missing_file`) |

## §3 PASS proofs (healthy tree, final code)

- `p0_scaffold_zero_leak` — 0.5 s:
  `0 scaffold leak in all 3 CSVs: OK metalcytotox_v2 rows=26632; OK metalcytotox_v3 rows=26632; OK metalcytotox_v4 rows=26632`
- `p0_nci60_split_no_leak` — 26.3 s:
  `m21/A10 loader: rows=7823 unique=5207 unique-by-split(train,val,test)=(3645, 521, 1041) row-by-split=(5415,804,1604) cross-cell-leaks=0 REAL-scaffold-overlap(train/val,train/test,val/test)=(165, 282, 121) splitter_scaffold_for_degenerate=True | HONEST-NEGATIVE: ... | chemprop scaffold helper: cells=9 seeds=[0, 1, 2] probes=27 fallbacks=0 test-vs-rest-scaffold-leaks=0 train/val-shared-scaffold-keys=1120 (...)`

## §4 SKIP proofs (env-aware)

- Multi-CSV gate, probe runtime without pandas -> `SKIP` with an env-gap
  note naming the canonical venv (`.venv/bin/python`, AGENTS.md §5.9);
  the gated CSVs ARE present and structurally readable.
- nci60 gate, helper module unimportable -> `SKIP` (env gap; CSV present).

## §5 Deliberate non-fixes + HONEST-NEGATIVE (MEASURED)

1. **MAJOR FINDING, NOT FIXED (out of file ownership):** the shipped nci60
   "scaffold" split is molecule-level, NOT scaffold-disjoint.  MEASURED
   real Bemis-Murcko overlap: train/val=165, train/test=282, val/test=121.
   Root cause: `molmetal/data/splits.py::ScaffoldSplitter._scaffold_for`
   (L595-609) calls `MurckoScaffold.GetScaffoldForMol(mol, includeChirality=...)`
   — a kwarg the installed rdkit rejects (TypeError) — and L604
   `except Exception: return ""` swallows it, keying every molecule to
   `__empty__<canon>`.  m21 (`train_nci60_10cell.py`) uses the same code,
   hence the byte-identical 3645/521/1041 sizes; the Wf-A10 config is
   pinned to m21's split for comparability.  Changing `splits.py` would
   change every consumer's split — out of this fix's ownership.  The gate
   reports the overlap loudly every run and offers
   `NCI60_SPLIT_STRICT_SCAFFOLD=1` to hard-FAIL once the splitter is fixed.
2. platinai_4cell / DrugOOD / bindingdb_ic50: NOT gated — no on-disk split
   manifest emitted at train time (bindingdb CSV resolves to a broken/empty
   symlink).  Stall until a split manifest exists; documented in the gate
   docstring (non-fix by design).
3. chemprop driver's default `--split temporal` (random mode) path is not
   covered by design — the gate verifies the scaffold-helper path only.
4. Helper probes exercise the auto-threshold path only;
   `--no-n-test-min-auto --n-test-min 50` (explicit override) is not probed.
5. v1 metalcytotox CSV excluded (pre-Wf-261 artifact, 29-48 leaks,
   not a trainer input).

## §6 Tests

- NEW `molmetal/tests/test_wf446_h5_nci60_split_gate.py` (124 lines,
  10 tests, CPU-only, no network/GPU, MEASURED 27.14 s): pure helpers;
  multi-CSV clean PASS / injected-leak FAIL / missing FAIL; nci60
  missing-CSV FAIL; healthy-tree integration PASS asserting the
  honest-negative markers (`REAL-scaffold-overlap`,
  `splitter_scaffold_for_degenerate=True`, `HONEST-NEGATIVE`).
  Result: 10 passed.
- `molmetal/tests/test_wf327f11_gate_count_consistency.py`: 3 passed.

## §7 Files touched + on-disk evidence (AGENTS.md §5.5)

```
$ ls -la molmetal/scripts/pre_flight_check.py molmetal/tests/test_wf446_h5_nci60_split_gate.py TODO/completed/321d_preflight_gate_audit.md
-rw------- 1 hugo hugo 429131 Oct  3 10:58 molmetal/scripts/pre_flight_check.py
-rw-r--r-- 1 hugo hugo   4888 Oct  3 10:55 molmetal/tests/test_wf446_h5_nci60_split_gate.py
-rw-r--r-- 1 hugo hugo  19100 Oct  3 10:58 TODO/completed/321d_preflight_gate_audit.md
```

(plus this file, stat captured after write:)

```
$ ls -la TODO/completed/h5_nci60_split_gate.md
-rw------- 1 hugo hugo 8322 Oct  3 11:00 TODO/completed/h5_nci60_split_gate.md
```

142 lines / 8322 bytes.

No commit (per task rules).  No `--pipeline all` full run (specific gate
functions called directly).  No `paper/*` / `docs/PAPER_*` touches.  No
pre-commit hook / `.pre-commit-config.yaml` edits.  No
`scripts/_validate_metrics.py` edits.  No gate BLOCKs commits — pre-flight
remains report-only.

Written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout`
(AGENTS.md §5.6).
