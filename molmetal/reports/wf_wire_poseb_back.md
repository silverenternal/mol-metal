# WF-Wire-PoseBusters: chemistry validity on docked candidates

## Goal

Wire the existing ``PoseBustersAdapter`` (full Python API wrapper at
``molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py``) into the
``r4_c_full_sweep`` pipeline as a *post-docking* chemistry validity
check on each generated candidate. The new ``--pb-check`` flag invokes
``PoseBustersAdapter.validate_mol`` on every generated candidate's
SMILES after Vina docking (gated by ``--physical-docking``) and
records ``n_pb_pass`` / ``pb_pass_rate`` columns in ``report.json``.

## What was already there

* ``molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`` —
  ``PoseBustersAdapter`` class wrapping ``posebusters.PoseBusters``
  with ``validate_mol`` / ``validate_list`` / ``pass_rate` /
  ``get_metadata`` methods; ``ValidityReport`` dataclass; full
  ETKDGv3 + MMFF94 conformer pipeline with UFF fallback.
* ``molmetal/validation/posebusters_runner.py`` —
  ``check_docked_pose`` runs the docked *coordinates* through PB
  ``dock`` mode inside ``evaluate_generated_poses``. This already
  populates ``physical.summary.n_pb_pass`` in the existing report.

The new ``--pb-check`` flag is a *separate* path: it runs the
**chemistry + geometry** PB check (``mode='mol'``) on each candidate's
own SMILES, independent of the docked pose. This separation matters
because:

1. ``check_docked_pose`` evaluates the docked coordinates (which
   depend on the receptor + box) and is useful for catching
   pose-vs-protein clashes.
2. ``PoseBustersAdapter.validate_mol`` evaluates the molecule *on its
   own*, asking "is this SMILES even a chemically valid drug-like
   molecule?" — independent of any receptor.

Together they answer two different questions. Both are wired now.

## What was added

### 1. ``--pb-check`` CLI flag

```text
--pb-check            Run PoseBustersAdapter chemistry validity check on
                      each generated candidate after Vina docking (requires
                      --physical-docking; CPU-only)
--pb-mode {mol,dock,redock}
                      PoseBusters adapter mode (default 'mol' = chemistry +
                      geometry, no protein)
```

* ``--pb-check`` is rejected at preflight if ``--physical-docking`` was
  not set (the PB check runs after Vina docking, so it requires the
  docking phase to be enabled).
* ``--pb-mode`` selects the adapter mode; default ``mol`` matches the
  de-novo generation setting (chemistry + geometry, no protein).

### 2. Worker-side execution path

The worker (when invoked via ``--worker-input``) consumes a new
``pb_config`` dict and, after the ``physical_config`` (Vina docking)
block runs, iterates over every generated candidate's SMILES and calls
``PoseBustersAdapter(mode=...).validate_list(...)``. The results are
recorded into the ``PocketResult`` under:

* ``n_pb_pass`` (int) — number of candidates that pass all PB checks
* ``pb_pass_rate`` (float | None) — ``n_pb_pass / n`` (None when no candidates)
* ``pb_status`` (str) — one of ``completed``, ``no_candidates``,
  ``unavailable``, ``error``, ``disabled``
* ``pb_check`` (dict) — full per-SMILES report (passed, n_checks,
  failed_checks, ...)

### 3. Graceful degradation

* **PB not installed** → ``ImportError`` caught → ``pb_status='unavailable'``,
  ``n_pb_pass=0``, ``pb_check={status: 'unavailable', reason: ...}``.
  This is consistent with how the runner-path (``check_docked_pose``)
  already handles the missing package.
* **No generated candidates** → ``pb_status='no_candidates'``,
  ``n_pb_pass=0``, ``pb_pass_rate=None``.
* **PB raises** → ``pb_status='error'``, ``pb_check={status: 'error',
  error: 'Type: msg'}``. The pipeline continues.

### 4. Aggregate columns in ``report.json`` ``summary``

Three new keys are added by ``aggregate()``:

* ``n_pb_pass`` (int) — total pass count across all pockets
* ``pb_pass_rate`` (float | None) — mean of per-pocket rates
* ``pb_n_jobs_with_check`` (int) — number of pockets that actually ran
  the PB check (excludes ``disabled``)

### 5. Markdown report

Two new rows appear in ``write_markdown``:

| Candidates passing PoseBusters (chemistry, post-dock) | N |
| PoseBusters (chemistry) pass rate | X.XXX |

### 6. Metadata

The JSON metadata's ``not_executed`` list now records
``posebusters_adapter_extra_check`` when ``--pb-check`` was *not* set,
so the audit trail is honest about which extra checks were skipped.

## Honest framing

* The PB check here is a **chemistry + geometry validity** check on
  the candidate's own conformer. It is **not** a docking-quality
  assessment. The docking-quality check (clash, intermolecular
  contacts) lives in the existing ``check_docked_pose`` path inside
  ``evaluate_generated_poses`` (recorded in
  ``physical.summary.n_pb_pass``).
* The flag requires ``posebusters`` to be installed. On environments
  where it is not, the worker degrades to ``pb_status='unavailable'``
  rather than crashing. This matches the existing
  ``check_docked_pose`` fallback path.
* The flag requires ``--physical-docking`` to be enabled. PB runs on
  generated candidates *after* Vina; without Vina there are no docked
  candidates to check.

## Tests

``molmetal/molmetal_lam/tests/test_poseb_backers_wire.py`` — 8 tests,
all passing:

1. ``test_posebusters_adapter_importable`` — adapter class + ValidityReport
2. ``test_pocket_result_has_pb_check_fields`` — new dataclass fields
3. ``test_aggregate_includes_pb_columns`` — summary keys
4. ``test_worker_records_pb_check_in_output_json`` — end-to-end worker
   wire with mocked PoseBustersAdapter
5. ``test_pb_check_requires_physical_docking`` — preflight ValueError
6. ``test_pb_check_handles_missing_posebusters`` — graceful
   ``unavailable`` fallback
7. ``test_validate_list_produces_per_smiles_reports`` — per-SMILES report
8. ``test_pb_adapter_metadata_shape`` — ``get_metadata`` shape

Test result: ``8 passed, 1 warning in 1.56s`` on uv-managed Python 3.12.

## Files modified

* ``molmetal/scripts/r4_c_full_sweep.py``
  - Added ``n_pb_pass`` / ``pb_pass_rate`` / ``pb_status`` / ``pb_check``
    fields to ``PocketResult``.
  - Added ``pb_config = kwargs.pop("pb_config", None)`` in the worker.
  - Added worker-side ``PoseBustersAdapter`` invocation block after
    the ``physical_config`` block.
  - Added ``--pb-check`` and ``--pb-mode`` CLI flags.
  - Added preflight ValueError when ``--pb-check`` is set without
    ``--physical-docking``.
  - Added ``pb_config`` dict construction in main() when ``--pb-check``
    is set.
  - Added ``n_pb_pass`` / ``pb_pass_rate`` / ``pb_n_jobs_with_check``
    to ``aggregate()``.
  - Added two new rows to ``write_markdown``.
  - Added ``pb_check`` to the JSON-dumped CSV columns.
  - Added ``posebusters_adapter_extra_check`` to ``not_executed``
    metadata when ``--pb-check`` was not set.
  - Added ``pb_check : ...`` line to dry-run output.

## Files added

* ``molmetal/molmetal_lam/tests/test_poseb_backers_wire.py`` —
  8 wire-up tests (CPU-only, mocks ``posebusters`` package).
* ``molmetal/reports/wf_wire_poseb_back.md`` — this report.

## How to run

```bash
# Smoke (CPU-only; uses mocked PB via tests):
uv run pytest molmetal/molmetal_lam/tests/test_poseb_backers_wire.py -v

# Real run (requires posebusters package + Vina installed):
uv run python -m molmetal.scripts.r4_c_full_sweep \
    --physical-docking --pb-check --pb-mode mol \
    --n-pockets 5 --seeds 42 0 1234 \
    --output-prefix molmetal/reports/wf_wire_poseb_smoke
```

## Dependency chain

``r4_c_full_sweep --pb-check`` → worker pops ``pb_config`` → invokes
``molmetal_lam.sbdd_env.posebusters_adapter.PoseBustersAdapter`` →
constructs ``PoseBusters(config=mode)`` → calls ``validate_list`` on
generated candidates' SMILES → per-SMILES ``ValidityReport``s
serialised into ``result.pb_check``.
