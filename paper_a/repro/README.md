# Paper A — metric reproducibility

Every quantitative claim in `paper_a/` traces to a JSON artifact under
`metrics/by_metric/`. This directory holds the map from artifact to producer, and
the driver that proves the map is honest: it re-runs the CPU-reproducible
producers into a scratch directory and diffs the result against the artifact that
is actually on disk.

## Files

| File | Role |
|---|---|
| `metric_manifest.json` | One entry per cited artifact: producer script, exact command, inputs, cost class, whether it is reproducible from committed code. |
| `../scripts/reproduce_paper_a.py` | Reads the manifest, runs the CPU subset, diffs with `scripts/_metric_diff.py`. Never writes to `metrics/by_metric/` (AGENTS.md §5.5). |
| `../scripts/reproduce_paper_a.sh` | Thin wrapper (single-venv, §5.9). |

## Usage

```bash
scripts/reproduce_paper_a.sh              # CPU subset: regenerate + diff
scripts/reproduce_paper_a.sh --dry-run    # print the plan, run nothing
scripts/reproduce_paper_a.sh --only metal_activity_transfer_tie_sensitivity
scripts/reproduce_paper_a.sh --include-gpu-long   # also GPU entries (hours)
# or:
make paper-a-repro
make paper-a-repro-plan
```

Output is `PASS` / `FAIL` / `SKIP` per entry (the repo's stdout invariant), with a
summary line. Exit code is 0 when every executed entry matches, 1 otherwise.

## Cost classes

* `cpu_seconds`, `cpu_minutes` — run by default.
* `gpu_minutes`, `gpu_hours` — skipped unless `--include-gpu-long`. GPU runs are
  not bit-reproducible run-to-run (bf16 reduction order), so their manifest
  entries state the expected drift and the comparison tolerance.
* `blocked` — no pinned producer; the entry records *why* (a superseded artifact,
  or a raw input that was ephemeral) so the gap is visible rather than silent.

## Tolerance, not byte-equality

Scorers are thread-nondeterministic, so the driver compares *structure* and
*values within a tolerance* (`abs_tol` per entry, default `1e-6`; some artifacts
round to fewer digits). `scripts/_metric_diff.py` also refuses to treat `True` as
`1`. A byte diff would either be permanently red or meaningless.

## Known gaps

`reproducible_from_committed_code: false` entries are the ones that would block a
"regenerate everything from scratch" claim. They are listed in the manifest with
the reason (unpinned producer / ephemeral input), not hidden.
