# DEFERRED — enable `_validate_metrics.py --strict` at project end

**Status:** DEFERRED on explicit user instruction, 2026-10-03.
**User's words:** *"第一点先不开，最后项目结束了再修"*

## §0 What was found

`scripts/_validate_metrics.py` has a `--strict` mode that turns "missing v1
schema fields" into `FAIL`. The pre-commit hook
(`.git/hooks/pre-commit` L47) invokes it **without** `--strict`:

```bash
if ! "${REPO_ROOT}/.venv/bin/python" "${VALIDATOR}"; then
```

MEASURED, on a real commit (2026-10-03):

| mode | result | exit |
|---|---|---|
| tolerant (what the hook runs) | `0 PASS / 154 WARN / 0 FAIL` | **0** |
| `--strict` | `0 PASS / 0 WARN / **154 FAIL**` | **1** |

So the provenance check that AGENTS.md §1 rule 1 and §5.2 depend on
("every numeric value MUST trace to a JSON / report path on disk") has an
enforcement mechanism that **has never fired**.

Fields actually missing, across `metrics/by_metric/*.json` (154 files):

| field | files carrying it |
|---|---:|
| `source_path` | **1** |
| `metric_id` | **1** |

## §1 Why it is deferred, not fixed now

Enabling `--strict` today would turn **all 154** files red and block **every**
commit. That is a project-wide stop, not a fix. It has to be sequenced after
the metric files are brought into schema — or after the schema is brought into
line with the key names actually in use (many files carry provenance under
`source` / `data_source` / `audit_provenance` rather than `source_path`).

**Important nuance (do not overstate this finding):** 153/154 files lacking
`source_path` does NOT mean those metrics are fabricated. Spot-checked, e.g.
`nci60_precious_9cell_auc.json` carries `source`, `timestamp`, `unit` and
`honest_framing` — the provenance is there, under different key names. The
defect is that **the check never runs**, not that the numbers are fake.

## §2 What to do at project end

Pick ONE, deliberately — this is a design decision, not a chore:

- **Option A — conform the files.** Add `metric_id` + `source_path` to all 154
  metric JSONs from their existing provenance fields. Then flip the hook to
  `--strict`. Highest integrity, most work.
- **Option B — conform the schema.** Extend `schemas/metric_v1.schema.json` to
  accept the key names already in use (`source` as an alias of `source_path`,
  etc.). Then flip the hook. Cheap, but weakens the schema.
- **Option C — leave tolerant, but make it loud.** Keep exit 0, but surface the
  WARN count somewhere a human will see it (the commit output is already
  hundreds of lines, so today nobody reads it).

Whichever is chosen, the hook line to change is
`.git/hooks/pre-commit` L47, and the change must be accompanied by a
`SKIP_METRICS=1` escape-hatch note so a bad metric cannot wedge the repo.

## §3 Related

- `TODO/completed/config_dead_key_remediation.md` — the config half of the
  same "declared but never enforced" class.
- AGENTS.md §5.6 (atomic write) + §5.7 (flag wire-in smoke) — the project's
  existing rules for this defect class; this item is the metrics-layer
  instance of it.
