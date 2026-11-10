# Retired launch wrappers

Moved here 2026-10-10 by user decision. **Not deleted** — the launch record is
the provenance for numbers that are cited in `todo.json` and
`metrics/by_metric/`, so removing it would make those numbers unreproducible.

## Why these seven

`scripts/_wrapper_audit.py` audits every `scripts/launch_wf*.sh` against six
canonical markers. These seven were the entire violation set (measured
`rc=1`, 7/32):

| wrapper | missing markers |
|---|---|
| `launch_wf_blk_d1_attentivefp_5seed.sh` | USER_OK gate, DRY_RUN emission, plan.json emission, USER_OK env propagation |
| `launch_wf_blk_d2_chemprop_v3_frozen_5seed.sh` | (same four) |
| `launch_wf_fix3lines_bace.sh` | (same four) |
| `launch_wf_fix3lines_cfm.sh` | (same four) |
| `launch_wf_fix3lines_cfm_100ep.sh` | (same four) |
| `launch_wf_fix3lines_cfm_100ep_qed.sh` | the four, plus **no `pre_flight_check` gate at all**, plus no PYTHONPATH export |
| `launch_wf_fix3lines_cfm_30ep_onecycle.sh` | the four, plus no PYTHONPATH export |

Two reasons for retiring rather than rewriting:

1. **Scale.** Compliant wrappers run 9,187–34,915 B (median 17,413 B). These
   seven were 2,464–5,176 B — roughly 1/3 to 1/6 of the template. Bringing them
   up to standard means rewriting each into the full template, which is real
   work for launchers nobody will run again.
2. **They're spent.** All seven drove 2026-10-09 waves whose results are already
   on disk in `metrics/by_metric/` and `todo.json`.

What *was* fixed before retiring, because it was a genuine robustness issue
rather than a style marker: five of them invoked the interpreter through the
**relative** path `.venv/bin/python`, relying on an earlier `cd` to resolve it.
AGENTS.md §5.9 requires the absolute `/home/hugo/codes/mol-metal/.venv/bin/python`.
All seven now use the absolute form (two via a canonical `PY=` assignment).
Verified with `bash -n` on each and by re-running `_wrapper_audit.py`.

## Why a subdirectory works

`_wrapper_audit.py::discover_wrappers` is
`sorted(scripts_dir.glob("launch_wf*.sh"))` — **non-recursive**. Files here are
out of its scope by construction, not by an exemption list.

## If you need to re-run one of these

`git mv scripts/_retired/<name>.sh scripts/<name>.sh` to bring it back into the
audit's scope. Bring it up to the canonical template at the same time, or the
gate goes red again — and that is the correct outcome, because a launcher that
cannot be audited is a launcher whose reproducibility claim should not be
trusted.

`launch_wf_fix3lines_cfm_c5_repair.sh` was **not** moved: it is compliant, and it
is cited by live code at
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:5849`.

## Reference note

These seven are named in `TODO/completed/*.md` and `todo.json`. Those are
historical records of completed waves, so the paths there now read
`scripts/_retired/<name>.sh`. `train_cfm_yuelbond_wasserstein.py:5849` is the
only *code* reference, and it points at `..._c5_repair.sh`, which did not move.

## 2026-11-04 — one CPU-lock line repaired (Wf-A269 wave-21)

`launch_wf_fix3lines_cfm.sh` line 61 read `export CUDA_VISIBLE_DEVICES=0`, which
**hard-forces GPU 0** and defeats an ambient `CUDA_VISIBLE_DEVICES=""` lock. It is
now

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
```

— the **no-colon** `${VAR-default}` form, which substitutes only when the variable is
**UNSET**, so an explicitly empty (locked) ambient mask survives.

**The historical launch record is unaffected.** With no ambient mask set — the
condition under which the 2026-10-09 wave actually ran — the value is still `0`,
byte-identical in behaviour. The line differs **only** when an operator has set
the ambient mask, which is exactly the case the repo's standing 0-GPU directive
requires to be honoured.

**Why a retired launcher was touched.** The re-run path documented above
(`git mv scripts/_retired/<name>.sh scripts/<name>.sh`) makes the file a **LIVE**
launcher under `scripts/`, where the old line was a live defect. The repair is
therefore load-bearing for that documented operation.

**Note for the next census.** `molmetal/scripts/wfa167_command_prefix_guard.py`
classifies `ARCHIVED` by a **path prefix** (`TODO/archive/`, `results/`) and does
**not** list `scripts/_retired/`, so this tree is scanned as **live** even though
this directory's own policy (and `scripts/_yaml_only_gate.py:108,131`) call it
archival. The correct repair is a **CITED** exclusion table, not a path heuristic
— recorded as an OPEN item in `docs/HONEST_NEGATIVE_CATALOG.md` §6.269.31.

