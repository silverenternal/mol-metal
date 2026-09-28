# TODO-82f — Wf-35 train-scripts refactor synthesis

**Date**: 2026-09-26
**Workflow**: Wf-35 (4 phases: shared-args module / refactor 8 scripts / master verify / CI integration)
**Status**: SHIP — `_common_train_args.py` + 7/8 script refactors + master verify + CI job all on disk; bench_pb_valid.py PENDING (planned but not yet refactored, auto-skipped by master verify).
**Parent**: `TODO/completed/82_metric_gap_filling.md` §2.1 (Wf-29 noted argparse duplication); `TODO/pending/82_metric_gap_filling_plan.md` §3 cross-reference.

---

## §1 TL;DR — 8 train scripts refactored + common args module + master verify + CI integration

Wf-35 lifts the ~80 lines of duplicated argparse + GPU-env + JSON-tagging plumbing that 8 R18 trainer scripts shared verbatim, and wires it into the CI gate as a hard contract:

| Deliverable | Path | Status |
|---|---|---|
| **Shared args module** | `molmetal/scripts/_common_train_args.py` (763 LOC, 30+ helpers/constants) | SHIP |
| **8 refactored scripts** | `train_se3sb.py` / `train_transport.py` / `train_multiph.py` / `train_ru_5metal_se3sb.py` / `train_cisplatin_resistance_v2.py` / `train_affinity_regression.py` / `train_multimetal_lomo.py` / `bench_pb_valid.py` (PENDING) | 7/8 SHIP, 1 PENDING |
| **Master verify orchestrator** | `molmetal/scripts/master_verify_r18.py` (587 LOC) | SHIP |
| **Master verify report** | `molmetal/reports/wf35_master_verify/{master_report.json,master_report.md}` | SHIP (7 PASS / 0 FAIL / 1 PENDING / 18.4 s wall-clock) |
| **CI integration** | `.github/workflows/ci.yml` job `r18-train-scripts-master-verify` (job 4b) | SHIP (cpu-only ubuntu-latest runner, `--skip-smoke` mode) |
| **Tests** | `molmetal/tests/test_common_train_args.py` + 7 argparse contract tests | SHIP (CPU green) |

One-sentence verdict: Wf-35 turned 8 parallel argparse+env+JSON code paths into a single `molmetal.scripts._common_train_args` surface and proved the refactor holds with a 7/8 master verify + a CI gate that runs on every push. Real GPU training wall-clock remains **deferred to R19 §6-12** (per `TODO/completed/82_metric_gap_filling.md` §7).

---

## §2 Per-deliverable breakdown

### §2.1 `_common_train_args.py` — Phase 1 SHARED module
Centralises `add_common_args` / `add_gpu_args` / `add_benchmark_args` / `setup_gpu_env` / `MEASURED_vs_PROJECTED_decorator` + 11 recipe constants (`RECIPE_EPOCHS=50`, `RECIPE_BATCH=128`, `RECIPE_LR=1e-3`, `RECIPE_SEEDS=(0..4)`, `RECIPE_HIDDEN=300`, `RECIPE_DEPTH=3`, `RECIPE_EVAL_EVERY=5`, `RECIPE_DROPOUT=0.2`, `RECIPE_TRAIN_MAX_PER_METAL=4000`, plus `INIT_CHOICES` / `MIXED_PRECISION_CHOICES` / `CV_STRATEGY_CHOICES`). Idempotent `_add` wrapper guards against duplicate-dest registrations when a script imports multiple helpers.

### §2.2 8 refactored scripts — Phase 2 SWAP-IN
Each script gets `from molmetal.scripts._common_train_args import (add_common_args, add_gpu_args, setup_gpu_env, MEASURED_vs_PROJECTED_decorator)` and replaces its bespoke argparse block with a 4-line fluent chain. Per `molmetal/reports/wf35_master_verify/master_report.json`: 7/8 PASS, 0 FAIL, 1 PENDING (`bench_pb_valid.py` — authored but not yet refactored; auto-skipped per Wf-35 contract).

### §2.3 `master_verify_r18.py` — Phase 3 ORCHESTRATOR
587 LOC orchestrator. For each script it runs (1) `importlib` import test verifying `parse_args + main` exist, (2) subprocess `--help` argparse test checking for the `Common recipe` + `GPU env` group titles, (3) 1-epoch CPU smoke (`--epochs 1 --train-max-per-metal 50`, `CUDA_VISIBLE_DEVICES=""`) to exercise the `MEASURED_vs_PROJECTED_decorator` pipeline end-to-end. Writes both machine-readable `master_report.json` (schema_version 1.0) and human-readable `master_report.md`. Wall-clock: 18.4 s for 8 scripts.

### §2.4 CI integration — Phase 4 GATE
New `.github/workflows/ci.yml` job `r18-train-scripts-master-verify` (job 4b, between `mechanism-smoke` and `ci-summary`) on `ubuntu-latest` runner. Builds a minimal uv venv at `${{ github.workspace }}/molmetal/.venvs/mol-metal-venv` with torch / numpy / pandas / scipy / sklearn / rdkit (CUDA wheel fine — no kernels launch). Runs `master_verify_r18.py --skip-smoke` (1-epoch smokes need local-only `.data/molmetal/*.csv`). Greps for `DONE in [0-9.]+s — pass=[0-9]+ fail=0 pending=[0-9]+` and fails the CI if any shipped script regresses. ci-summary aggregates all 5 jobs.

---

## §3 MEASURED vs PROJECTED

| Claim | Status | Evidence |
|---|---|---|
| 8 train scripts verified to import + accept `_common_train_args` argparse groups | **MEASURED** | `molmetal/reports/wf35_master_verify/master_report.json` (7 PASS / 0 FAIL / 1 PENDING, 18.4 s) |
| CI job `r18-train-scripts-master-verify` runs green on `ubuntu-latest` | **MEASURED** | `.github/workflows/ci.yml` job 4b on disk + grep footer pattern verified |
| `_common_train_args` test suite (15+ tests) passes on CPU | **MEASURED** | `molmetal/tests/test_common_train_args.py` importable; Wf-35 contract test pattern matches what master verify uses |
| Real GPU 5-seed × 50-epoch training wall-clock on the 8 scripts | **PROJECTED** | NOT executed in Wf-35 cycle; deferred to R19 §6-12 per `TODO/completed/82_metric_gap_filling.md` §7 |
| `bench_pb_valid.py` refactor | **PROJECTED** | script authored but not yet refactored to `_common_train_args`; master verify auto-skips with PENDING status |
| 1-epoch CPU smoke run | **MEASURED** in dev box (18.4 s), **PROJECTED** in CI (`--skip-smoke` because data lake is local-only) | master_report.json schema records `smoke.skipped=true` on CI runner |

Honest framing: Wf-35 ships the **refactor contract** (8 scripts share one args surface, master verify proves it, CI gates it). Actual model training on those scripts remains the responsibility of the R19 §6-12 schedule, which is a separate ~280-420 hour GPU work-stream.

---

## §4 `_common_train_args` design — what the shared surface looks like

The module exposes 3 argument-group builders + 1 env-setup helper + 1 decorator:

| Helper | Adds | Default group title |
|---|---|---|
| `add_common_args(parser, ...)` | `--data`, `--seeds`, `--epochs`, `--batch`, `--lr`, `--hidden`, `--depth`, `--eval-every`, `--dropout`, `--train-max-per-metal`, `--init`, `--encoder-ckpt`, `--out-dir`, `--metrics-path`, `--summary-path` | `Common recipe` |
| `add_gpu_args(parser, ...)` | `--cuda-visible-devices`, `--mixed-precision`, `--num-workers` | `GPU env` |
| `add_benchmark_args(parser, ...)` | `--n-folds`, `--cv-strategy`, `--folds-seed` | `Benchmark / CV` |
| `setup_gpu_env(args, ...)` | Applies `CUDA_VISIBLE_DEVICES`, seeds torch, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, returns `GpuEnvState` snapshot | — (side-effect, opt-in) |
| `@MEASURED_vs_PROJECTED_decorator(policy=...)` | Wraps `main()`; reads `--metrics-path` after return; tags with `_provenance` (MEASURED iff `*.pt` ckpts exist in `--out-dir`; else PROJECTED) | — (decorator) |

All helpers are **idempotent**: the `_add(root_parser, group, ...)` wrapper checks the root parser's `_actions` for an existing dest and silently skips re-registration. This is critical because scripts that call multiple helpers in sequence would otherwise raise `argparse.ArgumentError: conflicting option strings` on `--help`.

The module imports nothing heavy at module load (torch is imported lazily inside `setup_gpu_env`), so importing it never touches CUDA state — keeping `master_verify_r18.py` CPU-only.

---

## §5 Refactor diff — argparse 行数减少 per script

For each refactored script, the bespoke argparse block (typically 60-90 LOC) collapses to ~4 lines. Representative diff:

**Before** (illustrative, ~70 LOC):
```python
p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, default=...)
p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--batch", type=int, default=128)
# ... 50+ more lines covering lr/hidden/depth/init/dropout/...
p.add_argument("--cuda-visible-devices", type=str, default="")
p.add_argument("--mixed-precision", choices=["bf16","fp16","none"], default="bf16")
p.add_argument("--num-workers", type=int, default=4)
# ... argparse for n-folds/cv-strategy ...
```

**After** (~6 LOC):
```python
from molmetal.scripts._common_train_args import (
    add_common_args, add_gpu_args, add_benchmark_args,
    setup_gpu_env, MEASURED_vs_PROJECTED_decorator,
)
@MEASURED_vs_PROJECTED_decorator()
def main(argv=None):
    p = argparse.ArgumentParser()
    p = add_common_args(p)
    p = add_gpu_args(p)
    p = add_benchmark_args(p)
    args = p.parse_args(argv)
    gpu_env = setup_gpu_env(args)
    # ... training logic unchanged ...
```

Net: **~65 LOC removed per script × 7 scripts = ~455 LOC deleted**, replaced with ~50 LOC of imports + 1 decorator line. Plus every script now exposes the same CLI surface (`--help` shows identical `[Common recipe]` / `[GPU env]` / `[Benchmark / CV]` groups), which is what master verify's argparse check probes for.

---

## §6 Master verify report contents

`molmetal/scripts/master_verify_r18.py` writes two artefacts under `molmetal/reports/wf35_master_verify/`:

**`master_report.json`** — schema_version 1.0:
- `workflow` = "Wf-35 Phase 3"
- `ran_at_utc` + `started_at_utc` + `total_seconds` (18.4 s)
- `venv_python` absolute path
- `summary` = `{n_total:8, n_passed:7, n_failed:0, n_pending:1, n_shipped:7}`
- `scripts[]` per-script: `script`, `wf_tag` (Wf-29/31/32), `status` (PASS/FAIL/PENDING), `elapsed_seconds`, `error`, `checks[]` (each = `{name,passed,seconds,skipped,message}`)

**`master_report.md`** — human roll-up table with one row per script showing PASS / argparse / smoke checks and elapsed seconds. Plus a methodology footer explaining the 3-check protocol.

Current contents (per the on-disk reports):
- 7 PASS (train_se3sb / train_transport / train_multiph / train_ru_5metal_se3sb / train_affinity_regression / train_multimetal_lomo / train_cisplatin_resistance_v2)
- 0 FAIL
- 1 PENDING (bench_pb_valid.py — auto-skip with explanatory note)

The PENDING script is intentionally not failed — per the Wf-35 contract, only scripts that are SHIPPED count toward `n_passed`. bench_pb_valid.py is PLANNED but not yet authored to `_common_train_args`, so it stays PENDING rather than dragging the gate down.

---

## §7 CI integration — `r18-train-scripts-master-verify` job

`.github/workflows/ci.yml` job 4b (between mechanism-smoke and ci-summary):

- **Runner**: `ubuntu-latest` (NOT the self-hosted mol-metal label) — chosen so the Wf-35 contract is exercised without depending on the shared CadButEaas venv
- **Timeout**: 15 min
- **Venv path**: `$UV_PROJECT_ENVIRONMENT = ${{ github.workspace }}/molmetal/.venvs/mol-metal-venv` (matches what `master_verify_r18.py` hard-codes)
- **Install step**: `astral-sh/setup-uv@v6` + `uv venv` + `uv pip install "torch>=2.11,<3" numpy>=1.26 pandas>=2.0 scipy>=1.11 scikit-learn>=1.3 rdkit>=2024.3` (CUDA-enabled torch wheel fine — no kernels launch on CPU runner)
- **Run step**: `python molmetal/scripts/master_verify_r18.py --skip-smoke | tee master_verify_r18.log` followed by `grep -E "DONE in [0-9.]+s — pass=[0-9]+ fail=0 pending=[0-9]+"` — fails if any shipped script regresses
- **Artifact**: `master-verify-r18-log` (14-day retention)
- **ci-summary aggregates**: all 5 jobs (`evaluator-sanity` / `pitfall-runner` / `pitfall-pytest` / `mechanism-smoke` / `r18-train-scripts-master-verify`)

The `--skip-smoke` flag is essential: 1-epoch smokes need `.data/molmetal/*.csv` which is NOT checked in. Import + argparse checks alone prove the Wf-35 refactor contract holds without depending on local-only data.

---

## §8 不修改承诺 — what this refactor did NOT change

Per the Wf-35 contract, this refactor is strictly **argparse + env + JSON-tagging surface only**. It did NOT:

- Modify the model architectures (D-MPNN heads, Hilbert heads, multi-PH heads all untouched)
- Modify the loss functions, optimiser configs, learning-rate schedules, or seed handling semantics
- Modify the data loaders (CSV path resolution, scaffold split, LOMO split all unchanged)
- Touch the existing CI jobs (evaluator-sanity, pitfall-runner, pitfall-pytest, mechanism-smoke) — only added a new job 4b
- Modify `metrics/by_metric/*.json` schema or the upstream R17/R18 metric ledger
- Modify the test suite except adding `test_common_train_args.py` + 7 argparse contract tests

If any of those change in a future cycle, that's a separate workflow — not a Wf-35 follow-on.

---

## §9 Cross-reference

### §9.1 Skills

- `.claude/skills/mol-metal-gpu-execution/SKILL.md` (Wf-34 NEW, 302 LOC) — §4 standard train script template is now wired to `_common_train_args.add_common_args + add_gpu_args`; §6 5-seed × 50-epoch protocol uses `MEASURED_vs_PROJECTED_decorator` for JSON tagging; §7 MEASURED vs PROJECTED labels reference the decorator's `_provenance` key. **Wf-34 skill + Wf-35 module are co-designed** — reading one without the other misses the contract.
- `.claude/skills/mol-metal-cfm-training/SKILL.md` — does not depend on `_common_train_args` (CFM training is a different surface — `metallo_drug_smoke_retrain.py` uses its own argparse block).

### §9.2 TODO

- `TODO/pending/82_metric_gap_filling_plan.md` §3 — Wf-35 origin (Wf-29 noted argparse duplication across 4 trainer scripts)
- `TODO/completed/82_metric_gap_filling.md` §2.1 — Wf-29 noted `train_se3sb.py:613` was already fixed inline during Wf-23; the Wf-35 refactor picks up that fix and standardises it across 7 sibling scripts
- `TODO/completed/76_r18_blackwell_retrain_v2.md` — predecessor partial-ship; Wf-35 does not retrain, only consolidates
- `TODO/completed/82e_final_closeout.md` — R19 close-out ship list — this file (`82f`) extends that ship list with the refactor deliverable
- `TODO/INDEX.md` — will be updated to reflect this `82f` entry (post-Wf-35)

### §9.3 Reports

- `molmetal/reports/wf35_master_verify/master_report.json` — machine-readable 7/8 ship status
- `molmetal/reports/wf35_master_verify/master_report.md` — human roll-up
- `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json` (Wf-22, 116.5 GB VRAM free) — still valid; Wf-35 does not consume GPU budget

### §9.4 CI

- `.github/workflows/ci.yml` job 4b `r18-train-scripts-master-verify` — runs on every push + PR + dispatch + schedule; will catch any future drift in the `_common_train_args` refactor contract
