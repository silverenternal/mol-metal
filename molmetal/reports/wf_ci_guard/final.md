# WF-CI-Guard — Evaluator sanity + pitfall guards + mechanism smoke into CI

> Authoritative spec: `TODO/INDEX.md` (ultracode TODO entry), with the
> computed task text appended to the workflow invocation.  Status:
> **DONE — local verification 2026-09-25**.  Numbers below are MEASURED
> on the local dev box (RTX PRO 6000 Blackwell, Python 3.12.13,
> `/home/hugo/codes/CadButEaas/.venv-vlm` shared env, uv 0.11.x).

## 1. Goal

Wire the three existing CPU-only verification gates — evaluator sanity,
pitfall-guard matrix, and mechanism smoke — into a single CI workflow
that fires on every push and PR.  The gate must:

| Gate | Pass criterion |
|---|---|
| Evaluator sanity | `eval_all_evaluators.py` reports **14/14 PASS** |
| Pitfall runner | `wf_pitfall_guard/check_all.py` reports **7/7 PASS** |
| Pitfall pytest | `test_pitfall_guards.py` reports **≥11 passed** |
| Mechanism smoke | 8 named mechanism test files report **≥217 passed** |

All four gates MUST pass for the workflow to succeed.

## 2. Files added / modified

| Path | Change | Why |
|---|---|---|
| `.github/workflows/ci.yml` | **NEW** | The CI workflow itself. 4 jobs in series + 1 summary job. |
| `molmetal/tests/test_pitfall_guards.py` | EDITED | Replaced `__import__`-based `_try_import` with `importlib.import_module` so that dotted sub-module paths return the submodule (not the top-level package).  Without this fix, `test_p2_swap_smarts_idempotent_on_malformed_input` and `test_p4_amp_dtype_float32_supported` failed because `getattr(molmetal, "_swap_smarts")` is `None` even though `molmetal.molmetal_lam.sbdd_env.synflownet_env._swap_smarts` exists. |
| `molmetal/tests/test_lambda_reward_coupling.py` | EDITED | The test passed `--lambda-reward 0.5` to a `store_true` flag, so `0.5` was parsed as a positional and rejected.  Removed the bogus `0.5`; the script's `--lambda-reward` is a boolean. |

Files that were already in place and required no edits:

| Path | Role |
|---|---|
| `molmetal/scripts/eval_all_evaluators.py` | 14-check evaluator sanity harness (SA, QED, Vina, PoseBusters, BioPython, OpenBabel, ProDy, etc.). |
| `molmetal/scripts/wf_pitfall_guard/check_all.py` | Single-shot runner that invokes pytest on `test_pitfall_guards.py`, parses `-v` output, prints an ASCII PASS/FAIL table or `--json`. |
| `molmetal/tests/test_pitfall_guards.py` | 11 canonical sub-tests + 7-entry `test_pitfall_guard_matrix` parametrize. |

## 3. Workflow structure (`.github/workflows/ci.yml`)

The workflow defines **four parallel jobs** sharing the same env wiring:

```yaml
name: ci
on:
  push:        # + tags, + workflow_dispatch
    branches: [master, main, 'release/*', 'round*']
  pull_request:
    branches: [master, main, 'release/*', 'round*']
  workflow_dispatch:
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  evaluator-sanity:  # [1/4]
  pitfall-runner:    # [2/4]
  pitfall-pytest:    # [3/4]
  mechanism-smoke:   # [4/4]
  ci-summary:        # posts a single badge-style status
```

Each job runs on `[self-hosted, linux, x64, mol-metal]` (the dev box
that already has `UV_PROJECT_ENVIRONMENT=/home/hugo/codes/CadButEaas/.venv-vlm`)
and pins these env vars:

```yaml
UV_PROJECT_ENVIRONMENT: /home/hugo/codes/CadButEaas/.venv-vlm
VIRTUAL_ENV:            /home/hugo/codes/CadButEaas/.venv-vlm
MOLMETAL_DATA_DIR:      /home/hugo/codes/mol-metal/.data/molmetal
MOLMETAL_TMQM_DIR:      /home/hugo/codes/mol-metal/.data/molmetal/tmQM
TOKENIZERS_PARALLELISM: "false"
TRANSFORMERS_VERBOSITY: "error"
PYTHONUNBUFFERED:       "1"
```

For GitHub-hosted runners, replace the `runs-on` selector and the env
path with the project-local `.venv` (see `pyproject.toml`).

## 4. Per-job gates

### 4.1 evaluator-sanity (must show 14/14)

```
cd "$GITHUB_WORKSPACE"
$UV_PROJECT_ENVIRONMENT/bin/python molmetal/scripts/eval_all_evaluators.py \
    | tee eval_all_evaluators.log
tail -3 eval_all_evaluators.log | grep -q "14/14 evaluators PASS"
```

The tail-grep pattern enforces the strict "all 14 must PASS" contract —
any failure aborts the job.

### 4.2 pitfall-runner (must show 7/7)

```
$UV_PROJECT_ENVIRONMENT/bin/python molmetal/scripts/wf_pitfall_guard/check_all.py \
    | tee pitfall_check.log
grep -E "^\s*PASS=7\s+FAIL=0\s+SKIP=0\s*$" pitfall_check.log
```

The runner prints a footer of the form `PASS=7 FAIL=0 SKIP=0`; the
regex tolerates leading whitespace.

### 4.3 pitfall-pytest (≥11 passed)

```
$UV_PROJECT_ENVIRONMENT/bin/python -m pytest \
    molmetal/tests/test_pitfall_guards.py \
    -q --tb=short --no-header \
    | tee pitfall_pytest.log
grep -E "[0-9]+ passed" pitfall_pytest.log
```

`pytest -q` exits non-zero on any failure; the grep is purely
informational (the exit code is the gate).

### 4.4 mechanism-smoke (≥217 passed)

```
$UV_PROJECT_ENVIRONMENT/bin/python -m pytest \
    molmetal/tests/test_f2a_metal_ligand_exchange.py \
    molmetal/tests/test_synthemol_reactions.py \
    molmetal/tests/test_lambda_combinators_expansion.py \
    molmetal/tests/test_pll_signature.py \
    molmetal/tests/test_reducer.py \
    molmetal/tests/test_lambda_reward_coupling.py \
    molmetal/tests/test_biolm_score_adapter.py \
    molmetal/tests/test_synflownet_env.py \
    -q --tb=short --no-header \
    | tee mechanism_smoke.log
PASSED=$(grep -oE '[0-9]+ passed' mechanism_smoke.log | head -1 | awk '{print $1}')
[ "${PASSED:-0}" -lt 217 ] && exit 1
```

The hard pytest exit-on-any-failure still applies; the
`PASSED < 217` branch is a defence-in-depth guard in case the
threshold is ever raised.

### 4.5 ci-summary

Posts a single bullet-list into `$GITHUB_STEP_SUMMARY` with each job's
result, so the GitHub UI shows a one-glance green/red badge even when
multiple jobs run in parallel.

## 5. Measured results — local dev box (2026-09-25)

All four gates measured GREEN end-to-end.  Reproducible from the repo
root with `source env.sh && uv run --no-sync python ...`.

#### Gate 1 — Evaluator sanity

```
SUMMARY: 14/14 evaluators PASS
All evaluators ready for paper §4 evaluation suite
```

Detail (full table):

```
[1] SA score (Ertl-Schuffenhauer) ............ 3/3
[2] QED (Bickerton) ........................... 2/2
[3] Vina adapter (AutoDock) ................... 1/1
[4] Vina Python binding + meeko ............... 2/2
[5] PoseBusters (geometric sanity) ............ 1/1
[6] BioPython PDB (PLIP replacement) .......... 1/1
[7] OpenBabel .................................. 1/1
[8] ProDy ..................................... 1/1
[9] Reference drug sanity (cisplatin) ......... 2/2
                                       Total = 14/14
```

#### Gate 2 — Pitfall runner (`wf_pitfall_guard/check_all.py`)

```
Pitfall  Description                                                   Status
-------------------------------------------------------------------------------
P1       OOV atoms → vocab_mask + 20-element default vocab             PASS
P2       SynFlowNet shim TypeError → per-rule try/except               PASS
P3       MetalLigandExchange 84-rule matrix + cisplatin seed default   PASS
P4       BF16 atom_logits clamp ±50 + FP32 AMP fallback                PASS
P5       r4_lambda_only_run SAFETY_MAX ≥ 10000 floor                   PASS
P6       Reference clone location auto-detect (parent vs clone root)   PASS
P7       ModelWrapper keyword-only t fallback (functools.partial fix)  PASS
-------------------------------------------------------------------------------
PASS=7 FAIL=0 SKIP=0
pytest rc=0  wall=2.0s
```

#### Gate 3 — Pitfall pytest (`test_pitfall_guards.py`)

```
18 passed, 1 warning in 1.42s
```

Composition:
- 11 canonical sub-tests (P1×2, P2×2, P3×2, P4×2, P5, P6, P7)
- 7 `test_pitfall_guard_matrix[P#-…]` entries

The 11-vs-7 split is a design choice: each canonical sub-test
directly exercises the underlying guard, while the matrix entry is the
single-entry dispatch surface that the wf_pitfall_guard runner
parses.  Both must be green for CI to pass.

#### Gate 4 — Mechanism smoke (8 named test files)

```
434 passed, 22 skipped, 1 warning in 42.39s
```

| File | Tests | Role |
|---|---|---|
| `test_f2a_metal_ligand_exchange.py` | many | 84-rule Pt/Pd/Au/Ir/Ru exchange matrix sanity + CLI registration |
| `test_synthemol_reactions.py` | many | β-NF + click-rule reaction coverage |
| `test_lambda_combinators_expansion.py` | many | Lambda calculus combinator expansion + termination |
| `test_pll_signature.py` | many | PLL coupling signature contract |
| `test_reducer.py` | many | β-reduction correctness under metallocomplexes |
| `test_lambda_reward_coupling.py` | 8 | `--lambda-reward` CLI + dataset + sampler integration |
| `test_biolm_score_adapter.py` | many | BioLM adapter scoring contract |
| `test_synflownet_env.py` | many | SynFlowNet env action space + reward wiring |

The ≥217 threshold is comfortably met (434 actual).  Skipped tests are
data-fixture guards (training CSV / SDF files absent on the runner).

## 6. Pre-existing issues caught and fixed during CI bring-up

| Issue | Where | Fix |
|---|---|---|
| `_try_import` returned top-level package for sub-module paths, causing 2 pitfall sub-tests to falsely fail. | `molmetal/tests/test_pitfall_guards.py` | Replaced `__import__(modname)` with `importlib.import_module(modname)`. |
| `test_lambda_reward_cli_flag_accepted` passed a value to a `store_true` flag, causing argparse to reject the run. | `molmetal/tests/test_lambda_reward_coupling.py` | Removed the bogus `"0.5"` value. |

Both issues were latent (the tests passed silently in earlier rounds
because the CI gate did not yet exist), so surfacing them here is the
intended outcome of this work.

## 7. How to run the gate locally

```bash
cd /home/hugo/codes/mol-metal
source env.sh

# Gate 1 — evaluators
uv run --no-sync python molmetal/scripts/eval_all_evaluators.py

# Gate 2 — pitfall runner
uv run --no-sync python molmetal/scripts/wf_pitfall_guard/check_all.py

# Gate 3 — pitfall pytest
uv run --no-sync python -m pytest molmetal/tests/test_pitfall_guards.py \
    -q --tb=short --no-header

# Gate 4 — mechanism smoke
uv run --no-sync python -m pytest \
    molmetal/tests/test_f2a_metal_ligand_exchange.py \
    molmetal/tests/test_synthemol_reactions.py \
    molmetal/tests/test_lambda_combinators_expansion.py \
    molmetal/tests/test_pll_signature.py \
    molmetal/tests/test_reducer.py \
    molmetal/tests/test_lambda_reward_coupling.py \
    molmetal/tests/test_biolm_score_adapter.py \
    molmetal/tests/test_synflownet_env.py \
    -q --tb=line --no-header
```

(Replace `uv run --no-sync` with
`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` if uv is unavailable
in the active shell.)

## 8. Honest limitations

- **Self-hosted runner.** The `runs-on` selector
  `[self-hosted, linux, x64, mol-metal]` matches the dev box only.
  GitHub-hosted `ubuntu-latest` runners would need the
  `UV_PROJECT_ENVIRONMENT` path replaced with a project-local
  `.venv` (the existing `pyproject.toml` already declares the deps).
- **GPU not required.** All four gates are CPU-only.  GPU-dependent
  tests (CFM retrain, EGNN, Vina-GPU) are NOT exercised by this CI
  pass — they live behind the `gpu` pytest marker.
- **Hard pytest exit code.** Gates 3 and 4 use pytest's
  pass/fail exit code as the contract.  Adding new tests with a known
  failure mode requires either fixing them or marking them
  `@pytest.mark.skip` — there is no per-test tolerance.
- **No Docker image pinning yet.** The CI does not yet ship a
  containerised image of the env; that is a follow-up tracked under
  the round-18 infrastructure work (separate from this gate).