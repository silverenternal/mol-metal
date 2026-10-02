# Wf-282C Lint Pass Report

**Date:** 2026-09-29
**Scope:** 7 train/bench scripts + 1 metrics module (vs Wf-243..252 ship surface)
**Method:** `ast.parse` syntax check + `--help` CLI verification + AST walk for `%` in argparse help strings + import/TODO grep

## Summary Counts

| Metric | Count |
|---|---|
| Scripts checked | 8 (1 module + 7 scripts) |
| Syntax OK | 8 / 8 |
| `--help` works (scripts with argparse) | 5 / 6 |
| Issues found | 1 critical (1 minor: 4 hardcoded absolute paths in docstrings/example shells, not runtime) |

## Per-file Lint Results

| # | File | Syntax | `--help` | Issues |
|---|---|---|---|---|
| 1 | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` | OK | N/A (lib, no argparse) | none |
| 2 | `molmetal/scripts/bench_catherine_30metals.py` | OK | OK | none |
| 3 | `molmetal/scripts/metallo_drug_smoke_retrain.py` | OK | OK | none |
| 4 | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | OK | OK | none |
| 5 | `molmetal/scripts/train_chemprop_v2_v3.py` | OK | OK | none |
| 6 | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | OK | **CRASH** | 1 critical (see Issue #1) |
| 7 | `molmetal/scripts/wf229_ablation_launcher.py` | OK | OK | none |

## Issues

### Issue #1 (CRITICAL): Literal `% util` in `--cpu-num-workers` help string crashes argparse

- **File:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
- **Line:** 1190 (in help text for `--cpu-num-workers`, lines 1181-1192)
- **Symptom:** `--help` raises `TypeError: %u format: a real number is required, not dict`
- **Root cause:** Argparse performs `help % params` for substitution (`%(default)s`, `%(type)s`, …). A literal `%` in the help text is interpreted as a format spec — `% u` is `%u` which requires a real number, but `params` is a `dict`.
- **Trigger string:**
  ```
  "... leaving GPU at 15-30% util on RTX PRO 6000)."
  ```
- **Fix (one of):**
  1. Escape `%` → `%%` (writes as `15-30%%` in rendered help; argparse unescapes to `15-30%`).
  2. Rephrase to avoid `%`: `"... at 15-30 percent util on RTX PRO 6000)."`
- **Severity:** CRITICAL — `--help` is broken; downstream tooling that introspects CLI via `--help` (e.g. Wf-282C lint, Wf-256 apply scripts, log scraping) cannot see the flag.
- **Suggested owner:** Wf-260/261/262 (the change that introduced `--cpu-num-workers` flag in line 1181-1192).

### Minor Observations (non-blocking)

| File | Line(s) | Note |
|---|---|---|
| `molmetal/scripts/wf229_ablation_launcher.py` | 44, 57 | Hardcoded `/home/hugo/codes/mol-metal` in docstring shell block + `PROJECT_ROOT` constant. Used as runtime fallback for `Path(__file__).parents[2]` resolution; OK on this host but brittle if repo moves. |
| `molmetal/scripts/train_chemprop_v2_v3.py` | 81, 86 | Hardcoded `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` in docstring example block. Docstring only, not runtime. |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | 66 | Hardcoded `cd /home/hugo/codes/mol-metal` in docstring example. Docstring only. |
| `molmetal/scripts/metallo_drug_smoke_retrain.py` | 79 | Hardcoded `cd /home/hugo/codes/try_triton_on_rocm` in docstring. Docstring only. |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 823 | `cache_dir=Path("/tmp/wf131_mtw_fe_cache")` — uses `/tmp` rather than `tempfile.gettempdir()`. Acceptable for single-host runs but fragile on multi-tenant `/tmp`. |

None of these affect runtime correctness on the canonical host (`/home/hugo/codes/mol-metal`, `/home/hugo/codes/CadButEaas`). They are docstring examples (cosmetic) or shared-host hygiene, not lint defects per Wf-282C scope.

## Cross-Script Verification: Wf-243..252 Ship Surface Flags

| Flag / surface | Script | Verified |
|---|---|---|
| `m11 n_distinct` (Wf-253D) | `metrics_v2.py:737-810` | yes (`def n_distinct_from_smiles`, `n_distinct / min_count_for_ratio`) |
| `--ref-repo-path` (Wf-257) | `metallo_drug_smoke_retrain.py:262` | yes |
| Sign-flip fix (Wf-256) | `bench_catherine_30metals.py:148` | yes (commented "sign-flipped from previous buggy") |
| `--use-tanimoto-ood-filter` + `--ood-filter-{threshold,train-pool}` (Wf-268A + TODO-98) | `train_cfm_yuelbond_wasserstein.py:864-877, 974` | yes |
| `_LinearProbeHead` + 4 chemprop modes + 3 flags (Wf-263A/266/269A) | `train_chemprop_v2_v3.py` | yes (chemprop-mode `{v2,v3,v3-fallback,v3-fix,shared-encoder}`, --enhanced-head, --metal-attention, --use-click-aug) |
| DataLoader `num_workers` wire (Wf-260/261/262) | `train_dmpnn_multitask_wasserstein.py:488-520` | yes (`num_workers=num_workers` passed to DataLoader; `--cpu-num-workers` flag added but `--help` broken — see Issue #1) |
| `--output-dir` CLI flag (Wf-256) | `wf229_ablation_launcher.py:267-282` | yes |

## Hygiene Sweep Results

- **Unused imports:** 0 across 8 files (AST walk + literal-set diff).
- **TODO / FIXME / XXX / HACK markers:** 0 in any of the 8 files.
- **`%` literals in argparse help strings:** 1 (Issue #1, train_dmpnn_multitask_wasserstein.py line 1190).
- **Default-value consistency (seed lists):** `train_chemprop_v2_v3.py` and `train_dmpnn_multitask_wasserstein.py` both use `default=list(DEFAULT_SEEDS)` from `molmetal.scripts.train_constants` — consistent.
- **Honest-negative guard presence:** 3/7 scripts explicitly mark `honest-negative` / `MEASURED-vs-PROJECTED` (`bench_catherine_30metals.py`, `train_chemprop_v2_v3.py`, `wf229_ablation_launcher.py`). The other 4 emit JSON metrics to `metrics/by_metric/*.json` which is the canonical honest-negative sink — not a missing guard, by design.
- **Stale hardcoded paths:** None runtime-critical; all 5 hardcoded paths are in docstring example shells (cosmetic, see Minor Observations table).

## Recommended Next Step

Patch Issue #1 (escape `%` in `train_dmpnn_multitask_wasserstein.py` help text at line 1190), then re-run Wf-282C to confirm `--help` exits 0. Estimated time: 30 s. No other lint blockers for Wf-243..252 ship surface.

## Files Used (absolute paths)

- `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/sbdd_env/metrics_v2.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/bench_catherine_30metals.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/metallo_drug_smoke_retrain.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_chemprop_v2_v3.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
- `/home/hugo/codes/mol-metal/molmetal/scripts/wf229_ablation_launcher.py`
- `/home/hugo/codes/mol-metal/molmetal/reports/wf282c_lint_pass/lint_report.md` (this report)