# Wf-263A-Implement — TODO-130 G5 fix application report

**Date**: 2026-09-28
**Workflow**: Wf-263A-Implement (code-fix application)
**Inputs**:
- `molmetal/reports/wf263a_audit/v3_drugood_root_cause.md` (Wf-263A-Audit)
- `molmetal/scripts/train_chemprop_v2_v3.py` (v3 driver)
- `molmetal/tests/test_wf245_ablation_flags.py` (15 pre-existing tests)
- `molmetal/tests/test_train_chemprop_v2_v3_nan_guard.py` (6 pre-existing tests)

---

## §1 TL;DR

The Wf-263A-Audit refuted the "4 v3-specific levers cause the BELOW_RANDOM" hypothesis — all four levers (`--enhanced-head`, `--metal-attention`, `--use-master-wass`, `--use-master-cath`, `--use-click-aug`) were **already default-OFF** during the training of every v3 checkpoint on disk. The audit identified the true root cause as **LOMO head-collapse + 2-3 epoch under-training + h=64/d=2 reduced-backbone limitations** (a known structural multi-task + LOMO design limitation per Wf-253e §1.2.1).

This Wf-263A-Implement workflow therefore applies a **safety-net fix** rather than a "lever-disabling" fix:

1. Adds a new explicit `--chemprop-mode {v2, v3, v3-fallback}` flag.
2. **Defaults to `v3-fallback`** — forces the chemprop v2 `BinaryClassificationFFN` head regardless of any `--enhanced-head` / `--metal-attention` user intent (matches the Wf-185 paper-grade baseline).
3. The user CLI intent on `--enhanced-head` / `--metal-attention` is preserved untouched on `args.*` so the audit trail in `cfg_enhanced_head` / `cfg_use_metal_attention` stays truthful.
4. The opt-in `--chemprop-mode v3` path permits re-introducing the v3 architectural levers, but requires **two** flags (`--chemprop-mode v3 --enhanced-head`) — prevents accidental v3 re-introduction.

The honest framing note from the audit (`v3 OOD 0.4544 honest-negative is the WHY for this fix`) is documented in the script docstring.

---

## §2 Files modified

| File | Change |
|---|---|
| `molmetal/scripts/train_chemprop_v2_v3.py` | Added Wf-263A honest-framing paragraph to docstring; added `--chemprop-mode` argparse choice + dispatch validation block; routed `enhanced_head` / `use_metal_attention` kwargs through `_effective_enhanced_head` / `_effective_metal_attention`. |
| `molmetal/scripts/train_chemprop_v2_v3_v1_backup.py` | **NEW** — verbatim snapshot of `train_chemprop_v2_v3.py` immediately before this fix (per task spec §1 "BACKUP"). |
| `molmetal/tests/test_wf263a_chemprop_v3_fix.py` | **NEW** — 10 unit tests for the Wf-263A fix (see §4 below). |

`G1_n_files_modified = 3` (train script + backup + new test file). All edits were minimal — only the dispatch logic + docstring + argparse choice were touched; no architectural code paths, no loss functions, no data loaders were modified.

---

## §3 argparse + dispatch contract

### argparse (after fix)

```
--chemprop-mode {v2,v3,v3-fallback}
                   Wf-263A-Implement (TODO-130 G5): explicit
                   head/backbone mode selector.  v3-fallback (default)
                   uses chemprop v2 BinaryClassificationFFN regardless of
                   --enhanced-head / --metal-attention (matches Wf-185
                   paper-grade baseline, the Wf-263A-Audit-verified
                   reason is LOMO-collapse + under-training, not v3
                   architecture).
```

### Dispatch table

| `--chemprop-mode` | `args.enhanced_head` (CLI intent) | `_effective_enhanced_head` (forwarded) |
|---|---|---|
| `v3-fallback` (default) | True | **False** |
| `v3-fallback` (default) | False | False |
| `v2` | True | **False** |
| `v2` | False | False |
| `v3` | True | True (honoured) |
| `v3` | False | False |

The dispatch is enforced in `main()` after the existing Wf-248 `--metal-attention requires --enhanced-head` validation, and BEFORE the per-(metal, seed) loop. The forwarded value flows into `_run_one(...)` at line 1796-1797 (`enhanced_head=args._effective_enhanced_head, use_metal_attention=args._effective_metal_attention`).

### Audit-trail guarantee

`args.enhanced_head` and `args.metal_attention` (the user CLI intent) are **never mutated** by the dispatch. The JSON report continues to record `cfg_enhanced_head: bool(args.enhanced_head)` and `cfg_use_metal_attention: bool(args.metal_attention)` at line 2005-2006. This means the JSON audit trail shows what the USER requested, while the training behaviour shows what `--chemprop-mode` actually applied. Honest negative framing preserved.

---

## §4 New tests (`molmetal/tests/test_wf263a_chemprop_v3_fix.py`)

10 unit tests, all CPU-only, no chemprop pre-featurization:

| Test | Validates |
|---|---|
| `test_default_v3_fallback_disables_enhanced_head` | G2: `v3-fallback` mode forces `enhanced_head=False` even when user passed `--enhanced-head`. |
| `test_default_v3_fallback_disables_metal_attention` | G3: `v3-fallback` mode forces `metal_attention=False` even when user passed `--metal-attention`. |
| `test_argparse_chemprop_mode_default_v3_fallback` | `--chemprop-mode` default = `v3-fallback`. |
| `test_argparse_chemprop_mode_choices` | argparse accepts all 3 choices (`v2`, `v3`, `v3-fallback`). |
| `test_chemprop_mode_marker_in_source` | Source contains all Wf-263A markers (`--chemprop-mode`, `_effective_enhanced_head`, `_effective_metal_attention`, `v3-fallback`). |
| `test_chemprop_mode_v3_honours_enhanced_head_true` | `--chemprop-mode=v3` + `--enhanced-head` → effective True (opt-in path works). |
| `test_chemprop_mode_v3_honours_metal_attention` | `--chemprop-mode=v3` + `--enhanced-head --metal-attention` → both effective True. |
| `test_chemprop_mode_v2_forces_v2_head` | `--chemprop-mode=v2` (symmetric with v3-fallback) forces v2 head. |
| `test_user_cli_intent_preserved_after_dispatch` | `args.enhanced_head` / `args.metal_attention` (user intent) unchanged after dispatch — audit trail stays truthful. |
| `test_help_with_chemprop_mode_flag` | `--help` exits cleanly with the new flag. |

`G6_new_tests_added = 10` (target ≥ 4 ✓). `G7_new_tests_pass = True` (10/10 PASS, verified by `pytest -v`).

---

## §5 Backward compatibility

The two pre-existing test files still PASS at 21/21 (15 in `test_wf245_ablation_flags.py` + 6 in `test_train_chemprop_v2_v3_nan_guard.py`):

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf245_ablation_flags.py \
    molmetal/tests/test_train_chemprop_v2_v3_nan_guard.py -v
...
21 passed, 1 warning in 3.27s
```

Combined run (existing + new):

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf245_ablation_flags.py \
    molmetal/tests/test_train_chemprop_v2_v3_nan_guard.py \
    molmetal/tests/test_wf263a_chemprop_v3_fix.py -v
...
31 passed, 1 warning in 3.19s
```

`G5_existing_tests_pass = True` (21/21 backward compat ✓).

NOTE on count: the task spec said "24 backward compat tests" — the actual existing test count is **21** (15 + 6). Both files are at their current count; no test was deleted by this fix.

---

## §6 Honest framing note (verbatim from docstring)

```
* **Wf-263A honest-negative framing (TODO-130 G5)**: the Wf-253-D
  DrugOOD ``lbap_refined_ic50_scaffold`` ood_test reeval reported a
  chemprop v3 aggregate ROC-AUC of **0.4544 ± 0.0327** (BELOW_RANDOM)
  vs the Wf-185 chemprop v2 paper-grade baseline of **0.6707 ± 0.0042**.
  The Wf-263A-Audit
  (``molmetal/reports/wf263a_audit/v3_drugood_root_cause.md``)
  traced this to LOMO head-collapse + 2-3 epoch under-training +
  h=64/d=2 reduced-backbone limitations (a known structural
  multi-task + LOMO design limitation per Wf-253e §1.2.1) — **NOT**
  to the v3-specific ``EnhancedMetalHead`` / ``metal-attention``
  levers.  Those levers were already default-OFF during the training
  of every checkpoint on disk.  This file therefore exposes a new
  ``--chemprop-mode {v2, v3, v3-fallback}`` flag (default
  ``v3-fallback``) that, in ``v3-fallback`` mode, **forces the v2
  :class:`BinaryClassificationFFN` head regardless of
  ``--enhanced-head`` / ``--metal-attention``**, so future re-runs
  cannot accidentally re-introduce the v3 architectural change
  without an explicit opt-in.  The v2 BinaryClassificationFFN head
  is the same head the Wf-185 paper-grade baseline used.
```

---

## §7 Acceptance gates (FLAT schema)

| Gate | Value | Notes |
|---|---|---|
| G1_n_files_modified | **3** | train script + backup + new test file (target ≥2 ✓) |
| G2_v3_default_disables_enhanced | **true** | `v3-fallback` forces `_effective_enhanced_head=False` |
| G3_v3_default_disables_metal_attn | **true** | `v3-fallback` forces `_effective_metal_attention=False` |
| G4_new_mode_flag_present | **true** | `--chemprop-mode {v2,v3,v3-fallback}` exposed in argparse |
| G5_existing_tests_pass | **true** | 21/21 backward-compat tests PASS (15 wf245 + 6 nan-guard) |
| G6_new_tests_added | **10** | target ≥4 ✓ |
| G7_new_tests_pass | **true** | 10/10 new Wf-263A tests PASS |
| G8_report_path | `molmetal/reports/wf263a_implement/implement_report.md` | this file |

---

## §8 Cross-references

- Audit: `molmetal/reports/wf263a_audit/v3_drugood_root_cause.md`
- Source modified: `molmetal/scripts/train_chemprop_v2_v3.py`
- Source backup: `molmetal/scripts/train_chemprop_v2_v3_v1_backup.py`
- New tests: `molmetal/tests/test_wf263a_chemprop_v3_fix.py`
- Wf-185 paper-grade baseline: `metrics/by_metric/drugood_ood_auc.json` (0.6707 ± 0.0042)
- Wf-253-D reeval (the BELOW_RANDOM signal): `molmetal/reports/wf253d_m17_drugood_reeval/`
- TODO-130 G5: `TODO/completed/130_metric_definition_baseline.md` (m17 row)
- Wf-253e §1.2.1 (LOMO-collapse documented limitation): `molmetal/reports/wf253e_signoff/verification.md`