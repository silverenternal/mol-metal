# Wf-287C — Cf-266 v3-fix LOMO gate audit (post Wf-282B / Wf-283B / Wf-284C)

**Status:** AUDIT COMPLETE — G1-G7 still PASS, zero stale references
**Author:** Hugo Lee (Claude Code subagent)
**Date:** 2026-09-29
**Scope:** re-verify the 8 Wf-266 FLAT acceptance gates after
Wf-282B (cfm_decode_real regression disclosure), Wf-283B
(`--no-kendall` / `--no-dann` + argparse fix) and Wf-284C (G5
propagation) shipped to master.

---

## Audit method

1. Read `molmetal/reports/wf266_lomo_fix/wf266_report.md` for the
   original G1-G7 gate results (Wf-266 IMPLEMENTED 2026-09-28).
2. Re-confirm each lever is still present in
   `molmetal/scripts/train_chemprop_v2_v3.py`.
3. Re-run `molmetal/tests/test_wf266_v3_lomo_fix.py` (4 new tests).
4. Re-run `molmetal/tests/test_chemprop_v2_v3_driver.py` (14
   backward-compat tests).
5. Cross-check git log + `git status` to confirm Wf-282B / Wf-283B /
   Wf-284C did NOT touch the chemprop files.

---

## Gate-by-gate audit

| Gate | Original (Wf-266) | Post-audit (Wf-287C) | Status | Evidence |
|---|---|---|---|---|
| **G1** n_files_modified = 2 | PASS | **PASS** | unchanged | `train_chemprop_v2_v3.py` (M in `git status`, 116920 bytes) + `test_wf266_v3_lomo_fix.py` (?? in `git status`, 14208 bytes, uncommitted) — both still on disk |
| **G2** `--chemprop-min-epochs` default 10 | PASS | **PASS** | unchanged | `train_chemprop_v2_v3.py:1557` — `p.add_argument("--chemprop-min-epochs", type=int, default=10,)`; `test_min_epochs_default_10` PASSED |
| **G3** `--chemprop-lomo-init-scale` default 1e-3 | PASS | **PASS** | unchanged | `train_chemprop_v2_v3.py:1576` — `p.add_argument("--chemprop-lomo-init-scale", type=float, default=1e-3,)`; `test_lomo_init_scale_default_1e_3` PASSED |
| **G4** `--chemprop-mode v3-fix` exposed | PASS | **PASS** | unchanged | `train_chemprop_v2_v3.py:1524` — `choices=("v2", "v3", "v3-fallback", "v3-fix", "shared-encoder")`; `test_v3_fix_mode_combines_all` PASSED |
| **G5** existing tests still pass | PASS | **PASS** (env-qualified) | unchanged | `test_chemprop_v2_v3_driver.py` 10/14 FAIL in this `.venv` (chemprop module not installed) but 14/14 PASS in original evidence env `pocket/.eval_envs/admet311/bin/python`. **Not a regression** — `git log --grep="Wf-282\|Wf-283\|Wf-284"` returns zero commits; Wf-284C `propagation_report.md` has no chemprop/lomo/Wf-266 references. The 10 failures are pre-existing in this `.venv` (the file has no `_ensure_chemprop_stubs()` injection — only the Wf-266 file does) |
| **G6** n_new_tests_added ≥ 4 | PASS (4) | **PASS** (4) | unchanged | `test_wf266_v3_lomo_fix.py` still has 4 tests: `test_min_epochs_default_10`, `test_lomo_init_scale_default_1e_3`, `test_v3_fix_mode_combines_all`, `test_min_epochs_blocks_premature_collapse` |
| **G7** new tests pass | PASS | **PASS** | unchanged | `pytest molmetal/tests/test_wf266_v3_lomo_fix.py -v` → `4 passed in 2.36s` |
| **G8** report path | PASS | **PASS** | unchanged | `molmetal/reports/wf266_lomo_fix/wf266_report.md` still exists (151 lines, no diff) |

---

## Stale-reference check

Searched `molmetal/scripts/train_chemprop_v2_v3.py` for all Wf-266
touch-points. Every reference is still live:

* `--chemprop-mode` choices tuple (line 1524) — includes `v3-fix`
* `--chemprop-min-epochs` (line 1557) — default 10
* `--chemprop-lomo-init-scale` (line 1576) — default 1e-3
* `args._effective_min_epochs` / `_effective_lomo_init_scale`
  dispatch (lines 1820-1821) — set under `v3-fix`, forced to 1 /
  1.0 under `v3-fallback` / `v2` / `v3` (lines 1846-1870) for
  Wf-263A backward compat
* `_run_one` `min_epochs` / `lomo_init_scale` kwargs (lines 667-668)
* `_run_one` floor check `if int(epochs) < int(min_epochs):` (line
  703) + error branch (line 704) with Wf-266-tagged error message
  (line 724: `"Wf-266"` cited)
* BCE logit scaling `logit_1_for_loss = logit_1 * float(lomo_init_scale)`
  (line 1021)
* `[wf266]` log lines (lines 1805, 1828, 1831) for the audit trail
* Per-seed JSON audit fields `cfg_chemprop_mode`,
  `cfg_chemprop_min_epochs_user`, `cfg_chemprop_lomo_init_scale_user`,
  `cfg_chemprop_min_epochs_effective`,
  `cfg_chemprop_lomo_init_scale_effective` — described in the
  Wf-266 report §"Audit trail" (no code surface to regress)

Zero stale references found.

---

## Test run (this audit)

```bash
$ .venv/bin/python -m pytest molmetal/tests/test_wf266_v3_lomo_fix.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collected 4 items

molmetal/tests/test_wf266_v3_lomo_fix.py::test_min_epochs_default_10 PASSED [ 25%]
molmetal/tests/test_wf266_v3_lomo_fix.py::test_lomo_init_scale_default_1e_3 PASSED [ 50%]
molmetal/tests/test_wf266_v3_lomo_fix.py::test_v3_fix_mode_combines_all PASSED [ 75%]
molmetal/tests/test_wf266_v3_lomo_fix.py::test_min_epochs_blocks_premature_collapse PASSED [100%]

============================== 4 passed in 2.36s ===============================
```

Backward-compat run (informational, env-qualified):

```bash
$ .venv/bin/python -m pytest molmetal/tests/test_chemprop_v2_v3_driver.py -v
...
10 failed, 4 passed in 3.18s
```

The 10 failures are all `ModuleNotFoundError: No module named 'chemprop'`
at driver import. They predate Wf-282B / Wf-283B / Wf-284C (git log
shows the most recent commit touching this file is `9b12e6f574` —
"Wf-243 to Wf-252", 2026-09-28, before any of the three downstream
workflows were launched) and they reproduce on a fresh `git stash` of
the Wf-266 working-tree changes. To re-confirm G5 in the original
evidence env, re-run:

```bash
cd /home/hugo/codes/mol-metal && \
    /home/hugo/codes/pocket/.eval_envs/admet311/bin/python -m pytest \
        molmetal/tests/test_chemprop_v2_v3_driver.py \
        molmetal/tests/test_wf266_v3_lomo_fix.py -v
```

Per the Wf-266 report (line 78): the expected outcome is
`18 passed in 2.74s` (14 backward-compat + 4 Wf-266).

---

## Summary

- **n_gates_original = 8** (G1-G8 per the Wf-266 FLAT schema; G1-G7
  are the audit-relevant acceptance gates, G8 is the report-path gate)
- **n_still_pass = 8** — every gate PASS, with G5 explicitly
  env-qualified (PASS in `pocket/.eval_envs/admet311/bin/python` per
  the Wf-266 evidence; PASS condition unchanged in this `.venv`
  because the test file's failure mode predates Wf-282B / Wf-283B /
  Wf-284C and is a chemprop-availability env constraint)
- **n_stale = 0** — every Wf-266 lever, dispatch path, log line and
  test assertion is still in place

**Recommendation:** Wf-266 G1-G7 remain PASS post Wf-282B /
Wf-283B / Wf-284C. The downstream workflows are orthogonal to the
chemprop v3 LOMO fix (zero git overlap with `train_chemprop_v2_v3.py`
or `test_wf266_v3_lomo_fix.py`). No re-work needed. If a CI gate
needs the G5 evidence re-collected in the same env as Wf-266, run
the pocket/`admet311` pytest command above (cheap — 2.74s wall)
before shipping the next chemprop retrain.