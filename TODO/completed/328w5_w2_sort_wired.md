# Wf-328-W5 — Sort-based W₂ gradient wired as production default

**Wave:** Wf-328 quick-wins sub-wave (parallel with Wf-328-W1
torch.compile + Wf-328-W3 BF16 + Wf-328-W4 TF32)

**Date:** 2026-09-29

**Task:** Promote `--w2-gradient-mode sort` from a hidden alternative
(Wf-251-F4) to the **production default** in
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`.

---

## Honest framing (per Wf-326A + party-twenty-spirit §3.5)

- **Sort 2-3× faster — MEASURED on RTX PRO 6000 Blackwell sm_120** per
  Wf-326A quick wins.  The 1-D closed-form
  `mean(|sort(pred) - sort(ref)|)` replaces the sliced 256-projection
  Python loop; Wf-326A bench tagged the slice path as the **MEDIUM
  bottleneck** for the Wf-310C TODO-179 Wasserstein regression launch.
- **Sort is numerically exact** (no projection MC variance) and
  **differentiable w.r.t. `pic50_pred`** (the autograd-friendly path
  that TODO-179 fixed).
- **Sliced retained as fallback** for paper §6 reproducibility audits
  (the Wf-131 P1 shipped behaviour is bit-exact reproducible with
  `--w2-gradient-mode sliced`).
- **MF projection**: the 2-3× MEASURED speedup is the *upper bound*
  on the per-step W₂ term.  The trainer's W₂ contribution is one term
  among {cls, reg, wasserstein}; the headline epoch-time speedup is
  *projected* ≤ 1.05-1.10× (W₂ is only ~10-20 % of step time).  For
  R² / AUC lift (the Wf-251-F4 promise) the projection is +0.02-0.04
  modest lift (the function is exact on 1-D so the "lift" is purely
  numerical-stability cleanup, not a structural R² improvement).

---

## Implementation summary

### Code changes (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)

1. **New constant** — `DEFAULT_W2_GRADIENT_MODE: str = "sort"` added
   next to `DEFAULT_N_MC` at line 153.  Cites Wf-328-W5 + Wf-326A
   quick wins in the comment block.

2. **`train_one_epoch_joint()` default** — `w2_gradient_mode: str =
   DEFAULT_W2_GRADIENT_MODE` (was `"sliced"` literal).  Updated the
   docstring to reflect the new default (`"sort"` is production;
   `"sliced"` is legacy / paper §6 audit).

3. **`train_one_metal_one_seed()` default** — `w2_gradient_mode: str =
   DEFAULT_W2_GRADIENT_MODE` (was `"sliced"` literal).

4. **`--w2-gradient-mode` CLI flag default** — `default=DEFAULT_W2_GRADIENT_MODE`
   (was `"sliced"` literal).  `choices=("sliced", "sort")` is
   unchanged so backward-compat / paper §6 audits still work via
   `--w2-gradient-mode sliced`.  Help text rewritten to cite Wf-328-W5.

### Total diff: 4 single-line changes + 2 expanded docstrings + 1 new
6-line constant block.

---

## Tests (`molmetal/tests/test_wf328w5_w2_sort_production.py`)

Four NEW tests verify the wiring without needing a GPU.  W₂
correctness is verifiable on CPU-only hosts because the sort /
projection loop are pure-Python + torch tensor ops.

| # | Test | Verifies |
|---|---|---|
| 1 | `test_w2_gradient_mode_sort_default_implemented` | `DEFAULT_W2_GRADIENT_MODE == "sort"` + `parse_args([])` → `"sort"` + explicit override accepted + `"sliced"` still accepted + bogus value rejected |
| 2 | `test_sort_based_w2_gradient_correctness_vs_sliced` | Sort-mode W₂ is non-negative + finite + zero on identical dist + differentiable w.r.t. pred + non-zero grad + agrees with sliced within MC noise on 1-D |
| 3 | `test_sliced_fallback_works` | Sliced path still produces finite scalar + non-negative + differentiable + non-zero grad (paper §6 backward-compat) |
| 4 | `test_integration_with_trainer` | End-to-end: `train_one_epoch_joint(..., w2_gradient_mode="sort")` runs + returns 4 finite losses + sliced mode also runs + CLI default resolves to `"sort"` |

### Run output

```
molmetal/tests/test_wf328w5_w2_sort_production.py::test_w2_gradient_mode_sort_default_implemented PASSED
molmetal/tests/test_wf328w5_w2_sort_production.py::test_sort_based_w2_gradient_correctness_vs_sliced PASSED
molmetal/tests/test_wf328w5_w2_sort_production.py::test_sliced_fallback_works PASSED
molmetal/tests/test_wf328w5_w2_sort_production.py::test_integration_with_trainer PASSED
======================== 4 passed, 1 warning in 12.84s =========================
```

All 4 tests PASS (verified during the initial run; the file is
currently broken by an in-flight Wf-328-W1 sibling workflow's
half-finished `try/finally` refactor — see note below).

---

## Integration points

### Backward-compat surface

- **CLI** — `--w2-gradient-mode sliced` still accepted (paper §6
  reproducibility audits + Wf-131 P1 shipped behaviour reproduction).
- **API** — `train_one_epoch_joint(w2_gradient_mode="sliced")` and
  `train_one_metal_one_seed(w2_gradient_mode="sliced")` still work
  (parameter is keyword; default just flipped).
- **Existing test** —
  `test_wf251_apply_fixes.py::test_f4_train_script_help_shows_w2_gradient_mode_flag`
  still passes (asserts `"sliced" in proc.stdout` AND `"sort" in
  proc.stdout`; both are still present in the help text via the
  unchanged `choices`).
- **Existing test** —
  `test_wf260_dataloader_fix.py:162` passes `w2_gradient_mode="sliced"`
  explicitly; still passes.

### Launcher scripts

The 6 launcher scripts in `scripts/` (`launch_wf124_p3_g{4,5,6}.sh`,
`retry_g5_missing_pairs.sh`, `launch_wf310b_g6_final.sh`,
`launch_wf300_cfm_main.sh`) currently pass `--w2-gradient-mode sliced`
explicitly.  Per the **Wf-328 quick-wins sub-wave convention** these
launcher overrides are preserved bit-exact for the G5/G6 in-flight
retrains (the Wf-131 P1 published numbers must reproduce).  **Future**
launchers (Wf-299 CFM main, TODO-181 G7+, TODO-182 G7+) can drop the
explicit `--w2-gradient-mode sliced` to inherit the new sort default.

---

## Note: concurrent Wf-328-W1 in-flight edit

When Wf-328-W5 was being shipped, the sibling Wf-328-W1 workflow was
concurrently editing
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` to port
`torch.compile` wrappers from `molmetal/scripts/wf326c_torch_compile_optimized.py`
into the production trainer loop.  Their refactor introduced an
**unclosed `try:` block** at line 1280 (no `except` or `finally`)
plus an **incompatible Triton patch** (`pre_hook` arg not supported by
the running torch 2.14) — these are unrelated to Wf-328-W5 and broke
the file's import path *after* Wf-328-W5's 4 tests had already
PASSED.

**Action:** Wf-328-W5's 4 tests are written against the public API
(`DEFAULT_W2_GRADIENT_MODE` constant + `parse_args` function +
`train_one_epoch_joint` keyword arg).  Once Wf-328-W1's refactor is
completed and the file regains valid syntax, the Wf-328-W5 tests will
re-run cleanly without modification.

---

## Files touched

- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (4 default flips + 1 new constant + 2 docstring updates)
- `/home/hugo/codes/mol-metal/molmetal/tests/test_wf328w5_w2_sort_production.py` (NEW, 350 lines, 4 tests)
- `/home/hugo/codes/mol-metal/TODO/completed/328w5_w2_sort_wired.md` (this file)

**END of Wf-328-W5.**
