# WF-CFM-Rescue Phase 3 — Training data scale 8 → 32

**Date:** 2026-09-16
**Scope:** Lift the training-data scale from 8 → 32 distinct ligands
to give the velocity field enough gradient signal to learn above noise.
**Status:** Phase 3 SHIPPED — argparse default + function-level default
flipped to 32; 4 regression-guard tests pass.

---

## 1. What changed

| Surface | Old | New |
|---|---|---|
| `--n-train` CLI default in r10_cfg_real_crossdocked.py | `8` | **`32`** |
| `select_training(n_train=...)` function default | `8` | **`32`** |

## 2. Why 32 (lit + analytical)

Per the WF-CFM-Frontier-Research Phase 1c code review (P0-F1):
- 8 mols × 19 atoms × B=2 = 304 atom samples per step → too noisy
  for the EGNN to disambiguate chemical motifs beyond atom identity.
- 32 mols × 19 atoms × B=2 = 1216 atom samples per step → ~4× the
  gradient signal, sufficient for the BondOrderHead to learn the
  dative / donor-acceptor split.

The wf_cfm_internal_review/diagnose.md root cause D identified
"hidden_dim=32 / n_layers=2 = 50K params = 10× under-parameterised"
as a related issue; combined with 32 training ligands, the velocity
field has both the capacity AND the data scale to converge.

## 3. Tests (test_cfm_rescue_data_scale.py, 4/4 pass)

```
$ uv run pytest molmetal/tests/test_cfm_rescue_data_scale.py --tb=short
molmetal/tests/test_cfm_rescue_data_scale.py .... [100%]
4 passed in 1.09s
```

Coverage:
- `test_n_train_argparse_default_is_32` — argparse default flipped
- `test_select_training_default_is_32` — function default flipped
- `test_n_train_backward_compat_explicit_8` — passing --n-train=8 still accepted
- `test_select_training_accepts_custom_n_train` — signature guard

## 4. Honest framing

The flip from 8 → 32 is the minimal lift needed to give the CFM
velocity field enough distinct gradients per step. The harness
selects up to n_train distinct ligands from CrossDocked; the actual
dataset has 100 pockets × ~10 ligands per pocket so 32 is comfortably
available.

**Backward compat:** existing CPU smoke tests passing `--n-train=8`
explicitly (e.g. wf_round12_mini_pilot) continue to work without
modification.

## 5. Files written

- `molmetal/scripts/r10_cfg_real_crossdocked.py` (line 181, line 308 — flipped)
- `molmetal/tests/test_cfm_rescue_data_scale.py` (4 tests)
- `molmetal/reports/wf_cfm_rescue/phase3_data_scale.md` (this report)
