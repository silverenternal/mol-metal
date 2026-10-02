# Wf-290F2: Fix test_integration.py collection ImportError

**Date:** 2026-09-29
**Scope:** `molmetal/tests/test_integration.py` (5 tests)
**Status:** ImportError FIXED; collection succeeds.

---

## error_before

```
ImportError while importing test module
'/home/hugo/codes/mol-metal/molmetal/tests/test_integration.py'.

molmetal/tests/test_integration.py:137: in <module>
    from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
E   ImportError: cannot import name 'get_metal_ligand_exchange_matrix_patterns'
    from 'molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange'
    (/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py)

Interrupted: 1 error during collection
no tests collected, 1 error in 0.95s
```

The test file (`tests/test_integration.py:137-140`) imports two names from
`molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange`:

```python
from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
    get_metal_ligand_exchange_matrix_patterns,
    register_metal_ligand_exchange_rules,
)
```

---

## root_cause

**Stale `__pycache__` bytecode.** Both names ARE present in the current
source file:

```
$ grep -n "^def " molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py
308:def get_metal_ligand_exchange_patterns() -> List[Tuple[str, str, str]]:
321:def get_metal_ligand_exchange_matrix_patterns() -> List[Tuple[str, str, str]]:
354:def register_metal_ligand_exchange_rules(...
```

But the cached `.pyc` for Python 3.12 was compiled BEFORE these two
functions were added to the source. Inspection of the bytecode cache
directory revealed three conflicting bytecode files:

```
molmetal/molmetal_lam/lam_chem/__pycache__/pt_metal_ligand_exchange.cpython-311.pyc
molmetal/molmetal_lam/lam_chem/__pycache__/pt_metal_ligand_exchange.cpython-312.pyc
molmetal/molmetal_lam/lam_chem/__pycache__/pt_metal_ligand_exchange.cpython-314.pyc
```

The `cpython-312.pyc` matched the source mtime in the wrong direction —
Python's loader returned the stale 3.12 bytecode instead of recompiling
from the live source.

---

## fix_applied

**Removed the stale `__pycache__` directories** so Python recompiles
each `.py` from source against the running interpreter (Python 3.12.13
from `/home/hugo/codes/CadButEaas/.venv-vlm`):

```
find /home/hugo/codes/mol-metal/molmetal -name '__pycache__' -type d \
    -exec rm -rf {} +
```

After cache-clear, the bytecode regenerated cleanly:

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "
from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
    get_metal_ligand_exchange_matrix_patterns,
    register_metal_ligand_exchange_rules,
)
print('OK')"
OK
```

No source edits were needed — the symbols `get_metal_ligand_exchange_matrix_patterns`
(line 321) and `register_metal_ligand_exchange_rules` (line 354) already
existed in `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`
with the exact signatures the test imports.

---

## tests_after

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_integration.py -v --tb=line

collected 5 items

molmetal/tests/test_integration.py::test_001_lipman_to_vina_e2e          PASSED [ 20%]
molmetal/tests/test_integration.py::test_002_mcts_sa_qed_pipeline        PASSED [ 40%]
molmetal/tests/test_integration.py::test_003_lambda_astsr_coupling_adapter FAILED [ 60%]
molmetal/tests/test_integration.py::test_004_mle_matrix_click_compat     PASSED [ 80%]
molmetal/tests/test_integration.py::test_005_end_to_end_smoke            PASSED [100%]

============== 1 failed, 4 passed, 3 warnings in 85.91s (0:01:25) ==============
```

- 5 tests collected (was 0 before the fix).
- **4 tests PASSED**: `test_001`, `test_002`, `test_004`, `test_005`.
- **1 test FAILED** — but with a DIFFERENT error class (assertion, not
  ImportError). `test_003_lambda_astsr_coupling_adapter` fails the
  R² gate: the LambdaASTSR regressor on a 50-row toy problem hits
  R²=-0.3570 (asserted > 0.5). This is a numeric/algorithmic regression
  in `molmetal/molmetal_lam/reward/lambda_astsr.py` (the ASTSR
  symbolic-regression convergence), NOT the import problem this
  workflow was chartered to fix.

The `test_004_mle_matrix_click_compat` test — the one that actually
exercises the symbols this workflow was chartered to fix
(`register_metal_ligand_exchange_rules` + `list_rules_top(scope="metal-coord-matrix")`)
— now **PASSES**, confirming the fix is correct.

---

## summary

| Aspect | Before | After |
| --- | --- | --- |
| Collection | ERROR (ImportError) | 5 collected |
| Symbols reachable from `pt_metal_ligand_exchange` | 0/2 | 2/2 |
| Tests passing in this file | 0 | 4 of 5 |
| Tests failing in this file | 1 (collection) | 1 (assertion, orthogonal) |

The `test_integration.py` collection is restored. The remaining
`test_003` failure is a downstream ASTSR R² convergence problem
unrelated to this workflow and should be tracked in a separate Wf.

---

## report_path

`/home/hugo/codes/mol-metal/molmetal/reports/wf290f2_fix/fix_summary.md`
